"""Executable adapter for continuation inside a fixed (parent, region, option).

The outer region and option draws are supplied by the caller and never cloned.
Intermediate graphs come only from the supplied production executor. Exact
slot/context/lineage identity, not canonical SMILES, keys continuation work.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    ContinuationWork,
    ControlledDecision,
    FiniteHorizonContinuation,
    ReferenceRow,
    continuation_decision,
)
from compose_v4.control.fused_option import (
    BUILD_FUSED_RING_OPTION,
    FusedProgress,
    completed_fused_cycle,
    descriptor_indices,
    eligible_fusion_edges,
)
from compose_v4.control.macro_engine import state_contract_for
from compose_v4.control.option_selector import (
    ConditionedActionDistribution,
    conditioned_action_distribution,
    option_horizon,
    primitive_option_at_step,
)
from compose_v4.control.region_rewrite import (
    Lineage,
    RewriteContext,
    admissible_indices,
    context_preserved,
    created_slot,
    graph_connected,
)
from compose_v4.control.ring_expansion import (
    EXPAND_RING_OPTION,
    ExpansionProgress,
    completed_expansion,
    eligible_expansion_edges,
    expansion_descriptor_indices,
)
from compose_v4.control.ring_program import (
    RingProgress,
    completed_construction,
    construction_branches,
    construction_indices,
    ring_spec,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem, canonical_state_key


def exact_graph_key(graph: MolecularGraph) -> tuple:
    """Full persistent coordinates, including charge/H channels and array layouts."""
    return tuple(
        (a.dtype.str, a.shape, a.tobytes())
        for a in (
            np.asarray(graph.atom_types),
            np.asarray(graph.bonds),
            np.asarray(graph.formal_charges),
            np.asarray(graph.implicit_h_counts),
        )
    )


@dataclass(frozen=True)
class OptionState:
    graph: MolecularGraph
    origin: MolecularGraph
    context: RewriteContext
    lineage: Lineage
    option: str
    step: int
    horizon: int
    bundle_id: str
    fused_progress: FusedProgress | None = None
    expansion_progress: ExpansionProgress | None = None
    ring_progress: RingProgress | None = None

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("bundle_id is required; outer draws must retain their identity")
        if not 0 < self.graph.n_real_atoms <= 40 or not 0 < self.origin.n_real_atoms <= 40:
            raise ValueError("option states require 1..40 real atoms; null is not a molecule")
        for name, value in (("step", self.step), ("horizon", self.horizon)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        expected = option_horizon(self.option, self.horizon)
        if self.horizon < 1 or expected != self.horizon or self.step > self.horizon:
            raise ValueError("option phase/horizon does not match the registered program")
        if set(self.lineage.slot_of) != set(self.lineage.id_of.values()) or any(
            self.lineage.slot_of.get(identity) != slot
            for slot, identity in self.lineage.id_of.items()
        ):
            raise ValueError("lineage maps are not mutual inverses")
        real_slots = {
            i for i, atom_type in enumerate(self.graph.atom_types) if is_element(atom_type)
        }
        if set(self.lineage.id_of) != real_slots:
            raise ValueError("lineage must identify every real atom slot exactly once")
        if self.lineage.next_id <= max(self.lineage.slot_of, default=-1):
            raise ValueError("lineage next_id must exceed every existing atom identity")
        if self.option == BUILD_FUSED_RING_OPTION:
            if not isinstance(self.fused_progress, FusedProgress):
                raise ValueError("build_fused_ring requires explicit FusedProgress")
            self.fused_progress.validate(self.graph, self.origin, self.context.locus, self.step)
            if self.step == 0 and exact_graph_key(self.graph) != exact_graph_key(self.origin):
                raise ValueError("fused program must start from its exact origin state")
        elif self.fused_progress is not None:
            raise ValueError("fused progress is only valid for build_fused_ring")
        if self.option == EXPAND_RING_OPTION:
            if not isinstance(self.expansion_progress, ExpansionProgress):
                raise ValueError("expand_ring requires explicit ExpansionProgress")
            self.expansion_progress.validate(self.graph, self.origin, self.context.locus, self.step)
            if self.step == 0 and exact_graph_key(self.graph) != exact_graph_key(self.origin):
                raise ValueError("expansion program must start from its exact origin state")
        elif self.expansion_progress is not None:
            raise ValueError("expansion progress is only valid for expand_ring")
        spec = ring_spec(self.option)
        if spec is not None:
            if not isinstance(self.ring_progress, RingProgress):
                raise ValueError("parameterized construction requires RingProgress")
            self.ring_progress.validate(
                self.graph, self.origin, self.context.locus, self.step, spec
            )
            if self.step == 0 and exact_graph_key(self.graph) != exact_graph_key(self.origin):
                raise ValueError("ring program must start from its exact origin state")
        elif self.ring_progress is not None:
            raise ValueError("ring progress requires a parameterized construction option")

    @property
    def remaining(self) -> int:
        return self.horizon - self.step

    def key(self) -> tuple:
        ctx, lin = self.context, self.lineage
        legacy = (
            self.bundle_id,
            exact_graph_key(self.graph),
            exact_graph_key(self.origin),
            ctx.frozen,
            ctx.locus,
            ctx.terminals,
            ctx.interface,
            ctx.k_components,
            ctx.phase,
            tuple(sorted(lin.slot_of.items())),
            tuple(sorted(lin.id_of.items())),
            lin.next_id,
            self.option,
            self.step,
            self.horizon,
        )
        # Preserve every existing cache identity; only the opt-in state extends it.
        if self.ring_progress is not None:
            return legacy + (self.ring_progress,)
        if self.expansion_progress is not None:
            return legacy + (self.expansion_progress,)
        return legacy if self.fused_progress is None else legacy + (self.fused_progress,)


@dataclass
class OptionKernelWork:
    law_enumerations: int = 0
    executor_applications: int = 0
    rejected_products: int = 0
    legal_products: int = 0
    row_cache_hits: int = 0
    product_cache_hits: int = 0


class OptionContinuationKernel:
    """Existing option support and executor, with an explicit application ceiling.

    ``enumerate_law`` returns rule names, slot-addressed actions, probabilities.
    In production use ``from_runtime`` to delegate to the one model evaluator.
    Fixture callbacks are permitted only as explicitly labeled test references.
    No top-k or additional chemistry catalog is introduced here.
    """

    def __init__(
        self,
        enumerate_law: Callable[[MolecularGraph], tuple],
        system: RewriteSystem,
        *,
        max_executor_applications: int,
        macro_temperature: float = 2.0,
        macro_exploration: float = 0.15,
    ) -> None:
        if (
            isinstance(max_executor_applications, bool)
            or not isinstance(max_executor_applications, int)
            or max_executor_applications < 0
        ):
            raise ValueError("max_executor_applications must be a nonnegative integer")
        if macro_temperature != 2.0 or macro_exploration != 0.15:
            raise ValueError("the inherited macro temperature/exploration are frozen at 2.0/0.15")
        self.enumerate_law = enumerate_law
        self.system = system
        self.max_executor_applications = max_executor_applications
        self.work = OptionKernelWork()
        self._marks: dict[tuple, tuple[tuple[str, Any], ...]] = {}
        self._rows: dict[tuple, ReferenceRow[OptionState]] = {}
        self._fused_support: dict[tuple, dict] = {}
        self._expansion_support: dict[tuple, dict] = {}
        self._construction_support: dict[tuple, dict] = {}
        self._construction_products: dict[tuple, MolecularGraph | None] = {}
        self._primitive_products: dict[tuple, tuple[MolecularGraph, str] | None] = {}

    @classmethod
    def from_runtime(cls, model, system, *, time_point: float, max_executor_applications: int):
        from compose_v4.experiments.production_successor_kernel import (
            enumerate_factorized_marked_law,
        )

        def enumerate_law(graph):
            law = enumerate_factorized_marked_law(model, graph, time_point)
            return (
                tuple(m.executor_rule_name for m in law.marks),
                tuple(m.action for m in law.marks),
                tuple(m.probability for m in law.marks),
            )

        return cls(enumerate_law, system, max_executor_applications=max_executor_applications)

    def row(self, node: OptionState) -> ReferenceRow[OptionState]:
        cache_key = node.key()
        if cache_key in self._rows:
            self.work.row_cache_hits += 1
            return self._rows[cache_key]
        if node.remaining == 0:
            return ReferenceRow((), ())
        self.work.law_enumerations += 1
        families, actions, probabilities = self.enumerate_law(node.graph)
        p = np.asarray(probabilities, dtype=float)
        if (
            p.ndim != 1
            or len(p) != len(families)
            or len(p) != len(actions)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (len(p) and p.sum() <= 0)
        ):
            raise ValueError("marked law must have aligned finite nonnegative positive-mass rows")
        indices, _ = admissible_indices(families, actions, node.context)
        if (
            node.option in (BUILD_FUSED_RING_OPTION, EXPAND_RING_OPTION)
            or node.ring_progress is not None
        ):
            return self._descriptor_row(node, families, actions, p, indices)
        pre = conditioned_action_distribution(families, p, indices, node.option, step=node.step)
        active = primitive_option_at_step(node.option, node.step)
        contract = state_contract_for(active, node.graph) if active else None
        clean = np.zeros(len(p), dtype=bool)
        products: dict[int, OptionState] = {}
        for index in pre.indices:
            index = int(index)
            physical_key = (exact_graph_key(node.graph), families[index], repr(actions[index]))
            if physical_key in self._primitive_products:
                self.work.product_cache_hits += 1
                executed = self._primitive_products[physical_key]
            else:
                if self.work.executor_applications >= self.max_executor_applications:
                    raise ContinuationBudgetExceeded("executor-application budget exhausted")
                self.work.executor_applications += 1
                try:
                    product = self.system.apply(node.graph, families[index], actions[index])
                    executed = (product, canonical_state_key(product))
                except InvalidRewrite:
                    executed = None
                self._primitive_products[physical_key] = executed
            if executed is None:
                self.work.rejected_products += 1
                continue
            product, key = executed
            if not (
                0 < product.n_real_atoms <= 40
                and graph_connected(product)
                and charge_policy_preserved(node.graph, product)
                and is_valid(key)
                and context_preserved(
                    node.origin, product, node.context.frozen, node.context.terminal_context_slots
                )
                and (contract is None or contract(product))
            ):
                self.work.rejected_products += 1
                continue
            lineage = node.lineage.observe(families[index], actions[index])
            new_slot = created_slot(actions[index])
            context = node.context.with_locus(new_slot) if new_slot is not None else node.context
            products[index] = OptionState(
                product,
                node.origin,
                context,
                lineage,
                node.option,
                node.step + 1,
                node.horizon,
                node.bundle_id,
            )
            self.work.legal_products += 1
            clean[index] = True
        conditioned = conditioned_action_distribution(
            families, p, indices, node.option, step=node.step, clean=clean
        )
        self._marks[cache_key] = tuple(
            (families[int(i)], actions[int(i)]) for i in conditioned.indices
        )
        row = ReferenceRow(
            tuple(products[int(i)] for i in conditioned.indices),
            tuple(float(p) for p in conditioned.probabilities),
        )
        self._rows[cache_key] = row
        return row

    def _descriptor_row(self, node, families, actions, probabilities, admissible):
        """Joint edge/mark reference; share executor work, not augmented states."""
        spec = ring_spec(node.option)
        expansion = node.option == EXPAND_RING_OPTION
        progress = (
            node.ring_progress
            if spec
            else node.expansion_progress
            if expansion
            else node.fused_progress
        )
        eligible_edges = eligible_expansion_edges if expansion else eligible_fusion_edges
        match_indices = expansion_descriptor_indices if expansion else descriptor_indices
        if spec:
            edges = (
                construction_branches(node.origin, node.context.locus, spec)
                if not progress.anchors
                else ((progress.anchors, progress.pattern),)
            )
        else:
            edges = (
                eligible_edges(node.origin, node.context.locus)
                if progress.edge is None
                else (progress.edge,)
            )
        active = primitive_option_at_step(node.option, node.step)
        contract = state_contract_for(active, node.graph)

        def condition(matches):
            if not expansion and spec is None:
                return conditioned_action_distribution(families, probabilities, matches, active)
            # This NEW channel conditions descriptors before applying the
            # inherited cap. Unrelated marks must not erase its sole closure.
            # Existing fused/generic/macro laws retain their frozen cap order.
            idx = np.asarray(matches, dtype=int)
            local = conditioned_action_distribution(
                [families[i] for i in idx], probabilities[idx], range(len(idx)), active
            )
            return ConditionedActionDistribution(
                node.option, active, idx[local.indices], local.probabilities
            )

        in_region = set(admissible)
        pre_rows, audit = {}, []
        for edge in edges:
            matches = sorted(
                in_region.intersection(
                    construction_indices(
                        families,
                        actions,
                        probabilities,
                        node.graph,
                        spec,
                        progress,
                        edge,
                        node.step,
                    )
                    if spec
                    else match_indices(families, actions, probabilities, progress, edge, node.step)
                )
            )
            pre = condition(matches)
            pre_rows[edge] = pre
            audit.append(
                {
                    **(
                        {"branch": [list(items) for items in edge]}
                        if spec
                        else {"edge": list(edge)}
                    ),
                    "descriptor_region_marks": len(matches),
                    "after_inherited_cap": len(pre.indices),
                }
            )

        # One physical action can belong to two oriented-edge branches. Apply
        # each index once, then retain separate program states and joint mass.
        products = {}
        for index in sorted({int(i) for pre in pre_rows.values() for i in pre.indices}):
            # A parameter menu must not reexecute the same physical first edit
            # once per quota/pattern. Context checks remain bundle-specific.
            product_key = (exact_graph_key(node.graph), families[index], repr(actions[index]))
            if spec and product_key in self._construction_products:
                self.work.product_cache_hits += 1
                product = self._construction_products[product_key]
            else:
                if self.work.executor_applications >= self.max_executor_applications:
                    raise ContinuationBudgetExceeded("executor-application budget exhausted")
                self.work.executor_applications += 1
                try:
                    product = self.system.apply(node.graph, families[index], actions[index])
                    canonical_state_key(product)
                except InvalidRewrite:
                    product = None
                if spec:
                    self._construction_products[product_key] = product
            if product is None:
                self.work.rejected_products += 1
                continue
            key = canonical_state_key(product)
            if not (
                0 < product.n_real_atoms <= 40
                and graph_connected(product)
                and charge_policy_preserved(node.graph, product)
                and is_valid(key)
                and context_preserved(
                    node.origin, product, node.context.frozen, node.context.terminal_context_slots
                )
                and (contract is None or contract(product))
            ):
                self.work.rejected_products += 1
                continue
            products[index] = product

        branches = []
        for edge, receipt in zip(edges, audit):
            successors = {}
            for raw_index in pre_rows[edge].indices:
                index = int(raw_index)
                if index not in products:
                    continue
                product = products[index]
                slot = created_slot(actions[index])
                next_progress = (
                    RingProgress(
                        edge[0],
                        edge[1],
                        progress.path + (slot,) if slot is not None else progress.path,
                    )
                    if spec
                    else ExpansionProgress(edge, slot if slot is not None else progress.new_slot)
                    if expansion
                    else FusedProgress(
                        edge, progress.path + (slot,) if slot is not None else progress.path
                    )
                )
                completes = completed_expansion if expansion else completed_fused_cycle
                completion_failed = (
                    spec is not None
                    and node.step in (spec.growth, spec.horizon - 1)
                    and not completed_construction(
                        node.origin, product, next_progress, spec, refined=node.step > spec.growth
                    )
                ) or (
                    spec is None
                    and node.step == node.horizon - 1
                    and not completes(node.origin, product, next_progress)
                )
                if completion_failed:
                    self.work.rejected_products += 1
                    continue
                context = node.context.with_locus(slot) if slot is not None else node.context
                if expansion or spec:
                    try:
                        if spec:
                            next_progress.validate(
                                product, node.origin, context.locus, node.step + 1, spec
                            )
                        else:
                            next_progress.validate(
                                product, node.origin, context.locus, node.step + 1
                            )
                    except ValueError:
                        self.work.rejected_products += 1
                        continue
                successors[index] = OptionState(
                    product,
                    node.origin,
                    context,
                    node.lineage.observe(families[index], actions[index]),
                    node.option,
                    node.step + 1,
                    node.horizon,
                    node.bundle_id,
                    fused_progress=None if expansion or spec else next_progress,
                    expansion_progress=next_progress if expansion else None,
                    ring_progress=next_progress if spec else None,
                )
            receipt["after_product_contract"] = len(successors)
            conditioned = condition(sorted(successors))
            if len(conditioned.indices):
                branches.append((successors, conditioned))
        nodes, mass, marks = [], [], []
        for successors, conditioned in branches:
            for index, probability in zip(conditioned.indices, conditioned.probabilities):
                index = int(index)
                nodes.append(successors[index])
                mass.append(float(probability) / len(branches))
                marks.append((families[index], actions[index]))
        row = ReferenceRow(tuple(nodes), tuple(mass))
        cache_key = node.key()
        self.work.legal_products += len({i for successors, _ in branches for i in successors})
        self._marks[cache_key] = tuple(marks)
        support = (
            self._construction_support
            if spec
            else self._expansion_support
            if expansion
            else self._fused_support
        )
        support[cache_key] = {
            **(
                {"eligible_branches": len(edges), "product_applicable_branches": len(branches)}
                if spec
                else {
                    "eligible_oriented_edges": len(edges),
                    "product_applicable_oriented_edges": len(branches),
                }
            ),
            "augmented_successors": len(nodes),
            "physical_marks": len({i for successors, _ in branches for i in successors}),
            "edges": audit,
        }
        self._rows[cache_key] = row
        return row

    def fused_support(self, node: OptionState) -> dict:
        """Completed-row funnel only; never exposes partially normalized work."""
        from copy import deepcopy

        return deepcopy(self._fused_support[node.key()])

    def expansion_support(self, node: OptionState) -> dict:
        from copy import deepcopy

        return deepcopy(self._expansion_support[node.key()])

    def construction_support(self, node: OptionState) -> dict:
        from copy import deepcopy

        return deepcopy(self._construction_support[node.key()])

    def marks(self, node: OptionState) -> tuple[tuple[str, Any], ...]:
        return self._marks[node.key()]


def sample_option_trajectory(
    initial: OptionState,
    kernel: OptionContinuationKernel,
    terminal_weight: Callable[[OptionState], float],
    fallback_weight: Callable[[OptionState], float],
    *,
    snapshot_id: str,
    seed: int,
    max_expansions: int,
    max_terminal_evaluations: int,
    estimator: str = "exact",
    samples_per_successor: int = 32,
    confidence_alpha: float = 0.05,
    max_rollouts: int = 1024,
) -> dict:
    """Run one fixed bundle; no extra outer draws or hidden retry-until-success.

    Reference exploration branches are counterfactual executor products, not
    sampled offspring. Only the committed path's completed endpoint is emitted.
    A terminal callback is invoked only at the option's complete horizon.
    """
    from dataclasses import asdict
    from time import perf_counter

    from compose_v4.control.sampled_continuation import (
        SampledContinuation,
        sampled_continuation_decision,
    )

    started = perf_counter()
    common = {
        "snapshot_id": snapshot_id,
        "max_expansions": max_expansions,
        "max_terminal_evaluations": max_terminal_evaluations,
    }
    planner_seed = None
    if estimator == "reference":
        continuation = None
        rng = np.random.default_rng(seed)
    elif estimator == "exact":
        continuation = FiniteHorizonContinuation(
            kernel.row, terminal_weight, OptionState.key, **common
        )
        rng = np.random.default_rng(seed)  # preserve the existing exact fixture stream
    elif estimator == "sampled":
        planner_stream, path_stream = np.random.SeedSequence(seed).spawn(2)
        planner_seed = int(planner_stream.generate_state(1, dtype=np.uint64)[0])
        continuation = SampledContinuation(
            kernel.row,
            terminal_weight,
            OptionState.key,
            **common,
            seed=planner_seed,
            samples_per_successor=samples_per_successor,
            alpha=confidence_alpha,
            max_rollouts=max_rollouts,
        )
        rng = np.random.default_rng(path_stream)
    else:
        raise ValueError(f"unknown continuation estimator: {estimator!r}")
    node, trace, logq, status = initial, [], 0.0, "complete"
    while node.remaining:
        try:
            row = kernel.row(node)
        except ContinuationBudgetExceeded:
            status = "executor_budget_exhausted"
            break
        estimate = None
        kwargs = (
            {}
            if estimator == "reference"
            else {"fallback_values": tuple(fallback_weight(s) for s in row.successors)}
        )
        if estimator == "reference":
            decision = ControlledDecision(row.probabilities, "reference", None, 0.0, 0.0)
        elif estimator == "sampled":
            sampled = sampled_continuation_decision(row, node.remaining, continuation, **kwargs)
            decision, estimate = sampled.decision, sampled.estimate
        else:
            decision = continuation_decision(row, node.remaining, continuation, **kwargs)
        if not decision.probabilities:
            status = "no_admissible_action"
            break
        selected = int(rng.choice(len(row.successors), p=decision.probabilities))
        next_node = row.successors[selected]
        probability = decision.probabilities[selected]
        logq += math.log(probability)
        family, action = kernel.marks(node)[selected]
        trace.append(
            {
                "step": next_node.step,
                "before": canonical_state_key(node.graph),
                "after": canonical_state_key(next_node.graph),
                "family": family,
                "action": asdict(action),
                "reference_probability": row.probabilities[selected],
                "probability": probability,
                "decision_status": decision.status,
                "eta": decision.eta,
                "kl": decision.kl,
                "sampled_estimate": asdict(estimate) if estimate is not None else None,
            }
        )
        if next_node.fused_progress is not None:
            from compose_v4.rewrite.trace_shard import encode_state

            trace[-1]["fused_progress"] = next_node.fused_progress.payload()
            trace[-1]["exact_state"] = encode_state(next_node.graph)
            trace[-1]["support_funnel"] = kernel.fused_support(node)
        if next_node.expansion_progress is not None:
            from compose_v4.rewrite.trace_shard import encode_state

            trace[-1]["expansion_progress"] = next_node.expansion_progress.payload()
            trace[-1]["exact_state"] = encode_state(next_node.graph)
            trace[-1]["support_funnel"] = kernel.expansion_support(node)
        if next_node.ring_progress is not None:
            from compose_v4.rewrite.trace_shard import encode_state

            trace[-1]["ring_progress"] = next_node.ring_progress.payload()
            trace[-1]["exact_state"] = encode_state(next_node.graph)
            trace[-1]["support_funnel"] = kernel.construction_support(node)
        node = next_node
    result = {
        "bundle_id": initial.bundle_id,
        "option": initial.option,
        "estimator": estimator,
        "randomness": {
            "seed": seed,
            "planner_seed": planner_seed,
            "derivation": "SeedSequence(seed).spawn(2): planner uint64 seed, path stream"
            if estimator == "sampled"
            else "default_rng(seed), no planner"
            if estimator == "reference"
            else "default_rng(seed), exact deterministic planner",
        },
        "path_probability_role": "conditional on realized planner randomness, not marginalized over lookahead samples",
        "status": status,
        "endpoint": canonical_state_key(node.graph) if status == "complete" else None,
        "conditional_path_logq": logq,
        "trace": trace,
        "continuation_work": asdict(continuation.work if continuation else ContinuationWork()),
        "kernel_work": asdict(kernel.work),
        "seconds": perf_counter() - started,
    }
    if node.fused_progress is not None:
        result["final_fused_progress"] = node.fused_progress.payload()
        if status == "no_admissible_action":
            result["failure_support_funnel"] = kernel.fused_support(node)
    if node.expansion_progress is not None:
        result["final_expansion_progress"] = node.expansion_progress.payload()
        if status == "no_admissible_action":
            result["failure_support_funnel"] = kernel.expansion_support(node)
    if node.ring_progress is not None:
        result["final_ring_progress"] = node.ring_progress.payload()
        if status == "no_admissible_action":
            result["failure_support_funnel"] = kernel.construction_support(node)
    return result
