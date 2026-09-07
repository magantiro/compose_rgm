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
    FiniteHorizonContinuation,
    ReferenceRow,
    continuation_decision,
)
from compose_v4.control.macro_engine import contract_for
from compose_v4.control.option_selector import (
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

    @property
    def remaining(self) -> int:
        return self.horizon - self.step

    def key(self) -> tuple:
        ctx, lin = self.context, self.lineage
        return (
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


@dataclass
class OptionKernelWork:
    law_enumerations: int = 0
    executor_applications: int = 0
    rejected_products: int = 0
    legal_products: int = 0
    row_cache_hits: int = 0


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
        pre = conditioned_action_distribution(families, p, indices, node.option, step=node.step)
        active = primitive_option_at_step(node.option, node.step)
        contract = contract_for(active, canonical_state_key(node.graph)) if active else None
        clean = np.zeros(len(p), dtype=bool)
        products: dict[int, OptionState] = {}
        for index in pre.indices:
            index = int(index)
            if self.work.executor_applications >= self.max_executor_applications:
                raise ContinuationBudgetExceeded("executor-application budget exhausted")
            self.work.executor_applications += 1
            try:
                product = self.system.apply(node.graph, families[index], actions[index])
                key = canonical_state_key(product)
            except InvalidRewrite:
                self.work.rejected_products += 1
                continue
            if not (
                0 < product.n_real_atoms <= 40
                and graph_connected(product)
                and charge_policy_preserved(node.graph, product)
                and is_valid(key)
                and context_preserved(
                    node.origin, product, node.context.frozen, node.context.terminal_context_slots
                )
                and (contract is None or contract(key))
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
) -> dict:
    """Run one fixed bundle; no extra outer draws or hidden retry-until-success.

    Reference exploration branches are counterfactual executor products, not
    sampled offspring. Only the committed path's completed endpoint is emitted.
    A terminal callback is invoked only at the option's complete horizon.
    """
    from dataclasses import asdict
    from time import perf_counter

    started = perf_counter()
    continuation = FiniteHorizonContinuation(
        kernel.row,
        terminal_weight,
        OptionState.key,
        snapshot_id=snapshot_id,
        max_expansions=max_expansions,
        max_terminal_evaluations=max_terminal_evaluations,
    )
    rng = np.random.default_rng(seed)
    node, trace, logq, status = initial, [], 0.0, "complete"
    while node.remaining:
        try:
            row = kernel.row(node)
        except ContinuationBudgetExceeded:
            status = "executor_budget_exhausted"
            break
        decision = continuation_decision(
            row,
            node.remaining,
            continuation,
            fallback_values=tuple(fallback_weight(s) for s in row.successors),
        )
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
            }
        )
        node = next_node
    return {
        "bundle_id": initial.bundle_id,
        "option": initial.option,
        "status": status,
        "endpoint": canonical_state_key(node.graph) if status == "complete" else None,
        "conditional_path_logq": logq,
        "trace": trace,
        "continuation_work": asdict(continuation.work),
        "kernel_work": asdict(kernel.work),
        "seconds": perf_counter() - started,
    }
