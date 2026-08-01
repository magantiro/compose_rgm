"""True canonical-successor micro-overfit gates for the COMPOSE editor.

This module is deliberately independent of the full stochastic training
sampler.  It answers a narrower prelaunch question: given a small,
executor-verified panel of exact molecular jumps, can the configured model put
probability on the correct *canonical molecular successor*?

The gate reports family selection and within-family selection separately, but
optimizes the productive embedded molecular-successor NLL used by fixed-budget
editing.  It is therefore suitable for diagnosing support, labels,
factorization, and head capacity before any expensive mixed-corpus run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np
import torch
from torch import Tensor

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.experiments.factorized_mark_conditional import (
    assert_teachers_in_exact_candidates,
)
from compose_v4.experiments.factorized_successor_training import (
    TeacherSuccessorFiber,
    compile_teacher_successor_fiber,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.trace import RewriteTrace

CORE_EDITING_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
)

RINGCORE_EDITING_FAMILIES = (
    *CORE_EDITING_FAMILIES,
    "ring_system_restate",
)


class SuccessorMicroOverfitError(RuntimeError):
    """A development panel violates the successor-level gate contract."""


@dataclass(frozen=True)
class SuccessorSupervisionExample:
    """One exact executable molecular jump used by the bounded gate."""

    family_name: str
    state: MolecularGraph
    target: MolecularGraph
    teacher_rule_name: str
    teacher_action: Any
    time: float = 0.5
    teacher_rate: float = 1.0
    importance_weight: float = 1.0
    data_lane: str = "development"

    @property
    def source_key(self) -> str:
        return canonical_state_key(self.state)

    @property
    def target_key(self) -> str:
        return canonical_state_key(self.target)

    def __post_init__(self) -> None:
        expected_family = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(
            self.teacher_rule_name,
            self.teacher_rule_name,
        )
        if self.family_name != expected_family:
            raise ValueError(
                f"family {self.family_name!r} disagrees with teacher rule "
                f"{self.teacher_rule_name!r}"
            )
        if self.source_key == self.target_key:
            raise ValueError("a successor supervision example cannot be virtual")
        if not 0.0 < float(self.time) < 1.0:
            raise ValueError("micro-overfit time must lie strictly inside (0, 1)")
        if self.teacher_rate <= 0.0 or self.importance_weight <= 0.0:
            raise ValueError("micro-overfit rates and weights must be positive")


@dataclass(frozen=True)
class PreparedSuccessorPanel:
    """A fixed tensor batch and its precompiled teacher-successor fibers."""

    examples: tuple[SuccessorSupervisionExample, ...]
    batch: FactorizedMarkBatch
    fibers: tuple[TeacherSuccessorFiber, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("a successor panel must be nonempty")
        if self.batch.batch_size != len(self.examples):
            raise ValueError("successor panel batch does not align with examples")
        if len(self.fibers) != len(self.examples):
            raise ValueError("successor panel fibers do not align with examples")


def examples_from_traces(
    traces: Iterable[RewriteTrace],
    *,
    families: Iterable[str] = CORE_EDITING_FAMILIES,
    maximum_per_family: int | None = None,
    unique_source_molecules: bool = True,
    time: float = 0.5,
    data_lane: str = "development_trace",
    system: RewriteSystem | None = None,
) -> tuple[SuccessorSupervisionExample, ...]:
    """Extract executor-verified one-step examples from complete traces.

    Deduplication is by canonical source molecule within each family.  The
    exact persistent-slot state is retained in the example and never rebuilt
    from its canonical identity.
    """

    wanted = tuple(dict.fromkeys(str(name) for name in families))
    unknown = set(wanted) - set(RINGCORE_EDITING_FAMILIES)
    if unknown:
        raise ValueError(f"unknown editing families: {sorted(unknown)}")
    if maximum_per_family is not None and maximum_per_family <= 0:
        raise ValueError("maximum_per_family must be positive")
    runtime = system or de_novo_rewrite_system()
    counts = {name: 0 for name in wanted}
    seen_sources = {name: set() for name in wanted}
    examples: list[SuccessorSupervisionExample] = []

    for trace in traces:
        state = trace.source
        for step in trace.steps:
            try:
                successor = runtime.apply(state, step.rule_name, step.action)
            except Exception as error:
                raise SuccessorMicroOverfitError(
                    f"trace step {step.rule_name!r} is not executable"
                ) from error
            family_name = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(
                step.rule_name,
                step.rule_name,
            )
            if family_name in counts:
                source_key = canonical_state_key(state)
                at_limit = (
                    maximum_per_family is not None
                    and counts[family_name] >= maximum_per_family
                )
                duplicate = (
                    unique_source_molecules and source_key in seen_sources[family_name]
                )
                if not at_limit and not duplicate:
                    examples.append(
                        SuccessorSupervisionExample(
                            family_name=family_name,
                            state=state,
                            target=successor,
                            teacher_rule_name=step.rule_name,
                            teacher_action=step.action,
                            time=float(time),
                            data_lane=data_lane,
                        )
                    )
                    counts[family_name] += 1
                    seen_sources[family_name].add(source_key)
            state = successor
        if canonical_state_key(state) != canonical_state_key(trace.target):
            raise SuccessorMicroOverfitError(
                "trace endpoint does not match its recorded target"
            )
        if maximum_per_family is not None and all(
            count >= maximum_per_family for count in counts.values()
        ):
            break
    return tuple(examples)


def prepare_successor_panel(
    model: FactorizedTraceletRateModel,
    examples: Iterable[SuccessorSupervisionExample],
) -> PreparedSuccessorPanel:
    """Tensorize examples and compile the exact production successor fibers."""

    rows = tuple(examples)
    fibers = tuple(
        compile_teacher_successor_fiber(
            model,
            row.state,
            row.target,
            time=row.time,
        )
        for row in rows
    )
    return prepare_cached_successor_panel(model, rows, fibers)


def prepare_cached_successor_panel(
    model: FactorizedTraceletRateModel,
    examples: Iterable[SuccessorSupervisionExample],
    fibers: Iterable[TeacherSuccessorFiber],
) -> PreparedSuccessorPanel:
    """Tensorize examples against exact, externally validated cached fibers.

    The cache is accepted only when its exact persistent-slot source and
    target identities, canonical keys, and scored coordinates all agree with
    the current model batch.  This lets every T1 capacity arm share one
    immutable support compilation without silently trusting stale aliases.
    """

    rows = tuple(examples)
    cached_fibers = tuple(fibers)
    if not rows:
        raise ValueError("cannot prepare an empty successor panel")
    if len(cached_fibers) != len(rows):
        raise ValueError("cached successor fibers do not align with examples")
    slot_counts = {row.state.n_atoms for row in rows} | {
        row.target.n_atoms for row in rows
    }
    if len(slot_counts) != 1:
        raise ValueError("successor panel mixes persistent-slot capacities")
    capabilities = model.operator_capabilities
    batch = prepare_factorized_mark_batch(
        tuple(row.state for row in rows),
        tuple(float(row.time) for row in rows),
        tuple(row.teacher_action for row in rows),
        tuple(row.teacher_rule_name for row in rows),
        tuple(float(row.teacher_rate) for row in rows),
        tuple(float(row.importance_weight) for row in rows),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=(capabilities.atom_restate_action_semantics),
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
    )
    assert_teachers_in_exact_candidates(batch)

    for index, (row, fiber) in enumerate(zip(rows, cached_fibers, strict=True)):
        if (
            fiber.source_key != row.source_key
            or fiber.target_key != row.target_key
            or fiber.state_support.source_state_sha256
            != persistent_slot_state_sha256(row.state)
            or fiber.target_state_sha256 != persistent_slot_state_sha256(row.target)
        ):
            raise SuccessorMicroOverfitError(
                f"cached successor fiber disagrees with exact example identity at row {index}"
            )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        mark_prediction = model.forward_mark_batch(batch.to(model.device))
    if not bool(torch.isfinite(mark_prediction.selected_mark_log_probability).all()):
        failing = [
            {
                "index": index,
                "family_name": row.family_name,
                "source_key": row.source_key,
                "teacher_rule_name": row.teacher_rule_name,
                "teacher_action": repr(row.teacher_action),
            }
            for index, row in enumerate(rows)
            if not bool(
                torch.isfinite(mark_prediction.selected_mark_log_probability[index])
            )
        ]
        raise SuccessorMicroOverfitError(
            f"at least one recorded teacher mark is outside the scored support: {failing}"
        )
    # Indexing every cached alias through the differentiable production bridge
    # catches a cache built under another action-table layout even when the
    # molecular keys happen to agree.
    with torch.no_grad():
        forward_teacher_successor_batch(
            model,
            batch.to(model.device),
            cached_fibers,
        )
    model.train(was_training)
    return PreparedSuccessorPanel(rows, batch, cached_fibers)


def _weighted_mean(values: Tensor, weights: Tensor) -> float:
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise SuccessorMicroOverfitError("metric weights sum to zero")
    return float((values * weights).sum() / denominator)


@torch.no_grad()
def successor_panel_metrics(
    model: FactorizedTraceletRateModel,
    panel: PreparedSuccessorPanel,
    *,
    include_per_example: bool = True,
) -> dict[str, Any]:
    """Measure proper scoring, rank, aliasing, and candidate complexity."""

    was_training = model.training
    model.eval()
    device_batch = panel.batch.to(model.device)
    prediction = forward_teacher_successor_batch(
        model,
        device_batch,
        panel.fibers,
    )
    weights = device_batch.importance_weights
    conditioned_log_probability = (
        prediction.selected_productive_successor_log_probability
    )
    nll = -conditioned_log_probability
    teacher_probability = conditioned_log_probability.exp()
    family_probability = prediction.teacher_family_log_probability.exp()
    within_family_probability = (
        prediction.selected_within_teacher_family_log_probability.exp()
    )
    within_family_nll = -prediction.selected_within_teacher_family_log_probability
    teacher_family_indices = torch.tensor(
        [MARK_RULE_TO_INDEX[row.family_name] for row in panel.examples],
        dtype=torch.long,
        device=prediction.family_log_probabilities.device,
    )
    teacher_family_scores = prediction.family_log_probabilities.gather(
        1,
        teacher_family_indices[:, None],
    ).squeeze(1)
    family_ranks_tensor = 1 + (
        prediction.family_log_probabilities > teacher_family_scores[:, None] + 1e-12
    ).sum(dim=1)

    ranks: list[int] = []
    support_sizes: list[int] = []
    raw_mark_counts: list[int] = []
    alias_multiplicities: list[int] = []
    virtual_masses: list[float] = []
    uniform_nlls: list[float] = []
    per_example: list[dict[str, Any]] = []
    for index, row in enumerate(panel.examples):
        result = canonical_successor_result(
            model,
            row.state,
            row.time,
            prepared_batch=panel.batch.subbatch(index, index + 1),
        )
        target = next(
            (
                successor
                for successor in result.batch.successors
                if successor.key == row.target_key
            ),
            None,
        )
        if target is None:
            raise SuccessorMicroOverfitError(
                "compiled teacher successor disappeared from production support"
            )
        probability = float(target.probability)
        rank = 1 + sum(
            successor.probability > probability + 1e-12
            for successor in result.batch.successors
        )
        support_size = result.batch.support_size
        ranks.append(rank)
        support_sizes.append(support_size)
        raw_mark_counts.append(result.diagnostics.raw_mark_count)
        alias_multiplicities.append(target.alias_count)
        virtual_masses.append(result.diagnostics.virtual_self_mass)
        uniform_nlls.append(log(float(support_size)))
        if include_per_example:
            per_example.append(
                {
                    "index": index,
                    "family_name": row.family_name,
                    "data_lane": row.data_lane,
                    "source_key": row.source_key,
                    "target_key": row.target_key,
                    "teacher_successor_probability": probability,
                    "teacher_successor_nll": -log(max(probability, 1e-300)),
                    "teacher_successor_rank": rank,
                    "support_size": support_size,
                    "raw_mark_count": result.diagnostics.raw_mark_count,
                    "alias_multiplicity": target.alias_count,
                    "productive_mass": result.diagnostics.raw_productive_mass,
                    "virtual_self_mass": result.diagnostics.virtual_self_mass,
                    "teacher_family_probability": float(family_probability[index]),
                    "teacher_family_nll": float(
                        -prediction.teacher_family_log_probability[index]
                    ),
                    "teacher_family_rank": int(family_ranks_tensor[index]),
                    "within_teacher_family_successor_probability": float(
                        within_family_probability[index]
                    ),
                    "within_teacher_family_successor_nll": float(
                        within_family_nll[index]
                    ),
                }
            )

    rank_array = np.asarray(ranks, dtype=np.float64)
    support_array = np.asarray(support_sizes, dtype=np.float64)
    uniform_nll_tensor = torch.tensor(
        uniform_nlls,
        dtype=nll.dtype,
        device=nll.device,
    )
    excess_over_uniform = nll - uniform_nll_tensor
    normalized_lift = -excess_over_uniform / uniform_nll_tensor.clamp_min(1e-12)
    group_target_weights: dict[
        tuple[str, float],
        dict[str, float],
    ] = defaultdict(lambda: defaultdict(float))
    group_row_indices: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(panel.examples):
        group = (persistent_slot_state_sha256(row.state), float(row.time))
        group_target_weights[group][row.target_key] += float(weights[index])
        group_row_indices[group].append(index)

    multi_target_groups = {
        group
        for group, target_weights in group_target_weights.items()
        if len(target_weights) > 1
    }
    deterministic_groups = set(group_target_weights) - multi_target_groups

    def group_entropy(group: tuple[str, float]) -> float:
        target_weights = group_target_weights[group]
        total = sum(target_weights.values())
        return -sum(
            (weight / total) * log(weight / total)
            for weight in target_weights.values()
            if weight > 0.0
        )

    repeated_indices = tuple(
        index
        for group in sorted(multi_target_groups)
        for index in group_row_indices[group]
    )
    deterministic_indices = tuple(
        index
        for group in sorted(deterministic_groups)
        for index in group_row_indices[group]
    )

    def subset_weighted_mean(values: Tensor, indices: tuple[int, ...]) -> float:
        if not indices:
            return 0.0
        selected = torch.tensor(indices, dtype=torch.long, device=values.device)
        return _weighted_mean(values[selected], weights[selected])

    repeated_weight = sum(float(weights[index]) for index in repeated_indices)
    repeated_empirical_entropy = (
        sum(
            sum(group_target_weights[group].values()) * group_entropy(group)
            for group in multi_target_groups
        )
        / repeated_weight
        if repeated_weight > 0.0
        else 0.0
    )
    repeated_nll = subset_weighted_mean(nll, repeated_indices)
    deterministic_ranks = (
        rank_array[np.asarray(deterministic_indices, dtype=np.int64)]
        if deterministic_indices
        else np.asarray([], dtype=np.float64)
    )
    metrics: dict[str, Any] = {
        "n_examples": len(panel.examples),
        "families": sorted({row.family_name for row in panel.examples}),
        "canonical_successor_nll": _weighted_mean(nll, weights),
        "teacher_successor_probability": _weighted_mean(
            teacher_probability,
            weights,
        ),
        "teacher_family_probability": _weighted_mean(
            family_probability,
            weights,
        ),
        "teacher_family_nll": _weighted_mean(
            -prediction.teacher_family_log_probability,
            weights,
        ),
        "teacher_family_top1_recall": float(
            (family_ranks_tensor <= 1).to(torch.float32).mean()
        ),
        "teacher_family_top3_recall": float(
            (family_ranks_tensor <= 3).to(torch.float32).mean()
        ),
        "within_teacher_family_successor_probability": _weighted_mean(
            within_family_probability,
            weights,
        ),
        "within_teacher_family_successor_nll": _weighted_mean(
            within_family_nll,
            weights,
        ),
        "uniform_successor_nll": _weighted_mean(uniform_nll_tensor, weights),
        "excess_nll_over_uniform": _weighted_mean(excess_over_uniform, weights),
        "uniform_normalized_successor_lift": _weighted_mean(
            normalized_lift,
            weights,
        ),
        "teacher_successor_top1_recall": float(np.mean(rank_array <= 1)),
        "teacher_successor_top3_recall": float(np.mean(rank_array <= 3)),
        "teacher_successor_mrr": float(np.mean(1.0 / rank_array)),
        "mean_raw_mark_count": float(np.mean(raw_mark_counts)),
        "mean_canonical_successor_count": float(np.mean(support_array)),
        "maximum_canonical_successor_count": int(max(support_sizes)),
        "mean_alias_multiplicity": float(np.mean(alias_multiplicities)),
        "maximum_alias_multiplicity": int(max(alias_multiplicities)),
        "mean_virtual_self_mass": float(np.mean(virtual_masses)),
        "exact_state_time_group_count": len(group_target_weights),
        "deterministic_state_group_count": len(deterministic_groups),
        "repeated_multi_successor_state_group_count": len(multi_target_groups),
        "deterministic_state_canonical_successor_nll": (
            subset_weighted_mean(nll, deterministic_indices)
        ),
        "deterministic_state_teacher_successor_top1_recall": (
            float(np.mean(deterministic_ranks <= 1))
            if len(deterministic_ranks)
            else 0.0
        ),
        "repeated_state_canonical_successor_nll": repeated_nll,
        "repeated_state_empirical_entropy": repeated_empirical_entropy,
        "repeated_state_excess_nll_over_empirical_entropy": (
            repeated_nll - repeated_empirical_entropy if repeated_indices else 0.0
        ),
    }
    if include_per_example:
        metrics["per_example"] = per_example
    model.train(was_training)
    return metrics


_HEAD_PREFIXES_BY_FAMILY = {
    "atom_insert": ("grow_root_head.", "grow_query.", "grow_option."),
    "atom_delete": ("delete_head.",),
    "atom_restate": ("restate_head.",),
    "bond_reorder": ("reorder_head.",),
    "bond_reroute": ("graft_head.",),
    "cycle_insert": ("cycle_close_head.",),
    "cycle_attach": ("cycle_open_head.",),
    "ring_system_delete": ("ring_system_delete_head.",),
    "ring_system_restate": ("ring_restate_head.",),
}

_LOCAL_ADAPTER_PREFIXES_BY_FAMILY = {
    "bond_reorder": ("pair_project.",),
    "cycle_insert": ("pair_project.",),
    "cycle_attach": ("pair_project.",),
    "ring_system_restate": (
        "pair_project.",
        "restate_order_embedding.",
    ),
}

MICRO_OVERFIT_PARAMETER_SCOPES = (
    "heads_only",
    "heads_plus_local_adapter",
    "all",
)
MICRO_OVERFIT_LOCAL_ADAPTER_FAMILIES = tuple(_LOCAL_ADAPTER_PREFIXES_BY_FAMILY)


def require_micro_overfit_scope_applicable(
    families: Iterable[str],
    *,
    scope: str,
) -> tuple[str, ...]:
    """Require that a named scope adds a distinct trainable parameter surface."""

    ordered = tuple(dict.fromkeys(str(family) for family in families))
    if not ordered:
        raise ValueError("micro-overfit scope applicability requires a family")
    unknown = set(ordered) - set(_HEAD_PREFIXES_BY_FAMILY)
    if unknown:
        raise ValueError(f"unknown micro-overfit families: {sorted(unknown)}")
    if scope not in MICRO_OVERFIT_PARAMETER_SCOPES:
        raise ValueError(f"unknown micro-overfit parameter scope: {scope}")
    if scope == "heads_plus_local_adapter" and set(ordered).isdisjoint(
        MICRO_OVERFIT_LOCAL_ADAPTER_FAMILIES
    ):
        raise SuccessorMicroOverfitError(
            "heads_plus_local_adapter changes no parameters beyond heads_only "
            f"for families {list(ordered)!r}"
        )
    return ordered


def configure_micro_overfit_parameters(
    model: FactorizedTraceletRateModel,
    families: Iterable[str],
    *,
    scope: str,
) -> tuple[str, ...]:
    """Select the bounded architecture surface used by the capacity ladder."""

    family_set = set(
        require_micro_overfit_scope_applicable(
            families,
            scope=scope,
        )
    )
    prefixes = ["family_head."]
    for family in sorted(family_set):
        prefixes.extend(_HEAD_PREFIXES_BY_FAMILY[family])
    if scope == "heads_plus_local_adapter":
        for family in sorted(family_set):
            prefixes.extend(_LOCAL_ADAPTER_PREFIXES_BY_FAMILY.get(family, ()))

    selected: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = scope == "all" or name.startswith(tuple(prefixes))
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise SuccessorMicroOverfitError(
            "micro-overfit parameter scope selected nothing"
        )
    return tuple(selected)


def train_successor_micro_panel(
    model: FactorizedTraceletRateModel,
    panel: PreparedSuccessorPanel,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    scope: str = "all",
    seed: int = 0,
    report_points: Iterable[int] = (),
) -> dict[str, Any]:
    """Optimize a fixed development panel and return a forensic report."""

    if steps <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid micro-overfit optimization settings")
    torch.manual_seed(int(seed))
    families = tuple(sorted({row.family_name for row in panel.examples}))
    trainable_names = configure_micro_overfit_parameters(
        model,
        families,
        scope=scope,
    )
    required_component_prefixes = {
        "family_head": ("family_head.",),
        **{family: _HEAD_PREFIXES_BY_FAMILY[family] for family in families},
    }
    if scope == "heads_plus_local_adapter":
        required_component_prefixes.update(
            {
                f"{family}_local_adapter": prefixes
                for family in families
                if (prefixes := _LOCAL_ADAPTER_PREFIXES_BY_FAMILY.get(family))
            }
        )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    device_batch = panel.batch.to(model.device)
    requested_points = {
        int(point) for point in report_points if 0 < int(point) <= steps
    }
    requested_points.update({1, steps})
    history: list[dict[str, float]] = []
    finite_nonzero_gradient_seen = {name: False for name in trainable_names}
    gradient_update_counts = {name: 0 for name in trainable_names}
    component_gradient_update_counts = {
        component: 0 for component in required_component_prefixes
    }
    optimizer_steps_with_nonzero_gradient = 0

    initial = successor_panel_metrics(
        model,
        panel,
        include_per_example=False,
    )
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(
            model,
            device_batch,
            panel.fibers,
        )
        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise SuccessorMicroOverfitError(
                f"non-finite successor identity loss at step {step}"
            )
        loss.backward()
        squared_gradient_norm = 0.0
        step_had_nonzero_gradient = False
        nonzero_gradient_names: set[str] = set()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise SuccessorMicroOverfitError(
                    f"non-finite gradient for {name!r} at step {step}"
                )
            gradient_norm = float(parameter.grad.norm())
            squared_gradient_norm += gradient_norm * gradient_norm
            if gradient_norm > 0.0:
                finite_nonzero_gradient_seen[name] = True
                gradient_update_counts[name] += 1
                nonzero_gradient_names.add(name)
                step_had_nonzero_gradient = True
        if step_had_nonzero_gradient:
            optimizer_steps_with_nonzero_gradient += 1
        for component, prefixes in required_component_prefixes.items():
            if any(name.startswith(prefixes) for name in nonzero_gradient_names):
                component_gradient_update_counts[component] += 1
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=10.0)
        optimizer.step()
        if step in requested_points:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "gradient_norm": squared_gradient_norm**0.5,
                }
            )
    final = successor_panel_metrics(
        model,
        panel,
        include_per_example=True,
    )
    never_received_gradient = sorted(
        name for name, seen in finite_nonzero_gradient_seen.items() if not seen
    )
    required_components_without_gradient = sorted(
        component
        for component, count in component_gradient_update_counts.items()
        if count == 0
    )
    return {
        "scope": scope,
        "steps": steps,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "families": list(families),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "trainable_tensor_count": len(trainable_names),
        "never_received_nonzero_gradient": never_received_gradient,
        "gradient_update_counts": gradient_update_counts,
        "component_gradient_update_counts": (component_gradient_update_counts),
        "optimizer_steps_with_nonzero_gradient": (
            optimizer_steps_with_nonzero_gradient
        ),
        "required_components_without_gradient": (required_components_without_gradient),
        "initial": initial,
        "final": final,
        "history": history,
    }


__all__ = [
    "CORE_EDITING_FAMILIES",
    "MICRO_OVERFIT_LOCAL_ADAPTER_FAMILIES",
    "MICRO_OVERFIT_PARAMETER_SCOPES",
    "RINGCORE_EDITING_FAMILIES",
    "PreparedSuccessorPanel",
    "SuccessorMicroOverfitError",
    "SuccessorSupervisionExample",
    "configure_micro_overfit_parameters",
    "examples_from_traces",
    "prepare_cached_successor_panel",
    "prepare_successor_panel",
    "require_micro_overfit_scope_applicable",
    "successor_panel_metrics",
    "train_successor_micro_panel",
]
