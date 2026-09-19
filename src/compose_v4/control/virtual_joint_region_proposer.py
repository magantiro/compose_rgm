"""Deferred-validity joint complete-region proposals.

The frozen structural-goal proposer validates every bound region in isolation.
That loses jointly valid goals whose constituent targets temporarily violate
valence.  This additive proposer keeps same-root bound constituents as private
planning objects, combines one to four of them, and validates only a complete
STOP target.  A valid target is executed once from the original source through
the unchanged protected complete-region runtime.

Planning prefixes are never committed, evaluated, or exposed as molecular
successors.  Runtime inputs are only the source graph and the generic,
address-free route expert.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph, is_rdkit_valid
from compose_v4.chem.state import is_connected_or_null
from compose_v4.control.complete_region_program import program_from_structural_goal
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_policy import (
    REWRITE_SCALE_BANDS,
    StructuralDeltaTemplate,
    materialize_template,
    structural_rewrite_event_count,
    transfer_bindings,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_target,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

_SLOT_CAPACITY = 48
_MAXIMUM_ACTIVE_ATOMS = 40
_FLAT_ALLOCATION = "flat"
_TEMPLATE_BINDING_PARTICLE_ALLOCATION = "template_binding_particle"
_INCREMENTAL_BEAM_PLANNER = "incremental_beam"
_COMPLETE_COMBINATION_PARTICLE_PLANNER = "complete_combination_particle"


@dataclass(frozen=True)
class VirtualJointRegionBudgets:
    """Global deterministic work limits for one joint proposal call."""

    max_depth: int = 4
    beam_width: int = 64
    expansion_width: int = 48
    max_bindings_per_template: int = 8
    max_binding_visits: int = 16_384
    max_planning_expansions: int = 4_096
    max_targets: int = 128
    max_realization_attempts: int = 128
    max_realization_expansions: int = 65_536
    maximum_expansions_per_realization: int = 4_000
    maximum_primitives: int = 32

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "beam_width",
            "expansion_width",
            "max_bindings_per_template",
            "max_binding_visits",
            "max_planning_expansions",
            "max_targets",
            "max_realization_attempts",
            "max_realization_expansions",
            "maximum_expansions_per_realization",
            "maximum_primitives",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"virtual joint region {name} must be a positive integer"
                )
        if self.max_depth > 4:
            raise ValueError("virtual joint region depth must be within 1..4")
        if self.max_targets < self.max_depth:
            raise ValueError(
                "virtual joint region max_targets must cover every requested depth"
            )
        if self.maximum_primitives > 32:
            raise ValueError(
                "virtual joint region programs are limited to 32 primitives"
            )


@dataclass(frozen=True)
class VirtualCreatedRoleSlot:
    """Logical created-role identity and its private target-graph slot."""

    region_index: int
    output_index: int
    virtual_slot: int

    def __post_init__(self) -> None:
        if self.region_index < 0 or self.output_index < 0:
            raise ValueError("virtual created-role indices must be nonnegative")
        if not 0 <= self.virtual_slot < _SLOT_CAPACITY:
            raise ValueError("virtual created-role slot is outside persistent support")


@dataclass(frozen=True)
class VirtualJointRegionStep:
    """One same-root bound constituent and its exact planning provenance."""

    template: StructuralDeltaTemplate
    binding: tuple[int, ...]
    patch: StructuralSubgoal
    constituent_key: str
    created_roles: tuple[VirtualCreatedRoleSlot, ...]

    @property
    def template_id(self) -> str:
        return self.template.template_id


@dataclass(frozen=True)
class VirtualJointRegionProposal:
    """One exact, protected realization of a valid complete joint target."""

    endpoint: MolecularGraph
    steps: tuple[VirtualJointRegionStep, ...]
    log_probability: float
    program_id: str
    actions: tuple[dict[str, Any], ...]
    primitive_count: int
    compiler_strategy: str | None
    expanded: int
    attempted: int

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def templates(self) -> tuple[StructuralDeltaTemplate, ...]:
        return tuple(step.template for step in self.steps)

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(self.endpoint)

    @property
    def created_roles(self) -> tuple[VirtualCreatedRoleSlot, ...]:
        return tuple(role for step in self.steps for role in step.created_roles)


@dataclass(frozen=True)
class VirtualJointRegionProposalBatch:
    """Committed proposals and complete bounded-work telemetry."""

    proposals: tuple[VirtualJointRegionProposal, ...]
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class _BoundConstituent:
    template: StructuralDeltaTemplate
    binding: tuple[int, ...]
    patch: StructuralSubgoal
    constituent_key: str


@dataclass(frozen=True)
class _PlanningPrefix:
    constituents: tuple[_BoundConstituent, ...]
    log_probability: float

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(row.constituent_key for row in self.constituents)


@dataclass(frozen=True)
class _PlannedTarget:
    prefix: _PlanningPrefix
    endpoint: MolecularGraph
    target_receipt: dict[str, Any]

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(self.endpoint)


def _prefix_rank_key(prefix: _PlanningPrefix) -> tuple:
    return (-prefix.log_probability, prefix.keys)


def _target_rank_key(target: _PlannedTarget) -> tuple:
    return (-target.prefix.log_probability, target.endpoint_key, target.prefix.keys)


def _proposal_rank_key(proposal: VirtualJointRegionProposal) -> tuple:
    return (
        -proposal.log_probability,
        proposal.endpoint_key,
        tuple(step.constituent_key for step in proposal.steps),
    )


def _constituent_key(
    template: StructuralDeltaTemplate,
    binding: tuple[int, ...],
    patch: StructuralSubgoal,
) -> str:
    return identity(
        {
            "schema_version": "virtual_joint_bound_constituent_v1",
            "template_id": template.template_id,
            "binding": list(binding),
            "patch_id": patch.subgoal_id,
        }
    )


def _bound_constituents(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    budgets: VirtualJointRegionBudgets,
    telemetry: Counter,
    *,
    constituent_allocation: str,
    binding_particle_index: int | None,
) -> tuple[_BoundConstituent, ...]:
    """Bind generic templates on the root without isolated target validation."""

    rows: dict[str, _BoundConstituent] = {}
    for template in expert.templates:
        remaining = budgets.max_binding_visits - telemetry["binding_visits"]
        if remaining < 1:
            telemetry["binding_visit_budget_exhausted"] = 1
            break
        census = transfer_bindings(
            template,
            source,
            max_bindings=budgets.max_bindings_per_template,
            max_visits=remaining,
        )
        telemetry["binding_visits"] += census.visits
        telemetry["binding_truncations"] += int(census.truncated)
        telemetry["templates_with_bindings"] += int(bool(census.assignments))
        for binding in census.assignments:
            telemetry["materialization_attempts"] += 1
            try:
                patch = materialize_template(template, source, binding)
            except ValueError:
                telemetry["materialization_rejections"] += 1
                continue
            key = _constituent_key(template, binding, patch)
            if key in rows:
                telemetry["constituent_aliases"] += 1
                continue
            rows[key] = _BoundConstituent(template, binding, patch, key)
    if telemetry["binding_visits"] >= budgets.max_binding_visits:
        telemetry["binding_visit_budget_exhausted"] = 1
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            -expert.marginal.score((row.template,)),
            row.constituent_key,
        ),
    )
    telemetry["bound_constituents"] = len(ordered)
    if constituent_allocation == _FLAT_ALLOCATION:
        selected = tuple(ordered[: budgets.expansion_width])
    else:
        if binding_particle_index is None:
            raise AssertionError("validated binding particle index is missing")
        selected = _template_binding_particle_constituents(
            ordered,
            expert,
            expansion_width=budgets.expansion_width,
            binding_particle_index=binding_particle_index,
            telemetry=telemetry,
        )
    selected_keys = {row.constituent_key for row in selected}
    telemetry["selected_constituents"] = len(selected)
    telemetry["constituents_truncated_by_expansion_width"] = max(
        0, len(ordered) - len(selected)
    )
    telemetry["bound_constituent_ranks"] = [
        {
            "rank": rank,
            "constituent_key": row.constituent_key,
            "template_id": row.template.template_id,
            "binding": list(row.binding),
            "selected_for_expansion": row.constituent_key in selected_keys,
        }
        for rank, row in enumerate(ordered, 1)
    ]
    if constituent_allocation == _TEMPLATE_BINDING_PARTICLE_ALLOCATION:
        selection_ranks = {
            row.constituent_key: rank for rank, row in enumerate(selected, 1)
        }
        within_template_ranks = _within_template_ranks(ordered)
        for rank, row in enumerate(telemetry["bound_constituent_ranks"], 1):
            constituent_key = row["constituent_key"]
            template_id = row["template_id"]
            row.update(
                {
                    "global_rank": rank,
                    "within_template_rank": within_template_ranks[constituent_key],
                    "selection_rank": selection_ranks.get(constituent_key),
                    "scale": _template_scale(
                        next(
                            item.template
                            for item in ordered
                            if item.template.template_id == template_id
                        )
                    ),
                }
            )
    return selected


def _template_scale(template: StructuralDeltaTemplate) -> str:
    events = structural_rewrite_event_count(template)
    if events <= 3:
        return "local"
    if events <= 11:
        return "medium"
    return "large"


def _within_template_ranks(
    ordered: list[_BoundConstituent],
) -> dict[str, int]:
    grouped: dict[str, list[_BoundConstituent]] = {}
    for row in ordered:
        grouped.setdefault(row.template.template_id, []).append(row)
    return {
        row.constituent_key: rank
        for rows in grouped.values()
        for rank, row in enumerate(
            sorted(rows, key=lambda item: item.constituent_key), 1
        )
    }


def _template_binding_particle_constituents(
    ordered: list[_BoundConstituent],
    expert: RouteDistilledGoalExpert,
    *,
    expansion_width: int,
    binding_particle_index: int,
    telemetry: Counter,
) -> tuple[_BoundConstituent, ...]:
    """Allocate one rotated binding per template before repeated bindings."""

    grouped: dict[str, list[_BoundConstituent]] = {}
    templates: dict[str, StructuralDeltaTemplate] = {}
    for row in ordered:
        template_id = row.template.template_id
        grouped.setdefault(template_id, []).append(row)
        templates[template_id] = row.template
    for rows in grouped.values():
        rows.sort(key=lambda row: row.constituent_key)

    buckets: dict[str, list[str]] = {band: [] for band in REWRITE_SCALE_BANDS}
    for template_id, template in templates.items():
        buckets[_template_scale(template)].append(template_id)
    for rows in buckets.values():
        rows.sort(
            key=lambda template_id: (
                -expert.marginal.score((templates[template_id],)),
                template_id,
            )
        )
    group_order: list[str] = []
    positions = {band: 0 for band in REWRITE_SCALE_BANDS}
    while len(group_order) < len(grouped):
        for band in REWRITE_SCALE_BANDS:
            position = positions[band]
            if position >= len(buckets[band]):
                continue
            group_order.append(buckets[band][position])
            positions[band] += 1

    rotated: dict[str, tuple[_BoundConstituent, ...]] = {}
    for template_id in group_order:
        rows = grouped[template_id]
        offset = binding_particle_index % len(rows)
        rotated[template_id] = (*rows[offset:], *rows[:offset])
    selected: list[_BoundConstituent] = []
    round_index = 0
    while len(selected) < expansion_width:
        added = False
        for template_id in group_order:
            rows = rotated[template_id]
            if round_index >= len(rows):
                continue
            selected.append(rows[round_index])
            added = True
            if len(selected) >= expansion_width:
                break
        if not added:
            break
        round_index += 1

    telemetry["constituent_allocation_mode"] = _TEMPLATE_BINDING_PARTICLE_ALLOCATION
    telemetry["binding_particle_index"] = binding_particle_index
    telemetry["bound_template_group_count"] = len(grouped)
    telemetry["scale_census"] = {
        band: {
            "template_groups": len(buckets[band]),
            "bound_constituents": sum(
                len(grouped[template_id]) for template_id in buckets[band]
            ),
            "selected_constituents": sum(
                _template_scale(row.template) == band for row in selected
            ),
        }
        for band in REWRITE_SCALE_BANDS
    }
    return tuple(selected)


def _goal(
    prefix: _PlanningPrefix,
) -> tuple[StructuralGoal, tuple[tuple[int, ...], ...]]:
    return (
        StructuralGoal(tuple(row.patch for row in prefix.constituents)),
        tuple(row.binding for row in prefix.constituents),
    )


def _valid_stop_target(
    source: MolecularGraph,
    prefix: _PlanningPrefix,
) -> tuple[MolecularGraph, dict[str, Any]] | None:
    """Validate a complete STOP target, never an isolated planning constituent."""

    goal, bindings = _goal(prefix)
    try:
        endpoint, receipt = instantiate_goal(source, goal, bindings)
    except ValueError:
        return None
    if (
        endpoint.n_atoms != _SLOT_CAPACITY
        or endpoint.n_real_atoms > _MAXIMUM_ACTIVE_ATOMS
        or not is_rdkit_valid(endpoint)
        or not is_connected_or_null(endpoint)
        or not charge_policy_preserved(source, endpoint)
        or canonical_state_key(endpoint) == canonical_state_key(source)
    ):
        return None
    return endpoint, receipt


def _plan_targets(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    constituents: tuple[_BoundConstituent, ...],
    budgets: VirtualJointRegionBudgets,
    telemetry: Counter,
) -> tuple[_PlannedTarget, ...]:
    """Carry invalid prefixes through the bounded same-root combination beam."""

    frontier = (_PlanningPrefix((), 0.0),)
    best_by_endpoint: dict[str, _PlannedTarget] = {}
    depth_champions: dict[int, _PlannedTarget] = {}
    stop = False
    for depth in range(1, budgets.max_depth + 1):
        next_by_keys: dict[tuple[str, ...], _PlanningPrefix] = {}
        for prefix in frontier:
            last_key = prefix.keys[-1] if prefix.keys else None
            for constituent in constituents:
                if last_key is not None and constituent.constituent_key <= last_key:
                    continue
                if telemetry["planning_expansions"] >= budgets.max_planning_expansions:
                    telemetry["planning_expansion_budget_exhausted"] = 1
                    stop = True
                    break
                telemetry["planning_expansions"] += 1
                combined = (*prefix.constituents, constituent)
                candidate = _PlanningPrefix(
                    combined,
                    expert.marginal.score(tuple(row.template for row in combined)),
                )
                next_by_keys[candidate.keys] = candidate
            if stop:
                break
        next_rows = sorted(next_by_keys.values(), key=_prefix_rank_key)
        frontier = tuple(next_rows[: budgets.beam_width])
        telemetry[f"planning_prefixes_depth_{depth}"] = len(frontier)
        for prefix in frontier:
            telemetry["stop_target_attempts"] += 1
            target = _valid_stop_target(source, prefix)
            if target is None:
                telemetry["final_target_abstentions"] += 1
                telemetry[f"invalid_stop_targets_depth_{depth}"] += 1
                if depth == 1:
                    telemetry["standalone_invalid_constituents_retained"] += 1
                continue
            endpoint, target_receipt = target
            planned = _PlannedTarget(prefix, endpoint, target_receipt)
            telemetry[f"valid_stop_targets_depth_{depth}"] += 1
            champion = depth_champions.get(depth)
            if champion is None or _target_rank_key(planned) < _target_rank_key(
                champion
            ):
                depth_champions[depth] = planned
            previous = best_by_endpoint.get(planned.endpoint_key)
            if previous is not None:
                telemetry["canonical_target_aliases"] += 1
                if _target_rank_key(previous) <= _target_rank_key(planned):
                    continue
            best_by_endpoint[planned.endpoint_key] = planned
            ranked_targets = sorted(best_by_endpoint.values(), key=_target_rank_key)
            if len(ranked_targets) > budgets.max_targets:
                telemetry["target_budget_exhausted"] = 1
                telemetry[f"target_budget_drops_depth_{depth}"] += len(
                    ranked_targets[budgets.max_targets :]
                )
                best_by_endpoint = {
                    row.endpoint_key: row
                    for row in ranked_targets[: budgets.max_targets]
                }
        telemetry[f"target_quota_depth_{depth}"] = 1
        if stop or not frontier:
            break

    selected_by_endpoint: dict[str, _PlannedTarget] = {}
    for depth in sorted(depth_champions):
        planned = depth_champions[depth]
        previous = selected_by_endpoint.get(planned.endpoint_key)
        if previous is None or _target_rank_key(planned) < _target_rank_key(previous):
            selected_by_endpoint[planned.endpoint_key] = planned
        else:
            telemetry["cross_depth_canonical_target_aliases"] += 1
    for planned in sorted(best_by_endpoint.values(), key=_target_rank_key):
        if len(selected_by_endpoint) >= budgets.max_targets:
            break
        selected_by_endpoint.setdefault(planned.endpoint_key, planned)
    best_by_endpoint = selected_by_endpoint
    for depth in range(1, budgets.max_depth + 1):
        telemetry[f"retained_targets_depth_{depth}"] = sum(
            len(row.prefix.constituents) == depth for row in best_by_endpoint.values()
        )
    telemetry["valid_joint_endpoints_recovered"] = sum(
        len(row.prefix.constituents) >= 2 for row in best_by_endpoint.values()
    )
    telemetry["target_depths_with_valid_stops"] = sum(
        telemetry[f"valid_stop_targets_depth_{depth}"] > 0
        for depth in range(1, budgets.max_depth + 1)
    )
    telemetry["target_depths_retained"] = sum(
        telemetry[f"retained_targets_depth_{depth}"] > 0
        for depth in range(1, budgets.max_depth + 1)
    )
    return tuple(sorted(best_by_endpoint.values(), key=_target_rank_key))


def _plan_complete_combination_particle(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    constituents: tuple[_BoundConstituent, ...],
    budgets: VirtualJointRegionBudgets,
    telemetry: Counter,
    *,
    depth: int,
    particle_index: int,
    particle_count: int,
) -> tuple[_PlannedTarget, ...]:
    """Validate one disjoint bounded shard of complete combinations."""

    ordered = tuple(sorted(constituents, key=lambda row: row.constituent_key))
    total_combinations = math.comb(len(ordered), depth)
    shard_start = total_combinations * particle_index // particle_count
    shard_stop = total_combinations * (particle_index + 1) // particle_count
    shard_size = shard_stop - shard_start
    required_particle_count = max(
        1,
        math.ceil(total_combinations / budgets.max_planning_expansions),
    )
    telemetry["combination_planner_mode"] = _COMPLETE_COMBINATION_PARTICLE_PLANNER
    telemetry["combination_particle_depth"] = depth
    telemetry["combination_particle_index"] = particle_index
    telemetry["combination_particle_count"] = particle_count
    telemetry["combination_particle_total_combinations"] = total_combinations
    telemetry["combination_particle_shard_start"] = shard_start
    telemetry["combination_particle_shard_stop"] = shard_stop
    telemetry["combination_particle_assigned_combinations"] = shard_size
    telemetry["combination_particle_required_minimum_count"] = required_particle_count
    telemetry["combination_particle_raw_union_covers_requested_depth"] = int(
        particle_count >= required_particle_count
    )
    telemetry["combination_particle_coverage_requires_all_particles"] = 1
    telemetry["combination_particle_telemetry_sanitized"] = 1
    telemetry["combination_particle_empty_depth_abstention"] = 0
    telemetry["combination_particle_planning_capacity_abstention"] = 0
    telemetry["combination_particle_unvalidated_combinations"] = 0
    telemetry["combination_particle_target_capacity_abstention"] = 0
    telemetry["combination_particle_unretained_valid_targets"] = 0
    telemetry["planning_expansions"] += 0
    telemetry["realization_attempts"] += 0
    telemetry["realizer_expansions"] += 0
    for candidate_depth in range(1, budgets.max_depth + 1):
        telemetry[f"planning_prefixes_depth_{candidate_depth}"] = (
            shard_size if candidate_depth == depth else 0
        )

    if total_combinations == 0:
        telemetry["combination_particle_empty_depth_abstention"] = 1
        for candidate_depth in range(1, budgets.max_depth + 1):
            telemetry[f"retained_targets_depth_{candidate_depth}"] = 0
        telemetry["target_depths_with_valid_stops"] = 0
        telemetry["target_depths_retained"] = 0
        telemetry["valid_joint_endpoints_recovered"] = 0
        return ()
    if shard_size > budgets.max_planning_expansions:
        telemetry["combination_particle_planning_capacity_abstention"] = 1
        telemetry["combination_particle_unvalidated_combinations"] = shard_size
        for candidate_depth in range(1, budgets.max_depth + 1):
            telemetry[f"retained_targets_depth_{candidate_depth}"] = 0
        telemetry["target_depths_with_valid_stops"] = 0
        telemetry["target_depths_retained"] = 0
        telemetry["valid_joint_endpoints_recovered"] = 0
        return ()

    best_by_endpoint: dict[str, _PlannedTarget] = {}
    combinations = itertools.islice(
        itertools.combinations(ordered, depth),
        shard_start,
        shard_stop,
    )
    for combined in combinations:
        telemetry["planning_expansions"] += 1
        prefix = _PlanningPrefix(
            combined,
            expert.marginal.score(tuple(row.template for row in combined)),
        )
        telemetry["stop_target_attempts"] += 1
        target = _valid_stop_target(source, prefix)
        if target is None:
            telemetry["final_target_abstentions"] += 1
            telemetry[f"invalid_stop_targets_depth_{depth}"] += 1
            if depth == 1:
                telemetry["standalone_invalid_constituents_retained"] += 1
            continue
        endpoint, target_receipt = target
        planned = _PlannedTarget(prefix, endpoint, target_receipt)
        telemetry[f"valid_stop_targets_depth_{depth}"] += 1
        previous = best_by_endpoint.get(planned.endpoint_key)
        if previous is not None:
            telemetry["canonical_target_aliases"] += 1
            if _target_rank_key(previous) <= _target_rank_key(planned):
                continue
        best_by_endpoint[planned.endpoint_key] = planned

    unique_target_count = len(best_by_endpoint)
    telemetry["combination_particle_unique_valid_targets"] = unique_target_count
    target_capacity = min(
        budgets.max_targets,
        budgets.max_realization_attempts,
    )
    telemetry["combination_particle_target_capacity"] = target_capacity
    if unique_target_count > target_capacity:
        telemetry["combination_particle_target_capacity_abstention"] = 1
        telemetry["combination_particle_unretained_valid_targets"] = unique_target_count
        best_by_endpoint = {}

    for candidate_depth in range(1, budgets.max_depth + 1):
        telemetry[f"retained_targets_depth_{candidate_depth}"] = (
            len(best_by_endpoint) if candidate_depth == depth else 0
        )
    telemetry["valid_joint_endpoints_recovered"] = sum(
        len(row.prefix.constituents) >= 2 for row in best_by_endpoint.values()
    )
    telemetry["target_depths_with_valid_stops"] = int(unique_target_count > 0)
    telemetry["target_depths_retained"] = int(bool(best_by_endpoint))
    return tuple(sorted(best_by_endpoint.values(), key=_target_rank_key))


def _steps(target: _PlannedTarget) -> tuple[VirtualJointRegionStep, ...]:
    raw_output_slots = target.target_receipt.get("output_slots")
    if not isinstance(raw_output_slots, list) or len(raw_output_slots) != len(
        target.prefix.constituents
    ):
        raise RuntimeError("joint target omitted created-role slot provenance")
    steps = []
    for region_index, (constituent, slots) in enumerate(
        zip(target.prefix.constituents, raw_output_slots, strict=True)
    ):
        if not isinstance(slots, list) or len(slots) != len(
            constituent.patch.output_atoms
        ):
            raise RuntimeError("joint target created-role provenance is malformed")
        created = tuple(
            VirtualCreatedRoleSlot(region_index, output_index, int(slot))
            for output_index, slot in enumerate(slots)
        )
        steps.append(
            VirtualJointRegionStep(
                template=constituent.template,
                binding=constituent.binding,
                patch=constituent.patch,
                constituent_key=constituent.constituent_key,
                created_roles=created,
            )
        )
    return tuple(steps)


def _realize_targets(
    source: MolecularGraph,
    targets: tuple[_PlannedTarget, ...],
    budgets: VirtualJointRegionBudgets,
    telemetry: Counter,
) -> tuple[VirtualJointRegionProposal, ...]:
    proposals: list[VirtualJointRegionProposal] = []
    for target in targets:
        if telemetry["realization_attempts"] >= budgets.max_realization_attempts:
            telemetry["realization_attempt_budget_exhausted"] = 1
            break
        remaining = (
            budgets.max_realization_expansions - telemetry["realizer_expansions"]
        )
        if remaining < 1:
            telemetry["realization_expansion_budget_exhausted"] = 1
            break
        telemetry["realization_attempts"] += 1
        goal, _ = _goal(target.prefix)
        program = program_from_structural_goal(goal)
        candidate_expansion_limit = min(
            budgets.maximum_expansions_per_realization,
            remaining,
        )
        receipt = realize_target(
            source,
            target.endpoint,
            config=RealizerConfig(
                maximum_primitives=budgets.maximum_primitives,
                maximum_expansions=candidate_expansion_limit,
            ),
        )
        expanded = int(receipt.get("expanded", 0))
        if not 0 <= expanded <= candidate_expansion_limit:
            raise RuntimeError(
                "joint realizer exceeded its per-candidate expansion budget"
            )
        telemetry["realizer_expansions"] += expanded
        telemetry["realizer_action_attempts"] += int(receipt.get("attempted", 0))
        if telemetry["realizer_expansions"] >= budgets.max_realization_expansions:
            telemetry["realization_expansion_budget_exhausted"] = 1
        status = str(receipt["status"])
        telemetry[f"realization_status:{status}"] += 1
        if status != "realized":
            telemetry["compiler_abstentions"] += 1
            if expanded >= candidate_expansion_limit:
                telemetry["per_realization_expansion_cap_abstentions"] += 1
            continue
        telemetry["successful_realized_targets"] += 1
        if int(receipt.get("primitive_teacher_actions_used", -1)) != 0:
            raise RuntimeError("joint target realizer used teacher actions")
        states = receipt.get("states")
        if (
            not isinstance(states, list)
            or not states
            or not isinstance(states[-1], dict)
        ):
            raise RuntimeError("joint target realizer omitted its committed endpoint")
        endpoint = decode_state(states[-1])
        if canonical_state_key(endpoint) != target.endpoint_key:
            raise RuntimeError("joint target realizer changed the planned endpoint")
        actions = receipt.get("actions")
        primitive_count = len(actions) if isinstance(actions, list) else -1
        if (
            not isinstance(actions, list)
            or primitive_count != len(actions)
            or not 0 <= primitive_count <= budgets.maximum_primitives
        ):
            raise RuntimeError("joint target realizer omitted a bounded exact program")
        telemetry["realized_primitives"] += primitive_count
        proposals.append(
            VirtualJointRegionProposal(
                endpoint=endpoint,
                steps=_steps(target),
                log_probability=target.prefix.log_probability,
                program_id=program.program_id,
                actions=tuple(actions),
                primitive_count=primitive_count,
                compiler_strategy=receipt.get("compiler_strategy"),
                expanded=expanded,
                attempted=int(receipt.get("attempted", 0)),
            )
        )
        if telemetry["realization_expansion_budget_exhausted"]:
            break
    return tuple(sorted(proposals, key=_proposal_rank_key))


def propose_virtual_joint_region_paths(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    *,
    budgets: VirtualJointRegionBudgets | None = None,
    ranker: object | None = None,
    constituent_allocation: str = _FLAT_ALLOCATION,
    binding_particle_index: int | None = None,
    combination_planner: str = _INCREMENTAL_BEAM_PLANNER,
    combination_particle_depth: int | None = None,
    combination_particle_index: int | None = None,
    combination_particle_count: int | None = None,
) -> VirtualJointRegionProposalBatch:
    """Plan and exactly execute valid joint targets from generic same-root deltas.

    This function has no task, cell, route, endpoint, evaluator, or teacher
    action input.  Invalid planning prefixes remain private and can only affect
    which complete joint targets are attempted within the declared budgets.
    The opt-in complete-combination mode covers one requested depth only when
    every particle in its declared fixed particle count is run and unioned.
    """

    budgets = VirtualJointRegionBudgets() if budgets is None else budgets
    if not isinstance(budgets, VirtualJointRegionBudgets):
        raise TypeError("budgets must be VirtualJointRegionBudgets")
    if ranker is not None:
        raise ValueError("virtual joint region ranker support is not implemented")
    if constituent_allocation not in {
        _FLAT_ALLOCATION,
        _TEMPLATE_BINDING_PARTICLE_ALLOCATION,
    }:
        raise ValueError(
            "constituent_allocation must be 'flat' or " "'template_binding_particle'"
        )
    if constituent_allocation == _FLAT_ALLOCATION:
        if binding_particle_index is not None:
            raise ValueError(
                "binding_particle_index is only valid for "
                "template_binding_particle allocation"
            )
    elif (
        type(binding_particle_index) is not int
        or not 0 <= binding_particle_index < budgets.max_bindings_per_template
    ):
        raise ValueError(
            "template_binding_particle allocation requires an integer "
            "binding_particle_index within 0..max_bindings_per_template-1"
        )
    particle_arguments = (
        combination_particle_depth,
        combination_particle_index,
        combination_particle_count,
    )
    if combination_planner == _INCREMENTAL_BEAM_PLANNER:
        if any(value is not None for value in particle_arguments):
            raise ValueError(
                "combination particle arguments are only valid for "
                "complete_combination_particle planning"
            )
    elif combination_planner == _COMPLETE_COMBINATION_PARTICLE_PLANNER:
        if (
            type(combination_particle_depth) is not int
            or not 1 <= combination_particle_depth <= budgets.max_depth
        ):
            raise ValueError(
                "complete_combination_particle planning requires an integer "
                "combination_particle_depth within 1..max_depth"
            )
        if (
            type(combination_particle_count) is not int
            or combination_particle_count < 1
        ):
            raise ValueError(
                "complete_combination_particle planning requires a positive integer "
                "combination_particle_count"
            )
        if (
            type(combination_particle_index) is not int
            or not 0 <= combination_particle_index < combination_particle_count
        ):
            raise ValueError(
                "complete_combination_particle planning requires an integer "
                "combination_particle_index within 0..combination_particle_count-1"
            )
    else:
        raise ValueError(
            "combination_planner must be 'incremental_beam' or "
            "'complete_combination_particle'"
        )
    if source.n_atoms != _SLOT_CAPACITY:
        raise ValueError("virtual joint region source must have exactly 48 slots")
    if (
        source.n_real_atoms > _MAXIMUM_ACTIVE_ATOMS
        or not is_rdkit_valid(source)
        or not is_connected_or_null(source)
    ):
        raise ValueError("virtual joint region source is outside molecular support")
    expected = set(expert.marginal.template_ids)
    if {row.template_id for row in expert.templates} != expected:
        raise ValueError("runtime template vocabulary and marginal disagree")

    telemetry: Counter = Counter()
    constituents = _bound_constituents(
        source,
        expert,
        budgets,
        telemetry,
        constituent_allocation=constituent_allocation,
        binding_particle_index=binding_particle_index,
    )
    if combination_planner == _INCREMENTAL_BEAM_PLANNER:
        targets = _plan_targets(source, expert, constituents, budgets, telemetry)
    else:
        for row in telemetry["bound_constituent_ranks"]:
            row.pop("binding", None)
        if (
            combination_particle_depth is None
            or combination_particle_index is None
            or combination_particle_count is None
        ):
            raise AssertionError("validated combination particle arguments are missing")
        targets = _plan_complete_combination_particle(
            source,
            expert,
            constituents,
            budgets,
            telemetry,
            depth=combination_particle_depth,
            particle_index=combination_particle_index,
            particle_count=combination_particle_count,
        )
    proposals = _realize_targets(source, targets, budgets, telemetry)
    if combination_planner == _COMPLETE_COMBINATION_PARTICLE_PLANNER:
        telemetry["combination_particle_realization_coverage_numerator"] = telemetry[
            "realization_attempts"
        ]
        telemetry["combination_particle_realization_coverage_denominator"] = len(
            targets
        )
        telemetry["combination_particle_unattempted_targets"] = max(
            0,
            len(targets) - telemetry["realization_attempts"],
        )
    result_telemetry = {
        **dict(sorted(telemetry.items())),
        "max_depth": budgets.max_depth,
        "max_binding_visits": budgets.max_binding_visits,
        "max_planning_expansions": budgets.max_planning_expansions,
        "max_targets": budgets.max_targets,
        "max_realization_attempts": budgets.max_realization_attempts,
        "max_realization_expansions": budgets.max_realization_expansions,
        "maximum_expansions_per_realization": (
            budgets.maximum_expansions_per_realization
        ),
        "maximum_primitives": budgets.maximum_primitives,
        "unique_valid_targets": len(targets),
        "unique_committed_endpoints": len(proposals),
        "exact_realization_precision_numerator": len(proposals),
        "exact_realization_precision_denominator": telemetry[
            "successful_realized_targets"
        ],
        "final_target_abstentions": telemetry["final_target_abstentions"],
        "compiler_abstentions": telemetry["compiler_abstentions"],
        "virtual_prefix_commits": 0,
        "partial_endpoint_evaluations": 0,
        "task_cell_route_or_endpoint_input_used": False,
        "primitive_teacher_actions_used": 0,
    }
    return VirtualJointRegionProposalBatch(proposals, result_telemetry)


__all__ = [
    "VirtualCreatedRoleSlot",
    "VirtualJointRegionBudgets",
    "VirtualJointRegionProposal",
    "VirtualJointRegionProposalBatch",
    "VirtualJointRegionStep",
    "propose_virtual_joint_region_paths",
]
