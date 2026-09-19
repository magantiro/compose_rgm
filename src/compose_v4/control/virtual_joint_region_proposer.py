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
    StructuralDeltaTemplate,
    materialize_template,
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
            "maximum_primitives",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"virtual joint region {name} must be a positive integer"
                )
        if self.max_depth > 4:
            raise ValueError("virtual joint region depth must be within 1..4")
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
    return tuple(ordered[: budgets.expansion_width])


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
                telemetry[f"invalid_stop_targets_depth_{depth}"] += 1
                if depth == 1:
                    telemetry["standalone_invalid_constituents_retained"] += 1
                continue
            endpoint, target_receipt = target
            planned = _PlannedTarget(prefix, endpoint, target_receipt)
            previous = best_by_endpoint.get(planned.endpoint_key)
            if previous is not None:
                telemetry["canonical_target_aliases"] += 1
                if _target_rank_key(previous) <= _target_rank_key(planned):
                    continue
            best_by_endpoint[planned.endpoint_key] = planned
            if len(best_by_endpoint) >= budgets.max_targets:
                telemetry["target_budget_exhausted"] = 1
                stop = True
                break
        telemetry[f"valid_stop_targets_depth_{depth}"] = sum(
            len(row.prefix.constituents) == depth for row in best_by_endpoint.values()
        )
        if stop or not frontier:
            break
    telemetry["valid_joint_endpoints_recovered"] = sum(
        len(row.prefix.constituents) >= 2 for row in best_by_endpoint.values()
    )
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
        receipt = realize_target(
            source,
            target.endpoint,
            config=RealizerConfig(
                maximum_primitives=budgets.maximum_primitives,
                maximum_expansions=remaining,
            ),
        )
        expanded = int(receipt.get("expanded", 0))
        if not 0 <= expanded <= remaining:
            raise RuntimeError("joint realizer exceeded its global expansion budget")
        telemetry["realizer_expansions"] += expanded
        telemetry["realizer_action_attempts"] += int(receipt.get("attempted", 0))
        if telemetry["realizer_expansions"] >= budgets.max_realization_expansions:
            telemetry["realization_expansion_budget_exhausted"] = 1
        status = str(receipt["status"])
        telemetry[f"realization_status:{status}"] += 1
        if status != "realized":
            continue
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
) -> VirtualJointRegionProposalBatch:
    """Plan and exactly execute valid joint targets from generic same-root deltas.

    This function has no task, cell, route, endpoint, evaluator, or teacher
    action input.  Invalid planning prefixes remain private and can only affect
    which complete joint targets are attempted within the declared budgets.
    """

    budgets = VirtualJointRegionBudgets() if budgets is None else budgets
    if not isinstance(budgets, VirtualJointRegionBudgets):
        raise TypeError("budgets must be VirtualJointRegionBudgets")
    if ranker is not None:
        raise ValueError("virtual joint region ranker support is not implemented")
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
    constituents = _bound_constituents(source, expert, budgets, telemetry)
    targets = _plan_targets(source, expert, constituents, budgets, telemetry)
    proposals = _realize_targets(source, targets, budgets, telemetry)
    result_telemetry = {
        **dict(sorted(telemetry.items())),
        "max_depth": budgets.max_depth,
        "max_binding_visits": budgets.max_binding_visits,
        "max_planning_expansions": budgets.max_planning_expansions,
        "max_targets": budgets.max_targets,
        "max_realization_attempts": budgets.max_realization_attempts,
        "max_realization_expansions": budgets.max_realization_expansions,
        "maximum_primitives": budgets.maximum_primitives,
        "unique_valid_targets": len(targets),
        "unique_committed_endpoints": len(proposals),
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
