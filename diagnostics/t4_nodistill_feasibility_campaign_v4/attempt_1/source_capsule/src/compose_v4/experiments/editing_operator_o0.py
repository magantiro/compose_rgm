"""Bounded, checkpoint-independent Gate O0 for editing operator decisions.

Gate O0 is structural.  It compares declared executable supports on frozen
E3/E4/E7 tasks before any architecture change or long training run.  It cannot
establish learned probability, calibration, or search efficiency.

Every negative result is scoped to an event horizon.  If any wall-time, state,
or generated-mark limit is reached, the result is ``incomplete_resource_bound``
rather than an impossibility claim.

Canonical molecular identity remains the scientific quotient, while bounded
search also retains the exact persistent-slot digest.  This coordinate
refinement is required because later legal masks and protected-slot path
constraints can distinguish two representations of the same molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
import time
from typing import Callable, Hashable, Iterable, Literal

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.rewrite.kernel import RewriteSystem, canonical_state_key
from compose_v4.rewrite.tracelets import RingSystemGrow
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog

from .editing_operator_support import (
    OperatorBasis,
    OperatorSupportError,
    SupportLimits,
    SupportedAction,
    enumerate_production_editing_support,
)


O0_CONTRACT_SCHEMA = "compose.editing_operator_o0"
O0_CONTRACT_VERSION = 1
REQUIRED_EXPERIMENTS = ("E3", "E4", "E7")
_ALLOWED_STATUSES = (
    "reached",
    "reached_with_unsupported_lowering",
    "not_reached_within_complete_event_bound",
    "incomplete_resource_bound",
    "primitive_cost_undefined",
)

ConstraintTransition = Callable[
    [Hashable | None, MolecularGraph, SupportedAction],
    tuple[bool, Hashable | None],
]
TargetPredicate = Callable[[MolecularGraph, Hashable | None], bool]
GrowActionProvider = Callable[
    [MolecularGraph],
    Iterable[RingSystemGrow],
]


class EditingOperatorO0Error(RuntimeError):
    """The frozen O0 contract or one bounded search is invalid."""


@dataclass(frozen=True)
class SearchLimits:
    max_expanded_labels: int = 10_000
    max_generated_marks: int = 1_000_000
    max_wall_seconds: float = 60.0
    support_limits: SupportLimits = SupportLimits()

    def __post_init__(self) -> None:
        if self.max_expanded_labels <= 0:
            raise ValueError("max_expanded_labels must be positive")
        if self.max_generated_marks <= 0:
            raise ValueError("max_generated_marks must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")


@dataclass(frozen=True)
class OperatorTask:
    """One resolved, exact-state O0 task row.

    The frozen task artifact owns source/target state references, split and
    stratum metadata.  This runtime object owns the resolved persistent-slot
    state and optional path-dependent hard constraint.
    """

    task_id: str
    experiment: Literal["E3", "E4", "E7"]
    task_kind: str
    source: MolecularGraph
    max_events: int
    target_key: str | None = None
    target_predicate: TargetPredicate | None = None
    constraint_id: str = "none"
    initial_constraint_state: Hashable | None = None
    constraint_transition: ConstraintTransition | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.experiment not in REQUIRED_EXPERIMENTS:
            raise ValueError(f"unsupported experiment {self.experiment!r}")
        if not self.task_kind.strip():
            raise ValueError("task_kind must be non-empty")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if (self.target_key is None) == (self.target_predicate is None):
            raise ValueError("provide exactly one of target_key or target_predicate")
        if self.constraint_transition is not None and self.constraint_id == "none":
            raise ValueError("path constraint requires a non-default constraint_id")
        try:
            hash(self.initial_constraint_state)
        except TypeError as exc:
            raise ValueError("initial constraint state must be hashable") from exc


@dataclass(frozen=True)
class CertifiedPath:
    objective: Literal["events_then_primitive", "primitive_then_events"]
    event_cost: int
    primitive_equivalent_cost: int | None
    contains_unsupported_lowering: bool
    family_names: tuple[str, ...]
    successor_keys: tuple[str, ...]
    successor_state_sha256s: tuple[str, ...]
    actions: tuple[SupportedAction, ...]

    def __post_init__(self) -> None:
        if self.contains_unsupported_lowering != (self.primitive_equivalent_cost is None):
            raise ValueError("unsupported path must have undefined primitive cost")
        if len(self.successor_keys) != len(self.actions):
            raise ValueError("path successor keys do not align with actions")
        if len(self.successor_state_sha256s) != len(self.actions):
            raise ValueError("path persistent identities do not align with actions")


@dataclass(frozen=True)
class ObjectiveSearchResult:
    objective: Literal["events_then_primitive", "primitive_then_events"]
    status: str
    path: CertifiedPath | None
    expanded_labels: int
    generated_marks: int
    elapsed_seconds: float
    exhausted_complete_event_bound: bool
    incomplete_reason: str | None
    unsupported_lowering_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(f"unknown search status {self.status!r}")
        if self.status == "primitive_cost_undefined":
            if self.path is not None or not self.unsupported_lowering_families:
                raise ValueError("undefined primitive cost requires explicit unsupported families")
        elif self.unsupported_lowering_families:
            raise ValueError("unsupported families belong only to primitive_cost_undefined")
        if self.status == "reached_with_unsupported_lowering":
            if self.path is None or not self.path.contains_unsupported_lowering:
                raise ValueError("unsupported reachability requires an explicit unsupported path")


@dataclass(frozen=True)
class BasisReachabilityResult:
    task_id: str
    basis_id: str
    max_events: int
    events_then_primitive: ObjectiveSearchResult
    primitive_then_events: ObjectiveSearchResult

    @property
    def reached(self) -> bool:
        reached_statuses = {"reached", "reached_with_unsupported_lowering"}
        return (
            self.events_then_primitive.status in reached_statuses
            or self.primitive_then_events.status in reached_statuses
        )

    @property
    def incomplete(self) -> bool:
        return (
            self.events_then_primitive.status == "incomplete_resource_bound"
            or self.primitive_then_events.status == "incomplete_resource_bound"
        )

    @property
    def primitive_cost_undefined(self) -> bool:
        return self.primitive_then_events.status == "primitive_cost_undefined"


@dataclass
class _Label:
    state: MolecularGraph
    constraint_state: Hashable | None
    event_cost: int
    primitive_cost: int | None
    actions: tuple[SupportedAction, ...]
    successor_keys: tuple[str, ...]
    successor_state_sha256s: tuple[str, ...]


def _target_reached(
    task: OperatorTask,
    state: MolecularGraph,
    constraint_state: Hashable | None,
) -> bool:
    if task.target_predicate is not None:
        return bool(task.target_predicate(state, constraint_state))
    return canonical_state_key(state) == task.target_key


def _priority(
    label: _Label,
    objective: Literal["events_then_primitive", "primitive_then_events"],
) -> tuple[int, int, int]:
    if objective == "events_then_primitive":
        return (
            label.event_cost,
            int(label.primitive_cost is None),
            0 if label.primitive_cost is None else label.primitive_cost,
        )
    if label.primitive_cost is None:
        raise EditingOperatorO0Error("undefined primitive cost entered primitive-ordered search")
    return label.primitive_cost, label.event_cost, 0


def _augmented_key(
    state: MolecularGraph,
    constraint_state: Hashable | None,
    event_cost: int,
) -> tuple[str, str, Hashable | None, int]:
    # Event count is part of the bounded-search identity.  A lower-primitive
    # path that has consumed more events does not dominate a costlier path with
    # enough remaining budget to reach the target.  The persistent digest is
    # retained alongside canonical molecular identity because future
    # slot-addressed legal masks and constraints can differ across aliases.
    return (
        canonical_state_key(state),
        persistent_slot_state_sha256(state),
        constraint_state,
        int(event_cost),
    )


def _search_one_objective(
    task: OperatorTask,
    basis: OperatorBasis,
    objective: Literal["events_then_primitive", "primitive_then_events"],
    *,
    ring_catalog: TypedRingCatalog | None,
    grow_action_provider: GrowActionProvider | None,
    system: RewriteSystem | None,
    limits: SearchLimits,
) -> ObjectiveSearchResult:
    start = time.monotonic()
    serial = 0
    initial = _Label(
        state=task.source,
        constraint_state=task.initial_constraint_state,
        event_cost=0,
        primitive_cost=0,
        actions=(),
        successor_keys=(),
        successor_state_sha256s=(),
    )
    heap: list[tuple[int, int, int, int, _Label]] = [
        (*_priority(initial, objective), serial, initial)
    ]
    # Dijkstra needs only the best lexicographic objective at an augmented
    # state.  All costs are positive, so a popped target is certified optimal
    # for the selected objective even when the full horizon is not exhausted.
    best: dict[
        tuple[str, str, Hashable | None, int],
        tuple[int, int, int],
    ] = {
        _augmented_key(task.source, task.initial_constraint_state, 0): _priority(initial, objective)
    }
    expanded = 0
    generated = 0

    while heap:
        elapsed = time.monotonic() - start
        if elapsed > limits.max_wall_seconds:
            return ObjectiveSearchResult(
                objective=objective,
                status="incomplete_resource_bound",
                path=None,
                expanded_labels=expanded,
                generated_marks=generated,
                elapsed_seconds=elapsed,
                exhausted_complete_event_bound=False,
                incomplete_reason="wall_time_limit",
            )
        _, _, _, _, current = heapq.heappop(heap)
        identity = _augmented_key(
            current.state,
            current.constraint_state,
            current.event_cost,
        )
        if best.get(identity) != _priority(current, objective):
            continue
        if _target_reached(task, current.state, current.constraint_state):
            contains_unsupported = current.primitive_cost is None
            return ObjectiveSearchResult(
                objective=objective,
                status=("reached_with_unsupported_lowering" if contains_unsupported else "reached"),
                path=CertifiedPath(
                    objective=objective,
                    event_cost=current.event_cost,
                    primitive_equivalent_cost=current.primitive_cost,
                    contains_unsupported_lowering=contains_unsupported,
                    family_names=tuple(action.family_name for action in current.actions),
                    successor_keys=current.successor_keys,
                    successor_state_sha256s=current.successor_state_sha256s,
                    actions=current.actions,
                ),
                expanded_labels=expanded,
                generated_marks=generated,
                elapsed_seconds=time.monotonic() - start,
                exhausted_complete_event_bound=False,
                incomplete_reason=None,
            )
        if current.event_cost >= task.max_events:
            continue
        if expanded >= limits.max_expanded_labels:
            return ObjectiveSearchResult(
                objective=objective,
                status="incomplete_resource_bound",
                path=None,
                expanded_labels=expanded,
                generated_marks=generated,
                elapsed_seconds=time.monotonic() - start,
                exhausted_complete_event_bound=False,
                incomplete_reason="expanded_label_limit",
            )
        expanded += 1
        grow_actions = (
            tuple(grow_action_provider(current.state))
            if basis.ring_system_grow and grow_action_provider is not None
            else None
        )
        try:
            support = enumerate_production_editing_support(
                current.state,
                basis,
                ring_catalog=ring_catalog,
                ring_system_grow_actions=grow_actions,
                system=system,
                limits=limits.support_limits,
            )
        except OperatorSupportError as exc:
            return ObjectiveSearchResult(
                objective=objective,
                status="incomplete_resource_bound",
                path=None,
                expanded_labels=expanded,
                generated_marks=generated,
                elapsed_seconds=time.monotonic() - start,
                exhausted_complete_event_bound=False,
                incomplete_reason=f"support_enumeration:{exc}",
            )

        # Apply hard/path constraints before quotienting.  Equivalent marks can
        # update path memory differently, and canonical aliases can expose
        # different future slot-addressed masks, so both constraint state and
        # persistent digest remain in the grouping key.
        proposals: dict[
            tuple[str, str, Hashable | None],
            SupportedAction,
        ] = {}
        for action in support.marks:
            generated += 1
            if generated > limits.max_generated_marks:
                return ObjectiveSearchResult(
                    objective=objective,
                    status="incomplete_resource_bound",
                    path=None,
                    expanded_labels=expanded,
                    generated_marks=generated,
                    elapsed_seconds=time.monotonic() - start,
                    exhausted_complete_event_bound=False,
                    incomplete_reason="generated_mark_limit",
                )
            next_constraint_state = current.constraint_state
            if task.constraint_transition is not None:
                allowed, next_constraint_state = task.constraint_transition(
                    current.constraint_state,
                    current.state,
                    action,
                )
                if not allowed:
                    continue
                try:
                    hash(next_constraint_state)
                except TypeError as exc:
                    raise EditingOperatorO0Error(
                        "constraint transition returned an unhashable state"
                    ) from exc
            proposal_key = (
                action.successor_key,
                action.successor_state_sha256,
                next_constraint_state,
            )
            previous = proposals.get(proposal_key)
            if previous is None or (
                int(action.primitive_equivalent_cost is None),
                (
                    0
                    if action.primitive_equivalent_cost is None
                    else action.primitive_equivalent_cost
                ),
                action.family_name,
                repr(action.action),
            ) < (
                int(previous.primitive_equivalent_cost is None),
                (
                    0
                    if previous.primitive_equivalent_cost is None
                    else previous.primitive_equivalent_cost
                ),
                previous.family_name,
                repr(previous.action),
            ):
                proposals[proposal_key] = action

        unsupported_families = tuple(
            sorted(
                {
                    action.family_name
                    for action in proposals.values()
                    if action.primitive_equivalent_cost is None
                }
            )
        )
        if objective == "primitive_then_events" and unsupported_families:
            return ObjectiveSearchResult(
                objective=objective,
                status="primitive_cost_undefined",
                path=None,
                expanded_labels=expanded,
                generated_marks=generated,
                elapsed_seconds=time.monotonic() - start,
                exhausted_complete_event_bound=False,
                incomplete_reason=None,
                unsupported_lowering_families=unsupported_families,
            )

        for proposal_key, action in sorted(
            proposals.items(),
            key=lambda item: (item[0][0], item[0][1], repr(item[0][2])),
        ):
            next_events = current.event_cost + 1
            next_primitive = (
                None
                if current.primitive_cost is None or action.primitive_equivalent_cost is None
                else current.primitive_cost + action.primitive_equivalent_cost
            )
            candidate = _Label(
                state=action.successor,
                constraint_state=proposal_key[2],
                event_cost=next_events,
                primitive_cost=next_primitive,
                actions=(*current.actions, action),
                successor_keys=(*current.successor_keys, action.successor_key),
                successor_state_sha256s=(
                    *current.successor_state_sha256s,
                    action.successor_state_sha256,
                ),
            )
            candidate_priority = _priority(candidate, objective)
            augmented_proposal_key = (
                proposal_key[0],
                proposal_key[1],
                proposal_key[2],
                next_events,
            )
            previous_priority = best.get(augmented_proposal_key)
            if previous_priority is not None and previous_priority <= candidate_priority:
                continue
            best[augmented_proposal_key] = candidate_priority
            serial += 1
            heapq.heappush(
                heap,
                (*candidate_priority, serial, candidate),
            )

    return ObjectiveSearchResult(
        objective=objective,
        status="not_reached_within_complete_event_bound",
        path=None,
        expanded_labels=expanded,
        generated_marks=generated,
        elapsed_seconds=time.monotonic() - start,
        exhausted_complete_event_bound=True,
        incomplete_reason=None,
    )


def evaluate_basis_reachability(
    task: OperatorTask,
    basis: OperatorBasis,
    *,
    ring_catalog: TypedRingCatalog | None = None,
    grow_action_provider: GrowActionProvider | None = None,
    system: RewriteSystem | None = None,
    limits: SearchLimits | None = None,
) -> BasisReachabilityResult:
    """Compute two separately certified shortest paths for one task/basis."""

    search_limits = limits or SearchLimits()
    if basis.ring_system_delete and ring_catalog is None:
        raise EditingOperatorO0Error(
            "ring_system_delete basis requires an explicit typed ring catalog"
        )
    if basis.ring_system_grow and grow_action_provider is None:
        raise EditingOperatorO0Error(
            "ring_system_grow basis requires a complete audited action provider"
        )
    first = _search_one_objective(
        task,
        basis,
        "events_then_primitive",
        ring_catalog=ring_catalog,
        grow_action_provider=grow_action_provider,
        system=system,
        limits=search_limits,
    )
    second = _search_one_objective(
        task,
        basis,
        "primitive_then_events",
        ring_catalog=ring_catalog,
        grow_action_provider=grow_action_provider,
        system=system,
        limits=search_limits,
    )
    return BasisReachabilityResult(
        task_id=task.task_id,
        basis_id=basis.basis_id,
        max_events=task.max_events,
        events_then_primitive=first,
        primitive_then_events=second,
    )


def compare_operator_bases(
    task: OperatorTask,
    bases: Iterable[OperatorBasis],
    *,
    ring_catalog: TypedRingCatalog | None = None,
    grow_action_provider: GrowActionProvider | None = None,
    system: RewriteSystem | None = None,
    limits: SearchLimits | None = None,
) -> tuple[BasisReachabilityResult, ...]:
    declared = tuple(bases)
    if not declared:
        raise ValueError("at least one operator basis is required")
    ids = tuple(item.basis_id for item in declared)
    if len(set(ids)) != len(ids):
        raise ValueError("operator basis ids must be unique")
    return tuple(
        evaluate_basis_reachability(
            task,
            basis,
            ring_catalog=ring_catalog,
            grow_action_provider=grow_action_provider,
            system=system,
            limits=limits,
        )
        for basis in declared
    )


def validate_o0_contract(payload: dict) -> None:
    """Validate the frozen method contract without inventing pass thresholds."""

    if payload.get("schema") != O0_CONTRACT_SCHEMA:
        raise EditingOperatorO0Error("unexpected O0 contract schema")
    if payload.get("version") != O0_CONTRACT_VERSION:
        raise EditingOperatorO0Error("unexpected O0 contract version")
    if tuple(payload.get("required_experiments", ())) != REQUIRED_EXPERIMENTS:
        raise EditingOperatorO0Error("O0 must cover E3, E4, and E7")
    if payload.get("checkpoint_dependency") != "none":
        raise EditingOperatorO0Error("Gate O0 must remain checkpoint-independent")
    if payload.get("production_mutation_authorized") is not False:
        raise EditingOperatorO0Error("Gate O0 cannot mutate production operators")
    if "decision_thresholds" in payload:
        raise EditingOperatorO0Error("O0 structural tooling must not invent decision thresholds")
    bases = payload.get("operator_bases")
    if not isinstance(bases, list) or not bases:
        raise EditingOperatorO0Error("operator_bases must be a non-empty list")
    basis_ids = [item.get("basis_id") for item in bases if isinstance(item, dict)]
    if len(basis_ids) != len(bases) or len(set(basis_ids)) != len(basis_ids):
        raise EditingOperatorO0Error("operator basis ids must be present and unique")
    required_basis_ids = {
        "primitive7",
        "primitive7_plus_ring_system_delete",
        "primitive7_plus_two_neighbor_atom_insert",
        "primitive7_plus_ring_system_grow_screen",
    }
    if not required_basis_ids.issubset(set(basis_ids)):
        raise EditingOperatorO0Error("required operator comparisons are missing")
    delete_basis = next(
        item for item in bases if item.get("basis_id") == "primitive7_plus_ring_system_delete"
    )
    if delete_basis.get("support_provider") != ("explicit_typed_ring_catalog_required"):
        raise EditingOperatorO0Error("ring_system_delete comparison requires a typed ring catalog")
    cost_views = payload.get("cost_views")
    if not isinstance(cost_views, dict):
        raise EditingOperatorO0Error("cost_views must be an object")
    if cost_views.get("unsupported_lowering_policy") != (
        "report_explicitly_and_do_not_call_accelerator"
    ):
        raise EditingOperatorO0Error("unsupported-lowering policy must fail closed")
    if cost_views.get("unsupported_lowering_result") != {
        "event_search": "retain_action_with_null_primitive_cost",
        "event_target_status": "reached_with_unsupported_lowering",
        "primitive_search": "primitive_cost_undefined",
        "accelerator_claim": "forbidden",
    }:
        raise EditingOperatorO0Error("unsupported-lowering result semantics must be explicit")
    if payload.get("search_state_identity") != [
        "canonical_molecular_key",
        "persistent_slot_state_sha256",
        "constraint_state",
        "event_cost",
    ]:
        raise EditingOperatorO0Error("search identity must retain exact persistent-slot state")
    task_kinds = payload.get("required_task_kinds")
    if not isinstance(task_kinds, dict):
        raise EditingOperatorO0Error("required_task_kinds must be an object")
    for experiment in REQUIRED_EXPERIMENTS:
        values = task_kinds.get(experiment)
        if not isinstance(values, list) or not values:
            raise EditingOperatorO0Error(f"{experiment} requires at least one frozen task kind")
    resource_limits = payload.get("resource_limits")
    if not isinstance(resource_limits, dict):
        raise EditingOperatorO0Error("resource_limits must be frozen")
    for key in (
        "max_events",
        "max_expanded_labels",
        "max_generated_marks",
        "max_wall_seconds_per_search",
    ):
        if float(resource_limits.get(key, 0)) <= 0:
            raise EditingOperatorO0Error(f"invalid resource limit {key!r}")
    negative_semantics = payload.get("negative_result_semantics")
    if negative_semantics != {
        "complete": "not_reached_within_complete_event_bound",
        "resource_exhausted": "incomplete_resource_bound",
    }:
        raise EditingOperatorO0Error("negative-result semantics must fail closed")


def load_o0_contract(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EditingOperatorO0Error("O0 contract root must be an object")
    validate_o0_contract(payload)
    return payload


__all__ = [
    "BasisReachabilityResult",
    "CertifiedPath",
    "ConstraintTransition",
    "EditingOperatorO0Error",
    "GrowActionProvider",
    "O0_CONTRACT_SCHEMA",
    "O0_CONTRACT_VERSION",
    "ObjectiveSearchResult",
    "OperatorTask",
    "REQUIRED_EXPERIMENTS",
    "SearchLimits",
    "compare_operator_bases",
    "evaluate_basis_reachability",
    "load_o0_contract",
    "validate_o0_contract",
]
