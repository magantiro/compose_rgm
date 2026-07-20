"""Exact, validity-preserving rescheduling of executable rewrite traces.

The transport compiler emits one valid sequential program.  Some adjacent
events in that program are independent: they can execute in either order and
reach the identical padded molecular state.  This module detects that fact by
execution, not by a heuristic read/write approximation, and uses it to move
whole ring transactions toward the earliest state at which they are actually
executable.

The scheduler is deliberately conservative.  Cardinality changes are phase
barriers, and an event crosses another event only when both orders are legal
and array-exact after the pair.  Consequently all suffix actions retain their
original slot semantics and the final corpus endpoint is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace, execute_trace


DEFAULT_PRIORITY_RULES = frozenset({"ring_system_grow", "ring_system_delete"})
DEFAULT_PHASE_BARRIERS = frozenset({"atom_insert", "atom_delete"})


@dataclass(frozen=True)
class CommutingScheduleReport:
    """Diagnostics for one conservative trace rescheduling pass."""

    attempted_swaps: int
    accepted_swaps: int
    priority_positions_before: tuple[int, ...]
    priority_positions_after: tuple[int, ...]


def states_are_array_exact(left: MolecularGraph, right: MolecularGraph) -> bool:
    """Return whether two padded states are identical, including slot gauge."""

    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def swapped_adjacent_midpoint(
    state: MolecularGraph,
    first: RewriteStep,
    second: RewriteStep,
    *,
    system: RewriteSystem,
) -> MolecularGraph | None:
    """Return the swapped midpoint when two events commute exactly.

    The original order and the proposed reversed order must both execute and
    reach the same array-level successor after two events.  Returning the
    alternate midpoint lets the caller update its cached prefix states in
    constant time after an accepted swap.
    """

    try:
        original_midpoint = system.apply(state, first.rule_name, first.action)
        original_endpoint = system.apply(
            original_midpoint,
            second.rule_name,
            second.action,
        )
        swapped_midpoint = system.apply(state, second.rule_name, second.action)
        swapped_endpoint = system.apply(
            swapped_midpoint,
            first.rule_name,
            first.action,
        )
    except InvalidRewrite:
        return None
    if not states_are_array_exact(original_endpoint, swapped_endpoint):
        return None
    return swapped_midpoint


def schedule_priority_events_earliest(
    trace: RewriteTrace,
    *,
    priority_rule_names: Iterable[str] = DEFAULT_PRIORITY_RULES,
    phase_barrier_rule_names: Iterable[str] = DEFAULT_PHASE_BARRIERS,
    system: RewriteSystem | None = None,
) -> tuple[RewriteTrace, CommutingScheduleReport]:
    """Move selected transactions left across exactly commuting neighbors.

    This is a stable scheduler: non-priority events retain their mutual order,
    as do priority events that cannot legally commute.  Atom insertion and
    deletion are barriers by default, preserving the useful resize phase of
    the carbon-tree transport while allowing topology-local ring commitment.
    """

    runtime = system or de_novo_rewrite_system()
    priority = frozenset(priority_rule_names)
    barriers = frozenset(phase_barrier_rule_names)
    steps = list(trace.steps)
    endpoint, cached_states = execute_trace(
        trace.source,
        steps,
        system=runtime,
        return_states=True,
    )
    if not states_are_array_exact(endpoint, trace.target):
        raise ValueError("scheduler requires a trace with an array-exact endpoint")
    states = list(cached_states)
    before = tuple(
        index for index, step in enumerate(steps) if step.rule_name in priority
    )
    attempts = 0
    accepted = 0

    # Process each priority event once in its current left-to-right order.
    # Accepted swaps preserve the state after the pair, so only one cached
    # midpoint changes; every later cached state remains exact.
    priority_ordinal = 0
    while priority_ordinal < len(before):
        positions = [
            index for index, step in enumerate(steps) if step.rule_name in priority
        ]
        position = positions[priority_ordinal]
        while position > 0:
            predecessor = steps[position - 1]
            event = steps[position]
            if predecessor.rule_name in barriers:
                break
            attempts += 1
            swapped_midpoint = swapped_adjacent_midpoint(
                states[position - 1],
                predecessor,
                event,
                system=runtime,
            )
            if swapped_midpoint is None:
                break
            steps[position - 1], steps[position] = event, predecessor
            states[position] = swapped_midpoint
            accepted += 1
            position -= 1
        priority_ordinal += 1

    scheduled_endpoint, scheduled_states = execute_trace(
        trace.source,
        steps,
        system=runtime,
        return_states=True,
    )
    if not states_are_array_exact(scheduled_endpoint, trace.target):
        raise RuntimeError("commuting schedule changed the target endpoint")
    # The rewrite system already enforces chemical validity.  This second
    # replay is intentional: it is a cheap, explicit certificate stored with
    # every transformed path rather than an assumption about commutation.
    if len(scheduled_states) != len(steps) + 1:
        raise RuntimeError("commuting schedule replay returned incomplete states")

    after = tuple(
        index for index, step in enumerate(steps) if step.rule_name in priority
    )
    metadata = {
        **trace.metadata,
        "event_schedule": "exact_earliest_priority_v1",
        "schedule_priority_rules": tuple(sorted(priority)),
        "schedule_phase_barriers": tuple(sorted(barriers)),
        "schedule_attempted_swaps": attempts,
        "schedule_accepted_swaps": accepted,
        "schedule_priority_positions_before": before,
        "schedule_priority_positions_after": after,
    }
    scheduled = RewriteTrace(
        source=trace.source,
        target=trace.target,
        steps=tuple(steps),
        metadata=metadata,
    )
    return scheduled, CommutingScheduleReport(
        attempted_swaps=attempts,
        accepted_swaps=accepted,
        priority_positions_before=before,
        priority_positions_after=after,
    )


__all__ = [
    "CommutingScheduleReport",
    "schedule_priority_events_earliest",
    "states_are_array_exact",
    "swapped_adjacent_midpoint",
]
