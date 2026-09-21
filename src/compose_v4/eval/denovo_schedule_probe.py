"""Zero-training probe: does exact ring rescheduling repair the de-novo ring support?

The de-novo base overproduces 3- and 4-membered rings by roughly an order of
magnitude against its own corpus, and the defect is localized to one conditional
decision.  Uniform mass over the EXACT EXECUTABLE ring-template support, measured
at the states where ring events actually fire, is far above the catalog's own
unconditional small-ring mass, while the learned policy sits BELOW that uniform
baseline.  The action support is therefore wrong before the policy decides, which
is why more training and why an SA or small-ring penalty are both wrong tools.

``compose_v4.rewrite.commuting_schedule`` proposes a repair: move whole ring
transactions earlier across events that provably commute, leaving the compiled
endpoint array-exact.  It treats ``atom_insert``/``atom_delete`` as unconditional
phase barriers -- and atom insertion is exactly what consumes the free slots that
large ring templates need, so the repair may be inert against this defect.

This module measures that, on real training traces, without training anything.
It reports three quantities per schedule arm:

``ring_event_positions``
    Where ring transactions sit in the trace, RECOMPUTED FROM THE RETURNED TRACE.
    The scheduler reports its own positions; those are produced by the code under
    test, so they are cross-checked here rather than trusted.  A disagreement is
    itself a finding and is surfaced, never silently reconciled.

``free_slots``
    ``n_atoms - count(is_element(atom_types))`` at the state immediately preceding
    each ring event.  This is the mechanistic variable: a ring template is only
    executable when the slots it needs are free, so a schedule that does not raise
    this cannot widen the support no matter how far events move.

``small_ring_support_mass``
    Uniform mass on 3/4-ring templates over the exact executable support at each
    ring-grow state, computed through the production support predicate and the
    production category mask -- not a transcription of either.

Invariants maintained here:

* Every reported distribution carries its own explicit ``n``.  An empty arm
  reports ``None``, never a measured zero.
* A reordered trace that does not re-execute to an array-exact endpoint is
  DROPPED and COUNTED.  It is never repaired, and never silently kept.
* Arms are matched: all three see the identical ``(source, target)`` pair and the
  identical base trace, so an arm difference cannot come from a different sample.
* Positions are recomputed independently of the scheduler's report fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element, is_occupied
from compose_v4.eval.ring_calibration import (
    undesirable_small_ring_mask,
    uniform_category_mass,
)
from compose_v4.rewrite.commuting_schedule import (
    DEFAULT_PHASE_BARRIERS,
    DEFAULT_PRIORITY_RULES,
    schedule_priority_events_earliest,
    states_are_array_exact,
)
from compose_v4.rewrite.kernel import RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.trace import RewriteTrace, execute_trace

# The support question is asked of ring GROWTH specifically: it is the event that
# selects a template from the catalog, so it is the only event whose legal support
# has a small-ring mass at all.  Movement and free slots are asked of every
# priority transaction the scheduler is configured to move.
RING_GROW_RULE = "ring_system_grow"


# ---- Independent recomputation from a returned trace ----


def ring_event_indices(
    trace: RewriteTrace,
    *,
    priority_rule_names: Iterable[str] = DEFAULT_PRIORITY_RULES,
) -> tuple[int, ...]:
    """Return ring-transaction step indices, read from the trace's own steps.

    This deliberately does not consult ``trace.metadata``.  The scheduler writes
    its own before/after positions there, and evidence that ring events moved
    cannot come from the same code whose movement is in question.
    """

    priority = frozenset(priority_rule_names)
    return tuple(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name in priority
    )


def free_slots(state: MolecularGraph) -> int:
    """Return padded slots not holding a real chemical element.

    A SCAR occupies a slot without being an element, so this is an upper bound on
    slots an insertion could claim.  ``occupied_free_slots`` reports the stricter
    count; both are recorded so a divergence is visible rather than assumed away.
    """

    return int(state.n_atoms) - int(np.count_nonzero(is_element(state.atom_types)))


def occupied_free_slots(state: MolecularGraph) -> int:
    """Return padded slots that are wholly unoccupied, scars included as taken."""

    return int(state.n_atoms) - int(np.count_nonzero(is_occupied(state.atom_types)))


@dataclass(frozen=True)
class RingEventObservation:
    """One ring transaction, located and characterized inside one trace."""

    rule_name: str
    index: int
    trace_length: int
    free_slots: int
    occupied_free_slots: int
    state: MolecularGraph = field(compare=False, repr=False)

    @property
    def fraction_of_trace(self) -> float:
        """Position as a fraction of trace length.

        Rescheduling permutes steps and never changes their number, so the
        denominator is identical across arms and the choice cannot manufacture a
        difference between them.
        """

        return float(self.index) / float(self.trace_length)


def observe_ring_events(
    trace: RewriteTrace,
    *,
    priority_rule_names: Iterable[str] = DEFAULT_PRIORITY_RULES,
    system: RewriteSystem | None = None,
) -> tuple[RingEventObservation, ...]:
    """Locate every ring transaction and the state it fires from.

    The prefix states come from replaying the trace through the real executor, so
    the state a ring event is measured at is the state the event actually sees.
    """

    runtime = system or de_novo_rewrite_system()
    _, states = execute_trace(
        trace.source,
        trace.steps,
        system=runtime,
        return_states=True,
    )
    if len(states) != len(trace.steps) + 1:
        raise RuntimeError("trace replay returned incomplete prefix states")
    priority = frozenset(priority_rule_names)
    length = max(len(trace.steps), 1)
    return tuple(
        RingEventObservation(
            rule_name=step.rule_name,
            index=index,
            trace_length=length,
            free_slots=free_slots(states[index]),
            occupied_free_slots=occupied_free_slots(states[index]),
            state=states[index],
        )
        for index, step in enumerate(trace.steps)
        if step.rule_name in priority
    )


# ---- Schedule arms ----


@dataclass(frozen=True)
class ScheduleArmResult:
    """One arm applied to one base trace."""

    arm: str
    trace: RewriteTrace | None
    attempted_swaps: int
    accepted_swaps: int
    reported_positions_before: tuple[int, ...]
    reported_positions_after: tuple[int, ...]
    dropped: bool
    drop_reason: str | None


def apply_schedule_arm(
    trace: RewriteTrace,
    *,
    arm: str,
    phase_barrier_rule_names: Iterable[str],
    priority_rule_names: Iterable[str] = DEFAULT_PRIORITY_RULES,
    system: RewriteSystem | None = None,
) -> ScheduleArmResult:
    """Reschedule one trace, dropping and counting any arm that cannot stay exact.

    The exactness test inside the scheduler is untouched: a swap is still accepted
    only when both orders execute and agree array-exactly after the pair, and the
    reordered trace is still re-executed against the original target.  Relaxing
    the barrier set changes only WHICH adjacent pairs are offered to that test.
    """

    runtime = system or de_novo_rewrite_system()
    try:
        scheduled, report = schedule_priority_events_earliest(
            trace,
            priority_rule_names=priority_rule_names,
            phase_barrier_rule_names=phase_barrier_rule_names,
            system=runtime,
        )
    except (RuntimeError, ValueError) as error:
        return ScheduleArmResult(
            arm=arm,
            trace=None,
            attempted_swaps=0,
            accepted_swaps=0,
            reported_positions_before=(),
            reported_positions_after=(),
            dropped=True,
            drop_reason=f"{type(error).__name__}: {error}",
        )
    endpoint, _ = execute_trace(
        scheduled.source,
        scheduled.steps,
        system=runtime,
        return_states=True,
    )
    if not states_are_array_exact(endpoint, trace.target):
        return ScheduleArmResult(
            arm=arm,
            trace=None,
            attempted_swaps=report.attempted_swaps,
            accepted_swaps=report.accepted_swaps,
            reported_positions_before=report.priority_positions_before,
            reported_positions_after=report.priority_positions_after,
            dropped=True,
            drop_reason="rescheduled endpoint is not array-exact",
        )
    return ScheduleArmResult(
        arm=arm,
        trace=scheduled,
        attempted_swaps=report.attempted_swaps,
        accepted_swaps=report.accepted_swaps,
        reported_positions_before=report.priority_positions_before,
        reported_positions_after=report.priority_positions_after,
        dropped=False,
        drop_reason=None,
    )


RELAXED_PHASE_BARRIERS: frozenset[str] = frozenset()


# Arm A is the UNSCHEDULED base trace, so it is deliberately absent here: it is
# not a scheduling configuration and must never be produced by running the
# scheduler with some "identity" barrier set, which would not be byte-identical
# to the trace Lineage B trained on.
ARM_BARRIERS: dict[str, frozenset[str]] = {
    "B_exact_early_ring_default_barriers": DEFAULT_PHASE_BARRIERS,
    "C_exact_early_ring_barrier_relaxed": RELAXED_PHASE_BARRIERS,
}

ARM_A = "A_sequential"


# ---- Support measurement ----


def small_ring_support_mass(
    support: Sequence[bool],
    category: np.ndarray,
) -> tuple[float, int, int]:
    """Return (uniform small mass, legal templates, legal small templates).

    The mass itself is computed by the production helper; this function only
    adapts the support predicate's tuple return into the tensor form that helper
    expects, so the statistic is never recomputed here.
    """

    import torch

    support_tensor = torch.tensor(tuple(bool(item) for item in support), dtype=torch.bool)
    category_tensor = (
        category
        if isinstance(category, torch.Tensor)
        else torch.tensor(tuple(bool(item) for item in category), dtype=torch.bool)
    )
    legal = int(support_tensor.sum())
    legal_small = int((support_tensor & category_tensor).sum())
    return (
        uniform_category_mass(support_tensor, category_tensor),
        legal,
        legal_small,
    )


def small_ring_category_mask(templates: Sequence[object], *, maximum_size: int = 4):
    """Return the production 3/4-ring template mask for one catalog."""

    return undesirable_small_ring_mask(templates, maximum_size=maximum_size)


# ---- Aggregation ----


def _summarize(values: Sequence[float]) -> dict[str, object]:
    """Summarize one distribution, reporting ``None`` rather than a false zero."""

    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "p10": None, "p25": None, "p75": None, "p90": None, "stderr": None}
    ordered = sorted(float(value) for value in values)
    array = np.asarray(ordered, dtype=float)
    count = len(ordered)
    return {
        "n": count,
        "mean": float(array.mean()),
        "median": float(median(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "stderr": (float(array.std(ddof=1) / np.sqrt(count)) if count > 1 else None),
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    """Return an exact integer histogram, so a distribution is never only a mean."""

    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def summarize_arm(
    *,
    arm: str,
    observations: Sequence[RingEventObservation],
    support_masses: Sequence[float],
    legal_counts: Sequence[int],
    legal_small_counts: Sequence[int],
    attempted_swaps: int,
    accepted_swaps: int,
    traces_scheduled: int,
    traces_dropped: int,
    drop_reasons: Sequence[str],
    position_report_disagreements: int,
) -> dict[str, object]:
    """Reduce one arm's per-event records into the reported distributions."""

    return {
        "arm": arm,
        "traces_scheduled": traces_scheduled,
        "traces_dropped": traces_dropped,
        "drop_reasons": sorted(set(drop_reasons))[:8],
        "position_report_disagreements": position_report_disagreements,
        "attempted_swaps": attempted_swaps,
        "accepted_swaps": accepted_swaps,
        "ring_events": len(observations),
        "m1_position_index": _summarize([obs.index for obs in observations]),
        "m1_position_fraction": _summarize(
            [obs.fraction_of_trace for obs in observations]
        ),
        "m2_free_slots": _summarize([obs.free_slots for obs in observations]),
        "m2_free_slots_histogram": _histogram([obs.free_slots for obs in observations]),
        "m2_occupied_free_slots": _summarize(
            [obs.occupied_free_slots for obs in observations]
        ),
        "m3_small_ring_support_mass": _summarize(support_masses),
        "m3_legal_template_count": _summarize(legal_counts),
        "m3_legal_small_template_count": _summarize(legal_small_counts),
        "rule_census": _histogram_str([obs.rule_name for obs in observations]),
    }


def _histogram_str(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


__all__ = [
    "ARM_A",
    "ARM_BARRIERS",
    "RELAXED_PHASE_BARRIERS",
    "RING_GROW_RULE",
    "RingEventObservation",
    "ScheduleArmResult",
    "apply_schedule_arm",
    "free_slots",
    "observe_ring_events",
    "occupied_free_slots",
    "ring_event_indices",
    "small_ring_category_mask",
    "small_ring_support_mass",
    "states_are_array_exact",
    "summarize_arm",
]
