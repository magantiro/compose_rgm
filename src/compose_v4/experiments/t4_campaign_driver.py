"""Fail-closed phase-driver decisions for future shared T4 campaigns.

Frozen campaign apps retain their exact source identities.  New shared drivers
should use these decisions instead of inferring completion from one nonterminal
status such as ``status != "running"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

ACTIVE_PHASE_STATUSES = frozenset(
    {
        "running",
        "proposals_running",
        "queries_running",
    }
)
TERMINAL_PHASE_STATUSES = frozenset(
    {
        "complete_budget",
        "candidate_exhaustion",
        "operational_fail_closed",
        "root_oracle_failure",
        "failed",
    }
)

ContinuationState = Literal["none", "reserved", "running", "terminal"]
DriverAction = Literal["finish", "reserve_and_spawn", "wait"]


def campaign_finished(statuses: Iterable[str]) -> bool:
    """Return true only when every cell reports an explicit terminal status.

    Empty and unknown status sets are ambiguous operational state, so they fail
    closed instead of being treated as completed campaigns.
    """

    normalized = tuple(str(status) for status in statuses)
    if not normalized:
        raise ValueError("campaign driver received no cell statuses")
    unknown = sorted(set(normalized) - ACTIVE_PHASE_STATUSES - TERMINAL_PHASE_STATUSES)
    if unknown:
        raise ValueError(f"campaign driver received unknown cell statuses: {unknown}")
    return all(status in TERMINAL_PHASE_STATUSES for status in normalized)


def campaign_driver_action(
    statuses: Iterable[str], *, continuation_state: ContinuationState
) -> DriverAction:
    """Choose the only safe next action for a shared campaign driver.

    A driver must durably publish ``reserved`` before spawning a continuation.
    If it is preempted before recording the spawned call identifier, the
    reservation makes the ambiguous retry wait rather than create a duplicate
    call.  A later operator or supervisor may set ``terminal`` only after it has
    established that the prior continuation call is no longer live; that state
    permits one resumed spawn.
    """

    if continuation_state not in {"none", "reserved", "running", "terminal"}:
        raise ValueError(
            f"campaign driver received unknown continuation state: {continuation_state!r}"
        )
    finished = campaign_finished(statuses)
    if finished:
        return "finish"
    if continuation_state in {"reserved", "running"}:
        return "wait"
    return "reserve_and_spawn"


__all__ = [
    "ACTIVE_PHASE_STATUSES",
    "TERMINAL_PHASE_STATUSES",
    "campaign_driver_action",
    "campaign_finished",
]
