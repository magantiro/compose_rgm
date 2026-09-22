"""The ONE frozen, target-agnostic proposal-escalation ladder for the T4 panel.

WHAT THIS IS
------------
`compose_v4.experiments.t4_support_expansion` supplies the MECHANISM: a bounded
ladder that runs when a round's candidate pool comes back empty, and a stopping
rule that records which bound ended it.  That module reads its numbers from the
contract, which is correct for a development arm and is exactly what must not be
true of the evaluated algorithm: a contract that can carry numbers can carry
different numbers per target, and then the panel is a family of controllers
rather than one.

This module removes that freedom.  The ladder is a module-level CONSTANT, and a
contract may only NAME it and pin its content hash.  A contract that carries a
draw count, a lane list or a stopping threshold is REFUSED, so per-target tuning
is not merely discouraged, it is unrepresentable.

THE LADDER
----------
Rung 0  ``zero_support_fallback``  -- a structurally different proposal lane:
        excise each bridge-separated region of the parent through the production
        executor and gate the EXECUTED endpoint directly, never entering the
        goal-abstraction layer.  Adds a lane; costs one worker per parent.
Rung 1  960 draws per escalatable lane
Rung 2  1920 draws per escalatable lane
Rung 3  3840 draws per escalatable lane
Rung 4  terminate -- publish ``candidate_exhaustion`` carrying the bound that
        stopped the ladder.

Rungs 1-3 add EFFORT over the same primary lanes at the same per-draw law; rung 0
is the only rung that adds a lane.  Both are bounded before the run.

WHAT "TARGET-AGNOSTIC" MEANS HERE, PRECISELY
--------------------------------------------
Two different quantities, and conflating them is how a claim like this goes
wrong:

* RUNG SELECTION -- whether the ladder fires, which rung runs next, at what draw
  count, and when to stop -- is a pure function of the policy constant and the
  event's own history (fresh distinct endpoints so far, draws spent, elapsed
  seconds).  It reads no protein, no seed, no source molecule and no score.
* RUNG CONTENT -- which molecules a rung proposes -- is necessarily a function of
  the parent, because a proposal is a proposal ABOUT something.  A ladder whose
  content did not depend on the parent would not be a search.

The claim is the first, and it is what `tests/test_frozen_proposal_escalation.py`
tests: driven with an identical yield history, all fifteen panel cells at both
thresholds issue a byte-identical sequence of rung requests and stop for the same
recorded reason.

THE TRIGGER
-----------
`select_batch` returns empty exactly when `candidates` is empty, and `room` is at
least one whenever a round starts, so "this round proposed no eligible endpoint"
and "this cell terminates at candidate_exhaustion" are THE SAME EVENT.  The
ladder fires on entry to that branch and nowhere else, so on a round that wrote a
lock it is unreachable and the proof is structural rather than statistical: no
RNG is drawn and no state is mutated before the check.

WHAT IT SPENDS
--------------
CPU, never oracle calls.  It returns proposal records which the caller locks and
docks through the unchanged round path, so charged-call accounting, the per-query
receipts, the no-retry/no-replacement/no-backfill policy and the budget ceiling
are untouched.
"""

from __future__ import annotations

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_support_expansion import (
    CONTRACT_FIELD,
    SupportExpansionContractError,
    SupportExpansionPolicy,
)

SCHEMA_VERSION = "t4_frozen_escalation_v1"

#: The name a contract uses to request this ladder. A contract names the ladder;
#: it never describes one.
FROZEN_LADDER_ID = "t4_frozen_escalation_v1"

#: The ladder itself. Every number is fixed here, in code, before any cell runs.
#:
#: draw_ladder            two doublings from twice the primary lanes' own base
#:                        draw count, so a rung is always a whole number of
#:                        replicate workers at that base.
#: lanes                  the lanes whose proposal cost IS a draw count.
#:                        `route_complete_region` is a beam over a fitted expert
#:                        with no draw parameter, so escalating it would move a
#:                        different quantity and it is deliberately absent.
#: stop_at_distinct_eligible  the round's own batch is 8 and its exploration
#:                        allowance is 2; four distinct eligible endpoints is
#:                        enough for selection to be a choice rather than a
#:                        formality, and stopping there keeps the ladder bounded
#:                        in the common case.
#: max_extra_draws_per_event  the sum of the ladder, so the cap binds only if a
#:                        future ladder grows past its own declared total.
#: wall_seconds           two hours, against a 21-hour cell driver.
FROZEN_LADDER = SupportExpansionPolicy(
    draw_ladder=(960, 1920, 3840),
    lanes=("shallow", "anchored_replacement"),
    zero_support_fallback=True,
    stop_at_distinct_eligible=4,
    max_extra_draws_per_event=6720,
    wall_seconds=7200.0,
)

#: The content hash of the ladder above. A contract pins THIS, so a silent edit
#: to any ladder number invalidates every contract that names it.
FROZEN_LADDER_SHA256 = identity(FROZEN_LADDER.as_record())

#: The exact block a contract must carry. Nothing else is accepted, which is the
#: whole point: a block that cannot hold a number cannot hold a per-target one.
_REQUIRED_KEYS = frozenset({"policy", "policy_sha256"})


def frozen_escalation_block() -> dict:
    """The `support_expansion` block a frozen-panel contract must carry."""

    return {"policy": FROZEN_LADDER_ID, "policy_sha256": FROZEN_LADDER_SHA256}


def resolve_frozen_escalation(payload: dict) -> SupportExpansionPolicy:
    """The frozen ladder, or a refusal naming what the contract did wrong.

    Returns the module constant ITSELF, not a copy built from the contract, so a
    caller holding the result is holding the same object every other cell holds.
    """

    block = payload.get(CONTRACT_FIELD)
    if block is None:
        raise SupportExpansionContractError(
            f"the contract declares no {CONTRACT_FIELD!r} block; a frozen-panel "
            "contract must name the frozen ladder so an empty candidate pool "
            "escalates instead of ending the cell"
        )
    if not isinstance(block, dict):
        raise SupportExpansionContractError(
            f"{CONTRACT_FIELD!r} must be a mapping, got {type(block).__name__}"
        )
    keys = frozenset(block)
    if keys != _REQUIRED_KEYS:
        extra = sorted(keys - _REQUIRED_KEYS)
        missing = sorted(_REQUIRED_KEYS - keys)
        raise SupportExpansionContractError(
            f"{CONTRACT_FIELD!r} must be exactly {sorted(_REQUIRED_KEYS)} for the "
            f"frozen panel (unexpected {extra}, missing {missing}); a contract that "
            "can carry a ladder number can carry a different one per target, which "
            "is the freedom this panel exists to remove"
        )
    if block["policy"] != FROZEN_LADDER_ID:
        raise SupportExpansionContractError(
            f"{CONTRACT_FIELD}.policy must be {FROZEN_LADDER_ID!r}, "
            f"got {block['policy']!r}"
        )
    if block["policy_sha256"] != FROZEN_LADDER_SHA256:
        raise SupportExpansionContractError(
            f"{CONTRACT_FIELD}.policy_sha256 pins {block['policy_sha256']!r} but the "
            f"frozen ladder hashes to {FROZEN_LADDER_SHA256!r}; either the ladder "
            "moved or the contract was sealed against a different one"
        )
    return FROZEN_LADDER


def frozen_escalation_record() -> dict:
    """The ladder as it should appear in a round lock and a terminal result."""

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": FROZEN_LADDER_ID,
        "policy_sha256": FROZEN_LADDER_SHA256,
        **FROZEN_LADDER.as_record(),
    }
