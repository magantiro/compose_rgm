"""Bounded support expansion for a T4 round whose candidate pool came back empty.

WHAT THIS FIXES
---------------
Every T4 campaign app ends a cell the first time selection returns nothing::

    if not selected:
        result = {..., "status": "candidate_exhaustion", ...}
        return result

`select_batch` returns empty exactly when `candidates` is empty, and `room` is at
least one whenever a round starts, so "this round proposed no eligible endpoint"
and "this cell is over" are THE SAME EVENT.  A cell whose eligible yield is low
but positive therefore dies on one unlucky draw with its whole budget unspent.
That is what the blank `fa7_0` cell is: it terminated at `charged_calls=1` -- the
seed's own docking call -- with 247 authorized calls never issued.

This module is the bounded expansion that runs on entry to that branch, and
nowhere else.  Only an expansion that runs to its declared end without producing
an eligible endpoint is candidate exhaustion.

WHY THE TRIGGER IS THE TERMINAL CONDITION AND NOT A THRESHOLD
--------------------------------------------------------------
Every round that wrote a lock had at least one eligible candidate, so it never
entered the branch.  On any row that reached its budget the expansion is
therefore UNREACHABLE, and the proof is structural rather than statistical: no
RNG is drawn and no state is mutated before the check.  A weaker trigger -- say
"fewer than n candidates" -- would reach inside successful cells and would owe a
full re-run of the panel.

WHAT IT SPENDS
--------------
CPU, never oracle calls.  It returns proposal records; the caller locks and docks
them through the unchanged round path, so charged-call accounting, per-query
receipts and the budget ceiling are untouched.  Nothing here retries a query,
replaces a failed docking or backfills a receipt -- the policies that forbid
those are about ORACLE-CALL INTEGRITY and are not weakened by proposing more
molecules before any lock is written.

THE TWO STAGES, AND WHY BOTH
----------------------------
`zero_support_fallback` runs first.  It is structurally different rather than
merely larger: it excises each bridge-separated region through the production
executor and gates the EXECUTED endpoint directly, so it never enters the
goal-abstraction layer where `t4_fiber_campaign.expand` abstracts a program to a
structural goal, re-binds it and gates whatever `instantiate_goal` returns -- the
layer at which the program's own endpoint was measured to reach the gate with
frequency 0.000 on this cell.  It is cheap and its support is nearly
deterministic, so it either supplies candidates quickly or is ruled out quickly.

The draw ladder runs second.  It escalates the primary lanes' own draw budget,
which is the configuration whose yield is low-but-positive rather than zero, and
is therefore the stage that a rare-event cell is actually waiting on.  More
attempts alone do not fix a distribution that misses the transformation; this
buys the tail of a distribution that reaches it, and nothing more.

THE STOPPING RULE
-----------------
Stop at the first of:

``reached_target``     `stop_at_distinct_eligible` distinct endpoints accumulated
``wall_clock``         `wall_seconds` elapsed since the event began
``ladder_exhausted``   every ladder step ran

Only `reached_target` is not, by itself, grounds to end the cell; the caller
still has to select from what came back.  The other two are written verbatim
into the round lock, so a terminal result records WHICH bound stopped it instead
of a bare status string.

WHY THERE IS NO ``draw_cap`` STOP REASON, AND WHY THAT IS THE FIX
------------------------------------------------------------------
`max_extra_draws_per_event` is the event's declared total spend.  It used to be
checked mid-run -- the loop broke with ``stop_reason="draw_cap"`` the first time
the next step would cross it -- while `from_contract` only required the cap to
admit the LARGEST STEP.  Those two facts together are a defect with a dead half
and a live half:

DEAD.  Under the frozen ladder the cap IS the ladder's sum (960+1920+3840 =
6720), so `spent + step` reaches the cap exactly and never exceeds it.  The
branch could not fire, which made ``draw_cap`` an untestable production stop
reason -- a guard is only tested where it binds, and this one never bound.

LIVE, and worse.  A policy declaring ``draw_ladder=(960, 1920)`` with
``max_extra_draws_per_event=2000`` PASSED validation (1920 <= 2000) and then ran
exactly ONE of its two declared steps, reporting a bound its author never meant
to set.  Silently spending less than the declared ladder is the same class of
failure this module exists to prevent: a bounded search whose realized bound is
not the one anybody wrote down.

So the cap is now enforced where it can actually be checked -- at CONSTRUCTION,
against the ladder's SUM, in `__post_init__` so a hand-built policy is covered
too, and again in `from_contract` with a contract-shaped message.  A policy
whose cap cannot admit its own ladder is refused before a single CPU-second is
spent rather than discovered halfway through.  The cap keeps its stated purpose
-- it is the tripwire for a future ladder that grows past its own declared total
-- and now that tripwire actually fires.

The runtime check survives as defence in depth and RAISES rather than breaking:
reaching it means a policy bypassed its own construction invariant, which is a
bug and not a legitimate stop.  A ladder that should spend less must be declared
shorter; a cap and a ladder that disagree are two ways of saying one thing, and
carrying both is exactly the per-target tuning freedom
`compose_v4.control.frozen_proposal_escalation` exists to remove.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "t4_support_expansion_v1"

#: The contract block this policy is read from.
CONTRACT_FIELD = "support_expansion"

#: Lanes whose proposal cost is a draw count, and which can therefore be
#: escalated. `route_complete_region` is a beam over a fitted expert with no
#: draw parameter, so escalating it would mean moving a different quantity and
#: is deliberately not offered.
ESCALATABLE_LANES = ("shallow", "anchored_replacement")

#: The lane label the fallback's records carry, mirrored from
#: `compose_v4.control.zero_support_fallback`.
FALLBACK_LANE = "zero_support_fallback"


class SupportExpansionContractError(ValueError):
    """The contract's support-expansion block cannot be honoured as written."""


class SupportExpansionNotConsumed(SupportExpansionContractError):
    """The contract declares an expansion the running path never performs."""


@dataclass(frozen=True)
class SupportExpansionPolicy:
    """The bounded expansion a contract authorizes for one exhausted round.

    Every field is a bound.  There is no unbounded mode and no default that
    grows without one, because the failure being guarded against is a search
    that spends an authorized budget discovering that it should have stopped.
    """

    draw_ladder: tuple[int, ...]
    lanes: tuple[str, ...]
    zero_support_fallback: bool
    stop_at_distinct_eligible: int
    max_extra_draws_per_event: int
    wall_seconds: float

    def __post_init__(self) -> None:
        """Refuse a policy whose cap cannot admit its own declared ladder.

        This is the CONSTRUCTION-time form of the bound that used to be
        discovered mid-run as ``stop_reason="draw_cap"``.  It lives here rather
        than only in `from_contract` so a hand-built policy -- the form every
        test and `frozen_proposal_escalation.FROZEN_LADDER` itself use -- is
        covered by the same invariant a contract-built one is.

        After this, ``spent + step <= sum(draw_ladder) <= max_extra_draws_per_event``
        holds at every ladder step, so the runtime cap check is unreachable and
        a ladder always runs to the length it declares.
        """

        total = sum(self.draw_ladder)
        if self.max_extra_draws_per_event < total:
            raise SupportExpansionContractError(
                f"max_extra_draws_per_event={self.max_extra_draws_per_event} cannot "
                f"admit the declared draw_ladder {list(self.draw_ladder)} summing to "
                f"{total}; the event would silently run only part of its own ladder "
                "and report a bound nobody declared. Shorten the ladder or raise the "
                "cap -- the two must agree before anything is spent"
            )

    @staticmethod
    def from_contract(block: Any) -> SupportExpansionPolicy:
        """Build the policy a contract declares, refusing anything unbounded."""

        if not isinstance(block, dict):
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD!r} must be a mapping, got {type(block).__name__}"
            )
        allowed = {
            "draw_ladder",
            "lanes",
            "zero_support_fallback",
            "stop_at_distinct_eligible",
            "max_extra_draws_per_event",
            "wall_seconds",
        }
        unknown = set(block) - allowed
        if unknown:
            raise SupportExpansionContractError(
                f"unknown {CONTRACT_FIELD!r} keys {sorted(unknown)}"
            )
        ladder = tuple(int(step) for step in block.get("draw_ladder", ()))
        if not ladder:
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.draw_ladder must name at least one step"
            )
        if any(step < 1 for step in ladder):
            raise SupportExpansionContractError(
                f"every {CONTRACT_FIELD}.draw_ladder step must be positive"
            )
        lanes = tuple(str(lane) for lane in block.get("lanes", ()))
        if not lanes:
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.lanes must name at least one lane"
            )
        illegal = sorted(set(lanes) - set(ESCALATABLE_LANES))
        if illegal:
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.lanes {illegal} have no draw budget to escalate; "
                f"escalatable lanes are {list(ESCALATABLE_LANES)}"
            )
        fallback = block.get("zero_support_fallback")
        if not isinstance(fallback, bool):
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.zero_support_fallback must be an explicit boolean; "
                "an absent or coerced value would leave it unclear whether the stage ran"
            )
        target = int(block.get("stop_at_distinct_eligible", 0))
        if target < 1:
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.stop_at_distinct_eligible must be at least one"
            )
        cap = int(block.get("max_extra_draws_per_event", 0))
        if cap < sum(ladder):
            # The SUM, not the largest step. A cap that admits only some prefix
            # of the ladder validated under the old rule and then truncated the
            # event mid-flight, which is a bound the contract never declared.
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.max_extra_draws_per_event must admit the WHOLE "
                f"declared ladder {list(ladder)} summing to {sum(ladder)}, got {cap}; "
                "a cap below its own ladder would run only part of it and report a "
                "bound the contract never declared"
            )
        wall = float(block.get("wall_seconds", 0.0))
        if wall <= 0:
            raise SupportExpansionContractError(
                f"{CONTRACT_FIELD}.wall_seconds must be positive"
            )
        return SupportExpansionPolicy(
            draw_ladder=ladder,
            lanes=lanes,
            zero_support_fallback=fallback,
            stop_at_distinct_eligible=target,
            max_extra_draws_per_event=cap,
            wall_seconds=wall,
        )

    def as_record(self) -> dict:
        """The policy as it should appear in a round lock."""

        return {
            "schema_version": SCHEMA_VERSION,
            "draw_ladder": list(self.draw_ladder),
            "lanes": list(self.lanes),
            "zero_support_fallback": self.zero_support_fallback,
            "stop_at_distinct_eligible": self.stop_at_distinct_eligible,
            "max_extra_draws_per_event": self.max_extra_draws_per_event,
            "wall_seconds": self.wall_seconds,
        }


def support_expansion_request(payload: dict) -> Any:
    """The raw contract block, or ``None`` when the field is absent."""

    return payload.get(CONTRACT_FIELD)


def resolve_support_expansion(payload: dict) -> SupportExpansionPolicy | None:
    """The contract's policy, or ``None`` when the field is absent.

    ``None`` is returned only for an ABSENT field.  A field that is present but
    unusable raises, because a run that silently degrades to the historical hard
    termination while its contract says otherwise is the defect this surface
    exists to stop.
    """

    block = support_expansion_request(payload)
    if block is None:
        return None
    return SupportExpansionPolicy.from_contract(block)


@dataclass
class ExpansionOutcome:
    """What one expansion event produced, and which bound ended it."""

    records: list[dict] = field(default_factory=list)
    attempts: int = 0
    draws_spent: int = 0
    distinct_eligible: int = 0
    fallback_ran: bool = False
    fallback_eligible: int = 0
    fallback_work: dict = field(default_factory=dict)
    stop_reason: str = "not_run"
    attempt_log: list[dict] = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return bool(self.records)

    def as_record(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "attempts": self.attempts,
            "draws_spent": self.draws_spent,
            "distinct_eligible": self.distinct_eligible,
            "fallback_ran": self.fallback_ran,
            "fallback_eligible": self.fallback_eligible,
            "fallback_work": dict(self.fallback_work),
            "stop_reason": self.stop_reason,
            "attempt_log": list(self.attempt_log),
        }


def _endpoint_key(record: dict) -> str:
    """The identity an expanded endpoint is deduplicated on.

    `smiles` is the canonical endpoint the production gate returned, so two
    records carrying the same string are the same molecule however they were
    built.
    """

    key = record.get("smiles")
    if not isinstance(key, str) or not key:
        raise SupportExpansionContractError(
            "an expanded proposal record carries no endpoint smiles"
        )
    return key


def run_support_expansion(
    policy: SupportExpansionPolicy,
    *,
    escalate: Callable[[int, int], Iterable[dict]],
    fallback: Callable[[], tuple[Iterable[dict], dict]] | None = None,
    already_seen: Sequence[str] = (),
    clock: Callable[[], float] = time.monotonic,
) -> ExpansionOutcome:
    """Expand the proposal support until enough eligible endpoints exist.

    `escalate(draws, attempt)` runs ONE escalated fan-out over `policy.lanes` at
    `draws` per lane and returns the eligible records it produced.  `fallback()`
    runs the zero-support fallback once and returns its records and its
    load-independent work counters.  Both are the caller's production paths
    injected as callables, not transcriptions of them -- which is the whole
    reason this function takes callables rather than doing the fan-out itself,
    and is what lets a test drive the real stopping rule.

    `already_seen` names endpoints the caller has archived or will not query
    again, so an expansion that only re-finds them is correctly reported as
    having found nothing new.

    Charges nothing.
    """

    outcome = ExpansionOutcome()
    seen: set[str] = set(already_seen)
    found: dict[str, dict] = {}
    started = clock()

    def _absorb(produced: Iterable[dict]) -> int:
        fresh = 0
        for record in produced:
            key = _endpoint_key(record)
            if key in seen or key in found:
                continue
            found[key] = record
            fresh += 1
        return fresh

    if policy.zero_support_fallback and fallback is not None:
        produced, work = fallback()
        produced = list(produced)
        outcome.fallback_ran = True
        outcome.fallback_work = dict(work)
        fresh = _absorb(produced)
        outcome.fallback_eligible = len(produced)
        outcome.attempt_log.append(
            {
                "stage": FALLBACK_LANE,
                "returned": len(produced),
                "fresh_distinct": fresh,
                "cumulative_distinct": len(found),
                "elapsed_seconds": round(clock() - started, 3),
            }
        )

    spent = 0
    stop_reason = "ladder_exhausted"
    if len(found) >= policy.stop_at_distinct_eligible:
        stop_reason = "reached_target"
    else:
        for attempt, step in enumerate(policy.draw_ladder):
            if spent + step > policy.max_extra_draws_per_event:
                # Unreachable for any policy that passed `__post_init__`, which
                # requires the cap to admit the ladder's SUM. Reaching it means a
                # policy bypassed its own construction invariant, so this is a
                # bug and not a stop: truncating here is how an event used to
                # spend less than its declared ladder without saying so.
                raise SupportExpansionContractError(
                    f"ladder step {attempt} of {list(policy.draw_ladder)} would take "
                    f"spend to {spent + step} against a declared cap of "
                    f"{policy.max_extra_draws_per_event}; this policy violates its own "
                    "construction invariant and must not truncate its ladder silently"
                )
            if clock() - started >= policy.wall_seconds:
                stop_reason = "wall_clock"
                break
            produced = list(escalate(step, attempt))
            spent += step
            outcome.attempts += 1
            fresh = _absorb(produced)
            outcome.attempt_log.append(
                {
                    "stage": "draw_ladder",
                    "attempt": attempt,
                    "draws_per_lane": step,
                    "returned": len(produced),
                    "fresh_distinct": fresh,
                    "cumulative_distinct": len(found),
                    "elapsed_seconds": round(clock() - started, 3),
                }
            )
            if len(found) >= policy.stop_at_distinct_eligible:
                stop_reason = "reached_target"
                break

    outcome.records = list(found.values())
    outcome.draws_spent = spent
    outcome.distinct_eligible = len(found)
    outcome.stop_reason = stop_reason
    return outcome


def normalize_expansion_records(
    records: Iterable[dict], *, experts: Sequence[str] = ESCALATABLE_LANES
) -> list[dict]:
    """Give expansion records the provenance the round path requires.

    MEASURED FAILURE, run 727d9db5: the expansion reached its target with FOUR
    eligible endpoints and then died with `KeyError('proposal_experts')` while
    building the query rows, discarding all four.

    The cause is a difference between the two paths, not a missing field in
    `expand`. On a normal round the pool goes through
    `t4_integrated_route_fiber.merge_expert_pools`, which ATTACHES
    `proposal_experts` as part of merging; the expansion path deduplicates by
    endpoint itself and so never passes through that function, and `expand`'s own
    records carry only `proposal_lane`. This restores the one key the merge would
    have supplied, from the lane that actually produced the record.

    Records that already carry `proposal_experts` -- the zero-support fallback
    sets it to `[]` deliberately, because its lane is not in the frozen expert
    vocabulary -- are left exactly as they are.
    """

    known = set(experts)
    out: list[dict] = []
    for record in records:
        row = dict(record)
        if "proposal_experts" not in row:
            lane = row.get("proposal_lane")
            row["proposal_experts"] = [lane] if lane in known else []
        out.append(row)
    return out


def assert_support_expansion_is_consumed(outcome: ExpansionOutcome) -> None:
    """Refuse a terminal result produced without the declared expansion running.

    A contract that declares an expansion and a runtime that terminates without
    performing one is the inert-mechanism failure this repository has paid for
    four times.  Calling this on the terminal path means candidate exhaustion
    can only be published after an expansion event with a recorded stop reason.
    """

    if outcome.stop_reason == "not_run":
        raise SupportExpansionNotConsumed(
            "the contract declares support expansion but the terminal path was "
            "reached without an expansion event; the runtime cannot honour this "
            "contract and must not publish candidate exhaustion"
        )
    if not outcome.fallback_ran and outcome.attempts < 1:
        raise SupportExpansionNotConsumed(
            "a support-expansion event ran neither the fallback nor a single "
            f"ladder step (stop_reason={outcome.stop_reason!r}); its bounds are "
            "misconfigured and the cell must not terminate as exhausted"
        )
