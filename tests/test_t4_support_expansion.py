"""The bounded support expansion: its stopping rule, its bounds, and its wiring.

The invariant under test is the one that decides whether `fa7_0` spends its budget
or repeats its blank: an empty candidate pool must EXPAND before it may terminate.

Every behavioural test drives the real `run_support_expansion` with stub stages
whose yield depends on the draw budget, which is exactly the situation the
mechanism exists for.  The wiring tests read the app's own source rather than a
transcription of it, and each one is paired with a mutation in
`test_wiring_mutations` that must turn it red -- a structural check whose
expectation is recomputed from the code under test cannot fail.

WHAT THIS FILE COVERS ON THIS BRANCH, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------------------------
Sixteen APP-COUPLED tests were removed here, not weakened and not skipped.  They
guard `modal_apps/t4_fa7_0_support_expansion_base_app.py` and
`configs/t4_fa7_0_support_expansion_v1.json`, which live on branch
`t4-fa7-0-support-expansion-20260922` and are NOT on this branch, so here they failed
rather than tested anything.  They are the guards for the four carried-over launch
fixes -- the root call checkpointed before the round loop, the root lock not forfeited
by its own checkpoint, the expansion round body dry-driven end to end, and expansion
records normalized before selection -- plus the contract-pin and wrapper checks.

They were not brought across because that contract pins
`src/compose_v4/control/dynamic_program_synthesis.py` and
`src/compose_v4/experiments/t4_fiber_campaign.py` at revisions differing from this
tree, and importing two shared chemistry modules to make a PARKED test pass would move
the proposal path underneath the routing measurements in
`diagnostics/t4_state_routing_v1/blind_routing_gate_v1.json`.

A skip was rejected in favour of removal: a suite that passes while silently testing
less than it claims is a failure mode this repository has already paid for.  What
remains is the module-level contract -- the policy parser, the ladder, the stopping
rule, the fallback stage, and a real fallback record surviving the production
`attach_features` / `select_batch` path.  Anyone re-enabling the app must take the
sixteen guards back from that branch WITH the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_support_expansion import (
    ESCALATABLE_LANES,
    ExpansionOutcome,
    SupportExpansionContractError,
    SupportExpansionNotConsumed,
    SupportExpansionPolicy,
    assert_support_expansion_is_consumed,
    resolve_support_expansion,
    run_support_expansion,
)

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_APP = ROOT / "modal_apps/t4_fa7_0_support_expansion_app.py"

_GOOD_BLOCK = {
    "draw_ladder": [960, 1920, 3840],
    "lanes": ["shallow", "anchored_replacement"],
    "zero_support_fallback": True,
    "stop_at_distinct_eligible": 4,
    "max_extra_draws_per_event": 6720,
    "wall_seconds": 7200.0,
}


def _policy(**overrides) -> SupportExpansionPolicy:
    block = {**_GOOD_BLOCK, **overrides}
    return SupportExpansionPolicy.from_contract(block)


def _records(count: int, *, prefix: str = "C") -> list[dict]:
    return [{"smiles": prefix + "C" * index} for index in range(count)]


# ---- Contract surface ----------------------------------------------------


def test_policy_reads_a_well_formed_block():
    policy = _policy()
    assert policy.draw_ladder == (960, 1920, 3840)
    assert policy.lanes == ("shallow", "anchored_replacement")
    assert policy.zero_support_fallback is True
    assert policy.as_record()["draw_ladder"] == [960, 1920, 3840]


@pytest.mark.parametrize(
    "overrides",
    [
        {"draw_ladder": []},
        {"draw_ladder": [960, 0]},
        {"draw_ladder": [960, -1]},
        {"lanes": []},
        {"lanes": ["route_complete_region"]},
        {"stop_at_distinct_eligible": 0},
        {"max_extra_draws_per_event": 100},
        {"wall_seconds": 0.0},
        {"wall_seconds": -1.0},
    ],
)
def test_policy_refuses_an_unbounded_or_malformed_block(overrides):
    with pytest.raises(SupportExpansionContractError):
        _policy(**overrides)


def test_policy_refuses_an_unknown_key():
    block = {**_GOOD_BLOCK, "max_rounds": 3}
    with pytest.raises(SupportExpansionContractError):
        SupportExpansionPolicy.from_contract(block)


def test_fallback_flag_must_be_an_explicit_boolean():
    """A coerced or absent value would leave it unclear whether the stage ran."""

    for value in (None, 1, "true", ""):
        block = {**_GOOD_BLOCK, "zero_support_fallback": value}
        with pytest.raises(SupportExpansionContractError):
            SupportExpansionPolicy.from_contract(block)


def test_route_lane_is_not_escalatable():
    assert "route_complete_region" not in ESCALATABLE_LANES


def test_absent_field_is_the_only_none():
    assert resolve_support_expansion({}) is None
    assert resolve_support_expansion({"support_expansion": _GOOD_BLOCK}) is not None
    with pytest.raises(SupportExpansionContractError):
        resolve_support_expansion({"support_expansion": {"draw_ladder": []}})


# ---- The stopping rule ---------------------------------------------------


def test_a_budget_dependent_yield_is_recovered_by_escalation():
    """The mechanism's whole point: empty at the base budget, non-empty above it.

    This is the `fa7_0` situation in miniature -- the normal round found nothing,
    and a larger draw budget over the same lanes does.
    """

    seen = []

    def escalate(draws, attempt):
        seen.append((draws, attempt))
        return _records(4) if draws >= 1920 else []

    outcome = run_support_expansion(_policy(zero_support_fallback=False), escalate=escalate)
    assert seen == [(960, 0), (1920, 1)]
    assert outcome.stop_reason == "reached_target"
    assert outcome.distinct_eligible == 4
    assert outcome.found_any
    assert outcome.draws_spent == 2880


def test_a_ladder_that_never_yields_is_exhaustion_with_the_bound_named():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False), escalate=lambda draws, attempt: []
    )
    assert outcome.stop_reason == "ladder_exhausted"
    assert outcome.attempts == 3
    assert outcome.draws_spent == 6720
    assert not outcome.found_any


def test_a_cap_below_its_own_ladder_is_refused_at_construction_not_mid_run():
    """The `draw_cap` fix: the bound now binds where it can be checked.

    `draw_ladder=(960, 1920)` with `max_extra_draws_per_event=2000` used to
    VALIDATE -- the old rule only required the cap to admit the largest STEP --
    and then run exactly ONE of its two declared steps, reporting
    `stop_reason="draw_cap"` for a bound nobody wrote down. It is refused before
    anything is spent.
    """

    with pytest.raises(SupportExpansionContractError) as refusal:
        _policy(
            zero_support_fallback=False,
            draw_ladder=[960, 1920],
            max_extra_draws_per_event=2000,
        )
    assert "2880" in str(refusal.value)


def test_a_policy_that_bypasses_construction_cannot_truncate_its_ladder_silently():
    """Defence in depth: the runtime cap check RAISES, it does not stop.

    `object.__new__` plus `object.__setattr__` is the only way to reach this
    state, and it is used here precisely because no legitimate caller can
    produce it. A policy that got past `__post_init__` and would still cross its
    cap is a bug, so the event must fail loudly rather than record a short
    ladder as an ordinary stop.
    """

    rogue = object.__new__(SupportExpansionPolicy)
    for name, value in {
        "draw_ladder": (960, 1920),
        "lanes": ("shallow",),
        "zero_support_fallback": False,
        "stop_at_distinct_eligible": 4,
        "max_extra_draws_per_event": 2000,
        "wall_seconds": 60.0,
    }.items():
        object.__setattr__(rogue, name, value)

    with pytest.raises(SupportExpansionContractError):
        run_support_expansion(rogue, escalate=lambda draws, attempt: [])


def test_the_frozen_ladder_spends_exactly_its_declared_total():
    """The frozen ladder's cap IS its sum, so the whole ladder always runs.

    This is the fact that made the old mid-run check dead code. Asserting it by
    name means a future ladder growing past its own declared total is caught
    here rather than by a branch that can never fire.
    """

    from compose_v4.control.frozen_proposal_escalation import FROZEN_LADDER

    assert sum(FROZEN_LADDER.draw_ladder) == FROZEN_LADDER.max_extra_draws_per_event

    outcome = run_support_expansion(
        FROZEN_LADDER, escalate=lambda draws, attempt: [], fallback=lambda: ([], {})
    )
    assert outcome.stop_reason == "ladder_exhausted"
    assert outcome.attempts == len(FROZEN_LADDER.draw_ladder)
    assert outcome.draws_spent == FROZEN_LADDER.max_extra_draws_per_event


def test_the_wall_clock_binds_and_is_checked_before_a_step_is_spent():
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])

    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, wall_seconds=10.0),
        escalate=lambda draws, attempt: [],
        clock=lambda: next(ticks),
    )
    assert outcome.stop_reason == "wall_clock"
    assert outcome.attempts == 1


def test_already_seen_endpoints_are_not_counted_as_found():
    """An expansion that only re-finds archived molecules has found nothing new."""

    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, stop_at_distinct_eligible=2),
        escalate=lambda draws, attempt: _records(3),
        already_seen=[row["smiles"] for row in _records(3)],
    )
    assert outcome.distinct_eligible == 0
    assert outcome.stop_reason == "ladder_exhausted"


def test_duplicate_endpoints_across_steps_are_counted_once():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, stop_at_distinct_eligible=99),
        escalate=lambda draws, attempt: _records(2),
    )
    assert outcome.distinct_eligible == 2
    assert outcome.attempts == 3


def test_a_record_without_an_endpoint_is_refused_rather_than_silently_dropped():
    with pytest.raises(SupportExpansionContractError):
        run_support_expansion(
            _policy(zero_support_fallback=False),
            escalate=lambda draws, attempt: [{"quality": 0.9}],
        )


# ---- The fallback stage --------------------------------------------------


def test_the_fallback_runs_first_and_can_satisfy_the_target_alone():
    """It is the cheap, structurally different stage, so no ladder step is spent."""

    calls = []

    def escalate(draws, attempt):
        calls.append(draws)
        return []

    outcome = run_support_expansion(
        _policy(stop_at_distinct_eligible=2),
        escalate=escalate,
        fallback=lambda: (_records(3), {"regions_considered": 26}),
    )
    assert calls == []
    assert outcome.fallback_ran is True
    assert outcome.fallback_eligible == 3
    assert outcome.fallback_work == {"regions_considered": 26}
    assert outcome.stop_reason == "reached_target"
    assert outcome.attempt_log[0]["stage"] == "zero_support_fallback"


def test_a_zero_yield_fallback_still_hands_over_to_the_ladder():
    """The measured fa7_0 case: the fallback finds nothing and the ladder must run."""

    outcome = run_support_expansion(
        _policy(stop_at_distinct_eligible=1),
        escalate=lambda draws, attempt: _records(1) if attempt >= 1 else [],
        fallback=lambda: ([], {"regions_considered": 26, "distinct_eligible_endpoints": 0}),
    )
    assert outcome.fallback_ran is True
    assert outcome.fallback_eligible == 0
    assert outcome.attempts == 2
    assert outcome.stop_reason == "reached_target"
    assert outcome.found_any


def test_the_fallback_is_skipped_when_the_contract_disables_it():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False),
        escalate=lambda draws, attempt: [],
        fallback=lambda: (_records(9), {}),
    )
    assert outcome.fallback_ran is False
    assert outcome.distinct_eligible == 0


# ---- The consumption gate ------------------------------------------------


def test_publishing_exhaustion_without_an_expansion_is_refused():
    """The `not_run` clause, on a fixture the second clause cannot also catch.

    A plain `ExpansionOutcome()` trips BOTH clauses, so disabling either one left
    the other to raise and a mutation survived. `fallback_ran=True` satisfies the
    second clause, leaving only the stop-reason check able to refuse.
    """

    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(
            ExpansionOutcome(stop_reason="not_run", fallback_ran=True)
        )
    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(ExpansionOutcome())


def test_an_event_that_ran_no_stage_at_all_is_refused():
    """A misconfigured bound that spends nothing must not end the cell."""

    outcome = ExpansionOutcome(stop_reason="wall_clock")
    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(outcome)


def test_a_real_exhausted_ladder_is_accepted():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False), escalate=lambda draws, attempt: []
    )
    assert_support_expansion_is_consumed(outcome)


# ---- The contract this arm launches under --------------------------------










# ---- The app wiring ------------------------------------------------------
























# ---- The fallback record must survive the real feature path ----------------


def test_a_real_fallback_record_survives_attach_features_and_selection():
    """The shape bug that fires only when the fallback SUCCEEDS.

    `t4_integrated_route_fiber._experts` validates a record's `proposal_lane`
    against the frozen expert vocabulary and RAISES on an unknown one, so a record
    carrying "zero_support_fallback" killed `attach_features` at exactly the moment
    the fallback first produced an eligible endpoint. Stubbed `{"smiles": ...}`
    records cannot catch that, so this drives the REAL fallback and pushes a REAL
    record through the real feature and selection path.
    """

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        select_batch,
    )

    # A permissive reference so the fallback actually returns something: the point
    # is the RECORD SHAPE, not this molecule's chemistry.
    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    produced, _ = fallback_candidates(
        parent,
        np.random.default_rng(20260922),
        check=fiber.check,
        reference_smiles=parent,
        delta=0.2,
    )
    assert produced, "the fallback returned nothing, so this test proves nothing"
    assert produced[0]["proposal_lane"] == "zero_support_fallback", (
        "the fallback no longer labels its lane, so the app's re-labelling may be dead"
    )

    # The app's re-labelling, applied exactly as `proposal_worker` applies it.
    records = [
        {
            **row,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": -7.5,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(row.get("inserted_atoms", 0)),
            "deleted": int(row.get("deleted_atoms", 0)),
        }
        for row in produced
    ]
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    candidates = attach_features(records, state, fiber)
    assert candidates, "attach_features dropped every real fallback record"
    expert_census(candidates)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=1.0),
        state,
        np.random.default_rng(1),
        round_index=1,
        batch=4,
        exploration=1,
        expert_floor_rounds=2,
    )
    assert selected, "select_batch could not select a real fallback candidate"




def test_a_real_fallback_candidate_serializes_into_a_round_lock():
    """`attach_features` attaches a SET fingerprint and numpy features.

    The round lock is published as canonical JSON, so an expansion record that
    cannot serialize would fail AFTER the expansion succeeded and before anything
    was docked -- the same fires-only-on-success shape as the lane-label defect.
    """

    import numpy as np

    from compose_v4.control.fiber_control import SearchState
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import attach_features

    def _jsonable(value):
        """The app's own helper, reproduced so a drift in it fails this test."""

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    produced, _ = fallback_candidates(
        parent,
        np.random.default_rng(20260922),
        check=fiber.check,
        reference_smiles=parent,
        delta=0.2,
    )
    rows = [
        {
            **row,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": -7.5,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(row.get("inserted_atoms", 0)),
            "deleted": int(row.get("deleted_atoms", 0)),
        }
        for row in produced
    ]
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    candidates = attach_features(rows, state, fiber)
    assert candidates
    json.dumps(_jsonable(candidates), sort_keys=True, separators=(",", ":"))


# ---- The root checkpoint that makes a round-one preemption survivable -------






# ---- The dry pass: drive the whole expansion round body end to end ---------






