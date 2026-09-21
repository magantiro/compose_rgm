"""Guards for the three measured PMO allocation defects.

Each guard states a property that does NOT come from the code under test -- score
monotonicity, a bounded excursion length, an ordering between two cells at a named
reward scale -- so a mutation of the production code makes it red rather than moving
the expectation along with the observation.

Every test drives the production functions directly. Nothing here reimplements an
allocation rule in order to compare against it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.pmo_credit import (
    SCHEMA_VERSION,
    CreditKey,
    PopulationCredit,
)
from compose_v4.control.pmo_population_controller import (
    CHANNELS,
    ESCAPE_ROUNDS,
    JUMP_CHANNEL,
    PLATEAU_ROUNDS,
    PmoPopulationController,
    advance_plateau_state,
    initial_population_state,
)

# A drug-like leader, a near-tied runner-up, and three structurally isolated fragments.
# The fragments are what the production archive actually accumulated; the point of the
# panel is that each of them is alone enough in feature space to found its own niche.
LEADER = "CC(=O)Oc1ccccc1C(=O)O"
RUNNER_UP = "CC(=O)Oc1ccccc1C(=O)OC"
FRAGMENTS = ("O=C=O", "CCCF", "CCC=O")
PANEL = (LEADER, RUNNER_UP, *FRAGMENTS)


def _archive(allocation: str, scores: dict[str, float]) -> ProgramOptimizer:
    """A real ProgramOptimizer whose archive is exactly `scores`."""
    config = replace(
        ProgramSearchConfig(),
        parent_allocation=allocation,
        score_direction="maximize",
        require_broad_runtime=False,
        proposal_mode="program_only",
        channel_probabilities=(7 / 9, 2 / 9, 0.0),
    )
    optimizer = ProgramOptimizer(
        config, source_group="allocator_guard", oracle_protocol="allocator_guard"
    )
    for index, (endpoint, score) in enumerate(sorted(scores.items())):
        key = f"entry_{index}"
        optimizer.entries[key] = {"endpoint": endpoint, "static_score": float(score)}
        optimizer.observations[f"receipt_{index}"] = {
            "endpoint": endpoint,
            "score": float(score),
            "oracle_protocol": "allocator_guard",
        }
    return optimizer


def _mass(optimizer: ProgramOptimizer) -> dict[str, float]:
    keys, weights = optimizer.selection()
    mass: dict[str, float] = {}
    for key, weight in zip(keys, weights, strict=True):
        endpoint = optimizer.entries[key]["endpoint"]
        mass[endpoint] = mass.get(endpoint, 0.0) + float(weight)
    return mass


def _fragment_mass(allocation: str, fragment_score: float) -> float:
    scores = {LEADER: 0.90, RUNNER_UP: 0.85}
    scores.update({fragment: 0.02 for fragment in FRAGMENTS})
    scores[FRAGMENTS[0]] = fragment_score
    return _mass(_archive(allocation, scores))[FRAGMENTS[0]]


# ---- Defect 1: exploitation mass must depend on score ----


def test_evidence_allocation_mass_is_monotone_in_score():
    """Raising an endpoint's score must not lower, and must overall raise, its mass."""
    masses = [_fragment_mass("niche_evidence", score) for score in (0.01, 0.10, 0.40, 0.95)]
    assert masses == sorted(masses), f"mass must be non-decreasing in score, got {masses}"
    # The weight is the archive rank reciprocal, so it responds when the score moves the
    # endpoint past its neighbours. 0.95 overtakes both leaders; 0.01 is last.
    assert masses[-1] > masses[0] * 2, f"score must move mass materially, got {masses}"


def test_legacy_niche_allocation_is_score_insensitive():
    """The defect, pinned. `niche_score` is still live on T4 and must not silently move.

    A structurally isolated endpoint draws the same mass at 0.01 as at 0.80, because a
    niche's mass is uniform across niches and score only breaks ties WITHIN one.
    """
    low, high = _fragment_mass("niche_score", 0.01), _fragment_mass("niche_score", 0.80)
    assert low == pytest.approx(high, abs=1e-9), (
        "legacy niche_score is expected to be score-insensitive here; if this moved, the "
        "T4 niche_score arm changed behaviour and its campaigns must be re-checked"
    )


def test_evidence_allocation_prefers_the_better_of_two_admitted_endpoints():
    """Between two niche-admitted endpoints, the better-scoring one must draw more."""
    scores = {LEADER: 0.90, RUNNER_UP: 0.20}
    scores.update({fragment: 0.02 for fragment in FRAGMENTS})
    mass = _mass(_archive("niche_evidence", scores))
    assert mass[LEADER] > mass[RUNNER_UP]
    assert mass[LEADER] > max(mass[fragment] for fragment in FRAGMENTS)


def test_evidence_allocation_keeps_the_exploration_floor_intact():
    """No endpoint may be starved below the configured uniform floor."""
    scores = {LEADER: 0.90, RUNNER_UP: 0.85}
    scores.update({fragment: 0.0 for fragment in FRAGMENTS})
    optimizer = _archive("niche_evidence", scores)
    floor = optimizer.config.exploration / len(scores)
    for endpoint, value in _mass(optimizer).items():
        assert value >= floor * 0.999, f"{endpoint} fell below the exploration floor"


def test_niching_still_caps_how_many_endpoints_draw_exploitation_mass():
    """Evidence decides HOW MUCH; niching still decides WHO. Both halves are required.

    Without the niche mask the mode would collapse to `score_rank` and one structural
    basin could take every exploitation slot, which is the property niching exists to
    prevent.
    """
    scores = {f"C{'C' * index}O": 0.9 - index * 0.01 for index in range(12)}
    scores[LEADER] = 0.95
    optimizer = _archive("niche_evidence", scores)
    floor = optimizer.config.exploration / len(scores)
    mass = _mass(optimizer)
    above_floor = [e for e, v in mass.items() if v > floor * 1.001]
    assert len(above_floor) <= 4 * 2, (
        f"at most max_niches * per_niche endpoints may draw exploitation mass, "
        f"got {len(above_floor)}"
    )


def test_unknown_allocation_mode_is_refused():
    with pytest.raises(ValueError, match="parent allocation"):
        replace(ProgramSearchConfig(), parent_allocation="niche_whatever")


# ---- Defect 2: the plateau escape must drain ----


def _escape_pattern(improved_sequence) -> str:
    """Drive the production latch and the production decrement over a round sequence."""
    state = initial_population_state()
    pattern = []
    for improved in improved_sequence:
        active = state["escape_rounds_remaining"] > 0
        if active:  # the decrement `_allocate` performs, once per allocation
            state["escape_rounds_remaining"] -= 1
        pattern.append("E" if active else ".")
        advance_plateau_state(state, improved=improved)
    return "".join(pattern)


def test_escape_drains_on_an_unbroken_plateau():
    """An excursion must never exceed its declared length, however long the plateau."""
    pattern = _escape_pattern([False] * 40)
    longest = max(len(run) for run in pattern.split("."))
    assert longest <= ESCAPE_ROUNDS, f"escape latched for {longest} rounds: {pattern}"
    assert "." in pattern[pattern.index("E") :], "escape never released"


def test_escape_arms_only_after_the_declared_plateau():
    assert _escape_pattern([False] * PLATEAU_ROUNDS).count("E") == 0
    assert _escape_pattern([False] * (PLATEAU_ROUNDS + 1)).count("E") == 1


def test_improvement_clears_the_plateau_counter():
    state = initial_population_state()
    for _ in range(PLATEAU_ROUNDS - 1):
        advance_plateau_state(state, improved=False)
    advance_plateau_state(state, improved=True)
    assert state["rounds_without_improvement"] == 0
    assert state["escape_rounds_remaining"] == 0


# ---- Defect 3: the exploration bonus must be denominated in measured reward ----


@pytest.mark.parametrize("scale", [0.007, 0.05, 0.5])
def test_delivered_evidence_outranks_an_untried_cell_at_every_reward_scale(scale):
    """A cell that delivered one typical improvement must beat a cell with no evidence.

    Stated at three reward scales spanning two orders of magnitude, so a bonus sized in
    absolute units for any one of them fails the others.
    """
    credit = PopulationCredit()
    delivered = CreditKey("basin", "parent", "family", "refine")
    untried = CreditKey("basin", "parent", "family", "jump")
    credit.observe(delivered, scale)
    assert credit.value(delivered) > credit.value(untried)


def test_cold_start_allocation_is_uniform_and_unbiased():
    """With no evidence anywhere, every cell must still share the budget equally."""
    credit = PopulationCredit()
    keys = [CreditKey("b", "p", "f", scale) for scale in ("refine", "medium", "jump")]
    shares = credit.allocate(keys)
    assert shares == pytest.approx(np.full(len(keys), 1 / len(keys)))


def test_observed_scale_is_counted_not_configured():
    credit = PopulationCredit()
    assert credit.observed_scale() == 0.0
    credit.observe(CreditKey("b", "p", "f", "refine"), 0.10)
    credit.observe(CreditKey("b", "p", "f", "medium"), 0.0)
    assert credit.observed_scale() == pytest.approx(0.05)


def test_credit_snapshot_schema_refuses_the_reweighted_v1_payload():
    """`prior_weight` changed units, so a v1 snapshot must fail closed, not resume."""
    assert SCHEMA_VERSION == "pmo_credit_v2"
    with pytest.raises(ValueError, match="schema"):
        PopulationCredit.restore(
            {"schema_version": "pmo_credit_v1", "exploration_floor": 0.2,
             "prior_weight": 0.25, "cells": []}
        )


def test_parent_tilt_lets_measured_upside_outvote_novelty():
    """At PMO's measured reward scale a proven parent must outrank a never-tried one.

    With the bonus fixed at 0.25 absolute this required ~1000 children on one parent, so
    novelty won every comparison a 250-call task could actually produce.
    """
    scores = {LEADER: 0.50, RUNNER_UP: 0.50}
    scores.update({fragment: 0.50 for fragment in FRAGMENTS})
    optimizer = _archive("niche_evidence", scores)
    proven = next(k for k, v in optimizer.entries.items() if v["endpoint"] == LEADER)
    optimizer.population_state = initial_population_state()
    optimizer.population_state["parent_outcomes"] = {
        proven: {"children": 8, "positive_improvement_sum": 8 * 0.02}
    }
    optimizer.observed_upside_scale = PmoPopulationController.observed_upside_scale.__get__(
        optimizer
    )
    keys, weights = PmoPopulationController.selection(optimizer)
    mass = {optimizer.entries[k]["endpoint"]: w for k, w in zip(keys, weights, strict=True)}
    assert mass[LEADER] > mass[RUNNER_UP], (
        "a parent with eight measured wins must outrank an identically scored parent "
        "that has never been tried"
    )


def test_observed_upside_scale_is_zero_before_any_child_is_scored():
    optimizer = _archive("niche_evidence", {LEADER: 0.5, RUNNER_UP: 0.4})
    optimizer.population_state = initial_population_state()
    bound = PmoPopulationController.observed_upside_scale.__get__(optimizer)
    assert bound() == 0.0


# ---- Defect 2b: no reserved floor for the jump lane, but it stays reachable ----


def _bind_controller_methods(optimizer):
    """Bind the production controller methods `_allocate` reaches to this receiver.

    The methods themselves are never reimplemented; only their receiver is supplied.
    """
    for name in ("observed_upside_scale", "_fit_value", "_credit_allocate", "credit_report"):
        setattr(optimizer, name, getattr(PmoPopulationController, name).__get__(optimizer))
    return optimizer


def _candidate(index: int, channel: str, endpoint: str) -> dict:
    """A candidate row carrying exactly the fields `_allocate` reads."""
    return {
        "candidate_id": f"cand_{index}",
        "endpoint": endpoint,
        "fiber_features": [float(index), 1.0, 0.5],
        "fiber_fingerprint": [index, index + 1, index + 2],
        "provenance": {
            "planner_channel": channel,
            "entry_id": f"entry_{index % 2}",
            "parent_measured_score": 0.10,
        },
        "trace": {"actions": [{"executor_rule": "atom_insert"}]},
    }


def _allocation_detail(*, batches: int, escape: bool) -> dict:
    """Run the production `_allocate` over a pool holding all three channels."""
    optimizer = _archive("niche_evidence", {LEADER: 0.5, RUNNER_UP: 0.4})
    optimizer.population_state = initial_population_state()
    if escape:
        optimizer.population_state["escape_rounds_remaining"] = ESCAPE_ROUNDS
    optimizer.batches = batches
    optimizer.credit = PopulationCredit()
    _bind_controller_methods(optimizer)
    pool = []
    for index, channel in enumerate(CHANNELS * 6):
        pool.append(_candidate(index, channel, LEADER if index % 2 else RUNNER_UP))
    chosen, detail = PmoPopulationController._allocate(optimizer, pool)
    assert detail["mode"] != "empty_pool", "the guard must exercise a real allocation"
    assert chosen, "the guard must exercise a real allocation"
    return detail


@pytest.mark.parametrize(
    ("label", "batches", "escape"),
    [("early", 0, False), ("normal", 9, False), ("escape", 9, True)],
)
def test_jump_lane_holds_no_reserved_floor_in_any_quota_regime(label, batches, escape):
    """Driven through the production `_allocate`, in all three regimes it governs."""
    detail = _allocation_detail(batches=batches, escape=escape)
    assert detail["quota_by_channel"][JUMP_CHANNEL] == 0, (
        f"{label} regime still reserves "
        f"{detail['quota_by_channel'][JUMP_CHANNEL]} jump slots"
    )
    assert detail["available_by_channel"][JUMP_CHANNEL] > 0, "the pool must offer jump rows"


def test_jump_lane_is_still_reachable_without_a_reserved_floor():
    """Removing the floor must not remove the channel: jump rows stay selectable."""
    optimizer = _archive("niche_evidence", {LEADER: 0.5, RUNNER_UP: 0.4})
    optimizer.population_state = initial_population_state()
    optimizer.batches = 9
    optimizer.credit = PopulationCredit()
    _bind_controller_methods(optimizer)
    pool = [_candidate(index, JUMP_CHANNEL, LEADER) for index in range(6)]
    chosen, detail = PmoPopulationController._allocate(optimizer, pool)
    assert detail["quota_by_channel"][JUMP_CHANNEL] == 0
    assert [row for row in chosen if row["provenance"]["planner_channel"] == JUMP_CHANNEL], (
        "a jump candidate must remain selectable through the evidence path"
    )


def test_credit_floor_guarantees_a_jump_cell_positive_share():
    """The recoverable floor: a jump cell always keeps at least `eps / n` of budget."""
    credit = PopulationCredit()
    refine = CreditKey("basin", "parent", "family", "refine")
    jump = CreditKey("basin", "parent", "family", "jump")
    for _ in range(50):
        credit.observe(refine, 0.20)
    credit.observe(jump, 0.0)
    shares = credit.allocate([refine, jump])
    assert shares[1] >= credit.exploration_floor / 2 * 0.999
    assert shares[0] > shares[1]


# ---- Centre selection: a niche must be seeded on a PROMISING basin ----


def test_evidence_centres_prefer_scoring_molecules_over_distant_fragments():
    """The fragments are the most distant things in the archive, so distance alone picks
    them. A centre must earn its niche by carrying evidence as well as by being far."""
    from compose_v4.control.edit_learning_data import evidence_niches, exploration_niches

    scores = {LEADER: 0.90, RUNNER_UP: 0.85, "c1ccc(CCN)cc1": 0.70, "CC(=O)Nc1ccccc1O": 0.65}
    scores.update({fragment: 0.01 for fragment in FRAGMENTS})
    endpoints = sorted(scores)
    utilities = [scores[e] for e in endpoints]
    distance_centres = exploration_niches(endpoints, utilities)["centers"]
    evidence_centres = evidence_niches(endpoints, utilities)["centers"]
    assert sum(c in FRAGMENTS for c in evidence_centres) < sum(
        c in FRAGMENTS for c in distance_centres
    ), (
        f"evidence centres {evidence_centres} must seed fewer fragment niches than "
        f"distance centres {distance_centres}"
    )


def test_evidence_niches_degrade_to_max_min_when_utility_is_flat():
    """With no spread in utility there is no evidence to weight by, so the rule must
    reduce to the pure separation rule rather than collapsing onto one centre."""
    from compose_v4.control.edit_learning_data import evidence_niches, exploration_niches

    endpoints = sorted({LEADER, RUNNER_UP, *FRAGMENTS})
    flat = [0.5] * len(endpoints)
    assert evidence_niches(endpoints, flat)["centers"] == exploration_niches(
        endpoints, flat
    )["centers"]


def test_legacy_niche_score_still_uses_the_distance_only_partition():
    """T4 arms run on niche_score; its partition must not move."""
    import compose_v4.control.adaptive_program_optimizer as module

    calls = []
    original = module.exploration_niches
    module.exploration_niches = lambda *a, **k: calls.append("used") or original(*a, **k)
    try:
        _mass(_archive("niche_score", {LEADER: 0.9, RUNNER_UP: 0.8, FRAGMENTS[0]: 0.01}))
    finally:
        module.exploration_niches = original
    assert calls, "niche_score must still route through exploration_niches"


# ---- The allocator must move off a parent whose children stop working ----


def test_parent_mass_decays_as_children_fail_to_improve():
    """A stalled parent must lose mass. Otherwise the plateau defect returns as a latch.

    With the tilt written as `1 + upside + bonus`, both evidence terms sit at the reward
    scale (~0.007) against a base of 1, so 64 unproductive children moved a real parent's
    mass by 0.05-0.27% -- indistinguishable from a working parent.
    """
    scores = {LEADER: 0.90, RUNNER_UP: 0.88}
    scores.update({fragment: 0.40 for fragment in FRAGMENTS})
    masses = []
    for children in (0, 8, 64):
        optimizer = _archive("niche_evidence", scores)
        optimizer.population_state = initial_population_state()
        leader_id = next(k for k, v in optimizer.entries.items() if v["endpoint"] == LEADER)
        other_id = next(k for k, v in optimizer.entries.items() if v["endpoint"] == RUNNER_UP)
        optimizer.population_state["parent_outcomes"] = {
            # a second parent supplies the measured scale; the leader delivers nothing
            other_id: {"children": 8, "positive_improvement_sum": 8 * 0.02},
            leader_id: {"children": children, "positive_improvement_sum": 0.0},
        }
        _bind_controller_methods(optimizer)
        keys, weights = PmoPopulationController.selection(optimizer)
        masses.append(
            sum(
                float(w)
                for k, w in zip(keys, weights, strict=True)
                if optimizer.entries[k]["endpoint"] == LEADER
            )
        )
    assert masses[0] > masses[1] > masses[2], f"a stalled parent must lose mass, got {masses}"
    assert (masses[0] - masses[2]) / masses[0] > 0.05, (
        f"the decay must be material, not a rounding artifact: {masses}"
    )


def test_delivered_upside_still_outranks_a_stalled_parent():
    """The other direction: damping a stalled parent must not damp a productive one."""
    scores = {LEADER: 0.90, RUNNER_UP: 0.90}
    scores.update({fragment: 0.40 for fragment in FRAGMENTS})
    optimizer = _archive("niche_evidence", scores)
    optimizer.population_state = initial_population_state()
    working = next(k for k, v in optimizer.entries.items() if v["endpoint"] == LEADER)
    stalled = next(k for k, v in optimizer.entries.items() if v["endpoint"] == RUNNER_UP)
    optimizer.population_state["parent_outcomes"] = {
        working: {"children": 20, "positive_improvement_sum": 20 * 0.03},
        stalled: {"children": 20, "positive_improvement_sum": 0.0},
    }
    _bind_controller_methods(optimizer)
    keys, weights = PmoPopulationController.selection(optimizer)
    mass = {}
    for k, w in zip(keys, weights, strict=True):
        endpoint = optimizer.entries[k]["endpoint"]
        mass[endpoint] = mass.get(endpoint, 0.0) + float(w)
    assert mass[LEADER] > mass[RUNNER_UP] * 1.5, (
        "identically scored parents must separate on whether their children delivered"
    )


def test_evidence_mode_actually_routes_through_the_evidence_partition():
    """Wiring, not definition. A repaired partition that no caller reaches is inert.

    The companion test checks `evidence_niches` behaves; this checks the production
    selection REACHES it. A mutation that points the evidence mode back at the
    distance-only partition passes every behavioural test and fails only this one.
    """
    import compose_v4.control.adaptive_program_optimizer as module

    seen = []
    original = module.evidence_niches
    module.evidence_niches = lambda *a, **k: seen.append("used") or original(*a, **k)
    try:
        _mass(_archive("niche_evidence", {LEADER: 0.9, RUNNER_UP: 0.8, FRAGMENTS[0]: 0.01}))
    finally:
        module.evidence_niches = original
    assert seen, "niche_evidence must route through evidence_niches, not the legacy partition"
