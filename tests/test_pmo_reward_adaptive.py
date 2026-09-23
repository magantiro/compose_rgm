"""The feedback loop is the deliverable, so these tests assert BEHAVIOUR CHANGE, not fitting."""

from __future__ import annotations

import numpy as np

from compose_v4.control.pmo_reward_adaptive import (
    MINIMUM_OBSERVATIONS,
    RewardAdaptiveProgramController,
    RunningScale,
)


def _edge(parent, endpoint, *, families=("region_replace",), parent_score=0.2, primitives=11):
    return {
        "parent": parent, "endpoint": endpoint, "smiles": endpoint,
        "families": list(families), "requested_modules": 2, "module_count": 2,
        "primitives": primitives, "depth": 1, "generation": 1,
        "capacity_aware": False, "parent_score": parent_score,
    }


def test_the_learning_target_is_the_delta_not_the_endpoint():
    control = RewardAdaptiveProgramController()
    control.observe(_edge("CCO", "CCOC", parent_score=0.80), 0.85)
    record = control.observations[-1]
    assert np.isclose(record["delta_raw"], 0.05), "the training target must be u(child) - u(parent)"
    assert np.isclose(record["score"], 0.85), "the raw endpoint utility is stored separately"


def test_cross_parent_decisions_restore_the_parent_offset():
    """The T4 correction. A weak parent with a large predicted delta must NOT outrank a strong
    parent whose endpoint is genuinely better -- that defect was caught mid-campaign in T4.
    """
    control = RewardAdaptiveProgramController()
    rng = np.random.default_rng(0)
    for i in range(60):
        parent = 0.1 if i % 2 else 0.7
        control.observe(
            _edge(f"C{'C' * (i % 8)}O", f"C{'C' * (i % 8)}OC", parent_score=parent),
            parent + 0.05,
        )
    weak = _edge("CCCCO", "CCCCOC", parent_score=0.10)
    strong = _edge("CCCCO", "CCCCOC", parent_score=0.70)
    values = control._endpoint_hat(control.post, [weak, strong])
    assert values[1] > values[0], (
        "the candidate from the stronger parent must win on predicted ENDPOINT utility"
    )
    del rng


def test_a_reward_observation_changes_the_next_round_proposal_policy():
    """THE deliverable. If observations are logged but the policy does not move, this fails."""
    control = RewardAdaptiveProgramController(temperature=0.05)
    intents = [
        _edge("CCO", "CCOC", families=("region_replace",), parent_score=0.2),
        _edge("c1ccccc1", "c1ccccc1C", families=("append_ring",), parent_score=0.2),
        _edge("CCCCO", "CCCCOC", families=("segment_grow",), parent_score=0.2),
    ]
    for i in range(MINIMUM_OBSERVATIONS + 6):
        control.observe(_edge(f"C{'C' * (i % 5)}O", f"C{'C' * (i % 5)}OC"), 0.21)
    before = control.record_policy(intents, "before")

    # A strongly positive outcome for ONE context, repeated so it is not a single-point fluke.
    for _ in range(25):
        control.observe(
            _edge("c1ccccc1", "c1ccccc1CCN", families=("append_ring",), parent_score=0.2), 0.95
        )
    after = control.record_policy(intents, "after")

    shift = control.policy_shift(before, after)
    assert shift > 0.01, (
        f"reward did not move the proposal policy (total-variation shift {shift:.4f}); "
        "observations are being logged without changing future behaviour"
    )


def test_repeated_negative_outcomes_reduce_allocation():
    control = RewardAdaptiveProgramController(temperature=0.05)
    intents = [
        _edge("CCO", "CCOC", families=("region_replace",), parent_score=0.5),
        _edge("c1ccccc1", "c1ccccc1C", families=("append_ring",), parent_score=0.5),
    ]
    for i in range(MINIMUM_OBSERVATIONS + 6):
        control.observe(_edge(f"C{'C' * (i % 5)}O", f"C{'C' * (i % 5)}OC", parent_score=0.5), 0.51)
    before = control.record_policy(intents, "before")
    punished = before["parent_mass"]["c1ccccc1"]
    for _ in range(30):
        control.observe(
            _edge("c1ccccc1", "c1ccccc1O", families=("append_ring",), parent_score=0.5), 0.05
        )
    after = control.record_policy(intents, "after")
    assert after["parent_mass"]["c1ccccc1"] < punished, (
        "a context that repeatedly loses reward must lose proposal mass"
    )


def test_no_legal_family_or_live_lineage_is_ever_starved():
    control = RewardAdaptiveProgramController(temperature=0.01)
    intents = [
        _edge("CCO", "CCOC", families=("region_replace",)),
        _edge("c1ccccc1", "c1ccccc1C", families=("append_ring",)),
    ]
    for i in range(60):
        control.observe(
            _edge("CCO", f"CCO{'C' * (i % 6)}", families=("region_replace",)), 0.99
        )
    weights = control.intent_policy(intents)
    assert min(weights) > 0.0, "no legal macro intent may reach zero proposal mass"
    mass = control.record_policy(intents, "floored")["parent_mass"]
    assert min(mass.values()) > 0.0, "no live lineage may reach zero"


def test_an_unfitted_controller_is_uniform_and_selects_at_random():
    control = RewardAdaptiveProgramController()
    intents = [_edge("CCO", "CCOC"), _edge("c1ccccc1", "c1ccccc1C")]
    assert np.allclose(control.intent_policy(intents), 0.5)
    chosen, detail = control.acquire(intents, 0.3, batch=2, rng=np.random.default_rng(0))
    assert {d["reason"] for d in detail} == {"cold_start"}, "an unfitted model must not choose"


def test_the_batch_keeps_a_quarter_for_genuine_exploration():
    control = RewardAdaptiveProgramController(model_share=0.75)
    for i in range(60):
        control.observe(_edge(f"C{'C' * (i % 7)}O", f"C{'C' * (i % 7)}OC"), i / 60)
    pool = [_edge("CCO", f"CC{'N' * (i % 6)}O") for i in range(64)]
    chosen, detail = control.acquire(pool, 0.1, batch=16, rng=np.random.default_rng(3))
    reasons = [d["reason"] for d in detail]
    assert len(chosen) == 16
    assert reasons.count("model") == 12 and reasons.count("random") == 4
    assert all("propensity" in d for d in detail), "every selection must log its propensity"


def test_standardisation_uses_only_past_observations():
    scale = RunningScale()
    for value in (0.1, 0.2, 0.3):
        scale.update(value)
    assert np.isclose(scale.mean, 0.2)
    restored = scale.restore(scale.standardize([0.1, 0.2, 0.3]))
    assert np.allclose(restored, [0.1, 0.2, 0.3]), "the transform must round-trip to raw units"
