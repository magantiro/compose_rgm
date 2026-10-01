"""Behavior-level checks for the real PMO and T4 batch selectors, without oracles."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

# Reuse the checkpoint and exact one-edit program fixtures, not scoring logic.
from test_program_reference import (
    checkpoint as checkpoint,  # noqa: PLC0414 -- pytest fixture re-export
)
from test_program_reference import loaded as loaded  # noqa: PLC0414 -- pytest fixture re-export
from test_program_reference import programs

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.pmo_reward_adaptive import (
    MINIMUM_OBSERVATIONS,
    RewardAdaptiveProgramController,
)
from compose_v4.control.reference_guidance import (
    GuidanceConfig,
    ProgramScore,
    ScoredPanel,
    guide_panel,
)
from compose_v4.control.reference_programs import FrozenProgramReference, ProgramPanelGuidance
from compose_v4.control.reference_selection import draw_exploration
from compose_v4.experiments.t4_integrated_route_fiber import EXPERTS, select_batch


def make_guide(mode, *, missing=False, identity="candidate_id", unsupported="baseline"):
    config = GuidanceConfig(
        mode=mode, strength=0.5 if mode == "active" else 0, on_unsupported=unsupported
    )

    def guide(rows):
        ids = tuple(row[identity] for row in rows)

        def score():
            assert mode != "off", "off mode must not evaluate a reference"
            return ScoredPanel(
                "a" * 64,
                tuple(
                    ProgramScore(key, None, "missing_trace", "fixture missing trace")
                    if missing and index == 0
                    else ProgramScore(key, -float(index + 1), "scored")
                    for index, key in enumerate(ids)
                ),
                progress=0.5,
            )

        return guide_panel(ids, [1 / len(ids)] * len(ids), config=config, score=score)

    return guide


@pytest.mark.parametrize("mode,missing", [("off", False), ("shadow", False), ("active", True)])
def test_unguided_draws_preserve_exact_selection_and_rng(mode, missing):
    rows = [{"candidate_id": str(i)} for i in range(9)]
    for seed in range(12):
        expected_rng, actual_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        expected = expected_rng.choice(len(rows), 4, replace=False).tolist()
        receipts = []
        actual = draw_exploration(
            rows, 4, actual_rng, guide=make_guide(mode, missing=missing), receipts=receipts
        )
        assert actual == expected
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
        assert not receipts[0]["probabilities_changed"]


def test_active_draw_is_bounded_without_replacement_and_records_conditionals():
    rows = [{"candidate_id": str(i)} for i in range(7)]
    receipts = []
    chosen = draw_exploration(
        rows, 4, np.random.default_rng(23), guide=make_guide("active"), receipts=receipts
    )
    record = receipts[0]
    assert record["probabilities_changed"]
    assert len(set(chosen)) == 4
    assert min(record["probabilities"]) > 0
    assert max(record["probabilities"]) / min(record["probabilities"]) <= np.exp(1)
    remaining = list(record["probabilities"])
    for index, conditional in zip(chosen, record["conditional_draw_probabilities"], strict=True):
        assert conditional == pytest.approx(remaining[index] / sum(remaining))
        remaining[index] = 0
    assert record["selected_ids"] == [str(i) for i in chosen]


def test_partial_coverage_draw_keeps_unscored_candidates_and_exact_first_draw_mass():
    rows = [{"candidate_id": str(i)} for i in range(7)]
    receipts = []
    chosen = draw_exploration(
        rows,
        4,
        np.random.default_rng(23),
        guide=make_guide("active", missing=True, unsupported="preserve_mass"),
        receipts=receipts,
    )
    record = receipts[0]
    assert record["outcome"] == "active_partial"
    assert record["candidate_ids"] == tuple(str(i) for i in range(7))
    assert record["probabilities"][0] == 1 / 7
    assert record["probabilities_changed"]
    assert record["coverage"]["scored_count"] == 6
    assert len(set(chosen)) == 4
    assert all(value > 0 for value in record["probabilities"])


@pytest.mark.parametrize("count", [-1, 3, True, 1.5])
def test_bad_count_fails_before_rng(count):
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="count"):
        draw_exploration([{}, {}], count, rng)
    assert rng.bit_generator.state == before


def test_zero_draw_is_a_noop():
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    assert draw_exploration([], 0, rng, guide=lambda _: pytest.fail("unused scorer")) == []
    assert rng.bit_generator.state == before


def test_guidance_requires_a_receipt_and_a_valid_panel():
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="receipt sink"):
        draw_exploration(rows, 1, rng, guide=make_guide("active"))
    with pytest.raises(TypeError, match="GuidedPanel"):
        draw_exploration(rows, 1, rng, guide=lambda _: None, receipts=[])
    wrong_law = lambda _: guide_panel(("a", "b"), (0.2, 0.8))
    with pytest.raises(ValueError, match="actual uniform baseline"):
        draw_exploration(rows, 1, rng, guide=wrong_law, receipts=[])
    assert rng.bit_generator.state == before


def test_strict_coverage_failure_does_not_sample_or_publish_a_receipt():
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    rng, receipts = np.random.default_rng(2), []
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="coverage incomplete"):
        draw_exploration(
            rows,
            1,
            rng,
            guide=make_guide("active", missing=True, unsupported="error"),
            receipts=receipts,
        )
    assert rng.bit_generator.state == before
    assert receipts == []


def pmo_case(fitted):
    control = RewardAdaptiveProgramController()
    if fitted:
        control.observations = [{} for _ in range(MINIMUM_OBSERVATIONS)]
        control._endpoint_hat = lambda model, rows: np.asarray([row["test_value"] for row in rows])
    rows = [{"candidate_id": str(i), "test_value": i / 20} for i in range(16)]
    return control, rows


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_pmo_off_shadow_preserve_batch_details_state_and_rng(fitted, mode):
    control, rows = pmo_case(fitted)
    baseline_rng, rng = np.random.default_rng(23), np.random.default_rng(23)
    baseline = control.acquire(rows, 0.1, batch=8, rng=baseline_rng)
    before = copy.deepcopy(rows)
    receipts = []
    actual = control.acquire(
        rows,
        0.1,
        batch=8,
        rng=rng,
        reference_guide=make_guide(mode),
        reference_receipts=receipts,
    )
    assert actual == baseline
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    assert rows == before
    assert receipts[0]["outcome"] == mode


def test_pmo_active_preserves_model_selections_and_quota():
    control, rows = pmo_case(True)
    baseline, details = control.acquire(rows, 0.1, batch=8, rng=np.random.default_rng(23))
    receipts = []
    selected, guided_details = control.acquire(
        rows,
        0.1,
        batch=8,
        rng=np.random.default_rng(23),
        reference_guide=make_guide("active"),
        reference_receipts=receipts,
    )
    assert selected[:6] == baseline[:6]
    assert guided_details[:6] == details[:6]
    assert len(set(selected)) == 8
    assert receipts[0]["requested_count"] == 2
    assert receipts[0]["probabilities_changed"]
    assert set(receipts[0]["candidate_ids"]).isdisjoint(str(i) for i in selected[:6])
    assert all(row["propensity"] is None for row in guided_details[6:])


def t4_case(fitted):
    state = SearchState(archive={"root": -8.0})
    value = ProgramValue()
    if fitted:
        value.weights = np.ones(2)
        value.predict = lambda x: x[:, 0]
    rows = [
        {
            "smiles": str(i),
            "features": [i / 20, 1],
            "fingerprint": {i},
            "parent_score": -8.0,
            "proposal_lane": EXPERTS[i % 3],
            "realized_primitive_band": ("small", "medium", "large")[(i // 3) % 3],
            "route_proposal_rank": i,
        }
        for i in range(18)
    ]
    return rows, value, state


def choose_t4(rows, value, state, rng, **kwargs):
    return select_batch(
        rows,
        value,
        state,
        rng,
        round_index=1,
        batch=10,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=2,
        **kwargs,
    )


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_t4_off_shadow_preserve_floors_selections_and_rng(fitted, mode):
    rows, value, state = t4_case(fitted)
    baseline_rng, rng = np.random.default_rng(23), np.random.default_rng(23)
    baseline = choose_t4(rows, value, state, baseline_rng)
    receipts = []
    actual = choose_t4(
        rows,
        value,
        state,
        rng,
        reference_guide=make_guide(mode, identity="smiles"),
        reference_receipts=receipts,
    )
    assert actual == baseline
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    assert receipts[0]["outcome"] == mode


def test_t4_active_preserves_expert_scale_and_model_slots():
    rows, value, state = t4_case(True)
    baseline = choose_t4(rows, value, state, np.random.default_rng(23))
    receipts = []
    actual = choose_t4(
        rows,
        value,
        state,
        np.random.default_rng(23),
        reference_guide=make_guide("active", identity="smiles"),
        reference_receipts=receipts,
    )
    fixed = lambda chosen: [row for row in chosen if row["selection_kind"] != "exploration"]
    assert fixed(actual) == fixed(baseline)
    assert len({row["smiles"] for row in actual}) == 10
    assert receipts[0]["requested_count"] == 2
    assert receipts[0]["probabilities_changed"]


def test_program_join_preserves_subset_order_and_rejects_unknown_ids():
    exact = programs()
    seen = []

    def score(items):
        seen.extend(item.candidate_id for item in items)
        return ScoredPanel(
            "a" * 64, tuple(ProgramScore(p.candidate_id, -1, "scored") for p in items), 0.5
        )

    guide = ProgramPanelGuidance(exact, SimpleNamespace(score=score), GuidanceConfig("shadow"))
    panel = guide([{"candidate_id": "N"}, {"candidate_id": "O"}])
    assert panel.candidate_ids == ("N", "O")
    assert seen == ["N", "O"]
    with pytest.raises(ValueError, match="no exact program"):
        guide([{"candidate_id": "unknown"}])
    with pytest.raises(ValueError, match="nonempty"):
        guide([])
    with pytest.raises(ValueError, match="duplicate"):
        ProgramPanelGuidance((exact[0], exact[0]), None, GuidanceConfig())


def test_real_frozen_model_guides_both_actual_selectors(loaded):
    reference = FrozenProgramReference(loaded)
    exact = tuple(replace(p, candidate_id=p.endpoint) for p in programs())
    config = GuidanceConfig("active", strength=0.5)
    rows = [
        {
            "candidate_id": p.candidate_id,
            "smiles": p.endpoint,
            "features": [1.0, 1.0],
            "fingerprint": {i},
        }
        for i, p in enumerate(exact)
    ]
    pmo_receipts, t4_receipts = [], []
    RewardAdaptiveProgramController().acquire(
        rows,
        0.1,
        batch=1,
        rng=np.random.default_rng(9),
        reference_guide=ProgramPanelGuidance(exact, reference, config),
        reference_receipts=pmo_receipts,
    )
    select_batch(
        rows,
        ProgramValue(),
        SearchState(),
        np.random.default_rng(9),
        round_index=1,
        batch=1,
        exploration=1,
        expert_floor_rounds=0,
        reference_guide=ProgramPanelGuidance(exact, reference, config, identity_field="smiles"),
        reference_receipts=t4_receipts,
    )
    assert pmo_receipts == t4_receipts
    assert pmo_receipts[0]["probabilities_changed"]
    assert all(p.grad is None and not p.requires_grad for p in loaded.model.parameters())
