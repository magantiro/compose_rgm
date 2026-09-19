import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.t4_cold_start_allocation import cold_start_allocation_v1
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_protonation_integrated_route_fiber import (
    EXPERTS,
    PROTONATION_EXPERT,
    integrated_features,
    merge_expert_pools,
    select_batch,
)
from compose_v4.experiments.t4_protonation_rescue_contract import (
    PREPARED_STATUS,
    validate_rescue_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1.json"
APP = ROOT / "modal_apps/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1_app.py"


def _row(smiles: str, expert: str, *, score: float = -8.0) -> dict:
    return {
        "smiles": smiles,
        "proposal_lane": expert,
        "proposal_experts": [expert],
        "parent": "P",
        "parent_score": score,
        "similarity": 0.7,
        "delta": 0.6,
        "qed": 0.7,
        "sa": 3.0,
        "regions": 1,
        "created": 1,
        "deleted": 0,
        "families": [expert],
        "program_families": [expert],
    }


def test_policy_adds_exactly_one_general_expert_and_feature():
    assert EXPERTS == (
        "shallow",
        "anchored_replacement",
        "route_complete_region",
        "protonation_aware_retained_subgraph",
    )
    state = SearchState(archive={"P": -8.0})
    vectors = [integrated_features(_row(expert, expert), state) for expert in EXPERTS]
    assert all(len(vector) == 20 and vector[-1] == 1.0 for vector in vectors)
    assert len({tuple(vector) for vector in vectors}) == 4


def test_merge_preserves_protonation_provenance_and_best_parent():
    shallow = _row("C", "shallow", score=-7.0)
    protonation = _row("C", PROTONATION_EXPERT, score=-9.0)
    result = merge_expert_pools(
        {
            "shallow": [shallow],
            "anchored_replacement": [],
            "route_complete_region": [],
            PROTONATION_EXPERT: [protonation],
        }
    )
    assert len(result) == 1
    assert result[0]["parent_score"] == -9.0
    assert result[0]["proposal_experts"] == [PROTONATION_EXPERT, "shallow"]


def test_early_batch_guarantees_each_available_expert():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for expert_index, expert in enumerate(EXPERTS):
        for position in range(3):
            row = _row(f"{expert_index}-{position}", expert)
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {expert_index * 10 + position}
            candidates.append(row)
    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(17),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=2,
    )
    floor = [row for row in selected if row["selection_kind"] == "expert_floor"]
    assert {row["proposal_lane"] for row in floor} == set(EXPERTS)
    assert len({row["smiles"] for row in selected}) == len(selected) == 8


def test_versioned_floor_can_allocate_more_sparse_data_observations():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for expert_index, expert in enumerate(EXPERTS):
        for position in range(4):
            row = _row(f"{expert_index}-{position}", expert)
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {expert_index * 10 + position}
            candidates.append(row)
    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(23),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        expert_floor_counts={PROTONATION_EXPERT: 2},
    )
    floor = [row for row in selected if row["selection_kind"] == "expert_floor"]
    counts = {
        expert: sum(expert in row["proposal_experts"] for row in floor)
        for expert in EXPERTS
    }
    assert counts == {
        "shallow": 0,
        "anchored_replacement": 0,
        "route_complete_region": 0,
        PROTONATION_EXPERT: 2,
    }


@pytest.mark.parametrize("include_protonation", [True, False])
def test_frozen_cold_start_rule_uses_last_slot_for_protonation_or_exploration(
    include_protonation,
):
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for band_index, band in enumerate(("small", "medium", "large")):
        for rank in (1, 3, 6, 9):
            row = _row(f"route-{band_index}-{rank}", "route_complete_region")
            row["realized_primitive_band"] = band
            row["route_proposal_rank"] = rank
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {100 + band_index * 10 + rank}
            candidates.append(row)
    for expert_index, expert in enumerate(("shallow", "anchored_replacement")):
        row = _row(expert, expert)
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = {200 + expert_index}
        candidates.append(row)
    if include_protonation:
        row = _row("protonation", PROTONATION_EXPERT)
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = {300}
        candidates.append(row)

    available_experts = tuple(
        expert
        for expert in EXPERTS
        if include_protonation or expert != PROTONATION_EXPERT
    )
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=8,
        available_experts=available_experts,
    )
    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(29),
        round_index=1,
        batch=8,
        exploration=plan.exploration_slots,
        expert_floor_rounds=1,
        expert_floor_counts=plan.expert_floor_counts,
        route_scale_floor_rounds=1,
        route_scale_floor_counts=plan.route_scale_floor_counts,
    )
    assert len(selected) == 8
    route_floor = [
        row for row in selected if row["selection_kind"] == "route_scale_floor"
    ]
    assert [row["realized_primitive_band"] for row in route_floor].count("small") == 1
    assert [row["realized_primitive_band"] for row in route_floor].count("medium") == 1
    assert [row["realized_primitive_band"] for row in route_floor].count("large") == 3
    if include_protonation:
        assert (
            sum(
                PROTONATION_EXPERT in row["proposal_experts"]
                and row["selection_kind"] == "expert_floor"
                for row in selected
            )
            == 1
        )
    else:
        assert any(row["selection_kind"] == "exploration" for row in selected)


@pytest.mark.parametrize(
    "counts", [{"unknown": 1}, {PROTONATION_EXPERT: -1}, {PROTONATION_EXPERT: True}]
)
def test_expert_floor_rejects_invalid_versioned_allocations(counts):
    with pytest.raises(ValueError, match="expert floor"):
        select_batch(
            [],
            ProgramValue(),
            SearchState(archive={"P": -8.0}),
            np.random.default_rng(3),
            round_index=1,
            batch=8,
            exploration=2,
            expert_floor_rounds=2,
            expert_floor_counts=counts,
        )


def test_contract_is_one_fresh_49_call_rescue_and_not_authorized_yet():
    contract = unseal(CONTRACT)
    assert contract["status"] == PREPARED_STATUS
    assert [(row["cell"], row["source_global_index"]) for row in contract["cells"]] == [
        ("5ht1b_2", 8)
    ]
    assert contract["charged_calls_per_cell"] == 49
    assert contract["total_charged_call_ceiling"] == 49
    assert contract["automatic_retries"] == 0
    assert contract["launch_authorization"]["state"] == "REQUIRED_AND_NOT_YET_RECORDED"
    assert sorted(contract["proposal"]) == sorted(EXPERTS)


def test_preflight_is_fail_closed_and_hash_bound():
    prepared = validate_rescue_preflight(ROOT, CONTRACT, require_clean_runtime=False)
    assert prepared["status"] == "PREPARED_NO_SCORED_AUTHORIZATION"
    assert prepared["zero_oracle_primary_eligible"] == 14
    with pytest.raises(ValueError, match="authorization"):
        validate_rescue_preflight(
            ROOT,
            CONTRACT,
            authorization_payload_sha256="0" * 64,
            require_clean_runtime=False,
        )
    authorized = validate_rescue_preflight(
        ROOT,
        CONTRACT,
        authorization_payload_sha256=prepared["contract_payload_sha256"],
        require_clean_runtime=False,
    )
    assert authorized["status"] == "LAUNCH_AUTHORIZATION_BOUND"


def test_app_uses_additive_expert_and_fail_closed_continuation():
    source = APP.read_text()
    assert 'elif expert == "protonation_aware_retained_subgraph"' in source
    assert "propose_protonation_aware_candidates(" in source
    assert "cold_start_allocation_v1(" in source
    assert "available_experts = sorted(" in source
    assert '"allocation_plan": _jsonable(allocation_plan.__dict__)' in source
    assert 'campaign_driver_action(statuses, continuation_state="terminal")' in source
    assert source.index('"state": "reserved"') < source.index(
        "continuation = drive.spawn("
    )
    assert "a separately authorized contract payload SHA-256 is required" in source
    assert 'mode: str = "preflight"' in source


def test_contract_contains_no_known_endpoint_or_runtime_winner_map():
    contract = unseal(CONTRACT)
    serialized = json.dumps(contract, sort_keys=True).lower()
    assert "cell-to-winner lookup" in serialized  # explicit prohibition only
    assert "teacher candidate injection" in serialized  # explicit prohibition only
    for forbidden_key in (
        "known_endpoint",
        "teacher_endpoint",
        "winner_smiles",
        "target_to_program",
        "cell_to_winner",
    ):
        assert forbidden_key not in contract
