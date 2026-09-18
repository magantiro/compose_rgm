from pathlib import Path

import pytest

from compose_v4.experiments.pmo_route_fiber_prelaunch import (
    TASKS,
    build_prelaunch,
    contract_envelope,
    control_score_to_reward,
    reward_to_control_score,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pmo_reward_coordinate_round_trip():
    for reward in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert control_score_to_reward(reward_to_control_score(reward)) == reward
    assert reward_to_control_score(0.9) < reward_to_control_score(0.2)


def test_pmo_reward_coordinate_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="PMO reward"):
        reward_to_control_score(1.01)
    with pytest.raises(ValueError, match="control score"):
        control_score_to_reward(0.1)


def test_prelaunch_fails_closed_until_a_production_proposer_exists():
    contract = contract_envelope()
    assert contract["payload"]["oracle_calls_authorized"] == 0
    assert contract["payload"]["scored_pilot_requires_separate_authorization"] is True
    result = build_prelaunch(ROOT)
    payload = result["payload"]
    assert payload["new_oracle_calls"] == 0
    assert payload["scored_launch_authorized"] is False
    assert payload["decision"] == ("BLOCKED_MISSING_SPLIT_CLEAN_PRODUCTION_ROUTE_PROPOSER")
    assert payload["contract_payload_sha256"] == contract["payload_sha256"]
    assert payload["initialization"] == {
        "count": 16,
        "task_information_present": False,
    }
    for task in TASKS:
        fold = payload["leave_one_task_out_audit"]["folds"][task]
        assert fold["held_task_absent_from_training"] is True
        assert fold["lineage_leakage"] == 0
        assert fold["held_decisions"] > 0
        assert fold["train_decisions"] > 0
