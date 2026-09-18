import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
    ARMS,
    CONTRACT,
    INPUTS,
    TASKS,
    envelope,
    sha256_file,
    verify_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_self_hashed_and_binds_sealed_inputs():
    contract = verify_contract(ROOT, ROOT / CONTRACT)
    assert contract["payload_sha256"] == identity(contract["payload"])
    assert contract == envelope(ROOT)
    for relative, expected in INPUTS.items():
        assert sha256_file(ROOT / relative) == expected


def test_factorial_budget_and_loto_folds_are_frozen_without_launch_authority():
    payload = json.loads((ROOT / CONTRACT).read_text())["payload"]
    assert payload["tasks"] == TASKS
    assert payload["arms"] == ARMS
    assert payload["tasks"]["gsk3b"]["fold"] == 0
    assert payload["tasks"]["perindopril_mpo"]["fold"] == 2
    assert payload["budget"] == {
        "charged_calls_per_task_arm": 48,
        "initialization_calls_per_task_arm": 16,
        "post_initialization_rounds": 4,
        "calls_per_round": 8,
        "task_arm_units": 8,
        "total_charged_call_ceiling": 384,
        "automatic_retries": 0,
        "failed_or_unresolved_reservation_is_charged": True,
        "replacement_or_backfill_after_scoring": False,
    }
    assert payload["oracle_calls_authorized"] == 0
    assert payload["scored_launch_authorized"] is False
    assert payload["modal_launch_authorized"] is False


def test_candidate_locks_and_online_outcomes_cannot_leak_between_arms():
    payload = envelope()["payload"]
    assert payload["proposal_factor"][
        "all_candidate_pools_locked_before_first_oracle_call"
    ]
    assert payload["candidate_locks"]["pool_pairs"] == {
        "old_v0": ["old_v0_blind", "old_v0_fiber_control"],
        "additive_route": [
            "additive_route_blind",
            "additive_route_fiber_control",
        ],
    }
    assert payload["selection_factor"]["fiber_control"][
        "no_outcome_sharing_across_tasks_or_arms"
    ]
    assert payload["budget"]["replacement_or_backfill_after_scoring"] is False
    assert payload["split_discipline"]["runtime_route_or_endpoint_lookup"] is False


def test_promotion_and_kill_decisions_are_fixed_before_scoring():
    payload = envelope()["payload"]
    assert len(payload["promotion"]["all_required"]) == 4
    assert "on both tasks" in payload["kill"]["scientific"]
    assert "384" in payload["required_authorization_text"]
    assert "zero retries" in payload["required_authorization_text"]
