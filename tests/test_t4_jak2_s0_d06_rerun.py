import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_jak2_s0_d06_rerun_contract import (
    EXPECTED_EXPERTS,
    PREPARED_STATUS,
    validate_rerun_preflight,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_shared_retained_fiber_jak2_s0_d06_rerun_v1.json"
PRIMARY = ROOT / "configs/t4_shared_retained_fiber_jak2_v2.json"
APP = ROOT / "modal_apps/t4_shared_retained_fiber_jak2_s0_d06_rerun_v1_app.py"
PREFLIGHT = (
    ROOT / "diagnostics/t4_shared_retained_fiber_jak2_s0_d06_rerun_v1/preflight.json"
)


def test_contract_is_one_fresh_unlaunched_49_call_cell():
    contract = unseal(CONTRACT)
    assert contract["status"] == PREPARED_STATUS
    assert contract["scored_calls_authorized"] == 0
    assert [(row["cell"], row["source_global_index"]) for row in contract["cells"]] == [
        ("jak2_0", 12)
    ]
    assert contract["delta"] == 0.6
    assert contract["charged_calls_per_cell"] == 49
    assert contract["total_charged_call_ceiling"] == 49
    assert contract["automatic_retries"] == 0
    assert "legacy_resume" not in contract
    assert contract["launch_authorization"]["state"] == "REQUIRED_AND_NOT_RECORDED"


def test_controller_settings_match_current_primary_except_requested_cell_and_fiber():
    contract = unseal(CONTRACT)
    primary = unseal(PRIMARY)
    scalar_settings = (
        "support",
        "batch",
        "parents",
        "parent_explore",
        "exploration",
        "expert_floor_rounds",
        "route_scale_floor_rounds",
        "value_penalty",
        "docking_seed",
        "docking_box",
        "evaluator_sha256",
    )
    for key in scalar_settings:
        assert contract[key] == primary[key]
    assert contract["proposal"] == primary["proposal"]
    assert contract["cells"][0] == primary["cells"][0]


def test_preflight_is_hash_bound_and_prepared_only():
    prepared = validate_rerun_preflight(
        ROOT,
        CONTRACT,
        require_clean_runtime=False,
    )
    assert prepared["status"] == "PREPARED_NO_SCORED_AUTHORIZATION"
    assert prepared["contract_payload_sha256"] == identity(unseal(CONTRACT))
    assert prepared["contract_file_sha256"] == sha256_file(CONTRACT)
    assert prepared["proposal_experts"] == list(EXPECTED_EXPERTS)
    assert prepared["shared_route_training_routes"] == 77
    assert prepared["shared_route_training_regions"] == 147
    assert prepared["deferred_joint_planning"] is False
    assert prepared["charged_call_ceiling"] == 49
    with pytest.raises(ValueError, match="authorization"):
        validate_rerun_preflight(
            ROOT,
            CONTRACT,
            authorization_payload_sha256="0" * 64,
            require_clean_runtime=False,
        )


def test_persisted_preflight_exactly_replays_and_is_self_hashed():
    envelope = json.loads(PREFLIGHT.read_text())
    assert identity(envelope["payload"]) == envelope["payload_sha256"]
    replay = validate_rerun_preflight(ROOT, CONTRACT, require_clean_runtime=False)
    assert envelope["payload"] == replay


def test_checkpoint_binding_is_all_route_task_independent():
    contract = unseal(CONTRACT)
    binding = contract["shared_route_checkpoint"]
    checkpoint_path = ROOT / binding["path"]
    envelope = json.loads(checkpoint_path.read_text())
    assert sha256_file(checkpoint_path) == binding["sha256"]
    assert envelope["payload_sha256"] == binding["payload_sha256"]
    assert identity(envelope["payload"]) == envelope["payload_sha256"]
    assert envelope["payload"]["training_routes"] == 77
    assert envelope["payload"]["training_scope"] == (
        "all_locked_t4_routes_shared_task_independent"
    )
    assert envelope["payload"]["runtime_target_conditioning"] is False


def test_runtime_has_three_legacy_lanes_and_no_joint_or_comparator_path():
    contract = unseal(CONTRACT)
    assert tuple(contract["proposal"]) == EXPECTED_EXPERTS
    assert contract["deferred_joint_planning"] == {
        "enabled": False,
        "reason": "irrelevant_to_one_region_jak_routes",
    }
    for forbidden_key in (
        "reported_ivg_delta_0_6",
        "reported_ivg_source",
        "winner_smiles",
        "known_endpoint",
        "target_to_program",
        "cell_to_winner",
    ):
        assert forbidden_key not in contract

    source = APP.read_text().lower()
    for forbidden_token in (
        "virtual_joint_region_proposer",
        "deferred_joint",
        "reported_ivg",
        "winner_smiles",
        "target_to_program",
    ):
        assert forbidden_token not in source
    assert 'expert in {"shallow", "anchored_replacement"}' in source
    assert 'expert == "route_complete_region"' in source


def test_app_preserves_receipt_shards_and_fails_closed_on_duplicates():
    source = APP.read_text()
    assert '"retries": 0' in source
    assert 'status") == "complete"' in source
    assert "retry forbidden" in source
    assert 'folder / f"round_{round_index:03d}_lock.json"' in source
    assert 'folder / f"round_{round_index:03d}_result.json"' in source
    assert 'checkpoint_path = folder / "checkpoint.json"' in source
    assert '"state": "reserved"' in source
    assert source.index('"state": "reserved"') < source.index(
        "continuation = drive.spawn("
    )
    assert "launch reservation already exists" in source
    assert "manual advance requires authoritative confirmation" in source
    assert 'mode: str = "preflight"' in source
    assert "a separately authorized contract payload SHA-256 is required" in source


def test_required_authorization_is_exactly_one_hash_bound_sentence():
    contract = unseal(CONTRACT)
    sentence = contract["required_scored_authorization_template"].format(
        contract_payload_sha256=identity(contract)
    )
    assert "\n" not in sentence
    assert sentence.endswith(".")
    assert identity(contract) in sentence
    assert "49 charged docking calls" in sentence
    assert "zero retries" in sentence
    assert "runtime comparator" in sentence
