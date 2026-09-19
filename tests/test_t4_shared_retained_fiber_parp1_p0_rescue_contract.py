from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/t4_shared_retained_fiber_parp1_p0_rescue_v1.json"
REFERENCE_PATH = ROOT / "configs/t4_shared_retained_fiber_parp1_v2.json"
APP_PATH = ROOT / "modal_apps/t4_shared_retained_fiber_parp1_p0_rescue_v1_app.py"


def test_parp1_p0_rescue_is_a_fresh_single_cell_full_horizon_contract() -> None:
    contract = unseal(CONTRACT_PATH)

    assert contract["status"] == "AUTHORIZED_FROZEN_FRESH_P0_RESCUE"
    assert [row["cell"] for row in contract["cells"]] == ["parp1_0"]
    assert contract["charged_calls_per_cell"] == 49
    assert contract["total_charged_call_ceiling"] == 49
    assert contract["batch"] == 8
    assert contract["exploration"] == 2
    assert contract["fresh_start"] == {
        "root_docked_once": True,
        "prior_checkpoint_loaded": False,
        "prior_archive_loaded": False,
        "charged_calls_before_start": 0,
        "budget_remaining_before_start": 49,
    }
    assert "legacy_resume" not in contract
    assert contract["round_lock_policy"]["fresh_start_only"] is True
    assert contract["round_lock_policy"]["legacy_checkpoint_migration"] is False


def test_parp1_p0_rescue_changes_only_the_frozen_route_timeout_and_floor_policy() -> (
    None
):
    contract = unseal(CONTRACT_PATH)
    reference = unseal(REFERENCE_PATH)

    for field in (
        "delta",
        "support",
        "charged_calls_per_cell",
        "batch",
        "parents",
        "parent_explore",
        "exploration",
        "expert_floor_rounds",
        "value_penalty",
        "docking_seed",
        "proposal_wait_seconds",
        "query_wait_seconds",
        "phase_poll_seconds",
        "evaluator_sha256",
        "docking_box",
    ):
        assert contract[field] == reference[field]

    assert contract["proposal"]["shallow"] == reference["proposal"]["shallow"]
    assert (
        contract["proposal"]["anchored_replacement"]
        == reference["proposal"]["anchored_replacement"]
    )
    route = contract["proposal"]["route_complete_region"]
    reference_route = reference["proposal"]["route_complete_region"]
    assert {
        key: value
        for key, value in route.items()
        if key != "per_candidate_timeout_seconds"
    } == reference_route
    assert reference_route["maximum_expansions"] == 4000
    assert route["maximum_expansions"] == 4000
    assert route["per_candidate_timeout_seconds"] == 10.0
    assert contract["route_scale_floor_rounds"] == 1
    assert contract["route_scale_floor_counts"] == {
        "small": 1,
        "medium": 1,
        "large": 4,
    }


def test_parp1_p0_rescue_binds_all_runtime_inputs_and_new_selection_argument() -> None:
    contract = unseal(CONTRACT_PATH)
    for relative, expected in contract["runtime_inputs_sha256"].items():
        assert sha256_file(ROOT / relative) == expected

    source = APP_PATH.read_text()
    assert (
        'VOLUME_NAME = "compose-t4-shared-retained-fiber-parp1-p0-rescue-v1"' in source
    )
    assert (
        'CONTRACT = "configs/t4_shared_retained_fiber_parp1_p0_rescue_v1.json"'
        in source
    )
    assert 'route_scale_floor_counts=contract["route_scale_floor_counts"]' in source
    assert (
        'per_candidate_timeout_seconds=route["per_candidate_timeout_seconds"]' in source
    )
    assert "diagnostics/t4_shared_retained_fiber_parp1_p0_rescue_v1/launches" in source
    assert "t4_shared_retained_fiber_parp1_v2/launches" not in source
