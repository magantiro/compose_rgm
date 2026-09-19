import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/t4_shared_retained_fiber_parp1_v1.json",
    ROOT / "configs/t4_shared_retained_fiber_jak2_v1.json",
)
V2_CONFIGS = (
    ROOT / "configs/t4_shared_retained_fiber_parp1_v2.json",
    ROOT / "configs/t4_shared_retained_fiber_jak2_v2.json",
)


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for row in value.values():
            result.update(_keys(row))
        return result
    if isinstance(value, list):
        result = set()
        for row in value:
            result.update(_keys(row))
        return result
    return set()


def test_shared_scored_contracts_are_sealed_and_runtime_inputs_match():
    contracts = [unseal(path) for path in CONFIGS]
    for contract in contracts:
        assert contract["status"] == "AUTHORIZED_FROZEN_PRIMARY_CONTROLLER"
        assert len(contract["cells"]) == 3
        assert contract["charged_calls_per_cell"] == 49
        assert contract["total_charged_call_ceiling"] == 147
        assert contract["route_scale_floor_rounds"] == 2
        assert contract["proposal"]["route_complete_region"] == {
            "pool_size": 192,
            "realization_limit": 96,
            "beam_width": 48,
            "expansion_width": 48,
            "max_bindings_per_template": 4,
            "maximum_expansions": 4000,
            "scale_balanced": True,
            "training_scope": "one shared task-independent expert fit on all 77 locked routes",
        }
        for relative, expected in contract["runtime_inputs_sha256"].items():
            assert sha256_file(ROOT / relative) == expected

    invariant = (
        "support",
        "charged_calls_per_cell",
        "batch",
        "parents",
        "parent_explore",
        "exploration",
        "expert_floor_rounds",
        "route_scale_floor_rounds",
        "value_penalty",
        "docking_seed",
        "proposal",
    )
    assert all(contracts[0][key] == contracts[1][key] for key in invariant)
    assert {contract["delta"] for contract in contracts} == {0.4, 0.6}


def test_shared_checkpoint_contains_no_runtime_answer_map():
    checkpoint_path = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
    envelope = json.loads(checkpoint_path.read_text())
    assert identity(envelope["payload"]) == envelope["payload_sha256"]
    forbidden = {
        "cell",
        "endpoint",
        "route_id",
        "smiles",
        "source_graph",
        "source_group",
        "target",
        "target_name",
    }
    assert forbidden.isdisjoint(_keys(envelope["payload"]))


def test_shared_apps_enable_scale_balancing_and_route_scale_floor():
    for name in ("parp1", "jak2"):
        source = (
            ROOT / f"modal_apps/t4_shared_retained_fiber_{name}_app.py"
        ).read_text()
        assert 'scale_balanced=route["scale_balanced"]' in source
        assert 'route_scale_floor_rounds=contract["route_scale_floor_rounds"]' in source
        assert 'automatic_retries": 0' in source


def test_v2_runner_contracts_authorize_only_hash_bound_v1_continuation():
    contracts = [unseal(path) for path in V2_CONFIGS]
    for contract in contracts:
        assert contract["status"] == "AUTHORIZED_V1_CHECKPOINT_CONTINUATION"
        assert contract["proposal_wait_seconds"] == 2400
        assert contract["query_wait_seconds"] == 1200
        assert contract["phase_poll_seconds"] == 5
        assert contract["round_lock_policy"] == {
            "publish_before_docking": True,
            "per_query_receipts": True,
            "per_expert_proposal_receipts": True,
            "resume_complete_receipts_only": True,
            "never_resubmit_reserved_or_missing_query_receipt": True,
            "continue_after_deadline_with_unresolved_calls_charged": True,
            "no_backfill": True,
            "automatic_retries": 0,
        }
        for relative, expected in contract["runtime_inputs_sha256"].items():
            assert sha256_file(ROOT / relative) == expected
        assert set(contract["legacy_resume"]) == {
            row["cell"] for row in contract["cells"]
        }
        for cell, legacy in contract["legacy_resume"].items():
            assert cell in {row["cell"] for row in contract["cells"]}
            assert legacy["mode"] in {"checkpoint", "fail_closed"}
            assert len(legacy["round_lock_sha256"]) == 64
            assert len(legacy["round_lock_payload_sha256"]) == 64
            if legacy["mode"] == "checkpoint":
                assert len(legacy["checkpoint_sha256"]) == 64
                assert len(legacy["checkpoint_payload_sha256"]) == 64
            if "unresolved_round_lock_path" in legacy:
                assert legacy["mode"] == "checkpoint"
                assert len(legacy["unresolved_round_lock_sha256"]) == 64
                assert len(legacy["unresolved_round_lock_payload_sha256"]) == 64

        source = contract["reported_ivg_source"]
        assert source["upstream_revision"] == (
            "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
        )
        assert source["sha256"] == (
            "a5ae0d826fc458caa1bdfa0a4d8f23927da33cea39188c3e4375a55164de6b80"
        )
        assert source["evidence"].startswith("published lead-optimization table")

    parp1, jak2 = contracts
    assert parp1["reported_ivg_delta_0_6"] == {
        "parp1_0": -12.3,
        "parp1_1": -11.7,
        "parp1_2": -10.7,
    }
    assert jak2["reported_ivg_delta_0_4"] == {
        "jak2_0": -10.2,
        "jak2_1": -10.5,
        "jak2_2": -10.2,
    }


def test_v2_apps_shard_proposals_queries_and_phases():
    for name in ("parp1", "jak2"):
        source = (
            ROOT / f"modal_apps/t4_shared_retained_fiber_{name}_v2_app.py"
        ).read_text()
        assert "def durable_proposal_worker" in source
        assert '"status": "reserved"' in source
        assert '"status": "complete"' in source
        assert "def run_phase" in source
        assert "proposal_collection_action" in source
        assert "query_collection_action" in source
        assert "migrate_v1_checkpoint" in source
        assert "sealed_v1_checkpoint" in source
        assert 'route_scale_floor_rounds=contract["route_scale_floor_rounds"]' in source
        assert 'continuation = drive.spawn({**task, "continuation": True})' in source
