from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments import t4_nodistill_topology_four_call_runtime as runtime
from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    AUTHORIZATION_PAYLOAD_SHA256,
    AUTHORIZATION_RELATIVE_PATH,
    CANDIDATE_PROPOSAL_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
    SCORED_CONTRACT_RELATIVE_PATH,
    VOLUME_NAME,
    validate_execution_contract,
    validate_scored_authority,
)

ROOT = Path(__file__).resolve().parents[1]


class _Volume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_frozen_four_call_authority_and_candidates_are_exact():
    contract, contract_identity, authorization, authorization_identity = (
        validate_scored_authority(ROOT)
    )
    assert contract_identity == SCIENTIFIC_CONTRACT_PAYLOAD_SHA256
    assert authorization_identity == AUTHORIZATION_PAYLOAD_SHA256
    assert authorization["authorized_scored_calls"] == 4
    assert contract["budget"] == {
        "unique_query_count": 4,
        "total_charged_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
    }
    proposal = json.loads((ROOT / CANDIDATE_PROPOSAL_RELATIVE_PATH).read_text())[
        "payload"
    ]
    assert [row["canonical_smiles"] for row in contract["queries"]] == [
        row["canonical_smiles"] for row in proposal["candidates"]
    ]
    assert [row["endpoint_key_sha256"] for row in contract["queries"]] == [
        row["endpoint_key_sha256"] for row in proposal["candidates"]
    ]


def test_authority_validator_rejects_any_authorization_change(tmp_path):
    root = tmp_path / "repo"
    for relative in (
        SCORED_CONTRACT_RELATIVE_PATH,
        AUTHORIZATION_RELATIVE_PATH,
        CANDIDATE_PROPOSAL_RELATIVE_PATH,
        (
            "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1/"
            "scored_preparation_v1/source_capsule_manifest.json"
        ),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    authorization_path = root / AUTHORIZATION_RELATIVE_PATH
    envelope = json.loads(authorization_path.read_text())
    envelope["payload"]["authorized_scored_calls"] = 5
    authorization_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="self-hashed|authorization"):
        validate_scored_authority(root)


def _synthetic_runtime(tmp_path, monkeypatch):
    contract, _, _, _ = validate_scored_authority(ROOT)
    execution = {"execution_source_capsule": {"manifest_payload_sha256": "c" * 64}}
    task = {
        "run_id": "d" * 64,
        "query_ids": [row["query_id"] for row in contract["queries"]],
        "image_revision": {"identity": "test"},
    }
    monkeypatch.setattr(
        runtime,
        "validate_task",
        lambda _repository_root, _task: (execution, contract, "e" * 64),
    )
    monkeypatch.setattr(
        runtime,
        "_assert_evaluator_assets",
        lambda _scientific: {
            "qvina02_sha256": contract["evaluator"]["qvina02_sha256"],
            "receptor_sha256": contract["evaluator"]["receptor_sha256"],
        },
    )
    artifact_root = tmp_path / "private_volume"
    volume = _Volume()
    intent, reservations = runtime.prepare_dispatch(
        task,
        ROOT,
        artifact_root,
        volume,
        validate_revision=lambda _revision: None,
    )
    return contract, execution, task, artifact_root, volume, intent, reservations


def test_reserve_all_then_score_each_once_and_reduce_deterministically(
    tmp_path, monkeypatch
):
    contract, _execution, task, artifacts, volume, intent, reservations = (
        _synthetic_runtime(tmp_path, monkeypatch)
    )
    assert intent["all_reservations_published"] is True
    assert len(reservations) == 4
    run_root = artifacts / task["run_id"]
    assert (run_root / "query_lock.json").is_file()
    assert all(
        (run_root / "queries" / row["query_id"] / "reservation.json").is_file()
        for row in contract["queries"]
    )

    calls = []

    def dock(smiles, tag, seed, box):
        calls.append((smiles, tag, seed, box))
        return -7.0 - len(calls) / 10

    for query in contract["queries"]:
        terminal = runtime.run_query(
            task,
            query["query_id"],
            ROOT,
            artifacts,
            volume,
            validate_revision=lambda _revision: None,
            dock=dock,
        )
        assert terminal["status"] == "complete"
        assert terminal["first_score_call_charged"] == 1
    assert len(calls) == 4
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        runtime.run_query(
            task,
            contract["queries"][0]["query_id"],
            ROOT,
            artifacts,
            volume,
            validate_revision=lambda _revision: None,
            dock=dock,
        )
    assert len(calls) == 4

    reduced = runtime.reduce_run(
        task,
        ROOT,
        artifacts,
        volume,
        validate_revision=lambda _revision: None,
    )
    assert reduced["status"] == "complete"
    assert reduced["first_score_calls_charged"] == 4
    assert reduced["successful_scores"] == 4
    assert reduced["automatic_retries"] == 0
    assert reduced["replacement_queries"] == 0
    assert reduced["backfill_queries"] == 0
    assert [row["query_id"] for row in reduced["results"]] == task["query_ids"]


def test_reducer_refuses_to_freeze_an_interim_partial_result(tmp_path, monkeypatch):
    contract, _execution, task, artifacts, volume, _intent, _reservations = (
        _synthetic_runtime(tmp_path, monkeypatch)
    )
    runtime.run_query(
        task,
        contract["queries"][0]["query_id"],
        ROOT,
        artifacts,
        volume,
        validate_revision=lambda _revision: None,
        dock=lambda *_args: -7.1,
    )
    with pytest.raises(RuntimeError, match="interim reduction"):
        runtime.reduce_run(
            task,
            ROOT,
            artifacts,
            volume,
            validate_revision=lambda _revision: None,
        )
    assert not (artifacts / task["run_id"] / "reduction.json").exists()


def test_app_and_launcher_enforce_private_one_shot_driver_ordering():
    app_source = (
        ROOT / "modal_apps/t4_nodistill_topology_four_call_app.py"
    ).read_text()
    launcher_source = (
        ROOT / "tools/launch_t4_nodistill_topology_four_call.py"
    ).read_text()
    assert VOLUME_NAME == "compose-t4-nodistill-topology-four-call-v1"
    assert "modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)" in app_source
    assert "compose-v4-artifacts" not in app_source
    assert app_source.index("prepare_dispatch(") < app_source.index(
        "t4_nodistill_topology_four_call_worker.spawn"
    )
    assert app_source.count("retries=0") == 4
    launch_source = launcher_source.split("def launch(", 1)[1].split("def reduce(", 1)[
        0
    ]
    assert launch_source.index('publish_once(paths["intent"], intent)') < (
        launch_source.index("call = function.spawn(task)")
    )
    assert 'add_parser("resume")' not in launcher_source
    assert 'replacement_queries": 0' in launcher_source
    assert 'backfill_queries": 0' in launcher_source


def test_sealed_execution_package_when_present():
    path = ROOT / EXECUTION_CONTRACT_RELATIVE_PATH
    if not path.exists():
        pytest.skip("exact-commit execution capsule is sealed after the code commit")
    execution, _identity, scientific = validate_execution_contract(ROOT)
    assert execution["status"] == "SOURCE_BOUND_EXECUTABLE_UNLAUNCHED"
    assert execution["costs_spent_during_packaging"] == {
        "docking_calls": 0,
        "oracle_calls": 0,
        "modal_calls": 0,
    }
    assert scientific["budget"]["total_charged_call_ceiling"] == 4
