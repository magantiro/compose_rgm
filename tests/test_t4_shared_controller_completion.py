import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    QUERY_RECEIPT_SCHEMA,
    ReceiptStore,
    execute_reserved_query_once,
    make_query_lock,
    make_query_reservation,
    mark_driver_running,
    mark_driver_terminal,
    reserve_driver_generation,
    settle_query_lock,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    CAPSULE_MANIFEST_SCHEMA_VERSION,
    EXPECTED_CELLS,
    EXPECTED_CONTROLLER,
    EXPECTED_EXPERTS,
    FORBIDDEN_RUNTIME_KEYS,
    PREPARED_STATUS,
    assert_scored_launch_blocked,
    cell_runtime_payload,
    payload_identity,
    validate_preparation_contract,
    verify_exact_commit_capsule,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_shared_controller_completion_v1.json"
APP = ROOT / "modal_apps/t4_shared_controller_completion_v1_app.py"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text())


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for item in value.values():
            result.update(_keys(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_keys(item))
        return result
    return set()


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_preparation_contract_is_exactly_nine_cells_and_unlaunchable():
    report = validate_preparation_contract(ROOT, CONTRACT)
    contract = _contract()
    assert report["status"] == PREPARED_STATUS
    assert report["ready_for_scored_launch"] is False
    assert report["scored_calls_authorized"] == 0
    assert report["cell_keys"] == [row["cell_key"] for row in EXPECTED_CELLS]
    assert len(report["cell_keys"]) == len(set(report["cell_keys"])) == 9
    assert len(report["volumes"]) == len(set(report["volumes"])) == 9
    assert report["charged_call_ceiling_per_cell"] == 49
    assert report["total_charged_call_ceiling"] == 441
    assert contract["final_contract_payload_sha256"] is None
    assert contract["authorization"]["payload_sha256"] is None
    assert contract["zero_oracle_support_gate"]["artifact"] is None
    with pytest.raises(RuntimeError, match="PREPARED_PENDING_ZERO_ORACLE_GATE"):
        assert_scored_launch_blocked(ROOT, CONTRACT)


def test_matrix_and_all_77_controller_settings_are_frozen():
    contract = _contract()
    assert contract["cells"] == list(EXPECTED_CELLS)
    assert contract["controller"] == EXPECTED_CONTROLLER
    assert tuple(contract["controller"]["proposal"]) == EXPECTED_EXPERTS
    assert contract["automatic_retries"] == 0
    assert contract["replacement"] is False
    assert contract["backfill"] is False
    assert contract["shared_route_checkpoint"]["payload_sha256"] == (
        "5476be572dd40ee3f068cc8f1df238eec54e23a82cb84f803905ac891701e07d"
    )
    assert [row["cell_key"] for row in contract["cells"]] == [
        "parp1_0_d04",
        "parp1_1_d04",
        "parp1_2_d04",
        "braf_0_d04",
        "braf_1_d04",
        "braf_2_d04",
        "braf_0_d06",
        "braf_1_d06",
        "braf_2_d06",
    ]
    braf = [row for row in contract["cells"] if row["target"] == "braf"]
    by_source = {}
    for row in braf:
        by_source.setdefault(row["source_cell"], []).append(row)
    for rows in by_source.values():
        assert {row["delta"] for row in rows} == {0.4, 0.6}
        assert len({row["controller_seed"] for row in rows}) == 1


def test_runtime_payloads_are_sanitized_and_cell_local():
    contract = _contract()
    payloads = [
        cell_runtime_payload(contract, row["cell_key"]) for row in contract["cells"]
    ]
    for payload in payloads:
        assert payload["scored_calls_authorized"] == 0
        assert payload["preparation_status"] == PREPARED_STATUS
        assert not FORBIDDEN_RUNTIME_KEYS.intersection(_keys(payload))
        assert len(payload["cell"]["volume"]) > 0
        assert "cells" not in payload
        assert "published_ds" not in json.dumps(payload)
    assert len({payload["cell"]["volume"] for payload in payloads}) == 9
    assert len({payload_identity(payload) for payload in payloads}) == 9


def test_thin_app_has_no_image_volume_or_scored_runtime_path():
    source = APP.read_text()
    assert len(source.splitlines()) < 100
    for forbidden in (
        "modal.Image",
        "Volume.from_name",
        ".spawn(",
        "dock_t4",
        "GENMOL_T4_SEEDS",
        "legacy_resume",
        "add_local_dir",
    ):
        assert forbidden not in source
    assert "assert_scored_launch_blocked" in source


def test_runtime_dependencies_do_not_pull_broad_or_legacy_modules_into_capsule():
    contract_source = (
        ROOT / "src/compose_v4/experiments/t4_shared_controller_completion_contract.py"
    ).read_text()
    runtime_source = (
        ROOT / "src/compose_v4/experiments/t4_shared_controller_cell_runtime.py"
    ).read_text()
    assert "compose_v4.control.docking_value" not in contract_source + runtime_source
    assert "continuation_profile" not in contract_source + runtime_source
    assert "t4_integrated_route_fiber_v2" not in runtime_source
    assert "legacy_resume" not in runtime_source


def test_exact_commit_capsule_verifies_complete_census(tmp_path):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    files = {
        "modal_apps/app.py": b"print('prepared')\n",
        "src/compose_v4/runtime.py": b"VALUE = 1\n",
    }
    for relative, data in files.items():
        path = capsule / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    entries = {
        relative: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_oid": _git_blob_oid(data),
        }
        for relative, data in sorted(files.items())
    }
    payload = {
        "schema_version": CAPSULE_MANIFEST_SCHEMA_VERSION,
        "code_revision": "1" * 40,
        "git_tree": "2" * 40,
        "files": entries,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"payload": payload, "payload_sha256": payload_identity(payload)})
        + "\n"
    )
    result = verify_exact_commit_capsule(capsule, manifest)
    assert result["verified"] is True
    assert result["file_count"] == 2

    (capsule / "extra.py").write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="file census mismatch"):
        verify_exact_commit_capsule(capsule, manifest)


def test_capsule_rejects_tampering_and_unsafe_paths(tmp_path):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "runtime.py").write_text("VALUE = 2\n")
    payload = {
        "schema_version": CAPSULE_MANIFEST_SCHEMA_VERSION,
        "code_revision": "1" * 40,
        "git_tree": "2" * 40,
        "files": {
            "runtime.py": {
                "sha256": "0" * 64,
                "git_blob_oid": "0" * 40,
            }
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"payload": payload, "payload_sha256": payload_identity(payload)})
        + "\n"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_exact_commit_capsule(capsule, manifest)

    payload["files"] = {"../escape.py": payload["files"]["runtime.py"]}
    manifest.write_text(
        json.dumps({"payload": payload, "payload_sha256": payload_identity(payload)})
        + "\n"
    )
    with pytest.raises(ValueError, match="unsafe capsule path"):
        verify_exact_commit_capsule(capsule, manifest)


def test_capsule_rejects_symlinks(tmp_path):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    target = capsule / "runtime.py"
    target.write_text("VALUE = 1\n")
    linked = capsule / "alias.py"
    linked.symlink_to(target)
    data = target.read_bytes()
    payload = {
        "schema_version": CAPSULE_MANIFEST_SCHEMA_VERSION,
        "code_revision": "1" * 40,
        "git_tree": "2" * 40,
        "files": {
            "runtime.py": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_oid": _git_blob_oid(data),
            }
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"payload": payload, "payload_sha256": payload_identity(payload)})
        + "\n"
    )
    with pytest.raises(ValueError, match="may not contain symlinks"):
        verify_exact_commit_capsule(capsule, manifest)


def test_receipt_store_is_atomic_self_hashed_and_refuses_lock_overwrite(tmp_path):
    flushes = []
    store = ReceiptStore(tmp_path, flush=lambda: flushes.append("flush"))
    payload = {"schema_version": "fixture", "value": 1}
    receipt_hash = store.publish_once("cell/round_001_lock.json", payload)
    assert receipt_hash == payload_identity(payload)
    assert store.read("cell/round_001_lock.json") == payload
    assert flushes == ["flush"]
    with pytest.raises(FileExistsError, match="already exists"):
        store.publish_once("cell/round_001_lock.json", payload)

    store.replace("cell/checkpoint.json", {"calls": 1})
    store.replace("cell/checkpoint.json", {"calls": 9})
    assert store.read("cell/checkpoint.json") == {"calls": 9}
    assert flushes == ["flush", "flush", "flush"]


def test_query_crash_after_durable_reservation_cannot_repeat_evaluation(tmp_path):
    store = ReceiptStore(tmp_path)
    lock = make_query_lock(
        cell_key="braf_1_d06",
        round_index=0,
        charged_before=0,
        selected_rows=[{"smiles": "CC", "proposal_experts": ["shallow"]}],
        query_deadline=10.0,
    )
    lock_path = "rounds/000/lock.json"
    store.publish_once(lock_path, lock)
    calls = []

    def interrupted(smiles):
        calls.append(smiles)
        assert store.read("rounds/000/reservations/q00.json")["status"] == "reserved"
        raise RuntimeError("executor disappeared")

    arguments = {
        "lock_path": lock_path,
        "reservation_path": "rounds/000/reservations/q00.json",
        "receipt_path": "rounds/000/receipts/q00.json",
        "query_id": lock["queries"][0]["query_id"],
        "evaluate": interrupted,
    }
    with pytest.raises(RuntimeError, match="executor disappeared"):
        execute_reserved_query_once(store, **arguments)
    with pytest.raises(FileExistsError, match="already exists"):
        execute_reserved_query_once(store, **arguments)
    assert calls == ["CC"]
    reservation = store.read(arguments["reservation_path"])
    settled = settle_query_lock(lock, {arguments["query_id"]: reservation}, now=11.0)
    assert settled["charged_after"] == 1
    assert settled["observations"][0]["failure"].endswith("reserved")


def test_query_lock_waits_then_charges_every_locked_row_once():
    rows = [
        {
            "smiles": f"C{'C' * index}",
            "selection_kind": "model",
            "proposal_experts": ["shallow"],
            "parent": "C",
            "parent_score": -7.0,
        }
        for index in range(1, 9)
    ]
    lock = make_query_lock(
        cell_key="parp1_0_d04",
        round_index=6,
        charged_before=41,
        selected_rows=rows,
        query_deadline=100.0,
    )
    first = lock["queries"][0]
    receipt = {
        **make_query_reservation(lock, first["query_id"]),
        "schema_version": QUERY_RECEIPT_SCHEMA,
        "status": "complete",
        "answer": {"score": -8.1, "failure": None},
    }
    waiting = settle_query_lock(lock, {first["query_id"]: receipt}, now=99.0)
    assert waiting["action"] == "wait"
    assert waiting["charged_after"] == 41

    settled = settle_query_lock(lock, {first["query_id"]: receipt}, now=101.0)
    assert settled["action"] == "recover_with_unresolved"
    assert settled["charged_after"] == 49
    assert len(settled["observations"]) == 8
    assert settled["observations"][0]["score"] == -8.1
    assert all(row["score"] is None for row in settled["observations"][1:])
    assert all(
        row["failure"] == "unresolved_locked_query_missing"
        for row in settled["observations"][1:]
    )


def test_query_lock_rejects_budget_overrun_duplicates_and_foreign_receipts():
    row = {"smiles": "CC", "proposal_experts": ["shallow"]}
    with pytest.raises(ValueError, match="49-call"):
        make_query_lock(
            cell_key="braf_0_d04",
            round_index=7,
            charged_before=49,
            selected_rows=[row],
            query_deadline=1.0,
        )
    with pytest.raises(ValueError, match="duplicate molecules"):
        make_query_lock(
            cell_key="braf_0_d04",
            round_index=1,
            charged_before=1,
            selected_rows=[row, row],
            query_deadline=1.0,
        )
    lock = make_query_lock(
        cell_key="braf_0_d04",
        round_index=1,
        charged_before=1,
        selected_rows=[row],
        query_deadline=1.0,
    )
    with pytest.raises(ValueError, match="do not belong"):
        settle_query_lock(lock, {"other": {"status": "complete"}}, now=2.0)


def test_independent_driver_generations_fail_closed_without_cross_cell_state():
    parp = reserve_driver_generation(phase_status="running", existing_state=None)
    braf = reserve_driver_generation(phase_status="running", existing_state=None)
    assert parp["action"] == braf["action"] == "spawn"
    assert parp["state"] is not braf["state"]

    parp_running = mark_driver_running(parp["state"], "fc-parp")
    braf_running = mark_driver_running(braf["state"], "fc-braf")
    parp_terminal = mark_driver_terminal(parp_running)
    assert braf_running["state"] == "running"

    resumed = reserve_driver_generation(
        phase_status="queries_running",
        existing_state=parp_terminal,
        confirmed_prior_call_terminal=True,
    )
    assert resumed["action"] == "spawn"
    assert resumed["state"]["generation"] == 1
    waiting = reserve_driver_generation(
        phase_status="queries_running", existing_state=braf_running
    )
    assert waiting == {"action": "wait", "state": braf_running}


@pytest.mark.parametrize(
    "terminal_status",
    [
        "complete_budget",
        "candidate_exhaustion",
        "operational_fail_closed",
        "root_oracle_failure",
        "failed",
    ],
)
def test_terminal_cell_status_never_spawns_a_continuation(terminal_status):
    result = reserve_driver_generation(
        phase_status=terminal_status,
        existing_state=None,
    )
    assert result == {"action": "finish", "state": None}
