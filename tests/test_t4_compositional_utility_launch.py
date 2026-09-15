import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments import t4_compositional_utility_launch as launch_module
from compose_v4.experiments.t4_compositional_utility_launch import (
    make_run_id,
    publish_once,
    read_sealed,
    run_worker,
    validate_local_inputs,
)


class _Volume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_publish_once_is_self_hashed_and_refuses_state_change(tmp_path):
    path = tmp_path / "reservation.json"
    payload = {"request_id": "a" * 64, "attempt": 1}
    first = publish_once(path, payload)
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_sealed(path) == payload
    assert publish_once(path, payload) == first
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        publish_once(path, {**payload, "attempt": 2})


def test_run_identity_binds_launcher_and_sealed_input():
    contract = {
        "inputs": {
            "request_lock": {
                "physical_sha256": "1" * 64,
                "payload_sha256": "2" * 64,
            }
        }
    }
    launcher = {"worker.py": "3" * 64}
    observed = make_run_id(contract, launcher)
    assert observed == make_run_id(json.loads(json.dumps(contract)), dict(launcher))
    assert observed != make_run_id(contract, {"worker.py": "4" * 64})


def test_repository_contract_and_request_lock_are_exact():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/t4_compositional_generator_utility_launch_v1.json"
    envelope = json.loads(path.read_text())
    assert envelope["contract_sha256"] == identity(envelope["payload"])
    contract = envelope["payload"]
    request = contract["inputs"]["request_lock"]
    assert (
        hashlib.sha256((root / request["path"]).read_bytes()).hexdigest()
        == request["physical_sha256"]
    )
    loaded, lock, requests = validate_local_inputs(root)
    assert loaded == contract
    assert "candidate_trace_available" not in lock
    assert len(requests) == 4
    assert sum(len(row["membership_ids"]) for row in requests) == 8


def _worker_fixture(tmp_path, monkeypatch, score):
    monkeypatch.setattr(
        launch_module, "rdBase", type("PinnedRDKit", (), {"rdkitVersion": "2024.03.5"})
    )
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    repo.mkdir()
    implementation = repo / "worker.py"
    implementation.write_text("worker\n")
    implementation_sha = hashlib.sha256(implementation.read_bytes()).hexdigest()
    contract = {
        "inputs": {
            "request_lock": {
                "physical_sha256": "1" * 64,
                "payload_sha256": "2" * 64,
            }
        }
    }
    request = {
        "request_id": "3" * 64,
        "target": "braf",
        "canonical_smiles": "CC",
        "docking_seed": 1701,
        "evaluator": "4" * 64,
        "evaluator_payload": {
            "qvina02_sha256": "5" * 64,
            "receptor_sha256": "6" * 64,
            "score_direction": "minimize",
            "score_unit": "kcal/mol",
        },
    }
    task = {
        "run_id": "7" * 64,
        "contract_sha256": identity(contract),
        "launcher_identity": {"worker.py": implementation_sha},
        "image_revision": {"identity": "test"},
    }
    monkeypatch.setattr(
        launch_module, "_request_from_task", lambda _repo, _task: (contract, request)
    )
    original_sha256_file = launch_module.sha256_file

    def fake_sha256_file(path):
        path = Path(path)
        if path == Path("/opt/dock/qvina02"):
            return "5" * 64
        if path == Path("/opt/dock/receptors/braf.pdbqt"):
            return "6" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(launch_module, "sha256_file", fake_sha256_file)
    calls = []

    def dock(*args, **kwargs):
        calls.append((args, kwargs))
        return score

    volume = _Volume()
    terminal = run_worker(
        task,
        repo,
        artifacts,
        volume,
        validate_revision=lambda value: None,
        dock=dock,
    )
    folder = (
        artifacts
        / "t4_compositional_generator_utility_launch"
        / task["run_id"]
        / "requests"
        / request["request_id"]
    )
    return task, repo, artifacts, volume, terminal, folder, calls, dock


def test_worker_commits_one_success_and_refuses_a_second_attempt(tmp_path, monkeypatch):
    task, repo, artifacts, volume, terminal, folder, calls, dock = _worker_fixture(
        tmp_path, monkeypatch, -7.2
    )
    assert terminal["status"] == "complete"
    assert terminal["score"] == -7.2
    assert read_sealed(folder / "reservation.json")["maximum_attempts"] == 1
    assert read_sealed(folder / "started.json")["request_id"] == "3" * 64
    assert read_sealed(folder / "result.json") == terminal
    assert volume.commits == 3
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="no retry"):
        run_worker(
            task,
            repo,
            artifacts,
            volume,
            validate_revision=lambda value: None,
            dock=dock,
        )
    assert len(calls) == 1


def test_worker_records_no_score_as_terminal_failure_without_retry(
    tmp_path, monkeypatch
):
    *_unused, terminal, folder, calls, _dock = _worker_fixture(
        tmp_path, monkeypatch, None
    )
    assert terminal["status"] == "failed_no_retry"
    assert terminal["first_score_call_charged"] == 1
    assert terminal["retry_authorized"] is False
    assert read_sealed(folder / "failure.json") == terminal
    assert not (folder / "result.json").exists()
    assert len(calls) == 1
