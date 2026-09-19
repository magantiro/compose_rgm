from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_integrated_resume import (
    choose_resume_checkpoint,
    recovered_root_result,
    sha256_file,
)


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = identity(payload)
    path.write_text(json.dumps({"payload": payload, "payload_sha256": digest}, sort_keys=True))
    return digest


def _round(index: int, query_id: str, lock_hash: str, *, charged: int) -> dict:
    return {
        "round": index,
        "charged_calls": charged,
        "charged_this_round": 1,
        "candidate_lock_payload_sha256": lock_hash,
        "docked": [{"query_id": query_id, "smiles": f"C{index}", "score": -8.0}],
    }


def _checkpoint(contract: str, rounds: list[dict], *, charged: int) -> dict:
    return {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v1",
        "status": "running",
        "cell": "parp1_0",
        "contract_payload_sha256": contract,
        "charged_calls": charged,
        "budget_remaining": 49 - charged,
        "archive": {"ROOT": -7.3, "C1": -8.7},
        "features": [[0.0] * 15] * (charged - 1),
        "improvements": [0.0] * (charged - 1),
        "history": [{"round": row["round"], "improved": True} for row in rounds],
        "rounds": rounds,
        "rng_state": {"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
    }


def _original(folder: Path) -> dict:
    root_hash = _write(
        folder / "round_000_lock.json",
        {"queries": [{"query_id": "old_root", "smiles": "ROOT"}]},
    )
    query_id = "old_r1_q0"
    lock_hash = _write(
        folder / "round_001_lock.json",
        {"queries": [{"query_id": query_id, "smiles": "C1"}]},
    )
    checkpoint_hash = _write(
        folder / "checkpoint.json",
        _checkpoint("old-contract", [_round(1, query_id, lock_hash, charged=9)], charged=9),
    )
    return {
        "checkpoint_file_sha256": sha256_file(folder / "checkpoint.json"),
        "checkpoint_payload_sha256": checkpoint_hash,
        "round_000_file_sha256": sha256_file(folder / "round_000_lock.json"),
        "round_001_file_sha256": sha256_file(folder / "round_001_lock.json"),
        "root_payload_sha256": root_hash,
    }


def _choose(original: Path, rescue: Path, receipt: dict):
    return choose_resume_checkpoint(
        original,
        rescue,
        cell="parp1_0",
        original_contract_payload_sha256="old-contract",
        rescue_contract_payload_sha256="new-contract",
        charged_call_ceiling=49,
        receipt=receipt,
    )


def test_chooses_exact_original_completed_boundary(tmp_path: Path) -> None:
    original = tmp_path / "original"
    receipt = _original(original)

    checkpoint, source = _choose(original, tmp_path / "rescue", receipt)

    assert source == "original_round_1"
    assert checkpoint["charged_calls"] == 9
    root = recovered_root_result(checkpoint, original / "round_000_lock.json", "ROOT")
    assert root["score"] == -7.3
    assert root["recovered_from_completed_original_checkpoint"] is True


def test_chooses_advanced_rescue_checkpoint(tmp_path: Path) -> None:
    original = tmp_path / "original"
    receipt = _original(original)
    rescue = tmp_path / "rescue"
    original_checkpoint = json.loads((original / "checkpoint.json").read_text())["payload"]
    query_id = "new_r2_q0"
    lock_hash = _write(
        rescue / "round_002_lock.json",
        {"queries": [{"query_id": query_id, "smiles": "C2"}]},
    )
    second = _round(2, query_id, lock_hash, charged=17)
    advanced = {
        **original_checkpoint,
        "contract_payload_sha256": "new-contract",
        "charged_calls": 17,
        "budget_remaining": 32,
        "history": [*original_checkpoint["history"], {"round": 2, "improved": False}],
        "rounds": [*original_checkpoint["rounds"], second],
        "features": [[0.0] * 15] * 16,
        "improvements": [0.0] * 16,
    }
    _write(rescue / "checkpoint.json", advanced)

    checkpoint, source = _choose(original, rescue, receipt)

    assert source == "rescue_checkpoint"
    assert checkpoint["charged_calls"] == 17


def test_fails_closed_on_unresolved_rescue_lock(tmp_path: Path) -> None:
    original = tmp_path / "original"
    receipt = _original(original)
    rescue = tmp_path / "rescue"
    _write(rescue / "round_002_lock.json", {"queries": []})

    with pytest.raises(RuntimeError, match="locks but no completed checkpoint"):
        _choose(original, rescue, receipt)


def test_fails_closed_on_lock_beyond_completed_checkpoint(tmp_path: Path) -> None:
    original = tmp_path / "original"
    receipt = _original(original)
    rescue = tmp_path / "rescue"
    original_checkpoint = json.loads((original / "checkpoint.json").read_text())["payload"]
    query_id = "new_r2_q0"
    lock_hash = _write(
        rescue / "round_002_lock.json",
        {"queries": [{"query_id": query_id, "smiles": "C2"}]},
    )
    advanced = {
        **original_checkpoint,
        "contract_payload_sha256": "new-contract",
        "charged_calls": 17,
        "budget_remaining": 32,
        "history": [*original_checkpoint["history"], {"round": 2, "improved": False}],
        "rounds": [*original_checkpoint["rounds"], _round(2, query_id, lock_hash, charged=17)],
        "features": [[0.0] * 15] * 16,
        "improvements": [0.0] * 16,
    }
    _write(rescue / "checkpoint.json", advanced)
    _write(rescue / "round_003_lock.json", {"queries": []})

    with pytest.raises(RuntimeError, match="checkpoint/lock boundary mismatch"):
        _choose(original, rescue, receipt)
