"""Checkpoint-boundary resume validation for integrated T4 controllers.

A completed checkpoint is safe to resume because every charged query in its last
published lock already has a durable result in the checkpoint.  A later query lock
without a corresponding completed checkpoint is deliberately not resumable: calls may
already have been charged, so retrying would violate the prospective no-retry contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from compose_v4.control.docking_value import identity

_LOCK = re.compile(r"round_(\d{3})_lock\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_envelope(path: Path) -> dict:
    document = json.loads(path.read_text())
    if set(document) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid deterministic envelope: {path}")
    if identity(document["payload"]) != document["payload_sha256"]:
        raise ValueError(f"payload identity mismatch: {path}")
    return document


def _lock_rounds(folder: Path) -> dict[int, Path]:
    rounds: dict[int, Path] = {}
    if not folder.exists():
        return rounds
    for path in folder.glob("round_*_lock.json"):
        match = _LOCK.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected round-lock filename: {path}")
        rounds[int(match.group(1))] = path
    return rounds


def _validate_checkpoint(
    checkpoint_path: Path,
    *,
    cell: str,
    contract_payload_sha256: str,
    charged_call_ceiling: int,
) -> dict:
    checkpoint = load_envelope(checkpoint_path)["payload"]
    if checkpoint.get("schema_version") != "t4_integrated_route_fiber_checkpoint_v1":
        raise ValueError(f"checkpoint schema mismatch: {checkpoint_path}")
    if checkpoint.get("cell") != cell:
        raise ValueError(f"checkpoint cell mismatch: {checkpoint_path}")
    if checkpoint.get("contract_payload_sha256") != contract_payload_sha256:
        raise ValueError(f"checkpoint contract mismatch: {checkpoint_path}")
    charged = int(checkpoint.get("charged_calls", -1))
    remaining = int(checkpoint.get("budget_remaining", -1))
    rounds = checkpoint.get("rounds")
    if charged < 1 or remaining != charged_call_ceiling - charged:
        raise ValueError(f"checkpoint budget mismatch: {checkpoint_path}")
    if not isinstance(rounds, list):
        raise TypeError(f"checkpoint rounds malformed: {checkpoint_path}")
    observed_rounds = [row.get("round") for row in rounds]
    if observed_rounds != list(range(1, len(rounds) + 1)):
        raise ValueError(f"checkpoint round sequence mismatch: {checkpoint_path}")
    if int(checkpoint.get("history", []).__len__()) != len(rounds):
        raise ValueError(f"checkpoint history mismatch: {checkpoint_path}")
    if not isinstance(checkpoint.get("rng_state"), dict):
        raise TypeError(f"checkpoint RNG state missing: {checkpoint_path}")
    if len(checkpoint.get("features", [])) != len(checkpoint.get("improvements", [])):
        raise ValueError(f"checkpoint value-training rows mismatch: {checkpoint_path}")
    return checkpoint


def _validate_completed_locks(
    folder: Path,
    checkpoint: dict,
    *,
    first_round: int,
) -> None:
    completed = len(checkpoint["rounds"])
    locks = _lock_rounds(folder)
    expected = set(range(first_round, completed + 1))
    present_scored = {round_index for round_index in locks if round_index >= first_round}
    if present_scored != expected:
        raise RuntimeError(
            f"checkpoint/lock boundary mismatch in {folder}: expected {sorted(expected)}, "
            f"found {sorted(present_scored)}"
        )
    for round_index in expected:
        lock = load_envelope(locks[round_index])
        result = checkpoint["rounds"][round_index - 1]
        if lock["payload_sha256"] != result.get("candidate_lock_payload_sha256"):
            raise RuntimeError(f"round {round_index} lock/result mismatch in {folder}")
        locked_ids = [row["query_id"] for row in lock["payload"].get("queries", [])]
        observed_ids = [row["query_id"] for row in result.get("docked", [])]
        if locked_ids != observed_ids:
            raise RuntimeError(f"round {round_index} query census mismatch in {folder}")
        if int(result.get("charged_this_round", -1)) != len(locked_ids):
            raise RuntimeError(f"round {round_index} charged census mismatch in {folder}")


def validate_original_boundary(
    folder: Path,
    *,
    cell: str,
    original_contract_payload_sha256: str,
    charged_call_ceiling: int,
    receipt: dict,
) -> dict:
    """Validate the immutable original round-1 checkpoint and its physical hashes."""

    if (folder / "round_002_lock.json").exists():
        raise RuntimeError(f"original run has an unresolved round-2 lock: {folder}")
    required = {
        "checkpoint.json": receipt["checkpoint_file_sha256"],
        "round_000_lock.json": receipt["round_000_file_sha256"],
        "round_001_lock.json": receipt["round_001_file_sha256"],
    }
    for name, expected in required.items():
        path = folder / name
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"bound original source mismatch: {path}")
    checkpoint_envelope = load_envelope(folder / "checkpoint.json")
    if checkpoint_envelope["payload_sha256"] != receipt["checkpoint_payload_sha256"]:
        raise ValueError(f"bound checkpoint payload mismatch: {folder}")
    checkpoint = _validate_checkpoint(
        folder / "checkpoint.json",
        cell=cell,
        contract_payload_sha256=original_contract_payload_sha256,
        charged_call_ceiling=charged_call_ceiling,
    )
    if checkpoint["charged_calls"] != 9 or checkpoint["budget_remaining"] != 40:
        raise ValueError(f"original rescue boundary is not exactly call 9: {folder}")
    if len(checkpoint["rounds"]) != 1:
        raise ValueError(f"original rescue boundary is not exactly round 1: {folder}")
    _validate_completed_locks(folder, checkpoint, first_round=1)
    return checkpoint


def choose_resume_checkpoint(
    original_folder: Path,
    rescue_folder: Path,
    *,
    cell: str,
    original_contract_payload_sha256: str,
    rescue_contract_payload_sha256: str,
    charged_call_ceiling: int,
    receipt: dict,
) -> tuple[dict, str]:
    """Choose a safe completed boundary, failing on any unresolved next-round lock."""

    original = validate_original_boundary(
        original_folder,
        cell=cell,
        original_contract_payload_sha256=original_contract_payload_sha256,
        charged_call_ceiling=charged_call_ceiling,
        receipt=receipt,
    )
    rescue_checkpoint = rescue_folder / "checkpoint.json"
    rescue_locks = _lock_rounds(rescue_folder)
    if not rescue_checkpoint.exists():
        if rescue_locks:
            raise RuntimeError(
                f"rescue has query locks but no completed checkpoint: {rescue_folder}"
            )
        return original, "original_round_1"
    checkpoint = _validate_checkpoint(
        rescue_checkpoint,
        cell=cell,
        contract_payload_sha256=rescue_contract_payload_sha256,
        charged_call_ceiling=charged_call_ceiling,
    )
    if len(checkpoint["rounds"]) < 2 or checkpoint["charged_calls"] <= original["charged_calls"]:
        raise ValueError(f"rescue checkpoint did not advance the original boundary: {rescue_folder}")
    _validate_completed_locks(rescue_folder, checkpoint, first_round=2)
    return checkpoint, "rescue_checkpoint"


def recovered_root_result(checkpoint: dict, root_lock_path: Path, root_smiles: str) -> dict:
    lock = load_envelope(root_lock_path)["payload"]
    queries = lock.get("queries", [])
    if len(queries) != 1 or queries[0].get("smiles") != root_smiles:
        raise ValueError(f"root lock mismatch: {root_lock_path}")
    if root_smiles not in checkpoint["archive"]:
        raise ValueError(f"root score absent from checkpoint archive: {root_lock_path}")
    return {
        "query_id": queries[0]["query_id"],
        "smiles": root_smiles,
        "score": float(checkpoint["archive"][root_smiles]),
        "failure": None,
        "recovered_from_completed_original_checkpoint": True,
    }


__all__ = [
    "choose_resume_checkpoint",
    "load_envelope",
    "recovered_root_result",
    "sha256_file",
    "validate_original_boundary",
]
