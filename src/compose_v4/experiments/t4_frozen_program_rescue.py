"""Fail-closed recovery helpers for the frozen T4 benchmark.

This module does not dock molecules or alter controller state. It reconciles a
reserved query that has no durable observation into an explicit charged-failure
receipt so the original worker can continue from its exact checkpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA = "t4_frozen_program_rescue_contract_v1"
LOCK_SCHEMA = "t4_frozen_program_rescue_lock_v1"
RELAUNCH_LOCK_SCHEMA = "t4_frozen_program_rescue_relaunch_lock_v2"
RELAUNCH_V3_LOCK_SCHEMA = "t4_frozen_program_rescue_relaunch_lock_v3"
RELAUNCH_V4_LOCK_SCHEMA = "t4_frozen_program_rescue_relaunch_lock_v4"
TOMBSTONE_FAILURE = "ambiguous_charged_query_unobserved"
TOMBSTONE_COMPATIBILITY_ERROR = (
    "ValueError('missing score requires an explicit failure or unqueried status')"
)
REPAIRED_INPUT_PATH = "src/compose_v4/control/adaptive_program_optimizer.py"
REPAIRED_INPUT_REASON = "accept_exact_charged_missing_observation_status"
REPAIRED_INPUT_REASONS = {
    REPAIRED_INPUT_PATH: REPAIRED_INPUT_REASON,
    "src/compose_v4/control/program_transfer.py": (
        "direct_retrieval_disabled_by_frozen_zero_candidate_setting"
    ),
    "tools/t4_frozen_program_benchmark.py": "read_only_status_change_not_worker_runtime",
}
REQUIRED_STARTED_FIELDS = {
    "query_index",
    "round_index",
    "candidate_index",
    "candidate_id",
    "endpoint",
    "target",
    "docking_seed",
    "started_at_utc",
}


def raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unseal_bytes(raw: bytes, *, source: str) -> dict:
    envelope = json.loads(raw)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"sealed rescue input has unexpected keys: {source}")
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"sealed rescue input failed its payload hash: {source}")
    return payload


def read_status_bytes(raw: bytes, *, source: str) -> dict:
    value = json.loads(raw)
    if set(value) == {"payload", "payload_sha256"}:
        return unseal_bytes(raw, source=source)
    if not isinstance(value, dict) or not source.endswith("/progress.json"):
        raise ValueError(f"unsealed non-progress T4 artifact: {source}")
    return value


def read_gzip_bytes(raw: bytes, *, source: str) -> dict:
    return unseal_bytes(gzip.decompress(raw), source=source)


def sealed_bytes(payload: dict) -> bytes:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    return canonical_bytes(envelope) + b"\n"


def make_ambiguous_tombstone(started: dict, *, reconciled_at_utc: str) -> dict:
    missing = REQUIRED_STARTED_FIELDS - set(started)
    if missing:
        raise ValueError(f"ambiguous query reservation lacks fields: {sorted(missing)}")
    if type(started["query_index"]) is not int or started["query_index"] < 0:
        raise ValueError("ambiguous query index is invalid")
    if not started["candidate_id"] or not started["endpoint"]:
        raise ValueError("ambiguous query lost its frozen candidate identity")
    body = {
        **started,
        "status": "failed",
        "score": None,
        "failure": TOMBSTONE_FAILURE,
        "seconds": None,
        "pose_sha256": {},
        "completed_at_utc": reconciled_at_utc,
        "oracle_call_charged": True,
        "observation_recovered": False,
        "automatic_retry": False,
    }
    return {**body, "receipt_id": identity(body)}


def validate_tombstone(tombstone: dict, started: dict) -> None:
    for key, value in started.items():
        if tombstone.get(key) != value:
            raise ValueError(f"tombstone changed frozen reservation field: {key}")
    body = {key: value for key, value in tombstone.items() if key != "receipt_id"}
    if tombstone.get("receipt_id") != identity(body):
        raise ValueError("tombstone receipt identity is invalid")
    expected = {
        "status": "failed",
        "score": None,
        "failure": TOMBSTONE_FAILURE,
        "seconds": None,
        "pose_sha256": {},
        "oracle_call_charged": True,
        "observation_recovered": False,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if tombstone.get(key) != value:
            raise ValueError(f"tombstone changed recovery field: {key}")


def validate_locked_round(started: dict, batch: dict) -> None:
    candidates = batch.get("candidates")
    index = started["candidate_index"]
    if not isinstance(candidates, list) or not 0 <= index < len(candidates):
        raise ValueError("ambiguous query is outside its locked candidate batch")
    candidate = candidates[index]
    if (
        candidate.get("candidate_id") != started["candidate_id"]
        or candidate.get("endpoint") != started["endpoint"]
    ):
        raise ValueError("ambiguous reservation differs from its locked candidate")


def validate_resume_checkpoint(
    checkpoint: dict | None,
    *,
    completed_results: int,
    started_round: int,
    bootstrap: bool,
) -> int:
    if checkpoint is None:
        if started_round != 0 or bootstrap is not True:
            raise ValueError("non-bootstrap T4 interruption lost checkpoint")
        return 0
    if (
        checkpoint["query_count"] > completed_results
        or checkpoint["query_count"] != len(checkpoint["curve"])
        or checkpoint["next_round"] > started_round
    ):
        raise ValueError("interrupted T4 checkpoint cannot resume safely")
    return checkpoint["query_count"]


def validate_query_paths(started_indices: set[int], result_indices: set[int]) -> int:
    if not started_indices:
        raise ValueError("interrupted unit has no charged query reservation")
    expected = set(range(max(started_indices) + 1))
    if started_indices != expected:
        raise ValueError("interrupted unit query reservations are not contiguous")
    missing = started_indices - result_indices
    if result_indices - started_indices:
        raise ValueError("query result exists without a reservation")
    if missing != {max(started_indices)}:
        raise ValueError("unit does not have exactly one final ambiguous reservation")
    return max(started_indices)


def validate_relaunch_query_paths(
    started_indices: set[int], result_indices: set[int]
) -> int:
    """Require a fully observed, contiguous ledger after a rescue invocation."""
    if not started_indices or started_indices != result_indices:
        raise ValueError("relaunch query ledger contains an ambiguous reservation")
    expected = set(range(max(started_indices) + 1))
    if started_indices != expected:
        raise ValueError("relaunch query ledger is not contiguous")
    return len(started_indices)


def validate_tombstone_compatibility_failure(failure: dict, *, unit_id: str) -> None:
    if (
        failure.get("schema_version") != "t4_frozen_program_unit_failure_v1"
        or failure.get("status") != "failed"
        or failure.get("unit", {}).get("unit_id") != unit_id
        or failure.get("error") != TOMBSTONE_COMPATIBILITY_ERROR
        or failure.get("automatic_retry") is not False
    ):
        raise ValueError(f"unexpected first-relaunch outcome: {unit_id}")


def validate_repaired_input_override(
    override: dict,
    *,
    frozen_sha256: str,
    repaired_sha256: str,
) -> None:
    """Accept only the sealed one-file compatibility repair."""
    expected = {
        "path": REPAIRED_INPUT_PATH,
        "frozen_sha256": frozen_sha256,
        "repaired_sha256": repaired_sha256,
        "reason": REPAIRED_INPUT_REASON,
    }
    if override != expected:
        raise ValueError("T4 rescue input override changed scope or identity")


def validate_relaunch_prequery_identity_error(
    message: str,
    *,
    frozen_sha256: str,
    repaired_sha256: str,
) -> None:
    expected = (
        f"input identity mismatch: /root/compose/{REPAIRED_INPUT_PATH}: "
        f"expected {frozen_sha256}, got {repaired_sha256}"
    )
    if message != expected:
        raise ValueError("unexpected repaired-relaunch prequery failure")


def validate_prequery_input_identity_error(
    message: str,
    *,
    path: str,
    frozen_sha256: str,
    repaired_sha256: str,
) -> None:
    if path not in REPAIRED_INPUT_REASONS:
        raise ValueError("unexpected T4 rescue input path")
    expected = (
        f"input identity mismatch: /root/compose/{path}: "
        f"expected {frozen_sha256}, got {repaired_sha256}"
    )
    if message != expected:
        raise ValueError("unexpected repaired-relaunch input failure")


def verify_with_repaired_input(
    path: Path,
    expected_sha256: str,
    *,
    root: Path,
    override: dict,
    verify: Callable[[Path, str], None],
) -> None:
    """Delegate normal verification, replacing one exact frozen digest."""
    relative = str(path.relative_to(root))
    if relative != override["path"]:
        verify(path, expected_sha256)
        return
    if expected_sha256 != override["frozen_sha256"]:
        raise ValueError("T4 rescue repair was requested for an unexpected digest")
    verify(path, override["repaired_sha256"])


def validate_repaired_input_overrides(
    overrides: list[dict],
    *,
    frozen_sha256: dict[str, str],
    repaired_sha256: dict[str, str],
) -> None:
    expected = [
        {
            "path": path,
            "frozen_sha256": frozen_sha256[path],
            "repaired_sha256": repaired_sha256[path],
            "reason": reason,
        }
        for path, reason in sorted(REPAIRED_INPUT_REASONS.items())
    ]
    if overrides != expected:
        raise ValueError("T4 rescue input override set changed scope or identity")


def verify_with_repaired_inputs(
    path: Path,
    expected_sha256: str,
    *,
    root: Path,
    overrides: list[dict],
    verify: Callable[[Path, str], None],
) -> None:
    relative = str(path.relative_to(root))
    matched = [row for row in overrides if row["path"] == relative]
    if not matched:
        verify(path, expected_sha256)
        return
    if len(matched) != 1 or expected_sha256 != matched[0]["frozen_sha256"]:
        raise ValueError("T4 rescue repair was requested for an unexpected input")
    verify(path, matched[0]["repaired_sha256"])


def load_rescue_contract(root: Path, relative: str) -> dict:
    path = root / relative
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("T4 rescue contract schema changed")
    expected = contract["ambiguous_started_counts"]
    if (
        len(expected) != contract["ambiguous_query_count"]
        or sum(expected.values())
        != contract["charged_calls_in_rescue_units_before_recovery"]
        or sum(1000 - value for value in expected.values())
        != contract["new_search_call_ceiling"]
        or contract["automatic_retries"] != 0
        or contract["controller_changes"] is not False
        or contract["seed_changes"] is not False
        or contract["oracle_protocol_changes"] is not False
    ):
        raise ValueError("T4 rescue contract accounting or invariants changed")
    if (
        sha256_file(root / contract["source_contract"])
        != contract["source_contract_sha256"]
    ):
        raise ValueError("source T4 contract identity changed")
    return contract


def validate_rescue_lock(root: Path, lock_path: Path, contract: dict) -> dict:
    lock = unseal(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("T4 rescue lock schema changed")
    if lock.get("source_run_id") != contract["source_run_id"]:
        raise ValueError("T4 rescue lock changed source run")
    if lock.get("contract_sha256") != sha256_file(
        root / "configs/t4_frozen_program_rescue.json"
    ):
        raise ValueError("T4 rescue lock changed its contract")
    if set(lock.get("units", {})) != set(contract["ambiguous_started_counts"]):
        raise ValueError("T4 rescue lock changed interrupted unit census")
    for unit_id, row in lock["units"].items():
        tombstone_path = root / row["tombstone_path"]
        if sha256_file(tombstone_path) != row["tombstone_sha256"]:
            raise ValueError(f"T4 tombstone identity changed: {unit_id}")
        tombstone = unseal(tombstone_path)
        validate_tombstone(tombstone, row["started"])
    return lock
