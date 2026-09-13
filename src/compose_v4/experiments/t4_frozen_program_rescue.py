"""Fail-closed recovery helpers for the frozen T4 benchmark.

This module does not dock molecules or alter controller state. It reconciles a
reserved query that has no durable observation into an explicit charged-failure
receipt so the original worker can continue from its exact checkpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA = "t4_frozen_program_rescue_contract_v1"
LOCK_SCHEMA = "t4_frozen_program_rescue_lock_v1"
TOMBSTONE_FAILURE = "ambiguous_charged_query_unobserved"
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
