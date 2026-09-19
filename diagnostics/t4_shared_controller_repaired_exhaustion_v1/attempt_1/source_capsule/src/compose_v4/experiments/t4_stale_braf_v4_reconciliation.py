"""Reconcile stale BRAF-v4 charged queries without exposing prior outcomes.

This module is deliberately read-only with respect to the downloaded run.  It
validates every self-hashed JSON envelope, reconciles immutable query locks with
complete receipts and persisted checkpoints, and emits only canonical-molecule
SHA-256 identities.  Docking scores and raw SMILES are never written to the
reconciliation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
    sha256_file,
)

SCHEMA_VERSION = "t4_braf_v4_stale_query_reconciliation_v1"
EXPECTED_CELLS = ("braf_0", "braf_1")
EXPECTED_RECEIPTS = {"braf_0": 20, "braf_1": 23}
EXPECTED_SCHEMA_COUNTS = {
    "t4_integrated_phase_result_v1": 8,
    "t4_integrated_proposal_dispatch_v1": 8,
    "t4_integrated_proposal_manifest_v1": 8,
    "t4_integrated_proposal_receipt_v1": 104,
    "t4_integrated_query_receipt_v1": 43,
    "t4_integrated_round_lock_v2": 8,
    "t4_integrated_route_fiber_checkpoint_v2": 2,
    "t4_integrated_route_fiber_launch_v2": 1,
    "t4_integrated_route_fiber_phase_summary_v1": 2,
}
EXPECTED_JSON_ENVELOPES = sum(EXPECTED_SCHEMA_COUNTS.values())

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_OUTCOME_KEYS = {
    "answer",
    "best",
    "best_so_far",
    "final_best",
    "parent_score",
    "round_best",
    "score",
    "scores",
    "smiles",
    "winner",
}


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    if not isinstance(envelope, dict) or set(envelope) != {
        "payload",
        "payload_sha256",
    }:
        raise ValueError(f"invalid JSON envelope shape: {path}")
    payload = envelope["payload"]
    identity = envelope["payload_sha256"]
    if not isinstance(payload, dict) or payload_identity(payload) != identity:
        raise ValueError(f"invalid JSON envelope payload identity: {path}")
    return payload, str(identity)


def _mapping_identity(mapping: Mapping[str, str]) -> str:
    return payload_identity(dict(sorted(mapping.items())))


def canonical_smiles_sha256(smiles: str) -> str:
    """Return a score-free identity of the RDKit-canonical molecular graph."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("charged receipt contains an invalid SMILES value")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    repeated = Chem.MolFromSmiles(canonical)
    if (
        repeated is None
        or Chem.MolToSmiles(
            repeated,
            canonical=True,
            isomericSmiles=True,
        )
        != canonical
    ):
        raise ValueError("charged receipt canonicalization is not idempotent")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = {str(key) for key in value}
        for item in value.values():
            result.update(_walk_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_walk_keys(item))
        return result
    return set()


def validate_reconciliation_payload(payload: Mapping[str, Any]) -> None:
    """Validate the public score-free reconciliation schema and invariants."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected stale BRAF reconciliation schema")
    if payload.get("charged_complete_receipts") != 43:
        raise ValueError("stale BRAF charged receipt census drift")
    if payload.get("unresolved_locked_queries") != 0:
        raise ValueError("stale BRAF run has unresolved locked queries")
    leaked = sorted(_FORBIDDEN_OUTCOME_KEYS.intersection(_walk_keys(payload)))
    if leaked:
        raise ValueError(f"reconciliation leaks outcome fields: {leaked}")

    exclusions = payload.get("excluded_canonical_smiles_sha256")
    if not isinstance(exclusions, dict) or tuple(exclusions) != EXPECTED_CELLS:
        raise ValueError("stale BRAF exclusion cell census drift")
    for cell, expected in EXPECTED_RECEIPTS.items():
        values = exclusions.get(cell)
        if (
            not isinstance(values, list)
            or len(values) != expected
            or values != sorted(set(values))
            or any(
                not isinstance(value, str) or not _SHA256.fullmatch(value)
                for value in values
            )
        ):
            raise ValueError(f"invalid stale BRAF identities for {cell}")

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise TypeError("stale BRAF reconciliation counts are missing")
    if counts.get("json_envelopes") != EXPECTED_JSON_ENVELOPES:
        raise ValueError("stale BRAF JSON envelope census drift")
    if counts.get("schema_versions") != EXPECTED_SCHEMA_COUNTS:
        raise ValueError("stale BRAF JSON schema census drift")
    if counts.get("locked_queries") != 43 or counts.get("round_locks") != 8:
        raise ValueError("stale BRAF lock census drift")
    if counts.get("canonical_denylist_entries") != 43:
        raise ValueError("stale BRAF canonical denylist census drift")
    per_cell = counts.get("cells")
    if not isinstance(per_cell, dict) or tuple(per_cell) != EXPECTED_CELLS:
        raise ValueError("stale BRAF per-cell count census drift")
    for cell, expected in EXPECTED_RECEIPTS.items():
        if per_cell[cell] != {
            "canonical_denylist_entries": expected,
            "charged_complete_receipts": expected,
            "locked_queries": expected,
            "round_locks": 4,
        }:
            raise ValueError(f"stale BRAF per-cell counts drift for {cell}")


def _validate_launch_identity(
    *,
    run_id: str,
    downloaded_launch: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if downloaded_launch.get("schema_version") != "t4_integrated_route_fiber_launch_v2":
        raise ValueError("downloaded stale BRAF launch schema drift")
    if downloaded_launch.get("run_id") != run_id:
        raise ValueError("download root and stale BRAF run identity disagree")
    if downloaded_launch.get("charged_call_ceiling") != 98:
        raise ValueError("stale BRAF charged-call ceiling drift")
    if downloaded_launch.get("automatic_retries") != 0:
        raise ValueError("stale BRAF automatic-retry policy drift")
    if launch_receipt.get("schema_version") != (
        "t4_integrated_route_fiber_launch_receipt_v1"
    ):
        raise ValueError("repository launch receipt schema drift")
    if launch_receipt.get("task") != downloaded_launch:
        raise ValueError("repository and downloaded launch identities disagree")
    if launch_receipt.get("output_prefix") != run_id:
        raise ValueError("repository launch output prefix drift")
    required_strings = (
        "deployment",
        "function_call_id",
        "volume",
    )
    if any(
        not isinstance(launch_receipt.get(key), str) or not launch_receipt[key]
        for key in required_strings
    ):
        raise ValueError("remote launch identity is incomplete")
    return {
        "automatic_retries": downloaded_launch["automatic_retries"],
        "charged_call_ceiling": downloaded_launch["charged_call_ceiling"],
        "code_revision": downloaded_launch["code_revision"],
        "contract_file_sha256": downloaded_launch["contract_file_sha256"],
        "contract_payload_sha256": downloaded_launch["contract_payload_sha256"],
        "deployment": launch_receipt["deployment"],
        "function_call_id": launch_receipt["function_call_id"],
        "output_prefix": launch_receipt["output_prefix"],
        "run_id": run_id,
        "volume": launch_receipt["volume"],
    }


def _validate_summary(
    *,
    run_id: str,
    contract_payload_sha256: str,
    summary: Mapping[str, Any],
) -> dict[str, int]:
    if summary.get("schema_version") != "t4_integrated_route_fiber_phase_summary_v1":
        raise ValueError("stale BRAF summary schema drift")
    if summary.get("run_id") != run_id:
        raise ValueError("stale BRAF summary run identity drift")
    if summary.get("contract_payload_sha256") != contract_payload_sha256:
        raise ValueError("stale BRAF summary contract identity drift")
    if summary.get("finished") is not True or summary.get("new_oracle_calls") != 43:
        raise ValueError("stale BRAF summary charged-call census drift")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(EXPECTED_CELLS):
        raise ValueError("stale BRAF summary cell census drift")
    result: dict[str, int] = {}
    for row in records:
        if not isinstance(row, dict):
            raise TypeError("stale BRAF summary row is invalid")
        cell = row.get("cell")
        if cell not in EXPECTED_CELLS or cell in result:
            raise ValueError("stale BRAF summary cell identity drift")
        if row.get("phase") != 4 or row.get("status") != "proposals_running":
            raise ValueError("stale BRAF terminal proposal boundary drift")
        if row.get("final_best") is not None:
            raise ValueError("stale BRAF summary unexpectedly materializes an outcome")
        charged = row.get("charged_calls")
        if charged != EXPECTED_RECEIPTS[cell]:
            raise ValueError(f"stale BRAF summary charged calls drift for {cell}")
        result[cell] = int(charged)
    return result


def _validate_cell(
    *,
    input_root: Path,
    cell: str,
    contract_payload_sha256: str,
    envelopes: Mapping[str, tuple[dict[str, Any], str]],
) -> tuple[list[str], dict[str, int]]:
    cell_root = input_root / cell
    checkpoint_path = f"{cell}/checkpoint.json"
    checkpoint, _ = envelopes[checkpoint_path]
    if checkpoint.get("schema_version") != "t4_integrated_route_fiber_checkpoint_v2":
        raise ValueError(f"stale BRAF checkpoint schema drift for {cell}")
    if checkpoint.get("cell") != cell:
        raise ValueError(f"stale BRAF checkpoint cell drift for {cell}")
    if checkpoint.get("contract_payload_sha256") != contract_payload_sha256:
        raise ValueError(f"stale BRAF checkpoint contract drift for {cell}")
    expected_receipts = EXPECTED_RECEIPTS[cell]
    if (
        checkpoint.get("charged_calls") != expected_receipts
        or checkpoint.get("budget_remaining") != 49 - expected_receipts
        or checkpoint.get("rounds_completed") != 3
        or len(checkpoint.get("rounds", ())) != 3
        or len(checkpoint.get("archive", {})) != expected_receipts
        or checkpoint.get("status") != "running"
    ):
        raise ValueError(f"stale BRAF checkpoint census drift for {cell}")

    lock_paths = sorted(cell_root.glob("round_*_lock.json"))
    result_paths = sorted(cell_root.glob("round_*_result.json"))
    receipt_paths = sorted((cell_root / "receipts").glob("*.json"))
    if len(lock_paths) != 4 or len(result_paths) != 4:
        raise ValueError(f"stale BRAF round census drift for {cell}")

    locked: dict[str, str] = {}
    charged_before = 0
    lock_by_round: dict[int, tuple[dict[str, Any], str]] = {}
    for expected_round, path in enumerate(lock_paths):
        relative = path.relative_to(input_root).as_posix()
        lock, lock_identity = envelopes[relative]
        if (
            lock.get("schema_version") != "t4_integrated_round_lock_v2"
            or lock.get("cell") != cell
            or lock.get("round") != expected_round
            or lock.get("charged_before") != charged_before
            or lock.get("contract_payload_sha256") != contract_payload_sha256
        ):
            raise ValueError(f"stale BRAF query lock drift: {relative}")
        queries = lock.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"stale BRAF query lock is empty: {relative}")
        for query in queries:
            if not isinstance(query, dict):
                raise TypeError(f"stale BRAF lock query is invalid: {relative}")
            query_id = query.get("query_id")
            smiles = query.get("smiles")
            if (
                not isinstance(query_id, str)
                or not query_id
                or not isinstance(smiles, str)
                or not smiles
                or query_id in locked
            ):
                raise ValueError(f"stale BRAF lock query identity drift: {relative}")
            locked[query_id] = smiles
        charged_before += len(queries)
        lock_by_round[expected_round] = (lock, lock_identity)
    if charged_before != expected_receipts:
        raise ValueError(f"stale BRAF locked query census drift for {cell}")

    completed: dict[str, str] = {}
    hashes: list[str] = []
    for path in receipt_paths:
        relative = path.relative_to(input_root).as_posix()
        receipt, _ = envelopes[relative]
        query_id = receipt.get("query_id")
        smiles = receipt.get("smiles")
        if (
            receipt.get("schema_version") != "t4_integrated_query_receipt_v1"
            or receipt.get("status") != "complete"
            or not isinstance(receipt.get("answer"), dict)
            or not isinstance(query_id, str)
            or not isinstance(smiles, str)
            or query_id in completed
            or locked.get(query_id) != smiles
        ):
            raise ValueError(f"stale BRAF query receipt drift: {relative}")
        completed[query_id] = smiles
        hashes.append(canonical_smiles_sha256(smiles))
    if set(completed) != set(locked):
        raise ValueError(f"stale BRAF locked and complete query sets differ for {cell}")
    if len(hashes) != expected_receipts or len(set(hashes)) != expected_receipts:
        raise ValueError(
            f"stale BRAF canonical query identities are not unique for {cell}"
        )

    cumulative = 0
    for expected_round, path in enumerate(result_paths):
        relative = path.relative_to(input_root).as_posix()
        result, _ = envelopes[relative]
        lock, lock_identity = lock_by_round[expected_round]
        queries = lock["queries"]
        cumulative += len(queries)
        embedded_checkpoint = result.get("checkpoint")
        if (
            result.get("schema_version") != "t4_integrated_phase_result_v1"
            or result.get("round") != expected_round
            or not isinstance(embedded_checkpoint, dict)
            or embedded_checkpoint.get("charged_calls") != cumulative
        ):
            raise ValueError(f"stale BRAF round result drift: {relative}")
        expected_by_id = {row["query_id"]: row["smiles"] for row in queries}
        if expected_round == 0:
            root_result = embedded_checkpoint.get("root_result")
            if (
                not isinstance(root_result, dict)
                or {root_result.get("query_id"): root_result.get("smiles")}
                != expected_by_id
                or "round_result" in result
            ):
                raise ValueError(f"stale BRAF root lock/result mismatch: {relative}")
        else:
            round_result = result.get("round_result")
            if (
                not isinstance(round_result, dict)
                or round_result.get("round") != expected_round
                or round_result.get("charged_this_round") != len(queries)
                or round_result.get("charged_calls") != cumulative
                or round_result.get("candidate_lock_payload_sha256") != lock_identity
            ):
                raise ValueError(f"stale BRAF round result drift: {relative}")
            docked = round_result.get("docked")
            if not isinstance(docked, list) or len(docked) != len(queries):
                raise ValueError(f"stale BRAF docked result census drift: {relative}")
            docked_by_id = {
                row.get("query_id"): row.get("smiles")
                for row in docked
                if isinstance(row, dict)
            }
            if docked_by_id != expected_by_id:
                raise ValueError(f"stale BRAF lock/result query mismatch: {relative}")
        if expected_round == 3 and embedded_checkpoint != checkpoint:
            raise ValueError(f"stale BRAF final checkpoint mismatch for {cell}")

    if (cell_root / "round_004_lock.json").exists():
        raise ValueError(
            f"stale BRAF round 4 unexpectedly contains a query lock: {cell}"
        )
    if not (cell_root / "round_004_proposals.json").is_file():
        raise ValueError(f"stale BRAF terminal proposal boundary is missing for {cell}")
    return sorted(hashes), {
        "canonical_denylist_entries": len(hashes),
        "charged_complete_receipts": len(completed),
        "locked_queries": len(locked),
        "round_locks": len(lock_paths),
    }


def build_reconciliation_payload(
    *, input_root: Path, launch_receipt_path: Path
) -> dict[str, Any]:
    """Build a deterministic score-free reconciliation from immutable inputs."""

    input_root = input_root.resolve()
    launch_receipt_path = launch_receipt_path.resolve()
    if not input_root.is_dir() or not launch_receipt_path.is_file():
        raise FileNotFoundError("stale BRAF input root or launch receipt is missing")
    run_id = input_root.name
    if not _SHA256.fullmatch(run_id):
        raise ValueError("stale BRAF download root is not a run identity")

    all_files = sorted(path for path in input_root.rglob("*") if path.is_file())
    json_files = sorted(input_root.rglob("*.json"))
    if all_files != json_files or len(json_files) != EXPECTED_JSON_ENVELOPES:
        raise ValueError("stale BRAF download file census drift")
    if any(path.is_symlink() for path in json_files):
        raise ValueError("stale BRAF download may not contain symlinks")

    envelopes: dict[str, tuple[dict[str, Any], str]] = {}
    physical_hashes: dict[str, str] = {}
    payload_hashes: dict[str, str] = {}
    schema_counts: Counter[str] = Counter()
    for path in json_files:
        relative = path.relative_to(input_root).as_posix()
        payload, identity = _load_envelope(path)
        envelopes[relative] = (payload, identity)
        physical_hashes[relative] = sha256_file(path)
        payload_hashes[relative] = identity
        schema = payload.get("schema_version")
        if not isinstance(schema, str):
            raise TypeError(f"JSON envelope has no schema version: {relative}")
        schema_counts[schema] += 1
    if dict(sorted(schema_counts.items())) != EXPECTED_SCHEMA_COUNTS:
        raise ValueError("stale BRAF JSON schema census drift")

    downloaded_launch, _ = envelopes["launch.json"]
    launch_receipt, launch_receipt_payload_sha256 = _load_envelope(launch_receipt_path)
    remote_launch_identity = _validate_launch_identity(
        run_id=run_id,
        downloaded_launch=downloaded_launch,
        launch_receipt=launch_receipt,
    )
    summary, summary_payload_sha256 = envelopes["summary.json"]
    progress, progress_payload_sha256 = envelopes["progress.json"]
    if summary != progress or summary_payload_sha256 != progress_payload_sha256:
        raise ValueError("stale BRAF progress and summary payloads disagree")
    summary_counts = _validate_summary(
        run_id=run_id,
        contract_payload_sha256=str(downloaded_launch["contract_payload_sha256"]),
        summary=summary,
    )

    exclusions: dict[str, list[str]] = {}
    per_cell_counts: dict[str, dict[str, int]] = {}
    for cell in EXPECTED_CELLS:
        hashes, counts = _validate_cell(
            input_root=input_root,
            cell=cell,
            contract_payload_sha256=str(downloaded_launch["contract_payload_sha256"]),
            envelopes=envelopes,
        )
        if counts["charged_complete_receipts"] != summary_counts[cell]:
            raise ValueError(f"stale BRAF summary/receipt mismatch for {cell}")
        exclusions[cell] = hashes
        per_cell_counts[cell] = counts

    receipt_physical_hashes = {
        path: value for path, value in physical_hashes.items() if "/receipts/" in path
    }
    lock_physical_hashes = {
        path: value
        for path, value in physical_hashes.items()
        if path.endswith("_lock.json")
    }
    result_physical_hashes = {
        path: value
        for path, value in physical_hashes.items()
        if path.endswith("_result.json")
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "charged_complete_receipts": sum(EXPECTED_RECEIPTS.values()),
        "unresolved_locked_queries": 0,
        "excluded_canonical_smiles_sha256": exclusions,
        "counts": {
            "canonical_denylist_entries": sum(
                len(value) for value in exclusions.values()
            ),
            "cells": per_cell_counts,
            "json_envelopes": len(envelopes),
            "locked_queries": sum(
                value["locked_queries"] for value in per_cell_counts.values()
            ),
            "round_locks": sum(
                value["round_locks"] for value in per_cell_counts.values()
            ),
            "schema_versions": dict(sorted(schema_counts.items())),
        },
        "input_physical_hashes": {
            "checkpoints": {
                cell: physical_hashes[f"{cell}/checkpoint.json"]
                for cell in EXPECTED_CELLS
            },
            "download_json_payload_manifest_sha256": _mapping_identity(payload_hashes),
            "download_json_tree_manifest_sha256": _mapping_identity(physical_hashes),
            "launch": physical_hashes["launch.json"],
            "progress": physical_hashes["progress.json"],
            "query_receipt_manifest_sha256": _mapping_identity(receipt_physical_hashes),
            "repository_launch_receipt": sha256_file(launch_receipt_path),
            "round_lock_manifest_sha256": _mapping_identity(lock_physical_hashes),
            "round_result_manifest_sha256": _mapping_identity(result_physical_hashes),
            "summary": physical_hashes["summary.json"],
        },
        "remote_launch_identity": {
            **remote_launch_identity,
            "repository_launch_receipt_payload_sha256": (launch_receipt_payload_sha256),
        },
        "canonicalization": {
            "algorithm": (
                "RDKit MolFromSmiles then MolToSmiles(canonical=True,"
                "isomericSmiles=True), followed by SHA-256 of UTF-8 bytes"
            ),
            "rdkit_version": rdBase.rdkitVersion,
        },
        "content_policy": {
            "raw_smiles_materialized": False,
            "scores_materialized": False,
            "terminal_boundary": (
                "round_004_proposals_exist_without_any_round_004_query_lock"
            ),
        },
    }
    validate_reconciliation_payload(payload)
    return payload


def publish_reconciliation(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically publish one immutable deterministic reconciliation envelope."""

    validate_reconciliation_payload(payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite reconciliation: {path}")
    identity = payload_identity(dict(payload))
    encoded = (
        json.dumps(
            {"payload": dict(payload), "payload_sha256": identity},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return identity


def validate_reconciliation_artifact(path: Path) -> dict[str, Any]:
    payload, identity = _load_envelope(path)
    validate_reconciliation_payload(payload)
    return {
        "payload": payload,
        "payload_sha256": identity,
        "sha256": sha256_file(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_reconciliation_payload(
        input_root=args.input_root,
        launch_receipt_path=args.launch_receipt,
    )
    identity = publish_reconciliation(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "payload_sha256": identity,
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CELLS",
    "EXPECTED_JSON_ENVELOPES",
    "EXPECTED_RECEIPTS",
    "EXPECTED_SCHEMA_COUNTS",
    "SCHEMA_VERSION",
    "build_reconciliation_payload",
    "canonical_smiles_sha256",
    "publish_reconciliation",
    "validate_reconciliation_artifact",
    "validate_reconciliation_payload",
]
