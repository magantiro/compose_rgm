"""Audit, reconcile and resume interrupted frozen T4 benchmark units."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_rescue import (
    LOCK_SCHEMA,
    make_ambiguous_tombstone,
    raw_sha256,
    read_gzip_bytes,
    unseal_bytes,
    validate_locked_round,
    validate_query_paths,
    validate_rescue_lock,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/t4_frozen_program_rescue.json"
LOCK = Path("diagnostics/t4_frozen_program_rescue/rescue_lock.json")
LAUNCH = Path("diagnostics/t4_frozen_program_rescue/launch.json")
QUERY_PATTERN = re.compile(r"/queries/query_(\d{4})/(started|result)\.json$")


def _stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _remote_bytes(volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _paths(volume, prefix: str) -> set[str]:
    return {entry.path.lstrip("/") for entry in volume.listdir(prefix, recursive=True)}


def _contract() -> dict:
    from compose_v4.experiments.t4_frozen_program_rescue import load_rescue_contract

    return load_rescue_contract(ROOT, CONFIG)


def _source_contract(contract: dict) -> dict:
    source = unseal(ROOT / contract["source_contract"])
    if len(source["units"]) != 45:
        raise ValueError("source T4 contract lost its 45-unit census")
    return source


def _read_and_match(volume, path: str, expected_sha256: str) -> bytes:
    raw = _remote_bytes(volume, path)
    observed = raw_sha256(raw)
    if observed != expected_sha256:
        raise ValueError(f"remote T4 artifact changed: {path}: {observed}")
    return raw


def prepare() -> dict:
    import modal

    output = ROOT / LOCK
    if output.exists():
        raise ValueError("T4 rescue lock already exists; do not relock")
    contract = _contract()
    source_contract = _source_contract(contract)
    volume = modal.Volume.from_name(contract["source_volume"])
    prefix = contract["source_volume_path"].strip("/")

    started_raw = _read_and_match(
        volume, f"{prefix}/started.json", contract["source_run_started_sha256"]
    )
    task = unseal_bytes(started_raw, source=f"{prefix}/started.json")
    if (
        task.get("run_id") != contract["source_run_id"]
        or task.get("image_revision", {}).get("commit")
        != contract["source_deployed_commit"]
    ):
        raise ValueError("remote T4 start receipt changed run or deployed revision")
    _read_and_match(volume, f"{prefix}/result.json", contract["stale_aggregate_sha256"])
    late_path = f"{prefix}/units/{contract['late_completed_unit']}/result.json"
    _read_and_match(volume, late_path, contract["late_completed_unit_result_sha256"])

    unit_contracts = {row["unit_id"]: row for row in source_contract["units"]}
    expected_ambiguous = contract["ambiguous_started_counts"]
    reconciled_at = _stamp()
    locked_units: dict[str, dict] = {}
    complete_units: dict[str, dict] = {}
    for unit_id in sorted(unit_contracts):
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        result_path = f"{unit_prefix}/result.json"
        if result_path in paths:
            raw = _remote_bytes(volume, result_path)
            result = unseal_bytes(raw, source=result_path)
            if (
                result.get("status") != "complete"
                or result.get("unit", {}).get("unit_id") != unit_id
            ):
                raise ValueError(f"completed T4 unit identity is invalid: {unit_id}")
            complete_units[unit_id] = {
                "result_sha256": raw_sha256(raw),
                "oracle_calls": result["oracle_calls"],
                "termination": result["termination"],
            }
            continue
        if unit_id not in expected_ambiguous:
            raise ValueError(f"unexpected incomplete T4 unit: {unit_id}")

        started_indices, result_indices = set(), set()
        for path in paths:
            match = QUERY_PATTERN.search(path)
            if not match:
                continue
            index, kind = int(match.group(1)), match.group(2)
            (started_indices if kind == "started" else result_indices).add(index)
        ambiguous_index = validate_query_paths(started_indices, result_indices)
        if len(started_indices) != expected_ambiguous[unit_id]:
            raise ValueError(f"T4 interrupted ledger count changed: {unit_id}")

        query_prefix = f"{unit_prefix}/queries/query_{ambiguous_index:04d}"
        ambiguous_started_path = f"{query_prefix}/started.json"
        started_query_raw = _remote_bytes(volume, ambiguous_started_path)
        started_query = unseal_bytes(started_query_raw, source=ambiguous_started_path)
        unit = unit_contracts[unit_id]
        if (
            started_query["query_index"] != ambiguous_index
            or started_query["target"] != unit["target"]
            or started_query["docking_seed"] != unit["docking_seed"]
        ):
            raise ValueError(f"ambiguous T4 query changed unit identity: {unit_id}")

        batch_path = f"{unit_prefix}/rounds/round_{started_query['round_index']:04d}/batch.json.gz"
        batch_raw = _remote_bytes(volume, batch_path)
        batch_lock = read_gzip_bytes(batch_raw, source=batch_path)
        if batch_lock.get("round") != started_query["round_index"]:
            raise ValueError(f"ambiguous T4 query changed locked round: {unit_id}")
        validate_locked_round(started_query, batch_lock["batch"])

        checkpoint_path = f"{unit_prefix}/checkpoint.json.gz"
        checkpoint_raw = _remote_bytes(volume, checkpoint_path)
        checkpoint = read_gzip_bytes(checkpoint_raw, source=checkpoint_path)
        if (
            checkpoint["query_count"] > len(result_indices)
            or checkpoint["query_count"] != len(checkpoint["curve"])
            or checkpoint["next_round"] > started_query["round_index"]
        ):
            raise ValueError(
                f"interrupted T4 checkpoint cannot resume safely: {unit_id}"
            )

        tombstone = make_ambiguous_tombstone(
            started_query, reconciled_at_utc=reconciled_at
        )
        tombstone_path = (
            ROOT
            / "diagnostics/t4_frozen_program_rescue/tombstones"
            / unit_id
            / "result.json"
        )
        seal(tombstone_path, tombstone)
        locked_units[unit_id] = {
            "unit": unit,
            "started_count": len(started_indices),
            "completed_result_count": len(result_indices),
            "ambiguous_query_index": ambiguous_index,
            "started_path": ambiguous_started_path,
            "started_sha256": raw_sha256(started_query_raw),
            "started": started_query,
            "batch_path": batch_path,
            "batch_sha256": raw_sha256(batch_raw),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": raw_sha256(checkpoint_raw),
            "checkpoint_query_count": checkpoint["query_count"],
            "tombstone_path": str(tombstone_path.relative_to(ROOT)),
            "tombstone_sha256": sha256_file(tombstone_path),
            "remote_tombstone_path": f"{query_prefix}/result.json",
            "maximum_new_search_calls": unit["budget"] - len(started_indices),
        }

    if len(complete_units) != contract["expected_complete_units"]:
        raise ValueError(
            "live T4 completed-unit count changed during rescue preparation"
        )
    if set(locked_units) != set(expected_ambiguous):
        raise ValueError(
            "live T4 interrupted-unit census changed during rescue preparation"
        )
    if (
        sum(row["maximum_new_search_calls"] for row in locked_units.values())
        != contract["new_search_call_ceiling"]
    ):
        raise ValueError("live T4 remaining search ceiling changed")

    confirmation_paths = sorted(
        path
        for path in _paths(volume, f"{prefix}/confirmations")
        if path.endswith("/result.json")
    )
    if len(confirmation_paths) != 24:
        raise ValueError("stale T4 aggregate confirmation count changed")
    confirmations = {
        path: raw_sha256(_remote_bytes(volume, path)) for path in confirmation_paths
    }
    implementation = {
        path: sha256_file(ROOT / path)
        for path in (
            CONFIG,
            "docs/T4_FROZEN_PROGRAM_RESCUE.md",
            "src/compose_v4/experiments/t4_frozen_program_rescue.py",
            "tools/t4_frozen_program_rescue.py",
        )
    }
    body = {
        "schema_version": LOCK_SCHEMA,
        "source_run_id": contract["source_run_id"],
        "source_volume": contract["source_volume"],
        "source_volume_path": prefix,
        "source_task": task,
        "source_run_started_sha256": contract["source_run_started_sha256"],
        "stale_aggregate_sha256": contract["stale_aggregate_sha256"],
        "complete_units": complete_units,
        "units": locked_units,
        "historical_confirmation_results": confirmations,
        "historical_confirmation_calls": len(confirmations),
        "new_search_call_ceiling": contract["new_search_call_ceiling"],
        "new_confirmation_call_ceiling": contract["new_confirmation_call_ceiling"],
        "automatic_retries": 0,
        "contract_path": CONFIG,
        "contract_sha256": sha256_file(ROOT / CONFIG),
        "implementation_sha256": implementation,
        "prepared_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "prepared_at_utc": reconciled_at,
        "new_oracle_calls": 0,
    }
    seal(output, body)
    return body


def _assert_remote_lock(volume, lock: dict) -> None:
    prefix = lock["source_volume_path"]
    _read_and_match(volume, f"{prefix}/started.json", lock["source_run_started_sha256"])
    _read_and_match(volume, f"{prefix}/result.json", lock["stale_aggregate_sha256"])
    for unit_id, row in lock["units"].items():
        _read_and_match(volume, row["started_path"], row["started_sha256"])
        _read_and_match(volume, row["batch_path"], row["batch_sha256"])
        _read_and_match(volume, row["checkpoint_path"], row["checkpoint_sha256"])
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 unit completed after rescue lock: {unit_id}")
        if row["remote_tombstone_path"] in paths:
            raise ValueError(f"T4 ambiguous query already has a result: {unit_id}")


def launch() -> dict:
    import modal

    receipt_path = ROOT / LAUNCH
    if receipt_path.exists():
        raise ValueError("T4 rescue launch already exists; use status")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("T4 rescue launch requires clean committed source")
    contract = _contract()
    lock = validate_rescue_lock(ROOT, ROOT / LOCK, contract)
    volume = modal.Volume.from_name(contract["source_volume"])
    _assert_remote_lock(volume, lock)

    with volume.batch_upload(force=False) as upload:
        for row in lock["units"].values():
            upload.put_file(ROOT / row["tombstone_path"], row["remote_tombstone_path"])
    for unit_id, row in lock["units"].items():
        remote = _remote_bytes(volume, row["remote_tombstone_path"])
        if raw_sha256(remote) != row["tombstone_sha256"]:
            raise ValueError(f"uploaded T4 tombstone changed: {unit_id}")

    receipt = {
        "schema_version": "t4_frozen_program_rescue_launch_v1",
        "source_run_id": contract["source_run_id"],
        "source_app": contract["source_app"],
        "rescue_lock_sha256": sha256_file(ROOT / LOCK),
        "launched_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "units": {},
        "automatic_retries": 0,
        "launched_at_utc": _stamp(),
    }
    seal(receipt_path, receipt)
    worker = modal.Function.from_name(contract["source_app"], "worker")
    for unit_id in sorted(lock["units"]):
        receipt["units"][unit_id] = {"status": "spawn_reserved", "at": _stamp()}
        seal(receipt_path, receipt)
        call = worker.spawn({**lock["source_task"], "unit_id": unit_id})
        receipt["units"][unit_id] = {
            "status": "spawned",
            "call_id": call.object_id,
            "at": _stamp(),
        }
        seal(receipt_path, receipt)
    return receipt


def status() -> dict:
    import modal

    contract = _contract()
    source_contract = _source_contract(contract)
    volume = modal.Volume.from_name(contract["source_volume"])
    prefix = contract["source_volume_path"].strip("/")
    rows = []
    for unit in source_contract["units"]:
        unit_prefix = f"{prefix}/units/{unit['unit_id']}"
        paths = _paths(volume, unit_prefix)
        artifact = next(
            (
                path
                for path in (
                    f"{unit_prefix}/result.json",
                    f"{unit_prefix}/progress.json",
                    f"{unit_prefix}/failure.json",
                )
                if path in paths
            ),
            None,
        )
        payload = (
            {}
            if artifact is None
            else unseal_bytes(_remote_bytes(volume, artifact), source=artifact)
        )
        calls = payload.get(
            "oracle_calls", payload.get("queries_total", payload.get("queries", 0))
        )
        champion = payload.get("champion")
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "status": payload.get(
                    "status",
                    (
                        "running"
                        if artifact and artifact.endswith("progress.json")
                        else "pending"
                    ),
                ),
                "calls": calls,
                "best_score": None if champion is None else champion.get("score"),
                "termination": payload.get("termination"),
                "updated_at_utc": payload.get("completed_at_utc", payload.get("at")),
            }
        )
    result = {
        "source_run_id": contract["source_run_id"],
        "completed_units": sum(row["status"] == "complete" for row in rows),
        "running_units": sum(row["status"] == "running" for row in rows),
        "failed_units": sum(row["status"] == "failed" for row in rows),
        "total_charged_search_calls": sum(row["calls"] for row in rows),
        "units": rows,
        "checked_at_utc": _stamp(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "status"))
    args = parser.parse_args()
    result = (
        prepare()
        if args.action == "prepare"
        else launch() if args.action == "launch" else status()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
