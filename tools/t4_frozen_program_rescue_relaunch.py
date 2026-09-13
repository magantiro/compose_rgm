"""Freeze and launch the exact T4 tombstone-status compatibility repair."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_rescue import (
    RELAUNCH_LOCK_SCHEMA,
    TOMBSTONE_FAILURE,
    raw_sha256,
    read_gzip_bytes,
    unseal_bytes,
    validate_locked_round,
    validate_relaunch_query_paths,
    validate_tombstone,
    validate_tombstone_compatibility_failure,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]
FIRST_LOCK = Path("diagnostics/t4_frozen_program_rescue/rescue_lock.json")
FIRST_LAUNCH = Path("diagnostics/t4_frozen_program_rescue/launch.json")
RELAUNCH_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v2.json")
RELAUNCH_RECEIPT = Path("diagnostics/t4_frozen_program_rescue/relaunch_v2.json")
FAILURE_ARCHIVE = Path("diagnostics/t4_frozen_program_rescue/first_relaunch_failures")
APP = "modal_apps/t4_frozen_program_rescue_app.py"
APP_NAME = "compose-t4-frozen-program-rescue"
QUERY_PATTERN = re.compile(r"/queries/query_(\d{4})/(started|result)\.json$")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remote_bytes(volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _paths(volume, prefix: str) -> set[str]:
    return {entry.path.lstrip("/") for entry in volume.listdir(prefix, recursive=True)}


def _sealed_remote(volume, path: str) -> tuple[bytes, dict]:
    raw = _remote_bytes(volume, path)
    return raw, unseal_bytes(raw, source=path)


def _query_indices(paths: set[str]) -> tuple[set[int], set[int]]:
    started, results = set(), set()
    for path in paths:
        match = QUERY_PATTERN.search(path)
        if match:
            (started if match.group(2) == "started" else results).add(
                int(match.group(1))
            )
    return started, results


def _critical_delta(source_commit: str) -> dict:
    paths = (
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/edit_program_policy.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "modal_apps/t4_frozen_program_benchmark_app.py",
        "modal_apps/genmol_t4_opt_app.py",
        "configs/t4_frozen_program_benchmark_v2.json",
        "diagnostics/t4_frozen_program_benchmark/preflight_v2.json",
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", source_commit, "HEAD", "--", *paths],
        cwd=ROOT,
        text=True,
    ).splitlines()
    allowed = {
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/edit_program_policy.py",
        "src/compose_v4/control/program_transfer.py",
    }
    if not set(changed) <= allowed:
        raise ValueError(f"frozen T4 scientific inputs changed: {changed}")
    return {
        "source_commit": source_commit,
        "changed_critical_paths": changed,
        "diff_sha256": identity(
            subprocess.check_output(
                ["git", "diff", source_commit, "HEAD", "--", *paths], cwd=ROOT
            ).decode()
        ),
        "decision_equivalence": {
            "cold_start_retrieval_candidates": 0,
            "ranked_retrieval_enabled": False,
            "only_effective_change": "accept_exact_charged_missing_observation_status",
        },
    }


def prepare() -> dict:
    import modal

    if (ROOT / RELAUNCH_LOCK).exists():
        raise ValueError("T4 repaired relaunch lock already exists")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("prepare repaired relaunch from clean committed source")
    first_lock = unseal(ROOT / FIRST_LOCK)
    first_launch = unseal(ROOT / FIRST_LAUNCH)
    volume = modal.Volume.from_name(first_lock["source_volume"])
    prefix = first_lock["source_volume_path"]
    source_task = first_lock["source_task"]
    config = ProgramSearchConfig(
        **unseal(ROOT / "configs/t4_frozen_program_benchmark_v2.json")["controller"]
    )
    if config.cold_start_retrieval_candidates != 0:
        raise ValueError(
            "frozen T4 config unexpectedly enables later retrieval behavior"
        )

    units, active, completed = {}, {}, {}
    reconciled_at = _stamp()
    for unit_id, launch in sorted(first_launch["units"].items()):
        call = modal.FunctionCall.from_id(launch["call_id"])
        try:
            call_result = call.get(timeout=1)
        except TimeoutError:
            if first_lock["units"][unit_id]["checkpoint_path"] is not None:
                raise ValueError(
                    f"non-bootstrap first relaunch is unexpectedly active: {unit_id}"
                )
            active[unit_id] = {"call_id": launch["call_id"], "status": "running"}
            continue
        if call_result.get("status") == "complete":
            completed[unit_id] = {
                "call_id": launch["call_id"],
                "result_sha256": identity(call_result),
                "oracle_calls": call_result["oracle_calls"],
            }
            continue
        validate_tombstone_compatibility_failure(call_result, unit_id=unit_id)

        locked = first_lock["units"][unit_id]
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(
                f"failed first relaunch also has a final result: {unit_id}"
            )
        failure_path = f"{unit_prefix}/failure.json"
        failure_raw, failure = _sealed_remote(volume, failure_path)
        validate_tombstone_compatibility_failure(failure, unit_id=unit_id)
        if failure != call_result:
            raise ValueError(
                f"call result differs from durable unit failure: {unit_id}"
            )

        started_indices, result_indices = _query_indices(paths)
        current_count = validate_relaunch_query_paths(started_indices, result_indices)
        if not locked["started_count"] <= current_count <= locked["unit"]["budget"]:
            raise ValueError(
                f"relaunch query count is outside its frozen bounds: {unit_id}"
            )
        batch_raw = _remote_bytes(volume, locked["batch_path"])
        if raw_sha256(batch_raw) != locked["batch_sha256"]:
            raise ValueError(f"first relaunch changed its locked batch: {unit_id}")
        round_lock = read_gzip_bytes(batch_raw, source=locked["batch_path"])
        batch = round_lock["batch"]
        if locked["checkpoint_path"] is not None:
            checkpoint_raw = _remote_bytes(volume, locked["checkpoint_path"])
            if raw_sha256(checkpoint_raw) != locked["checkpoint_sha256"]:
                raise ValueError(
                    f"first relaunch changed pre-round checkpoint: {unit_id}"
                )

        observations = []
        for index in range(locked["ambiguous_query_index"], current_count):
            query_prefix = f"{unit_prefix}/queries/query_{index:04d}"
            started_raw, started = _sealed_remote(
                volume, f"{query_prefix}/started.json"
            )
            result_raw, result = _sealed_remote(volume, f"{query_prefix}/result.json")
            if any(result.get(key) != value for key, value in started.items()):
                raise ValueError(
                    f"relaunch result changed its reservation: {unit_id}/{index}"
                )
            if (
                started["query_index"] != index
                or started["round_index"] != round_lock["round"]
            ):
                raise ValueError(
                    f"relaunch query changed its locked round: {unit_id}/{index}"
                )
            validate_locked_round(started, batch)
            if index == locked["ambiguous_query_index"]:
                validate_tombstone(result, started)
                if result["failure"] != TOMBSTONE_FAILURE:
                    raise ValueError(f"relaunch tombstone status changed: {unit_id}")
            observations.append(
                {
                    "query_index": index,
                    "started_sha256": raw_sha256(started_raw),
                    "result_sha256": raw_sha256(result_raw),
                    "receipt_id": result["receipt_id"],
                    "score": result["score"],
                    "failure": result["failure"],
                }
            )
        if failure["oracle_calls"] != current_count:
            raise ValueError(f"failure/query accounting differs: {unit_id}")
        archive_path = ROOT / FAILURE_ARCHIVE / f"{unit_id}.json"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(failure_raw)
        units[unit_id] = {
            "unit": locked["unit"],
            "call_id": launch["call_id"],
            "failure_path": str(archive_path.relative_to(ROOT)),
            "failure_sha256": raw_sha256(failure_raw),
            "current_query_count": current_count,
            "first_relaunch_new_calls": current_count - locked["started_count"],
            "remaining_call_ceiling": locked["unit"]["budget"] - current_count,
            "locked_batch_path": locked["batch_path"],
            "locked_batch_sha256": locked["batch_sha256"],
            "checkpoint_path": locked["checkpoint_path"],
            "checkpoint_sha256": locked["checkpoint_sha256"],
            "reconciled_observations": observations,
        }

    if len(units) + len(active) + len(completed) != len(first_lock["units"]):
        raise ValueError("first relaunch census is incomplete")
    if len(units) + len(active) > 22:
        raise ValueError("repaired relaunch would exceed rescue concurrency")
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    implementation = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            "src/compose_v4/control/adaptive_program_optimizer.py",
            "src/compose_v4/experiments/t4_frozen_program_rescue.py",
            "tools/t4_frozen_program_rescue_relaunch.py",
            "docs/T4_FROZEN_PROGRAM_RESCUE.md",
        )
    }
    body = {
        "schema_version": RELAUNCH_LOCK_SCHEMA,
        "source_run_id": first_lock["source_run_id"],
        "source_volume": first_lock["source_volume"],
        "source_volume_path": prefix,
        "source_task": source_task,
        "first_rescue_lock_sha256": sha256_file(ROOT / FIRST_LOCK),
        "first_rescue_launch_sha256": sha256_file(ROOT / FIRST_LAUNCH),
        "units": units,
        "active_first_relaunch_units": active,
        "completed_first_relaunch_units": completed,
        "first_relaunch_new_calls": sum(
            row["first_relaunch_new_calls"] for row in units.values()
        ),
        "new_search_call_ceiling": sum(
            row["remaining_call_ceiling"] for row in units.values()
        ),
        "automatic_retries": 0,
        "controller_changes": False,
        "effective_repair": "accept_exact_charged_missing_observation_status",
        "critical_source_delta": _critical_delta(
            first_lock["source_task"]["image_revision"]["commit"]
        ),
        "implementation_sha256": implementation,
        "prepared_code_revision": code_revision,
        "prepared_at_utc": reconciled_at,
        "new_oracle_calls_during_prepare": 0,
    }
    seal(ROOT / RELAUNCH_LOCK, body)
    return body


def _validate_lock() -> dict:
    lock = unseal(ROOT / RELAUNCH_LOCK)
    if (
        lock.get("schema_version") != RELAUNCH_LOCK_SCHEMA
        or lock.get("automatic_retries") != 0
        or lock.get("controller_changes") is not False
        or lock.get("effective_repair")
        != "accept_exact_charged_missing_observation_status"
    ):
        raise ValueError("T4 repaired relaunch lock changed scope")
    for unit_id, row in lock["units"].items():
        if sha256_file(ROOT / row["failure_path"]) != row["failure_sha256"]:
            raise ValueError(f"preserved first-relaunch failure changed: {unit_id}")
    for path, digest in lock["implementation_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"T4 repaired relaunch implementation changed: {path}")
    return lock


def launch() -> dict:
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if (ROOT / RELAUNCH_RECEIPT).exists():
        raise ValueError("T4 repaired relaunch receipt already exists")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("T4 repaired relaunch requires clean committed source")
    lock = _validate_lock()
    volume = modal.Volume.from_name(lock["source_volume"])
    prefix = lock["source_volume_path"]
    for unit_id, row in lock["units"].items():
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 unit completed after repaired lock: {unit_id}")
        failure_raw = _remote_bytes(volume, f"{unit_prefix}/failure.json")
        if raw_sha256(failure_raw) != row["failure_sha256"]:
            raise ValueError(f"remote first-relaunch failure changed: {unit_id}")
        started, results = _query_indices(paths)
        if (
            validate_relaunch_query_paths(started, results)
            != row["current_query_count"]
        ):
            raise ValueError(f"remote relaunch ledger changed: {unit_id}")

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"]
    )
    files = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            str(RELAUNCH_LOCK),
            "modal_apps/t4_frozen_program_benchmark_app.py",
            "modal_apps/genmol_t4_opt_app.py",
        )
    }
    body = {
        "rescue_image_revision": revision,
        "relaunch_lock_sha256": sha256_file(ROOT / RELAUNCH_LOCK),
        "files_sha256": files,
    }
    task = {**body, "run_id": identity(body)}
    receipt = {
        "schema_version": "t4_frozen_program_rescue_relaunch_v2",
        "task": task,
        "units": {},
        "automatic_retries": 0,
        "launched_at_utc": _stamp(),
    }
    seal(ROOT / RELAUNCH_RECEIPT, receipt)
    worker = modal.Function.from_name(APP_NAME, "worker")
    for unit_id in sorted(lock["units"]):
        receipt["units"][unit_id] = {"status": "spawn_reserved", "at": _stamp()}
        seal(ROOT / RELAUNCH_RECEIPT, receipt)
        call = worker.spawn({**task, "unit_id": unit_id})
        receipt["units"][unit_id] = {
            "status": "spawned",
            "call_id": call.object_id,
            "at": _stamp(),
        }
        seal(ROOT / RELAUNCH_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch"))
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else launch()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
