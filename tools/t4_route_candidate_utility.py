#!/usr/bin/env python3
"""Prepare, launch and collect score-blind T4 route-policy candidate locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from modal_apps.run_process_v2_p50_app import local_image_revision

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/t4_route_policy_source_sharded/attempt_1"
OUTPUT = ROOT / "diagnostics/t4_route_candidate_utility/attempt_1"
TASKS = OUTPUT / "candidate_lock_tasks_v2.json"
LAUNCH = OUTPUT / "candidate_lock_launch_v2.json"
FAILED_LAUNCH = OUTPUT / "candidate_lock_launch.json"
FAILURE = OUTPUT / "prequery_failure_0001.json"
APP_NAME = "compose-t4-route-candidate-utility"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"
REMOTE_SOURCE = "/artifacts/t4_route_policy_source_sharded/attempt_1"
REMOTE_OUTPUT = "/t4_route_candidate_utility/attempt_1"
MATERIAL_FILES = (
    "AGENTS.md",
    "configs/t4_frozen_program_benchmark_v2.json",
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_route_candidate_utility_app.py",
    "src/compose_v4/experiments/t4_route_candidate_utility.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "tools/t4_route_policy_comparison.py",
)


def _clean_revision():
    from tools.preflight import assert_synced

    commit = assert_synced(strict=True)["commit"]
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _source_tasks(image_revision):
    launch = json.loads((SOURCE / "launch.json").read_text())
    files = {relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES}
    tasks = []
    for source_task in launch["tasks"]:
        folder = SOURCE / f"fold_{source_task['fold']}" / source_task["cell"]
        result_path, model_path = folder / "result.json", folder / "models.json.gz"
        if not result_path.exists() or not model_path.exists():
            continue
        remote_folder = (
            f"{REMOTE_SOURCE}/fold_{source_task['fold']}/{source_task['cell']}"
        )
        body = {
            "schema_version": "t4_route_candidate_utility_task_v1",
            "cell": source_task["cell"],
            "fold": source_task["fold"],
            "source_id": source_task["source_id"],
            "source_index": source_task["source_index"],
            "delta": 0.4,
            "result_path": f"{remote_folder}/result.json",
            "model_path": f"{remote_folder}/models.json.gz",
            "source_artifacts_sha256": {
                f"{remote_folder}/result.json": sha256_file(result_path),
                f"{remote_folder}/models.json.gz": sha256_file(model_path),
            },
            "files_sha256": files,
            "image_revision": image_revision,
            "oracle_calls": 0,
            "automatic_retry": 0,
        }
        tasks.append({**body, "run_id": identity(body)})
    if not 1 <= len(tasks) <= 15:
        raise ValueError(
            "candidate-utility task census must contain 1 to 15 durable sources"
        )
    return sorted(tasks, key=lambda row: row["cell"])


def prepare():
    if TASKS.exists():
        raise ValueError("candidate-lock task manifest exists; preserve it")
    commit, image_revision = _clean_revision()
    tasks = _source_tasks(image_revision)
    publish_json(
        TASKS,
        {
            "schema_version": "t4_route_candidate_utility_task_manifest_v1",
            "commit": commit,
            "tasks": tasks,
            "task_count": len(tasks),
            "maximum_locked_candidates": 2 * len(tasks),
            "costs": {"oracle_calls": 0, "docking_calls": 0},
        },
    )
    print(json.dumps({"status": "prepared", "tasks": len(tasks)}, sort_keys=True))


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError("candidate-lock launch exists; monitor instead")
    body = json.loads(TASKS.read_text())
    worker = modal.Function.from_name(APP_NAME, "lock_worker")
    calls = {task["cell"]: worker.spawn(task).object_id for task in body["tasks"]}
    receipt = {
        "schema_version": "t4_route_candidate_utility_launch_v1",
        "task_manifest_sha256": sha256_file(TASKS),
        "call_ids": calls,
        "automatic_retry": 0,
        "oracle_calls": 0,
    }
    receipt["receipt_sha256"] = identity(receipt)
    publish_json(LAUNCH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def status():
    import modal

    receipt = json.loads(LAUNCH.read_text())
    expected = receipt.pop("receipt_sha256")
    if identity(receipt) != expected:
        raise ValueError("candidate-lock launch receipt changed")
    rows = []
    for cell, call_id in sorted(receipt["call_ids"].items()):
        try:
            value = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"cell": cell, "status": "running"})
        except (ModuleNotFoundError, modal.exception.Error) as error:
            rows.append({"cell": cell, "status": "failed", "error": str(error)})
        else:
            rows.append({"cell": cell, "status": "complete", "result": value})
    print(json.dumps(rows, indent=2, sort_keys=True))


def seal_failure():
    import modal

    if FAILURE.exists():
        raise ValueError("prequery failure already sealed")
    receipt = json.loads(FAILED_LAUNCH.read_text())
    expected = receipt.pop("receipt_sha256")
    if identity(receipt) != expected:
        raise ValueError("failed candidate-lock launch receipt changed")
    failures = []
    for cell, call_id in sorted(receipt["call_ids"].items()):
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except ModuleNotFoundError as error:
            if "tools.t4_route_distillation" not in str(error):
                raise
            failures.append(
                {
                    "cell": cell,
                    "call_id": call_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            raise ValueError(f"failed launch call did not fail closed: {cell}")
    if len(failures) != 9:
        raise ValueError("prequery failure census changed")
    publish_json(
        FAILURE,
        {
            "schema_version": "t4_route_candidate_utility_prequery_failure_v1",
            "launch_sha256": sha256_file(FAILED_LAUNCH),
            "failures": failures,
            "failure_count": len(failures),
            "cause": "Modal image omitted unchanged tools.t4_route_distillation transitive import",
            "costs": {"oracle_calls": 0, "docking_calls": 0},
        },
    )
    print(json.dumps({"status": "sealed", "failures": len(failures)}, sort_keys=True))


def collect():
    import modal

    receipt = json.loads(LAUNCH.read_text())
    expected = receipt.pop("receipt_sha256")
    if identity(receipt) != expected:
        raise ValueError("candidate-lock launch receipt changed")
    volume = modal.Volume.from_name(VOLUME_NAME)
    locks = OUTPUT / "locks"
    collected, preserved = [], []
    for cell, call_id in sorted(receipt["call_ids"].items()):
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            continue
        destination = locks / cell / "candidate_lock.json"
        if destination.exists():
            lock = json.loads(destination.read_text())
            expected = lock.pop("lock_sha256")
            if identity(lock) != expected:
                raise ValueError(f"collected candidate lock changed: {destination}")
            preserved.append(cell)
            continue
        remote = f"{REMOTE_OUTPUT}/{cell}/candidate_lock.json"
        raw = b"".join(volume.read_file(remote))
        lock = json.loads(raw)
        expected = lock.pop("lock_sha256")
        if identity(lock) != expected:
            raise ValueError(f"remote candidate lock changed: {remote}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
        collected.append(cell)
    print(
        json.dumps(
            {
                "status": "incremental_collection_complete",
                "newly_collected": collected,
                "already_preserved": preserved,
                "durable_locks": len(collected) + len(preserved),
                "oracle_calls": 0,
            },
            sort_keys=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("seal-failure", "prepare", "launch", "status", "collect")
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "seal-failure": seal_failure,
        "launch": launch,
        "status": status,
        "collect": collect,
    }[args.action]()


if __name__ == "__main__":
    main()
