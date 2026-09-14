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
TASKS = OUTPUT / "candidate_lock_tasks.json"
LAUNCH = OUTPUT / "candidate_lock_launch.json"
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
            "result_path": f"/artifacts{remote_folder}/result.json",
            "model_path": f"/artifacts{remote_folder}/models.json.gz",
            "source_artifacts_sha256": {
                f"/artifacts{remote_folder}/result.json": sha256_file(result_path),
                f"/artifacts{remote_folder}/models.json.gz": sha256_file(model_path),
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
    manifest = json.loads(TASKS.read_text())
    body = manifest["payload"]
    if identity(body) != manifest.get("payload_sha256"):
        raise ValueError("candidate-lock task manifest changed")
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

    receipt = json.loads(LAUNCH.read_text())["payload"]
    rows = []
    for cell, call_id in sorted(receipt["call_ids"].items()):
        try:
            value = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"cell": cell, "status": "running"})
        except modal.exception.Error as error:
            rows.append({"cell": cell, "status": "failed", "error": str(error)})
        else:
            rows.append({"cell": cell, "status": "complete", "result": value})
    print(json.dumps(rows, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "status"))
    args = parser.parse_args()
    {"prepare": prepare, "launch": launch, "status": status}[args.action]()


if __name__ == "__main__":
    main()
