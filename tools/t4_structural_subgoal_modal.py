#!/usr/bin/env python3
"""Launch and query the durable 15-source structural-subgoal Modal fan-out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from modal_apps.run_process_v2_p50_app import local_image_revision
from modal_apps.t4_structural_subgoal_app import APP_NAME, MATERIAL_FILES, VOLUME_NAME
from tools.t4_structural_subgoal_audit import source_groups

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / "diagnostics/t4_structural_subgoal/attempt_2"
LAUNCH = LOCAL_ROOT / "launch.json"
REMOTE_ROOT = "/t4_structural_subgoal/attempt_2"


def _clean_revision():
    from tools.preflight import assert_synced

    commit = assert_synced(strict=True)["commit"]
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _tasks(image_revision):
    files = {relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES}
    rows = []
    for source_group in source_groups():
        body = {
            "schema_version": "t4_structural_subgoal_source_lock_v1",
            "source_group": source_group,
            "output": f"/artifacts{REMOTE_ROOT}/{source_group}/result.json",
            "oracle_calls": 0,
            "automatic_retry": 0,
            "image_revision": image_revision,
            "files_sha256": files,
        }
        rows.append({**body, "run_id": identity(body)})
    return rows


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError("structural-subgoal launch exists; query it instead")
    commit, image_revision = _clean_revision()
    tasks = _tasks(image_revision)
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {task["source_group"]: worker.spawn(task).object_id for task in tasks}
    body = {
        "schema_version": "t4_structural_subgoal_launch_v1",
        "commit": commit,
        "tasks": tasks,
        "call_ids": calls,
        "modal_volume": VOLUME_NAME,
        "concurrent_single_cpu_workers": 15,
        "oracle_calls_authorized": 0,
        "automatic_retry": 0,
    }
    publish_json(LAUNCH, {**body, "receipt_sha256": identity(body)})
    print(json.dumps({"status": "launched", "workers": len(calls), "commit": commit}))


def _receipt():
    receipt = json.loads(LAUNCH.read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if identity(body) != receipt.get("receipt_sha256"):
        raise ValueError("structural-subgoal launch receipt changed")
    return receipt


def _status_rows(receipt):
    import modal

    rows = []
    for source_group, call_id in sorted(receipt["call_ids"].items()):
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"source_group": source_group, "status": "running"})
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            rows.append(
                {
                    "source_group": source_group,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            rows.append(
                {"source_group": source_group, "status": "complete", "result": result}
            )
    return rows


def status():
    rows = _status_rows(_receipt())
    counts = {
        state: sum(row["status"] == state for row in rows)
        for state in ("running", "complete", "failed")
    }
    print(json.dumps({"counts": counts, "sources": rows}, indent=2, sort_keys=True))


def collect():
    import modal

    receipt = _receipt()
    rows = _status_rows(receipt)
    complete = {row["source_group"] for row in rows if row["status"] == "complete"}
    volume = modal.Volume.from_name(receipt["modal_volume"])
    downloaded = []
    for source_group in sorted(complete):
        destination = LOCAL_ROOT / "sources" / source_group / "result.json"
        if destination.exists():
            continue
        raw = b"".join(volume.read_file(f"{REMOTE_ROOT}/{source_group}/result.json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
        downloaded.append(source_group)
    print(json.dumps({"complete": len(complete), "downloaded": downloaded}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "status", "collect"))
    args = parser.parse_args()
    {"launch": launch, "status": status, "collect": collect}[args.action]()


if __name__ == "__main__":
    main()
