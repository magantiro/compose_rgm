"""Launch and monitor the three-fold zero-oracle T4 policy comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from modal_apps.run_process_v2_p50_app import local_image_revision

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "compose-t4-route-policy-parallel"
LAUNCH = ROOT / "diagnostics/t4_route_policy_parallel/attempt_1/launch.json"
REMOTE_OUTPUT_ROOT = "/artifacts/t4_route_policy_comparison_parallel/attempt_1"
MATERIAL_FILES = (
    "modal_apps/t4_route_policy_parallel_app.py",
    "tools/t4_route_policy_parallel.py",
    "tools/t4_route_policy_comparison.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "docs/GENMOL_T4_SEEDS.json",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
)


def _clean_revision():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise ValueError(f"parallel T4 launch requires clean source: {status}")
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _tasks(image_revision):
    files = {relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES}
    result = []
    for fold in range(3):
        body = {
            "schema_version": "t4_route_policy_parallel_fold_lock_v1",
            "fold": fold,
            "output": f"{REMOTE_OUTPUT_ROOT}/fold_{fold}",
            "oracle_calls": 0,
            "automatic_retry": 0,
            "image_revision": image_revision,
            "files_sha256": files,
        }
        result.append({**body, "run_id": identity(body)})
    return result


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError("parallel T4 launch exists; monitor rather than duplicate")
    commit, image_revision = _clean_revision()
    tasks = _tasks(image_revision)
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {str(task["fold"]): worker.spawn(task).object_id for task in tasks}
    body = {
        "schema_version": "t4_route_policy_parallel_launch_v1",
        "commit": commit,
        "tasks": tasks,
        "call_ids": calls,
        "modal_volume": "compose-v4-artifacts",
        "concurrent_single_cpu_workers": 3,
        "oracle_calls_authorized": 0,
    }
    receipt = {**body, "receipt_sha256": identity(body)}
    publish_json(LAUNCH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _receipt():
    value = json.loads(LAUNCH.read_text())
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if identity(body) != value.get("receipt_sha256"):
        raise ValueError("parallel T4 launch receipt changed")
    return value


def status():
    import modal

    receipt = _receipt()
    rows = []
    for fold, call_id in sorted(receipt["call_ids"].items()):
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"fold": int(fold), "status": "running"})
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            rows.append(
                {
                    "fold": int(fold),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            rows.append({"fold": int(fold), "status": "complete", "result": result})
    print(json.dumps({"folds": rows}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status"))
    args = parser.parse_args()
    {"launch": launch, "status": status}[args.action]()


if __name__ == "__main__":
    main()
