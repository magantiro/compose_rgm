"""Launch, monitor, collate, evaluate, and download PMO decoder shards."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from modal_apps.pmo_complete_program_decoder_app import (
    APP_NAME,
    MATERIAL_FILES,
    OUTPUT_ROOT,
    SOURCE_MANIFEST,
)
from modal_apps.run_process_v2_p50_app import local_image_revision
from tools.pmo_complete_program_decoder import _load_contract, _load_envelope

ROOT = Path(__file__).resolve().parents[1]
LOCAL_OUTPUT = ROOT / "diagnostics/pmo_complete_program_decoder/attempt_1"
LAUNCH = LOCAL_OUTPUT / "launch.json"
COLLATE_LAUNCH = LOCAL_OUTPUT / "collate_launch.json"
EVALUATE_LAUNCH = LOCAL_OUTPUT / "evaluate_launch.json"
VOLUME = "compose-v4-artifacts"
VOLUME_MOUNT = Path("/artifacts")


def _clean_revision() -> tuple[str, dict]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise ValueError(f"PMO decoder launch requires clean source: {status}")
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _common_body(image_revision: dict, contract_sha256: str, manifest_sha256: str):
    return {
        "contract_sha256": contract_sha256,
        "source_manifest_payload_sha256": manifest_sha256,
        "new_oracle_calls": 0,
        "automatic_retries": 0,
        "image_revision": image_revision,
        "files_sha256": {
            relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES
        },
    }


def _seal_task(body: dict) -> dict:
    return {**body, "run_id": identity(body)}


def _load_receipt(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("schema_version") != schema or value.get("receipt_sha256") != identity(
        body
    ):
        raise ValueError(f"invalid PMO decoder receipt: {path}")
    return value


def _publish_receipt(path: Path, body: dict) -> None:
    publish_json(path, {**body, "receipt_sha256": identity(body)})


def _relaunch_path(index: int) -> Path:
    return LOCAL_OUTPUT / f"case_{index:02d}_relaunch_1.json"


def _effective_call_ids(launch_receipt: dict) -> dict[str, str]:
    call_ids = dict(launch_receipt["call_ids"])
    tasks = {str(task["source_case_index"]): task for task in launch_receipt["tasks"]}
    for index, task in tasks.items():
        path = _relaunch_path(int(index))
        if not path.exists():
            continue
        receipt = _load_receipt(path, "pmo_complete_program_decoder_case_relaunch_v1")
        if (
            receipt.get("parent_launch_sha256") != launch_receipt["receipt_sha256"]
            or receipt.get("source_case_index") != int(index)
            or receipt.get("task_run_id") != task["run_id"]
            or receipt.get("manual_relaunch") != 1
            or receipt.get("automatic_retries") != 0
            or receipt.get("oracle_calls_authorized") != 0
        ):
            raise ValueError(f"invalid PMO decoder case relaunch lineage: {path}")
        call_ids[index] = receipt["call_id"]
    return call_ids


def launch() -> None:
    import modal

    if LAUNCH.exists():
        raise ValueError("PMO decoder launch exists; use status instead")
    commit, image_revision = _clean_revision()
    contract, contract_sha256 = _load_contract(ROOT)
    if contract.get("authoritative_execution_enabled") is not True:
        raise ValueError("PMO decoder authoritative execution is disabled")
    manifest, manifest_sha256 = _load_envelope(ROOT / SOURCE_MANIFEST)
    if manifest.get("source_case_count") != 9:
        raise ValueError("PMO decoder source manifest must contain nine cases")
    common = _common_body(image_revision, contract_sha256, manifest_sha256)
    tasks = []
    for index, source in enumerate(manifest["source_cases"]):
        output = (
            OUTPUT_ROOT / "shards" / f"case_{index:02d}_{source['source_case_id'][:12]}"
        )
        tasks.append(
            _seal_task(
                {
                    "schema_version": "pmo_complete_program_decoder_case_task_v1",
                    **common,
                    "source_case_index": index,
                    "source_case_id": source["source_case_id"],
                    "fold": int(source["fold"]),
                    "output": str(output),
                }
            )
        )
    worker = modal.Function.from_name(APP_NAME, "generate_case")
    call_ids = {
        str(task["source_case_index"]): worker.spawn(task).object_id for task in tasks
    }
    body = {
        "schema_version": "pmo_complete_program_decoder_parallel_launch_v1",
        "commit": commit,
        "contract_sha256": contract_sha256,
        "source_manifest_payload_sha256": manifest_sha256,
        "tasks": tasks,
        "call_ids": call_ids,
        "modal_app": APP_NAME,
        "modal_volume": VOLUME,
        "remote_output": str(OUTPUT_ROOT),
        "source_case_shards": 9,
        "concurrent_single_cpu_workers": 9,
        "restart_unit": "one_source_fold_case",
        "worker_timeout_seconds": 21600,
        "automatic_retries": 0,
        "oracle_calls_authorized": 0,
        "docking_calls_authorized": 0,
        "estimated_cpu_hours_ceiling": 54,
    }
    _publish_receipt(LAUNCH, body)
    print(json.dumps({**body, "receipt_sha256": identity(body)}, indent=2))


def relaunch(index: int) -> None:
    """Relaunch one interrupted case with the exact original sealed task."""

    import modal

    launch_receipt = _load_receipt(
        LAUNCH, "pmo_complete_program_decoder_parallel_launch_v1"
    )
    tasks = {int(task["source_case_index"]): task for task in launch_receipt["tasks"]}
    if index not in tasks:
        raise ValueError(f"PMO decoder source case does not exist: {index}")
    path = _relaunch_path(index)
    if path.exists():
        raise ValueError(f"PMO decoder case relaunch already exists: {path}")
    task = tasks[index]
    failure_path = LOCAL_OUTPUT / "shards" / Path(task["output"]).name / "failure.json"
    failure_envelope = json.loads(failure_path.read_text())
    failure = failure_envelope.get("payload")
    if (
        not isinstance(failure, dict)
        or failure_envelope.get("payload_sha256") != identity(failure)
        or failure.get("task_run_id") != task["run_id"]
        or failure.get("source_case_index") != index
        or failure.get("new_oracle_calls") != 0
    ):
        raise ValueError("PMO decoder case relaunch requires its sealed failure")
    call = modal.Function.from_name(APP_NAME, "generate_case").spawn(task)
    body = {
        "schema_version": "pmo_complete_program_decoder_case_relaunch_v1",
        "parent_launch_sha256": launch_receipt["receipt_sha256"],
        "source_case_index": index,
        "source_case_id": task["source_case_id"],
        "task_run_id": task["run_id"],
        "prior_call_id": launch_receipt["call_ids"][str(index)],
        "prior_failure_payload_sha256": failure_envelope["payload_sha256"],
        "call_id": call.object_id,
        "manual_relaunch": 1,
        "automatic_retries": 0,
        "oracle_calls_authorized": 0,
        "docking_calls_authorized": 0,
    }
    _publish_receipt(path, body)
    print(json.dumps({**body, "receipt_sha256": identity(body)}, indent=2))


def _volume_path(path: str | Path) -> str:
    return "/" + Path(path).relative_to(VOLUME_MOUNT).as_posix()


def _read_progress(volume, task: dict) -> dict | None:
    path = _volume_path(Path(task["output"]) / "progress.json")
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None
    except Exception as error:
        if error.__class__.__name__ == "NotFoundError":
            return None
        raise


def status() -> dict:
    import modal

    launch_receipt = _load_receipt(
        LAUNCH, "pmo_complete_program_decoder_parallel_launch_v1"
    )
    volume = modal.Volume.from_name(launch_receipt["modal_volume"])
    rows = []
    tasks = {str(task["source_case_index"]): task for task in launch_receipt["tasks"]}
    for index, call_id in sorted(
        _effective_call_ids(launch_receipt).items(), key=lambda row: int(row[0])
    ):
        task = tasks[index]
        progress = _read_progress(volume, task)
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append(
                {
                    "source_case_index": int(index),
                    "status": "running",
                    "progress": progress,
                }
            )
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            rows.append(
                {
                    "source_case_index": int(index),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "progress": progress,
                }
            )
        else:
            rows.append(
                {
                    "source_case_index": int(index),
                    "status": "complete",
                    "result": result,
                    "progress": progress,
                }
            )
    value = {"source_cases": rows}
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


def _all_generation_complete(receipt: dict) -> None:
    import modal

    failures = []
    for index, call_id in sorted(_effective_call_ids(receipt).items()):
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (
            TimeoutError,
            ValueError,
            RuntimeError,
            ImportError,
            modal.exception.Error,
        ) as error:
            failures.append((index, type(error).__name__, str(error)))
        else:
            if result.get("status") not in ("complete", "complete_reused"):
                failures.append((index, "status", repr(result)))
    if failures:
        raise RuntimeError(f"PMO decoder source shards are incomplete: {failures}")


def collate() -> None:
    import modal

    if COLLATE_LAUNCH.exists():
        raise ValueError("PMO decoder collate launch exists; use status instead")
    receipt = _load_receipt(LAUNCH, "pmo_complete_program_decoder_parallel_launch_v1")
    _all_generation_complete(receipt)
    common = {
        key: receipt["tasks"][0][key]
        for key in (
            "contract_sha256",
            "source_manifest_payload_sha256",
            "new_oracle_calls",
            "automatic_retries",
            "image_revision",
            "files_sha256",
        )
    }
    task = _seal_task(
        {
            "schema_version": "pmo_complete_program_decoder_collate_task_v1",
            **common,
            "output": str(OUTPUT_ROOT),
            "source_case_shards": 9,
            "case_task_run_ids": {
                str(source_task["source_case_index"]): source_task["run_id"]
                for source_task in receipt["tasks"]
            },
        }
    )
    call = modal.Function.from_name(APP_NAME, "collate").spawn(task)
    body = {
        "schema_version": "pmo_complete_program_decoder_collate_launch_v1",
        "parent_launch_sha256": receipt["receipt_sha256"],
        "task": task,
        "call_id": call.object_id,
        "automatic_retries": 0,
        "oracle_calls_authorized": 0,
    }
    _publish_receipt(COLLATE_LAUNCH, body)
    print(json.dumps({**body, "receipt_sha256": identity(body)}, indent=2))


def evaluate() -> None:
    import modal

    if EVALUATE_LAUNCH.exists():
        raise ValueError("PMO decoder evaluate launch exists; use status instead")
    launch_receipt = _load_receipt(
        LAUNCH, "pmo_complete_program_decoder_parallel_launch_v1"
    )
    collate_receipt = _load_receipt(
        COLLATE_LAUNCH, "pmo_complete_program_decoder_collate_launch_v1"
    )
    collated = modal.FunctionCall.from_id(collate_receipt["call_id"]).get(timeout=0)
    if collated.get("status") != "complete":
        raise RuntimeError("PMO decoder collation is not complete")
    common = {
        key: launch_receipt["tasks"][0][key]
        for key in (
            "contract_sha256",
            "source_manifest_payload_sha256",
            "new_oracle_calls",
            "automatic_retries",
            "image_revision",
            "files_sha256",
        )
    }
    task = _seal_task(
        {
            "schema_version": "pmo_complete_program_decoder_evaluate_task_v1",
            **common,
            "candidate_lock": str(OUTPUT_ROOT / "candidate_lock.json"),
            "generation_receipt": str(OUTPUT_ROOT / "generation_receipt.json"),
            "output": str(OUTPUT_ROOT / "result.json"),
        }
    )
    call = modal.Function.from_name(APP_NAME, "evaluate").spawn(task)
    body = {
        "schema_version": "pmo_complete_program_decoder_evaluate_launch_v1",
        "parent_collate_launch_sha256": collate_receipt["receipt_sha256"],
        "task": task,
        "call_id": call.object_id,
        "automatic_retries": 0,
        "oracle_calls_authorized": 0,
    }
    _publish_receipt(EVALUATE_LAUNCH, body)
    print(json.dumps({**body, "receipt_sha256": identity(body)}, indent=2))


def final_status() -> None:
    import modal

    value = {"generation": status()}
    for name, path, schema in (
        ("collate", COLLATE_LAUNCH, "pmo_complete_program_decoder_collate_launch_v1"),
        (
            "evaluate",
            EVALUATE_LAUNCH,
            "pmo_complete_program_decoder_evaluate_launch_v1",
        ),
    ):
        if not path.exists():
            value[name] = {"status": "not_launched"}
            continue
        receipt = _load_receipt(path, schema)
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
        except TimeoutError:
            value[name] = {"status": "running"}
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            value[name] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            value[name] = {"status": "complete", "result": result}
    print(json.dumps(value, indent=2, sort_keys=True))


def _download_file_once(volume, remote: str, destination: Path) -> bool:
    try:
        data = b"".join(volume.read_file(remote))
    except FileNotFoundError:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(
            f"refusing to replace local PMO decoder artifact: {destination}"
        )
    if not destination.exists():
        destination.write_bytes(data)
    return True


def collect() -> None:
    """Download immutable completed shards and preserved failure records."""

    import modal

    volume = modal.Volume.from_name(VOLUME)
    launch_receipt = _load_receipt(
        LAUNCH, "pmo_complete_program_decoder_parallel_launch_v1"
    )
    rows = []
    for task in launch_receipt["tasks"]:
        index = str(task["source_case_index"])
        call_id = _effective_call_ids(launch_receipt)[index]
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"source_case_index": int(index), "status": "running"})
            continue
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            names = ("failure.json", "progress.json")
            status_value = {
                "source_case_index": int(index),
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            if result.get("status") not in ("complete", "complete_reused"):
                raise RuntimeError(f"unexpected PMO decoder case result: {result}")
            names = ("case_shard.json", "generation_receipt.json", "progress.json")
            status_value = {"source_case_index": int(index), "status": "complete"}
        relative = Path("shards") / Path(task["output"]).name
        copied = []
        for name in names:
            remote = _volume_path(Path(task["output"]) / name)
            destination = LOCAL_OUTPUT / relative / name
            if _download_file_once(volume, remote, destination):
                copied.append(name)
        rows.append({**status_value, "copied": copied})
    print(json.dumps({"source_cases": rows}, indent=2, sort_keys=True))


def download() -> None:
    import modal

    collect()
    volume = modal.Volume.from_name(VOLUME)
    files = ["candidate_lock.json", "generation_receipt.json", "result.json"]
    for name in files:
        destination = LOCAL_OUTPUT / name
        if not _download_file_once(
            volume, _volume_path(OUTPUT_ROOT / name), destination
        ):
            raise FileNotFoundError(f"PMO decoder final artifact is absent: {name}")
    print(LOCAL_OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "launch",
            "relaunch",
            "status",
            "collect",
            "collate",
            "evaluate",
            "final-status",
            "download",
        ),
    )
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    actions = {
        "launch": launch,
        "status": status,
        "collect": collect,
        "collate": collate,
        "evaluate": evaluate,
        "final-status": final_status,
        "download": download,
    }
    if args.action == "relaunch":
        if args.case_index is None:
            parser.error("relaunch requires --case-index")
        relaunch(args.case_index)
    else:
        actions[args.action]()


if __name__ == "__main__":
    main()
