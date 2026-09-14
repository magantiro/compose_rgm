"""Durable zero-oracle PMO complete-program decoder shards on Modal."""

from __future__ import annotations

import json
import os
import platform
import resource
import tempfile
import time
from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-pmo-complete-program-decoder"
OUTPUT_ROOT = ARTIFACT_ROOT / "pmo_complete_program_decoder" / "attempt_1"
SOURCE_MANIFEST = Path(
    "diagnostics/pmo_complete_program_decoder/implementation/source_manifest.json"
)
MATERIAL_FILES = (
    "modal_apps/pmo_complete_program_decoder_app.py",
    "tools/pmo_complete_program_decoder_parallel.py",
    "tools/pmo_complete_program_decoder.py",
    "docs/PMO_COMPLETE_PROGRAM_DECODER.md",
    "tests/test_pmo_complete_program_decoder.py",
    str(SOURCE_MANIFEST),
    "diagnostics/pmo_exact_program_segmentation/attempt_2/training_exact_segmented_corpus.json.gz",
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/training_dependency_region_corpus.json.gz",
    "diagnostics/pmo_dependency_region_policy_comparison/attempt_2/runtime_fold_checkpoints.json.gz",
    "diagnostics/pmo_legal_action_policy/attempt_2/result.json",
    "diagnostics/pmo_legal_action_policy/attempt_2/runtime_fold_checkpoints.json.gz",
)

image = base_image.pip_install(
    "numpy==2.5.3",
    "scipy==1.18.1",
    "rdkit==2026.3.6",
).env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    }
)
for relative in MATERIAL_FILES:
    if relative.startswith(("configs/", "src/")):
        continue
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )

app = modal.App(APP_NAME)


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PMO decoder artifact: {path}")
    _replace_json(path, value)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_common(task: dict, *, schema: str) -> dict:
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file

    body = {key: value for key, value in task.items() if key != "run_id"}
    if task.get("schema_version") != schema or identity(body) != task.get("run_id"):
        raise ValueError("PMO decoder task identity changed")
    if task.get("new_oracle_calls") != 0 or task.get("automatic_retries") != 0:
        raise ValueError("PMO decoder tasks must remain zero-oracle without retry")
    if set(task.get("files_sha256", ())) != set(MATERIAL_FILES):
        raise ValueError("PMO decoder material-file inventory changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    _validate_remote_revision(task["image_revision"])
    return body


def _load_inputs(task: dict):
    import numpy
    import rdkit
    import scipy

    from tools.pmo_complete_program_decoder import (
        _decoder_config,
        _load_contract,
        _load_envelope,
        _load_pinned,
    )

    contract, contract_sha256 = _load_contract(REMOTE_ROOT)
    if contract_sha256 != task["contract_sha256"]:
        raise ValueError("PMO decoder contract identity changed")
    execution = contract["execution"]
    environment = {
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "rdkit_version": rdkit.__version__,
    }
    if any(environment[key] != execution[key] for key in environment):
        raise ValueError(
            f"PMO decoder environment changed: {environment} != "
            f"{ {key: execution[key] for key in environment} }"
        )
    manifest, manifest_sha256 = _load_envelope(REMOTE_ROOT / SOURCE_MANIFEST)
    if manifest_sha256 != task["source_manifest_payload_sha256"]:
        raise ValueError("PMO decoder source manifest identity changed")
    dependency = _load_pinned(
        REMOTE_ROOT, contract["inputs"]["dependency_policy_runtime"]
    )
    legal_specification = contract["inputs"]["legal_action_authoritative"]
    legal_result = _load_pinned(
        REMOTE_ROOT,
        {
            "path": legal_specification["result_path"],
            "sha256": legal_specification["result_sha256"],
            "payload_sha256": legal_specification["result_payload_sha256"],
        },
    )
    if (
        legal_result.get("code_revision") != legal_specification["code_revision"]
        or legal_result.get("new_oracle_calls") != 0
        or legal_result.get("runtime_checkpoint", {}).get("sha256")
        != legal_specification["sha256"]
        or legal_result.get("runtime_checkpoint", {}).get("payload_sha256")
        != legal_specification["payload_sha256"]
    ):
        raise ValueError("authoritative PMO legal-action result lineage changed")
    legal = _load_pinned(REMOTE_ROOT, legal_specification)
    return (
        contract,
        contract_sha256,
        manifest,
        dependency,
        legal,
        _decoder_config(contract),
    )


def _validate_existing_shard(directory: Path, task: dict) -> dict | None:
    from compose_v4.control.docking_value import identity
    from tools.pmo_complete_program_decoder import sha256_file

    shard_path = directory / "case_shard.json"
    receipt_path = directory / "generation_receipt.json"
    if not shard_path.exists() or not receipt_path.exists():
        return None
    envelope = _read_json(shard_path)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"existing PMO decoder shard is invalid: {shard_path}")
    if payload.get("task_run_id") != task["run_id"]:
        raise ValueError(f"existing PMO decoder shard task changed: {shard_path}")
    receipt_envelope = _read_json(receipt_path)
    receipt = receipt_envelope.get("payload")
    if (
        not isinstance(receipt, dict)
        or receipt_envelope.get("payload_sha256") != identity(receipt)
        or receipt.get("task_run_id") != task["run_id"]
        or receipt.get("source_case_index") != task["source_case_index"]
        or receipt.get("source_case_id") != task["source_case_id"]
        or receipt.get("case_shard_sha256") != sha256_file(shard_path)
        or receipt.get("case_shard_payload_sha256") != envelope["payload_sha256"]
        or receipt.get("new_oracle_calls") != 0
    ):
        raise ValueError(
            f"existing PMO decoder shard receipt is invalid: {receipt_path}"
        )
    return payload


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=6 * 3600,
    retries=0,
    max_containers=9,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def generate_case(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.pmo_complete_program_decoder import (
        build_candidate_case,
    )
    from tools.pmo_complete_program_decoder import sha256_file

    _validate_common(task, schema="pmo_complete_program_decoder_case_task_v1")
    index = int(task["source_case_index"])
    output = OUTPUT_ROOT / "shards" / f"case_{index:02d}_{task['source_case_id'][:12]}"
    expected = str(output)
    if task.get("output") != expected:
        raise ValueError("PMO decoder shard output path changed")
    shard_path = output / "case_shard.json"
    existing = _validate_existing_shard(output, task)
    if existing is not None:
        return {
            "status": "complete_reused",
            "source_case_index": index,
            "source_case_id": task["source_case_id"],
            "output": expected,
            "payload_sha256": identity(existing),
            "new_oracle_calls": 0,
        }
    contract, contract_sha256, manifest, dependency, legal, config = _load_inputs(task)
    if index < 0 or index >= len(manifest["source_cases"]):
        raise ValueError("PMO decoder source case index is out of range")
    source_case = manifest["source_cases"][index]
    if source_case["source_case_id"] != task["source_case_id"] or int(
        source_case["fold"]
    ) != int(task["fold"]):
        raise ValueError("PMO decoder source assignment changed")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_commit = [0.0]

    def progress(event: dict) -> None:
        row = {
            "schema_version": "pmo_complete_program_decoder_case_progress_v1",
            "status": "complete" if event.get("phase") == "complete" else "running",
            "task_run_id": task["run_id"],
            "source_case_index": index,
            "source_case_id": task["source_case_id"],
            "fold": int(task["fold"]),
            "elapsed_seconds": time.monotonic() - started,
            "event": event,
            "new_oracle_calls": 0,
        }
        _replace_json(output / "progress.json", row)
        now = time.monotonic()
        if now - last_commit[0] >= 30.0:
            artifact_volume.commit()
            last_commit[0] = now
        print(json.dumps(row, sort_keys=True), flush=True)

    progress(
        {"phase": "started", "depth": 0, "maximum_depth": config.maximum_primitives}
    )
    try:
        case = build_candidate_case(
            source_case,
            dependency,
            legal,
            config,
            progress_callback=progress,
        )
        payload = {
            "schema_version": "pmo_complete_program_decoder_case_shard_v1",
            "contract_sha256": contract_sha256,
            "source_manifest_payload_sha256": task["source_manifest_payload_sha256"],
            "task_run_id": task["run_id"],
            "source_case_index": index,
            "configuration": contract["decoder"],
            "source_case": case,
            "teacher_fields_present": False,
            "task_identity_present": False,
            "new_oracle_calls": 0,
        }
        sealed = {"payload": payload, "payload_sha256": identity(payload)}
        existing_shard = _read_json(shard_path) if shard_path.exists() else None
        if existing_shard is not None and existing_shard != sealed:
            raise ValueError("existing PMO decoder case shard differs")
        if existing_shard is None:
            _publish_once(shard_path, sealed)
        receipt_payload = {
            "schema_version": "pmo_complete_program_decoder_case_receipt_v1",
            "task_run_id": task["run_id"],
            "source_case_index": index,
            "source_case_id": task["source_case_id"],
            "case_shard_sha256": sha256_file(shard_path),
            "case_shard_payload_sha256": sealed["payload_sha256"],
            "wall_seconds": time.monotonic() - started,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "rdkit": __import__("rdkit").__version__,
            "new_oracle_calls": 0,
        }
        receipt = {
            "payload": receipt_payload,
            "payload_sha256": identity(receipt_payload),
        }
        receipt_path = output / "generation_receipt.json"
        existing_receipt = _read_json(receipt_path) if receipt_path.exists() else None
        if existing_receipt is not None and existing_receipt != receipt:
            raise ValueError("existing PMO decoder case receipt differs")
        if existing_receipt is None:
            _publish_once(receipt_path, receipt)
        progress(
            {
                "phase": "complete",
                "depth": config.maximum_primitives,
                "maximum_depth": config.maximum_primitives,
            }
        )
        artifact_volume.commit()
        return {
            "status": "complete",
            "source_case_index": index,
            "source_case_id": task["source_case_id"],
            "output": expected,
            "payload_sha256": identity(payload),
            "new_oracle_calls": 0,
        }
    except BaseException as error:
        failure_payload = {
            "schema_version": "pmo_complete_program_decoder_case_failure_v1",
            "task_run_id": task["run_id"],
            "source_case_index": index,
            "source_case_id": task["source_case_id"],
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.monotonic() - started,
            "new_oracle_calls": 0,
        }
        failure_path = output / "failure.json"
        if not failure_path.exists():
            _publish_once(
                failure_path,
                {
                    "payload": failure_payload,
                    "payload_sha256": identity(failure_payload),
                },
            )
        artifact_volume.commit()
        raise


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=30 * 60,
    retries=0,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def collate(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.pmo_complete_program_decoder import (
        candidate_lock_from_cases,
        seal_candidate_lock,
    )
    from tools.pmo_complete_program_decoder import sha256_file

    _validate_common(task, schema="pmo_complete_program_decoder_collate_task_v1")
    if task.get("output") != str(OUTPUT_ROOT):
        raise ValueError("PMO decoder collate output path changed")
    contract, contract_sha256, manifest, _, _, config = _load_inputs(task)
    cases = []
    shard_receipts = []
    for index, source in enumerate(manifest["source_cases"]):
        directory = (
            OUTPUT_ROOT / "shards" / f"case_{index:02d}_{source['source_case_id'][:12]}"
        )
        envelope = _read_json(directory / "case_shard.json")
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
            payload
        ):
            raise ValueError(f"PMO decoder shard seal failed: {directory}")
        if (
            payload.get("schema_version")
            != "pmo_complete_program_decoder_case_shard_v1"
            or payload.get("contract_sha256") != contract_sha256
            or payload.get("source_manifest_payload_sha256")
            != task["source_manifest_payload_sha256"]
            or payload.get("task_run_id") != task["case_task_run_ids"].get(str(index))
            or payload.get("source_case_index") != index
            or payload.get("configuration") != contract["decoder"]
            or payload.get("teacher_fields_present") is not False
            or payload.get("task_identity_present") is not False
            or payload.get("new_oracle_calls") != 0
        ):
            raise ValueError(f"PMO decoder shard provenance changed: {directory}")
        cases.append(payload["source_case"])
        receipt = _read_json(directory / "generation_receipt.json")
        receipt_payload = receipt.get("payload")
        if (
            not isinstance(receipt_payload, dict)
            or receipt.get("payload_sha256") != identity(receipt_payload)
            or receipt_payload.get("task_run_id")
            != task["case_task_run_ids"].get(str(index))
            or receipt_payload.get("source_case_index") != index
            or receipt_payload.get("source_case_id") != source["source_case_id"]
            or receipt_payload.get("case_shard_sha256")
            != sha256_file(directory / "case_shard.json")
            or receipt_payload.get("case_shard_payload_sha256")
            != envelope["payload_sha256"]
            or receipt_payload.get("new_oracle_calls") != 0
        ):
            raise ValueError(f"PMO decoder shard receipt seal failed: {directory}")
        shard_receipts.append(receipt_payload)
    payload = candidate_lock_from_cases(manifest, cases, config)
    payload["provenance"] = {
        "contract_sha256": contract_sha256,
        "code_revision": task["image_revision"]["commit"],
        "image_revision_sha256": task["image_revision"]["image_revision_sha256"],
        "source_manifest_payload_sha256": task["source_manifest_payload_sha256"],
        "dependency_policy_runtime_sha256": contract["inputs"][
            "dependency_policy_runtime"
        ]["sha256"],
        "dependency_policy_runtime_payload_sha256": contract["inputs"][
            "dependency_policy_runtime"
        ]["payload_sha256"],
        "legal_action_runtime_sha256": contract["inputs"]["legal_action_authoritative"][
            "sha256"
        ],
        "legal_action_runtime_payload_sha256": contract["inputs"][
            "legal_action_authoritative"
        ]["payload_sha256"],
        "implementation_file_sha256": task["files_sha256"],
        "execution": "nine_durable_source_case_shards",
    }
    sealed = seal_candidate_lock(payload)
    lock_path = OUTPUT_ROOT / "candidate_lock.json"
    existing = _read_json(lock_path) if lock_path.exists() else None
    if existing is not None and existing != sealed:
        raise ValueError("existing PMO decoder candidate lock differs")
    if existing is None:
        _publish_once(lock_path, sealed)
    receipt_payload = {
        "schema_version": "pmo_complete_program_decoder_generation_receipt_v1",
        "candidate_lock_path": lock_path.name,
        "candidate_lock_sha256": sha256_file(lock_path),
        "candidate_lock_payload_sha256": sealed["payload_sha256"],
        "wall_seconds": max(row["wall_seconds"] for row in shard_receipts),
        "aggregate_cpu_case_seconds": sum(
            row["wall_seconds"] for row in shard_receipts
        ),
        "peak_rss_raw": max(row["peak_rss_raw"] for row in shard_receipts),
        "platform": sorted({row["platform"] for row in shard_receipts}),
        "python": sorted({row["python"] for row in shard_receipts}),
        "numpy": sorted({row["numpy"] for row in shard_receipts}),
        "rdkit": sorted({row["rdkit"] for row in shard_receipts}),
        "source_case_shards": len(shard_receipts),
        "concurrent_single_cpu_workers": 9,
        "new_oracle_calls": 0,
    }
    receipt = {
        "payload": receipt_payload,
        "payload_sha256": identity(receipt_payload),
    }
    receipt_path = OUTPUT_ROOT / "generation_receipt.json"
    existing_receipt = _read_json(receipt_path) if receipt_path.exists() else None
    if existing_receipt is not None and existing_receipt != receipt:
        raise ValueError("existing PMO decoder generation receipt differs")
    if existing_receipt is None:
        _publish_once(receipt_path, receipt)
    artifact_volume.commit()
    return {
        "status": "complete",
        "candidate_lock_payload_sha256": sealed["payload_sha256"],
        "source_case_shards": len(shard_receipts),
        "new_oracle_calls": 0,
    }


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(6144, 6144),
    timeout=30 * 60,
    retries=0,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def evaluate(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.pmo_complete_program_decoder import (
        evaluate_candidate_lock,
        validate_candidate_lock,
    )
    from tools.pmo_complete_program_decoder import _load_pinned, sha256_file

    _validate_common(task, schema="pmo_complete_program_decoder_evaluate_task_v1")
    expected_paths = {
        "candidate_lock": str(OUTPUT_ROOT / "candidate_lock.json"),
        "generation_receipt": str(OUTPUT_ROOT / "generation_receipt.json"),
        "output": str(OUTPUT_ROOT / "result.json"),
    }
    if any(task.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("PMO decoder evaluation path changed")
    contract, contract_sha256, _, _, _, config = _load_inputs(task)
    lock_path = OUTPUT_ROOT / "candidate_lock.json"
    receipt_path = OUTPUT_ROOT / "generation_receipt.json"
    sealed = _read_json(lock_path)
    payload = validate_candidate_lock(sealed)
    receipt_envelope = _read_json(receipt_path)
    receipt = receipt_envelope.get("payload")
    if (
        not isinstance(receipt, dict)
        or receipt_envelope.get("payload_sha256") != identity(receipt)
        or payload.get("provenance", {}).get("contract_sha256") != contract_sha256
        or payload.get("provenance", {}).get("image_revision_sha256")
        != task["image_revision"]["image_revision_sha256"]
        or receipt.get("candidate_lock_sha256") != sha256_file(lock_path)
        or receipt.get("candidate_lock_payload_sha256") != sealed["payload_sha256"]
        or receipt.get("source_case_shards") != 9
        or receipt.get("new_oracle_calls") != 0
    ):
        raise ValueError("PMO decoder candidate lock or generation receipt changed")
    # Teacher-bearing artifacts are loaded only after the candidate lock is validated.
    panels = _load_pinned(REMOTE_ROOT, contract["inputs"]["panel_corpus"])
    dependency = _load_pinned(
        REMOTE_ROOT, contract["inputs"]["dependency_region_corpus"]
    )
    report = evaluate_candidate_lock(sealed, panels, dependency, config)
    report["generation_compute"] = {
        key: receipt[key]
        for key in (
            "wall_seconds",
            "aggregate_cpu_case_seconds",
            "peak_rss_raw",
            "platform",
            "python",
            "numpy",
            "rdkit",
            "source_case_shards",
            "concurrent_single_cpu_workers",
        )
    }
    report["provenance"] = payload.get("provenance", {})
    report["evaluation_inputs"] = {
        name: {
            "path": specification["path"],
            "sha256": specification["sha256"],
            "payload_sha256": specification["payload_sha256"],
        }
        for name, specification in (
            ("panel_corpus", contract["inputs"]["panel_corpus"]),
            (
                "dependency_region_corpus",
                contract["inputs"]["dependency_region_corpus"],
            ),
        )
    }
    envelope = {"payload": report, "payload_sha256": identity(report)}
    result_path = OUTPUT_ROOT / "result.json"
    existing = _read_json(result_path) if result_path.exists() else None
    if existing is not None and existing != envelope:
        raise ValueError("existing PMO decoder evaluation differs")
    if existing is None:
        _publish_once(result_path, envelope)
    artifact_volume.commit()
    return {
        "status": "complete",
        "engineering_gate": report["engineering_gate"],
        "scientific_where_how_gate": report["scientific_where_how_gate"],
        "scored_pmo_pilot_authorized": False,
        "new_oracle_calls": 0,
    }
