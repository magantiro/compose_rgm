"""Modal runner for the non-authorizing atom-insert V3 schedule diagnostic."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from collections.abc import Mapping
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REMOTE_PROJECT_ROOT = Path("/root/compose")
for import_root in (ROOT, SRC, REMOTE_PROJECT_ROOT, REMOTE_PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from compose_v4.data.immutable_artifact import ImmutableArtifactError, write_bytes_if_absent
from compose_v4.experiments.editing_t1_atom_insert_schedule_v3 import (
    load_atom_insert_schedule_v3_contract,
)
from compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 import (
    serialized_source_tree_sha256,
    stable_sha256,
)
from modal_apps.run_editing_t1_successor_gate import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    T1_SUCCESSOR_CACHE_ROOT,
    _editing_t1_cache_validation_command,
    _require_clean_serialized_tree,
    _require_worker_command_success,
    _resolve_active8_launch_binding,
    _validate_run_label,
    _validated_cache_receipt_payload,
    artifact_volume,
    image,
)
import scripts.run_editing_t1_atom_insert_schedule_v3 as v3_runner
from scripts.run_editing_t1_atom_insert_schedule_v3 import validate_atom_insert_v3_result

V3_CONTRACT_RELATIVE_PATH = "configs/editing_t1_atom_insert_schedule_v3.json"
V3_RESULT_ROOT = ARTIFACT_ROOT / "_editing_t1_atom_insert_schedule_v3"
LOCAL_RECEIPT_ROOT = ROOT / "results" / "_editing_t1_atom_insert_v3_launch_receipts"
RECEIPT_SCHEMA = "compose.editing.t1_atom_insert_schedule_v3_modal_receipt"
RECEIPT_VERSION = 1

app = modal.App("compose-v4-editing-t1-atom-insert-schedule-v3")


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        write_bytes_if_absent(path, encoded)
    except ImmutableArtifactError as error:
        raise FileExistsError(f"immutable atom-insert V3 receipt differs: {path}") from error


def _cache_paths(contract: Mapping[str, object]) -> tuple[Path, Path]:
    parents = contract["parents"]
    assert isinstance(parents, Mapping)
    cache_root = (
        T1_SUCCESSOR_CACHE_ROOT
        / str(parents["runtime_contract_sha256"])
        / "unique_state"
        / "atom_insert"
    )
    return cache_root, cache_root / "receipt.json"


@app.function(
    image=image,
    cpu=2,
    memory=32768,
    timeout=3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def validate_atom_insert_v3_cache_cpu(
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    artifact_volume.reload()
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory, active8_inventory_file_sha256
    )
    contract = load_atom_insert_schedule_v3_contract(REMOTE_ROOT / V3_CONTRACT_RELATIVE_PATH)
    cache_root, cache_receipt = _cache_paths(contract)
    if not cache_receipt.is_file():
        raise FileNotFoundError(f"atom-insert V3 requires frozen cache: {cache_receipt}")
    completed = _require_worker_command_success(
        subprocess.run(
            _editing_t1_cache_validation_command(
                active8_inventory=active8_inventory,
                active8_inventory_file_sha256=active8_inventory_file_sha256,
                family="atom_insert",
                panel_kind="unique_state",
                successor_cache_root=cache_root,
                receipt_path=cache_receipt,
            ),
            cwd=REMOTE_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    )
    parents = contract["parents"]
    assert isinstance(parents, Mapping)
    v2_result = ARTIFACT_ROOT / str(parents["v2_result_relative_path"])
    if (
        not v2_result.is_file()
        or hashlib.sha256(v2_result.read_bytes()).hexdigest() != parents["v2_result_file_sha256"]
    ):
        raise FileNotFoundError("atom-insert V3 exact failed V2 parent is absent or changed")
    return {
        "family": "atom_insert",
        "successor_cache_root": str(cache_root),
        "successor_cache_receipt": str(cache_receipt),
        "successor_cache_manifest_receipt": dict(_validated_cache_receipt_payload(cache_receipt)),
        "v2_result": str(v2_result),
        "worker_stdout_tail": completed.stdout[-1000:],
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_atom_insert_v3_l4(
    run_label: str,
    source_commit: str,
    source_tree_sha256: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    resume_checkpoint: str,
) -> dict[str, object]:
    artifact_volume.reload()
    _validate_run_label(run_label)
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory, active8_inventory_file_sha256
    )
    contract_path = REMOTE_ROOT / V3_CONTRACT_RELATIVE_PATH
    contract = load_atom_insert_schedule_v3_contract(contract_path)
    cache_root, cache_receipt = _cache_paths(contract)
    parents = contract["parents"]
    assert isinstance(parents, Mapping)
    v2_result = ARTIFACT_ROOT / str(parents["v2_result_relative_path"])
    run_root = V3_RESULT_ROOT / run_label
    output = run_root / "unique_state.atom_insert.all.json"
    checkpoint_root = run_root / "checkpoints"
    if output.exists():
        raise FileExistsError(f"atom-insert V3 output already exists: {output}")
    command = [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "run_editing_t1_atom_insert_schedule_v3.py"),
        "--source-commit",
        source_commit,
        "--source-tree-sha256",
        source_tree_sha256,
        "--transfer-root",
        str(ARTIFACT_ROOT),
        "--uniform-v2-contract",
        str(REMOTE_ROOT / "configs" / "editing_t1_uniform_successor_capacity_v2.json"),
        "--v3-contract",
        str(contract_path),
        "--v2-result",
        str(v2_result),
        "--active8-inventory",
        active8_inventory,
        "--active8-inventory-file-sha256",
        active8_inventory_file_sha256,
        "--family",
        "atom_insert",
        "--successor-cache-root",
        str(cache_root),
        "--successor-cache-receipt",
        str(cache_receipt),
        "--checkpoint-root",
        str(checkpoint_root),
        "--device",
        "cuda",
        "--output",
        str(output),
    ]
    if resume_checkpoint:
        resume_relative = Path(resume_checkpoint)
        if resume_relative.is_absolute() or ".." in resume_relative.parts:
            raise ValueError("atom-insert V3 resume checkpoint must be artifact-relative")
        resume_path = ARTIFACT_ROOT / resume_relative
        if not resume_path.is_file():
            raise FileNotFoundError(f"atom-insert V3 resume checkpoint is absent: {resume_path}")
        command.extend(("--resume-checkpoint", str(resume_path)))
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        returncode = v3_runner.main(
            command[2:],
            checkpoint_publish_callback=artifact_volume.commit,
        )
    if returncode != 0:
        raise RuntimeError(f"atom-insert V3 worker returned nonzero status: {returncode}")
    encoded = output.read_bytes()
    payload = json.loads(encoded)
    validate_atom_insert_v3_result(payload, contract=contract)
    if (
        payload.get("source_commit") != source_commit
        or payload.get("source_tree_sha256") != source_tree_sha256
    ):
        raise RuntimeError("atom-insert V3 result returned another source identity")
    artifact_volume.commit()
    return {
        "run_label": run_label,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "result_relative_path": output.relative_to(ARTIFACT_ROOT).as_posix(),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "result_sha256": payload["result_sha256"],
        "selected_update": payload["selected_update"],
        "terminal_update": payload["terminal_update"],
        "all_threshold_checks_pass": payload["all_threshold_checks_pass"],
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.local_entrypoint()
def main(
    run_label: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    resume_checkpoint: str = "",
) -> None:
    source_commit = _require_clean_serialized_tree()
    source_tree_sha256 = serialized_source_tree_sha256(ROOT)
    _validate_run_label(run_label)
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory, active8_inventory_file_sha256
    )
    preflight = validate_atom_insert_v3_cache_cpu.remote(
        active8_inventory, active8_inventory_file_sha256
    )
    result = run_atom_insert_v3_l4.remote(
        run_label,
        source_commit,
        source_tree_sha256,
        active8_inventory,
        active8_inventory_file_sha256,
        resume_checkpoint,
    )
    body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "status": "COMPLETE_ATOM_INSERT_V3_MODAL_RESULT",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "run_label": run_label,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "requested_gpu_class": "L4",
        "active8_inventory": active8_inventory,
        "active8_inventory_file_sha256": active8_inventory_file_sha256,
        "resume_checkpoint": resume_checkpoint or None,
        "cpu_preflight": preflight,
        "result": result,
    }
    receipt = {**body, "receipt_sha256": stable_sha256(body)}
    _write_receipt(LOCAL_RECEIPT_ROOT / f"{run_label}.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
