"""Parallel Modal launcher for the equal-coefficient T1 capacity diagnostic.

The launcher consumes the already-built immutable V8 successor caches.  It
does no CPU support compilation and emits no P50 or full-training authority.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REMOTE_PROJECT_ROOT = Path("/root/compose")
for import_root in (ROOT, SRC, REMOTE_PROJECT_ROOT, REMOTE_PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from modal_apps.run_editing_t1_successor_gate import (
    ARTIFACT_ROOT,
    FAMILIES,
    REMOTE_ROOT,
    T1_SUCCESSOR_CACHE_ROOT,
    _editing_t1_cache_validation_command,
    _failure_diagnostic,
    _require_clean_serialized_tree,
    _require_worker_command_success,
    _resolve_active8_launch_binding,
    _stable_sha256,
    _validate_run_label,
    _validated_cache_receipt_payload,
    artifact_volume,
    image,
)
from scripts.run_editing_t1_uniform_successor_capacity import (
    load_uniform_capacity_contract,
    validate_uniform_capacity_result,
)

UNIFORM_CONTRACT_RELATIVE_PATH = "configs/editing_t1_uniform_successor_capacity_v1.json"
UNIFORM_RESULT_ROOT = ARTIFACT_ROOT / "_editing_t1_uniform_successor_capacity"
LOCAL_RECEIPT_ROOT = ROOT / "results" / "_editing_t1_uniform_launch_receipts"
RECEIPT_SCHEMA = "compose.editing.t1_uniform_successor_capacity_modal_receipt"
RECEIPT_VERSION = 1
COMPLETE_RECEIPT_STATUS = "COMPLETE_UNIFORM_SUCCESSOR_CAPACITY_MODAL_RESULTS"
INCOMPLETE_RECEIPT_STATUS = "INCOMPLETE_UNIFORM_SUCCESSOR_CAPACITY_MODAL_RESULTS"

app = modal.App("compose-v4-editing-t1-uniform-successor-capacity")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        write_bytes_if_absent(path, content)
    except ImmutableArtifactError as error:
        raise FileExistsError(f"immutable uniform-capacity receipt differs: {path}") from error


def _worker_command(
    *,
    family: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    successor_cache_root: Path,
    successor_cache_receipt: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "run_editing_t1_uniform_successor_capacity.py"),
        "--transfer-root",
        str(ARTIFACT_ROOT),
        "--uniform-contract",
        str(REMOTE_ROOT / UNIFORM_CONTRACT_RELATIVE_PATH),
        "--active8-inventory",
        active8_inventory,
        "--active8-inventory-file-sha256",
        active8_inventory_file_sha256,
        "--family",
        family,
        "--successor-cache-root",
        str(successor_cache_root),
        "--successor-cache-receipt",
        str(successor_cache_receipt),
        "--device",
        "cuda",
        "--output",
        str(output),
    ]


def _cache_paths(
    contract: Mapping[str, object],
    family: str,
) -> tuple[Path, Path]:
    parents = contract["parents"]
    if not isinstance(parents, Mapping):
        raise TypeError("uniform-capacity contract parents are invalid")
    parent_runtime_sha256 = str(parents["runtime_contract_sha256"])
    cache_root = T1_SUCCESSOR_CACHE_ROOT / parent_runtime_sha256 / "unique_state" / family
    return cache_root, cache_root / "receipt.json"


@app.function(
    image=image,
    cpu=2,
    memory=32768,
    timeout=3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def validate_uniform_cache_cpu(
    family: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    """Validate every immutable V8 cache leaf before provisioning any GPU."""

    artifact_volume.reload()
    if family not in FAMILIES:
        raise ValueError(f"unknown Active8 family: {family}")
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory,
        active8_inventory_file_sha256,
    )
    contract = load_uniform_capacity_contract(REMOTE_ROOT / UNIFORM_CONTRACT_RELATIVE_PATH)
    cache_root, cache_receipt = _cache_paths(contract, family)
    if not cache_receipt.is_file():
        raise FileNotFoundError(
            f"uniform-capacity diagnostic requires the frozen V8 cache receipt: {cache_receipt}"
        )
    completed = _require_worker_command_success(
        subprocess.run(
            _editing_t1_cache_validation_command(
                active8_inventory=active8_inventory,
                active8_inventory_file_sha256=active8_inventory_file_sha256,
                family=family,
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
    return {
        "family": family,
        "successor_cache_root": str(cache_root),
        "successor_cache_receipt": str(cache_receipt),
        "successor_cache_manifest_receipt": dict(_validated_cache_receipt_payload(cache_receipt)),
        "worker_stdout_tail": completed.stdout[-1000:],
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


def _run_arm_impl(
    run_label: str,
    family: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    artifact_volume.reload()
    _validate_run_label(run_label)
    if family not in FAMILIES:
        raise ValueError(f"unknown Active8 family: {family}")
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory,
        active8_inventory_file_sha256,
    )
    contract = load_uniform_capacity_contract(REMOTE_ROOT / UNIFORM_CONTRACT_RELATIVE_PATH)
    cache_root, cache_receipt = _cache_paths(contract, family)
    if not cache_receipt.is_file():
        raise FileNotFoundError(
            f"uniform-capacity diagnostic requires the frozen V8 cache receipt: {cache_receipt}"
        )
    output = UNIFORM_RESULT_ROOT / run_label / f"unique_state.{family}.all.json"
    if output.exists():
        raise FileExistsError(f"uniform-capacity output already exists: {output}")
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    completed = subprocess.run(
        _worker_command(
            family=family,
            active8_inventory=active8_inventory,
            active8_inventory_file_sha256=active8_inventory_file_sha256,
            successor_cache_root=cache_root,
            successor_cache_receipt=cache_receipt,
            output=output,
        ),
        cwd=REMOTE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-8000:],
        }
        raise RuntimeError(
            f"uniform-capacity worker failed: {json.dumps(diagnostic, sort_keys=True)}"
        )
    encoded = output.read_bytes()
    payload = json.loads(encoded)
    cache_manifest_receipt = _validated_cache_receipt_payload(cache_receipt)
    validate_uniform_capacity_result(
        payload,
        contract=contract,
        expected_cache_manifest_receipt=cache_manifest_receipt,
    )
    if payload.get("family") != family:
        raise RuntimeError("uniform-capacity worker returned another family")
    artifact_volume.commit()
    return {
        "run_label": run_label,
        "family": family,
        "result_relative_path": output.relative_to(ARTIFACT_ROOT).as_posix(),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "result_sha256": payload["result_sha256"],
        "uniform_contract_sha256": payload["uniform_contract_sha256"],
        "parent_t1_runtime_contract_sha256": payload["parent_t1_runtime_contract_sha256"],
        "all_threshold_checks_pass": payload["all_threshold_checks_pass"],
        "observed_gpu_name": gpu_name,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_uniform_arm_l4(
    run_label: str,
    family: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        active8_inventory,
        active8_inventory_file_sha256,
    )


def _selected_families(raw: str, *, all_families: bool) -> tuple[str, ...]:
    if all_families:
        if raw.strip():
            raise ValueError("--all-families cannot be combined with --families")
        return FAMILIES
    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not selected:
        raise ValueError("select --families explicitly or use --all-families")
    if len(selected) != len(set(selected)):
        raise ValueError("uniform-capacity family selection contains duplicates")
    unknown = sorted(set(selected) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown Active8 families: {unknown}")
    return selected


def _build_receipt(
    *,
    run_label: str,
    source_commit: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    requested_families: Sequence[str],
    stage: str,
    cpu_preflight_bindings: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested = tuple(requested_families)
    preflight_families = tuple(str(binding["family"]) for binding in cpu_preflight_bindings)
    result_families = tuple(str(result["family"]) for result in results)
    failure_families = tuple(str(failure["family"]) for failure in failures)
    if (
        stage not in {"CPU_PREFLIGHT", "GPU_RESULTS"}
        or not requested
        or len(requested) != len(set(requested))
        or len(preflight_families) != len(set(preflight_families))
        or len(result_families) != len(set(result_families))
        or len(failure_families) != len(set(failure_families))
        or set(result_families).intersection(failure_families)
        or (
            stage == "CPU_PREFLIGHT"
            and (
                results
                or set(preflight_families).intersection(failure_families)
                or set(preflight_families).union(failure_families) != set(requested)
            )
        )
        or (
            stage == "GPU_RESULTS"
            and (
                set(preflight_families) != set(requested)
                or set(result_families).union(failure_families) != set(requested)
            )
        )
    ):
        raise ValueError("uniform-capacity receipt does not cover its requested family set")
    complete = stage == "GPU_RESULTS" and not failures and set(result_families) == set(requested)
    body: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "status": COMPLETE_RECEIPT_STATUS if complete else INCOMPLETE_RECEIPT_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "run_label": run_label,
        "source_commit": source_commit,
        "requested_gpu_class": "L4",
        "active8_inventory": active8_inventory,
        "active8_inventory_file_sha256": active8_inventory_file_sha256,
        "stage": stage,
        "requested_families": list(requested),
        "cpu_preflight_bindings": list(cpu_preflight_bindings),
        "results": list(results),
        "failures": list(failures),
    }
    return {**body, "receipt_sha256": _stable_sha256(body)}


@app.local_entrypoint()
def main(
    run_label: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    families: str = "",
    all_families: bool = False,
) -> None:
    source_commit = _require_clean_serialized_tree()
    _validate_run_label(run_label)
    selected = _selected_families(families, all_families=all_families)
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory,
        active8_inventory_file_sha256,
    )
    cache_tasks = tuple(
        (family, active8_inventory, active8_inventory_file_sha256) for family in selected
    )
    cache_outcomes = list(
        validate_uniform_cache_cpu.starmap(
            cache_tasks,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    cache_bindings: list[Mapping[str, object]] = []
    cache_failures: list[dict[str, str]] = []
    for family, outcome in zip(selected, cache_outcomes, strict=True):
        if isinstance(outcome, BaseException):
            cache_failures.append({"family": family, **_failure_diagnostic(outcome)})
        elif isinstance(outcome, Mapping) and outcome.get("family") == family:
            cache_bindings.append(outcome)
        else:
            raise RuntimeError("uniform-capacity CPU preflight returned an invalid outcome")
    if cache_failures:
        preflight_receipt = _build_receipt(
            run_label=run_label,
            source_commit=source_commit,
            active8_inventory=active8_inventory,
            active8_inventory_file_sha256=active8_inventory_file_sha256,
            requested_families=selected,
            stage="CPU_PREFLIGHT",
            cpu_preflight_bindings=cache_bindings,
            results=(),
            failures=cache_failures,
        )
        preflight_receipt_path = LOCAL_RECEIPT_ROOT / f"{run_label}.json"
        _write_receipt(preflight_receipt_path, preflight_receipt)
        print(
            json.dumps(
                {
                    "run_label": run_label,
                    "status": "UNIFORM_CAPACITY_CPU_PREFLIGHT_FAILED_NO_GPU_DISPATCH",
                    "cache_failures": cache_failures,
                    "launch_receipt": str(preflight_receipt_path),
                    "launch_receipt_sha256": preflight_receipt["receipt_sha256"],
                    "training_authorized": False,
                    "bounded_p50_authorized": False,
                    "gate_decision": None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise RuntimeError(
            f"{len(cache_failures)} of {len(selected)} cache preflights failed; no GPU dispatched"
        )
    tasks = tuple(
        (run_label, family, active8_inventory, active8_inventory_file_sha256) for family in selected
    )
    outcomes = list(
        run_uniform_arm_l4.starmap(
            tasks,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    results: list[Mapping[str, object]] = []
    failures: list[dict[str, str]] = []
    for family, outcome in zip(selected, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append({"family": family, **_failure_diagnostic(outcome)})
        elif isinstance(outcome, Mapping) and outcome.get("family") == family:
            results.append(outcome)
        else:
            raise RuntimeError("uniform-capacity worker returned an invalid outcome")
    receipt = _build_receipt(
        run_label=run_label,
        source_commit=source_commit,
        active8_inventory=active8_inventory,
        active8_inventory_file_sha256=active8_inventory_file_sha256,
        requested_families=selected,
        stage="GPU_RESULTS",
        cpu_preflight_bindings=cache_bindings,
        results=results,
        failures=failures,
    )
    receipt_path = LOCAL_RECEIPT_ROOT / f"{run_label}.json"
    _write_receipt(receipt_path, receipt)
    print(
        json.dumps(
            {
                "run_label": run_label,
                "source_commit": source_commit,
                "requested_family_count": len(selected),
                "successful_family_count": len(results),
                "failed_family_count": len(failures),
                "cpu_preflight_bindings": cache_bindings,
                "results": results,
                "failures": failures,
                "launch_receipt": str(receipt_path),
                "launch_receipt_status": receipt["status"],
                "launch_receipt_sha256": receipt["receipt_sha256"],
                "training_authorized": False,
                "bounded_p50_authorized": False,
                "gate_decision": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(selected)} uniform-capacity arms failed")
