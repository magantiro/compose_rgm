"""Restart-safe 62-shard/20-cell Editing-V2 role/lane continuation on Modal.

The driver publishes the same refined continuation contract and final
role/lane derivative as the reference materializer.  CPU work is split into
content-addressed source-shard map tasks and independently reusable lane/role
cell reductions.  Pre-plan progress and the completed plan pointer are durable,
and every worker commits the shared volume in ``finally`` so a failure preserves
its latest operational progress record.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LOCAL_SOURCE_ROOT = str(ROOT / "src")
if LOCAL_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_SOURCE_ROOT)

from compose_v4.data import (  # noqa: E402
    editing_v2_refined_role_lane_continuation as continuation,
)
from compose_v4.data import editing_v2_role_lane_mapreduce as mapreduce  # noqa: E402

EXPECTED_SOURCE_SHARDS = 62
# ``compose-v4-artifacts`` predates an explicit Volume-v2 declaration.  Keep
# concurrent writers/commits within Modal's legacy Volume-v1 guidance.
MAX_MAP_CONTAINERS = 5
MAX_CELL_CONTAINERS = 5
UPSTREAM_VALIDATION_RECEIPT_FILENAME = "ROLE_LANE_UPSTREAM_VALIDATION_RECEIPT.json"
PREPLAN_PROGRESS_FILENAME = "ROLE_LANE_PREPLAN_PROGRESS.json"
PLAN_POINTER_FILENAME = "ROLE_LANE_MAPREDUCE_PLAN_POINTER.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(
        ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_mapreduce_app.py",
        str(
            REMOTE_ROOT
            / "modal_apps/materialize_editing_v2_refined_role_lane_mapreduce_app.py"
        ),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_app.py",
        str(REMOTE_ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-refined-role-lane-mapreduce")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts", create_if_missing=False
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: _canonical_sha256(body)}


def _write_operational_json(path: Path, value: object) -> None:
    """Atomically replace one nonauthorizing operational status record."""

    encoded = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".staging",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_immutable_json(path: Path, value: object) -> bool:
    encoded = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable refined-role/lane collision at {path}")
        return False
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".staging",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError(f"immutable refined-role/lane collision at {path}")
            return False
        os.replace(temporary_name, path)
        temporary_name = None
        return True
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _artifact_path(value: str, *, field: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != value
    ):
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    mounted = ARTIFACT_ROOT / Path(*pure.parts[2:])
    resolved = mounted.resolve()
    if not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise RuntimeError(f"{field} resolves outside artifact root")
    # Keep the stable mounted coordinate for signed artifact identity.  Modal
    # resolves the volume mount to an internal /__modal/volumes/... path, which
    # is safe for the containment check above but is not the declared
    # /artifacts/... address recorded by upstream receipts.
    return mounted


def _project_path(value: str, *, field: str) -> Path:
    pure = PurePosixPath(value)
    remote = PurePosixPath(str(REMOTE_ROOT))
    if (
        not value
        or not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != value
        or not pure.is_relative_to(remote)
    ):
        raise RuntimeError(f"{field} must be below {REMOTE_ROOT}")
    return REMOTE_ROOT / Path(*pure.relative_to(remote).parts)


def _write_preplan_progress(
    *,
    run_root: Path,
    request: Mapping[str, Any],
    receipt_sha256: str,
    phase: str,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Publish the latest durable pre-plan boundary without granting authority."""

    body = {
        "schema": "compose.editing_v2_role_lane_preplan_progress",
        "schema_version": 1,
        "status": phase,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "continuation_run_identity_sha256": request["run_identity_sha256"],
        "source_revision": request["source_revision"],
        "upstream_validation_receipt_sha256": receipt_sha256,
        "updated_at": _utc_now(),
        "detail": dict(detail or {}),
    }
    _write_operational_json(
        run_root / PREPLAN_PROGRESS_FILENAME,
        _self_hashed(body, "progress_sha256"),
    )


def _build_plan_pointer(
    *,
    request: Mapping[str, Any],
    plan_path: Path,
    plan: Mapping[str, Any],
    upstream_validation_receipt_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": "compose.editing_v2_role_lane_mapreduce_plan_pointer",
        "schema_version": 1,
        "status": "COMPLETE_REUSABLE_PLAN_NO_TRAINING_AUTHORITY",
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "continuation_run_identity_sha256": request["run_identity_sha256"],
        "code_revision": request["source_revision"]["commit"],
        "upstream_validation_receipt_sha256": upstream_validation_receipt_sha256,
        "plan_artifact_path": continuation.artifact_address(
            plan_path,
            artifact_root=ARTIFACT_ROOT,
        ),
        "plan_file_sha256": _file_sha256(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "plan_run_identity_sha256": plan["run_identity_sha256"],
    }
    return _self_hashed(body, "pointer_sha256")


def _load_reusable_plan(
    *,
    pointer_path: Path,
    request: Mapping[str, Any],
    upstream_validation_receipt_sha256: str,
    expected_final_output_prefix: str,
    expected_candidate_identity: Mapping[str, Any],
    expected_source_stream: Mapping[str, Any],
    expected_split_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    """Strictly reopen a completed plan and all task inputs, or return absent."""

    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot load reusable plan pointer: {pointer_path}"
        ) from error
    if not isinstance(pointer, Mapping):
        raise RuntimeError("reusable plan pointer must be an object")
    pointer = dict(pointer)
    supplied = pointer.get("pointer_sha256")
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "gate_zero_authorized",
        "bounded_p50_authorized",
        "continuation_run_identity_sha256",
        "code_revision",
        "upstream_validation_receipt_sha256",
        "plan_artifact_path",
        "plan_file_sha256",
        "plan_sha256",
        "plan_run_identity_sha256",
        "pointer_sha256",
    }
    body = {key: item for key, item in pointer.items() if key != "pointer_sha256"}
    if (
        set(pointer) != expected_fields
        or pointer.get("schema")
        != "compose.editing_v2_role_lane_mapreduce_plan_pointer"
        or pointer.get("schema_version") != 1
        or pointer.get("status") != "COMPLETE_REUSABLE_PLAN_NO_TRAINING_AUTHORITY"
        or pointer.get("training_authorized") is not False
        or pointer.get("gate_zero_authorized") is not False
        or pointer.get("bounded_p50_authorized") is not False
        or supplied != _canonical_sha256(body)
        or pointer.get("continuation_run_identity_sha256")
        != request["run_identity_sha256"]
        or pointer.get("code_revision") != request["source_revision"]["commit"]
        or pointer.get("upstream_validation_receipt_sha256")
        != upstream_validation_receipt_sha256
    ):
        raise RuntimeError("reusable plan pointer identity or authority disagrees")
    plan_path = _artifact_path(
        str(pointer.get("plan_artifact_path", "")),
        field="reusable map/reduce plan",
    )
    if _file_sha256(plan_path) != pointer.get("plan_file_sha256"):
        raise RuntimeError("reusable map/reduce plan bytes disagree with pointer")
    plan = mapreduce.validate_role_lane_mapreduce_plan(
        plan_path,
        artifact_root=ARTIFACT_ROOT,
        validate_task_inputs=True,
    )
    reference_body = plan["reference_materialization"]["run_identity_body"]
    if (
        plan.get("plan_sha256") != pointer.get("plan_sha256")
        or plan.get("run_identity_sha256") != pointer.get("plan_run_identity_sha256")
        or plan.get("code_revision") != request["source_revision"]["commit"]
        or plan.get("upstream_validation_receipt_sha256")
        != upstream_validation_receipt_sha256
        or reference_body.get("output_artifact_prefix") != expected_final_output_prefix
        or reference_body.get("candidate_materialization")
        != dict(expected_candidate_identity)
        or reference_body.get("candidate_provenance_source_stream")
        != dict(expected_source_stream)
        or reference_body.get("split_assignment") != dict(expected_split_identity)
    ):
        raise RuntimeError(
            "reusable plan does not bind the current continuation inputs"
        )
    return plan, plan_path


def _task_progress_summary(plan: Mapping[str, Any]) -> dict[str, int]:
    root = _artifact_path(
        f"{plan['run_artifact_root']}/progress",
        field="map progress root",
    )
    counts: dict[str, int] = {}
    if root.is_dir():
        for progress in root.glob(f"*/{mapreduce.TASK_PROGRESS_FILENAME}"):
            try:
                status = json.loads(progress.read_bytes()).get("status", "INVALID")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                status = "INVALID"
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


@app.function(
    image=image,
    cpu=2.0,
    memory=16384,
    timeout=8 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def map_one_source(plan_path: str, task_id: str, max_source_row_bytes: int) -> dict:
    artifact_volume.reload()
    try:
        receipt = mapreduce.map_role_lane_source_shard(
            plan_path=plan_path,
            task_id=task_id,
            artifact_root=ARTIFACT_ROOT,
            max_source_row_bytes=max_source_row_bytes,
        )
        print(
            json.dumps(
                {
                    "phase": "role_lane_map_complete",
                    "task_identity_sha256": task_id,
                    "records": receipt["totals"]["records"],
                    "fragments": receipt["totals"]["fragments"],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return receipt
    finally:
        artifact_volume.commit()


@app.function(
    image=image,
    cpu=2.0,
    memory=16384,
    timeout=8 * 3600,
    max_containers=MAX_CELL_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_one_cell(plan_path: str, lane: str, role: str) -> dict:
    artifact_volume.reload()
    try:
        receipt = mapreduce.reduce_role_lane_cell(
            plan_path=plan_path,
            lane=lane,
            role=role,
            artifact_root=ARTIFACT_ROOT,
        )
        print(
            json.dumps(
                {
                    "phase": "role_lane_cell_complete",
                    "data_lane": lane,
                    "partition_role": role,
                    "records": receipt["records"],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return receipt
    finally:
        artifact_volume.commit()


@app.function(
    image=image,
    cpu=2.0,
    memory=32768,
    timeout=8 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def assemble_final(plan_path: str) -> dict:
    artifact_volume.reload()
    try:
        manifest = mapreduce.reduce_role_lane_mapreduce(
            plan_path=plan_path,
            artifact_root=ARTIFACT_ROOT,
        )
        print(
            json.dumps(
                {
                    "phase": "role_lane_final_complete",
                    "run_identity_sha256": manifest["run_identity_sha256"],
                    "records": manifest["totals"]["output_records"],
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return manifest
    finally:
        artifact_volume.commit()


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    *,
    source_revision: dict[str, object],
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    max_source_row_bytes: int,
    output_prefix: str,
) -> dict[str, Any]:
    revision = continuation.validate_source_revision(
        source_revision,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.reload()
    inputs = continuation.load_validated_refinement_inputs(
        parent_completion_artifact_path=parent_completion_artifact_path,
        refinement_artifact_path=refinement_artifact_path,
        artifact_root=ARTIFACT_ROOT,
    )
    request = continuation.build_continuation_request(
        source_revision=revision,
        python_runtime={
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        inputs=inputs,
        artifact_root=ARTIFACT_ROOT,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    run_root_address = f"{output_prefix}/{request['run_identity_sha256']}"
    run_root = _artifact_path(run_root_address, field="continuation run root")
    if _write_immutable_json(
        run_root / continuation.CONTINUATION_REQUEST_FILENAME,
        request,
    ):
        artifact_volume.commit()
    assignment_path = run_root / continuation.REFINED_ASSIGNMENT_FILENAME
    if _write_immutable_json(assignment_path, inputs.refined_assignment):
        artifact_volume.commit()
    assignment_record = continuation.refined_assignment_record(
        assignment_path,
        inputs.refined_assignment,
        artifact_root=ARTIFACT_ROOT,
    )

    parent_request = inputs.parent_completion["request"]
    candidate_root = _artifact_path(
        parent_request["candidate"]["path"], field="candidate path"
    )
    bridge_root = _artifact_path(
        parent_request["candidate_provenance_bridge"]["path"],
        field="candidate provenance bridge path",
    )
    registry_path = _artifact_path(
        parent_request["candidate_provenance_registry"]["path"],
        field="candidate provenance registry path",
    )
    contract_path = _project_path(
        parent_request["editing_corpus_contract"]["path"],
        field="editing corpus contract path",
    )
    role_prefix = f"{run_root_address}/{continuation.ROLE_LANE_OUTPUT_DIRECTORY}"
    execution_prefix = f"{run_root_address}/role_lane_mapreduce_execution"
    upstream_receipt = mapreduce.build_upstream_validation_receipt(
        parent_completion_identity=continuation.parent_identity(
            inputs,
            artifact_root=ARTIFACT_ROOT,
        ),
        parent_request=parent_request,
        source_stream=inputs.parent_assignment["source_stream"],
    )
    upstream_receipt_path = run_root / UPSTREAM_VALIDATION_RECEIPT_FILENAME
    if _write_immutable_json(upstream_receipt_path, upstream_receipt):
        artifact_volume.commit()
    expected_candidate_identity = {
        key: value
        for key, value in parent_request["candidate"].items()
        if key != "path"
    }
    expected_split_identity = {
        "file_sha256": assignment_record["file_sha256"],
        "assignment_sha256": inputs.refined_assignment["assignment_sha256"],
        "candidate_resolution_stream_sha256": inputs.refined_assignment[
            "candidate_resolution_stream_sha256"
        ],
        "source_stream_sha256": inputs.refined_assignment["source_stream"][
            "source_stream_sha256"
        ],
    }
    pointer_path = run_root / PLAN_POINTER_FILENAME
    reusable = _load_reusable_plan(
        pointer_path=pointer_path,
        request=request,
        upstream_validation_receipt_sha256=upstream_receipt["receipt_sha256"],
        expected_final_output_prefix=role_prefix,
        expected_candidate_identity=expected_candidate_identity,
        expected_source_stream=inputs.parent_assignment["source_stream"],
        expected_split_identity=expected_split_identity,
    )
    if reusable is None:
        last_phase = "UPSTREAM_VALIDATION_RECEIPT_PUBLISHED"
        _write_preplan_progress(
            run_root=run_root,
            request=request,
            receipt_sha256=upstream_receipt["receipt_sha256"],
            phase=last_phase,
            detail={"receipt_file_sha256": _file_sha256(upstream_receipt_path)},
        )
        artifact_volume.commit()

        def publish_progress(phase: str, detail: Mapping[str, Any]) -> None:
            nonlocal last_phase
            last_phase = phase
            _write_preplan_progress(
                run_root=run_root,
                request=request,
                receipt_sha256=upstream_receipt["receipt_sha256"],
                phase=phase,
                detail=detail,
            )
            if phase in {"PLAN_PUBLISHED", "PLAN_REUSED"}:
                published_plan_path = _artifact_path(
                    f"{detail['run_artifact_root']}/{mapreduce.PLAN_FILENAME}",
                    field="published map/reduce plan",
                )
                published_plan = mapreduce.validate_role_lane_mapreduce_plan(
                    published_plan_path,
                    artifact_root=ARTIFACT_ROOT,
                    validate_task_inputs=False,
                )
                published_pointer = _build_plan_pointer(
                    request=request,
                    plan_path=published_plan_path,
                    plan=published_plan,
                    upstream_validation_receipt_sha256=upstream_receipt[
                        "receipt_sha256"
                    ],
                )
                _write_immutable_json(pointer_path, published_pointer)
            artifact_volume.commit()

        try:
            plan = mapreduce.plan_role_lane_mapreduce(
                candidate_materialization_dir=candidate_root,
                candidate_provenance_bridge_dir=bridge_root,
                candidate_provenance_registry_path=registry_path,
                split_assignment_path=assignment_path,
                editing_corpus_contract_path=contract_path,
                artifact_root=ARTIFACT_ROOT,
                code_revision=revision["commit"],
                final_output_artifact_prefix=role_prefix,
                execution_artifact_prefix=execution_prefix,
                expected_source_shards=EXPECTED_SOURCE_SHARDS,
                upstream_validation_receipt=upstream_receipt,
                progress_callback=publish_progress,
            )
        except Exception as error:
            _write_preplan_progress(
                run_root=run_root,
                request=request,
                receipt_sha256=upstream_receipt["receipt_sha256"],
                phase="FAILED",
                detail={
                    "last_completed_phase": last_phase,
                    "error_type": type(error).__name__,
                },
            )
            artifact_volume.commit()
            raise
        plan_path = _artifact_path(
            f"{plan['run_artifact_root']}/{mapreduce.PLAN_FILENAME}",
            field="map/reduce plan",
        )
        pointer = _build_plan_pointer(
            request=request,
            plan_path=plan_path,
            plan=plan,
            upstream_validation_receipt_sha256=upstream_receipt["receipt_sha256"],
        )
        if _write_immutable_json(pointer_path, pointer):
            artifact_volume.commit()
    else:
        plan, plan_path = reusable
        _write_preplan_progress(
            run_root=run_root,
            request=request,
            receipt_sha256=upstream_receipt["receipt_sha256"],
            phase="PLAN_REUSED_FROM_POINTER",
            detail={"plan_sha256": plan["plan_sha256"]},
        )
        artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "role_lane_plan_complete",
                "plan_sha256": plan["plan_sha256"],
                "task_count": len(plan["tasks"]),
                "selected_candidates": plan["selected_candidates"],
                "max_map_containers": MAX_MAP_CONTAINERS,
                "progress": _task_progress_summary(plan),
            },
            sort_keys=True,
        )
    )

    map_results = list(
        map_one_source.starmap(
            [
                (str(plan_path), task_id, max_source_row_bytes)
                for task_id in plan["task_order"]
            ]
        )
    )
    if len(map_results) != EXPECTED_SOURCE_SHARDS:
        raise RuntimeError("role/lane map phase lost task results")
    lane_order = plan["reference_materialization"]["run_identity_body"]["lane_order"]
    role_order = plan["reference_materialization"]["run_identity_body"]["role_order"]
    cells = [(lane, role) for lane in lane_order for role in role_order]
    cell_results = list(
        reduce_one_cell.starmap([(str(plan_path), lane, role) for lane, role in cells])
    )
    if len(cell_results) != len(cells):
        raise RuntimeError("role/lane cell reduction lost results")
    manifest = assemble_final.remote(str(plan_path))
    artifact_volume.reload()
    role_root = _artifact_path(
        manifest["run_artifact_root"], field="role/lane output root"
    )
    validated, role_record, _ = continuation.validate_role_lane_output(
        role_root=role_root,
        candidate_root=candidate_root,
        split_assignment_path=assignment_path,
        editing_corpus_contract_path=contract_path,
        artifact_root=ARTIFACT_ROOT,
        expected_output_prefix=role_prefix,
        expected_code_revision=revision["commit"],
        expected_candidate_identity={
            key: value
            for key, value in parent_request["candidate"].items()
            if key != "path"
        },
        expected_source_stream=inputs.parent_assignment["source_stream"],
    )
    if validated != manifest:
        raise RuntimeError(
            "map/reduce result disagrees with strict continuation reopen"
        )
    completion = continuation.build_continuation_completion(
        request=request,
        inputs=inputs,
        artifact_root=ARTIFACT_ROOT,
        run_root=run_root_address,
        assignment_record=assignment_record,
        role_lane_record=role_record,
    )
    completion_path = run_root / continuation.CONTINUATION_COMPLETION_FILENAME
    if _write_immutable_json(completion_path, completion):
        artifact_volume.commit()
    observed, _, reopened_assignment_path, reopened_assignment = (
        continuation.validate_continuation_completion_header(
            continuation.artifact_address(
                completion_path,
                artifact_root=ARTIFACT_ROOT,
            ),
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        )
    )
    if (
        observed != completion
        or reopened_assignment_path != assignment_path
        or reopened_assignment != inputs.refined_assignment
    ):
        raise RuntimeError("published map/reduce continuation does not strictly reopen")
    return {
        "completion": completion,
        "completion_artifact_path": continuation.artifact_address(
            completion_path,
            artifact_root=ARTIFACT_ROOT,
        ),
        "plan_artifact_path": continuation.artifact_address(
            plan_path,
            artifact_root=ARTIFACT_ROOT,
        ),
        "map_task_count": len(map_results),
        "cell_task_count": len(cell_results),
        "task_progress": _task_progress_summary(plan),
    }


@app.local_entrypoint()
def main(
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    expected_commit: str,
    max_source_row_bytes: int = 16 * 1024 * 1024,
    output_prefix: str = continuation.OUTPUT_PREFIX,
) -> None:
    source_revision = continuation.build_source_revision(
        expected_commit=expected_commit,
        repo_root=ROOT,
    )
    result = driver.remote(
        source_revision=source_revision,
        parent_completion_artifact_path=parent_completion_artifact_path,
        refinement_artifact_path=refinement_artifact_path,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "EXPECTED_SOURCE_SHARDS",
    "MAX_CELL_CONTAINERS",
    "MAX_MAP_CONTAINERS",
    "assemble_final",
    "driver",
    "main",
    "map_one_source",
    "reduce_one_cell",
]
