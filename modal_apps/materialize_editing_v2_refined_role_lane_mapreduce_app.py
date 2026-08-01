"""Restart-safe 62-shard/20-cell Editing-V2 role/lane continuation on Modal.

The driver publishes the same refined continuation contract and final
role/lane derivative as the reference materializer.  CPU work is split into
content-addressed source-shard map tasks and independently reusable lane/role
cell reductions.  Every worker commits the shared volume in ``finally`` so a
failure preserves its latest operational progress record.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
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

from compose_v4.data import (
    editing_v2_refined_role_lane_continuation as continuation,
)
from compose_v4.data import editing_v2_role_lane_mapreduce as mapreduce

EXPECTED_SOURCE_SHARDS = 62
# ``compose-v4-artifacts`` predates an explicit Volume-v2 declaration.  Keep
# concurrent writers/commits within Modal's legacy Volume-v1 guidance.
MAX_MAP_CONTAINERS = 5
MAX_CELL_CONTAINERS = 5

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
    resolved = (ARTIFACT_ROOT / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise RuntimeError(f"{field} resolves outside artifact root")
    return resolved


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
    )
    plan_path = _artifact_path(
        f"{plan['run_artifact_root']}/{mapreduce.PLAN_FILENAME}",
        field="map/reduce plan",
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
