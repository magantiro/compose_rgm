"""Materialize the exact refined Editing-V2 role/lane derivative on Modal.

This CPU-only continuation consumes an immutable blocked structural-split
completion and a separately published deterministic refinement.  It replays
the refinement from the exact failed parent assignment, publishes that passing
schema-v2 assignment under a new content address, invokes the production
role/lane materializer, and emits a distinct nonauthorizing completion.
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
        ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_app.py",
        str(REMOTE_ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_app.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps/materialize_editing_v2_refined_role_lane_mapreduce_app.py",
        str(
            REMOTE_ROOT
            / "modal_apps/materialize_editing_v2_refined_role_lane_mapreduce_app.py"
        ),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-refined-role-lane")
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


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
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
    root = artifact_root.resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{field} resolves outside artifact_root")
    return resolved


def _project_path(value: str, *, remote_root: Path, field: str) -> Path:
    pure = PurePosixPath(value)
    remote = PurePosixPath(str(remote_root))
    if (
        not value
        or not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != value
        or not pure.is_relative_to(remote)
    ):
        raise RuntimeError(f"{field} must be a normalized path below {remote_root}")
    return remote_root / Path(*pure.relative_to(remote).parts)


def _load_mapping(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return dict(value)


def _materialize_or_reopen(
    *,
    output_prefix: str,
    inputs: continuation.ValidatedRefinementInputs,
    assignment_path: Path,
    artifact_root: Path,
    remote_root: Path,
    code_revision: str,
    max_source_row_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    from compose_v4.data.editing_v2_role_lane_packed_materializer import (
        materialize_editing_v2_role_lane_packed,
    )

    parent_request = inputs.parent_completion["request"]
    candidate_root = _artifact_path(
        parent_request["candidate"]["path"],
        artifact_root=artifact_root,
        field="candidate path",
    )
    bridge_root = _artifact_path(
        parent_request["candidate_provenance_bridge"]["path"],
        artifact_root=artifact_root,
        field="bridge path",
    )
    registry_path = _artifact_path(
        parent_request["candidate_provenance_registry"]["path"],
        artifact_root=artifact_root,
        field="provenance registry path",
    )
    contract_path = _project_path(
        parent_request["editing_corpus_contract"]["path"],
        remote_root=remote_root,
        field="editing corpus contract path",
    )
    prefix_parent = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="role/lane output prefix",
    ).parent
    existing = (
        sorted(
            path
            for path in prefix_parent.glob("*")
            if path.is_dir() and not path.name.startswith(".")
        )
        if prefix_parent.exists()
        else []
    )
    if len(existing) > 1:
        raise RuntimeError("role/lane output prefix contains multiple immutable runs")
    if not existing:
        manifest = materialize_editing_v2_role_lane_packed(
            candidate_materialization_dir=candidate_root,
            candidate_provenance_bridge_dir=bridge_root,
            candidate_provenance_registry_path=registry_path,
            split_assignment_path=assignment_path,
            editing_corpus_contract_path=contract_path,
            artifact_root=artifact_root,
            code_revision=code_revision,
            output_artifact_prefix=output_prefix,
            max_source_row_bytes=max_source_row_bytes,
        )
        role_root = _artifact_path(
            manifest["run_artifact_root"],
            artifact_root=artifact_root,
            field="role/lane run root",
        )
        reused = False
    else:
        role_root = existing[0]
        manifest = None
        reused = True
    validated, record, _ = continuation.validate_role_lane_output(
        role_root=role_root,
        candidate_root=candidate_root,
        split_assignment_path=assignment_path,
        editing_corpus_contract_path=contract_path,
        artifact_root=artifact_root,
        expected_output_prefix=output_prefix,
        expected_code_revision=code_revision,
        expected_candidate_identity={
            key: value
            for key, value in parent_request["candidate"].items()
            if key != "path"
        },
        expected_source_stream=inputs.parent_assignment["source_stream"],
    )
    if manifest is not None and manifest != validated:
        raise RuntimeError("role/lane materializer return disagrees with publication")
    return validated, record, reused


def _materialize_impl(
    *,
    source_revision: Mapping[str, Any],
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    max_source_row_bytes: int,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Any | None = None,
) -> dict[str, Any]:
    commit_stage = stage_commit or (lambda: None)
    revision = continuation.validate_source_revision(
        source_revision, repo_root=remote_root
    )
    inputs = continuation.load_validated_refinement_inputs(
        parent_completion_artifact_path=parent_completion_artifact_path,
        refinement_artifact_path=refinement_artifact_path,
        artifact_root=artifact_root,
    )
    request = continuation.build_continuation_request(
        source_revision=revision,
        python_runtime={
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        inputs=inputs,
        artifact_root=artifact_root,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    run_root_address = f"{output_prefix}/{request['run_identity_sha256']}"
    run_root = _artifact_path(
        run_root_address, artifact_root=artifact_root, field="continuation run root"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    if _write_immutable_json(
        run_root / continuation.CONTINUATION_REQUEST_FILENAME, request
    ):
        commit_stage()
    assignment_path = run_root / continuation.REFINED_ASSIGNMENT_FILENAME
    if _write_immutable_json(assignment_path, inputs.refined_assignment):
        commit_stage()
    assignment_record = continuation.refined_assignment_record(
        assignment_path,
        inputs.refined_assignment,
        artifact_root=artifact_root,
    )
    role_prefix = f"{run_root_address}/{continuation.ROLE_LANE_OUTPUT_DIRECTORY}"
    _, role_record, reused = _materialize_or_reopen(
        output_prefix=role_prefix,
        inputs=inputs,
        assignment_path=assignment_path,
        artifact_root=artifact_root,
        remote_root=remote_root,
        code_revision=revision["commit"],
        max_source_row_bytes=max_source_row_bytes,
    )
    commit_stage()
    completion = continuation.build_continuation_completion(
        request=request,
        inputs=inputs,
        artifact_root=artifact_root,
        run_root=run_root_address,
        assignment_record=assignment_record,
        role_lane_record=role_record,
    )
    completion_path = run_root / continuation.CONTINUATION_COMPLETION_FILENAME
    if _write_immutable_json(completion_path, completion):
        commit_stage()
    observed, _, reopened_assignment_path, reopened_assignment = (
        continuation.validate_continuation_completion_header(
            continuation.artifact_address(completion_path, artifact_root=artifact_root),
            artifact_root=artifact_root,
            repo_root=remote_root,
        )
    )
    if (
        observed != completion
        or reopened_assignment_path != assignment_path
        or reopened_assignment != inputs.refined_assignment
    ):
        raise RuntimeError("published continuation does not strictly reopen")
    return {
        "completion": completion,
        "completion_artifact_path": continuation.artifact_address(
            completion_path, artifact_root=artifact_root
        ),
        "role_lane_reused": reused,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_refined_role_lane(
    *,
    source_revision: dict[str, object],
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    max_source_row_bytes: int,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _materialize_impl(
        source_revision=source_revision,
        parent_completion_artifact_path=parent_completion_artifact_path,
        refinement_artifact_path=refinement_artifact_path,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
        stage_commit=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result


@app.local_entrypoint()
def main(
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    expected_commit: str,
    max_source_row_bytes: int = 16 * 1024 * 1024,
    output_prefix: str = continuation.OUTPUT_PREFIX,
) -> None:
    source_revision = continuation.build_source_revision(
        expected_commit=expected_commit, repo_root=ROOT
    )
    result = materialize_refined_role_lane.remote(
        source_revision=source_revision,
        parent_completion_artifact_path=parent_completion_artifact_path,
        refinement_artifact_path=refinement_artifact_path,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "_materialize_impl",
    "_materialize_or_reopen",
    "app",
    "main",
    "materialize_refined_role_lane",
]
