"""Exact continuation from a blocked split into the refined role/lane derivative.

The initial structural split run is immutable, including its negative gate
result.  This module validates that blocked completion, replays the separately
published train/validation-only refinement from the exact parent assignment,
and defines a distinct continuation completion.  It never rewrites the parent
run or represents the refined result as the original greedy assignment.

The continuation remains pre-Active8 and grants no training, Gate 0, T1, P50,
or final-test-selection authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data import editing_v2_role_lane_packed_materializer as role_lane_module
from compose_v4.data.editing_v2_active8_source_adapter import (
    resolve_editing_v2_active8_sources,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import canonical_sha256
from compose_v4.data.editing_v2_role_lane_packed_materializer import (
    LANE_REGISTRY_FILENAME,
    RESOLVED_MEMBERSHIP_FILENAME,
    ROLE_LANE_MATERIALIZATION_FILENAME,
    ROLE_LANE_MATERIALIZATION_SCHEMA,
    ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION,
    ROLE_LANE_MATERIALIZATION_STATUS,
)
from compose_v4.data.editing_v2_split_assignment_refinement import (
    validate_refinement_artifact,
    validate_split_assignment_v2,
)
from compose_v4.data.editing_v2_split_census import validate_split_component_census

PARENT_COMPLETION_SCHEMA = "compose.editing_v2_structural_split_completion"
PARENT_COMPLETION_SCHEMA_VERSION = 1
PARENT_BLOCKED_STATUS = "BLOCKED_SPLIT_GATES_NO_TRAINING_AUTHORITY"
PARENT_COMPLETION_FILENAME = "SPLIT_PIPELINE_COMPLETE.json"

CONTINUATION_REQUEST_SCHEMA = "compose.editing_v2_refined_role_lane_request"
CONTINUATION_REQUEST_SCHEMA_VERSION = 1
CONTINUATION_COMPLETION_SCHEMA = "compose.editing_v2_refined_role_lane_completion"
CONTINUATION_COMPLETION_SCHEMA_VERSION = 1
CONTINUATION_COMPLETION_STATUS = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"
CONTINUATION_REQUEST_FILENAME = "REFINED_ROLE_LANE_REQUEST.json"
REFINED_ASSIGNMENT_FILENAME = "EDITING_V2_REFINED_SPLIT_ASSIGNMENT.json"
CONTINUATION_COMPLETION_FILENAME = "REFINED_ROLE_LANE_COMPLETE.json"
ROLE_LANE_OUTPUT_DIRECTORY = "role_lane_packed"
OUTPUT_PREFIX = "/artifacts/editing_v2/refined_role_lane"
SOURCE_REVISION_SCHEMA = "compose.editing_v2_refined_role_lane_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
SERIALIZED_SOURCE_FILES = (
    "modal_apps/materialize_editing_v2_refined_role_lane_app.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_active8_source_adapter.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_lane_registry.py",
    "src/compose_v4/data/editing_v2_packed_candidate_materializer.py",
    "src/compose_v4/data/editing_v2_refined_role_lane_continuation.py",
    "src/compose_v4/data/editing_v2_role_lane_packed_materializer.py",
    "src/compose_v4/data/editing_v2_split_assignment.py",
    "src/compose_v4/data/editing_v2_split_assignment_refinement.py",
    "src/compose_v4/data/editing_v2_split_census.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/data/provenance_overlay.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PARENT_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "gate_zero_authorized",
    "bounded_p50_authorized",
    "active8_admission_status",
    "final_test_selection_use",
    "request",
    "run_root",
    "census",
    "assignment",
    "role_lane_packed",
    "blockers",
    "completion_sha256",
}
_PARENT_REQUEST_FIELDS = {
    "schema",
    "schema_version",
    "training_authorized",
    "gate_zero_authorized",
    "bounded_p50_authorized",
    "active8_authorized",
    "final_test_selection_authorized",
    "source_revision",
    "python_runtime",
    "candidate",
    "candidate_provenance_bridge",
    "candidate_provenance_registry",
    "split_census_policy",
    "split_assignment_policy",
    "editing_corpus_contract",
    "max_row_bytes",
    "max_source_row_bytes",
    "output_prefix",
    "run_identity_sha256",
}
_CENSUS_RECORD_FIELDS = {
    "artifact_path",
    "file_sha256",
    "census_sha256",
    "status",
    "census_structural_complete",
    "blockers",
}
_ASSIGNMENT_RECORD_FIELDS = {
    "artifact_path",
    "file_sha256",
    "assignment_sha256",
    "policy_schema_version",
    "artifact_schema_version",
    "gate_results",
    "blockers",
}
_ROLE_RECORD_FIELDS = {
    "artifact_root",
    "manifest_file_sha256",
    "manifest_sha256",
    "lane_registry_file_sha256",
    "lane_registry_sha256",
    "membership_file_sha256",
    "membership_receipt_sha256",
    "output_shards",
    "active8_admission_status",
}
_REFINED_ASSIGNMENT_RECORD_FIELDS = {
    "artifact_path",
    "file_sha256",
    "assignment_sha256",
    "candidate_resolution_stream_sha256",
    "gate_results",
    "training_authorized",
}
_CONTINUATION_REQUEST_FIELDS = {
    "schema",
    "schema_version",
    "training_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "active8_authorized",
    "final_test_selection_authorized",
    "source_revision",
    "python_runtime",
    "parent_blocked_completion",
    "split_assignment_refinement",
    "max_source_row_bytes",
    "output_prefix",
    "run_identity_sha256",
}
_ROLE_RUN_IDENTITY_FIELDS = (
    "schema",
    "schema_version",
    "code_revision",
    "materializer_implementation_sha256",
    "candidate_materialization",
    "candidate_provenance_source_stream",
    "split_assignment",
    "editing_corpus_contract",
    "lane_order",
    "role_order",
    "envelope_rewrite_contract",
    "sampler_contract",
    "output_artifact_prefix",
)
_COMPLETION_BLOCKERS = [
    "active8_whole_trace_admission.not_run",
    "gate_zero.not_run",
    "t1.not_run",
    "bounded_p50.not_run",
    "training.not_authorized",
    "final_test_selection.forbidden_not_performed",
]


class EditingV2RefinedRoleLaneContinuationError(ValueError):
    """The refined split continuation cannot be proved exact."""


@dataclass(frozen=True)
class ValidatedRefinementInputs:
    parent_completion_path: Path
    parent_completion: dict[str, Any]
    parent_completion_file_sha256: str
    parent_assignment_path: Path
    parent_assignment: dict[str, Any]
    refinement_path: Path
    refinement: dict[str, Any]
    refinement_file_sha256: str
    refined_assignment: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise EditingV2RefinedRoleLaneContinuationError(
            f"cannot establish continuation Git identity: git {' '.join(arguments)}"
        ) from error


def build_source_revision(
    *, expected_commit: str, repo_root: str | Path
) -> dict[str, Any]:
    """Bind the exact clean committed files serialized into the Modal image."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined role/lane continuation requires the exact clean committed worktree"
        )
    hashes = {
        relative: file_sha256(root / relative) for relative in SERIALIZED_SOURCE_FILES
    }
    body = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": canonical_sha256(hashes),
    }
    return {**body, "source_revision_sha256": canonical_sha256(body)}


def validate_source_revision(value: object, *, repo_root: str | Path) -> dict[str, Any]:
    revision = _require_mapping(
        value,
        fields={
            "schema",
            "schema_version",
            "commit",
            "tree",
            "worktree_clean",
            "serialized_source_hashes",
            "serialized_source_hashes_sha256",
            "source_revision_sha256",
        },
        field="continuation source revision",
    )
    body = {
        key: item for key, item in revision.items() if key != "source_revision_sha256"
    }
    hashes = revision["serialized_source_hashes"]
    root = Path(repo_root)
    observed = {
        relative: file_sha256(root / relative) for relative in SERIALIZED_SOURCE_FILES
    }
    if (
        revision["schema"] != SOURCE_REVISION_SCHEMA
        or revision["schema_version"] != SOURCE_REVISION_SCHEMA_VERSION
        or revision["worktree_clean"] is not True
        or revision["source_revision_sha256"] != canonical_sha256(body)
        or not isinstance(hashes, Mapping)
        or dict(hashes) != observed
        or revision["serialized_source_hashes_sha256"] != canonical_sha256(observed)
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "continuation serialized source revision disagrees"
        )
    _require_commit(revision["commit"], field="source_revision.commit")
    _require_commit(revision["tree"], field="source_revision.tree")
    return revision


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} must be a full lowercase Git commit"
        )
    return value


def _require_mapping(value: object, *, fields: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EditingV2RefinedRoleLaneContinuationError(f"{field} fields disagree")
    return dict(value)


def _artifact_path(value: object, *, artifact_root: Path, field: str) -> Path:
    raw = value if isinstance(value, str) else ""
    pure = PurePosixPath(raw)
    if (
        not raw
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != raw
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} must be a normalized path below /artifacts"
        )
    root = artifact_root.resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} resolves outside artifact_root"
        ) from error
    return resolved


def artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _project_path(value: object, *, repo_root: Path, field: str) -> Path:
    raw = value if isinstance(value, str) else ""
    pure = PurePosixPath(raw)
    declared_root = PurePosixPath(str(repo_root))
    if (
        not raw
        or not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != raw
        or not pure.is_relative_to(declared_root)
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} must be a normalized path below {repo_root}"
        )
    return repo_root / Path(*pure.relative_to(declared_root).parts)


def _load_mapping(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2RefinedRoleLaneContinuationError(
            f"cannot load {field}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EditingV2RefinedRoleLaneContinuationError(f"{field} must be an object")
    return dict(value)


def _self_hash(value: Mapping[str, Any], *, hash_field: str, field: str) -> str:
    supplied = _require_sha256(value.get(hash_field), field=f"{field}.{hash_field}")
    body = {key: item for key, item in value.items() if key != hash_field}
    if supplied != canonical_sha256(body):
        raise EditingV2RefinedRoleLaneContinuationError(f"{field} self-hash disagrees")
    return supplied


def _record_file(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    semantic_field: str,
    field: str,
) -> tuple[Path, dict[str, Any]]:
    path = _artifact_path(
        record.get("artifact_path"), artifact_root=artifact_root, field=field
    )
    if file_sha256(path) != _require_sha256(
        record.get("file_sha256"), field=f"{field}.file_sha256"
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} physical SHA-256 disagrees"
        )
    payload = _load_mapping(path, field=field)
    if payload.get(semantic_field) != record.get(semantic_field):
        raise EditingV2RefinedRoleLaneContinuationError(
            f"{field} semantic identity disagrees"
        )
    _self_hash(payload, hash_field=semantic_field, field=field)
    return path, payload


def load_validated_refinement_inputs(
    *,
    parent_completion_artifact_path: str,
    refinement_artifact_path: str,
    artifact_root: str | Path,
) -> ValidatedRefinementInputs:
    """Reopen the immutable negative parent and replay the exact refinement."""

    root = Path(artifact_root)
    parent_path = _artifact_path(
        parent_completion_artifact_path,
        artifact_root=root,
        field="parent_completion_artifact_path",
    )
    if parent_path.name != PARENT_COMPLETION_FILENAME:
        raise EditingV2RefinedRoleLaneContinuationError(
            "parent completion filename disagrees"
        )
    parent = _require_mapping(
        _load_mapping(parent_path, field="parent structural completion"),
        fields=_PARENT_COMPLETION_FIELDS,
        field="parent structural completion",
    )
    _self_hash(
        parent, hash_field="completion_sha256", field="parent structural completion"
    )
    request = _require_mapping(
        parent["request"], fields=_PARENT_REQUEST_FIELDS, field="parent request"
    )
    request_body = {
        key: item for key, item in request.items() if key != "run_identity_sha256"
    }
    expected_parent_root = (
        f"{request['output_prefix']}/{request['run_identity_sha256']}"
    )
    if (
        parent["schema"] != PARENT_COMPLETION_SCHEMA
        or parent["schema_version"] != PARENT_COMPLETION_SCHEMA_VERSION
        or parent["status"] != PARENT_BLOCKED_STATUS
        or parent["training_authorized"] is not False
        or parent["gate_zero_authorized"] is not False
        or parent["bounded_p50_authorized"] is not False
        or parent["active8_admission_status"] != "NOT_RUN"
        or parent["final_test_selection_use"] != "forbidden_not_performed"
        or parent["role_lane_packed"] is not None
        or request["schema"] != "compose.editing_v2_structural_split_modal_run"
        or request["schema_version"] != 1
        or request["run_identity_sha256"] != canonical_sha256(request_body)
        or parent["run_root"] != expected_parent_root
        or parent["run_root"]
        != artifact_address(parent_path.parent, artifact_root=root)
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "parent completion identity, status, or authority disagrees"
        )
    for authority_field in (
        "training_authorized",
        "gate_zero_authorized",
        "bounded_p50_authorized",
        "active8_authorized",
        "final_test_selection_authorized",
    ):
        if request[authority_field] is not False:
            raise EditingV2RefinedRoleLaneContinuationError(
                f"parent request unexpectedly grants {authority_field}"
            )

    census_record = _require_mapping(
        parent["census"], fields=_CENSUS_RECORD_FIELDS, field="parent census record"
    )
    _, census = _record_file(
        census_record,
        artifact_root=root,
        semantic_field="census_sha256",
        field="parent census",
    )
    validate_split_component_census(census)
    if (
        census.get("census_structural_complete") is not True
        or census.get("invalid_rows")
        or census_record["census_structural_complete"] is not True
        or census_record["status"] != census.get("status")
        or census_record["blockers"] != census.get("blockers")
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "parent census is incomplete or its record disagrees"
        )

    assignment_record = _require_mapping(
        parent["assignment"],
        fields=_ASSIGNMENT_RECORD_FIELDS,
        field="parent assignment record",
    )
    assignment_path, assignment = _record_file(
        assignment_record,
        artifact_root=root,
        semantic_field="assignment_sha256",
        field="parent assignment",
    )
    validated_parent = validate_split_assignment_v2(assignment, require_failed=True)
    if (
        assignment_record["gate_results"] != validated_parent["gate_results"]
        or assignment_record["blockers"] != validated_parent["blockers"]
        or assignment_record["artifact_schema_version"]
        != validated_parent["schema_version"]
        or assignment_record["policy_schema_version"]
        != validated_parent["policy"]["schema_version"]
        or validated_parent["source_census_sha256"] != census["census_sha256"]
        or validated_parent["source_component_inventory_sha256"]
        != census["hard_component_census"]["component_inventory_sha256"]
        or validated_parent["source_stream"] != census.get("source_stream")
        or parent["blockers"] != sorted(set(validated_parent["blockers"]))
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "parent assignment, census, or completion blockers disagree"
        )

    refinement_path = _artifact_path(
        refinement_artifact_path,
        artifact_root=root,
        field="refinement_artifact_path",
    )
    refinement = _load_mapping(refinement_path, field="split assignment refinement")
    validated_refinement = validate_refinement_artifact(
        refinement,
        parent_assignment=validated_parent,
        parent_assignment_file_sha256=file_sha256(assignment_path),
    )
    return ValidatedRefinementInputs(
        parent_completion_path=parent_path,
        parent_completion=parent,
        parent_completion_file_sha256=file_sha256(parent_path),
        parent_assignment_path=assignment_path,
        parent_assignment=validated_parent,
        refinement_path=refinement_path,
        refinement=validated_refinement,
        refinement_file_sha256=file_sha256(refinement_path),
        refined_assignment=dict(validated_refinement["refined_assignment"]),
    )


def parent_identity(
    inputs: ValidatedRefinementInputs, *, artifact_root: str | Path
) -> dict[str, Any]:
    parent = inputs.parent_completion
    request = parent["request"]
    return {
        "artifact_path": artifact_address(
            inputs.parent_completion_path, artifact_root=Path(artifact_root)
        ),
        "file_sha256": inputs.parent_completion_file_sha256,
        "completion_sha256": parent["completion_sha256"],
        "structural_run_identity_sha256": request["run_identity_sha256"],
        "candidate_manifest_sha256": request["candidate"]["manifest_sha256"],
        "candidate_source_stream_sha256": request["candidate_provenance_bridge"][
            "source_stream_sha256"
        ],
        "census_sha256": parent["census"]["census_sha256"],
        "parent_assignment_sha256": parent["assignment"]["assignment_sha256"],
    }


def refinement_identity(
    inputs: ValidatedRefinementInputs, *, artifact_root: str | Path
) -> dict[str, Any]:
    return {
        "artifact_path": artifact_address(
            inputs.refinement_path, artifact_root=Path(artifact_root)
        ),
        "file_sha256": inputs.refinement_file_sha256,
        "refinement_sha256": inputs.refinement["refinement_sha256"],
        "refined_assignment_sha256": inputs.refined_assignment["assignment_sha256"],
        "refined_candidate_resolution_stream_sha256": inputs.refined_assignment[
            "candidate_resolution_stream_sha256"
        ],
    }


def build_continuation_request(
    *,
    source_revision: Mapping[str, Any],
    python_runtime: Mapping[str, str],
    inputs: ValidatedRefinementInputs,
    artifact_root: str | Path,
    max_source_row_bytes: int,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Content-address the continuation by current code and exact upstreams."""

    _require_commit(source_revision.get("commit"), field="source_revision.commit")
    _require_commit(source_revision.get("tree"), field="source_revision.tree")
    if source_revision.get("worktree_clean") is not True:
        raise EditingV2RefinedRoleLaneContinuationError(
            "continuation source revision must be clean"
        )
    if (
        set(python_runtime) != {"implementation", "version"}
        or any(
            not isinstance(value, str) or not value for value in python_runtime.values()
        )
        or type(max_source_row_bytes) is not int
        or max_source_row_bytes <= 0
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "continuation runtime or row bound is invalid"
        )
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=Path(artifact_root),
        field="output_prefix",
    )
    body = {
        "schema": CONTINUATION_REQUEST_SCHEMA,
        "schema_version": CONTINUATION_REQUEST_SCHEMA_VERSION,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "active8_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision": dict(source_revision),
        "python_runtime": dict(python_runtime),
        "parent_blocked_completion": parent_identity(
            inputs, artifact_root=artifact_root
        ),
        "split_assignment_refinement": refinement_identity(
            inputs, artifact_root=artifact_root
        ),
        "max_source_row_bytes": max_source_row_bytes,
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": canonical_sha256(body)}


def refined_assignment_record(
    path: Path,
    assignment: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    return {
        "artifact_path": artifact_address(path, artifact_root=Path(artifact_root)),
        "file_sha256": file_sha256(path),
        "assignment_sha256": assignment["assignment_sha256"],
        "candidate_resolution_stream_sha256": assignment[
            "candidate_resolution_stream_sha256"
        ],
        "gate_results": dict(assignment["gate_results"]),
        "training_authorized": False,
    }


def build_continuation_completion(
    *,
    request: Mapping[str, Any],
    inputs: ValidatedRefinementInputs,
    artifact_root: str | Path,
    run_root: str,
    assignment_record: Mapping[str, Any],
    role_lane_record: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": CONTINUATION_COMPLETION_SCHEMA,
        "schema_version": CONTINUATION_COMPLETION_SCHEMA_VERSION,
        "status": CONTINUATION_COMPLETION_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": dict(request),
        "run_root": run_root,
        "parent_blocked_completion": parent_identity(
            inputs, artifact_root=artifact_root
        ),
        "split_assignment_refinement": refinement_identity(
            inputs, artifact_root=artifact_root
        ),
        "census": dict(inputs.parent_completion["census"]),
        "assignment": dict(assignment_record),
        "role_lane_packed": dict(role_lane_record),
        "blockers": list(_COMPLETION_BLOCKERS),
    }
    return {**body, "completion_sha256": canonical_sha256(body)}


def validate_continuation_completion_header(
    completion_artifact_path: str,
    *,
    artifact_root: str | Path,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], ValidatedRefinementInputs, Path, dict[str, Any]]:
    """Validate wrapper lineage and return the refined assignment for reopening."""

    root = Path(artifact_root)
    completion_path = _artifact_path(
        completion_artifact_path,
        artifact_root=root,
        field="refined continuation completion",
    )
    if completion_path.name != CONTINUATION_COMPLETION_FILENAME:
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined continuation completion filename disagrees"
        )
    completion = _load_mapping(completion_path, field="refined continuation completion")
    _self_hash(
        completion,
        hash_field="completion_sha256",
        field="refined continuation completion",
    )
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "active8_admission_status",
        "final_test_selection_use",
        "request",
        "run_root",
        "parent_blocked_completion",
        "split_assignment_refinement",
        "census",
        "assignment",
        "role_lane_packed",
        "blockers",
        "completion_sha256",
    }
    if set(completion) != expected_fields:
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined continuation completion fields disagree"
        )
    request = _require_mapping(
        completion.get("request"),
        fields=_CONTINUATION_REQUEST_FIELDS,
        field="refined continuation request",
    )
    request_body = {
        key: item for key, item in request.items() if key != "run_identity_sha256"
    }
    expected_run_root = (
        f"{request.get('output_prefix')}/{request.get('run_identity_sha256')}"
    )
    if (
        completion["schema"] != CONTINUATION_COMPLETION_SCHEMA
        or completion["schema_version"] != CONTINUATION_COMPLETION_SCHEMA_VERSION
        or completion["status"] != CONTINUATION_COMPLETION_STATUS
        or any(
            completion[field] is not False
            for field in (
                "training_authorized",
                "gate_zero_authorized",
                "t1_authorized",
                "bounded_p50_authorized",
            )
        )
        or completion["active8_admission_status"] != "NOT_RUN"
        or completion["final_test_selection_use"] != "forbidden_not_performed"
        or completion["blockers"] != _COMPLETION_BLOCKERS
        or request.get("schema") != CONTINUATION_REQUEST_SCHEMA
        or request.get("schema_version") != CONTINUATION_REQUEST_SCHEMA_VERSION
        or request.get("run_identity_sha256") != canonical_sha256(request_body)
        or completion["run_root"] != expected_run_root
        or expected_run_root
        != artifact_address(completion_path.parent, artifact_root=root)
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined continuation identity or authority disagrees"
        )
    for field in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "active8_authorized",
        "final_test_selection_authorized",
    ):
        if request.get(field) is not False:
            raise EditingV2RefinedRoleLaneContinuationError(
                f"refined continuation request unexpectedly grants {field}"
            )

    if repo_root is not None:
        validate_source_revision(request.get("source_revision"), repo_root=repo_root)

    parent_record = request.get("parent_blocked_completion")
    refinement_record = request.get("split_assignment_refinement")
    if not isinstance(parent_record, Mapping) or not isinstance(
        refinement_record, Mapping
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined continuation request lacks parent or refinement identity"
        )
    inputs = load_validated_refinement_inputs(
        parent_completion_artifact_path=str(parent_record.get("artifact_path", "")),
        refinement_artifact_path=str(refinement_record.get("artifact_path", "")),
        artifact_root=root,
    )
    if (
        dict(parent_record) != parent_identity(inputs, artifact_root=root)
        or dict(refinement_record) != refinement_identity(inputs, artifact_root=root)
        or completion["parent_blocked_completion"] != dict(parent_record)
        or completion["split_assignment_refinement"] != dict(refinement_record)
        or completion["census"] != inputs.parent_completion["census"]
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined continuation upstream lineage disagrees"
        )

    assignment_record = _require_mapping(
        completion.get("assignment"),
        fields=_REFINED_ASSIGNMENT_RECORD_FIELDS,
        field="refined assignment record",
    )
    _require_mapping(
        completion.get("role_lane_packed"),
        fields=_ROLE_RECORD_FIELDS,
        field="refined role/lane record",
    )
    assignment_path, assignment = _record_file(
        assignment_record,
        artifact_root=root,
        semantic_field="assignment_sha256",
        field="refined assignment",
    )
    validated_assignment = validate_split_assignment_v2(
        assignment, require_passing=True
    )
    if (
        validated_assignment != inputs.refined_assignment
        or assignment_record.get("candidate_resolution_stream_sha256")
        != validated_assignment["candidate_resolution_stream_sha256"]
        or assignment_record.get("gate_results") != validated_assignment["gate_results"]
        or assignment_record.get("training_authorized") is not False
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "published refined assignment disagrees with deterministic replay"
        )
    return completion, inputs, assignment_path, validated_assignment


def validate_role_lane_output(
    *,
    role_root: Path,
    candidate_root: Path,
    split_assignment_path: Path,
    editing_corpus_contract_path: Path,
    artifact_root: Path,
    expected_output_prefix: str,
    expected_code_revision: str,
    expected_candidate_identity: Mapping[str, Any],
    expected_source_stream: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Strictly reopen the production role/lane materialization and receipt."""

    manifest_path = role_root / ROLE_LANE_MATERIALIZATION_FILENAME
    registry_path = role_root / LANE_REGISTRY_FILENAME
    membership_path = role_root / RESOLVED_MEMBERSHIP_FILENAME
    manifest = _load_mapping(manifest_path, field="role/lane manifest")
    manifest_sha256 = _self_hash(
        manifest, hash_field="manifest_sha256", field="role/lane manifest"
    )
    assignment = _load_mapping(split_assignment_path, field="refined assignment")
    expected_split_identity = {
        "file_sha256": file_sha256(split_assignment_path),
        "assignment_sha256": assignment["assignment_sha256"],
        "candidate_resolution_stream_sha256": assignment[
            "candidate_resolution_stream_sha256"
        ],
        "source_stream_sha256": assignment["source_stream"]["source_stream_sha256"],
    }
    try:
        run_identity_body = {
            field: manifest[field] for field in _ROLE_RUN_IDENTITY_FIELDS
        }
    except KeyError as error:
        raise EditingV2RefinedRoleLaneContinuationError(
            "role/lane manifest lacks a run-identity field"
        ) from error
    expected_run_identity = canonical_sha256(run_identity_body)
    if (
        manifest.get("schema") != ROLE_LANE_MATERIALIZATION_SCHEMA
        or manifest.get("schema_version") != ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("status") != ROLE_LANE_MATERIALIZATION_STATUS
        or manifest.get("training_authorized") is not False
        or manifest.get("active8_admission_status") != "NOT_RUN"
        or manifest.get("code_revision") != expected_code_revision
        or manifest.get("materializer_implementation_sha256")
        != file_sha256(Path(role_lane_module.__file__))
        or manifest.get("candidate_materialization")
        != dict(expected_candidate_identity)
        or manifest.get("candidate_provenance_source_stream")
        != dict(expected_source_stream)
        or manifest.get("split_assignment") != expected_split_identity
        or manifest.get("output_artifact_prefix") != expected_output_prefix
        or manifest.get("run_identity_sha256") != expected_run_identity
        or manifest.get("run_artifact_root")
        != f"{expected_output_prefix}/{expected_run_identity}"
        or manifest.get("run_artifact_root")
        != artifact_address(role_root, artifact_root=artifact_root)
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "role/lane materialization identity or authority disagrees"
        )
    receipt = manifest.get("resolved_packed_membership")
    lane_registry = manifest.get("lane_registry")
    if not isinstance(receipt, Mapping) or not isinstance(lane_registry, Mapping):
        raise EditingV2RefinedRoleLaneContinuationError(
            "role/lane manifest lacks registry or membership identity"
        )
    resolved = resolve_editing_v2_active8_sources(
        candidate_materialization_dir=candidate_root,
        split_assignment_path=split_assignment_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        lane_registry_path=registry_path,
        artifact_root=artifact_root,
        membership_receipt_path=membership_path,
        expected_receipt_sha256=receipt.get("receipt_sha256"),
    )
    if (
        resolved.candidate_materialization_manifest_sha256
        != expected_candidate_identity.get("manifest_sha256")
        or resolved.candidate_provenance_source_stream != dict(expected_source_stream)
        or resolved.split_assignment_sha256 != assignment["assignment_sha256"]
        or resolved.lane_registry_sha256 != lane_registry.get("registry_sha256")
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "resolved role/lane source lineage disagrees"
        )
    record = {
        "artifact_root": artifact_address(role_root, artifact_root=artifact_root),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_sha256": manifest_sha256,
        "lane_registry_file_sha256": file_sha256(registry_path),
        "lane_registry_sha256": lane_registry["registry_sha256"],
        "membership_file_sha256": file_sha256(membership_path),
        "membership_receipt_sha256": receipt["receipt_sha256"],
        "output_shards": manifest["totals"]["output_shards"],
        "active8_admission_status": "NOT_RUN",
    }
    return manifest, record, resolved


def validate_for_semantic_v4_migration(
    completion_artifact_path: str,
    *,
    artifact_root: str | Path,
    repo_root: str | Path,
    max_row_bytes: int,
    candidates_module: Any,
    bridge_module: Any,
    expected_source_shards: int,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Reopen all refined lineage required by semantic-V4 migration."""

    root = Path(artifact_root)
    project_root = Path(repo_root)
    completion, inputs, assignment_path, assignment = (
        validate_continuation_completion_header(
            completion_artifact_path,
            artifact_root=root,
            repo_root=project_root,
        )
    )
    parent_request = inputs.parent_completion["request"]
    candidate_record = _require_mapping(
        parent_request["candidate"],
        fields={
            "path",
            "manifest_file_sha256",
            "manifest_sha256",
            "rows_file_sha256",
            "rows_semantic_sha256",
            "address_stream_sha256",
        },
        field="refined structural candidate identity",
    )
    bridge_record = _require_mapping(
        parent_request["candidate_provenance_bridge"],
        fields={
            "path",
            "manifest_file_sha256",
            "manifest_sha256",
            "source_stream_sha256",
            "outputs",
        },
        field="refined structural bridge identity",
    )
    registry_record = _require_mapping(
        parent_request["candidate_provenance_registry"],
        fields={"path", "file_sha256", "registry_sha256"},
        field="refined structural provenance registry identity",
    )
    contract_record = _require_mapping(
        parent_request["editing_corpus_contract"],
        fields={
            "path",
            "file_sha256",
            "semantic_sha256",
            "contract_id",
            "schema",
            "schema_version",
        },
        field="refined structural corpus contract identity",
    )
    candidate_root = _artifact_path(
        candidate_record["path"], artifact_root=root, field="candidate.path"
    )
    bridge_root = _artifact_path(
        bridge_record["path"], artifact_root=root, field="bridge.path"
    )
    registry_path = _artifact_path(
        registry_record["path"], artifact_root=root, field="registry.path"
    )
    contract_path = _project_path(
        contract_record["path"], repo_root=project_root, field="contract.path"
    )
    candidate_manifest = candidates_module.validate_packed_candidate_materialization(
        candidate_root,
        expected_manifest_sha256=candidate_record["manifest_sha256"],
        max_row_bytes=max_row_bytes,
    )
    candidate_rows = candidate_manifest.get("rows")
    observed_candidate = {
        "manifest_file_sha256": file_sha256(
            candidate_root / candidates_module.MATERIALIZATION_FILENAME
        ),
        "manifest_sha256": candidate_manifest.get("manifest_sha256"),
        "rows_file_sha256": (
            candidate_rows.get("file_sha256")
            if isinstance(candidate_rows, Mapping)
            else None
        ),
        "rows_semantic_sha256": (
            candidate_rows.get("semantic_sha256")
            if isinstance(candidate_rows, Mapping)
            else None
        ),
        "address_stream_sha256": (
            candidate_rows.get("address_stream_sha256")
            if isinstance(candidate_rows, Mapping)
            else None
        ),
    }
    if observed_candidate != {key: candidate_record[key] for key in observed_candidate}:
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined structural candidate identity disagrees"
        )
    bridge_manifest = bridge_module.validate_candidate_provenance_bridge(
        bridge_root,
        candidate_root=candidate_root,
        provenance_registry_path=registry_path,
        editing_corpus_contract_path=contract_path,
        max_row_bytes=max_row_bytes,
        _validated_candidate_materialization=candidate_manifest,
    )
    observed_bridge = {
        "manifest_file_sha256": file_sha256(
            bridge_root / bridge_module.BRIDGE_MANIFEST_FILENAME
        ),
        "manifest_sha256": bridge_manifest.get("manifest_sha256"),
        "source_stream_sha256": bridge_manifest.get("source_stream", {}).get(
            "source_stream_sha256"
        ),
        "outputs": bridge_manifest.get("outputs"),
    }
    if observed_bridge != {key: bridge_record[key] for key in observed_bridge}:
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined structural bridge identity disagrees"
        )
    if (
        file_sha256(registry_path) != registry_record["file_sha256"]
        or file_sha256(contract_path) != contract_record["file_sha256"]
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined structural registry or contract bytes disagree"
        )
    registry = _load_mapping(registry_path, field="candidate provenance registry")
    contract = _load_mapping(contract_path, field="editing corpus contract")
    if (
        registry.get("registry_sha256") != registry_record["registry_sha256"]
        or canonical_sha256(contract) != contract_record["semantic_sha256"]
        or contract.get("contract_id") != contract_record["contract_id"]
        or contract.get("schema") != contract_record["schema"]
        or contract.get("schema_version") != contract_record["schema_version"]
        or assignment.get("source_stream") != bridge_manifest.get("source_stream")
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined structural semantic lineage disagrees"
        )
    role_record = _require_mapping(
        completion["role_lane_packed"],
        fields=_ROLE_RECORD_FIELDS,
        field="refined structural role/lane record",
    )
    role_root = _artifact_path(
        role_record["artifact_root"],
        artifact_root=root,
        field="refined role_lane.artifact_root",
    )
    _, observed_role_record, resolved = validate_role_lane_output(
        role_root=role_root,
        candidate_root=candidate_root,
        split_assignment_path=assignment_path,
        editing_corpus_contract_path=contract_path,
        artifact_root=root,
        expected_output_prefix=(
            f"{completion['run_root']}/{ROLE_LANE_OUTPUT_DIRECTORY}"
        ),
        expected_code_revision=completion["request"]["source_revision"]["commit"],
        expected_candidate_identity={
            key: value for key, value in candidate_record.items() if key != "path"
        },
        expected_source_stream=bridge_manifest["source_stream"],
    )
    if (
        observed_role_record != role_record
        or role_record["output_shards"] != expected_source_shards
        or len(resolved.bindings) != expected_source_shards
    ):
        raise EditingV2RefinedRoleLaneContinuationError(
            "refined structural role/lane record disagrees"
        )
    completion_path = _artifact_path(
        completion_artifact_path,
        artifact_root=root,
        field="refined structural completion path",
    )
    structural_identity = {
        "completion_artifact_path": completion_artifact_path,
        "completion_file_sha256": file_sha256(completion_path),
        "completion_sha256": completion["completion_sha256"],
        "structural_run_identity_sha256": completion["request"]["run_identity_sha256"],
        "candidate_manifest_sha256": candidate_record["manifest_sha256"],
        "bridge_manifest_sha256": bridge_record["manifest_sha256"],
        "candidate_source_stream_sha256": bridge_record["source_stream_sha256"],
        "split_assignment_sha256": assignment["assignment_sha256"],
        "lane_registry_sha256": role_record["lane_registry_sha256"],
        "membership_receipt_sha256": role_record["membership_receipt_sha256"],
        "source_shard_count": len(resolved.bindings),
        "structural_input_kind": "refined_split_continuation_v1",
        "parent_structural_completion_sha256": inputs.parent_completion[
            "completion_sha256"
        ],
        "split_assignment_refinement_sha256": inputs.refinement["refinement_sha256"],
    }
    return completion, structural_identity, resolved


__all__ = [
    "CONTINUATION_COMPLETION_FILENAME",
    "CONTINUATION_COMPLETION_SCHEMA",
    "CONTINUATION_COMPLETION_SCHEMA_VERSION",
    "CONTINUATION_COMPLETION_STATUS",
    "CONTINUATION_REQUEST_FILENAME",
    "CONTINUATION_REQUEST_SCHEMA",
    "CONTINUATION_REQUEST_SCHEMA_VERSION",
    "OUTPUT_PREFIX",
    "REFINED_ASSIGNMENT_FILENAME",
    "ROLE_LANE_OUTPUT_DIRECTORY",
    "EditingV2RefinedRoleLaneContinuationError",
    "ValidatedRefinementInputs",
    "artifact_address",
    "build_continuation_completion",
    "build_continuation_request",
    "build_source_revision",
    "file_sha256",
    "load_validated_refinement_inputs",
    "parent_identity",
    "refined_assignment_record",
    "refinement_identity",
    "validate_continuation_completion_header",
    "validate_for_semantic_v4_migration",
    "validate_role_lane_output",
    "validate_source_revision",
]
