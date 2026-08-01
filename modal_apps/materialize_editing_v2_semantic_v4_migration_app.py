"""Migrate the exact completed Editing V2 structural split to semantic V4.

This CPU-only Modal job accepts one input: the immutable completion receipt of
the structural split pipeline.  It derives every candidate, provenance-bridge,
split-assignment, contract, lane-registry, membership, and physical-shard
address from that receipt.  The production resolver then reopens and validates
the exact ordered five-lane by four-role source grid before the existing
semantic migration map/reduce implementation is allowed to plan any work.

The wrapper, production plan, per-shard receipts, production reduction, and
wrapper completion are all content addressed and restart safe.  The job creates
corpus derivatives only.  It grants no Active8, Gate 0, T1, P50, training,
checkpoint-selection, or final-test authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")

OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_v4_migration"
# Keep legacy Volume-v1 commits within Modal's guidance of no more than five
# concurrent writers. Every map container publishes a durable receipt, so its
# container cap is a storage-safety limit rather than a compute preference.
MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5
MAX_MAP_CONTAINERS = MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS
EXPECTED_SOURCE_SHARDS = 20

RUN_SCHEMA = "compose.editing_v2_semantic_v4_modal_run"
RUN_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2_semantic_v4_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "compose.editing_v2_semantic_v4_modal_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"

STRUCTURAL_COMPLETION_SCHEMA = "compose.editing_v2_structural_split_completion"
STRUCTURAL_COMPLETION_SCHEMA_VERSION = 1
STRUCTURAL_COMPLETION_STATUS = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"
STRUCTURAL_COMPLETION_FILENAME = "SPLIT_PIPELINE_COMPLETE.json"
REFINED_STRUCTURAL_COMPLETION_SCHEMA = "compose.editing_v2_refined_role_lane_completion"

RUN_REQUEST_FILENAME = "SEMANTIC_V4_MIGRATION_REQUEST.json"
WRAPPER_COMPLETION_FILENAME = "SEMANTIC_V4_MIGRATION_COMPLETE.json"
LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_v4_migration_app.py"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_FIELDS = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "active8_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}

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
        ROOT / LAUNCHER_SOURCE,
        str(REMOTE_ROOT / LAUNCHER_SOURCE),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-semantic-v4-migration")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts", create_if_missing=False
)


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
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
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a full lowercase Git commit")
    return value


def _artifact_path(value: str | Path, *, artifact_root: Path, field: str) -> Path:
    raw = str(value)
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != raw
        or raw.endswith("/")
    ):
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{field} resolves outside the artifact root") from error
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(Path(artifact_root).resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _project_path(value: str | Path, *, remote_root: Path, field: str) -> Path:
    raw = str(value)
    pure = PurePosixPath(raw)
    expected = PurePosixPath(str(remote_root))
    if (
        not raw
        or "\\" in raw
        or not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != raw
        or not pure.is_relative_to(expected)
    ):
        raise RuntimeError(f"{field} must be a normalized path below {remote_root}")
    return Path(remote_root) / Path(*pure.relative_to(expected).parts)


def _load_mapping(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return dict(value)


def _write_immutable_json(path: Path, value: object) -> bool:
    """Publish exact bytes once, or prove that the existing stage is identical."""

    encoded = _canonical_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable semantic-v4 migration collision at {path}")
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
                raise RuntimeError(
                    f"immutable semantic-v4 migration collision at {path}"
                )
        else:
            os.replace(temporary_name, path)
            temporary_name = None
            return True
        return False
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


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
        raise RuntimeError(
            f"cannot establish semantic-v4 Git identity: git {' '.join(arguments)}"
        ) from error


def local_source_revision(
    *,
    expected_commit: str,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Bind the exact clean tree and all production migration source hashes."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic-v4 migration requires the exact clean committed serialized worktree"
        )

    import sys

    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        build_semantic_migration_source_revision,
    )

    core_revision = build_semantic_migration_source_revision(
        commit=commit,
        tree=tree,
        repo_root=root,
        worktree_clean=True,
    )
    launcher_sha256 = _file_sha256(root / LAUNCHER_SOURCE)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "launcher": {
            "relative_path": LAUNCHER_SOURCE,
            "file_sha256": launcher_sha256,
        },
        "semantic_core_revision": core_revision,
    }
    return {**body, "source_revision_sha256": _canonical_sha256(body)}


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data import editing_v2_candidate_provenance_bridge as bridge
    from compose_v4.data import editing_v2_packed_candidate_materializer as candidates
    from compose_v4.data import editing_v2_refined_role_lane_continuation as refined
    from compose_v4.data import editing_v2_role_lane_packed_materializer as role_lane
    from compose_v4.data.editing_v2_active8_source_adapter import (
        resolve_editing_v2_active8_sources,
    )
    from compose_v4.data.editing_v2_split_census import validate_split_component_census
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        COMPLETION_FILENAME as CORE_COMPLETION_FILENAME,
    )
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        PLAN_FILENAME as CORE_PLAN_FILENAME,
    )
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        completed_semantic_trace_migration_task_ids,
        execute_semantic_trace_migration_task,
        plan_semantic_trace_migration,
        reduce_semantic_trace_migration,
        validate_semantic_migration_source_revision,
        write_semantic_trace_migration_plan,
    )

    return {
        "bridge": bridge,
        "candidates": candidates,
        "role_lane": role_lane,
        "refined": refined,
        "resolve_editing_v2_active8_sources": resolve_editing_v2_active8_sources,
        "validate_split_component_census": validate_split_component_census,
        "CORE_COMPLETION_FILENAME": CORE_COMPLETION_FILENAME,
        "CORE_PLAN_FILENAME": CORE_PLAN_FILENAME,
        "completed_semantic_trace_migration_task_ids": (
            completed_semantic_trace_migration_task_ids
        ),
        "execute_semantic_trace_migration_task": execute_semantic_trace_migration_task,
        "plan_semantic_trace_migration": plan_semantic_trace_migration,
        "reduce_semantic_trace_migration": reduce_semantic_trace_migration,
        "validate_semantic_migration_source_revision": (
            validate_semantic_migration_source_revision
        ),
        "write_semantic_trace_migration_plan": write_semantic_trace_migration_plan,
    }


def _validate_source_revision(
    value: object,
    *,
    remote_root: Path,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic-v4 source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "launcher",
        "semantic_core_revision",
        "source_revision_sha256",
    }
    launcher = revision.get("launcher")
    body = {
        key: item for key, item in revision.items() if key != "source_revision_sha256"
    }
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or not isinstance(launcher, Mapping)
        or set(launcher) != {"relative_path", "file_sha256"}
        or launcher.get("relative_path") != LAUNCHER_SOURCE
        or launcher.get("file_sha256")
        != _file_sha256(Path(remote_root) / LAUNCHER_SOURCE)
        or revision.get("source_revision_sha256") != _canonical_sha256(body)
    ):
        raise RuntimeError("semantic-v4 serialized source revision disagrees")
    _require_commit(revision.get("commit"), field="source_revision.commit")
    _require_commit(revision.get("tree"), field="source_revision.tree")
    core = loaded["validate_semantic_migration_source_revision"](
        revision.get("semantic_core_revision"),
        repo_root=remote_root,
    )
    if core["commit"] != revision["commit"] or core["tree"] != revision["tree"]:
        raise RuntimeError("semantic-v4 wrapper and production core revisions disagree")
    return revision


def _require_exact_fields(
    value: object, *, fields: set[str], field: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError(f"{field} fields disagree")
    return dict(value)


def _validate_self_hash(
    value: Mapping[str, Any], *, hash_field: str, field: str
) -> str:
    supplied = _require_sha256(value.get(hash_field), field=f"{field}.{hash_field}")
    body = {key: item for key, item in value.items() if key != hash_field}
    if supplied != _canonical_sha256(body):
        raise RuntimeError(f"{field} self-hash disagrees")
    return supplied


def _validate_record_file(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    semantic_field: str,
    field: str,
) -> tuple[Path, dict[str, Any]]:
    path = _artifact_path(
        record.get("artifact_path", ""), artifact_root=artifact_root, field=field
    )
    if _file_sha256(path) != _require_sha256(
        record.get("file_sha256"), field=f"{field}.file_sha256"
    ):
        raise RuntimeError(f"{field} physical SHA-256 disagrees")
    payload = _load_mapping(path, field=field)
    if payload.get(semantic_field) != _require_sha256(
        record.get(semantic_field), field=f"{field}.{semantic_field}"
    ):
        raise RuntimeError(f"{field} semantic identity disagrees")
    _validate_self_hash(payload, hash_field=semantic_field, field=field)
    return path, payload


def _validate_structural_completion(
    completion_artifact_path: str,
    *,
    artifact_root: Path,
    remote_root: Path,
    max_row_bytes: int,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Reopen the complete structural lineage and resolve all 20 source shards."""

    completion_path = _artifact_path(
        completion_artifact_path,
        artifact_root=artifact_root,
        field="structural_completion_path",
    )
    raw_completion = _load_mapping(completion_path, field="structural split completion")
    if raw_completion.get("schema") == REFINED_STRUCTURAL_COMPLETION_SCHEMA:
        return loaded["refined"].validate_for_semantic_v4_migration(
            completion_artifact_path,
            artifact_root=artifact_root,
            repo_root=remote_root,
            max_row_bytes=max_row_bytes,
            candidates_module=loaded["candidates"],
            bridge_module=loaded["bridge"],
            expected_source_shards=EXPECTED_SOURCE_SHARDS,
        )
    if completion_path.name != STRUCTURAL_COMPLETION_FILENAME:
        raise RuntimeError("structural completion path has the wrong filename")
    completion = _require_exact_fields(
        raw_completion,
        fields={
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
        },
        field="structural split completion",
    )
    completion_sha256 = _validate_self_hash(
        completion,
        hash_field="completion_sha256",
        field="structural split completion",
    )
    if (
        completion["schema"] != STRUCTURAL_COMPLETION_SCHEMA
        or completion["schema_version"] != STRUCTURAL_COMPLETION_SCHEMA_VERSION
        or completion["status"] != STRUCTURAL_COMPLETION_STATUS
        or completion["training_authorized"] is not False
        or completion["gate_zero_authorized"] is not False
        or completion["bounded_p50_authorized"] is not False
        or completion["active8_admission_status"] != "NOT_RUN"
        or completion["final_test_selection_use"] != "forbidden_not_performed"
        or completion["run_root"]
        != _artifact_address(completion_path.parent, artifact_root=artifact_root)
    ):
        raise RuntimeError(
            "structural split completion identity or authority disagrees"
        )

    request = _require_exact_fields(
        completion["request"],
        fields={
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
        },
        field="structural split request",
    )
    request_body = {
        key: item for key, item in request.items() if key != "run_identity_sha256"
    }
    expected_structural_run_root = (
        f"{request.get('output_prefix')}/{request.get('run_identity_sha256')}"
    )
    if (
        request["schema"] != "compose.editing_v2_structural_split_modal_run"
        or request["schema_version"] != 1
        or request["training_authorized"] is not False
        or request["gate_zero_authorized"] is not False
        or request["bounded_p50_authorized"] is not False
        or request["active8_authorized"] is not False
        or request["final_test_selection_authorized"] is not False
        or request["run_identity_sha256"] != _canonical_sha256(request_body)
        or completion["run_root"] != expected_structural_run_root
    ):
        raise RuntimeError("structural split request identity or authority disagrees")

    candidate_record = _require_exact_fields(
        request["candidate"],
        fields={
            "path",
            "manifest_file_sha256",
            "manifest_sha256",
            "rows_file_sha256",
            "rows_semantic_sha256",
            "address_stream_sha256",
        },
        field="structural candidate identity",
    )
    bridge_record = _require_exact_fields(
        request["candidate_provenance_bridge"],
        fields={
            "path",
            "manifest_file_sha256",
            "manifest_sha256",
            "source_stream_sha256",
            "outputs",
        },
        field="structural bridge identity",
    )
    registry_record = _require_exact_fields(
        request["candidate_provenance_registry"],
        fields={"path", "file_sha256", "registry_sha256"},
        field="structural provenance registry identity",
    )
    contract_record = _require_exact_fields(
        request["editing_corpus_contract"],
        fields={
            "path",
            "file_sha256",
            "semantic_sha256",
            "contract_id",
            "schema",
            "schema_version",
        },
        field="structural corpus contract identity",
    )
    candidate_root = _artifact_path(
        candidate_record["path"], artifact_root=artifact_root, field="candidate.path"
    )
    bridge_root = _artifact_path(
        bridge_record["path"], artifact_root=artifact_root, field="bridge.path"
    )
    registry_path = _artifact_path(
        registry_record["path"], artifact_root=artifact_root, field="registry.path"
    )
    contract_path = _project_path(
        contract_record["path"], remote_root=remote_root, field="contract.path"
    )

    candidate_manifest = loaded["candidates"].validate_packed_candidate_materialization(
        candidate_root,
        expected_manifest_sha256=candidate_record["manifest_sha256"],
        max_row_bytes=max_row_bytes,
    )
    candidate_rows = candidate_manifest.get("rows")
    observed_candidate = {
        "manifest_file_sha256": _file_sha256(
            candidate_root / loaded["candidates"].MATERIALIZATION_FILENAME
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
        raise RuntimeError(
            "structural candidate physical or semantic identity disagrees"
        )

    bridge_manifest = loaded["bridge"].validate_candidate_provenance_bridge(
        bridge_root,
        candidate_root=candidate_root,
        provenance_registry_path=registry_path,
        editing_corpus_contract_path=contract_path,
        max_row_bytes=max_row_bytes,
        _validated_candidate_materialization=candidate_manifest,
    )
    observed_bridge = {
        "manifest_file_sha256": _file_sha256(
            bridge_root / loaded["bridge"].BRIDGE_MANIFEST_FILENAME
        ),
        "manifest_sha256": bridge_manifest.get("manifest_sha256"),
        "source_stream_sha256": bridge_manifest.get("source_stream", {}).get(
            "source_stream_sha256"
        ),
        "outputs": bridge_manifest.get("outputs"),
    }
    if observed_bridge != {key: bridge_record[key] for key in observed_bridge}:
        raise RuntimeError("structural provenance bridge identity disagrees")
    if _file_sha256(registry_path) != registry_record["file_sha256"]:
        raise RuntimeError("structural provenance registry physical identity disagrees")
    if _file_sha256(contract_path) != contract_record["file_sha256"]:
        raise RuntimeError("structural corpus contract physical identity disagrees")
    registry = _load_mapping(registry_path, field="candidate provenance registry")
    contract = _load_mapping(contract_path, field="editing corpus contract")
    if registry.get("registry_sha256") != registry_record["registry_sha256"]:
        raise RuntimeError("structural provenance registry semantic identity disagrees")
    if (
        _canonical_sha256(contract) != contract_record["semantic_sha256"]
        or contract.get("contract_id") != contract_record["contract_id"]
        or contract.get("schema") != contract_record["schema"]
        or contract.get("schema_version") != contract_record["schema_version"]
    ):
        raise RuntimeError("structural corpus contract semantic identity disagrees")

    census_record = _require_exact_fields(
        completion["census"],
        fields={
            "artifact_path",
            "file_sha256",
            "census_sha256",
            "status",
            "census_structural_complete",
            "blockers",
        },
        field="structural census record",
    )
    _, census = _validate_record_file(
        census_record,
        artifact_root=artifact_root,
        semantic_field="census_sha256",
        field="structural census",
    )
    loaded["validate_split_component_census"](census)
    if census.get("census_structural_complete") is not True or census.get(
        "invalid_rows"
    ):
        raise RuntimeError("structural census is not complete")

    assignment_record = _require_exact_fields(
        completion["assignment"],
        fields={
            "artifact_path",
            "file_sha256",
            "assignment_sha256",
            "policy_schema_version",
            "artifact_schema_version",
            "gate_results",
            "blockers",
        },
        field="structural assignment record",
    )
    assignment_path, assignment = _validate_record_file(
        assignment_record,
        artifact_root=artifact_root,
        semantic_field="assignment_sha256",
        field="structural assignment",
    )
    if (
        assignment.get("training_authorized") is not False
        or not isinstance(assignment.get("gate_results"), Mapping)
        or not assignment["gate_results"]
        or not all(value is True for value in assignment["gate_results"].values())
        or assignment_record["gate_results"] != assignment["gate_results"]
        or assignment.get("source_stream") != bridge_manifest.get("source_stream")
    ):
        raise RuntimeError("structural split assignment lineage or gates disagree")

    role_record = _require_exact_fields(
        completion["role_lane_packed"],
        fields={
            "artifact_root",
            "manifest_file_sha256",
            "manifest_sha256",
            "lane_registry_file_sha256",
            "lane_registry_sha256",
            "membership_file_sha256",
            "membership_receipt_sha256",
            "output_shards",
            "active8_admission_status",
        },
        field="structural role/lane record",
    )
    role_root = _artifact_path(
        role_record["artifact_root"],
        artifact_root=artifact_root,
        field="role_lane.artifact_root",
    )
    role_manifest_path = (
        role_root / loaded["role_lane"].ROLE_LANE_MATERIALIZATION_FILENAME
    )
    lane_registry_path = role_root / loaded["role_lane"].LANE_REGISTRY_FILENAME
    membership_path = role_root / loaded["role_lane"].RESOLVED_MEMBERSHIP_FILENAME
    role_manifest = _load_mapping(role_manifest_path, field="role/lane manifest")
    _validate_self_hash(
        role_manifest, hash_field="manifest_sha256", field="role/lane manifest"
    )
    if (
        role_record["output_shards"] != EXPECTED_SOURCE_SHARDS
        or role_record["active8_admission_status"] != "NOT_RUN"
        or role_manifest.get("training_authorized") is not False
        or role_manifest.get("active8_admission_status") != "NOT_RUN"
        or role_manifest.get("run_artifact_root") != role_record["artifact_root"]
        or role_manifest.get("manifest_sha256") != role_record["manifest_sha256"]
        or _file_sha256(role_manifest_path) != role_record["manifest_file_sha256"]
        or _file_sha256(lane_registry_path) != role_record["lane_registry_file_sha256"]
        or _file_sha256(membership_path) != role_record["membership_file_sha256"]
    ):
        raise RuntimeError(
            "structural role/lane output identity or authority disagrees"
        )

    resolved = loaded["resolve_editing_v2_active8_sources"](
        candidate_materialization_dir=candidate_root,
        split_assignment_path=assignment_path,
        editing_corpus_contract_path=contract_path,
        lane_registry_path=lane_registry_path,
        artifact_root=artifact_root,
        membership_receipt_path=membership_path,
        expected_receipt_sha256=role_record["membership_receipt_sha256"],
    )
    if (
        len(resolved.bindings) != EXPECTED_SOURCE_SHARDS
        or resolved.membership_receipt_file_sha256
        != role_record["membership_file_sha256"]
        or resolved.membership_receipt_sha256
        != role_record["membership_receipt_sha256"]
        or resolved.candidate_materialization_manifest_sha256
        != candidate_record["manifest_sha256"]
        or resolved.candidate_provenance_source_stream
        != bridge_manifest["source_stream"]
        or resolved.split_assignment_sha256 != assignment_record["assignment_sha256"]
        or resolved.lane_registry_sha256 != role_record["lane_registry_sha256"]
    ):
        raise RuntimeError(
            "resolved 20-shard source lineage disagrees with structural completion"
        )

    structural_identity = {
        "completion_artifact_path": completion_artifact_path,
        "completion_file_sha256": _file_sha256(completion_path),
        "completion_sha256": completion_sha256,
        "structural_run_identity_sha256": request["run_identity_sha256"],
        "candidate_manifest_sha256": candidate_record["manifest_sha256"],
        "bridge_manifest_sha256": bridge_record["manifest_sha256"],
        "candidate_source_stream_sha256": bridge_record["source_stream_sha256"],
        "split_assignment_sha256": assignment_record["assignment_sha256"],
        "lane_registry_sha256": role_record["lane_registry_sha256"],
        "membership_receipt_sha256": role_record["membership_receipt_sha256"],
        "source_shard_count": len(resolved.bindings),
    }
    return completion, structural_identity, resolved


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    structural_identity: Mapping[str, Any],
    python_runtime: Mapping[str, str],
    max_row_bytes: int,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Build one content address from the exact upstream and code identities."""

    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise RuntimeError("max_row_bytes must be a positive integer")
    if set(python_runtime) != {"implementation", "version"} or any(
        not isinstance(value, str) or not value.strip()
        for value in python_runtime.values()
    ):
        raise RuntimeError("Python runtime identity is incomplete")
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    if structural_identity.get("source_shard_count") != EXPECTED_SOURCE_SHARDS:
        raise RuntimeError(
            "semantic-v4 run requires exactly 20 structural source shards"
        )
    for field, value in structural_identity.items():
        if field.endswith("sha256"):
            _require_sha256(value, field=f"structural_identity.{field}")
    body: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        **_AUTHORITY_FIELDS,
        "source_revision": dict(source_revision),
        "python_runtime": dict(python_runtime),
        "structural_input": dict(structural_identity),
        "max_row_bytes": max_row_bytes,
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def _missing_task_ids(plan: Mapping[str, Any], completed: set[str]) -> list[str]:
    expected = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    if len(expected) != EXPECTED_SOURCE_SHARDS or len(set(expected)) != len(expected):
        raise RuntimeError("semantic-v4 plan must contain 20 unique task identities")
    unexpected = set(completed) - set(expected)
    if unexpected:
        raise RuntimeError(
            f"completed semantic task identities are unexpected: {sorted(unexpected)}"
        )
    return [task_id for task_id in expected if task_id not in completed]


def _core_completion_identity(
    completion: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    artifact_root: Path,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    completion_sha256 = _validate_self_hash(
        completion,
        hash_field="completion_sha256",
        field="semantic migration core completion",
    )
    completion_path = _artifact_path(
        f"{plan['run_artifact_root']}/{loaded['CORE_COMPLETION_FILENAME']}",
        artifact_root=artifact_root,
        field="core completion path",
    )
    if (
        completion.get("training_authorized") is not False
        or completion.get("gate_zero_run") is not False
        or completion.get("run_identity_sha256") != plan["run_identity_sha256"]
        or completion.get("plan_sha256") != plan["plan_sha256"]
        or completion.get("task_count") != EXPECTED_SOURCE_SHARDS
        or not completion_path.is_file()
        or _load_mapping(completion_path, field="core completion") != dict(completion)
    ):
        raise RuntimeError(
            "semantic migration core completion identity or authority disagrees"
        )
    plan_path = _artifact_path(
        f"{plan['run_artifact_root']}/{loaded['CORE_PLAN_FILENAME']}",
        artifact_root=artifact_root,
        field="core plan path",
    )
    return {
        "plan_artifact_path": _artifact_address(plan_path, artifact_root=artifact_root),
        "plan_file_sha256": _file_sha256(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "completion_artifact_path": _artifact_address(
            completion_path, artifact_root=artifact_root
        ),
        "completion_file_sha256": _file_sha256(completion_path),
        "completion_sha256": completion_sha256,
        "task_inventory_sha256": plan["task_inventory_sha256"],
        "task_count": completion["task_count"],
        "counts": dict(completion["counts"]),
    }


def _wrapper_completion(
    *,
    request: Mapping[str, Any],
    core_identity: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_AUTHORITY_FIELDS,
        "active8_admission_status": "NOT_RUN",
        "gate_zero_status": "NOT_RUN",
        "t1_status": "NOT_RUN",
        "bounded_p50_status": "NOT_RUN",
        "training_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": dict(request),
        "semantic_migration": dict(core_identity),
        "restart_safe": True,
        "blockers": [
            "active8_whole_trace_admission.not_run",
            "gate_zero.not_run",
            "t1.not_run",
            "bounded_p50.not_run",
            "training.not_authorized",
            "final_test_selection.forbidden_not_performed",
        ],
    }
    return {**body, "completion_sha256": _canonical_sha256(body)}


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=8 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def migrate_one_shard(
    plan: dict[str, object],
    task_identity_sha256: str,
) -> dict[str, object]:
    """Execute or strictly reopen one planned whole-shard migration."""

    loaded = _imports()
    artifact_volume.reload()
    result = loaded["execute_semantic_trace_migration_task"](
        plan,
        task_identity_sha256,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    print(json.dumps({"phase": "semantic_v4_map_complete", **result}, sort_keys=True))
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=2 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_complete(plan: dict[str, object]) -> dict[str, object]:
    """Reduce only the exact durable task namespace from the frozen plan."""

    loaded = _imports()
    artifact_volume.reload()
    completion = loaded["reduce_semantic_trace_migration"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "semantic_v4_reduce_complete",
                "run_identity_sha256": completion["run_identity_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "task_count": completion["task_count"],
                "training_authorized": completion["training_authorized"],
            },
            sort_keys=True,
        )
    )
    return completion


@app.function(
    image=image,
    cpu=2.0,
    memory=12288,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    *,
    source_revision: dict[str, object],
    structural_completion_path: str,
    max_row_bytes: int,
    output_prefix: str,
) -> dict[str, object]:
    """Validate the structural input, map only missing tasks, and reduce."""

    loaded = _imports()
    _validate_source_revision(source_revision, remote_root=REMOTE_ROOT, loaded=loaded)
    artifact_volume.reload()
    _, structural_identity, resolved = _validate_structural_completion(
        structural_completion_path,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        max_row_bytes=max_row_bytes,
        loaded=loaded,
    )
    request = build_run_request(
        source_revision=source_revision,
        structural_identity=structural_identity,
        python_runtime={
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        max_row_bytes=max_row_bytes,
        output_prefix=output_prefix,
    )
    run_root_address = f"{output_prefix}/{request['run_identity_sha256']}"
    run_root = _artifact_path(
        run_root_address,
        artifact_root=ARTIFACT_ROOT,
        field="semantic-v4 run root",
    )
    if _write_immutable_json(run_root / RUN_REQUEST_FILENAME, request):
        artifact_volume.commit()

    semantic_output_prefix = f"{run_root_address}/semantic_mapreduce"
    plan = loaded["plan_semantic_trace_migration"](
        resolved,
        source_revision=source_revision["semantic_core_revision"],
        repo_root=REMOTE_ROOT,
        output_artifact_prefix=semantic_output_prefix,
    )
    plan_path = loaded["write_semantic_trace_migration_plan"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    completed = loaded["completed_semantic_trace_migration_task_ids"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    missing = _missing_task_ids(plan, completed)
    print(
        json.dumps(
            {
                "phase": "semantic_v4_plan_complete",
                "request_sha256": request["run_identity_sha256"],
                "plan_path": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "expected_tasks": EXPECTED_SOURCE_SHARDS,
                "completed_tasks": len(completed),
                "missing_tasks": len(missing),
                "max_map_containers": MAX_MAP_CONTAINERS,
                "cpu_only": True,
            },
            sort_keys=True,
        )
    )
    if missing:
        results = list(
            migrate_one_shard.starmap(
                [(plan, task_identity_sha256) for task_identity_sha256 in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("semantic-v4 migration map lost task results")

    core_completion = reduce_complete.remote(plan)
    artifact_volume.reload()
    core_identity = _core_completion_identity(
        core_completion,
        plan=plan,
        artifact_root=ARTIFACT_ROOT,
        loaded=loaded,
    )
    completion = _wrapper_completion(
        request=request,
        core_identity=core_identity,
    )
    if _write_immutable_json(run_root / WRAPPER_COMPLETION_FILENAME, completion):
        artifact_volume.commit()
    return {
        "completion": completion,
        "run_root": run_root_address,
        "reused_task_count": len(completed),
        "materialized_task_count": len(missing),
    }


@app.local_entrypoint()
def main(
    commit: str,
    structural_completion_path: str,
    max_row_bytes: int = 2 * 1024 * 1024,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=commit)
    result = driver.remote(
        source_revision=source_revision,
        structural_completion_path=structural_completion_path,
        max_row_bytes=max_row_bytes,
        output_prefix=output_prefix,
    )
    print(
        json.dumps(
            {
                "phase": "editing_v2_semantic_v4_migration_complete",
                "source_revision": source_revision,
                **result,
                "cpu_only": True,
                "active8_launched": False,
                "training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
