"""Build the exact Editing V2 structural split derivative on Modal.

This CPU-only launcher starts from one validated candidate-provenance bridge,
builds the version-4 structural component census, applies the frozen version-1
assignment policy, and materializes the physical role/lane packed derivative
only when every assignment gate passes.

The census intentionally remains ``BLOCKED`` by external-authority requirements
even when its bounded structural computation is complete.  Every stage is an
immutable artifact below one content-addressed run root.  Existing stages are
strictly reopened on restart.  This launcher grants no training, Gate 0, P50,
Active8, or final-test-selection authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")

DEFAULT_CENSUS_POLICY_PATH = "/root/compose/configs/editing_v2_split_census_policy_v4.json"
DEFAULT_ASSIGNMENT_POLICY_PATH = "/root/compose/configs/editing_v2_split_assignment_policy_v1.json"
DEFAULT_CORPUS_CONTRACT_PATH = "/root/compose/configs/editing_corpus_v2_contract.json"
OUTPUT_PREFIX = "/artifacts/editing_v2/split_pipeline"

RUN_SCHEMA = "compose.editing_v2_structural_split_modal_run"
RUN_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2_structural_split_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "compose.editing_v2_structural_split_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETE_STATUS = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"
STRUCTURAL_BLOCKED_STATUS = "BLOCKED_STRUCTURAL_CENSUS_NO_TRAINING_AUTHORITY"
ASSIGNMENT_BLOCKED_STATUS = "BLOCKED_SPLIT_GATES_NO_TRAINING_AUTHORITY"

RUN_REQUEST_FILENAME = "SPLIT_PIPELINE_REQUEST.json"
CENSUS_FILENAME = "EDITING_V2_SPLIT_CENSUS_V4.json"
ASSIGNMENT_FILENAME = "EDITING_V2_SPLIT_ASSIGNMENT.json"
COMPLETION_FILENAME = "SPLIT_PIPELINE_COMPLETE.json"
ROLE_LANE_OUTPUT_DIRECTORY = "role_lane_packed"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SERIALIZED_SOURCE_FILES = (
    "modal_apps/materialize_editing_v2_split_pipeline_app.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_active8_source_adapter.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_lane_registry.py",
    "src/compose_v4/data/editing_v2_packed_candidate_materializer.py",
    "src/compose_v4/data/editing_v2_role_lane_packed_materializer.py",
    "src/compose_v4/data/editing_v2_split_assignment.py",
    "src/compose_v4/data/editing_v2_split_census.py",
)
_CENSUS_PROVENANCE_SOURCE_FILES = (
    "modal_apps/materialize_editing_v2_split_pipeline_app.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_split_assignment.py",
    "src/compose_v4/data/editing_v2_split_census.py",
)
_ROLE_LANE_RUN_IDENTITY_FIELDS = (
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
        ROOT / "modal_apps" / "materialize_editing_v2_split_pipeline_app.py",
        str(REMOTE_ROOT / "modal_apps/materialize_editing_v2_split_pipeline_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-structural-split-pipeline")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


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
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git commit")
    return value


def _artifact_path(value: str | Path, *, artifact_root: Path, field: str) -> Path:
    raw = str(value)
    pure = PurePosixPath(raw)
    if (
        not raw
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != raw
    ):
        raise ValueError(f"{field} must be a normalized path below /artifacts")
    root = artifact_root.resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} resolves outside the artifact root") from error
    return resolved


def _remote_project_path(value: str | Path, *, remote_root: Path, field: str) -> Path:
    raw = str(value)
    pure = PurePosixPath(raw)
    expected_root = PurePosixPath(str(remote_root))
    if (
        not raw
        or not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != raw
        or not pure.is_relative_to(expected_root)
    ):
        raise ValueError(f"{field} must be a normalized path below {remote_root}")
    return remote_root / Path(*pure.relative_to(expected_root).parts)


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _load_mapping(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a JSON object")
    return dict(value)


def _write_immutable_json(path: Path, value: object) -> bool:
    """Write once, or prove that the exact bytes already exist."""

    encoded = _canonical_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable split-pipeline collision at {path}")
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
                raise RuntimeError(f"immutable split-pipeline collision at {path}")
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
            f"cannot establish structural-split Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SERIALIZED_SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise RuntimeError(f"serialized structural-split source is absent: {source}")
        hashes[relative] = _file_sha256(source)
    return hashes


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind the exact clean committed tree serialized into the Modal image."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "structural split requires the exact clean committed serialized worktree"
        )
    source_hashes = _serialized_source_hashes(root)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": source_hashes,
        "serialized_source_hashes_sha256": _canonical_sha256(source_hashes),
    }
    return {**body, "source_revision_sha256": _canonical_sha256(body)}


def _validate_source_revision(
    value: object,
    *,
    serialized_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("structural-split source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "source_revision_sha256",
    }
    body = {key: item for key, item in revision.items() if key != "source_revision_sha256"}
    live_hashes = _serialized_source_hashes(serialized_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(revision.get("commit"))) is None
        or _COMMIT_RE.fullmatch(str(revision.get("tree"))) is None
        or revision.get("serialized_source_hashes") != live_hashes
        or revision.get("serialized_source_hashes_sha256") != _canonical_sha256(live_hashes)
        or revision.get("source_revision_sha256") != _canonical_sha256(body)
    ):
        raise RuntimeError("structural-split serialized source revision disagrees")
    return revision


def _file_identity(path: Path, *, remote_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(remote_root.resolve()))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _implementation_provenance(
    *,
    revision: Mapping[str, Any],
    census_policy_path: Path,
    contract_path: Path,
    remote_root: Path,
    split_census: Any,
) -> dict[str, Any]:
    return {
        "schema": split_census.IMPLEMENTATION_PROVENANCE_SCHEMA,
        "schema_version": split_census.IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
        "code_revision": {"commit_sha": revision["commit"], "dirty": False},
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "source_files": [
            _file_identity(remote_root / relative, remote_root=remote_root)
            for relative in _CENSUS_PROVENANCE_SOURCE_FILES
        ],
        "policy_file": _file_identity(census_policy_path, remote_root=remote_root),
        "editing_corpus_contract_file": _file_identity(
            contract_path,
            remote_root=remote_root,
        ),
    }


def _candidate_identity(
    candidate_root: Path,
    materialization: Mapping[str, Any],
    *,
    materialization_filename: str,
) -> dict[str, str]:
    rows = materialization.get("rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError("candidate materialization rows identity is absent")
    return {
        "manifest_file_sha256": _file_sha256(candidate_root / materialization_filename),
        "manifest_sha256": _require_sha256(
            materialization.get("manifest_sha256"), field="candidate.manifest_sha256"
        ),
        "rows_file_sha256": _require_sha256(
            rows.get("file_sha256"), field="candidate.rows.file_sha256"
        ),
        "rows_semantic_sha256": _require_sha256(
            rows.get("semantic_sha256"), field="candidate.rows.semantic_sha256"
        ),
        "address_stream_sha256": _require_sha256(
            rows.get("address_stream_sha256"),
            field="candidate.rows.address_stream_sha256",
        ),
    }


def _bridge_identity(
    bridge_root: Path,
    bridge_manifest: Mapping[str, Any],
    *,
    manifest_filename: str,
) -> dict[str, Any]:
    outputs = bridge_manifest.get("outputs")
    source_stream = bridge_manifest.get("source_stream")
    if not isinstance(outputs, Mapping) or not isinstance(source_stream, Mapping):
        raise RuntimeError("candidate provenance bridge lacks output or source-stream identity")
    normalized_outputs: dict[str, Any] = {}
    for name, value in sorted(outputs.items()):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"candidate provenance bridge output {name!r} is malformed")
        normalized_outputs[str(name)] = dict(value)
    return {
        "manifest_file_sha256": _file_sha256(bridge_root / manifest_filename),
        "manifest_sha256": _require_sha256(
            bridge_manifest.get("manifest_sha256"), field="bridge.manifest_sha256"
        ),
        "source_stream_sha256": _require_sha256(
            source_stream.get("source_stream_sha256"),
            field="bridge.source_stream.source_stream_sha256",
        ),
        "outputs": normalized_outputs,
    }


def _registry_identity(path: Path, registry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "file_sha256": _file_sha256(path),
        "registry_sha256": _require_sha256(
            registry.get("registry_sha256"), field="registry.registry_sha256"
        ),
    }


def _named_identity(
    path: Path,
    value: Mapping[str, Any],
    *,
    id_field: str,
) -> dict[str, Any]:
    identifier = value.get(id_field)
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{id_field} must be nonempty text")
    return {
        "file_sha256": _file_sha256(path),
        "semantic_sha256": _canonical_sha256(value),
        id_field: identifier,
        "schema": value.get("schema"),
        "schema_version": value.get("schema_version"),
    }


def _validate_hash_tree(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{field}.{key}"
            if str(key).endswith("sha256"):
                _require_sha256(item, field=child)
            else:
                _validate_hash_tree(item, field=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_hash_tree(item, field=f"{field}[{index}]")


def _require_identity_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    field: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{field} identity fields disagree; expected {sorted(required)}")
    _validate_hash_tree(value, field=field)


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    python_runtime: Mapping[str, str],
    candidate_root: str,
    candidate_identity: Mapping[str, Any],
    bridge_root: str,
    bridge_identity: Mapping[str, Any],
    registry_path: str,
    registry_identity: Mapping[str, Any],
    census_policy_path: str,
    census_policy_identity: Mapping[str, Any],
    assignment_policy_path: str,
    assignment_policy_identity: Mapping[str, Any],
    corpus_contract_path: str,
    corpus_contract_identity: Mapping[str, Any],
    max_row_bytes: int,
    max_source_row_bytes: int,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Name one immutable pipeline from every material input and runtime."""

    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise ValueError("max_row_bytes must be a positive integer")
    if type(max_source_row_bytes) is not int or max_source_row_bytes <= 0:
        raise ValueError("max_source_row_bytes must be a positive integer")
    _require_commit(source_revision.get("commit"), field="source_revision.commit")
    _require_commit(source_revision.get("tree"), field="source_revision.tree")
    if (
        source_revision.get("schema") != SOURCE_REVISION_SCHEMA
        or source_revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or source_revision.get("worktree_clean") is not True
        or set(python_runtime) != {"implementation", "version"}
        or any(
            not isinstance(python_runtime[field], str) or not python_runtime[field].strip()
            for field in python_runtime
        )
    ):
        raise ValueError("source revision or Python runtime identity is incomplete")
    _require_identity_fields(
        candidate_identity,
        required={
            "manifest_file_sha256",
            "manifest_sha256",
            "rows_file_sha256",
            "rows_semantic_sha256",
            "address_stream_sha256",
        },
        field="candidate_identity",
    )
    _require_identity_fields(
        bridge_identity,
        required={
            "manifest_file_sha256",
            "manifest_sha256",
            "source_stream_sha256",
            "outputs",
        },
        field="bridge_identity",
    )
    bridge_outputs = bridge_identity["outputs"]
    expected_bridge_outputs = {
        "candidate_audit_ledger",
        "candidate_audit_rows",
        "split_candidates",
    }
    if not isinstance(bridge_outputs, Mapping) or set(bridge_outputs) != expected_bridge_outputs:
        raise ValueError("bridge_identity.outputs fields disagree")
    for output_name, output_identity in bridge_outputs.items():
        _require_identity_fields(
            output_identity,
            required={"relative_path", "file_sha256", "semantic_sha256"},
            field=f"bridge_identity.outputs.{output_name}",
        )
    _require_identity_fields(
        registry_identity,
        required={"file_sha256", "registry_sha256"},
        field="registry_identity",
    )
    for field, identity, id_field, expected_version in (
        ("census_policy_identity", census_policy_identity, "policy_id", 4),
        ("assignment_policy_identity", assignment_policy_identity, "policy_id", 1),
        ("corpus_contract_identity", corpus_contract_identity, "contract_id", 2),
    ):
        _require_identity_fields(
            identity,
            required={
                "file_sha256",
                "semantic_sha256",
                id_field,
                "schema",
                "schema_version",
            },
            field=field,
        )
        if identity["schema_version"] != expected_version:
            raise ValueError(f"{field}.schema_version must equal {expected_version}")
    _validate_hash_tree(source_revision, field="source_revision")
    for field, path in (
        ("candidate_root", candidate_root),
        ("bridge_root", bridge_root),
        ("registry_path", registry_path),
    ):
        _artifact_path(path, artifact_root=ARTIFACT_ROOT, field=field)
    for field, path in (
        ("census_policy_path", census_policy_path),
        ("assignment_policy_path", assignment_policy_path),
        ("corpus_contract_path", corpus_contract_path),
    ):
        _remote_project_path(path, remote_root=REMOTE_ROOT, field=field)
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    body: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision": dict(source_revision),
        "python_runtime": dict(python_runtime),
        "candidate": {"path": candidate_root, **dict(candidate_identity)},
        "candidate_provenance_bridge": {"path": bridge_root, **dict(bridge_identity)},
        "candidate_provenance_registry": {
            "path": registry_path,
            **dict(registry_identity),
        },
        "split_census_policy": {
            "path": census_policy_path,
            **dict(census_policy_identity),
        },
        "split_assignment_policy": {
            "path": assignment_policy_path,
            **dict(assignment_policy_identity),
        },
        "editing_corpus_contract": {
            "path": corpus_contract_path,
            **dict(corpus_contract_identity),
        },
        "max_row_bytes": max_row_bytes,
        "max_source_row_bytes": max_source_row_bytes,
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def _iter_jsonl(path: Path, *, max_row_bytes: int) -> Iterator[object]:
    with path.open("rb") as handle:
        index = 0
        while True:
            raw_line = handle.readline(max_row_bytes + 1)
            if not raw_line:
                break
            if len(raw_line) > max_row_bytes or not raw_line.endswith(b"\n"):
                raise RuntimeError(
                    f"split candidate row {index} exceeds max_row_bytes or lacks newline"
                )
            if not raw_line.strip():
                continue
            try:
                yield json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"split candidate row {index} is invalid JSON") from error
            index += 1


def _census_allows_assignment(census: Mapping[str, Any], split_census: Any) -> bool:
    """Preserve structural-only PASS semantics without upgrading authority."""

    split_census.validate_split_component_census(census)
    if (
        census.get("status") != "BLOCKED"
        or census.get("status_scope") != "CENSUS_STRUCTURAL_COMPLETION_ONLY"
        or census.get("census_computation_complete") is not True
        or census.get("corpus_ready") is not False
        or census.get("split_ready") is not False
        or census.get("training_ready") is not False
        or census.get("authority")
        != {
            "corpus": "NO_CORPUS_READINESS_AUTHORITY",
            "split": "NO_SPLIT_AUTHORITY",
            "training": "NO_TRAINING_AUTHORITY",
        }
    ):
        raise RuntimeError("split census upgraded or lost its structural-only authority boundary")
    blockers = census.get("blockers")
    if not isinstance(blockers, list) or any(
        blocker not in blockers for blocker in split_census.REQUIRED_AUTHORITY_BLOCKERS
    ):
        raise RuntimeError("split census lost a mandatory external-authority blocker")
    return census.get("census_structural_complete") is True and not census.get("invalid_rows")


def _assignment_allows_packing(assignment: Mapping[str, Any]) -> bool:
    gates = assignment.get("gate_results")
    if not isinstance(gates, Mapping) or not gates:
        raise RuntimeError("split assignment does not contain gate results")
    if assignment.get("training_authorized") is not False:
        raise RuntimeError("split assignment unexpectedly grants training authority")
    return all(value is True for value in gates.values())


def _artifact_record(path: Path, *, hash_field: str) -> dict[str, Any]:
    payload = _load_mapping(path, field=path.name)
    return {
        "artifact_path": str(path),
        "file_sha256": _file_sha256(path),
        hash_field: _require_sha256(payload.get(hash_field), field=f"{path.name}.{hash_field}"),
    }


def _completion(
    *,
    status: str,
    request: Mapping[str, Any],
    run_root_address: str,
    census_record: Mapping[str, Any],
    assignment_record: Mapping[str, Any] | None,
    role_lane_record: Mapping[str, Any] | None,
    blockers: list[str],
) -> dict[str, Any]:
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": status,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": dict(request),
        "run_root": run_root_address,
        "census": dict(census_record),
        "assignment": None if assignment_record is None else dict(assignment_record),
        "role_lane_packed": None if role_lane_record is None else dict(role_lane_record),
        "blockers": sorted(set(blockers)),
    }
    return {**body, "completion_sha256": _canonical_sha256(body)}


def _imports(remote_root: Path) -> dict[str, Any]:
    import sys

    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data import editing_v2_active8_source_adapter as active8_adapter
    from compose_v4.data import editing_v2_candidate_provenance_bridge as bridge
    from compose_v4.data import editing_v2_lane_registry as lane_registry
    from compose_v4.data import editing_v2_packed_candidate_materializer as materializer
    from compose_v4.data import editing_v2_role_lane_packed_materializer as role_lane
    from compose_v4.data import editing_v2_split_assignment as split_assignment
    from compose_v4.data import editing_v2_split_census as split_census
    from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract

    return {
        "active8_adapter": active8_adapter,
        "bridge": bridge,
        "lane_registry": lane_registry,
        "load_editing_corpus_contract": load_editing_corpus_contract,
        "materializer": materializer,
        "role_lane": role_lane,
        "split_assignment": split_assignment,
        "split_census": split_census,
    }


def _validate_role_lane_output(
    output_root: Path,
    *,
    candidate_root: Path,
    split_assignment_path: Path,
    contract_path: Path,
    artifact_root: Path,
    expected_output_prefix: str,
    expected_code_revision: str,
    expected_candidate_identity: Mapping[str, Any],
    expected_source_stream: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    role_lane = loaded["role_lane"]
    manifest_path = output_root / role_lane.ROLE_LANE_MATERIALIZATION_FILENAME
    manifest = _load_mapping(manifest_path, field="role/lane packed materialization")
    assignment = _load_mapping(split_assignment_path, field="split assignment")
    expected_split_identity = {
        "file_sha256": _file_sha256(split_assignment_path),
        "assignment_sha256": _require_sha256(
            assignment.get("assignment_sha256"),
            field="split_assignment.assignment_sha256",
        ),
        "candidate_resolution_stream_sha256": _require_sha256(
            assignment.get("candidate_resolution_stream_sha256"),
            field="split_assignment.candidate_resolution_stream_sha256",
        ),
        "source_stream_sha256": _require_sha256(
            assignment.get("source_stream", {}).get("source_stream_sha256"),
            field="split_assignment.source_stream.source_stream_sha256",
        ),
    }
    contract = loaded["load_editing_corpus_contract"](contract_path)
    expected_contract_identity = loaded["lane_registry"].editing_corpus_contract_identity(
        contract,
        contract_file_sha256=_file_sha256(contract_path),
    )
    try:
        role_run_identity_body = {
            field: manifest[field] for field in _ROLE_LANE_RUN_IDENTITY_FIELDS
        }
    except KeyError as error:
        raise RuntimeError("role/lane packed materialization lacks run identity fields") from error
    expected_role_run_identity = _canonical_sha256(role_run_identity_body)
    supplied = _require_sha256(manifest.get("manifest_sha256"), field="role_lane.manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        supplied != _canonical_sha256(body)
        or manifest.get("schema") != role_lane.ROLE_LANE_MATERIALIZATION_SCHEMA
        or manifest.get("schema_version") != role_lane.ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("status") != role_lane.ROLE_LANE_MATERIALIZATION_STATUS
        or manifest.get("training_authorized") is not False
        or manifest.get("active8_admission_status") != "NOT_RUN"
        or manifest.get("code_revision") != expected_code_revision
        or manifest.get("candidate_materialization") != dict(expected_candidate_identity)
        or manifest.get("candidate_provenance_source_stream") != dict(expected_source_stream)
        or manifest.get("split_assignment") != expected_split_identity
        or manifest.get("editing_corpus_contract") != expected_contract_identity
        or manifest.get("materializer_implementation_sha256")
        != _file_sha256(Path(role_lane.__file__))
        or manifest.get("output_artifact_prefix") != expected_output_prefix
        or manifest.get("run_identity_sha256") != expected_role_run_identity
        or manifest.get("run_artifact_root")
        != f"{expected_output_prefix}/{expected_role_run_identity}"
        or manifest.get("run_artifact_root")
        != _artifact_address(
            output_root,
            artifact_root=artifact_root,
        )
    ):
        raise RuntimeError("role/lane packed materialization identity or authority disagrees")
    lane_registry_path = output_root / role_lane.LANE_REGISTRY_FILENAME
    membership_path = output_root / role_lane.RESOLVED_MEMBERSHIP_FILENAME
    receipt = manifest.get("resolved_packed_membership")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("role/lane packed materialization lacks membership identity")
    resolved = loaded["active8_adapter"].resolve_editing_v2_active8_sources(
        candidate_materialization_dir=candidate_root,
        split_assignment_path=split_assignment_path,
        editing_corpus_contract_path=contract_path,
        lane_registry_path=lane_registry_path,
        artifact_root=artifact_root,
        membership_receipt_path=membership_path,
        expected_receipt_sha256=receipt.get("receipt_sha256"),
    )
    if resolved.candidate_provenance_source_stream != dict(expected_source_stream):
        raise RuntimeError("role/lane packed output source-stream identity disagrees")
    record = {
        "artifact_root": _artifact_address(output_root, artifact_root=artifact_root),
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": supplied,
        "lane_registry_file_sha256": _file_sha256(lane_registry_path),
        "lane_registry_sha256": manifest["lane_registry"]["registry_sha256"],
        "membership_file_sha256": _file_sha256(membership_path),
        "membership_receipt_sha256": receipt["receipt_sha256"],
        "output_shards": manifest["totals"]["output_shards"],
        "active8_admission_status": "NOT_RUN",
    }
    return manifest, record


def _materialize_or_reuse_role_lane(
    *,
    output_prefix: str,
    candidate_root: Path,
    bridge_root: Path,
    registry_path: Path,
    assignment_path: Path,
    contract_path: Path,
    artifact_root: Path,
    code_revision: str,
    max_source_row_bytes: int,
    candidate_identity: Mapping[str, Any],
    source_stream: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    prefix_path = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="role_lane_output_prefix",
    ).parent
    existing = (
        sorted(
            path
            for path in prefix_path.glob("*")
            if path.is_dir() and not path.name.startswith(".")
        )
        if prefix_path.exists()
        else []
    )
    if len(existing) > 1:
        raise RuntimeError("role/lane output prefix contains multiple immutable run directories")
    if not existing:
        manifest = loaded["role_lane"].materialize_editing_v2_role_lane_packed(
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
        output_root = _artifact_path(
            manifest["run_artifact_root"],
            artifact_root=artifact_root,
            field="role_lane.run_artifact_root",
        )
        reused = False
    else:
        output_root = existing[0]
        reused = True
    validated, record = _validate_role_lane_output(
        output_root,
        candidate_root=candidate_root,
        split_assignment_path=assignment_path,
        contract_path=contract_path,
        artifact_root=artifact_root,
        expected_output_prefix=output_prefix,
        expected_code_revision=code_revision,
        expected_candidate_identity=candidate_identity,
        expected_source_stream=source_stream,
        loaded=loaded,
    )
    if not reused and validated != manifest:
        raise RuntimeError("role/lane materializer return value disagrees with published manifest")
    return validated, record, reused


def _materialize_impl(
    *,
    source_revision: Mapping[str, Any],
    candidate_root: str,
    candidate_provenance_bridge_dir: str,
    candidate_provenance_registry_path: str,
    census_policy_path: str,
    assignment_policy_path: str,
    editing_corpus_contract_path: str,
    max_row_bytes: int,
    max_source_row_bytes: int,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build or strictly reopen each content-bound structural split stage."""

    commit_stage = stage_commit or (lambda: None)
    revision = _validate_source_revision(source_revision, serialized_root=remote_root)
    candidate_path = _artifact_path(
        candidate_root,
        artifact_root=artifact_root,
        field="candidate_root",
    )
    bridge_path = _artifact_path(
        candidate_provenance_bridge_dir,
        artifact_root=artifact_root,
        field="candidate_provenance_bridge_dir",
    )
    registry_path = _artifact_path(
        candidate_provenance_registry_path,
        artifact_root=artifact_root,
        field="candidate_provenance_registry_path",
    )
    census_policy_file = _remote_project_path(
        census_policy_path,
        remote_root=remote_root,
        field="census_policy_path",
    )
    assignment_policy_file = _remote_project_path(
        assignment_policy_path,
        remote_root=remote_root,
        field="assignment_policy_path",
    )
    contract_file = _remote_project_path(
        editing_corpus_contract_path,
        remote_root=remote_root,
        field="editing_corpus_contract_path",
    )
    loaded = _imports(remote_root)
    materializer = loaded["materializer"]
    bridge_module = loaded["bridge"]
    split_census = loaded["split_census"]
    split_assignment = loaded["split_assignment"]

    candidate_manifest = materializer.validate_packed_candidate_materialization(
        candidate_path,
        max_row_bytes=max_row_bytes,
    )
    candidate_input_identity = _candidate_identity(
        candidate_path,
        candidate_manifest,
        materialization_filename=materializer.MATERIALIZATION_FILENAME,
    )
    bridge_manifest = bridge_module.validate_candidate_provenance_bridge(
        bridge_path,
        candidate_root=candidate_path,
        provenance_registry_path=registry_path,
        editing_corpus_contract_path=contract_file,
        max_row_bytes=max_row_bytes,
        _validated_candidate_materialization=candidate_manifest,
    )
    source_stream = split_assignment.validate_candidate_source_stream(
        bridge_manifest.get("source_stream"),
        expected_candidate_materialization=candidate_input_identity,
    )
    split_rows_path = bridge_path / bridge_module.SPLIT_ROWS_FILENAME
    if _file_sha256(split_rows_path) != source_stream["split_candidates"]["file_sha256"]:
        raise RuntimeError("bridge split-candidate bytes disagree with its source stream")

    contract = loaded["load_editing_corpus_contract"](contract_file)
    census_policy = _load_mapping(census_policy_file, field="split-census policy")
    expected_census_policy = split_census.default_split_census_policy(
        contract,
        identity_definitions=census_policy.get("identity_definitions"),
    )
    if census_policy != expected_census_policy:
        raise RuntimeError("frozen split-census policy is not the exact version-4 default")
    assignment_policy = split_assignment.load_split_assignment_policy(assignment_policy_file)
    registry = _load_mapping(registry_path, field="candidate provenance registry")
    provenance = _implementation_provenance(
        revision=revision,
        census_policy_path=census_policy_file,
        contract_path=contract_file,
        remote_root=remote_root,
        split_census=split_census,
    )
    runtime = dict(provenance["python_runtime"])
    request = build_run_request(
        source_revision=revision,
        python_runtime=runtime,
        candidate_root=candidate_root,
        candidate_identity=candidate_input_identity,
        bridge_root=candidate_provenance_bridge_dir,
        bridge_identity=_bridge_identity(
            bridge_path,
            bridge_manifest,
            manifest_filename=bridge_module.BRIDGE_MANIFEST_FILENAME,
        ),
        registry_path=candidate_provenance_registry_path,
        registry_identity=_registry_identity(registry_path, registry),
        census_policy_path=census_policy_path,
        census_policy_identity=_named_identity(
            census_policy_file,
            census_policy,
            id_field="policy_id",
        ),
        assignment_policy_path=assignment_policy_path,
        assignment_policy_identity=_named_identity(
            assignment_policy_file,
            assignment_policy,
            id_field="policy_id",
        ),
        corpus_contract_path=editing_corpus_contract_path,
        corpus_contract_identity=_named_identity(
            contract_file,
            contract,
            id_field="contract_id",
        ),
        max_row_bytes=max_row_bytes,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    output_root = _artifact_path(
        f"{output_prefix}/{request['run_identity_sha256']}",
        artifact_root=artifact_root,
        field="pipeline output root",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / RUN_REQUEST_FILENAME
    if _write_immutable_json(request_path, request):
        commit_stage()

    census_path = output_root / CENSUS_FILENAME
    if census_path.exists():
        census = _load_mapping(census_path, field="split census")
        split_census.validate_split_component_census(census)
        if (
            census.get("policy") != census_policy
            or census.get("source_stream") != source_stream
            or census.get("implementation_provenance") != provenance
        ):
            raise RuntimeError("reused split census disagrees with the exact run request")
        census_reused = True
    else:
        census = split_census.build_split_component_census(
            _iter_jsonl(split_rows_path, max_row_bytes=max_row_bytes),
            policy=census_policy,
            editing_corpus_contract=contract,
            implementation_provenance=provenance,
            source_stream=source_stream,
        )
        if _file_sha256(split_rows_path) != source_stream["split_candidates"]["file_sha256"]:
            raise RuntimeError("bridge split-candidate bytes changed during census construction")
        split_census.validate_split_component_census(census)
        _write_immutable_json(census_path, census)
        commit_stage()
        census_reused = False
    census_record = {
        **_artifact_record(census_path, hash_field="census_sha256"),
        "status": census["status"],
        "census_structural_complete": census["census_structural_complete"],
        "blockers": list(census["blockers"]),
    }

    if not _census_allows_assignment(census, split_census):
        completion = _completion(
            status=STRUCTURAL_BLOCKED_STATUS,
            request=request,
            run_root_address=_artifact_address(output_root, artifact_root=artifact_root),
            census_record=census_record,
            assignment_record=None,
            role_lane_record=None,
            blockers=list(census["blockers"]),
        )
        if _write_immutable_json(output_root / COMPLETION_FILENAME, completion):
            commit_stage()
        return {"completion": completion, "stage": "census", "reused": census_reused}

    assignment_path = output_root / ASSIGNMENT_FILENAME
    expected_assignment = split_assignment.build_split_assignment(
        census,
        policy=assignment_policy,
    )
    if assignment_path.exists():
        assignment = _load_mapping(assignment_path, field="split assignment")
        if assignment != expected_assignment:
            raise RuntimeError("reused split assignment disagrees with its census or policy")
        assignment_reused = True
    else:
        assignment = expected_assignment
        _write_immutable_json(assignment_path, assignment)
        commit_stage()
        assignment_reused = False
    assignment_record = {
        **_artifact_record(assignment_path, hash_field="assignment_sha256"),
        "policy_schema_version": assignment_policy["schema_version"],
        "artifact_schema_version": assignment["schema_version"],
        "gate_results": dict(assignment["gate_results"]),
        "blockers": list(assignment["blockers"]),
    }
    if not _assignment_allows_packing(assignment):
        completion = _completion(
            status=ASSIGNMENT_BLOCKED_STATUS,
            request=request,
            run_root_address=_artifact_address(output_root, artifact_root=artifact_root),
            census_record=census_record,
            assignment_record=assignment_record,
            role_lane_record=None,
            blockers=list(assignment["blockers"]),
        )
        if _write_immutable_json(output_root / COMPLETION_FILENAME, completion):
            commit_stage()
        return {
            "completion": completion,
            "stage": "assignment",
            "reused": census_reused and assignment_reused,
        }

    role_output_prefix = (
        f"{output_prefix}/{request['run_identity_sha256']}/{ROLE_LANE_OUTPUT_DIRECTORY}"
    )
    _, role_record, role_reused = _materialize_or_reuse_role_lane(
        output_prefix=role_output_prefix,
        candidate_root=candidate_path,
        bridge_root=bridge_path,
        registry_path=registry_path,
        assignment_path=assignment_path,
        contract_path=contract_file,
        artifact_root=artifact_root,
        code_revision=revision["commit"],
        max_source_row_bytes=max_source_row_bytes,
        candidate_identity=candidate_input_identity,
        source_stream=source_stream,
        loaded=loaded,
    )
    commit_stage()
    remaining_blockers = [
        blocker for blocker in assignment["blockers"] if blocker != "physical_lane_shards.not_built"
    ]
    remaining_blockers.extend(["gate_zero.not_run", "training.not_authorized"])
    completion = _completion(
        status=COMPLETE_STATUS,
        request=request,
        run_root_address=_artifact_address(output_root, artifact_root=artifact_root),
        census_record=census_record,
        assignment_record=assignment_record,
        role_lane_record=role_record,
        blockers=remaining_blockers,
    )
    if _write_immutable_json(output_root / COMPLETION_FILENAME, completion):
        commit_stage()
    return {
        "completion": completion,
        "stage": "role_lane_packed",
        "reused": census_reused and assignment_reused and role_reused,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_structural_split_pipeline(
    *,
    source_revision: dict[str, object],
    candidate_root: str,
    candidate_provenance_bridge_dir: str,
    candidate_provenance_registry_path: str,
    census_policy_path: str,
    assignment_policy_path: str,
    editing_corpus_contract_path: str,
    max_row_bytes: int,
    max_source_row_bytes: int,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    return _materialize_impl(
        source_revision=source_revision,
        candidate_root=candidate_root,
        candidate_provenance_bridge_dir=candidate_provenance_bridge_dir,
        candidate_provenance_registry_path=candidate_provenance_registry_path,
        census_policy_path=census_policy_path,
        assignment_policy_path=assignment_policy_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        max_row_bytes=max_row_bytes,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
        stage_commit=artifact_volume.commit,
    )


@app.local_entrypoint()
def main(
    commit: str,
    candidate_root: str,
    candidate_provenance_bridge_dir: str,
    candidate_provenance_registry_path: str,
    census_policy_path: str = DEFAULT_CENSUS_POLICY_PATH,
    assignment_policy_path: str = DEFAULT_ASSIGNMENT_POLICY_PATH,
    editing_corpus_contract_path: str = DEFAULT_CORPUS_CONTRACT_PATH,
    max_row_bytes: int = 2 * 1024 * 1024,
    max_source_row_bytes: int = 16 * 1024 * 1024,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=commit)
    result = materialize_structural_split_pipeline.remote(
        source_revision=source_revision,
        candidate_root=candidate_root,
        candidate_provenance_bridge_dir=candidate_provenance_bridge_dir,
        candidate_provenance_registry_path=candidate_provenance_registry_path,
        census_policy_path=census_policy_path,
        assignment_policy_path=assignment_policy_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        max_row_bytes=max_row_bytes,
        max_source_row_bytes=max_source_row_bytes,
        output_prefix=output_prefix,
    )
    print(
        json.dumps(
            {
                "phase": "editing_v2_structural_split_pipeline_complete",
                "source_revision": source_revision,
                **result,
                "cpu_only": True,
                "training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
