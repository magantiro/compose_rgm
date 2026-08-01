"""Refine one exact failed Editing V2 split assignment on Modal.

This CPU-only bridge reopens a physically and semantically pinned structural
split completion, validates its failed schema-v2 assignment, and delegates the
only scientific computation to the deterministic split-refinement module.  It
publishes the refinement, the passing refined assignment, and a self-hashed
completion receipt in one content-addressed namespace.

The bridge does not build role/lane shards and grants no corpus, Active8,
Gate 0, T1, pilot, checkpoint-selection, or training authority.
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

OUTPUT_PREFIX = "/artifacts/editing_v2/split_refinement"
PARENT_COMPLETION_SCHEMA = "compose.editing_v2_structural_split_completion"
PARENT_COMPLETION_SCHEMA_VERSION = 1
PARENT_BLOCKED_STATUS = "BLOCKED_SPLIT_GATES_NO_TRAINING_AUTHORITY"

RUN_SCHEMA = "compose.editing_v2_split_refinement_modal_run"
RUN_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2_split_refinement_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "compose.editing_v2_split_refinement_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_SPLIT_REFINEMENT_NO_TRAINING_AUTHORITY"

REQUEST_FILENAME = "SPLIT_REFINEMENT_REQUEST.json"
REFINEMENT_FILENAME = "EDITING_V2_SPLIT_ASSIGNMENT_REFINEMENT.json"
REFINED_ASSIGNMENT_FILENAME = "EDITING_V2_SPLIT_ASSIGNMENT_REFINED.json"
COMPLETION_FILENAME = "SPLIT_REFINEMENT_COMPLETE.json"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SERIALIZED_SOURCE_FILES = (
    "modal_apps/refine_editing_v2_split_assignment_app.py",
    "src/compose_v4/data/editing_candidate_audit_ledger.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_candidate_router.py",
    "src/compose_v4/data/editing_v2_packed_candidate_materializer.py",
    "src/compose_v4/data/editing_v2_split_assignment_refinement.py",
    "src/compose_v4/data/editing_v2_split_assignment.py",
    "src/compose_v4/data/editing_v2_split_census.py",
)
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
_RUN_FIELDS = {
    "schema",
    "schema_version",
    "training_authorized",
    "split_assignment_authorized",
    "corpus_authorized",
    "active8_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "final_test_selection_authorized",
    "source_revision",
    "python_runtime",
    "parent_completion",
    "parent_assignment",
    "output_prefix",
    "run_identity_sha256",
}
_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "split_assignment_authorized",
    "corpus_authorized",
    "active8_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "final_test_selection_authorized",
    "request",
    "run_root",
    "parent_completion",
    "parent_assignment",
    "refinement",
    "refined_assignment",
    "role_lane_packed",
    "blockers",
    "completion_sha256",
}
_NO_AUTHORITY = {
    "training_authorized": False,
    "split_assignment_authorized": False,
    "corpus_authorized": False,
    "active8_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "final_test_selection_authorized": False,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
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
    .add_local_file(
        ROOT / "modal_apps" / "refine_editing_v2_split_assignment_app.py",
        str(REMOTE_ROOT / "modal_apps/refine_editing_v2_split_assignment_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-split-assignment-refinement")
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


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _load_mapping(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return dict(value)


def _write_immutable_json(path: Path, value: object) -> bool:
    """Atomically write canonical bytes once, or prove exact reuse."""

    encoded = _canonical_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable split-refinement collision at {path}")
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
        try:
            # A same-filesystem hard link publishes the fully fsynced temporary
            # file with create-only semantics. Unlike ``os.replace``, it cannot
            # overwrite an artifact won by a concurrent writer.
            os.link(temporary_name, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"immutable split-refinement collision at {path}")
            return False
        Path(temporary_name).unlink()
        temporary_name = None
        return True
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
            f"cannot establish split-refinement Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SERIALIZED_SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise RuntimeError(
                f"serialized split-refinement source is absent: {source}"
            )
        hashes[relative] = _file_sha256(source)
    return hashes


def local_source_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind the exact clean committed tree serialized into the Modal image."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "split refinement requires the exact clean committed serialized worktree"
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
    value: object, *, serialized_root: Path
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("split-refinement source revision must be an object")
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
    body = {
        key: item for key, item in revision.items() if key != "source_revision_sha256"
    }
    live_hashes = _serialized_source_hashes(serialized_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(revision.get("commit"))) is None
        or _COMMIT_RE.fullmatch(str(revision.get("tree"))) is None
        or revision.get("serialized_source_hashes") != live_hashes
        or revision.get("serialized_source_hashes_sha256")
        != _canonical_sha256(live_hashes)
        or revision.get("source_revision_sha256") != _canonical_sha256(body)
    ):
        raise RuntimeError("split-refinement serialized source revision disagrees")
    return revision


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    python_runtime: Mapping[str, str],
    parent_completion_path: str,
    expected_parent_completion_file_sha256: str,
    expected_parent_completion_sha256: str,
    parent_assignment_path: str,
    expected_parent_assignment_file_sha256: str,
    expected_parent_assignment_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Name one immutable refinement from every material input."""

    _require_commit(source_revision.get("commit"), field="source_revision.commit")
    _require_commit(source_revision.get("tree"), field="source_revision.tree")
    if (
        source_revision.get("schema") != SOURCE_REVISION_SCHEMA
        or source_revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or source_revision.get("worktree_clean") is not True
        or set(python_runtime) != {"implementation", "version"}
        or any(
            not isinstance(python_runtime[field], str)
            or not python_runtime[field].strip()
            for field in python_runtime
        )
    ):
        raise ValueError("source revision or Python runtime identity is incomplete")
    for field, digest in (
        (
            "expected_parent_completion_file_sha256",
            expected_parent_completion_file_sha256,
        ),
        ("expected_parent_completion_sha256", expected_parent_completion_sha256),
        (
            "expected_parent_assignment_file_sha256",
            expected_parent_assignment_file_sha256,
        ),
        ("expected_parent_assignment_sha256", expected_parent_assignment_sha256),
    ):
        _require_sha256(digest, field=field)
    _artifact_path(
        parent_completion_path,
        artifact_root=ARTIFACT_ROOT,
        field="parent_completion_path",
    )
    _artifact_path(
        parent_assignment_path,
        artifact_root=ARTIFACT_ROOT,
        field="parent_assignment_path",
    )
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    body: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        **_NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "python_runtime": dict(python_runtime),
        "parent_completion": {
            "artifact_path": parent_completion_path,
            "file_sha256": expected_parent_completion_file_sha256,
            "completion_sha256": expected_parent_completion_sha256,
        },
        "parent_assignment": {
            "artifact_path": parent_assignment_path,
            "file_sha256": expected_parent_assignment_file_sha256,
            "assignment_sha256": expected_parent_assignment_sha256,
        },
        "output_prefix": output_prefix,
    }
    request = {**body, "run_identity_sha256": _canonical_sha256(body)}
    return validate_run_request(request)


def validate_run_request(value: object) -> dict[str, Any]:
    """Validate one content-addressed, strictly nonauthorizing request."""

    if not isinstance(value, Mapping) or set(value) != _RUN_FIELDS:
        raise ValueError("split-refinement request fields disagree")
    request = dict(value)
    supplied = _require_sha256(
        request.get("run_identity_sha256"), field="run_identity_sha256"
    )
    body = {key: item for key, item in request.items() if key != "run_identity_sha256"}
    revision = request.get("source_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("source_revision must be an object")
    revision_body = {
        key: item for key, item in revision.items() if key != "source_revision_sha256"
    }
    source_hashes = revision.get("serialized_source_hashes")
    parent_completion = request.get("parent_completion")
    parent_assignment = request.get("parent_assignment")
    runtime = request.get("python_runtime")
    if (
        supplied != _canonical_sha256(body)
        or request.get("schema") != RUN_SCHEMA
        or request.get("schema_version") != RUN_SCHEMA_VERSION
        or any(
            request.get(field) is not expected
            for field, expected in _NO_AUTHORITY.items()
        )
        or set(revision)
        != {
            "schema",
            "schema_version",
            "commit",
            "tree",
            "worktree_clean",
            "serialized_source_hashes",
            "serialized_source_hashes_sha256",
            "source_revision_sha256",
        }
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(revision.get("commit"))) is None
        or _COMMIT_RE.fullmatch(str(revision.get("tree"))) is None
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(
            not isinstance(path, str)
            or not path
            or _SHA256_RE.fullmatch(str(digest)) is None
            for path, digest in source_hashes.items()
        )
        or revision.get("serialized_source_hashes_sha256")
        != _canonical_sha256(source_hashes)
        or revision.get("source_revision_sha256") != _canonical_sha256(revision_body)
        or not isinstance(runtime, Mapping)
        or set(runtime) != {"implementation", "version"}
        or any(
            not isinstance(item, str) or not item.strip() for item in runtime.values()
        )
        or not isinstance(parent_completion, Mapping)
        or set(parent_completion)
        != {"artifact_path", "file_sha256", "completion_sha256"}
        or not isinstance(parent_assignment, Mapping)
        or set(parent_assignment)
        != {"artifact_path", "file_sha256", "assignment_sha256"}
    ):
        raise ValueError("split-refinement request identity or authority disagrees")
    for field, record, semantic_field in (
        ("parent_completion", parent_completion, "completion_sha256"),
        ("parent_assignment", parent_assignment, "assignment_sha256"),
    ):
        _artifact_path(
            record["artifact_path"],
            artifact_root=ARTIFACT_ROOT,
            field=f"{field}.artifact_path",
        )
        _require_sha256(record["file_sha256"], field=f"{field}.file_sha256")
        _require_sha256(record[semantic_field], field=f"{field}.{semantic_field}")
    _artifact_path(
        f"{request['output_prefix']}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    return request


def validate_parent_completion(
    value: object,
    *,
    expected_completion_sha256: str,
    parent_completion_path: str,
    parent_assignment_path: str,
    expected_parent_assignment_file_sha256: str,
    expected_parent_assignment_sha256: str,
) -> dict[str, Any]:
    """Validate the exact failed parent receipt and its assignment pointer."""

    if not isinstance(value, Mapping) or set(value) != _PARENT_COMPLETION_FIELDS:
        raise RuntimeError("parent structural-split completion fields disagree")
    completion = dict(value)
    supplied = _require_sha256(
        completion.get("completion_sha256"), field="parent.completion_sha256"
    )
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    parent_root = str(PurePosixPath(parent_completion_path).parent)
    assignment_record = completion.get("assignment")
    request = completion.get("request")
    if not isinstance(assignment_record, Mapping) or set(assignment_record) != {
        "artifact_path",
        "artifact_schema_version",
        "blockers",
        "file_sha256",
        "assignment_sha256",
        "gate_results",
        "policy_schema_version",
    }:
        raise RuntimeError("parent completion assignment record disagrees")
    if not isinstance(request, Mapping):
        raise TypeError("parent completion request is absent")
    request_body = {
        key: item for key, item in request.items() if key != "run_identity_sha256"
    }
    blockers = completion.get("blockers")
    assignment_blockers = assignment_record.get("blockers")
    recorded_assignment_path = str(assignment_record.get("artifact_path", ""))
    expected_assignment_suffix = "/".join(
        PurePosixPath(parent_assignment_path).parts[2:]
    )
    expected_failed_gates = {
        "all_observed_lanes_present_in_every_role": True,
        "all_roles_nonempty": True,
        "overall_mass_ratio_within_limit": True,
        "per_lane_mass_ratio_within_limit": False,
    }
    if (
        supplied != expected_completion_sha256
        or supplied != _canonical_sha256(body)
        or completion.get("schema") != PARENT_COMPLETION_SCHEMA
        or completion.get("schema_version") != PARENT_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != PARENT_BLOCKED_STATUS
        or completion.get("training_authorized") is not False
        or completion.get("gate_zero_authorized") is not False
        or completion.get("bounded_p50_authorized") is not False
        or completion.get("active8_admission_status") != "NOT_RUN"
        or completion.get("final_test_selection_use") != "forbidden_not_performed"
        or completion.get("run_root") != parent_root
        or completion.get("role_lane_packed") is not None
        or request.get("run_identity_sha256") != _canonical_sha256(request_body)
        or not recorded_assignment_path.endswith(f"/{expected_assignment_suffix}")
        or assignment_record.get("file_sha256")
        != expected_parent_assignment_file_sha256
        or assignment_record.get("assignment_sha256")
        != expected_parent_assignment_sha256
        or assignment_record.get("artifact_schema_version") != 2
        or assignment_record.get("policy_schema_version") != 1
        or assignment_record.get("gate_results") != expected_failed_gates
        or not isinstance(assignment_blockers, list)
        or assignment_blockers != sorted(set(assignment_blockers))
        or "split_gate.per_lane_mass_ratio_within_limit" not in assignment_blockers
        or not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or "split_gate.per_lane_mass_ratio_within_limit" not in blockers
    ):
        raise RuntimeError("parent structural-split completion identity disagrees")
    return completion


def build_completion(
    *,
    request: Mapping[str, Any],
    run_root: str,
    parent_completion: Mapping[str, Any],
    parent_assignment: Mapping[str, Any],
    refinement: Mapping[str, Any],
    refined_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed receipt that explicitly grants no authority."""

    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_NO_AUTHORITY,
        "request": dict(request),
        "run_root": run_root,
        "parent_completion": dict(parent_completion),
        "parent_assignment": dict(parent_assignment),
        "refinement": dict(refinement),
        "refined_assignment": dict(refined_assignment),
        "role_lane_packed": None,
        "blockers": [
            "active8.not_run",
            "gate_zero.not_run",
            "role_lane_packed.not_built",
            "semantic_sampling_policy.not_frozen",
            "t1.not_run",
            "training.not_authorized",
        ],
    }
    return {**body, "completion_sha256": _canonical_sha256(body)}


def validate_completion(
    value: object, *, expected_request: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on receipt tampering or an accidental authority upgrade."""

    if not isinstance(value, Mapping) or set(value) != _COMPLETION_FIELDS:
        raise RuntimeError("split-refinement completion fields disagree")
    request = validate_run_request(expected_request)
    completion = dict(value)
    supplied = _require_sha256(
        completion.get("completion_sha256"), field="completion_sha256"
    )
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    blockers = completion.get("blockers")
    run_root = f"{request['output_prefix']}/{request['run_identity_sha256']}"
    parent_completion = completion.get("parent_completion")
    parent_assignment = completion.get("parent_assignment")
    refinement = completion.get("refinement")
    refined_assignment = completion.get("refined_assignment")
    expected_records = (
        (
            parent_completion,
            {"artifact_path", "file_sha256", "completion_sha256"},
            "completion_sha256",
        ),
        (
            parent_assignment,
            {"artifact_path", "file_sha256", "assignment_sha256"},
            "assignment_sha256",
        ),
        (
            refinement,
            {"artifact_path", "file_sha256", "refinement_sha256"},
            "refinement_sha256",
        ),
        (
            refined_assignment,
            {
                "artifact_path",
                "file_sha256",
                "assignment_sha256",
                "candidate_resolution_stream_sha256",
                "all_frozen_split_gates_pass",
            },
            "assignment_sha256",
        ),
    )
    if (
        supplied != _canonical_sha256(body)
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("status") != COMPLETION_STATUS
        or any(
            completion.get(field) is not value for field, value in _NO_AUTHORITY.items()
        )
        or completion.get("request") != request
        or completion.get("run_root") != run_root
        or not all(
            isinstance(record, Mapping) and set(record) == fields
            for record, fields, _ in expected_records
        )
        or parent_completion != request["parent_completion"]
        or parent_assignment != request["parent_assignment"]
        or refinement.get("artifact_path") != f"{run_root}/{REFINEMENT_FILENAME}"
        or refined_assignment.get("artifact_path")
        != f"{run_root}/{REFINED_ASSIGNMENT_FILENAME}"
        or refined_assignment.get("all_frozen_split_gates_pass") is not True
        or completion.get("role_lane_packed") is not None
        or not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or "training.not_authorized" not in blockers
        or "role_lane_packed.not_built" not in blockers
    ):
        raise RuntimeError(
            "split-refinement completion identity or authority disagrees"
        )
    for record, _, semantic_field in expected_records:
        _require_sha256(record["file_sha256"], field="artifact.file_sha256")
        _require_sha256(record[semantic_field], field=f"artifact.{semantic_field}")
    _require_sha256(
        refined_assignment["candidate_resolution_stream_sha256"],
        field="refined_assignment.candidate_resolution_stream_sha256",
    )
    return completion


def _artifact_record(
    path: Path,
    *,
    artifact_root: Path,
    semantic_field: str,
    semantic_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_path": _artifact_address(path, artifact_root=artifact_root),
        "file_sha256": _file_sha256(path),
        semantic_field: _require_sha256(semantic_sha256, field=semantic_field),
    }


def _imports(remote_root: Path) -> Any:
    import sys

    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data import editing_v2_split_assignment_refinement as refinement

    return refinement


def _refine_impl(
    *,
    source_revision: Mapping[str, Any],
    parent_completion_path: str,
    expected_parent_completion_file_sha256: str,
    expected_parent_completion_sha256: str,
    parent_assignment_path: str,
    expected_parent_assignment_file_sha256: str,
    expected_parent_assignment_sha256: str,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Any | None = None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, serialized_root=remote_root)
    completion_path = _artifact_path(
        parent_completion_path,
        artifact_root=artifact_root,
        field="parent_completion_path",
    )
    assignment_path = _artifact_path(
        parent_assignment_path,
        artifact_root=artifact_root,
        field="parent_assignment_path",
    )
    if _file_sha256(completion_path) != expected_parent_completion_file_sha256:
        raise RuntimeError("parent completion physical SHA-256 disagrees")
    if _file_sha256(assignment_path) != expected_parent_assignment_file_sha256:
        raise RuntimeError("parent assignment physical SHA-256 disagrees")
    parent_completion = validate_parent_completion(
        _load_mapping(completion_path, field="parent structural-split completion"),
        expected_completion_sha256=expected_parent_completion_sha256,
        parent_completion_path=parent_completion_path,
        parent_assignment_path=parent_assignment_path,
        expected_parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
        expected_parent_assignment_sha256=expected_parent_assignment_sha256,
    )
    parent_assignment = _load_mapping(assignment_path, field="parent split assignment")
    refinement_module = _imports(remote_root)
    validated_parent = refinement_module.validate_split_assignment_v2(
        parent_assignment,
        require_failed=True,
    )
    if validated_parent["assignment_sha256"] != expected_parent_assignment_sha256:
        raise RuntimeError("parent assignment semantic SHA-256 disagrees")

    request = build_run_request(
        source_revision=revision,
        python_runtime={
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        parent_completion_path=parent_completion_path,
        expected_parent_completion_file_sha256=expected_parent_completion_file_sha256,
        expected_parent_completion_sha256=expected_parent_completion_sha256,
        parent_assignment_path=parent_assignment_path,
        expected_parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
        expected_parent_assignment_sha256=expected_parent_assignment_sha256,
        output_prefix=output_prefix,
    )
    prefix_root = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    output_root = prefix_root / request["run_identity_sha256"]

    refinement = refinement_module.refine_split_assignment(
        validated_parent,
        parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
    )
    refinement_module.validate_refinement_artifact(
        refinement,
        parent_assignment=validated_parent,
        parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
    )
    refined_assignment = refinement_module.validate_split_assignment_v2(
        refinement["refined_assignment"],
        require_passing=True,
    )
    if not all(refined_assignment["gate_results"].values()):
        raise RuntimeError("refined assignment did not pass every frozen split gate")

    def publish(path: Path, value: object) -> bool:
        created = _write_immutable_json(path, value)
        if created and stage_commit is not None:
            stage_commit()
        return created

    request_path = output_root / REQUEST_FILENAME
    refinement_path = output_root / REFINEMENT_FILENAME
    refined_assignment_path = output_root / REFINED_ASSIGNMENT_FILENAME
    created = [
        publish(request_path, request),
        publish(refinement_path, refinement),
        publish(refined_assignment_path, refined_assignment),
    ]

    # Reopen and independently replay every published semantic artifact before
    # recording completion. This makes a partial run safely resumable.
    if _load_mapping(request_path, field="published request") != request:
        raise RuntimeError("published split-refinement request disagrees")
    published_refinement = _load_mapping(refinement_path, field="published refinement")
    refinement_module.validate_refinement_artifact(
        published_refinement,
        parent_assignment=validated_parent,
        parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
    )
    published_assignment = _load_mapping(
        refined_assignment_path,
        field="published refined assignment",
    )
    refinement_module.validate_split_assignment_v2(
        published_assignment,
        require_passing=True,
    )
    if published_refinement["refined_assignment"] != published_assignment:
        raise RuntimeError("published refinement and refined assignment disagree")

    parent_completion_record = _artifact_record(
        completion_path,
        artifact_root=artifact_root,
        semantic_field="completion_sha256",
        semantic_sha256=parent_completion["completion_sha256"],
    )
    parent_assignment_record = _artifact_record(
        assignment_path,
        artifact_root=artifact_root,
        semantic_field="assignment_sha256",
        semantic_sha256=validated_parent["assignment_sha256"],
    )
    refinement_record = _artifact_record(
        refinement_path,
        artifact_root=artifact_root,
        semantic_field="refinement_sha256",
        semantic_sha256=published_refinement["refinement_sha256"],
    )
    refined_assignment_record = {
        **_artifact_record(
            refined_assignment_path,
            artifact_root=artifact_root,
            semantic_field="assignment_sha256",
            semantic_sha256=published_assignment["assignment_sha256"],
        ),
        "candidate_resolution_stream_sha256": published_assignment[
            "candidate_resolution_stream_sha256"
        ],
        "all_frozen_split_gates_pass": True,
    }
    completion = build_completion(
        request=request,
        run_root=_artifact_address(output_root, artifact_root=artifact_root),
        parent_completion=parent_completion_record,
        parent_assignment=parent_assignment_record,
        refinement=refinement_record,
        refined_assignment=refined_assignment_record,
    )
    validate_completion(completion, expected_request=request)
    completion_path = output_root / COMPLETION_FILENAME
    completion_created = publish(completion_path, completion)
    published_completion = validate_completion(
        _load_mapping(completion_path, field="published split-refinement completion"),
        expected_request=request,
    )
    return {
        "completion": published_completion,
        "run_root": _artifact_address(output_root, artifact_root=artifact_root),
        "refinement_sha256": published_refinement["refinement_sha256"],
        "refined_assignment_sha256": published_assignment["assignment_sha256"],
        "moves": len(published_refinement["moves"]),
        "all_frozen_split_gates_pass": True,
        "reused": not any(created) and not completion_created,
        "training_launched": False,
        "role_lane_materialization_launched": False,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=4 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def refine_split_assignment_remote(
    *,
    source_revision: dict[str, object],
    parent_completion_path: str,
    expected_parent_completion_file_sha256: str,
    expected_parent_completion_sha256: str,
    parent_assignment_path: str,
    expected_parent_assignment_file_sha256: str,
    expected_parent_assignment_sha256: str,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    return _refine_impl(
        source_revision=source_revision,
        parent_completion_path=parent_completion_path,
        expected_parent_completion_file_sha256=expected_parent_completion_file_sha256,
        expected_parent_completion_sha256=expected_parent_completion_sha256,
        parent_assignment_path=parent_assignment_path,
        expected_parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
        expected_parent_assignment_sha256=expected_parent_assignment_sha256,
        output_prefix=output_prefix,
        stage_commit=artifact_volume.commit,
    )


@app.local_entrypoint()
def main(
    commit: str,
    parent_completion_path: str,
    expected_parent_completion_file_sha256: str,
    expected_parent_completion_sha256: str,
    parent_assignment_path: str,
    expected_parent_assignment_file_sha256: str,
    expected_parent_assignment_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=commit)
    result = refine_split_assignment_remote.remote(
        source_revision=source_revision,
        parent_completion_path=parent_completion_path,
        expected_parent_completion_file_sha256=expected_parent_completion_file_sha256,
        expected_parent_completion_sha256=expected_parent_completion_sha256,
        parent_assignment_path=parent_assignment_path,
        expected_parent_assignment_file_sha256=expected_parent_assignment_file_sha256,
        expected_parent_assignment_sha256=expected_parent_assignment_sha256,
        output_prefix=output_prefix,
    )
    print(
        json.dumps(
            {
                "phase": "editing_v2_split_assignment_refinement_complete",
                "source_revision": source_revision,
                **result,
                "cpu_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
