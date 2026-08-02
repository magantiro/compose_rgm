"""Run exact semantic Editing-V2 Gate 0 structural evidence on Modal.

This CPU-only surface binds the exact semantic migration, chunk-cache, and
Active8-decision receipts to a clean committed source revision.  It delegates
the scientific computation to the production Gate 0 implementation and
publishes only structural evidence with no downstream authority.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_editing_v2_semantic_gate_zero_app.py"
MIGRATION_LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_corpus_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
IMAGE_SOURCE_FILES = (LAUNCHER_SOURCE, MIGRATION_LAUNCHER_SOURCE)
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_gate_zero_structural"
REQUEST_FILENAME = "GATE_ZERO_REQUEST.json"

SOURCE_REVISION_SCHEMA = "compose.editing.semantic_gate_zero_modal_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
REQUEST_SCHEMA = "compose.editing.semantic_gate_zero_modal_request"
REQUEST_SCHEMA_VERSION = 1
REQUEST_STATUS = "FROZEN_STRUCTURAL_EXECUTION_REQUEST_NO_DOWNSTREAM_AUTHORITY"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_INPUT_NAMES = (
    "migration_completion",
    "chunk_cache_plan",
    "chunk_cache_global_completion",
    "decision_plan",
    "decision_completion",
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
            "OMP_NUM_THREADS": "8",
        }
    )
)
for source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / source_directory,
        str(REMOTE_ROOT / source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
for source_file in IMAGE_SOURCE_FILES:
    image = image.add_local_file(
        ROOT / source_file,
        str(REMOTE_ROOT / source_file),
        copy=True,
    )

app = modal.App("compose-v4-editing-v2-semantic-gate-zero")
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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git commit")
    return value


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise ValueError(f"{field} must be a normalized path below /artifacts")
    root = artifact_root.resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} resolves outside artifact_root")
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


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
            f"cannot establish Gate 0 Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = list(IMAGE_SOURCE_FILES)
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "compose_v4").rglob("*.py"))
        if path.is_file()
    )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "configs").rglob("*.json"))
        if path.is_file()
    )
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {
        relative: _file_sha256(root / relative)
        for relative in _serialized_source_paths(root)
    }


def local_source_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind the exact clean tree and every file serialized into the image."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("Gate 0 requires the exact clean committed worktree")
    hashes = _serialized_source_hashes(root)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _canonical_sha256(hashes),
    }
    return {**body, "source_revision_sha256": _canonical_sha256(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Gate 0 source revision must be an object")
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
    live_hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or revision.get("serialized_source_hashes") != live_hashes
        or revision.get("serialized_source_hashes_sha256")
        != _canonical_sha256(live_hashes)
        or revision.get("source_revision_sha256") != _canonical_sha256(body)
    ):
        raise RuntimeError("Gate 0 serialized source revision disagrees")
    _require_commit(revision.get("commit"), field="source_revision.commit")
    _require_commit(revision.get("tree"), field="source_revision.tree")
    return revision


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, str]],
    contract_file_sha256: str,
    contract_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Content-address one nonauthorizing structural execution request."""

    if set(input_records) != set(_INPUT_NAMES):
        raise ValueError("Gate 0 input inventory disagrees")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for name in _INPUT_NAMES:
        record = input_records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "artifact_path",
            "file_sha256",
        }:
            raise ValueError(f"Gate 0 input {name} fields disagree")
        artifact_path = str(record["artifact_path"])
        _artifact_path(artifact_path, artifact_root=ARTIFACT_ROOT, field=name)
        normalized_inputs[name] = {
            "artifact_path": artifact_path,
            "file_sha256": _require_sha256(
                record["file_sha256"], field=f"{name}.file_sha256"
            ),
        }
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    body: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": REQUEST_STATUS,
        **_NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "inputs": normalized_inputs,
        "contract_file_sha256": _require_sha256(
            contract_file_sha256, field="contract_file_sha256"
        ),
        "contract_sha256": _require_sha256(contract_sha256, field="contract_sha256"),
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments import editing_v2_semantic_gate_zero as gate_zero

    return {
        "gate_zero": gate_zero,
        "write_bytes_if_absent": write_bytes_if_absent,
    }


def _driver_impl(
    *,
    source_revision: Mapping[str, Any],
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Any | None = None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    gate_zero = loaded["gate_zero"]
    supplied_paths = {
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
    }
    mounted = {
        name: _artifact_path(value, artifact_root=artifact_root, field=name)
        for name, value in supplied_paths.items()
    }
    contract = gate_zero.load_semantic_gate_zero_structural_contract(
        repo_root=remote_root
    )
    request = build_run_request(
        source_revision=revision,
        input_records={
            name: {
                "artifact_path": supplied_paths[name],
                "file_sha256": _file_sha256(path),
            }
            for name, path in mounted.items()
        },
        contract_file_sha256=contract.file_sha256,
        contract_sha256=contract.sha256,
        output_prefix=output_prefix,
    )
    prefix_parent = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    output_directory = prefix_parent / request["run_identity_sha256"]
    request_path = output_directory / REQUEST_FILENAME
    created = loaded["write_bytes_if_absent"](
        request_path, _canonical_bytes(request, pretty=True)
    )
    if created and stage_commit is not None:
        stage_commit()
    result = gate_zero.run_semantic_gate_zero_structural_evidence(
        migration_completion_path=mounted["migration_completion"],
        chunk_cache_plan_path=mounted["chunk_cache_plan"],
        chunk_cache_global_completion_path=mounted["chunk_cache_global_completion"],
        decision_plan_path=mounted["decision_plan"],
        decision_completion_path=mounted["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
        output_directory=output_directory,
    )
    if stage_commit is not None:
        stage_commit()
    published_paths = {
        "evidence": output_directory / gate_zero.EVIDENCE_FILENAME,
        "decision": output_directory / gate_zero.DECISION_FILENAME,
        "completion": output_directory / gate_zero.COMPLETION_FILENAME,
    }
    for name, path in published_paths.items():
        raw = path.read_bytes()
        if raw != _canonical_bytes(result[name]) + b"\n":
            raise RuntimeError(f"published Gate 0 {name} differs from returned evidence")
    return {
        "run_root": _artifact_address(output_directory, artifact_root=artifact_root),
        "request_artifact_path": _artifact_address(
            request_path, artifact_root=artifact_root
        ),
        "request_file_sha256": _file_sha256(request_path),
        "run_identity_sha256": request["run_identity_sha256"],
        "structural_result": result["completion"]["structural_result"],
        "completion": result["completion"],
        "completion_file_sha256": _file_sha256(published_paths["completion"]),
        "training_launched": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_gate_zero(
    *,
    source_revision: dict[str, object],
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _driver_impl(
        source_revision=source_revision,
        migration_completion=migration_completion,
        chunk_cache_plan=chunk_cache_plan,
        chunk_cache_global_completion=chunk_cache_global_completion,
        decision_plan=decision_plan,
        decision_completion=decision_completion,
        output_prefix=output_prefix,
        stage_commit=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result


@app.local_entrypoint()
def main(
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    expected_commit: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = run_gate_zero.remote(
        source_revision=source_revision,
        migration_completion=migration_completion,
        chunk_cache_plan=chunk_cache_plan,
        chunk_cache_global_completion=chunk_cache_global_completion,
        decision_plan=decision_plan,
        decision_completion=decision_completion,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "OUTPUT_PREFIX",
    "REQUEST_FILENAME",
    "app",
    "build_run_request",
    "local_source_revision",
    "main",
    "run_gate_zero",
]
