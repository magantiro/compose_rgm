"""Materialize the frozen Editing V2 candidate-provenance bridge on Modal.

This CPU-only job consumes one exact packed-candidate materialization and one
prospectively frozen provenance-decisions artifact.  It validates the large
candidate derivative exactly once per invocation, reuses that validated value
through registry construction and bridge validation/materialization, and
publishes into an immutable content-addressed namespace.

The resulting artifacts are corpus derivatives.  They grant no split, Gate 0,
P50, or training authority.
"""

from __future__ import annotations

import hashlib
import json
import os
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

DEFAULT_CANDIDATE_ROOT = (
    "/artifacts/editing_v2/packed_candidates/"
    "2f29883dafe6b2b7fd812c34a1a99d8c89270ce729387d37abc1111193f56c13/"
    "candidate_materialization"
)
DEFAULT_DECISIONS_PATH = "/root/compose/configs/editing_v2_candidate_provenance_decisions_v1.json"
DEFAULT_CORPUS_CONTRACT_PATH = "/root/compose/configs/editing_corpus_v2_contract.json"
OUTPUT_PREFIX = "/artifacts/editing_v2/candidate_provenance"

RUN_SCHEMA = "compose.editing_v2_candidate_provenance_modal_run"
RUN_SCHEMA_VERSION = 1
RUN_STATUS = "COMPLETE_CANDIDATE_PROVENANCE_NO_TRAINING_AUTHORITY"
SOURCE_REVISION_SCHEMA = "compose.editing_v2_candidate_provenance_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
COMPLETION_FILENAME = "CANDIDATE_PROVENANCE_COMPLETE.json"
BRIDGE_DIRECTORY = "candidate_provenance_bridge"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SERIALIZED_SOURCE_FILES = (
    "modal_apps/materialize_editing_v2_candidate_provenance_app.py",
    "src/compose_v4/data/editing_candidate_audit_ledger.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_decisions.py",
    "src/compose_v4/data/editing_v2_packed_candidate_materializer.py",
    "src/compose_v4/data/editing_v2_split_census.py",
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
        ROOT / "modal_apps" / "materialize_editing_v2_candidate_provenance_app.py",
        str(REMOTE_ROOT / "modal_apps/materialize_editing_v2_candidate_provenance_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-candidate-provenance")
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


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SERIALIZED_SOURCE_FILES:
        source = Path(root) / relative
        if not source.is_file():
            raise RuntimeError(f"serialized provenance source is absent: {source}")
        hashes[relative] = _file_sha256(source)
    return hashes


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
            f"cannot establish candidate-provenance Git identity: git {' '.join(arguments)}"
        ) from error


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind one clean full Git revision and the exact serialized sources."""

    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("candidate provenance requires a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "candidate provenance requires the exact clean committed serialized worktree"
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
        raise RuntimeError("candidate provenance source revision must be an object")
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
        raise RuntimeError("candidate provenance serialized source revision disagrees")
    return revision


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    candidate_root: str,
    candidate_identity: Mapping[str, Any],
    decisions_path: str,
    decisions_file_sha256: str,
    decisions_sha256: str,
    decisions_registry_inputs_sha256: str,
    editing_corpus_contract_path: str,
    editing_corpus_contract_file_sha256: str,
    registry_sha256: str,
    max_row_bytes: int,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Name one immutable provenance-bridge run from all material inputs."""

    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise ValueError("max_row_bytes must be a positive integer")
    _require_sha256(decisions_file_sha256, field="decisions_file_sha256")
    _require_sha256(decisions_sha256, field="decisions_sha256")
    _require_sha256(
        decisions_registry_inputs_sha256,
        field="decisions_registry_inputs_sha256",
    )
    _require_sha256(
        editing_corpus_contract_file_sha256,
        field="editing_corpus_contract_file_sha256",
    )
    _require_sha256(registry_sha256, field="registry_sha256")
    candidate = dict(candidate_identity)
    if not candidate or any(
        _SHA256_RE.fullmatch(str(candidate.get(field))) is None
        for field in (
            "manifest_file_sha256",
            "manifest_sha256",
            "rows_file_sha256",
            "rows_semantic_sha256",
            "address_stream_sha256",
        )
    ):
        raise ValueError("candidate_identity must contain the exact five SHA-256 bindings")
    body: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "source_revision": dict(source_revision),
        "candidate_root": candidate_root,
        "candidate_identity": candidate,
        "decisions_path": decisions_path,
        "decisions_file_sha256": decisions_file_sha256,
        "decisions_sha256": decisions_sha256,
        "decisions_registry_inputs_sha256": decisions_registry_inputs_sha256,
        "editing_corpus_contract_path": editing_corpus_contract_path,
        "editing_corpus_contract_file_sha256": editing_corpus_contract_file_sha256,
        "registry_sha256": registry_sha256,
        "max_row_bytes": max_row_bytes,
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def _candidate_identity(candidate_root: Path, materialization: Mapping[str, Any]) -> dict[str, str]:
    rows = materialization.get("rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError("candidate materialization rows identity is absent")
    return {
        "manifest_file_sha256": _file_sha256(
            candidate_root / "PACKED_CANDIDATE_MATERIALIZATION.json"
        ),
        "manifest_sha256": _require_sha256(
            materialization.get("manifest_sha256"), field="candidate manifest_sha256"
        ),
        "rows_file_sha256": _require_sha256(
            rows.get("file_sha256"), field="candidate rows_file_sha256"
        ),
        "rows_semantic_sha256": _require_sha256(
            rows.get("semantic_sha256"), field="candidate rows_semantic_sha256"
        ),
        "address_stream_sha256": _require_sha256(
            rows.get("address_stream_sha256"), field="candidate address_stream_sha256"
        ),
    }


def _validate_decisions_candidate_binding(
    decisions: object,
    *,
    candidate_root_address: str,
    candidate_identity: Mapping[str, str],
    materialization_filename: str,
    contract_file_sha256: str,
) -> None:
    """Fail closed if the decisions unit names another candidate derivative."""

    candidate_artifact = getattr(decisions, "candidate_artifact", None)
    declared = getattr(candidate_artifact, "candidate_materialization", None)
    declared_contract = getattr(candidate_artifact, "editing_corpus_contract", None)
    expected_path = f"{candidate_root_address}/{materialization_filename}"
    if (
        getattr(decisions, "training_authorized", None) is not False
        or getattr(declared, "path", None) != expected_path
        or any(
            getattr(declared, field, None) != candidate_identity[field]
            for field in candidate_identity
        )
        or getattr(declared_contract, "file_sha256", None) != contract_file_sha256
    ):
        raise RuntimeError(
            "candidate provenance decisions disagree with the exact candidate or contract"
        )


def _write_immutable_json(path: Path, value: object) -> None:
    encoded = _canonical_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable candidate-provenance collision at {path}")
        return
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
                raise RuntimeError(f"immutable candidate-provenance collision at {path}")
        else:
            os.replace(temporary_name, path)
            temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _imports(remote_root: Path) -> dict[str, Any]:
    import sys

    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data import editing_v2_candidate_provenance_bridge as bridge
    from compose_v4.data import editing_v2_packed_candidate_materializer as materializer
    from compose_v4.data.editing_v2_candidate_provenance_decisions import (
        load_candidate_provenance_decisions,
    )

    return {
        "bridge": bridge,
        "load_candidate_provenance_decisions": load_candidate_provenance_decisions,
        "materializer": materializer,
    }


def _materialize_impl(
    *,
    source_revision: Mapping[str, Any],
    candidate_root: str,
    decisions_path: str,
    editing_corpus_contract_path: str,
    max_row_bytes: int,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
) -> dict[str, Any]:
    """Build or strictly reopen one exact provenance derivative."""

    revision = _validate_source_revision(source_revision, serialized_root=remote_root)
    candidate_path = _artifact_path(
        candidate_root,
        artifact_root=artifact_root,
        field="candidate_root",
    )
    decisions_file = _remote_project_path(
        decisions_path,
        remote_root=remote_root,
        field="decisions_path",
    )
    contract_file = _remote_project_path(
        editing_corpus_contract_path,
        remote_root=remote_root,
        field="editing_corpus_contract_path",
    )
    output_root = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    loaded = _imports(remote_root)
    bridge = loaded["bridge"]
    materializer = loaded["materializer"]

    # This is deliberately the sole packed-candidate validator invocation in
    # the launcher.  Every downstream bridge call receives this exact value.
    validated_candidate = materializer.validate_packed_candidate_materialization(
        candidate_path,
        max_row_bytes=max_row_bytes,
    )
    decisions = loaded["load_candidate_provenance_decisions"](decisions_file)
    registry_inputs = decisions.registry_inputs()
    candidate_identity = _candidate_identity(candidate_path, validated_candidate)
    contract_file_sha256 = _file_sha256(contract_file)
    _validate_decisions_candidate_binding(
        decisions,
        candidate_root_address=candidate_root,
        candidate_identity=candidate_identity,
        materialization_filename=materializer.MATERIALIZATION_FILENAME,
        contract_file_sha256=contract_file_sha256,
    )
    registry = bridge.build_candidate_provenance_registry(
        candidate_root=candidate_path,
        editing_corpus_contract_path=contract_file,
        _validated_candidate_materialization=validated_candidate,
        **registry_inputs,
    )
    request = build_run_request(
        source_revision=revision,
        candidate_root=candidate_root,
        candidate_identity=candidate_identity,
        decisions_path=decisions_path,
        decisions_file_sha256=_file_sha256(decisions_file),
        decisions_sha256=decisions.decision_sha256,
        decisions_registry_inputs_sha256=_canonical_sha256(registry_inputs),
        editing_corpus_contract_path=editing_corpus_contract_path,
        editing_corpus_contract_file_sha256=contract_file_sha256,
        registry_sha256=registry["registry_sha256"],
        max_row_bytes=max_row_bytes,
        output_prefix=output_prefix,
    )
    run_root = output_root / request["run_identity_sha256"]
    registry_path = run_root / bridge.REGISTRY_FILENAME
    bridge_root = run_root / BRIDGE_DIRECTORY
    completion_path = run_root / COMPLETION_FILENAME
    run_root.mkdir(parents=True, exist_ok=True)

    expected_registry = _canonical_bytes(registry, pretty=True)
    if registry_path.exists():
        if registry_path.read_bytes() != expected_registry:
            raise RuntimeError(f"immutable candidate-provenance collision at {registry_path}")
    else:
        bridge.write_candidate_provenance_registry(registry, registry_path)

    if bridge_root.exists():
        bridge_manifest = bridge.validate_candidate_provenance_bridge(
            bridge_root,
            candidate_root=candidate_path,
            provenance_registry_path=registry_path,
            editing_corpus_contract_path=contract_file,
            max_row_bytes=max_row_bytes,
            _validated_candidate_materialization=validated_candidate,
        )
        reused = True
    else:
        bridge_manifest = bridge.materialize_candidate_provenance_bridge(
            candidate_root=candidate_path,
            provenance_registry_path=registry_path,
            editing_corpus_contract_path=contract_file,
            output_dir=bridge_root,
            max_row_bytes=max_row_bytes,
            _validated_candidate_materialization=validated_candidate,
        )
        reused = False
    if bridge_manifest.get("training_authorized") is not False:
        raise RuntimeError("candidate provenance bridge unexpectedly grants training authority")
    bridge_manifest_path = bridge_root / bridge.BRIDGE_MANIFEST_FILENAME
    completion_body = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "status": RUN_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "request": request,
        "run_root": f"{output_prefix}/{request['run_identity_sha256']}",
        "registry": {
            "relative_path": bridge.REGISTRY_FILENAME,
            "file_sha256": _file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "bridge": {
            "relative_path": f"{BRIDGE_DIRECTORY}/{bridge.BRIDGE_MANIFEST_FILENAME}",
            "file_sha256": _file_sha256(bridge_manifest_path),
            "manifest_sha256": bridge_manifest["manifest_sha256"],
            "counts": bridge_manifest["counts"],
            "blockers": bridge_manifest["blockers"],
        },
    }
    completion = {
        **completion_body,
        "completion_sha256": _canonical_sha256(completion_body),
    }
    _write_immutable_json(completion_path, completion)
    return {"completion": completion, "reused": reused}


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=12 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_candidate_provenance(
    *,
    source_revision: dict[str, object],
    candidate_root: str,
    decisions_path: str,
    editing_corpus_contract_path: str,
    max_row_bytes: int,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _materialize_impl(
        source_revision=source_revision,
        candidate_root=candidate_root,
        decisions_path=decisions_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        max_row_bytes=max_row_bytes,
        output_prefix=output_prefix,
    )
    artifact_volume.commit()
    return result


@app.local_entrypoint()
def main(
    commit: str,
    candidate_root: str = DEFAULT_CANDIDATE_ROOT,
    decisions_path: str = DEFAULT_DECISIONS_PATH,
    editing_corpus_contract_path: str = DEFAULT_CORPUS_CONTRACT_PATH,
    max_row_bytes: int = 2 * 1024 * 1024,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=commit)
    result = materialize_candidate_provenance.remote(
        source_revision=source_revision,
        candidate_root=candidate_root,
        decisions_path=decisions_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        max_row_bytes=max_row_bytes,
        output_prefix=output_prefix,
    )
    print(
        json.dumps(
            {
                "phase": "candidate_provenance_complete",
                "source_revision": source_revision,
                **result,
                "training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
