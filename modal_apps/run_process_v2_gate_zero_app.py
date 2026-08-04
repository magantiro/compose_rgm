"""Run the Process-V2 Gate 0 reducer against one completed Active8 run.

This is intentionally a thin, single-container surface.  Active8 owns the
expensive molecular pass and its release sentinel.  This app authenticates that
completion, invokes the existing deterministic Gate 0 reducer, and commits one
content-addressed request/decision pair.  A structural FAIL is a valid negative
result; an incomplete or malformed Active8 run publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_active8_reduce import (  # noqa: E402
    COMPLETION_FILENAME,
    load_process_v2_active8_completion,
)
from compose_v4.data.editing_v2_process_v2_gate_zero import (  # noqa: E402
    load_gate_zero_contracts,
    run_gate_zero,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (  # noqa: E402
    GATE_ZERO_DECISION_FILENAME,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (  # noqa: E402
    AUTHORITY_FIELDS,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_no_granted_authority,
    verify_self_hash,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_gate_zero_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_gate_zero"
REQUEST_FILENAME = "PROCESS_V2_GATE_ZERO_REQUEST.json"

REQUEST_SCHEMA = "compose.data.editing_v2_process_v2_gate_zero_request"
REQUEST_SCHEMA_VERSION = 1
IMAGE_REVISION_SCHEMA = "compose.data.process_v2_gate_zero_modal_image_revision"
IMAGE_REVISION_SCHEMA_VERSION = 1

GATE_ZERO_CPU = 1.0
GATE_ZERO_MEMORY_MB = 8 * 1024
GATE_ZERO_TIMEOUT_SECONDS = 60 * 60
GATE_ZERO_MAX_CONTAINERS = 1

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
)
for _source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / _source_directory,
        str(REMOTE_ROOT / _source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-process-v2-gate-zero")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("the Gate 0 serialized source inventory repeats a path")
    return tuple(paths)


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
            f"cannot establish Process-V2 Gate 0 Git identity: git {' '.join(arguments)}"
        ) from error


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind the exact clean commit and every byte serialized into the image."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "Process-V2 Gate 0 requires the exact clean committed worktree"
        )
    sources = {
        relative: _file_sha256(root / relative)
        for relative in _serialized_source_paths(root)
    }
    body = {
        "schema": IMAGE_REVISION_SCHEMA,
        "schema_version": IMAGE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": canonical_sha256(body)}


def _validate_remote_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    revision = dict(value)
    body = dict(revision)
    supplied = body.pop("image_revision_sha256", None)
    sources = revision.get("serialized_sources")
    expected = _serialized_source_paths(REMOTE_ROOT)
    if (
        revision.get("schema") != IMAGE_REVISION_SCHEMA
        or revision.get("schema_version") != IMAGE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or not isinstance(sources, Mapping)
        or set(sources) != set(expected)
        or supplied != canonical_sha256(body)
    ):
        raise RuntimeError("the Process-V2 Gate 0 image revision disagrees")
    for relative, digest in sources.items():
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(f"the serialized source digest is malformed: {relative}")
        if _file_sha256(REMOTE_ROOT / relative) != digest:
            raise RuntimeError(f"a serialized Gate 0 source differs remotely: {relative}")
    return revision


def _artifact_path(value: str, *, field: str) -> Path:
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
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    resolved = (ARTIFACT_ROOT / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise RuntimeError(f"{field} resolves outside /artifacts")
    return resolved


def build_run_request(
    *,
    image_revision: Mapping[str, Any],
    active8_run_root: str,
    active8_completion: Mapping[str, Any],
    active8_completion_file_sha256: str,
    contracts: Any,
    output_artifact_prefix: str,
) -> dict[str, Any]:
    """Content-address one Gate 0 execution from its complete evidence."""

    _artifact_path(active8_run_root, field="active8_run_root")
    _artifact_path(f"{output_artifact_prefix}/placeholder", field="output_artifact_prefix")
    image_sha = image_revision.get("image_revision_sha256")
    completion_sha = active8_completion.get("completion_sha256")
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in (image_sha, completion_sha, active8_completion_file_sha256)
    ):
        raise RuntimeError("the Gate 0 request carries a malformed SHA-256")
    body: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "image_revision_sha256": image_sha,
        "active8_run_artifact_root": active8_run_root,
        "active8_completion_sha256": completion_sha,
        "active8_completion_file_sha256": active8_completion_file_sha256,
        "gate_zero_structural_contract_sha256": contracts.contract_sha256,
        "contracts_binding_sha256": contracts.binding_sha256,
        "process_identity_sha256": contracts.process_identity_sha256,
        "output_artifact_prefix": output_artifact_prefix,
    }
    run_identity = canonical_sha256(body)
    request_body = {
        **body,
        "run_identity_sha256": run_identity,
        "run_artifact_root": f"{output_artifact_prefix}/{run_identity}",
    }
    return {**request_body, "request_sha256": canonical_sha256(request_body)}


def validate_run_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *AUTHORITY_FIELDS,
        "image_revision_sha256",
        "active8_run_artifact_root",
        "active8_completion_sha256",
        "active8_completion_file_sha256",
        "gate_zero_structural_contract_sha256",
        "contracts_binding_sha256",
        "process_identity_sha256",
        "output_artifact_prefix",
        "run_identity_sha256",
        "run_artifact_root",
        "request_sha256",
    }
    if set(request) != expected_fields:
        raise RuntimeError("the Gate 0 request field set disagrees")
    verify_self_hash(request, field="request_sha256", label="the Gate 0 request")
    require_no_granted_authority(request, label="the Gate 0 request")
    run_body = {
        key: request[key]
        for key in request
        if key not in {"run_identity_sha256", "run_artifact_root", "request_sha256"}
    }
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("status") != PIPELINE_STATUS_NO_AUTHORITY
        or request.get("run_identity_sha256") != canonical_sha256(run_body)
        or request.get("run_artifact_root")
        != f"{request.get('output_artifact_prefix')}/{request.get('run_identity_sha256')}"
    ):
        raise RuntimeError("the Gate 0 request identity disagrees")
    for field in (
        "image_revision_sha256",
        "active8_completion_sha256",
        "active8_completion_file_sha256",
        "gate_zero_structural_contract_sha256",
        "contracts_binding_sha256",
        "process_identity_sha256",
        "run_identity_sha256",
        "request_sha256",
    ):
        value = request[field]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"the Gate 0 request carries a malformed {field}")
    _artifact_path(str(request["active8_run_artifact_root"]), field="active8_run_root")
    _artifact_path(
        f"{request['output_artifact_prefix']}/placeholder",
        field="output_artifact_prefix",
    )
    _artifact_path(str(request["run_artifact_root"]), field="run_artifact_root")
    return request


def _publish_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    content = canonical_bytes(value) + b"\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise RuntimeError(f"immutable Gate 0 collision at {target}")
        return True
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return False


def _validate_decision(
    decision: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(decision)
    verify_self_hash(result, field="decision_sha256", label="the Gate 0 decision")
    require_no_granted_authority(result, label="the Gate 0 decision")
    for request_field, decision_field in (
        ("active8_completion_sha256", "active8_completion_sha256"),
        ("gate_zero_structural_contract_sha256", "gate_zero_structural_contract_sha256"),
        ("contracts_binding_sha256", "contracts_binding_sha256"),
        ("process_identity_sha256", "process_identity_sha256"),
    ):
        if result.get(decision_field) != request.get(request_field):
            raise RuntimeError(f"the Gate 0 decision disagrees through {decision_field}")
    if result.get("decision") not in {"PASS", "FAIL"}:
        raise RuntimeError("the Gate 0 decision outcome is malformed")
    return result


@app.function(
    image=image,
    cpu=GATE_ZERO_CPU,
    memory=GATE_ZERO_MEMORY_MB,
    timeout=GATE_ZERO_TIMEOUT_SECONDS,
    max_containers=GATE_ZERO_MAX_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def execute_gate_zero(
    active8_run_root: str,
    output_artifact_prefix: str,
    image_revision: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate Active8, execute the core reducer, and commit once."""

    revision = _validate_remote_revision(image_revision)
    artifact_volume.reload()
    completion = load_process_v2_active8_completion(
        active8_run_root,
        artifact_root=ARTIFACT_ROOT,
    )
    active8_path = _artifact_path(active8_run_root, field="active8_run_root")
    completion_file_sha256 = _file_sha256(active8_path / COMPLETION_FILENAME)
    contracts = load_gate_zero_contracts(repo_root=REMOTE_ROOT)
    request = validate_run_request(
        build_run_request(
            image_revision=revision,
            active8_run_root=active8_run_root,
            active8_completion=completion,
            active8_completion_file_sha256=completion_file_sha256,
            contracts=contracts,
            output_artifact_prefix=output_artifact_prefix,
        )
    )
    output_root = _artifact_path(str(request["run_artifact_root"]), field="run_artifact_root")
    request_path = output_root / REQUEST_FILENAME
    decision_path = output_root / GATE_ZERO_DECISION_FILENAME
    if request_path.exists() or decision_path.exists():
        if not request_path.is_file() or not decision_path.is_file():
            raise RuntimeError("a partial Gate 0 generation already exists")
        prior_request = validate_run_request(json.loads(request_path.read_bytes()))
        if prior_request != request:
            raise RuntimeError("the existing Gate 0 request disagrees")
        decision = _validate_decision(
            json.loads(decision_path.read_bytes()),
            request=request,
        )
        return {
            "phase": "process_v2_gate_zero_reused",
            "run_artifact_root": request["run_artifact_root"],
            "request_sha256": request["request_sha256"],
            "decision": decision["decision"],
            "decision_sha256": decision["decision_sha256"],
            **authority_false_block(),
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            dir=output_root.parent,
            prefix=f".{output_root.name}.staging.",
        )
    )
    try:
        staging_request_path = staging_root / REQUEST_FILENAME
        staging_decision_path = staging_root / GATE_ZERO_DECISION_FILENAME
        _publish_immutable(staging_request_path, request)
        returned = _validate_decision(
            run_gate_zero(
                active8_path,
                gate_zero_root=staging_root,
                repo_root=REMOTE_ROOT,
            ),
            request=request,
        )
        if not staging_decision_path.is_file():
            raise RuntimeError("Gate 0 returned without publishing its decision")
        decision = _validate_decision(
            json.loads(staging_decision_path.read_bytes()),
            request=request,
        )
        if decision != returned:
            raise RuntimeError("the returned and published Gate 0 decisions disagree")
        os.replace(staging_root, output_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    artifact_volume.commit()
    return {
        "phase": "process_v2_gate_zero_complete",
        "run_artifact_root": request["run_artifact_root"],
        "request_sha256": request["request_sha256"],
        "decision": decision["decision"],
        "decision_sha256": decision["decision_sha256"],
        **authority_false_block(),
    }


@app.local_entrypoint()
def main(
    active8_run_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
) -> None:
    """Spawn one disconnect-safe Gate 0 execution and print its call identity."""

    revision = local_image_revision(expected_commit=expected_commit)
    call = execute_gate_zero.spawn(
        active8_run_root,
        output_artifact_prefix,
        revision,
    )
    print(
        json.dumps(
            {
                "phase": "process_v2_gate_zero_launched",
                "function_call_id": call.object_id,
                "active8_run_root": active8_run_root,
                "image_revision": revision,
                **authority_false_block(),
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "GATE_ZERO_CPU",
    "GATE_ZERO_MAX_CONTAINERS",
    "GATE_ZERO_MEMORY_MB",
    "GATE_ZERO_TIMEOUT_SECONDS",
    "OUTPUT_ARTIFACT_PREFIX",
    "REQUEST_FILENAME",
    "build_run_request",
    "local_image_revision",
    "validate_run_request",
]
