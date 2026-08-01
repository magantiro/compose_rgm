"""Materialize exact Editing V2 candidate headers from frozen packed caches.

This CPU-only job consumes the immutable upstream-overlay completion and the
frozen layer-to-source registry. It builds the source manifest internally, so
the scientific launch cannot substitute a hand-assembled shard subset. Output
is content addressed, restart safe, and explicitly carries no training
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")

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
        ROOT / "modal_apps" / "materialize_editing_v2_candidates_app.py",
        str(REMOTE_ROOT / "modal_apps" / "materialize_editing_v2_candidates_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-candidates")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)

RUN_SCHEMA = "compose.editing_v2_candidate_materialization_run"
RUN_SCHEMA_VERSION = 1
RUN_STATUS = "COMPLETE_CANDIDATE_HEADERS_NO_TRAINING_AUTHORITY"
OUTPUT_PREFIX = "/artifacts/editing_v2/packed_candidates"
SOURCE_MANIFEST_FILENAME = "PACKED_CANDIDATE_SOURCES.json"
COMPLETION_FILENAME = "CANDIDATE_BUILD_COMPLETE.json"
SOURCE_BINDING_REGISTRY = "configs/editing_v2_overlay_source_bindings_v1.json"
ROUTING_POLICY = "configs/editing_v2_candidate_routing_policy_v1.json"
CORPUS_CONTRACT = "configs/editing_corpus_v2_contract.json"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_candidate_run_request(
    *,
    commit: str,
    overlay_completion_path: str,
    overlay_completion_file_sha256: str,
    source_binding_registry_file_sha256: str,
    source_manifest_file_sha256: str,
    source_manifest_sha256: str,
    routing_policy_file_sha256: str,
    routing_policy_sha256: str,
    corpus_contract_file_sha256: str,
    corpus_contract_sha256: str,
    materializer_source_sha256: str,
    launcher_source_sha256: str,
) -> dict[str, object]:
    """Build the exact immutable request that names one candidate derivative."""

    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be a full lowercase Git SHA")
    hashes = {
        "overlay_completion_file_sha256": overlay_completion_file_sha256,
        "source_binding_registry_file_sha256": source_binding_registry_file_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "routing_policy_file_sha256": routing_policy_file_sha256,
        "routing_policy_sha256": routing_policy_sha256,
        "corpus_contract_file_sha256": corpus_contract_file_sha256,
        "corpus_contract_sha256": corpus_contract_sha256,
        "materializer_source_sha256": materializer_source_sha256,
        "launcher_source_sha256": launcher_source_sha256,
    }
    for name, value in hashes.items():
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a full lowercase SHA-256")
    overlay_path = Path(overlay_completion_path)
    if (
        not overlay_path.is_absolute()
        or not overlay_path.is_relative_to(ARTIFACT_ROOT)
        or ".." in overlay_path.parts
    ):
        raise ValueError("overlay completion must be a normalized path below /artifacts")
    body = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "training_authorized": False,
        "commit": commit,
        "overlay_completion_path": str(overlay_path),
        "source_binding_registry_path": str(REMOTE_ROOT / SOURCE_BINDING_REGISTRY),
        "routing_policy_path": str(REMOTE_ROOT / ROUTING_POLICY),
        "corpus_contract_path": str(REMOTE_ROOT / CORPUS_CONTRACT),
        **hashes,
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def _require_clean_revision(expected_commit: str) -> None:
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("candidate materialization requires a full Git SHA")
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_commit or dirty:
        raise RuntimeError("candidate materialization requires the exact clean committed worktree")


def _write_immutable_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact collision at {path}")
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
                raise RuntimeError(f"immutable artifact collision at {path}")
        else:
            os.replace(temporary_name, path)
            temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=6 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_candidates(
    *,
    commit: str,
    overlay_completion_path: str,
    expected_overlay_completion_file_sha256: str,
    expected_source_binding_registry_file_sha256: str,
    expected_launcher_source_sha256: str,
) -> dict[str, object]:
    """Build or reopen one exact content-addressed candidate derivative."""

    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
    from compose_v4.data.editing_v2_candidate_router import (
        load_routing_policy,
        routing_policy_sha256,
    )
    from compose_v4.data import editing_v2_packed_candidate_materializer as materializer

    artifact_volume.reload()
    launcher_path = REMOTE_ROOT / "modal_apps" / "materialize_editing_v2_candidates_app.py"
    if _file_sha256(launcher_path) != expected_launcher_source_sha256:
        raise RuntimeError("serialized candidate launcher source identity disagrees")
    registry_path = REMOTE_ROOT / SOURCE_BINDING_REGISTRY
    routing_path = REMOTE_ROOT / ROUTING_POLICY
    contract_path = REMOTE_ROOT / CORPUS_CONTRACT
    if _file_sha256(registry_path) != expected_source_binding_registry_file_sha256:
        raise RuntimeError("serialized source-binding registry identity disagrees")

    source_manifest = materializer.build_packed_candidate_source_manifest_from_overlay_completion(
        overlay_completion_path=overlay_completion_path,
        expected_overlay_completion_file_sha256=(expected_overlay_completion_file_sha256),
        source_binding_registry_path=registry_path,
        expected_source_binding_registry_file_sha256=(expected_source_binding_registry_file_sha256),
    )
    source_manifest_bytes = (
        json.dumps(source_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    routing = load_routing_policy(routing_path)
    contract = load_editing_corpus_contract(contract_path)
    request = build_candidate_run_request(
        commit=commit,
        overlay_completion_path=overlay_completion_path,
        overlay_completion_file_sha256=expected_overlay_completion_file_sha256,
        source_binding_registry_file_sha256=expected_source_binding_registry_file_sha256,
        source_manifest_file_sha256=hashlib.sha256(source_manifest_bytes).hexdigest(),
        source_manifest_sha256=source_manifest["manifest_sha256"],
        routing_policy_file_sha256=_file_sha256(routing_path),
        routing_policy_sha256=routing_policy_sha256(routing),
        corpus_contract_file_sha256=_file_sha256(contract_path),
        corpus_contract_sha256=materializer.canonical_sha256(contract),
        materializer_source_sha256=_file_sha256(Path(materializer.__file__)),
        launcher_source_sha256=expected_launcher_source_sha256,
    )
    run_root = Path(OUTPUT_PREFIX) / str(request["run_identity_sha256"])
    source_manifest_path = run_root / SOURCE_MANIFEST_FILENAME
    candidate_root = run_root / "candidate_materialization"
    completion_path = run_root / COMPLETION_FILENAME
    _write_immutable_json(source_manifest_path, source_manifest)

    if candidate_root.exists():
        candidate = materializer.validate_packed_candidate_materialization(candidate_root)
    else:
        candidate = materializer.materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=ARTIFACT_ROOT,
            routing_policy_path=routing_path,
            editing_corpus_contract_path=contract_path,
            output_dir=candidate_root,
            code_revision=commit,
        )
    if (
        candidate["inputs"]["source_manifest"]
        != {
            "file_sha256": request["source_manifest_file_sha256"],
            "manifest_sha256": source_manifest["manifest_sha256"],
        }
        or candidate["inputs"]["routing_policy"]
        != {
            "file_sha256": request["routing_policy_file_sha256"],
            "semantic_sha256": request["routing_policy_sha256"],
        }
        or candidate["inputs"]["editing_corpus_contract"]["file_sha256"]
        != request["corpus_contract_file_sha256"]
        or candidate["inputs"]["editing_corpus_contract"]["semantic_sha256"]
        != request["corpus_contract_sha256"]
        or candidate["implementation"]["file_sha256"] != request["materializer_source_sha256"]
        or candidate["code_revision"] != commit
    ):
        raise RuntimeError("candidate materialization disagrees with its exact run request")
    completion_body = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "status": RUN_STATUS,
        "training_authorized": False,
        "request": request,
        "run_root": str(run_root),
        "source_manifest": {
            "path": str(source_manifest_path),
            "file_sha256": _file_sha256(source_manifest_path),
            "manifest_sha256": source_manifest["manifest_sha256"],
        },
        "candidate_materialization": {
            "path": str(candidate_root / materializer.MATERIALIZATION_FILENAME),
            "file_sha256": _file_sha256(candidate_root / materializer.MATERIALIZATION_FILENAME),
            "manifest_sha256": candidate["manifest_sha256"],
            "rows_file_sha256": candidate["rows"]["file_sha256"],
            "rows_semantic_sha256": candidate["rows"]["semantic_sha256"],
            "totals": candidate["rows"]["totals"],
        },
    }
    completion = {
        **completion_body,
        "completion_sha256": _canonical_sha256(completion_body),
    }
    _write_immutable_json(completion_path, completion)
    artifact_volume.commit()
    return completion


@app.local_entrypoint()
def main(
    commit: str,
    overlay_completion_path: str,
    overlay_completion_file_sha256: str,
) -> None:
    _require_clean_revision(commit)
    registry_path = ROOT / SOURCE_BINDING_REGISTRY
    launcher_path = ROOT / "modal_apps" / "materialize_editing_v2_candidates_app.py"
    completion = materialize_candidates.remote(
        commit=commit,
        overlay_completion_path=overlay_completion_path,
        expected_overlay_completion_file_sha256=overlay_completion_file_sha256,
        expected_source_binding_registry_file_sha256=_file_sha256(registry_path),
        expected_launcher_source_sha256=_file_sha256(launcher_path),
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
