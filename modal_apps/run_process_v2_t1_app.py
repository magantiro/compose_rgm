"""Launch the bounded Process-V2 T1 stages on Modal.

This module is orchestration only.  Scientific panel selection, successor
materialization, gradient probing, capacity optimization, decision semantics,
and immutable publication belong to the Process-V2 T1 library modules.  Importing
this module launches nothing.

Every remote stage receives one source-revision object produced from an exact,
clean Git commit.  A worker re-hashes every byte serialized into its image before
opening an artifact or constructing a model.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_t1_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

REVISION_SCHEMA = "compose.editing_v2.process_v2_t1_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
PROGRESS_SCHEMA = "compose.editing_v2.process_v2_t1_modal_progress"
PROGRESS_SCHEMA_VERSION = 1

PANEL_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_panel"
PANEL_CPU = 1.0
PANEL_MEMORY_MB = 16 * 1024
PANEL_TIMEOUT_SECONDS = 4 * 3600

NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}

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
for source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / source_directory,
        str(REMOTE_ROOT / source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-process-v2-t1")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)

_VALIDATED_REVISION_SHA256: str | None = None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish Process-V2 T1 Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / directory).rglob("*"))
            if path.is_file()
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("Process-V2 T1 serialized source inventory repeats a path")
    return tuple(paths)


def local_image_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind the exact clean commit and every byte serialized into the image."""

    if _COMMIT_RE.fullmatch(str(expected_commit)) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("Process-V2 T1 launch requires the exact clean commit")
    sources = {
        relative: _file_sha256(root / relative)
        for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not set(sources).issubset(tracked):
        raise RuntimeError("every serialized Process-V2 T1 source must be Git-tracked")
    body = {
        "schema": REVISION_SCHEMA,
        "schema_version": REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": _sha256(body)}


def _validate_remote_revision(value: object, *, remote_root: Path = REMOTE_ROOT) -> None:
    """Re-hash the serialized image once per warm container."""

    global _VALIDATED_REVISION_SHA256
    if not isinstance(value, dict):
        raise RuntimeError("Process-V2 T1 image revision must be a mapping")
    supplied = value.get("image_revision_sha256")
    if isinstance(supplied, str) and _VALIDATED_REVISION_SHA256 == supplied:
        return
    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    sources = value.get("serialized_sources")
    if (
        value.get("schema") != REVISION_SCHEMA
        or value.get("schema_version") != REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(value.get("commit"))) is None
        or _COMMIT_RE.fullmatch(str(value.get("tree"))) is None
        or not isinstance(supplied, str)
        or _SHA256_RE.fullmatch(supplied) is None
        or supplied != _sha256(body)
        or not isinstance(sources, dict)
        or set(sources) != set(_serialized_source_paths(remote_root))
    ):
        raise RuntimeError("Process-V2 T1 image revision disagrees")
    for relative, digest in sources.items():
        if _SHA256_RE.fullmatch(str(digest)) is None:
            raise RuntimeError(f"serialized Process-V2 T1 digest is malformed: {relative}")
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"serialized Process-V2 T1 source differs: {relative}")
    _VALIDATED_REVISION_SHA256 = supplied


def _require_artifact_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized absolute artifact path")
    try:
        path.relative_to(ARTIFACT_ROOT)
    except ValueError as error:
        raise ValueError(f"{field} must be below {ARTIFACT_ROOT}") from error
    return path


def _emit_progress(event: Mapping[str, Any]) -> None:
    """Emit one compact structured event without performing extra science."""

    print(
        json.dumps(
            {
                "schema": PROGRESS_SCHEMA,
                "schema_version": PROGRESS_SCHEMA_VERSION,
                **dict(event),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _panel_imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        PANEL_FILENAME,
        build_process_v2_t1_panel,
        load_process_v2_t1_panel,
        open_process_v2_t1_source,
        write_process_v2_t1_panel,
    )

    return {
        "PANEL_FILENAME": PANEL_FILENAME,
        "canonical_bytes": canonical_bytes,
        "open_source": open_process_v2_t1_source,
        "build_panel": build_process_v2_t1_panel,
        "write_panel": write_process_v2_t1_panel,
        "load_panel": load_process_v2_t1_panel,
    }


def _materialize_panel(
    *,
    image_revision: dict[str, Any],
    active8_run_root: str,
    gate_zero_decision: str,
    output_prefix: str,
    remote_root: Path,
    artifact_root: Path,
    reload_volume: Callable[[], None],
    commit_volume: Callable[[], None],
    emit_progress: Callable[[Mapping[str, Any]], None] = _emit_progress,
    loaded: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Delegate one authenticated panel publication to the native library."""

    _validate_remote_revision(image_revision, remote_root=remote_root)
    reload_volume()
    active8_root = _require_artifact_path(active8_run_root, field="active8_run_root")
    gate_zero_path = _require_artifact_path(
        gate_zero_decision,
        field="gate_zero_decision",
    )
    prefix = _require_artifact_path(output_prefix, field="output_prefix")
    emit_progress(
        {
            "stage": "panel",
            "event": "stage_start",
            "image_revision_sha256": image_revision["image_revision_sha256"],
            "active8_run_root": str(active8_root),
            "gate_zero_decision_path": str(gate_zero_path),
        }
    )
    interfaces = dict(loaded or _panel_imports(remote_root))
    source = interfaces["open_source"](
        active8_root,
        gate_zero_decision_path=gate_zero_path,
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    panel = interfaces["build_panel"](source)
    panel_sha256 = str(panel.get("panel_sha256"))
    if _SHA256_RE.fullmatch(panel_sha256) is None:
        raise RuntimeError("the Process-V2 T1 panel has no valid identity")
    entries = panel.get("entries")
    family_counts = panel.get("family_counts")
    cell_counts = panel.get("capability_cell_counts")
    if (
        not isinstance(entries, list)
        or not isinstance(family_counts, dict)
        or not isinstance(cell_counts, dict)
    ):
        raise RuntimeError("the Process-V2 T1 panel summary is malformed")
    emit_progress(
        {
            "stage": "panel",
            "event": "panel_built",
            "panel_sha256": panel_sha256,
            "entry_count": len(entries),
            "family_counts": dict(family_counts),
            "capability_cell_count": len(cell_counts),
        }
    )
    output_root = prefix / panel_sha256
    path = interfaces["write_panel"](
        panel,
        output_root=output_root,
        source=source,
    )
    expected_path = output_root / str(interfaces["PANEL_FILENAME"])
    if Path(path).resolve() != expected_path.resolve():
        raise RuntimeError("the Process-V2 T1 panel writer returned another path")
    commit_volume()
    reload_volume()
    reopened = interfaces["load_panel"](path, source=source)
    if interfaces["canonical_bytes"](reopened) != interfaces["canonical_bytes"](panel):
        raise RuntimeError("the Process-V2 T1 panel changed across publication")
    family_counts = reopened.get("family_counts")
    cell_counts = reopened.get("capability_cell_counts")
    entries = reopened.get("entries")
    if (
        not isinstance(family_counts, dict)
        or not isinstance(cell_counts, dict)
        or not isinstance(entries, list)
    ):
        raise RuntimeError("the Process-V2 T1 panel summary is malformed")
    result = {
        "phase": "process_v2_t1_panel_complete",
        "panel_artifact_path": str(path),
        "panel_file_sha256": _file_sha256(path),
        "panel_sha256": panel_sha256,
        "entry_count": len(entries),
        "family_counts": dict(family_counts),
        "capability_cell_count": len(cell_counts),
        "image_revision_sha256": image_revision["image_revision_sha256"],
        **NO_AUTHORITY,
    }
    emit_progress(
        {
            "stage": "panel",
            "event": "stage_end",
            "panel_artifact_path": result["panel_artifact_path"],
            "panel_file_sha256": result["panel_file_sha256"],
            "panel_sha256": result["panel_sha256"],
            "entry_count": result["entry_count"],
        }
    )
    return result


@app.function(
    image=image,
    cpu=PANEL_CPU,
    memory=PANEL_MEMORY_MB,
    timeout=PANEL_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_process_v2_t1_panel(
    *,
    image_revision: dict[str, Any],
    active8_run_root: str,
    gate_zero_decision: str,
    output_prefix: str = PANEL_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Materialize one deterministic Process-V2 unique-state T1 panel."""

    return _materialize_panel(
        image_revision=image_revision,
        active8_run_root=active8_run_root,
        gate_zero_decision=gate_zero_decision,
        output_prefix=output_prefix,
        remote_root=REMOTE_ROOT,
        artifact_root=ARTIFACT_ROOT,
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )


__all__ = [
    "app",
    "local_image_revision",
    "materialize_process_v2_t1_panel",
]
