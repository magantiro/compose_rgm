"""Offline, hash-checked fragment assets. No discovery, fetching, or fallback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import EvidenceError


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1\n")


def child_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not name or ".." in relative.parts:
        raise EvidenceError(f"unsafe relative path: {name!r}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise EvidenceError(f"path escapes root through a symlink: {target}")
    return target


def load_registry(root: Path) -> dict:
    path = root / "experiments/fragments/assets.json"
    registry = json.loads(path.read_text())
    if registry.get("schema") != "compose_fragment_assets_v1":
        raise EvidenceError(f"invalid asset registry schema: {path}")
    for name, record in registry["assets"].items():
        child_path(root, record["path"])
        if len(record["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in record["sha256"]
        ):
            raise EvidenceError(f"invalid SHA-256 for asset {name}")
    return registry


def required_assets(registry: dict, task: str, *, include_evaluator: bool = True) -> list[str]:
    if task not in registry["task_assets"]:
        raise EvidenceError(f"unknown fragment task: {task}")
    evaluator_assets = sorted(name for name in registry["assets"] if name.startswith("evaluator_"))
    return registry["task_assets"][task] + (evaluator_assets if include_evaluator else [])


def verify_assets(
    registry: dict,
    directory: Path,
    task: str | None = None,
    *,
    include_evaluator: bool = True,
) -> dict[str, Path]:
    paths, errors = {}, []
    names = (
        required_assets(registry, task, include_evaluator=include_evaluator)
        if task
        else sorted(registry["assets"])
    )
    for name in names:
        record = registry["assets"][name]
        path = child_path(directory, record["path"])
        if not path.is_file():
            errors.append(f"missing {name}: {path} (expected SHA-256 {record['sha256']})")
        elif sha256(path) != record["sha256"]:
            if _is_lfs_pointer(path):
                errors.append(f"Git LFS pointer for {name}: {path}; run git lfs pull")
            else:
                errors.append(f"hash mismatch for {name}: {path} (expected {record['sha256']})")
        else:
            paths[name] = path.resolve()
    if errors:
        raise EvidenceError("\n".join(errors))
    return paths


def install_asset(registry: dict, directory: Path, name: str, source: Path) -> Path:
    """Copy a known asset atomically without replacing any existing destination."""
    if name not in registry["assets"]:
        raise EvidenceError(f"unknown asset {name!r}; choose from {sorted(registry['assets'])}")
    record = registry["assets"][name]
    if sha256(source) != record["sha256"]:
        raise EvidenceError(f"wrong source hash for {name}: {source}; expected {record['sha256']}")
    target = child_path(directory, record["path"])
    if target.exists():
        if sha256(target) == record["sha256"]:
            return target
        raise FileExistsError(f"refusing to replace asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".asset-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != record["sha256"]:
            raise EvidenceError(f"asset changed while copying: {source}")
        # link() is an atomic no-clobber handoff, including concurrent writers.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
