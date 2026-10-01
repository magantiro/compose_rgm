"""Pinned chemistry and implementation identity for local molecular campaigns."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

CORE_ENVIRONMENT = {
    "torch": "2.4.0",
    "numpy": "1.26.4",
    "rdkit": "2024.3.5",
    "scipy": "1.13.1",
    "networkx": "3.3",
}


def core_runtime_identity() -> dict:
    actual = {name: version(name) for name in CORE_ENVIRONMENT}
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise ValueError("COMPOSE core requires Python 3.11")
    for name, expected in CORE_ENVIRONMENT.items():
        if actual[name].split("+")[0] != expected:
            raise ValueError(f"COMPOSE core requires {name}=={expected}, found {actual[name]}")
    package = Path(__file__).resolve().parents[2]
    root = package.parent
    commit = None
    if (root / ".git").exists():
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    return {
        "git_commit": commit,
        "source_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
        "software": {"python": platform.python_version(), **actual},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": 1,
        "device": "cpu",
        "dtype": "float32",
    }
