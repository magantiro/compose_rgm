"""The core and PMO oracle chemistry kernels must not be merged by packaging."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_package_targets_the_core_python_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11,<3.12"
    assert "oracle" not in project["optional-dependencies"]


def test_separate_pmo_oracle_kernel_is_pinned():
    core = (ROOT / "requirements/core.txt").read_text()
    oracle = (ROOT / "requirements/pmo-oracle.txt").read_text()
    assert "rdkit==2024.3.5" in core
    assert "rdkit==2023.9.6" in oracle
    assert "scikit-learn==1.2.2" in oracle
    assert "PyTDC==1.1.15" not in core
