"""Optional local-asset parity, with no private worktree on the import path.

Requires assets installed per experiments/fragments/GENERATION.md and the pinned
Python 3.11 environment. Skip only when those documented assets are absent;
present-but-wrong assets or environments fail. No network or benchmark campaign.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.fragments.assets import load_registry, required_assets

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_registry(ROOT)
ASSETS = ROOT / "local_assets/fragments"


@pytest.mark.parametrize("task", list(REGISTRY["task_assets"]))
def test_source_export_saved_attempt_parity(tmp_path, task):
    absent = [
        name
        for name in required_assets(REGISTRY, task)
        if not (ASSETS / REGISTRY["assets"][name]["path"]).is_file()
    ]
    if absent:
        pytest.skip(
            f"documented fragment assets absent: {absent}; see experiments/fragments/GENERATION.md"
        )
    export = tmp_path / "export"
    for relative in ("src/compose_v4/experiments/fragments", "experiments/fragments"):
        shutil.copytree(
            ROOT / relative, export / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    assert not (export / ".git").exists() and not (export / ".worktrees").exists()
    output = tmp_path / "result"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "compose_v4.experiments.fragments",
            "parity",
            "--task",
            task,
            "--assets",
            str(ASSETS),
            "--output",
            str(output),
        ],
        cwd=export,
        env={**os.environ, "PYTHONPATH": str(export / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if process.returncode:
        logs = "\n".join(
            p.read_text()[-4000:] for p in tmp_path.glob(".result.pending-*/worker.log")
        )
        pytest.fail(process.stdout + process.stderr + logs)
    assert json.loads((output / "parity.json").read_text())["passed"] is True
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["code_revision"] is None
    assert provenance["original_launch_authorization_reused"] is False
    result = json.loads((output / "result.json").read_text())
    assert result["execution_environment"]["PYTHONHASHSEED"] == "0"
