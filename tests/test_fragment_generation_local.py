"""One local-asset smoke from a plain source export.

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
ASSETS = Path(os.environ.get("COMPOSE_FRAGMENT_ASSETS", ROOT / "local_assets/fragments")).resolve()


@pytest.mark.external_artifact
def test_source_export_runs_one_nll_reference_attempt(tmp_path):
    absent = [
        name
        for name in required_assets(REGISTRY, "superstructure_generation", include_evaluator=False)
        if not (ASSETS / REGISTRY["assets"][name]["path"]).is_file()
    ]
    if absent:
        pytest.skip(f"documented fragment assets absent: {absent}")
    export = tmp_path / "export"
    for relative in ("src", "experiments/fragments"):
        shutil.copytree(
            ROOT / relative, export / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    output = tmp_path / "result"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "compose_v4.experiments.fragments",
            "generate",
            "--task",
            "superstructure_generation",
            "--assets",
            str(ASSETS),
            "--python",
            sys.executable,
            "--seed",
            "0",
            "--prompt",
            "BARICITINIB",
            "--attempts",
            "1",
            "--no-metrics",
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
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["checkpoint_sha256"] == REGISTRY["assets"]["checkpoint"]["sha256"]
    assert len(result["cells"][0]["attempts"]) == 1
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["code_revision"] is None
    assert provenance["original_launch_authorization_reused"] is False
    assert provenance["runtime"] == "library"
    assert provenance["source_sha256"]
    assert result["execution_environment"]["PYTHONHASHSEED"] == "0"
