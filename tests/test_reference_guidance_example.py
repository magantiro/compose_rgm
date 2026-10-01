from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_export_does_not_report_an_enclosing_checkout(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "reference_example", ROOT / "examples/reference_guidance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "export/examples/reference_guidance.py"
    source.parent.mkdir(parents=True)
    source.write_text("# export fixture\n")
    monkeypatch.setattr(module, "__file__", str(source))
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("must not probe enclosing Git")
    )
    identity = module.source_identity()
    assert identity["git_commit"] is None and identity["dirty"] is None
    assert "examples/reference_guidance.py" in identity["source_sha256"]


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
@pytest.mark.parametrize("selector", ["toy", "pmo", "t4"])
def test_example_is_runnable_and_publishes_no_clobber_output(tmp_path, mode, selector):
    checkpoint = os.environ.get("COMPOSE_REFERENCE_CHECKPOINT")
    if mode != "off" and not checkpoint:
        pytest.skip(
            "configure the documented COMPOSE_REFERENCE_CHECKPOINT for model-backed examples"
        )
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(ROOT / "examples/reference_guidance.py"),
        "--mode",
        mode,
        "--selector",
        selector,
        "--output",
        str(output),
    ]
    if mode != "off":
        command.extend(["--checkpoint", checkpoint])
    if mode == "active":
        command.extend(["--strength", "0.25"])
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
    process = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60, check=False
    )
    assert process.returncode == 0, process.stdout + process.stderr
    original = output.read_bytes()
    result = json.loads(original)
    assert result["oracle_calls"] == 0 and result["candidate_count"] == 2
    assert result["configuration"]["selector"] == selector
    assert result["selection"]["outcome"] == mode
    assert result["selection"]["probabilities_changed"] == (mode == "active")
    assert result["selection"]["reference_evaluated"] == (mode != "off")
    expected_progress = None if mode == "off" else result["selection"]["reference"]["progress"]
    assert result["configuration"]["reference_progress"] == expected_progress
    assert all(p["trace_sha256"] and len(p["trace"]["states"]) == 2 for p in result["programs"])
    assert (
        "src/compose_v4/model/reference_checkpoint.py" in result["implementation"]["source_sha256"]
    )
    if mode in ("off", "shadow"):
        assert result["baseline_index"] == result["selected_index"]
        assert result["baseline_rng_state"] == result["selected_rng_state"]
    repeated = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60, check=False
    )
    assert repeated.returncode != 0
    assert "refusing to replace output" in repeated.stderr
    assert output.read_bytes() == original
