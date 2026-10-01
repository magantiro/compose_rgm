from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/t4_reference_panel.py"
LEAD = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"


def command(output):
    return [
        sys.executable,
        str(SCRIPT),
        "--lead",
        LEAD,
        "--delta",
        "0.6",
        "--draws",
        "2",
        "--horizon",
        "1",
        "--output",
        str(output),
    ]


def environment():
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}


def test_generated_panel_example_is_offline_replayable_and_preserves_outputs(tmp_path):
    output = tmp_path / "panel.json"
    process = subprocess.run(
        command(output),
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "compose.t4_reference_panel"
    assert payload["status"] == "complete"
    assert payload["candidate_count"] > 0
    assert payload["candidate_count"] == len(payload["programs"])
    assert payload["oracle_calls"] == payload["excluded_after_generation"] == 0
    assert payload["checkpoint_path"] is None
    assert payload["configuration"]["reference_progress"] is None
    assert payload["chosen"] == payload["baseline_chosen"]
    assert payload["rng"]["selection"] == payload["rng"]["baseline_selection"]
    assert payload["configuration"]["parent_score_role"].startswith("synthetic fixture")
    for record in payload["programs"]:
        if record["trace"] is not None:
            serialized = json.dumps(record["trace"], sort_keys=True, separators=(",", ":"))
            assert hashlib.sha256(serialized.encode()).hexdigest() == record["trace_sha256"]
        else:
            assert record["missing_reason"]
    original = output.read_bytes()
    repeated = subprocess.run(
        command(output),
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert repeated.returncode != 0
    assert "refusing to replace output" in repeated.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--delta", "nan"], "--delta must be finite"),
        (["--delta", "-0.1"], "--delta must be finite"),
        (["--draws", "0"], "must be a positive integer"),
        (["--proposal-seed", "-1"], "seeds must be nonnegative"),
        (["--mode", "active", "--strength", "0.25"], "--checkpoint is required"),
        (["--proposal-lane", "route_complete_region"], "shallow-lane settings"),
        (["--program-template-prior", "absent.json"], "program template prior requires"),
    ],
)
def test_invalid_panel_configuration_fails_without_output(tmp_path, arguments, message):
    output = tmp_path / "invalid.json"
    process = subprocess.run(
        [*command(output), *arguments],
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode != 0
    assert message in process.stderr
    assert not output.exists()


def test_route_lane_requires_pinned_asset_without_generating(tmp_path):
    output = tmp_path / "panel.json"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lead",
            LEAD,
            "--delta",
            "0.6",
            "--proposal-lane",
            "route_complete_region",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode != 0
    assert "requires --program-template-prior and --program-template-prior-sha256" in process.stderr
    assert not output.exists()


def test_active_panel_records_the_reference_time_used_for_scoring(tmp_path):
    checkpoint = os.environ.get("COMPOSE_REFERENCE_CHECKPOINT")
    if not checkpoint:
        pytest.skip("configure COMPOSE_REFERENCE_CHECKPOINT for the model-backed example")
    output = tmp_path / "active.json"
    process = subprocess.run(
        [
            *command(output),
            "--mode",
            "active",
            "--strength",
            "0.25",
            "--checkpoint",
            checkpoint,
        ],
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(output.read_text())
    assert (
        payload["configuration"]["reference_progress"]
        == payload["selection"][0]["reference"]["progress"]
    )


def test_program_template_prior_flag_records_verified_input(tmp_path):
    output = tmp_path / "program-panel.json"
    asset = ROOT / "experiments/t4/assets/parp1_checkpoint.json"
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lead",
            LEAD,
            "--delta",
            "0.6",
            "--proposal-lane",
            "route_complete_region",
            "--program-template-prior",
            str(asset),
            "--program-template-prior-sha256",
            digest,
            "--route-pool-size",
            "1",
            "--route-realizations",
            "1",
            "--route-beam-width",
            "1",
            "--route-expansion-width",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 3
    assert payload["program_template_prior"]["sha256"] == digest
    assert payload["oracle_calls"] == 0
