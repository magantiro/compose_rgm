"""Offline orchestration checks for the QED value-head workflow."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_qed_shared_value_pipeline as pipeline


def test_pipeline_commands_bind_role_index_and_shared_checkpoint(tmp_path: Path) -> None:
    job = pipeline.SourceJob("train", 12)
    checkpoint = tmp_path / "reference.pt"
    rollout_command, rollout = pipeline.stage_command(
        "rollouts",
        job,
        tmp_path,
        horizon=24,
        replicates=2,
        time=0.5,
        checkpoint=checkpoint,
    )
    assert rollout.name == "train_0012.json"
    assert rollout_command[rollout_command.index("--checkpoint") + 1] == str(checkpoint)
    assert rollout_command[rollout_command.index("--role") + 1] == "train"
    assert rollout_command[rollout_command.index("--index") + 1] == "12"
    with pytest.raises(FileNotFoundError, match="rollout is absent"):
        pipeline.stage_command(
            "features",
            job,
            tmp_path,
            horizon=24,
            replicates=2,
            time=0.5,
            checkpoint=checkpoint,
        )
    rollout.parent.mkdir(parents=True)
    rollout.write_text("{}")
    feature_command, feature = pipeline.stage_command(
        "features",
        job,
        tmp_path,
        horizon=24,
        replicates=2,
        time=0.5,
        checkpoint=checkpoint,
    )
    assert feature.name == "train_0012.npz"
    assert feature_command[feature_command.index("--rollout") + 1] == str(rollout)


def test_resume_preserves_existing_outputs_and_runs_missing_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = (pipeline.SourceJob("train", 0), pipeline.SourceJob("validation", 0))
    existing = tmp_path / "rollouts/train_0000.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserved")
    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_stage(
            "rollouts",
            jobs,
            tmp_path,
            workers=1,
            resume=False,
            horizon=24,
            replicates=2,
            time=0.5,
            checkpoint=tmp_path / "reference.pt",
        )

    def write_output(_command: list[str], output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("new")

    monkeypatch.setattr(pipeline, "_run", write_output)
    assert pipeline.run_stage(
        "rollouts",
        jobs,
        tmp_path,
        workers=1,
        resume=True,
        horizon=24,
        replicates=2,
        time=0.5,
        checkpoint=tmp_path / "reference.pt",
    ) == (1, 1)
    assert existing.read_text() == "preserved"
    assert (tmp_path / "rollouts/validation_0000.json").read_text() == "new"


def test_invalid_reference_time_fails_before_generating(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(pipeline.ROOT / "tools/run_qed_shared_value_pipeline.py"),
            "--workspace",
            str(tmp_path),
            "--time",
            "nan",
        ],
        cwd=pipeline.ROOT,
        env={**os.environ, "PYTHONPATH": str(pipeline.ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "reference time must be finite" in result.stderr
    assert not list(tmp_path.iterdir())
