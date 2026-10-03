"""Offline orchestration checks for the QED value-head workflow."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import run_qed_shared_value_pipeline as pipeline


def _existing_rollout(role: str, index: int, *, horizon: int = 24, replicates: int = 2) -> dict:
    roles = pipeline.load_qed_source_roles(
        pipeline.ROOT, pipeline.ROOT / "experiments/qed/shared_sources.json"
    )
    sources = roles.train if role == "train" else roles.validation
    input_index = roles.train_input_indices[index] if role == "train" else index
    return {
        "schema_version": "compose.qed.shared_rollout.v1",
        "source_role": role,
        "source_index": index,
        "source_original": sources[index],
        "source_input_row_index": input_index,
        "source_split_sha256": roles.manifest_sha256,
        "configuration": {"horizon": horizon, "replicates": replicates},
        "reference": pipeline._reference_fields(0.5),
        "trajectories": [{} for _ in range(replicates)],
    }


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


def test_pipeline_defaults_to_frozen_rollout_replicate_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    monkeypatch.setattr(pipeline, "source_jobs", lambda: ())
    monkeypatch.setattr(
        pipeline, "run_stage", lambda *args, **kwargs: observed.append(kwargs) or (0, 0)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_qed_shared_value_pipeline.py", "--workspace", str(tmp_path), "--stage", "rollouts"],
    )

    pipeline.main()

    assert len(observed) == 1
    assert observed[0]["horizon"] == 24
    assert observed[0]["replicates"] == 16
    assert observed[0]["time"] == 0.5


def test_resume_preserves_existing_outputs_and_runs_missing_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = (pipeline.SourceJob("train", 0), pipeline.SourceJob("validation", 0))
    existing = tmp_path / "rollouts/train_0000.json"
    existing.parent.mkdir(parents=True)
    existing_payload = json.dumps(_existing_rollout("train", 0))
    existing.write_text(existing_payload)
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
    assert existing.read_text() == existing_payload
    assert (tmp_path / "rollouts/validation_0000.json").read_text() == "new"


def test_resume_rejects_different_replicate_count_before_running_missing_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "rollouts/train_0000.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_existing_rollout("train", 0)))
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args: pytest.fail("resume started a new job after an incompatible output"),
    )
    with pytest.raises(ValueError, match="incompatible configuration"):
        pipeline.run_stage(
            "rollouts",
            (pipeline.SourceJob("train", 0), pipeline.SourceJob("validation", 0)),
            tmp_path,
            workers=1,
            resume=True,
            horizon=24,
            replicates=8,
            time=0.5,
            checkpoint=tmp_path / "reference.pt",
        )
    assert not (tmp_path / "rollouts/validation_0000.json").exists()


def test_resume_rejects_feature_bound_to_another_rollout(tmp_path: Path) -> None:
    job = pipeline.SourceJob("train", 0)
    rollout = tmp_path / "rollouts/train_0000.json"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(json.dumps(_existing_rollout("train", 0)))
    feature = tmp_path / "features/train_0000.npz"
    feature.parent.mkdir(parents=True)
    roles = pipeline.load_qed_source_roles(
        pipeline.ROOT, pipeline.ROOT / "experiments/qed/shared_sources.json"
    )
    metadata = {
        "schema_version": "compose.qed.shared_features.v4",
        "role": "train",
        "source_index": 0,
        "source_input_row_index": roles.train_input_indices[0],
        "source_split_sha256": roles.manifest_sha256,
        "budget_max": 24,
        "target_semantics": "terminal_region",
        "rollout_sha256": "0" * 64,
        "builder_sha256": hashlib.sha256(
            (pipeline.ROOT / "src/compose_v4/experiments/qed_shared_training.py").read_bytes()
        ).hexdigest(),
        "reference": pipeline._reference_fields(0.5),
    }
    np.savez_compressed(feature, metadata=np.str_(json.dumps(metadata)))
    with pytest.raises(ValueError, match="incompatible rollout_sha256"):
        pipeline.run_stage(
            "features",
            (job,),
            tmp_path,
            workers=1,
            resume=True,
            horizon=24,
            replicates=2,
            time=0.5,
            checkpoint=tmp_path / "reference.pt",
        )


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
