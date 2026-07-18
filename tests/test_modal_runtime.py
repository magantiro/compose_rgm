from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("modal", reason="Modal runtime tests require the cloud extra")

modal_entrypoint = importlib.import_module("modal_apps.train_tracelet_gm")
_early_rollout_decision = modal_entrypoint._early_rollout_decision


def test_early_rollout_waits_for_warmup_and_material_improvement() -> None:
    payload = {
        "initial_validation": {"factorized_gm_loss": 10.0},
        "selected_validation": {
            "factorized_gm_loss": 9.4,
            "selected_step": 500.0,
        },
    }

    assert (
        _early_rollout_decision(
            payload,
            completed_steps=499,
            warmup_steps=500,
        )
        is None
    )
    decision = _early_rollout_decision(
        payload,
        completed_steps=500,
        warmup_steps=500,
    )

    assert decision is not None
    assert decision["selected_step"] == 500.0
    assert decision["relative_improvement"] == pytest.approx(0.06)


def test_early_rollout_rejects_marginal_or_malformed_checkpoints() -> None:
    marginal = {
        "initial_validation": {"factorized_gm_loss": 10.0},
        "selected_validation": {
            "factorized_gm_loss": 9.51,
            "selected_step": 500.0,
        },
    }
    assert (
        _early_rollout_decision(
            marginal,
            completed_steps=500,
            warmup_steps=500,
        )
        is None
    )
    assert (
        _early_rollout_decision(
            {},
            completed_steps=500,
            warmup_steps=500,
        )
        is None
    )


def test_early_rollout_launch_is_deduplicated_by_persistent_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    torch.save(
        {
            "initial_validation": {"factorized_gm_loss": 10.0},
            "selected_validation": {
                "factorized_gm_loss": 4.0,
                "selected_step": 500.0,
            },
        },
        run_dir / "checkpoint.best_so_far.pt",
    )

    class FakeVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    class FakeCall:
        object_id = "fc-test"

    class FakeStage:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def spawn(self, *args: object) -> FakeCall:
            self.calls.append(args)
            return FakeCall()

    volume = FakeVolume()
    stage = FakeStage()
    monkeypatch.setattr(modal_entrypoint, "artifact_volume", volume)
    monkeypatch.setattr(modal_entrypoint, "rollout_evaluate_stage", stage)
    recipe: dict[str, object] = {"arguments": {"warmup_steps": 500}}

    for _ in range(2):
        modal_entrypoint._maybe_launch_early_rollout(
            run_label="source-run",
            run_dir=run_dir,
            recipe=recipe,
            completed_steps=500,
        )

    assert len(stage.calls) == 1
    assert stage.calls[0][-1] == 100
    assert stage.calls[0][2] == "checkpoint.early_step500.pt"
    marker = json.loads((run_dir / "early_eval_launch.json").read_text())
    assert marker["status"] == "spawned"
    assert marker["function_call_id"] == "fc-test"
    assert len(marker["source_checkpoint_sha256"]) == 64
    assert (run_dir / marker["source_checkpoint"]).is_file()
    assert volume.commits == 2


def test_early_rollout_marker_retries_failed_and_stale_reservations(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "early_eval_launch.json"
    marker_path.write_text(
        json.dumps(
            {
                "status": "reserved",
                "updated_at_unix": modal_entrypoint.time.time(),
            }
        )
    )
    assert modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(
        json.dumps(
            {
                "status": "reserved",
                "updated_at_unix": modal_entrypoint.time.time() - 901.0,
            }
        )
    )
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(json.dumps({"status": "failed"}))
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at_unix": modal_entrypoint.time.time() - 901.0,
            }
        )
    )
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)
