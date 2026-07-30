from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compose_v4.experiments import p50_completion as completion
from compose_v4.experiments.p50_completion import (
    P50CompletionError,
    assert_p50_completion_targets_absent,
    p50_completion_targets,
    publish_p50_completion,
    validate_p50_completion_member,
)


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    selected = {
        "checkpoint_kind": "selected_evaluation_model",
        "state_dict": {"weight": torch.tensor([1.0])},
        "selected_validation": {"loss": 0.5},
    }
    recovery = {
        "checkpoint_kind": "exact_training_recovery",
        "completed_steps": 50,
        "current_state_dict": {"weight": torch.tensor([2.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "best_state_dict": {"weight": torch.tensor([1.0])},
        "best_metrics": {"loss": 0.5},
        "history": [],
    }
    return selected, recovery


@pytest.mark.parametrize("target_index", range(7))
def test_preexisting_p50_target_is_rejected_without_overwrite(
    tmp_path: Path,
    target_index: int,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    occupied = p50_completion_targets(checkpoint)[target_index]
    if occupied.suffix:
        occupied.write_text("do not overwrite")
    else:
        occupied.mkdir()

    with pytest.raises(P50CompletionError, match="already exist"):
        assert_p50_completion_targets_absent(checkpoint)
    assert occupied.exists()


def test_p50_completion_publishes_one_manifest_complete_namespace(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    selected, recovery = _payloads()
    published = publish_p50_completion(
        checkpoint,
        selected_payload=selected,
        recovery_payload=recovery,
    )

    assert not checkpoint.exists()
    assert published.namespace.is_dir()
    assert published.manifest.is_file()
    selected_payload = torch.load(
        published.selected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    recovery_payload = torch.load(
        published.recovery_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    selected_manifest = validate_p50_completion_member(
        published.selected_checkpoint,
        selected_payload,
    )
    recovery_manifest = validate_p50_completion_member(
        published.recovery_checkpoint,
        recovery_payload,
    )
    assert selected_manifest == recovery_manifest
    assert selected_manifest["status"] == "PASS"
    assert selected_manifest["completed_optimizer_steps"] == 50

    with pytest.raises(P50CompletionError, match="already exist"):
        publish_p50_completion(
            checkpoint,
            selected_payload=selected,
            recovery_payload=recovery,
        )


def test_serialization_failure_leaves_no_completion_or_staging_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    selected, recovery = _payloads()
    real_save = completion.torch.save
    calls = 0

    def fail_second_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected recovery serialization failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(completion.torch, "save", fail_second_save)
    with pytest.raises(
        RuntimeError,
        match="injected recovery serialization failure",
    ):
        publish_p50_completion(
            checkpoint,
            selected_payload=selected,
            recovery_payload=recovery,
        )

    assert all(
        not path.exists() for path in p50_completion_targets(checkpoint)
    )


def test_copied_or_manifestless_member_cannot_be_interpreted_as_complete(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    selected, recovery = _payloads()
    published = publish_p50_completion(
        checkpoint,
        selected_payload=selected,
        recovery_payload=recovery,
    )
    payload = torch.load(
        published.selected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    copied = tmp_path / published.selected_checkpoint.name
    torch.save(payload, copied)
    with pytest.raises(P50CompletionError, match="outside its declared"):
        validate_p50_completion_member(copied, payload)

    published.manifest.unlink()
    with pytest.raises(P50CompletionError, match="manifest is absent"):
        validate_p50_completion_member(
            published.selected_checkpoint,
            payload,
        )
