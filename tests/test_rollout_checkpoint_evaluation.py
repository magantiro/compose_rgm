from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from scripts.evaluate_tracelet_rollouts import (
    load_factorized_rollout_checkpoint,
    load_reusable_rollouts,
)


def _checkpoint(path: Path) -> FactorizedTraceletRateModel:
    records = build_tracelet_path_records(
        ("CCO", "c1ccccc1"),
        n_slots=12,
        typed_ring_payloads=True,
    )
    catalog = build_typed_ring_catalog(
        (record.path.trace for record in records),
        max_cycle_templates=16,
        max_attach_templates=16,
        max_ear_templates=16,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    torch.save(
        {
            "checkpoint_kind": "selected_evaluation_model",
            "training_backend": "factorized_marks",
            "source_prior": "carbon_tree",
            "ring_catalog": catalog,
            "tree_source_prior": DegreeBoundedCarbonTreePrior(sizes=(3, 4, 5)),
            "hidden_dim": 16,
            "message_passing_steps": 1,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return model


def test_rollout_checkpoint_is_self_contained(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = _checkpoint(path)
    loaded, payload = load_factorized_rollout_checkpoint(path)

    assert payload["checkpoint_kind"] == "selected_evaluation_model"
    assert not loaded.training
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_interim_best_checkpoint_is_rollout_ready(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected.pt"
    expected = _checkpoint(selected_path)
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "checkpoint_kind": "interim_best_evaluation_model",
            "selected_validation": {
                "factorized_gm_loss": 2.5,
                "selected_step": 500.0,
            },
            "completed_steps": 500,
        }
    )
    interim_path = tmp_path / "checkpoint.best_so_far.pt"
    torch.save(payload, interim_path)

    loaded, loaded_payload = load_factorized_rollout_checkpoint(interim_path)

    assert loaded_payload["checkpoint_kind"] == "interim_best_evaluation_model"
    assert loaded_payload["selected_validation"]["selected_step"] == 500.0
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_rollout_checkpoint_rejects_missing_inference_metadata(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="lacks rollout metadata"):
        load_factorized_rollout_checkpoint(path)


def test_rollout_checkpoint_uses_selected_best_state_from_recovery(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected.pt"
    expected = _checkpoint(selected_path)
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    current_state = {
        key: torch.zeros_like(value)
        for key, value in expected.state_dict().items()
    }
    recovery_path = tmp_path / "checkpoint.recovery.pt"
    torch.save(
        {
            **{
                key: value
                for key, value in selected_payload.items()
                if key != "state_dict"
            },
            "checkpoint_kind": "exact_training_recovery",
            "current_state_dict": current_state,
            "best_state_dict": expected.state_dict(),
            "best_metrics": {
                "factorized_gm_loss": 1.25,
                "selected_step": 750.0,
            },
        },
        recovery_path,
    )

    loaded, payload = load_factorized_rollout_checkpoint(recovery_path)

    assert payload["checkpoint_kind"] == "exact_training_recovery"
    assert payload["rollout_state_source"] == "recovery_best_state_dict"
    assert payload["selected_validation"]["selected_step"] == 750.0
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
        assert not torch.equal(loaded.state_dict()[key], current_state[key]) or not torch.any(value)


def test_rollout_checkpoint_rejects_incomplete_recovery_state(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.recovery.pt"
    torch.save(
        {
            "checkpoint_kind": "exact_training_recovery",
            "best_state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="lacks selected-best rollout state"):
        load_factorized_rollout_checkpoint(path)


def test_rollout_cache_reuses_only_exact_complete_signature(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.pt"
    signature = {
        "format": "compose_v4_rollout_cache_v2",
        "checkpoint_sha256": "abc",
        "rollout_samples": 2,
    }

    assert load_reusable_rollouts(path, signature=signature, samples=2) is None
    torch.save(
        {"signature": signature, "rollouts": ("rollout-a", "rollout-b")},
        path,
    )
    assert load_reusable_rollouts(
        path,
        signature=signature,
        samples=2,
    ) == ("rollout-a", "rollout-b")

    with pytest.raises(ValueError, match="signature"):
        load_reusable_rollouts(
            path,
            signature={**signature, "checkpoint_sha256": "different"},
            samples=2,
        )
    with pytest.raises(ValueError, match="size"):
        load_reusable_rollouts(path, signature=signature, samples=3)
