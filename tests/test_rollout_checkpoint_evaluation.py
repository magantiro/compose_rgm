from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.cnof_conditional import CompactTrajectoryDiagnostics
from compose_v4.experiments.p50_completion import publish_p50_completion
from compose_v4.experiments.tracelet_conditional import (
    TraceletRollout,
    build_tracelet_path_records,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkEmpiricalPriors,
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


def test_rollout_loader_requires_p50_completion_manifest_membership(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    expected = _checkpoint(source)
    selected = torch.load(source, map_location="cpu", weights_only=False)
    published = publish_p50_completion(
        tmp_path / "candidate.pt",
        selected_payload=selected,
        recovery_payload={
            "checkpoint_kind": "exact_training_recovery",
            "completed_steps": 50,
            "current_state_dict": expected.state_dict(),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "best_state_dict": expected.state_dict(),
            "best_metrics": {"loss": 0.0},
            "history": [],
        },
    )
    loaded, _ = load_factorized_rollout_checkpoint(
        published.selected_checkpoint
    )
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)

    payload = torch.load(
        published.selected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    copied = tmp_path / published.selected_checkpoint.name
    torch.save(payload, copied)
    with pytest.raises(
        RuntimeError,
        match="outside its declared immutable namespace",
    ):
        load_factorized_rollout_checkpoint(copied)


def test_rollout_checkpoint_ring_delete_flag_is_strict_and_legacy_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    payload["enable_ring_system_delete"] = False
    torch.save(payload, path)
    loaded, _ = load_factorized_rollout_checkpoint(path)
    assert loaded.enable_ring_system_delete is False

    payload.pop("enable_ring_system_delete")
    torch.save(payload, path)
    loaded, _ = load_factorized_rollout_checkpoint(path)
    assert loaded.enable_ring_system_delete is True

    payload["enable_ring_system_delete"] = "false"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="literal Boolean"):
        load_factorized_rollout_checkpoint(path)


def test_rollout_checkpoint_ring_restate_flag_is_strict_and_legacy_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["corrupted_prior_mix"] = True

    payload["enable_ring_restates"] = False
    torch.save(payload, path)
    loaded, _ = load_factorized_rollout_checkpoint(path)
    assert loaded.enable_ring_restates is False

    payload.pop("enable_ring_restates")
    torch.save(payload, path)
    loaded, _ = load_factorized_rollout_checkpoint(path)
    assert loaded.enable_ring_restates is True

    payload["enable_ring_restates"] = "false"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="literal Boolean"):
        load_factorized_rollout_checkpoint(path)


@pytest.mark.parametrize("malformed", ["false", 0, 1, None])
def test_rollout_checkpoint_rejects_malformed_legacy_restate_fallback(
    tmp_path: Path,
    malformed,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("enable_ring_restates", None)
    payload["corrupted_prior_mix"] = malformed
    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match="corrupted_prior_mix.*literal Boolean",
    ):
        load_factorized_rollout_checkpoint(path)


def test_rollout_checkpoint_restores_empirical_base_measure_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    uniform_four = (-1.3862943611198906,) * 4
    uniform_three = (-1.0986122886681098,) * 3
    uniform_eight = (-2.0794415416798357,) * 8
    uniform_twelve = (-2.4849066497880004,) * 12
    priors = FactorizedMarkEmpiricalPriors(
        root_atom_log_probabilities=uniform_four,
        connected_atom_order_log_probabilities=tuple(
            uniform_twelve[index : index + 4] for index in range(0, 12, 4)
        ),
        atom_restate_log_probabilities=uniform_four,
        bond_reorder_log_probabilities=uniform_three,
        ring_electronic_log_probabilities=uniform_eight,
    )
    configured = FactorizedTraceletRateModel(
        payload["ring_catalog"],
        hidden_dim=16,
        message_passing_steps=1,
        empirical_mark_prior_mode="corpus_residual_v1",
        empirical_mark_priors=priors,
        ring_family_mass_mode="catalog_topology_local_support",
    )
    configured.load_state_dict(expected.state_dict(), strict=True)
    payload.update(
        {
            "state_dict": configured.state_dict(),
            "empirical_mark_prior_mode": "corpus_residual_v1",
            "empirical_mark_priors": priors.to_dict(),
            "ring_family_mass_mode": "catalog_topology_local_support",
        }
    )
    torch.save(payload, path)

    loaded, _ = load_factorized_rollout_checkpoint(path)

    assert loaded.empirical_mark_prior_mode == "corpus_residual_v1"
    assert loaded.ring_family_mass_mode == "catalog_topology_local_support"
    assert torch.equal(loaded.bond_reorder_log_prior, configured.bond_reorder_log_prior)
    assert torch.equal(
        loaded.ring_electronic_log_prior,
        configured.ring_electronic_log_prior,
    )


def test_rollout_checkpoint_restores_hierarchical_ring_template_factorization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    configured = FactorizedTraceletRateModel(
        payload["ring_catalog"],
        hidden_dim=16,
        message_passing_steps=1,
        ring_template_factorization="topology_cycle_hierarchical",
    )
    payload.update(
        {
            "state_dict": configured.state_dict(),
            "ring_template_factorization": "topology_cycle_hierarchical",
        }
    )
    torch.save(payload, path)

    loaded, loaded_payload = load_factorized_rollout_checkpoint(path)

    assert loaded_payload["ring_template_factorization"] == (
        "topology_cycle_hierarchical"
    )
    assert loaded.ring_template_factorization == "topology_cycle_hierarchical"
    assert loaded.ring_topology_group_head is not None
    for key, value in configured.state_dict().items():
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


def test_legacy_ring_checkpoint_loads_with_neutral_role_logits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    model = _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("ring_electronic_mode", None)
    payload["ring_catalog"] = replace(
        payload["ring_catalog"],
        ring_system_electronic_alias_version=0,
        ring_system_electronic_aliases=(),
        ring_system_electronic_alias_counts=(),
    )
    payload["state_dict"] = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("ring_system_role_head.")
    }
    torch.save(payload, path)

    loaded, loaded_payload = load_factorized_rollout_checkpoint(path)

    assert (
        loaded_payload["checkpoint_compatibility"]
        == "legacy_v0_neutral_ring_role_logits"
    )
    role_parameters = tuple(
        parameter
        for name, parameter in loaded.named_parameters()
        if name.startswith("ring_system_role_head.")
    )
    assert role_parameters
    assert all(torch.count_nonzero(parameter) == 0 for parameter in role_parameters)
    assert loaded.virtualize_legacy_self_grafts is True


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
        "format": "compose_v4_rollout_cache_v3_compact_trajectories",
        "checkpoint_sha256": "abc",
        "rollout_samples": 2,
    }
    rollout = TraceletRollout(
        final_state=smiles_to_molecular_graph("CC"),
        event_times=(),
        event_rules=(),
        exhausted_event_budget=False,
        diagnostics=CompactTrajectoryDiagnostics(
            canonical_state_keys=("CC",),
            atom_counts=(2,),
            state_valid=(True,),
            state_connected_or_null=(True,),
        ),
    )

    assert load_reusable_rollouts(path, signature=signature, samples=2) is None
    torch.save(
        {"signature": signature, "rollouts": (rollout, rollout)},
        path,
    )
    reused = load_reusable_rollouts(
        path,
        signature=signature,
        samples=2,
    )
    assert reused is not None
    assert len(reused) == 2
    assert all(item.diagnostics == rollout.diagnostics for item in reused)

    with pytest.raises(ValueError, match="signature"):
        load_reusable_rollouts(
            path,
            signature={**signature, "checkpoint_sha256": "different"},
            samples=2,
        )
    with pytest.raises(ValueError, match="size"):
        load_reusable_rollouts(path, signature=signature, samples=3)

    torch.save(
        {"signature": signature, "rollouts": ("terminal-only", "terminal-only")},
        path,
    )
    with pytest.raises(ValueError, match="trajectory diagnostics"):
        load_reusable_rollouts(path, signature=signature, samples=2)

    corrupt_diagnostics = CompactTrajectoryDiagnostics(
        canonical_state_keys=("CC",),
        atom_counts=(2,),
        state_valid=(True,),
        state_connected_or_null=(True,),
    )
    object.__setattr__(corrupt_diagnostics, "atom_counts", (-1,))
    corrupt_rollout = TraceletRollout.__new__(TraceletRollout)
    object.__setattr__(corrupt_rollout, "final_state", smiles_to_molecular_graph("CC"))
    object.__setattr__(corrupt_rollout, "event_times", ())
    object.__setattr__(corrupt_rollout, "event_rules", ())
    object.__setattr__(corrupt_rollout, "exhausted_event_budget", False)
    object.__setattr__(corrupt_rollout, "diagnostics", corrupt_diagnostics)
    torch.save(
        {"signature": signature, "rollouts": (corrupt_rollout, corrupt_rollout)},
        path,
    )
    with pytest.raises(ValueError, match="trajectory diagnostics"):
        load_reusable_rollouts(path, signature=signature, samples=2)
