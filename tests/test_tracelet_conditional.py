from __future__ import annotations

import numpy as np
import pytest
import torch

from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.tracelet_conditional import (
    _cpu_byte_rng_state,
    _sample_tracelet_progress,
    build_tracelet_path_records,
    build_tree_transport_path_records,
    sample_tracelet_ancestral,
    sample_tracelet_conditional_batch,
    tracelet_conditional_batch_loss,
    tracelet_conditional_metrics,
    train_tracelet_conditional_model,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.tracelet_corpus_marginal import (
    fit_tracelet_corpus_marginal_rate_model,
)
from compose_v4.experiments.tracelet_prior_tilted import (
    PriorTiltedTraceletRateModel,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel


SMILES = ("c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CC2CCC1C2")


def test_serialized_rng_state_is_normalized_for_torch_restore() -> None:
    serialized = torch.tensor([1, 2, 255], dtype=torch.int64)

    restored = _cpu_byte_rng_state(serialized)

    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    assert restored.is_contiguous()
    assert restored.tolist() == [1, 2, 255]


def test_serialized_rng_state_rejects_non_tensor_payload() -> None:
    with pytest.raises(ValueError, match="must be tensors"):
        _cpu_byte_rng_state([1, 2, 3])


def test_training_recovery_resumes_exactly() -> None:
    records = build_tracelet_path_records(("CCO", "CCN"), n_slots=6)
    validation_examples = sample_tracelet_conditional_batch(
        records,
        batch_size=2,
        rng=np.random.default_rng(10),
        fiber_cache={},
    )
    torch.manual_seed(99)
    template = TraceletRateModel(hidden_dim=8, message_passing_steps=1)
    initial_state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }

    uninterrupted = TraceletRateModel(hidden_dim=8, message_passing_steps=1)
    uninterrupted.load_state_dict(initial_state)
    full_history, full_best = train_tracelet_conditional_model(
        uninterrupted,
        records,
        validation_examples,
        steps=4,
        batch_size=1,
        learning_rate=1e-3,
        seed=11,
        fiber_cache={},
        evaluation_points=4,
    )

    interrupted = TraceletRateModel(hidden_dim=8, message_passing_steps=1)
    interrupted.load_state_dict(initial_state)
    captured: dict[str, object] = {}

    class ExpectedInterruption(RuntimeError):
        pass

    def interrupt_after_two_steps(state: dict[str, object]) -> None:
        captured.update(state)
        raise ExpectedInterruption

    with pytest.raises(ExpectedInterruption):
        train_tracelet_conditional_model(
            interrupted,
            records,
            validation_examples,
            steps=4,
            batch_size=1,
            learning_rate=1e-3,
            seed=11,
            fiber_cache={},
            evaluation_points=4,
            checkpoint_interval=2,
            checkpoint_callback=interrupt_after_two_steps,
        )
    assert captured["completed_steps"] == 2

    resumed = TraceletRateModel(hidden_dim=8, message_passing_steps=1)
    resumed_history, resumed_best = train_tracelet_conditional_model(
        resumed,
        records,
        validation_examples,
        steps=4,
        batch_size=1,
        learning_rate=1e-3,
        seed=11,
        fiber_cache={},
        evaluation_points=4,
        resume_state=captured,
    )

    assert resumed_history == full_history
    assert resumed_best == full_best
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)


def test_tracelet_prior_has_smoothed_support_and_exact_initial_tilt() -> None:
    records = build_tracelet_path_records(SMILES, n_slots=12)
    prior = fit_tracelet_corpus_marginal_rate_model(records, time_bins=8)
    example = sample_tracelet_conditional_batch(
        records,
        batch_size=1,
        rng=np.random.default_rng(3),
        fiber_cache={},
    )[0]
    baseline = prior.predict_tracelet_fiber(example.state, example.time, fiber=example.fiber)
    tilted = PriorTiltedTraceletRateModel(
        prior,
        hidden_dim=16,
        message_passing_steps=1,
    ).predict_tracelet_fiber(example.state, example.time, fiber=example.fiber)

    assert bool((baseline.marked_rates > 0).all())
    assert torch.allclose(tilted.marked_rates.cpu(), baseline.marked_rates, atol=1e-6)


def test_tracelet_loss_and_target_free_ancestral_sampler_are_executable() -> None:
    torch.manual_seed(4)
    records = build_tracelet_path_records(SMILES, n_slots=12)
    examples = sample_tracelet_conditional_batch(
        records,
        batch_size=3,
        rng=np.random.default_rng(4),
        fiber_cache={},
    )
    model = TraceletRateModel(hidden_dim=16, message_passing_steps=1)
    loss = tracelet_conditional_batch_loss(model, examples)
    loss.backward()
    rollout = sample_tracelet_ancestral(
        model,
        rng=np.random.default_rng(5),
        n_slots=12,
        operational_horizon=0.25,
        time_step=0.1,
        max_events=4,
    )

    assert torch.isfinite(loss)
    assert is_valid_state(rollout.final_state)
    assert is_connected_or_null(rollout.final_state)


def test_tree_transport_records_train_on_primitive_delete_and_regrow_paths() -> None:
    records = build_tree_transport_path_records(
        ("CC(C)CO", "c1ccncc1"),
        n_slots=12,
        source_prior=DegreeBoundedCarbonTreePrior(sizes=(3, 8)),
        seed=41,
        couplings_per_target=2,
    )
    families = {
        step.rule_name
        for record in records
        for step in record.path.trace.steps
    }
    examples = sample_tracelet_conditional_batch(
        records,
        batch_size=32,
        rng=np.random.default_rng(42),
        fiber_cache={},
    )
    assert "atom_delete" in families
    assert {"atom_insert", "ring_ear_insert"} & families
    assert len(records) == 4
    assert examples


def test_size_matched_graft_records_use_the_target_size_without_grow_or_shrink() -> None:
    smiles = ("CC(C)CO", "c1ccncc1", "C1CCC2(CC1)CCCC2")
    records = build_tree_transport_path_records(
        smiles,
        n_slots=14,
        source_prior=DegreeBoundedCarbonTreePrior(sizes=(3, 8, 12)),
        seed=51,
        couplings_per_target=3,
        transport_mode="size_matched_graft",
    )

    assert len(records) == 9
    assert all(
        record.path.trace.source.n_real_atoms
        == record.path.trace.target.n_real_atoms
        for record in records
    )
    assert all(
        step.rule_name not in {"atom_insert", "atom_delete"}
        for record in records
        for step in record.path.trace.steps
    )
    assert any(
        step.rule_name == "bond_reroute"
        for record in records
        for step in record.path.trace.steps
    )


def test_flexible_graft_records_supervise_both_grow_and_shrink() -> None:
    records = build_tree_transport_path_records(
        ("CC(C)CO", "c1ccncc1O"),
        n_slots=12,
        source_prior=DegreeBoundedCarbonTreePrior(
            sizes=(3, 10),
            probabilities=(0.5, 0.5),
        ),
        seed=530,
        couplings_per_target=8,
        transport_mode="flexible_size_graft",
    )

    assert len(records) == 16
    deltas = {
        record.path.trace.target.n_real_atoms
        - record.path.trace.source.n_real_atoms
        for record in records
    }
    families = {
        step.rule_name
        for record in records
        for step in record.path.trace.steps
    }
    assert any(delta > 0 for delta in deltas)
    assert any(delta < 0 for delta in deltas)
    assert {"atom_insert", "atom_delete"} <= families
    assert all(
        record.path.trace.metadata["compiler"]
        == "flexible_size_graft_transport_v1"
        for record in records
    )


def test_corpus_marginal_prior_supports_graft_transport_records() -> None:
    records = build_tree_transport_path_records(
        ("CC(C)CO", "c1ccncc1"),
        n_slots=12,
        source_prior=DegreeBoundedCarbonTreePrior(sizes=(5, 6)),
        seed=52,
        couplings_per_target=2,
        transport_mode="size_matched_graft",
    )

    prior = fit_tracelet_corpus_marginal_rate_model(records, time_bins=8)
    prediction = prior.predict_tracelet_fiber(
        records[0].path.trace.source,
        0.25,
    )

    assert "bond_reroute" in prior.table.rule_families
    assert any(
        transition.rule_name == "bond_reroute"
        for transition in prediction.transitions
    )
    assert bool((prediction.marked_rates > 0).all())


def test_tree_transport_rejects_nonpositive_coupling_count() -> None:
    with pytest.raises(ValueError, match="couplings_per_target"):
        build_tree_transport_path_records(
            ("CCO",),
            n_slots=8,
            source_prior=DegreeBoundedCarbonTreePrior(sizes=(3,)),
            seed=9,
            couplings_per_target=0,
        )


def test_quotient_energy_aromatic_view_has_trainable_gm_loss() -> None:
    torch.manual_seed(12)
    records = build_tracelet_path_records(SMILES, n_slots=12)
    examples = sample_tracelet_conditional_batch(
        records,
        batch_size=3,
        rng=np.random.default_rng(12),
        fiber_cache={},
        late_time_fraction=1.0,
    )
    model = TraceletRateModel(
        hidden_dim=16,
        message_passing_steps=1,
        rate_factorization="quotient_energy",
        use_aromatic_bond_view=True,
    )
    loss = tracelet_conditional_batch_loss(model, examples)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_causal_frontier_batch_trains_against_all_enabled_successors() -> None:
    torch.manual_seed(23)
    records = build_tracelet_path_records(("CC(C)(N)O",), n_slots=12)
    examples = sample_tracelet_conditional_batch(
        records,
        batch_size=128,
        rng=np.random.default_rng(23),
        fiber_cache={},
        causal_teachers=True,
        causal_cache={},
    )
    branched = [item for item in examples if len(item.teacher_successor_rates) > 1]
    assert branched
    assert all(
        np.isclose(
            item.teacher_rate,
            sum(rate for _, rate in item.teacher_successor_rates),
        )
        for item in branched
    )

    model = TraceletRateModel(hidden_dim=16, message_passing_steps=1)
    loss = tracelet_conditional_batch_loss(model, examples)
    loss.backward()
    metrics = tracelet_conditional_metrics(model, examples)

    assert torch.isfinite(loss)
    assert np.isfinite(metrics["conditional_gm_loss"])
    assert 0.0 <= metrics["mean_teacher_successor_mass"] <= 1.0
    assert metrics["family/<CAUSAL_FRONTIER>/examples"] > 0


def test_family_stratified_progress_is_importance_corrected() -> None:
    records = build_tracelet_path_records(("c1ccc2ccccc2c1",), n_slots=12)
    path = records[0].path
    time = 0.81
    marginal = path.marginal(time)
    rng = np.random.default_rng(91)
    estimate = np.zeros_like(marginal)
    draws = 20_000
    for _ in range(draws):
        progress, weight = _sample_tracelet_progress(
            path,
            time=time,
            rng=rng,
            stratification_fraction=0.75,
        )
        estimate[progress] += weight / draws

    assert np.allclose(estimate, marginal, atol=0.01)
    assert abs(float(estimate.sum()) - 1.0) < 0.01
