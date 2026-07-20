from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from compose_v4.experiments.factorized_mark_conditional import (
    attach_property_conditions,
    sample_factorized_mark_batch,
)
from compose_v4.experiments.property_conditioned_sampling import (
    PropertyConditionedRewriteSampler,
)
from compose_v4.experiments.molecular_property_conditioning import (
    fit_property_condition_normalizer,
    standardized_record_conditions,
)
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    factorized_mark_bregman_loss,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from scripts.evaluate_property_conditioned_rollouts import _rank_correlation


def _case():
    records = build_tracelet_path_records(
        ("CCO", "CCN"),
        n_slots=12,
        typed_ring_payloads=True,
    )
    catalog = build_typed_ring_catalog(
        (record.path.trace for record in records),
        max_cycle_templates=16,
        max_attach_templates=16,
        max_ear_templates=16,
    )
    conditions = {
        record.target_key: (float(index),)
        for index, record in enumerate(records)
    }
    return records, catalog, conditions


def test_property_conditions_are_deterministic_and_classifier_free_droppable() -> None:
    records, catalog, conditions = _case()
    conditioned = sample_factorized_mark_batch(
        records,
        batch_size=8,
        seed=73,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
        target_property_conditions=conditions,
        condition_dropout_probability=0.25,
    )
    replay = sample_factorized_mark_batch(
        records,
        batch_size=8,
        seed=73,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
        target_property_conditions=conditions,
        condition_dropout_probability=0.25,
    )
    assert torch.equal(
        conditioned.property_condition_values,
        replay.property_condition_values,
    )
    assert torch.equal(
        conditioned.property_condition_mask,
        replay.property_condition_mask,
    )

    dropped = sample_factorized_mark_batch(
        records,
        batch_size=8,
        seed=73,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
        target_property_conditions=conditions,
        condition_dropout_probability=1.0,
    )
    assert dropped.property_condition_values is not None
    assert dropped.property_condition_mask is not None
    assert not bool(dropped.property_condition_mask.any())


def test_cached_batch_condition_upgrade_matches_direct_sampling() -> None:
    records, catalog, conditions = _case()
    base = sample_factorized_mark_batch(
        records,
        batch_size=8,
        seed=31,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
    )
    direct = sample_factorized_mark_batch(
        records,
        batch_size=8,
        seed=31,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
        target_property_conditions=conditions,
        condition_dropout_probability=0.25,
    )
    upgraded = attach_property_conditions(
        base,
        records,
        seed=31,
        target_property_conditions=conditions,
        condition_dropout_probability=0.25,
    )
    assert torch.equal(
        upgraded.property_condition_values,
        direct.property_condition_values,
    )
    assert torch.equal(
        upgraded.property_condition_mask,
        direct.property_condition_mask,
    )


def test_endpoint_properties_are_standardized_from_training_targets() -> None:
    records, _, _ = _case()
    normalizer = fit_property_condition_normalizer(
        records,
        ("qed", "logp", "molecular_weight"),
    )
    conditions = standardized_record_conditions(records, normalizer)
    matrix = np.asarray(tuple(conditions.values()), dtype=np.float64)
    assert np.allclose(matrix.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(matrix.std(axis=0, ddof=1), 1.0, atol=1e-7)
    assert normalizer.to_dict()["names"] == ["qed", "logp", "molecular_weight"]


def test_condition_adapter_is_a_zero_initialized_residual_with_gradient() -> None:
    records, catalog, conditions = _case()
    batch = sample_factorized_mark_batch(
        records,
        batch_size=4,
        seed=79,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        ring_catalog=catalog,
        target_property_conditions=conditions,
        condition_dropout_probability=0.0,
    )
    torch.manual_seed(11)
    base = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    conditioned = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
        property_condition_dim=1,
    )
    compatible = {
        name: value
        for name, value in base.state_dict().items()
        if name in conditioned.state_dict()
        and conditioned.state_dict()[name].shape == value.shape
    }
    conditioned.load_state_dict(compatible, strict=False)

    unconditioned_batch = replace(
        batch,
        property_condition_values=None,
        property_condition_mask=None,
    )
    base_prediction = base.forward_mark_batch(unconditioned_batch)
    initial_prediction = conditioned.forward_mark_batch(batch)
    assert torch.allclose(base_prediction.total_hazard, initial_prediction.total_hazard)
    assert torch.allclose(
        base_prediction.family_log_probabilities,
        initial_prediction.family_log_probabilities,
    )

    loss = factorized_mark_bregman_loss(initial_prediction, batch)
    loss.backward()
    assert conditioned.property_condition_encoder is not None
    last_layer = conditioned.property_condition_encoder[-1]
    assert last_layer.weight.grad is not None
    assert float(last_layer.weight.grad.abs().sum()) > 0.0


def test_property_conditioned_sampler_uses_only_the_models_legal_support() -> None:
    records, catalog, _ = _case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
        property_condition_dim=1,
    )
    sampler = PropertyConditionedRewriteSampler(model, (0.5,))
    state = records[0].path.trace.source
    mark = sampler.sample_rewrite_mark(state, 0.5, np.random.default_rng(83))
    assert mark.total_hazard >= 0.0
    assert isinstance(mark.rule_name, str)

    with pytest.raises(ValueError, match="wrong dimension"):
        PropertyConditionedRewriteSampler(model, (0.1, 0.2))


def test_unconditional_model_rejects_accidental_property_inputs() -> None:
    _, catalog, _ = _case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    with pytest.raises(ValueError, match="cannot accept property"):
        model.sample_rewrite_mark_conditioned(
            _case()[0][0].path.trace.source,
            0.5,
            np.random.default_rng(89),
            property_values=(0.0,),
            property_mask=(True,),
        )


def test_property_rollout_rank_correlation_averages_tied_targets() -> None:
    requested = [0.3, 0.3, 0.7, 0.7]
    achieved = [0.1, 0.2, 0.8, 0.9]
    assert _rank_correlation(requested, achieved) == pytest.approx(
        0.8944271909999159
    )
