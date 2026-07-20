from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    CanonicalHistoryThinningSampler,
    FrozenExactRDKitResidualAdapter,
    FrozenExactRDKitResidualConfig,
    FrozenQEDResidualAdapter,
    FrozenQEDResidualConfig,
    MarkedRateRecord,
    aggregate_canonical_successor_rates,
    build_calibrated_pancake_quotient_target,
    build_pancake_quotient_state_features,
    canonical_successor_distillation_excess,
    predict_factorized_quotient_rates,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    SampledRewriteMark,
    _legacy_prequotient_graft_tables,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


def _empty_catalog() -> TypedRingCatalog:
    return TypedRingCatalog((), (), ())


def _raw_graft_records(smiles: str = "CCCCCCCC"):
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)
    raw_mask, removed_neighbors = _legacy_prequotient_graft_tables(state)
    records = []
    for moved, target in np.argwhere(raw_mask):
        removed = int(removed_neighbors[moved, target])
        records.append(
            MarkedRateRecord(
                "bond_reroute",
                BondReroute(
                    a=int(moved),
                    b=removed,
                    u=int(moved),
                    v=int(target),
                ),
                1.0,
            )
        )
    return state, tuple(records)


def test_explicit_graft_quotient_matches_compiled_small_state_equivalence() -> None:
    state, records = _raw_graft_records()
    aggregated = aggregate_canonical_successor_rates(state, records)
    support = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
    )

    assert len(records) == 42
    assert aggregated.virtual_self_rate == pytest.approx(12.0)
    assert aggregated.productive_total_rate == pytest.approx(30.0)
    assert aggregated.input_total_rate == pytest.approx(42.0)
    assert aggregated.mass_balance_error == pytest.approx(0.0)
    assert len(aggregated.successor_rates) == 4
    assert max(aggregated.successor_alias_counts.values()) > 1
    assert int(support.graft_mask[0].sum()) == 30
    assert len(
        torch.unique(
            support.graft_successor_groups[0][support.graft_mask[0]],
        )
    ) == len(aggregated.successor_rates)


def test_family_mass_is_partitioned_not_renormalized() -> None:
    state, records = _raw_graft_records()
    previous_key = next(
        key
        for key, count in aggregate_canonical_successor_rates(
            state,
            records,
        ).successor_alias_counts.items()
        if count > 1
    )
    aggregated = aggregate_canonical_successor_rates(
        state,
        records,
        previous_state_key=previous_key,
    )

    assert aggregated.virtual_self_rate == pytest.approx(12.0)
    assert aggregated.virtual_backtrack_rate > 1.0
    assert aggregated.input_total_rate == pytest.approx(
        aggregated.productive_total_rate
        + aggregated.virtual_self_rate
        + aggregated.virtual_backtrack_rate
    )
    assert aggregated.input_family_rates["bond_reroute"] == pytest.approx(42.0)
    assert aggregated.productive_family_rates["bond_reroute"] < 30.0


def test_calibrated_teacher_target_preserves_productive_family_mass() -> None:
    torch.manual_seed(17)
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(),
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    target = build_calibrated_pancake_quotient_target(teacher, state, 0.5)
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]
    delete_index = MARK_RULE_TO_INDEX["atom_delete"]

    assert target.raw_graft_mark_count == 42
    assert target.productive_graft_mark_count == 30
    assert len(target.graft_successor_keys) == 4
    assert torch.allclose(
        target.productive_family_rates[graft_index],
        target.graft_successor_rates.sum(),
    )
    assert target.productive_family_rates[graft_index] < target.raw_family_rates[
        graft_index
    ]
    assert target.productive_family_rates[delete_index] < target.raw_family_rates[
        delete_index
    ]
    assert float(target.mass_balance_error) < 1e-6


def test_analytic_quotient_sampler_matches_target_and_never_commits_self() -> None:
    torch.manual_seed(19)
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(),
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    reference = build_calibrated_pancake_quotient_target(teacher, state, 0.5)
    sampler = AnalyticPancakeQuotientSampler(teacher)
    analytic = sampler.rate_table(state, 0.5)
    features = build_pancake_quotient_state_features(teacher, state, 0.5)
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]

    assert torch.allclose(
        analytic.productive_family_rates,
        reference.productive_family_rates,
        rtol=1e-5,
        atol=1e-7,
    )
    assert torch.allclose(
        analytic.graft_successor_rates,
        reference.graft_successor_rates,
        rtol=1e-5,
        atol=1e-7,
    )
    assert torch.allclose(
        features.productive_survival_fractions[graft_index],
        reference.productive_family_rates[graft_index]
        / reference.raw_family_rates[graft_index],
        rtol=1e-5,
        atol=1e-7,
    )

    source_key = canonical_state_key(state)
    runtime = de_novo_rewrite_system()
    rng = np.random.default_rng(91)
    for _ in range(100):
        sampled = sampler.sample_rewrite_mark(state, 0.5, rng)
        assert sampled.total_hazard == pytest.approx(
            float(reference.productive_total_hazard),
            rel=1e-5,
        )
        assert sampled.action is not None
        successor = runtime.apply(state, sampled.rule_name, sampled.action)
        assert canonical_state_key(successor) != source_key


def test_frozen_qed_sidecar_is_zero_identity_and_excludes_base_weights() -> None:
    torch.manual_seed(101)
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(),
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    base = AnalyticPancakeQuotientSampler(teacher)
    adapter = FrozenQEDResidualAdapter(
        base,
        FrozenQEDResidualConfig(
            qed_mean=0.75,
            qed_standard_deviation=0.05,
            hidden_dim=12,
        ),
    )
    reference = base.rate_table(state, 0.5)
    conditioned = adapter.rate_table(state, 0.5, target_qed=0.9)

    assert torch.allclose(
        conditioned.family_rates,
        reference.productive_family_rates,
        rtol=0.0,
        atol=1e-8,
    )
    assert torch.allclose(
        conditioned.total_hazard,
        reference.productive_total_hazard,
        rtol=0.0,
        atol=1e-8,
    )
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all("base" not in key for key in adapter.state_dict())
    assert adapter.sidecar_contract()["base_weights_in_sidecar_state_dict"] is False

    rng_base = np.random.default_rng(8)
    rng_conditioned = np.random.default_rng(8)
    for _ in range(20):
        sampled_base = base.sample_rewrite_mark(state, 0.5, rng_base)
        sampled_conditioned = adapter.sample_rewrite_mark_conditioned(
            state,
            0.5,
            rng_conditioned,
            target_qed=0.9,
        )
        assert sampled_conditioned.total_hazard == pytest.approx(
            sampled_base.total_hazard,
            rel=0.0,
            abs=1e-7,
        )
        assert sampled_conditioned.rule_name == sampled_base.rule_name
        assert sampled_conditioned.action == sampled_base.action


def test_frozen_qed_sidecar_missing_condition_remains_exact_identity_after_update() -> None:
    torch.manual_seed(103)
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(),
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    base = AnalyticPancakeQuotientSampler(teacher)
    adapter = FrozenQEDResidualAdapter(
        base,
        FrozenQEDResidualConfig(
            qed_mean=0.75,
            qed_standard_deviation=0.05,
            hidden_dim=12,
        ),
    )
    with torch.no_grad():
        adapter.family_log_hazard_residual[-1].bias.fill_(0.4)
        adapter.node_residual[-1].bias.fill_(0.2)
        adapter.global_residual[-1].bias.fill_(-0.1)
        adapter.pair_residual[-1].bias.fill_(0.3)

    reference = base.rate_table(state, 0.25)
    missing = adapter.rate_table(state, 0.25, target_qed=None)
    present = adapter.rate_table(state, 0.25, target_qed=0.9)
    assert torch.allclose(
        missing.family_rates,
        reference.productive_family_rates,
        rtol=0.0,
        atol=1e-8,
    )
    assert not torch.allclose(present.family_rates, missing.family_rates)

    sampled = base.sample_rewrite_mark(state, 0.25, np.random.default_rng(13))
    prediction, batch = adapter.forward_mark_example(
        state,
        0.25,
        target_qed=0.9,
        teacher_rule_name=sampled.rule_name,
        teacher_action=sampled.action,
        teacher_rate=sampled.total_hazard,
    )
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in adapter.parameters()
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_exact_rdkit_property_facade_reuses_frozen_sidecar_contract() -> None:
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(), hidden_dim=16, message_passing_steps=1
    ).eval()
    adapter = FrozenExactRDKitResidualAdapter(
        AnalyticPancakeQuotientSampler(teacher),
        FrozenExactRDKitResidualConfig(
            property_name="molecular_weight",
            mean=350.0,
            standard_deviation=75.0,
            hidden_dim=12,
        ),
    )
    table = adapter.rate_table_target(state, 0.4, target_value=450.0)
    reference = adapter.base_sampler.rate_table(state, 0.4)
    assert torch.allclose(table.family_rates, reference.productive_family_rates)
    assert adapter.sidecar_contract()["property_names"] == ["molecular_weight"]
    assert adapter.sidecar_contract()["base_parameters_frozen"] is True


def test_tiny_distillation_matches_teacher_rates_without_topology_scalar() -> None:
    torch.manual_seed(23)
    state, _ = _raw_graft_records()
    teacher = FactorizedTraceletRateModel(
        _empty_catalog(),
        hidden_dim=16,
        message_passing_steps=1,
        ring_family_mass_mode="boolean",
    ).eval()
    student = copy.deepcopy(teacher).train()
    target = build_calibrated_pancake_quotient_target(teacher, state, 0.5)
    optimizer = torch.optim.Adam(
        (
            *student.total_hazard_head.parameters(),
            *student.family_head.parameters(),
        ),
        lr=0.03,
    )

    initial_prediction = predict_factorized_quotient_rates(student, state, 0.5)
    initial_excess = float(
        canonical_successor_distillation_excess(
            initial_prediction,
            target,
        ).detach()
    )
    for _ in range(100):
        prediction = predict_factorized_quotient_rates(student, state, 0.5)
        loss = canonical_successor_distillation_excess(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = predict_factorized_quotient_rates(student, state, 0.5)
    final_excess = float(
        canonical_successor_distillation_excess(final, target).detach()
    )

    assert initial_excess > 1e-4
    assert final_excess < initial_excess * 0.01
    assert float(
        (final.family_rates - target.productive_family_rates).abs().sum().detach()
    ) < 1e-3
    assert float(
        (
            final.graft_successor_rates - target.graft_successor_rates
        ).abs().sum().detach()
    ) < 1e-3
    assert student.ring_family_mass_mode == "boolean"


class _UnusedSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return SampledRewriteMark(0.0, "<TERMINAL>", None)


def test_history_thinning_blocks_canonical_self_without_changing_hazard() -> None:
    state, records = _raw_graft_records()
    runtime = de_novo_rewrite_system()
    self_record = next(
        record
        for record in records
        if canonical_state_key(
            runtime.apply(state, record.rule_name, record.action)
        )
        == canonical_state_key(state)
    )
    sampler = CanonicalHistoryThinningSampler(_UnusedSampler())
    sampler.begin_rollout_audit()
    sampled = sampler.resample_rewrite_mark_after_event(
        state,
        0.2,
        np.random.default_rng(0),
        SampledRewriteMark(9.0, self_record.rule_name, self_record.action),
    )
    audit = sampler.end_rollout_audit()

    assert sampled.rule_name == "<VIRTUAL_CANONICAL_SELF>"
    assert sampled.action is None
    assert sampled.total_hazard == pytest.approx(9.0)
    assert audit["self_rejections"] == 1
    assert audit["accepted_events"] == 0


def test_history_thinning_blocks_reverse_jump_without_changing_hazard() -> None:
    runtime = de_novo_rewrite_system()
    state_a = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 8)
    forward_action = BondReroute(a=0, b=1, u=0, v=2)
    state_b = runtime.apply(state_a, "bond_reroute", forward_action)
    reverse_action = BondReroute(a=0, b=2, u=0, v=1)
    assert canonical_state_key(state_a) != canonical_state_key(state_b)
    assert canonical_state_key(
        runtime.apply(state_b, "bond_reroute", reverse_action)
    ) == canonical_state_key(state_a)

    sampler = CanonicalHistoryThinningSampler(_UnusedSampler())
    sampler.begin_rollout_audit()
    forward = sampler.resample_rewrite_mark_after_event(
        state_a,
        0.3,
        np.random.default_rng(1),
        SampledRewriteMark(7.0, "bond_reroute", forward_action),
    )
    reverse = sampler.resample_rewrite_mark_after_event(
        state_b,
        0.4,
        np.random.default_rng(2),
        SampledRewriteMark(7.0, "bond_reroute", reverse_action),
    )
    audit = sampler.end_rollout_audit()

    assert forward.action == forward_action
    assert reverse.rule_name == "<VIRTUAL_IMMEDIATE_BACKTRACK>"
    assert reverse.action is None
    assert reverse.total_hazard == pytest.approx(7.0)
    assert audit["accepted_events"] == 1
    assert audit["immediate_backtrack_rejections"] == 1
