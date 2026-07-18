from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.state import empty_molecular_graph
from compose_v4.experiments.cnof_conditional import (
    build_path_records,
    sample_factorized_ancestral,
)
from compose_v4.experiments.corpus_marginal import fit_corpus_marginal_rate_model


def test_corpus_marginal_policy_is_normalized_and_samples_legally() -> None:
    records = build_path_records(
        ("CC", "CO", "C1CC1", "C=C"),
        n_slots=4,
        seed=31,
        traces_per_molecule=2,
    )
    model = fit_corpus_marginal_rate_model(records, time_bins=16)
    state = empty_molecular_graph(4)
    prediction = model.predict_factorized_fiber(state, 0.25)
    expected_hazard = model.table.mean_path_length * 0.75
    assert torch.allclose(
        prediction.total_hazard,
        torch.tensor(expected_hazard, dtype=prediction.marked_rates.dtype),
    )
    assert all(
        rate > 0.0
        for transition, rate in zip(prediction.transitions, prediction.marked_rates)
        if transition.rule_name == "atom_insert"
    )

    endpoint = model.predict_factorized_fiber(state, 1.0)
    assert float(endpoint.total_hazard) == 0.0
    rollout = sample_factorized_ancestral(
        model,
        rng=np.random.default_rng(32),
        n_slots=4,
        operational_horizon=7.0,
    )
    assert rollout.final_state.n_atoms == 4


def test_deferred_bond_order_traces_activate_reorder_rates() -> None:
    records = build_path_records(
        ("C=C", "C#N", "c1ccccc1"),
        n_slots=7,
        seed=33,
        traces_per_molecule=2,
        defer_bond_orders=True,
    )
    model = fit_corpus_marginal_rate_model(records, time_bins=16)
    assert any(
        table.get("bond_reorder", 0.0) > 0.0
        for table in model.table.family_probabilities
    )
