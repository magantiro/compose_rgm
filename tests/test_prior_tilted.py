from __future__ import annotations

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cnof_conditional import build_path_records
from compose_v4.experiments.corpus_marginal import fit_corpus_marginal_rate_model
from compose_v4.experiments.prior_tilted import PriorTiltedRewriteRateModel


def test_prior_tilted_model_initializes_exactly_at_corpus_policy() -> None:
    records = build_path_records(
        ("CC", "CO", "C1CC1", "C=C"),
        n_slots=5,
        seed=41,
        traces_per_molecule=2,
    )
    prior = fit_corpus_marginal_rate_model(records, time_bins=16)
    torch.manual_seed(42)
    model = PriorTiltedRewriteRateModel(
        prior,
        hidden_dim=24,
        message_passing_steps=2,
        use_topology_context=True,
    )
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 5)
    expected = prior.predict_factorized_fiber(state, 0.45)
    actual = model.predict_factorized_fiber(state, 0.45)
    assert torch.allclose(actual.marked_rates, expected.marked_rates, atol=1e-6)
    assert actual.grouped_indices.keys() == expected.grouped_indices.keys()
    assert abs(float(actual.action_kl_to_prior.detach())) < 1e-6
    assert float(actual.squared_log_hazard_tilt.detach()) == 0.0

    endpoint = model.predict_factorized_fiber(state, 1.0)
    assert float(endpoint.total_hazard) == 0.0


def test_prior_tilt_reports_positive_trust_region_cost_after_perturbation() -> None:
    records = build_path_records(
        ("CCCC", "C1CC1", "C1CCCCC1", "CCO"),
        n_slots=7,
        seed=51,
        traces_per_molecule=2,
    )
    prior = fit_corpus_marginal_rate_model(records, time_bins=16)
    torch.manual_seed(52)
    model = PriorTiltedRewriteRateModel(
        prior,
        hidden_dim=24,
        message_passing_steps=2,
        use_topology_context=True,
    )
    with torch.no_grad():
        model.action_residual_head[-1].weight.normal_(mean=0.0, std=0.2)
        model.hazard_residual_head[-1].bias.fill_(0.25)
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 7)
    prediction = model.predict_factorized_fiber(state, 0.45)
    assert float(prediction.action_kl_to_prior.detach()) > 0.0
    assert float(prediction.squared_log_hazard_tilt.detach()) > 0.0
