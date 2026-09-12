"""The existing timed sampler must consume the same conditional mark law."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import FixedMolecularStatePrior
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    sample_tracelet_ancestral,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.kernel import InvalidRewrite
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog_from_paths


@pytest.fixture
def model():
    paths = build_tracelet_path_records(("CCO", "C1CC1"), n_slots=12)
    catalog = build_typed_ring_catalog_from_paths([record.path for record in paths])
    torch.manual_seed(908330)
    return FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=2, mark_dim=8, scaffold_conditioning=True
    ).eval()


@pytest.mark.parametrize("smiles,ports", [("", ()), ("NCCO", (3,))])
def test_context_forwarding_preserves_existing_clock_and_rng(model, smiles, ports):
    source = (
        pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)
        if smiles
        else empty_molecular_graph(12)
    )
    context = ScaffoldContext.from_source(source, ports)
    prior = FixedMolecularStatePrior(source)
    calls = []

    def direct(state, time, rng):
        calls.append((context.identity, time))
        assert context.accepts(state)
        return model.sample_rewrite_mark_conditioned(
            state, time, rng, property_values=None, scaffold_context=context
        )

    proxy = SimpleNamespace(sample_rewrite_mark=direct)
    kwargs = {
        "n_slots": 12,
        "operational_horizon": 1.0,
        "time_step": 0.1,
        "max_events": 6,
        "source_prior": prior,
    }
    with torch.no_grad():
        expected = sample_tracelet_ancestral(proxy, rng=np.random.default_rng(77), **kwargs)
        observed = sample_tracelet_ancestral(
            model, rng=np.random.default_rng(77), scaffold_context=context, **kwargs
        )
    assert calls
    assert observed.event_times == expected.event_times
    assert observed.event_rules == expected.event_rules
    assert context.accepts(observed.final_state)
    for name in ("atom_types", "formal_charges", "implicit_h_counts", "bonds"):
        np.testing.assert_array_equal(
            getattr(expected.final_state, name), getattr(observed.final_state, name)
        )


def test_conditioned_model_cannot_silently_run_without_condition(model):
    with pytest.raises(ValueError, match="explicit condition"):
        sample_tracelet_ancestral(model, rng=np.random.default_rng(0), n_slots=12)


def test_incompatible_source_rejected_before_sampling(model):
    source = pad_molecular_graph(smiles_to_molecular_graph("NCCO"), 12)
    context = ScaffoldContext.from_source(source, (3,))
    with pytest.raises(ValueError, match="source does not satisfy"):
        sample_tracelet_ancestral(
            model, rng=np.random.default_rng(0), n_slots=12, scaffold_context=context
        )


def test_executor_enforces_condition_at_every_committed_event():
    source = pad_molecular_graph(smiles_to_molecular_graph("NCCO"), 12)
    context = ScaffoldContext.from_source(source, (3,))

    def illegal(*args, **kwargs):
        return SimpleNamespace(total_hazard=10000.0, rule_name="atom_delete", action=AtomDelete(0))

    model = SimpleNamespace(scaffold_conditioning=True, sample_rewrite_mark_conditioned=illegal)
    with pytest.raises(InvalidRewrite):
        sample_tracelet_ancestral(
            model,
            rng=np.random.default_rng(0),
            n_slots=12,
            scaffold_context=context,
            source_prior=FixedMolecularStatePrior(source),
        )
