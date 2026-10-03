from __future__ import annotations

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.conditioned_mark_law import conditioned_mark_law
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, AtomRestate
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog_from_paths


def graph(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)


@pytest.fixture
def model():
    records = build_tracelet_path_records(("CCO", "C1CCCCC1"), n_slots=16)
    catalog = build_typed_ring_catalog_from_paths(tuple(row.path for row in records))
    torch.manual_seed(1204)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=2,
        mark_dim=8,
        scaffold_conditioning=True,
    ).eval()


def test_conditioned_law_is_normalized_and_all_actions_execute(model):
    state = graph("CC")
    context = ScaffoldContext.from_source(graph("C"), (0,))
    law = conditioned_mark_law(
        model,
        state,
        0.35,
        allowed_families=("atom_insert",),
        condition_id="grow-from-slot-1",
        action_predicate=lambda family, action: (
            family == "atom_insert"
            and isinstance(action, AtomInsert)
            and action.neighbors
            and action.neighbors[0][0] == 1
        ),
        scaffold_context=context,
    )
    assert law.candidates
    assert sum(item.conditioned_probability for item in law.candidates) == pytest.approx(1.0)
    assert 0.0 < law.conditioning_mass <= 1.0
    system = context.rewrite_system(de_novo_rewrite_system())
    for candidate in law.candidates:
        assert candidate.rule_name == "atom_insert"
        system.apply(state, candidate.rule_name, candidate.action)


def test_conditioned_sample_reports_exact_candidate_density(model):
    state = graph("CCC")
    context = ScaffoldContext.from_source(graph("C"), (0,))
    law = conditioned_mark_law(
        model,
        state,
        0.4,
        allowed_families=("atom_restate",),
        condition_id="restate-last-atom",
        action_predicate=lambda _family, action: (
            isinstance(action, AtomRestate) and action.v == 2
        ),
        scaffold_context=context,
    )
    draw = law.sample(np.random.default_rng(52))
    match = next(item for item in law.candidates if item.action == draw.action)
    assert draw.condition_id == law.condition_id
    assert draw.base_log_probability == match.base_log_probability
    assert draw.conditioned_log_probability == match.conditioned_log_probability
    assert draw.conditioning_log_mass == law.conditioning_log_mass


def test_empty_or_unsupported_condition_is_explicit(model):
    state = graph("CC")
    context = ScaffoldContext.from_source(graph("C"), (0,))
    with pytest.raises(ValueError, match="zero model support"):
        conditioned_mark_law(
            model,
            state,
            0.2,
            allowed_families=("atom_insert",),
            condition_id="empty",
            action_predicate=lambda _family, _action: False,
            scaffold_context=context,
        )
    with pytest.raises(ValueError, match="lack a public finite discrete law"):
        conditioned_mark_law(
            model,
            state,
            0.2,
            allowed_families=("ring_system_grow",),
            condition_id="unsupported",
            scaffold_context=context,
        )
