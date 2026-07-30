"""Production factorized marked law -> canonical molecular successor integration.

These tests use real model masks, the real executor, and real molecular
canonicalization.  The independent dictionary implementation is imported only
here as an oracle; reported results use the segmented production path.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    canonical_successor_result,
    enumerate_factorized_marked_law,
)
from compose_v4.experiments.reference_successor_kernel import (
    compare_against_reference,
    reference_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 12
_SYSTEM = de_novo_rewrite_system()


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    def trace(smiles: str):
        target = _state(smiles)
        source = DegreeBoundedCarbonTreePrior(
            sizes=(target.n_real_atoms,)
        ).sample(np.random.default_rng(1), n_slots=_SLOTS)
        return compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )

    return build_typed_ring_catalog(
        tuple(trace(smiles) for smiles in ("c1ccccc1", "C1CCNCC1"))
    )


@pytest.fixture(scope="module")
def model(ring_catalog):
    torch.manual_seed(7)
    return FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


@pytest.mark.parametrize("smiles", ("CCO", "c1ccccc1", "C1CCNCC1"))
def test_complete_marked_law_is_normalized_and_executable(model, smiles):
    state = _state(smiles)
    law = enumerate_factorized_marked_law(model, state, 0.37)

    assert law.marks
    assert law.total_probability == pytest.approx(1.0, abs=2e-5)
    assert law.total_hazard > 0.0
    for mark in law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        assert canonical_state_key(successor)


def test_null_root_marked_law_contains_only_executable_valence_classes(model):
    state = empty_molecular_graph(_SLOTS)
    law = enumerate_factorized_marked_law(model, state, 0.37)

    assert law.marks
    assert {mark.family_name for mark in law.marks} == {"atom_insert"}
    assert law.total_probability == pytest.approx(1.0, abs=2e-5)
    for mark in law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        assert canonical_state_key(successor)


def test_segmented_production_path_matches_dictionary_oracle(model):
    state = _state("c1ccccc1")
    result = canonical_successor_result(model, state, 0.41)
    reference = reference_successor_batch(
        state,
        [
            (
                mark.executor_rule_name,
                mark.action,
                mark.probability,
            )
            for mark in result.marked_law.marks
        ],
        system=_SYSTEM,
        identity=result.batch.identity,
    )

    produced_probabilities = {
        successor.key: successor.probability
        for successor in result.batch.successors
    }
    reference_probabilities = {
        successor.key: successor.probability
        for successor in reference.successors
    }
    assert compare_against_reference(
        produced_probabilities,
        reference_probabilities,
        tolerance=2e-7,
    ) == []
    assert {
        successor.key: successor.alias_count
        for successor in result.batch.successors
    } == {
        successor.key: successor.alias_count
        for successor in reference.successors
    }
    assert result.batch.virtual_mass == pytest.approx(
        reference.virtual_mass,
        abs=2e-7,
    )


def test_alias_mass_is_summed_before_productive_conditioning(model):
    state = _state("c1ccccc1")
    result = canonical_successor_result(model, state, 0.29)
    aliased = [
        successor
        for successor in result.batch.successors
        if successor.alias_count > 1
    ]
    assert aliased, "symmetric benzene should expose at least one aliased successor"

    raw_mass_by_successor: dict[str, float] = defaultdict(float)
    raw_count_by_successor: dict[str, int] = defaultdict(int)
    for mark in result.marked_law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        key = canonical_state_key(successor)
        if key == result.batch.source_key:
            continue
        raw_mass_by_successor[key] += mark.probability
        raw_count_by_successor[key] += 1

    for successor in aliased:
        assert successor.alias_count == raw_count_by_successor[successor.key]
        assert successor.probability == pytest.approx(
            raw_mass_by_successor[successor.key]
            / result.diagnostics.raw_productive_mass,
            abs=2e-7,
        )


def test_compositional_ring_close_and_open_are_both_in_the_marked_law(model):
    acyclic = enumerate_factorized_marked_law(model, _state("CCCCCC"), 0.5)
    cyclic = enumerate_factorized_marked_law(model, _state("C1CCCCC1"), 0.5)

    assert any(mark.family_name == "cycle_insert" for mark in acyclic.marks)
    assert any(mark.family_name == "cycle_attach" for mark in cyclic.marks)
    assert not any(mark.family_name == "ring_system_grow" for mark in cyclic.marks)


def test_legacy_ring_macro_fails_closed(ring_catalog):
    model = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=8,
        message_passing_steps=1,
        enable_ring_grow_macro=True,
        enable_cycle_ops=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    with pytest.raises(ProductionSuccessorKernelError, match="legacy ring_system_grow"):
        enumerate_factorized_marked_law(model, _state("CCO"), 0.5)
