from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, is_element
from compose_v4.chem.source_prior import (
    DegreeBoundedCarbonTreePrior,
    NullSourcePrior,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state


def test_null_source_prior_is_the_formal_empty_graph() -> None:
    state = NullSourcePrior().sample(np.random.default_rng(1), n_slots=8)
    assert state.n_real_atoms == 0
    assert is_valid_state(state)


def test_degree_bounded_carbon_tree_prior_is_reproducible_and_valid() -> None:
    prior = DegreeBoundedCarbonTreePrior(
        sizes=(4, 7, 10),
        probabilities=(0.2, 0.3, 0.5),
    )
    first = prior.sample(np.random.default_rng(7), n_slots=12)
    second = prior.sample(np.random.default_rng(7), n_slots=12)

    assert np.array_equal(first.atom_types, second.atom_types)
    assert np.array_equal(first.bonds, second.bonds)
    assert is_valid_state(first)
    assert is_connected_or_null(first)
    real = np.flatnonzero(is_element(first.atom_types))
    assert set(int(first.atom_types[v]) for v in real) == {ELEMENT_TO_IDX["C"]}
    assert int(np.triu(first.bonds != 0, k=1).sum()) == len(real) - 1
    assert int((first.bonds != 0).sum(axis=1).max()) <= 4


def test_tree_prior_from_size_counts_matches_declared_support() -> None:
    prior = DegreeBoundedCarbonTreePrior.from_size_counts({3: 2, 5: 6, 8: 0})
    observed = {
        prior.sample(np.random.default_rng(seed), n_slots=8).n_real_atoms
        for seed in range(40)
    }
    assert observed == {3, 5}


def test_tree_prior_can_sample_an_exact_conditional_size() -> None:
    prior = DegreeBoundedCarbonTreePrior(
        sizes=(3, 7, 11),
        probabilities=(0.2, 0.3, 0.5),
    )
    state = prior.sample_size(
        np.random.default_rng(17),
        n_slots=14,
        size=9,
    )

    assert state.n_real_atoms == 9
    assert is_valid_state(state)
    assert is_connected_or_null(state)
    real = np.flatnonzero(is_element(state.atom_types))
    assert int(np.triu(state.bonds != 0, k=1).sum()) == len(real) - 1
    assert int((state.bonds != 0).sum(axis=1).max()) <= 4
