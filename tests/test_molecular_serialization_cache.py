"""Exact-state cache must not change chemistry, alias handling, or mutation semantics."""

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    SCAR_IDX,
    molecular_graph_to_smiles,
    molecular_serialization_cache,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph


@pytest.mark.parametrize(
    "smiles", ["CCO", "c1ccncc1", "c1ccc2ccccc2c1", "C[N+](C)(C)C", "CS(=O)(=O)C"]
)
def test_cached_roundtrip_matches_uncached_for_permuted_padded_states(smiles):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 40)
    order = np.random.default_rng(2).permutation(40)
    graph.atom_types = graph.atom_types[order]
    graph.formal_charges = graph.formal_charges[order]
    graph.implicit_h_counts = graph.implicit_h_counts[order]
    graph.bonds = graph.bonds[np.ix_(order, order)]
    expected = molecular_graph_to_smiles(graph)
    assert expected == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    with molecular_serialization_cache(2) as cache:
        assert molecular_graph_to_smiles(graph) == expected
        # Equal content in a new object hits; object identity is not a cache key.
        assert molecular_graph_to_smiles(pad_molecular_graph(graph, 40)) == expected
        assert (cache.hits, cache.misses, len(cache.entries)) == (1, 1, 1)
    assert not cache.entries


def test_every_mutable_array_is_part_of_the_key_and_invalid_results_are_cached():
    for field, index, value in [
        ("atom_types", 1, ELEMENT_TO_IDX["N"]),
        ("formal_charges", 1, 1),
        ("implicit_h_counts", 1, 4),
        ("bonds", (0, 1), 9),
    ]:
        graph = pad_molecular_graph(smiles_to_molecular_graph("CO"), 40)
        with molecular_serialization_cache() as cache:
            original = molecular_graph_to_smiles(graph)
            array = getattr(graph, field)
            previous = array[index]
            array[index] = value
            if field == "bonds":
                array[1, 0] = value
            changed = molecular_graph_to_smiles(graph)
            assert cache.misses == 2
            assert changed != original
            assert molecular_graph_to_smiles(graph) == changed
            assert cache.hits == 1
            array[index] = previous
            if field == "bonds":
                array[1, 0] = previous
            assert molecular_graph_to_smiles(graph) == original


def test_null_scar_and_lru_scope_cleanup():
    null = empty_molecular_graph(40)
    scar = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 40)
    scar.atom_types[1] = SCAR_IDX
    scar.implicit_h_counts[1] = 0
    bad = pad_molecular_graph(smiles_to_molecular_graph("CC"), 40)
    bad.bonds[0, 1] = bad.bonds[1, 0] = 9
    expected = [molecular_graph_to_smiles(g) for g in (null, scar, bad)]
    assert expected == ["", "CC", None]
    with molecular_serialization_cache(2) as cache:
        assert [molecular_graph_to_smiles(g) for g in (null, scar, bad)] == expected
        assert len(cache.entries) == 2
        assert molecular_graph_to_smiles(null) == ""
        assert cache.misses == 4  # oldest exact state was evicted, not retained forever
        assert len(cache.entries) == 2
        with pytest.raises(RuntimeError), molecular_serialization_cache(1) as inner:
            assert molecular_graph_to_smiles(scar) == "CC"
            raise RuntimeError("interrupted")
        assert not inner.entries
        assert molecular_graph_to_smiles(null) == ""
        assert cache.hits == 1
    assert not cache.entries
    with pytest.raises(ValueError), molecular_serialization_cache(0):
        pass
