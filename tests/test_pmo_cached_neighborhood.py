"""Only new adapter boundaries; executor census has its own focused regression."""

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.continuation_profile import encode_action
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.trace_shard import encode_state
from tools.pmo_cached_neighborhood import census, query_subset


def test_exact_cached_adapter_deduplicates_and_rejects_bad_mass():
    state = encode_state(pad_molecular_graph(smiles_to_molecular_graph("CC"), 48))
    marks = [encode_action("atom_delete", AtomDelete(i)) for i in (0, 1)]
    payload = {"source": state, "marks": marks, "probabilities": [0.5, 0.5]}
    result = census(payload)
    assert len(result["products"]) == 1
    assert result["products"][0]["smiles"] == "C"
    assert len(result["products"][0]["witnesses"]) == 2
    with pytest.raises(ValueError, match="probabilities"):
        census({**payload, "probabilities": [0.5, 0.6]})


def test_query_cap_is_deterministic_and_does_not_drop_support():
    products = [{"smiles": str(i)} for i in range(20)]
    known = {"0": 1.0}
    selected = query_subset(products, known, limit=8)
    assert selected == query_subset(list(reversed(products)), known, limit=8)
    assert len(selected) == len(set(selected)) == 8
    assert "0" not in selected
    assert len(products) == 20
    assert len(query_subset(products, known, limit=30)) == 19
