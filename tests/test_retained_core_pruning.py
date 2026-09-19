from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.retained_core_pruning import enumerate_retained_core_prunes


def test_retained_core_pruning_exactly_removes_a_pendant_complement():
    source = pad_molecular_graph(smiles_to_molecular_graph("CCOC(=O)NCC"), 48)
    proposals = enumerate_retained_core_prunes(
        source,
        maximum_fragment_atoms=4,
        maximum_stages=1,
        maximum_primitives=8,
    )
    endpoints = {molecular_graph_to_smiles(row.product) for row in proposals}
    assert endpoints
    assert any(sum(stage["deleted_atoms"] for stage in row.stages) >= 2 for row in proposals)
    assert all(row.actions and row.stages for row in proposals)


def test_retained_core_pruning_composes_two_protected_deletions():
    source = pad_molecular_graph(smiles_to_molecular_graph("CCN(CC)CC"), 48)
    one = enumerate_retained_core_prunes(
        source, maximum_fragment_atoms=2, maximum_stages=1
    )
    two = enumerate_retained_core_prunes(
        source, maximum_fragment_atoms=2, maximum_stages=2
    )
    assert len(two) > len(one)
    assert any(len(row.stages) == 2 for row in two)


def test_retained_core_pruning_rejects_silent_support_expansion():
    source = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 48)
    try:
        enumerate_retained_core_prunes(source, maximum_primitives=33)
    except ValueError as error:
        assert "maximum_primitives" in str(error)
    else:
        raise AssertionError("support above 32 primitives must fail explicitly")
