"""Actual-executor inverse curriculum fixtures, with no oracle calls."""

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.option_demonstrations import recognize_trace
from compose_v4.experiments.inverse_ring_demonstrations import inverse_ring_demonstrations
from compose_v4.experiments.whole_ring_plan import verify_trace
from compose_v4.rewrite.trace_shard import encode_state


@pytest.mark.parametrize(
    "smiles,topology,electronic",
    [
        ("CCc1ccccc1", "pendant", "aromatic"),
        ("c1ccc2ccccc2c1", "fused", "aromatic"),
        ("O=C1CCCc2ccccc21", "fused", "nonaromatic"),
    ],
)
def test_inverse_ring_reconstructs_exact_endpoint(smiles, topology, electronic):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    result = inverse_ring_demonstrations(graph)
    matching = [
        example
        for example in result["examples"]
        if f":{topology}:" in example["option"] and f":{electronic}:" in example["option"]
    ]
    assert matching, result["attempts"]
    for example in matching:
        assert verify_trace(
            encode_state(graph),
            [example["reverse"], example["forward"], example["redecoration"]],
            result["target"],
        )["exact_endpoint"]
        segments = recognize_trace(example["forward"]["states"], example["forward"]["actions"])
        assert len(segments) == 1 and segments[0].option == example["option"]
        if smiles.startswith("O="):
            assert example["redecoration"]["primitive_edits"] == 1


def test_standalone_ring_is_not_misreported_as_attachable():
    graph = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 48)
    result = inverse_ring_demonstrations(graph)
    assert not result["examples"]
    assert result["attempts"]
