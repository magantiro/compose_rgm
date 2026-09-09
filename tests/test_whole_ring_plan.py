"""Small checks for the actual program compiler, not a second chemistry implementation."""

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.ring_program import RingSpec
from compose_v4.experiments.whole_ring_plan import (
    RingRequest,
    compile_ring,
    first_winner_plan,
    verify_trace,
)
from compose_v4.rewrite.trace_shard import encode_state


def benzene():
    return pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 48)


@pytest.mark.parametrize(
    "topology,size,elements,pattern,anchors",
    [
        ("pendant", 6, ("C",) * 6, (1, 2, 1, 2, 1, 2), (0,)),
        ("pendant", 5, ("C", "C", "N", "C", "C"), (1, 1, 1, 1, 1), (0,)),
        ("fused", 6, ("C",) * 4, (1, 1, 1, 1, 1, 2), (0, 1)),
    ],
)
def test_compiled_ring_replays_exactly(topology, size, elements, pattern, anchors):
    graph = benzene()
    if topology == "fused":
        pattern = pattern[:-1] + (int(graph.bonds[anchors]),)
    counts = tuple(
        (2 if topology == "fused" and element == "C" else 0) + elements.count(element)
        for element in ("C", "N", "O")
    )
    electronic = "aromatic" if topology == "pendant" and size == 6 else "nonaromatic"
    request = RingRequest(RingSpec(topology, size, counts, electronic), anchors, elements, pattern)
    endpoint, result = compile_ring(graph, request)
    assert result["primitive_edits"] == request.spec.horizon
    assert endpoint.n_real_atoms == graph.n_real_atoms + request.spec.growth
    check = verify_trace(encode_state(graph), [result], result["endpoint"])
    assert check["exact_states"] and check["exact_endpoint"]
    corrupt = deepcopy(result)
    corrupt["states"][-1] = encode_state(graph)
    with pytest.raises(ValueError, match="exact replay"):
        verify_trace(encode_state(graph), [corrupt], result["endpoint"])


def test_incompatible_descriptors_fail_without_mutating_source():
    graph = benzene()
    before = encode_state(graph)
    request = RingRequest(
        RingSpec("fused", 6, (6, 0, 0), "aromatic"), (0, 3), ("C",) * 4, (1, 2, 1, 2, 1, 2)
    )
    with pytest.raises(ValueError, match="not applicable"):
        compile_ring(graph, request)
    assert encode_state(graph) == before
    request = RingRequest(
        RingSpec("pendant", 6, (6, 0, 0), "aromatic"), (0,), ("N",) * 6, (1, 2, 1, 2, 1, 2)
    )
    with pytest.raises(ValueError, match="descriptor contract"):
        compile_ring(graph, request)
    assert encode_state(graph) == before


def test_declared_winner_plan_exact_replay_and_determinism():
    receipt = Path(__file__).resolve().parents[1] / (
        "diagnostics/ivg_winner_paths/pairs/"
        "76467b6ddd4b5b74e571a1ba415e47a46d51c0cb94b40e03fa2059bdeab2bbae.json.gz"
    )
    if not receipt.exists():
        pytest.skip("vendored IVG development path receipt absent; see T4_WHOLE_RING_PLAN.md")
    raw = receipt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "b6e2fb41d505289e67d30c80ffcd8c1eafc8cc9b9c962e921fd84be541af675f"
    )
    path = json.loads(gzip.decompress(raw))["payload"]["path"]
    stages = first_winner_plan(path)
    assert stages == first_winner_plan(path)
    assert [s["primitive_edits"] for s in stages] == [4, 7, 5, 1, 4]
    assert verify_trace(path["source_state"], stages, path["target_2d"]) == {
        "exact_endpoint": True,
        "exact_states": True,
        "primitive_edits": 21,
        "execution_programs": 5,
    }
    assert all(stages[i]["existing_ring_contract"] for i in (1, 2))
