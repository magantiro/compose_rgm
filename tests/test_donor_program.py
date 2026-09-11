import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.donor_program import (
    PendantCut,
    compile_transplant,
    pendant_cuts,
    transplant_plan,
)
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.pmo_donor_comparison import (
    ReferenceComponent,
    donor_candidate,
    load_contract,
)
from compose_v4.experiments.winner_paths import PathConfig
from compose_v4.rewrite.kernel import canonical_state_key


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def test_pendant_ring_exchange_replays_and_preserves_exact_source():
    source, donor = graph("CCO"), graph("Cc1ccccc1")
    before = source.bonds.copy()
    small = next(c for c in pendant_cuts(source) if c.component == (2,))
    ring = next(c for c in pendant_cuts(donor) if len(c.component) == 6)
    result = compile_transplant(source, donor, small, ring)
    assert result["status"] == "compiled"
    assert result["smiles"] == canonical_state_key(graph("CCc1ccccc1"))
    assert len(result["states"]) == result["primitive_steps"] + 1
    assert result["primitive_steps"] >= 8
    assert np.array_equal(source.bonds, before)
    node = MolecularSearchState.start(source, budget=64, root_id="fixture")
    parent = {
        "id": "fixture",
        "node": encode_search_state(node),
        "smiles": canonical_state_key(source),
        "score": 0.0,
        "chain": [],
        "primitives": 0,
    }
    candidate = donor_candidate(parent, result, "child", 0, small, ring)
    child = decode_search_state(candidate["node"])
    assert child.graph.n_real_atoms == 8
    assert child.lineage.next_id == 9
    assert 2 not in child.lineage.slot_of
    assert candidate["chain"] == ["child"]
    assert candidate["structural_change"]["n_inserted"] == 6
    json.dumps(candidate)
    unresolved = compile_transplant(source, donor, small, ring, config=PathConfig(max_expansions=1))
    assert unresolved["status"] == "search_unresolved"
    assert json.loads(json.dumps(unresolved))["best_residual"] > 0


def test_cut_validation_and_size_are_explicit():
    source, donor = graph("CC"), graph("C" * 40)
    assert pendant_cuts(graph("c1ccccc1")) == ()
    a = pendant_cuts(source)[0]
    b = max(pendant_cuts(donor), key=lambda c: len(c.component))
    longer = graph("CCC")
    leaf = next(c for c in pendant_cuts(longer) if len(c.component) == 1)
    assert transplant_plan(longer, donor, leaf, b)["status"] == "unsupported_size"
    with pytest.raises(ValueError, match="bridge"):
        transplant_plan(source, donor, PendantCut(0, 1, (0,)), b)
    assert compile_transplant(source, source, a, a)["status"] == "self_proposal"


def test_reference_component_does_not_reweight_or_consume_extra_draws():
    policy = ReferenceComponent()
    reference = np.array([0.1, 0.3, 0.6])
    row = SimpleNamespace(reference=reference)
    q, audit = policy.distribution(None, row)
    assert np.array_equal(q, reference)
    assert audit["kl"] == 0
    hierarchy = SimpleNamespace(sample_reference=lambda node, rng: int(rng.choice(3, p=reference)))
    left, right = np.random.default_rng(71), np.random.default_rng(71)
    for _ in range(20):
        assert policy.sample(hierarchy, None, left)[0] == hierarchy.sample_reference(None, right)


def test_registered_donor_comparison_contract():
    root = Path(__file__).resolve().parents[1]
    c = load_contract(root)
    assert c["compute"]["max_workers"] + c["compute"]["driver_containers"] == 30
    assert c["particles"] * c["boundaries"] * len(c["arms"]) == c["new_oracle_limit"]
