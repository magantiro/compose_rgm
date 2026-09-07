"""Two fixed engineering witnesses; no learned law or oracle is used."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.ring_construction_probe import ProbeRuntime, run_probe, topology_witness
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert
from compose_v4.rewrite.trace_shard import decode_state


@pytest.fixture(scope="module")
def outcomes():
    return {topology: run_probe(topology) for topology in ("pendant", "fused")}


@pytest.mark.parametrize("topology", ["pendant", "fused"])
def test_existing_constructors_have_exact_valid_aromatic_witnesses(outcomes, topology):
    result = outcomes[topology]
    assert result["status"] == "verified_witness", result["constructor"]
    assert result["topology_witness"]["passed"]
    assert result["work"]["executor_applications"] <= 256
    current = decode_state(result["initial_state"])
    for step in result["execution_path"]:
        before, after = decode_state(step["source"]), decode_state(step["product"])
        assert exact_graph_key(before) == exact_graph_key(current)
        assert is_valid_state(after)
        assert charge_policy_preserved(before, after)
        current = after
    assert canonical_state_key(current) == result["constructor"]["smiles"]
    receipts = result["executor_receipts"]
    assert sorted(r["call_index"] for r in receipts) == list(
        range(result["work"]["executor_applications"])
    )


@pytest.mark.parametrize("topology,wrong", [("pendant", "fused"), ("fused", "pendant")])
def test_counts_alone_cannot_substitute_the_wrong_topology(outcomes, topology, wrong):
    result = outcomes[topology]
    source = decode_state(result["initial_state"])
    product = decode_state(result["execution_path"][-1]["product"])
    assert not topology_witness(source, product, wrong)["passed"]


def test_nonadjacent_old_attachments_are_not_a_fused_edge():
    source = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 48)
    system = editing_v2_rewrite_system()
    current, tip = source, 0
    for slot in range(6, 10):
        current = system.apply(
            current, "atom_insert", AtomInsert(slot, ELEMENT_TO_IDX["C"], 0, 3, ((tip, 1),))
        )
        tip = slot
    product = system.apply(current, "bond_insert", BondInsert(tip, 2, 1))
    audit = topology_witness(source, product, "fused")
    assert audit["deltas"]["cycle_rank"] == 1
    assert audit["deltas"]["ring_systems"] == 0
    assert not audit["checks"]["requested_attachment"]
    assert not audit["passed"]


def test_interrupted_construction_is_not_chemical_unsat():
    result = run_probe("pendant", limit=2)
    assert result["status"] == "budget_abstention"
    assert result["work"]["executor_applications"] == 2
    assert len(result["executor_receipts"]) == 2
    assert result["topology_witness"] is None


def test_deadline_abstains_before_new_executor_work():
    result = run_probe("fused", safety_seconds=0)
    assert result["status"] == "timeout_incomplete"
    assert result["work"]["executor_applications"] == 0


def test_uniform_candidate_cache_preserves_exact_slot_identity():
    source = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 8)
    permutation = np.array([2, 1, 0, 3, 4, 5, 6, 7])
    permuted = replace(
        source,
        atom_types=source.atom_types[permutation],
        bonds=source.bonds[np.ix_(permutation, permutation)],
        formal_charges=source.formal_charges[permutation],
        implicit_h_counts=source.implicit_h_counts[permutation],
    )
    runtime = ProbeRuntime(source, 4, 30)
    assert runtime.enumerate(source) is runtime.enumerate(source)
    runtime.enumerate(permuted)
    assert canonical_state_key(source) == canonical_state_key(permuted)
    assert len(runtime.cache) == 2
    assert len(runtime.enumerated_counts) == 2
    assert runtime.cache_hits == 1
    assert runtime.meter.calls == 0


def test_fixed_probe_repeats_structural_receipts_exactly(outcomes):
    first, second = deepcopy(outcomes["fused"]), run_probe("fused")
    first["work"].pop("wall_seconds")
    second["work"].pop("wall_seconds")
    assert first == second


def test_bad_probe_inputs_fail_loudly():
    with pytest.raises(ValueError, match="topology"):
        run_probe("unknown")
    source = pad_molecular_graph(smiles_to_molecular_graph("C"), 8)
    with pytest.raises(ValueError, match="finite"):
        ProbeRuntime(source, 1, float("nan"))
