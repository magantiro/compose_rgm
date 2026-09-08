"""Direct expansion uses real executor edits and keeps the old controller law."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.continuation import FiniteHorizonContinuation, continuation_decision
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    exact_graph_key,
    sample_option_trajectory,
)
from compose_v4.control.option_selector import (
    OPTIONS,
    applicable_options,
    balanced_option_prior,
    conditioned_action_distribution,
)
from compose_v4.control.ring_expansion import (
    EXPAND_RING_OPTION,
    ExpansionProgress,
    completed_expansion,
    eligible_expansion_edges,
)
from compose_v4.experiments.continuation_profile import state_payload
from compose_v4.rewrite.kernel import editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert, CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state
from tools.ring_expansion_audit import fixture_law, initial


def run(node, budget=128, seed=0):
    process = OptionContinuationKernel(
        fixture_law,
        editing_v2_rewrite_system(),
        max_executor_applications=budget,
    )
    result = sample_option_trajectory(
        node,
        process,
        lambda _: 1.0,
        lambda _: 1.0,
        snapshot_id="expansion-fixture-v1",
        seed=seed,
        max_expansions=0,
        max_terminal_evaluations=0,
        estimator="reference",
    )
    return result, process


@pytest.mark.parametrize("smiles", ["C1CCCC1", "C1CCCCC1", "c1ccc2c(c1)CCCCC2", "c1ccc2c(c1)CCCC2"])
def test_three_edits_expand_and_replay_without_extra_ring_topology(smiles):
    node = initial(smiles)
    result, _ = run(node)
    assert result["status"] == "complete"
    assert [r["family"] for r in result["trace"]] == ["cycle_open", "atom_insert", "bond_insert"]
    current, system = node.graph, editing_v2_rewrite_system()
    classes = {"cycle_open": CycleOpenEdge, "atom_insert": AtomInsert, "bond_insert": BondInsert}
    for row in result["trace"]:
        current = system.apply(current, row["family"], classes[row["family"]](**row["action"]))
        assert is_valid_state(current)
        assert exact_graph_key(current) == exact_graph_key(decode_state(row["exact_state"]))
    progress = ExpansionProgress.from_payload(result["final_expansion_progress"])
    assert completed_expansion(node.graph, current, progress)
    before = Chem.MolFromSmiles(smiles)
    after = Chem.MolFromSmiles(result["endpoint"])
    assert after.GetNumAtoms() == before.GetNumAtoms() + 1
    assert after.GetNumBonds() == before.GetNumBonds() + 1
    assert rdMolDescriptors.CalcNumBridgeheadAtoms(after) == 0
    assert rdMolDescriptors.CalcNumSpiroAtoms(after) == 0
    assert rdMolDescriptors.CalcNumAromaticRings(after) == rdMolDescriptors.CalcNumAromaticRings(
        before
    )


def test_consecutive_expansion_continues_the_exact_executed_state():
    node = initial("c1ccc2c(c1)CCCCC2")
    first, _ = run(node)
    product = decode_state(first["trace"][-1]["exact_state"])
    second, _ = run(initial(graph=product))
    final = Chem.MolFromSmiles(second["endpoint"])
    assert sorted(len(r) for r in final.GetRingInfo().AtomRings()) == [6, 9]
    assert second["trace"][0]["before"] == first["endpoint"]


def test_opening_executes_once_per_physical_edge_but_keeps_orientation_mass():
    node = initial("C1CCCCC1")
    process = OptionContinuationKernel(
        fixture_law, editing_v2_rewrite_system(), max_executor_applications=128
    )
    row = process.row(node)
    assert len(row.successors) == 12
    assert process.work.executor_applications == 6
    assert row.probabilities == pytest.approx([1 / 12] * 12)
    assert len({s.key() for s in row.successors}) == 12
    assert len({exact_graph_key(s.graph) for s in row.successors}) == 6
    assert process.row(node) is row
    assert "expansion_progress" in state_payload(row.successors[0])
    planner = FiniteHorizonContinuation(
        process.row,
        lambda _: 1.0,
        OptionState.key,
        snapshot_id="synthetic-equal-utility",
        max_expansions=128,
        max_terminal_evaluations=128,
    )
    decision = continuation_decision(row, 3, planner, fallback_values=[1.0] * len(row.successors))
    assert decision.kl <= 1.0
    assert sum(decision.probabilities) == pytest.approx(1.0)


def test_default_prior_and_registry_are_unchanged_and_opt_in_has_a_floor():
    assert EXPAND_RING_OPTION not in OPTIONS
    assert EXPAND_RING_OPTION not in applicable_options(["cycle_open"], [0], n_free_slots=1)
    assert EXPAND_RING_OPTION in applicable_options(
        ["cycle_open"], [0], n_free_slots=1, include_expansion=True
    )
    assert EXPAND_RING_OPTION not in applicable_options(
        ["cycle_open"], [0], n_free_slots=0, include_expansion=True
    )
    opts = ("generic", "grow", "cyclize", "open", EXPAND_RING_OPTION)
    assert balanced_option_prior(opts, exploration=0) == pytest.approx(
        [0.25, 0.25, 0.125, 0.25, 0.125]
    )
    assert min(balanced_option_prior(opts)) >= 0.1 / len(opts)
    with pytest.raises(ValueError, match="stateful"):
        conditioned_action_distribution(["cycle_open"], [1.0], [0], EXPAND_RING_OPTION)


def test_region_size_and_ring_junction_restrictions():
    aromatic = initial("c1ccccc1")
    assert eligible_expansion_edges(aromatic.graph, aromatic.context.locus) == ()
    acyclic = initial("CCCC")
    assert eligible_expansion_edges(acyclic.graph, acyclic.context.locus) == ()
    node = initial()
    assert eligible_expansion_edges(node.graph, {0}) == ()
    crowded = initial("C1" + "C" * 39 + "1")
    assert eligible_expansion_edges(crowded.graph, crowded.context.locus) == ()
    no_slots = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 6)
    assert eligible_expansion_edges(no_slots, range(6)) == ()
    fused = initial("C1CCC2CCCCC2C1")
    degrees = np.count_nonzero(fused.graph.bonds, axis=0)
    for u, v in eligible_expansion_edges(fused.graph, fused.context.locus):
        assert not (degrees[u] == degrees[v] == 3)


def test_sparse_slots_and_context_cannot_be_bypassed():
    node = initial("C1CCCCC1")
    permutation = np.roll(np.arange(48), 5)
    graph = replace(
        node.graph,
        **{
            field: getattr(node.graph, field)[permutation]
            for field in ("atom_types", "formal_charges", "implicit_h_counts")
        },
        bonds=node.graph.bonds[np.ix_(permutation, permutation)],
    )
    sparse = initial(graph=graph)
    result, _ = run(sparse)
    assert result["status"] == "complete"
    assert all(i >= 5 for i in result["final_expansion_progress"]["edge"])
    restricted = replace(
        node, context=replace(node.context, locus=frozenset({0}), frozen=frozenset(range(1, 6)))
    )
    result, _ = run(restricted)
    assert result["status"] == "no_admissible_action"
    assert result["endpoint"] is None


def test_incomplete_program_is_not_a_dockable_endpoint():
    result, _ = run(initial(), budget=1)
    assert result["status"] == "executor_budget_exhausted"
    assert result["endpoint"] is None
    assert result["trace"] == []


def test_progress_and_exact_topology_fail_closed():
    with pytest.raises(ValueError, match="explicit ExpansionProgress"):
        replace(initial(), expansion_progress=None)
    for kwargs in ({"edge": (0, 0)}, {"new_slot": 2}, {"edge": (True, 2)}):
        with pytest.raises(ValueError):
            ExpansionProgress(**kwargs)
    with pytest.raises(ValueError, match="schema"):
        ExpansionProgress.from_payload({"schema_version": "wrong", "edge": None, "new_slot": None})
    node = initial("C1CCCCC1")
    result, _ = run(node)
    product = decode_state(result["trace"][-1]["exact_state"])
    progress = ExpansionProgress.from_payload(result["final_expansion_progress"])
    assert not completed_expansion(node.graph, product, replace(progress, edge=(0, 3)))


def test_isolated_eight_ring_remains_rejected_by_the_unchanged_gate():
    result, process = run(initial("C1CCCCCC1"))
    assert result["status"] == "no_admissible_action"
    assert result["endpoint"] is None
    assert len(result["trace"]) == 2
    assert process.work.rejected_products == 1
    from compose_v4.gates.med_chem_gate import validity_reasons

    assert validity_reasons("C1CCCCCCC1") == ["isolated_ring:8"]
