"""Production-executor integration on explicit model-free engineering fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    FiniteHorizonContinuation,
    continuation_decision,
)
from compose_v4.control.molecular_search_codec import decode_option
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    exact_graph_key,
    sample_option_trajectory,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert


def source(option="generic", horizon=2):
    graph = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 6)
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), frozenset(range(4)), (), "pendant", 0),
        Lineage.initial(range(4)),
        option,
        0,
        horizon,
        "synthetic-engineering-bundle",
    )


def fixture_law(graph):
    # A hand-declared reference for testing, NOT learned R_theta probabilities.
    if graph.n_real_atoms == 4:
        return (
            ("atom_insert", "atom_insert"),
            (
                AtomInsert(4, ELEMENT_TO_IDX["C"], 0, 3, ((3, 1),)),
                AtomInsert(4, ELEMENT_TO_IDX["F"], 0, 0, ((3, 1),)),
            ),
            (0.5, 0.5),
        )
    if graph.atom_types[4] == ELEMENT_TO_IDX["C"]:
        return ("bond_insert",), (BondInsert(0, 4, 1),), (1.0,)
    return ("atom_insert",), (AtomInsert(5, ELEMENT_TO_IDX["F"], 0, 0, ((0, 1),)),), (1.0,)


def ring_goal(node):
    if node.remaining:
        return 0.0
    edges = int(np.count_nonzero(np.triu(node.graph.bonds, 1)))
    return float(edges - node.graph.n_real_atoms + 1 >= 1)


def kernel(budget=100):
    return OptionContinuationKernel(
        fixture_law, editing_v2_rewrite_system(), max_executor_applications=budget
    )


def test_production_executor_ring_plateau_has_downstream_value():
    node, process = source(), kernel()
    row = process.row(node)
    assert len(row.successors) == 2
    assert all(is_valid_state(s.graph) for s in row.successors)
    assert all(ring_goal(s) == 0 for s in row.successors)
    estimator = FiniteHorizonContinuation(
        process.row,
        ring_goal,
        OptionState.key,
        snapshot_id="fixture-ring-goal-v1",
        max_expansions=10,
        max_terminal_evaluations=10,
    )
    result = continuation_decision(row, 2, estimator, fallback_values=[1, 1])
    assert result.successor_values == (1.0, 0.0)
    assert result.probabilities == pytest.approx((0.95, 0.05))
    assert process.work.executor_applications == 4


def test_registered_macro_support_excludes_terminal_decoration():
    row = kernel().row(source("scaffold_extend", 1))
    assert len(row.successors) == 1
    assert row.successors[0].graph.atom_types[4] == ELEMENT_TO_IDX["C"]


def test_sampled_path_probability_and_executor_trace():
    node, process = source(), kernel()
    result = sample_option_trajectory(
        node,
        process,
        ring_goal,
        lambda s: 1.0,
        snapshot_id="fixture-ring-goal-v1",
        seed=0,
        max_expansions=10,
        max_terminal_evaluations=10,
    )
    assert result["status"] == "complete"
    assert len(result["trace"]) == 2
    assert result["endpoint"] == "C1CCCC1"
    assert result["conditional_path_logq"] == pytest.approx(
        sum(np.log(step["probability"]) for step in result["trace"])
    )
    assert result["conditional_reference_path_logp"] == pytest.approx(
        sum(np.log(step["reference_probability"]) for step in result["trace"])
    )
    assert decode_option(result["final_option_state"]).key() != node.key()
    assert all(step["kl"] <= 1 for step in result["trace"])
    assert node.graph.n_real_atoms == 4  # source was never mutated


def test_executor_budget_fails_without_publishing_partial_row():
    process = kernel(1)
    with pytest.raises(ContinuationBudgetExceeded):
        process.row(source())
    assert process.work.executor_applications == 1


def test_cached_rows_reuse_exact_state_and_count_no_new_execution():
    node, process = source(), kernel()
    first = process.row(node)
    calls = process.work.executor_applications
    second = process.row(node)
    assert first is second
    assert process.work.executor_applications == calls
    assert process.work.row_cache_hits == 1


def test_keys_distinguish_context_phase_and_lineage():
    node = source()
    assert node.key() != replace(node, step=1).key()
    assert node.key() != replace(node, horizon=3).key()
    assert node.key() != replace(node, context=node.context.with_phase("finish")).key()
    assert node.key() != replace(node, context=node.context.with_locus(5)).key()
    other_lineage = Lineage.initial(range(4))
    other_lineage.next_id += 1
    assert node.key() != replace(node, lineage=other_lineage).key()


def test_exact_key_distinguishes_slot_layout_charge_and_hydrogen():
    original = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 4)
    permutation = np.array([2, 1, 0, 3])
    permuted = replace(
        original,
        atom_types=original.atom_types[permutation],
        bonds=original.bonds[np.ix_(permutation, permutation)],
        formal_charges=original.formal_charges[permutation],
        implicit_h_counts=original.implicit_h_counts[permutation],
    )
    assert canonical_state_key(original) == canonical_state_key(permuted)
    assert exact_graph_key(original) != exact_graph_key(permuted)
    for field in ("formal_charges", "implicit_h_counts"):
        values = getattr(original, field).copy()
        values[0] += 1
        assert exact_graph_key(original) != exact_graph_key(replace(original, **{field: values}))


def test_invalid_program_phase_fails():
    with pytest.raises(ValueError, match="registered program"):
        source("build_ring_system", 1)
    with pytest.raises(ValueError, match="registered program"):
        replace(source(), step=3)


def test_missing_lineage_identity_is_rejected():
    with pytest.raises(ValueError, match="every real atom"):
        replace(source(), lineage=Lineage())


def test_context_only_actions_have_no_admissible_support():
    node = source()
    node = replace(node, context=RewriteContext(frozenset(range(4)), frozenset(), (), "pendant", 1))
    assert kernel().row(node).successors == ()


def test_runtime_adapter_delegates_to_the_production_evaluator(monkeypatch):
    from types import SimpleNamespace

    from compose_v4.experiments import production_successor_kernel as production

    node = source("scaffold_extend", 1)
    calls = []
    model = object()

    def evaluator(actual_model, graph, time_point):
        calls.append((actual_model, graph, time_point))
        families, actions, probabilities = fixture_law(graph)
        return SimpleNamespace(
            marks=[
                SimpleNamespace(executor_rule_name=f, action=a, probability=p)
                for f, a, p in zip(families, actions, probabilities)
            ]
        )

    monkeypatch.setattr(production, "enumerate_factorized_marked_law", evaluator)
    process = OptionContinuationKernel.from_runtime(
        model, editing_v2_rewrite_system(), time_point=0.5, max_executor_applications=10
    )
    row = process.row(node)
    assert len(row.successors) == 1
    assert len(calls) == 1 and calls[0][0] is model and calls[0][1] is node.graph
    assert calls[0][2] == 0.5
