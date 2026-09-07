"""Opt-in fused program invariants; explicit model-free reference fixtures."""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, SCAR_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    FiniteHorizonContinuation,
    continuation_decision,
)
from compose_v4.control.fused_option import (
    BUILD_FUSED_RING_OPTION,
    FusedProgress,
    completed_fused_cycle,
    eligible_fusion_edges,
)
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
    retain_product_applicable_options,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.experiments.continuation_profile import state_payload
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert


def source():
    graph = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), frozenset(range(6)), (), "multi", 0),
        Lineage.initial(range(6)),
        BUILD_FUSED_RING_OPTION,
        0,
        5,
        "synthetic-fused-bundle",
        FusedProgress(),
    )


def fixture_law(graph):
    """Declared reference, not learned probabilities or an endpoint constructor.

    Six initial attachment marks; alternating carbon-chain growth thereafter;
    all single/double closures back to old atoms. The production option chooses
    among them and must reject incompatible edges/products independently.
    """
    count = graph.n_real_atoms
    if count == 6:
        actions = tuple(AtomInsert(6, ELEMENT_TO_IDX["C"], 0, 3, ((i, 1),)) for i in range(6))
        return ("atom_insert",) * 6, actions, (1 / 6,) * 6
    if count < 10:
        order = {7: 2, 8: 1, 9: 2}[count]
        action = AtomInsert(count, ELEMENT_TO_IDX["C"], 0, 4 - order, ((count - 1, order),))
        return ("atom_insert",), (action,), (1.0,)
    actions = tuple(BondInsert(i, 9, order) for i in range(6) for order in (1, 2))
    return ("bond_insert",) * len(actions), actions, (1 / len(actions),) * len(actions)


def kernel(law=fixture_law, budget=512):
    return OptionContinuationKernel(
        law, editing_v2_rewrite_system(), max_executor_applications=budget
    )


def complete(node):
    return float(
        node.remaining == 0 and completed_fused_cycle(node.origin, node.graph, node.fused_progress)
    )


def test_default_registry_and_prior_do_not_enable_fused():
    assert BUILD_FUSED_RING_OPTION not in OPTIONS
    assert BUILD_FUSED_RING_OPTION not in applicable_options(["atom_insert"], [0], n_free_slots=8)
    opt_in = applicable_options(["atom_insert"], [0], n_free_slots=4, include_fused=True)
    assert BUILD_FUSED_RING_OPTION in opt_in
    assert "generic" in retain_product_applicable_options(opt_in, lambda _: False)
    assert BUILD_FUSED_RING_OPTION not in applicable_options(
        ["atom_insert"], [0], n_free_slots=3, include_fused=True
    )
    a = ("generic", "grow", "cyclize", "open")
    b = a + (BUILD_FUSED_RING_OPTION,)
    assert balanced_option_prior(a, exploration=0) == pytest.approx([0.25] * 4)
    assert balanced_option_prior(b, exploration=0) == pytest.approx(
        [0.25, 0.25, 0.125, 0.25, 0.125]
    )
    assert np.min(balanced_option_prior(b)) >= 0.1 / len(b)


def test_descriptor_free_call_fails_closed():
    with pytest.raises(ValueError, match="stateful"):
        conditioned_action_distribution(["atom_insert"], [1.0], [0], BUILD_FUSED_RING_OPTION)


def test_shared_physical_mark_keeps_distinct_edge_mass_and_state():
    node, process = source(), kernel()
    row = process.row(node)
    assert len(row.successors) == 12
    assert process.work.executor_applications == 6  # not twelve repeated applications
    assert row.probabilities == pytest.approx([1 / 12] * 12)
    assert len({s.key() for s in row.successors}) == 12
    assert len({exact_graph_key(s.graph) for s in row.successors}) == 6
    assert {s.bundle_id for s in row.successors} == {node.bundle_id}
    by_edge = defaultdict(float)
    for successor, probability in zip(row.successors, row.probabilities):
        assert is_valid_state(successor.graph)
        by_edge[successor.fused_progress.edge] += probability
    assert list(by_edge.values()) == pytest.approx([1 / 12] * 12)
    assert process.row(node) is row
    assert process.work.executor_applications == 6
    assert process.fused_support(node)["physical_marks"] == 6


def test_joint_reference_balances_edges_before_multiple_marks():
    def unequal_law(graph):
        families, actions, _ = fixture_law(graph)
        # Another persistent-slot mark at attachment 0, not a second edge draw.
        extra = AtomInsert(7, ELEMENT_TO_IDX["C"], 0, 3, ((0, 1),))
        return families + ("atom_insert",), actions + (extra,), (1 / 7,) * 7

    node, process = source(), kernel(unequal_law)
    row = process.row(node)
    by_edge = defaultdict(float)
    for successor, p in zip(row.successors, row.probabilities):
        by_edge[successor.fused_progress.edge] += p
    assert list(by_edge.values()) == pytest.approx([1 / 12] * 12)
    assert len(row.successors) == 14
    assert process.work.executor_applications == 7


def test_continuation_can_credit_the_fusion_edge_without_changing_qm():
    def asymmetric_law(graph):
        if graph.n_real_atoms == 10:
            # Explicit synthetic support asymmetry. Symmetric benzene under
            # fixture_law completes from EVERY edge and has no planning signal.
            return ("bond_insert",), (BondInsert(0, 9, 1),), (1.0,)
        return fixture_law(graph)

    node, process = source(), kernel(asymmetric_law)
    row = process.row(node)
    planner = FiniteHorizonContinuation(
        process.row,
        complete,
        OptionState.key,
        snapshot_id="synthetic-fused-completion-v1",
        max_expansions=100,
        max_terminal_evaluations=100,
    )
    decision = continuation_decision(row, node.remaining, planner, fallback_values=[1] * 12)
    values = np.asarray(decision.successor_values)
    assert set(values) == {0.0, 1.0}
    assert np.dot(decision.probabilities, values) > np.dot(row.probabilities, values)
    assert decision.kl <= 1.0
    assert sum(decision.probabilities) == pytest.approx(1)
    assert all(s.bundle_id == node.bundle_id for s in row.successors)


def test_symmetric_fixture_has_no_earned_edge_preference():
    node, process = source(), kernel()
    row = process.row(node)
    planner = FiniteHorizonContinuation(
        process.row,
        complete,
        OptionState.key,
        snapshot_id="symmetric-fixture-v1",
        max_expansions=100,
        max_terminal_evaluations=100,
    )
    decision = continuation_decision(row, 5, planner, fallback_values=[1] * 12)
    assert decision.successor_values == (1.0,) * 12
    assert decision.probabilities == pytest.approx(row.probabilities)


def test_guided_path_is_five_real_edits_and_an_exact_fused_cycle():
    node, process = source(), kernel()
    result = sample_option_trajectory(
        node,
        process,
        complete,
        lambda _: 1.0,
        snapshot_id="synthetic-fused-completion-v1",
        seed=0,
        max_expansions=100,
        max_terminal_evaluations=100,
    )
    assert result["status"] == "complete"
    assert result["endpoint"] == "c1ccc2ccccc2c1"
    assert len(result["trace"]) == 5
    assert [x["family"] for x in result["trace"]] == ["atom_insert"] * 4 + ["bond_insert"]
    assert all(x["kl"] <= 1.0 for x in result["trace"])
    assert result["conditional_path_logq"] == pytest.approx(
        sum(np.log(x["probability"]) for x in result["trace"])
    )
    system, graph = editing_v2_rewrite_system(), node.graph
    for record in result["trace"]:
        assert canonical_state_key(graph) == record["before"]
        values = record["action"]
        action = AtomInsert(**values) if record["family"] == "atom_insert" else BondInsert(**values)
        graph = system.apply(graph, record["family"], action)
        assert is_valid_state(graph)
        assert canonical_state_key(graph) == record["after"]
    progress = FusedProgress.from_payload(result["trace"][-1]["fused_progress"])
    assert completed_fused_cycle(node.graph, graph, progress)
    assert not completed_fused_cycle(node.graph, graph, FusedProgress((0, 2), progress.path))
    assert node.graph.n_real_atoms == 6


def test_reference_only_never_calls_guidance_or_terminal():
    def forbidden(_):
        pytest.fail("reference-only sampling must not evaluate guidance")

    result = sample_option_trajectory(
        source(),
        kernel(),
        forbidden,
        forbidden,
        snapshot_id="fixture-reference-v1",
        seed=0,
        max_expansions=0,
        max_terminal_evaluations=0,
        estimator="reference",
    )
    assert len(result["trace"]) >= 4  # can explicitly fail aromatic closure at the chosen edge
    assert all(value == 0 for value in result["continuation_work"].values())
    assert all(x["reference_probability"] == x["probability"] for x in result["trace"])


def test_sampled_planner_retains_program_state_and_reproducible_path():
    def run():
        return sample_option_trajectory(
            source(),
            kernel(),
            complete,
            lambda _: 1.0,
            snapshot_id="synthetic-fused-sampled-v1",
            seed=0,
            max_expansions=100,
            max_terminal_evaluations=100,
            estimator="sampled",
            samples_per_successor=4,
            max_rollouts=64,
        )

    first, second = run(), run()
    assert first["status"] == second["status"] == "complete"
    assert first["trace"] == second["trace"]
    assert first["randomness"] == second["randomness"]
    assert all(step["kl"] <= 1.0 for step in first["trace"])


def test_no_mutable_ring_edge_is_explicit_empty_support():
    node = source()
    context = RewriteContext(frozenset(range(1, 6)), frozenset({0}), (), "pendant", 1)
    node = replace(node, context=context)
    process = kernel()
    assert process.row(node).successors == ()
    assert process.fused_support(node)["eligible_oriented_edges"] == 0
    assert process.work.executor_applications == 0


def test_invalid_growth_products_are_removed_before_edge_normalization():
    def invalid_law(_):
        action = AtomInsert(6, ELEMENT_TO_IDX["C"], 0, 0, ((0, 1),))  # inconsistent H count
        return ("atom_insert",), (action,), (1.0,)

    node, process = source(), kernel(invalid_law)
    assert process.row(node).successors == ()
    assert process.work.executor_applications == 1
    assert process.fused_support(node)["product_applicable_oriented_edges"] == 0


def test_cap_exclusion_is_visible_and_not_bypassed():
    def capped_law(graph):
        _, actions, _ = fixture_law(graph)
        # High-probability decoration occupies the inherited 300/20 cap.
        decoration = AtomInsert(6, ELEMENT_TO_IDX["F"], 0, 0, ((0, 1),))
        return (
            ("atom_insert",) * 306,
            (decoration,) * 300 + actions,
            (1 / 301,) * 300 + (1 / 1806,) * 6,
        )

    node, process = source(), kernel(capped_law)
    assert process.row(node).successors == ()
    audit = process.fused_support(node)
    assert audit["eligible_oriented_edges"] == 12
    assert sum(e["descriptor_region_marks"] for e in audit["edges"]) == 12
    assert sum(e["after_inherited_cap"] for e in audit["edges"]) == 0
    assert process.work.executor_applications == 0


def test_budget_interruption_does_not_publish_partial_fused_row():
    node, process = source(), kernel(budget=1)
    with pytest.raises(ContinuationBudgetExceeded):
        process.row(node)
    with pytest.raises(KeyError):
        process.marks(node)
    with pytest.raises(KeyError):
        process.fused_support(node)
    assert process.work.executor_applications == 1


def test_sparse_slots_and_scar_do_not_become_ring_atoms():
    graph = source().graph
    permutation = np.array([6, 7, 0, 1, 8, 2, 3, 9, 4, 5, 10, 11])
    graph = replace(
        graph,
        atom_types=graph.atom_types[permutation].copy(),
        formal_charges=graph.formal_charges[permutation],
        implicit_h_counts=graph.implicit_h_counts[permutation],
        bonds=graph.bonds[np.ix_(permutation, permutation)],
    )
    graph.atom_types[0] = SCAR_IDX
    edges = eligible_fusion_edges(graph, range(12))
    assert len(edges) == 12
    assert not any(0 in e or 1 in e for e in edges)
    assert eligible_fusion_edges(graph, (2,)) == ()
    acyclic = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 12)
    assert eligible_fusion_edges(acyclic, range(6)) == ()


def test_missing_progress_bad_phase_and_wrong_option_fail_closed():
    node = source()
    with pytest.raises(ValueError, match="explicit FusedProgress"):
        replace(node, fused_progress=None)
    with pytest.raises(ValueError, match="phase/path"):
        replace(node, step=1)
    with pytest.raises(ValueError, match="only valid"):
        replace(node, option="generic")
    with pytest.raises(ValueError, match="tuples"):
        FusedProgress([0, 1])
    with pytest.raises(ValueError, match="distinct"):
        FusedProgress((0, 0))


def test_program_codec_extends_only_opt_in_state_payloads():
    node = source()
    assert state_payload(node)["fused_progress"] == FusedProgress().payload()
    assert FusedProgress.from_payload(FusedProgress((0, 1), (6,)).payload()) == FusedProgress(
        (0, 1), (6,)
    )
    generic = replace(node, option="generic", fused_progress=None)
    assert "fused_progress" not in state_payload(generic)
    assert len(node.key()) == len(generic.key()) + 1
    with pytest.raises(ValueError, match="schema"):
        FusedProgress.from_payload({"schema_version": "future", "edge": None, "path": []})
