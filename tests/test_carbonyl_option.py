"""Focused actual-kernel checks; reference probabilities here are synthetic."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    NULL_IDX,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.carbonyl_option import (
    ADD_CARBONYL_OPTION,
    INSERT_RING_CARBONYL_OPTION,
    CarbonylProgress,
    completed_core_carbonyl,
    eligible_core_edges,
)
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    OptionState,
    exact_graph_key,
)
from compose_v4.control.option_selector import (
    applicable_options,
    balanced_option_prior,
    option_horizon,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, CycleCloseEdge, CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def engineering_law(graph):
    """Small state-derived C/O/F reference, no target input or learned-law claim."""
    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]
    empty = np.flatnonzero(graph.atom_types == NULL_IDX)
    marks = []
    if len(empty):
        slot = int(empty[0])
        for anchor in real:
            for element in ("C", "O", "F"):
                for order in (1, 2):
                    h = ALLOWED_VALENCES[element][0] - order
                    if h >= 0:
                        marks.append(
                            (
                                "atom_insert",
                                AtomInsert(slot, ELEMENT_TO_IDX[element], 0, h, ((anchor, order),)),
                            )
                        )
    for a in real:
        for b in real:
            if a >= b:
                continue
            marks.append(
                ("cycle_open", CycleOpenEdge(a, b))
                if graph.bonds[a, b]
                else ("cycle_close", CycleCloseEdge(a, b, 1))
            )
    return tuple(f for f, _ in marks), tuple(a for _, a in marks), (1 / len(marks),) * len(marks)


def initial(smiles="C1CCCCC1", option=INSERT_RING_CARBONYL_OPTION, graph=None):
    graph = (
        graph if graph is not None else pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    )
    real = frozenset(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), real, (), "multi", 0),
        Lineage.initial(real),
        option,
        0,
        option_horizon(option, 3),
        "engineering",
        carbonyl_progress=CarbonylProgress() if option == INSERT_RING_CARBONYL_OPTION else None,
    )


def kernel():
    return OptionContinuationKernel(
        engineering_law,
        editing_v2_semantic_rewrite_system(),
        max_executor_applications=None,
        product_gate=EXECUTABLE_PRODUCT_GATE,
    )


@pytest.mark.parametrize("smiles", ["C1CCCC1", "C1CCCCC1", "c1ccc2c(c1)CCNC2"])
def test_core_completes_in_four_edits_and_preserves_rank(smiles):
    node, process = initial(smiles), kernel()
    origin = node.graph
    for step in range(4):
        row = process.row(node)
        assert row.successors and sum(row.probabilities) == pytest.approx(1)
        assert all(n.option == INSERT_RING_CARBONYL_OPTION for n in row.successors)
        node = row.successors[0]
        assert node.step == step + 1
    assert not node.remaining
    assert completed_core_carbonyl(origin, node.graph, node.carbonyl_progress)
    assert node.graph.n_real_atoms == origin.n_real_atoms + 2
    assert (
        np.count_nonzero(np.triu(node.graph.bonds)) == np.count_nonzero(np.triu(origin.bonds)) + 2
    )
    assert CarbonylProgress.from_payload(node.carbonyl_progress.payload()) == node.carbonyl_progress


def test_carbonyl_addition_is_separate_and_excludes_decoration():
    row = kernel().row(initial("CCC", ADD_CARBONYL_OPTION))
    assert row.successors
    assert {canonical_state_key(s.graph) for s in row.successors} == {"CCC=O", "CC(C)=O"}
    assert all(not n.remaining for n in row.successors)
    assert kernel().row(initial("NN", ADD_CARBONYL_OPTION)).successors == ()


def test_prior_opt_in_and_single_what_then_four_how_decisions():
    assert INSERT_RING_CARBONYL_OPTION not in applicable_options(["cycle_open"], [0])
    assert INSERT_RING_CARBONYL_OPTION in applicable_options(
        ["cycle_open"], [0], include_carbonyl=True
    )
    assert INSERT_RING_CARBONYL_OPTION not in applicable_options(
        ["cycle_open"], [0], include_carbonyl=True, n_free_slots=1
    )
    names = ("generic", "grow", ADD_CARBONYL_OPTION, "open", INSERT_RING_CARBONYL_OPTION)
    assert balanced_option_prior(names, exploration=0) == pytest.approx(
        [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    )
    assert min(balanced_option_prior(names)) >= 0.1 / len(names)
    process = kernel()
    old = MolecularHierarchy(process, lazy_applicability=True)
    new = MolecularHierarchy(process, lazy_applicability=True, include_carbonyl_options=True)
    root = MolecularSearchState.start(initial().graph, budget=8, root_id="synthetic")
    where = old.row(root)
    assert where == new.row(root)
    at_what = max(where.successors, key=lambda n: n.region.size)
    assert INSERT_RING_CARBONYL_OPTION not in old.row(at_what).labels
    options = new.row(at_what)
    assert INSERT_RING_CARBONYL_OPTION in options.labels and ADD_CARBONYL_OPTION in options.labels
    node = options.successors[options.labels.index(INSERT_RING_CARBONYL_OPTION)]
    for step in range(4):
        assert node.stage == "how" and node.active.option == INSERT_RING_CARBONYL_OPTION
        node = new.row(node).successors[0]
        assert node.key() == decode_search_state(encode_search_state(node)).key()
    assert node.stage == "where" and node.budget == 4


def test_lazy_matches_explicit_rows_and_reuses_products():
    process, node = kernel(), initial("C1CCCC1")
    exact = process.row(node)
    before = process.work.executor_applications
    lazy = process.lazy_row(node)
    rng = np.random.default_rng(0)
    counts = Counter(lazy.sample(rng).key() for _ in range(2000))
    for child, probability in zip(exact.successors, exact.probabilities, strict=True):
        assert counts[child.key()] / 2000 == pytest.approx(probability, abs=0.035)
    assert process.work.executor_applications == before


def test_context_capacity_and_malformed_progress_fail_closed():
    for smiles in ("CCC", "c1ccccc1", "C1" + "C" * 38 + "1"):
        node = initial(smiles)
        assert eligible_core_edges(node.graph, node.context.locus) == ()
        assert kernel().row(node).successors == ()
    node = initial()
    restricted = replace(
        node, context=replace(node.context, locus=frozenset({0}), frozen=frozenset(range(1, 6)))
    )
    assert kernel().row(restricted).successors == ()
    with pytest.raises(ValueError, match="explicit CarbonylProgress"):
        replace(node, carbonyl_progress=None)
    with pytest.raises(ValueError, match="phase/progress"):
        replace(node, step=2)
    for kwargs in ({"edge": (0, 0)}, {"carbon": 3}, {"oxygen": 3}, {"edge": (True, 1)}):
        with pytest.raises(ValueError):
            CarbonylProgress(**kwargs)


def test_known_winner_suffix_has_kernel_support_not_learned_law_evidence():
    path = Path(__file__).resolve().parents[1] / "diagnostics/t4_whole_ring_plan/result.json"
    if not path.exists():
        pytest.skip(
            "vendored known-winner program result absent; see CARBONYL_OPTION_INTEGRATION.md"
        )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "252dccee4785d3a8e972df30c4a075e7e98bdf02e120300990bcb1de9d21cee9"
    )
    report = json.loads(raw)
    suffix = report["attempts"][0]["stages"][-1]
    node, process = initial(graph=decode_state(suffix["states"][0])), kernel()
    for step, saved in enumerate(suffix["actions"]):
        row = process.row(node)
        family, action = decode_action(saved)
        candidates = [
            s
            for s, (f, a) in zip(row.successors, process.marks(node), strict=True)
            if (f, a) == (family, action) and s.carbonyl_progress.edge == (7, 17)
        ]
        assert len(candidates) == 1
        node = candidates[0]
        assert exact_graph_key(node.graph) == exact_graph_key(
            decode_state(suffix["states"][step + 1])
        )
    assert canonical_state_key(node.graph) == report["target"]
    assert encode_state(node.graph) == suffix["states"][-1]
