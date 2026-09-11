"""Small real-executor fixtures, not learned-law or benchmark evidence."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from test_ring_program import uniform_law

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.molecular_search_codec import decode_option
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    OptionState,
)
from compose_v4.control.option_selector import balanced_option_prior
from compose_v4.control.region_replacement import (
    PREFIX,
    ReplacementProgress,
    applicable,
    maximum_horizon,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext, context_preserved
from compose_v4.control.ring_program import RingSpec, real_slots
from compose_v4.experiments.continuation_profile import state_payload
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system


def fixture(smiles="C" * 40, frozen_count=34):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    frozen = frozenset(range(frozen_count))
    locus = real_slots(graph) - frozen
    terminals = tuple(
        (i, j, int(graph.bonds[i, j]))
        for i in sorted(frozen)
        for j in sorted(locus)
        if graph.bonds[i, j]
    )
    context = RewriteContext(frozen, locus, terminals, "pendant", 1)
    spec = RingSpec("pendant", 5, (4, 1, 0), "aromatic")
    return OptionState(
        graph,
        graph,
        context,
        Lineage.initial(real_slots(graph)),
        PREFIX + spec.option,
        0,
        maximum_horizon(graph, context, spec),
        "replacement-fixture",
        replacement_progress=ReplacementProgress(),
    )


def kernel(law=uniform_law):
    return OptionContinuationKernel(
        law,
        editing_v2_rewrite_system(),
        max_executor_applications=None,
        product_gate=EXECUTABLE_PRODUCT_GATE,
    )


def lazy_measure(row):
    result = {}
    branches = [branch for branch in row.branches.values() if branch.has_product()]
    for branch in branches:
        valid = [
            (branch.resolve(i), w)
            for i, w in zip(branch.indices, branch.weights)
            if branch.resolve(i) is not None
        ]
        total = sum(w for _, w in valid)
        for node, weight in valid:
            p = (
                (1 - branch.uniform_fraction) * weight / total
                + branch.uniform_fraction / len(valid)
            ) / len(branches)
            result[node.key()] = result.get(node.key(), 0) + p
    return result


def test_replacement_at_capacity_reuses_slots_and_preserves_context():
    node, process = fixture(), kernel()
    origin = node
    sizes, deleted_ids, stages = [40], set(), []
    rng = np.random.default_rng(2)
    while node.remaining:
        assert decode_option(state_payload(node)).key() == node.key()
        child = process.lazy_row(node).sample(rng)
        assert child is not None, (node.step, canonical_state_key(node.graph))
        stages.append("prune" if node.replacement_progress.build_origin is None else "build")
        deleted_ids |= set(node.lineage.slot_of) - set(child.lineage.slot_of)
        assert context_preserved(
            origin.graph, child.graph, origin.context.frozen, origin.context.terminal_context_slots
        )
        sizes.append(child.graph.n_real_atoms)
        node = child
    assert sizes == list(range(40, 33, -1)) + list(range(35, 40)) + [39]
    assert stages == ["prune"] * 6 + ["build"] * 6
    assert node.step == 12 and node.remaining == 0
    assert deleted_ids.isdisjoint(node.lineage.slot_of)
    assert node.lineage.next_id == 45  # reused slots have new persistent identities
    mol = Chem.MolFromSmiles(canonical_state_key(node.graph))
    assert sorted(map(len, mol.GetRingInfo().AtomRings())) == [5]
    assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in mol.GetRingInfo().AtomRings()[0])
    assert decode_option(state_payload(node)).key() == node.key()


def test_eager_lazy_match_through_prune_and_build():
    node, process = fixture("CCCCCC", 2), kernel()
    rng = np.random.default_rng(0)
    while node.remaining:
        eager = process.row(node)
        expected = {}
        for child, p in zip(eager.successors, eager.probabilities):
            expected[child.key()] = expected.get(child.key(), 0) + p
        assert expected
        assert lazy_measure(process.lazy_row(node)) == pytest.approx(expected)
        node = process.lazy_row(node).sample(rng)


def test_cycle_opening_is_enabling_not_a_decoration_escape():
    node = fixture("Cc1ccccc1", 1)

    # Explicit model-free support fixture exposing opening but no deletion at
    # the source. Production executor must still admit every selected action.
    def opening_law(graph):
        families, actions, p = uniform_law(graph)
        keep = [i for i, family in enumerate(families) if family != "atom_delete"]
        return (
            tuple(families[i] for i in keep),
            tuple(actions[i] for i in keep),
            tuple(p[i] for i in keep),
        )

    process = kernel(opening_law)
    row = process.row(node)
    assert row.successors and all(
        f in ("cycle_open", "bond_delete") for f, _ in process.marks(node)
    )
    assert all(child.replacement_progress.opened == 1 for child in row.successors)
    assert all(child.graph.n_real_atoms == node.graph.n_real_atoms for child in row.successors)


def test_domain_prior_and_bad_resume():
    node = fixture()
    spec = RingSpec("pendant", 5, (4, 1, 0), "aromatic")
    assert applicable(node.graph, node.context, spec)
    assert not applicable(node.graph, replace(node.context, k_components=2), spec)
    assert not applicable(node.graph, replace(node.context, terminals=()), spec)
    assert not applicable(node.graph, replace(node.context, locus=frozenset({39})), spec)
    options = ("generic", "rebuild", node.option)
    assert balanced_option_prior(options, exploration=0) == pytest.approx([0.5, 0.25, 0.25])
    assert np.min(balanced_option_prior(options)) >= 0.1 / len(options)
    payload = state_payload(node)
    payload["replacement_progress"]["removed"] = 1
    with pytest.raises(ValueError, match="clock"):
        decode_option(payload)


def test_opt_in_does_not_change_where_and_keeps_generic():
    graph = fixture("CCCCCC", 2).graph
    root = MolecularSearchState.start(graph, budget=40, root_id="fixture")
    spec = RingSpec("pendant", 5, (4, 1, 0), "aromatic")
    base = MolecularHierarchy(kernel(), ring_options=(spec.option,), lazy_applicability=True)
    new = MolecularHierarchy(
        kernel(),
        ring_options=(spec.option,),
        lazy_applicability=True,
        include_region_replacement=True,
    )
    a, b = base.row(root), new.row(root)
    assert np.array_equal(a.reference, b.reference) and a.labels == b.labels
    assert [n.key() for n in a.successors] == [n.key() for n in b.successors]
    what = next(
        n for n in a.successors if n.region.size == 2 and n.region.n_context_components == 1
    )
    old, enabled = base.row(what), new.row(what)
    assert "generic" in old.labels and "generic" in enabled.labels
    assert not any(o.startswith(PREFIX) for o in old.labels)
    assert PREFIX + spec.option in enabled.labels
    for label, state in zip(old.labels, old.successors):
        assert state.key() == enabled.successors[enabled.labels.index(label)].key()
