"""The de-novo ring dependency-block schedule, and the invariants it must keep.

The compiler used to defer every ring transaction behind the whole graft phase
AND the whole non-ring decoration phase, so the ring decision was supervised on
a state whose legal ring-template support had already collapsed.  The
dependency-block schedule commits each ring system at the earliest graft prefix
the executor accepts it at.

Three properties have to hold together, and each test below is written so that
breaking exactly one of them turns exactly one named test red:

* ``sequential`` is untouched.  The repair is opt-in; the corpus Lineage B
  trained on must still compile byte-for-byte identically.
* The dependency block reaches the ARRAY-EXACT target.  A schedule that moves
  an endpoint is a different corpus, not a repair.
* The dependency block actually moves ring events earlier.  A schedule that is
  silently inert would pass the first two properties perfectly.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.eval.denovo_schedule_probe import ring_size_support_histogram
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

# Real ring-bearing molecules; several carry more than one ring system, which is
# the case the block schedule has to keep exact while reordering.
_RING_MOLECULES = (
    "c1ccncc1O",
    "C1CCC2CCCCC2C1",
    "Cc1ccc(C(=O)NCc2ccco2)c(C)c1",
    "O=C(Cc1ccccc1O)Nc1ccccc1",
    "c1ccc2[nH]ccc2c1",
)

_GRAFT = {
    "use_bond_reroute": True,
    "flexible_size": True,
    "typed_ring_payloads": True,
}


def _pair(smiles: str, seed: int):
    """Return one (source, target) pair on the flexible Graft transport path."""

    target_raw = smiles_to_molecular_graph(smiles)
    target = pad_molecular_graph(target_raw, target_raw.n_real_atoms + 4)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(seed), n_slots=target.n_atoms)
    return source, target


def _rule_names(trace) -> tuple[str, ...]:
    return tuple(step.rule_name for step in trace.steps)


def _ring_indices(trace) -> tuple[int, ...]:
    return tuple(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )


# ---- The repair must not disturb the schedule the corpus was compiled under ----


@pytest.mark.parametrize("smiles", _RING_MOLECULES)
def test_sequential_schedule_is_untouched_by_the_dependency_block(smiles: str) -> None:
    """``sequential`` still emits grafts, then decoration, then every ring.

    The expectation is a STRUCTURAL property of the legacy order -- all ring
    transactions last -- not a value recomputed from the code under test, so a
    change to the default emission order fails here rather than moving with it.
    """

    source, target = _pair(smiles, seed=4113)
    trace = compile_carbon_tree_to_target(
        source, target, event_schedule="sequential", **_GRAFT
    )
    names = _rule_names(trace)
    rings = _ring_indices(trace)

    assert rings, "fixture must carry a ring system"
    assert list(rings) == list(range(len(names) - len(rings), len(names))), (
        "under the sequential schedule every ring transaction is emitted last"
    )
    assert trace.metadata["ring_emission_schedule"] == "sequential"
    assert "bond_reroute" not in names[rings[0]:]


# ---- Exactness: the hard constraint ----


@pytest.mark.parametrize("smiles", _RING_MOLECULES)
def test_dependency_block_reaches_the_array_exact_target(smiles: str) -> None:
    """Every reordered trace replays to the identical padded endpoint.

    Array equality is asserted per field rather than through a canonical key
    alone: the key is a chemical identity and would accept a state whose slot
    gauge had drifted, which later compilation stages do not tolerate.
    """

    source, target = _pair(smiles, seed=907)
    trace = compile_carbon_tree_to_target(
        source, target, event_schedule="ring_dependency_block", **_GRAFT
    )
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)

    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.formal_charges, target.formal_charges)
    assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert canonical_state_key(endpoint) == canonical_state_key(target)
    # The reordering may not pass through an unrepresentable intermediate.
    assert all(is_valid_state(state) for state in states)
    assert all(is_connected_or_null(state) for state in states)


@pytest.mark.parametrize("smiles", _RING_MOLECULES)
def test_dependency_block_preserves_the_ring_system_inventory(smiles: str) -> None:
    """Reordering may move ring transactions; it may not add or drop one."""

    source, target = _pair(smiles, seed=907)
    sequential = compile_carbon_tree_to_target(
        source, target, event_schedule="sequential", **_GRAFT
    )
    block = compile_carbon_tree_to_target(
        source, target, event_schedule="ring_dependency_block", **_GRAFT
    )

    assert len(_ring_indices(block)) == len(_ring_indices(sequential))
    assert sorted(
        tuple(sorted(step.action.system_atoms))
        for step in block.steps
        if step.rule_name == "ring_system_grow"
    ) == sorted(
        tuple(sorted(step.action.system_atoms))
        for step in sequential.steps
        if step.rule_name == "ring_system_grow"
    )
    committed = block.metadata["ring_dependency_block_commits"]
    deferred = block.metadata["ring_dependency_block_deferred"]
    assert committed + deferred == len(_ring_indices(block))


# ---- The repair must not be inert ----


def test_dependency_block_commits_every_ring_before_any_decoration() -> None:
    """Decoration never precedes a ring transaction any more.

    This is the property the repair exists for, and it is exact rather than
    statistical: ``atom_restate`` and ``bond_reorder`` are emitted only for
    atoms and bonds OUTSIDE every ring system, so no decoration step can be a
    dependency of a ring transaction.  An inert schedule satisfies exactness
    and inventory perfectly, so this is the test that fails when the block
    stops reordering.
    """

    decorated = 0
    for smiles in _RING_MOLECULES:
        source, target = _pair(smiles, seed=907)
        block = compile_carbon_tree_to_target(
            source, target, event_schedule="ring_dependency_block", **_GRAFT
        )
        names = _rule_names(block)
        first_ring = _ring_indices(block)[0]
        prefix = names[:first_ring]
        assert "atom_restate" not in prefix, smiles
        assert "bond_reorder" not in prefix, smiles
        if "atom_restate" in names or "bond_reorder" in names:
            decorated += 1
    assert decorated >= 3, "fixtures must include molecules that carry decoration"


def test_a_ring_moves_earlier_exactly_when_the_molecule_has_non_ring_work() -> None:
    """Movement is predicted by structure, not asserted as a blanket win.

    A molecule that is ENTIRELY one ring system has no step the ring does not
    depend on, so its transaction provably cannot move and reporting that as a
    failure would be wrong.  Every other molecule must move.
    """

    for smiles in _RING_MOLECULES:
        source, target = _pair(smiles, seed=907)
        sequential = compile_carbon_tree_to_target(
            source, target, event_schedule="sequential", **_GRAFT
        )
        block = compile_carbon_tree_to_target(
            source, target, event_schedule="ring_dependency_block", **_GRAFT
        )
        names = _rule_names(sequential)
        has_non_ring_work = "atom_restate" in names or "bond_reorder" in names
        moved = _ring_indices(block)[0] < _ring_indices(sequential)[0]
        assert moved == has_non_ring_work, (
            f"{smiles}: moved={moved} but non-ring work={has_non_ring_work}"
        )


def test_dependency_block_interleaves_a_ring_with_the_graft_phase() -> None:
    """A ring transaction lands strictly inside the graft phase.

    This separates the dependency block from the shipped adjacent-commutation
    scheduler, which cannot cross a ``bond_reroute`` at all and therefore can
    only ever park a ring event immediately after the last graft.
    """

    interleaved = 0
    for smiles in _RING_MOLECULES:
        source, target = _pair(smiles, seed=907)
        block = compile_carbon_tree_to_target(
            source, target, event_schedule="ring_dependency_block", **_GRAFT
        )
        names = _rule_names(block)
        first_ring = _ring_indices(block)[0]
        if "bond_reroute" in names[first_ring:]:
            interleaved += 1
    assert interleaved >= 1, (
        "no ring transaction was placed inside the graft phase; the block is inert"
    )


def test_unknown_event_schedule_is_refused() -> None:
    source, target = _pair("c1ccncc1O", seed=11)
    with pytest.raises(ValueError, match="unknown tree event schedule"):
        compile_carbon_tree_to_target(
            source, target, event_schedule="commit_rings_whenever", **_GRAFT
        )


# ---- The reporting instrument ----


class _Template:
    """Minimal stand-in carrying the two fields the size helper reads."""

    def __init__(self, span: int, bonds: tuple[tuple[int, int, int], ...]) -> None:
        self.span = span
        self.target_bonds = bonds


def _cycle(size: int) -> _Template:
    return _Template(
        size,
        tuple((index, (index + 1) % size, 1) for index in range(size)),
    )


def test_ring_size_support_histogram_counts_only_legal_templates() -> None:
    """The histogram reports the SUPPORT, not the catalog.

    Written against hand-built templates whose ring sizes are known
    independently of the production catalog, so the expectation cannot be
    recomputed from the code under test.
    """

    templates = (_cycle(3), _cycle(6), _cycle(4), _cycle(6))
    assert ring_size_support_histogram((True, True, False, True), templates) == {
        "3": 1,
        "6": 2,
    }
    assert ring_size_support_histogram((False, False, False, False), templates) == {}
    assert ring_size_support_histogram((True,) * 4, templates) == {
        "3": 1,
        "4": 1,
        "6": 2,
    }


def test_ring_size_support_histogram_labels_an_acyclic_template() -> None:
    acyclic = _Template(3, ((0, 1, 1), (1, 2, 1)))
    assert ring_size_support_histogram((True,), (acyclic,)) == {"acyclic": 1}


def test_ring_size_support_histogram_refuses_a_misaligned_support() -> None:
    with pytest.raises(ValueError, match="must align"):
        ring_size_support_histogram((True, False), (_cycle(6),))


def test_ring_size_support_histogram_agrees_with_the_small_ring_mask() -> None:
    """The histogram and the 3/4 mask must never disagree about a template."""

    from compose_v4.eval.denovo_schedule_probe import small_ring_category_mask

    templates = (_cycle(3), _cycle(5), _cycle(4), _cycle(7))
    mask = small_ring_category_mask(templates, maximum_size=4)
    histogram = ring_size_support_histogram((True,) * len(templates), templates)
    small_from_histogram = sum(
        count for key, count in histogram.items() if key.isdigit() and int(key) <= 4
    )
    assert small_from_histogram == int(mask.sum())
