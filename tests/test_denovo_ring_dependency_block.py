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


# A ring system whose tree edges are only completed by the final Graft cannot
# be placed early.  Measured on 116 real training molecules, 30% defer at least
# one system, so this is a common path and not a corner case.
_DEFERRING_MOLECULES = (
    "O=C1C=C(c2cccc(O)c2)CCC1",
    "COc1ccc(C(=O)N2CCCC2=O)cn1",
    "CN(C)CCN1C(=O)c2ccccc2C1=O",
)


@pytest.mark.parametrize("smiles", _DEFERRING_MOLECULES)
def test_a_deferred_ring_system_still_reaches_the_exact_target(smiles: str) -> None:
    """The fallback path is exercised, not merely believed in.

    A system the block cannot place early keeps its original position at the
    end of the route.  That is a real limitation of the repair -- such a system
    is still decided against a constricted support -- but it must never cost a
    molecule or move an endpoint.
    """

    source, target = _pair(smiles, seed=907)
    block = compile_carbon_tree_to_target(
        source, target, event_schedule="ring_dependency_block", **_GRAFT
    )
    assert block.metadata["ring_dependency_block_deferred"] >= 1, (
        "fixture must exercise the deferred path"
    )

    endpoint = execute_trace(block.source, block.steps)
    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
    assert canonical_state_key(endpoint) == canonical_state_key(target)

    sequential = compile_carbon_tree_to_target(
        source, target, event_schedule="sequential", **_GRAFT
    )
    # Every system is emitted exactly once, deferred or not.
    assert len(_ring_indices(block)) == len(_ring_indices(sequential))
    assert (
        block.metadata["ring_dependency_block_commits"]
        + block.metadata["ring_dependency_block_deferred"]
        == len(_ring_indices(block))
    )
    # The deferred system is last, which is exactly the position the repair
    # does NOT improve. Asserted so the limitation cannot be quietly lost.
    assert _rule_names(block)[-1] == "ring_system_grow"


# Real training molecules on which committing a ring system early invalidates a
# LATER graft that touches one of its atoms.  Found by compiling a 150-molecule
# random draw of the train partition under both schedules; without the
# trace-level fallback these three compiled under ``sequential`` and raised
# under the block, which would have silently shrunk the corpus by ~2%.
_FALLBACK_MOLECULES = (
    ("c1ccc(-c2ccc3[nH]c(-c4ccc5nc(-c6ccccc6)[nH]c5c4)nc3c2)cc1", 20_283_871),
    ("CC1(C)CC(=O)C(=NNc2cccnc2)C(=O)C1", 20_287_352),
    ("Cc1cc(C2CC(CO)CN2C)on1", 20_291_145),
)


def _production_pair(smiles: str, seed: int):
    """Reproduce the trainer's own 40-slot source draw for one molecule."""

    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 40)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(seed), n_slots=40)
    return source, target


@pytest.mark.parametrize(("smiles", "seed"), _FALLBACK_MOLECULES)
def test_a_molecule_the_block_cannot_reorder_is_kept_not_dropped(
    smiles: str, seed: int
) -> None:
    """The schedule may decline to reorder; it may never cost a molecule.

    A schedule that compiles fewer molecules produces a different corpus, not a
    repair, and the difference would be invisible in any per-arm support mean
    computed over whatever survived.
    """

    source, target = _production_pair(smiles, seed)
    sequential = compile_carbon_tree_to_target(
        source, target, event_schedule="sequential", ring_catalog=None,
        align_source=False, **_GRAFT
    )
    block = compile_carbon_tree_to_target(
        source, target, event_schedule="ring_dependency_block", ring_catalog=None,
        align_source=False, **_GRAFT
    )

    assert block.metadata["ring_dependency_block_fell_back_to_sequential"] is True
    # Falling back means falling back: the emitted route is the legacy one.
    assert _rule_names(block) == _rule_names(sequential)
    endpoint = execute_trace(block.source, block.steps)
    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert canonical_state_key(endpoint) == canonical_state_key(target)


def test_a_reorderable_molecule_does_not_report_a_fallback() -> None:
    """The fallback flag must discriminate, not be pinned to one value."""

    source, target = _pair("O=C(Cc1ccccc1O)Nc1ccccc1", seed=907)
    block = compile_carbon_tree_to_target(
        source, target, event_schedule="ring_dependency_block", **_GRAFT
    )
    assert block.metadata["ring_dependency_block_fell_back_to_sequential"] is False
    assert block.metadata["ring_dependency_block_commits"] >= 1


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
