"""Process-V2 ``atom_delete`` at the resolver and executor level.

Process V2 decides EVERY candidate through one admission authority.  There is
no inherited rule and no exempt slot, so the tests here are organised around the
two things that can go wrong with a uniform authority: a candidate that should
be excluded is admitted, or a capability that must stay reachable is withdrawn.

The first Process-V2 branch failed the first way.  It preserved the legacy
root/singleton/leaf admission set bit-for-bit and thereby preserved the legacy
exemption from the charge policy: measured over the 800 Jin-QED leads, 125 of
3,319 inherited candidates across 105 molecules changed a protected charged
centre and were admitted anyway.  The charged fixtures below are that defect,
reduced to reachable single molecules.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    NULL_IDX,
    SCAR_IDX,
    MolecularGraph,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.operators import (
    AtomDelete,
    apply_atom_delete,
    is_valid_atom_delete,
)
from compose_v4.rewrite.process_v2_atom_delete import (
    CONNECTED_NONLEAF_MINIMUM_DEGREE,
    ProcessV2AtomDeleteCandidateSource,
    ProcessV2AtomDeleteRejectionCode,
    enumerate_process_v2_atom_deletes,
    process_v2_atom_delete_mask,
    resolve_process_v2_atom_delete,
)

_SLOTS = 24
_CARBON = ELEMENT_TO_IDX["C"]
_OXYGEN = ELEMENT_TO_IDX["O"]
_INHERITED = ProcessV2AtomDeleteCandidateSource.INHERITED_ROOT_SINGLETON_LEAF
_CONNECTED_NONLEAF = ProcessV2AtomDeleteCandidateSource.CONNECTED_NONLEAF

# Verified reference behaviour of the complete effective fiber, recorded as
# absolute slot literals.
_REFERENCE_PANEL = (
    # Saturated rings: the fiber Process V2 newly reaches.
    ("C1CCCCC1", (0, 1, 2, 3, 4, 5)),
    ("C1CCOCC1", (0, 1, 2, 3, 4, 5)),
    ("C1CCSCC1", (0, 1, 2, 3, 4, 5)),
    ("C1CC2CCC1CC2", (0, 1, 2, 3, 4, 5, 6, 7)),
    # Aromatic connected-nonleaf stays excluded; the neutral leaf stays.
    ("c1ccccc1", ()),
    ("Cc1ccccc1", (0,)),
    ("c1ccc2ccccc2c1", ()),
    ("c1ccc(C2CCCCC2)cc1", (5, 6, 7, 8, 9)),
    # Articulation points stay excluded.
    ("CC1CCCCC1", (0, 2, 3, 4, 5, 6)),
    ("C1CCC2(CC1)CCCC2", (0, 1, 2, 4, 5, 6, 7, 8, 9)),
    ("O=C1NC(O)C2CCCCC12", (0, 2, 4, 5, 6, 7, 8, 9, 10)),
    # Leaves, a singleton root and a two-atom state stay reachable.
    ("CCO", (0, 2)),
    ("C", (0,)),
    ("CC", (0, 1)),
    # Charged chemistry: every charged centre and every neighbour of one is
    # excluded, whatever its structural source.
    ("C[N+](C)(C)CC(=O)[O-]", (6,)),
    ("C1CC[NH2+]CC1", (0, 1, 5)),
    ("[O-]C(=O)C1CCCCC1", (2, 4, 5, 6, 7, 8)),
    ("[NH4+]", ()),
    ("[OH-]", ()),
)

# Diverse bounded panel: charged, fused, bridged, spiro, S/Cl bearing and
# aromatic-plus-saturated mixtures.
_DIVERSE_PANEL = tuple(smiles for smiles, _ in _REFERENCE_PANEL) + (
    "ClC1CCCCC1",
    "CC(=O)NC1CCCCC1",
    "O=S1(=O)CCCC1",
    "CSC1CCCCC1",
    "OC1CCC(N)CC1",
    "N#CC1CCCCC1",
    "CC(=O)Oc1ccccc1C(=O)O",
)


def _state(smiles: str, slots: int = _SLOTS) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def _scarred(smiles: str, scar_slot: int, attachment: int, slots: int = 16):
    """A state carrying one SCAR bonded to ``attachment``.

    SCAR slots are occupied but are not elements, so a SCAR never enters the
    real-atom graph and never contributes real-atom degree.
    """

    base = _state(smiles, slots)
    atom_types = base.atom_types.copy()
    bonds = base.bonds.copy()
    hydrogens = base.implicit_h_counts.copy()
    atom_types[scar_slot] = SCAR_IDX
    bonds[scar_slot, attachment] = bonds[attachment, scar_slot] = 1
    hydrogens[attachment] -= 1
    scarred = MolecularGraph(atom_types, base.formal_charges.copy(), hydrogens, bonds)
    assert is_valid_state(scarred) and is_connected_or_null(scarred)
    return scarred


def _relabel_slots(state: MolecularGraph, order: np.ndarray) -> MolecularGraph:
    """Fixture-only persistent-slot relabeling: ``new[i] = old[order[i]]``."""

    return MolecularGraph(
        state.atom_types[order].copy(),
        state.formal_charges[order].copy(),
        state.implicit_h_counts[order].copy(),
        state.bonds[np.ix_(order, order)].copy(),
    )


def _real_slots(state: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))


def _real_degree(state: MolecularGraph, slot: int) -> int:
    return sum(
        1
        for other in _real_slots(state)
        if other != slot and int(state.bonds[slot, other]) != 0
    )


def _admitted_slots(state: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(slot) for slot in np.flatnonzero(process_v2_atom_delete_mask(state)))


# ---- The corrected semantics ----


def test_reference_admissions_match_the_frozen_panel() -> None:
    for smiles, expected in _REFERENCE_PANEL:
        assert _admitted_slots(_state(smiles)) == expected, smiles


def test_enumeration_mask_and_resolver_agree_and_are_deterministic() -> None:
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        actions = enumerate_process_v2_atom_deletes(state)
        assert actions == enumerate_process_v2_atom_deletes(state)
        enumerated = tuple(int(action.v) for action in actions)
        assert enumerated == tuple(sorted(enumerated)), smiles
        assert enumerated == _admitted_slots(state), smiles
        mask = process_v2_atom_delete_mask(state)
        assert mask.dtype == np.bool_
        assert mask.shape == (state.n_atoms,)
        for slot in range(state.n_atoms):
            resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
            assert bool(mask[slot]) is resolution.admitted, (smiles, slot)


def test_candidate_source_is_the_real_atom_degree_classification() -> None:
    """The source selects extra gates; it never exempts a slot from a gate."""

    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        for slot in range(state.n_atoms):
            resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
            if not bool(is_element(state.atom_types[slot])):
                assert resolution.real_degree == 0
                assert resolution.candidate_source is _INHERITED
                continue
            assert resolution.real_degree == _real_degree(state, slot), (smiles, slot)
            expected = (
                _CONNECTED_NONLEAF
                if resolution.real_degree >= CONNECTED_NONLEAF_MINIMUM_DEGREE
                else _INHERITED
            )
            assert resolution.candidate_source is expected, (smiles, slot)


# ---- The defect this round removes ----


def test_a_charged_leaf_deletion_is_now_excluded() -> None:
    """THE DEFECT.  A neutral leaf bonded to a charged centre must be excluded.

    Deleting slot 0, a methyl on the quaternary ammonium of
    ``C[N+](C)(C)CC(=O)[O-]``, changes the protected charged centre's bond row
    and implicit hydrogens.  The legacy dense rule admitted it and the first
    Process-V2 branch inherited that admission unchanged.
    """

    state = _state("C[N+](C)(C)CC(=O)[O-]")
    assert int(state.formal_charges[1]) == 1

    resolution = resolve_process_v2_atom_delete(state, AtomDelete(0))
    assert resolution.candidate_source is _INHERITED
    assert resolution.real_degree == 1
    assert resolution.admitted is False
    assert (
        resolution.rejection_code
        is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
    )
    assert resolution.successor is None and resolution.successor_key is None

    # The executor accepts it and the successor is connected, so the charge
    # policy is the only thing excluding it.
    assert is_valid_atom_delete(state, AtomDelete(0)) is True
    successor = apply_atom_delete(state, AtomDelete(0))
    assert is_connected_or_null(successor) is True
    assert charge_policy_preserved(state, successor) is False

    assert 0 not in _admitted_slots(state)


def test_deletion_adjacent_to_a_charged_centre_is_excluded() -> None:
    """Every neighbour of a charged centre, from either structural source."""

    inherited = _state("C[N+](C)(C)CC(=O)[O-]")
    for slot in (0, 2, 3):
        resolution = resolve_process_v2_atom_delete(inherited, AtomDelete(slot))
        assert resolution.candidate_source is _INHERITED
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
        ), slot

    connected = _state("C1CC[NH2+]CC1")
    assert int(connected.formal_charges[3]) == 1
    for slot in (2, 4):
        resolution = resolve_process_v2_atom_delete(connected, AtomDelete(slot))
        assert resolution.candidate_source is _CONNECTED_NONLEAF
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
        ), slot
    assert _admitted_slots(connected) == (0, 1, 5)


def test_a_charged_centre_is_never_deleted_from_either_source() -> None:
    charged_leaf = _state("[O-]C(=O)C1CCCCC1")
    assert int(charged_leaf.formal_charges[0]) == -1
    resolution = resolve_process_v2_atom_delete(charged_leaf, AtomDelete(0))
    assert resolution.admitted is False
    assert (
        resolution.rejection_code
        is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
    )

    charged_ring = _state("C1CC[NH2+]CC1")
    resolution = resolve_process_v2_atom_delete(charged_ring, AtomDelete(3))
    assert resolution.candidate_source is _CONNECTED_NONLEAF
    assert resolution.admitted is False
    assert (
        resolution.rejection_code
        is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
    )


def test_a_charged_singleton_is_excluded_but_a_neutral_one_still_reaches_null() -> None:
    """The capability is preserved; the unfiltered admission set is not.

    Deleting the only atom of a charged singleton empties the formal-charge
    array, which the charge policy refuses.  The neutral singleton still deletes
    to the null state, which the connectivity predicate accepts: the null state
    is a legal reversible source and is not a molecule.
    """

    for smiles in ("[NH4+]", "[OH-]"):
        state = _state(smiles)
        assert int(state.n_real_atoms) == 1
        assert int(state.formal_charges[0]) != 0
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(0))
        assert resolution.candidate_source is _INHERITED
        assert resolution.real_degree == 0
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
        ), smiles
        # The executor accepts it, so only the charge policy excludes it.
        assert is_valid_atom_delete(state, AtomDelete(0)) is True
        assert _admitted_slots(state) == (), smiles

    neutral = _state("C")
    resolution = resolve_process_v2_atom_delete(neutral, AtomDelete(0))
    assert resolution.candidate_source is _INHERITED
    assert resolution.real_degree == 0
    assert resolution.admitted is True
    successor = apply_atom_delete(neutral, AtomDelete(0))
    assert int(successor.n_real_atoms) == 0
    assert is_connected_or_null(successor) is True
    assert _admitted_slots(neutral) == (0,)


def test_a_neutral_leaf_is_still_admitted() -> None:
    """The inherited capability is preserved wherever the predicates pass."""

    for smiles, expected in (("CCO", (0, 2)), ("CC", (0, 1)), ("Cc1ccccc1", (0,))):
        state = _state(smiles)
        assert _admitted_slots(state) == expected, smiles
        for slot in expected:
            resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
            assert resolution.candidate_source is _INHERITED
            assert resolution.real_degree <= 1
            assert resolution.admitted is True, (smiles, slot)
            assert resolution.rejection_code is None
            assert resolution.successor is not None
            assert resolution.successor_key is not None


# ---- Connected-nonleaf gates ----


def test_carbocycle_delete_executes_to_the_exact_persistent_slot_successor() -> None:
    state = _state("C1CCCCC1", slots=8)
    assert _admitted_slots(state) == (0, 1, 2, 3, 4, 5)

    resolution = resolve_process_v2_atom_delete(state, AtomDelete(0))
    assert resolution.admitted
    assert resolution.rejection_code is None
    assert resolution.candidate_source is _CONNECTED_NONLEAF
    assert resolution.real_degree == CONNECTED_NONLEAF_MINIMUM_DEGREE
    assert resolution.semantic_aromatic_atom is False
    assert resolution.articulation_point is False
    assert resolution.scar_incident is False

    successor = apply_atom_delete(state, AtomDelete(0))
    expected_bonds = np.zeros((8, 8), dtype=np.int32)
    for left, right in ((1, 2), (2, 3), (3, 4), (4, 5)):
        expected_bonds[left, right] = expected_bonds[right, left] = 1
    assert np.array_equal(
        successor.atom_types,
        np.array(
            [NULL_IDX, _CARBON, _CARBON, _CARBON, _CARBON, _CARBON, NULL_IDX, NULL_IDX],
            dtype=np.int32,
        ),
    )
    assert np.array_equal(successor.formal_charges, np.zeros(8, dtype=np.int32))
    assert np.array_equal(
        successor.implicit_h_counts,
        np.array([0, 3, 2, 2, 2, 3, 0, 0], dtype=np.int32),
    )
    assert np.array_equal(successor.bonds, expected_bonds)
    assert molecular_graph_to_smiles(successor) == "CCCCC"
    assert np.array_equal(resolution.successor.atom_types, successor.atom_types)
    assert np.array_equal(resolution.successor.bonds, successor.bonds)


def test_saturated_heterocycle_delete_preserves_the_heteroatom() -> None:
    state = _state("C1CCOCC1", slots=8)
    assert int(state.atom_types[3]) == _OXYGEN
    assert _admitted_slots(state) == (0, 1, 2, 3, 4, 5)

    successor = apply_atom_delete(state, AtomDelete(0))
    expected_bonds = np.zeros((8, 8), dtype=np.int32)
    for left, right in ((1, 2), (2, 3), (3, 4), (4, 5)):
        expected_bonds[left, right] = expected_bonds[right, left] = 1
    assert np.array_equal(
        successor.atom_types,
        np.array(
            [NULL_IDX, _CARBON, _CARBON, _OXYGEN, _CARBON, _CARBON, NULL_IDX, NULL_IDX],
            dtype=np.int32,
        ),
    )
    assert np.array_equal(
        successor.implicit_h_counts,
        np.array([0, 3, 2, 0, 2, 3, 0, 0], dtype=np.int32),
    )
    assert np.array_equal(successor.bonds, expected_bonds)
    assert molecular_graph_to_smiles(successor) == "CCOCC"


def test_aromatic_connected_nonleaf_delete_remains_excluded() -> None:
    for smiles in ("c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1"):
        state = _state(smiles)
        for slot in _real_slots(state):
            if _real_degree(state, slot) < CONNECTED_NONLEAF_MINIMUM_DEGREE:
                continue
            resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
            assert resolution.candidate_source is _CONNECTED_NONLEAF
            assert resolution.admitted is False, (smiles, slot)
            assert resolution.semantic_aromatic_atom is True, (smiles, slot)
        assert not any(
            _real_degree(state, slot) >= CONNECTED_NONLEAF_MINIMUM_DEGREE
            for slot in _admitted_slots(state)
        ), smiles

    # Benzene's ring atoms are not articulation points and delete to a connected
    # successor, so aromaticity is the only gate that excludes them.
    benzene = _state("c1ccccc1")
    for slot in range(6):
        assert is_valid_atom_delete(benzene, AtomDelete(slot)) is True
        assert is_connected_or_null(apply_atom_delete(benzene, AtomDelete(slot))) is True
        resolution = resolve_process_v2_atom_delete(benzene, AtomDelete(slot))
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.AROMATIC_ATOM


def test_articulation_connected_nonleaf_delete_remains_excluded() -> None:
    """Excluded, and flagged as an articulation point, under the frozen order.

    The frozen gate order evaluates successor connectivity (gate 3) before the
    articulation gate (gate 8).  On a connected real-atom graph removing a cut
    vertex disconnects it by definition, and ``is_connected_or_null`` is
    evaluated over exactly that graph, so gate 3 always fires first and
    ``ARTICULATION_POINT`` is unreachable.  The exclusion and the structural fact
    are both asserted here; this test is what fails if the order ever moves.
    """

    for smiles, slot in (("CC1CCCCC1", 1), ("C1CCC2(CC1)CCCC2", 3)):
        state = _state(smiles)
        action = AtomDelete(slot)
        assert is_valid_atom_delete(state, action) is True, smiles
        successor = apply_atom_delete(state, action)
        assert is_valid_state(successor) is True, smiles
        assert is_connected_or_null(successor) is False, smiles

        resolution = resolve_process_v2_atom_delete(state, action)
        assert resolution.candidate_source is _CONNECTED_NONLEAF
        assert resolution.admitted is False, smiles
        assert resolution.articulation_point is True, smiles
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.SUCCESSOR_DISCONNECTED
        ), smiles
        assert slot not in _admitted_slots(state)


def test_a_scar_incident_connected_nonleaf_delete_is_excluded() -> None:
    """The gate this round adds: newly reached, and deferred.

    ``apply_atom_delete`` writes implicit hydrogen onto a SCAR neighbour, which
    is unsettled, so a ring deletion Process V2 newly reaches is held back
    pending a separate SCAR semantic decision.
    """

    for smiles, scar_slot, attachment, expected in (
        ("C1CCCCC1", 8, 0, (1, 2, 3, 4, 5)),
        ("C1CCCCC1", 8, 3, (0, 1, 2, 4, 5)),
        ("CC1CCCCC1", 10, 4, (0, 2, 3, 5, 6)),
    ):
        state = _scarred(smiles, scar_slot, attachment)
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(attachment))
        assert resolution.candidate_source is _CONNECTED_NONLEAF
        assert resolution.scar_incident is True, smiles
        assert resolution.admitted is False, smiles
        assert (
            resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.SCAR_INCIDENT
        ), smiles
        # Every other gate passes, which is what makes this a reachable witness.
        assert is_valid_atom_delete(state, AtomDelete(attachment)) is True
        assert is_connected_or_null(apply_atom_delete(state, AtomDelete(attachment)))
        assert resolution.semantic_aromatic_atom is False
        assert resolution.articulation_point is False
        assert _admitted_slots(state) == expected, smiles


def test_a_scar_adjacent_leaf_is_still_admitted() -> None:
    """The inherited capability the recorded decision does not withdraw.

    Gate 9 is deliberately connected-nonleaf only: a SCAR-adjacent leaf deletion
    was already reachable before Process V2, so excluding it would remove an
    inherited capability rather than defer a new one.
    """

    for attachment in (0, 3):
        state = _scarred("CCCC", 8, attachment)
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(attachment))
        assert resolution.candidate_source is _INHERITED
        assert resolution.real_degree == 1
        assert resolution.scar_incident is True, attachment
        assert resolution.admitted is True, attachment
        assert resolution.rejection_code is None
        assert attachment in _admitted_slots(state)


def test_scar_and_non_element_slots_are_never_admitted() -> None:
    state = _scarred("C1CCCCC1", 8, 0, slots=12)
    mask = process_v2_atom_delete_mask(state)
    assert bool(mask[8]) is False
    assert _admitted_slots(state) == (1, 2, 3, 4, 5)
    for slot in (8, 10):
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.NOT_A_REAL_ELEMENT
        ), slot


# ---- Boundaries ----


def test_out_of_range_slots_are_rejected_without_touching_the_executor() -> None:
    state = _state("C1CCCCC1", slots=8)
    for slot in (-1, 8, 99):
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
        assert resolution.admitted is False
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.INVALID_SLOT
        assert resolution.source_key is not None


def test_a_disconnected_source_state_is_rejected_as_an_invalid_source() -> None:
    left = smiles_to_molecular_graph("C1CCCCC1")
    right = smiles_to_molecular_graph("C1CCCCC1")
    n = left.n_atoms + right.n_atoms
    atom_types = np.concatenate((left.atom_types, right.atom_types))
    charges = np.concatenate((left.formal_charges, right.formal_charges))
    hydrogens = np.concatenate((left.implicit_h_counts, right.implicit_h_counts))
    bonds = np.zeros((n, n), dtype=np.int32)
    bonds[: left.n_atoms, : left.n_atoms] = left.bonds
    bonds[left.n_atoms :, left.n_atoms :] = right.bonds
    disconnected = MolecularGraph(atom_types, charges, hydrogens, bonds)
    assert is_connected_or_null(disconnected) is False

    assert enumerate_process_v2_atom_deletes(disconnected) == ()
    resolution = resolve_process_v2_atom_delete(disconnected, AtomDelete(0))
    assert resolution.admitted is False
    assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.INVALID_SOURCE
    assert resolution.source_key is None


# ---- Properties of the whole admitted fiber ----


def test_every_admission_yields_a_valid_connected_charge_preserving_successor() -> None:
    admitted_total = 0
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        for slot in _admitted_slots(state):
            action = AtomDelete(slot)
            admitted_total += 1
            assert is_valid_atom_delete(state, action) is True, (smiles, slot)
            successor = apply_atom_delete(state, action)
            assert is_valid_state(successor) is True, (smiles, slot)
            assert is_connected_or_null(successor) is True, (smiles, slot)
            assert charge_policy_preserved(state, successor) is True, (smiles, slot)
    assert admitted_total > 0


def test_no_slot_failing_a_common_gate_is_admitted_from_either_source() -> None:
    """The uniform contract, checked against the primitives on every slot."""

    excluded_for_connectivity = 0
    excluded_for_charge = 0
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        mask = process_v2_atom_delete_mask(state)
        for slot in _real_slots(state):
            action = AtomDelete(slot)
            executable = is_valid_atom_delete(state, action)
            successor = apply_atom_delete(state, action)
            connected = is_connected_or_null(successor)
            charge_ok = charge_policy_preserved(state, successor)
            if executable and connected and charge_ok:
                continue
            assert bool(mask[slot]) is False, (smiles, slot)
            if executable and not connected:
                excluded_for_connectivity += 1
            if executable and connected and not charge_ok:
                excluded_for_charge += 1
    assert excluded_for_connectivity > 0
    assert excluded_for_charge > 0


def test_admission_is_slot_relabeling_covariant() -> None:
    """The mask depends on persistent slots only, never on SMILES atom order."""

    rng = np.random.default_rng(20260802)
    for smiles in ("C1CCCCC1", "CC1CCCCC1", "O=C1NC(O)C2CCCCC12", "C1CC[NH2+]CC1"):
        state = _state(smiles)
        mask = process_v2_atom_delete_mask(state)
        for _ in range(3):
            order = rng.permutation(state.n_atoms)
            relabeled = _relabel_slots(state, order)
            relabeled_mask = process_v2_atom_delete_mask(relabeled)
            assert np.array_equal(relabeled_mask, mask[order]), smiles


@pytest.mark.parametrize("smiles", ("C1CCCCC1", "CC1CCCCC1", "C1CC2CCC1CC2"))
def test_connected_nonleaf_admissions_have_the_expected_degree(smiles: str) -> None:
    state = _state(smiles)
    for slot in _admitted_slots(state):
        resolution = resolve_process_v2_atom_delete(state, AtomDelete(slot))
        if resolution.candidate_source is _CONNECTED_NONLEAF:
            assert resolution.real_degree >= CONNECTED_NONLEAF_MINIMUM_DEGREE
        else:
            assert resolution.real_degree < CONNECTED_NONLEAF_MINIMUM_DEGREE
