"""Process-V2 connected-nonleaf atom deletion at the resolver and executor level.

Process V2 keeps the V1 root, singleton and leaf ``atom_delete`` behaviour
bit-for-bit and additionally admits a real slot of real-atom degree at least two
when every declared condition holds.  These tests pin the declared behaviour of
that additional fiber against the unchanged production executor.
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
    ProcessV2AtomDeleteRejectionCode,
    enumerate_process_v2_connected_nonleaf_atom_deletes,
    process_v2_connected_nonleaf_atom_delete_mask,
    resolve_process_v2_connected_nonleaf_atom_delete,
)

_SLOTS = 24
_CARBON = ELEMENT_TO_IDX["C"]
_OXYGEN = ELEMENT_TO_IDX["O"]

# Verified reference behaviour for the additional connected-nonleaf fiber.
_REFERENCE_PANEL = (
    ("C1CCCCC1", (0, 1, 2, 3, 4, 5)),
    ("C1CCOCC1", (0, 1, 2, 3, 4, 5)),
    ("c1ccccc1", ()),
    ("Cc1ccccc1", ()),
    ("c1ccc2ccccc2c1", ()),
    ("CC1CCCCC1", (2, 3, 4, 5, 6)),
    ("C1CC2CCC1CC2", (0, 1, 2, 3, 4, 5, 6, 7)),
    ("O=C1NC(O)C2CCCCC12", (2, 5, 6, 7, 8, 9, 10)),
    ("CCO", ()),
    ("C", ()),
    ("C[N+](C)(C)CC(=O)[O-]", ()),
)

# Diverse bounded panel: charged, fused, bridged, spiro, S/Cl bearing and
# aromatic-plus-saturated mixtures.
_DIVERSE_PANEL = (
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCSCC1",
    "C1CCNCC1",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccc2ccccc2c1",
    "CC1CCCCC1",
    "C1CC2CCC1CC2",
    "C1CCC2(CC1)CCCC2",
    "O=C1NC(O)C2CCCCC12",
    "ClC1CCCCC1",
    "CC(=O)NC1CCCCC1",
    "O=S1(=O)CCCC1",
    "C1CC[NH2+]CC1",
    "C[N+](C)(C)CC(=O)[O-]",
    "c1ccc(C2CCCCC2)cc1",
    "CSC1CCCCC1",
    "CCO",
    "C",
)


def _state(smiles: str, slots: int = _SLOTS) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


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
    return tuple(
        int(slot)
        for slot in np.flatnonzero(process_v2_connected_nonleaf_atom_delete_mask(state))
    )


def test_reference_connected_nonleaf_admissions_match_the_frozen_panel() -> None:
    for smiles, expected in _REFERENCE_PANEL:
        assert _admitted_slots(_state(smiles)) == expected, smiles


def test_enumeration_mask_and_resolver_agree_and_are_deterministic() -> None:
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        actions = enumerate_process_v2_connected_nonleaf_atom_deletes(state)
        assert actions == enumerate_process_v2_connected_nonleaf_atom_deletes(state)
        enumerated = tuple(int(action.v) for action in actions)
        assert enumerated == tuple(sorted(enumerated)), smiles
        assert enumerated == _admitted_slots(state), smiles
        mask = process_v2_connected_nonleaf_atom_delete_mask(state)
        assert mask.dtype == np.bool_
        assert mask.shape == (state.n_atoms,)
        for slot in range(state.n_atoms):
            resolution = resolve_process_v2_connected_nonleaf_atom_delete(
                state,
                AtomDelete(slot),
            )
            assert bool(mask[slot]) is resolution.admitted, (smiles, slot)


def test_carbocycle_delete_executes_to_the_exact_persistent_slot_successor() -> None:
    state = _state("C1CCCCC1", slots=8)
    assert _admitted_slots(state) == (0, 1, 2, 3, 4, 5)

    resolution = resolve_process_v2_connected_nonleaf_atom_delete(state, AtomDelete(0))
    assert resolution.admitted
    assert resolution.rejection_code is None
    assert resolution.real_degree == CONNECTED_NONLEAF_MINIMUM_DEGREE
    assert resolution.semantic_aromatic_atom is False
    assert resolution.articulation_point is False

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
    assert np.array_equal(
        successor.formal_charges,
        np.zeros(8, dtype=np.int32),
    )
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
        assert _admitted_slots(state) == (), smiles
        for slot in _real_slots(state):
            if _real_degree(state, slot) < CONNECTED_NONLEAF_MINIMUM_DEGREE:
                continue
            resolution = resolve_process_v2_connected_nonleaf_atom_delete(
                state,
                AtomDelete(slot),
            )
            assert resolution.admitted is False
            assert resolution.semantic_aromatic_atom is True
            assert (
                resolution.rejection_code
                is ProcessV2AtomDeleteRejectionCode.AROMATIC_ATOM
            ), (smiles, slot)


def test_articulation_delete_is_excluded_although_the_executor_accepts_it() -> None:
    """``is_valid_atom_delete`` is not a connectivity predicate.

    Both fixtures below are accepted by the unchanged executor and yield a
    valence-valid but DISCONNECTED successor, so the articulation and
    connectivity conditions are independently load bearing.
    """

    for smiles, slot in (("CC1CCCCC1", 1), ("C1CCC2(CC1)CCCC2", 3)):
        state = _state(smiles)
        action = AtomDelete(slot)
        assert is_valid_atom_delete(state, action) is True, smiles
        successor = apply_atom_delete(state, action)
        assert is_valid_state(successor) is True, smiles
        assert is_connected_or_null(successor) is False, smiles
        resolution = resolve_process_v2_connected_nonleaf_atom_delete(state, action)
        assert resolution.admitted is False
        assert resolution.articulation_point is True
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.ARTICULATION_POINT
        ), smiles
        assert slot not in _admitted_slots(state)


def test_charge_policy_violating_deletes_remain_excluded() -> None:
    state = _state("C1CC[NH2+]CC1")
    charged = tuple(int(slot) for slot in np.flatnonzero(state.formal_charges != 0))
    assert charged == (3,)
    for slot in (2, 3, 4):
        resolution = resolve_process_v2_connected_nonleaf_atom_delete(
            state,
            AtomDelete(slot),
        )
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
        ), slot
    assert _admitted_slots(state) == (0, 1, 5)


def test_leaf_singleton_and_root_slots_stay_outside_the_expansion() -> None:
    for smiles, slots in (
        ("CCO", (0, 2)),
        ("C", (0,)),
        ("Cc1ccccc1", (0,)),
    ):
        state = _state(smiles)
        for slot in slots:
            assert _real_degree(state, slot) < CONNECTED_NONLEAF_MINIMUM_DEGREE
            resolution = resolve_process_v2_connected_nonleaf_atom_delete(
                state,
                AtomDelete(slot),
            )
            assert resolution.admitted is False
            assert (
                resolution.rejection_code
                is ProcessV2AtomDeleteRejectionCode.OUTSIDE_CONNECTED_NONLEAF_EXPANSION
            ), (smiles, slot)


def test_scar_and_non_element_slots_are_never_admitted() -> None:
    base = _state("C1CCCCC1", slots=12)
    atom_types = base.atom_types.copy()
    atom_types[7] = SCAR_IDX
    scarred = MolecularGraph(
        atom_types,
        base.formal_charges.copy(),
        base.implicit_h_counts.copy(),
        base.bonds.copy(),
    )
    assert is_valid_state(scarred) and is_connected_or_null(scarred)

    mask = process_v2_connected_nonleaf_atom_delete_mask(scarred)
    assert bool(mask[7]) is False
    assert _admitted_slots(scarred) == (0, 1, 2, 3, 4, 5)
    for slot in (7, 9):
        resolution = resolve_process_v2_connected_nonleaf_atom_delete(
            scarred,
            AtomDelete(slot),
        )
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is ProcessV2AtomDeleteRejectionCode.NOT_A_REAL_ELEMENT
        ), slot


def test_out_of_range_slots_are_rejected_without_touching_the_executor() -> None:
    state = _state("C1CCCCC1", slots=8)
    for slot in (-1, 8, 99):
        resolution = resolve_process_v2_connected_nonleaf_atom_delete(
            state,
            AtomDelete(slot),
        )
        assert resolution.admitted is False
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.INVALID_SLOT


def test_every_admission_yields_a_valid_connected_charge_preserving_successor() -> None:
    """Invalid-valence and disconnected successors are excluded by construction."""

    admitted_total = 0
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        for slot in _real_slots(state):
            action = AtomDelete(slot)
            admitted = bool(
                process_v2_connected_nonleaf_atom_delete_mask(state)[slot]
            )
            if not admitted:
                continue
            admitted_total += 1
            assert is_valid_atom_delete(state, action) is True, (smiles, slot)
            successor = apply_atom_delete(state, action)
            assert is_valid_state(successor) is True, (smiles, slot)
            assert is_connected_or_null(successor) is True, (smiles, slot)
            assert charge_policy_preserved(state, successor) is True, (smiles, slot)
    assert admitted_total > 0


def test_no_slot_with_an_invalid_or_disconnected_successor_is_admitted() -> None:
    excluded_for_connectivity = 0
    for smiles in _DIVERSE_PANEL:
        state = _state(smiles)
        mask = process_v2_connected_nonleaf_atom_delete_mask(state)
        for slot in _real_slots(state):
            action = AtomDelete(slot)
            successor = apply_atom_delete(state, action)
            executable = is_valid_atom_delete(state, action)
            connected = is_connected_or_null(successor)
            if executable and connected:
                continue
            assert bool(mask[slot]) is False, (smiles, slot)
            if executable and not connected:
                excluded_for_connectivity += 1
    assert excluded_for_connectivity > 0


def test_admission_is_a_slot_addressed_relabeling_covariant() -> None:
    """The mask depends on persistent slots only, never on SMILES atom order."""

    rng = np.random.default_rng(20260802)
    for smiles in ("C1CCCCC1", "CC1CCCCC1", "O=C1NC(O)C2CCCCC12", "C1CC[NH2+]CC1"):
        state = _state(smiles)
        mask = process_v2_connected_nonleaf_atom_delete_mask(state)
        for _ in range(3):
            order = rng.permutation(state.n_atoms)
            relabeled = _relabel_slots(state, order)
            relabeled_mask = process_v2_connected_nonleaf_atom_delete_mask(relabeled)
            assert np.array_equal(relabeled_mask, mask[order]), smiles


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

    assert enumerate_process_v2_connected_nonleaf_atom_deletes(disconnected) == ()
    resolution = resolve_process_v2_connected_nonleaf_atom_delete(
        disconnected,
        AtomDelete(0),
    )
    assert resolution.admitted is False
    assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.INVALID_SOURCE


@pytest.mark.parametrize("smiles", ("C1CCCCC1", "CC1CCCCC1", "C1CC2CCC1CC2"))
def test_admitted_slots_all_have_connected_nonleaf_degree(smiles: str) -> None:
    state = _state(smiles)
    for slot in _admitted_slots(state):
        assert _real_degree(state, slot) >= CONNECTED_NONLEAF_MINIMUM_DEGREE
