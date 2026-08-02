"""Representation invariance of the Process-V2 connected-nonleaf deletion fiber.

Two properties are load-bearing for the successor-level quotient and are pinned
here rather than argued:

* **Slot-relabel equivariance.** The persistent-slot tensor is a coordinate
  representation, not the semantic dimension. Relabelling the occupied slots of
  one molecule must permute the admission mask by exactly the same permutation.
* **Kekule-alias invariance of the successor fiber.** Executable states are
  Kekule encodings. Two aliases of one molecule must expose the same set of
  canonical molecular successors, even though their exact successor arrays
  differ.

Neither property holds automatically: the mask is built from an RDKit
perception pass, a networkx articulation computation, and executor replay
against exact stored bond orders.
"""

from __future__ import annotations

import itertools

import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.aromatic_kekule import (
    enumerate_component_factored_kekule_assignments,
    instantiate_component_factored_kekule_alias,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomDelete, apply_atom_delete
from compose_v4.rewrite.process_v2_atom_delete import (
    process_v2_connected_nonleaf_atom_delete_mask,
)

_PANEL = (
    "C1CCCCC1",
    "C1CCOCC1",
    "CC1CCCCC1",
    "O=C1NC(O)C2CCCCC12",
    "C1CC2CCC1CC2",
    "c1ccc2c(c1)CCCC2",
    "C1CCC2(CC1)CCCCC2",
    "OC1CCC(N)CC1",
    "N#CC1CCCCC1",
    "CC(=O)Oc1ccccc1C(=O)O",
)


def _state(smiles: str, n_slots: int = 24) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def _relabelled(state: MolecularGraph, permutation: dict[int, int]) -> MolecularGraph:
    atom_types = np.zeros_like(state.atom_types)
    formal_charges = np.zeros_like(state.formal_charges)
    implicit_h = np.zeros_like(state.implicit_h_counts)
    bonds = np.zeros_like(state.bonds)
    for old, new in permutation.items():
        atom_types[new] = state.atom_types[old]
        formal_charges[new] = state.formal_charges[old]
        implicit_h[new] = state.implicit_h_counts[old]
    for left, new_left in permutation.items():
        for right, new_right in permutation.items():
            bonds[new_left, new_right] = state.bonds[left, right]
    return MolecularGraph(atom_types, formal_charges, implicit_h, bonds)


def _successor_keys(state: MolecularGraph) -> frozenset[str]:
    mask = process_v2_connected_nonleaf_atom_delete_mask(state)
    return frozenset(
        canonical_state_key(apply_atom_delete(state, AtomDelete(int(slot))))
        for slot in np.flatnonzero(mask)
    )


def test_admission_mask_is_slot_relabel_equivariant() -> None:
    generator = np.random.default_rng(20260802)
    compared = 0
    for smiles in _PANEL:
        state = _state(smiles)
        expected_mask = process_v2_connected_nonleaf_atom_delete_mask(state)
        occupied = [int(slot) for slot in np.flatnonzero(is_element(state.atom_types))]
        for _ in range(6):
            targets = generator.choice(
                state.n_atoms,
                size=len(occupied),
                replace=False,
            ).tolist()
            permutation = {old: int(new) for old, new in zip(occupied, targets)}
            relabelled = _relabelled(state, permutation)
            assert canonical_state_key(relabelled) == canonical_state_key(state)
            permuted_expectation = np.zeros_like(expected_mask)
            for old, new in permutation.items():
                permuted_expectation[new] = expected_mask[old]
            observed = process_v2_connected_nonleaf_atom_delete_mask(relabelled)
            assert np.array_equal(observed, permuted_expectation), smiles
            compared += 1
    assert compared == 6 * len(_PANEL)


def test_successor_fiber_is_invariant_across_kekule_aliases() -> None:
    compared = 0
    for smiles in _PANEL:
        state = _state(smiles)
        components = enumerate_component_factored_kekule_assignments(state)
        if not components:
            continue
        source_key = canonical_state_key(state)
        expected: frozenset[str] | None = None
        for selection in itertools.islice(
            itertools.product(*(component.bond_orders for component in components)),
            8,
        ):
            alias = instantiate_component_factored_kekule_alias(
                state,
                components,
                tuple(selection),
            )
            if canonical_state_key(alias) != source_key:
                continue
            observed = _successor_keys(alias)
            if expected is None:
                expected = observed
            else:
                assert observed == expected, smiles
            compared += 1
    assert compared > 0
