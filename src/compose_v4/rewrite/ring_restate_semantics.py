"""Resonance-invariant successor groups for ring-system restatement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.kernel import RewriteSystem, canonical_state_key
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_transitions,
)
from compose_v4.rewrite.tracelets import RingSystemRestate

RingRestateEdgeTransition = tuple[int, int, int, int]


@dataclass(frozen=True)
class RingRestateSemanticGroups:
    """Raw executable actions aligned to canonical semantic successor groups."""

    actions: tuple[RingSystemRestate, ...]
    successor_group_ids: tuple[int, ...]
    successor_keys: tuple[str, ...]
    group_descriptors: tuple[tuple[RingRestateEdgeTransition, ...], ...]
    group_multiplicities: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.successor_group_ids):
            raise ValueError("ring-restatement action/group alignment differs")
        if not (
            len(self.successor_keys)
            == len(self.group_descriptors)
            == len(self.group_multiplicities)
        ):
            raise ValueError("ring-restatement semantic group metadata differs")
        if self.successor_group_ids and (
            min(self.successor_group_ids) < 0
            or max(self.successor_group_ids) >= len(self.successor_keys)
        ):
            raise ValueError("ring-restatement group ID lies outside metadata")
        if any(not descriptor for descriptor in self.group_descriptors):
            raise ValueError("ring-restatement semantic descriptors must be nonempty")
        observed = tuple(
            self.successor_group_ids.count(index)
            for index in range(len(self.successor_keys))
        )
        if observed != self.group_multiplicities:
            raise ValueError("ring-restatement multiplicities disagree with group IDs")


def ring_restate_semantic_transition_descriptor(
    source: MolecularGraph,
    successor: MolecularGraph,
) -> tuple[RingRestateEdgeTransition, ...]:
    """Describe every resonance-invariant source-to-successor bond change."""

    if source.n_atoms != successor.n_atoms:
        raise ValueError("ring restatement cannot change persistent-slot capacity")
    if not np.array_equal(source.atom_types, successor.atom_types):
        raise ValueError("ring restatement cannot change atom types")
    if not np.array_equal(source.formal_charges, successor.formal_charges):
        raise ValueError("ring restatement cannot change formal charges")
    source_classes = resonance_invariant_bond_classes(source)
    target_classes = resonance_invariant_bond_classes(successor)
    real_slots = tuple(
        int(slot) for slot in np.flatnonzero(is_element(source.atom_types))
    )
    return tuple(
        (
            left,
            right,
            int(source_classes[left, right]),
            int(target_classes[left, right]),
        )
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(source_classes[left, right]) != int(target_classes[left, right])
    )


def enumerate_ring_restate_semantic_groups(
    state: MolecularGraph,
    *,
    system: RewriteSystem,
) -> RingRestateSemanticGroups:
    """Execute once, charge-filter, canonical-group, and describe restatements."""

    retained: list[
        tuple[RingSystemRestate, str, tuple[RingRestateEdgeTransition, ...]]
    ] = []
    for action, successor in enumerate_ring_system_restate_transitions(
        state,
        system=system,
    ):
        if not charge_policy_preserved(state, successor):
            continue
        descriptor = ring_restate_semantic_transition_descriptor(state, successor)
        if not descriptor:
            raise ValueError(
                "productive ring restatement lacks a semantic edge transition"
            )
        retained.append((action, canonical_state_key(successor), descriptor))

    successor_keys = tuple(sorted({key for _, key, _ in retained}))
    group_index = {key: index for index, key in enumerate(successor_keys)}
    descriptors_by_group: list[set[tuple[RingRestateEdgeTransition, ...]]] = [
        set() for _ in successor_keys
    ]
    group_ids: list[int] = []
    for _, key, descriptor in retained:
        index = group_index[key]
        group_ids.append(index)
        descriptors_by_group[index].add(descriptor)
    if any(len(descriptors) != 1 for descriptors in descriptors_by_group):
        raise ValueError(
            "raw aliases within one canonical ring-restatement successor "
            "have divergent semantic descriptors"
        )
    group_descriptors = tuple(
        next(iter(descriptors)) for descriptors in descriptors_by_group
    )
    group_ids_tuple = tuple(group_ids)
    multiplicities = tuple(
        group_ids_tuple.count(index) for index in range(len(successor_keys))
    )
    return RingRestateSemanticGroups(
        actions=tuple(action for action, _, _ in retained),
        successor_group_ids=group_ids_tuple,
        successor_keys=successor_keys,
        group_descriptors=group_descriptors,
        group_multiplicities=multiplicities,
    )


__all__ = [
    "RingRestateEdgeTransition",
    "RingRestateSemanticGroups",
    "enumerate_ring_restate_semantic_groups",
    "ring_restate_semantic_transition_descriptor",
]
