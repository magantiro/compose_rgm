"""Generic protected pruning around an exactly retained molecular core.

The operation chooses what survives before it chooses what is removed.  A proposal
deletes one or two bridge-separated complements while preserving every atom, bond and
attribute in the retained component.  The complete deletion schedule is executed
privately, so endpoint constraints are never imposed on intermediate states.

This module is task and oracle blind.  It enumerates exact executable programs only;
QED, similarity, synthetic accessibility and docking belong to callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.dynamic_program_synthesis import _pendant_fragments
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.operators import AtomDelete, CycleOpenEdge


@dataclass(frozen=True)
class RetainedCoreProposal:
    """One exactly executed protected retained-core pruning program."""

    product: MolecularGraph
    actions: tuple[dict, ...]
    stages: tuple[dict, ...]


def _real_slots(graph: MolecularGraph) -> frozenset[int]:
    return frozenset(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))


def _compile_fragment(
    source: MolecularGraph,
    fragment_slots: tuple[int, ...],
    retained_anchor: int,
    *,
    primitive_room: int,
) -> tuple[MolecularGraph, tuple[dict, ...]]:
    """Compile one named complement into a deterministic exact deletion schedule."""

    fragment = frozenset(fragment_slots)
    real = _real_slots(source)
    if not fragment or not fragment < real or retained_anchor not in real:
        raise ValueError("retained-core fragment is absent or consumes the whole graph")
    boundary = {
        int(neighbor)
        for slot in fragment
        for neighbor in np.flatnonzero(source.bonds[slot])
        if int(neighbor) not in fragment
    }
    if boundary != {retained_anchor}:
        raise ValueError("retained-core fragment does not have exactly one boundary")

    current = source
    actions: list[dict] = []
    remaining = set(fragment)

    # Cyclic pendant lobes must first be opened internally.  CycleOpenEdge rejects
    # non-cycle edges, so accepting the first exact executable edge is deterministic
    # and cannot sever the retained boundary.
    while True:
        opened = False
        for left in sorted(remaining):
            for right in sorted(remaining):
                if left >= right or not current.bonds[left, right]:
                    continue
                record = encode_action("cycle_open", CycleOpenEdge(left, right))
                try:
                    following, _ = execute_program(current, [record])
                except ValueError:
                    continue
                actions.append(record)
                current = following
                opened = True
                break
            if opened:
                break
        if not opened:
            break
        if len(actions) >= primitive_room:
            raise ValueError("retained-core cycle opening exhausts the primitive budget")

    while remaining:
        accepted = False
        for slot in sorted(remaining):
            if int(np.count_nonzero(current.bonds[slot])) != 1:
                continue
            record = encode_action("atom_delete", AtomDelete(slot))
            try:
                following, _ = execute_program(current, [record])
            except ValueError:
                continue
            actions.append(record)
            current = following
            remaining.remove(slot)
            accepted = True
            break
        if not accepted:
            raise ValueError("retained-core fragment has no exact legal deletion order")
        if len(actions) > primitive_room:
            raise ValueError("retained-core deletion exceeds the primitive budget")
    return current, tuple(actions)


def enumerate_retained_core_prunes(
    source: MolecularGraph,
    *,
    maximum_fragment_atoms: int = 16,
    maximum_stages: int = 2,
    maximum_primitives: int = 32,
    maximum_prefixes: int = 4096,
) -> tuple[RetainedCoreProposal, ...]:
    """Enumerate distinct exact retained-core pruning endpoints.

    The enumeration is exhaustive within the declared one-boundary fragment, stage and
    primitive limits.  Every returned proposal has already replayed through the ordinary
    COMPOSE executor.  Prefixes are canonicalized by exact molecular state, retaining
    the shortest primitive realization.
    """

    if not 1 <= maximum_fragment_atoms <= 32:
        raise ValueError("maximum_fragment_atoms must be between 1 and 32")
    if not 1 <= maximum_stages <= 4:
        raise ValueError("maximum_stages must be between 1 and 4")
    if not 1 <= maximum_primitives <= 32:
        raise ValueError("maximum_primitives must be between 1 and 32")
    if maximum_prefixes < 1:
        raise ValueError("maximum_prefixes must be positive")

    from compose_v4.rewrite.kernel import canonical_state_key

    prefixes = [(source, (), ())]
    accepted: dict[str, RetainedCoreProposal] = {}
    for _stage_index in range(maximum_stages):
        following_prefixes: dict[str, tuple[MolecularGraph, tuple[dict, ...], tuple[dict, ...]]] = {}
        for current, prior_actions, prior_stages in prefixes:
            room = maximum_primitives - len(prior_actions)
            if room < 1:
                continue
            for fragment, anchor in _pendant_fragments(
                current, maximum=maximum_fragment_atoms
            ):
                try:
                    product, actions = _compile_fragment(
                        current, fragment, anchor, primitive_room=room
                    )
                except ValueError:
                    continue
                combined = (*prior_actions, *actions)
                if len(combined) > maximum_primitives:
                    continue
                key = canonical_state_key(product)
                stage = {
                    "family": "retained_core_prune",
                    "fragment_slots": list(fragment),
                    "retained_anchor": int(anchor),
                    "deleted_atoms": len(fragment),
                    "primitive_edits": len(actions),
                }
                stages = (*prior_stages, stage)
                proposal = RetainedCoreProposal(product, combined, stages)
                previous = accepted.get(key)
                if previous is None or len(combined) < len(previous.actions):
                    accepted[key] = proposal
                queued = following_prefixes.get(key)
                if queued is None or len(combined) < len(queued[1]):
                    following_prefixes[key] = (product, combined, stages)
        prefixes = sorted(
            following_prefixes.values(),
            key=lambda row: (len(row[1]), canonical_state_key(row[0])),
        )[:maximum_prefixes]
        if not prefixes:
            break
    return tuple(accepted[key] for key in sorted(accepted))
