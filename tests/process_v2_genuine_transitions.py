"""Genuine Editing-V2 Active8 transitions, one per capability cell, by discovery.

Not a test module.  It is the fixture source the Process-V2 Gate-0 suites build
their stand-in decision index from, and the reason it exists is that Gate 0 no
longer consumes a classified transition: it consumes the RAW evidence the index
publishes -- the exact persistent-slot source and successor states, the ActionV4
record, the executor rule, the model family and the raw candidate counts -- and
derives the capability cell, the family context and the three strata itself.  A
fixture that invents ``capability_cell_id`` therefore proves nothing about the
path production takes.

**Discovery, not construction.**  Nothing below chooses a capability cell and
then builds chemistry to land in it.  The search is:

1. take small seed molecules and pad them to :data:`SLOTS` persistent slots;
2. enumerate the legal Active8 marks at each seed through the PRODUCTION
   legal-event fibers -- ``enumerate_process_v2_atom_deletes``,
   ``enumerate_semantic_atom_restates``, ``enumerate_pendant_graft_actions``,
   ``enumerate_cycle_close_edges``, ``enumerate_cycle_open_edges``,
   ``enumerate_ring_system_restate_actions``, and the slot/order sweeps the
   micro families are defined over;
3. execute every candidate through the production Active8 runtime
   ``editing_v2_semantic_rewrite_system()``, keeping only what it admits;
4. encode the action with ``action_codec_v4.encode_action``;
5. classify the resulting exact transition with the PRODUCTION classifier
   ``classify_action_family_context`` and BUCKET it under the cell it genuinely
   lands in, first hit wins in seed and enumeration order.

So the cells are an OUTPUT of the search.  If a seed stops reaching a cell the
fixture loses that cell and says so; it cannot quietly keep the label.

The candidate counts are derived from the same enumeration, with the production
meanings: ``raw_mark_count`` is every legal Active8 mark at the exact source,
``canonical_successor_count`` the distinct canonical successors those marks
reach, ``matching_mark_count`` the marks sharing the teacher's action identity,
``exact_successor_mark_count`` those of them that reach the teacher's exact
persistent-slot successor, and ``successor_alias_count`` every mark landing on
the teacher's canonical successor.  The production checker measures the same
quantities over the MODEL's marked law, which is a subset of the legal fiber;
the fiber is the model-free upper bound of it, and the frozen aggregation
invariants Gate 0 checks (``1 <= aliases <= raw``, ``successors <= raw``,
``exact <= matching``) are asserted here rather than assumed.

The search costs a few seconds and is cached for the process, so it runs once
per session and is re-derived every session rather than committed as a table.
Run this module directly to print what it found.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    NULL_IDX,
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellError,
    classify_action_family_context,
    load_semantic_capability_cell_registry,
)
from compose_v4.rewrite.action_codec_v4 import ActionCodecV4Error, encode_action
from compose_v4.rewrite.factorized_fiber import enumerate_pendant_graft_actions
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    MICRO_BOND_CLASSES,
    AtomInsert,
    BondReorder,
    enumerate_cycle_close_edges,
    enumerate_cycle_open_edges,
    enumerate_semantic_atom_restates,
)
from compose_v4.rewrite.process_v2_atom_delete import enumerate_process_v2_atom_deletes
from compose_v4.rewrite.semantic_trace import RewriteStep
from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions

#: Persistent-slot capacity of every discovered state.  Sixteen holds the widest
#: seed with room for a birth, and matches the V1 migration fixture's capacity.
SLOTS = 16

#: The seeds the search runs over, smallest first.  They are chosen for STRUCTURE
#: -- an empty state, a single atom, a chain, a ring, an aromatic ring, pendant
#: substituents, a chain long enough to spiro-close, a fused and a spiro ring
#: system, an unsaturated fused system, and one sulfur so a same-element valence
#: change exists -- never for a target cell.  Which cell each one reaches is
#: measured, and a seed that reaches nothing new simply costs time.
SEED_SMILES: tuple[str, ...] = (
    "",
    "C",
    "CC",
    "CCO",
    "CCCCCC",
    "C1CCCCC1",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CSC",
    "C1CCCCC1CCCCC",
    "C1CCC2CCCCC2C1",
    "C1CCC2(CC1)CCCCC2",
    "C1=CC2CCCCC2C1",
)

#: The elements and hydrogen counts the ``atom_insert`` sweep offers.  The other
#: families need no such declaration: their production enumerators are complete.
_BIRTH_ELEMENTS: tuple[str, ...] = ("C", "N", "O")
_BIRTH_HYDROGEN_COUNTS: tuple[int, ...] = (0, 1, 2, 3, 4)


class GenuineTransitionError(RuntimeError):
    """The discovery search produced something that is not a usable teacher."""


@dataclass(frozen=True)
class GenuineTransition:
    """One executed Active8 transition and everything the index would publish.

    ``model_family`` and ``family_context`` are what the production classifier
    RETURNED for this transition; they are recorded, not requested.
    """

    seed_smiles: str
    source_state: MolecularGraph
    successor_state: MolecularGraph
    executor_rule: str
    action: Any
    action_record: dict[str, Any]
    action_sha256: str
    source_state_sha256: str
    successor_state_sha256: str
    canonical_successor_key: str
    model_family: str
    family_context: str
    raw_mark_count: int
    canonical_successor_count: int
    matching_mark_count: int
    exact_successor_mark_count: int
    successor_alias_count: int

    @property
    def cell(self) -> str:
        """The unnamespaced capability cell, as ``family:context``."""

        return f"{self.model_family}:{self.family_context}"

    @property
    def capability_cell_id(self) -> str:
        """The cell id a pipeline stage publishes: ``namespace:family:context``.

        The namespace is read from the live registry, never spelled here, so a
        namespace change cannot leave this fixture agreeing with a registry it
        no longer matches.  This accessor exists because the unnamespaced
        ``cell`` above and the published ``capability_cell_id`` are two spellings
        of one value: comparing the wrong pair reports 0 of 22 cells covered
        while every cell is in fact present, which is a silent join failure
        rather than a loud one.
        """

        return f"{_registry_namespace()}:{self.cell}"



@lru_cache(maxsize=1)
def _registry_namespace() -> str:
    """The capability-cell namespace the live registry declares."""

    return load_semantic_capability_cell_registry().namespace


@dataclass(frozen=True)
class _Mark:
    executor_rule: str
    action: Any
    successor: MolecularGraph
    record: dict[str, Any]
    action_sha256: str
    successor_state_sha256: str
    canonical_key: str


def seed_state(smiles: str) -> MolecularGraph:
    """The padded persistent-slot state one seed SMILES denotes."""

    return pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)


def _birth_candidates(state: MolecularGraph) -> Iterator[tuple[str, Any]]:
    """Every single-neighbour (or root) atom birth into the first free slot.

    Editing-V2 admits at most one existing neighbour, so this is the whole
    ``atom_insert`` shape; the codec refuses anything wider.
    """

    null = tuple(int(slot) for slot in np.flatnonzero(state.atom_types == NULL_IDX))
    if not null:
        return
    slot = null[0]
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    for element in _BIRTH_ELEMENTS:
        atom_type = ELEMENT_TO_IDX[element]
        for hydrogen_count in _BIRTH_HYDROGEN_COUNTS:
            if not real:
                yield "atom_insert", AtomInsert(slot, atom_type, 0, hydrogen_count, ())
                continue
            for neighbor in real:
                for order in MICRO_BOND_CLASSES:
                    yield (
                        "atom_insert",
                        AtomInsert(
                            slot, atom_type, 0, hydrogen_count, ((neighbor, order),)
                        ),
                    )


def _reorder_candidates(state: MolecularGraph) -> Iterator[tuple[str, Any]]:
    """Every productive bond-order change, over the real bonds of the state."""

    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    for left, right in combinations(real, 2):
        current = int(state.bonds[left, right])
        if current == 0:
            continue
        for order in MICRO_BOND_CLASSES:
            if order != current:
                yield "bond_reorder", BondReorder(left, right, order)


def candidate_actions(state: MolecularGraph) -> Iterator[tuple[str, Any]]:
    """Every Active8 candidate at ``state``, in a fixed family order.

    The eight families are exactly ``ACTIVE8_EXECUTOR_RULES``.  Six of them have
    a production enumerator and are taken from it verbatim; ``atom_insert`` and
    ``bond_reorder`` are per-coordinate families whose candidate space is the
    slot/order sweep above.  Nothing is filtered here -- admission is the
    runtime's answer, taken in :func:`_marks_at`.
    """

    yield from _birth_candidates(state)
    for action in enumerate_process_v2_atom_deletes(state):
        yield "atom_delete", action
    for action in enumerate_semantic_atom_restates(state):
        yield "atom_restate_semantic", action
    yield from _reorder_candidates(state)
    for action in enumerate_pendant_graft_actions(state):
        yield "bond_reroute", action
    for action in enumerate_cycle_close_edges(state):
        yield "cycle_close", action
    for action in enumerate_cycle_open_edges(state):
        yield "cycle_open", action
    for action in enumerate_ring_system_restate_actions(state):
        yield "ring_system_restate", action


def _marks_at(state: MolecularGraph) -> tuple[_Mark, ...]:
    """Execute every candidate and keep the ones the production runtime admits."""

    system = editing_v2_semantic_rewrite_system()
    marks: list[_Mark] = []
    for rule, action in candidate_actions(state):
        try:
            successor = system.apply(state, rule, action)
        except (InvalidRewrite, ValueError):
            continue
        try:
            record = encode_action(rule, action)
        except ActionCodecV4Error:
            continue
        marks.append(
            _Mark(
                executor_rule=rule,
                action=action,
                successor=successor,
                record=record,
                action_sha256=canonical_sha256(record),
                successor_state_sha256=persistent_slot_state_sha256(successor),
                canonical_key=canonical_state_key(successor),
            )
        )
    return tuple(marks)


def _transition_from(
    seed_smiles: str,
    state: MolecularGraph,
    source_sha256: str,
    marks: tuple[_Mark, ...],
    mark: _Mark,
    family: str,
    context: str,
) -> GenuineTransition:
    raw = len(marks)
    matching = tuple(item for item in marks if item.action_sha256 == mark.action_sha256)
    exact = sum(
        1
        for item in matching
        if item.successor_state_sha256 == mark.successor_state_sha256
    )
    aliases = sum(1 for item in marks if item.canonical_key == mark.canonical_key)
    successors = len({item.canonical_key for item in marks})
    if len(matching) != 1 or exact != 1:
        raise GenuineTransitionError(
            f"the teacher for {family}:{context} is not a unique exact Action-V4 mark"
        )
    if not 1 <= aliases <= raw or successors > raw:
        raise GenuineTransitionError(
            f"the candidate counts for {family}:{context} do not aggregate"
        )
    return GenuineTransition(
        seed_smiles=seed_smiles,
        source_state=state,
        successor_state=mark.successor,
        executor_rule=mark.executor_rule,
        action=mark.action,
        action_record=dict(mark.record),
        action_sha256=mark.action_sha256,
        source_state_sha256=source_sha256,
        successor_state_sha256=mark.successor_state_sha256,
        canonical_successor_key=mark.canonical_key,
        model_family=family,
        family_context=context,
        raw_mark_count=raw,
        canonical_successor_count=successors,
        matching_mark_count=len(matching),
        exact_successor_mark_count=exact,
        successor_alias_count=aliases,
    )


@lru_cache(maxsize=1)
def discover_genuine_transitions() -> Mapping[str, GenuineTransition]:
    """Search the seeds and return one genuine transition per cell reached.

    Keyed by the unnamespaced ``family:context`` the production classifier
    returned.  Deterministic: seeds in declared order, candidates in family
    order, first hit wins.
    """

    found: dict[str, GenuineTransition] = {}
    for seed_smiles in SEED_SMILES:
        state = seed_state(seed_smiles)
        source_sha256 = persistent_slot_state_sha256(state)
        marks = _marks_at(state)
        for mark in marks:
            try:
                family, context = classify_action_family_context(
                    state,
                    mark.successor,
                    RewriteStep(rule_name=mark.executor_rule, action=mark.action),
                )
            except SemanticCapabilityCellError:
                continue
            cell = f"{family}:{context}"
            if cell in found:
                continue
            found[cell] = _transition_from(
                seed_smiles, state, source_sha256, marks, mark, family, context
            )
    return dict(found)


def genuine_transition(cell_id: str) -> GenuineTransition:
    """The discovered transition for ``[namespace:]family:context``."""

    parts = cell_id.split(":")
    cell = ":".join(parts[-2:])
    discovered = discover_genuine_transitions()
    if cell not in discovered:
        raise GenuineTransitionError(
            f"discovery reached no genuine transition for the capability cell {cell!r}"
        )
    return discovered[cell]


def main() -> None:  # pragma: no cover - a developer entry point
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

    discovered = discover_genuine_transitions()
    for cell in sorted(discovered):
        item = discovered[cell]
        print(
            f"{cell:62s} {item.seed_smiles!r:20s} {item.executor_rule:22s} "
            f"-> {molecular_graph_to_smiles(item.successor_state)!r} "
            f"raw={item.raw_mark_count} successors={item.canonical_successor_count} "
            f"aliases={item.successor_alias_count}"
        )
    print(f"\n{len(discovered)} capability cells reached from {len(SEED_SMILES)} seeds")


if __name__ == "__main__":  # pragma: no cover - a developer entry point
    main()
