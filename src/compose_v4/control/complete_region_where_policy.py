"""Implicit full-support WHERE grammar for complete-region policies.

The generated object is a nonempty set of current molecular roles.  The mask
stream is an internal decoder for one semantic WHERE decision, never an
executor-action sequence.  Every live-role subset is representable, including
the disconnected boundary sets required by some complete structural patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.edit_program import atom_signature, environment

WHERE_MASK_SCHEMA = "complete_region_where_mask_v1"


def canonical_role_order(graph: MolecularGraph) -> tuple[int, ...]:
    """Return the deterministic current-state role order used by the decoder."""

    live = tuple(int(value) for value in np.flatnonzero(is_element(graph.atom_types)))
    return tuple(
        sorted(
            live,
            key=lambda slot: (
                atom_signature(graph, slot),
                environment(graph, slot),
                slot,
            ),
        )
    )


@dataclass(frozen=True)
class WhereMaskDecision:
    """One complete WHERE decision in canonical current-role order."""

    selected: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.selected or any(value not in (0, 1) for value in self.selected):
            raise ValueError("WHERE mask must be nonempty and binary")
        if not any(self.selected):
            raise ValueError("WHERE mask cannot select the empty role set")

    def payload(self) -> dict:
        return {"schema_version": WHERE_MASK_SCHEMA, "selected": list(self.selected)}

    @classmethod
    def from_payload(cls, payload: dict) -> WhereMaskDecision:
        if payload.get("schema_version") != WHERE_MASK_SCHEMA or set(payload) != {
            "schema_version",
            "selected",
        }:
            raise ValueError("WHERE mask schema mismatch")
        return cls(tuple(map(int, payload["selected"])))


def encode_where_mask(graph: MolecularGraph, selected_slots: tuple[int, ...]) -> WhereMaskDecision:
    """Encode a current-state role set without persisting addresses in a model."""

    order = canonical_role_order(graph)
    selected = set(map(int, selected_slots))
    if not selected or not selected <= set(order):
        raise ValueError("WHERE roles must be a nonempty subset of live current roles")
    return WhereMaskDecision(tuple(int(slot in selected) for slot in order))


def decode_where_mask(graph: MolecularGraph, decision: WhereMaskDecision) -> tuple[int, ...]:
    order = canonical_role_order(graph)
    if len(order) != len(decision.selected):
        raise ValueError("WHERE mask length differs from current live-role count")
    return tuple(slot for slot, selected in zip(order, decision.selected, strict=True) if selected)


def declared_where_support_size(graph: MolecularGraph) -> int:
    """Exact cardinality of the implicit nonempty current-role-set support."""

    return (1 << len(canonical_role_order(graph))) - 1


def source_component_count(graph: MolecularGraph, selected_slots: tuple[int, ...]) -> int:
    """Count source-graph connected components in a selected WHERE role set."""

    remaining = set(map(int, selected_slots))
    if not remaining:
        raise ValueError("component count requires a nonempty WHERE region")
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            slot = stack.pop()
            neighbors = {
                int(value) for value in np.flatnonzero(graph.bonds[slot]) if int(value) in remaining
            }
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors))
    return components


__all__ = [
    "WhereMaskDecision",
    "canonical_role_order",
    "declared_where_support_size",
    "decode_where_mask",
    "encode_where_mask",
    "source_component_count",
]
