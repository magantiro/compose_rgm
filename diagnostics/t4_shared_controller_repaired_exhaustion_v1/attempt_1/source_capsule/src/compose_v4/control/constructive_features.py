"""Compact features for the coupled constructive proposal law.

The unit is one decision: at committed state `G` the teacher started a constructive
dependency region, attaching at site `r` in a mode `m`. A prior has to put mass on the
PAIR, so both sides need a description that lets an unseen pair borrow strength from
seen ones -- which is the whole reason for features rather than categorical bins. With
105 observed decisions, 59 distinct exact `(mode, site)` identities and only 46% of them
recurring across cells, a lookup keyed on identity has nothing to say about a new
combination; a model over `element`, `degree`, `free valence`, `ring membership` and
`atoms created`, `sites used`, `ring closed` can score one it has never seen.

Features stay small and interpretable on purpose. Held-out power at n=105 under
leave-one-target-out folds is the binding constraint, not expressiveness, and a learned
embedding over the same 105 rows would be fit to the folds rather than to chemistry.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT

SCHEMA_VERSION = "constructive_features_v1"

# Elements carried explicitly; everything else shares an "other" coordinate, so a
# halogen site is scored rather than dropped.
NAMED_ELEMENTS = ("C", "N", "O", "S")

SITE_FEATURE_NAMES = (
    *(f"element_{e}" for e in NAMED_ELEMENTS),
    "element_other",
    "degree_1",
    "degree_2",
    "degree_3",
    "degree_4plus",
    "free_valence_0",
    "free_valence_1",
    "free_valence_2plus",
    "in_ring",
    "ring_adjacent",
    "heteroatom_neighbours",
    "unsaturated",
    "bias",
)

MODE_FEATURE_NAMES = (
    "sites_1",
    "sites_2",
    "sites_3plus",
    "created_0",
    "created_1",
    "created_2_3",
    "created_4_6",
    "created_7plus",
    "closes_ring",
    "opens_ring",
    "log_primitives",
    "bias",
)


def _adjacency(state) -> dict[int, list[tuple[int, int]]]:
    """Neighbour lists with bond order, from the receipt's edge-list form."""
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for row in np.asarray(state["bonds"]).reshape(-1, 3):
        u, v, order = int(row[0]), int(row[1]), int(row[2])
        neighbours.setdefault(u, []).append((v, order))
        neighbours.setdefault(v, []).append((u, order))
    return neighbours


def _ring_slots(state, neighbours) -> set[int]:
    """Slots on a cycle, by iteratively peeling degree-one atoms off the heavy graph."""
    degree = {slot: len(edges) for slot, edges in neighbours.items()}
    remaining = {slot for slot, d in degree.items() if d > 0}
    changed = True
    while changed:
        changed = False
        for slot in sorted(remaining):
            live = sum(1 for other, _ in neighbours[slot] if other in remaining)
            if live <= 1:
                remaining.discard(slot)
                changed = True
    return remaining


def occupied_slots(state) -> list[int]:
    """Real atoms, excluding null padding and scar slots."""
    types = np.asarray(state["atom_types"])
    return [
        i for i, t in enumerate(types) if IDX_TO_ELEMENT.get(int(t)) not in (None, "null", "SCAR")
    ]


def site_features(state) -> tuple[np.ndarray, list[int]]:
    """Feature matrix over every real atom of `state`, plus the slot each row describes."""
    types = np.asarray(state["atom_types"])
    hydrogens = np.asarray(state["implicit_h_counts"])
    neighbours = _adjacency(state)
    rings = _ring_slots(state, neighbours)
    slots = occupied_slots(state)

    rows = []
    for slot in slots:
        edges = neighbours.get(slot, [])
        element = IDX_TO_ELEMENT.get(int(types[slot]), "other")
        degree = len(edges)
        free = int(hydrogens[slot]) if slot < hydrogens.size else 0
        hetero = sum(
            1
            for other, _ in edges
            if IDX_TO_ELEMENT.get(int(types[other]), "other") not in ("C", "null")
        )
        rows.append(
            [
                *(1.0 if element == e else 0.0 for e in NAMED_ELEMENTS),
                1.0 if element not in NAMED_ELEMENTS else 0.0,
                1.0 if degree == 1 else 0.0,
                1.0 if degree == 2 else 0.0,
                1.0 if degree == 3 else 0.0,
                1.0 if degree >= 4 else 0.0,
                1.0 if free == 0 else 0.0,
                1.0 if free == 1 else 0.0,
                1.0 if free >= 2 else 0.0,
                1.0 if slot in rings else 0.0,
                1.0 if slot not in rings and any(o in rings for o, _ in edges) else 0.0,
                min(hetero, 3) / 3.0,
                1.0 if any(order > 1 for _, order in edges) else 0.0,
                1.0,
            ]
        )
    return np.asarray(rows, dtype=float), slots


def mode_vector(mode) -> np.ndarray:
    """One medium-granularity construction mode as features."""
    sites = int(mode["attachment_count"])
    created = int(mode["created_atoms"])
    primitives = max(int(mode.get("primitive_count", 1)), 1)
    return np.asarray(
        [
            1.0 if sites <= 1 else 0.0,
            1.0 if sites == 2 else 0.0,
            1.0 if sites >= 3 else 0.0,
            1.0 if created == 0 else 0.0,
            1.0 if created == 1 else 0.0,
            1.0 if 2 <= created <= 3 else 0.0,
            1.0 if 4 <= created <= 6 else 0.0,
            1.0 if created >= 7 else 0.0,
            1.0 if mode["closes_ring"] else 0.0,
            1.0 if mode.get("opens_ring") else 0.0,
            np.log(primitives) / np.log(32.0),
            1.0,
        ],
        dtype=float,
    )


def mode_identity(mode) -> tuple[int, int, bool]:
    """The medium granularity the mode vocabulary is enumerated at."""
    return (int(mode["attachment_count"]), int(mode["created_atoms"]), bool(mode["closes_ring"]))


def mode_matrix(modes) -> np.ndarray:
    return np.asarray([mode_vector(m) for m in modes], dtype=float)
