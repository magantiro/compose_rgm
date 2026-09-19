"""Parameterized ring proposals, compiled to existing primitive descriptors.

No molecule templates or endpoint construction: these descriptors specify a
cycle's size, attachment, C/N/O quota and electronic state. The shared option
kernel samples and executes every mark. Legacy programs remain unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise, product

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_AROMATIC, NULL_IDX, MolecularGraph, is_element
from compose_v4.control.macro_engine import (
    ELEMENT_CODE,
    match_aromatisation_descriptors,
    match_closure_descriptors,
    match_growth_descriptors,
)

PREFIX = "construct:"
ELEMENTS = tuple(ELEMENT_CODE[element] for element in ("C", "N", "O"))


@dataclass(frozen=True)
class RingSpec:
    topology: str
    size: int
    counts: tuple[int, int, int]
    electronic: str
    refine: int = 0

    def __post_init__(self):
        if self.topology not in ("pendant", "fused") or self.size not in (5, 6):
            raise ValueError("ring program requires pendant/fused topology and size 5 or 6")
        if (
            not isinstance(self.counts, tuple)
            or len(self.counts) != 3
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in self.counts)
            or sum(self.counts) != self.size
        ):
            raise ValueError("C/N/O counts must be nonnegative integers summing to ring size")
        if self.electronic not in ("aromatic", "nonaromatic", "saturated"):
            raise ValueError("unknown ring electronic state")
        if (
            isinstance(self.refine, bool)
            or not isinstance(self.refine, int)
            or not 0 <= self.refine <= 2
        ):
            raise ValueError("ring-scoped refinement length must be 0, 1 or 2")

    @property
    def option(self) -> str:
        return f"{PREFIX}{self.topology}:{self.size}:{','.join(map(str, self.counts))}:{self.electronic}:{self.refine}"

    @property
    def growth(self) -> int:
        return self.size - (2 if self.topology == "fused" else 0)

    @property
    def horizon(self) -> int:
        return self.growth + 1 + self.refine

    def phase(self, step: int) -> str | None:
        if 0 <= step < self.growth:
            return "scaffold_extend"
        if step == self.growth:
            return "cyclize"
        return "restate" if self.growth < step < self.horizon else None


@lru_cache(maxsize=128)
def ring_spec(option: str) -> RingSpec | None:
    if not option.startswith(PREFIX):
        return None
    try:
        _, topology, size, counts, electronic, refine = option.split(":")
        spec = RingSpec(
            topology, int(size), tuple(map(int, counts.split(","))), electronic, int(refine)
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid ring option {option!r}: {error}") from error
    if spec.option != option:
        raise ValueError(f"ring option is not canonically serialized: {option!r}")
    return spec


def default_ring_options() -> tuple[str, ...]:
    """Small data-independent parameter menu; no fragments or winner frequencies.

    The API also accepts other explicit C/N/O quotas and refinement lengths.
    Zero-refinement programs can emit a construction immediately; ordinary
    region options remain available for subsequent optimization/refinement.
    """
    out = []
    for topology in ("pendant", "fused"):
        for size in (5, 6):
            for n, o in ((0, 0), (1, 0), (0, 1)):
                counts = (size - n - o, n, o)
                out.append(RingSpec(topology, size, counts, "nonaromatic").option)
                if (size == 5 and n + o == 1) or (size == 6 and o == 0):
                    out.append(RingSpec(topology, size, counts, "aromatic").option)
    return tuple(out)


def real_slots(graph):
    return frozenset(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))


def _ring_edges(graph, locus):
    # Work in persistent slots; the historical macro helper assumes dense slots.
    real = real_slots(graph)
    adjacency = {i: {j for j in real if graph.bonds[i, j]} for i in real}
    eligible = sorted(i for i in real & set(locus) if graph.implicit_h_counts[i] >= 1)
    for u in eligible:
        for v in eligible:
            if u >= v or v not in adjacency[u]:
                continue
            seen, stack = {u}, [u]
            while stack:
                at = stack.pop()
                for nb in adjacency[at] - seen:
                    if {at, nb} != {u, v}:
                        seen.add(nb)
                        stack.append(nb)
            if v in seen:
                yield (u, v)
                yield (v, u)


def _patterns(spec, shared_order):
    if spec.electronic != "aromatic":
        if spec.electronic == "saturated" and shared_order not in (None, 1):
            return ()
        order = shared_order if shared_order in (1, 2) else 1
        return ((1,) * (spec.size - 1) + (order,),)
    return tuple(
        pattern
        for pattern in product((1, 2), repeat=spec.size)
        if pattern.count(2) == spec.size // 2
        and all(pattern[i - 1] + pattern[i] <= 3 for i in range(spec.size))
        and (shared_order in (None, BOND_AROMATIC) or pattern[-1] == shared_order)
    )


def _allowed(spec, pattern, position):
    valence = pattern[position - 1] + pattern[position]
    external = int(spec.topology == "pendant" and position == 0)
    donor = spec.electronic == "aromatic" and spec.size == 5 and valence == 2
    return tuple(
        element
        for element, max_valence in zip(ELEMENTS, (4, 3, 2))
        if valence + external <= max_valence and (not donor or element != ELEMENTS[0])
    )


@lru_cache(maxsize=2048)
def _quota_possible(allowed, counts):
    if not allowed:
        return not any(counts)
    if any(n < 0 for n in counts) or sum(counts) != len(allowed):
        return False
    return any(
        counts[i] > 0
        and element in allowed[0]
        and _quota_possible(allowed[1:], counts[:i] + (counts[i] - 1,) + counts[i + 1 :])
        for i, element in enumerate(ELEMENTS)
    )


def residual(spec, graph, anchors, path):
    inherited = anchors if spec.topology == "fused" else ()
    used = Counter(int(graph.atom_types[i]) for i in inherited + path)
    return tuple(n - used[element] for element, n in zip(ELEMENTS, spec.counts))


def _positions(spec, path_length):
    start = path_length + int(spec.topology == "fused")
    end = spec.size - int(spec.topology == "fused")
    return range(start, end)


def construction_branches(graph, locus, spec):
    if (
        graph.n_real_atoms + spec.growth > 40
        or np.count_nonzero(graph.atom_types == NULL_IDX) < spec.growth
    ):
        return ()
    anchors = (
        tuple(_ring_edges(graph, locus))
        if spec.topology == "fused"
        else tuple(
            (i,) for i in sorted(real_slots(graph) & set(locus)) if graph.implicit_h_counts[i] >= 1
        )
    )
    classes = resonance_invariant_bond_classes(graph)
    branches = []
    for anchor in anchors:
        shared = int(classes[anchor]) if spec.topology == "fused" else None
        if spec.electronic == "nonaromatic" and shared == BOND_AROMATIC:
            shared = int(graph.bonds[anchor])
        for pattern in _patterns(spec, shared):
            if spec.topology == "fused" and any(
                int(graph.atom_types[slot]) not in _allowed(spec, pattern, position)
                for slot, position in ((anchor[0], 0), (anchor[1], spec.size - 1))
            ):
                continue
            allowed = tuple(_allowed(spec, pattern, p) for p in _positions(spec, 0))
            if _quota_possible(allowed, residual(spec, graph, anchor, ())):
                branches.append((anchor, pattern))
    return tuple(branches)


@dataclass(frozen=True)
class RingProgress:
    anchors: tuple[int, ...] = ()
    pattern: tuple[int, ...] = ()
    path: tuple[int, ...] = ()

    def __post_init__(self):
        if any(not isinstance(items, tuple) for items in (self.anchors, self.pattern, self.path)):
            raise ValueError("ring progress must contain immutable tuples")
        slots = self.anchors + self.path
        if len(slots) != len(set(slots)) or any(type(i) is not int or i < 0 for i in slots):
            raise ValueError("ring anchors/path require distinct nonnegative persistent slots")

    def cycle(self, spec):
        return (
            (self.anchors[:1] + self.path + self.anchors[1:])
            if spec.topology == "fused"
            else self.path
        )

    def payload(self):
        return {
            "schema_version": "ring_progress_v1",
            "anchors": list(self.anchors),
            "pattern": list(self.pattern),
            "path": list(self.path),
        }

    @classmethod
    def from_payload(cls, payload):
        if (
            set(payload) != {"schema_version", "anchors", "pattern", "path"}
            or payload["schema_version"] != "ring_progress_v1"
        ):
            raise ValueError("unknown ring progress schema")
        return cls(*(tuple(payload[key]) for key in ("anchors", "pattern", "path")))

    def validate(self, graph, origin, locus, step, spec):
        if not 0 <= step <= spec.horizon or len(self.path) != min(step, spec.growth):
            raise ValueError("ring program phase/path mismatch")
        if step == 0:
            if self.anchors or self.pattern or self.path:
                raise ValueError("initial ring program must not preselect a branch")
            return
        if (self.anchors, self.pattern) not in construction_branches(origin, locus, spec):
            raise ValueError("ring branch is not eligible in its original region")
        old, real = real_slots(origin), real_slots(graph)
        if not old <= real or real - old != set(self.path) or not set(self.path) <= set(locus):
            raise ValueError("ring path does not identify exactly the new mutable atoms")
        old_slots = sorted(old)
        if not np.array_equal(origin.atom_types[old_slots], graph.atom_types[old_slots]):
            raise ValueError("ring construction changed original atom identities")
        if not np.array_equal(origin.formal_charges[old_slots], graph.formal_charges[old_slots]):
            raise ValueError("ring construction changed original charges")
        if not np.array_equal(
            origin.bonds[np.ix_(old_slots, old_slots)] != 0,
            graph.bonds[np.ix_(old_slots, old_slots)] != 0,
        ):
            raise ValueError("ring construction changed original induced connectivity")
        chain = self.anchors[:1] + self.path
        expected = {tuple(sorted(pair)) for pair in pairwise(chain)}
        if step > spec.growth:
            endpoint = self.anchors[1] if spec.topology == "fused" else self.path[0]
            expected.add(tuple(sorted((self.path[-1], endpoint))))
        actual = {tuple(sorted((i, j))) for i in self.path for j in real if graph.bonds[i, j]}
        if actual != expected:
            raise ValueError("ring program has a missing bond or an unintended attachment")
        if step <= spec.growth + 1:
            allowed = tuple(
                _allowed(spec, self.pattern, p) for p in _positions(spec, len(self.path))
            )
            if not _quota_possible(allowed, residual(spec, graph, self.anchors, self.path)):
                raise ValueError("ring composition quota is no longer satisfiable")


def construction_indices(families, actions, probabilities, graph, spec, progress, branch, step):
    anchors, pattern = branch
    if step < spec.growth:
        remaining = residual(spec, graph, anchors, progress.path)
        positions = tuple(_positions(spec, len(progress.path)))
        future = tuple(_allowed(spec, pattern, p) for p in positions[1:])
        allowed = {
            element
            for i, element in enumerate(ELEMENTS)
            if element in _allowed(spec, pattern, positions[0])
            and remaining[i] > 0
            and _quota_possible(future, remaining[:i] + (remaining[i] - 1,) + remaining[i + 1 :])
        }
        order = 1 if spec.topology == "pendant" and step == 0 else pattern[positions[0] - 1]
        return match_growth_descriptors(
            families,
            actions,
            probabilities,
            progress.path[-1] if progress.path else None,
            allowed,
            anchors=anchors[:1],
            bond_order=order,
        )
    if step == spec.growth:
        endpoint = anchors[1] if spec.topology == "fused" else progress.path[0]
        order = pattern[-2] if spec.topology == "fused" else pattern[-1]
        return [
            i
            for i in match_closure_descriptors(families, actions, [(progress.path[-1], endpoint)])
            if int(getattr(actions[i], "order", 1)) == order
        ]
    # Refine only newly built ring atoms/bonds. Shared original atoms remain
    # unchanged; strict product validation rejects attempted changes to them.
    return match_aromatisation_descriptors(families, actions, probabilities, progress.cycle(spec))


def completed_construction(
    origin: MolecularGraph,
    graph: MolecularGraph,
    progress: RingProgress,
    spec: RingSpec,
    *,
    refined=False,
):
    try:
        progress.validate(
            graph, origin, real_slots(graph), spec.horizon if refined else spec.growth + 1, spec
        )
    except ValueError:
        return False
    edges_before = np.count_nonzero(np.triu(origin.bonds != 0, 1))
    edges_after = np.count_nonzero(np.triu(graph.bonds != 0, 1))
    if edges_after - edges_before - (graph.n_real_atoms - origin.n_real_atoms) != 1:
        return False
    if refined and spec.refine:
        return True  # Electronic/composition changes are the declared refinement, not construction failure.
    cycle = progress.cycle(spec)
    classes = resonance_invariant_bond_classes(graph)
    orders = [int(classes[a, b]) for a, b in zip(cycle, cycle[1:] + cycle[:1])]
    if spec.electronic == "aromatic":
        return all(order == BOND_AROMATIC for order in orders)
    if spec.electronic == "saturated":
        return all(order == 1 for order in orders)
    return not all(order == BOND_AROMATIC for order in orders)
