"""Lipid-shaped carbon-tree source prior.

The generic `DegreeBoundedCarbonTreePrior` samples uniform-ish degree-bounded
random trees, so its skeletons are not lipid-shaped and de-novo tails come out
structureless. Lipids have a specific carbon topology: a compact hub (the head +
linker-adjacent carbons) with a few LONG LINEAR tails. This prior samples exactly
that -- still a valid, connected, degree-<=4 carbon tree (so it is a legitimate
CTMC source prior and passes the `flexible_size_graft` type check), just drawn from
a lipid-structured distribution instead of a generic one.

Tail length/count is decided at the skeleton (the source), which is where the
generated tail structure actually comes from in the carbon-tree architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    NULL_IDX,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state

_C = ELEMENT_TO_IDX["C"]


def _lipid_tree_edges(
    rng: np.random.Generator,
    size: int,
    *,
    max_degree: int,
    tail_counts: tuple[int, ...],
    tail_count_weights: tuple[float, ...],
    tail_len_lo: int,
    tail_len_hi: int,
    min_hub: int,
) -> list[tuple[int, int]]:
    """A hub path with `n_tails` long linear tails grafted onto it. Exactly `size`
    carbons, connected, every degree <= max_degree."""
    if size <= 2:
        return [(i, i + 1) for i in range(size - 1)]
    weights = np.asarray(tail_count_weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_tails = int(rng.choice(np.asarray(tail_counts, dtype=np.int64), p=weights))
    n_tails = max(1, min(n_tails, max(1, size - min_hub)))
    # sample tail lengths, then shrink longest until a hub of >= min_hub fits
    lengths = [int(rng.integers(tail_len_lo, tail_len_hi + 1)) for _ in range(n_tails)]
    while sum(lengths) > size - min_hub:
        i = int(np.argmax(lengths))
        if lengths[i] <= 1:
            break
        lengths[i] -= 1
    hub_size = size - sum(lengths)
    if hub_size < 1:  # degenerate fallback: pure chain
        return [(i, i + 1) for i in range(size - 1)]
    # hub is a simple path 0..hub_size-1
    edges: list[tuple[int, int]] = [(i, i + 1) for i in range(hub_size - 1)]
    degree = [0] * size
    for i in range(hub_size - 1):
        degree[i] += 1
        degree[i + 1] += 1
    nxt = hub_size
    for length in lengths:
        candidates = [h for h in range(hub_size) if degree[h] < max_degree]
        if not candidates:  # no room -> extend from the last tail atom instead
            candidates = [nxt - 1] if nxt > hub_size else [0]
        anchor = int(candidates[int(rng.integers(0, len(candidates)))])
        prev = anchor
        for _ in range(length):
            edges.append((prev, nxt))
            degree[prev] += 1
            degree[nxt] += 1
            prev = nxt
            nxt += 1
    return edges


def _edges_to_carbon_graph(edges, size: int, n_slots: int) -> MolecularGraph:
    atom_types = np.full(n_slots, NULL_IDX, dtype=np.int32)
    atom_types[:size] = _C
    formal_charges = np.zeros(n_slots, dtype=np.int32)
    bonds = np.zeros((n_slots, n_slots), dtype=np.int32)
    degree = np.zeros(size, dtype=np.int32)
    for a, b in edges:
        bonds[a, b] = bonds[b, a] = BOND_SINGLE
        degree[a] += 1
        degree[b] += 1
    implicit_h = np.zeros(n_slots, dtype=np.int32)
    implicit_h[:size] = 4 - degree
    state = MolecularGraph(atom_types, formal_charges, implicit_h, bonds)
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise RuntimeError("lipid carbon-tree sampler violated its invariant")
    return state


@dataclass(frozen=True)
class LipidCarbonTreePrior(DegreeBoundedCarbonTreePrior):
    """Carbon-tree prior that samples lipid skeletons (hub + long linear tails).
    Subclass of DegreeBoundedCarbonTreePrior so it slots into graft transport."""

    tail_counts: tuple[int, ...] = (2, 3, 4)
    tail_count_weights: tuple[float, ...] = (0.35, 0.40, 0.25)
    tail_len_lo: int = 6
    tail_len_hi: int = 18
    min_hub: int = 3

    def sample_size(self, rng: np.random.Generator, *, n_slots: int, size: int) -> MolecularGraph:
        size = int(size)
        if size <= 0 or size > int(n_slots):
            raise ValueError("lipid tree size out of range")
        edges = _lipid_tree_edges(
            rng, size, max_degree=int(self.max_degree),
            tail_counts=self.tail_counts, tail_count_weights=self.tail_count_weights,
            tail_len_lo=int(self.tail_len_lo), tail_len_hi=int(self.tail_len_hi),
            min_hub=int(self.min_hub),
        )
        return _edges_to_carbon_graph(edges, size, n_slots)

    @classmethod
    def from_size_counts(cls, counts, *, smoothing: float = 0.0, max_degree: int = 4, **kwargs):
        base = DegreeBoundedCarbonTreePrior.from_size_counts(counts, smoothing=smoothing, max_degree=max_degree)
        return cls(sizes=base.sizes, probabilities=base.probabilities, max_degree=base.max_degree, **kwargs)


__all__ = ["LipidCarbonTreePrior"]
