"""Explicit source distributions for de novo molecular transport.

The source prior is deliberately separated from empirical rewrite-rate priors
and from the typed ring proposal catalog.  A source prior samples the complete
state at model time zero; it does not score later actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Protocol

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    NULL_IDX,
)
from compose_v4.chem.state import empty_molecular_graph, is_connected_or_null, is_valid_state
from compose_v4.chem.state import pad_molecular_graph


class MolecularSourcePrior(Protocol):
    """Sample a padded valid source state for an ancestral trajectory."""

    def sample(self, rng: np.random.Generator, *, n_slots: int) -> MolecularGraph:
        ...


@dataclass(frozen=True)
class NullSourcePrior:
    """Delta mass at the formal all-NULL source used by the original model."""

    def sample(self, rng: np.random.Generator, *, n_slots: int) -> MolecularGraph:
        del rng
        return empty_molecular_graph(n_slots)


@dataclass(frozen=True)
class FixedMolecularStatePrior:
    """Delta source at one complete molecule for editing experiments.

    The state is validated when the prior is constructed and copied on every
    draw so a downstream executor can never mutate the retained source.  This
    prior changes only the initial distribution; it does not expose a target
    molecule to the rate model or alter any rewrite application condition.
    """

    state: MolecularGraph

    def __post_init__(self) -> None:
        if not is_valid_state(self.state) or not is_connected_or_null(self.state):
            raise ValueError("fixed source must be a valid connected molecule")

    def sample(self, rng: np.random.Generator, *, n_slots: int) -> MolecularGraph:
        del rng
        if int(n_slots) < self.state.n_atoms:
            raise ValueError(
                f"fixed source uses {self.state.n_atoms} slots but only {n_slots} "
                "were requested"
            )
        if int(n_slots) > self.state.n_atoms:
            return pad_molecular_graph(self.state, int(n_slots))
        return MolecularGraph(
            atom_types=self.state.atom_types.copy(),
            formal_charges=self.state.formal_charges.copy(),
            implicit_h_counts=self.state.implicit_h_counts.copy(),
            bonds=self.state.bonds.copy(),
        )


@dataclass(frozen=True)
class DegreeBoundedCarbonTreePrior:
    """A tractable structured-noise prior over valid alkane trees.

    Sizes are sampled from the declared categorical distribution.  Conditional
    on size, a capped Prüfer process samples a labeled tree whose maximum degree
    cannot exceed ``max_degree``.  The process is explicit and inexpensive, but
    is not claimed to be uniform over degree-constrained unlabeled trees.
    """

    sizes: tuple[int, ...]
    probabilities: tuple[float, ...] | None = None
    max_degree: int = 4

    def __post_init__(self) -> None:
        if not self.sizes or any(int(size) <= 0 for size in self.sizes):
            raise ValueError("tree-prior sizes must be positive")
        if len(set(int(size) for size in self.sizes)) != len(self.sizes):
            raise ValueError("tree-prior sizes must be unique")
        if not 2 <= int(self.max_degree) <= 4:
            raise ValueError("carbon-tree max_degree must lie in [2, 4]")
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.sizes):
                raise ValueError("one tree-prior probability is required per size")
            weights = np.asarray(self.probabilities, dtype=np.float64)
            if not bool(np.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("tree-prior probabilities must be finite and nonnegative")
            if float(weights.sum()) <= 0.0:
                raise ValueError("tree-prior probabilities must have positive mass")

    @classmethod
    def from_size_counts(
        cls,
        counts: dict[int, int],
        *,
        smoothing: float = 0.0,
        max_degree: int = 4,
    ) -> "DegreeBoundedCarbonTreePrior":
        """Build a transparent categorical size prior from training-only counts."""

        if smoothing < 0.0:
            raise ValueError("size smoothing must be nonnegative")
        sizes = tuple(sorted(int(size) for size, count in counts.items() if count > 0))
        if not sizes:
            raise ValueError("size counts contain no positive observations")
        weights = np.asarray(
            [float(counts[size]) + float(smoothing) for size in sizes],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return cls(sizes, tuple(float(value) for value in weights), max_degree)

    def sample(self, rng: np.random.Generator, *, n_slots: int) -> MolecularGraph:
        probabilities = None
        if self.probabilities is not None:
            weights = np.asarray(self.probabilities, dtype=np.float64)
            probabilities = weights / weights.sum()
        size = int(rng.choice(np.asarray(self.sizes, dtype=np.int64), p=probabilities))
        return self.sample_size(rng, n_slots=n_slots, size=size)

    def sample_size(
        self,
        rng: np.random.Generator,
        *,
        n_slots: int,
        size: int,
    ) -> MolecularGraph:
        """Sample the conditional tree law at an explicitly chosen size.

        This is useful for size-matched training couplings.  If target sizes
        follow the same empirical categorical law used by :meth:`sample`, the
        unconditional source marginal is unchanged; only the coupling between
        source and target is lower variance.
        """

        size = int(size)
        if size <= 0:
            raise ValueError("tree size must be positive")
        if size > int(n_slots):
            raise ValueError(
                f"sampled tree size {size} exceeds the {n_slots} available slots"
            )
        edges = _sample_capped_prufer_tree(
            rng,
            size=size,
            max_degree=int(self.max_degree),
        )
        carbon = int(ELEMENT_TO_IDX["C"])
        atom_types = np.full(n_slots, NULL_IDX, dtype=np.int32)
        atom_types[:size] = carbon
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
            raise RuntimeError("carbon-tree source sampler violated its invariant")
        return state


def _sample_capped_prufer_tree(
    rng: np.random.Generator,
    *,
    size: int,
    max_degree: int,
) -> tuple[tuple[int, int], ...]:
    if size == 1:
        return ()
    counts = np.zeros(size, dtype=np.int32)
    sequence: list[int] = []
    for _ in range(size - 2):
        available = np.flatnonzero(counts < max_degree - 1)
        selected = int(available[int(rng.integers(0, len(available)))])
        sequence.append(selected)
        counts[selected] += 1

    degree = np.ones(size, dtype=np.int32)
    for vertex in sequence:
        degree[vertex] += 1
    leaves = [int(vertex) for vertex in np.flatnonzero(degree == 1)]
    heapq.heapify(leaves)
    edges: list[tuple[int, int]] = []
    for vertex in sequence:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, int(vertex)))
        degree[leaf] -= 1
        degree[vertex] -= 1
        if degree[vertex] == 1:
            heapq.heappush(leaves, int(vertex))
    left = heapq.heappop(leaves)
    right = heapq.heappop(leaves)
    edges.append((left, right))
    return tuple(edges)


__all__ = [
    "DegreeBoundedCarbonTreePrior",
    "FixedMolecularStatePrior",
    "MolecularSourcePrior",
    "NullSourcePrior",
]
