"""A probability law over the REPLACEMENT half of a segment-replace module.

WHY THIS EXISTS
---------------
``dynamic_program_synthesis.compile_generic_module`` builds a ``segment_replace``
module in two halves: it excises a bridge-separated region, then constructs a
replacement at the retained interface.  The excision half is conditioned --
:mod:`compose_v4.control.bridge_region_law` tilts it by the child's free gate
margin -- and the completion half is not.  It draws

    growth = rng.integers(1, capacity + 1)

uniformly over ``1..min(8, 40 - n)`` and then a chain of that many atoms, each
drawn uniformly from ``(C, N, O)`` and attached by a single bond.

MEASURED, on the T4 cell that motivated this (60 production region draws x 64
completions, zero oracle calls): every eligible completion inserted ONE or TWO
atoms, while the blind length draw puts only about a quarter of its mass there.
The pooled per-completion eligible fraction was 0.00859, so the replacement the
cell needs is expressible and merely improbable.  That is a proposal-weighting
defect, not a support defect, and this law is the smallest change that fixes it.

WHAT THIS LAW IS, AND IS NOT
----------------------------
It is a RE-RANKING over completions the caller has already drawn from the
EXISTING blind distribution.  Every candidate keeps a strictly positive weight
(``floor``), so the reachable set is exactly the reachable set of the blind draw
-- this law can make a completion likelier or unlikelier and can never make one
impossible.  It is NOT a filter, and it does not enumerate a new completion
space.  A filter here would delete the completions some other cell depends on,
which is precisely the failure that rejected an earlier region-size change.

It scores an ENDPOINT, not a plan: the caller executes each candidate and hands
over the resulting state, so the law never has to reimplement growth or the
executor, and cannot drift from them.

Nothing here is task-specific beyond the free gate the benchmark itself
declares.  The margin callable is supplied by the caller and, for T4, is the
same ``FreeFeasibilityGate`` margin the region law already consumes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.bridge_region_law import (
    MARGIN_TEMPERATURE,
    SUPPORT_FLOOR,
)

SCHEMA_VERSION = "t4_replace_completion_law_v1"

#: Completions drawn per firing before re-ranking.  Chosen from the measured
#: per-completion eligible fraction rather than by taste: at p = 0.00859 a
#: K-candidate draw contains an eligible completion with probability about
#: ``1 - (1 - p) ** K``, which is ~13% at K = 16.  Larger K costs one executor
#: replay each and buys progressively less, so the realized rate is reported by
#: the gate harness rather than assumed here.
DEFAULT_CANDIDATES = 16


@dataclass(frozen=True)
class ReplaceCompletionLaw:
    """A law over already-drawn replacement completions.

    ``candidates``   completions the caller draws per firing before re-ranking
    ``margin``       endpoint state -> feasibility margin, or ``None`` for uniform
    ``floor``        minimum relative weight, preserving support
    ``temperature``  Boltzmann scale in normalized gate units

    ``margin is None`` makes this a uniform law over the drawn candidates. It is
    retained so a counterfactual can name it, and it is deliberately NOT the OFF
    state -- see :mod:`compose_v4.control.completion_law_contract`, where OFF is
    an absent contract field and is byte-identical to the historical behaviour.
    """

    candidates: int = DEFAULT_CANDIDATES
    margin: Callable[[MolecularGraph], float] | None = None
    floor: float = SUPPORT_FLOOR
    temperature: float = MARGIN_TEMPERATURE

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("completion law needs at least one candidate")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("support floor must lie in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("margin temperature must be positive")

    @property
    def conditioned(self) -> bool:
        return self.margin is not None

    def weights(self, endpoints: Sequence[MolecularGraph]) -> np.ndarray:
        """Relative weight of each drawn completion; strictly positive everywhere.

        A candidate whose margin cannot be computed is held AT THE FLOOR rather
        than dropped, for the same reason the region law holds an unrealizable
        region at the floor: the law must never narrow what the blind draw could
        reach.
        """

        if self.margin is None:
            return np.ones(len(endpoints), dtype=float)
        out = np.empty(len(endpoints), dtype=float)
        for position, endpoint in enumerate(endpoints):
            try:
                value = float(self.margin(endpoint))
            except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
                out[position] = self.floor
                continue
            if not math.isfinite(value):
                out[position] = self.floor
                continue
            out[position] = max(self.floor, math.exp(value / self.temperature))
        return out

    def order(self, endpoints: Sequence[MolecularGraph], rng) -> list[int]:
        """Candidate INDICES in weighted draw order, sampled without replacement.

        Efraimidis-Spirakis, exactly as :meth:`BridgeRegionLaw.order`: sorting by
        ``-log(u_i) / w_i`` ascending is successive weighted selection without
        replacement, so the head is one draw from the law and the tail is the law
        conditioned on the head having been rejected. Uniform weights reduce to a
        plain permutation.
        """

        if not len(endpoints):
            return []
        weights = self.weights(endpoints)
        keys = -np.log(np.clip(rng.random(len(endpoints)), 1e-300, 1.0)) / weights
        return [int(at) for at in np.argsort(keys, kind="stable")]


def free_gate_margin_completion_law(
    margin: Callable[[MolecularGraph], float],
    *,
    candidates: int = DEFAULT_CANDIDATES,
    floor: float = SUPPORT_FLOOR,
    temperature: float = MARGIN_TEMPERATURE,
) -> ReplaceCompletionLaw:
    """The repair: draw ``candidates`` blind completions, tilt by the endpoint margin.

    ``margin`` maps an executed endpoint state to its signed free-gate slack. For
    T4 that is ``FreeFeasibilityGate.margin`` against the ORIGINAL benchmark
    source, never the current parent, so the law cannot drift its own target as a
    campaign walks away from the root.
    """

    return ReplaceCompletionLaw(
        candidates=candidates,
        margin=margin,
        floor=floor,
        temperature=temperature,
    )
