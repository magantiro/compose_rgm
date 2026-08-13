"""Claim 2: decompose the cycling signal. Definitions frozen BEFORE any reading.

The 8-source smoke measured, on held-in sources, an ``R_theta`` revisit rate of
0.281 and an immediate-reversal rate of 0.300 against roughly 0.01 for the
empirical-family law, together with LOWER structural displacement.  That is too
large to leave unexplained and too small to act on.  This module decomposes it.

Every definition here is fixed before it is evaluated on any trajectory.  That
ordering is the whole point: "what counts as a cancelled edit" is exactly the
kind of choice that, made after seeing the numbers, can turn any result into
any other result.

The two readings this is built to separate
------------------------------------------
The same measurement supports opposite conclusions, and they have opposite
consequences:

* **Inherited reversibility.** If the reference corpus itself frequently
  contains both ``x -> y`` and ``y -> x``, a locally reversible ``R_theta`` is
  faithfully modelling a locally reversible process.  The cycling is then a
  property of the reference law, not a defect of the network, and the honest
  framing is "learned local reference dynamics" rather than "goal-directed
  transport".
* **Compositional pathology.** If the corpus is directional and ``R_theta`` is
  not, then one-step likelihood is not composing into usable dynamics -- which
  is precisely the failure this lane exists to detect.

``corpus_reversibility`` answers that question; it needs the corpus and
therefore the Modal runtime.  Everything else here is pure and runs locally on
saved traces.

What cannot be computed from the smoke's saved traces
-----------------------------------------------------
``reverse_edge_probability`` -- ``R_theta(x | y)`` after sampling ``x -> y`` --
requires the successor row at ``y``, which the smoke shards do not carry.  It
is recorded by the rollout app from the 36-source run onward.  Saying so
explicitly is better than substituting a proxy and calling it the same thing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Which operator families can undo which. Frozen from the process-V2
#: capability-cell registry, whose family contexts make the pairing explicit:
#: ``cycle_insert`` closes to a ring system, ``cycle_attach`` opens from one.
#:
#: A family paired with ITSELF is self-inverting: a bond-order increase is
#: undone by a bond-order decrease, an aromatization by a dearomatization, an
#: element retype by the reverse retype.  Self-inversion makes the family-level
#: test necessary but NOT sufficient, which is why exact state return is
#: measured separately and is the quantity that carries the argument.
INVERSE_FAMILIES: dict[str, frozenset[str]] = {
    "atom_insert": frozenset({"atom_delete"}),
    "atom_delete": frozenset({"atom_insert"}),
    "cycle_insert": frozenset({"cycle_attach"}),
    "cycle_attach": frozenset({"cycle_insert"}),
    "bond_reorder": frozenset({"bond_reorder"}),
    "atom_restate": frozenset({"atom_restate"}),
    "ring_system_restate": frozenset({"ring_system_restate"}),
    "bond_reroute": frozenset({"bond_reroute"}),
}

SELF_INVERTING_FAMILIES: frozenset[str] = frozenset(
    name for name, inverses in INVERSE_FAMILIES.items() if name in inverses
)


class CycleAttributionError(ValueError):
    """A cycle diagnostic was asked for something it cannot honestly compute."""


@dataclass(frozen=True)
class CycleAttribution:
    """Cycle decomposition of one realized trajectory.

    ``committed_edits`` counts every accepted transition.  ``net_edits`` counts
    what survives cancelling immediate out-and-back moves.  The gap between
    them is the quantity the headline mobility number must be discounted by:
    a trajectory that makes six edits and cancels four has not moved six units.
    """

    committed_edits: int
    net_edits: int
    cancelled_edits: int
    cancelled_fraction: float
    immediate_two_cycles: int
    immediate_two_cycle_rate: float
    longer_revisits: int
    longer_revisit_rate: float
    distinct_states: int
    inverse_family_pairs: int
    inverse_family_pair_rate: float
    inverse_family_pairs_that_cancelled: int
    reversal_by_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def reduce_out_and_back(states: Sequence[str]) -> tuple[str, ...]:
    """Cancel immediate out-and-back moves, repeatedly, like free-group reduction.

    ``a b a`` reduces to ``a``; the reduction is applied to a stack so nested
    cancellations collapse too -- ``a b c b a`` reduces to ``a`` because once
    ``c b c`` cancels the outer pair becomes adjacent.

    Only IMMEDIATE returns are cancelled.  A trajectory that wanders away and
    comes back five steps later did genuinely traverse those states, and
    erasing that would overstate the cancellation.  That path is counted as a
    longer revisit instead.
    """
    if not states:
        raise CycleAttributionError("cannot reduce an empty trajectory")
    stack: list[str] = [states[0]]
    for state in states[1:]:
        if len(stack) >= 2 and stack[-2] == state:
            stack.pop()
        else:
            stack.append(state)
    return tuple(stack)


def attribute_cycles(
    states: Sequence[str],
    families: Sequence[Sequence[str]],
) -> CycleAttribution:
    """Decompose one trajectory's cycling. Definitions frozen before any reading.

    * **immediate two-cycle** -- ``x_t -> x_{t+1} -> x_t``: the step was undone
      by the next one.
    * **longer revisit** -- a committed state equal to some earlier state that
      is NOT the immediately preceding out-and-back return. Counted separately
      because wandering back is a different behaviour from oscillating.
    * **inverse-family pair** -- consecutive steps whose committed families
      could undo one another under ``INVERSE_FAMILIES``. Necessary but not
      sufficient for cancellation, so the subset that actually returned to the
      prior state is reported alongside.
    """
    if not states:
        raise CycleAttributionError("a trajectory must contain at least its source")
    committed = len(states) - 1
    if committed and len(families) != committed:
        raise CycleAttributionError(
            f"{len(families)} family labels for {committed} committed edits"
        )

    two_cycles = [t for t in range(committed - 1) if states[t + 2] == states[t]]

    seen: set[str] = {states[0]}
    longer = 0
    for index in range(1, len(states)):
        state = states[index]
        if state in seen and not (index >= 2 and states[index - 2] == state):
            longer += 1
        seen.add(state)

    inverse_pairs, inverse_cancelled = 0, 0
    for t in range(committed - 1):
        left = set(families[t])
        right = set(families[t + 1])
        if any(right & INVERSE_FAMILIES.get(name, frozenset()) for name in left):
            inverse_pairs += 1
            if states[t + 2] == states[t]:
                inverse_cancelled += 1

    by_family: dict[str, dict[str, float]] = {}
    committed_counts: Counter[str] = Counter()
    undone_counts: Counter[str] = Counter()
    for t in range(committed):
        undone = t + 2 <= committed and states[t + 2] == states[t]
        for name in set(families[t]):
            committed_counts[name] += 1
            if undone:
                undone_counts[name] += 1
    for name, total in committed_counts.items():
        by_family[name] = {
            "committed": float(total),
            "undone_next_step": float(undone_counts[name]),
            "rate": undone_counts[name] / total if total else 0.0,
        }

    net = len(reduce_out_and_back(states)) - 1
    cancelled = committed - net
    return CycleAttribution(
        committed_edits=committed,
        net_edits=net,
        cancelled_edits=cancelled,
        cancelled_fraction=cancelled / committed if committed else 0.0,
        immediate_two_cycles=len(two_cycles),
        immediate_two_cycle_rate=len(two_cycles) / max(committed - 1, 1) if committed > 1 else 0.0,
        longer_revisits=longer,
        longer_revisit_rate=longer / committed if committed else 0.0,
        distinct_states=len(set(states)),
        inverse_family_pairs=inverse_pairs,
        inverse_family_pair_rate=(
            inverse_pairs / max(committed - 1, 1) if committed > 1 else 0.0
        ),
        inverse_family_pairs_that_cancelled=inverse_cancelled,
        reversal_by_family=by_family,
    )


def net_displacement_efficiency(
    endpoint_distance: float | None,
    attribution: CycleAttribution,
) -> dict[str, float | None]:
    """Structural motion per edit, before and after cancelling out-and-back moves.

    A model that makes six edits and cancels four must not get credit for six
    units of motion. ``per_net_edit`` is the honest denominator; ``per_committed_edit``
    is what an analysis that ignored cancellation would have reported.
    """
    if endpoint_distance is None:
        return {"per_committed_edit": None, "per_net_edit": None, "inflation": None}
    per_committed = (
        endpoint_distance / attribution.committed_edits
        if attribution.committed_edits
        else None
    )
    per_net = endpoint_distance / attribution.net_edits if attribution.net_edits else None
    return {
        "per_committed_edit": per_committed,
        "per_net_edit": per_net,
        # How much an analysis that ignored cancellation would have understated
        # the cost of each unit of motion.
        "inflation": (
            per_net / per_committed if per_committed and per_net else None
        ),
    }


# ---- the central causal question ------------------------------------------


@dataclass(frozen=True)
class ReversibilityCensus:
    """How reversible a set of directed transitions is.

    Applied to the training corpus this answers whether ``R_theta`` inherited
    its reversibility or invented it. Applied to controlled trajectories it
    answers whether goal control suppresses the backtracking that the
    uncontrolled reference law exhibits.
    """

    directed_edges: int
    distinct_directed_edges: int
    mutual_pairs: int
    mutual_edge_fraction: float
    source_states: int
    label: str = ""

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def reversibility_census(
    edges: Sequence[tuple[str, str]],
    *,
    label: str = "",
) -> ReversibilityCensus:
    """Fraction of directed transitions whose reverse also occurs in the set.

    ``mutual_edge_fraction`` is over DISTINCT directed edges, not over
    occurrences, so a single heavily repeated transition cannot dominate the
    statistic.

    Interpretation is preregistered: a HIGH fraction in the training corpus
    means a locally reversible reference process and makes ``R_theta``'s cycling
    inherited rather than pathological. A LOW fraction means the corpus is
    directional and ``R_theta`` is not, which is a compositional pathology.
    """
    distinct = {(a, b) for a, b in edges if a != b}
    mutual = sum(1 for a, b in distinct if (b, a) in distinct)
    return ReversibilityCensus(
        directed_edges=len(edges),
        distinct_directed_edges=len(distinct),
        mutual_pairs=mutual // 2,
        mutual_edge_fraction=mutual / len(distinct) if distinct else 0.0,
        source_states=len({a for a, _b in distinct}),
        label=label,
    )


def edges_from_trajectory(states: Sequence[str]) -> list[tuple[str, str]]:
    return [(states[i], states[i + 1]) for i in range(len(states) - 1)]


# ---- preregistered interpretation -----------------------------------------

INHERITED = "inherited_reversibility"
MODEL_SPECIFIC = "model_specific_cycling"
INCONCLUSIVE = "inconclusive"

#: How much more reversible the corpus must be than a directional process for
#: "inherited" to be the reading. Fixed before measurement. A corpus in which
#: more than a third of distinct transitions are mutual is not a directional
#: process, and a model reproducing that is not inventing the behaviour.
INHERITED_CORPUS_THRESHOLD = 0.33
#: Below this the corpus is directional enough that model reversibility is not
#: explained by it.
DIRECTIONAL_CORPUS_THRESHOLD = 0.10


def attribution_verdict(
    corpus_mutual_fraction: float | None,
    model_cancelled_fraction: float,
) -> str:
    """Which reading the corpus census licenses. Thresholds frozen above.

    Returns ``inconclusive`` rather than guessing when the corpus census is
    unavailable or lands between the thresholds. The whole value of this
    diagnostic is that it can come out either way.
    """
    if corpus_mutual_fraction is None:
        return INCONCLUSIVE
    if corpus_mutual_fraction >= INHERITED_CORPUS_THRESHOLD:
        return INHERITED
    if corpus_mutual_fraction <= DIRECTIONAL_CORPUS_THRESHOLD and model_cancelled_fraction > 0.0:
        return MODEL_SPECIFIC
    return INCONCLUSIVE


def summarize(attributions: Sequence[CycleAttribution]) -> dict[str, float]:
    """Mean of each scalar across trajectories. Callers average within source first."""
    if not attributions:
        return {}
    keys = (
        "committed_edits",
        "net_edits",
        "cancelled_edits",
        "cancelled_fraction",
        "immediate_two_cycles",
        "immediate_two_cycle_rate",
        "longer_revisits",
        "longer_revisit_rate",
        "inverse_family_pairs",
        "inverse_family_pair_rate",
        "inverse_family_pairs_that_cancelled",
    )
    return {
        key: sum(float(getattr(a, key)) for a in attributions) / len(attributions)
        for key in keys
    }


__all__ = [
    "CycleAttribution",
    "CycleAttributionError",
    "DIRECTIONAL_CORPUS_THRESHOLD",
    "INCONCLUSIVE",
    "INHERITED",
    "INHERITED_CORPUS_THRESHOLD",
    "INVERSE_FAMILIES",
    "MODEL_SPECIFIC",
    "ReversibilityCensus",
    "SELF_INVERTING_FAMILIES",
    "attribute_cycles",
    "attribution_verdict",
    "edges_from_trajectory",
    "net_displacement_efficiency",
    "reduce_out_and_back",
    "reversibility_census",
    "summarize",
]
