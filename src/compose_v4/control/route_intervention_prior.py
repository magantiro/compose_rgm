"""The route-derived reference law over anchor-intervention decisions.

One stochastic proposal is not "a program". It is the latent triple

    Z = (A, C, z'_C)      ->      O' = I(A, C, z'_C)

an ANCHOR program, a grammar-induced minimal CLOSURE to intervene on, and the VALUES that
closure takes. The anchor is explicit rather than hidden inside a conditional kernel,
because the accessibility audit showed the only tractable broad-support reference is
local around some existing program -- and a law that is local around something must say
what that something is.

    q_ref(Z | G) = q_anchor(A | G) * q_block(C | G, A) * q_value(z'_C | G, A, C)

Keeping `Z` in the path rather than marginalising it buys two things. The likelihood
ratio stays a product over explicit coordinates, so hierarchical credit is a property of
the factorisation instead of an extra assumption. And one executable patch reachable
through several anchors does not force the likelihood into a sum over latent histories.

This module learns the reference law from successful routes ONLY. It never sees a docking
score. That separation is deliberate and load-bearing:

    routes give the prior dynamics -- what coherent optimisation moves look like;
    oracle outcomes give the control field -- which of them this target rewards.

Collapsing the two is how the earlier `q_theta` came to rank elite JAK2 constructions at
median 156 and poor ones at 11: it was fitted on what the search generated, which is a
statement about the searcher, not about chemistry or about the target.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

SCHEMA_VERSION = "route_intervention_prior_v1"

# Coarse anchor description. Deliberately not the exact patch: the objective is mass
# around productive neighbourhoods, not reproduction of a particular teacher answer.
SCALE_BANDS = (
    (0, 0, "none"),
    (1, 1, "single"),
    (2, 3, "small"),
    (4, 6, "medium"),
    (7, 99, "large"),
)


def scale_band(created: int) -> str:
    for low, high, name in SCALE_BANDS:
        if low <= created <= high:
            return name
    return "large"


def heteroatom_pattern(subgoal) -> str:
    """Which non-carbon elements the patch creates, as a sorted signature."""
    created = [int(atom[0]) for atom in subgoal.output_atoms]
    hetero = sorted({e for e in created if e not in (2,)})
    if not created:
        return "none"
    if not hetero:
        return "carbon_only"
    return "+".join(str(e) for e in hetero)


def topology_class(subgoal) -> str:
    """Whether the patch closes a cycle among its own roles, and how densely."""
    n_input = len(subgoal.input_atoms)
    size = len(subgoal.target_bonds)
    edges = sum(1 for i in range(size) for j in range(i + 1, size) if subgoal.target_bonds[i][j])
    live = sum(1 for a in subgoal.target_atoms if a is not None) + len(subgoal.output_atoms)
    if live <= 1:
        return "point"
    # a connected acyclic patch has live-1 edges; anything more closes a cycle
    if edges >= live:
        return "cyclic"
    if edges == live - 1:
        return "tree"
    return "fragmented" if n_input else "tree"


def anchor_class(subgoal) -> tuple[str, str, str]:
    """The coarse anchor identity a reference law places mass on."""
    created = len(subgoal.output_atoms)
    return (scale_band(created), topology_class(subgoal), heteroatom_pattern(subgoal))


def fit_anchor_prior(subgoals, *, alpha: float = 0.5) -> dict:
    """Additively smoothed marginal over anchor classes.

    Smoothing is not cosmetic. An unsmoothed empirical prior assigns zero to every class
    the corpus happens not to contain, which reintroduces exactly the reachability
    failure the mixture exists to prevent -- and 147 subgoals cannot have covered the
    class space.
    """
    counts = Counter(anchor_class(s) for s in subgoals)
    classes = sorted(counts)
    total = sum(counts.values()) + alpha * max(len(classes), 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": classes,
        "log_probability": {cls: math.log((counts[cls] + alpha) / total) for cls in classes},
        "alpha": alpha,
        "observations": sum(counts.values()),
        "smoothed_floor": math.log(alpha / total),
    }


def anchor_log_probability(prior: dict, cls) -> float:
    return prior["log_probability"].get(cls, prior["smoothed_floor"])


def fit_block_prior(observations, *, alpha: float = 0.5) -> dict:
    """q_block(C | A): which closure a proposal intervenes on, given the anchor class.

    `observations` are `(anchor_class, closure_name)` pairs. Backs off to the pooled
    closure distribution when an anchor class is thin, which is the same partial-pooling
    argument that made the additive site/mode terms transfer where a categorical lookup
    did not.
    """
    joint = defaultdict(Counter)
    pooled = Counter()
    for cls, closure in observations:
        joint[cls][closure] += 1
        pooled[closure] += 1
    closures = sorted(pooled)
    pooled_total = sum(pooled.values()) + alpha * max(len(closures), 1)
    table = {}
    for cls, counts in joint.items():
        total = sum(counts.values()) + alpha * max(len(closures), 1)
        table[cls] = {c: math.log((counts[c] + alpha) / total) for c in closures}
    return {
        "closures": closures,
        "per_anchor": table,
        "pooled": {c: math.log((pooled[c] + alpha) / pooled_total) for c in closures},
        "alpha": alpha,
    }


def block_log_probability(prior: dict, cls, closure: str) -> float:
    table = prior["per_anchor"].get(cls)
    if table and closure in table:
        return table[closure]
    return prior["pooled"].get(closure, math.log(prior["alpha"]))


def transition_statistics(sequences) -> dict:
    """How successful routes move between global and local options.

    `sequences` are per-route lists of option labels in order. This is where the
    compounding behaviour that made BRAF and 5HT1B work has to survive into the option
    formulation: a controller that can only make one global jump cannot express them.
    """
    counts = defaultdict(Counter)
    first = Counter()
    lengths = []
    for labels in sequences:
        if not labels:
            continue
        lengths.append(len(labels))
        first[labels[0]] += 1
        for before, after in pairwise(labels):
            counts[before][after] += 1
    matrix = {}
    for before, after_counts in counts.items():
        total = sum(after_counts.values())
        matrix[before] = {a: n / total for a, n in after_counts.items()}
    return {
        "first_option": dict(first.most_common()),
        "transition": matrix,
        "options_per_route": {
            "n": len(lengths),
            "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def expected_draws_to_class(prior: dict, cls) -> float:
    """How many proposals until this anchor neighbourhood appears.

    This, not exact-patch likelihood, is the quantity that decides whether a reference
    law is usable: a prior that ranks the right neighbourhood first is worthless if
    reaching it still takes more draws than the budget allows.
    """
    return math.exp(-anchor_log_probability(prior, cls))
