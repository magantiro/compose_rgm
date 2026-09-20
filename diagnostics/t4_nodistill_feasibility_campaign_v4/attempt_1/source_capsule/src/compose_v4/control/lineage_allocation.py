"""Choose which archived molecule gets the next expensive evaluation.

The measured failure this addresses: on the cells Dynamic loses, a charged call
reaches back a median of 146-162 calls for its parent and continues recent work only
2.2-2.4% of the time, against 67-77 and 7.5-8.9% on the cells it wins. Parent
selection is score-blind, so a node whose descendant later scored -14 carries exactly
the weight of one whose descendants all scored -8. Nothing propagates.

The allocator here is the smallest change that fixes that: give every archived node
the value of the best molecule found anywhere beneath it, and select parents by that
backed-up value traded against how much budget the node has already had.

Three choices are deliberate and are consequences of the regime, which was measured
at roughly one evaluation per archive node with the most-reused parent receiving
about four children:

* **max-backup, not mean.** This is optimization, not game-value estimation, and at
  one sample per node a mean is a single noisy draw. Repeated dockings of one
  molecule were measured with a 90th-percentile spread of 2.9 units.
* **rank, not raw score.** One pathological docking outlier must not be able to
  capture the budget.
* **uniform prior for now.** `prior` is the slot a learned continuation prior fills
  later; leaving it uniform makes this a strict generalization of the current
  score-blind allocator rather than a different thing.
"""

from __future__ import annotations

import math
from collections import defaultdict

SCHEMA_VERSION = "lineage_allocation_v1"


def backed_up_values(nodes, *, better=min) -> dict:
    """Best objective found at a node or anywhere beneath it.

    `nodes` maps node id to a record carrying `parent` and `score`. Ancestry is
    walked upward per node, so a node with no scored descendant keeps its own score
    and a node whose descendant scored well inherits that.
    """
    value = {key: record["score"] for key, record in nodes.items()}
    for key, record in nodes.items():
        current, seen = record.get("parent"), {key}
        while current is not None and current in nodes and current not in seen:
            seen.add(current)
            value[current] = better(value[current], record["score"])
            current = nodes[current].get("parent")
    return value


def rank_scores(values, *, lower_is_better=True) -> dict:
    """Map objective values onto [0, 1] by rank, best at 1.

    Ties share a rank, and a single outlier moves one rank step rather than
    dominating an arithmetic scale.
    """
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not lower_is_better)
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    ranks, position = {}, 0
    while position < len(ordered):
        stop = position
        while stop + 1 < len(ordered) and ordered[stop + 1][1] == ordered[position][1]:
            stop += 1
        shared = (position + stop) / 2
        for index in range(position, stop + 1):
            ranks[ordered[index][0]] = 1.0 - shared / (len(ordered) - 1)
        position = stop + 1
    return ranks


def select_parent(nodes, visits, *, exploration: float, prior=None, lower_is_better=True):
    """PUCT-shaped selection over archived nodes, returning the chosen node id.

    With a uniform prior and a large exploration constant this reduces to selecting
    among the least-used nodes, which is what the current score-blind allocator does.
    """
    if not nodes:
        raise ValueError("cannot allocate a call against an empty archive")
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError("exploration constant must be finite and non-negative")
    values = backed_up_values(nodes, better=min if lower_is_better else max)
    ranks = rank_scores(values, lower_is_better=lower_is_better)
    prior = prior or {key: 1.0 / len(nodes) for key in nodes}
    total = sum(visits.get(key, 0) for key in nodes)
    best, chosen = -math.inf, None
    for key in sorted(nodes):
        used = visits.get(key, 0)
        bonus = exploration * prior.get(key, 0.0) * math.sqrt(total + 1) / (1 + used)
        score = ranks[key] + bonus
        if score > best:
            best, chosen = score, key
    return chosen


def score_blind_parent(nodes, visits, rng):
    """The current allocator: equal weight before duplicate penalties."""
    if not nodes:
        raise ValueError("cannot allocate a call against an empty archive")
    keys = sorted(nodes)
    return keys[int(rng.integers(len(keys)))]


def lineage_of(node, nodes) -> list:
    """A node and its ancestors, nearest first."""
    chain, current, seen = [], node, set()
    while current is not None and current in nodes and current not in seen:
        seen.add(current)
        chain.append(current)
        current = nodes[current].get("parent")
    return chain


def replay(records, *, exploration: float, rng, lower_is_better=True) -> dict:
    """Counterfactual: which parent would each allocator have chosen, call by call?

    This cannot produce counterfactual scores, because the molecule a different
    parent would have produced was never made. What it does measure is whether an
    allocator concentrates budget on the ancestry that actually produced the best
    molecule, which is a free leading indicator and is reported as exactly that.
    """
    ordered = sorted(records, key=lambda r: r["call"])
    champion = min(ordered, key=lambda r: r["score"] * (1 if lower_is_better else -1))
    nodes, visits = {}, defaultdict(int)
    winning, blind_hits, backed_hits, decisions = set(), 0, 0, 0
    for record in ordered:
        if nodes:
            decisions += 1
            blind = score_blind_parent(nodes, visits, rng)
            backed = select_parent(
                nodes, visits, exploration=exploration, lower_is_better=lower_is_better
            )
            blind_hits += blind in winning
            backed_hits += backed in winning
        nodes[record["id"]] = {"parent": record.get("parent"), "score": record["score"]}
        if record.get("parent") is not None:
            visits[record["parent"]] += 1
        if record["id"] == champion["id"]:
            winning = set(lineage_of(champion["id"], nodes))
    return {
        "schema_version": SCHEMA_VERSION,
        "calls": len(ordered),
        "decisions": decisions,
        "champion_score": champion["score"],
        "champion_lineage_length": len(winning),
        "score_blind_on_winning_lineage": blind_hits,
        "backed_up_on_winning_lineage": backed_hits,
        "score_blind_rate": blind_hits / decisions if decisions else None,
        "backed_up_rate": backed_hits / decisions if decisions else None,
        "exploration": exploration,
        "interpretation": (
            "the winning lineage is only known after the fact, so this measures "
            "concentration on it, not a counterfactual score; no molecule that was "
            "never made can be evaluated here"
        ),
        "new_oracle_calls": 0,
    }
