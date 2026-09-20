"""Empirical archive-improvement allocation, not an exact value function.

No chemistry, oracle, or persistence lives here. Full reference rows remain
available; only their selection probabilities change.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from compose_v4.control.continuation import _tilted


def top10_sum(observed: dict[str, float]) -> float:
    return float(sum(sorted(observed.values(), reverse=True)[:10]))


def scale_cell(release: float) -> str:
    if not np.isfinite(release) or not 0 <= release <= 1:
        raise ValueError(f"invalid intended release: {release}")
    return "local" if release <= 0.2 else "regional" if release <= 0.5 else "global"


def parent_distribution(archive: list[dict], observed: dict[str, float], exploration=0.2):
    """Canonical mass first, then equal mass among exact-state variants."""
    groups = defaultdict(list)
    for index, row in enumerate(archive):
        groups[row["smiles"]].append(index)
    identities = sorted(groups)
    if not identities or not 0 < exploration <= 1:
        raise ValueError("parent allocation requires an archive and positive exploration")
    scores = np.asarray([observed[s] for s in identities])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("parent scores must be finite PMO values in [0,1]")
    ranks = 1 + (scores[None, :] > scores[:, None]).sum(axis=1)
    weight = 1.0 / ranks
    weight = (1 - exploration) * weight / weight.sum() + exploration / len(weight)
    probabilities = np.zeros(len(archive))
    for smiles, mass in zip(identities, weight):
        probabilities[groups[smiles]] = mass / len(groups[smiles])
    return probabilities


def sample_archive_parents(archive: dict[str, dict], n: int, rng, *, exploration=0.2):
    """One incumbent slot plus stochastic full-archive branching, not an SMC row.

    The archive has one exact representative per canonical molecule. Rank-based
    sampling is an optimization heuristic; its exploration mass is not uncertainty.
    """
    if type(n) is not int or n < 2 or not archive:
        raise ValueError("archive branching requires at least two slots and a nonempty archive")
    if any(s != row["smiles"] for s, row in archive.items()):
        raise ValueError("archive key differs from molecular identity")
    rows = [archive[s] for s in sorted(archive)]
    probabilities = parent_distribution(
        rows, {s: r["score"] for s, r in archive.items()}, exploration
    )
    best = min(range(len(rows)), key=lambda i: (-rows[i]["score"], rows[i]["smiles"]))
    indices = [best, *map(int, rng.choice(len(rows), size=n - 1, p=probabilities))]
    return [rows[i] for i in indices], {
        "schema_version": "archive_parent_selection_v1",
        "canonical_order": [r["smiles"] for r in rows],
        "probabilities": probabilities.tolist(),
        "indices": indices,
        "incumbent_index": best,
        "exploration": exploration,
        "interpretation": "slot zero retains incumbent; remaining slots sample rank-weighted archive with uniform exploration; not a Doob or SMC transition",
    }


class ArchiveCredit:
    """One observation per attempted option; update its realized descendant credit.

    This is an online search heuristic. Pseudocounts are regularization, not
    Bayesian evidence or uncertainty bounds. Ancestors share reward deliberately.
    """

    def __init__(self, *, discount=0.9, pseudocount=2.0):
        if not 0 < discount <= 1 or pseudocount <= 0:
            raise ValueError("invalid credit discount or pseudocount")
        self.discount, self.pseudocount = discount, pseudocount
        self.edges: dict[str, dict] = {}

    def observe(self, edge_id, *, parent_chain, scale, option, reward):
        if edge_id in self.edges or not np.isfinite(reward) or not 0 <= reward <= 1 + 1e-12:
            raise ValueError("duplicate edge or invalid archive reward")
        if len(set(parent_chain)) != len(parent_chain) or any(
            ancestor not in self.edges for ancestor in parent_chain
        ):
            raise ValueError("credit requires an already observed acyclic ancestry")
        self.edges[edge_id] = {
            "where": scale,
            "what": option,
            "direct": float(reward),
            "credit": float(reward),
        }
        for depth, ancestor in enumerate(reversed(parent_chain), start=1):
            self.edges[ancestor]["credit"] += self.discount**depth * reward

    def values(self, stage, cells):
        if stage not in ("where", "what"):
            raise ValueError("task allocation acts only on WHERE and WHAT")
        rows = [r for r in self.edges.values() if r[stage] is not None]
        pooled = sum(r["credit"] for r in rows) / len(rows) if rows else 0.0
        counts, totals = defaultdict(int), defaultdict(float)
        for row in rows:
            counts[row[stage]] += 1
            totals[row[stage]] += row["credit"]
        estimates = np.asarray(
            [
                (totals[c] + self.pseudocount * pooled) / (counts[c] + self.pseudocount)
                for c in cells
            ]
        )
        return estimates, [counts[c] for c in cells]

    def distribution(self, reference, stage, cells, *, adaptive):
        p = np.asarray(reference, dtype=float)
        if p.shape != (len(cells),) or not len(p) or np.any(p <= 0) or not np.isclose(p.sum(), 1):
            raise ValueError("allocation requires a normalized full-support reference row")
        estimates, counts = self.values(stage, cells)
        q, eta, kl = p.copy(), 0.0, 0.0
        if adaptive:
            q, eta, kl = _tilted(p, np.exp(estimates), kappa=1.0, exploration=0.1)
        if not np.isclose(q.sum(), 1) or np.any(q < 0.1 * p - 1e-12) or kl > 1 + 1e-10:
            raise ValueError("allocation violated normalization, exploration or KL")
        return q, {
            "reference": p.tolist(),
            "probabilities": q.tolist(),
            "cells": list(cells),
            "estimated_credit": estimates.tolist(),
            "cell_trials": counts,
            "eta": eta,
            "kl": kl,
            "total_variation": float(np.abs(q - p).sum() / 2),
        }
