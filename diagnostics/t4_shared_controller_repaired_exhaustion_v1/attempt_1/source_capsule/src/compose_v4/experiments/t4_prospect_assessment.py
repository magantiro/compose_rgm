"""Assess a prospected candidate pool without spending a docking call.

A pool from `constructive_composition.prospect` is eligible and unscored. Two things
can still be learned about it for free, and both are worth knowing before any charged
call is allocated.

**Free labels.** The recovered historical corpus holds 35,895 charged observations. An
endpoint the pool rediscovers already has a measured score, at zero new cost. A hit is
a rediscovery, not a discovery, and it says nothing about the pool's novel members --
but it is the only unbiased read on pool quality available before docking, because
section 7 of the strategy report measured structural proximity against score gap at
Spearman 0.12 and it is not a usable surrogate.

**Shape.** Whether the pool occupies the same size and construction regime as the
cell's historically high-scoring constructions. This is descriptive: matching the shape
of good molecules is not evidence of being good, and is reported as shape, not merit.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter

import numpy as np


def endpoint_sha256(smiles: str) -> str:
    return hashlib.sha256(smiles.encode()).hexdigest()


def free_labels(pool, records, *, cell: str) -> dict:
    """Historical scores for pool endpoints that were already charged, at no cost.

    Matching is by canonical endpoint identity within the same cell, because a score
    belongs to a molecule under one evaluator protocol and cells do not share
    endpoints in this corpus.
    """
    scored = {}
    for record in records:
        if record["cell"] != cell:
            continue
        current = scored.get(record["endpoint_sha256"])
        if current is None or record["score_mean"] < current:
            scored[record["endpoint_sha256"]] = record["score_mean"]
    hits = []
    for row in pool:
        digest = endpoint_sha256(row["endpoint"])
        if digest in scored:
            hits.append(
                {
                    "endpoint": row["endpoint"],
                    "primitives": row["primitives"],
                    "historical_score": scored[digest],
                }
            )
    hits.sort(key=lambda hit: hit["historical_score"])
    return {
        "pool_size": len(pool),
        "historical_endpoints_in_cell": len(scored),
        "rediscovered": len(hits),
        "rediscovery_rate": len(hits) / len(pool) if pool else 0.0,
        "best_rediscovered_score": hits[0]["historical_score"] if hits else None,
        "median_rediscovered_score": (
            statistics.median(hit["historical_score"] for hit in hits) if hits else None
        ),
        "hits": hits[:25],
        "unscored_members": len(pool) - len(hits),
        "interpretation": (
            "a rediscovered endpoint carries the score of the molecule, measured "
            "historically; it is evidence about this pool's quality and is not a new "
            "result, and it says nothing about the members that were never charged"
        ),
        "new_oracle_calls": 0,
    }


def shape_comparison(pool, records, *, cell: str, top: int = 30) -> dict:
    """Pool size and construction shape against the cell's best historical rows."""
    rows = sorted((r for r in records if r["cell"] == cell), key=lambda r: r["score_mean"])[:top]
    if not rows or not pool:
        return {"comparable": False, "pool_size": len(pool), "reference_rows": len(rows)}

    def summary(primitives, changed, created):
        return {
            "median_primitives": float(np.median(primitives)),
            "median_changed_originals": float(np.median(changed)),
            "median_created": float(np.median(created)),
        }

    return {
        "comparable": True,
        "pool": summary(
            [row["primitives"] for row in pool],
            [row["changed_originals"] for row in pool],
            [row["created"] for row in pool],
        ),
        "historical_best": summary(
            [r["primitive_count"] for r in rows],
            [r["changed_slot_count"] for r in rows],
            [r["net_created"] for r in rows],
        ),
        "reference_rows": len(rows),
        "reference_best_score": rows[0]["score_mean"],
        "pool_families": dict(sorted(Counter(f for row in pool for f in row["families"]).items())),
        "interpretation": (
            "shape only: occupying the same size regime as high-scoring constructions "
            "is not evidence of scoring well, and structural proximity was measured to "
            "be a poor predictor of docking utility"
        ),
    }
