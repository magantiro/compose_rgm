"""How the search spends its generations, and whether change compounds.

Dynamic's two autonomous wins were not single large programs. BRAF's champion
accumulated roughly 25 primitive edits over seven generations and finished with a
three-primitive refinement; 5HT1B's accumulated nine over four and finished with one.
The two cells it loses on reach comparable generation depth and accumulate far less.

This module measures that directly, per cell and per arm, so the hypothesis can be
checked rather than repeated: generation depth, cumulative ancestral edits, score
change per accepted generation, how concentrated parent selection is, and how often a
charged call continues recent work versus reaching back to an older parent.

All of it is read-only over recovered historical observations. Scores are compared
only within one evaluator protocol, because that is what the corpus contract requires.
"""

from __future__ import annotations

import statistics
from collections import Counter

import numpy as np

SCHEMA_VERSION = "t4_lineage_audit_v1"

# Arms that ran the delta-0.4 protocol with no route bank at runtime.
AUTONOMOUS_ARMS = frozenset({"v0", "v1", "v21"})


def index_by_entry(rows) -> dict:
    """Entry lookup keyed by cell, because entry ids are only unique inside one."""
    index = {}
    for row in rows:
        index.setdefault((row["cell"], row["entry_id"]), row)
    return index


def lineage_chain(row, index) -> list[dict]:
    """The construction and its ancestors, nearest first, stopping at a cycle."""
    chain, current, seen = [row], row, {(row["cell"], row["entry_id"])}
    while current is not None and current.get("genealogical_parent"):
        key = (current["cell"], current["genealogical_parent"])
        if key in seen:
            break
        seen.add(key)
        current = index.get(key)
        if current is None:
            break
        chain.append(current)
    return chain


def generation_depth(row, index) -> int:
    return len(lineage_chain(row, index)) - 1


def _quantiles(values, keys=(50, 90)) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        **{f"p{k}": float(np.percentile(values, k)) for k in keys},
        "max": max(values),
    }


def continuation_profile(rows, index) -> dict:
    """How far back a charged call reaches for its parent.

    A call whose parent was scored a moment ago is continuing a lineage. One whose
    parent was scored hundreds of calls earlier is returning to old material. Only
    rows carrying an exact call index participate; the rest are reported as excluded
    rather than imputed.
    """
    scored_at = {
        (row["cell"], row["entry_id"]): row["query"] for row in rows if row.get("query") is not None
    }
    gaps, without_index, parent_unscored = [], 0, 0
    for row in rows:
        if row.get("query") is None:
            without_index += 1
            continue
        parent = row.get("genealogical_parent")
        key = (row["cell"], parent) if parent else None
        if key is None or key not in scored_at:
            parent_unscored += 1
            continue
        gaps.append(row["query"] - scored_at[key])
    positive = [g for g in gaps if g > 0]
    return {
        "measured_calls": len(gaps),
        "excluded_no_call_index": without_index,
        "excluded_parent_never_scored": parent_unscored,
        "call_gap_to_parent": _quantiles(positive, keys=(25, 50, 75, 90)),
        "continued_within_10_calls": (
            sum(1 for g in positive if g <= 10) / len(positive) if positive else None
        ),
        "interpretation": (
            "the gap is how many charged calls separate a proposal from the call that "
            "scored its parent; small gaps mean the search is continuing a lineage"
        ),
    }


def parent_concentration(rows) -> dict:
    """How concentrated parent selection is, and how much of the archive gets reused."""
    parents = Counter(
        (row["cell"], row["genealogical_parent"]) for row in rows if row.get("genealogical_parent")
    )
    if not parents:
        return {"parents_used": 0}
    counts = sorted(parents.values(), reverse=True)
    total = sum(counts)
    return {
        "parents_used": len(parents),
        "selections": total,
        "selections_per_parent_median": statistics.median(counts),
        "share_of_selections_top_10_parents": sum(counts[:10]) / total,
        "distinct_parents_per_charged_call": len(parents) / len(rows),
    }


def score_progress(rows, index) -> dict:
    """Score change from parent to child, within the same protocol."""
    deltas = []
    for row in rows:
        parent = index.get((row["cell"], row.get("genealogical_parent")))
        if parent is None or parent["protocol"] != row["protocol"]:
            continue
        deltas.append(row["score"] - parent["score"])
    improved = [d for d in deltas if d < 0]
    return {
        "measured_edges": len(deltas),
        "improving_edges": len(improved),
        "improving_fraction": len(improved) / len(deltas) if deltas else None,
        "median_improvement_when_improving": (float(np.median(improved)) if improved else None),
        "median_delta": float(np.median(deltas)) if deltas else None,
    }


def cell_profile(rows, index) -> dict:
    """Everything the lineage hypothesis needs, for one cell and arm group."""
    depths = [generation_depth(row, index) for row in rows]
    ancestral = [
        row["ancestral_primitives"]
        for row in rows
        if isinstance(row.get("ancestral_primitives"), int)
    ]
    families = Counter(rule for row in rows for rule in (row.get("rule_counts") or {}))
    best = min(rows, key=lambda row: row["score"])
    return {
        "charged_calls": len(rows),
        "best_score": best["score"],
        "best_program_primitives": best["primitive_count"],
        "best_generation_depth": generation_depth(best, index),
        "best_ancestral_primitives": best.get("ancestral_primitives"),
        "generation_depth": _quantiles(depths),
        "ancestral_primitives": _quantiles(ancestral),
        "ancestral_primitives_reported_for": len(ancestral),
        "score_progress": score_progress(rows, index),
        "parent_concentration": parent_concentration(rows),
        "continuation": continuation_profile(rows, index),
        "channels": dict(Counter(row.get("channel") for row in rows).most_common()),
        "executor_rules": dict(families.most_common(8)),
    }


def audit(rows, protocols, *, cells) -> dict:
    """Per-cell autonomous-versus-bank profiles, inside each cell's own protocol."""
    index = index_by_entry(rows)
    result = {}
    for cell in cells:
        pool = [r for r in rows if r["cell"] == cell and r["protocol"] == protocols[cell]]
        groups = {
            "autonomous": [r for r in pool if r["arm"] in AUTONOMOUS_ARMS],
            "bank": [r for r in pool if r["arm"] == "full146"],
        }
        result[cell] = {
            name: cell_profile(group, index) if group else {"charged_calls": 0}
            for name, group in groups.items()
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "cells": result,
        "protocol_rule": "scores compared only within one evaluator protocol",
        "new_oracle_calls": 0,
    }
