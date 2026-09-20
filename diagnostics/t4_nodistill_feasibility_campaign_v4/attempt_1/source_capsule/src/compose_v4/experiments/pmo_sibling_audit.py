"""Matched-parent diagnostics on existing predictions; no fitting or oracle calls."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isfinite
from statistics import mean

MODELS = ("parent_copy", "option_delta", "context_delta", "endpoint_score", "edit_delta")
TOLERANCE = 1e-12


def joined_rows(predictions: list[dict], original: dict) -> list[dict]:
    archive = {row["id"]: row for row in original["archive"]}
    attempts = {f"attempts/{r['attempt']:04}": r for r in original["attempts"]}
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("duplicate prediction identity")
    rows = []
    for row in predictions:
        attempt = attempts[row["id"]]
        parent = archive[attempt["parent"]]
        if (
            attempt["status"] != "complete"
            or not attempt["new_canonical"]
            or attempt["calls"] != row["query"]
            or attempt["score"] != row["score"]
            or attempt["option"] != row["option"]
            or attempt["scale"] != row["scale"]
        ):
            raise ValueError(f"prediction/attempt mismatch: {row['id']}")
        rows.append(
            {
                **row,
                "parent_id": attempt["parent"],
                "parent_smiles": parent["smiles"],
                "smiles": archive[row["id"]]["smiles"],
            }
        )
    return rows


def group_record(rows: list[dict]) -> dict:
    pairs = []
    for left, right in combinations(rows, 2):
        delta = left["score"] - right["score"]
        if abs(delta) <= TOLERANCE:
            continue
        outcomes = {}
        for model in MODELS:
            predicted = left["predictions"][model] - right["predictions"][model]
            outcomes[model] = 0.5 if abs(predicted) <= TOLERANCE else float(delta * predicted > 0)
        pairs.append(
            {
                "left": left["id"],
                "right": right["id"],
                "score_difference": delta,
                "cycle_comparison": "both_cycle"
                if left["cycle_rank_increase"] and right["cycle_rank_increase"]
                else "cycle_vs_other"
                if left["cycle_rank_increase"] != right["cycle_rank_increase"]
                else "neither_cycle",
                "concordant": outcomes,
            }
        )
    best, average = max(r["score"] for r in rows), mean(r["score"] for r in rows)
    selection = {}
    for model in MODELS:
        maximum = max(r["predictions"][model] for r in rows)
        winners = [r for r in rows if maximum - r["predictions"][model] <= TOLERANCE]
        selected = mean(r["score"] for r in winners)
        selection[model] = {
            "selected_ids": [r["id"] for r in winners],
            "selected_score": selected,
            "regret": best - selected,
            "gain_vs_uniform": selected - average,
            "best_probability": mean(best - r["score"] <= TOLERANCE for r in winners),
        }
    first = rows[0]
    return {
        "parent_id": first["parent_id"],
        "parent_smiles": first["parent_smiles"],
        "fit_through_query": first["fit_through_query"],
        "parent_available_at_snapshot": first["parent_query"] <= first["fit_through_query"],
        "mixed_cycle": len({r["cycle_rank_increase"] for r in rows}) > 1,
        "rows": rows,
        "pairs": pairs,
        "uniform_score": average,
        "best_score": best,
        "selection": selection,
    }


def summary(groups: list[dict]) -> dict:
    pairs = [p for g in groups for p in g["pairs"]]
    metrics = {}
    for model in MODELS:
        outcomes = [p["concordant"][model] for p in pairs]
        decisive = [x for x in outcomes if x != 0.5]
        metrics[model] = {
            "concordance": mean(outcomes) if outcomes else None,
            "parent_window_concordance": mean(
                mean(p["concordant"][model] for p in g["pairs"]) for g in groups if g["pairs"]
            )
            if pairs
            else None,
            "decisive_pairs": len(decisive),
            "decisive_coverage": len(decisive) / len(pairs) if pairs else None,
            "decisive_accuracy": mean(decisive) if decisive else None,
            **{
                f"mean_{key}": mean(g["selection"][model][key] for g in groups) if groups else None
                for key in ("selected_score", "regret", "gain_vs_uniform", "best_probability")
            },
        }
    return {
        "groups": len(groups),
        "parents": len({g["parent_id"] for g in groups}),
        "candidates": sum(len(g["rows"]) for g in groups),
        "ranking_pairs": len(pairs),
        "equal_score_groups": sum(not g["pairs"] for g in groups),
        "metrics": metrics,
    }


def audit_rows(rows: list[dict]) -> dict:
    buckets, parents = defaultdict(list), defaultdict(list)
    seen = set()
    for row in sorted(rows, key=lambda r: r["query"]):
        if row["id"] in seen or row["fit_through_query"] >= row["query"]:
            raise ValueError(f"duplicate or nonchronological prediction: {row['id']}")
        if row["parent_query"] >= row["query"]:
            raise ValueError(f"unavailable parent label: {row['id']}")
        if set(row["predictions"]) != set(MODELS) or not all(
            isfinite(v) and 0 <= v <= 1
            for v in (row["score"], row["parent_score"], *row["predictions"].values())
        ):
            raise ValueError(f"invalid prediction/score schema: {row['id']}")
        seen.add(row["id"])
        buckets[row["parent_id"], row["fit_through_query"]].append(row)
        parents[row["parent_id"]].append(row)
    for parent, children in parents.items():
        if len({(r["parent_index"], r["parent_query"], r["parent_score"]) for r in children}) != 1:
            raise ValueError(f"inconsistent parent label or identity: {parent}")
    groups = [group_record(value) for _, value in sorted(buckets.items()) if len(value) > 1]
    return {
        "prediction_count": len(rows),
        "parent_count": len(parents),
        "parents_with_multiple_children_across_snapshots": sum(
            len(v) > 1 for v in parents.values()
        ),
        "primary": summary(groups),
        "parent_observed_by_snapshot": summary(
            [g for g in groups if g["parent_available_at_snapshot"]]
        ),
        "parent_observed_after_snapshot": summary(
            [g for g in groups if not g["parent_available_at_snapshot"]]
        ),
        "mixed_cycle": summary([g for g in groups if g["mixed_cycle"]]),
        "excluded": [
            {"id": value[0]["id"], "reason": "singleton_exact_parent_snapshot"}
            for _, value in sorted(buckets.items())
            if len(value) == 1
        ],
        "groups": groups,
    }
