"""Reduce the completed A/B cells printed in the submitted PMO ablation.

The saved source reduction is not a replacement for raw oracle receipts. This
module only recomputes the submitted completed-pair arithmetic from its cells.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping

EXPECTED_COMPLETED = {
    "albuterol_similarity": 2,
    "celecoxib_rediscovery": 2,
    "gsk3b": 3,
    "isomers_c9h10n2o2pf2cl": 3,
    "ranolazine_mpo": 1,
    "scaffold_hop": 3,
}


class AblationError(ValueError):
    """The source reduction cannot support the declared completed-pair table."""


def _score(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationError(f"{context} must be a numeric score")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AblationError(f"{context} must be finite and in [0, 1]")
    return result


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def reduce_ablation(document: object, expected_completed: Mapping[str, int]) -> dict:
    """Use only cells where both arms completed both 1,000-receipt metrics."""
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "pmo_abc_ablation_reduction_v1"
    ):
        raise AblationError("source.schema_version is not pmo_abc_ablation_reduction_v1")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("budget_charged_calls") != 1008:
        raise AblationError("source.protocol.budget_charged_calls is not 1008")
    if protocol.get("reporting_prefix") != 1000:
        raise AblationError("source.protocol.reporting_prefix is not 1000")
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise AblationError("source.cells must be a list")
    seen: set[tuple[str, int]] = set()
    selected: list[dict] = []
    excluded: list[dict] = []
    for row in cells:
        if not isinstance(row, dict):
            raise AblationError("source.cells contains a non-object row")
        task, seed = row.get("task"), row.get("seed")
        if task not in expected_completed or isinstance(seed, bool) or not isinstance(seed, int):
            raise AblationError(f"unexpected objective or seed: {task!r}, {seed!r}")
        identity = (task, seed)
        if identity in seen:
            raise AblationError(f"duplicate objective-seed cell: {identity}")
        seen.add(identity)
        arm_a, arm_b = row.get("A"), row.get("B")
        if not isinstance(arm_a, dict) or not isinstance(arm_b, dict):
            raise AblationError(f"{identity}: A and B must be objects")
        if arm_a.get("state") != "COMPLETE":
            raise AblationError(f"{identity}: reused arm A did not complete")
        if arm_b.get("state") != "COMPLETE":
            excluded.append({"task": task, "seed": seed, "B_state": arm_b.get("state")})
            continue
        selected.append(
            {
                "task": task,
                "seed": seed,
                "A_top10": _score(arm_a.get("top10"), f"{identity} A.top10"),
                "B_top10": _score(arm_b.get("top10"), f"{identity} B.top10"),
                "A_auc": _score(arm_a.get("auc"), f"{identity} A.auc"),
                "B_auc": _score(arm_b.get("auc"), f"{identity} B.auc"),
            }
        )
    counts = Counter(row["task"] for row in selected)
    if counts != Counter(expected_completed):
        raise AblationError(
            f"completed-pair counts {dict(counts)} != declared {dict(expected_completed)}"
        )
    if len(cells) != 18 or len(selected) != 14 or len(excluded) != 4:
        raise AblationError(
            "submitted panel must contain 18 planned, 14 complete, and 4 excluded cells"
        )
    selected.sort(key=lambda row: (row["task"], row["seed"]))
    excluded.sort(key=lambda row: (row["task"], row["seed"]))

    per_objective: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["task"]].append(row)
    for task in sorted(grouped):
        rows = grouped[task]
        per_objective[task] = {
            field: _summary([row[field] for row in rows])
            for field in ("A_top10", "B_top10", "A_auc", "B_auc")
        }

    aggregate = {
        field: _summary([row[field] for row in selected])
        for field in ("A_top10", "B_top10", "A_auc", "B_auc")
    }
    paired = {}
    for metric in ("top10", "auc"):
        differences = [row[f"B_{metric}"] - row[f"A_{metric}"] for row in selected]
        paired[metric] = {
            "n": len(differences),
            "mean_B_minus_A": statistics.fmean(differences),
            "median_B_minus_A": statistics.median(differences),
            "population_sd_B_minus_A": statistics.pstdev(differences),
            "A_higher": sum(value < 0 for value in differences),
            "B_higher": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }
    return {
        "schema_version": "compose_submitted_pmo_ablation_subset_v1",
        "scope": "Saved 14 completed matched A/B cells; not raw oracle-receipt verification",
        "source_status": document.get("STATUS"),
        "selection_rule": "Both A and B COMPLETE, with Top-10 and AUC at the 1,000-receipt prefix",
        "planned_cells": len(cells),
        "completed_matched_cells": len(selected),
        "aggregate": aggregate,
        "paired": paired,
        "per_objective": per_objective,
        "cells": selected,
        "excluded": excluded,
    }
