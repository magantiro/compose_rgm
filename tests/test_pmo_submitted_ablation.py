"""The paper-specific PMO reducer keeps incomplete B cells out of both metrics."""

from __future__ import annotations

import copy

import pytest

from compose_v4.experiments.pmo_submitted_ablation import AblationError, reduce_ablation


def _source() -> dict:
    cells = []
    for seed, score in enumerate((0.2, 0.4, 0.6)):
        cells.append(
            {
                "task": "objective",
                "seed": seed,
                "A": {"state": "COMPLETE", "top10": score + 0.1, "auc": score},
                "B": {
                    "state": "COMPLETE" if seed < 2 else "PARTIAL@512",
                    "top10": score if seed < 2 else 0.99,
                    "auc": score - 0.1 if seed < 2 else None,
                },
            }
        )
    return {
        "schema_version": "pmo_abc_ablation_reduction_v1",
        "STATUS": "PROVISIONAL",
        "protocol": {"budget_charged_calls": 1008, "reporting_prefix": 1000},
        "cells": cells,
    }


def test_partial_top10_is_excluded_from_both_metrics() -> None:
    # The production reducer has a fixed 18/14 census. Build a matching fixture
    # to exercise the same selection rule without importing external result data.
    source = _source()
    source["cells"] = [
        {**copy.deepcopy(row), "task": task, "seed": offset * 3 + row["seed"]}
        for offset, task in enumerate(("a", "b", "c", "d", "e", "f"))
        for row in source["cells"]
    ]
    # Four incomplete cells leave the declared 14 completed pairs.
    for row in source["cells"]:
        if row["task"] not in ("a", "b", "c", "d") and row["B"]["state"] != "COMPLETE":
            row["B"] = {"state": "COMPLETE", "top10": 0.5, "auc": 0.4}
    counts = {"a": 2, "b": 2, "c": 2, "d": 2, "e": 3, "f": 3}
    result = reduce_ablation(source, counts)
    assert result["completed_matched_cells"] == 14
    assert len(result["excluded"]) == 4
    assert all(row["B_top10"] != 0.99 for row in result["cells"])
    assert result["aggregate"]["B_top10"]["n"] == 14
    assert result["aggregate"]["B_auc"]["n"] == 14


def test_duplicate_cell_fails() -> None:
    source = _source()
    source["cells"].append(copy.deepcopy(source["cells"][0]))
    with pytest.raises(AblationError, match="duplicate"):
        reduce_ablation(source, {"objective": 2})


def test_wrong_protocol_fails() -> None:
    source = _source()
    source["protocol"]["reporting_prefix"] = 512
    with pytest.raises(AblationError, match="reporting_prefix"):
        reduce_ablation(source, {"objective": 2})
