from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_fiber_route_recognition import (
    _archive_rows,
    choose_probe,
    rank_probe,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_probe_rule_selects_only_after_model_can_fit() -> None:
    archive = json.loads(
        (ROOT / "diagnostics/t4_program_curriculum/attempt_1/braf_1_archive.json").read_text()
    )
    root, score, rows = _archive_rows(archive, delta=0.4)
    probe = choose_probe(rows, minimum_prior_outcomes=4)
    assert root
    assert score == -9.4
    assert probe["query_index"] == 26
    assert probe["score"] == -11.1
    assert probe["trace_complete"] is True
    assert sum(row["query_index"] < probe["query_index"] for row in rows) == 7


def test_rank_reports_tie_interval_without_using_outcomes() -> None:
    root = "CCO"
    fiber = Fiber(root, 0.0, support="benchmark_only")
    state = SearchState(archive={root: -1.0})
    value = ProgramValue()
    value.weights = np.zeros(15)
    value._mean = np.zeros(15)
    value._scale = np.ones(15)
    value.n = 4
    base = {
        "parent": root,
        "parent_score": -1.0,
        "similarity": 1.0,
        "qed": 0.5,
        "sa": 2.0,
        "delta": 0.0,
        "regions": 1,
        "created": 0,
        "deleted": 0,
        "families": [],
    }
    menu = [
        {**base, "endpoint": "CCN"},
        {**base, "endpoint": "CCC"},
        {**base, "endpoint": "CCCl"},
    ]
    rank = rank_probe(
        menu,
        probe_endpoint="CCN",
        value=value,
        state=state,
        fiber=fiber,
        tolerance=1e-12,
    )
    assert rank.optimistic == 1
    assert rank.pessimistic == 3
    assert rank.tied == 3
    assert rank.denominator == 3
