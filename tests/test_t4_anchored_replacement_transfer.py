from __future__ import annotations

import copy

import pytest

from compose_v4.experiments.t4_anchored_replacement_transfer import (
    compare_transfer_cells,
)


def _arm(lane: str, count: int, *, failures: int = 0) -> dict:
    return {
        "lane": lane,
        "unique_eligible_endpoints": count,
        "worker_failures": failures,
        "new_oracle_calls": 0,
    }


def _summaries() -> dict:
    return {
        "jak2_0": {
            "shallow": _arm("shallow", 9),
            "anchored_replacement": _arm("anchored_replacement", 15),
        },
        "jak2_2": {
            "shallow": _arm("shallow", 10),
            "anchored_replacement": _arm("anchored_replacement", 12),
        },
    }


def test_transfer_requires_nonzero_support_on_every_cell() -> None:
    value = compare_transfer_cells(_summaries(), minimum_panel_yield=12)
    assert value["transfer_support_gate"] == "PASS"
    assert value["full_panel_yield_gate"] == "PASS"
    failed = _summaries()
    failed["jak2_2"]["anchored_replacement"]["unique_eligible_endpoints"] = 0
    assert compare_transfer_cells(failed, minimum_panel_yield=12)["transfer_support_gate"] == "FAIL"


def test_transfer_fails_engineering_on_worker_failure() -> None:
    failed = copy.deepcopy(_summaries())
    failed["jak2_0"]["anchored_replacement"]["worker_failures"] = 1
    value = compare_transfer_cells(failed, minimum_panel_yield=12)
    assert value["engineering_gate"] == "FAIL"
    assert value["transfer_support_gate"] == "FAIL"


def test_transfer_rejects_missing_lane_and_charged_calls() -> None:
    missing = _summaries()
    del missing["jak2_0"]["shallow"]
    with pytest.raises(ValueError, match="two frozen"):
        compare_transfer_cells(missing, minimum_panel_yield=12)
    charged = _summaries()
    charged["jak2_0"]["shallow"]["new_oracle_calls"] = 1
    with pytest.raises(ValueError, match="charged call"):
        compare_transfer_cells(charged, minimum_panel_yield=12)
