"""Pendant support gates count all draws and all attempted outputs."""

import copy

import numpy as np
import pytest
from run_fragment_pendant_support_v1 import support_summary

from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel


def _rows():
    prompts = [str(index) for index in range(10)]
    rows = []
    for drug in prompts:
        for index in range(2):
            offered = [
                {"draw": draw, "status": "compiler_or_constraint_abstention"} for draw in range(8)
            ]
            if drug == "0" and index == 0:
                offered[0] = {
                    "draw": 0,
                    "status": "model_supported",
                    "provenance": {"pendant_plan": {"draws": [{"heavy_atoms": 1, "rings": 1}]}},
                }
            rows.append(
                {
                    "drug": drug,
                    "attempt_index": index,
                    "panel": {"offered_count": 8, "offered": offered, "output_count": 1},
                    "selected_valid_connected": True,
                    "selected_constraint_fidelity": True,
                }
            )
    return rows, {
        "prompts": prompts,
        "required_output_coverage": 0.9,
        "required_one_atom_model_supported": 1,
        "required_ring_bearing_model_supported": 1,
    }


def test_support_summary_preserves_full_denominators_and_falsifiers():
    rows, contract = _rows()
    result = support_summary(rows, contract)
    assert result["support_pass"]
    assert result["offered_draws"] == 160
    assert result["model_supported_one_atom_draws"] == 1
    assert result["model_supported_ring_bearing_draws"] == 1
    missing = copy.deepcopy(rows)
    for row in missing[:3]:
        row["panel"]["output_count"] = 0
    assert not support_summary(missing, contract)["checks"]["output_at_least_90_percent"]
    missing[1]["selected_constraint_fidelity"] = False
    assert not support_summary(missing, contract)["support_pass"]
    with pytest.raises(ValueError, match="two attempts"):
        support_summary(rows[:-1], contract)
    rows[0]["panel"]["offered"].pop()
    with pytest.raises(ValueError, match="eight-draw"):
        support_summary(rows, contract)


def test_empty_pendant_panel_has_eight_recorded_refusals(monkeypatch):
    import compose_v4.benchmark.fragment_pendant_programs as module

    def refuse(*_args):
        raise ValueError("fixture refusal")

    monkeypatch.setattr(module, "propose_pendant_decoration", refuse)
    rng = np.random.default_rng(7)
    before = copy.deepcopy(rng.bit_generator.state)
    panel = sample_pendant_panel(None, None, None, rng)
    assert panel.selected is None
    assert panel.receipt["output_count"] == 0
    assert panel.receipt["offered_count"] == 8
    assert [row["draw"] for row in panel.receipt["offered"]] == list(range(8))
    assert panel.receipt["rng_state_before"] == before
