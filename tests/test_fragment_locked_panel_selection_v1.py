"""Focused invariants for locked fragment selection replay."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/analyze_fragment_locked_panel_selection_v1.py"
SPEC = importlib.util.spec_from_file_location("fragment_locked_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _raw(*entries):
    offered = []
    for draw in range(8):
        if draw in entries:
            offered.append({
                "draw": draw, "status": "model_supported",
                "endpoint": f"C{draw}", "mean_log_mark": float(draw),
            })
        else:
            offered.append({"draw": draw, "status": "compiler_or_constraint_abstention"})
    return offered


def test_prefix_counts_attempted_offers_including_failures():
    raw = _raw(1, 3, 7)
    assert module.admitted_offers(raw, 1) == ()
    assert [offer.draw for offer in module.admitted_offers(raw, 2)] == [1]
    assert [offer.draw for offer in module.admitted_offers(raw, 4)] == [1, 3]
    assert [offer.draw for offer in module.admitted_offers(raw, 8)] == [1, 3, 7]


def test_uniform_and_novelty_probabilities():
    offers = module.admitted_offers(_raw(0, 1), 8)
    assert np.allclose(module.probabilities(offers, selector="uniform", emitted=frozenset()), [0.5, 0.5])
    learned = module.probabilities(offers, selector="learned", emitted=frozenset())
    assert math.isclose(float(learned.sum()), 1.0)
    assert learned[1] > learned[0]
    novelty = module.probabilities(
        offers, selector="learned_novelty4", emitted=frozenset({"C1"})
    )
    expected_first = 4.0 / (4.0 + math.e)
    assert math.isclose(float(novelty[0]), expected_first)


def test_singleton_selection_is_identical_and_empty_is_failure():
    offers = module.admitted_offers(_raw(5), 8)
    for selector in ("learned", "learned_novelty4", "uniform"):
        assert module.draw_endpoint(
            offers, selector=selector, emitted=frozenset(), seed=("test", selector)
        ) == "C5"
        assert module.draw_endpoint(
            (), selector=selector, emitted=frozenset(), seed=("test", selector)
        ) == ""


def test_deployed_receipt_fails_on_probability_mismatch():
    offers = module.admitted_offers(_raw(0, 1), 8)
    panel = {
        "selected_smiles": "C1",
        "selection": {
            "selected_index": 1, "selected_draw": 1,
            "selected_probability": 0.5,
            "mean_native_log_mark_probability": 1.0,
            "unique_model_supported_endpoints": 2,
        },
    }
    with pytest.raises(ValueError, match="probability"):
        module.verify_deployed_selection(
            panel, offers, selector="learned", emitted=frozenset()
        )
