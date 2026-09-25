from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_fragment_linker_novelty_headroom_v1 import (
    summarize_prompt,
    unique_supported_offers,
    unseen_probability,
)


def test_unique_supported_offers_preserves_first_canonical_score() -> None:
    panel = {
        "offered": [
            {"status": "model_supported", "endpoint": "A", "mean_log_mark": -2.0},
            {"status": "model_supported", "endpoint": "A", "mean_log_mark": -1.0},
            {"status": "compiler_or_constraint_abstention"},
            {"status": "model_supported", "endpoint": "B", "mean_log_mark": float("-inf")},
        ]
    }
    assert unique_supported_offers(panel) == [("A", -2.0)]


def test_novelty_weighting_increases_conditional_unseen_probability() -> None:
    offers = [("A", -1.0), ("B", -1.0)]
    assert unseen_probability(offers, {"A"}, 1.0) == 0.5
    assert unseen_probability(offers, {"A"}, 4.0) == 0.8
    assert unseen_probability([], {"A"}, 4.0) == 0.0


def test_repeated_selection_with_unseen_offer_is_counted() -> None:
    def attempt(index: int, selected: str) -> dict:
        return {
            "attempt_index": index,
            "panel": {
                "selected_smiles": selected,
                "offered": [
                    {"status": "model_supported", "endpoint": "A", "mean_log_mark": -1.0},
                    {"status": "model_supported", "endpoint": "B", "mean_log_mark": -1.0},
                ],
            },
        }

    result = summarize_prompt([attempt(0, "A"), attempt(1, "A")])
    assert result["selected_distinct"] == 1
    assert result["selected_repeated"] == 1
    assert result["selected_repeated_despite_unseen_offer"] == 1
    assert math.isclose(result["conditional_expected_unseen_novelty4"], 1.8)
