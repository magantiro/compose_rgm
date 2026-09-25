"""Tests for the frozen quality-aware finite-panel selection rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_fragment_linker_quality_guided_replay_v1 import select_quality_guided


def test_unseen_quality_precedes_higher_scoring_other_offers():
    class CaptureRng:
        def choice(self, count, *, p):
            assert count == 1
            assert np.allclose(p, (1.0,))
            return 0

    selected, tier = select_quality_guided(
        (("good_old", 100.0), ("bad_new", 90.0), ("good_new", -2.0)),
        {"good_old"},
        {"good_old": True, "bad_new": False, "good_new": True},
        CaptureRng(),
    )
    assert (selected, tier) == ("good_new", "unseen_quality")


def test_native_ratios_apply_within_priority_tier():
    class CaptureRng:
        def choice(self, count, *, p):
            assert count == 2
            assert np.allclose(p, (0.25, 0.75))
            return 1

    selected, tier = select_quality_guided(
        (("good_a", 0.0), ("good_b", np.log(3.0)), ("bad", 100.0)),
        set(),
        {"good_a": True, "good_b": True, "bad": False},
        CaptureRng(),
    )
    assert (selected, tier) == ("good_b", "unseen_quality")


def test_forced_repeat_remains_counted_when_panel_is_exhausted():
    selected, tier = select_quality_guided(
        (("old", 0.0),), {"old"}, {"old": False}, np.random.default_rng(0)
    )
    assert (selected, tier) == ("old", "forced_repeat")
    with pytest.raises(ValueError, match="empty or duplicate"):
        select_quality_guided((), set(), {}, np.random.default_rng(0))
