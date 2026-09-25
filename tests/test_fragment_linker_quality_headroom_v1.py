"""Focused tests for the public fragment-quality predicate accounting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_fragment_linker_quality_headroom_v1 import quality_flags


def test_quality_thresholds_are_inclusive():
    assert quality_flags(4.0, 0.6) == (False, False, True)


def test_quality_failure_categories_remain_distinct():
    assert quality_flags(3.9, 0.59) == (True, False, False)
    assert quality_flags(4.1, 0.61) == (False, True, False)
    assert quality_flags(4.1, 0.59) == (True, True, False)


def test_nonfinite_property_values_do_not_pass_quality():
    assert quality_flags(float("nan"), 0.9) == (False, True, False)
    assert quality_flags(2.0, float("nan")) == (True, False, False)
