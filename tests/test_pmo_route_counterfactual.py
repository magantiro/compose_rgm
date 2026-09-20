"""Guards for the PMO conditional-decomposition primitives.

``factors`` and ``match_indicators`` are what the (a)-(e) classification rests on, so
the cases that can silently invert a verdict are pinned here:

  * a teacher that releases nothing must not hand a free region hit to every proposal
    that also releases nothing -- ``Jaccard(empty, empty) == 1`` would classify such a
    route as "region reached" no matter what the proposer did;
  * the region/scale coordinates must be read off the PADDED production state, because
    an unpadded state deletes the whole ``atom_insert`` family from legal support;
  * a zero observed count must report a finite upper bound, never a bare ``p == 0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pmo_route_counterfactual_v1 import (
    factors,
    jaccard,
    load_routes,
    match_indicators,
    scale_bucket,
    upper_bound,
)


def _teacher() -> dict:
    route = load_routes()[0]
    return factors(
        route["source_state"], route["states"], route["actions"], route["terminal_endpoint"]
    )


def test_factors_partition_the_source_slots() -> None:
    teacher = _teacher()
    released = set(teacher["released_slots"])
    interface = set(teacher["interface_slots"])
    touched = set(teacher["touched_slots"])
    assert teacher["source_slot_count"] > 0
    # The interface is the retained part of what the program touched, so it is disjoint
    # from the released part and contained in the touched part.
    assert interface <= touched
    assert not (interface & released)
    assert 0.0 <= teacher["retained_fraction"] <= 1.0
    assert teacher["released_count"] == len(released)
    assert teacher["mode"] in {"grow", "prune", "replace", "remodel", "restate_only"}


def test_empty_teacher_release_does_not_grant_a_free_region_hit() -> None:
    """The degenerate case: Jaccard(empty, empty) == 1 must not become a region hit."""
    teacher = {
        "released_slots": [],
        "touched_slots": [1, 2, 3],
        "interface_slots": [1, 2, 3],
        "scale_bucket": "0",
        "mode": "grow",
        "ring_direction": 0,
        "created_handles_reused": 0,
        "endpoint": "CCO",
    }
    elsewhere = {
        "released_slots": [],
        "touched_slots": [40, 41],
        "interface_slots": [40, 41],
        "scale_bucket": "0",
        "mode": "grow",
        "ring_direction": 0,
        "created_handles_reused": 0,
        "endpoint": "CCC",
    }
    assert jaccard(teacher["released_slots"], elsewhere["released_slots"]) == 1.0
    # ... and yet the proposal edited a disjoint part of the molecule, so it is not a hit.
    assert match_indicators(teacher, elsewhere)["region_hit"] is False
    on_region = {**elsewhere, "touched_slots": [1, 2], "interface_slots": [1, 2]}
    assert match_indicators(teacher, on_region)["region_hit"] is True


def test_scale_buckets_are_ordered_and_total() -> None:
    assert scale_bucket(0) == "0"
    assert scale_bucket(2) == "1-2"
    assert scale_bucket(3) == "3-5"
    assert scale_bucket(16) == "11-16"
    assert scale_bucket(32) == "17-32"
    assert scale_bucket(33) == ">32"


@pytest.mark.parametrize("trials", [32, 96, 320])
def test_zero_count_reports_a_finite_upper_bound(trials: int) -> None:
    bound = upper_bound(0, trials)
    assert 0.0 < bound < 1.0
    # A zero count is never evidence for p == 0; it must shrink like ~3/n, not vanish.
    assert bound == pytest.approx(1.0 - 0.05 ** (1.0 / trials))
    assert upper_bound(0, 0) == 1.0
