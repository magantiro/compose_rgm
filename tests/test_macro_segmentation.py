"""The segmentation DP, including the two bugs that made it unable to report a failure."""
from __future__ import annotations

from compose_v4.experiments.macro_segmentation import (
    describe,
    min_segments,
    primitive_dips_below_source,
    source_relative_tolerance,
)


def test_monotone_route_is_one_segment():
    assert min_segments([0.1, 0.2, 0.3, 0.4]) == (1, [0, 3])


def test_interior_dip_is_absorbed_by_one_macro():
    # the recorded perindopril and qed shapes: a deep midpoint, both endpoints fine
    assert min_segments([0.360, 0.009, 0.809])[0] == 1
    assert min_segments([0.796, 0.251, 0.948])[0] == 1


def test_endpoint_below_source_admits_no_staging():
    assert min_segments([0.5, 0.6, 0.2]) == (None, None)


def test_length_cap_can_make_a_route_infeasible():
    # absorbing the dip needs a 2-primitive macro; capping at 1 forces a boundary on it
    assert min_segments([0.360, 0.009, 0.809], 1) == (None, None)


def test_cap_forces_a_boundary_and_the_dp_finds_a_legal_one():
    scores = [0.10, 0.50, 0.02, 0.60]
    k, bounds = min_segments(scores, 2)
    assert k == 2 and bounds == [0, 1, 3]


def test_tolerance_is_zero_exactly_when_a_monotone_staging_exists():
    assert source_relative_tolerance([0.1, 0.2, 0.3], 23) == 0.0


def test_tolerance_is_positive_and_correct_when_none_exists():
    # REGRESSION: min_segments returns a TUPLE, so `is not None` was always True and the
    # early return fired on every route -- the instrument could only ever report 0.0.
    eps = source_relative_tolerance([0.5, 0.1, 0.2], 1)
    assert eps > 0.0
    assert abs(eps - 0.4) < 1e-3


def test_tolerance_is_measured_against_the_source_not_per_boundary():
    # REGRESSION: a per-boundary allowance accumulates across boundaries, so a long
    # staircase would read as needing ~0 tolerance.  Against the source it does not.
    staircase = [0.50] + [0.50 - 0.02 * i for i in range(1, 12)] + [0.90]
    eps = source_relative_tolerance(staircase, 2)
    assert eps > 0.15, "a descending staircase must require a real drop below the source"


def test_primitive_dip_detection():
    assert primitive_dips_below_source([0.3, 0.1, 0.9])
    assert not primitive_dips_below_source([0.3, 0.4, 0.9])


def test_describe_reports_macro_sizes_that_sum_to_the_route():
    scores = [0.10, 0.50, 0.02, 0.60]
    d = describe(scores, max_len=2)
    assert sum(d["macro_sizes"]) == d["n_primitives"]
    assert d["macro_feasible_within_five"] is True
