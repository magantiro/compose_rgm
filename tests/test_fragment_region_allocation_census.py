"""Check exact context permutation and remaining-capacity integration."""

import pytest

from tools.census_fragment_region_allocation import allocation_law


def test_order_averaging_and_minimum_reservation():
    cells = {"A": {(1, 0): 1.0, (3, 1): 1.0}, "B": {(2, 0): 1.0, (4, 2): 1.0}}
    # Either order permits all three feasible size/ring totals, but the
    # larger first draw forces the minimum at the second boundary.
    law = allocation_law(["A", "B"], cells, 5)
    assert law == pytest.approx({(3, 0): 0.25, (5, 1): 0.375, (5, 2): 0.375})
    assert allocation_law(["B", "A"], cells, 5) == law
    with pytest.raises(ValueError, match="do not fit"):
        allocation_law(["A", "B"], cells, 2)
