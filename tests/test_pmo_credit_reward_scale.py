"""The PMO credit direction, and the SCALE its optimism bonus is calibrated against.

Two different invariants live here and they fail for different reasons.

DIRECTION is a correctness invariant and must always hold: a child that improved the parent
must end up preferred over one that did not, under celecoxib's maximize convention. A flipped
sign here produces a controller that confidently allocates budget away from every improvement
it finds, which is exactly what `pmo_credit.improvement` warns about.

SCALE is a calibration fact, not a law, and is pinned so it cannot drift silently. The optimism
bonus `prior_weight / sqrt(trials + 1)` is an ABSOLUTE constant while realized PMO improvements
are ~0.008 per productive child. Measured, an untried cell is therefore preferred over a cell
that just delivered a typical improvement, and a cell must deliver roughly ten times the typical
signal merely to break even. That was recorded as an observation long before its consequence was
measured: on real frozen pools the resulting allocation selects BELOW the pool mean, i.e. worse
than choosing at random.

The break-even assertion is deliberately a band rather than an equality. Changing `prior_weight`
is a legitimate calibration decision; doing it without noticing that it moves the exploration
scale by an order of magnitude is not.
"""

from __future__ import annotations

import math

from compose_v4.control.pmo_credit import (
    DEFAULT_PRIOR_WEIGHT,
    CreditKey,
    PopulationCredit,
    improvement,
)

#: Mean positive improvement per productive child, measured across six channel x task cells of
#: real PMO runs (0.0052-0.0114). The scale the exploration bonus ought to be comparable to.
MEASURED_TYPICAL_IMPROVEMENT = 0.0077


def _key(family: str) -> CreditKey:
    return CreditKey(basin="b", parent="p", family=family, scale="refine")


def _break_even(credit: PopulationCredit, untried_value: float) -> float:
    """Smallest single delivered improvement that ties an untried cell."""
    low, high = 0.0, 5.0
    for _ in range(60):
        middle = (low + high) / 2
        probe = _key(f"probe_{middle!r}")
        credit.observe(probe, middle)
        low, high = (middle, high) if credit.value(probe) < untried_value else (low, middle)
    return high


def test_credit_direction_prefers_the_improving_child():
    assert improvement(0.45, 0.30, direction="maximize") > 0
    assert improvement(0.15, 0.30, direction="maximize") < 0

    credit = PopulationCredit()
    credit.observe(_key("gained"), improvement(0.45, 0.30, direction="maximize"))
    credit.observe(_key("lost"), improvement(0.15, 0.30, direction="maximize"))
    assert credit.value(_key("gained")) > credit.value(_key("lost"))

    shares = credit.allocate([_key("gained"), _key("lost")])
    assert shares[0] > shares[1], "budget must favour the cell that delivered an improvement"


def test_a_negative_outcome_never_counts_as_upside():
    """A failed child costs one call; the parent survives, so `df < 0` contributes zero."""
    credit = PopulationCredit()
    credit.observe(_key("lost"), -0.5)
    cell = credit.cell(_key("lost"))
    assert cell.trials == 1
    assert cell.positive_improvement_sum == 0.0
    assert cell.improvements == 0


def test_optimism_bonus_is_pinned_against_the_measured_reward_scale():
    """An untried cell currently outranks one delivering a TYPICAL improvement.

    This is the calibration defect, pinned. It is not a law: if the optimism term is rescaled
    to the empirical improvement scale this test must be updated deliberately, with the new
    break-even stated.
    """
    credit = PopulationCredit()
    untried = credit.value(_key("never_tried"))
    assert math.isclose(untried, DEFAULT_PRIOR_WEIGHT, rel_tol=1e-9)

    credit.observe(_key("typical"), MEASURED_TYPICAL_IMPROVEMENT)
    assert credit.value(_key("typical")) < untried, (
        "a cell delivering the measured typical improvement currently ranks BELOW an untried "
        "cell; if this assertion fails the optimism scale has been changed"
    )

    break_even = _break_even(credit, untried)
    assert 0.07 <= break_even <= 0.08, f"break-even moved to {break_even:.4f}"
    ratio = break_even / MEASURED_TYPICAL_IMPROVEMENT
    assert ratio > 8.0, (
        f"optimism currently demands {ratio:.0f}x the typical signal to break even; a rescaling "
        "that lowers this is the intended repair, and this bound documents the starting point"
    )
