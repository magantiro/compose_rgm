"""Budget-explicit PMO AUC aggregation: the guards that stop a 250 vs 10k mix-up.

The reference for the AUC value itself is the PRODUCTION ``pmo_top_ten_auc``,
not a transcription of it, so a drift in the wrapper cannot move both sides of
the comparison together.
"""

from __future__ import annotations

import pytest

from compose_v4.control.program_task import pmo_top_ten_auc
from compose_v4.eval.pmo_budget_aggregation import (
    PMO_OFFICIAL_BUDGET,
    aggregate_suite,
    frozen_tail_projection,
    score_calls_log,
    top_ten_auc,
)


def _rising(n: int) -> list[float]:
    return [min(1.0, 0.2 + 0.7 * i / max(1, n - 1)) for i in range(n)]


# ---- score-vs-calls logging ----


def test_score_calls_log_is_monotone_and_matches_curve_fields():
    curve = score_calls_log(_rising(30))
    assert len(curve) == 30
    assert set(curve[0]) == {"charged_queries", "best_score", "top10_score"}
    assert [row["charged_queries"] for row in curve] == list(range(1, 31))
    bests = [row["best_score"] for row in curve]
    assert bests == sorted(bests)
    # top10_score is the mean of the up-to-10 best seen so far
    assert curve[0]["top10_score"] == pytest.approx(curve[0]["best_score"])


def test_score_calls_log_refuses_out_of_range_reward():
    with pytest.raises(ValueError):
        score_calls_log([0.5, 1.5])
    with pytest.raises(ValueError):
        score_calls_log([float("nan")])


# ---- the value delegates to production, it does not reimplement ----


def test_value_equals_production_top_auc():
    scores = _rising(250)
    reading = top_ten_auc(scores, budget=250, authorized_budget=250)
    assert reading.value == pytest.approx(
        pmo_top_ten_auc(scores, budget=250, finish=True)
    )


# ---- budget is inseparable from the number ----


def test_metric_id_carries_the_budget():
    assert top_ten_auc(_rising(250), budget=250, authorized_budget=250).metric_id == (
        "auc_top10@250"
    )
    assert top_ten_auc(_rising(900), budget=1000, authorized_budget=1000).metric_id == (
        "auc_top10@1000"
    )


def test_development_budgets_are_not_official_comparable():
    for budget in (250, 1000):
        reading = top_ten_auc(_rising(budget), budget=budget, authorized_budget=budget)
        assert reading.official_pmo_comparable is False
        assert "NOT comparable" in reading.comparability
        assert reading.to_dict()["official_pmo_budget"] == PMO_OFFICIAL_BUDGET


def test_full_official_budget_is_comparable():
    reading = top_ten_auc(
        _rising(PMO_OFFICIAL_BUDGET),
        budget=PMO_OFFICIAL_BUDGET,
        authorized_budget=PMO_OFFICIAL_BUDGET,
    )
    assert reading.official_pmo_comparable is True


def test_reading_a_short_run_at_a_longer_budget_is_refused():
    with pytest.raises(ValueError, match="frozen_tail_projection"):
        top_ten_auc(_rising(250), budget=PMO_OFFICIAL_BUDGET, authorized_budget=250)


def test_charged_calls_may_not_exceed_the_budget_read():
    with pytest.raises(ValueError):
        top_ten_auc(_rising(300), budget=250, authorized_budget=250)


def test_log_points_are_recorded_and_are_coarse_at_250():
    reading = top_ten_auc(_rising(250), budget=250, authorized_budget=250)
    assert reading.log_points == (100, 200, 250)
    full = top_ten_auc(
        _rising(PMO_OFFICIAL_BUDGET),
        budget=PMO_OFFICIAL_BUDGET,
        authorized_budget=PMO_OFFICIAL_BUDGET,
    )
    assert len(full.log_points) == 100


# ---- projections label themselves ----


def test_projection_is_never_official_comparable():
    projection = frozen_tail_projection(_rising(250), authorized_budget=250)
    assert projection.is_projection is True
    assert projection.budget == PMO_OFFICIAL_BUDGET
    assert projection.official_pmo_comparable is False
    assert "PROJECTION ONLY" in projection.comparability
    assert "lower bound" in (projection.assumption or "")


def test_constant_run_obeys_the_ramp_loss_closed_form():
    # top_auc trapezoids up from (0, 0), so a constant top-10 level c scores
    # c * (1 - frequency / (2 * budget)) exactly.  This is why a short-budget AUC
    # is DEPRESSED relative to a published one at identical performance.
    level = 0.4
    for budget, expected_loss in ((250, 0.20), (1000, 0.05), (10_000, 0.005)):
        flat = [level] * min(250, budget)
        reading = top_ten_auc(flat, budget=budget, authorized_budget=budget) if budget == 250 \
            else frozen_tail_projection(flat, authorized_budget=250, projected_budget=budget)
        assert reading.ramp_loss_fraction == pytest.approx(expected_loss)
        assert reading.value == pytest.approx(level * (1 - expected_loss))


def test_short_budget_reading_is_depressed_relative_to_the_projection():
    # A constant run scores LOWER at 250 than the same run flat-extended to 10k,
    # because the ramp is a larger fraction of a small budget.  Comparing a
    # 250-call AUC to a published 10k value understates performance.
    flat = [0.4] * 250
    assert top_ten_auc(flat, budget=250, authorized_budget=250).value < (
        frozen_tail_projection(flat, authorized_budget=250).value
    )

def test_projection_cannot_shorten_the_budget():
    with pytest.raises(ValueError):
        frozen_tail_projection(_rising(250), authorized_budget=1000, projected_budget=250)


# ---- suite aggregation ----


def _reading(n: int, budget: int):
    return top_ten_auc(_rising(n), budget=budget, authorized_budget=budget)


def test_suite_refuses_mixed_budgets():
    with pytest.raises(ValueError, match="mixed budgets"):
        aggregate_suite({"a": _reading(250, 250), "b": _reading(900, 1000)})


def test_suite_refuses_projections():
    # Same budget on both sides, so the refusal can only come from the projection.
    readings = {
        "a": _reading(PMO_OFFICIAL_BUDGET, PMO_OFFICIAL_BUDGET),
        "b": frozen_tail_projection(_rising(250), authorized_budget=250),
    }
    with pytest.raises(ValueError, match="projections"):
        aggregate_suite(readings)


def test_suite_reports_absent_baselines_and_stays_incomparable():
    readings = {"gsk3b": _reading(250, 250), "jnk3": _reading(250, 250)}
    agg = aggregate_suite(readings, baseline={"gsk3b": None, "jnk3": 0.825})
    assert agg.baseline_absent == ("gsk3b",)
    assert agg.baseline_covered == ("jnk3",)
    assert agg.official_pmo_comparable is False
    assert agg.sum_auc == pytest.approx(sum(agg.per_task.values()))
    assert agg.n_tasks == 2


def test_suite_with_full_budget_and_full_baseline_is_comparable():
    readings = {
        "a": _reading(PMO_OFFICIAL_BUDGET, PMO_OFFICIAL_BUDGET),
        "b": _reading(PMO_OFFICIAL_BUDGET, PMO_OFFICIAL_BUDGET),
    }
    agg = aggregate_suite(readings, baseline={"a": 0.5, "b": 0.6})
    assert agg.official_pmo_comparable is True
    assert agg.baseline_absent == ()


def test_suite_with_full_budget_but_absent_baseline_is_not_comparable():
    readings = {
        "a": _reading(PMO_OFFICIAL_BUDGET, PMO_OFFICIAL_BUDGET),
        "b": _reading(PMO_OFFICIAL_BUDGET, PMO_OFFICIAL_BUDGET),
    }
    agg = aggregate_suite(readings, baseline={"a": 0.5, "b": None})
    assert agg.official_pmo_comparable is False
    assert "partial baseline" in agg.comparability


def test_empty_suite_is_refused():
    with pytest.raises(ValueError):
        aggregate_suite({})
