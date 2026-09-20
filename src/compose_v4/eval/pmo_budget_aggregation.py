"""Budget-explicit PMO top-10 AUC aggregation and score-vs-calls logging.

The PMO leaderboard metric is ``top_auc(finish=True)`` evaluated at a budget of
10,000 oracle calls with a logging frequency of 100.  COMPOSE development runs
are far smaller (250 / 1,000 calls), and a top-10 AUC computed at one budget is
NOT comparable to one computed at another: the metric divides by its budget and
flat-extends the last logged value out to that budget, so a short run padded to
10,000 silently asserts "no further improvement for the remaining 9,750 calls".

That confusion has a history in this project, so every number produced here
carries its budget in its own identifier and an explicit comparability verdict.
The invariants maintained and tested:

* ``metric_id`` always embeds the budget (``auc_top10@250``), so a bare float can
  never be lifted out of one artifact and dropped into another.
* ``official_pmo_comparable`` is True only when the run was *authorized and
  charged* at the official budget with the official logging frequency.  A
  projection is never comparable, whatever its arithmetic.
* ``top_ten_auc`` refuses a budget that differs from the authorized budget.
  Reading a short run at a longer budget requires the separate, self-labelling
  ``frozen_tail_projection``.

The AUC itself is delegated to ``pmo_top_ten_auc`` (the audited transcription of
mol_opt's ``top_auc``); nothing here reimplements the trapezoid rule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from compose_v4.control.program_task import pmo_top_ten_auc

# ---- Official PMO constants ----

PMO_OFFICIAL_BUDGET = 10_000
PMO_OFFICIAL_LOG_FREQUENCY = 100
PMO_OFFICIAL_METRIC = "top_auc(finish=True)"


# ---- Score-vs-calls logging ----


def score_calls_log(scores: Iterable[float]) -> list[dict]:
    """Canonical score-vs-calls curve, one row per charged oracle call.

    Field names match ``pmo_dynamic_v21._score_curve`` so downstream readers do
    not have to branch on which driver produced the run.  Unlike that helper this
    takes bare scores, because manifests and replays do not carry ledger rows.
    """

    values: list[float] = []
    curve: list[dict] = []
    for index, raw in enumerate(scores, start=1):
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("PMO score curve received an invalid reward")
        values.append(value)
        top = sorted(values, reverse=True)[:10]
        curve.append({
            "charged_queries": index,
            "best_score": max(values),
            "top10_score": sum(top) / len(top),
        })
    return curve


# ---- Budget-explicit AUC ----


@dataclass(frozen=True)
class TopTenAuc:
    """One top-10 AUC reading, inseparable from the budget it was read at."""

    value: float
    budget: int
    authorized_budget: int
    charged_calls: int
    log_frequency: int
    log_points: tuple[int, ...]
    is_projection: bool
    official_pmo_comparable: bool
    comparability: str
    assumption: str | None = None

    @property
    def ramp_loss_fraction(self) -> float:
        """Fraction of a constant top-10 level that the (0, 0) ramp removes."""

        return self.log_frequency / (2 * self.budget)

    @property
    def metric_id(self) -> str:
        return f"auc_top10@{self.budget}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric_id": self.metric_id,
            "metric_definition": PMO_OFFICIAL_METRIC,
            "value": self.value,
            "budget": self.budget,
            "authorized_budget": self.authorized_budget,
            "charged_calls": self.charged_calls,
            "log_frequency": self.log_frequency,
            "log_points": list(self.log_points),
            "n_log_points": len(self.log_points),
            "ramp_loss_fraction": self.ramp_loss_fraction,
            "ramp_loss_note": (
                "top_auc trapezoids up from (0, 0), so a constant-top10 run scores "
                "level * (1 - frequency / (2 * budget)). At budget "
                f"{self.budget} that removes {self.ramp_loss_fraction:.1%} of the level; "
                f"at the official {PMO_OFFICIAL_BUDGET}-call budget it removes "
                f"{PMO_OFFICIAL_LOG_FREQUENCY / (2 * PMO_OFFICIAL_BUDGET):.1%}. A short-budget "
                "AUC is therefore depressed relative to a published one at identical "
                "performance."
            ),
            "is_projection": self.is_projection,
            "official_pmo_budget": PMO_OFFICIAL_BUDGET,
            "official_pmo_comparable": self.official_pmo_comparable,
            "comparability": self.comparability,
        }
        if self.assumption is not None:
            payload["assumption"] = self.assumption
        return payload


def _log_points(n_charged: int, frequency: int) -> tuple[int, ...]:
    """The counts at which pmo_top_ten_auc evaluates the top-10 mean."""

    if n_charged < 1:
        return ()
    return (*range(frequency, n_charged, frequency), n_charged)


def _validate(scores: Sequence[float], budget: int, frequency: int) -> None:
    if type(budget) is not int or budget < 1:
        raise ValueError("positive integer PMO budget required")
    if type(frequency) is not int or frequency < 1:
        raise ValueError("positive integer logging frequency required")
    if len(scores) > budget:
        raise ValueError(
            f"{len(scores)} charged calls exceed the {budget}-call budget being read"
        )


def top_ten_auc(
    scores: Sequence[float],
    *,
    budget: int,
    authorized_budget: int,
    frequency: int = PMO_OFFICIAL_LOG_FREQUENCY,
) -> TopTenAuc:
    """Top-10 AUC of a run read at the budget it was authorized to spend.

    Refuses ``budget != authorized_budget``: reading a 250-call run as if it had
    a 10,000-call budget is a projection and has to say so, via
    ``frozen_tail_projection``.
    """

    values = [float(v) for v in scores]
    _validate(values, budget, frequency)
    if budget != authorized_budget:
        raise ValueError(
            f"refusing to read a {authorized_budget}-call run at budget {budget}; "
            "use frozen_tail_projection, which labels itself as a projection"
        )
    area = pmo_top_ten_auc(values, budget=budget, finish=True)
    if area is None:
        raise ValueError("no charged calls: top-10 AUC is undefined")
    comparable = (
        budget == PMO_OFFICIAL_BUDGET
        and authorized_budget == PMO_OFFICIAL_BUDGET
        and frequency == PMO_OFFICIAL_LOG_FREQUENCY
        and len(values) == PMO_OFFICIAL_BUDGET
    )
    if comparable:
        note = (
            "Comparable to published PMO tables: official budget, official logging "
            "frequency, and the full budget was charged."
        )
    else:
        note = (
            f"NOT comparable to published PMO tables. This is a {budget}-call "
            f"development reading; published values are {PMO_OFFICIAL_BUDGET}-call "
            f"{PMO_OFFICIAL_METRIC}. Do not place this value in a column headed by a "
            "published baseline."
        )
    return TopTenAuc(
        value=float(area),
        budget=budget,
        authorized_budget=authorized_budget,
        charged_calls=len(values),
        log_frequency=frequency,
        log_points=_log_points(len(values), frequency),
        is_projection=False,
        official_pmo_comparable=comparable,
        comparability=note,
    )


def frozen_tail_projection(
    scores: Sequence[float],
    *,
    authorized_budget: int,
    projected_budget: int = PMO_OFFICIAL_BUDGET,
    frequency: int = PMO_OFFICIAL_LOG_FREQUENCY,
) -> TopTenAuc:
    """Read a short run at a longer budget, flat-extending its final top-10 mean.

    This is an explicitly-labelled LOWER BOUND under a no-further-improvement
    assumption, never a leaderboard number.  ``official_pmo_comparable`` is
    always False, including when ``projected_budget`` is 10,000.
    """

    values = [float(v) for v in scores]
    _validate(values, projected_budget, frequency)
    if projected_budget < authorized_budget:
        raise ValueError("projected budget cannot be shorter than the authorized budget")
    area = pmo_top_ten_auc(values, budget=projected_budget, finish=True)
    if area is None:
        raise ValueError("no charged calls: top-10 AUC is undefined")
    return TopTenAuc(
        value=float(area),
        budget=projected_budget,
        authorized_budget=authorized_budget,
        charged_calls=len(values),
        log_frequency=frequency,
        log_points=_log_points(len(values), frequency),
        is_projection=True,
        official_pmo_comparable=False,
        comparability=(
            f"PROJECTION ONLY. A {authorized_budget}-call run flat-extended to "
            f"{projected_budget} calls. Not a PMO leaderboard value and not comparable "
            "to a published baseline."
        ),
        assumption=(
            f"Assumes the top-10 mean never improves over the remaining "
            f"{projected_budget - len(values)} calls, so it is a lower bound on the "
            "AUC a full-budget run would score."
        ),
    )


# ---- Suite aggregation ----


@dataclass(frozen=True)
class SuiteAggregate:
    """Sum/mean of per-task AUCs at one budget, plus baseline coverage."""

    budget: int
    per_task: dict[str, float] = field(default_factory=dict)
    sum_auc: float = 0.0
    mean_auc: float = 0.0
    n_tasks: int = 0
    official_pmo_comparable: bool = False
    comparability: str = ""
    baseline_covered: tuple[str, ...] = ()
    baseline_absent: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": f"sum_auc_top10@{self.budget}",
            "budget": self.budget,
            "n_tasks": self.n_tasks,
            "per_task": dict(self.per_task),
            "sum_auc": self.sum_auc,
            "mean_auc": self.mean_auc,
            "official_pmo_comparable": self.official_pmo_comparable,
            "comparability": self.comparability,
            "baseline_covered": list(self.baseline_covered),
            "baseline_absent": list(self.baseline_absent),
        }


def aggregate_suite(
    readings: dict[str, TopTenAuc],
    *,
    baseline: dict[str, float | None] | None = None,
) -> SuiteAggregate:
    """Aggregate per-task readings, refusing to mix budgets in one sum.

    A sum over tasks read at different budgets is meaningless, so it is rejected
    rather than silently produced.  Tasks whose baseline is ABSENT are reported
    separately: a suite sum compared against a partial baseline sum is the exact
    shape of error that inverts a headline.
    """

    if not readings:
        raise ValueError("no per-task AUC readings to aggregate")
    if any(r.is_projection for r in readings.values()):
        raise ValueError("refusing to aggregate projections into a suite total")
    budgets = {r.budget for r in readings.values()}
    if len(budgets) != 1:
        raise ValueError(f"refusing to aggregate across mixed budgets: {sorted(budgets)}")
    budget = budgets.pop()
    per_task = {task: r.value for task, r in sorted(readings.items())}
    total = float(sum(per_task.values()))
    comparable = all(r.official_pmo_comparable for r in readings.values())
    baseline = baseline or {}
    covered = tuple(sorted(t for t in per_task if baseline.get(t) is not None))
    absent = tuple(sorted(t for t in per_task if baseline.get(t) is None))
    if comparable and not absent:
        note = "Suite total comparable to a published full-budget baseline sum."
    elif comparable:
        note = (
            "Per-task values are full-budget, but "
            f"{len(absent)} task(s) have no published baseline: a suite-sum comparison "
            "would compare against a partial baseline."
        )
    else:
        note = (
            f"NOT comparable to a published PMO sum: {budget}-call development "
            "readings. Published sums are 10,000-call."
        )
    return SuiteAggregate(
        budget=budget,
        per_task=per_task,
        sum_auc=total,
        mean_auc=total / len(per_task),
        n_tasks=len(per_task),
        official_pmo_comparable=comparable and not absent,
        comparability=note,
        baseline_covered=covered,
        baseline_absent=absent,
    )
