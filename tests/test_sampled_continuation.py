"""Bounded, analytic fixtures, not learned-model or docking performance tests."""

from dataclasses import asdict

import numpy as np
import pytest

from compose_v4.control.continuation import ReferenceRow
from compose_v4.control.option_continuation import sample_option_trajectory
from compose_v4.control.sampled_continuation import (
    SampledContinuation,
    sampled_continuation_decision,
)
from compose_v4.experiments.continuation_gate import engineering_fixture, ring_terminal_weight


def estimator(reference=None, terminal=None, **kwargs):
    options = {
        "snapshot_id": "synthetic-v1",
        "seed": 17,
        "samples_per_successor": 32,
        "alpha": 0.05,
        "max_expansions": 10000,
        "max_terminal_evaluations": 10000,
        "max_rollouts": 10000,
    }
    options.update(kwargs)
    return SampledContinuation(
        reference or (lambda s: ReferenceRow((s,), (1.0,))),
        terminal or (lambda s: float(s)),
        lambda s: s,
        **options,
    )


def decide(value, row=None, remaining=2):
    row = row or ReferenceRow((0, 1), (0.5, 0.5))
    return sampled_continuation_decision(
        row, remaining, value, fallback_values=[1] * len(row.successors)
    )


def test_complete_balanced_samples_guide_with_frozen_kl_and_support():
    value = estimator()
    result = decide(value, ReferenceRow((0, 1, 99), (0.9, 0.1, 0)))
    assert result.decision.status == "guided_sampled"
    assert result.estimate.means == (0, 1, 0)
    assert result.estimate.samples == (32, 32, 0)
    assert result.decision.probabilities[2] == 0
    assert result.decision.probabilities[1] > 0.1
    assert result.decision.kl <= 1 + 1e-10
    assert sum(result.decision.probabilities) == pytest.approx(1)
    assert all(q >= 0.1 * p for q, p in zip(result.decision.probabilities, (0.9, 0.1, 0)))
    assert value.work.rollouts_started == value.work.rollouts_completed == 64
    assert value.work.row_visits == 64
    assert value.work.expansions == value.work.terminal_evaluations == 2


def test_incomplete_comparison_discards_partial_estimates_and_uses_nonuniform_baseline():
    value = estimator(max_terminal_evaluations=1)
    row = ReferenceRow((0, 1), (0.7, 0.3))
    result = decide(value, row)
    assert result.decision.status == "budget_abstention"
    assert result.estimate is None and result.decision.successor_values is None
    assert result.decision.probabilities == pytest.approx(row.probabilities)
    assert value.work.rollouts_started == 2
    assert value.work.rollouts_completed == 1
    assert value.work.budget_abstentions == 1


def test_predeclared_rollout_ceiling_abstains_before_spending_partial_panel():
    value = estimator(max_rollouts=63)
    assert decide(value).decision.status == "budget_abstention"
    assert value.work.rollouts_started == value.work.expansions == 0


@pytest.mark.parametrize(
    "reward,status", [(0, "no_observed_success"), (0.5, "insufficient_evidence")]
)
def test_sampled_null_is_not_exact_unreachability(reward, status):
    result = decide(estimator(terminal=lambda s: reward))
    assert result.decision.status == status
    assert result.decision.probabilities == (0.5, 0.5)
    assert result.estimate is not None
    if reward == 0:
        assert min(result.estimate.upper) > 0


def test_empty_and_single_support_spend_no_lookahead():
    value = estimator()
    assert decide(value, ReferenceRow((), ())).decision.status == "no_admissible_action"
    assert decide(value, ReferenceRow((0, 1), (1.0, 0.0))).decision.status == "only_one_successor"
    assert value.work.rollouts_started == 0


def test_empty_future_support_counts_as_unsuccessful_rollout_not_a_terminal_molecule():
    value = estimator(reference=lambda s: ReferenceRow((), ()), terminal=lambda s: float("nan"))
    result = decide(value)
    assert result.decision.status == "no_observed_success"
    assert value.work.dead_ends == value.work.rollouts_completed == 64
    assert value.work.terminal_evaluations == 0


def test_exact_terminal_needs_no_rollouts_or_statistical_bound():
    value = estimator(max_rollouts=0)
    result = decide(value, remaining=1)
    assert result.decision.status == "guided_terminal"
    assert result.estimate.lower == result.estimate.upper == (0, 1)
    assert value.work.rollouts_started == 0
    assert value.work.terminal_evaluations == 2


def test_alias_refinement_shares_samples_and_preserves_aggregate_control():
    original, refined = (
        decide(estimator()),
        decide(estimator(), ReferenceRow((0, 0, 1), (0.2, 0.3, 0.5))),
    )
    assert sum(refined.decision.probabilities[:2]) == pytest.approx(
        original.decision.probabilities[0]
    )
    assert refined.decision.probabilities[2] == pytest.approx(original.decision.probabilities[1])
    assert refined.estimate.lower[0] == refined.estimate.lower[1]


def test_cached_uncached_sampling_laws_match_exact_matrix_expectation():
    matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    reference = lambda s: ReferenceRow((0, 1), tuple(matrix[s]))
    root = ReferenceRow((0, 1), (0.5, 0.5))
    expected = np.linalg.matrix_power(matrix, 3) @ np.array([0, 1])
    results = []
    for memoize in (True, False):
        value = estimator(
            reference=reference, samples_per_successor=2048, memoize=memoize, max_expansions=13000
        )
        estimate = value.estimate(root, 4)
        assert np.all(np.asarray(estimate.lower) <= expected)
        assert np.all(np.asarray(estimate.upper) >= expected)
        assert estimate.means == pytest.approx(expected, abs=0.04)
        results.append(estimate)
    assert results[0] == results[1]


def test_simultaneous_bounds_cover_fixed_analytic_reference_panel():
    # Regression against an analytic expectation, not empirical calibration of
    # a docking model. Inputs, seeds, N and tolerance are fixed here.
    expected = np.array([0.25, 0.75])
    reference = lambda s: ReferenceRow((0, 1), (1 - expected[s], expected[s]))
    covered = 0
    for seed in range(128):
        estimate = estimator(reference=reference, seed=seed).estimate(
            ReferenceRow((0, 1), (0.5, 0.5)), 2
        )
        covered += bool(
            np.all(np.asarray(estimate.lower) <= expected)
            and np.all(np.asarray(estimate.upper) >= expected)
        )
    assert covered >= 122  # >=95% on this fixed engineering panel, not a universal claim


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples_per_successor": 0},
        {"samples_per_successor": True},
        {"alpha": 0},
        {"alpha": float("nan")},
        {"seed": -1},
        {"max_expansions": -1},
        {"max_rollouts": 1.5},
    ],
)
def test_malformed_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        estimator(**kwargs)


@pytest.mark.parametrize("weight", [-1, 2, float("nan")])
def test_malformed_terminal_weight_is_not_suppressed(weight):
    with pytest.raises(ValueError, match="terminal weight"):
        decide(estimator(terminal=lambda s: weight))


def test_executor_trajectory_has_separate_reproducible_randomness_and_full_program():
    runs = []
    for _ in range(2):
        node, kernel = engineering_fixture()
        result = sample_option_trajectory(
            node,
            kernel,
            ring_terminal_weight,
            lambda s: 1,
            snapshot_id="synthetic-ring-v1",
            seed=0,
            max_expansions=64,
            max_terminal_evaluations=64,
            estimator="sampled",
            max_rollouts=64,
        )
        result.pop("seconds")
        runs.append(result)
    assert runs[0] == runs[1]
    assert result["status"] == "complete"
    assert len(result["trace"]) == 2
    assert result["trace"][0]["decision_status"] == "guided_sampled"
    assert result["randomness"]["planner_seed"] != result["randomness"]["seed"]
    assert result["conditional_path_logq"] == pytest.approx(
        sum(np.log(s["probability"]) for s in result["trace"])
    )
    assert result["kernel_work"]["executor_applications"] == 4
    assert all(s["kl"] <= 1 for s in result["trace"])


def test_unknown_estimator_is_rejected_before_molecular_work():
    node, kernel = engineering_fixture()
    with pytest.raises(ValueError, match="unknown continuation estimator"):
        sample_option_trajectory(
            node,
            kernel,
            ring_terminal_weight,
            lambda s: 1,
            snapshot_id="synthetic",
            seed=0,
            max_expansions=2,
            max_terminal_evaluations=2,
            estimator="silent-new-law",
        )
    assert asdict(kernel.work)["executor_applications"] == 0


def test_one_lucky_sample_is_not_confident_guidance():
    result = decide(estimator(samples_per_successor=1))
    assert result.estimate.means == (0.0, 1.0)
    assert result.decision.status == "insufficient_evidence"
    assert result.decision.probabilities == (0.5, 0.5)


def test_sampled_controller_rejects_kappa_changes_and_invalid_horizons():
    row, value = ReferenceRow((0, 1), (0.5, 0.5)), estimator()
    with pytest.raises(ValueError, match="freezes kappa"):
        sampled_continuation_decision(row, 2, value, fallback_values=(1, 1), kappa=2)
    for horizon in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            decide(value, remaining=horizon)


def test_engineering_comparison_reports_cost_without_claiming_molecular_speedup():
    from compose_v4.experiments.sampled_continuation_gate import evaluate

    report = evaluate()
    assert report["passed"]
    assert report["chemistry"]["sampled"]["kernel_work"]["executor_applications"] == 4
    assert report["tree"]["sampled"]["work"]["row_visits"] == 320
