"""Independent path sums and failure semantics for bounded continuation."""

from dataclasses import asdict
from itertools import product

import numpy as np
import pytest

from compose_v4.control.continuation import (
    FiniteHorizonContinuation,
    ReferenceRow,
    continuation_decision,
    exact_doob_distribution,
)


def engine(matrix, reward, *, memoize=True, expansions=1000, terminals=1000):
    return FiniteHorizonContinuation(
        lambda s: ReferenceRow(tuple(range(len(matrix))), tuple(matrix[s])),
        lambda s: reward[s],
        lambda s: s,
        snapshot_id="explicit-finite-matrix-fixture-v1",
        max_expansions=expansions,
        max_terminal_evaluations=terminals,
        memoize=memoize,
    )


def test_backup_matches_independent_complete_path_sum_and_terminal_tilt():
    matrix = np.array([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5], [0.4, 0.5, 0.1]])
    rewards = [0.05, 0.4, 1.0]
    value = engine(matrix, rewards)
    for horizon in range(1, 5):
        target = np.zeros(3)
        controlled = np.zeros(3)
        for path in product(range(3), repeat=horizon):
            src, probability, guided = 0, 1.0, 1.0
            for step, dst in enumerate(path):
                row = ReferenceRow((0, 1, 2), tuple(matrix[src]))
                h = value.successor_values(row, horizon - step)
                q = exact_doob_distribution(row, h)
                guided *= q[dst]
                probability *= matrix[src, dst]
                src = dst
            target[path[-1]] += probability * rewards[path[-1]]
            controlled[path[-1]] += guided
        assert value.value(0, horizon) == pytest.approx(target.sum(), abs=1e-12)
        np.testing.assert_allclose(controlled, target / target.sum(), atol=1e-12)


def test_future_value_crosses_plateau_and_preserves_reference_support():
    matrix = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    value = engine(matrix, [0, 0, 0, 1])
    row = ReferenceRow((0, 1, 2, 3), tuple(matrix[0]))
    immediate = continuation_decision(row, 1, value, fallback_values=[1] * 4)
    future = continuation_decision(row, 2, value, fallback_values=[1] * 4)
    assert immediate.status == "zero_terminal_mass"
    assert immediate.probabilities[1] == pytest.approx(0.5)
    assert future.probabilities[1] == pytest.approx(0.95)
    assert future.probabilities[0] == future.probabilities[3] == 0
    assert future.kl <= 1
    assert sum(future.probabilities) == pytest.approx(1)


def test_cache_matches_uncached_and_reduces_computation():
    matrix = np.array([[0.3, 0.7], [0.4, 0.6]])
    cached, fresh = engine(matrix, [0.2, 1]), engine(matrix, [0.2, 1], memoize=False)
    assert cached.value(0, 5) == pytest.approx(fresh.value(0, 5), abs=1e-12)
    assert cached.work.expansions < fresh.work.expansions
    assert cached.work.terminal_evaluations < fresh.work.terminal_evaluations
    assert cached.work.value_cache_hits > 0


def test_budget_abstention_discards_partial_ranking_and_counts_work():
    matrix = np.array([[0.2, 0.8], [0.7, 0.3]])
    row = ReferenceRow((0, 1), (0.2, 0.8))
    short = engine(matrix, [0, 1], terminals=1)
    decision = continuation_decision(row, 1, short, fallback_values=[1, 1])
    assert decision.status == "budget_abstention"
    assert decision.successor_values is None
    assert decision.probabilities == pytest.approx((0.2, 0.8))
    assert short.work.terminal_evaluations == 1
    assert short.work.budget_abstentions == 1
    assert asdict(short.work)["expansions"] == 0


def test_zero_mass_is_not_an_epsilon_patched_exact_transform():
    assert exact_doob_distribution(ReferenceRow((0, 1), (0.5, 0.5)), (0, 0)) is None
    value = engine(np.eye(2), [0, 0])
    assert (
        continuation_decision(
            ReferenceRow((0, 1), (0.5, 0.5)), 2, value, fallback_values=[1, 1]
        ).status
        == "zero_terminal_mass"
    )


def test_no_support_is_explicit_not_absorbing_success():
    value = FiniteHorizonContinuation(
        lambda s: ReferenceRow((), ()),
        lambda s: 1,
        lambda s: s,
        snapshot_id="empty-support-fixture-v1",
        max_expansions=10,
        max_terminal_evaluations=10,
    )
    assert value.value(0, 1) == 0
    assert value.value(0, 0) == 1
    result = continuation_decision(ReferenceRow((), ()), 1, value, fallback_values=[])
    assert result.status == "no_admissible_action"


@pytest.mark.parametrize("probabilities", [(0.3, 0.3), (-0.1, 1.1), (float("nan"), 1)])
def test_bad_reference_rows_fail(probabilities):
    with pytest.raises(ValueError):
        ReferenceRow((0, 1), probabilities)


@pytest.mark.parametrize("budget", [-1, 1.2, True])
def test_invalid_horizon_fails(budget):
    with pytest.raises(ValueError):
        engine(np.eye(2), [0, 1]).value(0, budget)


@pytest.mark.parametrize("reward", [-1, 1.2, float("nan")])
def test_invalid_terminal_weight_fails(reward):
    with pytest.raises(ValueError, match="terminal weight"):
        engine(np.eye(2), [reward, 1]).value(0, 0)


def test_zero_reference_branch_does_not_consume_terminal_budget():
    value = engine(np.eye(2), [1, float("nan")], terminals=1)
    result = continuation_decision(
        ReferenceRow((0, 1), (1.0, 0.0)), 1, value, fallback_values=[1, 1]
    )
    assert result.probabilities == (1.0, 0.0)
    assert value.work.terminal_evaluations == 1


def test_frozen_kappa_rejects_changes():
    with pytest.raises(ValueError, match="freezes kappa"):
        continuation_decision(
            ReferenceRow((0, 1), (0.5, 0.5)),
            1,
            engine(np.eye(2), [0, 1]),
            fallback_values=[1, 1],
            kappa=2.0,
        )


def test_refining_an_identical_successor_alias_preserves_aggregate_control():
    value = engine(np.eye(2), [0.1, 1.0])
    original = continuation_decision(
        ReferenceRow((0, 1), (0.99, 0.01)), 1, value, fallback_values=[1, 1]
    )
    refined = continuation_decision(
        ReferenceRow((0, 0, 1), (0.33, 0.66, 0.01)),
        1,
        value,
        fallback_values=[1, 1, 1],
    )
    assert sum(refined.probabilities[:2]) == pytest.approx(original.probabilities[0], abs=1e-10)
    assert refined.probabilities[2] == pytest.approx(original.probabilities[1], abs=1e-10)
    assert refined.kl == pytest.approx(original.kl, abs=1e-10)
