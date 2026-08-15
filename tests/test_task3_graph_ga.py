"""The Graph-GA baseline. The failure mode here is silence, not an exception."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.baselines import task3_graph_ga as ga
from compose_v4.benchmark.init_sets import DEFAULT_INIT_DIR, development_init_set
from compose_v4.benchmark.oracles import DEFAULT_BUNDLE_DIR

pytestmark = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").exists()
    or not (DEFAULT_INIT_DIR / "development_init_sets.json").exists(),
    reason="frozen oracle bundle or init sets not present")


def test_the_size_prior_globals_are_actually_set(tmp_path):
    """Upstream `crossover.mol_OK` reads two module globals its driver sets.

    Unset, it raises NameError, a bare `except` swallows it, EVERY candidate is
    rejected, crossover always returns None and the run spins forever producing
    nothing -- with no error message anywhere. This is the check that the driver
    installed them.
    """
    from compose_v4.benchmark.task3_run import Task3Run

    co, _ = ga._upstream()
    for attribute in ("average_size", "size_stdev"):
        if hasattr(co, attribute):
            delattr(co, attribute)

    run = Task3Run.open(tmp_path / "run", seed=100, budget=400, policy="graph-ga")
    try:
        state = ga.initialize(run, development_init_set(100)[:60])
        assert co.average_size == pytest.approx(state.average_size)
        assert co.size_stdev == pytest.approx(state.size_stdev)
        assert state.average_size > 5, "a degenerate size prior rejects everything"
    finally:
        run.close()


def test_reproduction_actually_produces_offspring(tmp_path):
    """The silent-failure mode is an empty generation, so require a full one."""
    from compose_v4.benchmark.task3_run import Task3Run

    run = Task3Run.open(tmp_path / "run", seed=100, budget=400, policy="graph-ga")
    try:
        state = ga.initialize(run, development_init_set(100))
        offspring, failures = ga.reproduce(state.population, 20, ga.MUTATION_RATE,
                                           run.rng)
        assert len(offspring) == 20, (
            f"only {len(offspring)} offspring after {failures} failures; the "
            f"upstream operators are rejecting everything")
    finally:
        run.close()


def test_the_baseline_stops_at_the_budget_rather_than_overrunning(tmp_path):
    from compose_v4.benchmark.task3_run import Task3Run

    budget = 500
    run = Task3Run.open(tmp_path / "run", seed=101, budget=budget, policy="graph-ga")
    try:
        state = ga.initialize(run, development_init_set(101))
        for _ in range(40):
            if run.remaining == 0:
                break
            state = ga.step(run, state, offspring_size=40)
        assert run.spent <= budget
        assert run.spent >= budget - ga.POPULATION_SIZE, "the budget went unspent"
    finally:
        run.close()


def test_fitness_is_the_sum_of_the_five_transformed_objectives():
    """MOLLEO's own scalarization (pareto_optimizer.py:87-96), not a variant."""
    assert ga.fitness((0.1, 0.2, 0.3, 0.4, 0.5)) == pytest.approx(1.5)
    assert ga.fitness((1.0,) * 5) == pytest.approx(5.0)


def test_sanitize_dedupes_then_keeps_the_best():
    population = ["A", "B", "A", "C", "D"]
    scores = [0.1, 0.9, 5.0, 0.5, 0.2]
    kept, kept_scores = ga.sanitize(population, scores, 2)
    # The first "A" wins the dedupe, as upstream does, so its 0.1 is the score
    # that survives -- and B then outranks it.
    assert kept == ["B", "C"]
    assert kept_scores == [0.9, 0.5]


def test_roulette_selection_prefers_fitter_parents():
    rng = np.random.default_rng(0)
    pool = ga.mating_pool(["good", "bad"], [9.0, 1.0], 400, rng)
    assert 0.8 < pool.count("good") / len(pool) < 0.98


def test_roulette_survives_an_all_zero_population():
    """Fitness is a sum of non-negative objectives, so zero is reachable in
    principle and must not divide by zero."""
    pool = ga.mating_pool(["a", "b"], [0.0, 0.0], 10, np.random.default_rng(0))
    assert len(pool) == 10
