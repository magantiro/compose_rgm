"""Small delayed-credit, probability-law and real-executor regressions."""

from dataclasses import replace

import numpy as np
import pytest

from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.task_search import ReturnEstimate, SearchRow, TaskSearch


def binary_row(state):
    return SearchRow((state + (0,), state + (1,)), ("left", "right"), (0.5, 0.5), (0.5, 0.5), 0.1)


def binary_planner(target, **kwargs):
    return TaskSearch(
        binary_row,
        lambda s: (1.0 if s == target else 0.05) if len(s) == 3 else None,
        lambda s: s,
        snapshot_id="synthetic-delayed-credit",
        seed=11,
        max_rollouts=1000,
        **kwargs,
    )


def test_one_terminal_objective_steers_all_three_levels_and_reverses():
    for preferred in (0, 1):
        planner = binary_planner((preferred,) * 3)
        assert planner.decision(())["probabilities"] == pytest.approx([0.5, 0.5])
        assert planner.plan((), 1000) == 1000
        for stage in range(3):  # synthetic WHERE, WHAT, HOW
            state = (preferred,) * stage
            decision = planner.decision(state)
            assert decision["probabilities"][preferred] > 0.65
            assert decision["kl"] <= 1 + 1e-10
            assert min(decision["probabilities"]) >= 0.05
        assert planner.work.rows == 7  # reuse the same finite tree


def test_importance_correction_recovers_reference_returns_not_visit_counts():
    estimate = ReturnEstimate()
    # Proposal samples useful return 9x as often; reference probability is 1/2.
    for _ in range(90):
        estimate.add(1, np.log(0.5 / 0.9))
    for _ in range(10):
        estimate.add(0, np.log(0.5 / 0.1))
    assert estimate.mean(0) == pytest.approx(0.5)
    assert estimate.count == 100
    huge = ReturnEstimate()
    huge.add(0.4, 10000)
    assert huge.mean(0) == pytest.approx(0.4)


def test_budget_interruption_does_not_publish_partial_returns():
    planner = binary_planner((0, 0, 0), max_rows=1)
    assert planner.plan((), 1) == 0
    assert planner.work.rollouts_interrupted == 1
    assert not planner.returns
    assert planner.decision(())["probabilities"] == pytest.approx([0.5, 0.5])


def test_completed_evidence_survives_later_interruption():
    planner = binary_planner((0, 0, 0))
    planner.plan((), 1)
    old = {key: value.payload() for key, value in planner.returns.items()}
    planner.max_terminals = 0
    planner.terminals.clear()
    assert not planner.rollout(())
    assert old == {key: value.payload() for key, value in planner.returns.items()}


def test_scale_floor_is_preserved_and_zero_support_not_created():
    row = SearchRow(
        (0, 1, 2), ("local", "global", "unsupported"), (0.999, 0.001, 0), (0.5, 0.5, 0), 0.2
    )
    planner = TaskSearch(lambda _: row, lambda _: None, lambda s: s, snapshot_id="floor", seed=0)
    planner.terminals.update({0: 0.0, 1: 1.0, 2: 1.0})
    decision = planner.decision("root")
    assert decision["probabilities"][2] == 0
    assert all(p >= 0.1 for p in decision["probabilities"][:2])
    assert decision["kl"] <= 1 + 1e-10


def test_production_options_cross_boundaries_without_teleportation():
    from test_option_continuation import fixture_law, source

    from compose_v4.chem.state import is_valid_state
    from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
    from compose_v4.control.option_continuation import OptionContinuationKernel
    from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system

    graph = source().graph
    root = MolecularSearchState.start(graph, budget=2, root_id="fixture-only")
    kernel = OptionContinuationKernel(
        fixture_law, editing_v2_rewrite_system(), max_executor_applications=200
    )
    hierarchy = MolecularHierarchy(kernel, generic_horizon=1, ring_options=())
    region_row = hierarchy.row(root)
    selected = max(region_row.successors, key=lambda s: s.region.size)
    option_row = hierarchy.row(selected)
    grow = option_row.successors[option_row.labels.index("grow")]
    primitive = hierarchy.row(grow)
    carbon = next(s for s in primitive.successors if canonical_state_key(s.graph) == "CCCCC")
    assert carbon.stage == "where" and carbon.budget == 1  # returns to WHERE, not terminal
    assert is_valid_state(carbon.graph)
    next_regions = hierarchy.row(carbon)
    next_region = max(next_regions.successors, key=lambda s: s.region.size)
    next_options = hierarchy.row(next_region)
    close = next_options.successors[next_options.labels.index("cyclize")]
    endpoint = hierarchy.row(close).successors[0]
    assert endpoint.stage == "where" and endpoint.budget == 0
    assert canonical_state_key(endpoint.graph) == "C1CCCC1"
    assert is_valid_state(endpoint.graph)
    assert root.key() != replace(root, budget=1).key()
    before = kernel.work.executor_applications
    hierarchy.row(close)
    assert kernel.work.executor_applications == before


def test_bad_rows_and_empty_support_are_explicit():
    with pytest.raises(ValueError, match="sum to one"):
        SearchRow((1,), ("a",), (0.2,), (1.0,), 0.1)
    planner = TaskSearch(
        lambda _: SearchRow((), (), (), (), 0.1),
        lambda _: None,
        lambda s: s,
        snapshot_id="empty",
        seed=0,
    )
    assert planner.decision(0)["status"] == "no_admissible_action"
    assert planner.rollout(0)
    assert planner.work.dead_ends == 1
    assert planner.returns[0].mean(1) == 0
    with pytest.raises(ContinuationBudgetExceeded):
        binary_planner((0, 0, 0), max_rows=0).decision(())


def test_physical_cache_shares_work_without_sharing_option_laws():
    from test_option_continuation import kernel, source

    shared = kernel()
    grow, backbone = source("grow", 1), source("scaffold_extend", 1)
    first = shared.row(grow)
    before = shared.work.executor_applications
    second = shared.row(backbone)
    fresh = kernel().row(backbone)
    assert before == 2 and shared.work.executor_applications == before
    assert len(first.successors) == 2 and len(second.successors) == 1
    assert second.probabilities == fresh.probabilities
    assert tuple(s.key() for s in second.successors) == tuple(s.key() for s in fresh.successors)


def test_region_distribution_refactor_preserves_sampled_row_and_rng():
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control.region_selector import region_distribution, sample_region

    regions = enumerate_regions("CCc1ccccc1")
    options, scores, probabilities = region_distribution(regions)
    rng1, rng2 = np.random.default_rng(51), np.random.default_rng(51)
    chosen, score = sample_region(regions, rng1)
    expected = int(rng2.choice(len(options), p=probabilities))
    assert chosen == options[expected] and score == scores[expected]
    assert rng1.bit_generator.state == rng2.bit_generator.state
