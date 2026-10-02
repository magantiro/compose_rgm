"""The value head and evaluator must agree on the fixed goal-grid order."""

from compose_v4.experiments.qed_goal_regions import BENCHMARK_REGION, registered_regions


def test_goal_grid_contains_benchmark_region_at_stable_index() -> None:
    regions = registered_regions()
    assert len(regions) == 20
    assert len(set(regions)) == len(regions)
    assert regions.index(BENCHMARK_REGION) == 13
