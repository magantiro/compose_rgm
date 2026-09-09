"""Empirical regressions for useful option completions before the last edit."""

import pytest

from compose_v4.control.task_search import SearchRow, TaskSearch
from compose_v4.experiments.t4_task_search import PreparationConfig


def chain_row(state):
    if state == ():
        return SearchRow(((0,), (1,)), ("a", "b"), (0.5, 0.5), (0.5, 0.5), 0.1)
    return SearchRow((state + (0,),), ("continue",), (1.0,), (1.0,), 0.1)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("preferred", [0, 1])
def test_useful_early_completion_is_not_erased_by_bad_final_edit(lazy, preferred):
    def draw(state, rng):
        row = chain_row(state)
        return row.successors[int(rng.choice(len(row.successors), p=row.reference))]

    common = {
        "snapshot_id": "synthetic-endpoint-credit",
        "seed": 9,
        "max_rollouts": 200,
        "reference_draw": draw if lazy else None,
    }
    endpoint = lambda s: (0.9 if s[0] == preferred else 0.1) if len(s) == 1 else None
    terminal = lambda s: 0.0 if len(s) == 3 else None
    old = TaskSearch(chain_row, terminal, lambda s: s, **common)
    new = TaskSearch(chain_row, terminal, lambda s: s, endpoint=endpoint, **common)
    old.plan((), 200)
    new.plan((), 200)
    assert old.decision(())["probabilities"] == pytest.approx([0.5, 0.5])
    decision = new.decision(())
    assert decision["probabilities"][preferred] > 0.8
    assert decision["kl"] <= 1 + 1e-10
    assert min(decision["probabilities"]) >= 0.05
    # A reward already left behind must not leak into the later state's suffix.
    assert new.returns[(preferred, 0)].mean(1) == 0
    assert new.receipt()["positive_endpoint_evaluations"] == 2


def test_later_improvement_survives_low_value_completion_and_partial_returns_do_not():
    planner = TaskSearch(
        lambda s: SearchRow((s + 1,), ("continue",), (1.0,), (1.0,), 0.1),
        lambda s: 0.0 if s == 4 else None,
        lambda s: s,
        snapshot_id="synthetic-low-value-valley",
        seed=0,
        endpoint=lambda s: {1: 0.2, 2: 0.0, 3: 0.8}.get(s),
    )
    assert planner.rollout(0)
    assert planner.returns[0].mean(0) == pytest.approx(0.8)
    assert planner.returns[2].mean(0) == pytest.approx(0.8)
    assert planner.returns[4].mean(1) == 0
    interrupted = TaskSearch(
        planner.reference,
        planner.terminal,
        lambda s: s,
        snapshot_id="synthetic-interruption",
        seed=0,
        max_path_steps=2,
        endpoint=planner.endpoint,
    )
    assert not interrupted.rollout(0)
    assert interrupted.endpoints[1] == 0.2
    assert not interrupted.returns
    assert interrupted.work.rollouts_interrupted == 1


def test_policy_requires_pathwise_split_and_preserves_legacy_configuration():
    assert "return_policy" not in PreparationConfig().payload()
    with pytest.raises(ValueError, match="separate pathwise"):
        PreparationConfig(return_policy="anytime_options_v1")
    config = PreparationConfig(
        product_gate="executable_intermediates_v1", return_policy="anytime_options_v1"
    )
    assert PreparationConfig(**config.payload()) == config
    with pytest.raises(ValueError, match="return policy"):
        PreparationConfig(return_policy="silently_changed")
