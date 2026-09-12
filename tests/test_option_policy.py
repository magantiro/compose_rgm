import numpy as np
import pytest

from compose_v4.control.option_policy import (
    OptionActorExample,
    OptionActorFitConfig,
    actor_distribution,
    conservative_option_distribution,
    fit_option_actor,
)


def example(policy="policy-a", *, weight=None):
    return OptionActorExample(
        source_id="source-a",
        decision_id="decision-a",
        behavior_policy_id=policy,
        state_features=(0.0,),
        options=("generic", "cyclize"),
        option_features=((0.0,), (1.0,)),
        reference=(0.5, 0.5),
        behavior=(0.5, 0.5),
        selected=1,
        advantage=1.0,
        importance_weight=weight,
    )


def test_conservative_option_law_keeps_generic_floor_and_kl():
    decision = conservative_option_distribution(
        ["generic", "cyclize", "grow"], [0.8, 0.15, 0.05], [-10, 5, 1]
    )
    assert sum(decision.proposal) == pytest.approx(1)
    assert np.all(np.asarray(decision.proposal) >= 0.1 * np.asarray(decision.reference))
    assert decision.kl <= 1 + 1e-10
    assert decision.proposal[1] > decision.reference[1]
    with pytest.raises(ValueError, match="generic"):
        conservative_option_distribution(["cyclize"], [1], [2])


def test_advantage_weighted_actor_learns_declared_choice():
    actor, snapshot = fit_option_actor(
        [example()],
        behavior_policy_id="policy-a",
        config=OptionActorFitConfig(hidden=8, updates=40, learning_rate=0.01, seed=9),
    )
    decision = actor_distribution(actor, example())
    assert decision.proposal[1] > 0.5
    assert snapshot["behavior_policy_id"] == "policy-a"
    assert snapshot["policy_id"]
    with pytest.raises(ValueError, match="behavior-policy"):
        fit_option_actor([example("policy-b")], behavior_policy_id="policy-a")
    fit_option_actor(
        [example("policy-b", weight=0.2)],
        behavior_policy_id="policy-a",
        config=OptionActorFitConfig(hidden=4, updates=1),
    )


def test_extreme_advantage_is_capped_before_exponentiation():
    row = example()
    row = OptionActorExample(
        **{**row.__dict__, "advantage": 1e300},
    )
    _, receipt = fit_option_actor(
        [row],
        behavior_policy_id="policy-a",
        config=OptionActorFitConfig(hidden=4, updates=1),
    )
    assert np.isfinite(receipt["history"][-1]["loss"])
