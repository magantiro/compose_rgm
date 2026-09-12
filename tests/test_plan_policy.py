import numpy as np
import torch

from compose_v4.control.docking_value import identity
from compose_v4.control.plan_policy import (
    RECIPE,
    EndpointNetwork,
    PlanPolicy,
    actor_network,
    policy_snapshot,
    weights,
)


def fixture_policy():
    torch.manual_seed(17)
    network = EndpointNetwork().double()
    actor = actor_network()
    torch.nn.init.zeros_(actor[-1].weight)
    torch.nn.init.zeros_(actor[-1].bias)
    body = {"recipe": RECIPE, "weights": weights(network), "training_smiles": []}
    encoder = {**body, "encoder_id": identity(body)}
    snapshot = policy_snapshot(actor)
    return encoder, snapshot, PlanPolicy(encoder, snapshot)


def test_plan_distribution_is_normalized_with_exploration_floor():
    _, _, policy = fixture_policy()
    q = policy.distribution("CC", ["CCC", "CCN", "CCO"], [0.2, 0.3, 0.4])
    assert np.isclose(q.sum(), 1)
    assert np.all(q >= RECIPE["exploration"] / 3 - 1e-12)


def test_positive_observed_gain_increases_selected_plan_probability():
    encoder, snapshot, policy = fixture_policy()
    products, release = ["CCC", "CCN", "CCO"], [0.2, 0.3, 0.4]
    before = policy.distribution("CC", products, release)
    decision = {
        "decision_id": "paid-fixture",
        "policy_id": snapshot["policy_id"],
        "source": "CC",
        "products": products,
        "release": release,
        "probabilities": before.tolist(),
        "selected": 1,
        "gain": 0.1,
    }
    updated, audit = policy.update([decision])
    after = PlanPolicy(encoder, updated).distribution("CC", products, release)
    assert audit["changed"]
    assert audit["scored_decisions"] == 1
    assert after[1] > before[1]


def test_policy_update_rejects_mismatched_behavior_probabilities():
    _, snapshot, policy = fixture_policy()
    decision = {
        "decision_id": "bad-fixture",
        "policy_id": snapshot["policy_id"],
        "source": "CC",
        "products": ["CCC", "CCO"],
        "release": [0.2, 0.4],
        "probabilities": [0.9, 0.1],
        "selected": 0,
        "gain": 0.1,
    }
    try:
        policy.update([decision])
    except ValueError as error:
        assert "behavior probabilities" in str(error)
    else:
        raise AssertionError("mismatched behavior policy was accepted")
