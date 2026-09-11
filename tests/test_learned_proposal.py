"""Bounded policy-law checks; no oracle or neural reference calls."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.learned_proposal import RECIPE, ProposalPolicy, fit
from compose_v4.experiments.pmo_branch_policy import worker_identity


def zero_policy():
    payload = {
        "recipe": RECIPE,
        "coefficients": [0.0] * RECIPE["dimension"],
        "data_sha256": "fixture",
    }
    payload["model_sha256"] = identity(payload)
    return ProposalPolicy(payload)


def test_return_weights_change_fit_without_new_labels():
    examples = [
        {"features": [{0: 1.0}, {}, {}, {}, {}], "weight": 9.0},
        {"features": [{0: -1.0}, {}, {}, {}, {}], "weight": 1.0},
    ]
    model = fit(examples, data_sha256="fixture")
    assert model.coefficients[0] > 0
    assert model.payload == fit(examples, data_sha256="fixture").payload
    assert model.payload["fit"]["objective"] < model.payload["fit"]["initial_objective"]
    changed = deepcopy(model.payload)
    changed["coefficients"][0] += 1
    with pytest.raises(ValueError, match="hash"):
        ProposalPolicy(changed)
    with pytest.raises(ValueError, match="positive"):
        fit([{**examples[0], "weight": float("nan")}], data_sha256="bad")


def test_what_floor_and_how_rejection_match_same_declared_mixture():
    policy = zero_policy()
    base = np.array([0.75, 0.2499, 0.0001])
    energies = np.array([-RECIPE["energy_bound"], RECIPE["energy_bound"], 0.0])
    children = [
        SimpleNamespace(graph=i, active=SimpleNamespace(option="generic")) for i in range(3)
    ]
    policy.energy = lambda node, option, graph: energies[graph]
    row = SimpleNamespace(reference=base, successors=children)
    q, audit = policy.distribution(SimpleNamespace(stage="what"), row)
    assert q.sum() == pytest.approx(1)
    assert np.all(q >= 0.1 * base) and audit["kl"] <= 1
    calls = 0

    def sample_reference(node, rng):
        nonlocal calls
        calls += 1
        return children[rng.choice(3, p=base)]

    hierarchy = SimpleNamespace(sample_reference=sample_reference)
    node = SimpleNamespace(stage="how", active=SimpleNamespace(option="generic"))
    rng = np.random.default_rng(91)
    samples = np.zeros(3)
    for _ in range(4000):
        child, receipt = policy.sample(hierarchy, node, rng)
        samples[child.graph] += 1
        assert receipt["attempts"] >= 1
    assert np.max(np.abs(samples / samples.sum() - q)) < 0.025
    assert calls > 4000
    dead = SimpleNamespace(sample_reference=lambda node, rng: None)
    child, receipt = policy.sample(dead, node, rng)
    assert child is None and receipt["status"] == "empty_reference"


def test_policy_identity_separates_workers_without_changing_legacy_identity():
    parent = {"id": "root"}
    assert worker_identity(1, 2, parent) == identity({"phase": 1, "slot": 2, "parent": parent})
    assert worker_identity(1, 2, parent, "learned") != worker_identity(1, 2, parent)


def test_registered_comparison_binds_frozen_policy_and_training():
    from compose_v4.experiments.pmo_learned_proposal import load_contract

    contract = load_contract(Path(__file__).resolve().parents[1])
    assert contract["selection_modes"] == {"baseline": "immediate", "learned": "immediate"}
    assert contract["proposal_ids"]["baseline"] is None
    assert contract["proposal_ids"]["learned"]
    assert contract["proposal_training_authorized"]
    assert not contract["reference_training_authorized"]
