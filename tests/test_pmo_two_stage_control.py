"""The two-stage controller's guarantees: no leakage, no collapse, no self-confirming data."""

from __future__ import annotations

import numpy as np

from compose_v4.control.pmo_contextual_macro import (
    POST_EXECUTION_BLOCKS,
    PRE_EXECUTION_BLOCKS,
    edge_features,
)
from compose_v4.control.pmo_two_stage_control import TwoStageControl


def _edge(parent, endpoint, *, families=("region_replace",), score=0.3, parent_score=0.2):
    return {
        "parent": parent,
        "endpoint": endpoint,
        "smiles": endpoint,
        "families": list(families),
        "requested_modules": 2,
        "module_count": 2,
        "primitives": 11,
        "depth": 1,
        "generation": 1,
        "capacity_aware": False,
        "parent_score": parent_score,
        "score": score,
    }


def test_q_pre_cannot_see_the_endpoint():
    """The load-bearing guarantee: changing ONLY the realized endpoint must not move Q_pre.

    If it did, the controller would be choosing what to propose using the molecule it has not
    proposed yet -- and every upstream allocation claim would be circular.
    """
    a = _edge("CCO", "CCOC")
    b = _edge("CCO", "c1ccccc1C(=O)NCCS(N)(=O)=O")
    pre_a = edge_features(a, blocks=PRE_EXECUTION_BLOCKS)
    pre_b = edge_features(b, blocks=PRE_EXECUTION_BLOCKS)
    assert np.allclose(pre_a, pre_b), "a pre-execution feature moved with the endpoint"

    post_a = edge_features(a, blocks=POST_EXECUTION_BLOCKS)
    post_b = edge_features(b, blocks=POST_EXECUTION_BLOCKS)
    assert not np.allclose(post_a, post_b), "Q_post must see the endpoint it is ranking"


def test_q_pre_does_see_the_parent_and_the_intent():
    base = _edge("CCO", "CCOC")
    other_parent = edge_features(_edge("c1ccccc1", "CCOC"), blocks=PRE_EXECUTION_BLOCKS)
    other_family = edge_features(
        _edge("CCO", "CCOC", families=("append_ring",)), blocks=PRE_EXECUTION_BLOCKS
    )
    reference = edge_features(base, blocks=PRE_EXECUTION_BLOCKS)
    assert not np.allclose(reference, other_parent), "Q_pre must see the parent"
    assert not np.allclose(reference, other_family), "Q_pre must see the macro intent"


def test_an_unfitted_controller_is_uniform_rather_than_opinionated():
    control = TwoStageControl()
    intents = [_edge("CCO", "CCOC"), _edge("c1ccccc1", "CCOC", families=("append_ring",))]
    weights = control.intent_weights(intents)
    assert np.allclose(weights, 0.5), "an unfitted model must not express a preference"


def test_no_live_lineage_is_ever_starved():
    """A previous PMO run collapsed onto one initialization lineage; the floor exists for that."""
    control = TwoStageControl(lineage_floor=0.2)
    rng = np.random.default_rng(0)
    for i in range(40):
        control.observe(
            _edge("CCO", f"CCO{'C' * (i % 7)}", score=0.9 if i % 2 else 0.01), 0.9 if i % 2 else 0.01
        )
    intents = [_edge("CCO", "CCOC"), _edge("c1ccccc1", "CCOC", families=("append_ring",))]
    shares = control.parent_shares(intents, ["CCO", "c1ccccc1"])
    assert min(shares.values()) > 0.0, "no live parent may receive zero proposal mass"
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    del rng


def test_a_slice_of_calls_is_drawn_without_consulting_the_model():
    """Without an audit quota the training set becomes a fixpoint of the model's own opinion."""
    control = TwoStageControl(audit_fraction=0.25)
    for i in range(40):
        control.observe(_edge("CCO", f"CCO{'C' * (i % 9)}", score=i / 40), i / 40)
    candidates = [_edge("CCO", f"CC{'N' * (i % 5)}O", score=0.0) for i in range(12)]
    chosen, detail = control.acquire(
        candidates, [("m", 0.5)], batch=8, rng=np.random.default_rng(1)
    )
    reasons = {d["reason"] for d in detail}
    assert "audit" in reasons, "some calls must be drawn without reference to Q_post"
    assert len(chosen) == 8
    assert all(0.0 <= d["propensity"] <= 1.0 for d in detail), "propensities must be recorded"


def test_every_selection_records_the_probability_it_was_chosen_with():
    control = TwoStageControl()
    candidates = [_edge("CCO", f"CC{'N' * i}O") for i in range(6)]
    chosen, detail = control.acquire(
        candidates, [("m", 0.5)], batch=3, rng=np.random.default_rng(2)
    )
    assert len(detail) == len(chosen) == 3
    assert control.propensities, "propensities must be retained for later correction"
