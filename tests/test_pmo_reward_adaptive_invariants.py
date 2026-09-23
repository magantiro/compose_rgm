"""Invariants an online controller must hold or it is learning from corrupted labels.

Each of these guards a failure that is INVISIBLE in aggregate telemetry: a partly-attributed
round, an imputed parent score, and an upstream policy that cannot express the choice the
downstream head learned about.
"""
from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.pmo_contextual_macro import (
    PRE_EXECUTION_BLOCKS,
    edge_features,
    region_replace_labels,
)
from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _row(option, parent_score=0.4):
    return {
        "parent": ASPIRIN,
        "endpoint": "Oc1ccccc1",
        "smiles": "Oc1ccccc1",
        "parent_score": parent_score,
        "families": region_replace_labels(option),
        "realized_families": region_replace_labels(option),
        "requested_modules": 2,
        "module_count": 2,
        "primitives": 0,
        "depth": 1,
        "generation": 0,
        "capacity_aware": False,
    }


def test_q_pre_distinguishes_the_intended_rebuild_option():
    """The whole point of an UPSTREAM value is that it can ask for a specific rebuild.

    If the pre-execution features collapse every rebuild into a bare `region_replace`, the
    controller can learn after the fact that a fused-ring rebuild was good and still have no
    way to increase its probability before generation.
    """
    fuse = edge_features(_row("fuse_ring"), blocks=PRE_EXECUTION_BLOCKS)
    regrow = edge_features(_row("regrow"), blocks=PRE_EXECUTION_BLOCKS)
    assert not np.allclose(fuse, regrow)
    # and the distinction must survive into the policy, not merely into the features
    brain = RewardAdaptiveProgramController()
    rows = [_row("fuse_ring"), _row("regrow")] * 20
    for index, row in enumerate(rows[:30]):
        brain.observe(row, 0.9 if "fuse_ring" in row["families"] else 0.1)
    assert brain.fitted
    weights = brain.intent_policy(rows)
    fuse_mass = sum(
        w for r, w in zip(rows, weights, strict=True) if "fuse_ring" in r["families"]
    )
    regrow_mass = sum(
        w for r, w in zip(rows, weights, strict=True) if "regrow" in r["families"]
    )
    assert fuse_mass > regrow_mass, (fuse_mass, regrow_mass)


def test_partial_reward_attribution_is_a_failure_not_a_warning():
    """Losing even one label per round biases exactly the macro learning under test, and
    'most of them arrived' is indistinguishable from 'all of them' in every aggregate."""
    scored = [{"candidate_id": f"c{i}", "score": 0.5} for i in range(16)]
    pending = {f"c{i}": _row("regrow") for i in range(14)}  # two missing

    attributed = sum(1 for o in scored if pending.get(o["candidate_id"]) is not None)
    assert attributed == 14
    with pytest.raises(RuntimeError, match="online learning requires all of them"):
        if attributed != len(scored):
            raise RuntimeError(
                f"reward feedback attributed {attributed} of {len(scored)} "
                "charged outcomes; online learning requires all of them"
            )


def test_a_decision_is_clipped_while_the_stored_label_stays_raw():
    """Clip the DECISION, never the reward. An early ridge fit can predict far outside
    [0, 1]; normalising the labels instead would change what the regression is fitting."""
    brain = RewardAdaptiveProgramController()
    rows = [_row("regrow", parent_score=0.9) for _ in range(30)]
    for row in rows:
        brain.observe(row, 0.95)
    assert brain.fitted
    # every stored label is the raw delta, untouched
    assert all(o["delta_raw"] == pytest.approx(0.95 - 0.9) for o in brain.observations)
    decision = brain._decision_utility(brain.post, rows)
    assert float(decision.min()) >= 0.0 and float(decision.max()) <= 1.0
