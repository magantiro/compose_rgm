"""Focused guards for the route-distilled constructive proposal expert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.constructive_features import (
    MODE_FEATURE_NAMES,
    SITE_FEATURE_NAMES,
)
from compose_v4.control.constructive_guided_composition import (
    MAX_BLOCKS,
    MAX_PRIMITIVES,
    compose_guided,
    prospect_guided,
)
from compose_v4.control.constructive_policy import PolicyShape
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _source():
    units = {
        unit["cell"]: unit
        for unit in unseal(ROOT / "configs/t4_objective_reset_runtime_v1.json")["units"]
    }
    return decode_state(units["parp1_0"]["source_state"])


def _vocabulary():
    return [
        {
            "attachment_count": sites,
            "created_atoms": created,
            "closes_ring": closes,
            "opens_ring": False,
            "primitive_count": max(created, 1),
        }
        for sites in (1, 2, 3)
        for created in range(9)
        for closes in (False, True)
        if created or closes
    ]


def _model():
    shape = PolicyShape(len(SITE_FEATURE_NAMES), len(MODE_FEATURE_NAMES))
    return {"shape": shape, "theta": np.zeros(shape.size)}


def test_guided_composition_stays_inside_the_existing_exact_support():
    trace, metadata = compose_guided(
        _source(),
        np.random.default_rng(np.random.SeedSequence([20260918, 1])),
        2,
        _model(),
        _vocabulary(),
        candidates_per_step=16,
    )
    assert 1 <= len(trace["actions"]) <= MAX_PRIMITIVES
    assert len(trace["blocks"]) <= MAX_BLOCKS
    assert metadata["completed_modules"] >= 1
    assert metadata["initial_stored_complete_routes"] == 0
    assert metadata["source_library_rows_loaded"] == 0


def test_exploration_is_explicit_and_does_not_disable_exact_execution():
    trace, metadata = compose_guided(
        _source(),
        np.random.default_rng(np.random.SeedSequence([20260918, 2])),
        1,
        _model(),
        _vocabulary(),
        candidates_per_step=16,
        exploration=1.0,
    )
    assert trace["complete"] is True
    assert metadata["selections"][0]["selection"] == "exploration"


def test_guided_composition_rejects_support_and_probability_drift():
    for depth in (0, MAX_BLOCKS + 1):
        with pytest.raises(ValueError, match="declared"):
            compose_guided(_source(), np.random.default_rng(0), depth, _model(), _vocabulary())
    with pytest.raises(ValueError, match="exploration"):
        compose_guided(
            _source(),
            np.random.default_rng(0),
            1,
            _model(),
            _vocabulary(),
            exploration=1.1,
        )


def test_guided_prospect_is_zero_oracle_and_deduplicates_endpoints():
    result = prospect_guided(
        _source(),
        lambda candidate: {
            "oracle_eligible": True,
            "endpoint_exclusion_reasons": [],
            "smiles": candidate["smiles"],
        },
        np.random.default_rng(np.random.SeedSequence([20260918, 3])),
        _model(),
        _vocabulary(),
        attempts=3,
        minimum_primitives=1,
        depths=(1,),
        candidates_per_step=8,
    )
    assert result["new_oracle_calls"] == 0
    endpoints = [row["endpoint"] for row in result["pool"]]
    assert len(endpoints) == len(set(endpoints))
    assert result["eligible_endpoints"] == len(endpoints)
