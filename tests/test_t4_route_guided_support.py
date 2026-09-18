"""Scientific-invariant tests for target-held-out route-guided support probing."""

from __future__ import annotations

import numpy as np

from compose_v4.control.constructive_features import MODE_FEATURE_NAMES, SITE_FEATURE_NAMES
from compose_v4.control.constructive_policy import PolicyShape
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_guided_support import (
    aggregate_attempt_results,
    fit_held_target_prior,
    jak2_motif_flags,
    run_arm,
)
from compose_v4.rewrite.trace_shard import decode_state


def _rows():
    state = unseal.__module__  # stable non-molecular value is not used by the fit fixture
    del state
    return [
        {
            "target": target,
            "state": {
                "atom_types": [2, 2, 0],
                "formal_charges": [0, 0, 0],
                "implicit_h_counts": [3, 3, 0],
                "bonds": [[0, 1, 1]],
                "n_slots": 3,
            },
            "site": 0,
            "mode": {
                "attachment_count": 1,
                "created_atoms": 1,
                "closes_ring": False,
                "opens_ring": False,
                "primitive_count": 1,
            },
        }
        for target in ("jak2", "braf", "fa7")
    ]


def test_held_target_is_removed_before_vocabulary_and_fit():
    _, vocabulary, audit = fit_held_target_prior(_rows(), "jak2")
    assert audit["held_target_absent_from_training"] is True
    assert audit["excluded_held_target_rows"] == 1
    assert audit["training_targets"] == ["braf", "fa7"]
    assert len(vocabulary) == 1


def test_jak2_basin_requires_both_amide_and_diamine_ring():
    assert jak2_motif_flags("O=C(CC)N1CCNCC1")["basin"] is True
    assert jak2_motif_flags("O=C(CC)NCC")["basin"] is False
    assert jak2_motif_flags("N1CCNCC1")["basin"] is False


def test_run_arm_has_zero_oracle_calls_and_explicit_same_pool_control(monkeypatch):
    unit = unseal(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs/t4_objective_reset_runtime_v1.json"
    )["units"][0]
    source = decode_state(unit["source_state"])
    shape = PolicyShape(len(SITE_FEATURE_NAMES), len(MODE_FEATURE_NAMES))
    model = {"shape": shape, "theta": np.zeros(shape.size)}
    vocabulary = [
        {
            "attachment_count": 1,
            "created_atoms": 1,
            "closes_ring": False,
            "opens_ring": False,
            "primitive_count": 1,
        }
    ]
    result = run_arm(
        source,
        unit["original_seed"],
        unit["target"],
        model,
        vocabulary,
        arm="uniform_same_pool",
        attempts=1,
        seed=17,
        delta=0.4,
        depths=(1,),
        candidates_per_step=4,
        route_exploration=0.1,
        temperature=1.0,
    )
    assert result["new_oracle_calls"] == 0
    assert result["attempted"] == 1
    assert result["arm"] == "uniform_same_pool"


def test_independent_attempt_reduction_preserves_counts_and_unique_endpoints():
    base = {
        "arm": "route_prior",
        "attempted": 1,
        "completed": 1,
        "generation_failures": {},
        "feasible_programs": 1,
        "raw_factor_rates": {"basin": {"count": 1, "rate": 1.0}},
        "feasible_factor_rates": {"basin": {"count": 1, "rate": 1.0}},
        "elapsed_seconds": 2.0,
        "new_oracle_calls": 0,
    }
    results = []
    for seed, endpoint in ((11, "CC"), (12, "CC"), (13, "CCC")):
        results.append(
            {
                **base,
                "worker_seed": seed,
                "ledger": [
                    {
                        "attempt": 0,
                        "status": "complete",
                        "endpoint": endpoint,
                        "audit": {"official_compose_valid": True},
                    }
                ],
            }
        )
    merged = aggregate_attempt_results(results, arm="route_prior", attempt_budget=3)
    assert merged["completed"] == 3
    assert merged["feasible_programs"] == 3
    assert merged["unique_raw_endpoints"] == 2
    assert merged["unique_feasible_endpoints"] == 2
    assert merged["raw_factor_rates"]["basin"]["count"] == 3
    assert merged["new_oracle_calls"] == 0
