from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.route_distilled_dynamic_optimizer import (
    RouteDistilledDynamicOptimizer,
    initial_route_distilled_program_batch,
)
from compose_v4.control.route_distilled_program_policy import (
    RouteDistilledProgramPolicy,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_route_distilled_full import SEED_PAIRS, full_payload
from compose_v4.experiments.t4_route_distilled_qualification import (
    qualification_payload,
)
from tools.t4_route_policy_select import actor_selection_gate

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "diagnostics/t4_route_distillation/attempt_3/actor.json.gz"


def _checkpoint():
    with gzip.open(ACTOR, "rt") as handle:
        envelope = json.load(handle)
    return envelope["payload"]


def test_route_distilled_initial_batch_is_route_empty_and_deterministic():
    source = production_state_from_smiles("CCNCC", max_atoms=48)
    policy = RouteDistilledProgramPolicy.from_checkpoint(_checkpoint())
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=29),
        attempts_per_batch=16,
        candidates_per_batch=4,
        wall_seconds=10,
    )
    kwargs = {
        "source_group": "test-source",
        "oracle_protocol": "structural-test",
        "eligibility": lambda _: {"oracle_eligible": True},
        "policy": policy,
    }

    first = initial_route_distilled_program_batch(source, (), config, **kwargs)
    second = initial_route_distilled_program_batch(source, (), config, **kwargs)

    assert first["batch_id"] == second["batch_id"]
    assert first["initial_route_archive"] == []
    assert first["source_library_rows_loaded"] == 0
    assert first["new_oracle_calls"] == 0
    assert {row["channel"] for row in first["attempts"]} >= {
        "dynamic_generic_composition",
        "route_distilled_program",
    }


def test_route_distilled_optimizer_snapshot_binds_policy_and_rng_state():
    checkpoint = _checkpoint()
    RouteDistilledDynamicOptimizer.configure_policy(checkpoint)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=31),
        attempts_per_batch=8,
        candidates_per_batch=2,
    )
    optimizer = RouteDistilledDynamicOptimizer(
        config,
        source_group="test-source",
        oracle_protocol="structural-test",
        hierarchy=None,
    )

    snapshot = optimizer.snapshot(include_history=False)
    restored = RouteDistilledDynamicOptimizer.restore(snapshot, hierarchy=None)

    assert restored.snapshot(include_history=False) == snapshot


def test_route_distilled_optimizer_requires_frozen_policy():
    prior_policy = RouteDistilledDynamicOptimizer._configured_policy
    prior_identity = RouteDistilledDynamicOptimizer._configured_checkpoint_identity
    try:
        RouteDistilledDynamicOptimizer._configured_policy = None
        RouteDistilledDynamicOptimizer._configured_checkpoint_identity = None
        config = ProgramSearchConfig.program_only_recipe(seed=37)
        try:
            RouteDistilledDynamicOptimizer(
                config,
                source_group="test-source",
                oracle_protocol="structural-test",
                hierarchy=None,
            )
        except ValueError as error:
            assert "policy is not configured" in str(error)
        else:
            raise AssertionError("missing route-distilled policy was accepted")
    finally:
        RouteDistilledDynamicOptimizer._configured_policy = prior_policy
        RouteDistilledDynamicOptimizer._configured_checkpoint_identity = prior_identity


def test_actor_selection_gate_does_not_trade_recovery_for_low_yield():
    autonomous = {
        "generic_marginal": {
            "cutoffs": {"128": {"transformation_recall": 0.4}},
            "source_balanced_transformation_mrr": 0.2,
            "execution_precision": 0.8,
            "unique_endpoint_yield": 0.6,
        },
        "context_module_prototype": {
            "cutoffs": {"128": {"transformation_recall": 0.5}},
            "source_balanced_transformation_mrr": 0.3,
            "execution_precision": 0.39,
            "unique_endpoint_yield": 0.31,
        },
    }

    gate = actor_selection_gate(autonomous)

    assert gate == {
        "transformation_recall_at_128_not_lower": True,
        "transformation_mrr_strictly_higher": True,
        "execution_precision_retains_half": False,
        "unique_endpoint_yield_retains_half": True,
    }


def test_full_wave_uses_paired_seeds_and_disables_plateau_stopping():
    assert SEED_PAIRS == (
        (20260913, 1701),
        (20260914, 1702),
        (20260915, 1703),
    )
    for delta in (0.4, 0.6):
        payload = full_payload(delta)
        assert payload["plateau_stopping"] is False
        assert payload["controller_docking_seed_pairs"] == [
            list(row) for row in SEED_PAIRS
        ]
        assert payload["units"] == 45
    assert qualification_payload()["qualification_gate"] == {
        "durable_units": 5,
        "minimum_scored_endpoints_per_cell": 1,
        "minimum_cells_at_or_better_than_ivg_mean": 4,
    }
