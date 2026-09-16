from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import DynamicProgramOptimizer
from compose_v4.experiments.t4_hybrid_top3_v0_stage20 import (
    FULL_FROZEN_CEILING,
    REPORT_CALLS,
    STAGE_CEILING,
    _configure_refinement_rng,
    _macro_batch,
    _summary_points,
    load_contract,
    run_identity,
)
from modal_apps.t4_hybrid_top3_v0_stage20_app import serialized
from tools.t4_hybrid_top3_v0_stage20 import material_files

ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_self_hashed_and_stage_is_bounded() -> None:
    envelope = json.loads(
        (ROOT / "configs/t4_hybrid_top3_v0_stage20_launch_v1.json").read_text()
    )
    payload = envelope["payload"]
    assert envelope["contract_sha256"] == identity(payload)
    assert STAGE_CEILING == 20
    assert FULL_FROZEN_CEILING == 100
    assert REPORT_CALLS == (1, 5, 10, 20)
    assert payload["protocol"]["stage_total_call_ceiling"] == 100
    assert payload["protocol"]["maximum_concurrent_single_cpu_workers"] == 5
    assert payload["protocol"]["automatic_retry"] is False
    assert payload["protocol"]["replacement"] is False
    assert payload["protocol"]["backfill"] is False
    assert payload["protocol"]["confirmation_calls"] == 0
    assert payload["protocol"]["plateau_stopping"] is False


def test_inputs_and_exact_five_replicate_zero_units_are_bound() -> None:
    contract = load_contract(ROOT)
    assert contract["launch"]["units"] == [
        "5ht1b_0_r0",
        "braf_1_r0",
        "jak2_1_r0",
        "parp1_0_r0",
        "fa7_0_r0",
    ]
    assert [row["unit_id"] for row in contract["units"]] == contract["launch"]["units"]
    assert all(
        row["replicate"] == 0 and row["budget"] == 1000 for row in contract["units"]
    )
    assert len(run_identity(contract)) == 64


def test_modal_image_serializes_every_task_validated_material() -> None:
    contract = load_contract(ROOT)
    assert serialized == material_files(contract)
    assert "docs/T4_HYBRID_TOP3_V0_STAGE20_LAUNCH.md" in serialized
    assert "tests/test_t4_hybrid_top3_v0_stage20.py" in serialized


def test_locked_macro_batch_preserves_calls_and_endpoints() -> None:
    candidate = json.loads(
        (
            ROOT
            / "diagnostics/t4_hybrid_top3_v0_controller/attempt_1/candidate_lock.json"
        ).read_text()
    )["payload"]
    schedule = next(row for row in candidate["cells"] if row["cell"] == "jak2_1")
    batch = _macro_batch(schedule, source_group="source", oracle_protocol="oracle")
    assert [row["candidate_id"] for row in batch["candidates"]] == [
        row["candidate_id"] for row in schedule["initial_macro_calls"]
    ]
    assert [row["endpoint"] for row in batch["candidates"]] == [
        row["canonical_smiles"] for row in schedule["initial_macro_calls"]
    ]
    assert [row["provenance"]["locked_call"] for row in batch["candidates"]] == [
        1,
        2,
        3,
    ]
    assert batch["new_oracle_calls"] == 0


def test_refinement_rng_namespace_is_deterministic_and_snapshot_complete() -> None:
    from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig

    left = DynamicProgramOptimizer(
        ProgramSearchConfig.program_only_recipe(seed=7),
        source_group="source",
        oracle_protocol="oracle",
        hierarchy=None,
    )
    right = DynamicProgramOptimizer(
        ProgramSearchConfig.program_only_recipe(seed=999),
        source_group="source",
        oracle_protocol="oracle",
        hierarchy=None,
    )
    words = [1, 2, 3, 4]
    _configure_refinement_rng(left, words)
    _configure_refinement_rng(right, words)
    assert np.array_equal(left.rng.random(8), right.rng.random(8))
    snapshot = left.snapshot(include_history=False)
    restored = DynamicProgramOptimizer.restore(snapshot, hierarchy=None)
    assert restored.rng.bit_generator.state == left.rng.bit_generator.state


def test_summary_points_never_interpolate_missing_calls() -> None:
    curve = [{"query": index, "best_score": -float(index)} for index in range(1, 11)]
    assert _summary_points(curve) == {
        "1": -1.0,
        "5": -5.0,
        "10": -10.0,
        "20": None,
    }
