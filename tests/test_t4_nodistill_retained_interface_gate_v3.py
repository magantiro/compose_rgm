from __future__ import annotations

import inspect
import json
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_retained_interface_composer_v3 import (
    CYCLE_GAIN_GRID,
    HEAVY_GROWTH_BANDS,
    MACRO_MODES,
    allocate_retained_interface_macro_plans,
    propose_generic_retained_interface_programs,
)
from compose_v4.experiments import t4_nodistill_retained_interface_gate_v3 as gate

ROOT = Path(__file__).resolve().parents[1]


def test_v3_contract_is_self_hashed_zero_oracle_and_prospective() -> None:
    envelope = json.loads((ROOT / gate.CONTRACT_RELATIVE_PATH).read_text())
    assert envelope["payload_sha256"] == identity(envelope["payload"])
    contract, contract_identity = gate.load_contract(ROOT)
    assert contract_identity == envelope["payload_sha256"]
    assert contract["status"] == "FROZEN_ZERO_ORACLE_SUPPORT_GATE"
    assert contract["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }
    support_gate = contract["evaluation"]["prospective_support_gate"]
    assert support_gate["exact_replay_precision"] == 1.0
    assert (
        support_gate["minimum_archive_ready_unique_endpoints_per_motivating_cell"] == 2
    )
    assert (
        support_gate["minimum_archive_ready_macro_plan_identities_per_motivating_cell"]
        == 2
    )


def test_v3_grid_is_balanced_and_source_state_allocated() -> None:
    assert tuple(HEAVY_GROWTH_BANDS) == ("small", "medium", "large")
    assert CYCLE_GAIN_GRID == (1, 2, 3)
    assert MACRO_MODES == ("grow", "replace", "open_grow")
    source = pad_molecular_graph(smiles_to_molecular_graph("CCc1ccccc1"), 48)
    features, plans, telemetry = allocate_retained_interface_macro_plans(
        source, candidate_quota_per_particle=2
    )
    assert telemetry["predeclared_particle_count"] == 27
    assert telemetry["allocated_particle_count"] == len(plans)
    assert features.available_heavy_capacity == 32
    assert features.cycle_rank == 1
    assert {plan.desired_heavy_band for plan in plans} == set(HEAVY_GROWTH_BANDS)
    assert {plan.desired_delta_cycle_rank for plan in plans} == set(CYCLE_GAIN_GRID)
    assert {plan.mode for plan in plans} == set(MACRO_MODES)
    assert all(len(plan.families) <= 8 for plan in plans)


def test_v3_near_capacity_source_abstains_before_compilation() -> None:
    registry = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    source = pad_molecular_graph(smiles_to_molecular_graph(registry[9]["smiles"]), 48)
    features, plans, telemetry = allocate_retained_interface_macro_plans(
        source, candidate_quota_per_particle=2
    )
    assert features.n_heavy == 40
    assert features.available_heavy_capacity == 0
    assert plans == ()
    assert telemetry["allocated_particle_count"] == 0
    assert telemetry["skipped_particle_count"] == 27


def test_v3_runtime_api_has_no_task_route_teacher_delta_or_fiber_input() -> None:
    signature = inspect.signature(propose_generic_retained_interface_programs)
    assert tuple(signature.parameters) == (
        "source",
        "seed",
        "attempts_per_particle",
        "candidate_quota_per_particle",
        "maximum_heavy_atoms",
        "maximum_primitives",
        "maximum_blocks",
    )
    for forbidden in (
        "target",
        "cell",
        "delta",
        "fiber",
        "objective",
        "route",
        "template",
        "teacher",
        "endpoint",
        "weights",
    ):
        assert forbidden not in signature.parameters


def test_v3_composer_is_deterministic_and_every_returned_program_is_exact() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCc1ccccc1"), 48)
    arguments = {
        "seed": 9917,
        "attempts_per_particle": 1,
        "candidate_quota_per_particle": 1,
    }
    first = propose_generic_retained_interface_programs(source, **arguments)
    second = propose_generic_retained_interface_programs(source, **arguments)
    assert [row.endpoint_key for row in first.proposals] == [
        row.endpoint_key for row in second.proposals
    ]
    assert first.telemetry == second.telemetry
    assert all(
        row.metadata["schema_version"].endswith("_v3") for row in first.proposals
    )
    assert all(len(row.actions) <= 32 for row in first.proposals)
    assert all(len(row.program["blocks"]) <= 8 for row in first.proposals)


def test_v3_teacher_loading_and_endpoint_admission_are_post_generation_only() -> None:
    lock_source = inspect.getsource(gate.lock_cell)
    evaluate_source = inspect.getsource(gate.evaluate_locks)
    assert "_teacher_descriptors" not in lock_source
    assert "_teacher_descriptors" in evaluate_source
    assert "_apply_post_generation_production_admission" in lock_source
    assert lock_source.index(
        "propose_generic_retained_interface_programs"
    ) < lock_source.index("_apply_post_generation_production_admission")


def test_v3_sealed_result_preserves_the_preregistered_gate() -> None:
    root = ROOT / gate.ARTIFACT_ROOT_RELATIVE_PATH
    result = json.loads((root / "result.json").read_text())
    assert result["payload_sha256"] == identity(result["payload"])
    payload = result["payload"]
    assert payload["decision"] in {
        "PASS_RETAINED_INTERFACE_SUPPORT_GATE",
        "NO_PROMOTION",
    }
    assert payload["gates"]["values"]["exact_replay_precision_one"] is True
    assert payload["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }
