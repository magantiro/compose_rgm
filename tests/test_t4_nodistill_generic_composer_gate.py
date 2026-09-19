from __future__ import annotations

import inspect
import json
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer import (
    GENERIC_FAMILY_SCHEDULES,
    primitive_scale,
    propose_generic_complete_programs,
)
from compose_v4.experiments import t4_nodistill_generic_composer_gate as gate

ROOT = Path(__file__).resolve().parents[1]


def test_generic_composer_contract_is_self_hashed_and_zero_oracle() -> None:
    envelope = json.loads((ROOT / gate.CONTRACT_RELATIVE_PATH).read_text())
    assert envelope["payload_sha256"] == identity(envelope["payload"])
    contract, contract_identity = gate.load_contract(ROOT)
    assert contract_identity == envelope["payload_sha256"]
    assert contract["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }
    assert len(contract["cells"]) == 6


def test_scale_bands_cover_exact_declared_program_support() -> None:
    assert primitive_scale(1) == primitive_scale(3) == "small"
    assert primitive_scale(4) == primitive_scale(11) == "medium"
    assert primitive_scale(12) == primitive_scale(32) == "large"


def test_generic_schedules_include_joint_cross_family_composition() -> None:
    assert all(GENERIC_FAMILY_SCHEDULES.values())
    assert any(len(row) >= 2 for row in GENERIC_FAMILY_SCHEDULES["medium"])
    assert any(len(row) == 3 for row in GENERIC_FAMILY_SCHEDULES["large"])
    flattened = {
        family for rows in GENERIC_FAMILY_SCHEDULES.values() for row in rows for family in row
    }
    assert {
        "segment_grow",
        "segment_replace",
        "substituent_delete",
        "construct_substituted_ring",
        "ring_path_remodel",
        "append_ring",
        "fuse_ring",
    } <= flattened


def test_generic_composer_uses_no_route_or_task_input() -> None:
    signature = inspect.signature(propose_generic_complete_programs)
    assert tuple(signature.parameters) == (
        "source",
        "seed",
        "per_scale",
        "attempts_per_scale",
        "maximum_primitives",
        "maximum_blocks",
    )
    for forbidden in (
        "target_name",
        "cell_key",
        "teacher_endpoint",
        "route_checkpoint",
        "comparator_score",
    ):
        assert forbidden not in signature.parameters


def test_small_composer_fixture_is_exact_and_deterministic() -> None:
    molecule = pad_molecular_graph(smiles_to_molecular_graph("CCNCCc1ccccc1"), 48)
    arguments = {
        "seed": 713,
        "per_scale": 1,
        "attempts_per_scale": 24,
    }
    first = propose_generic_complete_programs(molecule, **arguments)
    second = propose_generic_complete_programs(molecule, **arguments)
    assert [row.endpoint_key for row in first.proposals] == [
        row.endpoint_key for row in second.proposals
    ]
    assert all(1 <= len(row.actions) <= 32 for row in first.proposals)
    assert all(row.requested_scale == row.realized_scale for row in first.proposals)
    assert first.telemetry["trajectory_distilled_templates_loaded"] == 0
    assert first.telemetry["route_weights_loaded"] == 0
    assert first.telemetry["runtime_task_cell_or_target_input"] is False
    assert isinstance(len(first.proposals), int)


def test_teacher_import_is_confined_to_post_lock_diagnostic() -> None:
    lock_source = inspect.getsource(gate.lock_cell)
    teacher_source = inspect.getsource(gate._teacher_descriptors)
    assert "teacher_traces" not in lock_source
    assert "teacher_traces" in teacher_source
