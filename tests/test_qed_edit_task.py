"""Invariants of the dedicated constrained-QED editing task.

These tests exist because the three defects they cover were all SILENT: an
inherited SA ceiling that refuses benchmark solutions, a direction forced by a
label that inverts a maximization objective, and two different slot counts
sharing one name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.program_task import ProgramTask
from compose_v4.tasks.qed_edit_task import (
    ALLOCATED_GRAPH_SLOTS,
    MAX_ACTIVE_HEAVY_ATOMS,
    QedEditTask,
    load_qed_edit_contract,
    production_source_state,
    resolve_score_direction,
    task_from_contract,
)

CONTRACT = Path(__file__).resolve().parents[1] / "configs" / "qed_dedicated_task_v1.json"
SOURCE = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"


def _task(**overrides):
    contract = load_qed_edit_contract(CONTRACT)
    for section, key, value in overrides.pop("region_overrides", []):
        contract[section][key] = value
    return task_from_contract(
        contract,
        name=overrides.get("name", "test"),
        oracle_protocol=overrides.get("oracle_protocol", "local:test"),
        source=overrides.get("source", SOURCE),
    )


# ---- Direction ----


def test_direction_is_resolved_from_the_frozen_guard_not_assumed():
    assert resolve_score_direction(_task()) == "maximize"


def test_routing_qed_through_t4_would_invert_the_objective():
    t4 = ProgramTask("t4", "local:test", "t4", SOURCE, 0.4)
    assert resolve_score_direction(t4) == "minimize"
    # Higher QED must not rank worse.
    assert t4.utility(0.95) < t4.utility(0.50)
    task = _task()
    assert task.utility(0.95) > task.utility(0.50)


def test_utility_refuses_bools_and_out_of_range_scores():
    task = _task()
    with pytest.raises(ValueError):
        task.utility(True)
    with pytest.raises(ValueError):
        task.utility(1.5)
    with pytest.raises(ValueError):
        task.utility(float("nan"))


# ---- Contract governance ----


def test_thresholds_have_no_defaults_so_a_partial_contract_cannot_pass_silently():
    with pytest.raises(TypeError):
        QedEditTask(name="x", oracle_protocol="p", original_source=SOURCE)  # type: ignore[call-arg]


def test_changing_a_threshold_changes_the_task_identity():
    base = _task()
    moved = _task(region_overrides=[("region", "qed_target", 0.95)])
    assert base.qed_target == 0.9 and moved.qed_target == 0.95
    assert base.task_id != moved.task_id


def test_contract_loader_refuses_a_slot_count_the_executor_would_refuse(tmp_path):
    payload = json.loads(CONTRACT.read_text())
    payload["slots"]["allocated_graph_slots"] = 40
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="allocated_graph_slots"):
        load_qed_edit_contract(path)


def test_contract_loader_refuses_a_missing_region(tmp_path):
    payload = json.loads(CONTRACT.read_text())
    payload["region"].pop("similarity_floor")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="similarity_floor"):
        load_qed_edit_contract(path)


# ---- Support versus reward ----


def test_similarity_floor_is_an_admission_gate_and_the_source_admits_itself():
    task = _task()
    row = task.endpoint_evaluator()({"smiles": SOURCE})
    assert row["oracle_eligible"] is True
    assert row["sim"] == pytest.approx(1.0)


def test_a_dissimilar_molecule_is_refused_by_the_support_not_scored_zero():
    task = _task()
    row = task.endpoint_evaluator()({"smiles": "CCO"})
    assert row["oracle_eligible"] is False
    assert row["reason"] == "below_similarity_floor"
    # The reward is QED itself; the indicator is NOT folded into it.
    assert task.reward()("CCO") == pytest.approx(row["qed"])
    assert task.reward()("CCO") > 0.0


def test_an_invalid_molecule_is_refused_with_an_explicit_reason():
    task = _task()
    row = task.endpoint_evaluator()({"smiles": "not-a-molecule"})
    assert row["oracle_eligible"] is False
    assert row["reason"] == "invalid_molecule"


def test_the_qed_target_is_not_an_admission_gate():
    """Output feasibility must stay separate from intermediate program validity."""
    task = _task()
    measured = task.measure(SOURCE)
    assert measured["qed"] < task.qed_target  # the source does not solve the benchmark
    assert task.endpoint_evaluator()({"smiles": SOURCE})["oracle_eligible"] is True
    assert task.is_success(measured) is False


def test_success_requires_both_qed_and_similarity():
    task = _task()
    assert task.is_success({"valid": True, "qed": 0.95, "sim": 0.45}) is True
    assert task.is_success({"valid": True, "qed": 0.95, "sim": 0.35}) is False
    assert task.is_success({"valid": True, "qed": 0.85, "sim": 0.95}) is False
    assert task.is_success({"valid": False, "qed": 0.99, "sim": 0.99}) is False


def test_no_synthetic_accessibility_term_enters_the_dedicated_gate():
    """The T4 gate refuses panel sources on SA alone; this task must not."""
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    high_sa = "C[C@@H](C(=O)C1=c2ccccc2=[NH+]C1)[NH+]1CCC[C@@H]1[C@@H]1CC=CS1"
    assert sascorer.calculateScore(Chem.MolFromSmiles(high_sa)) > 4.0
    t4 = ProgramTask("t4", "local:test", "t4", high_sa, 0.4)
    assert t4.endpoint_evaluator()({"smiles": high_sa})["oracle_eligible"] is False
    task = _task(source=high_sa)
    assert task.endpoint_evaluator()({"smiles": high_sa})["oracle_eligible"] is True


# ---- Slot semantics ----


def test_allocated_slots_and_active_heavy_atom_ceiling_are_distinct():
    assert ALLOCATED_GRAPH_SLOTS == 48
    assert MAX_ACTIVE_HEAVY_ATOMS == 40
    assert ALLOCATED_GRAPH_SLOTS != MAX_ACTIVE_HEAVY_ATOMS


def test_production_source_state_pads_to_the_width_the_executor_requires():
    from compose_v4.experiments.whole_ring_plan import execute_program

    state = production_source_state(SOURCE)
    assert state.n_atoms == ALLOCATED_GRAPH_SLOTS
    assert 1 <= state.n_real_atoms <= MAX_ACTIVE_HEAVY_ATOMS
    assert state.n_atoms - state.n_real_atoms > 0  # free slots for birth operations
    execute_program(state, [])  # must not raise


def test_a_tight_graph_has_no_free_slot_and_the_executor_refuses_it():
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.whole_ring_plan import execute_program

    tight = smiles_to_molecular_graph(SOURCE)
    assert tight.n_atoms == tight.n_real_atoms  # zero free slots -> no atom_insert
    with pytest.raises(ValueError, match="48-slot"):
        execute_program(tight, [])


def test_padding_to_production_max_atoms_is_refused_by_the_program_executor():
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )
    from compose_v4.experiments.whole_ring_plan import execute_program

    state = production_state_from_smiles(SOURCE, MAX_ACTIVE_HEAVY_ATOMS)
    assert state.n_atoms == 40
    with pytest.raises(ValueError, match="48-slot"):
        execute_program(state, [])


def test_a_charged_drug_like_source_is_admitted_and_scored():
    """29% of this benchmark's leads are charged; a neutral-only smoke would miss them."""
    charged = "C[C@@H](C(=O)C1=c2ccccc2=[NH+]C1)[NH+]1CCC[C@@H]1[C@@H]1CC=CS1"
    task = _task(source=charged)
    row = task.endpoint_evaluator()({"smiles": charged})
    assert row["oracle_eligible"] is True
    assert row["sim"] == pytest.approx(1.0)
    assert 0.0 < task.reward()(charged) <= 1.0
    state = production_source_state(charged)
    assert state.n_atoms == ALLOCATED_GRAPH_SLOTS


def test_every_panel_source_is_admitted_by_the_dedicated_task():
    """The T4 gate refuses 143 of these; the benchmark's own starts must all be legal."""
    panel_path = Path(__file__).resolve().parents[1] / "data" / "jin" / "qed_test.txt"
    panel = [line.strip() for line in panel_path.read_text().split("\n") if line.strip()]
    assert len(panel) == 800
    refused = []
    for smiles in panel[::40]:  # every 40th, keeps the test fast
        task = _task(source=smiles)
        if not task.endpoint_evaluator()({"smiles": smiles})["oracle_eligible"]:
            refused.append(smiles)
    assert refused == []
