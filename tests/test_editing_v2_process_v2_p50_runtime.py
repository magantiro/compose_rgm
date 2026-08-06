"""Focused tests for the minimal Process-V2 P50 fast path."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.factorized_successor_training import (
    CanonicalSuccessorAliasGroup,
    CompiledStateSuccessorMap,
    CompiledSuccessorMark,
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_payload,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50ScopedPrerequisites,
)
from compose_v4.experiments import editing_v2_process_v2_p50_runtime as runtime


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _family(cell: str) -> str:
    return cell.split(":", 2)[1]


def _candidate(*, cell: str, role: str, index: int) -> dict[str, object]:
    body: dict[str, object] = {
        "v1_task_identity_sha256": _sha(f"v1-{role}-{index // 32}"),
        "task_identity_sha256": _sha(f"task-{role}-{index // 32}"),
        "entry_index": index,
        "trace_id": f"trace-{role}-{index}",
        "step_index": 0,
        "progress_index": 0,
        "partition_role": role,
        "data_lane": "observed_local_analogue",
        "executor_rule": "atom_restate",
        "model_family": _family(cell),
        "capability_cell_id": cell,
        "source_state_sha256": _sha(f"source-{role}-{index}"),
        "target_state_sha256": _sha(f"target-{role}-{index}"),
        "source_canonical_key": f"source-key-{role}-{index}",
        "canonical_successor_key": f"target-key-{role}-{index}",
        "action_sha256": _sha(f"action-{role}-{index}"),
        "assignment_sha256": _sha(f"assignment-{role}-{index}"),
    }
    return {**body, "p50_entry_sha256": runtime._candidate_identity(body)}


def _prerequisites(families: tuple[str, ...]) -> ProcessV2P50ScopedPrerequisites:
    return ProcessV2P50ScopedPrerequisites(
        process_identity_sha256=_sha("process"),
        active8_completion_sha256=_sha("active8"),
        gate_zero_decision_sha256=_sha("gate0"),
        t1_capacity_policy_sha256=_sha("t1-policy"),
        t1_result_sha256=_sha("t1-result"),
        t1_decision_sha256=_sha("t1-decision"),
        t1_initial_model_state_sha256=_sha("initial"),
        t1_score_revision_receipt_sha256=_sha("score-revision"),
        p50_recipe_policy_sha256=_sha("p50-policy"),
        active_families=families,
        optimizer_steps=50,
        batch_size=64,
    )


def test_selection_is_exactly_50_by_64_and_uses_a_bounded_validation_sentinel(
    monkeypatch,
) -> None:
    cells = (
        "ns:atom_insert:one_neighbor_birth",
        "ns:atom_delete:leaf_death",
        "ns:atom_delete:connected_nonleaf_death",
        "ns:atom_restate:element_identity_change",
        "ns:atom_restate:valence_state_change",
        "ns:bond_reorder:increase",
        "ns:bond_reorder:decrease",
        "ns:bond_reroute:single_atom_acyclic",
        "ns:bond_reroute:multi_atom_acyclic",
        "ns:bond_reroute:single_atom_cyclic",
        "ns:bond_reroute:multi_atom_cyclic",
        "ns:cycle_insert:monocyclic",
        "ns:cycle_insert:nonarticulated_polycyclic",
        "ns:cycle_attach:monocyclic",
        "ns:cycle_attach:nonarticulated_polycyclic",
        "ns:ring_system_restate:aromatization",
        "ns:ring_system_restate:dearomatization",
    )
    families = tuple(dict.fromkeys(_family(cell) for cell in cells))
    policy = {
        "contract_sha256": _sha("p50-policy"),
        "active_families": list(families),
        "optimization": {
            "scheduled_nonterminal_examples": 3200,
            "optimizer_steps": 50,
            "batch_size": 64,
            "seed": 31,
        },
    }
    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    index = 0
    unsupported = "ns:atom_delete:connected_nonleaf_death"
    for cell in cells:
        for _ in range(200):
            train.append(_candidate(cell=cell, role="train", index=index))
            index += 1
        if cell != unsupported:
            for _ in range(4):
                validation.append(_candidate(cell=cell, role="validation", index=index))
                index += 1

    def candidates(_source, *, partition_role):
        return iter(train if partition_role == "train" else validation)

    monkeypatch.setattr(runtime, "_iter_role_candidates", candidates)
    monkeypatch.setattr(runtime, "_required_cells", lambda _root: cells)
    monkeypatch.setattr(
        runtime,
        "load_process_v2_chain_artifact",
        lambda _name, *, repo_root: policy,
    )
    source = SimpleNamespace(repo_root="/unused", plan={"run_identity_sha256": _sha("run")})
    selected = runtime.build_process_v2_p50_selection(
        source, prerequisites=_prerequisites(families)
    )
    assert runtime.validate_process_v2_p50_selection(selected) == selected

    stream = selected["training_stream"]
    assert len(stream) == 3200
    assert {row["optimizer_step"] for row in stream} == set(range(50))
    assert {row["batch_offset"] for row in stream} == set(range(64))
    assert set(selected["training_cell_opportunities"]) == set(cells)
    assert set(selected["training_family_opportunities"]) == set(families)
    assert all(0.0 < float.fromhex(row["time_hex"]) < 1.0 for row in stream)
    assert selected["validation_unsupported_required_cells"] == [unsupported]
    assert len(selected["validation_entry_sha256s"]) == 64
    validation_ids = set(selected["validation_entry_sha256s"])
    validation_rows = [
        entry for entry in selected["entries"] if entry["p50_entry_sha256"] in validation_ids
    ]
    assert len({row["source_state_sha256"] for row in validation_rows}) == 64
    assert all(row["partition_role"] == "validation" for row in validation_rows)
    assert all(
        row["partition_role"] == "train"
        for row in selected["entries"]
        if row["p50_entry_sha256"] not in validation_ids
    )

    tampered = copy.deepcopy(selected)
    tampered["training_stream"][0]["time_hex"] = "not-a-hex-float"
    body = {key: item for key, item in tampered.items() if key != "selection_sha256"}
    tampered["training_stream_sha256"] = runtime.canonical_sha256(tampered["training_stream"])
    tampered["selection_sha256"] = runtime.canonical_sha256(body)
    # Re-seal after updating the dependent stream hash as well.
    body = {key: item for key, item in tampered.items() if key != "selection_sha256"}
    tampered["selection_sha256"] = runtime.canonical_sha256(body)
    with pytest.raises(runtime.ProcessV2P50RuntimeError, match="training time"):
        runtime.validate_process_v2_p50_selection(tampered)


def test_time_derivation_binds_stream_position_and_entry() -> None:
    first = {"p50_entry_sha256": _sha("one")}
    second = {"p50_entry_sha256": _sha("two")}
    observed = runtime._time_hex(stream_index=0, candidate=first, seed=31)
    assert observed == runtime._time_hex(stream_index=0, candidate=first, seed=31)
    assert observed != runtime._time_hex(stream_index=1, candidate=first, seed=31)
    assert observed != runtime._time_hex(stream_index=0, candidate=second, seed=31)


def test_compact_prepared_input_round_trips_only_the_teacher_fiber() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 8)
    source_sha = persistent_slot_state_sha256(state)
    alias = TeacherSuccessorAlias(
        family_name="atom_restate",
        table_name="atom_restate",
        coordinate=(2, 0),
    )
    fiber = TeacherSuccessorFiber(
        source_key=canonical_state_key(state),
        target_key="CCN",
        target_state_sha256=_sha("target-state"),
        aliases=(alias,),
        state_support=StateProductiveSupport(
            source_key=canonical_state_key(state),
            source_state_sha256=source_sha,
        ),
    )
    identifier = _sha("compact-entry")
    entry_body = {
        "p50_entry_sha256": identifier,
        "model_family": "atom_restate",
        "capability_cell_id": "ns:atom_restate:element_identity_change",
        "partition_role": "train",
        "support_time_hex": float(0.5).hex(),
        "source_state_sha256": source_sha,
        "target_state_sha256": fiber.target_state_sha256,
        "successor_canonical_key": fiber.target_key,
        "teacher_action_sha256": _sha("teacher-action"),
        "objective_coefficient": 1,
        "raw_mark_count": 7,
        "decoded_mark_count": 1,
        "marks_by_family": {"atom_restate": 7},
        "production_successor_alias_multiplicity": 1,
        "virtual_alias_count": 0,
        "exact_state": encode_state(state),
        "teacher_successor_fiber": runtime._teacher_fiber_payload(fiber),
        "exact_teacher_alias": runtime._alias_payload(alias),
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    entry = {
        **entry_body,
        "p50_compiled_entry_sha256": runtime.canonical_sha256(entry_body),
    }
    stream = [{"p50_entry_sha256": identifier}]
    body = {
        "schema": runtime.PREPARED_SCHEMA,
        "schema_version": runtime.PREPARED_SCHEMA_VERSION,
        "status": runtime.PREPARED_STATUS,
        **runtime.authority_false_block(),
        "selection_sha256": _sha("selection"),
        "prerequisites_binding_sha256": _sha("prerequisites"),
        "initial_model_state_sha256": _sha("initial"),
        "training_stream": stream,
        "training_stream_sha256": runtime.canonical_sha256(stream),
        "required_cells": [entry_body["capability_cell_id"]],
        "training_family_step_opportunities": {"atom_restate": 1},
        "training_cell_step_opportunities": {entry_body["capability_cell_id"]: 1},
        "validation_entry_sha256s": [],
        "validation_inventory_sha256": runtime.canonical_sha256([]),
        "validation_unsupported_required_cells": [],
        "entry_count": 1,
        "entries": [entry],
    }
    prepared = {**body, "prepared_sha256": runtime.canonical_sha256(body)}
    loaded = runtime.load_process_v2_p50_inputs(prepared)
    assert loaded.fibers_by_id[identifier] == fiber
    assert not hasattr(loaded, "partitions_by_id")


def test_exhaustive_v1_leaf_converts_without_molecular_reenumeration() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 8)
    source_sha = persistent_slot_state_sha256(state)
    target_sha = _sha("target-state")
    action_sha = _sha("teacher-action")
    alias = TeacherSuccessorAlias(
        family_name="atom_restate",
        table_name="atom_restate",
        coordinate=(2, 0),
    )
    mark = CompiledSuccessorMark(
        alias=alias,
        successor_state_sha256=target_sha,
        action_sha256=action_sha,
    )
    partition = CompiledStateSuccessorMap(
        state_support=StateProductiveSupport(
            source_key=canonical_state_key(state),
            source_state_sha256=source_sha,
        ),
        successor_groups=(CanonicalSuccessorAliasGroup(target_key="CCN", marks=(mark,)),),
    )
    task_id = _sha("task")
    candidate_body = {
        "v1_task_identity_sha256": _sha("v1-task"),
        "task_identity_sha256": task_id,
        "entry_index": 0,
        "trace_id": "trace",
        "step_index": 0,
        "progress_index": 0,
        "partition_role": "train",
        "data_lane": "observed_local_analogue",
        "executor_rule": "atom_restate_semantic",
        "model_family": "atom_restate",
        "capability_cell_id": "ns:atom_restate:element_identity_change",
        "source_state_sha256": source_sha,
        "target_state_sha256": target_sha,
        "source_canonical_key": canonical_state_key(state),
        "canonical_successor_key": "CCN",
        "action_sha256": action_sha,
        "assignment_sha256": _sha("assignment"),
    }
    identifier = runtime._candidate_identity(candidate_body)
    candidate = {**candidate_body, "p50_entry_sha256": identifier}
    selection = {
        "selection_sha256": _sha("selection"),
        "entries": [candidate],
    }
    old_body = {
        "p50_entry_sha256": identifier,
        "model_family": "atom_restate",
        "capability_cell_id": candidate["capability_cell_id"],
        "partition_role": "train",
        "support_time_hex": float(0.5).hex(),
        "source_state_sha256": source_sha,
        "target_state_sha256": target_sha,
        "successor_canonical_key": "CCN",
        "teacher_action_sha256": action_sha,
        "objective_coefficient": 1,
        "raw_mark_count": 1,
        "canonical_successor_count": 1,
        "productive_alias_count": 1,
        "production_successor_alias_multiplicity": 1,
        "virtual_alias_count": 0,
        "exact_state": encode_state(state),
        "successor_partition": compiled_successor_map_payload(partition),
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    old_entry = {
        **old_body,
        "p50_compiled_entry_sha256": runtime.canonical_sha256(old_body),
    }
    leaf_body = {
        "task_identity_sha256": task_id,
        "selection_sha256": selection["selection_sha256"],
        "entry_count": 1,
        "entries": [old_entry],
    }
    legacy_leaf = {**leaf_body, "leaf_sha256": runtime.canonical_sha256(leaf_body)}
    converted = runtime.convert_process_v2_p50_v1_leaf(selection, legacy_leaf)
    entry = converted["entries"][0]
    assert "successor_partition" not in entry
    assert entry["teacher_successor_fiber"]["aliases"] == [runtime._alias_payload(alias)]
    assert entry["exact_teacher_alias"] == runtime._alias_payload(alias)
    assert entry["decoded_mark_count"] == 1

    corrupted = copy.deepcopy(legacy_leaf)
    corrupted_entry = corrupted["entries"][0]
    corrupted_entry["raw_mark_count"] = 2
    corrupted_body = {
        key: value for key, value in corrupted_entry.items() if key != "p50_compiled_entry_sha256"
    }
    corrupted_entry["p50_compiled_entry_sha256"] = runtime.canonical_sha256(corrupted_body)
    corrupted_leaf_body = {key: value for key, value in corrupted.items() if key != "leaf_sha256"}
    corrupted["leaf_sha256"] = runtime.canonical_sha256(corrupted_leaf_body)
    with pytest.raises(runtime.ProcessV2P50RuntimeError, match="identity disagrees"):
        runtime.convert_process_v2_p50_v1_leaf(selection, corrupted)
