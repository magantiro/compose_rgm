"""Focused tests for the minimal Process-V2 P50 fast path."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

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
