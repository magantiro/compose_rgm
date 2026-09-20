"""Frozen zero-oracle gate for the route-free feasibility-headroom v4 union."""

from __future__ import annotations

import copy
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer import primitive_scale
from compose_v4.control.generic_feasibility_headroom_composer_v4 import (
    FreeFeasibilitySpec,
    allocate_feasibility_headroom_plans,
    free_endpoint_headrooms,
    propose_feasibility_headroom_plan,
)
from compose_v4.control.generic_retained_interface_composer_v3 import (
    _created_attachment_interface_count,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.control.nodistill_joint_stop_adapter_v1 import (
    JointStopAdapterSettings,
    propose_joint_stop_candidates,
)
from compose_v4.experiments.t4_nodistill_generic_composer_gate import (
    _arm_summary,
    _publish_once,
    _read_envelope,
    _teacher_descriptors,
    sha256_file,
)
from compose_v4.experiments.t4_nodistill_retained_interface_gate_v3 import (
    _apply_post_generation_production_admission,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT_RELATIVE_PATH = "configs/t4_nodistill_feasibility_headroom_gate_v4.json"
ARTIFACT_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_headroom_gate_v4/attempt_1"
)
PLAN_LOCK_SCHEMA = "t4_nodistill_feasibility_headroom_plan_lock_v4"
JOINT_LOCK_SCHEMA = "t4_nodistill_feasibility_headroom_joint_lock_v4"
CELL_LOCK_SCHEMA = "t4_nodistill_feasibility_headroom_candidate_lock_v4"
RESULT_SCHEMA = "t4_nodistill_feasibility_headroom_gate_result_v4"


def load_contract(repository_root: Path) -> tuple[dict[str, Any], str]:
    contract, contract_identity = _read_envelope(
        repository_root / CONTRACT_RELATIVE_PATH
    )
    if contract.get("schema_version") != (
        "t4_nodistill_feasibility_headroom_gate_contract_v4"
    ):
        raise ValueError("unexpected feasibility-headroom v4 contract schema")
    if contract.get("status") != "FROZEN_ZERO_ORACLE_SUPPORT_GATE":
        raise ValueError("feasibility-headroom v4 contract is not frozen")
    if contract.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }:
        raise ValueError("feasibility-headroom v4 contract is not zero-oracle")
    if not all(contract.get("runtime_prohibitions", {}).values()):
        raise ValueError("feasibility-headroom v4 runtime prohibition was weakened")
    for binding in contract["inputs"].values():
        path = repository_root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feasibility-headroom v4 input changed: {path}")
        if binding.get("payload_sha256") is not None:
            _payload, observed = _read_envelope(path)
            if observed != binding["payload_sha256"]:
                raise ValueError(f"v4 input payload changed: {path}")
    return contract, contract_identity


def _source_cells(repository_root: Path, contract: dict[str, Any]) -> dict[str, dict]:
    registry = json.loads(
        (repository_root / contract["inputs"]["source_registry"]["path"]).read_text()
    )
    selected = {}
    for declared in contract["cells"]:
        index = int(declared["source_index"])
        source = registry[index]
        if int(source["idx"]) != index:
            raise ValueError(f"source-registry index drift: {index}")
        selected[declared["cell_key"]] = {
            **copy.deepcopy(declared),
            "source_smiles": str(source["smiles"]),
            "source_target": str(source["target"]),
        }
    if len(selected) != 5:
        raise ValueError("feasibility-headroom v4 requires five panel cells")
    return selected


def _source(repository_root: Path, cell_key: str) -> tuple[dict, dict, Any, str]:
    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the frozen v4 panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    return contract, cell, source, contract_identity


def allocated_plan_names(*, repository_root: Path, cell_key: str) -> tuple[str, ...]:
    contract, cell, source, _identity = _source(repository_root, cell_key)
    support = contract["support"]
    proposal = contract["proposal"]
    _features, _headrooms, plans, _allocation = allocate_feasibility_headroom_plans(
        source,
        feasibility=FreeFeasibilitySpec(
            similarity_minimum=float(cell["similarity_minimum"]),
            qed_minimum=float(support["qed_minimum"]),
            sa_maximum=float(support["sa_maximum"]),
            heavy_atom_maximum=int(support["representable_heavy_atom_ceiling"]),
        ),
        attempts_per_plan=int(proposal["attempts_per_local_plan"]),
        candidate_quota_per_plan=int(proposal["candidate_quota_per_local_plan"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    return tuple(plan.name for plan in plans)


def _row(
    source,
    candidate,
    expert: str,
    *,
    feasibility: FreeFeasibilitySpec | None = None,
    available_cycle_capacity: int = 2,
) -> dict[str, Any]:
    before = topology(source)
    after = topology(candidate.endpoint)
    changes = candidate.actual_changes
    retained = 1.0 - int(changes["deleted_original_atoms"]) / max(
        1, int(before["n_heavy"])
    )
    interfaces = _created_attachment_interface_count(source, candidate.actions)
    metadata = copy.deepcopy(candidate.metadata)
    macro_plan = copy.deepcopy(metadata["macro_plan"])
    macro_identity = str(macro_plan.get("identity") or identity(macro_plan))
    mode = str(macro_plan.get("mode") or expert)
    primitive_count = len(candidate.actions)
    block_count = len(candidate.program["blocks"])
    extent = max(
        1,
        int(changes["surviving_new_atoms"]) + int(changes["deleted_original_atoms"]),
        len(changes["changed_original_slots"]),
    )
    if feasibility is not None and "endpoint_headrooms" not in metadata:
        desired_cycle = max(0, int(after["cycle_rank"] - before["cycle_rank"]))
        metadata["endpoint_headrooms"] = free_endpoint_headrooms(
            source,
            candidate.endpoint,
            feasibility=feasibility,
            retained_fraction=retained,
            interface_count=interfaces,
            desired_cycle_rank=desired_cycle,
            available_cycle_capacity=available_cycle_capacity,
        )
        metadata["endpoint_admission_applied_during_generation"] = False
    return {
        "expert": expert,
        "endpoint_state": encode_state(candidate.endpoint),
        "endpoint_key_sha256": identity(canonical_state_key(candidate.endpoint)),
        "canonical_smiles": molecular_graph_to_smiles(candidate.endpoint),
        "actions": list(candidate.actions),
        "program": candidate.program,
        "program_graph": candidate.program_graph,
        "families": list(candidate.families),
        "module_count": len(candidate.families),
        "requested_scale": getattr(
            candidate, "requested_scale", primitive_scale(primitive_count)
        ),
        "realized_primitive_count": primitive_count,
        "realized_primitive_band": getattr(
            candidate, "realized_scale", primitive_scale(primitive_count)
        ),
        "structural_extent": extent,
        "structural_extent_band": (
            "small" if extent <= 3 else ("medium" if extent <= 11 else "large")
        ),
        "delta_heavy_atoms": int(after["n_heavy"] - before["n_heavy"]),
        "delta_cycle_rank": int(after["cycle_rank"] - before["cycle_rank"]),
        "delta_ring_systems": int(after["n_ring_systems"] - before["n_ring_systems"]),
        "endpoint_heavy_atoms": int(after["n_heavy"]),
        "actual_changes": changes,
        "created_handle_dependencies": int(
            metadata.get("created_handle_dependencies", 0)
        ),
        "retained_fraction": retained,
        "created_attachment_interface_count": interfaces,
        "block_count": block_count,
        "macro_plan_identity": macro_identity,
        "macro_mode": mode,
        "macro_plan": macro_plan,
        "exact_execution_verified": bool(
            metadata.get("exact_execution_verified", True)
        ),
        "eligible": False,
        "fiber_admitted": False,
        "archive_ready": False,
        "endpoint_properties": None,
        "fingerprint": None,
        "metadata": metadata,
    }


def _structural_signature(row: dict[str, Any]) -> str:
    molecule = Chem.MolFromSmiles(row["canonical_smiles"])
    if molecule is None:
        raise RuntimeError("exact v4 endpoint did not parse during signature audit")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return identity(
        {
            "murcko_scaffold": scaffold,
            "delta_heavy_atoms": int(row["delta_heavy_atoms"]),
            "delta_cycle_rank": int(row["delta_cycle_rank"]),
            "created_attachment_interfaces": int(
                row["created_attachment_interface_count"]
            ),
        }
    )


def lock_plan(
    *, repository_root: Path, cell_key: str, plan_name: str, output_path: Path
) -> dict[str, Any]:
    contract, cell, source, contract_identity = _source(repository_root, cell_key)
    support, proposal = contract["support"], contract["proposal"]
    batch = propose_feasibility_headroom_plan(
        source,
        seed=int(cell["controller_seed"]),
        feasibility=FreeFeasibilitySpec(
            similarity_minimum=float(cell["similarity_minimum"]),
            qed_minimum=float(support["qed_minimum"]),
            sa_maximum=float(support["sa_maximum"]),
            heavy_atom_maximum=int(support["representable_heavy_atom_ceiling"]),
        ),
        plan_name=plan_name,
        attempts_per_plan=int(proposal["attempts_per_local_plan"]),
        candidate_quota_per_plan=int(proposal["candidate_quota_per_local_plan"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    rows = [
        _row(source, candidate, "feasibility_headroom_v4")
        for candidate in batch.proposals
    ]
    payload = {
        "schema_version": PLAN_LOCK_SCHEMA,
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "plan_name": plan_name,
        "source_key_sha256": identity(canonical_state_key(source)),
        "telemetry": batch.telemetry,
        "candidates": rows,
        "endpoint_admission_applied": False,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_identity = _publish_once(output_path, payload)
    return {
        "kind": "plan",
        "cell_key": cell_key,
        "unit": plan_name,
        "path": str(output_path),
        "payload_sha256": payload_identity,
        "exact_candidates": len(rows),
        "raw_compile_successes": int(batch.telemetry["raw_compile_successes"]),
    }


def lock_joint_stop(
    *, repository_root: Path, cell_key: str, output_path: Path
) -> dict[str, Any]:
    contract, cell, source, contract_identity = _source(repository_root, cell_key)
    support = contract["support"]
    feasibility = FreeFeasibilitySpec(
        similarity_minimum=float(cell["similarity_minimum"]),
        qed_minimum=float(support["qed_minimum"]),
        sa_maximum=float(support["sa_maximum"]),
        heavy_atom_maximum=int(support["representable_heavy_atom_ceiling"]),
    )
    _features, source_headrooms, _plans, _allocation = (
        allocate_feasibility_headroom_plans(
            source,
            feasibility=feasibility,
            attempts_per_plan=int(contract["proposal"]["attempts_per_local_plan"]),
            candidate_quota_per_plan=int(
                contract["proposal"]["candidate_quota_per_local_plan"]
            ),
            maximum_primitives=int(support["maximum_primitives"]),
            maximum_blocks=int(support["maximum_blocks"]),
        )
    )
    settings = JointStopAdapterSettings(**contract["proposal"]["joint_stop"])
    batch = propose_joint_stop_candidates(
        source, seed=int(cell["controller_seed"]), settings=settings
    )
    rows = [
        _row(
            source,
            candidate,
            "joint_stop_v1",
            feasibility=feasibility,
            available_cycle_capacity=int(source_headrooms["available_cycle_capacity"]),
        )
        for candidate in batch.candidates
    ]
    payload = {
        "schema_version": JOINT_LOCK_SCHEMA,
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "source_key_sha256": identity(canonical_state_key(source)),
        "telemetry": batch.telemetry,
        "candidates": rows,
        "endpoint_admission_applied": False,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_identity = _publish_once(output_path, payload)
    return {
        "kind": "joint_stop",
        "cell_key": cell_key,
        "unit": "joint_stop",
        "path": str(output_path),
        "payload_sha256": payload_identity,
        "exact_candidates": len(rows),
        "raw_compile_successes": int(batch.telemetry["raw_exact_unique"]),
    }


def _range(rows: list[dict], field: str) -> list[float] | None:
    return (
        None
        if not rows
        else [min(row[field] for row in rows), max(row[field] for row in rows)]
    )


def _summary(rows: list[dict], admission: dict, raw_compile: int) -> dict[str, Any]:
    ready = [row for row in rows if row["archive_ready"]]
    return {
        "raw_compile_successes": raw_compile,
        "exact_unique_candidates": len(rows),
        "fiber_admitted_candidates": sum(bool(row["fiber_admitted"]) for row in rows),
        "archive_ready_unique_candidates": len(ready),
        "archive_ready_macro_plan_identities": sorted(
            {row["macro_plan_identity"] for row in ready}
        ),
        "archive_ready_macro_plan_identity_count": len(
            {row["macro_plan_identity"] for row in ready}
        ),
        "archive_ready_modes": sorted({row["macro_mode"] for row in ready}),
        "archive_ready_mode_count": len({row["macro_mode"] for row in ready}),
        "archive_ready_structural_signatures": sorted(
            {row["structural_signature"] for row in ready}
        ),
        "archive_ready_structural_signature_count": len(
            {row["structural_signature"] for row in ready}
        ),
        "archive_ready_expert_counts": dict(
            sorted(Counter(row["expert"] for row in ready).items())
        ),
        "all_candidate_delta_heavy_atom_range": _range(rows, "delta_heavy_atoms"),
        "archive_ready_delta_heavy_atom_range": _range(ready, "delta_heavy_atoms"),
        "archive_ready_delta_cycle_rank_range": _range(ready, "delta_cycle_rank"),
        "archive_ready_retained_fraction_range": _range(ready, "retained_fraction"),
        "archive_ready_interface_count_range": _range(
            ready, "created_attachment_interface_count"
        ),
        "primitive_count_range": _range(rows, "realized_primitive_count"),
        "block_count_range": _range(rows, "block_count"),
        "production_endpoint_admission": admission,
    }


def merge_cell_shards(
    *, repository_root: Path, cell_key: str, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    _contract, cell, source, contract_identity = _source(repository_root, cell_key)
    rows: list[dict] = []
    receipts = []
    raw_compile = 0
    for plan_name in allocated_plan_names(
        repository_root=repository_root, cell_key=cell_key
    ):
        path = artifact_root / "plans" / cell_key / f"{plan_name}.json.gz"
        payload, payload_identity = _read_envelope(path)
        if (
            payload.get("schema_version") != PLAN_LOCK_SCHEMA
            or payload.get("contract_payload_sha256") != contract_identity
            or payload.get("cell_key") != cell_key
            or payload.get("plan_name") != plan_name
            or payload.get("endpoint_admission_applied") is not False
        ):
            raise ValueError(f"v4 plan lock changed: {path}")
        rows.extend(payload["candidates"])
        raw_compile += int(payload["telemetry"]["raw_compile_successes"])
        receipts.append(
            {
                "kind": "plan",
                "unit": plan_name,
                "path": str(path.relative_to(repository_root)),
                "sha256": sha256_file(path),
                "payload_sha256": payload_identity,
            }
        )
    joint_path = artifact_root / "joint_stop" / f"{cell_key}.json.gz"
    joint, joint_identity = _read_envelope(joint_path)
    if (
        joint.get("schema_version") != JOINT_LOCK_SCHEMA
        or joint.get("contract_payload_sha256") != contract_identity
        or joint.get("cell_key") != cell_key
        or joint.get("endpoint_admission_applied") is not False
    ):
        raise ValueError(f"v4 joint-STOP lock changed: {joint_path}")
    rows.extend(joint["candidates"])
    raw_compile += int(joint["telemetry"]["raw_exact_unique"])
    receipts.append(
        {
            "kind": "joint_stop",
            "unit": "joint_stop",
            "path": str(joint_path.relative_to(repository_root)),
            "sha256": sha256_file(joint_path),
            "payload_sha256": joint_identity,
        }
    )

    # Canonical union occurs only after both route-free experts finish.  Preserve all
    # colliding origins so plan-coverage evidence is not erased by endpoint dedupe.
    by_endpoint: dict[str, dict] = {}
    for row in rows:
        key = row["endpoint_key_sha256"]
        if key not in by_endpoint:
            selected = copy.deepcopy(row)
            selected["proposal_origins"] = [
                {
                    "expert": row["expert"],
                    "macro_plan_identity": row["macro_plan_identity"],
                    "macro_mode": row["macro_mode"],
                }
            ]
            by_endpoint[key] = selected
        else:
            by_endpoint[key]["proposal_origins"].append(
                {
                    "expert": row["expert"],
                    "macro_plan_identity": row["macro_plan_identity"],
                    "macro_mode": row["macro_mode"],
                }
            )
    raw_rows = list(by_endpoint.values())
    admitted_rows, admission = _apply_post_generation_production_admission(
        raw_rows,
        source_smiles=cell["source_smiles"],
        delta=float(cell["similarity_minimum"]),
    )
    for row in admitted_rows:
        row["structural_signature"] = _structural_signature(row)
    summary = _summary(admitted_rows, admission, raw_compile)
    payload = {
        "schema_version": CELL_LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle v4 lock sealed before teacher loading",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "evidence_role": cell["evidence_role"],
        "similarity_minimum": float(cell["similarity_minimum"]),
        "controller_seed": int(cell["controller_seed"]),
        "source_state": encode_state(source),
        "source_key_sha256": identity(canonical_state_key(source)),
        "source_smiles_sha256": identity(cell["source_smiles"]),
        "proposal_inputs": {
            "source_graph": True,
            "seed": True,
            "generic_numeric_constraint_thresholds": True,
            "task_cell_or_target": False,
            "route_or_template": False,
            "teacher_action_endpoint_or_proximity": False,
            "objective_comparator_or_docking_score": False,
            "fitted_weights": False,
        },
        "generation_then_admission": {
            "both_experts_finished_before_canonical_union": True,
            "generation_completed_before_endpoint_admission": True,
            "candidate_count_before_union": len(rows),
            "candidate_count_after_union": len(raw_rows),
            "admission": admission,
            "durable_receipts": receipts,
        },
        "preteacher_support": summary,
        "candidates": admitted_rows,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_identity = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_identity,
        "exact_candidates": len(admitted_rows),
        "archive_ready": summary["archive_ready_unique_candidates"],
        "plans": summary["archive_ready_macro_plan_identity_count"],
        "modes": summary["archive_ready_mode_count"],
        "structural_signatures": summary["archive_ready_structural_signature_count"],
        "receipts": len(receipts),
    }


def _validate_lock(lock: dict, cell_key: str, contract_identity: str) -> None:
    if (
        lock.get("schema_version") != CELL_LOCK_SCHEMA
        or lock.get("cell_key") != cell_key
        or lock.get("contract_payload_sha256") != contract_identity
        or not lock["generation_then_admission"][
            "both_experts_finished_before_canonical_union"
        ]
        or not lock["generation_then_admission"][
            "generation_completed_before_endpoint_admission"
        ]
    ):
        raise ValueError(f"v4 candidate lock changed: {cell_key}")


def evaluate_locks(
    *, repository_root: Path, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    locks = {}
    for cell_key in cells:
        path = artifact_root / "locks" / f"{cell_key}.json.gz"
        lock, lock_identity = _read_envelope(path)
        _validate_lock(lock, cell_key, contract_identity)
        locks[cell_key] = (lock, lock_identity, path)

    # Teacher descriptors are unavailable to generation and are first read here.
    teachers_by_source = _teacher_descriptors(repository_root)
    result_cells = []
    all_candidates, all_ready, all_teachers = [], [], []
    for cell_key, (lock, lock_identity, path) in locks.items():
        source = decode_state(lock["source_state"])
        teachers = teachers_by_source.get(canonical_state_key(source), [])
        if not teachers:
            raise ValueError(f"no teacher diagnostic matches {cell_key}")
        candidates = lock["candidates"]
        ready = [row for row in candidates if row["archive_ready"]]
        all_candidates.extend(candidates)
        all_ready.extend(ready)
        all_teachers.extend(teachers)
        result_cells.append(
            {
                "cell_key": cell_key,
                "evidence_role": lock["evidence_role"],
                "similarity_minimum": lock["similarity_minimum"],
                "candidate_lock": {
                    "path": str(path.relative_to(repository_root)),
                    "sha256": sha256_file(path),
                    "payload_sha256": lock_identity,
                },
                "preteacher_support": lock["preteacher_support"],
                "teacher_routes": len(teachers),
                "post_lock_teacher_comparison": {
                    "all_exact_candidates": _arm_summary(candidates, teachers),
                    "archive_ready_candidates": _arm_summary(ready, teachers),
                },
            }
        )
    motivating = [
        row for row in result_cells if row["evidence_role"] == "motivating_growth_miss"
    ]
    gate = contract["evaluation"]["prospective_support_gate"]
    exact_denominator = len(all_candidates)
    gates = {
        "complete_five_cell_lock_census": len(result_cells) == 5,
        "exact_replay_precision_one": bool(
            exact_denominator
            and sum(bool(row["exact_execution_verified"]) for row in all_candidates)
            / exact_denominator
            == float(gate["exact_replay_precision"])
        ),
        "minimum_archive_ready_support_on_all_three_motivating_cells": len(motivating)
        == 3
        and all(
            row["preteacher_support"]["archive_ready_unique_candidates"]
            >= int(gate["minimum_archive_ready_unique_endpoints_per_motivating_cell"])
            for row in motivating
        ),
        "minimum_three_plan_identities_on_all_three_motivating_cells": len(motivating)
        == 3
        and all(
            row["preteacher_support"]["archive_ready_macro_plan_identity_count"]
            >= int(
                gate["minimum_archive_ready_macro_plan_identities_per_motivating_cell"]
            )
            for row in motivating
        ),
        "minimum_three_modes_on_all_three_motivating_cells": len(motivating) == 3
        and all(
            row["preteacher_support"]["archive_ready_mode_count"]
            >= int(gate["minimum_archive_ready_macro_modes_per_motivating_cell"])
            for row in motivating
        ),
        "minimum_three_structural_signatures_on_all_three_motivating_cells": len(
            motivating
        )
        == 3
        and all(
            row["preteacher_support"]["archive_ready_structural_signature_count"]
            >= int(
                gate["minimum_archive_ready_structural_signatures_per_motivating_cell"]
            )
            for row in motivating
        ),
        "all_endpoints_within_40_heavy_atoms": all(
            int(row["endpoint_heavy_atoms"]) <= 40 for row in all_candidates
        ),
        "all_programs_within_32_primitives_and_8_blocks": all(
            1 <= int(row["realized_primitive_count"]) <= 32
            and 1 <= int(row["block_count"]) <= 8
            for row in all_candidates
        ),
        "generation_union_and_admission_order_unchanged": all(
            lock[0]["generation_then_admission"][
                "both_experts_finished_before_canonical_union"
            ]
            and lock[0]["generation_then_admission"][
                "generation_completed_before_endpoint_admission"
            ]
            for lock in locks.values()
        ),
        "no_teacher_route_objective_or_cell_runtime_input": all(
            not any(
                lock[0]["proposal_inputs"][field]
                for field in (
                    "task_cell_or_target",
                    "route_or_template",
                    "teacher_action_endpoint_or_proximity",
                    "objective_comparator_or_docking_score",
                    "fitted_weights",
                )
            )
            for lock in locks.values()
        ),
        "post_lock_teacher_descriptor_coverage_reported": all(
            "teacher_coarse_descriptor_covered"
            in row["post_lock_teacher_comparison"]["archive_ready_candidates"][
                "teacher_descriptor_support"
            ]
            for row in result_cells
        ),
    }
    passed = all(gates.values())
    code_inputs = (
        "src/compose_v4/control/generic_feasibility_headroom_composer_v4.py",
        "src/compose_v4/experiments/t4_nodistill_feasibility_headroom_gate_v4.py",
        "tools/t4_nodistill_feasibility_headroom_gate_v4.py",
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle route-free feasibility-headroom support gate",
        "decision": (
            "PASS_FEASIBILITY_HEADROOM_SUPPORT_GATE" if passed else "NO_PROMOTION"
        ),
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(repository_root / CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "candidate_lock_ordering": (
            "all five autonomous locks sealed and validated before teacher descriptors"
        ),
        "cells": result_cells,
        "aggregate": {
            "all_exact_candidates": _arm_summary(all_candidates, all_teachers),
            "archive_ready_candidates": _arm_summary(all_ready, all_teachers),
            "archive_ready_by_cell": {
                row["cell_key"]: row["preteacher_support"][
                    "archive_ready_unique_candidates"
                ]
                for row in result_cells
            },
        },
        "gates": {"values": gates, "passed": passed},
        "implementation": {
            "code_revision_before_result_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "code_sha256": {
                path: sha256_file(repository_root / path) for path in code_inputs
            },
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
        "claim_boundary": (
            "Proposal support, exact replay and free post-generation admission only. "
            "No scored selection, production integration or exact-teacher recovery is "
            "claimed or required."
        ),
    }
    payload_identity = _publish_once(output_path, payload)
    return {
        "decision": payload["decision"],
        "payload_sha256": payload_identity,
        "gates": payload["gates"],
    }


__all__ = [
    "ARTIFACT_ROOT_RELATIVE_PATH",
    "allocated_plan_names",
    "evaluate_locks",
    "load_contract",
    "lock_joint_stop",
    "lock_plan",
    "merge_cell_shards",
]
