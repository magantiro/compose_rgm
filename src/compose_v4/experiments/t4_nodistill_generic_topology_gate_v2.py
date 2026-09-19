"""Zero-oracle gate for route-free repeated topology-closing macro plans."""

from __future__ import annotations

import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer_v2 import (
    propose_generic_topology_macro_programs,
)
from compose_v4.experiments.t4_fiber_campaign import COMPOSE_VALID, Fiber
from compose_v4.experiments.t4_nodistill_generic_composer_gate import (
    _arm_summary,
    _candidate_row,
    _publish_once,
    _read_envelope,
    _source_cells,
    _teacher_descriptors,
    sha256_file,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

CONTRACT_RELATIVE_PATH = "configs/t4_nodistill_generic_topology_gate_v2.json"
ARTIFACT_ROOT_RELATIVE_PATH = "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1"
LOCK_SCHEMA = "t4_nodistill_generic_topology_candidate_lock_v2"
RESULT_SCHEMA = "t4_nodistill_generic_topology_gate_result_v2"


def load_contract(repository_root: Path) -> tuple[dict[str, Any], str]:
    contract, contract_identity = _read_envelope(repository_root / CONTRACT_RELATIVE_PATH)
    if contract.get("schema_version") != ("t4_nodistill_generic_topology_gate_contract_v2"):
        raise ValueError("unexpected generic topology-gate schema")
    if contract.get("status") != "FROZEN_ZERO_ORACLE_SUPPORT_GATE":
        raise ValueError("generic topology-gate contract is not frozen")
    if contract.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }:
        raise ValueError("generic topology-gate contract is not zero-oracle")
    if not all(contract.get("runtime_prohibitions", {}).values()):
        raise ValueError("generic topology-gate runtime prohibition was weakened")
    for row in contract["inputs"].values():
        path = repository_root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"generic topology-gate input changed: {path}")
    attempt_1, attempt_1_identity = _read_envelope(
        repository_root / contract["inputs"]["attempt_1_result"]["path"]
    )
    if (
        attempt_1_identity != contract["inputs"]["attempt_1_result"]["payload_sha256"]
        or attempt_1.get("decision") != "PASS_SUPPORT_GATE"
    ):
        raise ValueError("attempt-1 support result changed")
    return contract, contract_identity


def _attempt_1_locks(repository_root: Path, contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result, _ = _read_envelope(repository_root / contract["inputs"]["attempt_1_result"]["path"])
    locks = {}
    for cell in result["cells"]:
        path = repository_root / cell["candidate_lock"]["path"]
        if sha256_file(path) != cell["candidate_lock"]["sha256"]:
            raise ValueError(f"attempt-1 candidate lock changed: {path}")
        payload, payload_identity = _read_envelope(path)
        if payload_identity != cell["candidate_lock"]["payload_sha256"]:
            raise ValueError(f"attempt-1 candidate payload changed: {path}")
        locks[cell["cell_key"]] = payload
    return locks


def _macro_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    large_growth_multicycle = [
        row
        for row in candidates
        if row["realized_primitive_band"] == "large"
        and int(row["delta_heavy_atoms"]) >= 7
        and int(row["delta_cycle_rank"]) >= 2
    ]
    eligible = [row for row in large_growth_multicycle if row["eligible"]]
    plans = Counter(
        row["metadata"]["macro_plan"]["name"]
        for row in candidates
        if "macro_plan" in row["metadata"]
    )
    fields = [
        row["metadata"]["observed_macro_fields"]
        for row in candidates
        if "observed_macro_fields" in row["metadata"]
    ]
    return {
        "candidate_count": len(candidates),
        "unique_endpoint_count": len({row["endpoint_key_sha256"] for row in candidates}),
        "exact_execution_precision": (
            sum(bool(row["exact_execution_verified"]) for row in candidates) / len(candidates)
            if candidates
            else None
        ),
        "cycle_gain_counts": dict(
            sorted(Counter(int(row["delta_cycle_rank"]) for row in candidates).items())
        ),
        "macro_plan_counts": dict(sorted(plans.items())),
        "macro_plan_diversity": len(plans),
        "large_growth_multicycle_candidates": len(large_growth_multicycle),
        "eligible_large_growth_multicycle_candidates": len(eligible),
        "eligible_large_growth_multicycle_endpoint_sha256": [
            row["endpoint_key_sha256"] for row in eligible
        ],
        "observed_retained_fraction_range": (
            [
                min(float(row["retained_fraction"]) for row in fields),
                max(float(row["retained_fraction"]) for row in fields),
            ]
            if fields
            else None
        ),
        "observed_interface_count_range": (
            [
                min(int(row["created_attachment_interface_count"]) for row in fields),
                max(int(row["created_attachment_interface_count"]) for row in fields),
            ]
            if fields
            else None
        ),
    }


def lock_cell(*, repository_root: Path, cell_key: str, output_path: Path) -> dict[str, Any]:
    """Seal a route-free macro-plan pool before teacher data is imported."""

    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the frozen topology panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    fiber = Fiber(cell["source_smiles"], float(cell["delta"]), support=COMPOSE_VALID)
    proposal = propose_generic_topology_macro_programs(
        source,
        seed=int(cell["controller_seed"]),
        attempts_per_plan=int(contract["proposal"]["attempts_per_plan"]),
        maximum_primitives=int(contract["support"]["maximum_primitives"]),
        maximum_blocks=int(contract["support"]["maximum_blocks"]),
    )
    v2_rows = [
        _candidate_row(
            arm="generic_topology_macro_v2",
            source=source,
            endpoint=row.endpoint,
            actions=row.actions,
            families=row.families,
            requested_scale=row.requested_scale,
            metadata=row.metadata,
            program=row.program,
            program_graph=row.program_graph,
            fiber=fiber,
        )
        for row in proposal.proposals
    ]

    attempt_1 = _attempt_1_locks(repository_root, contract)[cell_key]
    old_rows = [
        row
        for row in attempt_1["arms"]["generic_complete_program_composer"]["candidates"]
        if row["realized_primitive_band"] in {"medium", "large"}
    ]
    matched_rows = old_rows[: len(v2_rows)]
    v2_rows = v2_rows[: len(matched_rows)]
    payload = {
        "schema_version": LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle candidate lock before teacher loading",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "delta": float(cell["delta"]),
        "controller_seed": int(cell["controller_seed"]),
        "source_state": attempt_1["source_state"],
        "source_key_sha256": identity(canonical_state_key(source)),
        "proposal_inputs": {
            "source_graph": True,
            "seed": True,
            "task_cell_or_target": False,
            "trajectory_distilled_templates": False,
            "route_weights_or_ids": False,
            "teacher_or_winner_endpoints": False,
            "objective_or_comparator_scores": False,
        },
        "arms": {
            "generic_topology_macro_v2": {
                "telemetry": proposal.telemetry,
                "candidates": v2_rows,
            },
            "attempt_1_matched_medium_large": {"candidates": matched_rows},
        },
        "preteacher_macro_support": _macro_summary(v2_rows),
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "candidate_count_per_arm": len(v2_rows),
        "eligible_large_growth_multicycle": payload["preteacher_macro_support"][
            "eligible_large_growth_multicycle_candidates"
        ],
    }


def evaluate_locks(
    *, repository_root: Path, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    locks = {}
    for cell_key in cells:
        path = artifact_root / "locks" / f"{cell_key}.json.gz"
        payload, payload_identity = _read_envelope(path)
        if (
            payload.get("schema_version") != LOCK_SCHEMA
            or payload.get("cell_key") != cell_key
            or payload.get("contract_payload_sha256") != contract_identity
        ):
            raise ValueError(f"generic topology candidate lock changed: {path}")
        locks[cell_key] = (payload, payload_identity, path)

    teachers_by_source = _teacher_descriptors(repository_root)
    result_cells = []
    aggregate_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_teachers = []
    for cell_key, (lock, lock_identity, path) in locks.items():
        source = decode_state(lock["source_state"])
        teachers = teachers_by_source.get(canonical_state_key(source), [])
        if not teachers:
            raise ValueError(f"no teacher diagnostic matches {cell_key}")
        summaries = {}
        for arm, arm_payload in lock["arms"].items():
            candidates = arm_payload["candidates"]
            summaries[arm] = _arm_summary(candidates, teachers)
            aggregate_candidates[arm].extend(candidates)
        aggregate_teachers.extend(teachers)
        result_cells.append(
            {
                "cell_key": cell_key,
                "delta": lock["delta"],
                "candidate_lock": {
                    "path": str(path.relative_to(repository_root)),
                    "sha256": sha256_file(path),
                    "payload_sha256": lock_identity,
                },
                "preteacher_macro_support": lock["preteacher_macro_support"],
                "teacher_routes": len(teachers),
                "arms": summaries,
            }
        )

    aggregate = {
        arm: _arm_summary(rows, aggregate_teachers)
        for arm, rows in sorted(aggregate_candidates.items())
    }
    parp = {row["cell_key"]: row for row in result_cells if row["cell_key"].startswith("parp1_")}
    cycle_counts = Counter()
    for row in parp.values():
        cycle_counts.update(
            {
                int(key): int(value)
                for key, value in row["preteacher_macro_support"]["cycle_gain_counts"].items()
            }
        )
    gates = {
        "complete_six_cell_lock_census": len(result_cells) == 6,
        "matched_candidate_budgets": all(
            len(lock[0]["arms"]["generic_topology_macro_v2"]["candidates"])
            == len(lock[0]["arms"]["attempt_1_matched_medium_large"]["candidates"])
            for lock in locks.values()
        ),
        "exact_execution_precision_one": (
            aggregate["generic_topology_macro_v2"]["exact_execution"]["precision"] == 1.0
        ),
        "preteacher_plus_2_and_plus_3_cycle_support_on_parp_panel": (
            cycle_counts[2] > 0 and cycle_counts[3] > 0
        ),
        "eligible_large_growth_multicycle_support_on_parp1_0_and_parp1_1": all(
            parp[key]["preteacher_macro_support"]["eligible_large_growth_multicycle_candidates"] > 0
            for key in ("parp1_0_d04", "parp1_1_d04")
        ),
        "nonzero_macro_plan_diversity": (
            aggregate["generic_topology_macro_v2"]["unique_composition_count"] > 1
        ),
        "no_route_or_objective_runtime_inputs": all(
            all(
                not bool(lock[0]["proposal_inputs"][field])
                for field in (
                    "task_cell_or_target",
                    "trajectory_distilled_templates",
                    "route_weights_or_ids",
                    "teacher_or_winner_endpoints",
                    "objective_or_comparator_scores",
                )
            )
            for lock in locks.values()
        ),
    }
    passed = all(gates.values())
    code_inputs = (
        "src/compose_v4/control/generic_complete_program_composer_v2.py",
        "src/compose_v4/experiments/t4_nodistill_generic_topology_gate_v2.py",
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle route-free topology support gate",
        "decision": "PASS_TOPOLOGY_SUPPORT_GATE" if passed else "NO_PROMOTION",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(repository_root / CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "candidate_lock_ordering": (
            "all six candidate locks were sealed and validated before teacher "
            "descriptors were imported or read"
        ),
        "cells": result_cells,
        "aggregate": aggregate,
        "gates": {"values": gates, "passed": passed},
        "implementation": {
            "code_revision_before_result_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "code_sha256": {path: sha256_file(repository_root / path) for path in code_inputs},
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "claim_boundary": (
            "Support and post-lock descriptor evidence only. Eligibility is a free "
            "endpoint property; no docking utility was measured."
        ),
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "decision": payload["decision"],
        "payload_sha256": payload_sha256,
        "gates": payload["gates"],
    }


__all__ = [
    "ARTIFACT_ROOT_RELATIVE_PATH",
    "CONTRACT_RELATIVE_PATH",
    "evaluate_locks",
    "load_contract",
    "lock_cell",
]
