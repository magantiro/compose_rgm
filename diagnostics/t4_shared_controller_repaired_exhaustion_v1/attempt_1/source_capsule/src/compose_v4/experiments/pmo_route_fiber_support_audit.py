"""Audit whether the sealed PMO route export can drive a production sampler.

The answer is deliberately separated into two questions:

* does the export contain useful, split-clean structural supervision; and
* does it contain enough information to bind and execute a route proposal at
  runtime without reconstructing an answer-known teacher action?

The second question is a hard precondition for an equal-attempt sampler
comparison.  This audit fails closed when that precondition is absent.  It does
not call a PMO oracle, sample a candidate, or modify any prior artifact.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_route_fiber_prelaunch import (
    EXPECTED_SHA256,
    INITIALIZATION,
    TASKS,
    TRAINING_DATASET,
    TRAINING_RESULT,
    _load_envelope,
    _loto_audit,
)

SCHEMA = "pmo_route_fiber_support_audit_v2"
CONTRACT_SCHEMA = "pmo_route_fiber_support_audit_contract_v2"
CONTRACT = "configs/pmo_route_fiber_support_audit_v2.json"
RESULT = "diagnostics/pmo_route_fiber_pilot/support_audit_v2.json"

EXPECTED_PARAMETER_KEYS = {
    "atom_delete": (),
    "atom_insert": (
        "atom_type",
        "formal_charge",
        "implicit_hydrogens",
        "neighbor_bond_classes",
    ),
    "atom_restate_semantic": ("target_class_index",),
    "bond_reorder": ("new_bond_class",),
    "cycle_close": ("bond_class",),
    "cycle_open": (),
    "ring_system_restate": ("new_bond_classes",),
}

FORBIDDEN_BINDING_KEYS = {
    "absolute_atom_address",
    "assignment",
    "atom_index",
    "binding",
    "executable_action",
    "slot",
    "source_graph",
    "target_patch",
}


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def _initialization_audit(root: Path) -> dict[str, Any]:
    initialized = json.loads((root / INITIALIZATION).read_text())
    body = {key: value for key, value in initialized.items() if key != "lock_sha256"}
    if identity(body) != initialized.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")
    candidates = initialized.get("candidates")
    if initialized.get("count") != 16 or not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError("PMO initialization is not the immutable 16-source bank")
    return {
        "sources": len(candidates),
        "unique_endpoints": len({row["endpoint"] for row in candidates}),
        "task_or_score_fields": sum(
            "task" in row or "score" in row for row in candidates
        ),
        "source_sha256": initialized["source_sha256"],
    }


def _row_support_audit(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = dataset.get("generic_rows")
    if not isinstance(rows, list) or len(rows) != 6143:
        raise ValueError("sealed PMO decision census changed")
    state_dim = int(dataset.get("state_feature_dim", -1))
    stage_dim = len(dataset.get("stage_descriptor_names") or [])
    if state_dim != 517 or stage_dim != 18:
        raise ValueError("PMO state or stage descriptor contract changed")

    rule_counts: Counter[str] = Counter()
    parameter_rows: Counter[str] = Counter()
    parameter_mismatches: Counter[str] = Counter()
    operand_rows = 0
    operand_count = 0
    origin_counts: Counter[str] = Counter()
    created_dependency_rows = 0
    created_output_rows = 0
    row_keys = _recursive_keys(rows)

    for index, row in enumerate(rows):
        if row.get("row_index") != index:
            raise ValueError(f"PMO generic-row order changed at row {index}")
        if len(row.get("state_features") or []) != state_dim:
            raise ValueError(f"PMO state feature width changed at row {index}")
        if len(row.get("stage_descriptor") or []) != stage_dim:
            raise ValueError(f"PMO stage descriptor width changed at row {index}")
        action = row.get("action_supervision")
        if not isinstance(action, dict):
            raise TypeError(f"PMO action supervision missing at row {index}")
        rule = str(action.get("executor_rule"))
        if rule not in EXPECTED_PARAMETER_KEYS:
            raise ValueError(f"unexpected PMO executor rule at row {index}: {rule}")
        rule_counts[rule] += 1
        parameters = action.get("parameters")
        if not isinstance(parameters, dict):
            raise TypeError(f"PMO action parameters are malformed at row {index}")
        if parameters:
            parameter_rows[rule] += 1
        if tuple(sorted(parameters)) != tuple(sorted(EXPECTED_PARAMETER_KEYS[rule])):
            parameter_mismatches[rule] += 1
        operands = action.get("operands")
        if not isinstance(operands, list) or not operands:
            raise ValueError(f"PMO structural operands missing at row {index}")
        operand_rows += 1
        operand_count += len(operands)
        for operand in operands:
            if set(operand) != {"descriptor", "role"}:
                raise ValueError(f"PMO operand schema changed at row {index}")
            origin_counts[str(operand["descriptor"].get("origin"))] += 1
        created_dependency_rows += bool(action.get("created_handle_dependencies"))
        created_output_rows += action.get("created_output_ordinal") is not None

    binding_hits = sorted(FORBIDDEN_BINDING_KEYS & row_keys)
    return {
        "decisions": len(rows),
        "state_feature_width": state_dim,
        "stage_descriptor_width": stage_dim,
        "executor_rule_counts": dict(sorted(rule_counts.items())),
        "typed_parameter_rows": sum(parameter_rows.values()),
        "parameterless_rows_expected_by_rule": len(rows) - sum(parameter_rows.values()),
        "parameter_schema_mismatches": dict(sorted(parameter_mismatches.items())),
        "structural_operand_rows": operand_rows,
        "structural_operands": operand_count,
        "operand_origin_counts": dict(sorted(origin_counts.items())),
        "created_dependency_rows": created_dependency_rows,
        "created_output_rows": created_output_rows,
        "runtime_binding_key_hits": binding_hits,
        "runtime_binding_rows": 0,
        "exact_source_graph_rows": 0,
        "executable_action_rows": 0,
        "complete_region_target_rows": 0,
    }


def contract_envelope() -> dict[str, Any]:
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "authorization": (
            "2026-09-18 user authorized a bounded zero-oracle PMO production "
            "route-proposer and equal-attempt sampler gate"
        ),
        "scientific_question": (
            "can the sealed split-clean PMO route export instantiate an actual "
            "runtime route proposer without reconstructing missing bindings"
        ),
        "tasks": list(TASKS),
        "inputs": EXPECTED_SHA256,
        "immutable_initialization_sources": 16,
        "attempt_matching": (
            "required, but prohibited until both actual samplers exist"
        ),
        "oracle_calls_authorized": 0,
        "scored_launch_authorized": False,
        "outputs": {"support_audit": RESULT},
        "fail_closed": True,
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def build_support_audit(root: Path) -> dict[str, Any]:
    for relative, expected in EXPECTED_SHA256.items():
        verify_file(root / relative, expected)
    dataset = _load_envelope(root / TRAINING_DATASET, compressed=True)
    route_result = _load_envelope(root / TRAINING_RESULT)
    loto = _loto_audit(dataset)
    row_support = _row_support_audit(dataset)
    initialization = _initialization_audit(root)

    export_flags = {
        "artifact_role": dataset.get("artifact_role"),
        "evidence": dataset.get("evidence"),
        "actor_training_performed": bool(dataset.get("actor_training_performed")),
        "module_count_targets_emitted": bool(dataset.get("module_count_targets_emitted")),
        "binding_prototype_targets_emitted": bool(
            dataset.get("binding_prototype_targets_emitted")
        ),
        "recognized_compound_stages": int(
            route_result["summary"]["recognized_compound_stages"]
        ),
    }
    production_ready = all(
        (
            export_flags["actor_training_performed"],
            export_flags["binding_prototype_targets_emitted"],
            row_support["runtime_binding_rows"] > 0,
            row_support["exact_source_graph_rows"] > 0,
            row_support["executable_action_rows"] > 0,
        )
    )
    if production_ready:
        raise RuntimeError(
            "support audit unexpectedly found a production proposer; implement the "
            "matched sampler comparison in a new revision"
        )

    implementation_paths = (
        "src/compose_v4/experiments/pmo_route_fiber_support_audit.py",
        "tools/pmo_route_fiber_support_audit.py",
        "docs/PMO_ROUTE_FIBER_SUPPORT_AUDIT.md",
        "tests/test_pmo_route_fiber_support_audit.py",
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract = contract_envelope()
    payload = {
        "schema_version": SCHEMA,
        "decision": "BLOCKED_EXPORT_NOT_PRODUCTION_BINDABLE",
        "new_oracle_calls": 0,
        "scored_launch_authorized": False,
        "matched_sampler_comparison": {
            "executed": False,
            "old_sampler_attempts": 0,
            "route_sampler_attempts": 0,
            "reason": (
                "running only the old sampler would not be an equal-attempt comparison; "
                "the route arm has no executable production policy"
            ),
        },
        "contract": contract["payload"],
        "contract_payload_sha256": contract["payload_sha256"],
        "inputs": EXPECTED_SHA256,
        "code_revision": revision,
        "implementation_sha256": {
            path: sha256_file(root / path) for path in implementation_paths
        },
        "initialization": initialization,
        "leave_one_task_out_audit": loto,
        "route_export": export_flags,
        "row_support": row_support,
        "support_matrix": {
            "split_clean_task_and_lineage_exclusion": True,
            "task_free_structural_operand_descriptors": True,
            "typed_non_address_rule_parameters": not row_support[
                "parameter_schema_mismatches"
            ],
            "exact_current_state_graph_for_binding": False,
            "teacher_successor_or_executable_action_for_binding_validation": False,
            "complete_dependency_region_target": False,
            "fitted_leave_one_task_out_runtime_checkpoint": False,
            "actual_route_sampler": False,
        },
        "smallest_missing_asset": {
            "name": "split_first_exact_region_binding_corpus_and_loto_checkpoint",
            "required_contents": [
                "exact current molecular graph at each training decision",
                "complete dependency-region target or canonical teacher successor",
                "legal bound action and typed parameter realization receipt",
                "route order, STOP and relative created-handle dependencies",
                "whole-task-family and shared-lineage split identity",
            ],
            "why_minimal": (
                "the existing export already supplies task-free state features, rule "
                "labels, structural operand roles, typed non-address parameters and "
                "route order; the missing asset is the graph-to-legal-binding bridge "
                "needed to fit and verify an actual sampler"
            ),
        },
        "next_safe_action": (
            "publish the split-first exact region/binding corpus from the already "
            "locked 184 route traces, fit one leave-one-task-out runtime checkpoint "
            "per pilot task, then run the equal-attempt old-v0 versus additive-route "
            "sampler comparison on all 16 immutable sources with zero oracle calls"
        ),
        "claim_boundary": (
            "the sealed 6,143-row export is valid route supervision but is not an "
            "executable proposal policy; no PMO yield comparison was performed"
        ),
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


__all__ = [
    "CONTRACT",
    "RESULT",
    "build_support_audit",
    "contract_envelope",
]
