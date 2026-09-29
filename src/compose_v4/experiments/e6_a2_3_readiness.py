"""Fail-closed readiness audit for E6 milestone A2.3.

The exact graph and its action accounting are frozen by A2.2b.  The next
milestone cannot be executed scientifically until the *control problem* is
also frozen: a finite horizon, terminal desirability, initial law, numeric
tolerances, and the independent path-check instance.

This module deliberately does not contain a Doob solver.  It verifies the
already-frozen graph and the stage-1 canonical-uniform kernel specification,
then records the unresolved control semantics.  A blocked audit is preferable
to silently borrowing the E7 dynamic horizon or inventing a favorable target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.experiments.e6_graph_audit import (
    load_a2_2b_contract,
    validate_a2_2b_artifact,
)
from compose_v4.experiments.registry import load_registry

A2_3_CONTRACT_SCHEMA = "compose.experiments.e6_a2_3_readiness_contract"
A2_3_CONTRACT_VERSION = 1
A2_3_ARTIFACT_SCHEMA = "compose.experiments.e6_a2_3_readiness"
A2_3_ARTIFACT_VERSION = 1
A2_3_BLOCKED_STATUS = "BLOCKED_MISSING_FROZEN_CONTROL_SEMANTICS"

_ROOT = Path(__file__).resolve().parents[3]
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/e6_a2_3_readiness.py",
    "src/compose_v4/experiments/e6_graph_audit.py",
    "src/compose_v4/experiments/registry.py",
)
_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "milestone",
    "paper_claim_authorized",
    "solver_execution_authorized",
    "inputs",
    "confirmed_stage_1_kernel",
    "required_unresolved_decisions",
    "required_task_artifacts",
    "fail_closed_policy",
    "scope_caveat",
    "contract_sha256",
}
_ARTIFACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "milestone",
    "paper_claim_authorized",
    "solver_execution_authorized",
    "control_solver_run",
    "solver_kernel_materialized",
    "terminal_law_metrics_computed",
    "scope_caveat",
    "confirmed",
    "missing_decisions",
    "missing_task_artifacts",
    "blocked_operations",
    "provenance",
    "invariants",
    "artifact_sha256",
}
_REQUIRED_DECISION_IDS = (
    "finite_control_horizon",
    "terminal_desirability",
    "initial_law",
    "solver_tolerances",
    "explicit_path_check",
)


class E6A23ReadinessError(RuntimeError):
    """A2.3 readiness contract, dependency, or provenance is invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E6A23ReadinessError("A2.3 metadata is not finite canonical JSON") from error


def stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def contract_self_hash(contract: Mapping[str, Any]) -> str:
    body = dict(contract)
    body.pop("contract_sha256", None)
    return stable_sha256(body)


def artifact_self_hash(artifact: Mapping[str, Any]) -> str:
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    return stable_sha256(body)


def implementation_sha256(repo_root: str | Path = _ROOT) -> str:
    root = Path(repo_root)
    digest = hashlib.sha256()
    for relative in sorted(_IMPLEMENTATION_SOURCES):
        path = root / relative
        if not path.is_file():
            raise E6A23ReadinessError(f"A2.3 implementation source is absent: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_input(repo_root: str | Path, relative: str) -> Path:
    path = Path(repo_root) / relative
    if not path.is_file():
        raise E6A23ReadinessError(f"required A2.3 input is absent: {relative}")
    return path


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if set(contract) != _CONTRACT_FIELDS:
        raise E6A23ReadinessError("A2.3 contract has missing or unknown top-level fields")
    if (
        contract["schema"] != A2_3_CONTRACT_SCHEMA
        or contract["schema_version"] != A2_3_CONTRACT_VERSION
        or contract["experiment_id"] != "E6"
        or contract["milestone"] != "A2.3"
        or contract["paper_claim_authorized"] is not False
        or contract["solver_execution_authorized"] is not False
    ):
        raise E6A23ReadinessError("A2.3 contract identity or fail-closed status is invalid")
    if contract["contract_sha256"] != contract_self_hash(contract):
        raise E6A23ReadinessError("A2.3 contract self-hash does not match")
    decision_ids = tuple(
        decision["decision_id"] for decision in contract["required_unresolved_decisions"]
    )
    if decision_ids != _REQUIRED_DECISION_IDS:
        raise E6A23ReadinessError("A2.3 unresolved-decision list changed")
    if any(
        decision["current_value"] is not None
        for decision in contract["required_unresolved_decisions"]
    ):
        raise E6A23ReadinessError("blocked A2.3 contract cannot contain resolved values")
    kernel = contract["confirmed_stage_1_kernel"]
    if (
        kernel["level"] != "canonical_molecular_successor"
        or kernel["is_a2_2b_diagnostic_mark_law"] is not False
    ):
        raise E6A23ReadinessError("A2.3 solver-law level is not canonical-successor")
    policy = contract["fail_closed_policy"]
    if (
        policy["blocked_status"] != A2_3_BLOCKED_STATUS
        or policy["forbid_solver_execution"] is not True
        or policy["forbid_kernel_materialization"] is not True
        or policy["forbid_terminal_law_metrics"] is not True
        or policy["forbid_paper_claim"] is not True
    ):
        raise E6A23ReadinessError("A2.3 fail-closed policy has weakened")


def load_a2_3_readiness_contract(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise E6A23ReadinessError(f"cannot read A2.3 readiness contract: {path}") from error
    if not isinstance(payload, dict):
        raise E6A23ReadinessError("A2.3 readiness contract must be an object")
    _validate_contract(payload)
    return payload


def _artifact_path(repo_root: str | Path, relative: str) -> Path:
    return Path(repo_root).resolve() / relative


def run_a2_3_readiness_audit(
    contract: Mapping[str, Any],
    *,
    repo_root: str | Path = _ROOT,
) -> dict[str, Any]:
    """Validate dependencies and emit a non-claiming blocked readiness record."""

    _validate_contract(contract)
    root = Path(repo_root).resolve()
    inputs = contract["inputs"]
    registry_path = _resolve_input(root, inputs["registry_path"])
    plan_path = _resolve_input(root, inputs["infrastructure_plan_path"])
    handoff_path = _resolve_input(root, inputs["handoff_path"])
    a2_2b_contract_path = _resolve_input(root, inputs["a2_2b_contract_path"])
    a2_2b_artifact_path = _resolve_input(root, inputs["a2_2b_artifact_path"])

    registry = load_registry(registry_path)
    e6 = registry["experiments"]["E6"]
    selected = registry["exact_sizing"]["selected"]
    a2_2b_contract = load_a2_2b_contract(
        a2_2b_contract_path,
        registry_path=registry_path,
    )
    a2_2b_artifact = json.loads(a2_2b_artifact_path.read_text())
    validate_a2_2b_artifact(
        a2_2b_artifact,
        contract=a2_2b_contract,
        registry_path=registry_path,
        repo_root=root,
    )

    if a2_2b_contract["contract_sha256"] != inputs["expected_a2_2b_contract_sha256"]:
        raise E6A23ReadinessError("A2.2b contract identity drifted")
    if a2_2b_artifact["artifact_sha256"] != inputs["expected_a2_2b_artifact_sha256"]:
        raise E6A23ReadinessError("A2.2b artifact identity drifted")
    if a2_2b_artifact["graph_audit_hash"] != inputs["expected_graph_audit_hash"]:
        raise E6A23ReadinessError("A2.2b graph audit identity drifted")
    benchmark = a2_2b_artifact["benchmark"]
    if benchmark["structural_graph_fingerprint"] != inputs["expected_structural_graph_fingerprint"]:
        raise E6A23ReadinessError("A2.2b structural graph identity drifted")
    if a2_2b_artifact["state_index"]["state_index_sha256"] != inputs["expected_state_index_sha256"]:
        raise E6A23ReadinessError("A2.2b state index identity drifted")

    task_paths = tuple(str(e6["task_artifacts"][kind]) for kind in ("development", "final"))
    if task_paths != tuple(contract["required_task_artifacts"]):
        raise E6A23ReadinessError("registry E6 task-artifact paths drifted")
    missing_tasks = [
        relative for relative in task_paths if not _artifact_path(root, relative).is_file()
    ]

    # Names of tilt families are not definitions of g.  The E7 dynamic horizon
    # is intentionally not searched or inherited here.
    e6_horizon_fields = {
        key: e6[key]
        for key in ("control_horizon_steps", "finite_horizon_steps", "horizon_steps")
        if key in e6
    }
    e6_desirability_fields = {
        key: e6[key]
        for key in ("terminal_desirability", "desirability_builder", "terminal_values")
        if key in e6
    }
    if e6_horizon_fields or e6_desirability_fields or not missing_tasks:
        raise E6A23ReadinessError(
            "A2.3 freeze state changed; replace this blocked contract with a reviewed "
            "resolved control contract before executing a solver"
        )

    confirmed = {
        "registry_protocol_content_hash": registry["protocol"]["protocol_freeze"]["content_hash"],
        "selected_benchmark": {
            "candidate_id": selected["candidate_id"],
            "n_states": selected["n_states"],
            "n_edges": selected["n_edges"],
            "structural_graph_fingerprint": selected["graph_fingerprint"],
            "state_index_sha256": a2_2b_artifact["state_index"]["state_index_sha256"],
        },
        "a2_2b_complete": a2_2b_artifact["status"],
        "a2_2b_diagnostic_law_is_solver_kernel": a2_2b_artifact["benchmark"]["diagnostic_mark_law"][
            "is_solver_kernel"
        ],
        "stage_1_solver_kernel_specification": contract["confirmed_stage_1_kernel"],
        "registry_tilt_names_are_unparameterized": list(e6["tilts"]),
        "e6_horizon_fields_present": e6_horizon_fields,
        "e6_desirability_fields_present": e6_desirability_fields,
    }
    missing_decisions = [dict(decision) for decision in contract["required_unresolved_decisions"]]
    invariants = {
        "a2_2b_artifact_validated": True,
        "selected_graph_identity_unchanged": (
            selected["graph_fingerprint"] == inputs["expected_structural_graph_fingerprint"]
        ),
        "stage_1_law_is_successor_level": (
            contract["confirmed_stage_1_kernel"]["level"] == "canonical_molecular_successor"
        ),
        "diagnostic_mark_law_not_promoted_to_solver_kernel": (
            a2_2b_artifact["benchmark"]["diagnostic_mark_law"]["is_solver_kernel"] is False
        ),
        "e7_dynamic_horizon_not_imported": not e6_horizon_fields,
        "unparameterized_tilt_names_not_treated_as_g": not e6_desirability_fields,
        "all_required_control_decisions_reported": (
            tuple(item["decision_id"] for item in missing_decisions) == _REQUIRED_DECISION_IDS
        ),
        "solver_not_run": True,
        "solver_kernel_not_materialized": True,
        "terminal_law_metrics_not_computed": True,
        "paper_claim_not_authorized": True,
    }
    if not all(invariants.values()):
        failed = sorted(name for name, passed in invariants.items() if not passed)
        raise E6A23ReadinessError(f"A2.3 readiness invariants failed: {failed}")

    body: dict[str, Any] = {
        "schema": A2_3_ARTIFACT_SCHEMA,
        "schema_version": A2_3_ARTIFACT_VERSION,
        "status": A2_3_BLOCKED_STATUS,
        "experiment_id": "E6",
        "milestone": "A2.3",
        "paper_claim_authorized": False,
        "solver_execution_authorized": False,
        "control_solver_run": False,
        "solver_kernel_materialized": False,
        "terminal_law_metrics_computed": False,
        "scope_caveat": contract["scope_caveat"],
        "confirmed": confirmed,
        "missing_decisions": missing_decisions,
        "missing_task_artifacts": missing_tasks,
        "blocked_operations": [
            "budget_indexed_kernel_materialization",
            "backward_value_recursion",
            "doob_kernel_construction",
            "h_zero_reachability_classification",
            "support_inclusion_measurement",
            "terminal_law_tv_measurement",
            "explicit_path_enumeration",
        ],
        "provenance": {
            "contract_sha256": contract["contract_sha256"],
            "implementation_sha256": implementation_sha256(root),
            "implementation_sources": list(_IMPLEMENTATION_SOURCES),
            "registry_path": inputs["registry_path"],
            "registry_file_sha256": file_sha256(registry_path),
            "registry_protocol_content_hash": registry["protocol"]["protocol_freeze"][
                "content_hash"
            ],
            "infrastructure_plan_path": inputs["infrastructure_plan_path"],
            "infrastructure_plan_file_sha256": file_sha256(plan_path),
            "handoff_path": inputs["handoff_path"],
            "handoff_file_sha256": file_sha256(handoff_path),
            "a2_2b_contract_path": inputs["a2_2b_contract_path"],
            "a2_2b_contract_sha256": a2_2b_contract["contract_sha256"],
            "a2_2b_artifact_path": inputs["a2_2b_artifact_path"],
            "a2_2b_artifact_sha256": a2_2b_artifact["artifact_sha256"],
            "a2_2b_artifact_file_sha256": file_sha256(a2_2b_artifact_path),
            "graph_audit_hash": a2_2b_artifact["graph_audit_hash"],
        },
        "invariants": invariants,
    }
    return {**body, "artifact_sha256": stable_sha256(body)}


def validate_a2_3_readiness_artifact(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    repo_root: str | Path = _ROOT,
) -> None:
    _validate_contract(contract)
    if set(artifact) != _ARTIFACT_FIELDS:
        raise E6A23ReadinessError("A2.3 artifact has missing or unknown top-level fields")
    if artifact["artifact_sha256"] != artifact_self_hash(artifact):
        raise E6A23ReadinessError("A2.3 artifact self-hash does not match")
    expected = run_a2_3_readiness_audit(contract, repo_root=repo_root)
    if artifact != expected:
        raise E6A23ReadinessError("A2.3 artifact does not match current frozen inputs")


def freeze_a2_3_readiness_artifact(
    artifact: Mapping[str, Any],
    output_path: str | Path,
    *,
    contract: Mapping[str, Any],
    repo_root: str | Path = _ROOT,
) -> None:
    """Write one immutable canonical readiness artifact without replacement."""

    validate_a2_3_readiness_artifact(
        artifact,
        contract=contract,
        repo_root=repo_root,
    )
    output = Path(output_path)
    encoded = _canonical_json_bytes(artifact) + b"\n"
    if output.exists():
        if output.read_bytes() != encoded:
            raise E6A23ReadinessError(f"immutable A2.3 readiness artifact collision: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = output.open("xb")
    except FileExistsError as error:
        raise E6A23ReadinessError(
            f"immutable A2.3 readiness artifact collision: {output}"
        ) from error
    with handle:
        handle.write(encoded)
