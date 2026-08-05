"""Fail-closed Process-V2 prerequisite boundary for the bounded P50 pilot.

The historical semantic P50 path accepts the superseded V1 Gate-0 and T1
artifacts.  Process V2 has different, independently validated artifacts.  This
module is the narrow join between the current Process-V2 Gate-0 decision, the
current Process-V2 T1 result/decision pair, and the frozen Process-V2 P50 recipe
policy.  It performs no sampling, successor compilation, optimization, or
launch.

The T1 selected checkpoint is evidence that the architecture can fit the
bounded panel.  It is deliberately *not* the P50 initialization.  P50 starts
from the same scratch initialization bound by T1, as required by the recipe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import (
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CHAIN_SCHEMA_VERSION,
    P50_RECIPE_POLICY,
    T1_CAPACITY_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_t1_result import (
    ProcessV2T1ResultError,
    validate_process_v2_t1_capacity_decision,
    validate_process_v2_t1_capacity_result,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    load_process_v2_t1_capacity_policy,
)


class ProcessV2P50PrerequisiteError(ValueError):
    """The current artifacts do not authorize the bounded Process-V2 P50 pilot."""


@dataclass(frozen=True)
class ProcessV2P50Prerequisites:
    """Small immutable identity bundle consumed by future P50 preparation."""

    process_identity_sha256: str
    active8_completion_sha256: str
    gate_zero_decision_sha256: str
    t1_capacity_policy_sha256: str
    t1_result_sha256: str
    t1_decision_sha256: str
    t1_initial_model_state_sha256: str
    t1_selected_model_state_sha256: str
    p50_recipe_policy_sha256: str
    active_families: tuple[str, ...]
    optimizer_steps: int
    batch_size: int

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.as_payload())

    def as_payload(self) -> dict[str, Any]:
        return {
            "process_identity_sha256": self.process_identity_sha256,
            "active8_completion_sha256": self.active8_completion_sha256,
            "gate_zero_decision_sha256": self.gate_zero_decision_sha256,
            "t1_capacity_policy_sha256": self.t1_capacity_policy_sha256,
            "t1_result_sha256": self.t1_result_sha256,
            "t1_decision_sha256": self.t1_decision_sha256,
            "t1_initial_model_state_sha256": self.t1_initial_model_state_sha256,
            "t1_selected_model_state_sha256": self.t1_selected_model_state_sha256,
            "p50_recipe_policy_sha256": self.p50_recipe_policy_sha256,
            "active_families": list(self.active_families),
            "optimizer_steps": self.optimizer_steps,
            "batch_size": self.batch_size,
        }


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, *, field: str) -> str:
    if not _is_sha(value):
        raise ProcessV2P50PrerequisiteError(f"{field} must be a full lowercase SHA-256")
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2P50PrerequisiteError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ProcessV2P50PrerequisiteError(
            f"{label} is not canonical newline-framed JSON"
        )
    return value


def validate_process_v2_p50_prerequisite_relationships(
    *,
    p50_policy: Mapping[str, Any],
    capacity_policy: Mapping[str, Any],
    gate_zero_decision: Mapping[str, Any],
    t1_result: Mapping[str, Any],
    t1_decision: Mapping[str, Any],
    t1_result_file_sha256: str,
) -> ProcessV2P50Prerequisites:
    """Validate the only current path from T1 evidence to bounded-P50 scope."""

    try:
        result = validate_process_v2_t1_capacity_result(
            t1_result, capacity_policy=capacity_policy
        )
        decision = validate_process_v2_t1_capacity_decision(
            t1_decision,
            result=result,
            capacity_policy=capacity_policy,
            result_file_sha256=t1_result_file_sha256,
            require_p50_go=True,
        )
    except ProcessV2T1ResultError as error:
        raise ProcessV2P50PrerequisiteError(str(error)) from error

    policy = dict(p50_policy)
    gate = dict(gate_zero_decision)
    try:
        verify_self_hash(
            policy,
            field="contract_sha256",
            label="the Process-V2 P50 recipe policy",
        )
        require_authority_false(policy, label="the Process-V2 P50 recipe policy")
        verify_self_hash(
            gate,
            field="decision_sha256",
            label="the Process-V2 Gate-0 decision",
        )
        require_authority_false(gate, label="the Process-V2 Gate-0 decision")
    except ValueError as error:
        raise ProcessV2P50PrerequisiteError(str(error)) from error

    process = policy.get("process_identity")
    optimization = policy.get("optimization")
    parents = policy.get("parents")
    t1_parent = (
        parents.get("t1_capacity_policy") if isinstance(parents, Mapping) else None
    )
    if (
        policy.get("schema") != "compose.editing_v2.process_v2_p50_recipe_policy"
        or policy.get("schema_version") != CHAIN_SCHEMA_VERSION
        or policy.get("status")
        != "FROZEN_PROCESS_V2_P50_RECIPE_POLICY_NO_DOWNSTREAM_AUTHORITY"
        or not isinstance(process, Mapping)
        or not isinstance(optimization, Mapping)
        or not isinstance(t1_parent, Mapping)
    ):
        raise ProcessV2P50PrerequisiteError("Process-V2 P50 recipe policy shape disagrees")

    process_sha = _require_sha(
        process.get("process_identity_sha256"), field="P50 process identity"
    )
    families = tuple(str(item) for item in policy.get("active_families", ()))
    required_families = tuple(
        str(item) for item in capacity_policy.get("required_families", ())
    )
    provenance = result["provenance"]
    t1_parent_semantic = t1_parent.get("semantic")
    if (
        not families
        or families != required_families
        or tuple(decision["required_families"]) != families
        or process_sha != decision["process_identity_sha256"]
        or process_sha != provenance["process_identity_sha256"]
        or not isinstance(t1_parent_semantic, Mapping)
        or t1_parent_semantic.get("sha256")
        != capacity_policy.get("contract_sha256")
        or gate.get("schema") != GATE_ZERO_DECISION_SCHEMA
        or gate.get("schema_version") != GATE_ZERO_DECISION_SCHEMA_VERSION
        or gate.get("status") != PIPELINE_STATUS_NO_AUTHORITY
        or gate.get("decision") != "PASS"
        or gate.get("process_identity_sha256") != process_sha
        or gate.get("active8_completion_sha256") != decision["active8_completion_sha256"]
        or gate.get("decision_sha256") != decision["gate_zero_decision_sha256"]
        or gate.get("decision_sha256") != provenance["gate_zero_decision_sha256"]
        or decision["active8_completion_sha256"]
        != provenance["active8_completion_sha256"]
    ):
        raise ProcessV2P50PrerequisiteError(
            "Process-V2 P50, Gate-0, T1, or Active8 identity disagrees"
        )

    expected_optimization = {
        "optimizer_steps": 50,
        "batch_size": 64,
        "initialization": "scratch_from_t1_bound_initial_model_state",
        "resume": False,
        "dtype": "float32",
        "mixed_precision": False,
    }
    if any(optimization.get(key) != value for key, value in expected_optimization.items()):
        raise ProcessV2P50PrerequisiteError(
            "Process-V2 P50 optimization is outside the bounded 50-step scratch pilot"
        )
    if (
        policy.get("objective", {}).get("unit")
        != "productive_embedded_canonical_successor"
        or policy.get("objective", {}).get("hazard_included") is not False
        or policy.get("scientific_scope")
        != "scratch_active8_stage_a_capability_pilot_not_production_law_calibration"
        or policy.get("p500_authorized") is not False
    ):
        raise ProcessV2P50PrerequisiteError("Process-V2 P50 scientific scope disagrees")

    return ProcessV2P50Prerequisites(
        process_identity_sha256=process_sha,
        active8_completion_sha256=decision["active8_completion_sha256"],
        gate_zero_decision_sha256=decision["gate_zero_decision_sha256"],
        t1_capacity_policy_sha256=decision["capacity_policy_sha256"],
        t1_result_sha256=decision["result_sha256"],
        t1_decision_sha256=decision["decision_sha256"],
        t1_initial_model_state_sha256=provenance["initial_model_state_sha256"],
        t1_selected_model_state_sha256=decision["selected_model_state_sha256"],
        p50_recipe_policy_sha256=policy["contract_sha256"],
        active_families=families,
        optimizer_steps=int(optimization["optimizer_steps"]),
        batch_size=int(optimization["batch_size"]),
    )


def load_process_v2_p50_prerequisites(
    *,
    gate_zero_decision_path: Path,
    t1_result_path: Path,
    t1_decision_path: Path,
    repo_root: Path,
) -> ProcessV2P50Prerequisites:
    """Physically reopen and validate the current Process-V2 prerequisite chain."""

    root = Path(repo_root).resolve()
    p50_policy = load_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=root)
    capacity_policy, _ = load_process_v2_t1_capacity_policy(
        root / T1_CAPACITY_POLICY, repo_root=root
    )
    gate = _load_canonical(gate_zero_decision_path, label="the Process-V2 Gate-0 decision")
    result = _load_canonical(t1_result_path, label="the Process-V2 T1 result")
    decision = _load_canonical(t1_decision_path, label="the Process-V2 T1 decision")
    return validate_process_v2_p50_prerequisite_relationships(
        p50_policy=p50_policy,
        capacity_policy=capacity_policy,
        gate_zero_decision=gate,
        t1_result=result,
        t1_decision=decision,
        t1_result_file_sha256=_file_sha256(t1_result_path),
    )


__all__ = [
    "ProcessV2P50PrerequisiteError",
    "ProcessV2P50Prerequisites",
    "load_process_v2_p50_prerequisites",
    "validate_process_v2_p50_prerequisite_relationships",
]
