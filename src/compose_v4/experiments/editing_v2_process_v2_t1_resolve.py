"""Resolve the exact Process-V2 structural evidence into a bounded T1 panel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_active8_plan import (
    PLAN_FILENAME,
    load_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    PREPARATION_FILENAME,
    validate_process_v2_active8_completion,
    validate_process_v2_active8_reduction_preparation,
)
from compose_v4.data.editing_v2_process_v2_gate_zero import (
    load_gate_zero_contracts,
    read_active8_decision_index,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    GATE_ZERO_DECISION_FILENAME,
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    canonical_bytes,
    canonical_sha256,
    require_no_granted_authority,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    T1_PANEL_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1CandidateSelection,
    ProcessV2T1PanelError,
    ProcessV2T1TargetCensus,
)
from compose_v4.experiments.editing_v2_process_v2_t1_source import (
    select_authenticated_process_v2_t1_panel,
)


def _load_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except OSError as error:
        raise ProcessV2T1PanelError(f"{label} is absent: {path}") from error
    except json.JSONDecodeError as error:
        raise ProcessV2T1PanelError(f"{label} is not JSON: {path}") from error
    if not isinstance(value, Mapping) or canonical_bytes(value) + b"\n" != raw:
        raise ProcessV2T1PanelError(f"{label} is not canonical newline-terminated JSON")
    return dict(value), hashlib.sha256(raw).hexdigest()


def _validate_gate_zero_pass(
    value: Mapping[str, Any],
    *,
    completion: Mapping[str, Any],
    source_index_sha256: str,
    contracts: Any,
) -> dict[str, Any]:
    decision = dict(value)
    try:
        verify_self_hash(decision, field="decision_sha256", label="the Gate 0 decision")
        require_no_granted_authority(decision, label="the Gate 0 decision")
    except ValueError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    if (
        decision.get("schema") != GATE_ZERO_DECISION_SCHEMA
        or decision.get("schema_version") != GATE_ZERO_DECISION_SCHEMA_VERSION
        or decision.get("status") != PIPELINE_STATUS_NO_AUTHORITY
        or decision.get("decision") != "PASS"
    ):
        raise ProcessV2T1PanelError("T1 requires an exact Process-V2 Gate 0 PASS")
    expected = {
        "active8_completion_sha256": completion["completion_sha256"],
        "active8_sentinel_sha256": completion["sentinel"]["sentinel_sha256"],
        "contracts_binding_sha256": contracts.binding_sha256,
        "gate_zero_structural_contract_sha256": contracts.contract_sha256,
        "process_identity_sha256": contracts.process_identity_sha256,
        "source_index_sha256": source_index_sha256,
    }
    for field, expected_value in expected.items():
        if decision.get(field) != expected_value:
            raise ProcessV2T1PanelError(
                f"the Gate 0 PASS disagrees with the current {field}"
            )
    checks = decision.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(check is not True for check in checks.values())
        or decision.get("total_violations") != 0
        or decision.get("missing_active8_families") != []
        or decision.get("missing_required_cells") != []
    ):
        raise ProcessV2T1PanelError("the Gate 0 PASS contradicts its structural evidence")
    return decision


@dataclass(frozen=True, slots=True)
class ProcessV2T1ResolvedPanel:
    """Nonauthorizing, provenance-bound T1 candidate selection."""

    provenance: Mapping[str, str]
    target_census: ProcessV2T1TargetCensus
    selection: ProcessV2T1CandidateSelection
    resolved_sha256: str

    def identity_body(self) -> dict[str, object]:
        return {
            "provenance": dict(sorted(self.provenance.items())),
            "target_census_sha256": self.target_census.census_sha256,
            "selection_sha256": self.selection.selection_sha256,
        }


def resolve_process_v2_t1_panel(
    *,
    active8_run_root: Path,
    gate_zero_run_root: Path,
    repo_root: Path,
) -> ProcessV2T1ResolvedPanel:
    """Require the completed Active8 and Gate-0 chain, then select T1 states."""

    active8_root = Path(active8_run_root)
    repository = Path(repo_root)
    plan = load_process_v2_active8_plan(
        active8_root / PLAN_FILENAME,
        repo_root=repository,
    )
    prepared_raw, prepared_file_sha256 = _load_canonical(
        active8_root / PREPARATION_FILENAME,
        label="the Active8 reduction preparation",
    )
    try:
        prepared = validate_process_v2_active8_reduction_preparation(
            prepared_raw,
            active8_plan=plan,
        )
    except ValueError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    completion_raw, completion_file_sha256 = _load_canonical(
        active8_root / COMPLETION_FILENAME,
        label="the Active8 completion",
    )
    try:
        completion = validate_process_v2_active8_completion(completion_raw)
    except ValueError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    for field in (
        "binding_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "task_inventory_sha256",
        "result_inventory_sha256",
        "accepted_transitions",
    ):
        if completion.get(field) != prepared.get(field):
            raise ProcessV2T1PanelError(
                f"the Active8 preparation and completion disagree through {field}"
            )

    contracts = load_gate_zero_contracts(repo_root=repository)
    source_index = read_active8_decision_index(active8_root, contracts=contracts)
    decision_raw, decision_file_sha256 = _load_canonical(
        Path(gate_zero_run_root) / GATE_ZERO_DECISION_FILENAME,
        label="the Process-V2 Gate 0 decision",
    )
    decision = _validate_gate_zero_pass(
        decision_raw,
        completion=completion,
        source_index_sha256=source_index.index_sha256,
        contracts=contracts,
    )
    panel_policy = load_process_v2_chain_artifact(
        T1_PANEL_POLICY,
        repo_root=repository,
    )
    if panel_policy["process_identity"]["process_identity_sha256"] != (
        contracts.process_identity_sha256
    ):
        raise ProcessV2T1PanelError("the T1 panel policy binds another process identity")
    census, selection = select_authenticated_process_v2_t1_panel(
        active8_run_root=active8_root,
        source_index=source_index,
        sentinel_plan=prepared["sentinel_plan"],
        panel_policy=panel_policy,
        required_cell_ids=contracts.required_cell_ids,
    )
    provenance = {
        "active8_plan_sha256": str(plan["plan_sha256"]),
        "active8_preparation_sha256": str(prepared["preparation_sha256"]),
        "active8_preparation_file_sha256": prepared_file_sha256,
        "active8_completion_sha256": str(completion["completion_sha256"]),
        "active8_completion_file_sha256": completion_file_sha256,
        "active8_sentinel_sha256": str(completion["sentinel"]["sentinel_sha256"]),
        "gate_zero_decision_sha256": str(decision["decision_sha256"]),
        "gate_zero_decision_file_sha256": decision_file_sha256,
        "gate_zero_source_index_sha256": source_index.index_sha256,
        "t1_panel_policy_sha256": str(panel_policy["contract_sha256"]),
    }
    identity = {
        "provenance": dict(sorted(provenance.items())),
        "target_census_sha256": census.census_sha256,
        "selection_sha256": selection.selection_sha256,
    }
    return ProcessV2T1ResolvedPanel(
        provenance=provenance,
        target_census=census,
        selection=selection,
        resolved_sha256=canonical_sha256(identity),
    )


__all__ = ["ProcessV2T1ResolvedPanel", "resolve_process_v2_t1_panel"]
