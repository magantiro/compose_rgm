"""Compose independent Process-V2 T1 family receipts for bounded P50.

T1 capacity is a property of a family, panel, objective, threshold set,
initialization semantics, and trainable parameter route.  It is not a single
checkpoint requirement.  This module validates the original joint diagnostic
and only the four family repairs it identified, then publishes one typed
manifest without pretending those independent optimizations share model state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_t1_failure_scope import (
    failing_families_from_capacity_result,
    validate_failure_scope_result,
)
from compose_v4.experiments.editing_v2_process_v2_t1_result import (
    validate_process_v2_t1_capacity_result,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    semantic_t1_threshold_checks,
    summarize_semantic_t1_metrics,
)

RESULT_FILENAME = "PROCESS_V2_T1_SCOPED_CAPACITY_RESULT.json"
DECISION_FILENAME = "PROCESS_V2_T1_SCOPED_CAPACITY_DECISION.json"
RESULT_SCHEMA = "compose.editing_v2.process_v2_t1_scoped_capacity_result"
RESULT_SCHEMA_VERSION = 1
RESULT_STATUS = "COMPLETE_PROCESS_V2_T1_SCOPED_CAPACITY_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing_v2.process_v2_t1_scoped_capacity_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_GO_STATUS = "GO_PROCESS_V2_BOUNDED_P50_SCOPED_T1_CAPACITY"
DECISION_NO_GO_STATUS = "NO_GO_PROCESS_V2_BOUNDED_P50_SCOPED_T1_CAPACITY"

ACTIVE_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
)
BASE_PASS_FAMILIES = (
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "cycle_insert",
)
REPAIRED_FAMILIES = (
    "atom_insert",
    "bond_reroute",
    "cycle_attach",
    "ring_system_restate",
)
_EXPECTED_SCOPE = {
    "atom_insert": "all",
    "bond_reroute": "all",
    "cycle_attach": "all",
    "ring_system_restate": "heads_plus_local_adapter",
}
_POLICY_PROJECTION_FIELDS = (
    "active_families",
    "empirical_repeated_state_gate",
    "hazard_included",
    "objective_unit",
    "optimization",
    "panel_cardinality",
    "panel_kind",
    "required_families",
    "sampling_law",
    "support_time_hex",
    "thresholds",
)
_HISTORICAL_POLICY_SEMANTIC_SHA256 = (
    "6940710e8104764376ae85999fdfbe346a8f76aa98899aac65ab19f1acca9ef3"
)
_HISTORICAL_POLICY_FILE_SHA256 = (
    "1e8bb119487c2b401600b2d9e993ac366de30aea36fc3283a6fd801fff9cc330"
)
_HISTORICAL_POLICY_PROJECTION_SHA256 = (
    "8d2838e92a1b96997adec0f61488ec6927c94df8973ee1cfc42d0a8a8771412a"
)


class ProcessV2T1ScopedResultError(ValueError):
    """The per-family capacity receipts do not authorize bounded P50."""


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _policy_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    if any(field not in policy for field in _POLICY_PROJECTION_FIELDS):
        raise ProcessV2T1ScopedResultError("T1 policy lacks a scientific projection field")
    return {field: policy[field] for field in _POLICY_PROJECTION_FIELDS}


def _has_historical_lineage(policy: Mapping[str, Any]) -> bool:
    lineage = policy.get("superseded_design_lineage")
    if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
        return False
    return any(
        isinstance(row, Mapping)
        and isinstance(row.get("physical"), Mapping)
        and isinstance(row.get("semantic"), Mapping)
        and row["physical"].get("sha256") == _HISTORICAL_POLICY_FILE_SHA256
        and row["semantic"].get("sha256") == _HISTORICAL_POLICY_SEMANTIC_SHA256
        for row in lineage
    )


def validate_t1_evidence_source_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1ScopedResultError("T1 evidence source manifest is not an object")
    manifest = dict(value)
    try:
        verify_self_hash(
            manifest, field="manifest_sha256", label="the P50 T1 evidence source manifest"
        )
        require_authority_false(manifest, label="the P50 T1 evidence source manifest")
    except ValueError as error:
        raise ProcessV2T1ScopedResultError(str(error)) from error
    historical = manifest.get("historical_capacity_policy")
    sources = manifest.get("input_sources")
    expected_sources = {"base", *REPAIRED_FAMILIES}
    if (
        manifest.get("schema")
        != "compose.editing_v2.process_v2_p50_t1_evidence_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("status")
        != "FROZEN_PROCESS_V2_P50_T1_EVIDENCE_INPUTS_NO_DOWNSTREAM_AUTHORITY"
        or tuple(manifest.get("active_families", ())) != ACTIVE_FAMILIES
        or not isinstance(historical, Mapping)
        or dict(historical)
        != {
            "file_sha256": _HISTORICAL_POLICY_FILE_SHA256,
            "scientific_projection_sha256": _HISTORICAL_POLICY_PROJECTION_SHA256,
            "semantic_sha256": _HISTORICAL_POLICY_SEMANTIC_SHA256,
        }
        or not isinstance(sources, Mapping)
        or set(sources) != expected_sources
    ):
        raise ProcessV2T1ScopedResultError("T1 evidence source manifest identity disagrees")
    for name, raw in sources.items():
        if not isinstance(raw, Mapping) or set(raw) != {
            "file_sha256",
            "kind",
            "path",
            "result_sha256",
        }:
            raise ProcessV2T1ScopedResultError(f"T1 evidence source {name!r} fields disagree")
        path = raw["path"]
        if (
            raw["kind"] != ("joint_capacity_result" if name == "base" else "family_repair")
            or not _is_sha(raw["file_sha256"])
            or not _is_sha(raw["result_sha256"])
            or not isinstance(path, str)
            or not path.startswith("/artifacts/")
            or "/../" in path
        ):
            raise ProcessV2T1ScopedResultError(f"T1 evidence source {name!r} is malformed")
    return manifest


def validate_score_revision_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1ScopedResultError("score-revision receipt is not an object")
    receipt = dict(value)
    try:
        verify_self_hash(receipt, field="receipt_sha256", label="the score-revision receipt")
    except ValueError as error:
        raise ProcessV2T1ScopedResultError(str(error)) from error
    residuals = receipt.get("zero_initialized_residuals")
    if (
        set(receipt)
        != {
            "schema",
            "schema_version",
            "base_initial_model_state_sha256",
            "relational_initial_model_state_sha256",
            "current_initial_model_state_sha256",
            "support_geometry_sha256",
            "implementation_sha256",
            "zero_initialized_residuals",
            "receipt_sha256",
        }
        or receipt["schema"]
        != "compose.editing_v2.process_v2_score_revision_containment"
        or receipt["schema_version"] != 1
        or any(
            not _is_sha(receipt[field])
            for field in (
                "base_initial_model_state_sha256",
                "relational_initial_model_state_sha256",
                "current_initial_model_state_sha256",
                "support_geometry_sha256",
                "implementation_sha256",
            )
        )
        or residuals
        != [
            {
                "affected_family": "bond_reroute",
                "parameter": "graft_relation_head.weight",
                "predecessor": "base",
                "successor": "relational",
                "zero_initialized": True,
            },
            {
                "affected_family": "ring_system_restate",
                "parameter": "ring_restate_context_head.weight",
                "predecessor": "relational",
                "successor": "current",
                "zero_initialized": True,
            },
        ]
    ):
        raise ProcessV2T1ScopedResultError("score-revision containment receipt disagrees")
    return receipt


def _semantic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "panel_entry_sha256": row["panel_entry_sha256"],
            "model_family": row.get("model_family", row.get("family")),
            "capability_cell_id": row.get(
                "capability_cell_id", row.get("semantic_cell_id")
            ),
            "teacher_successor_probability": row["teacher_successor_probability"],
            "teacher_successor_nll": row.get(
                "teacher_successor_nll", row.get("canonical_successor_nll")
            ),
            "teacher_successor_rank": row["teacher_successor_rank"],
            "teacher_successor_top1": row["teacher_successor_top1"],
            "canonical_successor_count": row.get("canonical_successor_count"),
        }
        for row in rows
    ]


def _base_gradient(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family_route_finite_nonzero_seen": row["family_gate_gradient_finite"] is True
        and float(row["family_gate_cumulative_l2"]) > 0.0
        and int(row["family_gate_nonzero_update_steps"]) > 0,
        "family_route_nonzero_steps": row["family_gate_nonzero_update_steps"],
        "family_route_cumulative_gradient_norm": row["family_gate_cumulative_l2"],
        "action_route_finite_nonzero_seen": row["action_route_gradient_finite"] is True
        and float(row["action_route_cumulative_l2"]) > 0.0
        and int(row["action_route_nonzero_update_steps"]) > 0,
        "action_route_nonzero_steps": row["action_route_nonzero_update_steps"],
        "action_route_cumulative_gradient_norm": row["action_route_cumulative_l2"],
    }


def _family_receipt(
    *,
    family: str,
    source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gradient: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    panel_inventory_sha256: str,
    initial_model_state_sha256: str,
    selected_step: int,
    scope: str,
) -> dict[str, Any]:
    normalized = _semantic_rows(rows)
    metrics = summarize_semantic_t1_metrics(normalized)
    gradients = {family: dict(gradient)}
    checks = semantic_t1_threshold_checks(
        metrics, thresholds=thresholds, gradient_evidence=gradients
    )
    if set(metrics["by_family"]) != {family} or not all(checks.values()):
        raise ProcessV2T1ScopedResultError(f"T1 family {family!r} does not pass capacity")
    identifiers = [str(row["panel_entry_sha256"]) for row in normalized]
    if canonical_sha256(identifiers) != panel_inventory_sha256:
        raise ProcessV2T1ScopedResultError(f"T1 family {family!r} panel identity disagrees")
    summary = metrics["by_family"][family]
    return {
        "family": family,
        "source_kind": source["kind"],
        "source_file_sha256": source["file_sha256"],
        "source_result_sha256": source["result_sha256"],
        "scope": scope,
        "initial_model_state_sha256": initial_model_state_sha256,
        "panel_entry_count": len(normalized),
        "panel_entry_inventory_sha256": panel_inventory_sha256,
        "selected_step": selected_step,
        "minimum_teacher_successor_probability": summary[
            "minimum_teacher_successor_probability"
        ],
        "every_entry_teacher_successor_top1": summary[
            "every_entry_teacher_successor_top1"
        ],
        "threshold_checks": checks,
    }


def build_process_v2_t1_scoped_capacity_result(
    *,
    source_manifest: Mapping[str, Any],
    source_file_sha256s: Mapping[str, str],
    base_result: Mapping[str, Any],
    repair_results: Mapping[str, Mapping[str, Any]],
    capacity_policy: Mapping[str, Any],
    score_revision_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one non-authorizing result from the minimum sufficient receipts."""

    manifest = validate_t1_evidence_source_manifest(source_manifest)
    score = validate_score_revision_receipt(score_revision_receipt)
    try:
        verify_self_hash(
            dict(capacity_policy), field="contract_sha256", label="the current T1 policy"
        )
        require_authority_false(dict(capacity_policy), label="the current T1 policy")
    except ValueError as error:
        raise ProcessV2T1ScopedResultError(str(error)) from error
    if (
        tuple(capacity_policy.get("required_families", ())) != ACTIVE_FAMILIES
        or canonical_sha256(_policy_projection(capacity_policy))
        != _HISTORICAL_POLICY_PROJECTION_SHA256
        or not _has_historical_lineage(capacity_policy)
        or set(source_file_sha256s) != set(manifest["input_sources"])
        or any(
            source_file_sha256s[name] != source["file_sha256"]
            for name, source in manifest["input_sources"].items()
        )
    ):
        raise ProcessV2T1ScopedResultError("T1 evidence policy or physical inputs disagree")

    historical_policy = {
        **dict(capacity_policy),
        "policy_sha256": _HISTORICAL_POLICY_SEMANTIC_SHA256,
    }
    base = validate_process_v2_t1_capacity_result(
        base_result, capacity_policy=historical_policy
    )
    base_source = manifest["input_sources"]["base"]
    if (
        base["result_sha256"] != base_source["result_sha256"]
        or base["provenance"]["capacity_policy_file_sha256"]
        != _HISTORICAL_POLICY_FILE_SHA256
        or base["provenance"]["initial_model_state_sha256"]
        != score["base_initial_model_state_sha256"]
        or failing_families_from_capacity_result(
            base, capacity_policy=historical_policy
        )
        != ("atom_insert", "bond_reroute", "cycle_attach", "ring_system_restate")
        or set(repair_results) != set(REPAIRED_FAMILIES)
    ):
        raise ProcessV2T1ScopedResultError("the original T1 diagnostic identity disagrees")

    thresholds = capacity_policy["thresholds"]
    base_rows = base["entry_metrics"]
    base_gradients = {row["family"]: row for row in base["gradient_evidence"]}
    family_receipts: dict[str, dict[str, Any]] = {}
    for family in BASE_PASS_FAMILIES:
        rows = [row for row in base_rows if row["family"] == family]
        inventory = canonical_sha256([row["panel_entry_sha256"] for row in rows])
        family_receipts[family] = _family_receipt(
            family=family,
            source=base_source,
            rows=rows,
            gradient=_base_gradient(base_gradients[family]),
            thresholds=thresholds,
            panel_inventory_sha256=inventory,
            initial_model_state_sha256=score["base_initial_model_state_sha256"],
            selected_step=int(base["run_integrity"]["selected_step"]),
            scope="joint_all_trainable",
        )

    expected_initial = {
        "atom_insert": score["base_initial_model_state_sha256"],
        "cycle_attach": score["relational_initial_model_state_sha256"],
        "bond_reroute": score["current_initial_model_state_sha256"],
        "ring_system_restate": score["current_initial_model_state_sha256"],
    }
    for family in REPAIRED_FAMILIES:
        repair = validate_failure_scope_result(repair_results[family])
        source = manifest["input_sources"][family]
        base_family_rows = [row for row in base_rows if row["family"] == family]
        repair_identifiers = [
            row["panel_entry_sha256"]
            for row in repair["selected_metrics"]["per_entry"]
        ]
        base_identifiers = [row["panel_entry_sha256"] for row in base_family_rows]
        metrics = summarize_semantic_t1_metrics(
            _semantic_rows(repair["selected_metrics"]["per_entry"])
        )
        checks = semantic_t1_threshold_checks(
            metrics,
            thresholds=thresholds,
            gradient_evidence=repair["gradient_evidence"],
        )
        if (
            repair["family"] != family
            or repair["scope"] != _EXPECTED_SCOPE[family]
            or repair["diagnostic_passed"] is not True
            or repair["result_sha256"] != source["result_sha256"]
            or repair["input_capacity_result_file_sha256"] != base_source["file_sha256"]
            or repair["input_capacity_result_sha256"] != base_source["result_sha256"]
            or repair["panel_entry_count"] != len(base_family_rows)
            or len(repair_identifiers) != len(set(repair_identifiers))
            or set(repair_identifiers) != set(base_identifiers)
            or repair["initial_model_state_sha256"] != expected_initial[family]
            or repair["capacity_policy_sha256"]
            not in {
                _HISTORICAL_POLICY_SEMANTIC_SHA256,
                capacity_policy["contract_sha256"],
            }
            or repair["selected_threshold_checks"] != checks
            or not all(checks.values())
        ):
            raise ProcessV2T1ScopedResultError(
                f"T1 repair for {family!r} does not recompute against the frozen panel"
            )
        family_receipts[family] = _family_receipt(
            family=family,
            source=source,
            rows=repair["selected_metrics"]["per_entry"],
            gradient=repair["gradient_evidence"][family],
            thresholds=thresholds,
            panel_inventory_sha256=repair["panel_entry_inventory_sha256"],
            initial_model_state_sha256=repair["initial_model_state_sha256"],
            selected_step=int(repair["selected_step"]),
            scope=str(repair["scope"]),
        )

    ordered = [family_receipts[family] for family in ACTIVE_FAMILIES]
    body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": RESULT_STATUS,
        **authority_false_block(),
        "objective_unit": "exact_source_frozen_time_canonical_successor",
        "panel_kind": "unique_state_single_target_canonical_successor_capacity",
        "process_identity_sha256": base["provenance"]["process_identity_sha256"],
        "active8_completion_sha256": base["provenance"]["active8_completion_sha256"],
        "gate_zero_decision_sha256": base["provenance"]["gate_zero_decision_sha256"],
        "panel_sha256": base["provenance"]["panel_sha256"],
        "panel_entry_inventory_sha256": base["provenance"][
            "panel_entry_inventory_sha256"
        ],
        "capacity_policy_sha256": capacity_policy["contract_sha256"],
        "historical_capacity_policy_sha256": _HISTORICAL_POLICY_SEMANTIC_SHA256,
        "policy_scientific_projection_sha256": _HISTORICAL_POLICY_PROJECTION_SHA256,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "score_revision_receipt": score,
        "required_families": list(ACTIVE_FAMILIES),
        "unique_entry_count": sum(row["panel_entry_count"] for row in ordered),
        "family_receipts": ordered,
        "all_required_families_passed": all(
            all(row["threshold_checks"].values()) for row in ordered
        ),
        "repeated_state_empirical_law_evaluated": False,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_scoped_capacity_result(result)


def validate_process_v2_t1_scoped_capacity_result(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1ScopedResultError("scoped T1 result is not an object")
    result = dict(value)
    try:
        verify_self_hash(result, field="result_sha256", label="the scoped T1 result")
        require_authority_false(result, label="the scoped T1 result")
    except ValueError as error:
        raise ProcessV2T1ScopedResultError(str(error)) from error
    receipts = result.get("family_receipts")
    receipt_checks_are_boolean = isinstance(receipts, list) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("threshold_checks"), Mapping)
        and row["threshold_checks"]
        and all(type(check) is bool for check in row["threshold_checks"].values())
        for row in receipts
    )
    recomputed_pass = receipt_checks_are_boolean and all(
        all(row["threshold_checks"].values())
        and row.get("every_entry_teacher_successor_top1") is True
        for row in receipts
    )
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != RESULT_STATUS
        or result.get("objective_unit")
        != "exact_source_frozen_time_canonical_successor"
        or result.get("panel_kind")
        != "unique_state_single_target_canonical_successor_capacity"
        or tuple(result.get("required_families", ())) != ACTIVE_FAMILIES
        or result.get("unique_entry_count") != 512
        or type(result.get("all_required_families_passed")) is not bool
        or result.get("all_required_families_passed") is not recomputed_pass
        or result.get("repeated_state_empirical_law_evaluated") is not False
        or not isinstance(receipts, list)
        or [row.get("family") for row in receipts if isinstance(row, Mapping)]
        != list(ACTIVE_FAMILIES)
        or any(
            type(row.get("panel_entry_count")) is not int
            or row["panel_entry_count"] != 64
            or not _is_sha(row.get("source_file_sha256"))
            or not _is_sha(row.get("source_result_sha256"))
            or not _is_sha(row.get("initial_model_state_sha256"))
            or not _is_sha(row.get("panel_entry_inventory_sha256"))
            or type(row.get("every_entry_teacher_successor_top1")) is not bool
            for row in receipts
        )
        or not receipt_checks_are_boolean
    ):
        raise ProcessV2T1ScopedResultError("scoped T1 result identity disagrees")
    validate_score_revision_receipt(result.get("score_revision_receipt"))
    for field in (
        "process_identity_sha256",
        "active8_completion_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_entry_inventory_sha256",
        "capacity_policy_sha256",
        "historical_capacity_policy_sha256",
        "policy_scientific_projection_sha256",
        "source_manifest_sha256",
    ):
        if not _is_sha(result.get(field)):
            raise ProcessV2T1ScopedResultError(f"scoped T1 {field} is not a SHA-256")
    return result


def build_process_v2_t1_scoped_capacity_decision(
    result: Mapping[str, Any], *, result_file_sha256: str
) -> dict[str, Any]:
    validated = validate_process_v2_t1_scoped_capacity_result(result)
    if not _is_sha(result_file_sha256):
        raise ProcessV2T1ScopedResultError("scoped T1 result file hash is invalid")
    passed = validated["all_required_families_passed"] is True
    authority = authority_false_block()
    authority["t1_authorized"] = passed
    authority["bounded_p50_authorized"] = passed
    score = validated["score_revision_receipt"]
    body = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_GO_STATUS if passed else DECISION_NO_GO_STATUS,
        **authority,
        "process_identity_sha256": validated["process_identity_sha256"],
        "active8_completion_sha256": validated["active8_completion_sha256"],
        "gate_zero_decision_sha256": validated["gate_zero_decision_sha256"],
        "capacity_policy_sha256": validated["capacity_policy_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "panel_entry_inventory_sha256": validated["panel_entry_inventory_sha256"],
        "current_initial_model_state_sha256": score[
            "current_initial_model_state_sha256"
        ],
        "score_revision_receipt_sha256": score["receipt_sha256"],
        "result_file_sha256": result_file_sha256,
        "result_sha256": validated["result_sha256"],
        "required_families": list(ACTIVE_FAMILIES),
        "unique_entry_count": validated["unique_entry_count"],
        "failed_checks": [] if passed else ["not_all_required_families_passed"],
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def validate_process_v2_t1_scoped_capacity_decision(
    value: object,
    *,
    result: Mapping[str, Any],
    result_file_sha256: str,
    require_p50_go: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1ScopedResultError("scoped T1 decision is not an object")
    expected = build_process_v2_t1_scoped_capacity_decision(
        result, result_file_sha256=result_file_sha256
    )
    if dict(value) != expected:
        raise ProcessV2T1ScopedResultError("scoped T1 decision differs from recomputation")
    if require_p50_go and (
        expected["status"] != DECISION_GO_STATUS
        or expected["t1_authorized"] is not True
        or expected["bounded_p50_authorized"] is not True
        or expected["failed_checks"]
    ):
        raise ProcessV2T1ScopedResultError("scoped T1 decision is not a bounded-P50 GO")
    return expected


__all__ = [
    "ACTIVE_FAMILIES",
    "DECISION_FILENAME",
    "DECISION_GO_STATUS",
    "RESULT_FILENAME",
    "ProcessV2T1ScopedResultError",
    "build_process_v2_t1_scoped_capacity_decision",
    "build_process_v2_t1_scoped_capacity_result",
    "validate_process_v2_t1_scoped_capacity_decision",
    "validate_process_v2_t1_scoped_capacity_result",
    "validate_score_revision_receipt",
    "validate_t1_evidence_source_manifest",
]
