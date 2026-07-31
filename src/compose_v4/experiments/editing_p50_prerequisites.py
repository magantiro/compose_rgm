"""Semantic prerequisite contracts for the bounded COMPOSE P50 pilot.

Physical SHA-256 equality is necessary but not sufficient for launch
authorization.  This module validates the meaning of the frozen Gate-0
evidence, the explicit T1 decision, and the exact P50 recipe, then proves that
all three name the same Active8 corpus and operator contract.

The artifacts authorize only the exactly-50-step development pilot.  They
never authorize P500, P2000, or full scientific training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.active8_trace_inventory import Active8TraceAdmission
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.experiments.editing_gate_zero_runtime import (
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION,
    EDITING_GATE_ZERO_RUNTIME_STATUS,
    GATE_ZERO_ACTIVE8_CENSUS_FIELDS,
    GATE_ZERO_ACTIVE8_IDENTITY_FIELDS,
)
from compose_v4.experiments.editing_t1_successor_runtime import (
    EDITING_T1_LOCAL_ADAPTER_FAMILIES,
    EDITING_T1_RESULT_STATUS,
    EDITING_T1_RUNTIME_CONTRACT_VERSION,
    EditingT1RuntimeContract,
    EditingT1RuntimeError,
    editing_t1_numeric_thresholds_sha256,
    load_editing_t1_result,
    load_editing_t1_runtime_contract,
    require_editing_t1_family_scope_applicable,
    validate_editing_t1_numeric_thresholds,
)
from compose_v4.experiments.editing_t1_panel import (
    ACTIVE8_T1_IDENTITY_FIELDS,
    EDITING_T1_GLOBAL_FAMILY_SELECTOR,
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_REQUIRED_SCOPES,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

T1_P50_DECISION_SCHEMA = "compose.editing.t1_successor_gate_decision"
T1_P50_DECISION_SCHEMA_VERSION = 3
T1_P50_DECISION_PASS_STATUS = "PASS_BOUNDED_P50_PREREQUISITE"
T1_P50_DECISION_NO_GO_STATUS = "NO_GO_BOUNDED_P50_PREREQUISITE"
T1_ARM_RESULTS_MANIFEST_SCHEMA = "compose.editing.t1_arm_results_manifest"
T1_ARM_RESULTS_MANIFEST_VERSION = 1
T1_ARM_RESULTS_MANIFEST_STATUS = "COMPLETE_PHYSICAL_T1_ARM_RESULTS"
P50_RECIPE_SCHEMA = "compose.editing.p50_launch_recipe"
P50_RECIPE_SCHEMA_VERSION = 2
P50_RECIPE_STATUS = "FROZEN_BOUNDED_P50_RECIPE"
P50_DISABLED_FAMILIES = ("ring_system_delete", "ring_system_grow")

P50_RECIPE_LAUNCH_FIELDS = (
    "optimizer_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "schedule_steps",
    "minimum_learning_rate_fraction",
    "seed",
    "initialization_regime",
    "factorized_training_objective",
    "successor_objective_mode",
    "successor_hazard_weight",
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "rate_factorization",
    "empirical_mark_prior_mode",
    "empirical_mark_prior_smoothing",
    "ring_family_mass_mode",
    "ring_template_factorization",
    "trainable_parameter_scope",
    "late_time_fraction",
    "operational_horizon",
    "progress_stratification_fraction",
    "bond_representation",
    "ring_electronic_mode",
    "teacher_ordering",
    "condition_dropout_probability",
    "property_conditions",
    "use_bf16",
    "denovo_weight",
    "corrupted_prior_mix",
    "cycle_op_mix",
    "disable_ring_grow_macro",
    "enable_ring_system_delete",
    "organic_vocabulary",
    "evaluation_every",
    "evaluation_batch_size",
    "validation_examples",
    "validation_size",
    "max_atoms",
    "required_families",
    "disabled_families",
    "resume",
    "early_stopping_patience",
    "benchmark_steps",
    "snapshot_checkpoints",
    "source_corpus_inventory_sha256",
    "successor_cache_inventory_sha256",
    "successor_cache_compatibility_sha256",
    "semantic_sidecar_manifest_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
)

_GATE_ZERO_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "quality_thresholds",
    "training_decision",
    "optimizer_steps",
    "runtime_contract_sha256",
    "active8_admission",
    "source",
    "model",
    "cache_contract",
    "probe",
    "cache_artifacts",
    "support_audit",
    "evidence_sha256",
}
_GATE_ZERO_SOURCE_FIELDS = {
    "semantic_sidecar_manifest_sha256",
    "semantic_sidecar_source_sha256",
    "semantic_census_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    "validation_trace_count",
    "validation_progress_row_count",
    "validation_shards",
}
_GATE_ZERO_CACHE_FIELDS = {
    "support_signature",
    "support_signature_sha256",
    "operator_registry_hash",
    "operator_registry_source_sha256",
    "canonicalizer_contract_sha256",
    "tensorization_implementation_hash",
    "fiber_compiler_implementation_hash",
}
_T1_DECISION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "bounded_p50_authorized",
    "full_training_authorized",
    "thresholds_frozen",
    "pre_result_runtime_contract_relative_path",
    "pre_result_runtime_contract_file_sha256",
    "pre_result_runtime_contract_sha256",
    "numeric_thresholds",
    "numeric_thresholds_sha256",
    "prerequisites",
    "required_families",
    "disabled_families",
    "arm_results_manifest_relative_path",
    "arm_results_manifest_file_sha256",
    "arm_results_manifest_sha256",
    "evaluation",
    "decision_sha256",
}
_T1_ARM_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "runtime_contract_file_sha256",
    "runtime_contract_sha256",
    "numeric_thresholds_sha256",
    "panel_artifact_sha256",
    "panel_selection_sha256",
    "panel_census_sha256",
    "panel_capacity_strata_sha256",
    *ACTIVE8_T1_IDENTITY_FIELDS,
    "planned_arms",
    "results",
    "manifest_sha256",
}
_T1_ARM_FIELDS = {"family", "panel_kind", "scope"}
_T1_ARM_RESULT_FIELDS = {
    *_T1_ARM_FIELDS,
    "relative_path",
    "file_sha256",
    "result_sha256",
    "cache_receipts",
}
_T1_RETAINED_RESULT_RECEIPT_FIELDS = {
    "file_sha256",
    "result_sha256",
    "cache_receipts",
}
_T1_EVALUATION_FIELDS = {
    "planned_arm_count",
    "validated_arm_count",
    "core_unique_all_scope",
    "global_repeated_entropy_all_scope",
    "required_diagnostic_arm_count",
    "failed_checks",
}
_PARENT_IDENTITY_FIELDS = {
    "source_corpus_inventory_file_sha256",
    "source_corpus_inventory_sha256",
    "gate_zero_structural_evidence_file_sha256",
    "unified_packed_manifest_sha256",
    "support_contract_sha256",
}
_RECIPE_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "bounded_p50_authorized",
    "full_training_authorized",
    "prerequisites",
    "launch",
    "launch_sha256",
    "planned_stream",
    "recipe_sha256",
}
_PLANNED_STREAM_FIELDS = {
    "ordered_address_stream_sha256",
    "ordered_training_stream_sha256",
}
_RECIPE_PARENT_IDENTITY_FIELDS = {
    *_PARENT_IDENTITY_FIELDS,
    "t1_successor_gate_decision_file_sha256",
}


class EditingP50PrerequisiteError(ValueError):
    """A bounded-pilot prerequisite is malformed or refers to another run."""


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
        raise EditingP50PrerequisiteError(
            "P50 prerequisite is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise EditingP50PrerequisiteError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _require_exact_mapping(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingP50PrerequisiteError(f"{field} must be an object")
    observed = set(value)
    if observed != expected:
        raise EditingP50PrerequisiteError(
            f"{field} has missing or unknown fields: "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _require_positive_family_census(
    value: object,
    *,
    field: str,
) -> None:
    if not isinstance(value, Mapping):
        raise EditingP50PrerequisiteError(f"{field} must be an object")
    for family in RINGCORE_EDITING_FAMILIES:
        count = value.get(family)
        if type(count) is not int or count <= 0:
            raise EditingP50PrerequisiteError(f"{field} lacks a positive {family} witness")


def _sha256_file(path: Path) -> str:
    if not Path(path).is_file():
        raise EditingP50PrerequisiteError(f"T1 prerequisite artifact is absent: {path}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, field: str) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise EditingP50PrerequisiteError(f"{field} is absent: {source}")
    try:
        payload = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingP50PrerequisiteError(f"{field} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise EditingP50PrerequisiteError(f"{field} must contain a JSON object")
    return payload


def _resolve_relative_artifact(
    parent: Path,
    relative: object,
    *,
    field: str,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EditingP50PrerequisiteError(f"{field} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise EditingP50PrerequisiteError(f"{field} is not a safe relative path")
    root = Path(parent).resolve()
    candidate = root.joinpath(*pure.parts).resolve()
    if not candidate.is_relative_to(root):
        raise EditingP50PrerequisiteError(f"{field} escapes its artifact directory")
    return candidate


def planned_t1_arms(
    contract: EditingT1RuntimeContract,
) -> tuple[Mapping[str, str], ...]:
    """Return the complete mandatory capacity and diagnostic arm matrix."""

    if not isinstance(contract, EditingT1RuntimeContract):
        raise TypeError("planned T1 arms require a validated v4 runtime contract")

    def applicable_scopes(family: str) -> tuple[str, ...]:
        result = []
        for scope in EDITING_T1_REQUIRED_SCOPES:
            if (
                scope == "heads_plus_local_adapter"
                and family != EDITING_T1_GLOBAL_FAMILY_SELECTOR
                and family not in EDITING_T1_LOCAL_ADAPTER_FAMILIES
            ):
                continue
            try:
                require_editing_t1_family_scope_applicable(family, scope)
            except EditingT1RuntimeError as error:
                raise EditingP50PrerequisiteError(
                    f"T1 scope ladder failed for {family}/{scope}: {error}"
                ) from error
            result.append(scope)
        if "all" not in result or "heads_only" not in result:
            raise EditingP50PrerequisiteError(
                f"T1 family {family} lacks mandatory all/heads-only scopes"
            )
        return tuple(result)

    arms: list[Mapping[str, str]] = []
    for family in RINGCORE_EDITING_FAMILIES:
        for scope in applicable_scopes(family):
            arms.append(
                {
                    "family": family,
                    "panel_kind": EDITING_T1_UNIQUE_PANEL_KIND,
                    "scope": scope,
                }
            )
    for scope in applicable_scopes(EDITING_T1_GLOBAL_FAMILY_SELECTOR):
        arms.append(
            {
                "family": EDITING_T1_GLOBAL_FAMILY_SELECTOR,
                "panel_kind": EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
                "scope": scope,
            }
        )
    for family in contract.payload["within_family_repeated_families"]:
        for scope in applicable_scopes(str(family)):
            arms.append(
                {
                    "family": str(family),
                    "panel_kind": (EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND),
                    "scope": scope,
                }
            )
    keys = [(arm["family"], arm["panel_kind"], arm["scope"]) for arm in arms]
    if len(keys) != len(set(keys)):
        raise EditingP50PrerequisiteError("planned T1 arm matrix contains duplicates")
    return tuple(arms)


def _arm_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    arm = _require_exact_mapping(value, _T1_ARM_FIELDS, field="T1 arm")
    family = arm["family"]
    panel_kind = arm["panel_kind"]
    scope = arm["scope"]
    if not all(isinstance(item, str) and item for item in (family, panel_kind, scope)):
        raise EditingP50PrerequisiteError("T1 arm fields must be nonempty strings")
    return str(family), str(panel_kind), str(scope)


def build_t1_arm_results_manifest(
    *,
    runtime_contract_path: Path,
    result_paths: Mapping[tuple[str, str, str], Path],
    retained_result_receipts: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    manifest_directory: Path,
) -> dict[str, Any]:
    """Build a physical manifest from a complete matrix and retained run receipts.

    ``retained_result_receipts`` must come from the launch orchestrator rather
    than being reconstructed from the durable result files. This keeps physical
    file, result, and cache identities independent of the payload being checked.
    """

    contract_path = Path(runtime_contract_path)
    contract = load_editing_t1_runtime_contract(contract_path)
    planned = planned_t1_arms(contract)
    planned_keys = tuple(_arm_key(arm) for arm in planned)
    if set(result_paths) != set(planned_keys):
        raise EditingP50PrerequisiteError(
            "T1 result paths must exactly cover the complete planned arm matrix"
        )
    if set(retained_result_receipts) != set(planned_keys):
        raise EditingP50PrerequisiteError(
            "T1 retained result receipts must exactly cover the complete planned arm matrix"
        )
    root = Path(manifest_directory).resolve()
    active8_identity = {field: contract.payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS}
    results: list[dict[str, Any]] = []
    for arm, key in zip(planned, planned_keys, strict=True):
        path = Path(result_paths[key]).resolve()
        if not path.is_relative_to(root):
            raise EditingP50PrerequisiteError(
                "T1 result path must be inside the manifest directory"
            )
        receipt = _require_exact_mapping(
            retained_result_receipts[key],
            _T1_RETAINED_RESULT_RECEIPT_FIELDS,
            field="retained T1 arm result receipt",
        )
        file_sha256 = _require_sha256(
            receipt["file_sha256"],
            field="retained T1 arm result file_sha256",
        )
        result_sha256 = _require_sha256(
            receipt["result_sha256"],
            field="retained T1 arm result result_sha256",
        )
        if _sha256_file(path) != file_sha256:
            raise EditingP50PrerequisiteError(
                "T1 arm result disagrees with its independently retained physical SHA-256"
            )
        try:
            load_editing_t1_result(
                path,
                expected_result_sha256=result_sha256,
                expected_contract_sha256=contract.sha256,
                expected_numeric_thresholds_sha256=(contract.numeric_thresholds_sha256),
                expected_panel_artifact_sha256=str(contract.payload["panel_artifact_sha256"]),
                expected_panel_selection_sha256=str(contract.payload["panel_selection_sha256"]),
                expected_panel_census_sha256=str(contract.payload["panel_census_sha256"]),
                expected_panel_capacity_strata_sha256=str(
                    contract.payload["panel_capacity_strata_sha256"]
                ),
                expected_active8_identity=active8_identity,
                expected_cache_receipts=receipt["cache_receipts"],
            )
        except EditingT1RuntimeError as error:
            raise EditingP50PrerequisiteError(
                f"T1 arm result disagrees with its retained run receipt: {error}"
            ) from error
        results.append(
            {
                **dict(arm),
                "relative_path": path.relative_to(root).as_posix(),
                "file_sha256": file_sha256,
                "result_sha256": result_sha256,
                "cache_receipts": receipt["cache_receipts"],
            }
        )
    body = {
        "schema": T1_ARM_RESULTS_MANIFEST_SCHEMA,
        "schema_version": T1_ARM_RESULTS_MANIFEST_VERSION,
        "status": T1_ARM_RESULTS_MANIFEST_STATUS,
        "training_authorized": False,
        "runtime_contract_file_sha256": _sha256_file(contract_path),
        "runtime_contract_sha256": contract.sha256,
        "numeric_thresholds_sha256": contract.numeric_thresholds_sha256,
        "panel_artifact_sha256": contract.payload["panel_artifact_sha256"],
        "panel_selection_sha256": contract.payload["panel_selection_sha256"],
        "panel_census_sha256": contract.payload["panel_census_sha256"],
        "panel_capacity_strata_sha256": contract.payload["panel_capacity_strata_sha256"],
        **{field: contract.payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS},
        "planned_arms": [dict(arm) for arm in planned],
        "results": results,
    }
    return {**body, "manifest_sha256": _stable_sha256(body)}


def validate_t1_arm_results_manifest(
    payload: object,
    *,
    manifest_path: Path,
    runtime_contract: EditingT1RuntimeContract,
    runtime_contract_file_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[tuple[str, str, str], Mapping[str, Any]]]:
    """Validate every physical result and return the exact arm-result index."""

    manifest = _require_exact_mapping(
        payload,
        _T1_ARM_MANIFEST_FIELDS,
        field="T1 arm-results manifest",
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != T1_ARM_RESULTS_MANIFEST_SCHEMA
        or manifest["schema_version"] != T1_ARM_RESULTS_MANIFEST_VERSION
        or manifest["status"] != T1_ARM_RESULTS_MANIFEST_STATUS
        or manifest["training_authorized"] is not False
        or manifest["manifest_sha256"] != _stable_sha256(body)
        or manifest["runtime_contract_file_sha256"] != runtime_contract_file_sha256
        or manifest["runtime_contract_sha256"] != runtime_contract.sha256
        or manifest["numeric_thresholds_sha256"] != runtime_contract.numeric_thresholds_sha256
    ):
        raise EditingP50PrerequisiteError(
            "T1 arm-results manifest identity, status, or runtime binding is invalid"
        )
    for field in (
        "panel_artifact_sha256",
        "panel_selection_sha256",
        "panel_census_sha256",
        "panel_capacity_strata_sha256",
        *ACTIVE8_T1_IDENTITY_FIELDS,
    ):
        if manifest[field] != runtime_contract.payload[field]:
            raise EditingP50PrerequisiteError(
                f"T1 arm-results manifest disagrees with runtime contract for {field}"
            )
    planned = planned_t1_arms(runtime_contract)
    if manifest["planned_arms"] != [dict(arm) for arm in planned]:
        raise EditingP50PrerequisiteError(
            "T1 arm-results manifest planned matrix is incomplete or reordered"
        )
    entries = manifest["results"]
    if not isinstance(entries, list) or len(entries) != len(planned):
        raise EditingP50PrerequisiteError(
            "T1 arm-results manifest lacks the complete result matrix"
        )
    active8_identity = {
        field: runtime_contract.payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS
    }
    result_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    root = Path(manifest_path).parent
    for planned_arm, raw_entry in zip(planned, entries, strict=True):
        entry = _require_exact_mapping(
            raw_entry,
            _T1_ARM_RESULT_FIELDS,
            field="T1 arm-results entry",
        )
        key = _arm_key({field: entry[field] for field in _T1_ARM_FIELDS})
        if key != _arm_key(planned_arm) or key in result_index:
            raise EditingP50PrerequisiteError(
                "T1 arm-results entry is duplicated, missing, or reordered"
            )
        path = _resolve_relative_artifact(
            root,
            entry["relative_path"],
            field="T1 arm result relative_path",
        )
        if _sha256_file(path) != entry["file_sha256"]:
            raise EditingP50PrerequisiteError("T1 arm result physical SHA-256 mismatch")
        try:
            result = load_editing_t1_result(
                path,
                expected_result_sha256=str(entry["result_sha256"]),
                expected_contract_sha256=runtime_contract.sha256,
                expected_numeric_thresholds_sha256=(runtime_contract.numeric_thresholds_sha256),
                expected_panel_artifact_sha256=str(
                    runtime_contract.payload["panel_artifact_sha256"]
                ),
                expected_panel_selection_sha256=str(
                    runtime_contract.payload["panel_selection_sha256"]
                ),
                expected_panel_census_sha256=str(runtime_contract.payload["panel_census_sha256"]),
                expected_panel_capacity_strata_sha256=str(
                    runtime_contract.payload["panel_capacity_strata_sha256"]
                ),
                expected_active8_identity=active8_identity,
                expected_cache_receipts=entry["cache_receipts"],
            )
        except EditingT1RuntimeError as error:
            raise EditingP50PrerequisiteError(
                f"T1 arm result failed v4 validation: {error}"
            ) from error
        if (
            result["status"] != EDITING_T1_RESULT_STATUS
            or (result["family"], result["panel_kind"], result["scope"]) != key
            or result["training_report"]["steps"] != runtime_contract.optimization["steps"]
        ):
            raise EditingP50PrerequisiteError(
                "T1 arm result role or optimization length is off-contract"
            )
        result_index[key] = result
    return manifest, result_index


def validate_gate_zero_structural_evidence(
    payload: object,
    *,
    source_admission: Active8TraceAdmission,
) -> Mapping[str, Any]:
    """Validate structural evidence and bind it to the Active8 source."""

    evidence = _require_exact_mapping(
        payload,
        _GATE_ZERO_FIELDS,
        field="Gate0 structural evidence",
    )
    body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if (
        evidence["schema"] != EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA
        or evidence["schema_version"] != EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION
        or evidence["status"] != EDITING_GATE_ZERO_RUNTIME_STATUS
        or evidence["training_authorized"] is not False
        or evidence["quality_thresholds"] is not None
        or evidence["training_decision"] is not None
        or evidence["optimizer_steps"] != 0
        or evidence["evidence_sha256"] != _stable_sha256(body)
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 structural evidence status or self-hash is invalid"
        )
    runtime_contract_sha256 = _require_sha256(
        evidence["runtime_contract_sha256"],
        field="Gate0 runtime_contract_sha256",
    )
    if runtime_contract_sha256 != source_admission.support_contract_sha256:
        raise EditingP50PrerequisiteError(
            "Gate0 evidence and Active8 inventory name different support contracts"
        )

    active8 = _require_exact_mapping(
        evidence["active8_admission"],
        {
            *GATE_ZERO_ACTIVE8_IDENTITY_FIELDS,
            *GATE_ZERO_ACTIVE8_CENSUS_FIELDS,
        },
        field="Gate0 Active8 admission",
    )
    expected_active8_identity = {
        "active8_inventory_manifest_file_sha256": (
            source_admission.manifest_file_sha256
        ),
        "active8_inventory_sha256": source_admission.inventory_sha256,
        "active8_effective_source_corpus_cache_sha256": (
            source_admission.effective_source_corpus_cache_sha256
        ),
        "active8_unified_packed_manifest_sha256": (
            source_admission.unified_packed_manifest_sha256
        ),
        "active8_support_contract_sha256": (
            source_admission.support_contract_sha256
        ),
    }
    observed_active8_identity = {
        field: active8[field] for field in GATE_ZERO_ACTIVE8_IDENTITY_FIELDS
    }
    if observed_active8_identity != expected_active8_identity:
        raise EditingP50PrerequisiteError(
            "Gate0 evidence and Active8 inventory identities disagree"
        )
    if active8["partition"] != "validation":
        raise EditingP50PrerequisiteError(
            "Gate0 Active8 admission must describe the validation partition"
        )
    count_fields = (
        "source_shard_count",
        "source_trace_count",
        "source_progress_row_count",
        "admitted_trace_count",
        "excluded_trace_count",
        "admitted_progress_row_count",
        "admitted_nonterminal_row_count",
        "admitted_terminal_row_count",
        "admitted_nonempty_semantic_cell_count",
    )
    if any(type(active8[field]) is not int or active8[field] < 0 for field in count_fields):
        raise EditingP50PrerequisiteError(
            "Gate0 Active8 admission counts must be nonnegative integers"
        )
    if (
        active8["source_shard_count"] <= 0
        or active8["source_trace_count"] <= 0
        or active8["source_progress_row_count"] <= 0
        or active8["admitted_trace_count"] <= 0
        or active8["admitted_progress_row_count"] <= 0
        or active8["admitted_nonempty_semantic_cell_count"] <= 0
        or active8["source_trace_count"]
        != active8["admitted_trace_count"] + active8["excluded_trace_count"]
        or active8["admitted_terminal_row_count"]
        != active8["admitted_trace_count"]
        or active8["admitted_progress_row_count"]
        != active8["admitted_nonterminal_row_count"]
        + active8["admitted_terminal_row_count"]
        or active8["source_progress_row_count"]
        < active8["admitted_progress_row_count"]
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 Active8 admission census is internally inconsistent"
        )
    _require_positive_family_census(
        active8["admitted_teacher_examples_by_family"],
        field="Gate0 Active8 admitted teacher census",
    )

    source = _require_exact_mapping(
        evidence["source"],
        _GATE_ZERO_SOURCE_FIELDS,
        field="Gate0 source",
    )
    if source["unified_packed_manifest_sha256"] != source_admission.unified_packed_manifest_sha256:
        raise EditingP50PrerequisiteError(
            "Gate0 evidence and Active8 inventory name different unified corpora"
        )
    for field in (
        "semantic_sidecar_manifest_sha256",
        "semantic_sidecar_source_sha256",
        "semantic_census_sha256",
        "unified_packed_manifest_sha256",
        "representability_overlay_sha256",
    ):
        _require_sha256(source[field], field=f"Gate0 source.{field}")
    for field in ("validation_trace_count", "validation_progress_row_count"):
        if type(source[field]) is not int or source[field] <= 0:
            raise EditingP50PrerequisiteError(f"Gate0 source.{field} must be positive")
    if not isinstance(source["validation_shards"], list) or not source["validation_shards"]:
        raise EditingP50PrerequisiteError("Gate0 source must name at least one validation shard")
    if (
        active8["source_shard_count"] != len(source["validation_shards"])
        or active8["admitted_trace_count"] != source["validation_trace_count"]
        or active8["admitted_progress_row_count"]
        != source["validation_progress_row_count"]
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 source and Active8 admitted census disagree"
        )

    cache_contract = _require_exact_mapping(
        evidence["cache_contract"],
        _GATE_ZERO_CACHE_FIELDS,
        field="Gate0 cache contract",
    )
    support_signature = cache_contract["support_signature"]
    if not isinstance(support_signature, Mapping):
        raise EditingP50PrerequisiteError("Gate0 support signature must be an object")
    if cache_contract["support_signature_sha256"] != _stable_sha256(support_signature):
        raise EditingP50PrerequisiteError("Gate0 support-signature self-hash is invalid")
    if (
        support_signature.get("charge_policy") != CHARGE_POLICY_VERSION
        or support_signature.get("enable_ring_restates") is not True
        or support_signature.get("enable_cycle_ops") is not True
        or support_signature.get("enable_ring_opening") is not True
        or support_signature.get("enable_ring_grow_macro") is not False
        or support_signature.get("enable_ring_system_delete") is not False
        or support_signature.get("max_atoms") != 40
        or support_signature.get("atom_insert_arity_support") != [0, 1]
        or support_signature.get("embedded_jump_chain_policy") != "productive_canonical_successors"
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 support signature is not the bounded Active8 editing object"
        )
    ordered_families = support_signature.get("ordered_family_vocabulary")
    if not isinstance(ordered_families, list) or not set(RINGCORE_EDITING_FAMILIES).issubset(
        ordered_families
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 support signature lacks the required Active8 family vocabulary"
        )

    support_audit = evidence["support_audit"]
    if not isinstance(support_audit, Mapping):
        raise EditingP50PrerequisiteError("Gate0 support audit must be an object")
    for field in (
        "teacher_examples_by_family",
        "candidate_marks_by_family",
        "productive_successors_by_family",
    ):
        _require_positive_family_census(
            support_audit.get(field),
            field=f"Gate0 support_audit.{field}",
        )
    if not isinstance(evidence["cache_artifacts"], list) or not evidence["cache_artifacts"]:
        raise EditingP50PrerequisiteError("Gate0 evidence must contain immutable cache receipts")
    return evidence


def _finite_metric(
    metrics: Mapping[str, Any],
    field: str,
    *,
    arm: tuple[str, str, str],
) -> float:
    value = metrics.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise EditingP50PrerequisiteError(f"T1 arm {arm!r} lacks finite final metric {field}")
    return float(value)


def build_t1_p50_decision(
    *,
    runtime_contract_path: Path,
    arm_results_manifest_path: Path,
    decision_directory: Path,
    source_admission: Active8TraceAdmission,
    source_inventory_file_sha256: str,
    gate_zero_evidence_file_sha256: str,
) -> dict[str, Any]:
    """Recompute the bounded-P50 decision from complete physical T1 evidence."""

    runtime_path = Path(runtime_contract_path).resolve()
    manifest_path = Path(arm_results_manifest_path).resolve()
    decision_root = Path(decision_directory).resolve()
    for path, field in (
        (runtime_path, "T1 runtime contract"),
        (manifest_path, "T1 arm-results manifest"),
    ):
        if not path.is_relative_to(decision_root):
            raise EditingP50PrerequisiteError(
                f"{field} must be inside the decision artifact directory"
            )
    try:
        runtime_contract = load_editing_t1_runtime_contract(runtime_path)
    except EditingT1RuntimeError as error:
        raise EditingP50PrerequisiteError(
            f"T1 v4 runtime contract failed validation: {error}"
        ) from error
    if runtime_contract.payload["schema_version"] != EDITING_T1_RUNTIME_CONTRACT_VERSION:
        raise EditingP50PrerequisiteError("T1 decision requires the frozen v4 contract")
    runtime_file_sha256 = _sha256_file(runtime_path)
    manifest_payload = _load_json(
        manifest_path,
        field="T1 arm-results manifest",
    )
    manifest, results = validate_t1_arm_results_manifest(
        manifest_payload,
        manifest_path=manifest_path,
        runtime_contract=runtime_contract,
        runtime_contract_file_sha256=runtime_file_sha256,
    )
    thresholds = runtime_contract.numeric_thresholds
    failures: list[str] = []
    core_unique: dict[str, Any] = {}
    for family in RINGCORE_EDITING_FAMILIES:
        key = (family, EDITING_T1_UNIQUE_PANEL_KIND, "all")
        result = results[key]
        final = result["training_report"]["final"]
        top1 = _finite_metric(
            final,
            "teacher_successor_top1_recall",
            arm=key,
        )
        probability = _finite_metric(
            final,
            "teacher_successor_probability",
            arm=key,
        )
        nll = _finite_metric(final, "canonical_successor_nll", arm=key)
        checks = {
            "minimum_unique_state_teacher_successor_top1": (
                top1 >= thresholds["minimum_unique_state_teacher_successor_top1"]
            ),
            "minimum_unique_state_teacher_successor_probability": (
                probability >= thresholds["minimum_unique_state_teacher_successor_probability"]
            ),
            "maximum_unique_state_teacher_successor_nll": (
                nll <= thresholds["maximum_unique_state_teacher_successor_nll"]
            ),
        }
        core_unique[family] = {
            "result_sha256": result["result_sha256"],
            "teacher_successor_top1_recall": top1,
            "teacher_successor_probability": probability,
            "canonical_successor_nll": nll,
            "checks": checks,
            "pass": all(checks.values()),
        }
        if not all(checks.values()):
            failures.append(f"core_unique_all_scope:{family}")

    entropy_key = (
        EDITING_T1_GLOBAL_FAMILY_SELECTOR,
        EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
        "all",
    )
    entropy_result = results[entropy_key]
    entropy_value = _finite_metric(
        entropy_result["training_report"]["final"],
        "repeated_state_excess_nll_over_empirical_entropy",
        arm=entropy_key,
    )
    entropy_pass = (
        entropy_value
        <= thresholds["maximum_global_repeated_state_excess_nll_over_empirical_entropy"]
    )
    if not entropy_pass:
        failures.append("global_repeated_entropy_all_scope")

    core_keys = {
        (family, EDITING_T1_UNIQUE_PANEL_KIND, "all") for family in RINGCORE_EDITING_FAMILIES
    }
    core_keys.add(entropy_key)
    evaluation = {
        "planned_arm_count": len(manifest["planned_arms"]),
        "validated_arm_count": len(results),
        "core_unique_all_scope": core_unique,
        "global_repeated_entropy_all_scope": {
            "result_sha256": entropy_result["result_sha256"],
            "repeated_state_excess_nll_over_empirical_entropy": entropy_value,
            "pass": entropy_pass,
        },
        "required_diagnostic_arm_count": len(set(results) - core_keys),
        "failed_checks": failures,
    }
    passed = not failures
    body = {
        "schema": T1_P50_DECISION_SCHEMA,
        "schema_version": T1_P50_DECISION_SCHEMA_VERSION,
        "status": (T1_P50_DECISION_PASS_STATUS if passed else T1_P50_DECISION_NO_GO_STATUS),
        "bounded_p50_authorized": passed,
        "full_training_authorized": False,
        "thresholds_frozen": True,
        "pre_result_runtime_contract_relative_path": (
            runtime_path.relative_to(decision_root).as_posix()
        ),
        "pre_result_runtime_contract_file_sha256": runtime_file_sha256,
        "pre_result_runtime_contract_sha256": runtime_contract.sha256,
        "numeric_thresholds": dict(thresholds),
        "numeric_thresholds_sha256": (runtime_contract.numeric_thresholds_sha256),
        "prerequisites": {
            "source_corpus_inventory_file_sha256": source_inventory_file_sha256,
            "source_corpus_inventory_sha256": source_admission.inventory_sha256,
            "gate_zero_structural_evidence_file_sha256": (gate_zero_evidence_file_sha256),
            "unified_packed_manifest_sha256": (source_admission.unified_packed_manifest_sha256),
            "support_contract_sha256": source_admission.support_contract_sha256,
        },
        "required_families": list(RINGCORE_EDITING_FAMILIES),
        "disabled_families": list(P50_DISABLED_FAMILIES),
        "arm_results_manifest_relative_path": (manifest_path.relative_to(decision_root).as_posix()),
        "arm_results_manifest_file_sha256": _sha256_file(manifest_path),
        "arm_results_manifest_sha256": manifest["manifest_sha256"],
        "evaluation": evaluation,
    }
    return {**body, "decision_sha256": _stable_sha256(body)}


def validate_t1_p50_decision(
    payload: object,
    *,
    decision_path: Path,
    source_admission: Active8TraceAdmission,
    source_inventory_file_sha256: str,
    gate_zero_evidence_file_sha256: str,
) -> Mapping[str, Any]:
    """Require an explicit passing T1 decision, never a decision-free arm."""

    decision = _require_exact_mapping(
        payload,
        _T1_DECISION_FIELDS,
        field="T1 P50 decision",
    )
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    if (
        decision["schema"] != T1_P50_DECISION_SCHEMA
        or decision["schema_version"] != T1_P50_DECISION_SCHEMA_VERSION
        or decision["full_training_authorized"] is not False
        or decision["thresholds_frozen"] is not True
        or decision["decision_sha256"] != _stable_sha256(body)
        or tuple(decision["required_families"]) != RINGCORE_EDITING_FAMILIES
        or tuple(decision["disabled_families"]) != P50_DISABLED_FAMILIES
    ):
        raise EditingP50PrerequisiteError(
            "T1 decision identity, family support, or self-hash is invalid"
        )
    if (
        decision["status"] != T1_P50_DECISION_PASS_STATUS
        or decision["bounded_p50_authorized"] is not True
    ):
        raise EditingP50PrerequisiteError(
            "T1 decision is not an exact passing bounded-P50 decision"
        )
    _require_sha256(
        decision["pre_result_runtime_contract_file_sha256"],
        field="T1 pre_result_runtime_contract_file_sha256",
    )
    _require_sha256(
        decision["pre_result_runtime_contract_sha256"],
        field="T1 pre_result_runtime_contract_sha256",
    )
    _require_sha256(
        decision["numeric_thresholds_sha256"],
        field="T1 numeric_thresholds_sha256",
    )
    try:
        numeric_thresholds = validate_editing_t1_numeric_thresholds(decision["numeric_thresholds"])
        observed_thresholds_sha256 = editing_t1_numeric_thresholds_sha256(numeric_thresholds)
    except EditingT1RuntimeError as error:
        raise EditingP50PrerequisiteError(
            f"T1 decision numeric thresholds are invalid: {error}"
        ) from error
    if decision["numeric_thresholds_sha256"] != observed_thresholds_sha256:
        raise EditingP50PrerequisiteError(
            "T1 decision numeric thresholds disagree with their prospective hash"
        )
    parents = _require_exact_mapping(
        decision["prerequisites"],
        _PARENT_IDENTITY_FIELDS,
        field="T1 decision prerequisites",
    )
    expected = {
        "source_corpus_inventory_file_sha256": source_inventory_file_sha256,
        "source_corpus_inventory_sha256": source_admission.inventory_sha256,
        "gate_zero_structural_evidence_file_sha256": (gate_zero_evidence_file_sha256),
        "unified_packed_manifest_sha256": (source_admission.unified_packed_manifest_sha256),
        "support_contract_sha256": source_admission.support_contract_sha256,
    }
    if dict(parents) != expected:
        raise EditingP50PrerequisiteError(
            "T1 decision parent identities disagree with Active8 or Gate0"
        )
    _require_sha256(
        decision["arm_results_manifest_file_sha256"],
        field="T1 arm_results_manifest_file_sha256",
    )
    _require_sha256(
        decision["arm_results_manifest_sha256"],
        field="T1 arm_results_manifest_sha256",
    )
    _require_exact_mapping(
        decision["evaluation"],
        _T1_EVALUATION_FIELDS,
        field="T1 decision evaluation",
    )

    decision_root = Path(decision_path).resolve().parent
    runtime_path = _resolve_relative_artifact(
        decision_root,
        decision["pre_result_runtime_contract_relative_path"],
        field="T1 runtime contract relative path",
    )
    manifest_path = _resolve_relative_artifact(
        decision_root,
        decision["arm_results_manifest_relative_path"],
        field="T1 arm-results manifest relative path",
    )
    if (
        _sha256_file(runtime_path) != decision["pre_result_runtime_contract_file_sha256"]
        or _sha256_file(manifest_path) != decision["arm_results_manifest_file_sha256"]
    ):
        raise EditingP50PrerequisiteError("T1 decision physical runtime or manifest hash mismatch")
    recomputed = build_t1_p50_decision(
        runtime_contract_path=runtime_path,
        arm_results_manifest_path=manifest_path,
        decision_directory=decision_root,
        source_admission=source_admission,
        source_inventory_file_sha256=source_inventory_file_sha256,
        gate_zero_evidence_file_sha256=gate_zero_evidence_file_sha256,
    )
    if dict(decision) != recomputed:
        raise EditingP50PrerequisiteError(
            "T1 decision disagrees with physical v4 arm-result recomputation"
        )
    if (
        decision["status"] != T1_P50_DECISION_PASS_STATUS
        or decision["bounded_p50_authorized"] is not True
        or decision["evaluation"]["failed_checks"] != []
    ):
        raise EditingP50PrerequisiteError(
            "T1 decision is not an exact passing bounded-P50 decision"
        )
    return decision


def validate_p50_recipe(
    payload: object,
    *,
    source_admission: Active8TraceAdmission,
    source_inventory_file_sha256: str,
    gate_zero_evidence_file_sha256: str,
    t1_decision_file_sha256: str,
    expected_launch: Mapping[str, object],
) -> Mapping[str, Any]:
    """Validate the frozen recipe and its exact live launcher projection."""

    recipe = _require_exact_mapping(
        payload,
        _RECIPE_FIELDS,
        field="P50 recipe",
    )
    body = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    if (
        recipe["schema"] != P50_RECIPE_SCHEMA
        or recipe["schema_version"] != P50_RECIPE_SCHEMA_VERSION
        or recipe["status"] != P50_RECIPE_STATUS
        or recipe["bounded_p50_authorized"] is not True
        or recipe["full_training_authorized"] is not False
        or recipe["recipe_sha256"] != _stable_sha256(body)
    ):
        raise EditingP50PrerequisiteError("P50 recipe status or self-hash is invalid")
    parents = _require_exact_mapping(
        recipe["prerequisites"],
        _RECIPE_PARENT_IDENTITY_FIELDS,
        field="P50 recipe prerequisites",
    )
    expected_parents = {
        "source_corpus_inventory_file_sha256": source_inventory_file_sha256,
        "source_corpus_inventory_sha256": source_admission.inventory_sha256,
        "gate_zero_structural_evidence_file_sha256": (gate_zero_evidence_file_sha256),
        "unified_packed_manifest_sha256": (source_admission.unified_packed_manifest_sha256),
        "support_contract_sha256": source_admission.support_contract_sha256,
        "t1_successor_gate_decision_file_sha256": t1_decision_file_sha256,
    }
    if dict(parents) != expected_parents:
        raise EditingP50PrerequisiteError(
            "P50 recipe parent identities disagree with the frozen prerequisites"
        )
    launch = _require_exact_mapping(
        recipe["launch"],
        set(P50_RECIPE_LAUNCH_FIELDS),
        field="P50 recipe launch",
    )
    expected_launch_mapping = _require_exact_mapping(
        expected_launch,
        set(P50_RECIPE_LAUNCH_FIELDS),
        field="live P50 launch projection",
    )
    if recipe["launch_sha256"] != _stable_sha256(launch):
        raise EditingP50PrerequisiteError("P50 recipe launch self-hash is invalid")
    successor_mode = launch["successor_objective_mode"]
    successor_hazard_weight = launch["successor_hazard_weight"]
    if (
        successor_mode
        not in {
            "productive_identity",
            "productive_identity_plus_hazard",
        }
        or not isinstance(successor_hazard_weight, (int, float))
        or isinstance(successor_hazard_weight, bool)
        or not math.isfinite(float(successor_hazard_weight))
        or float(successor_hazard_weight) < 0.0
        or (successor_mode == "productive_identity" and float(successor_hazard_weight) != 0.0)
        or (
            successor_mode == "productive_identity_plus_hazard"
            and float(successor_hazard_weight) <= 0.0
        )
    ):
        raise EditingP50PrerequisiteError(
            "P50 successor identity and separately weighted hazard settings are inconsistent"
        )
    if (
        launch["optimizer_steps"] != 50
        or launch["resume"] is not False
        or launch["max_atoms"] != 40
        or launch["hidden_dim"] != 256
        or launch["message_passing_steps"] != 6
        or launch["mark_dim"] != 32
        or launch["rate_factorization"] != "hierarchical"
        or launch["empirical_mark_prior_mode"] != "none"
        or launch["ring_family_mass_mode"] != "boolean"
        or launch["ring_template_factorization"] != "flat"
        or tuple(launch["required_families"]) != RINGCORE_EDITING_FAMILIES
        or tuple(launch["disabled_families"]) != P50_DISABLED_FAMILIES
        or launch["factorized_training_objective"] != "canonical_successor"
        or launch["organic_vocabulary"] is not True
        or launch["disable_ring_grow_macro"] is not True
        or launch["enable_ring_system_delete"] is not False
        or launch["corrupted_prior_mix"] is not True
        or launch["cycle_op_mix"] is not True
        or isinstance(launch["denovo_weight"], bool)
        or launch["denovo_weight"] != 0.0
        or launch["trainable_parameter_scope"] != "all"
        or launch["property_conditions"] != []
        or launch["condition_dropout_probability"] != 0.0
        or launch["source_corpus_inventory_sha256"] != source_admission.inventory_sha256
        or launch["unified_packed_manifest_sha256"]
        != source_admission.unified_packed_manifest_sha256
    ):
        raise EditingP50PrerequisiteError(
            "P50 recipe is not the exact non-resumable Active8 successor pilot"
        )
    if dict(launch) != dict(expected_launch_mapping):
        raise EditingP50PrerequisiteError(
            "live P50 launcher arguments differ from the frozen recipe"
        )
    planned_stream = _require_exact_mapping(
        recipe["planned_stream"],
        _PLANNED_STREAM_FIELDS,
        field="P50 recipe planned stream",
    )
    for field in _PLANNED_STREAM_FIELDS:
        _require_sha256(
            planned_stream[field],
            field=f"P50 recipe planned_stream.{field}",
        )
    return recipe


@dataclass(frozen=True)
class VerifiedP50Prerequisites:
    """Cross-linked semantic identities safe to record in P50 metadata."""

    physical_sha256: Mapping[str, str]
    source_inventory_sha256: str
    unified_packed_manifest_sha256: str
    support_contract_sha256: str
    gate_zero_evidence_sha256: str
    t1_decision_sha256: str
    t1_pre_result_runtime_contract_file_sha256: str
    t1_pre_result_runtime_contract_sha256: str
    t1_numeric_thresholds_sha256: str
    t1_arm_results_manifest_file_sha256: str
    t1_arm_results_manifest_sha256: str
    recipe_sha256: str
    launch_sha256: str
    ordered_address_stream_sha256: str
    ordered_training_stream_sha256: str

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "physical_sha256": dict(self.physical_sha256),
            "source_inventory_sha256": self.source_inventory_sha256,
            "unified_packed_manifest_sha256": (self.unified_packed_manifest_sha256),
            "support_contract_sha256": self.support_contract_sha256,
            "gate_zero_evidence_sha256": self.gate_zero_evidence_sha256,
            "t1_decision_sha256": self.t1_decision_sha256,
            "t1_pre_result_runtime_contract_file_sha256": (
                self.t1_pre_result_runtime_contract_file_sha256
            ),
            "t1_pre_result_runtime_contract_sha256": (self.t1_pre_result_runtime_contract_sha256),
            "t1_numeric_thresholds_sha256": (self.t1_numeric_thresholds_sha256),
            "t1_arm_results_manifest_file_sha256": (self.t1_arm_results_manifest_file_sha256),
            "t1_arm_results_manifest_sha256": (self.t1_arm_results_manifest_sha256),
            "recipe_sha256": self.recipe_sha256,
            "launch_sha256": self.launch_sha256,
            "ordered_address_stream_sha256": (self.ordered_address_stream_sha256),
            "ordered_training_stream_sha256": (self.ordered_training_stream_sha256),
            "bounded_p50_authorized": True,
            "full_training_authorized": False,
        }


__all__ = [
    "P50_DISABLED_FAMILIES",
    "P50_RECIPE_LAUNCH_FIELDS",
    "P50_RECIPE_SCHEMA",
    "P50_RECIPE_SCHEMA_VERSION",
    "P50_RECIPE_STATUS",
    "T1_P50_DECISION_PASS_STATUS",
    "T1_P50_DECISION_NO_GO_STATUS",
    "T1_P50_DECISION_SCHEMA",
    "T1_P50_DECISION_SCHEMA_VERSION",
    "T1_ARM_RESULTS_MANIFEST_SCHEMA",
    "T1_ARM_RESULTS_MANIFEST_STATUS",
    "T1_ARM_RESULTS_MANIFEST_VERSION",
    "EditingP50PrerequisiteError",
    "VerifiedP50Prerequisites",
    "build_t1_arm_results_manifest",
    "build_t1_p50_decision",
    "planned_t1_arms",
    "validate_gate_zero_structural_evidence",
    "validate_p50_recipe",
    "validate_t1_arm_results_manifest",
    "validate_t1_p50_decision",
]
