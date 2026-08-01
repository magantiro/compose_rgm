"""Semantic Editing-V2 Gate 0 structural evidence.

This layer consumes the verified Action-V4 Active8 decision source and the
frozen semantic capability-cell classifier.  It proves identity and coverage
facts only.  A structural PASS does not authorize T1, P50, optimization,
checkpoint selection, final-test use, or training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_active8_admission import (
    build_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    EditingV2SemanticActive8DecisionIndex,
    resolve_editing_v2_semantic_active8_decision_source,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellError,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)

CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_gate_zero_structural_v1.json"
CONTRACT_SCHEMA = "compose.editing.semantic_gate_zero_structural_contract"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_STATUS = "FROZEN_STRUCTURAL_EVIDENCE_CONTRACT_NO_DOWNSTREAM_AUTHORITY"
FROZEN_CONTRACT_SHA256 = (
    "5875cf1ab5d8fb2d77b9e6983c7a3ac770ead8ddf69d86e287d756dd6cdcd16f"
)
EVIDENCE_SCHEMA = "compose.editing.semantic_gate_zero_structural_evidence"
EVIDENCE_SCHEMA_VERSION = 2
EVIDENCE_STATUS = "STRUCTURAL_EVIDENCE_COMPLETE_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing.semantic_gate_zero_structural_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_STATUS = "STRUCTURAL_RESULT_RECORDED_NO_DOWNSTREAM_AUTHORITY"
EVIDENCE_FILENAME = "STRUCTURAL_EVIDENCE.json"
DECISION_FILENAME = "STRUCTURAL_DECISION.json"
COMPLETION_FILENAME = "COMPLETE.json"

_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "contract_id",
    "status",
    "training_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "long_training_authorized",
    "checkpoint_selection_authorized",
    "final_test_selection_authorized",
    "parents",
    "required_architecture",
    "structural_checks",
    "decision_policy",
    "contract_sha256",
}
_PARENT_FIELDS = {"path", "file_sha256", "semantic_sha256"}
_PARENT_NAMES = {
    "decision_runtime",
    "semantic_model_process",
    "semantic_process",
    "capability_cell_registry",
}
_MODEL_RUNTIME_FIELDS = {
    "schema",
    "schema_version",
    "runtime_contract_sha256",
    "semantic_model_process_contract_sha256",
    "semantic_model_identity",
    "process_identity_sha256",
    "architecture",
    "initialization_seed",
    "initial_model_state_sha256",
    "software",
    "source_revision_sha256",
    "identity_sha256",
}
_ARCHITECTURE_FIELDS = {
    "max_atoms",
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "dtype",
    "parameter_dtypes",
    "atom_vocabulary_class_count",
    "catalog_fingerprint",
    "operator_capability_fingerprint",
}
_EVIDENCE_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_AUTHORITY,
    "structural_result",
    "contract_file_sha256",
    "contract_sha256",
    "implementation_file_sha256",
    "decision_source_identity",
    "decision_source_inventory_sha256",
    "decision_plan_file_sha256",
    "decision_plan_sha256",
    "model_runtime_identity",
    "active8_policy",
    "capability_registry",
    "active8_families",
    "checks",
    "counts",
    "decision_eligible_partition_roles",
    "sealed_nondecision_partition_roles",
    "decision_eligible_teacher_counts_by_family",
    "excluded_action_counts_by_family",
    "decision_eligible_teacher_counts_by_lane",
    "sealed_nondecision_role_inventory_sha256",
    "classification_failure_count",
    "classification_failure_receipts",
    "excluded_trace_reasons",
    "capability_cell_counts",
    "empty_capability_cell_ids",
    "accepted_trace_stream_sha256",
    "structural_assignment_inventory_sha256",
    "evidence_sha256",
}
_COUNT_FIELDS = {
    "accepted_traces",
    "excluded_traces",
    "accepted_actions",
    "excluded_actions",
    "accepted_progress_rows",
    "decision_eligible_accepted_traces",
    "decision_eligible_actions",
    "decision_eligible_progress_rows",
    "structural_assignments",
    "terminal_progress_rows",
    "terminal_assignments",
    "registered_capability_cells",
    "observed_capability_cells",
}
_CHECK_FIELDS = {
    "decision_source_identity_bound",
    "semantic_runtime_identity_bound",
    "active8_policy_identity_bound",
    "capability_registry_identity_bound",
    "decision_eligible_roles_bound",
    "sealed_nondecision_roles_not_disclosed",
    "accepted_trace_census_matches",
    "excluded_trace_census_matches",
    "one_structural_assignment_per_decision_eligible_action",
    "accepted_progress_census_matches",
    "decision_eligible_progress_census_matches",
    "terminal_progress_census_matches",
    "terminal_assignment_count_is_zero",
    "structural_assignment_identities_are_unique",
    "complete_action_census_matches",
    "complete_action_family_census_matches",
    "complete_exclusion_reason_census_matches",
    "all_active8_families_have_decision_eligible_teachers",
    "classification_failure_count_is_zero",
    "every_decision_eligible_teacher_matches_exactly_one_action_v4_mark",
    "every_decision_eligible_teacher_has_exact_candidate_support",
    "legacy_action_v2_evidence_used",
}
_CELL_ROW_FIELDS = {
    "capability_cell_id",
    "model_family",
    "family_context",
    "teacher_count",
    "unique_source_state_count",
    "unique_target_state_count",
    "unique_action_count",
    "raw_candidate_mark_sum",
    "canonical_candidate_successor_sum",
    "matching_candidate_mark_sum",
    "successor_alias_sum",
    "raw_mark_strata",
    "canonical_successor_strata",
    "successor_alias_strata",
    "lane_counts",
}

_EXPECTED_STRUCTURAL_CHECKS = {
    "trace_selection": "accepted_whole_traces_only",
    "teacher_unit": "accepted_nonterminal_action_v4_transition",
    "decision_eligible_partition_roles": ["train"],
    "sealed_nondecision_partition_roles": [
        "controller_validation",
        "final_test",
        "validation",
    ],
    "require_every_active8_family": True,
    "require_every_teacher_supported": True,
    "require_positive_raw_mark_count": True,
    "require_positive_canonical_successor_count": True,
    "require_positive_successor_alias_count": True,
    "require_exactly_one_matching_mark": True,
    "require_one_assignment_per_accepted_action": True,
    "require_zero_terminal_assignments": True,
    "capability_context_policy": (
        "classify_every_decision_eligible_teacher_report_all_registered_empty_"
        "contexts_no_context_minimum"
    ),
    "excluded_trace_policy": (
        "preserve_and_report_never_reclassify_as_teacher_coverage"
    ),
    "classification_failure_policy": (
        "publish_bounded_typed_negative_receipts_and_fail"
    ),
    "classification_failure_receipt_limit": 100,
    "legacy_action_v2_evidence": "forbidden",
}
_EXPECTED_DECISION_POLICY = {
    "pass_meaning": (
        "structural_evidence_complete_for_all_active8_families_in_train_only"
    ),
    "pass_grants_gate_zero_authority": False,
    "pass_grants_t1_authority": False,
    "pass_grants_p50_authority": False,
    "pass_grants_training_authority": False,
    "pass_grants_long_training_authority": False,
    "pass_grants_checkpoint_selection_authority": False,
    "pass_grants_final_test_selection_authority": False,
    "failure_policy": "publish_fail_closed_negative_evidence",
}


class SemanticGateZeroStructuralError(RuntimeError):
    """A structural contract, source, or evidence invariant is invalid."""


@dataclass(frozen=True, slots=True)
class FrozenSemanticGateZeroStructuralContract:
    source: Path
    payload: dict[str, Any]
    file_sha256: str

    @property
    def sha256(self) -> str:
        return str(self.payload["contract_sha256"])


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticGateZeroStructuralError(f"{field} must be a lowercase SHA-256")
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticGateZeroStructuralError(
            f"{field} is absent or invalid: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SemanticGateZeroStructuralError(f"{field} must be an object")
    return value, raw


def load_semantic_gate_zero_structural_contract(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> FrozenSemanticGateZeroStructuralContract:
    """Load the exact self-hashed contract and all physical parent bindings."""

    root = Path(repo_root or _repository_root()).resolve()
    source = Path(path) if path is not None else root / CONTRACT_RELATIVE_PATH
    payload, raw = _load_json(source, field="semantic Gate 0 structural contract")
    if set(payload) != _CONTRACT_FIELDS:
        raise SemanticGateZeroStructuralError("structural contract fields disagree")
    body = {key: item for key, item in payload.items() if key != "contract_sha256"}
    if (
        payload["schema"] != CONTRACT_SCHEMA
        or payload["schema_version"] != CONTRACT_SCHEMA_VERSION
        or payload["contract_id"] != "editing_v2_semantic_gate_zero_structural_v1"
        or payload["status"] != CONTRACT_STATUS
        or any(
            payload.get(field) is not expected for field, expected in _AUTHORITY.items()
        )
        or payload["contract_sha256"] != _sha(body)
        or payload["contract_sha256"] != FROZEN_CONTRACT_SHA256
    ):
        raise SemanticGateZeroStructuralError(
            "structural contract identity, self-hash, or authority disagrees"
        )
    parents = payload.get("parents")
    if not isinstance(parents, dict) or set(parents) != _PARENT_NAMES:
        raise SemanticGateZeroStructuralError("structural parent inventory disagrees")
    semantic_field = {
        "decision_runtime": "runtime_contract_sha256",
        "semantic_model_process": "contract_sha256",
        "semantic_process": "contract_sha256",
        "capability_cell_registry": "registry_sha256",
    }
    for name in sorted(_PARENT_NAMES):
        parent = parents[name]
        if not isinstance(parent, dict) or set(parent) != _PARENT_FIELDS:
            raise SemanticGateZeroStructuralError(f"parent {name} fields disagree")
        parent_path = (root / str(parent["path"])).resolve()
        if (
            not parent_path.is_relative_to(root)
            or _file_sha(parent_path) != parent["file_sha256"]
        ):
            raise SemanticGateZeroStructuralError(
                f"parent {name} physical bytes disagree"
            )
        parent_payload, _ = _load_json(parent_path, field=f"parent {name}")
        if parent_payload.get(semantic_field[name]) != parent["semantic_sha256"]:
            raise SemanticGateZeroStructuralError(
                f"parent {name} semantic identity disagrees"
            )
    checks = payload.get("structural_checks")
    if checks != _EXPECTED_STRUCTURAL_CHECKS:
        raise SemanticGateZeroStructuralError("structural check policy disagrees")
    decision = payload.get("decision_policy")
    if decision != _EXPECTED_DECISION_POLICY:
        raise SemanticGateZeroStructuralError("structural decision policy disagrees")
    return FrozenSemanticGateZeroStructuralContract(
        source=source.resolve(),
        payload=payload,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _load_bound_parent(
    contract: FrozenSemanticGateZeroStructuralContract,
    name: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    parent = contract.payload["parents"][name]
    payload, _ = _load_json(repo_root / parent["path"], field=f"parent {name}")
    return payload


def _load_decision_plan(
    index: EditingV2SemanticActive8DecisionIndex,
    *,
    decision_plan_path: Path,
) -> dict[str, Any]:
    plan, raw = _load_json(decision_plan_path, field="semantic Active8 decision plan")
    if (
        _canonical_bytes(plan, newline=True) != raw
        or hashlib.sha256(raw).hexdigest() != index.decision_plan_file_sha256
        or plan.get("plan_sha256") != index.decision_plan_sha256
        or plan.get("run_identity_sha256") != index.decision_run_identity_sha256
    ):
        raise SemanticGateZeroStructuralError(
            "decision plan differs from decision-source index"
        )
    return plan


def _validate_model_runtime(
    index: EditingV2SemanticActive8DecisionIndex,
    plan: Mapping[str, Any],
    contract: FrozenSemanticGateZeroStructuralContract,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    runtime = plan.get("model_runtime_identity")
    if not isinstance(runtime, dict) or set(runtime) != _MODEL_RUNTIME_FIELDS:
        raise SemanticGateZeroStructuralError("decision model runtime fields disagree")
    body = {key: item for key, item in runtime.items() if key != "identity_sha256"}
    if (
        runtime.get("schema") != "compose.data.semantic_active8_exact_model_runtime"
        or runtime.get("schema_version") != 1
        or runtime.get("identity_sha256") != _sha(body)
        or runtime.get("identity_sha256") != index.model_runtime_identity_sha256
    ):
        raise SemanticGateZeroStructuralError(
            "decision model runtime identity disagrees"
        )
    decision_runtime = _load_bound_parent(
        contract, "decision_runtime", repo_root=repo_root
    )
    semantic_model = _load_bound_parent(
        contract, "semantic_model_process", repo_root=repo_root
    )
    architecture = runtime.get("architecture")
    required = contract.payload["required_architecture"]
    expected_architecture = {
        "max_atoms": required["max_atoms"],
        "hidden_dim": required["hidden_dim"],
        "message_passing_steps": required["message_passing_steps"],
        "mark_dim": required["mark_dim"],
        "dtype": required["dtype"],
        "parameter_dtypes": [required["dtype"]],
        "atom_vocabulary_class_count": required["atom_vocabulary_class_count"],
        "catalog_fingerprint": required["catalog_fingerprint"],
        "operator_capability_fingerprint": semantic_model["model_identity"][
            "operator_capability_fingerprint"
        ],
    }
    if (
        not isinstance(architecture, dict)
        or set(architecture) != _ARCHITECTURE_FIELDS
        or architecture != expected_architecture
        or runtime["runtime_contract_sha256"]
        != decision_runtime["runtime_contract_sha256"]
        or runtime["semantic_model_process_contract_sha256"]
        != semantic_model["contract_sha256"]
        or runtime["semantic_model_identity"] != semantic_model["model_identity"]
        or runtime["process_identity_sha256"] != index.process_identity_sha256
        or runtime["initialization_seed"] != required["initialization_seed"]
        or runtime["software"] != decision_runtime["software"]
    ):
        raise SemanticGateZeroStructuralError(
            "decision runtime differs from exact semantic architecture/process"
        )
    _require_sha(
        runtime["initial_model_state_sha256"], field="initial_model_state_sha256"
    )
    _require_sha(runtime["source_revision_sha256"], field="source_revision_sha256")
    return runtime


def _empty_cell_row(cell_id: str, family: str, context: str) -> dict[str, object]:
    return {
        "capability_cell_id": cell_id,
        "model_family": family,
        "family_context": context,
        "teacher_count": 0,
        "unique_source_state_count": 0,
        "unique_target_state_count": 0,
        "unique_action_count": 0,
        "raw_candidate_mark_sum": 0,
        "canonical_candidate_successor_sum": 0,
        "matching_candidate_mark_sum": 0,
        "successor_alias_sum": 0,
        "raw_mark_strata": {},
        "canonical_successor_strata": {},
        "successor_alias_strata": {},
        "lane_counts": {},
    }


def build_semantic_gate_zero_structural_evidence(
    index: EditingV2SemanticActive8DecisionIndex,
    *,
    decision_plan_path: Path,
    contract: FrozenSemanticGateZeroStructuralContract | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Classify train-role teachers and bind the complete source in one pass."""

    root = Path(repo_root or _repository_root()).resolve()
    selected = contract or load_semantic_gate_zero_structural_contract(repo_root=root)
    plan = _load_decision_plan(index, decision_plan_path=Path(decision_plan_path))
    runtime = _validate_model_runtime(index, plan, selected, repo_root=root)
    policy = build_semantic_active8_admission_policy()
    if (
        policy.policy_sha256 != index.policy_sha256
        or policy.process_identity_sha256 != index.process_identity_sha256
    ):
        raise SemanticGateZeroStructuralError(
            "live Active8 policy differs from decision source"
        )
    registry_path = (
        root / selected.payload["parents"]["capability_cell_registry"]["path"]
    )
    registry = load_semantic_capability_cell_registry(registry_path)
    if (
        registry.registry_sha256
        != selected.payload["parents"]["capability_cell_registry"]["semantic_sha256"]
        or registry.process_identity_sha256 != index.process_identity_sha256
        or tuple(registry.contexts_by_family) != tuple(ACTIVE8_FAMILIES)
    ):
        raise SemanticGateZeroStructuralError(
            "capability registry differs from semantic source"
        )

    cell_rows: dict[str, dict[str, Any]] = {}
    cell_sets: dict[str, dict[str, set[str]]] = {}
    for family, contexts in registry.family_contexts:
        for context in contexts:
            cell_id = f"{registry.namespace}:{family}:{context}"
            cell_rows[cell_id] = _empty_cell_row(cell_id, family, context)
            cell_sets[cell_id] = {"sources": set(), "targets": set(), "actions": set()}
    decision_roles = tuple(
        selected.payload["structural_checks"]["decision_eligible_partition_roles"]
    )
    sealed_roles = tuple(
        selected.payload["structural_checks"]["sealed_nondecision_partition_roles"]
    )
    if decision_roles != ("train",) or set(decision_roles).intersection(sealed_roles):
        raise SemanticGateZeroStructuralError(
            "decision-eligible partition policy disagrees"
        )
    allowed_roles = set(decision_roles) | set(sealed_roles)

    family_counts: Counter[str] = Counter()
    global_family_counts: Counter[str] = Counter()
    decision_lane_counts: Counter[str] = Counter()
    accepted_trace_keys: set[tuple[str, int, str]] = set()
    accepted_transition_keys: set[tuple[str, int, str, int]] = set()
    accepted_assignment_sha256s: set[str] = set()
    sealed_streams = {role: hashlib.sha256() for role in sealed_roles}
    classification_failures: list[dict[str, object]] = []
    classification_failure_count = 0
    failure_limit = int(
        selected.payload["structural_checks"]["classification_failure_receipt_limit"]
    )
    accepted_trace_count = 0
    accepted_action_count = 0
    accepted_progress_count = 0
    decision_trace_count = 0
    decision_action_count = 0
    decision_progress_count = 0
    terminal_progress_count = 0
    terminal_assignment_count = 0
    excluded_trace_count = 0
    excluded_action_count = 0
    excluded_family_counts: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    accepted_trace_stream = hashlib.sha256()
    accepted_assignment_stream = hashlib.sha256()

    for accepted, resolved in index.iter_resolved_traces():
        address = resolved.addressed_trace.address
        role = address.partition
        if role not in allowed_roles:
            raise SemanticGateZeroStructuralError(
                f"decision trace uses undeclared partition role: {role}"
            )
        trace_address_payload = {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "packed_shard_name": address.packed_shard_name,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
            "layer": address.layer,
            "partition": address.partition,
            "source_key": address.source_key,
            "target_key": address.target_key,
            "path_length": address.path_length,
        }
        trace_address_sha256 = _sha(trace_address_payload)
        if not accepted:
            excluded_trace_count += 1
            excluded_action_count += len(resolved.action_decisions)
            excluded_family_counts.update(
                item.classification.model_family
                for item in resolved.action_decisions
                if item.classification.model_family is not None
            )
            exclusion_reasons.update(item.reason for item in resolved.exclusions)
            continue

        trace_key = (
            address.packed_shard_content_sha256,
            address.entry_index,
            address.trace_id,
        )
        if trace_key in accepted_trace_keys:
            raise SemanticGateZeroStructuralError(
                "accepted trace stream contains a duplicate"
            )
        accepted_trace_keys.add(trace_key)
        accepted_trace_count += 1
        accepted_action_count += len(resolved.action_decisions)
        accepted_progress_count += len(resolved.progress_addresses)
        terminal_progress_count += sum(
            item.terminal for item in resolved.progress_addresses
        )
        global_family_counts.update(
            item.classification.model_family
            for item in resolved.action_decisions
            if item.classification.model_family is not None
        )
        accepted_trace_stream.update(
            _canonical_bytes(
                {
                    "trace_address_sha256": trace_address_sha256,
                    "decision_sha256": resolved.decision_sha256,
                },
                newline=True,
            )
        )
        if role not in decision_roles:
            sealed_streams[role].update(
                _canonical_bytes(
                    {
                        "trace_address_sha256": trace_address_sha256,
                        "decision_sha256": resolved.decision_sha256,
                    },
                    newline=True,
                )
            )
            continue

        decision_trace_count += 1
        decision_action_count += len(resolved.action_decisions)
        decision_progress_count += len(resolved.progress_addresses)
        for transition in index.accepted_transitions_for(resolved):
            if transition.source_progress_address.terminal:
                terminal_assignment_count += 1
            try:
                assignment = classify_verified_structural_transition(
                    index,
                    transition,
                    registry=registry,
                )
            except SemanticCapabilityCellError as error:
                classification_failure_count += 1
                if len(classification_failures) < failure_limit:
                    classification_failures.append(
                        {
                            "trace_address_sha256": trace_address_sha256,
                            "decision_sha256": resolved.decision_sha256,
                            "progress_index": transition.step_index,
                            "model_family": transition.action_decision.classification.model_family,
                            "failure_type": type(error).__name__,
                            "reason": str(error),
                        }
                    )
                continue
            transition_key = (*trace_key, assignment.progress_index)
            if (
                transition_key in accepted_transition_keys
                or assignment.assignment_sha256 in accepted_assignment_sha256s
            ):
                raise SemanticGateZeroStructuralError(
                    "verified structural assignment stream contains a duplicate"
                )
            accepted_transition_keys.add(transition_key)
            accepted_assignment_sha256s.add(assignment.assignment_sha256)
            accepted_assignment_stream.update(
                _canonical_bytes(assignment.as_payload(), newline=True)
            )
            cell_id = assignment.capability_cell_id
            row = cell_rows.get(cell_id)
            if row is None:
                raise SemanticGateZeroStructuralError(
                    "teacher classified outside the frozen capability registry"
                )
            sets = cell_sets[cell_id]
            row["teacher_count"] += 1
            row["raw_candidate_mark_sum"] += assignment.raw_mark_count
            row[
                "canonical_candidate_successor_sum"
            ] += assignment.canonical_successor_count
            row["matching_candidate_mark_sum"] += assignment.matching_mark_count
            row["successor_alias_sum"] += assignment.successor_alias_multiplicity
            for field, stratum in (
                ("raw_mark_strata", assignment.raw_mark_count_stratum),
                (
                    "canonical_successor_strata",
                    assignment.canonical_successor_count_stratum,
                ),
                (
                    "successor_alias_strata",
                    assignment.successor_alias_multiplicity_stratum,
                ),
            ):
                row[field][stratum] = row[field].get(stratum, 0) + 1
            lane = assignment.data_lane
            row["lane_counts"][lane] = row["lane_counts"].get(lane, 0) + 1
            sets["sources"].add(assignment.source_state_sha256)
            sets["targets"].add(assignment.target_state_sha256)
            sets["actions"].add(assignment.action_sha256)
            family_counts[assignment.model_family] += 1
            decision_lane_counts[lane] += 1

    for cell_id, row in cell_rows.items():
        sets = cell_sets[cell_id]
        row["unique_source_state_count"] = len(sets["sources"])
        row["unique_target_state_count"] = len(sets["targets"])
        row["unique_action_count"] = len(sets["actions"])
        for field in (
            "raw_mark_strata",
            "canonical_successor_strata",
            "successor_alias_strata",
            "lane_counts",
        ):
            row[field] = dict(sorted(row[field].items()))
    count_map = index.count_map
    total_family_counts = Counter(global_family_counts)
    total_family_counts.update(excluded_family_counts)
    checks = {
        "decision_source_identity_bound": True,
        "semantic_runtime_identity_bound": True,
        "active8_policy_identity_bound": True,
        "capability_registry_identity_bound": True,
        "decision_eligible_roles_bound": decision_roles == ("train",),
        "sealed_nondecision_roles_not_disclosed": all(
            role not in decision_roles for role in sealed_roles
        ),
        "accepted_trace_census_matches": accepted_trace_count
        == count_map["accepted_traces"],
        "excluded_trace_census_matches": excluded_trace_count
        == count_map["excluded_traces"],
        "one_structural_assignment_per_decision_eligible_action": (
            len(accepted_assignment_sha256s) + classification_failure_count
            == decision_action_count
        ),
        "accepted_progress_census_matches": accepted_progress_count
        == count_map["progress_rows"]
        == accepted_action_count + accepted_trace_count,
        "decision_eligible_progress_census_matches": decision_progress_count
        == decision_action_count + decision_trace_count,
        "terminal_progress_census_matches": terminal_progress_count
        == accepted_trace_count,
        "terminal_assignment_count_is_zero": terminal_assignment_count == 0,
        "structural_assignment_identities_are_unique": (
            len(accepted_assignment_sha256s)
            == len(accepted_transition_keys)
            == decision_action_count - classification_failure_count
        ),
        "complete_action_census_matches": accepted_action_count + excluded_action_count
        == count_map["actions"],
        "complete_action_family_census_matches": dict(
            sorted(total_family_counts.items())
        )
        == dict(index.action_family_histogram),
        "complete_exclusion_reason_census_matches": dict(
            sorted(exclusion_reasons.items())
        )
        == dict(index.active8_exclusion_reason_histogram),
        "all_active8_families_have_decision_eligible_teachers": all(
            family_counts[family] > 0 for family in ACTIVE8_FAMILIES
        ),
        "classification_failure_count_is_zero": classification_failure_count == 0,
        "every_decision_eligible_teacher_matches_exactly_one_action_v4_mark": all(
            row["matching_candidate_mark_sum"] == row["teacher_count"]
            for row in cell_rows.values()
        ),
        "every_decision_eligible_teacher_has_exact_candidate_support": True,
        "legacy_action_v2_evidence_used": False,
    }
    structural_result = (
        "PASS"
        if all(
            value is True
            for key, value in checks.items()
            if key != "legacy_action_v2_evidence_used"
        )
        and checks["legacy_action_v2_evidence_used"] is False
        else "FAIL"
    )
    identity_payload = index.identity_payload()
    evidence_body: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": EVIDENCE_STATUS,
        **_AUTHORITY,
        "structural_result": structural_result,
        "contract_file_sha256": selected.file_sha256,
        "contract_sha256": selected.sha256,
        "implementation_file_sha256": _file_sha(Path(__file__)),
        "decision_source_identity": identity_payload,
        "decision_source_inventory_sha256": index.inventory_sha256,
        "decision_plan_file_sha256": index.decision_plan_file_sha256,
        "decision_plan_sha256": index.decision_plan_sha256,
        "model_runtime_identity": runtime,
        "active8_policy": policy.as_payload(),
        "capability_registry": {
            "registry_id": registry.registry_id,
            "namespace": registry.namespace,
            "registry_sha256": registry.registry_sha256,
            "process_identity_sha256": registry.process_identity_sha256,
            "corpus_contract_file_sha256": registry.corpus_contract_file_sha256,
            "classifier_implementation_sha256": registry.classifier_implementation_sha256,
            "family_contexts": {
                family: list(contexts) for family, contexts in registry.family_contexts
            },
        },
        "active8_families": list(ACTIVE8_FAMILIES),
        "checks": checks,
        "counts": {
            "accepted_traces": accepted_trace_count,
            "excluded_traces": excluded_trace_count,
            "accepted_actions": accepted_action_count,
            "excluded_actions": excluded_action_count,
            "accepted_progress_rows": accepted_progress_count,
            "decision_eligible_accepted_traces": decision_trace_count,
            "decision_eligible_actions": decision_action_count,
            "decision_eligible_progress_rows": decision_progress_count,
            "structural_assignments": len(accepted_assignment_sha256s),
            "terminal_progress_rows": terminal_progress_count,
            "terminal_assignments": terminal_assignment_count,
            "registered_capability_cells": len(cell_rows),
            "observed_capability_cells": sum(
                row["teacher_count"] > 0 for row in cell_rows.values()
            ),
        },
        "decision_eligible_partition_roles": list(decision_roles),
        "sealed_nondecision_partition_roles": list(sealed_roles),
        "decision_eligible_teacher_counts_by_family": {
            family: family_counts[family] for family in ACTIVE8_FAMILIES
        },
        "excluded_action_counts_by_family": {
            family: excluded_family_counts[family] for family in ACTIVE8_FAMILIES
        },
        "decision_eligible_teacher_counts_by_lane": dict(
            sorted(decision_lane_counts.items())
        ),
        "sealed_nondecision_role_inventory_sha256": {
            role: sealed_streams[role].hexdigest() for role in sealed_roles
        },
        "classification_failure_count": classification_failure_count,
        "classification_failure_receipts": classification_failures,
        "excluded_trace_reasons": dict(sorted(exclusion_reasons.items())),
        "capability_cell_counts": [cell_rows[key] for key in sorted(cell_rows)],
        "empty_capability_cell_ids": [
            key for key in sorted(cell_rows) if cell_rows[key]["teacher_count"] == 0
        ],
        "accepted_trace_stream_sha256": accepted_trace_stream.hexdigest(),
        "structural_assignment_inventory_sha256": (
            accepted_assignment_stream.hexdigest()
        ),
    }
    return {**evidence_body, "evidence_sha256": _sha(evidence_body)}


def _count_mapping(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key or type(count) is not int or count < 0
        for key, count in value.items()
    ):
        raise SemanticGateZeroStructuralError(
            f"structural evidence {field} is not a nonnegative count map"
        )
    return dict(value)


def _validate_structural_evidence(
    evidence: Mapping[str, Any],
    *,
    index: EditingV2SemanticActive8DecisionIndex,
    contract: FrozenSemanticGateZeroStructuralContract,
    decision_plan_path: Path,
    repo_root: Path,
) -> None:
    body = dict(evidence)
    supplied = body.pop("evidence_sha256", None)
    if (
        set(evidence) != _EVIDENCE_FIELDS
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != EVIDENCE_STATUS
        or supplied != _sha(body)
        or any(
            evidence.get(field) is not expected
            for field, expected in _AUTHORITY.items()
        )
        or not isinstance(evidence.get("counts"), Mapping)
        or set(evidence["counts"]) != _COUNT_FIELDS
        or not isinstance(evidence.get("checks"), Mapping)
        or set(evidence["checks"]) != _CHECK_FIELDS
        or evidence.get("active8_families") != list(ACTIVE8_FAMILIES)
    ):
        raise SemanticGateZeroStructuralError("structural evidence is malformed")
    _count_mapping(evidence["counts"], field="counts")
    expected = build_semantic_gate_zero_structural_evidence(
        index,
        decision_plan_path=decision_plan_path,
        contract=contract,
        repo_root=repo_root,
    )
    if dict(evidence) != expected:
        raise SemanticGateZeroStructuralError(
            "structural evidence differs from the recomputed exact source snapshot"
        )


def structural_decision_from_evidence(
    evidence: Mapping[str, Any],
    *,
    index: EditingV2SemanticActive8DecisionIndex,
    contract: FrozenSemanticGateZeroStructuralContract,
    decision_plan_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute PASS or FAIL without converting structural evidence to authority."""

    root = Path(repo_root or _repository_root()).resolve()
    _validate_structural_evidence(
        evidence,
        index=index,
        contract=contract,
        decision_plan_path=Path(decision_plan_path),
        repo_root=root,
    )
    return _structural_decision_from_verified_evidence(evidence)


def _structural_decision_from_verified_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the non-authorizing decision for a freshly constructed snapshot."""

    supplied = evidence["evidence_sha256"]
    decision_body: dict[str, object] = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_STATUS,
        **_AUTHORITY,
        "structural_result": evidence["structural_result"],
        "evidence_sha256": supplied,
        "contract_sha256": evidence["contract_sha256"],
        "decision_source_inventory_sha256": evidence[
            "decision_source_inventory_sha256"
        ],
        "next_authorized_stage": None,
        "required_next_action": (
            "record_separate_human_or_registered_gate_zero_authorization"
            if evidence["structural_result"] == "PASS"
            else "repair_missing_structural_coverage_and_rerun_gate_zero"
        ),
    }
    return {**decision_body, "decision_sha256": _sha(decision_body)}


def _publish(path: Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    if target.exists():
        if _file_sha(target) != digest or target.read_bytes() != content:
            raise SemanticGateZeroStructuralError(
                f"immutable structural-evidence collision at {target}"
            )
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_semantic_gate_zero_structural_artifacts(
    *,
    output_directory: Path,
    index: EditingV2SemanticActive8DecisionIndex,
    decision_plan_path: Path,
    contract: FrozenSemanticGateZeroStructuralContract,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fully revalidate persisted Gate 0 bytes against the exact live source."""

    root = Path(repo_root or _repository_root()).resolve()
    output = Path(output_directory).resolve()
    evidence, evidence_bytes = _load_json(
        output / EVIDENCE_FILENAME,
        field="semantic Gate 0 structural evidence",
    )
    decision, decision_bytes = _load_json(
        output / DECISION_FILENAME,
        field="semantic Gate 0 structural decision",
    )
    completion, completion_bytes = _load_json(
        output / COMPLETION_FILENAME,
        field="semantic Gate 0 structural completion",
    )
    if any(
        raw != _canonical_bytes(payload, newline=True)
        for payload, raw in (
            (evidence, evidence_bytes),
            (decision, decision_bytes),
            (completion, completion_bytes),
        )
    ):
        raise SemanticGateZeroStructuralError(
            "persisted structural artifacts are not canonical JSON"
        )
    completion_body = dict(completion)
    supplied_completion_sha256 = completion_body.pop("completion_sha256", None)
    expected_completion_fields = {
        "schema",
        "schema_version",
        "status",
        *_AUTHORITY,
        "structural_result",
        "contract_sha256",
        "evidence_file_sha256",
        "evidence_sha256",
        "decision_file_sha256",
        "decision_sha256",
        "completion_sha256",
    }
    if (
        set(completion) != expected_completion_fields
        or completion.get("schema")
        != "compose.editing.semantic_gate_zero_structural_completion"
        or completion.get("schema_version") != 1
        or completion.get("status")
        != "COMPLETE_STRUCTURAL_ARTIFACTS_NO_DOWNSTREAM_AUTHORITY"
        or any(
            completion.get(field) is not expected
            for field, expected in _AUTHORITY.items()
        )
        or supplied_completion_sha256 != _sha(completion_body)
        or completion.get("contract_sha256") != contract.sha256
        or completion.get("evidence_file_sha256")
        != hashlib.sha256(evidence_bytes).hexdigest()
        or completion.get("decision_file_sha256")
        != hashlib.sha256(decision_bytes).hexdigest()
    ):
        raise SemanticGateZeroStructuralError(
            "persisted structural completion identity disagrees"
        )
    _validate_structural_evidence(
        evidence,
        index=index,
        contract=contract,
        decision_plan_path=Path(decision_plan_path),
        repo_root=root,
    )
    expected_decision = _structural_decision_from_verified_evidence(evidence)
    if (
        decision != expected_decision
        or completion.get("structural_result") != evidence["structural_result"]
        or completion.get("evidence_sha256") != evidence["evidence_sha256"]
        or completion.get("decision_sha256") != decision["decision_sha256"]
    ):
        raise SemanticGateZeroStructuralError(
            "persisted structural decision differs from exact evidence"
        )
    return {
        "evidence": evidence,
        "decision": decision,
        "completion": completion,
    }


def run_semantic_gate_zero_structural_evidence(
    *,
    migration_completion_path: Path,
    chunk_cache_plan_path: Path,
    chunk_cache_global_completion_path: Path,
    decision_plan_path: Path,
    decision_completion_path: Path,
    artifact_root: Path,
    repo_root: Path,
    output_directory: Path,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve exact semantic upstreams and publish evidence plus non-authority decision."""

    index = resolve_editing_v2_semantic_active8_decision_source(
        migration_completion_path=migration_completion_path,
        chunk_cache_plan_path=chunk_cache_plan_path,
        chunk_cache_global_completion_path=chunk_cache_global_completion_path,
        decision_plan_path=decision_plan_path,
        decision_completion_path=decision_completion_path,
        artifact_root=artifact_root,
        repo_root=repo_root,
    )
    contract = load_semantic_gate_zero_structural_contract(
        contract_path, repo_root=repo_root
    )
    evidence = build_semantic_gate_zero_structural_evidence(
        index,
        decision_plan_path=decision_plan_path,
        contract=contract,
        repo_root=repo_root,
    )
    decision = _structural_decision_from_verified_evidence(evidence)
    output = Path(output_directory).resolve()
    artifact = Path(artifact_root).resolve()
    if not output.is_relative_to(artifact):
        raise SemanticGateZeroStructuralError(
            "output_directory lies outside artifact_root"
        )
    evidence_bytes = _canonical_bytes(evidence, newline=True)
    decision_bytes = _canonical_bytes(decision, newline=True)
    _publish(output / EVIDENCE_FILENAME, evidence_bytes)
    _publish(output / DECISION_FILENAME, decision_bytes)
    completion_body: dict[str, object] = {
        "schema": "compose.editing.semantic_gate_zero_structural_completion",
        "schema_version": 1,
        "status": "COMPLETE_STRUCTURAL_ARTIFACTS_NO_DOWNSTREAM_AUTHORITY",
        **_AUTHORITY,
        "structural_result": evidence["structural_result"],
        "contract_sha256": contract.sha256,
        "evidence_file_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "evidence_sha256": evidence["evidence_sha256"],
        "decision_file_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "decision_sha256": decision["decision_sha256"],
    }
    completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    _publish(output / COMPLETION_FILENAME, _canonical_bytes(completion, newline=True))
    return {"evidence": evidence, "decision": decision, "completion": completion}


__all__ = [
    "COMPLETION_FILENAME",
    "CONTRACT_RELATIVE_PATH",
    "DECISION_FILENAME",
    "EVIDENCE_FILENAME",
    "FrozenSemanticGateZeroStructuralContract",
    "SemanticGateZeroStructuralError",
    "build_semantic_gate_zero_structural_evidence",
    "load_semantic_gate_zero_structural_artifacts",
    "load_semantic_gate_zero_structural_contract",
    "run_semantic_gate_zero_structural_evidence",
    "structural_decision_from_evidence",
]
