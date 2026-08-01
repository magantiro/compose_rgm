"""Immutable exact-state and full-successor inputs for semantic T1.

The existing semantic successor cache deliberately stores only the teacher
fiber and virtual coordinates.  T1 additionally needs exact slot-addressed
source states and the complete productive canonical-successor partition to
measure rank/top-1 without replaying chemistry on the GPU.  This module owns a
separate, nonauthorizing prerequisite artifact for precisely that information.
It never stores a model score, probability, hazard, loss, or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheRecord,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.editing_v2_semantic_t1_panel_cache import (
    SemanticT1PanelArtifact,
)
from compose_v4.experiments.editing_v2_semantic_t1_successor_cache import (
    SemanticT1SuccessorCache,
)
from compose_v4.experiments.factorized_successor_training import (
    CanonicalSuccessorAliasGroup,
    CompiledStateSuccessorMap,
    CompiledSuccessorMark,
    StateProductiveSupport,
    TeacherSuccessorAlias,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "compose.editing_v2.semantic_t1_prepared_inputs"
SCHEMA_VERSION = 1
STATUS = "PREPARED_EXACT_STATE_FULL_SUCCESSOR_SUPPORT_NO_T1_AUTHORITY"
FILENAME = "SEMANTIC_T1_PREPARED_INPUTS.json"
MAX_BYTES = 2 << 30
NO_AUTHORITY = {
    "training_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
SOURCE_STATE_RECEIPT_SCHEMA = "compose.editing_v2.semantic_t1_exact_source_state_receipt"
SOURCE_STATE_RECEIPT_SCHEMA_VERSION = 1
SOURCE_STATE_RECEIPT_STATUS = "EXACT_SLOT_STATES_REOPENED_NO_T1_AUTHORITY"
_ROOT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_AUTHORITY,
    "source_revision",
    "cache_source_revision_sha256",
    "implementation_sha256",
    "capacity_policy_sha256",
    "capacity_policy_file_sha256",
    "panel_artifact_sha256",
    "panel_entry_inventory_sha256",
    "cache_completion_sha256",
    "cache_manifest_sha256",
    "decision_source_inventory_sha256",
    "initial_model_state_sha256",
    "semantic_model_process_contract",
    "source_state_receipt",
    "state_encoding",
    "tensor_contract",
    "successor_partition_contract",
    "entry_count",
    "entry_inventory_sha256",
    "entries",
    "artifact_sha256",
}
_ENTRY_FIELDS = {
    "panel_entry_sha256",
    "model_family",
    "capability_cell_id",
    "support_time_hex",
    "source_state_sha256",
    "successor_canonical_key",
    "objective_coefficient",
    "raw_mark_count",
    "canonical_successor_count",
    "production_successor_alias_multiplicity",
    "cache_record_sha256",
    "exact_state",
    "successor_partition",
    "productive_alias_count",
    "virtual_alias_count",
    "model_scores_or_probabilities_stored",
    "hazard_included",
    "entry_sha256",
}
_TENSOR_CONTRACT = {
    "model_dtype": "torch.float32",
    "mixed_precision": False,
    "collation": "production_factorized_mark_batch_semantic_active8",
}
_PARTITION_CONTRACT = {
    "productive_groups_complete": True,
    "productive_aliases_disjoint": True,
    "virtual_aliases_disjoint": True,
    "self_events_excluded_from_embedded_jump_chain": True,
    "model_scores_or_probabilities_stored": False,
    "hazard_included": False,
}
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_t1_prepared_inputs.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_panel_cache.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_successor_cache.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/rewrite/trace_shard.py",
)


class SemanticT1PreparedInputError(RuntimeError):
    """Prepared T1 input bytes are missing, stale, or scientifically invalid."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticT1PreparedInputError(
            "semantic T1 prepared inputs are not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, *, field: str) -> str:
    if not _is_sha(value):
        raise SemanticT1PreparedInputError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _source_revision_is_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    body = dict(value)
    supplied = body.pop("source_revision_sha256", None)
    return bool(
        body
        and _is_sha(supplied)
        and supplied == _sha(body)
        and value.get("worktree_clean") is True
    )


def semantic_t1_prepared_input_implementation_sha256(*, repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticT1PreparedInputError(
                f"prepared-input implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _alias_payload(alias: TeacherSuccessorAlias) -> dict[str, object]:
    return {
        "family_name": alias.family_name,
        "table_name": alias.table_name,
        "coordinate": list(alias.coordinate),
    }


def _alias_from_payload(value: object) -> TeacherSuccessorAlias:
    if not isinstance(value, Mapping) or set(value) != {
        "family_name",
        "table_name",
        "coordinate",
    }:
        raise SemanticT1PreparedInputError("prepared successor alias is malformed")
    coordinate = value["coordinate"]
    family_name = value["family_name"]
    table_name = value["table_name"]
    if (
        not isinstance(family_name, str)
        or not family_name
        or not isinstance(table_name, str)
        or not table_name
        or not isinstance(coordinate, list)
        or not coordinate
        or any(type(item) is not int or item < 0 for item in coordinate)
    ):
        raise SemanticT1PreparedInputError("prepared successor coordinate is malformed")
    return TeacherSuccessorAlias(
        family_name=family_name,
        table_name=table_name,
        coordinate=tuple(coordinate),
    )


def _mark_payload(mark: CompiledSuccessorMark) -> dict[str, object]:
    return {
        "alias": _alias_payload(mark.alias),
        "successor_state_sha256": mark.successor_state_sha256,
        "action_sha256": mark.action_sha256,
    }


def _mark_from_payload(value: object) -> CompiledSuccessorMark:
    if not isinstance(value, Mapping) or set(value) != {
        "alias",
        "successor_state_sha256",
        "action_sha256",
    }:
        raise SemanticT1PreparedInputError("prepared successor mark is malformed")
    return CompiledSuccessorMark(
        alias=_alias_from_payload(value["alias"]),
        successor_state_sha256=_require_sha(
            value["successor_state_sha256"], field="successor_state_sha256"
        ),
        action_sha256=_require_sha(value["action_sha256"], field="action_sha256"),
    )


def compiled_successor_map_payload(
    value: CompiledStateSuccessorMap,
) -> dict[str, object]:
    """Encode one complete score-independent coordinate partition."""

    return {
        "source_key": value.source_key,
        "source_state_sha256": value.source_state_sha256,
        "virtual_marks": [_mark_payload(mark) for mark in value.virtual_marks],
        "successor_groups": [
            {
                "target_key": group.target_key,
                "marks": [_mark_payload(mark) for mark in group.marks],
            }
            for group in value.successor_groups
        ],
    }


def compiled_successor_map_from_payload(
    value: object,
) -> CompiledStateSuccessorMap:
    """Decode and revalidate one complete coordinate partition."""

    if not isinstance(value, Mapping) or set(value) != {
        "source_key",
        "source_state_sha256",
        "virtual_marks",
        "successor_groups",
    }:
        raise SemanticT1PreparedInputError("prepared successor partition is malformed")
    virtual_payloads = value["virtual_marks"]
    group_payloads = value["successor_groups"]
    if not isinstance(virtual_payloads, list) or not isinstance(group_payloads, list):
        raise SemanticT1PreparedInputError("prepared successor rows must be lists")
    virtual_marks = tuple(_mark_from_payload(item) for item in virtual_payloads)
    groups: list[CanonicalSuccessorAliasGroup] = []
    for group in group_payloads:
        if not isinstance(group, Mapping) or set(group) != {"target_key", "marks"}:
            raise SemanticT1PreparedInputError("prepared successor group is malformed")
        marks = group["marks"]
        if not isinstance(marks, list):
            raise SemanticT1PreparedInputError("prepared successor marks must be a list")
        target_key = group["target_key"]
        if not isinstance(target_key, str) or not target_key:
            raise SemanticT1PreparedInputError("prepared successor target key is malformed")
        groups.append(
            CanonicalSuccessorAliasGroup(
                target_key=target_key,
                marks=tuple(_mark_from_payload(item) for item in marks),
            )
        )
    source_key = value["source_key"]
    if not isinstance(source_key, str) or not source_key:
        raise SemanticT1PreparedInputError("prepared successor source key is malformed")
    source_state_sha256 = _require_sha(value["source_state_sha256"], field="source_state_sha256")
    support = StateProductiveSupport(
        source_key=source_key,
        source_state_sha256=source_state_sha256,
        virtual_aliases=tuple(sorted(mark.alias for mark in virtual_marks)),
    )
    return CompiledStateSuccessorMap(
        state_support=support,
        successor_groups=tuple(groups),
        virtual_marks=virtual_marks,
    )


def _entry_payload(
    *,
    panel_entry: Any,
    cache_record: SuccessorFiberCacheRecord,
    source_state: Any,
    partition: CompiledStateSuccessorMap,
) -> dict[str, object]:
    state_payload = encode_state(source_state)
    decoded = decode_state(state_payload)
    if encode_state(decoded) != state_payload:
        raise SemanticT1PreparedInputError("exact state codec is not byte-stable")
    source_sha256 = persistent_slot_state_sha256(decoded)
    teacher = cache_record.teacher_fiber
    if teacher is None:
        raise SemanticT1PreparedInputError("selected T1 cache row is terminal")
    teacher_group = next(
        (group for group in partition.successor_groups if group.target_key == teacher.target_key),
        None,
    )
    if (
        source_sha256 != panel_entry.source_state_sha256
        or source_sha256 != cache_record.source_state_sha256
        or partition.state_support != cache_record.state_support
        or teacher.target_key != panel_entry.successor_canonical_key
        or teacher.target_state_sha256 != panel_entry.representative_target_state_sha256
        or teacher_group is None
        or teacher_group.aliases != tuple(sorted(teacher.aliases))
        or teacher.target_state_sha256 not in teacher_group.successor_state_sha256s
    ):
        raise SemanticT1PreparedInputError(
            "prepared state, panel target, cache teacher, or full partition disagrees"
        )
    record_payload = successor_fiber_cache_record_payload(cache_record)
    partition_payload = compiled_successor_map_payload(partition)
    body = {
        "panel_entry_sha256": panel_entry.panel_entry_sha256,
        "model_family": panel_entry.model_family,
        "capability_cell_id": panel_entry.capability_cell_id,
        "support_time_hex": panel_entry.support_time_hex,
        "source_state_sha256": source_sha256,
        "successor_canonical_key": panel_entry.successor_canonical_key,
        "objective_coefficient": panel_entry.objective_coefficient,
        "raw_mark_count": panel_entry.raw_mark_count,
        "canonical_successor_count": panel_entry.canonical_successor_count,
        "production_successor_alias_multiplicity": (
            panel_entry.production_successor_alias_multiplicity
        ),
        "cache_record_sha256": _sha(record_payload),
        "exact_state": state_payload,
        "successor_partition": partition_payload,
        "productive_alias_count": sum(len(group.marks) for group in partition.successor_groups),
        "virtual_alias_count": len(partition.virtual_marks),
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    if body["canonical_successor_count"] != len(partition.successor_groups):
        raise SemanticT1PreparedInputError(
            "prepared full successor count differs from the frozen panel audit"
        )
    if body["raw_mark_count"] != (body["productive_alias_count"] + body["virtual_alias_count"]):
        raise SemanticT1PreparedInputError(
            "prepared full mark census differs from the frozen panel audit"
        )
    if body["production_successor_alias_multiplicity"] != len(teacher_group.marks):
        raise SemanticT1PreparedInputError(
            "prepared teacher alias multiplicity differs from the frozen panel audit"
        )
    return {**body, "entry_sha256": _sha(body)}


def _source_state_receipt(
    *,
    panel: SemanticT1PanelArtifact,
    cache: SemanticT1SuccessorCache,
    source_states_by_panel_entry_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    bindings: list[dict[str, str]] = []
    for panel_entry in panel.entries:
        panel_id = panel_entry.panel_entry_sha256
        record = cache.record_for_panel_entry_sha256(panel_id)
        bindings.append(
            {
                "panel_entry_sha256": panel_id,
                "source_cache_record_sha256": _sha(successor_fiber_cache_record_payload(record)),
                "source_state_sha256": persistent_slot_state_sha256(
                    source_states_by_panel_entry_sha256[panel_id]
                ),
            }
        )
    bindings.sort(key=lambda item: item["panel_entry_sha256"])
    body = {
        "schema": SOURCE_STATE_RECEIPT_SCHEMA,
        "schema_version": SOURCE_STATE_RECEIPT_SCHEMA_VERSION,
        "status": SOURCE_STATE_RECEIPT_STATUS,
        **NO_AUTHORITY,
        "panel_artifact_sha256": panel.artifact_sha256,
        "cache_completion_sha256": cache.completion["completion_sha256"],
        "entry_count": len(bindings),
        "entry_state_binding_sha256": _sha(bindings),
        "exact_persistent_slot_states_reopened": True,
        "canonical_smiles_reconstruction_used": False,
    }
    return {**body, "receipt_sha256": _sha(body)}


def _validate_source_state_receipt(
    value: object,
    *,
    panel_artifact_sha256: str,
    cache_completion_sha256: str,
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "panel_artifact_sha256",
        "cache_completion_sha256",
        "entry_count",
        "entry_state_binding_sha256",
        "exact_persistent_slot_states_reopened",
        "canonical_smiles_reconstruction_used",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticT1PreparedInputError("exact source-state receipt is malformed")
    receipt = dict(value)
    body = dict(receipt)
    supplied_sha256 = body.pop("receipt_sha256")
    bindings = sorted(
        (
            {
                "panel_entry_sha256": entry["panel_entry_sha256"],
                "source_cache_record_sha256": entry["cache_record_sha256"],
                "source_state_sha256": entry["source_state_sha256"],
            }
            for entry in entries
        ),
        key=lambda item: item["panel_entry_sha256"],
    )
    if (
        receipt.get("schema") != SOURCE_STATE_RECEIPT_SCHEMA
        or receipt.get("schema_version") != SOURCE_STATE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != SOURCE_STATE_RECEIPT_STATUS
        or supplied_sha256 != _sha(body)
        or receipt.get("panel_artifact_sha256") != panel_artifact_sha256
        or receipt.get("cache_completion_sha256") != cache_completion_sha256
        or receipt.get("entry_count") != len(bindings)
        or receipt.get("entry_state_binding_sha256") != _sha(bindings)
        or receipt.get("exact_persistent_slot_states_reopened") is not True
        or receipt.get("canonical_smiles_reconstruction_used") is not False
        or any(receipt.get(name) is not expected for name, expected in NO_AUTHORITY.items())
    ):
        raise SemanticT1PreparedInputError("exact source-state receipt disagrees")
    return receipt


def build_semantic_t1_prepared_inputs(
    *,
    panel: SemanticT1PanelArtifact,
    cache: SemanticT1SuccessorCache,
    source_states_by_panel_entry_sha256: Mapping[str, Any],
    successor_partitions_by_panel_entry_sha256: Mapping[str, CompiledStateSuccessorMap],
    capacity_policy: Mapping[str, Any],
    capacity_policy_file_sha256: str,
    source_revision: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Seal exact states and exhaustive coordinate partitions for one panel."""

    panel_ids = tuple(entry.panel_entry_sha256 for entry in panel.entries)
    if (
        set(source_states_by_panel_entry_sha256) != set(panel_ids)
        or set(successor_partitions_by_panel_entry_sha256) != set(panel_ids)
        or cache.manifest.get("panel_binding", {}).get("panel_artifact_sha256")
        != panel.artifact_sha256
        or capacity_policy.get("hazard_included") is not False
        or capacity_policy.get("training_authorized") is not False
    ):
        raise SemanticT1PreparedInputError(
            "prepared-input panel, cache, or prospective policy identity disagrees"
        )
    _require_sha(capacity_policy.get("policy_sha256"), field="policy_sha256")
    _require_sha(capacity_policy_file_sha256, field="capacity_policy_file_sha256")
    if not _source_revision_is_valid(source_revision):
        raise SemanticT1PreparedInputError("source revision identity is absent")
    cache_source_revision = cache.manifest.get("source_revision")
    if not isinstance(cache_source_revision, Mapping) or not _is_sha(
        cache_source_revision.get("source_revision_sha256")
    ):
        raise SemanticT1PreparedInputError("cache compiler source revision is absent")

    entries = [
        _entry_payload(
            panel_entry=entry,
            cache_record=cache.record_for_panel_entry_sha256(entry.panel_entry_sha256),
            source_state=source_states_by_panel_entry_sha256[entry.panel_entry_sha256],
            partition=successor_partitions_by_panel_entry_sha256[entry.panel_entry_sha256],
        )
        for entry in panel.entries
    ]
    entries.sort(
        key=lambda item: (
            item["model_family"],
            item["capability_cell_id"],
            item["panel_entry_sha256"],
        )
    )
    body = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        **NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "cache_source_revision_sha256": cache_source_revision["source_revision_sha256"],
        "implementation_sha256": (
            semantic_t1_prepared_input_implementation_sha256(repo_root=repo_root)
        ),
        "capacity_policy_sha256": capacity_policy["policy_sha256"],
        "capacity_policy_file_sha256": capacity_policy_file_sha256,
        "panel_artifact_sha256": panel.artifact_sha256,
        "panel_entry_inventory_sha256": _sha(sorted(panel_ids)),
        "cache_completion_sha256": cache.completion["completion_sha256"],
        "cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "decision_source_inventory_sha256": cache.completion["decision_source_inventory_sha256"],
        "initial_model_state_sha256": cache.completion["initial_model_state_sha256"],
        "semantic_model_process_contract": cache.manifest["semantic_model_process_contract"],
        "source_state_receipt": _source_state_receipt(
            panel=panel,
            cache=cache,
            source_states_by_panel_entry_sha256=source_states_by_panel_entry_sha256,
        ),
        "state_encoding": "compose.rewrite.trace.encoded_state_v2_exact_slots",
        "tensor_contract": {
            **_TENSOR_CONTRACT,
            "support_time_hex": panel.request.support_time_hex,
        },
        "successor_partition_contract": dict(_PARTITION_CONTRACT),
        "entry_count": len(entries),
        "entry_inventory_sha256": _sha(entries),
        "entries": entries,
    }
    return {**body, "artifact_sha256": _sha(body)}


def validate_semantic_t1_prepared_inputs(
    value: Mapping[str, Any],
    *,
    expected_capacity_policy_sha256: str,
    expected_panel_artifact_sha256: str,
    expected_cache_completion_sha256: str,
    expected_initial_model_state_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    artifact = dict(value)
    body = dict(artifact)
    supplied_sha256 = body.pop("artifact_sha256", None)
    entries = artifact.get("entries")
    if (
        set(artifact) != _ROOT_FIELDS
        or artifact.get("schema") != SCHEMA
        or artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("status") != STATUS
        or supplied_sha256 != _sha(body)
        or artifact.get("capacity_policy_sha256") != expected_capacity_policy_sha256
        or artifact.get("panel_artifact_sha256") != expected_panel_artifact_sha256
        or artifact.get("cache_completion_sha256") != expected_cache_completion_sha256
        or artifact.get("initial_model_state_sha256") != expected_initial_model_state_sha256
        or artifact.get("implementation_sha256")
        != semantic_t1_prepared_input_implementation_sha256(repo_root=repo_root)
        or not isinstance(entries, list)
        or not _is_sha(artifact.get("cache_source_revision_sha256"))
        or not _source_revision_is_valid(artifact.get("source_revision"))
        or artifact.get("entry_count") != len(entries)
        or artifact.get("entry_inventory_sha256") != _sha(entries)
        or any(artifact.get(name) is not expected for name, expected in NO_AUTHORITY.items())
        or artifact.get("state_encoding") != "compose.rewrite.trace.encoded_state_v2_exact_slots"
        or artifact.get("successor_partition_contract") != _PARTITION_CONTRACT
        or not isinstance(artifact.get("tensor_contract"), Mapping)
        or set(artifact["tensor_contract"]) != {*_TENSOR_CONTRACT, "support_time_hex"}
        or any(
            artifact["tensor_contract"].get(name) != expected
            for name, expected in _TENSOR_CONTRACT.items()
        )
        or not isinstance(artifact["tensor_contract"].get("support_time_hex"), str)
        or not artifact["tensor_contract"]["support_time_hex"]
    ):
        raise SemanticT1PreparedInputError(
            "semantic T1 prepared-input schema, authority, or lineage disagrees"
        )
    observed_ids: list[str] = []
    observed_order: list[tuple[str, str, str]] = []
    source_receipt_entries: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != _ENTRY_FIELDS:
            raise SemanticT1PreparedInputError("prepared entry must be an object")
        entry = dict(raw)
        entry_body = dict(entry)
        entry_sha256 = entry_body.pop("entry_sha256", None)
        if entry_sha256 != _sha(entry_body):
            raise SemanticT1PreparedInputError("prepared entry self-hash disagrees")
        state = decode_state(entry["exact_state"])
        partition = compiled_successor_map_from_payload(entry["successor_partition"])
        if (
            encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
            or partition.source_state_sha256 != entry["source_state_sha256"]
            or entry.get("canonical_successor_count") != len(partition.successor_groups)
            or type(entry.get("productive_alias_count")) is not int
            or type(entry.get("virtual_alias_count")) is not int
            or type(entry.get("raw_mark_count")) is not int
            or entry.get("productive_alias_count")
            != sum(len(group.marks) for group in partition.successor_groups)
            or entry.get("virtual_alias_count") != len(partition.virtual_marks)
            or entry.get("raw_mark_count")
            != entry.get("productive_alias_count") + entry.get("virtual_alias_count")
            or entry.get("model_scores_or_probabilities_stored") is not False
            or entry.get("hazard_included") is not False
            or not _is_sha(entry.get("panel_entry_sha256"))
            or not _is_sha(entry.get("cache_record_sha256"))
            or not isinstance(entry.get("model_family"), str)
            or not entry["model_family"]
            or not isinstance(entry.get("capability_cell_id"), str)
            or not entry["capability_cell_id"]
            or entry.get("objective_coefficient") != 1
            or entry["raw_mark_count"] <= 0
            or type(entry.get("production_successor_alias_multiplicity")) is not int
            or entry["production_successor_alias_multiplicity"] <= 0
        ):
            raise SemanticT1PreparedInputError("prepared entry state or successor census disagrees")
        teacher = next(
            (
                group
                for group in partition.successor_groups
                if group.target_key == entry["successor_canonical_key"]
            ),
            None,
        )
        if teacher is None:
            raise SemanticT1PreparedInputError(
                "prepared teacher successor is absent from the full partition"
            )
        if len(teacher.marks) != entry["production_successor_alias_multiplicity"]:
            raise SemanticT1PreparedInputError("prepared teacher alias multiplicity disagrees")
        observed_ids.append(str(entry["panel_entry_sha256"]))
        observed_order.append(
            (
                str(entry["model_family"]),
                str(entry["capability_cell_id"]),
                str(entry["panel_entry_sha256"]),
            )
        )
        source_receipt_entries.append(
            {
                "panel_entry_sha256": entry["panel_entry_sha256"],
                "cache_record_sha256": entry["cache_record_sha256"],
                "source_state_sha256": entry["source_state_sha256"],
            }
        )
    if (
        len(set(observed_ids)) != len(observed_ids)
        or observed_order != sorted(observed_order)
        or artifact.get("panel_entry_inventory_sha256") != _sha(sorted(observed_ids))
    ):
        raise SemanticT1PreparedInputError(
            "prepared entries are not a canonical unique panel inventory"
        )
    _validate_source_state_receipt(
        artifact["source_state_receipt"],
        panel_artifact_sha256=artifact["panel_artifact_sha256"],
        cache_completion_sha256=artifact["cache_completion_sha256"],
        entries=source_receipt_entries,
    )
    return artifact


@dataclass(frozen=True, slots=True)
class LoadedSemanticT1PreparedInputs:
    artifact: Mapping[str, Any]
    states_by_panel_entry_sha256: Mapping[str, Any]
    partitions_by_panel_entry_sha256: Mapping[str, CompiledStateSuccessorMap]


def authenticate_semantic_t1_prepared_inputs(
    prepared: LoadedSemanticT1PreparedInputs,
    *,
    panel: SemanticT1PanelArtifact,
    cache: SemanticT1SuccessorCache,
) -> LoadedSemanticT1PreparedInputs:
    """Authenticate every prepared row against the reopened panel and cache.

    Prepared-input self-hashes establish byte integrity, not scientific
    authenticity.  This boundary reopens the two authorities that selected each
    row and verifies the exact source, teacher fiber, canonical teacher group,
    action identity, and complete mark census.  The successor cache does not
    contain nonteacher successor coordinates, so it cannot independently
    authenticate them.  Their authority is the separately published,
    content-addressed CPU-preparation completion receipt.
    """

    if not isinstance(prepared, LoadedSemanticT1PreparedInputs):
        raise TypeError("prepared inputs must be a loaded semantic T1 artifact")
    artifact = prepared.artifact
    panel_entries = {entry.panel_entry_sha256: entry for entry in panel.entries}
    prepared_entries = {str(entry["panel_entry_sha256"]): entry for entry in artifact["entries"]}
    panel_ids = set(panel_entries)
    if (
        artifact.get("panel_artifact_sha256") != panel.artifact_sha256
        or artifact.get("cache_completion_sha256") != cache.completion.get("completion_sha256")
        or artifact.get("cache_manifest_sha256") != cache.manifest.get("manifest_sha256")
        or cache.manifest.get("panel_binding", {}).get("panel_artifact_sha256")
        != panel.artifact_sha256
        or set(prepared_entries) != panel_ids
        or set(prepared.states_by_panel_entry_sha256) != panel_ids
        or set(prepared.partitions_by_panel_entry_sha256) != panel_ids
        or set(cache.records_by_panel_entry_sha256) != panel_ids
    ):
        raise SemanticT1PreparedInputError(
            "prepared inputs, reopened panel, and successor-cache inventories disagree"
        )

    for panel_id in sorted(panel_ids):
        entry = prepared_entries[panel_id]
        panel_entry = panel_entries[panel_id]
        try:
            record = cache.record_for_panel_entry_sha256(panel_id)
        except KeyError as error:
            raise SemanticT1PreparedInputError(
                f"prepared panel entry is absent from the reopened cache: {panel_id}"
            ) from error
        state = prepared.states_by_panel_entry_sha256[panel_id]
        partition = prepared.partitions_by_panel_entry_sha256[panel_id]
        teacher = record.teacher_fiber
        if teacher is None:
            raise SemanticT1PreparedInputError(
                f"prepared panel entry has a terminal cache record: {panel_id}"
            )
        teacher_groups = tuple(
            group
            for group in partition.successor_groups
            if group.target_key == panel_entry.successor_canonical_key
        )
        if len(teacher_groups) != 1:
            raise SemanticT1PreparedInputError(
                f"prepared teacher canonical grouping disagrees: {panel_id}"
            )
        teacher_group = teacher_groups[0]
        record_sha256 = _sha(successor_fiber_cache_record_payload(record))
        exact_source_sha256 = persistent_slot_state_sha256(state)
        productive_alias_count = sum(len(group.marks) for group in partition.successor_groups)
        virtual_alias_count = len(partition.virtual_marks)
        expected_entry_fields = {
            "model_family": panel_entry.model_family,
            "capability_cell_id": panel_entry.capability_cell_id,
            "support_time_hex": panel_entry.support_time_hex,
            "source_state_sha256": panel_entry.source_state_sha256,
            "successor_canonical_key": panel_entry.successor_canonical_key,
            "objective_coefficient": panel_entry.objective_coefficient,
            "raw_mark_count": panel_entry.raw_mark_count,
            "canonical_successor_count": panel_entry.canonical_successor_count,
            "production_successor_alias_multiplicity": (
                panel_entry.production_successor_alias_multiplicity
            ),
        }
        if any(entry.get(name) != expected for name, expected in expected_entry_fields.items()):
            raise SemanticT1PreparedInputError(
                f"prepared row differs from its reopened panel entry: {panel_id}"
            )
        if (
            entry.get("cache_record_sha256") != record_sha256
            or exact_source_sha256 != panel_entry.source_state_sha256
            or record.source_state_sha256 != panel_entry.source_state_sha256
            or partition.source_state_sha256 != panel_entry.source_state_sha256
            or partition.source_key != panel_entry.source_canonical_key
            or record.state_support != partition.state_support
        ):
            raise SemanticT1PreparedInputError(
                f"prepared exact source or cache-record identity disagrees: {panel_id}"
            )
        if (
            teacher.target_key != panel_entry.successor_canonical_key
            or teacher.target_state_sha256 != panel_entry.representative_target_state_sha256
            or teacher_group.aliases != tuple(sorted(teacher.aliases))
            or teacher.target_state_sha256 not in teacher_group.successor_state_sha256s
            or not teacher_group.contains_exact_action(
                successor_state_sha256=teacher.target_state_sha256,
                action_sha256=panel_entry.representative_action_sha256,
            )
        ):
            raise SemanticT1PreparedInputError(
                f"prepared teacher target, aliases, or canonical grouping disagrees: {panel_id}"
            )
        if (
            entry.get("productive_alias_count") != productive_alias_count
            or entry.get("virtual_alias_count") != virtual_alias_count
            or panel_entry.raw_mark_count != productive_alias_count + virtual_alias_count
            or panel_entry.canonical_successor_count != len(partition.successor_groups)
            or panel_entry.production_successor_alias_multiplicity != len(teacher_group.marks)
        ):
            raise SemanticT1PreparedInputError(
                f"prepared full-partition census or teacher multiplicity disagrees: {panel_id}"
            )
    return prepared


def write_semantic_t1_prepared_inputs(path: Path, artifact: Mapping[str, Any]) -> bool:
    return write_bytes_if_absent(Path(path), _canonical_bytes(dict(artifact), newline=True))


def load_semantic_t1_prepared_inputs(
    path: Path,
    *,
    expected_capacity_policy_sha256: str,
    expected_panel_artifact_sha256: str,
    expected_cache_completion_sha256: str,
    expected_initial_model_state_sha256: str,
    repo_root: Path,
) -> LoadedSemanticT1PreparedInputs:
    source = Path(path)
    try:
        if not 0 < source.stat().st_size <= MAX_BYTES:
            raise SemanticT1PreparedInputError(
                "semantic T1 prepared-input file exceeds its byte bound"
            )
        raw = source.read_bytes()
        payload = json.loads(raw)
    except SemanticT1PreparedInputError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1PreparedInputError(
            f"semantic T1 prepared inputs are absent or invalid: {source}"
        ) from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticT1PreparedInputError("semantic T1 prepared inputs are not canonical JSON")
    validated = validate_semantic_t1_prepared_inputs(
        payload,
        expected_capacity_policy_sha256=expected_capacity_policy_sha256,
        expected_panel_artifact_sha256=expected_panel_artifact_sha256,
        expected_cache_completion_sha256=expected_cache_completion_sha256,
        expected_initial_model_state_sha256=expected_initial_model_state_sha256,
        repo_root=repo_root,
    )
    states: dict[str, Any] = {}
    partitions: dict[str, CompiledStateSuccessorMap] = {}
    for entry in validated["entries"]:
        panel_id = entry["panel_entry_sha256"]
        states[panel_id] = decode_state(entry["exact_state"])
        partitions[panel_id] = compiled_successor_map_from_payload(entry["successor_partition"])
    return LoadedSemanticT1PreparedInputs(
        artifact=MappingProxyType(validated),
        states_by_panel_entry_sha256=MappingProxyType(states),
        partitions_by_panel_entry_sha256=MappingProxyType(partitions),
    )


__all__ = [
    "FILENAME",
    "LoadedSemanticT1PreparedInputs",
    "SemanticT1PreparedInputError",
    "authenticate_semantic_t1_prepared_inputs",
    "build_semantic_t1_prepared_inputs",
    "compiled_successor_map_from_payload",
    "compiled_successor_map_payload",
    "load_semantic_t1_prepared_inputs",
    "semantic_t1_prepared_input_implementation_sha256",
    "validate_semantic_t1_prepared_inputs",
    "write_semantic_t1_prepared_inputs",
]
