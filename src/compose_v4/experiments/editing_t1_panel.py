"""Immutable exact-address panels for the editing T1 capacity gate.

T1 is a bounded *development* diagnostic.  It asks whether the configured
scratch editor can memorize exact canonical molecular successors before any
mixed-corpus pilot is allowed.  This module derives two disjoint panel views
from the already-frozen RingCore validation family-forensics panel:

* exactly 64 globally deterministic, unique persistent-slot source states for
  every active family;
* every observed exact state having more than one teacher successor under the
  full mixed-family editing law; and
* explicitly secondary within-family repeated-state views.

The deterministic panels exclude any source that has multiple target
successors anywhere in the active eight-family projection of the complete
frozen forensics panel, including targets recorded under another active
operator family. The frozen ``ring_system_delete`` rows are explicitly counted
but excluded before grouping under the bounded-pilot operator freeze. Thus an
operator-specific capacity arm cannot turn a genuinely stochastic active-family
choice into an arbitrary one-hot label. No example is synthesized or duplicated
to satisfy a count. The artifact contains immutable source-row references and is
re-derived during loading, so changing the source panel, semantic sidecar, or
selection implementation fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    Active8TraceInventoryError,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

EDITING_T1_PANEL_SCHEMA = "compose.editing.t1_successor_panel"
EDITING_T1_PANEL_SCHEMA_VERSION = 3
EDITING_T1_PANEL_STATUS = "FROZEN_DEVELOPMENT_CAPACITY_PANEL_NO_DECISION"
EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY = 64
EDITING_T1_SUPPORT_TIME = 0.5
EDITING_T1_SELECTION_ALGORITHM = (
    "physical_active8_whole_trace_admission_then_forensics_stream_first_globally_"
    "deterministic_unique_exact_state_64_per_family_plus_all_global_and_within_"
    "family_multitarget_exact_state_groups_v3"
)
EDITING_T1_GLOBAL_FAMILY_SELECTOR = "all_families"
EDITING_T1_UNIQUE_PANEL_KIND = "unique_state"
EDITING_T1_GLOBAL_REPEATED_PANEL_KIND = "global_repeated_state_distribution"
EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND = "within_family_repeated_state_distribution"
EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES: tuple[str, ...] = ()
EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES = ("ring_system_delete",)
EDITING_T1_REQUIRED_SCOPES = (
    "heads_only",
    "heads_plus_local_adapter",
    "all",
)

_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "source",
    "selection",
    "architecture",
    "thresholds",
    "unique_state_panels",
    "global_repeated_state_panel",
    "within_family_repeated_state_panels",
    "census",
    "artifact_sha256",
}
_SOURCE_FIELDS = {
    "forensics_file_sha256",
    "forensics_artifact_sha256",
    "leaderboard_protocol_sha256",
    "frozen_inventory_sha256",
    "semantic_sidecar_file_sha256",
    "semantic_sidecar_manifest_file_sha256",
    "semantic_sidecar_manifest_sha256",
    "semantic_sidecar_config_sha256",
    "semantic_sidecar_provenance_sha256",
    "charge_policy_audit_file_sha256",
    "charge_policy_exclusions_file_sha256",
    "charge_policy_exclusion_payload_sha256",
    "charge_policy_source_input_inventory_sha256",
    "charge_policy_audit_implementation_sha256",
    "charge_policy_excluded_unique_trace_addresses",
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
}
_SELECTION_FIELDS = {
    "algorithm",
    "active_families",
    "diagnostic_only_families",
    "operator_freeze_excluded_families",
    "unique_examples_per_family",
    "global_determinism_policy",
    "primary_repeated_state_policy",
    "conditional_repeated_state_policy",
    "conditional_repeated_state_relationship",
    "operator_freeze_exclusion_policy",
    "charge_policy_exclusion_policy",
    "active8_whole_trace_admission_policy",
    "support_time",
}
_ARCHITECTURE_FIELDS = {
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "atom_vocabulary_class_count",
    "enable_ring_restates",
    "enable_cyclic_graft",
    "enable_heteroatom_scan",
    "enable_ring_opening",
    "enable_cycle_ops",
    "enable_ring_grow_macro",
    "enable_ring_system_delete",
    "required_scope_ladder",
}
_THRESHOLD_FIELDS = {
    "minimum_unique_state_teacher_successor_top1",
    "minimum_unique_state_teacher_successor_probability",
    "maximum_unique_state_teacher_successor_nll",
    "maximum_global_repeated_state_excess_nll_over_empirical_entropy",
}
_ROW_REF_FIELDS = {"source_row_index", "source_row_sha256"}
_WITHIN_FAMILY_REPEATED_GROUP_FIELDS = {
    "source_state_sha256",
    "distinct_teacher_successor_count",
    "rows",
}
_GLOBAL_REPEATED_GROUP_FIELDS = {
    *_WITHIN_FAMILY_REPEATED_GROUP_FIELDS,
    "distinct_teacher_family_count",
}
_CHARGE_EXCLUSION_FIELDS = {
    "schema",
    "schema_version",
    "selection",
    "source_input_inventory_sha256",
    "charge_policy_audit_implementation_sha256",
    "excluded_unique_trace_addresses",
    "violation_type_bit_order",
    "shards",
    "payload_sha256",
}
_CHARGE_EXCLUSION_SHARD_FIELDS = {
    "manifest_layer",
    "envelope_layer",
    "partition",
    "relative_path",
    "packed_shard_content_sha256",
    "excluded_entry_count",
    "violating_entry_indices",
    "entries",
}
_CHARGE_EXCLUSION_ENTRY_FIELDS = {
    "entry_index",
    "trace_id",
    "violation_type_mask",
}


class EditingT1PanelError(RuntimeError):
    """The T1 panel or one of its frozen sources is invalid."""


ACTIVE8_T1_IDENTITY_FIELDS = (
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
)


@dataclass(frozen=True)
class _ForensicsTraceAddress:
    packed_shard_content_sha256: str
    entry_index: int
    trace_id: str


def active8_t1_identity(
    admission: Active8TraceAdmission,
) -> Mapping[str, str]:
    """Project the five non-interchangeable Active8 identities used by T1."""

    if not isinstance(admission, Active8TraceAdmission):
        raise TypeError("T1 requires a verified Active8TraceAdmission")
    return MappingProxyType(
        {
            "active8_inventory_manifest_file_sha256": (
                admission.manifest_file_sha256
            ),
            "active8_inventory_sha256": admission.inventory_sha256,
            "active8_effective_source_corpus_cache_sha256": (
                admission.effective_source_corpus_cache_sha256
            ),
            "active8_unified_packed_manifest_sha256": (
                admission.unified_packed_manifest_sha256
            ),
            "active8_support_contract_sha256": (
                admission.support_contract_sha256
            ),
        }
    )


def _forensics_trace_address(
    row: Mapping[str, Any],
    admission: Active8TraceAdmission,
) -> _ForensicsTraceAddress:
    state_ref = row.get("exact_state_ref")
    if not isinstance(state_ref, Mapping):
        raise EditingT1PanelError(
            "forensics row lacks an exact packed-state reference"
        )
    shard_digest = state_ref.get("shard_sha256")
    record_index = state_ref.get("record_index")
    shard_name = state_ref.get("shard_name")
    record_key = row.get("record_key")
    partition = row.get("partition")
    if (
        not _is_sha256(shard_digest)
        or type(record_index) is not int
        or record_index < 0
        or not isinstance(shard_name, str)
        or not shard_name
        or not isinstance(record_key, str)
        or not record_key
        or not isinstance(partition, str)
        or not partition
    ):
        raise EditingT1PanelError(
            "forensics row lacks a complete Active8 trace address"
        )
    shard_path = PurePosixPath(shard_name)
    if (
        shard_path.is_absolute()
        or ".." in shard_path.parts
        or len(shard_path.parts) < 3
        or shard_path.parts[-2] != partition
    ):
        raise EditingT1PanelError(
            "forensics packed shard path disagrees with its partition"
        )
    prefix = f"{shard_name}:{record_index}:"
    if not record_key.startswith(prefix) or len(record_key) == len(prefix):
        raise EditingT1PanelError(
            "forensics record key disagrees with its exact packed address"
        )
    trace_id = record_key[len(prefix) :]
    try:
        expected_digest = admission.expected_source_digest(
            packed_shard_name=shard_path.name,
            layer=shard_path.parts[-3],
            partition=partition,
        )
    except Active8TraceInventoryError as error:
        raise EditingT1PanelError(
            "forensics row names a shard absent from the physical Active8 inventory"
        ) from error
    if expected_digest != shard_digest:
        raise EditingT1PanelError(
            "forensics row shard digest disagrees with the physical Active8 inventory"
        )
    return _ForensicsTraceAddress(
        packed_shard_content_sha256=shard_digest,
        entry_index=record_index,
        trace_id=trace_id,
    )


def filter_active8_forensics_rows(
    rows: Sequence[Mapping[str, Any]],
    admission: Active8TraceAdmission,
) -> tuple[tuple[Mapping[str, Any], ...], int, int]:
    """Apply the immutable whole-trace decision before any T1 row selection.

    Returns ``(accepted_rows, excluded_row_count, excluded_trace_count)``.
    One excluded middle action therefore removes every otherwise usable
    prefix, suffix, and terminal observation from the trace.
    """

    if not isinstance(admission, Active8TraceAdmission):
        raise TypeError("T1 requires a verified Active8TraceAdmission")
    accepted_rows: list[Mapping[str, Any]] = []
    excluded_traces: set[tuple[str, int, str]] = set()
    for row in rows:
        address = _forensics_trace_address(row, admission)
        try:
            accepted = admission.is_accepted(address)
        except Active8TraceInventoryError as error:
            raise EditingT1PanelError(
                "forensics row disagrees with its whole-trace Active8 decision"
            ) from error
        if accepted:
            accepted_rows.append(row)
        else:
            excluded_traces.add(
                (
                    address.packed_shard_content_sha256,
                    address.entry_index,
                    address.trace_id,
                )
            )
    return (
        tuple(accepted_rows),
        len(rows) - len(accepted_rows),
        len(excluded_traces),
    )


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
        raise EditingT1PanelError("T1 panel metadata is not finite canonical JSON") from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EditingT1PanelError(f"{name} must be an object")
    if set(value) != expected:
        raise EditingT1PanelError(
            f"{name} fields disagree; "
            f"missing={sorted(expected - set(value))!r}, "
            f"unexpected={sorted(set(value) - expected)!r}"
        )
    return value


def validate_charge_policy_exclusions(
    *,
    audit: Mapping[str, Any],
    audit_file_sha256: str,
    exclusions: Mapping[str, Any],
    exclusions_file_sha256: str,
) -> Mapping[tuple[str, int], str]:
    """Validate the frozen whole-trace exclusion index and return trace IDs."""

    for name, value in (
        ("audit_file_sha256", audit_file_sha256),
        ("exclusions_file_sha256", exclusions_file_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    payload = _require_exact_fields(
        exclusions,
        _CHARGE_EXCLUSION_FIELDS,
        name="charge-policy exclusion payload",
    )
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload["schema"] != "compose.data.packed_charge_policy_exclusions"
        or payload["schema_version"] != 1
        or payload["selection"] != "exclude_whole_trace_if_any_consecutive_state_pair_violates"
        or not _is_sha256(payload["source_input_inventory_sha256"])
        or not _is_sha256(payload["charge_policy_audit_implementation_sha256"])
        or not isinstance(payload["violation_type_bit_order"], list)
        or not payload["violation_type_bit_order"]
        or any(
            not isinstance(value, str) or not value for value in payload["violation_type_bit_order"]
        )
        or len(payload["violation_type_bit_order"]) != len(set(payload["violation_type_bit_order"]))
        or payload["payload_sha256"] != _stable_sha256(body)
    ):
        raise EditingT1PanelError(
            "charge-policy exclusion schema, selection, or self-hash is invalid"
        )
    expected_count = payload["excluded_unique_trace_addresses"]
    if type(expected_count) is not int or expected_count < 0:
        raise EditingT1PanelError("charge-policy excluded trace count must be nonnegative")
    shards = payload["shards"]
    if not isinstance(shards, list):
        raise EditingT1PanelError("charge-policy exclusion shards must be a list")
    index: dict[tuple[str, int], str] = {}
    for shard in shards:
        shard = _require_exact_fields(
            shard,
            _CHARGE_EXCLUSION_SHARD_FIELDS,
            name="charge-policy exclusion shard",
        )
        shard_digest = shard["packed_shard_content_sha256"]
        entries = shard["entries"]
        indices = shard["violating_entry_indices"]
        if (
            not _is_sha256(shard_digest)
            or type(shard["excluded_entry_count"]) is not int
            or not isinstance(entries, list)
            or not isinstance(indices, list)
            or shard["excluded_entry_count"] != len(entries)
            or any(type(index_value) is not int or index_value < 0 for index_value in indices)
            or indices != sorted(set(indices))
        ):
            raise EditingT1PanelError("charge-policy exclusion shard is malformed")
        observed_indices: list[int] = []
        for entry in entries:
            entry = _require_exact_fields(
                entry,
                _CHARGE_EXCLUSION_ENTRY_FIELDS,
                name="charge-policy exclusion entry",
            )
            entry_index = entry["entry_index"]
            trace_id = entry["trace_id"]
            violation_type_mask = entry["violation_type_mask"]
            if (
                type(entry_index) is not int
                or entry_index < 0
                or not isinstance(trace_id, str)
                or not trace_id
                or type(violation_type_mask) is not int
                or violation_type_mask <= 0
                or violation_type_mask >= (1 << len(payload["violation_type_bit_order"]))
            ):
                raise EditingT1PanelError(
                    "charge-policy exclusion entry lacks an exact trace identity"
                )
            key = (str(shard_digest), entry_index)
            if key in index:
                raise EditingT1PanelError(
                    "charge-policy exclusion index repeats an exact trace address"
                )
            index[key] = trace_id
            observed_indices.append(entry_index)
        if observed_indices != indices:
            raise EditingT1PanelError(
                "charge-policy exclusion entry identities disagree with their shard index"
            )
    if len(index) != expected_count:
        raise EditingT1PanelError(
            "charge-policy exclusion count disagrees with exact indexed traces"
        )

    audit_exclusions = audit.get("exclusion_payload")
    audit_counts = audit.get("counts")
    audit_implementation = audit.get("implementation")
    if (
        audit.get("schema") != "compose.data.packed_charge_policy_audit"
        or audit.get("schema_version") != 1
        or audit.get("input_inventory_sha256") != payload["source_input_inventory_sha256"]
        or not isinstance(audit_counts, Mapping)
        or audit_counts.get("traces_with_violations") != expected_count
        or not isinstance(audit_implementation, Mapping)
        or audit_implementation.get("implementation_sha256")
        != payload["charge_policy_audit_implementation_sha256"]
        or audit_exclusions != dict(payload)
    ):
        raise EditingT1PanelError("charge-policy audit and standalone exclusion index disagree")
    return MappingProxyType(index)


def _source_row_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    index = row.get("row_index")
    digest = row.get("row_sha256")
    if type(index) is not int or index < 0 or not _is_sha256(digest):
        raise EditingT1PanelError("forensics row lacks a valid row index or self-hash")
    return {
        "source_row_index": index,
        "source_row_sha256": digest,
    }


def _validate_forensics_rows(
    forensics: Mapping[str, Any],
    sidecar: LoadedSemanticCellSidecar,
) -> tuple[Mapping[str, Any], ...]:
    rows = forensics.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise EditingT1PanelError("forensics artifact lacks a row sequence")
    materialized = tuple(rows)
    if not materialized:
        raise EditingT1PanelError("forensics artifact is empty")
    source_families = set(
        (*RINGCORE_EDITING_FAMILIES, *EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES)
    )
    sidecar_cells = sidecar.semantic_cell_ids
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise EditingT1PanelError("forensics row is not an object")
        if row.get("row_index") != index:
            raise EditingT1PanelError("forensics rows are not in their frozen index order")
        family = row.get("teacher_family")
        if family not in source_families:
            raise EditingT1PanelError(f"forensics row references an unknown family: {family!r}")
        if row.get("terminal") is not False:
            raise EditingT1PanelError("T1 family forensics may contain nonterminal rows only")
        source_digest = row.get("source_state_sha256")
        source_key = row.get("source_state_key")
        target_key = row.get("teacher_successor_key")
        if (
            not _is_sha256(source_digest)
            or not isinstance(source_key, str)
            or not source_key
            or not isinstance(target_key, str)
            or not target_key
            or source_key == target_key
        ):
            raise EditingT1PanelError(
                "forensics row lacks a productive exact source/target identity"
            )
        state_ref = row.get("exact_state_ref")
        if not isinstance(state_ref, Mapping):
            raise EditingT1PanelError("forensics row lacks an exact packed-state reference")
        exact_key = (
            state_ref.get("shard_sha256"),
            state_ref.get("record_index"),
            state_ref.get("progress_index"),
        )
        if exact_key not in sidecar_cells:
            raise EditingT1PanelError(f"semantic sidecar is missing forensics row {index}")
        if sidecar_cells[exact_key] != row.get("semantic_cell_id"):
            raise EditingT1PanelError(f"semantic sidecar cell disagrees for forensics row {index}")
    return materialized


def _select_unique_panels(
    rows: tuple[Mapping[str, Any], ...],
) -> dict[str, list[dict[str, Any]]]:
    targets_by_state: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        targets_by_state[str(row["source_state_sha256"])].add(str(row["teacher_successor_key"]))
    selected = {family: [] for family in RINGCORE_EDITING_FAMILIES}
    seen = {family: set() for family in RINGCORE_EDITING_FAMILIES}
    for row in rows:
        family = str(row["teacher_family"])
        source_digest = str(row["source_state_sha256"])
        if (
            len(selected[family]) < EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY
            and source_digest not in seen[family]
            and len(targets_by_state[source_digest]) == 1
        ):
            selected[family].append(_source_row_ref(row))
            seen[family].add(source_digest)
    shortfalls = {
        family: EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY - len(panel)
        for family, panel in selected.items()
        if len(panel) != EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY
    }
    if shortfalls:
        raise EditingT1PanelError(
            f"forensics artifact cannot fill exact unique-state panels: {shortfalls}"
        )
    return selected


def _group_repeated_rows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_family_count: bool,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for source_digest, occurrences in grouped.items():
        target_count = len({str(row["teacher_successor_key"]) for row in occurrences})
        if target_count <= 1:
            continue
        ordered = sorted(
            occurrences,
            key=lambda row: int(row["row_index"]),
        )
        group = {
            "source_state_sha256": source_digest,
            "distinct_teacher_successor_count": target_count,
            "rows": [_source_row_ref(row) for row in ordered],
        }
        if include_family_count:
            group["distinct_teacher_family_count"] = len(
                {str(row["teacher_family"]) for row in occurrences}
            )
        groups.append(group)
    return sorted(
        groups,
        key=lambda group: (
            group["rows"][0]["source_row_index"],
            group["source_state_sha256"],
        ),
    )


def _select_repeated_panels(
    rows: tuple[Mapping[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    global_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    within_family_grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        family: defaultdict(list) for family in RINGCORE_EDITING_FAMILIES
    }
    for row in rows:
        source_digest = str(row["source_state_sha256"])
        global_grouped[source_digest].append(row)
        within_family_grouped[str(row["teacher_family"])][source_digest].append(row)

    global_panel = _group_repeated_rows(
        global_grouped,
        include_family_count=True,
    )
    conditional_panels: dict[str, list[dict[str, Any]]] = {}
    for family in RINGCORE_EDITING_FAMILIES:
        conditional_panels[family] = _group_repeated_rows(
            within_family_grouped[family],
            include_family_count=False,
        )
    return global_panel, conditional_panels


def _panel_census(
    unique: Mapping[str, Sequence[Mapping[str, Any]]],
    global_repeated: Sequence[Mapping[str, Any]],
    conditional_repeated: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_row_count: int,
    active8_eligible_row_count: int,
    active8_excluded_row_count: int,
    active8_excluded_trace_count: int,
    charge_eligible_row_count: int,
    charge_excluded_row_count: int,
    pilot_eligible_row_count: int,
    operator_freeze_excluded_row_count: int,
    charge_excluded_forensics_trace_count: int,
) -> dict[str, Any]:
    conditional_group_counts = {
        family: len(conditional_repeated[family]) for family in RINGCORE_EDITING_FAMILIES
    }
    conditional_row_counts = {
        family: sum(len(group["rows"]) for group in conditional_repeated[family])
        for family in RINGCORE_EDITING_FAMILIES
    }
    global_row_total = sum(len(group["rows"]) for group in global_repeated)
    cross_family_groups = [
        group for group in global_repeated if int(group["distinct_teacher_family_count"]) > 1
    ]
    unique_references = [
        reference for family in RINGCORE_EDITING_FAMILIES for reference in unique[family]
    ]
    global_references = [
        reference for group in global_repeated for reference in group["rows"]
    ]
    conditional_references = [
        reference
        for family in RINGCORE_EDITING_FAMILIES
        for group in conditional_repeated[family]
        for reference in group["rows"]
    ]

    def identity_sets(
        references: Sequence[Mapping[str, Any]],
    ) -> tuple[set[str], set[tuple[str, int, int]], set[tuple[str, int]], set[tuple[int, str]]]:
        selected_rows = [source_rows[int(reference["source_row_index"])] for reference in references]
        return (
            {str(row["source_state_sha256"]) for row in selected_rows},
            {
                (
                    str(row["exact_state_ref"]["shard_sha256"]),
                    int(row["exact_state_ref"]["record_index"]),
                    int(row["exact_state_ref"]["progress_index"]),
                )
                for row in selected_rows
            },
            {
                (
                    str(row["exact_state_ref"]["shard_sha256"]),
                    int(row["exact_state_ref"]["record_index"]),
                )
                for row in selected_rows
            },
            {
                (int(reference["source_row_index"]), str(reference["source_row_sha256"]))
                for reference in references
            },
        )

    unique_identities = identity_sets(unique_references)
    global_identities = identity_sets(global_references)
    conditional_identities = identity_sets(conditional_references)

    def intersection_counts(
        left: tuple[
            set[str],
            set[tuple[str, int, int]],
            set[tuple[str, int]],
            set[tuple[int, str]],
        ],
        right: tuple[
            set[str],
            set[tuple[str, int, int]],
            set[tuple[str, int]],
            set[tuple[int, str]],
        ],
    ) -> dict[str, int]:
        return {
            "shared_source_states": len(left[0] & right[0]),
            "shared_progress_addresses": len(left[1] & right[1]),
            "shared_trace_addresses": len(left[2] & right[2]),
            "shared_source_row_references": len(left[3] & right[3]),
        }

    return {
        "source_forensics_rows_total": source_row_count,
        "active8_eligible_forensics_rows_total": active8_eligible_row_count,
        "active8_excluded_forensics_rows_total": active8_excluded_row_count,
        "active8_excluded_unique_forensics_trace_addresses": (
            active8_excluded_trace_count
        ),
        "charge_policy_eligible_forensics_rows_total": charge_eligible_row_count,
        "charge_policy_excluded_forensics_rows_total": charge_excluded_row_count,
        "charge_policy_excluded_unique_forensics_trace_addresses": (
            charge_excluded_forensics_trace_count
        ),
        "pilot_family_eligible_forensics_rows_total": pilot_eligible_row_count,
        "operator_freeze_excluded_forensics_rows_total": operator_freeze_excluded_row_count,
        "unique_rows_by_family": {
            family: len(unique[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "unique_rows_total": sum(len(unique[family]) for family in unique),
        "global_repeated_multitarget_groups_total": len(global_repeated),
        "global_repeated_rows_total": global_row_total,
        "global_repeated_cross_family_groups_total": len(cross_family_groups),
        "global_repeated_cross_family_rows_total": sum(
            len(group["rows"]) for group in cross_family_groups
        ),
        "global_repeated_panel_meets_64_example_minimum": (
            global_row_total >= EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY
        ),
        "within_family_repeated_multitarget_groups_by_family": (conditional_group_counts),
        "within_family_repeated_rows_by_family": conditional_row_counts,
        "within_family_repeated_multitarget_groups_total": sum(conditional_group_counts.values()),
        "within_family_repeated_rows_total": sum(conditional_row_counts.values()),
        "families_without_observed_within_family_multitarget_repeats": [
            family for family in RINGCORE_EDITING_FAMILIES if conditional_group_counts[family] == 0
        ],
        "panel_overlap": {
            "unique_state_vs_global_repeated": intersection_counts(
                unique_identities,
                global_identities,
            ),
            "unique_state_vs_within_family_repeated": intersection_counts(
                unique_identities,
                conditional_identities,
            ),
            "global_repeated_vs_within_family_repeated": intersection_counts(
                global_identities,
                conditional_identities,
            ),
            "within_family_repeated_is_exact_row_subset_of_global_repeated": (
                conditional_identities[3] <= global_identities[3]
            ),
            "statistically_independent_panel_claim": False,
        },
    }


def _artifact_body(
    *,
    source: Mapping[str, Any],
    unique: Mapping[str, Sequence[Mapping[str, Any]]],
    global_repeated: Sequence[Mapping[str, Any]],
    conditional_repeated: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rows: Sequence[Mapping[str, Any]],
    source_row_count: int,
    active8_eligible_row_count: int,
    active8_excluded_row_count: int,
    active8_excluded_trace_count: int,
    charge_eligible_row_count: int,
    charge_excluded_row_count: int,
    pilot_eligible_row_count: int,
    operator_freeze_excluded_row_count: int,
    charge_excluded_forensics_trace_count: int,
) -> dict[str, Any]:
    return {
        "schema": EDITING_T1_PANEL_SCHEMA,
        "schema_version": EDITING_T1_PANEL_SCHEMA_VERSION,
        "status": EDITING_T1_PANEL_STATUS,
        "training_authorized": False,
        "source": dict(source),
        "selection": {
            "algorithm": EDITING_T1_SELECTION_ALGORITHM,
            "active_families": list(RINGCORE_EDITING_FAMILIES),
            "diagnostic_only_families": list(EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES),
            "operator_freeze_excluded_families": list(
                EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES
            ),
            "unique_examples_per_family": (EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY),
            "global_determinism_policy": (
                "after whole-trace Active8 admission, operator-freeze, and "
                "charge-policy exclusions, unique family panels exclude an exact "
                "source whenever the full active eight-family pilot projection "
                "records more than one distinct canonical teacher successor"
            ),
            "primary_repeated_state_policy": (
                "all exact persistent-slot source groups with more than one distinct "
                "canonical teacher successor under the active eight-family pilot law; "
                "operator-freeze-excluded families are omitted before grouping; every "
                "retained frozen observation and its importance weight are preserved, "
                "including repeated observations of one packed progress address; no "
                "synthetic padding"
            ),
            "conditional_repeated_state_policy": (
                "secondary diagnostic views group the same frozen rows within one "
                "recorded teacher family; they are not the primary T1 successor law"
            ),
            "conditional_repeated_state_relationship": (
                "every within-family repeated row is an exact conditional subset/view "
                "of the primary global repeated evidence; these panels deliberately "
                "reuse observations and must not be treated as statistically independent"
            ),
            "operator_freeze_exclusion_policy": (
                "exclude every row whose recorded teacher family is ring_system_delete "
                "before unique or repeated-state selection; the superseded source "
                "forensics remains immutable and is counted explicitly"
            ),
            "charge_policy_exclusion_policy": (
                "exclude a complete packed trace before panel selection whenever "
                "the frozen exact-state charge-policy index contains its "
                "(packed_shard_content_sha256, entry_index); selected addresses "
                "are checked again against indexed trace_id at runtime"
            ),
            "active8_whole_trace_admission_policy": (
                "load and physically hash the immutable Active8 inventory and every "
                "decision shard, then require the exact "
                "(packed_shard_content_sha256, entry_index, trace_id) decision before "
                "any T1 selection; if any middle action is excluded, no neighboring "
                "row from that trace may enter any T1 panel"
            ),
            "support_time": EDITING_T1_SUPPORT_TIME,
        },
        "architecture": {
            "hidden_dim": 256,
            "message_passing_steps": 6,
            "mark_dim": 32,
            "atom_vocabulary_class_count": 15,
            "enable_ring_restates": True,
            "enable_cyclic_graft": True,
            "enable_heteroatom_scan": True,
            "enable_ring_opening": True,
            "enable_cycle_ops": True,
            "enable_ring_grow_macro": False,
            "enable_ring_system_delete": False,
            "required_scope_ladder": list(EDITING_T1_REQUIRED_SCOPES),
        },
        "thresholds": {
            "minimum_unique_state_teacher_successor_top1": None,
            "minimum_unique_state_teacher_successor_probability": None,
            "maximum_unique_state_teacher_successor_nll": None,
            "maximum_global_repeated_state_excess_nll_over_empirical_entropy": None,
        },
        "unique_state_panels": {
            family: list(unique[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "global_repeated_state_panel": list(global_repeated),
        "within_family_repeated_state_panels": {
            family: list(conditional_repeated[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "census": _panel_census(
            unique,
            global_repeated,
            conditional_repeated,
            source_rows=source_rows,
            source_row_count=source_row_count,
            active8_eligible_row_count=active8_eligible_row_count,
            active8_excluded_row_count=active8_excluded_row_count,
            active8_excluded_trace_count=active8_excluded_trace_count,
            charge_eligible_row_count=charge_eligible_row_count,
            charge_excluded_row_count=charge_excluded_row_count,
            pilot_eligible_row_count=pilot_eligible_row_count,
            operator_freeze_excluded_row_count=operator_freeze_excluded_row_count,
            charge_excluded_forensics_trace_count=(
                charge_excluded_forensics_trace_count
            ),
        ),
    }


def build_editing_t1_panel(
    *,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_unified_packed_manifest_sha256: str,
) -> dict[str, Any]:
    """Derive the exact T1 panels from validated frozen inputs."""

    for name, value in (
        ("forensics_file_sha256", forensics_file_sha256),
        ("semantic_sidecar_file_sha256", semantic_sidecar_file_sha256),
        (
            "semantic_sidecar_manifest_file_sha256",
            semantic_sidecar_manifest_file_sha256,
        ),
        ("charge_policy_audit_file_sha256", charge_policy_audit_file_sha256),
        (
            "charge_policy_exclusions_file_sha256",
            charge_policy_exclusions_file_sha256,
        ),
        (
            "gate_zero_unified_packed_manifest_sha256",
            gate_zero_unified_packed_manifest_sha256,
        ),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    forensics_artifact_sha256 = forensics.get("artifact_sha256")
    leaderboard_protocol_sha256 = forensics.get("leaderboard_protocol_sha256")
    frozen_inventory_sha256 = forensics.get("frozen_inventory_sha256")
    for name, value in (
        ("forensics_artifact_sha256", forensics_artifact_sha256),
        ("leaderboard_protocol_sha256", leaderboard_protocol_sha256),
        ("frozen_inventory_sha256", frozen_inventory_sha256),
    ):
        if not _is_sha256(value):
            raise EditingT1PanelError(f"forensics source lacks {name}")
    manifest = sidecar.manifest
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("unified_packed_manifest_sha256")
        != gate_zero_unified_packed_manifest_sha256
        or active8_admission.unified_packed_manifest_sha256
        != gate_zero_unified_packed_manifest_sha256
    ):
        raise EditingT1PanelError(
            "Gate0 validation source, semantic sidecar, and Active8 inventory "
            "name different unified packed manifests"
        )
    rows = _validate_forensics_rows(forensics, sidecar)
    (
        active8_eligible_rows,
        active8_excluded_row_count,
        active8_excluded_trace_count,
    ) = filter_active8_forensics_rows(rows, active8_admission)
    excluded_trace_ids = validate_charge_policy_exclusions(
        audit=charge_policy_audit,
        audit_file_sha256=charge_policy_audit_file_sha256,
        exclusions=charge_policy_exclusions,
        exclusions_file_sha256=charge_policy_exclusions_file_sha256,
    )
    charge_eligible_rows = tuple(
        row
        for row in active8_eligible_rows
        if (
            str(row["exact_state_ref"]["shard_sha256"]),
            int(row["exact_state_ref"]["record_index"]),
        )
        not in excluded_trace_ids
    )
    active_families = set(RINGCORE_EDITING_FAMILIES)
    eligible_rows = tuple(
        row for row in charge_eligible_rows if str(row["teacher_family"]) in active_families
    )
    operator_freeze_excluded_row_count = sum(
        str(row["teacher_family"]) in EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES
        for row in charge_eligible_rows
    )
    excluded_forensics_trace_count = len(
        {
            (
                str(row["exact_state_ref"]["shard_sha256"]),
                int(row["exact_state_ref"]["record_index"]),
            )
            for row in active8_eligible_rows
            if (
                str(row["exact_state_ref"]["shard_sha256"]),
                int(row["exact_state_ref"]["record_index"]),
            )
            in excluded_trace_ids
        }
    )
    unique = _select_unique_panels(eligible_rows)
    global_repeated, conditional_repeated = _select_repeated_panels(eligible_rows)
    source = {
        "forensics_file_sha256": forensics_file_sha256,
        "forensics_artifact_sha256": forensics_artifact_sha256,
        "leaderboard_protocol_sha256": leaderboard_protocol_sha256,
        "frozen_inventory_sha256": frozen_inventory_sha256,
        "semantic_sidecar_file_sha256": semantic_sidecar_file_sha256,
        "semantic_sidecar_manifest_file_sha256": (semantic_sidecar_manifest_file_sha256),
        "semantic_sidecar_manifest_sha256": manifest["manifest_sha256"],
        "semantic_sidecar_config_sha256": manifest["config_sha256"],
        "semantic_sidecar_provenance_sha256": (manifest["provenance_sha256"]),
        "charge_policy_audit_file_sha256": charge_policy_audit_file_sha256,
        "charge_policy_exclusions_file_sha256": (charge_policy_exclusions_file_sha256),
        "charge_policy_exclusion_payload_sha256": charge_policy_exclusions["payload_sha256"],
        "charge_policy_source_input_inventory_sha256": (
            charge_policy_exclusions["source_input_inventory_sha256"]
        ),
        "charge_policy_audit_implementation_sha256": (
            charge_policy_exclusions["charge_policy_audit_implementation_sha256"]
        ),
        "charge_policy_excluded_unique_trace_addresses": (
            charge_policy_exclusions["excluded_unique_trace_addresses"]
        ),
        **active8_t1_identity(active8_admission),
    }
    body = _artifact_body(
        source=source,
        unique=unique,
        global_repeated=global_repeated,
        conditional_repeated=conditional_repeated,
        source_rows=rows,
        source_row_count=len(rows),
        active8_eligible_row_count=len(active8_eligible_rows),
        active8_excluded_row_count=active8_excluded_row_count,
        active8_excluded_trace_count=active8_excluded_trace_count,
        charge_eligible_row_count=len(charge_eligible_rows),
        charge_excluded_row_count=(
            len(active8_eligible_rows) - len(charge_eligible_rows)
        ),
        pilot_eligible_row_count=len(eligible_rows),
        operator_freeze_excluded_row_count=operator_freeze_excluded_row_count,
        charge_excluded_forensics_trace_count=excluded_forensics_trace_count,
    )
    return {**body, "artifact_sha256": _stable_sha256(body)}


def _validate_row_ref(
    value: object,
    *,
    source_rows: tuple[Mapping[str, Any], ...],
) -> None:
    payload = _require_exact_fields(
        value,
        _ROW_REF_FIELDS,
        name="T1 source-row reference",
    )
    index = payload["source_row_index"]
    digest = payload["source_row_sha256"]
    if (
        type(index) is not int
        or not 0 <= index < len(source_rows)
        or not _is_sha256(digest)
        or source_rows[index].get("row_sha256") != digest
    ):
        raise EditingT1PanelError("T1 source-row reference does not resolve exactly")


def validate_editing_t1_panel(
    panel: Mapping[str, Any],
    *,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_unified_packed_manifest_sha256: str,
) -> None:
    """Validate the artifact and re-derive its complete selection."""

    payload = _require_exact_fields(
        panel,
        _TOP_LEVEL_FIELDS,
        name="T1 panel",
    )
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if (
        payload["schema"] != EDITING_T1_PANEL_SCHEMA
        or payload["schema_version"] != EDITING_T1_PANEL_SCHEMA_VERSION
        or payload["status"] != EDITING_T1_PANEL_STATUS
        or payload["training_authorized"] is not False
        or not _is_sha256(payload["artifact_sha256"])
        or payload["artifact_sha256"] != _stable_sha256(body)
    ):
        raise EditingT1PanelError("T1 panel schema, status, or self-hash is invalid")
    _require_exact_fields(payload["source"], _SOURCE_FIELDS, name="T1 source")
    _require_exact_fields(
        payload["selection"],
        _SELECTION_FIELDS,
        name="T1 selection",
    )
    _require_exact_fields(
        payload["architecture"],
        _ARCHITECTURE_FIELDS,
        name="T1 architecture",
    )
    thresholds = _require_exact_fields(
        payload["thresholds"],
        _THRESHOLD_FIELDS,
        name="T1 thresholds",
    )
    if any(value is not None for value in thresholds.values()):
        raise EditingT1PanelError("T1 numeric thresholds must remain explicitly unfrozen")
    source_rows = _validate_forensics_rows(forensics, sidecar)
    for family in RINGCORE_EDITING_FAMILIES:
        unique = payload["unique_state_panels"].get(family)
        conditional = payload["within_family_repeated_state_panels"].get(family)
        if not isinstance(unique, list) or not isinstance(conditional, list):
            raise EditingT1PanelError(f"T1 panel is missing family {family!r}")
        for row_ref in unique:
            _validate_row_ref(row_ref, source_rows=source_rows)
        for group in conditional:
            group = _require_exact_fields(
                group,
                _WITHIN_FAMILY_REPEATED_GROUP_FIELDS,
                name="T1 within-family repeated-state group",
            )
            if not _is_sha256(group["source_state_sha256"]):
                raise EditingT1PanelError("T1 repeated group source digest is invalid")
            if (
                type(group["distinct_teacher_successor_count"]) is not int
                or group["distinct_teacher_successor_count"] <= 1
                or not isinstance(group["rows"], list)
                or len(group["rows"]) < 2
            ):
                raise EditingT1PanelError("T1 repeated-state group is not multi-successor")
            for row_ref in group["rows"]:
                _validate_row_ref(row_ref, source_rows=source_rows)
    global_repeated = payload["global_repeated_state_panel"]
    if not isinstance(global_repeated, list):
        raise EditingT1PanelError("T1 global repeated-state panel must be a list")
    for group in global_repeated:
        group = _require_exact_fields(
            group,
            _GLOBAL_REPEATED_GROUP_FIELDS,
            name="T1 global repeated-state group",
        )
        if (
            not _is_sha256(group["source_state_sha256"])
            or type(group["distinct_teacher_successor_count"]) is not int
            or group["distinct_teacher_successor_count"] <= 1
            or type(group["distinct_teacher_family_count"]) is not int
            or group["distinct_teacher_family_count"] <= 0
            or not isinstance(group["rows"], list)
            or len(group["rows"]) < 2
        ):
            raise EditingT1PanelError("T1 global repeated-state group is invalid")
        for row_ref in group["rows"]:
            _validate_row_ref(row_ref, source_rows=source_rows)
    if set(payload["unique_state_panels"]) != set(RINGCORE_EDITING_FAMILIES) or set(
        payload["within_family_repeated_state_panels"]
    ) != set(RINGCORE_EDITING_FAMILIES):
        raise EditingT1PanelError("T1 panel family maps have extra or missing families")

    expected = build_editing_t1_panel(
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        sidecar=sidecar,
        semantic_sidecar_file_sha256=semantic_sidecar_file_sha256,
        semantic_sidecar_manifest_file_sha256=(semantic_sidecar_manifest_file_sha256),
        charge_policy_audit=charge_policy_audit,
        charge_policy_audit_file_sha256=charge_policy_audit_file_sha256,
        charge_policy_exclusions=charge_policy_exclusions,
        charge_policy_exclusions_file_sha256=(charge_policy_exclusions_file_sha256),
        active8_admission=active8_admission,
        gate_zero_unified_packed_manifest_sha256=(
            gate_zero_unified_packed_manifest_sha256
        ),
    )
    if dict(payload) != expected:
        raise EditingT1PanelError("T1 panel does not equal the deterministic source re-derivation")


def load_editing_t1_panel(
    path: Path,
    *,
    expected_artifact_sha256: str,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_unified_packed_manifest_sha256: str,
    max_file_bytes: int = 1 << 24,
) -> Mapping[str, Any]:
    """Load a self-hashed T1 artifact and bind it to every exact source."""

    if not _is_sha256(expected_artifact_sha256):
        raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256")
    source = Path(path)
    if not source.is_file() or source.stat().st_size > max_file_bytes:
        raise EditingT1PanelError("T1 panel is absent or exceeds its bound")
    try:
        panel = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1PanelError("T1 panel is invalid JSON") from error
    validate_editing_t1_panel(
        panel,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        sidecar=sidecar,
        semantic_sidecar_file_sha256=semantic_sidecar_file_sha256,
        semantic_sidecar_manifest_file_sha256=(semantic_sidecar_manifest_file_sha256),
        charge_policy_audit=charge_policy_audit,
        charge_policy_audit_file_sha256=charge_policy_audit_file_sha256,
        charge_policy_exclusions=charge_policy_exclusions,
        charge_policy_exclusions_file_sha256=(charge_policy_exclusions_file_sha256),
        active8_admission=active8_admission,
        gate_zero_unified_packed_manifest_sha256=(
            gate_zero_unified_packed_manifest_sha256
        ),
    )
    if panel["artifact_sha256"] != expected_artifact_sha256:
        raise EditingT1PanelError("T1 panel artifact hash mismatch")
    return MappingProxyType(panel)


def selected_source_rows(
    panel: Mapping[str, Any],
    forensics: Mapping[str, Any],
    *,
    family: str,
    panel_kind: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve one exact family/role panel after artifact validation."""

    rows = tuple(forensics["rows"])
    if panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND:
        if family != EDITING_T1_GLOBAL_FAMILY_SELECTOR:
            raise ValueError(
                "the primary global repeated-state panel requires family='all_families'"
            )
        references = [
            reference
            for group in panel["global_repeated_state_panel"]
            for reference in group["rows"]
        ]
        if not references:
            raise EditingT1PanelError("primary global repeated-state panel is empty")
    elif family not in RINGCORE_EDITING_FAMILIES:
        raise ValueError(f"unknown T1 family {family!r}")
    elif panel_kind == EDITING_T1_UNIQUE_PANEL_KIND:
        references = panel["unique_state_panels"][family]
    elif panel_kind == EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND:
        references = [
            reference
            for group in panel["within_family_repeated_state_panels"][family]
            for reference in group["rows"]
        ]
        if not references:
            raise EditingT1PanelError(
                f"family {family!r} has no observed repeated multi-successor state"
            )
    else:
        raise ValueError(f"unknown T1 panel kind {panel_kind!r}")
    selected = tuple(rows[int(reference["source_row_index"])] for reference in references)
    if panel_kind == EDITING_T1_UNIQUE_PANEL_KIND and len(
        {row["source_state_sha256"] for row in selected}
    ) != len(selected):
        raise EditingT1PanelError("resolved unique-state panel contains an exact-state duplicate")
    return selected


def write_editing_t1_panel(
    panel: Mapping[str, Any],
    path: Path,
) -> None:
    """Atomically freeze canonical JSON without overwriting a different file."""

    content = (
        json.dumps(
            panel,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise EditingT1PanelError(
                    "immutable T1 panel already exists with different bytes"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "ACTIVE8_T1_IDENTITY_FIELDS",
    "EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES",
    "EDITING_T1_GLOBAL_FAMILY_SELECTOR",
    "EDITING_T1_GLOBAL_REPEATED_PANEL_KIND",
    "EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES",
    "EDITING_T1_PANEL_SCHEMA",
    "EDITING_T1_PANEL_SCHEMA_VERSION",
    "EDITING_T1_PANEL_STATUS",
    "EDITING_T1_REQUIRED_SCOPES",
    "EDITING_T1_SELECTION_ALGORITHM",
    "EDITING_T1_SUPPORT_TIME",
    "EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY",
    "EDITING_T1_UNIQUE_PANEL_KIND",
    "EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND",
    "EditingT1PanelError",
    "active8_t1_identity",
    "build_editing_t1_panel",
    "filter_active8_forensics_rows",
    "load_editing_t1_panel",
    "selected_source_rows",
    "validate_charge_policy_exclusions",
    "validate_editing_t1_panel",
    "write_editing_t1_panel",
]
