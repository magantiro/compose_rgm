"""Deterministic streaming census of charge-policy violations in packed traces.

This audit reads the immutable exact states stored in packed editing shards. It
does not reconstruct from SMILES, invoke RDKit, execute actions, or mutate the
corpus. Every failure is addressed by packed-shard digest, shard row, trace,
progress position, and the complete versioned action record.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.charge_policy import (
    CHARGE_POLICY_VIOLATION_TYPES,
    audit_charge_policy_transition,
)
from compose_v4.data.packed_trace_store import (
    manifest_path_for,
    read_addressed_packed_shard,
)
from compose_v4.rewrite.action_codec import (
    canonical_family,
    encode_action,
    public_operator_name,
)

PACKED_CHARGE_POLICY_AUDIT_SCHEMA = "compose.data.packed_charge_policy_audit"
PACKED_CHARGE_POLICY_AUDIT_SCHEMA_VERSION = 1

_IMPLEMENTATION_SOURCE_PATHS = (
    "src/compose_v4/data/charge_policy.py",
    "src/compose_v4/data/packed_charge_policy_audit.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/chem/persistent_state_identity.py",
    "src/compose_v4/rewrite/action_codec.py",
)
_PARTITION_ORDER = {"train": 0, "validation": 1, "test": 2}
_MANIFEST_LAYER_ROOT = {
    "general_corruption": "audit_layers",
    "cycle_operations": "audit_layers",
    "mmp_analogue": "mmp_layer",
}
_MANIFEST_LAYER_ENVELOPE = {
    "general_corruption": "corruption",
    "cycle_operations": "cycle_ops",
    "mmp_analogue": "mmp_analogue",
}


class PackedChargePolicyAuditError(RuntimeError):
    """The requested census cannot establish exact packed-corpus coverage."""


@dataclass(frozen=True)
class DeclaredPackedShard:
    """One manifest-declared shard and its expected semantic lane."""

    manifest_layer: str
    envelope_layer: str
    partition: str
    relative_path: str
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def implementation_provenance(*, repo_root: Path | None = None) -> dict[str, object]:
    """Hash every source file defining decoding, identity, and this predicate."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    sources: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise PackedChargePolicyAuditError(
                f"audit implementation source is absent: {path}"
            )
        sources[relative] = _sha256(path)
    return {
        "sources": sources,
        "implementation_sha256": hashlib.sha256(
            _canonical_json_bytes(sources)
        ).hexdigest(),
    }


def resolve_unified_manifest_shards(
    manifest_path: Path,
    *,
    audit_root: Path | None = None,
    mmp_root: Path | None = None,
) -> tuple[dict[str, object], tuple[DeclaredPackedShard, ...]]:
    """Resolve every shard declared by one unified packed-corpus manifest."""

    manifest_path = Path(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PackedChargePolicyAuditError(
            f"cannot read unified packed manifest {manifest_path}: {exc}"
        ) from exc
    if payload.get("artifact") != "unified_packed_corpus_manifest":
        raise PackedChargePolicyAuditError(
            "expected artifact='unified_packed_corpus_manifest'"
        )
    layers = payload.get("layers")
    roots = payload.get("roots")
    if not isinstance(layers, dict) or not isinstance(roots, dict):
        raise PackedChargePolicyAuditError(
            "unified packed manifest must contain object-valued roots and layers"
        )

    resolved_roots: dict[str, Path] = {}
    for key, override in (
        ("audit_layers", audit_root),
        ("mmp_layer", mmp_root),
    ):
        stored = roots.get(key)
        if override is None and (not isinstance(stored, str) or not stored):
            raise PackedChargePolicyAuditError(
                f"unified packed manifest has no non-empty root {key!r}"
            )
        resolved_roots[key] = Path(override) if override is not None else Path(stored)
    declared: list[DeclaredPackedShard] = []
    seen_paths: set[Path] = set()
    for manifest_layer in sorted(layers):
        if manifest_layer not in _MANIFEST_LAYER_ROOT:
            raise PackedChargePolicyAuditError(
                f"unrecognized unified-manifest layer {manifest_layer!r}"
            )
        partitions = layers[manifest_layer]
        if not isinstance(partitions, dict):
            raise PackedChargePolicyAuditError(
                f"layer {manifest_layer!r} must map partitions to shard lists"
            )
        unexpected = set(partitions) - set(_PARTITION_ORDER)
        if unexpected:
            raise PackedChargePolicyAuditError(
                f"layer {manifest_layer!r} has unknown partitions {sorted(unexpected)}"
            )
        root_key = _MANIFEST_LAYER_ROOT[manifest_layer]
        root = resolved_roots[root_key]
        for partition in sorted(partitions, key=_PARTITION_ORDER.__getitem__):
            relative_paths = partitions[partition]
            if not isinstance(relative_paths, list):
                raise PackedChargePolicyAuditError(
                    f"{manifest_layer}/{partition} shard inventory must be a list"
                )
            for relative_path in relative_paths:
                if (
                    not isinstance(relative_path, str)
                    or not relative_path
                    or Path(relative_path).is_absolute()
                    or ".." in Path(relative_path).parts
                ):
                    raise PackedChargePolicyAuditError(
                        f"unsafe packed-shard relative path {relative_path!r}"
                    )
                path = root / relative_path
                resolved = path.resolve()
                if resolved in seen_paths:
                    raise PackedChargePolicyAuditError(
                        f"packed shard is declared more than once: {path}"
                    )
                if not path.is_file():
                    raise PackedChargePolicyAuditError(
                        f"manifest-declared packed shard is absent: {path}"
                    )
                seen_paths.add(resolved)
                declared.append(
                    DeclaredPackedShard(
                        manifest_layer=manifest_layer,
                        envelope_layer=_MANIFEST_LAYER_ENVELOPE[manifest_layer],
                        partition=partition,
                        relative_path=relative_path,
                        path=path,
                    )
                )

    expected_shards = (payload.get("totals") or {}).get("shards")
    if type(expected_shards) is not int or expected_shards != len(declared):
        raise PackedChargePolicyAuditError(
            "unified packed manifest shard total disagrees with its inventory: "
            f"{expected_shards!r} != {len(declared)}"
        )
    return payload, tuple(declared)


def audit_packed_charge_policy(
    shards: Iterable[DeclaredPackedShard],
    *,
    max_examples: int = 32,
    source_manifest_path: Path | None = None,
    source_manifest: dict[str, object] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Stream exact packed states and return one deterministic audit report."""

    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative integer")
    ordered_shards = tuple(
        sorted(
            shards,
            key=lambda shard: (
                shard.manifest_layer,
                _PARTITION_ORDER.get(shard.partition, 99),
                shard.relative_path,
            ),
        )
    )
    if not ordered_shards:
        raise PackedChargePolicyAuditError("at least one packed shard is required")

    counts = Counter(
        shards=0,
        entries=0,
        states=0,
        transitions=0,
        traces_with_violations=0,
        transitions_with_violations=0,
        violation_events=0,
    )
    violations_by_type = Counter(
        {violation: 0 for violation in CHARGE_POLICY_VIOLATION_TYPES}
    )
    unique_violating_traces_by_type = Counter(
        {violation: 0 for violation in CHARGE_POLICY_VIOLATION_TYPES}
    )
    transitions_by_group = Counter(
        formal_charge_coordinate_mutation=0,
        protected_charged_center_mutation=0,
    )
    unique_traces_by_group = Counter(
        formal_charge_coordinate_mutation=0,
        protected_charged_center_mutation=0,
    )
    coordinate_mutation_transitions = Counter(
        formal_charge_created=0,
        formal_charge_deleted=0,
        formal_charge_nonzero_value_changed=0,
    )
    coordinate_mutation_slot_events = Counter(
        formal_charge_created=0,
        formal_charge_deleted=0,
        formal_charge_nonzero_value_changed=0,
    )
    violating_traces_by_lane: Counter[str] = Counter()
    violating_transitions_by_family: Counter[str] = Counter()
    example_candidates: dict[tuple[object, ...], dict[str, object]] = {}
    shard_reports: list[dict[str, object]] = []
    exclusion_shards: list[dict[str, object]] = []

    for declared in ordered_shards:
        shard_path = declared.path
        sidecar_path = manifest_path_for(shard_path)
        if not sidecar_path.is_file():
            raise PackedChargePolicyAuditError(
                f"packed shard has no manifest: {shard_path}"
            )
        shard_counts = Counter(
            entries=0,
            states=0,
            transitions=0,
            traces_with_violations=0,
            transitions_with_violations=0,
            violation_events=0,
        )
        shard_violations = Counter(
            {violation: 0 for violation in CHARGE_POLICY_VIOLATION_TYPES}
        )
        shard_unique_traces_by_type = Counter(
            {violation: 0 for violation in CHARGE_POLICY_VIOLATION_TYPES}
        )
        shard_exclusions: list[dict[str, object]] = []
        observed_digest: str | None = None

        for addressed in read_addressed_packed_shard(shard_path):
            address = addressed.address
            if address.layer != declared.envelope_layer:
                raise PackedChargePolicyAuditError(
                    f"{shard_path} row {address.entry_index} declares layer "
                    f"{address.layer!r}, expected {declared.envelope_layer!r}"
                )
            if address.partition != declared.partition:
                raise PackedChargePolicyAuditError(
                    f"{shard_path} row {address.entry_index} declares partition "
                    f"{address.partition!r}, expected {declared.partition!r}"
                )
            if observed_digest is None:
                observed_digest = address.packed_shard_content_sha256
            elif observed_digest != address.packed_shard_content_sha256:
                raise PackedChargePolicyAuditError(
                    f"{shard_path} produced inconsistent packed-shard digests"
                )

            path = addressed.path
            shard_counts["entries"] += 1
            shard_counts["states"] += path.path_length + 1
            shard_counts["transitions"] += path.path_length
            trace_has_violation = False
            trace_violation_types: set[str] = set()
            trace_violation_groups: set[str] = set()
            for progress_index, step in enumerate(addressed.trace.steps):
                before = path.state_at(progress_index)
                after = path.state_at(progress_index + 1)
                transition = audit_charge_policy_transition(before, after)
                if transition.preserved:
                    continue
                trace_has_violation = True
                shard_counts["transitions_with_violations"] += 1
                family = canonical_family(step.rule_name)
                violating_transitions_by_family[family] += 1
                trace_violation_types.update(transition.violation_types)
                for violation in transition.violation_types:
                    shard_violations[violation] += 1
                    violations_by_type[violation] += 1
                    shard_counts["violation_events"] += 1
                if transition.formal_charge_coordinate_mutated:
                    group = "formal_charge_coordinate_mutation"
                    transitions_by_group[group] += 1
                    trace_violation_groups.add(group)
                if transition.protected_charged_center_mutated:
                    group = "protected_charged_center_mutation"
                    transitions_by_group[group] += 1
                    trace_violation_groups.add(group)
                for kind, slots in (
                    ("formal_charge_created", transition.formal_charge_created_slots),
                    ("formal_charge_deleted", transition.formal_charge_deleted_slots),
                    (
                        "formal_charge_nonzero_value_changed",
                        transition.formal_charge_value_changed_slots,
                    ),
                ):
                    if slots:
                        coordinate_mutation_transitions[kind] += 1
                        coordinate_mutation_slot_events[kind] += len(slots)

                example_bucket = (
                    declared.manifest_layer,
                    address.partition,
                    family,
                    transition.violation_types,
                )
                if example_bucket not in example_candidates:
                    action = encode_action(step.rule_name, step.action)
                    example_candidates[example_bucket] = {
                        "address": {
                            "manifest_layer": declared.manifest_layer,
                            "layer": address.layer,
                            "partition": address.partition,
                            "packed_shard_name": address.packed_shard_name,
                            "packed_shard_content_sha256": (
                                address.packed_shard_content_sha256
                            ),
                            "entry_index": address.entry_index,
                            "trace_id": address.trace_id,
                            "source_key": address.source_key,
                            "target_key": address.target_key,
                            "path_length": address.path_length,
                            "progress_index": progress_index,
                        },
                        "action": {
                            **action,
                            "public_operator_name": public_operator_name(family),
                        },
                        "before_persistent_state_sha256": (
                            persistent_slot_state_sha256(before)
                        ),
                        "after_persistent_state_sha256": (
                            persistent_slot_state_sha256(after)
                        ),
                        "differences": transition.to_json(),
                    }
            if trace_has_violation:
                shard_counts["traces_with_violations"] += 1
                violating_traces_by_lane[
                    f"{declared.manifest_layer}/{declared.partition}"
                ] += 1
                for violation in trace_violation_types:
                    unique_violating_traces_by_type[violation] += 1
                    shard_unique_traces_by_type[violation] += 1
                for group in trace_violation_groups:
                    unique_traces_by_group[group] += 1
                shard_exclusions.append(
                    {
                        "entry_index": address.entry_index,
                        "trace_id": address.trace_id,
                        "violation_type_mask": sum(
                            1 << index
                            for index, violation in enumerate(
                                CHARGE_POLICY_VIOLATION_TYPES
                            )
                            if violation in trace_violation_types
                        ),
                    }
                )

        manifest_payload = json.loads(sidecar_path.read_text())
        expected_entries = manifest_payload.get("entries")
        expected_states = manifest_payload.get("states")
        if expected_entries != shard_counts["entries"]:
            raise PackedChargePolicyAuditError(
                f"{shard_path} manifest entries {expected_entries!r} != "
                f"audited {shard_counts['entries']}"
            )
        if expected_states != shard_counts["states"]:
            raise PackedChargePolicyAuditError(
                f"{shard_path} manifest states {expected_states!r} != "
                f"audited {shard_counts['states']}"
            )
        if observed_digest is None:
            observed_digest = _sha256(shard_path)

        shard_report = {
            "manifest_layer": declared.manifest_layer,
            "envelope_layer": declared.envelope_layer,
            "partition": declared.partition,
            "relative_path": declared.relative_path,
            "packed_shard_content_sha256": observed_digest,
            "packed_manifest_sha256": _sha256(sidecar_path),
            "counts": dict(shard_counts),
            "violating_transitions_by_type": dict(shard_violations),
            "unique_violating_traces_by_type": dict(
                shard_unique_traces_by_type
            ),
            "violating_entry_indices": [
                entry["entry_index"] for entry in shard_exclusions
            ],
        }
        if shard_exclusions:
            exclusion_shards.append(
                {
                    "manifest_layer": declared.manifest_layer,
                    "envelope_layer": declared.envelope_layer,
                    "partition": declared.partition,
                    "relative_path": declared.relative_path,
                    "packed_shard_content_sha256": observed_digest,
                    "excluded_entry_count": len(shard_exclusions),
                    "violating_entry_indices": [
                        entry["entry_index"] for entry in shard_exclusions
                    ],
                    "entries": shard_exclusions,
                }
            )
        shard_reports.append(shard_report)
        counts.update(shard_counts)
        counts["shards"] += 1
        if progress_callback is not None:
            progress_callback(shard_report)

    inventory = [
        {
            key: shard[key]
            for key in (
                "manifest_layer",
                "envelope_layer",
                "partition",
                "relative_path",
                "packed_shard_content_sha256",
                "packed_manifest_sha256",
            )
        }
        for shard in shard_reports
    ]
    source_manifest_record: dict[str, object] | None = None
    if source_manifest_path is not None:
        source_manifest_path = Path(source_manifest_path)
        source_manifest_record = {
            "path": str(source_manifest_path),
            "sha256": _sha256(source_manifest_path),
            "manifest_checksum": (
                None
                if source_manifest is None
                else source_manifest.get("manifest_checksum")
            ),
        }

    input_inventory_sha256 = hashlib.sha256(
        _canonical_json_bytes(inventory)
    ).hexdigest()
    implementation = implementation_provenance()
    exclusion_payload_body = {
        "schema": "compose.data.packed_charge_policy_exclusions",
        "schema_version": 1,
        "selection": "exclude_whole_trace_if_any_consecutive_state_pair_violates",
        "violation_type_bit_order": list(CHARGE_POLICY_VIOLATION_TYPES),
        "source_input_inventory_sha256": input_inventory_sha256,
        "charge_policy_audit_implementation_sha256": implementation[
            "implementation_sha256"
        ],
        "excluded_unique_trace_addresses": counts["traces_with_violations"],
        "shards": exclusion_shards,
    }
    exclusion_payload = {
        **exclusion_payload_body,
        "payload_sha256": hashlib.sha256(
            _canonical_json_bytes(exclusion_payload_body)
        ).hexdigest(),
    }

    examples = sorted(
        example_candidates.values(),
        key=lambda example: (
            str(example["address"]["manifest_layer"]),
            _PARTITION_ORDER.get(str(example["address"]["partition"]), 99),
            str(example["address"]["packed_shard_name"]),
            int(example["address"]["entry_index"]),
            int(example["address"]["progress_index"]),
        ),
    )[:max_examples]
    report = {
        "schema": PACKED_CHARGE_POLICY_AUDIT_SCHEMA,
        "schema_version": PACKED_CHARGE_POLICY_AUDIT_SCHEMA_VERSION,
        "status": (
            "PASS"
            if counts["transitions_with_violations"] == 0
            else "FAIL_CHARGE_POLICY_VIOLATIONS"
        ),
        "policy": {
            "formal_charge_array": "exactly_preserved",
            "source_charged_center_element": "exactly_preserved",
            "source_charged_center_implicit_h": "exactly_preserved",
            "source_charged_center_bond_row": "exactly_preserved",
            "violation_types": list(CHARGE_POLICY_VIOLATION_TYPES),
        },
        "method": {
            "state_source": "exact_packed_consecutive_states",
            "canonical_smiles_used": False,
            "rdkit_used": False,
            "executor_replay_used": False,
            "corpus_mutated": False,
        },
        "source_manifest": source_manifest_record,
        "implementation": implementation,
        "input_inventory_sha256": input_inventory_sha256,
        "counts": dict(counts),
        "violating_transitions_by_type": dict(violations_by_type),
        "unique_violating_traces_by_type": dict(
            unique_violating_traces_by_type
        ),
        "violation_groups": {
            "formal_charge_coordinate_mutation": {
                "definition": (
                    "one or more exact formal-charge coordinates changed; "
                    "creation/deletion/nonzero-value subtypes can overlap within "
                    "one transition"
                ),
                "transitions": transitions_by_group[
                    "formal_charge_coordinate_mutation"
                ],
                "unique_traces": unique_traces_by_group[
                    "formal_charge_coordinate_mutation"
                ],
                "transition_subtypes": dict(coordinate_mutation_transitions),
                "slot_events_by_subtype": dict(coordinate_mutation_slot_events),
            },
            "protected_charged_center_mutation": {
                "definition": (
                    "source-charged slot changed element, implicit H, or bond row"
                ),
                "transitions": transitions_by_group[
                    "protected_charged_center_mutation"
                ],
                "unique_traces": unique_traces_by_group[
                    "protected_charged_center_mutation"
                ],
            },
            "unique_trace_counts_note": (
                "headline traces_with_violations counts each exact packed "
                "(shard digest, entry index) once; per-reason trace counts overlap"
            ),
        },
        "violating_traces_by_lane": dict(sorted(violating_traces_by_lane.items())),
        "violating_transitions_by_model_family": dict(
            sorted(violating_transitions_by_family.items())
        ),
        "max_examples": max_examples,
        "example_selection": (
            "first_exact_address_per_"
            "(manifest_layer,partition,model_family,violation_type_tuple)"
        ),
        "examples": examples,
        "shards": shard_reports,
        "exclusion_payload": exclusion_payload,
    }

    if source_manifest is not None:
        expected_totals = source_manifest.get("totals") or {}
        for field in ("shards", "entries", "states"):
            expected = expected_totals.get(field)
            if expected != counts[field]:
                raise PackedChargePolicyAuditError(
                    f"unified manifest total {field}={expected!r} != "
                    f"audited {counts[field]}"
                )
    return report


__all__ = [
    "PACKED_CHARGE_POLICY_AUDIT_SCHEMA",
    "PACKED_CHARGE_POLICY_AUDIT_SCHEMA_VERSION",
    "DeclaredPackedShard",
    "PackedChargePolicyAuditError",
    "audit_packed_charge_policy",
    "implementation_provenance",
    "resolve_unified_manifest_shards",
]
