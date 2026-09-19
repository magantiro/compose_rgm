"""Deterministic four-role assignment of indivisible Editing-V2 components.

The input census defines hard leakage components. This module assigns complete
components while balancing total progress-state mass and each evidence lane.
It does not resolve the census's external membership or relationship blockers,
build packed shards, or authorize training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    SOURCE_STREAM_SCHEMA,
    SOURCE_STREAM_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_split_census import (
    PARTITION_ROLES,
    canonical_sha256,
    validate_split_component_census,
)

SPLIT_ASSIGNMENT_POLICY_SCHEMA = "compose.editing_v2_split_assignment_policy"
SPLIT_ASSIGNMENT_POLICY_VERSION = 1
SPLIT_ASSIGNMENT_SCHEMA = "compose.editing_v2_split_assignment"
SPLIT_ASSIGNMENT_VERSION = 2
SPLIT_ASSIGNMENT_STATUS = "PROPOSED_ASSIGNMENT_NO_CORPUS_AUTHORITY"
ASSIGNMENT_ALGORITHM = "largest_component_exact_deficit_greedy_v1"

_POLICY_FIELDS = {
    "schema",
    "schema_version",
    "policy_id",
    "partition_roles",
    "target_mass_ratios",
    "balance_axes",
    "gates",
    "assignment_algorithm",
    "split_salt",
    "sealed_final_test_policy",
}
_SOURCE_STREAM_FIELDS = {
    "schema",
    "schema_version",
    "nonempty_jsonl_rows",
    "candidate_materialization",
    "provenance_registry",
    "candidate_audit_ledger",
    "split_candidates",
    "source_stream_sha256",
}
_CANDIDATE_MATERIALIZATION_FIELDS = {
    "manifest_file_sha256",
    "manifest_sha256",
    "rows_file_sha256",
    "rows_semantic_sha256",
    "address_stream_sha256",
}
_PROVENANCE_REGISTRY_FIELDS = {"file_sha256", "registry_sha256"}
_CANDIDATE_AUDIT_LEDGER_FIELDS = {
    "file_sha256",
    "semantic_sha256",
    "rows_file_sha256",
    "rows_sha256",
}
_SPLIT_CANDIDATE_FIELDS = {"file_sha256", "semantic_sha256"}
_SHA256_HEX = frozenset("0123456789abcdef")


class EditingV2SplitAssignmentError(ValueError):
    """A four-role assignment cannot be constructed or verified."""


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
        raise EditingV2SplitAssignmentError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_exact_hash_mapping(
    value: object,
    *,
    fields: set[str],
    field: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EditingV2SplitAssignmentError(f"{field} fields disagree")
    return {name: _require_sha256(value[name], field=f"{field}.{name}") for name in sorted(fields)}


def validate_candidate_source_stream(
    source_stream: object,
    *,
    expected_candidate_materialization: Mapping[str, Any] | None = None,
    expected_nonempty_rows: int | None = None,
) -> dict[str, Any]:
    """Validate the exact candidate-provenance bridge source identity."""

    if not isinstance(source_stream, Mapping) or set(source_stream) != _SOURCE_STREAM_FIELDS:
        raise EditingV2SplitAssignmentError(
            "candidate provenance source-stream fields disagree with the current schema"
        )
    normalized = json.loads(_canonical_json_bytes(source_stream))
    rows = normalized["nonempty_jsonl_rows"]
    if (
        normalized["schema"] != SOURCE_STREAM_SCHEMA
        or normalized["schema_version"] != SOURCE_STREAM_SCHEMA_VERSION
        or type(rows) is not int
        or rows <= 0
    ):
        raise EditingV2SplitAssignmentError(
            "candidate provenance source-stream identity or row count disagrees"
        )
    candidate_identity = _require_exact_hash_mapping(
        normalized["candidate_materialization"],
        fields=_CANDIDATE_MATERIALIZATION_FIELDS,
        field="candidate provenance source-stream candidate_materialization",
    )
    _require_exact_hash_mapping(
        normalized["provenance_registry"],
        fields=_PROVENANCE_REGISTRY_FIELDS,
        field="candidate provenance source-stream provenance_registry",
    )
    _require_exact_hash_mapping(
        normalized["candidate_audit_ledger"],
        fields=_CANDIDATE_AUDIT_LEDGER_FIELDS,
        field="candidate provenance source-stream candidate_audit_ledger",
    )
    _require_exact_hash_mapping(
        normalized["split_candidates"],
        fields=_SPLIT_CANDIDATE_FIELDS,
        field="candidate provenance source-stream split_candidates",
    )
    supplied_sha256 = _require_sha256(
        normalized["source_stream_sha256"],
        field="candidate provenance source-stream source_stream_sha256",
    )
    stream_body = {key: value for key, value in normalized.items() if key != "source_stream_sha256"}
    if supplied_sha256 != canonical_sha256(stream_body):
        raise EditingV2SplitAssignmentError("candidate provenance source-stream SHA-256 disagrees")
    if expected_candidate_materialization is not None and candidate_identity != dict(
        expected_candidate_materialization
    ):
        raise EditingV2SplitAssignmentError(
            "candidate provenance source-stream candidate materialization disagrees"
        )
    if expected_nonempty_rows is not None and rows != expected_nonempty_rows:
        raise EditingV2SplitAssignmentError(
            "candidate provenance source-stream row count disagrees"
        )
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2SplitAssignmentError(
            "split assignment is not finite deterministic JSON"
        ) from error


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise EditingV2SplitAssignmentError(f"{field} must be a positive integer")
    return value


def _normalized_ratio(
    value: object,
    *,
    field: str,
) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise EditingV2SplitAssignmentError(f"{field} must contain numerator and denominator")
    numerator = _positive_int(value["numerator"], field=f"{field}.numerator")
    denominator = _positive_int(value["denominator"], field=f"{field}.denominator")
    if numerator > denominator:
        raise EditingV2SplitAssignmentError(f"{field} must be at most one")
    return numerator, denominator


def validate_split_assignment_policy(policy: object) -> dict[str, Any]:
    """Validate the complete prospective split decision."""

    if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
        raise EditingV2SplitAssignmentError(
            "split-assignment policy fields disagree with schema version 1"
        )
    normalized = json.loads(_canonical_json_bytes(policy))
    if (
        normalized["schema"] != SPLIT_ASSIGNMENT_POLICY_SCHEMA
        or normalized["schema_version"] != SPLIT_ASSIGNMENT_POLICY_VERSION
        or tuple(normalized["partition_roles"]) != PARTITION_ROLES
        or normalized["assignment_algorithm"] != ASSIGNMENT_ALGORITHM
    ):
        raise EditingV2SplitAssignmentError(
            "split-assignment policy identity or roles are unsupported"
        )
    ratios = normalized["target_mass_ratios"]
    if not isinstance(ratios, Mapping) or set(ratios) != {
        "denominator",
        "numerators",
    }:
        raise EditingV2SplitAssignmentError("target_mass_ratios fields disagree")
    denominator = _positive_int(
        ratios["denominator"],
        field="target_mass_ratios.denominator",
    )
    numerators = ratios["numerators"]
    if not isinstance(numerators, Mapping) or set(numerators) != set(PARTITION_ROLES):
        raise EditingV2SplitAssignmentError(
            "target mass-ratio numerators must contain exactly the four roles"
        )
    values = [
        _positive_int(numerators[role], field=f"target_mass_ratios.{role}")
        for role in PARTITION_ROLES
    ]
    if sum(values) != denominator:
        raise EditingV2SplitAssignmentError("target mass ratios must sum to one")
    normalized["target_mass_ratios"]["numerators"] = {
        role: int(numerators[role]) for role in PARTITION_ROLES
    }
    axes = normalized["balance_axes"]
    if not isinstance(axes, Mapping) or set(axes) != {
        "overall_mass_units_weight",
        "per_lane_mass_units_weight",
    }:
        raise EditingV2SplitAssignmentError("balance_axes fields disagree")
    for name, value in axes.items():
        _positive_int(value, field=f"balance_axes.{name}")
    gates = normalized["gates"]
    if not isinstance(gates, Mapping) or set(gates) != {
        "maximum_overall_absolute_ratio_deviation",
        "maximum_per_lane_absolute_ratio_deviation",
        "all_roles_nonempty",
        "all_observed_lanes_present_in_every_role",
    }:
        raise EditingV2SplitAssignmentError("split-assignment gates fields disagree")
    _normalized_ratio(
        gates["maximum_overall_absolute_ratio_deviation"],
        field="gates.maximum_overall_absolute_ratio_deviation",
    )
    _normalized_ratio(
        gates["maximum_per_lane_absolute_ratio_deviation"],
        field="gates.maximum_per_lane_absolute_ratio_deviation",
    )
    if (
        gates["all_roles_nonempty"] is not True
        or gates["all_observed_lanes_present_in_every_role"] is not True
    ):
        raise EditingV2SplitAssignmentError("role and lane non-emptiness gates cannot be weakened")
    for field in ("policy_id", "split_salt", "sealed_final_test_policy"):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise EditingV2SplitAssignmentError(f"{field} must be nonempty text")
    return normalized


def load_split_assignment_policy(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2SplitAssignmentError(
            f"cannot load split-assignment policy {path}"
        ) from error
    return validate_split_assignment_policy(value)


def _tie_break(split_salt: str, component_id: str, role: str) -> str:
    return hashlib.sha256(f"{split_salt}\0{component_id}\0{role}".encode()).hexdigest()


def _component_rows(census: Mapping[str, Any]) -> list[dict[str, Any]]:
    components = census["hard_component_census"]["components"]
    if not isinstance(components, list) or len(components) < len(PARTITION_ROLES):
        raise EditingV2SplitAssignmentError("at least four hard components are required")
    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            raise EditingV2SplitAssignmentError("hard component must be an object")
        candidate_ids = tuple(component.get("candidate_ids") or ())
        if (
            not candidate_ids
            or len(candidate_ids) != len(set(candidate_ids))
            or seen_candidates.intersection(candidate_ids)
        ):
            raise EditingV2SplitAssignmentError(
                "hard components must contain disjoint candidate identities"
            )
        seen_candidates.update(candidate_ids)
        lane_mass = component.get("mass_units_by_lane")
        if not isinstance(lane_mass, Mapping) or not lane_mass:
            raise EditingV2SplitAssignmentError(
                "hard component lacks per-lane mass needed for balanced assignment"
            )
        mass = _positive_int(component.get("mass_units"), field="component.mass_units")
        if sum(int(value) for value in lane_mass.values()) != mass:
            raise EditingV2SplitAssignmentError("component per-lane mass does not equal total mass")
        rows.append(
            {
                "component_id": str(component["component_id"]),
                "candidate_ids": list(candidate_ids),
                "candidate_count": int(component["candidate_count"]),
                "mass_units": mass,
                "mass_units_by_lane": {
                    str(lane): int(value) for lane, value in sorted(lane_mass.items())
                },
            }
        )
    return rows


def _ratio_cost_scales(numerators: Mapping[str, int]) -> dict[str, int]:
    squares = [int(value) ** 2 for value in numerators.values()]
    common = math.lcm(*squares)
    return {role: common // (int(numerators[role]) ** 2) for role in PARTITION_ROLES}


def _axis_delta(
    *,
    assigned: int,
    added: int,
    total: int,
    target_numerator: int,
    target_denominator: int,
    cost_scale: int,
) -> int:
    old_residual = assigned * target_denominator - total * target_numerator
    new_residual = (assigned + added) * target_denominator - total * target_numerator
    return (new_residual * new_residual - old_residual * old_residual) * cost_scale


def _assignment_cost_delta(
    *,
    role: str,
    component: Mapping[str, Any],
    assigned_mass: Mapping[str, int],
    assigned_lane_mass: Mapping[str, Mapping[str, int]],
    total_mass: int,
    total_lane_mass: Mapping[str, int],
    numerators: Mapping[str, int],
    denominator: int,
    cost_scales: Mapping[str, int],
    overall_weight: int,
    lane_weight: int,
) -> int:
    delta = overall_weight * _axis_delta(
        assigned=assigned_mass[role],
        added=int(component["mass_units"]),
        total=total_mass,
        target_numerator=int(numerators[role]),
        target_denominator=denominator,
        cost_scale=cost_scales[role],
    )
    for lane, added in component["mass_units_by_lane"].items():
        delta += lane_weight * _axis_delta(
            assigned=assigned_lane_mass[role].get(lane, 0),
            added=int(added),
            total=total_lane_mass[lane],
            target_numerator=int(numerators[role]),
            target_denominator=denominator,
            cost_scale=cost_scales[role],
        )
    return delta


def _deviation(
    assigned: int,
    total: int,
    numerator: int,
    denominator: int,
) -> dict[str, int]:
    return {
        "numerator": abs(assigned * denominator - total * numerator),
        "denominator": total * denominator,
    }


def _within_limit(deviation: Mapping[str, int], limit: tuple[int, int]) -> bool:
    return int(deviation["numerator"]) * limit[1] <= int(deviation["denominator"]) * limit[0]


def build_split_assignment(
    census: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign each complete hard component to exactly one frozen role."""

    validate_split_component_census(census)
    if census.get("census_structural_complete") is not True or census.get("invalid_rows"):
        raise EditingV2SplitAssignmentError(
            "split assignment requires a structurally complete census"
        )
    normalized_policy = validate_split_assignment_policy(policy)
    components = _component_rows(census)
    source_stream = validate_candidate_source_stream(
        census.get("source_stream"),
        expected_nonempty_rows=int(census["input_summary"]["valid_vertices"]),
    )
    ratios = normalized_policy["target_mass_ratios"]
    denominator = int(ratios["denominator"])
    numerators = {role: int(ratios["numerators"][role]) for role in PARTITION_ROLES}
    cost_scales = _ratio_cost_scales(numerators)
    total_mass = sum(component["mass_units"] for component in components)
    lanes = sorted({lane for component in components for lane in component["mass_units_by_lane"]})
    total_lane_mass = {
        lane: sum(component["mass_units_by_lane"].get(lane, 0) for component in components)
        for lane in lanes
    }
    assigned_mass = {role: 0 for role in PARTITION_ROLES}
    assigned_lane_mass = {role: {lane: 0 for lane in lanes} for role in PARTITION_ROLES}
    assigned_components: dict[str, list[dict[str, Any]]] = {role: [] for role in PARTITION_ROLES}
    ordered = sorted(
        components,
        key=lambda component: (
            -int(component["mass_units"]),
            -int(component["candidate_count"]),
            _tie_break(
                normalized_policy["split_salt"],
                component["component_id"],
                "component_order",
            ),
        ),
    )
    overall_weight = int(normalized_policy["balance_axes"]["overall_mass_units_weight"])
    lane_weight = int(normalized_policy["balance_axes"]["per_lane_mass_units_weight"])
    for component in ordered:
        role = min(
            PARTITION_ROLES,
            key=lambda candidate_role: (
                _assignment_cost_delta(
                    role=candidate_role,
                    component=component,
                    assigned_mass=assigned_mass,
                    assigned_lane_mass=assigned_lane_mass,
                    total_mass=total_mass,
                    total_lane_mass=total_lane_mass,
                    numerators=numerators,
                    denominator=denominator,
                    cost_scales=cost_scales,
                    overall_weight=overall_weight,
                    lane_weight=lane_weight,
                ),
                _tie_break(
                    normalized_policy["split_salt"],
                    component["component_id"],
                    candidate_role,
                ),
            ),
        )
        assigned_components[role].append(component)
        assigned_mass[role] += int(component["mass_units"])
        for lane, mass in component["mass_units_by_lane"].items():
            assigned_lane_mass[role][lane] += int(mass)

    empty_roles = [role for role in PARTITION_ROLES if not assigned_components[role]]
    if empty_roles:
        raise EditingV2SplitAssignmentError(f"greedy split left empty roles: {empty_roles}")

    component_assignments = sorted(
        (
            {
                "component_id": component["component_id"],
                "assigned_role": role,
                "candidate_count": component["candidate_count"],
                "mass_units": component["mass_units"],
                "mass_units_by_lane": component["mass_units_by_lane"],
            }
            for role in PARTITION_ROLES
            for component in assigned_components[role]
        ),
        key=lambda row: row["component_id"],
    )
    candidate_resolutions = sorted(
        (
            {
                "candidate_id": candidate_id,
                "component_id": component["component_id"],
                "status": "assigned",
                "assigned_role": role,
                "source_endpoint_role": role,
                "target_endpoint_role": role,
            }
            for role in PARTITION_ROLES
            for component in assigned_components[role]
            for candidate_id in component["candidate_ids"]
        ),
        key=lambda row: row["candidate_id"],
    )
    overall_deviations = {
        role: _deviation(
            assigned_mass[role],
            total_mass,
            numerators[role],
            denominator,
        )
        for role in PARTITION_ROLES
    }
    lane_deviations = {
        lane: {
            role: _deviation(
                assigned_lane_mass[role][lane],
                total_lane_mass[lane],
                numerators[role],
                denominator,
            )
            for role in PARTITION_ROLES
        }
        for lane in lanes
    }
    gates = normalized_policy["gates"]
    overall_limit = _normalized_ratio(
        gates["maximum_overall_absolute_ratio_deviation"],
        field="gates.maximum_overall_absolute_ratio_deviation",
    )
    lane_limit = _normalized_ratio(
        gates["maximum_per_lane_absolute_ratio_deviation"],
        field="gates.maximum_per_lane_absolute_ratio_deviation",
    )
    gate_results = {
        "all_roles_nonempty": all(assigned_components[role] for role in PARTITION_ROLES),
        "all_observed_lanes_present_in_every_role": all(
            assigned_lane_mass[role][lane] > 0 for role in PARTITION_ROLES for lane in lanes
        ),
        "overall_mass_ratio_within_limit": all(
            _within_limit(deviation, overall_limit) for deviation in overall_deviations.values()
        ),
        "per_lane_mass_ratio_within_limit": all(
            _within_limit(deviation, lane_limit)
            for role_deviations in lane_deviations.values()
            for deviation in role_deviations.values()
        ),
    }
    blockers = [
        blocker
        for blocker in census["blockers"]
        if blocker != "split_assignment.not_performed_by_component_census"
    ]
    blockers.extend(f"split_gate.{name}" for name, passed in gate_results.items() if not passed)
    blockers.extend(
        [
            "physical_lane_shards.not_built",
            "whole_trace_active8.not_built",
            "semantic_sampling_sidecar.not_built",
        ]
    )
    summaries = {
        role: {
            "components": len(assigned_components[role]),
            "candidates": sum(
                int(component["candidate_count"]) for component in assigned_components[role]
            ),
            "mass_units": assigned_mass[role],
            "mass_units_by_lane": assigned_lane_mass[role],
            "overall_absolute_ratio_deviation": overall_deviations[role],
        }
        for role in PARTITION_ROLES
    }
    body = {
        "schema": SPLIT_ASSIGNMENT_SCHEMA,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
        "status": SPLIT_ASSIGNMENT_STATUS,
        "training_authorized": False,
        "split_assignment_selected": True,
        "split_assignment_authorized": False,
        "blockers": sorted(set(blockers)),
        "policy": normalized_policy,
        "policy_sha256": canonical_sha256(normalized_policy),
        "source_stream": source_stream,
        "source_census_sha256": census["census_sha256"],
        "source_component_inventory_sha256": census["hard_component_census"][
            "component_inventory_sha256"
        ],
        "partition_roles": list(PARTITION_ROLES),
        "observed_lanes": lanes,
        "total_mass_units": total_mass,
        "total_mass_units_by_lane": total_lane_mass,
        "role_summaries": summaries,
        "lane_absolute_ratio_deviations": lane_deviations,
        "gate_results": gate_results,
        "component_assignments": component_assignments,
        "candidate_resolutions": candidate_resolutions,
        "candidate_resolution_stream_sha256": canonical_sha256(candidate_resolutions),
        "sealed_final_test_policy": normalized_policy["sealed_final_test_policy"],
    }
    return {**body, "assignment_sha256": canonical_sha256(body)}


__all__ = [
    "ASSIGNMENT_ALGORITHM",
    "SPLIT_ASSIGNMENT_POLICY_SCHEMA",
    "SPLIT_ASSIGNMENT_POLICY_VERSION",
    "SPLIT_ASSIGNMENT_SCHEMA",
    "SPLIT_ASSIGNMENT_STATUS",
    "SPLIT_ASSIGNMENT_VERSION",
    "EditingV2SplitAssignmentError",
    "build_split_assignment",
    "load_split_assignment_policy",
    "validate_candidate_source_stream",
    "validate_split_assignment_policy",
]
