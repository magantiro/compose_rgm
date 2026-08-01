"""Fail-closed local refinement of a completed Editing-V2 split assignment.

The initial prospective greedy assignment can miss a per-lane balance gate even
when a nearby component-complete assignment satisfies every frozen constraint.
This module repairs only that bounded failure mode.  It validates the complete
schema-v2 parent assignment, freezes ``controller_validation`` and
``final_test``, and permits each hard component to move at most once between
``train`` and ``validation``.

Search decisions use exact integer arithmetic.  The primary score is the
weighted squared excess beyond the already-frozen split gates.  The secondary
score is the original weighted squared target-residual objective.  A refined
artifact is emitted only when all original gates pass.  Nothing in this module
grants corpus or training authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_split_assignment import (
    SPLIT_ASSIGNMENT_SCHEMA,
    SPLIT_ASSIGNMENT_STATUS,
    SPLIT_ASSIGNMENT_VERSION,
    EditingV2SplitAssignmentError,
    validate_candidate_source_stream,
    validate_split_assignment_policy,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES, canonical_sha256

REFINEMENT_SCHEMA = "compose.editing_v2_split_assignment_refinement"
REFINEMENT_SCHEMA_VERSION = 1
REFINEMENT_STATUS = "PASS_NO_CORPUS_AUTHORITY"
REFINEMENT_ALGORITHM_ID = "sealed_roles_exact_gate_penalty_descent_v1"

_MOVABLE_ROLES = ("train", "validation")
_FROZEN_ROLES = ("controller_validation", "final_test")
_STATIC_DOWNSTREAM_BLOCKERS = {
    "physical_lane_shards.not_built",
    "whole_trace_active8.not_built",
    "semantic_sampling_sidecar.not_built",
}
_ASSIGNMENT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "split_assignment_selected",
    "split_assignment_authorized",
    "blockers",
    "policy",
    "policy_sha256",
    "source_stream",
    "source_census_sha256",
    "source_component_inventory_sha256",
    "partition_roles",
    "observed_lanes",
    "total_mass_units",
    "total_mass_units_by_lane",
    "role_summaries",
    "lane_absolute_ratio_deviations",
    "gate_results",
    "component_assignments",
    "candidate_resolutions",
    "candidate_resolution_stream_sha256",
    "sealed_final_test_policy",
    "assignment_sha256",
}
_COMPONENT_FIELDS = {
    "component_id",
    "assigned_role",
    "candidate_count",
    "mass_units",
    "mass_units_by_lane",
}
_RESOLUTION_FIELDS = {
    "candidate_id",
    "component_id",
    "status",
    "assigned_role",
    "source_endpoint_role",
    "target_endpoint_role",
}
_REFINEMENT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "split_assignment_authorized",
    "parent_assignment",
    "refinement_algorithm",
    "objective",
    "moves",
    "refined_assignment",
    "refinement_sha256",
}
_SHA256_HEX = frozenset("0123456789abcdef")


class EditingV2SplitAssignmentRefinementError(ValueError):
    """A split assignment cannot be validated or safely refined."""


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
        raise EditingV2SplitAssignmentRefinementError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _require_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise EditingV2SplitAssignmentRefinementError(
            f"{field} must be a positive integer"
        )
    return value


def _require_exact_mapping(
    value: object,
    *,
    fields: set[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EditingV2SplitAssignmentRefinementError(f"{field} fields disagree")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_ratio(value: Mapping[str, Any], *, field: str) -> tuple[int, int]:
    if set(value) != {"numerator", "denominator"}:
        raise EditingV2SplitAssignmentRefinementError(f"{field} fields disagree")
    return (
        _require_positive_int(value["numerator"], field=f"{field}.numerator"),
        _require_positive_int(value["denominator"], field=f"{field}.denominator"),
    )


def _deviation(
    assigned: int, total: int, numerator: int, denominator: int
) -> dict[str, int]:
    return {
        "numerator": abs(assigned * denominator - total * numerator),
        "denominator": total * denominator,
    }


def _within_limit(deviation: Mapping[str, int], limit: tuple[int, int]) -> bool:
    return (
        int(deviation["numerator"]) * limit[1]
        <= int(deviation["denominator"]) * limit[0]
    )


def _ratio_cost_scales(numerators: Mapping[str, int]) -> dict[str, int]:
    common = math.lcm(*(int(value) ** 2 for value in numerators.values()))
    return {role: common // (int(numerators[role]) ** 2) for role in PARTITION_ROLES}


def _summaries_and_gates(
    components: list[dict[str, Any]],
    *,
    lanes: tuple[str, ...],
    total_mass: int,
    total_lane_mass: Mapping[str, int],
    policy: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, int]]],
    dict[str, bool],
]:
    assigned_components = {role: 0 for role in PARTITION_ROLES}
    assigned_candidates = {role: 0 for role in PARTITION_ROLES}
    assigned_mass = {role: 0 for role in PARTITION_ROLES}
    assigned_lane_mass = {role: {lane: 0 for lane in lanes} for role in PARTITION_ROLES}
    for component in components:
        role = component["assigned_role"]
        assigned_components[role] += 1
        assigned_candidates[role] += int(component["candidate_count"])
        assigned_mass[role] += int(component["mass_units"])
        for lane, mass in component["mass_units_by_lane"].items():
            assigned_lane_mass[role][lane] += int(mass)

    ratios = policy["target_mass_ratios"]
    denominator = int(ratios["denominator"])
    numerators = {role: int(ratios["numerators"][role]) for role in PARTITION_ROLES}
    overall_deviations = {
        role: _deviation(assigned_mass[role], total_mass, numerators[role], denominator)
        for role in PARTITION_ROLES
    }
    lane_deviations = {
        lane: {
            role: _deviation(
                assigned_lane_mass[role][lane],
                int(total_lane_mass[lane]),
                numerators[role],
                denominator,
            )
            for role in PARTITION_ROLES
        }
        for lane in lanes
    }
    gates = policy["gates"]
    overall_limit = _normalized_ratio(
        gates["maximum_overall_absolute_ratio_deviation"],
        field="gates.maximum_overall_absolute_ratio_deviation",
    )
    lane_limit = _normalized_ratio(
        gates["maximum_per_lane_absolute_ratio_deviation"],
        field="gates.maximum_per_lane_absolute_ratio_deviation",
    )
    gate_results = {
        "all_roles_nonempty": all(
            assigned_components[role] > 0 for role in PARTITION_ROLES
        ),
        "all_observed_lanes_present_in_every_role": all(
            assigned_lane_mass[role][lane] > 0
            for role in PARTITION_ROLES
            for lane in lanes
        ),
        "overall_mass_ratio_within_limit": all(
            _within_limit(deviation, overall_limit)
            for deviation in overall_deviations.values()
        ),
        "per_lane_mass_ratio_within_limit": all(
            _within_limit(deviation, lane_limit)
            for role_deviations in lane_deviations.values()
            for deviation in role_deviations.values()
        ),
    }
    summaries = {
        role: {
            "components": assigned_components[role],
            "candidates": assigned_candidates[role],
            "mass_units": assigned_mass[role],
            "mass_units_by_lane": assigned_lane_mass[role],
            "overall_absolute_ratio_deviation": overall_deviations[role],
        }
        for role in PARTITION_ROLES
    }
    return summaries, lane_deviations, gate_results


def _validate_components_and_resolutions(
    assignment: Mapping[str, Any],
    *,
    lanes: tuple[str, ...],
) -> list[dict[str, Any]]:
    raw_components = assignment["component_assignments"]
    if not isinstance(raw_components, list) or not raw_components:
        raise EditingV2SplitAssignmentRefinementError(
            "component assignments must be nonempty"
        )
    components: list[dict[str, Any]] = []
    component_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_components):
        component = _require_exact_mapping(
            raw,
            fields=_COMPONENT_FIELDS,
            field=f"component_assignments[{index}]",
        )
        component_id = component["component_id"]
        role = component["assigned_role"]
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_by_id
            or role not in PARTITION_ROLES
        ):
            raise EditingV2SplitAssignmentRefinementError(
                f"component assignment {index} has duplicate or invalid identity"
            )
        candidate_count = _require_positive_int(
            component["candidate_count"],
            field=f"component {component_id}.candidate_count",
        )
        mass = _require_positive_int(
            component["mass_units"], field=f"component {component_id}.mass_units"
        )
        lane_mass = component["mass_units_by_lane"]
        if (
            not isinstance(lane_mass, Mapping)
            or not lane_mass
            or not set(lane_mass) <= set(lanes)
        ):
            raise EditingV2SplitAssignmentRefinementError(
                f"component {component_id} has invalid lane-mass fields"
            )
        normalized_lane_mass = {
            str(lane): _require_positive_int(
                value, field=f"component {component_id}.mass_units_by_lane.{lane}"
            )
            for lane, value in sorted(lane_mass.items())
        }
        if sum(normalized_lane_mass.values()) != mass:
            raise EditingV2SplitAssignmentRefinementError(
                f"component {component_id} lane mass does not equal total mass"
            )
        normalized = {
            "component_id": component_id,
            "assigned_role": role,
            "candidate_count": candidate_count,
            "mass_units": mass,
            "mass_units_by_lane": normalized_lane_mass,
        }
        components.append(normalized)
        component_by_id[component_id] = normalized
    if [row["component_id"] for row in components] != sorted(component_by_id):
        raise EditingV2SplitAssignmentRefinementError(
            "component assignments must be sorted by component_id"
        )

    resolutions = assignment["candidate_resolutions"]
    if not isinstance(resolutions, list) or not resolutions:
        raise EditingV2SplitAssignmentRefinementError(
            "candidate resolutions must be nonempty"
        )
    if assignment["candidate_resolution_stream_sha256"] != canonical_sha256(
        resolutions
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "candidate resolution stream SHA-256 disagrees"
        )
    observed_counts: Counter[str] = Counter()
    observed_candidates: set[str] = set()
    ordered_candidate_ids: list[str] = []
    for index, raw in enumerate(resolutions):
        resolution = _require_exact_mapping(
            raw,
            fields=_RESOLUTION_FIELDS,
            field=f"candidate_resolutions[{index}]",
        )
        candidate_id = resolution["candidate_id"]
        component_id = resolution["component_id"]
        component = component_by_id.get(component_id)
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in observed_candidates
            or component is None
            or resolution["status"] != "assigned"
            or resolution["assigned_role"] != component["assigned_role"]
            or resolution["source_endpoint_role"] != component["assigned_role"]
            or resolution["target_endpoint_role"] != component["assigned_role"]
        ):
            raise EditingV2SplitAssignmentRefinementError(
                f"candidate resolution {index} is duplicate or component-inconsistent"
            )
        observed_candidates.add(candidate_id)
        ordered_candidate_ids.append(candidate_id)
        observed_counts[component_id] += 1
    if ordered_candidate_ids != sorted(ordered_candidate_ids):
        raise EditingV2SplitAssignmentRefinementError(
            "candidate resolutions must be sorted by candidate_id"
        )
    expected_counts = {
        row["component_id"]: int(row["candidate_count"]) for row in components
    }
    if dict(observed_counts) != expected_counts:
        raise EditingV2SplitAssignmentRefinementError(
            "candidate resolutions do not preserve complete hard components"
        )
    return components


def validate_split_assignment_v2(
    assignment: object,
    *,
    require_failed: bool = False,
    require_passing: bool = False,
) -> dict[str, Any]:
    """Validate the exact self-hashed schema-v2 assignment and all summaries."""

    value = _require_exact_mapping(
        assignment,
        fields=_ASSIGNMENT_FIELDS,
        field="split assignment",
    )
    supplied_sha256 = _require_sha256(
        value["assignment_sha256"], field="assignment_sha256"
    )
    body = {key: item for key, item in value.items() if key != "assignment_sha256"}
    if supplied_sha256 != canonical_sha256(body):
        raise EditingV2SplitAssignmentRefinementError("assignment SHA-256 disagrees")
    if (
        value["schema"] != SPLIT_ASSIGNMENT_SCHEMA
        or value["schema_version"] != SPLIT_ASSIGNMENT_VERSION
        or value["status"] != SPLIT_ASSIGNMENT_STATUS
        or value["training_authorized"] is not False
        or value["split_assignment_selected"] is not True
        or value["split_assignment_authorized"] is not False
        or tuple(value["partition_roles"]) != PARTITION_ROLES
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "split assignment identity, roles, or authority disagrees"
        )
    try:
        policy = validate_split_assignment_policy(value["policy"])
        source_stream = validate_candidate_source_stream(value["source_stream"])
    except EditingV2SplitAssignmentError as error:
        raise EditingV2SplitAssignmentRefinementError(
            "split assignment policy or candidate source stream is invalid"
        ) from error
    if value["policy_sha256"] != canonical_sha256(policy):
        raise EditingV2SplitAssignmentRefinementError("policy SHA-256 disagrees")
    for field in ("source_census_sha256", "source_component_inventory_sha256"):
        _require_sha256(value[field], field=field)
    lanes = value["observed_lanes"]
    if (
        not isinstance(lanes, list)
        or not lanes
        or lanes != sorted(lanes)
        or len(lanes) != len(set(lanes))
        or any(not isinstance(lane, str) or not lane for lane in lanes)
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "observed lanes must be nonempty, unique, sorted text"
        )
    lane_tuple = tuple(lanes)
    total_mass = _require_positive_int(
        value["total_mass_units"], field="total_mass_units"
    )
    total_lane_mass = value["total_mass_units_by_lane"]
    if not isinstance(total_lane_mass, Mapping) or set(total_lane_mass) != set(
        lane_tuple
    ):
        raise EditingV2SplitAssignmentRefinementError("total lane-mass fields disagree")
    normalized_total_lane_mass = {
        lane: _require_positive_int(
            total_lane_mass[lane], field=f"total_mass_units_by_lane.{lane}"
        )
        for lane in lane_tuple
    }
    if sum(normalized_total_lane_mass.values()) != total_mass:
        raise EditingV2SplitAssignmentRefinementError(
            "total per-lane mass does not equal total mass"
        )
    components = _validate_components_and_resolutions(value, lanes=lane_tuple)
    if sum(row["mass_units"] for row in components) != total_mass:
        raise EditingV2SplitAssignmentRefinementError(
            "component mass does not equal declared total mass"
        )
    component_lane_mass = {
        lane: sum(row["mass_units_by_lane"].get(lane, 0) for row in components)
        for lane in lane_tuple
    }
    if component_lane_mass != normalized_total_lane_mass:
        raise EditingV2SplitAssignmentRefinementError(
            "component lane mass does not equal declared lane totals"
        )
    if int(source_stream["nonempty_jsonl_rows"]) != len(value["candidate_resolutions"]):
        raise EditingV2SplitAssignmentRefinementError(
            "candidate source-stream row count disagrees with resolutions"
        )
    summaries, lane_deviations, gate_results = _summaries_and_gates(
        components,
        lanes=lane_tuple,
        total_mass=total_mass,
        total_lane_mass=normalized_total_lane_mass,
        policy=policy,
    )
    if value["role_summaries"] != summaries:
        raise EditingV2SplitAssignmentRefinementError("role summaries disagree")
    if value["lane_absolute_ratio_deviations"] != lane_deviations:
        raise EditingV2SplitAssignmentRefinementError("lane deviations disagree")
    if value["gate_results"] != gate_results:
        raise EditingV2SplitAssignmentRefinementError("gate results disagree")
    if value["sealed_final_test_policy"] != policy["sealed_final_test_policy"]:
        raise EditingV2SplitAssignmentRefinementError(
            "sealed final-test policy disagrees"
        )
    blockers = value["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not _STATIC_DOWNSTREAM_BLOCKERS <= set(blockers)
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "split assignment blockers are incomplete or nondeterministic"
        )
    expected_gate_blockers = {
        f"split_gate.{name}" for name, passed in gate_results.items() if not passed
    }
    observed_gate_blockers = {
        blocker for blocker in blockers if blocker.startswith("split_gate.")
    }
    if observed_gate_blockers != expected_gate_blockers:
        raise EditingV2SplitAssignmentRefinementError(
            "split-gate blockers disagree with recomputed gates"
        )
    all_pass = all(gate_results.values())
    if require_failed and all_pass:
        raise EditingV2SplitAssignmentRefinementError(
            "refinement requires a failed parent assignment"
        )
    if require_passing and not all_pass:
        raise EditingV2SplitAssignmentRefinementError(
            "refined assignment does not pass every frozen gate"
        )
    return dict(value)


def _state_from_components(
    components: list[dict[str, Any]], lanes: tuple[str, ...]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    assigned_mass = {role: 0 for role in PARTITION_ROLES}
    assigned_lane_mass = {role: {lane: 0 for lane in lanes} for role in PARTITION_ROLES}
    for component in components:
        role = component["assigned_role"]
        assigned_mass[role] += int(component["mass_units"])
        for lane, mass in component["mass_units_by_lane"].items():
            assigned_lane_mass[role][lane] += int(mass)
    return assigned_mass, assigned_lane_mass


def _objective_cost(
    assigned_mass: Mapping[str, int],
    assigned_lane_mass: Mapping[str, Mapping[str, int]],
    *,
    total_mass: int,
    total_lane_mass: Mapping[str, int],
    policy: Mapping[str, Any],
) -> int:
    ratios = policy["target_mass_ratios"]
    denominator = int(ratios["denominator"])
    numerators = {role: int(ratios["numerators"][role]) for role in PARTITION_ROLES}
    scales = _ratio_cost_scales(numerators)
    overall_weight = int(policy["balance_axes"]["overall_mass_units_weight"])
    lane_weight = int(policy["balance_axes"]["per_lane_mass_units_weight"])
    cost = 0
    for role in PARTITION_ROLES:
        residual = assigned_mass[role] * denominator - total_mass * numerators[role]
        cost += overall_weight * residual * residual * scales[role]
        for lane, lane_total in total_lane_mass.items():
            residual = (
                assigned_lane_mass[role][lane] * denominator
                - int(lane_total) * numerators[role]
            )
            cost += lane_weight * residual * residual * scales[role]
    return cost


def _gate_penalty(
    assigned_mass: Mapping[str, int],
    assigned_lane_mass: Mapping[str, Mapping[str, int]],
    *,
    total_mass: int,
    total_lane_mass: Mapping[str, int],
    policy: Mapping[str, Any],
) -> int:
    """Return exact weighted squared residual only beyond each frozen gate."""

    ratios = policy["target_mass_ratios"]
    denominator = int(ratios["denominator"])
    numerators = {role: int(ratios["numerators"][role]) for role in PARTITION_ROLES}
    scales = _ratio_cost_scales(numerators)
    gates = policy["gates"]
    overall_limit = _normalized_ratio(
        gates["maximum_overall_absolute_ratio_deviation"],
        field="gates.maximum_overall_absolute_ratio_deviation",
    )
    lane_limit = _normalized_ratio(
        gates["maximum_per_lane_absolute_ratio_deviation"],
        field="gates.maximum_per_lane_absolute_ratio_deviation",
    )
    overall_weight = int(policy["balance_axes"]["overall_mass_units_weight"])
    lane_weight = int(policy["balance_axes"]["per_lane_mass_units_weight"])

    def excess(
        assigned: int,
        total: int,
        target_numerator: int,
        limit: tuple[int, int],
    ) -> int:
        residual = abs(assigned * denominator - total * target_numerator)
        return max(residual * limit[1] - total * denominator * limit[0], 0)

    penalty = 0
    for role in PARTITION_ROLES:
        value = excess(assigned_mass[role], total_mass, numerators[role], overall_limit)
        penalty += overall_weight * value * value * scales[role]
        for lane, lane_total in total_lane_mass.items():
            value = excess(
                assigned_lane_mass[role][lane],
                int(lane_total),
                numerators[role],
                lane_limit,
            )
            penalty += lane_weight * value * value * scales[role]
    return penalty


def _moved_state(
    assigned_mass: Mapping[str, int],
    assigned_lane_mass: Mapping[str, Mapping[str, int]],
    *,
    component: Mapping[str, Any],
    destination: str,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    source = str(component["assigned_role"])
    next_mass = dict(assigned_mass)
    next_lane_mass = {
        role: dict(lane_mass) for role, lane_mass in assigned_lane_mass.items()
    }
    next_mass[source] -= int(component["mass_units"])
    next_mass[destination] += int(component["mass_units"])
    for lane, mass in component["mass_units_by_lane"].items():
        next_lane_mass[source][lane] -= int(mass)
        next_lane_mass[destination][lane] += int(mass)
    return next_mass, next_lane_mass


def _algorithm_identity() -> dict[str, Any]:
    specification = {
        "algorithm_id": REFINEMENT_ALGORITHM_ID,
        "arithmetic": "exact_integer_only",
        "frozen_roles": list(_FROZEN_ROLES),
        "movable_roles": list(_MOVABLE_ROLES),
        "move": "one_complete_hard_component_crosses_train_validation_at_most_once",
        "selection": [
            "minimum_next_exact_weighted_squared_gate_excess",
            "minimum_next_original_frozen_weighted_squared_target_residual",
            "sha256_split_salt_algorithm_component_source_destination",
        ],
        "progress_requirement": "gate_penalty_strictly_decreases",
        "success": "all_original_gate_results_true",
        "failure": "no_unmoved_strictly_improving_component",
        "authority": "none",
    }
    source_path = Path(__file__).resolve()
    return {
        **specification,
        "algorithm_spec_sha256": canonical_sha256(specification),
        "implementation_file": "src/compose_v4/data/editing_v2_split_assignment_refinement.py",
        "implementation_sha256": _file_sha256(source_path),
    }


def _tie_break(
    *, policy: Mapping[str, Any], component: Mapping[str, Any], destination: str
) -> str:
    source = component["assigned_role"]
    payload = (
        f"{policy['split_salt']}\0{REFINEMENT_ALGORITHM_ID}\0"
        f"{component['component_id']}\0{source}\0{destination}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rebuild_assignment(
    parent: Mapping[str, Any],
    *,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    lanes = tuple(parent["observed_lanes"])
    component_roles = {row["component_id"]: row["assigned_role"] for row in components}
    resolutions = [
        {
            **dict(row),
            "assigned_role": component_roles[row["component_id"]],
            "source_endpoint_role": component_roles[row["component_id"]],
            "target_endpoint_role": component_roles[row["component_id"]],
        }
        for row in parent["candidate_resolutions"]
    ]
    summaries, lane_deviations, gate_results = _summaries_and_gates(
        components,
        lanes=lanes,
        total_mass=int(parent["total_mass_units"]),
        total_lane_mass=parent["total_mass_units_by_lane"],
        policy=parent["policy"],
    )
    blockers = {
        blocker
        for blocker in parent["blockers"]
        if not blocker.startswith("split_gate.")
    }
    blockers.update(
        f"split_gate.{name}" for name, passed in gate_results.items() if not passed
    )
    body = {
        **{
            key: value
            for key, value in parent.items()
            if key
            not in {
                "assignment_sha256",
                "blockers",
                "role_summaries",
                "lane_absolute_ratio_deviations",
                "gate_results",
                "component_assignments",
                "candidate_resolutions",
                "candidate_resolution_stream_sha256",
            }
        },
        "blockers": sorted(blockers),
        "role_summaries": summaries,
        "lane_absolute_ratio_deviations": lane_deviations,
        "gate_results": gate_results,
        "component_assignments": sorted(
            components, key=lambda row: row["component_id"]
        ),
        "candidate_resolutions": resolutions,
        "candidate_resolution_stream_sha256": canonical_sha256(resolutions),
    }
    return {**body, "assignment_sha256": canonical_sha256(body)}


def refine_split_assignment(
    parent_assignment: Mapping[str, Any],
    *,
    parent_assignment_file_sha256: str,
) -> dict[str, Any]:
    """Refine one failed assignment or raise without publishing a partial result."""

    parent = validate_split_assignment_v2(parent_assignment, require_failed=True)
    parent_file_sha256 = _require_sha256(
        parent_assignment_file_sha256, field="parent_assignment_file_sha256"
    )
    components = [
        {
            **dict(row),
            "mass_units_by_lane": dict(row["mass_units_by_lane"]),
        }
        for row in parent["component_assignments"]
    ]
    lanes = tuple(parent["observed_lanes"])
    total_mass = int(parent["total_mass_units"])
    total_lane_mass = parent["total_mass_units_by_lane"]
    policy = parent["policy"]
    assigned_mass, assigned_lane_mass = _state_from_components(components, lanes)
    initial_mass = dict(assigned_mass)
    initial_lane_mass = {role: dict(mass) for role, mass in assigned_lane_mass.items()}
    current_penalty = _gate_penalty(
        assigned_mass,
        assigned_lane_mass,
        total_mass=total_mass,
        total_lane_mass=total_lane_mass,
        policy=policy,
    )
    initial_penalty = current_penalty
    initial_cost = _objective_cost(
        assigned_mass,
        assigned_lane_mass,
        total_mass=total_mass,
        total_lane_mass=total_lane_mass,
        policy=policy,
    )
    current_cost = initial_cost
    moved_components: set[str] = set()
    moves: list[dict[str, Any]] = []
    while current_penalty > 0:
        best: (
            tuple[
                tuple[int, int, str],
                dict[str, Any],
                str,
                dict[str, int],
                dict[str, dict[str, int]],
            ]
            | None
        ) = None
        for component in components:
            component_id = component["component_id"]
            source = component["assigned_role"]
            if component_id in moved_components or source not in _MOVABLE_ROLES:
                continue
            destination = "validation" if source == "train" else "train"
            next_mass, next_lane_mass = _moved_state(
                assigned_mass,
                assigned_lane_mass,
                component=component,
                destination=destination,
            )
            next_penalty = _gate_penalty(
                next_mass,
                next_lane_mass,
                total_mass=total_mass,
                total_lane_mass=total_lane_mass,
                policy=policy,
            )
            if next_penalty >= current_penalty:
                continue
            next_cost = _objective_cost(
                next_mass,
                next_lane_mass,
                total_mass=total_mass,
                total_lane_mass=total_lane_mass,
                policy=policy,
            )
            score = (
                next_penalty,
                next_cost,
                _tie_break(policy=policy, component=component, destination=destination),
            )
            candidate = (score, component, destination, next_mass, next_lane_mass)
            if best is None or score < best[0]:
                best = candidate
        if best is None:
            raise EditingV2SplitAssignmentRefinementError(
                "no all-gates-pass solution exists within the frozen local refinement algorithm"
            )
        score, component, destination, next_mass, next_lane_mass = best
        source = component["assigned_role"]
        move = {
            "move_index": len(moves),
            "component_id": component["component_id"],
            "from_role": source,
            "to_role": destination,
            "candidate_count": component["candidate_count"],
            "mass_units": component["mass_units"],
            "mass_units_by_lane": component["mass_units_by_lane"],
            "gate_penalty_before": current_penalty,
            "gate_penalty_after": score[0],
            "objective_cost_before": current_cost,
            "objective_cost_after": score[1],
            "tie_break_sha256": score[2],
        }
        moves.append(move)
        moved_components.add(component["component_id"])
        component["assigned_role"] = destination
        assigned_mass = next_mass
        assigned_lane_mass = next_lane_mass
        current_penalty = score[0]
        current_cost = score[1]

    for role in _FROZEN_ROLES:
        if (
            assigned_mass[role] != initial_mass[role]
            or assigned_lane_mass[role] != initial_lane_mass[role]
        ):
            raise EditingV2SplitAssignmentRefinementError(
                f"refinement mutated frozen role {role}"
            )
    refined_assignment = _rebuild_assignment(parent, components=components)
    validate_split_assignment_v2(refined_assignment, require_passing=True)
    algorithm = _algorithm_identity()
    body = {
        "schema": REFINEMENT_SCHEMA,
        "schema_version": REFINEMENT_SCHEMA_VERSION,
        "status": REFINEMENT_STATUS,
        "training_authorized": False,
        "split_assignment_authorized": False,
        "parent_assignment": {
            "file_sha256": parent_file_sha256,
            "assignment_sha256": parent["assignment_sha256"],
            "candidate_resolution_stream_sha256": parent[
                "candidate_resolution_stream_sha256"
            ],
            "source_census_sha256": parent["source_census_sha256"],
            "source_component_inventory_sha256": parent[
                "source_component_inventory_sha256"
            ],
        },
        "refinement_algorithm": algorithm,
        "objective": {
            "initial_gate_penalty": initial_penalty,
            "final_gate_penalty": current_penalty,
            "initial_target_residual_cost": initial_cost,
            "final_target_residual_cost": current_cost,
            "all_original_gates_pass": True,
        },
        "moves": moves,
        "refined_assignment": refined_assignment,
    }
    return {**body, "refinement_sha256": canonical_sha256(body)}


def refine_split_assignment_file(path: str | Path) -> dict[str, Any]:
    """Load one immutable parent assignment and bind both of its hashes."""

    source = Path(path)
    try:
        value = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2SplitAssignmentRefinementError(
            f"cannot load parent split assignment {source}"
        ) from error
    return refine_split_assignment(
        value,
        parent_assignment_file_sha256=_file_sha256(source),
    )


def validate_refinement_artifact(
    value: object,
    *,
    parent_assignment: Mapping[str, Any],
    parent_assignment_file_sha256: str,
) -> dict[str, Any]:
    """Validate an artifact by exact deterministic replay from its bound parent."""

    artifact = _require_exact_mapping(
        value, fields=_REFINEMENT_FIELDS, field="split assignment refinement"
    )
    supplied_sha256 = _require_sha256(
        artifact["refinement_sha256"], field="refinement_sha256"
    )
    body = {key: item for key, item in artifact.items() if key != "refinement_sha256"}
    if supplied_sha256 != canonical_sha256(body):
        raise EditingV2SplitAssignmentRefinementError("refinement SHA-256 disagrees")
    if (
        artifact["schema"] != REFINEMENT_SCHEMA
        or artifact["schema_version"] != REFINEMENT_SCHEMA_VERSION
        or artifact["status"] != REFINEMENT_STATUS
        or artifact["training_authorized"] is not False
        or artifact["split_assignment_authorized"] is not False
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "refinement identity or authority disagrees"
        )
    parent = _require_exact_mapping(
        artifact["parent_assignment"],
        fields={
            "file_sha256",
            "assignment_sha256",
            "candidate_resolution_stream_sha256",
            "source_census_sha256",
            "source_component_inventory_sha256",
        },
        field="parent_assignment",
    )
    for field, digest in parent.items():
        _require_sha256(digest, field=f"parent_assignment.{field}")
    algorithm = artifact["refinement_algorithm"]
    if algorithm != _algorithm_identity():
        raise EditingV2SplitAssignmentRefinementError(
            "refinement algorithm identity disagrees with this implementation"
        )
    objective = _require_exact_mapping(
        artifact["objective"],
        fields={
            "initial_gate_penalty",
            "final_gate_penalty",
            "initial_target_residual_cost",
            "final_target_residual_cost",
            "all_original_gates_pass",
        },
        field="objective",
    )
    for field in (
        "initial_gate_penalty",
        "final_gate_penalty",
        "initial_target_residual_cost",
        "final_target_residual_cost",
    ):
        if type(objective[field]) is not int or int(objective[field]) < 0:
            raise EditingV2SplitAssignmentRefinementError(
                f"objective.{field} must be a nonnegative integer"
            )
    if (
        objective["final_gate_penalty"] != 0
        or objective["all_original_gates_pass"] is not True
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "refinement objective does not establish gate passage"
        )
    moves = artifact["moves"]
    if not isinstance(moves, list) or not moves:
        raise EditingV2SplitAssignmentRefinementError(
            "successful refinement must contain at least one move"
        )
    seen: set[str] = set()
    previous_penalty = int(objective["initial_gate_penalty"])
    for index, move in enumerate(moves):
        if not isinstance(move, Mapping):
            raise EditingV2SplitAssignmentRefinementError(
                f"move {index} must be an object"
            )
        component_id = move.get("component_id")
        if (
            move.get("move_index") != index
            or not isinstance(component_id, str)
            or component_id in seen
            or (move.get("from_role"), move.get("to_role"))
            not in {("train", "validation"), ("validation", "train")}
            or move.get("gate_penalty_before") != previous_penalty
            or type(move.get("gate_penalty_after")) is not int
            or int(move["gate_penalty_after"]) >= previous_penalty
        ):
            raise EditingV2SplitAssignmentRefinementError(
                f"move {index} violates deterministic whole-component progress"
            )
        seen.add(component_id)
        previous_penalty = int(move["gate_penalty_after"])
    if previous_penalty != 0:
        raise EditingV2SplitAssignmentRefinementError(
            "move sequence does not end at zero gate penalty"
        )
    refined = validate_split_assignment_v2(
        artifact["refined_assignment"], require_passing=True
    )
    if (
        refined["source_census_sha256"] != parent["source_census_sha256"]
        or refined["source_component_inventory_sha256"]
        != parent["source_component_inventory_sha256"]
    ):
        raise EditingV2SplitAssignmentRefinementError(
            "refined assignment changed its hard-component source identity"
        )
    expected = refine_split_assignment(
        parent_assignment,
        parent_assignment_file_sha256=parent_assignment_file_sha256,
    )
    if dict(artifact) != expected:
        raise EditingV2SplitAssignmentRefinementError(
            "refinement artifact disagrees with exact deterministic parent replay"
        )
    return dict(artifact)


def write_refinement_artifact(path: str | Path, value: Mapping[str, Any]) -> bool:
    """Publish canonical bytes once; reuse exact bytes and reject collisions.

    Returns ``True`` when this call publishes the artifact and ``False`` when
    the exact bytes already exist.
    """

    target = Path(path)
    try:
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2SplitAssignmentRefinementError(
            "refinement artifact is not finite deterministic JSON"
        ) from error
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == payload:
            return False
        raise EditingV2SplitAssignmentRefinementError(
            f"immutable refinement artifact collision at {target}"
        )
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise EditingV2SplitAssignmentRefinementError(
                    f"immutable refinement artifact collision at {target}"
                )
            return False
        Path(temporary).unlink()
        temporary = None
        return True
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine one failed self-hashed Editing-V2 split assignment"
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = refine_split_assignment_file(args.parent)
    validate_refinement_artifact(
        artifact,
        parent_assignment=json.loads(args.parent.read_bytes()),
        parent_assignment_file_sha256=_file_sha256(args.parent),
    )
    write_refinement_artifact(args.output, artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "training_authorized": False,
                "refinement_sha256": artifact["refinement_sha256"],
                "refined_assignment_sha256": artifact["refined_assignment"][
                    "assignment_sha256"
                ],
                "gate_results": artifact["refined_assignment"]["gate_results"],
                "moves": len(artifact["moves"]),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "REFINEMENT_ALGORITHM_ID",
    "REFINEMENT_SCHEMA",
    "REFINEMENT_SCHEMA_VERSION",
    "REFINEMENT_STATUS",
    "EditingV2SplitAssignmentRefinementError",
    "refine_split_assignment",
    "refine_split_assignment_file",
    "validate_refinement_artifact",
    "validate_split_assignment_v2",
    "write_refinement_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
