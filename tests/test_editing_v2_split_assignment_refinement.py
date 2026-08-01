from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.data import editing_v2_split_assignment_refinement as refinement
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    SOURCE_STREAM_SCHEMA,
    SOURCE_STREAM_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_split_assignment import (
    SPLIT_ASSIGNMENT_SCHEMA,
    SPLIT_ASSIGNMENT_STATUS,
    SPLIT_ASSIGNMENT_VERSION,
    load_split_assignment_policy,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "editing_v2_split_assignment_policy_v1.json"
SHA = "a" * 64
LANES = (
    "linker_positional_topology_analogue",
    "observed_local_analogue",
    "operator_aware_real_endpoint",
    "real_endpoint_multistep_path",
    "reversible_synthetic_walk",
)


def _source_stream(rows: int) -> dict:
    body = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": rows,
        "candidate_materialization": {
            "manifest_file_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "rows_file_sha256": "3" * 64,
            "rows_semantic_sha256": "4" * 64,
            "address_stream_sha256": "5" * 64,
        },
        "provenance_registry": {
            "file_sha256": "6" * 64,
            "registry_sha256": "7" * 64,
        },
        "candidate_audit_ledger": {
            "file_sha256": "8" * 64,
            "semantic_sha256": "9" * 64,
            "rows_file_sha256": "b" * 64,
            "rows_sha256": "c" * 64,
        },
        "split_candidates": {
            "file_sha256": "d" * 64,
            "semantic_sha256": "e" * 64,
        },
    }
    return {**body, "source_stream_sha256": canonical_sha256(body)}


def _component(
    component_id: str,
    role: str,
    lane_mass: tuple[int, int, int, int, int],
) -> dict:
    return {
        "component_id": component_id,
        "assigned_role": role,
        "candidate_count": 1,
        "mass_units": sum(lane_mass),
        "mass_units_by_lane": {
            lane: mass for lane, mass in zip(LANES, lane_mass, strict=True) if mass > 0
        },
    }


def _failed_observed_pattern(*, freeze_failure: bool = False) -> dict:
    """Compact the observed 695,638-mass failure into six hard components."""

    helpful_role = "controller_validation" if freeze_failure else "validation"
    components = sorted(
        [
            _component(
                "component-controller",
                "controller_validation",
                (862, 806, 773, 24441, 16276),
            ),
            _component("component-final", "final_test", (1061, 954, 785, 17399, 16618)),
            _component(
                "component-helpful-multistep",
                helpful_role,
                (192, 72, 306, 8851, 1530),
            ),
            _component(
                "component-train-compensation",
                "train",
                (200, 100, 160, 700, 840),
            ),
            _component(
                "component-train-remainder",
                "train",
                (23139, 21220, 20208, 206999, 304071),
            ),
            _component(
                "component-validation-remainder",
                "validation",
                (490, 541, 585, 10973, 14486),
            ),
        ],
        key=lambda row: row["component_id"],
    )
    resolutions = sorted(
        [
            {
                "candidate_id": f"candidate-{component['component_id']}",
                "component_id": component["component_id"],
                "status": "assigned",
                "assigned_role": component["assigned_role"],
                "source_endpoint_role": component["assigned_role"],
                "target_endpoint_role": component["assigned_role"],
            }
            for component in components
        ],
        key=lambda row: row["candidate_id"],
    )
    policy = load_split_assignment_policy(POLICY_PATH)
    total_lane_mass = {
        lane: sum(
            component["mass_units_by_lane"].get(lane, 0) for component in components
        )
        for lane in LANES
    }
    total_mass = sum(total_lane_mass.values())
    summaries, deviations, gates = refinement._summaries_and_gates(
        components,
        lanes=LANES,
        total_mass=total_mass,
        total_lane_mass=total_lane_mass,
        policy=policy,
    )
    blockers = [
        "physical_lane_shards.not_built",
        "semantic_sampling_sidecar.not_built",
        "whole_trace_active8.not_built",
    ]
    blockers.extend(
        f"split_gate.{name}" for name, passed in gates.items() if not passed
    )
    body = {
        "schema": SPLIT_ASSIGNMENT_SCHEMA,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
        "status": SPLIT_ASSIGNMENT_STATUS,
        "training_authorized": False,
        "split_assignment_selected": True,
        "split_assignment_authorized": False,
        "blockers": sorted(blockers),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "source_stream": _source_stream(len(resolutions)),
        "source_census_sha256": SHA,
        "source_component_inventory_sha256": "f" * 64,
        "partition_roles": list(PARTITION_ROLES),
        "observed_lanes": list(LANES),
        "total_mass_units": total_mass,
        "total_mass_units_by_lane": total_lane_mass,
        "role_summaries": summaries,
        "lane_absolute_ratio_deviations": deviations,
        "gate_results": gates,
        "component_assignments": components,
        "candidate_resolutions": resolutions,
        "candidate_resolution_stream_sha256": canonical_sha256(resolutions),
        "sealed_final_test_policy": policy["sealed_final_test_policy"],
    }
    return {**body, "assignment_sha256": canonical_sha256(body)}


def test_refinement_repairs_observed_pattern_and_freezes_held_roles(
    tmp_path: Path,
) -> None:
    parent = _failed_observed_pattern()
    first = refinement.refine_split_assignment(
        parent, parent_assignment_file_sha256="0" * 64
    )
    second = refinement.refine_split_assignment(
        parent, parent_assignment_file_sha256="0" * 64
    )
    assert first == second
    assert first["moves"][0]["component_id"] == "component-helpful-multistep"
    assert all(first["refined_assignment"]["gate_results"].values())
    assert first["objective"]["final_gate_penalty"] == 0
    assert first["training_authorized"] is False
    assert first["split_assignment_authorized"] is False
    for role in ("controller_validation", "final_test"):
        assert (
            first["refined_assignment"]["role_summaries"][role]
            == parent["role_summaries"][role]
        )
    assert all(
        move["from_role"] in {"train", "validation"}
        and move["to_role"] in {"train", "validation"}
        for move in first["moves"]
    )
    assert len({move["component_id"] for move in first["moves"]}) == len(first["moves"])
    refinement.validate_refinement_artifact(
        first,
        parent_assignment=parent,
        parent_assignment_file_sha256="0" * 64,
    )
    output = tmp_path / "refinement.json"
    assert refinement.write_refinement_artifact(output, first) is True
    first_bytes = output.read_bytes()
    assert refinement.write_refinement_artifact(output, first) is False
    assert output.read_bytes() == first_bytes
    collision = copy.deepcopy(first)
    collision["status"] = "different"
    with pytest.raises(
        refinement.EditingV2SplitAssignmentRefinementError,
        match="immutable refinement artifact collision",
    ):
        refinement.write_refinement_artifact(output, collision)
    assert output.read_bytes() == first_bytes


def test_rehashed_forged_refinement_fails_exact_parent_replay() -> None:
    parent = _failed_observed_pattern()
    artifact = refinement.refine_split_assignment(
        parent, parent_assignment_file_sha256="0" * 64
    )
    forged = copy.deepcopy(artifact)
    forged["moves"][0]["tie_break_sha256"] = "0" * 64
    body = {key: value for key, value in forged.items() if key != "refinement_sha256"}
    forged["refinement_sha256"] = canonical_sha256(body)
    with pytest.raises(
        refinement.EditingV2SplitAssignmentRefinementError,
        match="exact deterministic parent replay",
    ):
        refinement.validate_refinement_artifact(
            forged,
            parent_assignment=parent,
            parent_assignment_file_sha256="0" * 64,
        )


def test_parent_tampering_and_frozen_role_infeasibility_fail_closed() -> None:
    parent = _failed_observed_pattern()
    tampered = copy.deepcopy(parent)
    tampered["component_assignments"][0]["mass_units"] += 1
    body = {key: value for key, value in tampered.items() if key != "assignment_sha256"}
    tampered["assignment_sha256"] = canonical_sha256(body)
    with pytest.raises(
        refinement.EditingV2SplitAssignmentRefinementError,
        match="lane mass does not equal|summaries disagree",
    ):
        refinement.refine_split_assignment(
            tampered, parent_assignment_file_sha256="0" * 64
        )

    infeasible = _failed_observed_pattern(freeze_failure=True)
    with pytest.raises(
        refinement.EditingV2SplitAssignmentRefinementError,
        match="no all-gates-pass solution",
    ):
        refinement.refine_split_assignment(
            infeasible, parent_assignment_file_sha256="0" * 64
        )


def test_parent_file_hash_is_bound() -> None:
    parent = _failed_observed_pattern()
    artifact = refinement.refine_split_assignment(
        parent, parent_assignment_file_sha256="0" * 64
    )
    with pytest.raises(
        refinement.EditingV2SplitAssignmentRefinementError,
        match="exact deterministic parent replay",
    ):
        refinement.validate_refinement_artifact(
            artifact,
            parent_assignment=parent,
            parent_assignment_file_sha256="1" * 64,
        )
