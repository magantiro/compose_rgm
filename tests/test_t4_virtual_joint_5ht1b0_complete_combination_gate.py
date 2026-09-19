from types import SimpleNamespace

import numpy as np

from compose_v4.control.virtual_joint_region_proposer import VirtualJointRegionBudgets
from tools.t4_virtual_joint_5ht1b0_complete_combination_forensics import (
    _primitive_atom_event_lower_bound,
)
from tools.t4_virtual_joint_5ht1b0_complete_combination_gate import (
    _capacity_abstention_reasons,
    _deduplicate_union,
    _required_particle_count,
)


def _proposal(endpoint_key: str, score: float, program_id: str):
    step = SimpleNamespace(constituent_key=f"constituent-{program_id}")
    return SimpleNamespace(
        endpoint_key=endpoint_key,
        log_probability=score,
        steps=(step,),
        program_id=program_id,
    )


def test_gate_derives_minimum_five_particles_from_frozen_budgets() -> None:
    assert _required_particle_count(VirtualJointRegionBudgets()) == (17_296, 5)


def test_union_deduplicates_endpoints_by_deterministic_rank() -> None:
    lower = _proposal("shared", -2.0, "lower")
    higher = _proposal("shared", -1.0, "higher")
    unique = _proposal("unique", -3.0, "unique")
    rows = _deduplicate_union(
        [
            {"particle_index": 0, "batch": SimpleNamespace(proposals=(lower, unique))},
            {"particle_index": 1, "batch": SimpleNamespace(proposals=(higher,))},
        ]
    )
    assert {row["proposal"].endpoint_key for row in rows} == {"shared", "unique"}
    assert (
        next(
            row["proposal"].program_id
            for row in rows
            if row["proposal"].endpoint_key == "shared"
        )
        == "higher"
    )


def test_capacity_abstention_reports_every_loss_mode() -> None:
    telemetry = {
        "combination_particle_planning_capacity_abstention": 1,
        "combination_particle_unvalidated_combinations": 4,
        "combination_particle_unattempted_targets": 2,
        "per_realization_expansion_cap_abstentions": 1,
    }
    assert _capacity_abstention_reasons(telemetry) == [
        "combination_particle_planning_capacity_abstention",
        "per_realization_expansion_cap_abstentions",
        "unattempted_targets",
        "unvalidated_combinations",
    ]


def test_atom_event_lower_bound_counts_birth_delete_and_restate() -> None:
    source = SimpleNamespace(
        atom_types=np.array([2, 2, 0, 3]),
        formal_charges=np.array([0, 0, 0, 0]),
        implicit_h_counts=np.array([3, 2, 0, 1]),
    )
    target = SimpleNamespace(
        atom_types=np.array([2, 0, 4, 3]),
        formal_charges=np.array([0, 0, 0, 1]),
        implicit_h_counts=np.array([3, 0, 1, 0]),
    )
    assert _primitive_atom_event_lower_bound(source, target) == {
        "births": 1,
        "deletions": 1,
        "live_atom_restatements": 1,
        "primitive_lower_bound": 3,
    }
