import numpy as np
import pytest

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.t4_cold_start_allocation import (
    PROTONATION_EXPERT,
    SCHEMA_VERSION,
    cold_start_allocation_v1,
)
from compose_v4.experiments.t4_integrated_route_fiber import select_batch

BASE_EXPERTS = ("shallow", "anchored_replacement", "route_complete_region")


def _candidate(smiles, expert, *, band=None, rank=None):
    row = {
        "smiles": smiles,
        "proposal_lane": expert,
        "proposal_experts": [expert],
        "features": np.asarray([0.0, 1.0]),
        "fingerprint": {1, 2},
    }
    if band is not None:
        row["realized_primitive_band"] = band
        row["route_proposal_rank"] = rank
    return row


def test_base_plan_preserves_one_exploration_slot():
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=8,
        available_experts=BASE_EXPERTS,
    )

    assert plan.schema_version == SCHEMA_VERSION
    assert plan.active
    assert plan.route_scale_floor_counts == {"small": 1, "medium": 1, "large": 3}
    assert plan.expert_floor_counts == {"shallow": 1, "anchored_replacement": 1}
    assert plan.exploration_slots == 1


def test_nonempty_optional_protonation_pool_uses_final_slot():
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=8,
        available_experts=(*BASE_EXPERTS, PROTONATION_EXPERT),
    )

    assert plan.expert_floor_counts == {
        "shallow": 1,
        "anchored_replacement": 1,
        PROTONATION_EXPERT: 1,
    }
    assert plan.exploration_slots == 0


def test_missing_route_pool_releases_all_route_slots():
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=8,
        available_experts=("shallow", "anchored_replacement"),
    )

    assert plan.route_scale_floor_counts == {}
    assert plan.expert_floor_counts == {"shallow": 1, "anchored_replacement": 1}
    assert plan.exploration_slots == 6


def test_later_round_has_no_cold_start_floors():
    plan = cold_start_allocation_v1(
        round_index=2,
        batch_size=8,
        available_experts=BASE_EXPERTS,
    )

    assert not plan.active
    assert plan.route_scale_floor_counts == {}
    assert plan.expert_floor_counts == {}
    assert plan.exploration_slots == 8


@pytest.mark.parametrize(
    ("round_index", "batch_size", "experts"),
    [(0, 8, BASE_EXPERTS), (1, 7, BASE_EXPERTS), (1, 8, (*BASE_EXPERTS, "unknown"))],
)
def test_invalid_contract_values_fail_loudly(round_index, batch_size, experts):
    with pytest.raises(ValueError):
        cold_start_allocation_v1(
            round_index=round_index,
            batch_size=batch_size,
            available_experts=experts,
        )


def test_plan_selects_three_top_large_routes_and_preserves_base_experts():
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=8,
        available_experts=BASE_EXPERTS,
    )
    candidates = [
        _candidate("small", "route_complete_region", band="small", rank=4),
        _candidate("medium", "route_complete_region", band="medium", rank=12),
        _candidate("large-9", "route_complete_region", band="large", rank=9),
        _candidate("large-3", "route_complete_region", band="large", rank=3),
        _candidate("large-6", "route_complete_region", band="large", rank=6),
        _candidate("large-20", "route_complete_region", band="large", rank=20),
        _candidate("shallow", "shallow"),
        _candidate("anchored", "anchored_replacement"),
        _candidate("explore", "shallow"),
    ]

    selected = select_batch(
        candidates,
        ProgramValue(),
        SearchState(),
        np.random.default_rng(19),
        round_index=1,
        batch=8,
        exploration=plan.exploration_slots,
        expert_floor_rounds=1,
        route_scale_floor_rounds=1,
        route_scale_floor_counts=plan.route_scale_floor_counts,
    )

    selected_smiles = {row["smiles"] for row in selected}
    assert {"large-3", "large-6", "large-9"} <= selected_smiles
    selected_lanes = {row["proposal_lane"] for row in selected}
    assert {"shallow", "anchored_replacement"} <= selected_lanes
    assert len(selected) == 8
