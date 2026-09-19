import numpy as np

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    integrated_features,
    merge_expert_pools,
    select_batch,
)


def _row(smiles, expert, *, parent="P", score=-8.0):
    return {
        "smiles": smiles,
        "proposal_lane": expert,
        "proposal_experts": [expert],
        "parent": parent,
        "parent_score": score,
        "similarity": 0.7,
        "delta": 0.6,
        "qed": 0.7,
        "sa": 3.0,
        "regions": 1,
        "created": 1,
        "deleted": 0,
        "families": [expert],
        "program_families": [expert],
        "fingerprint": {len(smiles)},
    }


def test_merge_preserves_all_experts_and_best_parent():
    pools = {
        "shallow": [_row("C", "shallow", parent="weak", score=-7.0)],
        "anchored_replacement": [],
        "route_complete_region": [
            _row("C", "route_complete_region", parent="strong", score=-9.0),
            _row("N", "route_complete_region"),
        ],
    }
    result = merge_expert_pools(pools)
    assert [row["smiles"] for row in result] == ["C", "N"]
    carbon = result[0]
    assert carbon["parent"] == "strong"
    assert carbon["proposal_experts"] == ["route_complete_region", "shallow"]
    assert [row["expert"] for row in carbon["proposal_origins"]] == [
        "route_complete_region",
        "shallow",
    ]


def test_integrated_features_keep_intercept_last_and_distinguish_experts():
    state = SearchState(archive={"P": -8.0})
    rows = [_row("C", expert) for expert in EXPERTS]
    features = [integrated_features(row, state) for row in rows]
    assert all(vector[-1] == 1.0 for vector in features)
    assert len({tuple(vector) for vector in features}) == len(EXPERTS)
    assert all(len(vector) == 19 for vector in features)


def test_early_selection_seeds_every_available_expert_then_releases_floor():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for expert_index, expert in enumerate(EXPERTS):
        for item in range(3):
            row = _row(f"{expert_index}{item}", expert)
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {expert_index * 10 + item}
            candidates.append(row)
    value = ProgramValue()
    early = select_batch(
        candidates,
        value,
        state,
        np.random.default_rng(11),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
    )
    floor = [row for row in early if row["selection_kind"] == "expert_floor"]
    assert len(floor) == 3
    assert {row["proposal_lane"] for row in floor} == set(EXPERTS)
    assert len({row["smiles"] for row in early}) == len(early) == 8

    late = select_batch(
        candidates,
        value,
        state,
        np.random.default_rng(11),
        round_index=3,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
    )
    assert {row["selection_kind"] for row in late} == {"exploration"}


def test_route_scale_floor_uses_prior_rank_before_online_value_is_fit():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for band_index, band in enumerate(("small", "medium", "large")):
        for rank in (9, 3):
            row = _row(f"route-{band_index}-{rank}", "route_complete_region")
            row["realized_primitive_band"] = band
            row["route_proposal_rank"] = rank
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {band_index * 10 + rank}
            candidates.append(row)
    for expert in ("shallow", "anchored_replacement"):
        row = _row(expert, expert)
        row["features"] = integrated_features(row, state)
        candidates.append(row)

    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=2,
    )
    scale_floor = [
        row for row in selected if row["selection_kind"] == "route_scale_floor"
    ]

    assert {row["realized_primitive_band"] for row in scale_floor} == {
        "small",
        "medium",
        "large",
    }
    assert {row["route_proposal_rank"] for row in scale_floor} == {3}
    assert sum(row["selection_kind"] == "expert_floor" for row in selected) == 2


def test_generic_scale_floor_spans_experts_without_route_prior_ranking():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for index, (band, expert) in enumerate(
        zip(("small", "medium", "large"), EXPERTS, strict=True)
    ):
        row = _row(f"generic-{band}", expert)
        row["proposal_scale_band"] = band
        row["route_proposal_rank"] = 999 - index
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = {index}
        candidates.append(row)

    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=3,
        exploration=0,
        expert_floor_rounds=2,
        route_scale_floor_rounds=2,
        scale_floor_scope="all_generic",
    )

    assert {row["selection_kind"] for row in selected} == {"generic_scale_floor"}
    assert {row["proposal_scale_band"] for row in selected} == {
        "small",
        "medium",
        "large",
    }
    assert {row["proposal_lane"] for row in selected} == set(EXPERTS)


def test_route_scale_floor_accepts_per_band_counts_without_changing_defaults():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for band_index, band in enumerate(("small", "medium", "large")):
        for rank in (9, 3, 6, 1):
            row = _row(f"route-{band_index}-{rank}", "route_complete_region")
            row["realized_primitive_band"] = band
            row["route_proposal_rank"] = rank
            row["features"] = integrated_features(row, state)
            row["fingerprint"] = {band_index * 10 + rank}
            candidates.append(row)
    for expert in ("shallow", "anchored_replacement"):
        row = _row(expert, expert)
        row["features"] = integrated_features(row, state)
        candidates.append(row)

    defaults = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=1,
    )
    explicit_defaults = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=1,
        route_scale_floor_counts={"small": 1, "medium": 1, "large": 1},
    )
    assert explicit_defaults == defaults

    expanded = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=1,
        route_scale_floor_counts={"small": 1, "medium": 1, "large": 4},
    )
    scale_floor = [
        row for row in expanded if row["selection_kind"] == "route_scale_floor"
    ]
    assert [
        (row["realized_primitive_band"], row["route_proposal_rank"])
        for row in scale_floor
    ] == [
        ("small", 1),
        ("medium", 1),
        ("large", 1),
        ("large", 3),
        ("large", 6),
        ("large", 9),
    ]
    assert sum(row["selection_kind"] == "expert_floor" for row in expanded) == 2
    assert len({row["smiles"] for row in expanded}) == len(expanded) == 8

    duplicated = [*candidates, dict(candidates[-3])]
    deduplicated = select_batch(
        duplicated,
        ProgramValue(),
        state,
        np.random.default_rng(7),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=1,
        route_scale_floor_counts={"small": 1, "medium": 1, "large": 4},
    )
    assert len({row["smiles"] for row in deduplicated}) == len(deduplicated)


def test_route_scale_floor_counts_reject_invalid_contract_values():
    state = SearchState(archive={"P": -8.0})
    common = {
        "candidates": [],
        "value": ProgramValue(),
        "state": state,
        "rng": np.random.default_rng(7),
        "round_index": 1,
        "batch": 8,
        "exploration": 2,
        "expert_floor_rounds": 2,
        "route_scale_floor_rounds": 1,
    }
    for counts in ({"giant": 1}, {"large": -1}, {"large": True}, {"large": 1.5}):
        try:
            select_batch(**common, route_scale_floor_counts=counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid route scale floor counts accepted: {counts}")
