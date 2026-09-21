"""Invariants of the PMO discovery allocator (arm C) and its wiring.

Zero oracle calls. Improvements and frontier gains are supplied directly, because the
allocation law has to be correct before any budget is spent against it.

Two halves, and the second is the one this repository keeps needing. The first half is
module algebra. The second drives the LIVE controller, because the failure mode here is
not a wrong allocator -- it is a correct allocator nothing consults.
:meth:`OnlineProposalMemory.allocation_priority` was measured to have zero production
call sites on arm B despite carrying its own passing test, which is exactly the state a
definition-site test cannot detect.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_credit import (
    CreditKey,
    PopulationCredit,
    basin_label,
)
from compose_v4.control.pmo_discovery import (
    SCHEMA_VERSION,
    DiscoveryConfig,
    DiscoveryCredit,
    basin_stratified_shares,
    discovery_quota,
)
from compose_v4.control.pmo_population_controller import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    PmoPopulationController,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

ANALOGUES = (
    "O=C(NC1CCNCC1)c1ccccc1",
    "O=C(NC1CCN(C)CC1)c1ccccc1",
    "O=C(NC1CCN(CC)CC1)c1ccccc1",
)
#: Structurally distinct from ANALOGUES: a different Bemis-Murcko scaffold.
RIVALS = ("c1ccc2ccccc2c1", "C1CCOCC1", "c1ccncc1")


def _dominant_cells(n=10):
    return [
        CreditKey(basin="DOM", parent=f"p{i}", family="atom_insert+cycle_close", scale=s)
        for i in range(n)
        for s in ("refine", "medium", "jump")
    ]


def _rival_cells():
    return [
        CreditKey(basin=f"RIV{i}", parent="q0", family="atom_insert", scale="refine")
        for i in range(3)
    ]


# ---- Module algebra ------------------------------------------------------


def test_the_exploration_floor_is_captured_by_whoever_opens_more_cells():
    """The defect arm C fixes, as a direct B-against-C comparison.

    `PopulationCredit` spreads its exploration mass uniformly over CELLS, and a cell is
    not a basin, so a lineage that has opened thirty cells receives thirty times the
    exploration mass of a structurally distinct basin holding one. The floor is real;
    it simply protects the wrong unit.
    """
    dominant, rivals = _dominant_cells(), _rival_cells()
    keys = dominant + rivals

    def rival_mass(credit):
        credit.observe(dominant[0], 0.30)
        for key in dominant[1:6]:
            credit.observe(key, 0.02)
        return float(credit.allocate(keys)[len(dominant):].sum())

    baseline = rival_mass(PopulationCredit())
    discovery = rival_mass(DiscoveryCredit())
    assert baseline < 0.10, f"fixture no longer reproduces the defect: {baseline:.4f}"
    assert discovery > 2.0 * baseline, (
        f"basin stratification bought nothing: {baseline:.4f} -> {discovery:.4f}"
    )


@pytest.mark.parametrize("exploit", [1.0, 1e3, 1e6, 1e12])
def test_every_basin_keeps_its_floor_under_an_arbitrarily_attractive_exploit(exploit):
    """The HARD guarantee: the floor does not depend on the credit values at all.

    An adversarial reward that makes one early exploit arbitrarily attractive drives
    `q_credit` for every rival basin to a rounding error. The floor term is ADDED after
    the credit term is normalized rather than multiplied into it, so it survives.
    """
    dominant, rivals = _dominant_cells(), _rival_cells()
    keys = dominant + rivals
    credit = DiscoveryCredit()
    for _ in range(500):
        credit.observe(dominant[0], exploit)
    shares = credit.allocate(keys)
    n_basins = len({key.basin for key in keys})
    guarantee = credit.exploration_floor / n_basins
    for index, key in enumerate(keys):
        if key.basin.startswith("RIV"):
            assert float(shares[index]) >= guarantee * 0.999, (
                f"{key.basin} fell below its floor at exploit={exploit:g}"
            )


def test_the_floor_is_a_basin_guarantee_not_a_cell_guarantee():
    """Stated explicitly so it cannot be misread as a per-cell promise.

    Cells inside a crowded basin legitimately receive LESS each than a cell that is the
    sole occupant of its own basin -- that is the mechanism, not a regression.
    """
    dominant, rivals = _dominant_cells(), _rival_cells()
    keys = dominant + rivals
    shares = DiscoveryCredit().allocate(keys)
    per_basin: dict[str, float] = {}
    for key, share in zip(keys, shares, strict=True):
        per_basin[key.basin] = per_basin.get(key.basin, 0.0) + float(share)
    guarantee = DiscoveryCredit().exploration_floor / len(per_basin)
    assert min(per_basin.values()) >= guarantee * 0.999
    # The crowded basin's individual cells sit below the per-basin guarantee.
    assert min(float(s) for s, k in zip(shares, keys, strict=True) if k.basin == "DOM") < guarantee


def test_basin_stratified_shares_sum_to_one_and_refuse_bad_input():
    keys = _rival_cells()
    shares = basin_stratified_shares(keys, np.ones(3), floor=0.2)
    assert float(shares.sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one value per key"):
        basin_stratified_shares(keys, np.ones(2), floor=0.2)
    with pytest.raises(ValueError, match="finite"):
        basin_stratified_shares(keys, np.asarray([1.0, np.nan, 1.0]), floor=0.2)
    with pytest.raises(ValueError, match="strictly inside"):
        basin_stratified_shares(keys, np.ones(3), floor=1.0)


@pytest.mark.parametrize(
    ("room", "fraction", "expected"),
    [(14, 0.25, 4), (14, 0.0, 0), (0, 0.25, 0), (1, 0.25, 1), (3, 0.99, 3), (8, 0.1, 1)],
)
def test_discovery_quota_rounds_up_and_never_exceeds_room(room, fraction, expected):
    """Rounded UP, so a fractional entitlement is a slot rather than a rounding loss."""
    assert discovery_quota(room, fraction=fraction) == expected


def test_a_basin_cannot_look_thin_by_opening_fresh_cells():
    """Discovery eligibility is counted at the BASIN, summing over every other axis.

    Counting at the cell would let a heavily-worked basin qualify for the reserved
    discovery slots by opening one new parent -- the precise move that captures a
    cell-level floor.
    """
    credit = DiscoveryCredit(config=DiscoveryConfig(basin_trial_threshold=3))
    worked = _dominant_cells(n=10)
    for key in worked[:8]:
        credit.observe(key, 0.01)
    fresh = CreditKey(basin="DOM", parent="brand_new", family="atom_delete", scale="jump")
    assert credit.basin_trials()["DOM"] == 8
    assert not credit.is_discovery_cell(fresh), "a fresh cell laundered a worked basin"
    assert credit.is_discovery_cell(_rival_cells()[0])


def test_frontier_evidence_sits_beside_parent_relative_and_raises_the_value():
    """Added, never substituted: parent-relative stays the learning evidence."""
    key = _rival_cells()[0]
    credit = DiscoveryCredit()
    credit.observe(key, 0.05)
    without = credit.value(key)
    credit.observe_frontier(key, 0.04)
    assert credit.value(key) > without
    # Both records survive independently.
    assert credit.cell(key).trials == 1
    assert credit.frontier_cell(key).trials == 1
    assert credit.cell(key).positive_improvement_sum == pytest.approx(0.05)
    assert credit.frontier_cell(key).positive_improvement_sum == pytest.approx(0.04)


def test_without_frontier_evidence_the_value_is_exactly_the_base_allocator():
    """Arm C reduces to arm B's cell value until frontier evidence exists.

    So any difference measured between the arms is attributable to the two mechanisms
    this module adds, not to an incidental change in the value function.
    """
    keys = _dominant_cells(n=3) + _rival_cells()
    base, discovery = PopulationCredit(), DiscoveryCredit()
    for credit in (base, discovery):
        for index, key in enumerate(keys[:5]):
            credit.observe(key, 0.01 * (index + 1))
    for key in keys:
        assert discovery.value(key) == pytest.approx(base.value(key))


def test_a_frontier_gain_must_be_finite_and_non_negative():
    credit = DiscoveryCredit()
    key = _rival_cells()[0]
    with pytest.raises(ValueError, match="non-negative"):
        credit.observe_frontier(key, -0.01)
    with pytest.raises(ValueError, match="finite"):
        credit.observe_frontier(key, float("inf"))
    with pytest.raises(TypeError):
        credit.observe_frontier("not-a-key", 0.01)


def test_snapshot_round_trips_and_refuses_a_plain_credit_payload():
    credit = DiscoveryCredit(config=DiscoveryConfig(discovery_fraction=0.3))
    key = _rival_cells()[0]
    credit.observe(key, 0.05)
    credit.observe_frontier(key, 0.04)
    payload = credit.payload()
    assert payload["schema_version"] == SCHEMA_VERSION
    restored = DiscoveryCredit.restore(json.loads(json.dumps(payload)))
    assert restored.payload() == payload
    assert restored.config.discovery_fraction == pytest.approx(0.3)
    assert restored.frontier_cell(key).positive_improvement_sum == pytest.approx(0.04)
    # A v2 payload carries no frontier evidence. Restoring it would read as "this cell
    # contributed nothing to the frontier" rather than "this was never measured".
    with pytest.raises(ValueError, match="unexpected schema"):
        DiscoveryCredit.restore(PopulationCredit().payload())


def test_discovery_config_refuses_settings_outside_its_range():
    with pytest.raises(ValueError, match="discovery fraction"):
        DiscoveryConfig(discovery_fraction=1.0)
    with pytest.raises(ValueError, match="basin trial threshold"):
        DiscoveryConfig(basin_trial_threshold=-1)
    with pytest.raises(ValueError, match="frontier weight"):
        DiscoveryConfig(frontier_weight=-0.5)


# ---- Controller wiring: the consumption guards ---------------------------


def _controller(*, memory=True, discovery=True, seed=23):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=32,
        candidates_per_batch=8,
        wall_seconds=5,
        parent_allocation="niche_score",
    )
    return PmoPopulationController(
        config,
        source_group="fixture-source",
        oracle_protocol="fixture-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
        enable_online_memory=memory,
        enable_discovery=discovery,
    )


def _candidate(endpoint, *, parent, channel, rules=("atom_insert",), parent_score=1.0):
    return {
        "endpoint": endpoint,
        "program": {"blocks": []},
        "trace": {"actions": [{"executor_rule": rule} for rule in rules]},
        "provenance": {
            "planner_channel": channel,
            "entry_id": parent,
            "parent_measured_score": parent_score,
        },
    }


def test_arm_c_requires_arm_b():
    """Discovery without the memory would silently allocate as arm B under C's identity."""
    with pytest.raises(ValueError, match="requires the online memory"):
        _controller(memory=False, discovery=True)


def test_the_controller_actually_constructs_the_discovery_allocator():
    assert isinstance(_controller().credit, DiscoveryCredit)
    assert type(_controller(discovery=False).credit) is PopulationCredit


def test_observe_batch_writes_frontier_credit_on_the_production_path():
    """CONSUMPTION guard: drives the live `observe_batch`, not the credit object.

    Dropping the `observe_frontier` call from the controller turns this red while every
    module-algebra test above stays green -- which is the whole failure this repository
    has hit three times.
    """
    controller = _controller()
    rows = [
        controller._augment(
            _candidate(smiles, parent=f"p{i}", channel=SHALLOW_CHANNEL, parent_score=1.0)
        )
        for i, smiles in enumerate(ANALOGUES)
    ]
    controller.pending = {"batch_id": "b0", "candidates": rows}
    outcomes = [
        {"candidate_id": row["candidate_id"], "score": 5.0, "endpoint": row["endpoint"]}
        for row in rows
    ]
    # The hand-built pending batch trips the archive lock AFTER the credit writes. If
    # the archive ever rejected earlier, the assertions below would measure nothing and
    # this line fails first.
    with pytest.raises(ValueError, match="pending candidate lock"):
        controller.observe_batch("b0", outcomes)
    assert controller.credit.frontier_cells, "no frontier credit recorded from a scored batch"
    assert sum(c.trials for c in controller.credit.frontier_cells.values()) == len(rows)
    # Both evidence kinds were written for the same cells, and neither replaced the other.
    assert set(controller.credit.frontier_cells) == set(controller.credit.cells)


def test_arm_b_records_no_frontier_evidence():
    """The off state is the absence of the mechanism, not a zeroed version of it."""
    controller = _controller(discovery=False)
    rows = [
        controller._augment(_candidate(s, parent=f"p{i}", channel=SHALLOW_CHANNEL))
        for i, s in enumerate(ANALOGUES)
    ]
    controller.pending = {"batch_id": "b0", "candidates": rows}
    with pytest.raises(ValueError, match="pending candidate lock"):
        controller.observe_batch(
            "b0",
            [{"candidate_id": r["candidate_id"], "score": 5.0, "endpoint": r["endpoint"]}
             for r in rows],
        )
    assert controller.credit.cells, "arm B stopped recording parent-relative credit"
    assert not hasattr(controller.credit, "frontier_cells")


def _pool(controller):
    rows = []
    for index, smiles in enumerate(ANALOGUES):
        rows.append(
            controller._augment(_candidate(smiles, parent=f"p{index}", channel=SHALLOW_CHANNEL))
        )
    for index, smiles in enumerate(RIVALS):
        rows.append(
            controller._augment(
                _candidate(smiles, parent=f"q{index}", channel=STRUCTURED_CHANNEL)
            )
        )
    return rows


def test_the_discovery_reservation_is_consumed_and_reaches_a_thin_basin():
    """The reserved slots are realized counts, not an expectation over one batch."""
    controller = _controller()
    rows = _pool(controller)
    # Pour trials into the analogue basin so pure credit would never leave it.
    worked = basin_label(ANALOGUES[0])
    for index in range(40):
        controller.credit.observe(
            CreditKey(basin=worked, parent=f"p{index % 3}", family="atom_insert", scale="refine"),
            5.0,
        )
    _, detail = controller._credit_allocate(rows, controller._fit_value()[0], room=6)
    assert detail["mode"].endswith("with_discovery_reservation")
    assert detail["discovery_reserved_slots"] >= 1
    assert detail["discovery_slots_drawn"] == detail["discovery_reserved_slots"]
    drawn_basins = {row["basin"] for row in detail["drawn_cells"] if row["role"] == "discovery"}
    assert drawn_basins, "the reservation drew nothing"
    assert worked not in drawn_basins, "a reserved discovery slot went to the worked basin"


def test_arm_b_allocation_detail_keeps_its_schema_and_reserves_nothing():
    """Arm B must be unchanged by the wiring: no reservation, no role key, same mode."""
    controller = _controller(discovery=False)
    rows = _pool(controller)
    controller.credit.observe(
        CreditKey(basin=basin_label(ANALOGUES[0]), parent="p0", family="atom_insert",
                  scale="refine"),
        5.0,
    )
    _, detail = controller._credit_allocate(rows, controller._fit_value()[0], room=6)
    assert detail["mode"] == "joint_credit_cell_allocation"
    assert "discovery_reserved_slots" not in detail
    assert all("role" not in row for row in detail["drawn_cells"]), (
        "arm B's draw record grew a discovery field"
    )


def test_restore_refuses_to_change_the_arm_in_either_direction():
    """A silent arm change under a run identity that says otherwise is the hazard."""
    state = {"credit": DiscoveryCredit().payload()}
    assert state["credit"]["schema_version"] == SCHEMA_VERSION
    # The controller-level guard is exercised through the same predicate the restore
    # uses, without rebuilding a full snapshot: a discovery payload into a plain arm.
    plain = _controller(discovery=False)
    assert type(plain.credit) is PopulationCredit
    with pytest.raises(ValueError, match="unexpected schema"):
        PopulationCredit.restore(DiscoveryCredit().payload())
    with pytest.raises(ValueError, match="unexpected schema"):
        DiscoveryCredit.restore(PopulationCredit().payload())
