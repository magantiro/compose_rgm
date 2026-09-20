"""The joint credit object is wired into the live PMO controller.

These tests drive the controller's own `_augment`, `_allocate`, `_credit_allocate` and
`observe_batch`, rather than the credit module in isolation: the failure this guards
against is a correct credit object that nothing consults, which is exactly the state the
controller was in before wiring (`basin_id` was computed and had zero readers).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_credit import CreditKey, basin_label
from compose_v4.control.pmo_population_controller import (
    JUMP_CHANNEL,
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    PmoPopulationController,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

# One shared Bemis-Murcko scaffold, decorated five different ways.
ANALOGUES = (
    "O=C(NC1CCNCC1)c1ccccc1",
    "O=C(NC1CCN(C)CC1)c1ccccc1",
    "O=C(NC1CCN(CC)CC1)c1ccccc1",
    "O=C(NC1CCN(CCO)CC1)c1ccccc1",
    "O=C(NC1CCNCC1)c1ccc(F)cc1",
)


def _controller(seed=23):
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


def test_analogues_sharing_a_scaffold_share_one_basin_id():
    """The defect this replaces: a per-molecule basin key makes every cell a singleton."""
    controller = _controller()
    rows = [
        controller._augment(_candidate(smiles, parent="p0", channel=SHALLOW_CHANNEL))
        for smiles in ANALOGUES
    ]
    basins = {row["provenance"]["basin_id"] for row in rows}
    assert len(basins) == 1, f"analogues split across {len(basins)} basins"
    assert basins == {basin_label(ANALOGUES[0])}
    # Distinct chemistry must still separate, or the coarsening has gone too far.
    other = controller._augment(_candidate("c1ccc2ccccc2c1", parent="p0", channel=SHALLOW_CHANNEL))
    assert other["provenance"]["basin_id"] not in basins


def test_observe_batch_writes_joint_credit_cells():
    controller = _controller()
    rows = [
        controller._augment(
            _candidate(smiles, parent=f"p{index}", channel=SHALLOW_CHANNEL, parent_score=1.0)
        )
        for index, smiles in enumerate(ANALOGUES[:3])
    ]
    controller.pending = {"batch_id": "b0", "candidates": rows}
    outcomes = [
        {"candidate_id": row["candidate_id"], "score": 5.0, "endpoint": row["endpoint"]}
        for row in rows
    ]
    # The hand-built pending batch trips the parent archive's lock check, which is raised
    # AFTER the credit write.  Asserting the specific failure keeps this test honest: if
    # the archive ever started rejecting earlier, the credit assertions below would be
    # measuring nothing and this line would fail first.
    with pytest.raises(ValueError, match="pending candidate lock"):
        controller.observe_batch("b0", outcomes)
    assert controller.credit.cells, "no joint credit recorded from a scored batch"
    keys = list(controller.credit.cells)
    assert all(isinstance(key, CreditKey) for key in keys)
    assert len({key.basin for key in keys}) == 1, "analogues should share the basin axis"
    assert len({key.parent for key in keys}) == 3, "parent must stay a separate axis"
    assert all(cell.improvements > 0 for cell in controller.credit.cells.values())


def test_credit_allocation_never_starves_an_unrewarded_cell():
    """The epsilon floor must survive the wiring, not just the credit module.

    Drawn one slot at a time: with more than one slot the winning cell simply runs out of
    candidates and the others get drawn for that reason instead, which would make this
    test pass even with the floor removed.  room=1 isolates the floor as the only
    mechanism that can reach an unrewarded lineage.
    """
    controller = _controller()
    rows = []
    for index, channel in enumerate((SHALLOW_CHANNEL, STRUCTURED_CHANNEL, JUMP_CHANNEL)):
        rows.extend(
            controller._augment(_candidate(smiles, parent=f"p{index}", channel=channel))
            for smiles in ANALOGUES
        )
    # Pour credit into exactly one cell, so pure credit would allocate everything to it.
    winner = CreditKey(
        basin=basin_label(ANALOGUES[0]), parent="p0", family="atom_insert", scale="refine"
    )
    for _ in range(500):
        controller.credit.observe(winner, 1000.0)

    value, _ = controller._fit_value()
    drawn = []
    for _ in range(400):
        chosen, detail = controller._credit_allocate(rows, value, 1)
        assert detail["mode"] == "joint_credit_cell_allocation"
        assert detail["minimum_share"] > 0.0
        drawn.extend(row["provenance"]["entry_id"] for row in chosen)
    assert set(drawn) == {"p0", "p1", "p2"}, f"a lineage was starved out: {sorted(set(drawn))}"
    off_winner = sum(entry != "p0" for entry in drawn) / len(drawn)
    assert off_winner > 0.02, f"exploration floor is not reaching unrewarded cells: {off_winner}"


def test_cold_start_allocation_is_unchanged():
    controller = _controller()
    rows = [
        controller._augment(_candidate(smiles, parent="p0", channel=SHALLOW_CHANNEL))
        for smiles in ANALOGUES
    ]
    assert not controller.credit.cells
    _, report = controller._allocate(rows)
    assert report["credit_allocation"]["mode"] in {"cold_start_acquisition", "floor_only"}
    assert report["credit_summary"]["active_cells"] == 0


def test_credit_report_exposes_the_requested_diagnostics():
    controller = _controller()
    key = CreditKey(basin="c1ccccc1", parent="p0", family="atom_insert", scale="jump")
    controller.credit.observe(key, 2.0)
    report = controller.credit_report()
    assert report["active_basins"] == 1
    assert report["cells_with_positive_credit"] == 1
    assert report["productive_parents"] == 1
    assert report["trial_fraction_by_scale"] == {"jump": 1.0}
    assert report["top_earning_cells"][0]["parent"] == "p0"
