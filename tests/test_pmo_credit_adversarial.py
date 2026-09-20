"""Adversarial counterexamples against the PMO joint-credit allocation.

Every test here is a MINIMAL reproducer for a defect found by
`scripts/pmo_controller_adversarial_audit.py`.  None of them fixes anything: the
controller and the credit module are pinned by content hash into sealed contracts, so
these record the defect and fail the moment somebody claims it is gone.

WHY `xfail(strict=True)` AND NOT A HARD FAILURE.  A scored relaunch is pinned against a
green `pytest tests/`, and turning that red would block the launch over findings the
owner has not yet triaged.  `strict=True` still executes every assertion and still fails
the suite if the behaviour ever changes -- an XPASS is an error -- so the evidence is
preserved in both directions.  To watch them fail directly:

    pytest tests/test_pmo_credit_adversarial.py --runxfail

The three fixtures below are pure credit-module algebra with no RDKit and no controller,
except `test_joint_credit_is_dropped_by_snapshot_restore`, which needs the real
controller because the defect lives in its `snapshot`/`restore` pair.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_credit import CreditKey, PopulationCredit
from compose_v4.control.pmo_population_controller import PmoPopulationController

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]


def _controller(seed=20260920):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        parent_allocation="niche_score",
        attempts_per_batch=128,
        candidates_per_batch=16,
        wall_seconds=45.0,
        proposal_cache_entries=128,
    )
    return PmoPopulationController(
        config,
        source_group="adversarial",
        oracle_protocol="free-synthetic",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
    )


# FIXED: snapshot/restore now persist both `credit` and `_pool_continuity`, so this is a
# live regression test rather than a defect reproducer.  It was found as DEFECT 1 -- a
# resume silently reverted the controller to the cold-start acquisition path and re-learnt
# from nothing, which a matched A/B measured as worse than the credit path.
def test_joint_credit_is_dropped_by_snapshot_restore():
    """`run_program_campaign` restores a snapshot on resume AND per completed round.

    `program_campaign.run_program_campaign` calls `optimizer_type.restore(...)` for every
    round whose `complete.json` already exists, and again for a `warm_start`.  Modal
    preemption makes that an expected event, not a hypothetical one, so a controller that
    loses its allocation authority on restore loses it in production.
    """
    controller = _controller()
    key = CreditKey(basin="c1ccccc1", parent="p0", family="atom_insert", scale="jump")
    for _ in range(37):
        controller.credit.observe(key, 3.0)
    assert controller.credit.cell(key).trials == 37

    restored = PmoPopulationController.restore(
        controller.snapshot(), hierarchy=None, jump_checkpoint=CHECKPOINT
    )
    # PopulationCredit.payload/restore exist and round-trip; nothing calls them.
    assert restored.credit.cell(key).trials == 37, "joint credit did not survive restore"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT 2: PopulationCredit.allocate spreads budget over CELLS, so a basin that "
        "fragments into many zero-credit cells outbids a cell with measured credit"
    ),
)
def test_cell_proliferation_cannot_outbid_measured_credit():
    """40 trials at +4.0 each must not lose the budget to 30 cells that never won.

    Each untried cell is worth `prior_weight / sqrt(1)` = 0.25 regardless of the basin it
    sits in, and the uniform floor adds `eps / n` on top, so total share to a lineage
    scales with how many DISTINCT cells it opens.  The jump channel invents novel
    scaffolds, and a novel scaffold is a novel basin, hence a novel cell every round.
    """
    credit = PopulationCredit()
    productive = CreditKey(basin="PROD", parent="p0", family="atom_insert", scale="refine")
    for _ in range(40):
        credit.observe(productive, 4.0)
    fragments = [
        CreditKey(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
        for i in range(30)
    ]
    shares = credit.allocate([productive, *fragments])
    assert credit.value(productive) > 4.0 > credit.value(fragments[0])
    assert float(shares[0]) > float(shares[1:].sum()), (
        f"30 zero-credit cells took {float(shares[1:].sum()):.3f} of the budget against "
        f"{float(shares[0]):.3f} for a cell with 40 trials at +4.0"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT 3: the allocation key includes parent entry_id and allocate() never backs "
        "off to marginal(), so a child of a productive lineage inherits nothing"
    ),
)
def test_child_of_a_productive_lineage_outranks_a_child_of_a_barren_one():
    """A new generation opens a new cell, and a new cell is worth the flat prior.

    `PopulationCredit.marginal('basin')` knows PROVEN earned +300 over 60 trials while
    BARREN earned nothing over 60, but `allocate` reads `value(key)` on the joint cell
    only.  So a descendant that stays inside the proven basin starts level with a
    descendant of a lineage that has never produced anything -- which is exactly the
    compounding the joint credit was introduced to enable.
    """
    credit = PopulationCredit()
    for _ in range(60):
        credit.observe(
            CreditKey(basin="PROVEN", parent="gen0", family="atom_insert", scale="refine"), 5.0
        )
        credit.observe(
            CreditKey(basin="BARREN", parent="b0", family="atom_insert", scale="refine"), 0.0
        )
    assert credit.marginal("basin")["PROVEN"].positive_improvement_sum == 300.0
    assert credit.marginal("basin")["BARREN"].positive_improvement_sum == 0.0

    child_of_proven = CreditKey(
        basin="PROVEN", parent="gen1", family="atom_insert", scale="refine"
    )
    child_of_barren = CreditKey(
        basin="BARREN", parent="b1", family="atom_insert", scale="refine"
    )
    assert credit.value(child_of_proven) > credit.value(child_of_barren), (
        "a child inside a proven basin starts at the same flat prior "
        f"({credit.value(child_of_proven)}) as a child inside a barren one"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT 4: credit_report()['trial_fraction_by_scale'] is a lifetime trial count, "
        "so it reports history rather than the current allocation regime"
    ),
)
def test_trial_fraction_by_scale_reflects_the_current_regime():
    """The diagnostic offered for 'has the scale mix shifted' cannot answer that question.

    After the controller has switched entirely to refine, every one of the last 50 trials
    is refine, yet the cumulative counter still calls jump the majority scale.  Measured
    across 15 seeds on a jump-early / refine-late landscape, the realized per-round mix
    moved in the intended direction 15/15 times while this field showed the shift 0/15.
    """
    controller = _controller()
    jump = CreditKey(basin="B", parent="p0", family="cycle_close", scale="jump")
    refine = CreditKey(basin="B", parent="p0", family="atom_insert", scale="refine")
    for _ in range(200):
        controller.credit.observe(jump, 1.0)
    for _ in range(50):  # the regime has changed: recent policy is 100% refine
        controller.credit.observe(refine, 1.0)

    fractions = controller.credit_report()["trial_fraction_by_scale"]
    assert fractions["refine"] > fractions["jump"], (
        f"after a complete switch to refine the diagnostic still reports {fractions}"
    )


def test_exploration_floor_reaches_every_live_cell_at_one_slot_per_draw():
    """Regression for what DOES work: the floor is real and matches eps/n.

    Not adversarial -- this pins the one guarantee the audit confirmed, so a future change
    to `allocate` that quietly decays the floor is caught.  Drawn at room=1 because a
    multi-slot draw empties the winning cell and reaches the others by exhaustion.
    """
    credit = PopulationCredit()
    winner = CreditKey(basin="W", parent="p0", family="atom_insert", scale="refine")
    for _ in range(2000):
        credit.observe(winner, 1000.0)
    others = [
        CreditKey(basin=f"O{i}", parent=f"p{i}", family="atom_insert", scale="jump")
        for i in range(14)
    ]
    keys = [winner, *others]
    shares = credit.allocate(keys)
    floor = credit.exploration_floor / len(keys)
    assert min(float(s) for s in shares) >= floor * 0.999
    assert float(shares[0]) == pytest.approx(
        (1 - credit.exploration_floor) * (credit.value(winner) / sum(credit.value(k) for k in keys))
        + floor,
        rel=1e-6,
    )
