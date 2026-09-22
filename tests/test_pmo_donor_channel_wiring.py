"""The donor lane is WIRED, and switching it off changes nothing.

This file exists because this repository has shipped three mechanisms that were built,
tested, merged and INERT -- the T4 region law behind a keyword no caller passed,
`OnlineProposalMemory.allocation_priority` with zero production call sites, and
`donor_program` itself, absent from the scored entry point's whole import closure.  Every
one of them would have passed a signature check, because in each case the seam existed and
was dropped one hop later.

So consumption is established by EXECUTION: a law that raises is installed where the
controller resolves it, `propose_batch` is driven, and the probe exception is required to
escape.  `DonorLawProbe` derives from `BaseException` precisely because the lane catches
`(ValueError, RuntimeError)` per attempt exactly as the jump lane does -- a probe raising
either would be swallowed by the code being observed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.bridge_region_law import bridge_separated_regions
from compose_v4.control.donor_program import pendant_cuts
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_donor_channel import (
    DONOR_CHANNEL,
    DonorLawProbe,
    assert_donor_law_is_consumed,
    donor_region_law,
    pendant_cut_from_region,
)
from compose_v4.control.pmo_population_controller import (
    CHANNELS,
    PmoPopulationController,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

SOURCE = "CC(=O)Nc1ccc(O)cc1"


def _eligibility(row):
    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def _controller(*, donor: bool, seed: int = 11, attempts: int = 6):
    source = production_state_from_smiles(SOURCE, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=attempts,
        candidates_per_batch=4,
        wall_seconds=20,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source, (), config,
        source_group="donor-test", oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    kwargs = {"enable_donor_channel": True} if donor else {}
    controller = PmoPopulationController(
        config, source_group="donor-test", oracle_protocol="free-no-oracle",
        hierarchy=None, jump_checkpoint=CHECKPOINT, enable_online_memory=True, **kwargs,
    )
    for index, candidate in enumerate(batch["candidates"][:4]):
        controller.add_measured_program(candidate, receipt_id=f"seed{index}", score=0.5)
    return controller


# ---- The correspondence the reuse rests on -------------------------------


def test_pendant_cuts_are_exactly_the_single_bond_bridge_regions() -> None:
    """The measured claim that licenses reusing `BridgeRegionLaw` instead of writing one."""
    graph = production_state_from_smiles("CC(=O)Nc1ccc(OCC)cc1", max_atoms=48)
    cuts = {(tuple(sorted(c.component)), c.anchor) for c in pendant_cuts(graph)}
    regions = bridge_separated_regions(graph, maximum=None)
    single = {(tuple(sorted(r.fragment)), r.anchor) for r in regions if r.bond_order == 1}
    assert cuts == single
    # A strict superset: the law also enumerates multi-bond bridges, which the transplant
    # planner refuses. If this ever becomes an equality the conversion's refusal is untested.
    assert len(regions) > len(single), "fixture no longer exercises a non-single bridge"


def test_the_conversion_recovers_root_and_refuses_a_multi_bond_bridge() -> None:
    graph = production_state_from_smiles("CC(=O)Nc1ccc(OCC)cc1", max_atoms=48)
    by_key = {(tuple(sorted(c.component)), c.anchor): c for c in pendant_cuts(graph)}
    refused = converted = 0
    for region in bridge_separated_regions(graph, maximum=None):
        cut = pendant_cut_from_region(graph, region)
        if region.bond_order != 1:
            assert cut is None
            refused += 1
            continue
        converted += 1
        assert cut is not None
        assert cut == by_key[(tuple(sorted(region.fragment)), region.anchor)]
    assert refused > 0, "the refusal branch was never reached"
    assert converted > 0


def test_the_retentive_law_reranks_and_never_filters() -> None:
    graph = production_state_from_smiles("CC(=O)Nc1ccc(OCC)cc1", max_atoms=48)
    law = donor_region_law(graph)
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    assert (weights > 0).all(), "a zero weight would make the tilt a filter"
    uniform = bridge_separated_regions(graph, maximum=None)
    # Same support as the uniform arm: the arms differ in probability, never in reach.
    assert {(r.fragment, r.anchor) for r in regions} == {(r.fragment, r.anchor) for r in uniform}
    sizes = [len(r.fragment) for r in regions]
    smallest = weights[sizes.index(min(sizes))]
    largest = weights[sizes.index(max(sizes))]
    assert smallest > largest, "the retentive law must prefer the smaller release"


def test_the_margin_reads_nothing_but_atom_counts() -> None:
    """Task independence, asserted on the object rather than promised in prose."""
    graph = production_state_from_smiles(SOURCE, max_atoms=48)
    law = donor_region_law(graph)
    closure = law.margin.__closure__ or ()
    values = [cell.cell_contents for cell in closure]
    assert values and all(isinstance(value, int) for value in values), (
        f"the margin closed over {values!r}; it must capture an atom COUNT and nothing "
        "else, or it could carry task information"
    )


# ---- THE HARD GATE: consumption at the call site -------------------------


def test_the_donor_law_is_consumed_by_the_production_propose_batch() -> None:
    """No scored run until this passes. Drives the REAL entry point, not the lane."""
    controller = _controller(donor=True)

    def draw(law, _rng):
        controller.donor_law = law
        controller.propose_batch(_eligibility)

    report = assert_donor_law_is_consumed(draw)
    assert report["consumed"] is True


def test_the_consumption_gate_fails_when_the_law_is_dropped() -> None:
    """The gate's own negative control: a draw that ignores the law must NOT pass."""
    with pytest.raises(AssertionError, match="NEVER consulted"):
        assert_donor_law_is_consumed(lambda _law, _rng: None, seeds=3)


def test_the_probe_exception_survives_the_lane_s_own_handler() -> None:
    """A ValueError/RuntimeError probe would be swallowed by the per-attempt handler."""
    assert issubclass(DonorLawProbe, BaseException)
    assert not issubclass(DonorLawProbe, Exception), (
        "an Exception subclass would be caught by the lane's own except clause and the "
        "consumption gate would silently measure nothing"
    )


# ---- ABSENT IS THE ONLY OFF ---------------------------------------------


def test_off_is_byte_identical_and_draws_no_extra_parents() -> None:
    """Off must not perturb the RNG, and extra parent draws are how it would.

    `propose_batch` draws `attempts_per_batch` parents PER CHANNEL, and `self._parent()`
    consumes `self.rng`. So channel-set membership -- not a uniform law -- is the only
    switch that leaves every existing draw where it was.
    """
    off = _controller(donor=False)
    assert off.channels is CHANNELS
    assert off.donor_rng is None
    assert set(off.population_state["channels"]) == set(CHANNELS)
    assert DONOR_CHANNEL not in off.population_state["channels"]

    def _count_parent_draws(donor: bool) -> int:
        controller = _controller(donor=donor)
        original, drawn = controller._parent, []

        def counting():
            drawn.append(1)
            return original()

        controller._parent = counting
        controller.propose_batch(_eligibility)
        return len(drawn)

    counts = {donor: _count_parent_draws(donor) for donor in (False, True)}
    # The DELTA is the invariant, not the absolute count: `_parent` is also called
    # outside the per-channel schedules, equally in both arms, so pinning the total
    # would assert something this test does not mean. Turning the lane on must add
    # EXACTLY one schedule -- `attempts_per_batch` draws -- and nothing else.
    assert counts[True] - counts[False] == 6, (
        f"turning the donor lane on moved the parent draw count by "
        f"{counts[True] - counts[False]}, not by one schedule of 6; the lanes are "
        "coupled through the RNG"
    )


def test_the_donor_lane_has_its_own_rng_stream() -> None:
    """A shared stream would make the new lane perturb the jump lane's draws.

    Checked by DRAWING, not by comparing seed arguments: two generators seeded
    differently are only useful if they actually diverge.
    """
    controller = _controller(donor=True)
    donor = [controller.donor_rng.random() for _ in range(8)]
    jump = [controller.jump_rng.random() for _ in range(8)]
    assert donor != jump, (
        "the donor lane shares the jump lane's RNG stream; enabling it would move "
        "every jump-lane draw"
    )


def test_the_snapshot_carries_the_donor_rng_only_when_the_lane_is_on() -> None:
    off = _controller(donor=False).snapshot(include_history=False)
    on = _controller(donor=True).snapshot(include_history=False)
    assert "donor_rng" not in off["pmo_population"], (
        "an off controller must not add a key to the snapshot payload"
    )
    assert "donor_rng" in on["pmo_population"]
