"""The SCORED entry point reaches the donor lane, and switching it off costs nothing.

`test_pmo_donor_channel_wiring.py` proves the CONTROLLER consumes its law. This file
proves the three hops ABOVE it, which is where every inert mechanism this repository has
shipped actually failed: `bridge_region_law` had a working `region_law=` keyword and no
production caller passing it, and a rescue launched on that commit would have spent up to
1,241 oracle calls reproducing the failure it was built to fix.

    execute_task  ->  optimizer_kwargs  ->  __init__      (fresh round)
                                       ->  restore        (every resumed round)

`run_program_campaign` splats the SAME dict into both, so a flag accepted by only one of
them makes the fresh path work and the resume path raise -- which is how this exact defect
surfaced once before, on the first 1k extension, after the ledger had already been widened.

Zero oracle calls: the gate's evaluator RAISES, so a charged call would abort rather than
merely be absent.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control import donor_memory
from compose_v4.control.donor_program import pendant_cuts
from compose_v4.control.pmo_donor_channel import (
    CUT_LAWS,
    DEFAULT_CUT_LAW,
    DONOR_CHANNEL,
    DONOR_TAG,
    RETENTIVE_RELEASED_FRACTION,
    UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM,
    DonorStratum,
    donor_region_law,
    donor_transplant_draw,
    resolve_cut_law,
)
from compose_v4.control.pmo_online_memory import ELITE
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments import pmo_donor_scored_gate as scored_gate
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

ROOT = Path(__file__).resolve().parents[1]

#: Two drug-like molecules, each carrying a small pendant and a large one, so a draw that
#: ignores the retentive tilt is visible in the SIZE of what it takes and releases.
PARENT = "CC(=O)Nc1ccc(OCCCCC)cc1"
DONOR = "COc1ccc(CCCCCCC)cc1C"


def _state(smiles: str):
    return production_state_from_smiles(smiles, max_atoms=48)


# ---- Hop 1: the scored entry point builds the kwargs ---------------------


def test_the_scored_entry_point_passes_the_donor_arm_to_the_campaign(tmp_path) -> None:
    """Drives the REAL `execute_task`; only `run_program_campaign` is replaced."""
    captured = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )
    kwargs = captured["optimizer_kwargs"]
    assert kwargs.get("enable_donor_channel") is True, (
        "execute_task did not forward the donor arm -- the lane is unreachable from "
        "every scored launch, exactly as the region law was from its production caller"
    )
    assert kwargs["donor_cut_law"] == DEFAULT_CUT_LAW
    assert captured["optimizer_type"] is PmoPopulationController


def test_the_off_arm_carries_no_donor_key_at_all(tmp_path) -> None:
    """Absence is the only byte-identical off HERE TOO, and for a concrete reason.

    `run_program_campaign` hashes `optimizer_kwargs` into `optimizer_kwargs_sha256`,
    writes it into the campaign manifest, and refuses a resume whose recipe moved
    ("campaign recipe changed during resume"). An unconditional donor key would therefore
    break every in-flight arm A/B/C run at its next preemption retry while changing
    nothing those runs compute.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(ROOT, tmp_path / "off")[
        "optimizer_kwargs"
    ]
    assert set(kwargs) == {"jump_checkpoint", "enable_online_memory", "enable_discovery"}, (
        f"the unflagged scored path now passes {sorted(kwargs)}; any addition moves "
        "optimizer_kwargs_sha256 and refuses an in-flight run's resume"
    )


# ---- Hops 2 and 3: constructor AND restore -------------------------------


def test_restore_accepts_the_donor_arm_like_the_constructor() -> None:
    init = inspect.signature(PmoPopulationController.__init__).parameters
    restore = inspect.signature(PmoPopulationController.restore).parameters
    for name in ("enable_donor_channel", "donor_cut_law"):
        assert name in init, f"{name} missing from __init__"
        assert name in restore, f"{name} missing from restore -- resume will raise"


def test_the_full_scored_chain_reaches_the_law_and_restores(tmp_path) -> None:
    """THE HARD GATE. No scored donor run until this passes."""
    report = scored_gate.gate(ROOT, tmp_path / "gate")
    assert report["verdict"] == "SCORED_ENTRY_POINT_REACHES_THE_DONOR_CUT_LAW"
    assert report["law_consumed"]["consumed"] is True
    assert report["restore_accepts_scored_kwargs"] is True
    assert DONOR_CHANNEL in report["live_channels"]
    assert report["oracle_calls"] == 0


def test_the_gate_fails_when_the_scored_hop_is_dropped(tmp_path, monkeypatch) -> None:
    """The gate's own negative control: a scored path that drops the key must not pass."""
    original = scored_gate.capture_scored_optimizer_kwargs

    def _stripped(root, folder, **arm):
        captured = original(root, folder, **arm)
        captured["optimizer_kwargs"] = {
            k: v for k, v in captured["optimizer_kwargs"].items() if "donor" not in k
        }
        return captured

    monkeypatch.setattr(scored_gate, "capture_scored_optimizer_kwargs", _stripped)
    with pytest.raises(AssertionError, match="never passed"):
        scored_gate.gate(ROOT, tmp_path / "gate")


# ---- The named arms ------------------------------------------------------


def test_the_production_default_is_the_retentive_cut_law() -> None:
    assert DEFAULT_CUT_LAW == RETENTIVE_RELEASED_FRACTION
    assert resolve_cut_law(DEFAULT_CUT_LAW) is donor_region_law


def test_the_uniform_arm_is_the_shipped_recipe_draw_by_name_and_by_behaviour() -> None:
    """The counterfactual arm must BE the shipped draw, not a lookalike.

    Two independent facts. Its NAME is the literal value `donor_memory.RECIPE` ships, so
    the arm cannot drift away from the recipe it is meant to reproduce. And it resolves to
    `None`, which is `donor_transplant_draw`'s unlawed `rng.permutation` branch -- a
    `BridgeRegionLaw(margin=None)` would reproduce the uniform SUPPORT while consuming
    `rng.random`, and so would not be the same draw.
    """
    assert UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM == donor_memory.RECIPE["cut_distribution"]
    assert resolve_cut_law(UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM) is None


def test_an_unknown_cut_law_is_refused_rather_than_defaulted() -> None:
    """A silent fallback would run the other arm under the asked-for arm's name."""
    for name in ("retentive", "uniform", ""):
        with pytest.raises(ValueError, match="unknown donor cut law"):
            resolve_cut_law(name)
    assert set(CUT_LAWS) == {RETENTIVE_RELEASED_FRACTION, UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM}


def test_the_controller_refuses_an_unknown_arm_at_construction(tmp_path) -> None:
    """At CONSTRUCTION, not at the first draw: a scored run must not get that far."""
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    with pytest.raises(ValueError, match="unknown donor cut law"):
        scored_gate._seeded_controller(
            dict(kwargs, donor_cut_law="retentive"),
            scored_gate._initialization_source(ROOT),
            seed=11,
        )


def _one_stratum(smiles: str, graph) -> list[DonorStratum]:
    """A single-member elite stratum: the smallest bank a draw can be made from."""
    return [DonorStratum(name=ELITE, weight=1.0, members=((smiles, graph),))]


# ---- THE SHARED SINK: the law is used TWICE and one use can be dropped ----


def test_the_law_is_consulted_for_the_DONOR_as_well_as_the_source() -> None:
    """Two hops share one sink, so a consultation gate alone cannot see this.

    `donor_transplant_draw` calls the law factory once for the parent and once per donor.
    A probe that raises fires on the FIRST call -- the parent -- so dropping the DONOR
    side leaves the consumption gate fully green while half the mechanism is gone. This
    is the exact near-miss that bit the region-law wiring: `segment_replace` kept
    threading the law after `substituent_delete` stopped, and only a behavioural test
    caught it.
    """
    source, donor = _state(PARENT), _state(DONOR)
    seen: list[int] = []

    def counting(graph):
        seen.append(int(graph.n_real_atoms))
        return donor_region_law(graph)

    donor_transplant_draw(
        source, _one_stratum(DONOR, donor), np.random.default_rng(7),
        law=counting, max_attempts=4,
    )
    assert int(source.n_real_atoms) in seen, "the law was never asked about the parent"
    assert int(donor.n_real_atoms) in seen, (
        "the law was asked about the parent but NOT about the donor; the donor cut is "
        "still drawn uniformly and the consumption gate cannot tell"
    )


def test_the_retentive_law_takes_a_smaller_graft_from_the_donor() -> None:
    """The behavioural half: the donor-side tilt must change what is INSTALLED.

    Distributional over fixed seeds rather than a single draw, because both arms share a
    support and differ only in probability -- one draw cannot separate them.
    """
    source, donor = _state(PARENT), _state(DONOR)
    pool = _one_stratum(DONOR, donor)

    def added(law):
        sizes = []
        for seed in range(40):
            proposal, _ = donor_transplant_draw(
                source, pool, np.random.default_rng([4242, seed]), law=law, max_attempts=4
            )
            if proposal is not None:
                sizes.append(proposal.added_atoms)
        return sizes

    retentive, uniform = added(donor_region_law), added(None)
    assert len(retentive) >= 10 and len(uniform) >= 10, "too few compiled draws to compare"
    assert float(np.mean(retentive)) < float(np.mean(uniform)), (
        f"retentive installed {np.mean(retentive):.2f} atoms against uniform's "
        f"{np.mean(uniform):.2f}; the donor-side tilt is not reaching the draw"
    )


def test_the_conversion_never_narrows_the_support() -> None:
    """Re-ranking, not filtering: every oriented single bridge stays drawable."""
    for smiles in (PARENT, DONOR):
        graph = _state(smiles)
        law = donor_region_law(graph)
        regions = [r for r in law.regions(graph) if r.bond_order == 1]
        assert len(regions) == len(pendant_cuts(graph))
        assert (law.weights(graph, law.regions(graph)) > 0).all()


# ---- A resume cannot silently change arms --------------------------------


def test_a_resume_cannot_swap_the_cut_law_under_a_running_arm(tmp_path) -> None:
    """The arms differ in PROBABILITY over a shared support, so a swap is invisible.

    A resumed run under the other law would look exactly like the arm it was asked for
    and measure the other one, with nothing in the artifact to show it.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    controller = scored_gate._seeded_controller(
        kwargs, scored_gate._initialization_source(ROOT), seed=11
    )
    snapshot = json.loads(json.dumps(controller.snapshot(include_history=True)))
    assert snapshot["pmo_population"]["donor_cut_law"] == RETENTIVE_RELEASED_FRACTION
    swapped = dict(kwargs, donor_cut_law=UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM)
    with pytest.raises(ValueError, match="donor cut law changed across resume"):
        PmoPopulationController.restore(snapshot, hierarchy=None, **swapped)


# ---- The launch hop: the worker selects the arm from the sealed payload ---


def test_the_worker_selects_the_donor_arm_from_the_sealed_payload() -> None:
    """Derived from the CALL SITE, not from a name written twice in two files.

    The Modal worker is the only thing that turns a sealed authorization into a runtime
    arm, and it is not importable here (it lives inside a Modal decorator and imports at
    call time). So the check parses the app, finds its ONE ``execute_task`` call, and
    requires the donor keyword to be present AND to be computed from the contract
    envelope's ``arm`` block -- never from the spawn spec, which a caller controls.
    """
    import ast

    tree = ast.parse((ROOT / "modal_apps/pmo_population_v1_app.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "execute_task"
    ]
    assert len(calls) == 1, f"expected one execute_task call site, found {len(calls)}"
    call = calls[0]
    text = ast.unparse(call)
    assert "enable_donor_channel" in text, (
        "the worker never selects the donor arm, so no sealed payload can turn it on "
        "and the lane is unreachable from a real launch"
    )
    donor = [
        segment
        for segment in text.splitlines()
        if "enable_donor_channel" in segment or "donor_cut_law" in segment
    ]
    assert donor, "donor keys vanished from the call site"
    assert '"arm"' in text or "'arm'" in text, (
        "the donor arm is not read from the contract envelope's arm block"
    )
    assert "spec" not in "".join(donor), (
        "the donor arm is selected from the spawn spec; a caller could then run a "
        "runtime the owner never authorized"
    )


# ---- Attribution: a donor proposal must be identifiable by a downstream ablation ----


def test_a_donor_candidate_is_tagged_at_synthesis_and_survives_to_the_snapshot(
    tmp_path,
) -> None:
    """The ablation reads `entry["provenance"]["metadata"]`; the tag must BE there.

    Two hops, and the second is the one that gets dropped: the lane writes the tag when it
    builds the candidate, and the archive entry the snapshot publishes has to still carry
    it. `pmo_ab_1k_checkpoints` attributes a frontier improvement by exactly
    `MEMORY_TAG in (entry["provenance"] or {}).get("metadata")`, so this channel is read
    the same way or it is invisible to the B-vs-C comparison.

    The tag also has to DISCRIMINATE: a key every entry carries attributes nothing.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    controller = scored_gate._seeded_controller(
        kwargs, scored_gate._initialization_source(ROOT), seed=11
    )
    schedule = [controller._parent() for _ in range(6)]
    _, candidates, _ = controller._generate_donor_pool(
        scored_gate._eligibility, set(), schedule
    )
    assert candidates, "the donor lane produced nothing to attribute"

    tag = candidates[0]["provenance"]["metadata"][DONOR_TAG]
    assert set(tag) >= {
        "donor",
        "endpoint",
        "retained_fraction",
        "removed_atoms",
        "added_atoms",
        "primitive_steps",
    }, f"the attribution payload lost fields: {sorted(tag)}"
    assert tag["donor"], "the tag must name the scored donor molecule"
    assert candidates[0]["provenance"]["planner_channel"] == DONOR_CHANNEL

    controller.add_measured_program(candidates[0], receipt_id="attribution", score=0.9)
    entries = controller.snapshot(include_history=True)["entries"]
    tagged = [
        entry_id
        for entry_id, entry in entries.items()
        if DONOR_TAG in ((entry.get("provenance") or {}).get("metadata") or {})
    ]
    assert len(tagged) == 1, (
        f"{len(tagged)} of {len(entries)} archive entries carry the donor tag; it must "
        "mark exactly the donor-derived one, or the ablation cannot separate this "
        "mechanism from broad exploration"
    )


def test_the_donor_tag_is_read_the_same_way_the_memory_tag_is() -> None:
    """Same key shape, same place, so one reader serves both channels."""
    from compose_v4.control.pmo_online_memory import SCHEMA  # noqa: F401

    assert DONOR_TAG == DONOR_CHANNEL
    assert isinstance(DONOR_TAG, str) and DONOR_TAG
