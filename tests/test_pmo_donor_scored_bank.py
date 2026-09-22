"""The donor lane's structural material: the run's own STRATIFIED scored bank.

WHAT IS BEING GUARDED, AND WHY EACH GUARD EXISTS
------------------------------------------------
The superseded donor pool was the 24 best-scoring molecules of the run.  This
repository measured blind PMO search leaving the drug-like manifold on almost every
task, so the top of the score ranking is the top of the drift and recombining it
recombines the drift.  The bank replaces that ranking with three disjoint strata --
score, demonstrated productivity as a PARENT, and structural coverage -- drawn
stratum-first so each one's mass is DECLARED rather than proportional to its size.

Every guard below is paired with a mutation in
``scripts/pmo_discovery_mutation_battery.py``.  A passing suite is not evidence that
its guards bite; the battery is.

INFORMATION REGIME.  Every quantity the bank holds is a counted observation of the run
in progress: a charged score, the difference of two charged scores, and the Bemis-Murcko
basin of a molecule the run already produced.  No declared target, no prescreened bank,
no cross-run history, and no molecular property computed off-ledger for selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.pmo_donor_channel import (
    DONOR_CHANNEL,
    DonorStratum,
    draw_stratified_donor,
)
from compose_v4.control.pmo_online_memory import (
    BANK_CAPACITY,
    DIVERSE,
    ELITE,
    ELITE_CAPACITY,
    PROMISING,
    STRATA,
    OnlineProposalMemory,
    ScoredMoleculeBank,
)
from compose_v4.control.pmo_population_controller import (
    DONOR_POOL_SIZE,
)
from compose_v4.experiments import pmo_donor_scored_gate as scored_gate
from compose_v4.experiments import pmo_population_v1

ROOT = Path(__file__).resolve().parents[1]

#: The pool size this bank supersedes. Named here rather than imported, because the
#: widening guard must compare against the value that actually shipped -- importing a
#: constant the same edit could move would let the two sides drift together.
SUPERSEDED_TOP_N_POOL = 24


def _bank(rows: list[tuple[str, float, str | None]]) -> ScoredMoleculeBank:
    bank = ScoredMoleculeBank()
    for endpoint, score, basin in rows:
        bank.observe(endpoint=endpoint, score=score, basin=basin)
    return bank


# ---- The selection ------------------------------------------------------


def test_the_bank_is_a_widening_and_never_a_narrowing() -> None:
    """Every molecule the superseded top-N pool offered is still offered.

    This is the property that makes the change a re-weighting rather than a filter: a
    difference between the arms cannot be one arm reaching material the other cannot.
    It holds because the ELITE stratum alone is larger than the whole superseded pool.
    """
    rows = [(f"M{i:04d}", 1.0 - i / 1000.0, f"basin{i % 37}") for i in range(600)]
    bank = _bank(rows)
    top_n = [endpoint for endpoint, _, _ in rows[:SUPERSEDED_TOP_N_POOL]]

    selected = set().union(*bank.strata(capacity=BANK_CAPACITY).values())
    missing = [endpoint for endpoint in top_n if endpoint not in selected]
    assert not missing, (
        f"{len(missing)} molecules the superseded top-{SUPERSEDED_TOP_N_POOL} pool "
        "offered are absent from the stratified bank; the change would be a filter, "
        "not a re-weighting, and an arm difference could be pure reach"
    )
    assert ELITE_CAPACITY >= SUPERSEDED_TOP_N_POOL


def test_the_strata_are_disjoint_and_precedence_decides_ties() -> None:
    """A molecule that qualifies twice occupies ONE slot, in the declared order.

    Disjointness is load-bearing because the draw picks a stratum and then a member: a
    molecule listed in two strata would receive two shares of the declared mass.
    """
    bank = _bank([("top", 0.9, "a"), ("mid", 0.4, "b"), ("low", 0.1, "c")])
    # `top` is both the best scorer AND a productive parent.
    bank.observe_lineage(parent_endpoint="top", parent_score=0.9, child_score=0.99)
    bank.observe_lineage(parent_endpoint="mid", parent_score=0.4, child_score=0.7)

    strata = bank.strata(capacity=3)
    everything = [e for name in STRATA for e in strata[name]]
    assert len(everything) == len(set(everything)), f"strata overlap: {strata}"
    assert "top" in strata[ELITE] and "top" not in strata[PROMISING], (
        "precedence broke: a molecule in the elite stratum is also drawing promising mass"
    )
    assert "mid" in strata[PROMISING]


def test_the_diverse_stratum_covers_a_basin_the_higher_strata_do_not() -> None:
    """Coverage, not ranking: the stratum exists to hold structure score discards.

    The fixture is the case the motivation describes -- a run whose high scorers have
    converged on one scaffold while a lower-scoring molecule holds a different one.
    """
    # The converged basin must have a member left OVER after the elite fills, or the
    # "already occupied" test is vacuous: with every member of that basin taken, any
    # implementation returns the same answer. `drifted_d` is that leftover, and it
    # OUTSCORES the molecule the stratum is supposed to pick.
    bank = _bank(
        [
            ("drifted_a", 0.90, "converged"),
            ("drifted_b", 0.89, "converged"),
            ("drifted_c", 0.88, "converged"),
            ("drifted_d", 0.50, "converged"),  # left over, and outscores `other_basin`
            ("other_basin", 0.05, "elsewhere"),  # the only molecule of its scaffold
        ]
    )
    strata = bank.strata(capacity=9)  # 3 slots per stratum
    assert strata[ELITE] == ["drifted_a", "drifted_b", "drifted_c"]
    assert strata[DIVERSE] == ["other_basin"], (
        f"the diverse stratum selected {strata[DIVERSE]}; it must represent a basin the "
        "higher strata do not already occupy, never a second member of one they do"
    )


def test_the_promising_stratum_is_evidence_about_a_molecule_as_a_SOURCE() -> None:
    """Counted improvement DELIVERED BY ITS OWN CHILDREN, not its own score.

    Two molecules with the same score and the same basin; only one has a child that
    improved on it.  The other must not be selected, or the stratum is measuring reuse
    rather than productivity.
    """
    # The stratum must have ROOM to spare, or an admit-everything predicate is
    # invisible: with one slot the positive-improvement molecule still ranks first and
    # the extra admissions never appear in the answer. Three slots and two unproductive
    # molecules make the difference observable.
    bank = _bank(
        [
            ("elite_a", 0.95, "top"),
            ("elite_b", 0.94, "top"),
            ("elite_c", 0.93, "top"),
            ("productive", 0.10, "shared"),
            ("barren", 0.20, "shared"),  # outscores `productive`
            ("barren_too", 0.15, "shared"),
        ]
    )
    bank.observe_lineage(parent_endpoint="productive", parent_score=0.10, child_score=0.40)
    # Used as parents just as often, and their children were WORSE.
    bank.observe_lineage(parent_endpoint="barren", parent_score=0.20, child_score=0.02)
    bank.observe_lineage(parent_endpoint="barren_too", parent_score=0.15, child_score=0.01)

    strata = bank.strata(capacity=9)  # 3 slots per stratum, so two go spare
    assert strata[PROMISING] == ["productive"], (
        f"promising selected {strata[PROMISING]}; a parent whose child did not improve "
        "carries no evidence that its structural material is productive"
    )


def test_a_parent_the_ledger_never_charged_cannot_enter_the_bank() -> None:
    """A bank row must be a molecule this run paid for.

    `observe_lineage` refuses to CREATE a row.  Inventing one from a provenance field
    would put an unscored molecule into a pool whose whole claim is that every member
    was counted -- and would then offer it as donor material.
    """
    bank = _bank([("scored", 0.5, "a")])
    assert bank.observe_lineage(
        parent_endpoint="never_scored", parent_score=0.1, child_score=0.9
    ) is False
    assert "never_scored" not in bank.rows


def test_the_bank_refuses_a_changed_counted_score() -> None:
    """Two counted values for one endpoint means the ledger join is wrong."""
    bank = _bank([("m", 0.5, "a")])
    bank.observe(endpoint="m", score=0.5, basin="a")  # idempotent
    with pytest.raises(ValueError, match="counted score"):
        bank.observe(endpoint="m", score=0.6, basin="a")


def test_the_selection_does_not_depend_on_observation_ORDER() -> None:
    """A selection that moved with dict insertion order would not survive a resume."""
    rows = [(f"M{i}", (i * 37 % 101) / 101.0, f"b{i % 11}") for i in range(200)]
    forward = _bank(rows).strata()
    backward = _bank(list(reversed(rows))).strata()
    assert forward == backward


# ---- The draw -----------------------------------------------------------


def test_the_draw_gives_each_stratum_its_DECLARED_mass_not_its_size_share() -> None:
    """Stratum-first, which is the entire reason `DonorStratum` exists.

    A uniform draw over the union would hand the elite slice mass in proportion to its
    SIZE.  The fixture makes the two unmistakable: 100 elite members against one diverse
    member.  Size-proportional would draw the diverse member ~1% of the time; declared
    shares draw it ~50%, because those are the two non-empty strata.
    """
    strata = [
        DonorStratum(ELITE, 0.5, tuple((f"E{i}", object()) for i in range(100))),
        DonorStratum(DIVERSE, 0.5, (("D0", object()),)),
    ]
    rng = np.random.default_rng(20260921)
    hits = sum(draw_stratified_donor(strata, rng)[1] == "D0" for _ in range(4000))
    assert 0.40 < hits / 4000 < 0.60, (
        f"the single diverse member was drawn {hits / 4000:.3f} of the time; at the "
        "declared 1/2 share it should be ~0.5, and a flat draw over the union would "
        "give ~0.01 -- the draw is size-proportional, not stratified"
    )


def test_an_empty_stratum_yields_its_mass_rather_than_shrinking_the_draw() -> None:
    bank = _bank([("a", 0.5, "x"), ("b", 0.4, "x")])
    strata = bank.strata(capacity=6)
    assert not strata[PROMISING] and not strata[DIVERSE]
    weights = bank.weights(strata)
    assert weights == {ELITE: 1.0}
    assert draw_stratified_donor(
        [DonorStratum(ELITE, 1.0, ()), DonorStratum(DIVERSE, 1.0, (("d", object()),))],
        np.random.default_rng(1),
    )[0] == DIVERSE


def test_nothing_offered_is_None_and_not_a_crash() -> None:
    assert draw_stratified_donor([], np.random.default_rng(0)) is None
    assert draw_stratified_donor(
        [DonorStratum(ELITE, 1.0, ())], np.random.default_rng(0)
    ) is None


# ---- The update hops ----------------------------------------------------


def test_the_lineage_is_banked_even_when_the_edit_cannot_be_attributed() -> None:
    """Recorded ABOVE the attribution early-return, and that placement is the guard.

    `observe_transition` gives up when no region overlaps the slots the edit touched,
    and returns early.  The lineage evidence is valid regardless -- the parent produced a
    better child whether or not the edit localizes to a region -- so banking it after
    that return would silently drop the promising stratum's evidence for exactly the
    transitions that are hardest to attribute.
    """
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )

    memory = OnlineProposalMemory()
    parent = production_state_from_smiles("CC(=O)Nc1ccc(OCC)cc1", max_atoms=48)
    memory.observe_scored_molecule(endpoint="parent", score=0.2, basin="b")
    attributed = memory.observe_transition(
        parent_graph=parent,
        parent_endpoint="parent",
        parent_score=0.2,
        child_endpoint="child",
        child_score=0.8,
        child_heavy=int(parent.n_real_atoms),
        family="atom_restate_semantic",
        touched_slots=(),  # unattributable by construction
    )
    assert attributed is False, "the fixture must exercise the UNATTRIBUTED path"
    assert memory.bank.rows["parent"]["improvement"] == pytest.approx(0.6), (
        "an unattributable transition lost its lineage evidence; the bank call sits "
        "below the attribution early-return"
    )


def test_a_bootstrap_scored_molecule_reaches_the_bank_AND_NOTHING_ELSE(tmp_path) -> None:
    """The campaign charges bootstrap candidates outside `observe_batch`.

    Two halves, and both matter.  The bank MUST see them: under the cold-start PMO
    recipe that is every initialization molecule plus roughly a fifth of later rounds,
    and the initialization molecules are the on-manifold material the diverse stratum
    exists to keep available.  Nothing else may move: `frontier`, `edits`, `donors` and
    `size` feed arm B's region law and arm C's frontier credit, both of which are
    already running scored.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    controller = scored_gate._seeded_controller(
        kwargs, scored_gate._initialization_source(ROOT), seed=11
    )
    memory = controller.online_memory
    assert memory.bank.rows, (
        "the bootstrap adds never reached the bank, so the donor lane starts from an "
        "empty pool on every cold-start run"
    )
    assert not memory.frontier.scores, (
        "a bootstrap add moved the FRONTIER; arm B's memory and arm C's frontier credit "
        "are not byte-identical any more, and both are running scored"
    )
    assert not memory.edits.counts and not memory.donors.counts and not memory.size.counts


def test_the_bank_rides_the_resume_and_an_old_payload_restores_EMPTY() -> None:
    """Round trip, plus the refusal that matters more than the round trip.

    A payload written before the bank existed carries none.  Rebuilding it from the
    frontier would be silently WRONG rather than merely partial: the frontier holds no
    basin and no lineage, so every row would land in the elite stratum and the resumed
    run would draw from a top-N pool under the stratified arm's name.
    """
    memory = OnlineProposalMemory()
    memory.observe_scored_molecule(endpoint="m1", score=0.5, basin="x")
    memory.observe_scored_molecule(endpoint="m2", score=0.1, basin="y")
    memory.bank.observe_lineage(parent_endpoint="m2", parent_score=0.1, child_score=0.4)

    restored = OnlineProposalMemory()
    restored.restore_payload(json.loads(json.dumps(memory.payload())))
    assert restored.bank.rows == memory.bank.rows
    assert restored.bank.strata() == memory.bank.strata()

    legacy = json.loads(json.dumps(memory.payload()))
    legacy.pop("bank")
    old = OnlineProposalMemory()
    old.restore_payload(legacy)
    assert old.bank.rows == {}, (
        "a pre-bank payload reconstructed a bank; every row would be basinless and "
        "lineage-free, which is a top-N pool wearing the stratified arm's name"
    )
    assert old.frontier.scores, "the rest of the memory must still restore"


# ---- The controller hop -------------------------------------------------


def test_the_donor_pool_IS_the_bank_selection_and_not_a_score_ranking(tmp_path) -> None:
    """The join from banked endpoint to archived 48-slot state, checked end to end.

    The pool must present the bank's own strata, in the bank's own order.  A pool that
    re-sorted by score would run the superseded ranking behind the new names.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    controller = scored_gate._seeded_controller(
        kwargs, scored_gate._initialization_source(ROOT), seed=11
    )
    # Give the bank material score alone would rank differently. The elite capacity has
    # to BIND for that to be visible, so the bank is filled with counted rows the archive
    # cannot build -- they occupy the elite stratum and the shared basin, and the join
    # then drops them, which is also the behaviour a malformed archive row must get.
    endpoints = [entry["endpoint"] for _, entry in sorted(controller.entries.items())]
    assert len(endpoints) >= 3
    bank = controller.online_memory.bank
    for filler in range(ELITE_CAPACITY):
        bank.observe(endpoint=f"unbuildable{filler}", score=1.0, basin="shared")
    for index, endpoint in enumerate(endpoints):
        bank.rows[endpoint]["score"] = 0.9 - 0.1 * index
        bank.rows[endpoint]["basin"] = "shared"
    bank.rows[endpoints[-1]]["basin"] = "lonely"
    bank.observe_lineage(
        parent_endpoint=endpoints[-2], parent_score=0.1, child_score=0.9
    )

    selection = bank.strata(capacity=DONOR_POOL_SIZE)
    strata = controller._donor_pool(exclude=None)
    assert [s.name for s in strata] == list(STRATA)
    offered = {s.name: [smiles for smiles, _ in s.members] for s in strata}
    for name in STRATA:
        assert offered[name] == [e for e in selection[name] if e in set(endpoints)], (
            f"the {name} stratum the pool offered is not the bank's own selection: "
            f"{offered[name]} vs {selection[name]}"
        )
    assert offered[PROMISING] == [endpoints[-2]]
    assert offered[DIVERSE] == [endpoints[-1]]
    for stratum in strata:
        for _, graph in stratum.members:
            assert len(graph.atom_types) == 48, "PMO donor states must be 48-slot"


def test_a_donor_candidate_names_the_stratum_its_donor_came_from(tmp_path) -> None:
    """Attribution an ablation can act on, written at SYNTHESIS time.

    A transplant endpoint does not determine which molecule the graft came from, still
    less which stratum, so this cannot be reconstructed afterwards.
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
    tag = candidates[0]["provenance"]["metadata"][DONOR_CHANNEL]
    assert tag["donor_stratum"] in STRATA, (
        f"the tag names stratum {tag.get('donor_stratum')!r}, which is not one of "
        f"{STRATA}; an ablation cannot ask whether elite, promising and diverse donors "
        "behave differently"
    )


# ---- Arm D is arm B plus the donor lane ---------------------------------


def test_arm_D_without_arm_B_is_REFUSED_at_the_controller(tmp_path) -> None:
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    with pytest.raises(ValueError, match="requires the online memory"):
        scored_gate._seeded_controller(
            dict(kwargs, enable_online_memory=False),
            scored_gate._initialization_source(ROOT),
            seed=11,
        )


def test_arm_D_without_arm_B_is_REFUSED_before_the_ledger_is_built(tmp_path) -> None:
    """Before the first charged call, so a misconfigured launch stays re-runnable."""
    with pytest.raises(ValueError, match="arm D requires the online memory"):
        pmo_population_v1.execute_task(
            scored_gate._runtime_contract(ROOT),
            ROOT,
            tmp_path / "bad",
            "celecoxib_rediscovery",
            evaluate=scored_gate._evaluate_must_not_be_called,
            charged_calls_per_task=250,
            enable_online_memory=False,
            enable_donor_channel=True,
        )


def test_the_promising_stratum_is_reachable_from_a_REAL_propose_observe_cycle(
    tmp_path,
) -> None:
    """The stratum is fed by the PRODUCTION loop, not only by a hand-seeded lineage.

    This is the guard that matters most here, and it caught a real defect.  Every other
    promising test constructs its lineage by calling `observe_lineage` directly, and all
    of them passed while the production path banked NOTHING: `_parent` writes provenance
    `{entry_id, parent_probability, parent_measured_score}` and no `parent_endpoint` --
    only the v22 optimizer writes that, and this controller descends from v21 -- so the
    lookup resolved to the empty string on every counted transition and the stratum was
    permanently empty in production.

    So the cycle is driven for real: propose a batch through the production
    `propose_batch`, charge every candidate a score that beats its parent, and require the
    bank to have learned that those parents are productive.
    """
    kwargs = scored_gate.capture_scored_optimizer_kwargs(
        ROOT, tmp_path / "on", enable_online_memory=True, enable_donor_channel=True
    )["optimizer_kwargs"]
    controller = scored_gate._seeded_controller(
        kwargs, scored_gate._initialization_source(ROOT), seed=11
    )
    batch = controller.propose_batch(scored_gate._eligibility)
    assert batch["candidates"], "the production proposal produced nothing to charge"
    controller.observe_batch(
        batch["batch_id"],
        [
            {
                "candidate_id": row["candidate_id"],
                "receipt_id": f"promising{index}",
                "endpoint": row["endpoint"],
                "score": 0.99,  # every child beats its 0.5 parent
                "oracle_protocol": controller.oracle_protocol,
            }
            for index, row in enumerate(batch["candidates"])
        ],
    )
    bank = controller.online_memory.bank
    productive = {e: r for e, r in bank.rows.items() if r["improvement"] > 0.0}
    assert productive, (
        "a full production propose/observe cycle in which EVERY child improved on its "
        "parent left the bank with no productive parent at all; the promising stratum "
        "is unreachable from the scored loop and is inert however well it is tested"
    )
    # And the parents it learned are real archived molecules, not a placeholder key.
    assert set(productive) <= {
        entry["endpoint"] for entry in controller.entries.values()
    }, f"the bank credited improvement to endpoints outside the archive: {set(productive)}"
