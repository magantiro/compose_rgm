"""Invariants of the T4-v2 feasibility-conditioned proposal law.

The properties tested here are the ones a wrong result would hide behind: that the
headroom vector agrees with the production gate, that the move vocabulary preserves net
formal charge and connectivity, that the padded-slot semantics the repo's known failure
mode turns on are actually exercised, and that the reported eligible pool is the
production ``Fiber``'s verdict rather than this module's reconstruction of it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import t4_v2_feasibility_proposal as law

#: A small drug-like molecule, not one of the T4 seeds: the law must not be tuned to them.
PROBE = "CC(=O)Nc1ccc(O)cc1"
#: A charged, ringed, stereocentre-carrying probe -- the three features that broke earlier
#: editing work in this repo.
HARD_PROBE = "C[C@H](N)C(=O)N1CCC[C@H]1C(=O)[O-]"


def _mol(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_slot_preflight_distinguishes_tight_from_padded():
    """A TIGHT graph silently deletes the atom_insert family; the padded one must not."""

    context = law.FeasibilityContext(PROBE, 0.6)
    evidence = context.slot_preflight()
    assert evidence["passed"]
    assert evidence["tight_graph_would_forbid_atom_birth"] is True
    assert evidence["padded_graph_slots"] == law.T4_PROPOSAL_SLOTS
    assert evidence["free_slots_in_padded_source"] > 0


def test_headroom_margins_agree_with_the_thresholds():
    context = law.FeasibilityContext(PROBE, 0.6)
    row = context.evaluate(Chem.MolToSmiles(_mol(PROBE)))
    head = row["headroom"]
    assert head.m_qed == pytest.approx(head.qed - law.QED_MIN)
    assert head.m_sa == pytest.approx(law.SA_MAX - head.sa)
    assert head.m_sim == pytest.approx(head.similarity - 0.6)
    assert head.z_qed == pytest.approx(head.m_qed / law.Z_SCALE["qed"])
    # The source is its own fingerprint, so similarity is exactly 1.
    assert head.similarity == pytest.approx(1.0)
    assert head.feasible() == (min(head.vector()) >= 0)


def test_feasibility_is_exactly_the_min_margin_being_nonnegative():
    """The scalarisation must not be able to call an infeasible state feasible."""

    context = law.FeasibilityContext(PROBE, 0.6)
    rng = random.Random(0)
    checked = 0
    for smiles, _ in law.enumerate_mode(context.source_mol, "replace", rng)[:60]:
        row = context.evaluate(smiles)
        if row is None or not row.get("reached_gate"):
            continue
        head = row["headroom"]
        capped_min = min(min(value, law.SLACK_CAP) for value in head.vector())
        assert head.feasible() == (capped_min >= 0)
        checked += 1
    assert checked > 5


def test_eligible_rows_are_the_production_fiber_verdict():
    """Every row is cross-checked against ``Fiber.check``; disagreement is a hard failure."""

    result = law.run_search(PROBE, 0.4, budget=400, seed=3)
    assert result["gate_reconstruction_disagreements"] == 0
    for row in result["eligible"]:
        assert result_fiber_agrees(row["smiles"])


def result_fiber_agrees(smiles: str) -> bool:
    from compose_v4.experiments.t4_fiber_campaign import Fiber

    return bool(Fiber(PROBE, 0.4, support="compose_valid").check(smiles))


@pytest.mark.parametrize("probe", [PROBE, HARD_PROBE])
def test_moves_preserve_net_charge_and_connectivity(probe):
    molecule = _mol(probe)
    before = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    rng = random.Random(1)
    produced = 0
    for mode in law.MODES:
        for smiles, meta in law.enumerate_mode(molecule, mode, rng):
            candidate = Chem.MolFromSmiles(smiles)
            assert candidate is not None, smiles
            assert "." not in smiles, smiles
            assert meta["mode"] == mode
            if mode != "prune":
                after = sum(atom.GetFormalCharge() for atom in candidate.GetAtoms())
                assert after == before, f"{mode} moved net charge on {smiles}"
            produced += 1
    assert produced > 20


def test_prune_never_returns_a_fragment_below_the_floor():
    molecule = _mol("CC(C)CCN(Cc1ccccc1)C(=O)c1ccccc1")
    for smiles, meta in law.prune_moves(molecule, random.Random(0)):
        candidate = Chem.MolFromSmiles(smiles)
        assert candidate.GetNumHeavyAtoms() >= law.MIN_FRAGMENT_HEAVY
        assert meta["atoms_touched"] >= 1


def test_prune_offers_both_sides_of_a_cut():
    """WHICH region is kept is the decision the attribution says loses the mass."""

    molecule = _mol("c1ccccc1CCCCc1ccncc1")
    produced = {smiles for smiles, _ in law.prune_moves(molecule, random.Random(0))}
    keeps_benzene = any(
        Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1"))
        and not Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts("c1ccncc1"))
        for smiles in produced
    )
    keeps_pyridine = any(
        Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts("c1ccncc1"))
        for smiles in produced
    )
    assert keeps_benzene and keeps_pyridine


def test_move_utility_rewards_the_binding_constraint_only():
    """Slack in a satisfied axis must not pay for a violation in another."""

    context = law.FeasibilityContext(PROBE, 0.6)

    def head(qed, sa, sim, size):
        return law.Headroom(
            qed=qed, sa=sa, similarity=sim, heavy=size, rings=1, aromatic_rings=1,
            rotatable=1, stereocentres=0, retained_bit_fraction=1.0,
            m_qed=qed - 0.6, m_sa=4.0 - sa, m_sim=sim - 0.6, m_size=40.0 - size,
            z_qed=(qed - 0.6) / law.Z_SCALE["qed"], z_sa=(4.0 - sa) / law.Z_SCALE["sa"],
            z_sim=(sim - 0.6) / law.Z_SCALE["sim"], z_size=(40.0 - size) / law.Z_SCALE["size"],
        )

    parent = head(0.30, 2.0, 0.90, 30)          # QED binds hard
    relieves = head(0.45, 2.0, 0.85, 26)        # buys QED, spends a little similarity
    buys_slack = head(0.30, 1.0, 0.95, 30)      # buys only SA and similarity slack
    assert law.move_utility(parent, relieves) > law.move_utility(parent, buys_slack)
    assert law.move_utility(parent, buys_slack) == pytest.approx(0.0, abs=0.06)
    assert context.delta == 0.6


def test_freezing_an_axis_removes_it_from_the_context_and_the_utility():
    # Both states are held INFEASIBLE on SA so the feasibility bonus cannot fire and the
    # test reads the scalarisation itself rather than the attainment term.
    def head(qed):
        return law.Headroom(
            qed=qed, sa=5.0, similarity=0.9, heavy=30.0, rings=1, aromatic_rings=1,
            rotatable=1, stereocentres=0, retained_bit_fraction=1.0,
            m_qed=qed - 0.6, m_sa=-1.0, m_sim=0.3, m_size=10.0,
            z_qed=(qed - 0.6) / law.Z_SCALE["qed"], z_sa=-1.0, z_sim=1.5, z_size=1.25,
        )

    low, high = head(0.20), head(0.90)
    assert low.context() != high.context()
    assert low.context(("qed",)) == high.context(("qed",))
    assert law.move_utility(low, high, frozen=("qed",)) == pytest.approx(0.0)
    assert law.move_utility(low, high) > 0.5


def test_policy_without_conditioning_collapses_to_one_context():
    conditioned = law.ProposalPolicy(conditioned=True)
    blind = law.ProposalPolicy(conditioned=False)
    rng = random.Random(0)
    context = law.FeasibilityContext(PROBE, 0.6)
    row = context.evaluate(Chem.MolToSmiles(context.source_mol))
    head = row["headroom"]
    assert blind.context_of(head) == "*"
    assert conditioned.context_of(head) != "*"
    for _ in range(20):
        blind.sample(head, rng)
        conditioned.sample(head, rng)
    # However many distinct headroom signatures it meets, the blind policy has one key.
    assert {key[0] for key in blind.stats} == {"*"}
    assert {key[0] for key in conditioned.stats} == {conditioned.context_of(head)}
    blind.update("*", "prune", 0, 1.0)
    assert blind.stats[("*", "prune", 0)].pulls == 1


def test_search_is_reproducible_under_a_seed():
    first = law.run_search(PROBE, 0.5, budget=250, seed=7)
    second = law.run_search(PROBE, 0.5, budget=250, seed=7)
    assert [row["smiles"] for row in first["rows"]] == [row["smiles"] for row in second["rows"]]


def test_removable_arms_rank_by_qed_bought_per_similarity_spent():
    arms = law.removable_arms("CC(C)CCN(Cc1ccc2ccccc2c1)C(=O)c1ccccc1", 0.6)
    assert arms
    ratios = [row["qed_gain_per_similarity_cost"] for row in arms]
    assert ratios == sorted(ratios, reverse=True)
    for row in arms:
        assert 0.0 <= row["similarity"] <= 1.0
        assert row["removed_heavy"] >= 1


def test_the_law_names_no_target_no_cell_and_no_seed_molecule():
    """A fix that only works on five identifiers is worthless; make that checkable.

    The proposal law must see a SMILES, a delta and three thresholds -- never a target, a
    cell, or a molecule from the panel.  This scans its source for every T4 seed SMILES
    and every target name in the campaign, and fails if any appears.
    """

    import json

    source = (REPO / "scripts/t4_v2_feasibility_proposal.py").read_text()
    audit = json.loads((REPO / "diagnostics/t4_support_stage_audit_v1.json").read_text())
    offenders = []
    for cell, payload in audit["cells"].items():
        seed = payload["slot_semantics_preflight"]["smiles"]
        if seed in source:
            offenders.append(f"seed molecule of {cell}")
        if cell in source:
            offenders.append(f"cell identifier {cell}")
    for target in {cell.rsplit("_", 1)[0] for cell in audit["cells"]}:
        if target in source.lower():
            offenders.append(f"target identifier {target}")
    assert not offenders, offenders


def test_every_arm_and_freeze_the_report_claims_is_actually_run():
    """The ablation table must not be able to describe an arm the driver never ran."""

    import t4_v2_feasibility_gate as gate

    assert set(gate.ARMS) == {
        "v2_full",
        "v2_blind_proposal",
        "v2_unguided_selection",
        "v2_moveset_only",
    }
    assert gate.ARMS["v2_full"] == {"conditioned_proposal": True, "guided_selection": True}
    assert gate.ARMS["v2_moveset_only"] == {
        "conditioned_proposal": False,
        "guided_selection": False,
    }
    # Every freeze names a real axis of the headroom vector.
    context = law.FeasibilityContext(PROBE, 0.6)
    row = context.evaluate(Chem.MolToSmiles(context.source_mol))
    assert set(gate.FREEZES) <= set(row["headroom"].z())
    assert set(gate.FAILED) & set(gate.CONTROLS) == set()


# ---- Narrow chemistry-sanity filter ----

#: Motifs no chemist would make, one per SMARTS in the table.  Each is a heteroatom-halogen
#: bond or a halogenated imine carbon.
PATHOLOGICAL = (
    ("CC(=O)OF", "O-F"),
    ("CC(=O)OCl", "O-Cl"),
    ("c1ccccc1OF", "O-F"),
    ("CCN(F)CC", "N-F"),
    ("CCN(Cl)C", "N-Cl"),
    ("CCN(Br)C", "N-Br"),
    ("CCOBr", "O-Br"),
    ("CCSF", "S-F"),
    ("CCSCl", "S-Cl"),
    ("CC(=N)F", "C(=N)-halogen"),
    ("CC(=NC)Cl", "C(=N)-halogen"),
)

#: Ordinary medicinal chemistry that must survive.  Every entry carries a halogen, so a
#: filter that merely banned F or Cl would fail this list outright.
ORDINARY_HALOGEN_CHEMISTRY = (
    "Fc1ccccc1",
    "FC(F)(F)c1ccccc1",
    "CCF",
    "Clc1ccccc1",
    "Brc1ccccc1",
    "Ic1ccccc1",
    "CCCCl",
    "CC(=O)Cl",
    "Oc1ccc(F)cc1",
    "FC(F)Oc1ccccc1",
    "O=c1[nH]cc(F)c(=O)[nH]1",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
)


@pytest.mark.parametrize(("smiles", "motif"), PATHOLOGICAL)
def test_pathological_heteroatom_halogen_motifs_are_named(smiles, motif):
    assert motif in law.pathological_motifs(_mol(smiles))
    assert not law.chemistry_sane(_mol(smiles))


@pytest.mark.parametrize("smiles", ORDINARY_HALOGEN_CHEMISTRY)
def test_ordinary_halogen_chemistry_is_untouched(smiles):
    """The filter is not a halogen ban: every probe here carries F, Cl, Br or I."""

    molecule = _mol(smiles)
    assert any(atom.GetSymbol() in {"F", "Cl", "Br", "I"} for atom in molecule.GetAtoms())
    assert law.pathological_motifs(molecule) == ()
    assert law.chemistry_sane(molecule)


def test_every_declared_smarts_compiles_and_is_exercised():
    """A table entry nothing tests is a table entry that can rot."""

    declared = {name for name, _ in law.PATHOLOGICAL_MOTIF_SMARTS}
    assert len(declared) == len(law.PATHOLOGICAL_MOTIF_SMARTS)
    assert all(pattern is not None for _, pattern in law._PATHOLOGICAL_PATTERNS)
    assert declared == {motif for _, motif in PATHOLOGICAL}


def test_the_filter_narrows_the_proposer_and_never_the_benchmark_gate():
    """``eligible`` may shrink; the Fiber verdict recorded beside it must not move."""

    unfiltered = law.run_search(PROBE, 0.6, budget=160, seed=3, chemistry_filter=False)
    filtered = law.run_search(PROBE, 0.6, budget=160, seed=3, chemistry_filter=True)
    for result in (unfiltered, filtered):
        assert result["gate_reconstruction_disagreements"] == 0
        for row in result["rows"]:
            if not row.get("reached_gate") or "benchmark_eligible" not in row:
                continue
            # The published gate is sim/QED/SA plus the production med-chem gate, and the
            # filter is applied strictly after it -- never instead of it.
            assert row["benchmark_eligible"] == (
                row["sim_ok"] and row["qed_ok"] and row["sa_ok"] and row["med_chem_ok"]
            )
            if row["eligible"]:
                assert row["benchmark_eligible"]


def test_a_filtered_run_emits_no_pathological_endpoint():
    result = law.run_search(PROBE, 0.6, budget=240, seed=5, chemistry_filter=True)
    assert all(law.chemistry_sane(_mol(row["smiles"])) for row in result["eligible"])


def test_disabling_the_filter_reproduces_the_unfiltered_pool_exactly():
    """Filter OFF must be behaviour-preserving, so the baseline stays comparable."""

    first = law.run_search(PROBE, 0.6, budget=200, seed=7, chemistry_filter=False)
    second = law.run_search(PROBE, 0.6, budget=200, seed=7, chemistry_filter=False)
    assert [row["smiles"] for row in first["eligible"]] == [
        row["smiles"] for row in second["eligible"]
    ]
    assert first["gate_calls"] == second["gate_calls"]
