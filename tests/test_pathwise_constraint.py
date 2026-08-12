"""Exact labeled-subgraph preservation: the pathwise constraint predicate.

The tests that matter here are the NEGATIVE ones. A predicate that always
returns True would make every pathwise arm trivially feasible and every
violation count zero, which is exactly the "statistic whose sign was fixed
before any data existed" failure this workstream is required to avoid. So each
positive case is paired with a mutation that must be rejected.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.experiments.pathwise_constraints import (
    HEAVY_ATOM_BAND,
    MOTIF_RULE_VERSION,
    derive_protected_motif,
    eligibility,
    fragment_smarts,
    mask_successors,
    motif_embedding_count,
    path_violation_summary,
    preserves_motif,
    ring_systems,
)

REPO = Path(__file__).resolve().parents[1]
RESERVE = REPO / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz"

# A 4-aminoquinazoline-like source: fused bicyclic core plus a real tail.
FUSED = "Cc1ccccc1-c1noc(-c2cc3ccccc3oc2=O)n1"
PIPERAZINE = "CCN1CCN(c2ccccc2)CC1"
STEROID = "CC(=O)CCC(C)C1CCC2C3=C(CCC21C)C1(C)CCC(=O)C(C)C1CC3=O"


# ---------------------------------------------------------------- ring systems


def test_ring_systems_are_fused_components():
    mol = Chem.MolFromSmiles(FUSED)
    systems = sorted(len(s) for s in ring_systems(mol))
    # benzene(6), oxadiazole(5), coumarin bicycle(10)
    assert systems == [5, 6, 10]


def test_acyclic_source_has_no_motif():
    assert derive_protected_motif("CCCCCCO") is None


def test_unparseable_source_has_no_motif():
    assert derive_protected_motif("not-a-smiles((") is None


# ---------------------------------------------------------------- derivation


def test_largest_ring_system_is_selected():
    motif = derive_protected_motif(FUSED)
    assert motif is not None
    assert motif.atom_count == 10  # the fused bicycle, not the lone benzene
    assert motif.ring_count == 2
    assert motif.bond_count == 11
    assert motif.rule == MOTIF_RULE_VERSION


def test_derivation_is_deterministic_across_input_spelling():
    """The rule must not depend on how the source SMILES happens to be written."""
    mol = Chem.MolFromSmiles(FUSED)
    a = derive_protected_motif(Chem.MolToSmiles(mol))
    b = derive_protected_motif(Chem.MolToSmiles(mol, canonical=False))
    assert a is not None and b is not None
    assert a.atom_count == b.atom_count
    # the SMARTS strings may be written from different roots, but each must
    # accept exactly the states the other accepts on this molecule
    assert preserves_motif(a.smarts, FUSED)
    assert preserves_motif(b.smarts, FUSED)


def test_motif_geometry_fields():
    motif = derive_protected_motif(FUSED)
    assert motif.source_heavy_atoms == 23
    assert motif.free_atoms == 13
    assert motif.fraction == pytest.approx(10 / 23)


# ------------------------------------------------- the predicate: POSITIVES


def test_source_preserves_its_own_motif():
    motif = derive_protected_motif(FUSED)
    assert preserves_motif(motif.smarts, FUSED)


def test_substitution_outside_the_motif_is_allowed():
    """The room to act: decorating non-motif atoms must NOT violate."""
    motif = derive_protected_motif(PIPERAZINE)
    assert motif.atom_count == 6  # the benzene
    # add a chlorine on the benzene ring -- ring atoms keep element/aroma/charge
    assert preserves_motif(motif.smarts, "CCN1CCN(c2ccccc2Cl)CC1")
    # extend the ethyl tail
    assert preserves_motif(motif.smarts, "CCCCN1CCN(c2ccccc2)CC1")


def test_deleting_the_rest_of_the_molecule_is_allowed():
    """Only the motif is protected; everything else may go."""
    motif = derive_protected_motif(PIPERAZINE)
    assert preserves_motif(motif.smarts, "c1ccccc1")


# ------------------------------------------------- the predicate: NEGATIVES


def test_ring_opening_violates():
    motif = derive_protected_motif(PIPERAZINE)
    # open the benzene ring into an open chain: the closure bond is gone
    assert not preserves_motif(motif.smarts, "CCN1CCN(C=CC=CC=C)CC1")


def test_dearomatisation_violates():
    """`c` must not match `C`: aromaticity is part of the label."""
    motif = derive_protected_motif(PIPERAZINE)
    assert not preserves_motif(motif.smarts, "CCN1CCN(C2CCCCC2)CC1")


def test_element_substitution_in_the_motif_violates():
    motif = derive_protected_motif(PIPERAZINE)
    # benzene -> pyridine: one ring carbon becomes nitrogen
    assert not preserves_motif(motif.smarts, "CCN1CCN(c2ccccn2)CC1")


def test_formal_charge_change_in_the_motif_violates():
    """This is the case `Chem.MolFragmentToSmarts` would have let through."""
    motif = derive_protected_motif("C[N+](C)(C)CC1CCCCC1")
    assert motif.atom_count == 6  # the cyclohexane
    assert preserves_motif(motif.smarts, "C[N+](C)(C)CC1CCCCC1")
    charged_ring = derive_protected_motif("c1cc[n+](C)cc1")
    assert charged_ring is not None
    assert preserves_motif(charged_ring.smarts, "c1cc[n+](C)cc1")
    # neutralising the ring nitrogen must break the protected pattern
    assert not preserves_motif(charged_ring.smarts, "c1ccncc1")


def test_bond_order_change_in_the_motif_violates():
    motif = derive_protected_motif("C1CCCCC1CO")
    assert motif is not None and motif.atom_count == 6
    assert preserves_motif(motif.smarts, "C1CCCCC1CO")
    # saturated ring -> cyclohexene: one single bond becomes double
    assert not preserves_motif(motif.smarts, "C1=CCCCC1CO")


def test_ring_contraction_violates():
    motif = derive_protected_motif("C1CCCCC1CO")
    assert not preserves_motif(motif.smarts, "C1CCCC1CO")


def test_unparseable_state_counts_as_a_violation():
    motif = derive_protected_motif(PIPERAZINE)
    assert not preserves_motif(motif.smarts, "((((")


# ------------------------------------------------- trajectory-level auditing


def test_path_violation_summary_flags_the_decisive_cell():
    """endpoint-valid + path-invalid is the trajectory endpoint-only accepts."""
    motif = derive_protected_motif(PIPERAZINE)
    trajectory = [
        PIPERAZINE,                     # source: valid
        "CCN1CCN(C2CCCCC2)CC1",         # dearomatised: VIOLATION
        "CCN1CCN(c2ccccc2)CC1",         # restored: valid endpoint
    ]
    summary = path_violation_summary(motif.smarts, trajectory)
    assert summary["endpoint_valid"] is True
    assert summary["any_violation"] is True
    assert summary["first_violation_index"] == 1
    assert summary["endpoint_valid_path_invalid"] is True
    assert summary["source_violates_own_motif"] is False


def test_path_violation_summary_is_zero_on_a_clean_path():
    """The statistic must be able to take the value that falsifies the claim."""
    motif = derive_protected_motif(PIPERAZINE)
    trajectory = [PIPERAZINE, "CCN1CCN(c2ccccc2Cl)CC1", "CCCN1CCN(c2ccccc2Cl)CC1"]
    summary = path_violation_summary(motif.smarts, trajectory)
    assert summary["any_violation"] is False
    assert summary["endpoint_valid_path_invalid"] is False
    assert summary["violation_count"] == 0


def test_endpoint_invalid_path_invalid_is_distinguished():
    motif = derive_protected_motif(PIPERAZINE)
    trajectory = [PIPERAZINE, "CCN1CCN(C2CCCCC2)CC1", "CCN1CCN(C2CCCCC2)CC1C"]
    summary = path_violation_summary(motif.smarts, trajectory)
    assert summary["any_violation"] is True
    assert summary["endpoint_valid"] is False
    assert summary["endpoint_valid_path_invalid"] is False


# ------------------------------------------------- masking


def test_mask_removes_only_violating_successors():
    motif = derive_protected_motif(PIPERAZINE)
    rows = [
        ("CCN1CCN(c2ccccc2Cl)CC1", 0.4),   # keep
        ("CCN1CCN(C2CCCCC2)CC1", 0.3),     # drop: dearomatised
        ("CCCN1CCN(c2ccccc2)CC1", 0.2),    # keep
        ("CCN1CCN(c2ccccn2)CC1", 0.1),     # drop: element change
    ]
    kept, census = mask_successors(motif.smarts, rows)
    assert [k for k, _ in kept] == [rows[0][0], rows[2][0]]
    assert census["candidates"] == 4
    assert census["kept"] == 2
    assert census["removed"] == 2
    assert census["removed_fraction"] == pytest.approx(0.5)
    assert census["kept_reference_mass"] == pytest.approx(0.6)
    assert census["removed_reference_mass"] == pytest.approx(0.4)


def test_mask_on_empty_support_is_well_defined():
    motif = derive_protected_motif(PIPERAZINE)
    kept, census = mask_successors(motif.smarts, [])
    assert kept == []
    assert census["removed_fraction"] == 0.0


def test_embedding_count_is_a_diagnostic_not_the_predicate():
    motif = derive_protected_motif(PIPERAZINE)
    # a molecule with two independent benzenes has more embeddings, but the
    # predicate is >= 1 either way
    assert motif_embedding_count(motif.smarts, "c1ccccc1") >= 1
    assert motif_embedding_count(motif.smarts, "c1ccccc1-c1ccccc1") >= 2
    assert motif_embedding_count(motif.smarts, "CCCC") == 0


# ------------------------------------------------- eligibility


def test_eligibility_rejects_a_motif_that_is_the_whole_molecule():
    smiles = "c1ccc2[nH]ccc2c1"  # indole: motif fraction 1.0
    motif = derive_protected_motif(smiles)
    verdict = eligibility(smiles, motif)
    assert verdict["eligible"] is False
    assert "motif_fraction" in verdict["reasons"]


def test_eligibility_rejects_a_source_already_at_the_goal():
    motif = derive_protected_motif(FUSED)
    verdict = eligibility(FUSED, motif, already_satisfies_goal=True)
    assert verdict["eligible"] is False
    assert "already_satisfies_goal" in verdict["reasons"]


def test_eligibility_accepts_a_well_shaped_source():
    motif = derive_protected_motif(FUSED)
    verdict = eligibility(FUSED, motif, already_satisfies_goal=False)
    assert verdict["eligible"] is True
    assert verdict["reasons"] == []
    assert HEAVY_ATOM_BAND[0] <= verdict["heavy_atoms"] <= HEAVY_ATOM_BAND[1]


def test_eligibility_rejects_out_of_band_size():
    verdict = eligibility(PIPERAZINE, derive_protected_motif(PIPERAZINE))
    assert verdict["eligible"] is False
    assert "size_band" in verdict["reasons"]  # 14 heavy atoms, band starts at 18


def test_eligibility_never_reads_a_successor_set():
    """Guard the outcome-independence contract at the signature level.

    If a future edit adds a kernel-derived argument to `eligibility`, this test
    fails and forces the change through DECISION_LOG.md, because selecting
    sources on "the kernel offers motif-destroying moves" would make the
    vacuity gate true by construction.
    """
    import inspect

    names = set(inspect.signature(eligibility).parameters)
    assert names == {"smiles", "motif", "already_satisfies_goal"}


# ------------------------------------------------- the writer, at scale


def test_fragment_smarts_rejects_a_bond_leaving_the_atom_set():
    mol = Chem.MolFromSmiles(PIPERAZINE)
    with pytest.raises(Exception):
        fragment_smarts(mol, [0, 1], [b.GetIdx() for b in mol.GetBonds()])


def test_polycyclic_writer_round_trips():
    motif = derive_protected_motif(STEROID)
    assert motif.ring_count == 4
    assert motif.atom_count == 17
    assert preserves_motif(motif.smarts, STEROID)


@pytest.mark.skipif(not RESERVE.exists(), reason="reserve ids not available")
def test_every_held_in_source_matches_its_own_derived_motif():
    """A writer bug must fail loudly here, not silently widen the support."""
    reserve = json.load(gzip.open(RESERVE, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    random.Random(20260813).shuffle(held_in)
    checked = 0
    for smiles in held_in[:1500]:
        motif = derive_protected_motif(smiles)
        if motif is None:
            continue
        assert preserves_motif(motif.smarts, smiles), smiles
        checked += 1
    assert checked > 1000
