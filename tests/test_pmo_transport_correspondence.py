"""Step 1 of the transport planner: the (G, T) correspondence and its offline validation.

Zero oracle calls, zero execution. Two of these tests exist because the validation caught
a defect in the representation rather than confirming it:

  * RDKit's ``completeRingsOnly`` still admits a LONE ring atom as an attachment point, so
    the celecoxib core came back as benzene PLUS one atom of a pyrazole whose other four
    are deleted. Counting that atom as retained understates the install work.
  * An atom-only representation reported ``scale = 0`` for benzene -> cyclohexane, i.e.
    "no transport needed" between two different molecules.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from compose_v4.control.pmo_transport_correspondence import (
    core_is_ring_complete_substructure,
    correspondences,
    deletion_keeps_source_connected,
    partitions_source,
    prune_to_ring_complete,
    validate,
    zero_scale_implies_identical,
)

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
INIT_LIKE = "CC(C)n1cc(S(=O)(=O)[N-]c2ccccc2-c2ncnn2C)cn1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARACETAMOL_OME = "CC(=O)Nc1ccc(OC)cc1"

#: Keys `validate` reports for DIAGNOSIS rather than as verdicts. `requires_reattachment`
#: is False for a healthy connected core, so `all(checks.values())` would read a good
#: outcome as a failure -- which it did, on five tests, the moment the key was added.
DIAGNOSTIC_KEYS = ("deletion_keeps_source_connected", "requires_reattachment")


def decisive(checks: dict) -> dict:
    return {key: value for key, value in checks.items() if key not in DIAGNOSTIC_KEYS}


REAL_PAIRS = (
    (INIT_LIKE, CELECOXIB),
    (PARACETAMOL, PARACETAMOL_OME),
    ("O=C(NC1CCNCC1)c1ccccc1", "O=C(NC1CCN(C)CC1)c1ccccc1"),
    ("COc1ccc(CCN)cc1", "COc1ccc(CCNC)cc1"),
)


@pytest.mark.parametrize(("source", "target"), REAL_PAIRS)
def test_retain_core_and_r_delete_exactly_partition_the_source(source, target):
    for correspondence in correspondences(source, target):
        assert partitions_source(correspondence)


@pytest.mark.parametrize(("source", "target"), REAL_PAIRS)
def test_the_core_is_ring_complete_in_both_endpoints(source, target):
    for correspondence in correspondences(source, target):
        assert core_is_ring_complete_substructure(correspondence)


@pytest.mark.parametrize(("source", "target"), REAL_PAIRS)
def test_every_offline_check_passes_on_real_pairs(source, target):
    found = correspondences(source, target)
    assert found, "a real drug-like pair produced no correspondence"
    for correspondence in found:
        assert all(decisive(validate(correspondence)).values())


def test_a_lone_ring_atom_is_pruned_out_of_the_core():
    """The celecoxib case. Without pruning the core is 7 atoms, one of them a single
    member of a 5-ring that is otherwise deleted."""
    molecule = Chem.MolFromSmiles(INIT_LIKE)
    ring = molecule.GetRingInfo().AtomRings()[0]
    # A core holding the whole first ring plus ONE atom of another is pruned back.
    intruder = next(
        index
        for index in range(molecule.GetNumAtoms())
        if molecule.GetAtomWithIdx(index).IsInRing() and index not in ring
    )
    pruned = prune_to_ring_complete(molecule, set(ring) | {intruder})
    assert intruder not in pruned
    assert set(ring).issubset(pruned) or not pruned


def test_pruning_reaches_a_fixed_point_not_just_one_pass():
    """A second pass is needed when the FIRST pass breaks a ring that was fully covered.

    Naphthalene's two rings share two atoms. Start from one complete ring plus a single
    atom of the other: pass 1 drops the partial ring's touched atoms, which includes the
    two SHARED atoms, so the previously complete ring becomes partial and pass 2 must drop
    it too. A one-pass implementation stops at four atoms and leaves a partial ring, which
    is exactly the thing the core is not allowed to contain.
    """
    molecule = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    first, second = molecule.GetRingInfo().AtomRings()
    intruder = next(index for index in second if index not in first)
    seed = set(first) | {intruder}
    pruned = prune_to_ring_complete(molecule, seed)
    assert pruned == set(), f"one pass would have stopped early, leaving {sorted(pruned)}"
    # And the property that makes it correct: no ring is left partly in, partly out.
    for ring in molecule.GetRingInfo().AtomRings():
        touched = pruned.intersection(ring)
        assert not touched or len(touched) == len(ring)


def test_a_bond_order_only_difference_is_counted_as_transport():
    """benzene -> cyclohexane maps all six atoms; an atom-only scale reports zero."""
    found = correspondences("c1ccccc1", "C1CCCCC1")
    assert found
    correspondence = found[0]
    assert len(correspondence.retain_core) == 6
    assert not correspondence.r_delete and not correspondence.h_install
    assert len(correspondence.core_bond_changes) == 6
    assert correspondence.scale == 6, "a bond-order-only transport must not be free"
    assert all(decisive(validate(correspondence)).values())


def test_zero_scale_is_reserved_for_genuinely_identical_molecules():
    identical = correspondences("c1ccccc1", "c1ccccc1")[0]
    assert identical.scale == 0
    assert zero_scale_implies_identical(identical)
    different = correspondences("c1ccccc1", "C1CCCCC1")[0]
    assert different.scale > 0


def test_attachment_interfaces_are_enumerated_on_both_sides():
    """`alpha` is a SET. A T4 measurement showed alternative placements build genuinely
    different molecules (+11.0%); they lose there only because a similarity ball punishes
    moving away from the reference, which PMO does not have."""
    correspondence = correspondences(INIT_LIKE, CELECOXIB)[0]
    sides = {attachment.side for attachment in correspondence.alpha}
    assert sides == {"source", "target"}, f"alpha covers only {sides}"
    assert len(correspondence.alpha) > 1


def test_deleting_in_the_recorded_order_keeps_every_intermediate_connected():
    """The real assertion: EXACTLY ONE fragment survives at every step.

    The first version of this test asserted `len(fragments) >= 1`, which is always true
    and therefore proved nothing. Once it was made real it FAILED: a lowest-in-set-degree
    peel reached two fragments on the celecoxib pair, because in-set degree ignores how an
    atom connects to the retained core. The ordering was changed, not the assertion.
    """
    for source, target in REAL_PAIRS:
        correspondence = correspondences(source, target)[0]
        molecule = Chem.MolFromSmiles(correspondence.source_smiles)
        dead: set[int] = set()
        for index in correspondence.delete_order:
            dead.add(index)
            editable = Chem.RWMol(molecule)
            for victim in sorted(dead, reverse=True):
                editable.RemoveAtom(victim)
            survivors = editable.GetMol()
            if survivors.GetNumAtoms() <= 1:
                continue
            fragments = Chem.GetMolFrags(survivors)
            assert len(fragments) == 1, (
                f"deleting {index} left {len(fragments)} fragments for {source!r}"
            )
        assert deletion_keeps_source_connected(correspondence)


def test_a_disconnected_core_is_declared_as_requiring_reattachment():
    """Found by running the validation at scale: 7 of 108 correspondences failed the
    connectivity check, all on median1, whose target is the bridged bicyclic camphor.

    When the retained core's pieces are joined only through atoms being deleted, NO
    deletion order preserves connectivity -- the transport is an excision PLUS a
    reattachment bond, the move class the T4 5ht1b_2 witness needed. It must be declared,
    not reported as an ordering defect.
    """
    camphor = "CC1(C)C2CCC1(C)C(=O)C2"
    source = "CC1(C(=O)[O-])CCCN(C(=O)C2CCCO2)C1"
    flagged = [c for c in correspondences(source, camphor) if c.requires_reattachment]
    assert flagged, "the known disconnected-core case was not flagged"
    for correspondence in flagged:
        # The declaration is what makes the verdict correct, not a passing connectivity check.
        checks = validate(correspondence)
        assert checks["requires_reattachment"]
        assert checks["deletion_connectivity_satisfied_or_reattachment_declared"]
        assert checks["partitions_source"]


def test_a_connected_core_is_not_flagged_for_reattachment():
    """The negative control: without it, a predicate that always returns True would pass."""
    correspondence = correspondences(PARACETAMOL, PARACETAMOL_OME)[0]
    assert not correspondence.requires_reattachment
    assert validate(correspondence)["deletion_keeps_source_connected"]


def test_install_order_places_ring_atoms_before_acyclic_ones():
    correspondence = correspondences(INIT_LIKE, CELECOXIB)[0]
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    in_ring = [target.GetAtomWithIdx(i).IsInRing() for i in correspondence.install_order]
    assert in_ring == sorted(in_ring, reverse=True), "an acyclic atom precedes a ring atom"


def test_unparseable_input_raises_rather_than_returning_no_core():
    """An empty result must mean "no shared ring-complete core", never "it did not parse"."""
    with pytest.raises(ValueError, match="source does not parse"):
        correspondences("not-a-molecule", CELECOXIB)
    with pytest.raises(ValueError, match="target does not parse"):
        correspondences(CELECOXIB, "not-a-molecule")


def test_molecules_with_no_shared_ring_complete_core_return_empty():
    assert correspondences("CCCC", "c1ccccc1") == []


def test_the_payload_round_trips_through_json():
    import json

    correspondence = correspondences(PARACETAMOL, PARACETAMOL_OME)[0]
    payload = correspondence.payload()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["scale"] == correspondence.scale
