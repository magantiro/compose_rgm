"""The stage splitter: does a staged transport path exist, and does it approach?

Zero oracle calls, zero program execution. Intermediates are built with RDKit from the
correspondence, so a failure here is a failure of the PLAN and cannot be blamed on the
executor.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from compose_v4.control.pmo_transport_correspondence import correspondences
from compose_v4.control.pmo_transport_staging import (
    DEFAULT_MAX_PRIMITIVES,
    build_intermediate,
    split,
    validate_staging,
)

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
INIT_LIKE = "CC(C)n1cc(S(=O)(=O)[N-]c2ccccc2-c2ncnn2C)cn1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARACETAMOL_OME = "CC(=O)Nc1ccc(OC)cc1"


def _staged(source, target, *, ceiling=DEFAULT_MAX_PRIMITIVES):
    correspondence = correspondences(source, target)[0]
    stages = split(correspondence, max_primitives=ceiling)
    return correspondence, stages, validate_staging(
        correspondence, stages, max_primitives=ceiling
    )


def test_no_stage_exceeds_the_ceiling():
    for ceiling in (8, 16, 23):
        _, stages, checks = _staged(INIT_LIKE, CELECOXIB, ceiling=ceiling)
        assert checks["respects_ceiling"], (
            f"a stage of {checks['max_stage_size']} exceeded a ceiling of {ceiling}"
        )
        assert all(stage.size <= ceiling for stage in stages)


def test_every_stage_endpoint_is_a_valid_connected_molecule():
    _, stages, checks = _staged(INIT_LIKE, CELECOXIB)
    assert checks["all_intermediates_valid"]
    assert checks["all_intermediates_connected"]
    for stage in stages:
        molecule = Chem.MolFromSmiles(stage.endpoint_smiles)
        assert molecule is not None
        assert len(Chem.GetMolFrags(molecule)) == 1


def test_the_staged_path_reaches_the_target():
    _, _, checks = _staged(INIT_LIKE, CELECOXIB)
    assert checks["reaches_target"]


def test_a_long_transport_needs_more_than_one_stage():
    """The point of staging. If this ever passes in one stage the ceiling moved."""
    correspondence, stages, _ = _staged(INIT_LIKE, CELECOXIB)
    assert correspondence.scale > DEFAULT_MAX_PRIMITIVES
    assert len(stages) > 1


def test_a_one_primitive_transport_is_a_single_stage():
    correspondence, stages, checks = _staged(PARACETAMOL, PARACETAMOL_OME)
    assert correspondence.scale == 1
    assert len(stages) == 1
    assert checks["reaches_target"]


def test_monotonicity_over_fewer_than_two_points_is_reported_unevaluated():
    """A verdict over one point is True by construction and says nothing. Reporting it as
    a pass let a single-point 'monotone' sit beside an invalid intermediate."""
    _, _, checks = _staged(PARACETAMOL, PARACETAMOL_OME)
    assert checks["similarity_points"] >= 1
    if checks["similarity_points"] < 2:
        assert checks["similarity_monotone_nondecreasing"] is None


def test_the_trajectory_starts_at_the_source():
    """Measuring monotonicity over stage endpoints alone cannot see an initial DIP: the
    prune phase strips structure before the install phase rebuilds it."""
    _, stages, checks = _staged(INIT_LIKE, CELECOXIB)
    assert checks["source_similarity"] is not None
    assert checks["similarity_trajectory"][0] == pytest.approx(checks["source_similarity"])
    assert checks["similarity_points"] == len(stages) + 1


#: A transport measured to dip. Anchored growth REMOVED the dip on celecoxib
#: (0.1486 -> 0.1591 -> 1.0, monotone), so the case that demonstrates the finding had to
#: be taken from the measured set rather than assumed -- 17 of 44 staged transports dip,
#: and this is one of them.
DIPPING_SOURCE = "CCCC(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1"
DIPPING_TARGET = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C"


def test_a_staged_transport_can_dip_below_its_own_starting_similarity():
    """MEASURED, and it is the finding: the path exists and the SIGNAL does not guide you
    along it. A controller selecting parents on score would abandon the intermediate.
    """
    _, _, checks = _staged(DIPPING_SOURCE, DIPPING_TARGET)
    assert checks["dips_below_source"] is True
    assert checks["minimum_similarity_on_path"] < checks["source_similarity"]
    assert checks["similarity_monotone_nondecreasing"] is False


def test_the_dip_detector_does_not_fire_on_a_monotone_transport():
    """The negative control: a detector that always fires would pass the test above."""
    _, _, checks = _staged(INIT_LIKE, CELECOXIB)
    assert checks["dips_below_source"] is False
    assert checks["similarity_monotone_nondecreasing"] is True


def test_interleaving_removes_the_dip_on_a_dipping_transport():
    """The dip has a MECHANISM and it is the ORDER, not the transport.

    Prune-then-install strips structure the target does not want before adding structure
    it does, so the midpoint is smaller than both ends and scores below the source.
    Installing before deleting keeps the molecule out of that trough. MEASURED across the
    declared-target set: 9 of 10 dipping transports become monotone under interleaving.
    """
    correspondence = correspondences(DIPPING_SOURCE, DIPPING_TARGET)[0]
    prune_first = validate_staging(
        correspondence,
        split(correspondence, max_primitives=DEFAULT_MAX_PRIMITIVES),
        max_primitives=DEFAULT_MAX_PRIMITIVES,
    )
    woven = validate_staging(
        correspondence,
        split(correspondence, max_primitives=DEFAULT_MAX_PRIMITIVES, interleave=True),
        max_primitives=DEFAULT_MAX_PRIMITIVES,
    )
    assert prune_first["dips_below_source"] is True
    assert woven["dips_below_source"] is False
    assert woven["dip_depth_relative"] == 0.0
    # The endpoint must not change: a reordering that reaches somewhere else is not a
    # rescue. MEASURED: interleaving preserves it on 6 of 10, so it is a per-transport
    # choice under a validity constraint, never a blanket default.
    assert woven["final_smiles"] == prune_first["final_smiles"]


def test_dip_depth_and_width_are_reported_not_just_a_boolean():
    """A 2% dip for one stage is navigable; a 40% dip for two stages needs protected
    budget. MEASURED distribution: median depth 65.9% but median width ONE stage."""
    correspondence = correspondences(DIPPING_SOURCE, DIPPING_TARGET)[0]
    checks = validate_staging(
        correspondence,
        split(correspondence, max_primitives=DEFAULT_MAX_PRIMITIVES),
        max_primitives=DEFAULT_MAX_PRIMITIVES,
    )
    assert checks["dip_depth_relative"] > 0.1
    assert checks["dip_width_stages"] >= 1
    assert checks["dip_depth_absolute"] > 0.0


def test_dip_shape_is_unevaluated_when_there_is_nothing_to_compare():
    correspondence = correspondences(PARACETAMOL, PARACETAMOL_OME)[0]
    checks = validate_staging(
        correspondence, split(correspondence), max_primitives=DEFAULT_MAX_PRIMITIVES
    )
    if checks["similarity_points"] < 2:
        assert checks["dip_depth_relative"] is None
        assert checks["dip_width_stages"] is None


def test_stereochemistry_is_excluded_from_the_target_comparison():
    """COMPOSE's MolecularGraph carries no stereo, so a rebuilt molecule can never
    reproduce [C@H]. Both comparisons are reported; the stereo-blind one is the verdict."""
    _, _, checks = _staged(INIT_LIKE, CELECOXIB)
    assert "reaches_target_with_stereochemistry" in checks


def test_build_intermediate_returns_none_rather_than_raising_on_a_non_molecule():
    """A staged plan passing through a non-molecule is a FINDING, not an error to
    swallow, so it must be reportable rather than fatal."""
    correspondence = correspondences(INIT_LIKE, CELECOXIB)[0]
    # A deliberately mid-ring cut: partial aromatic rings do not sanitize.
    result = build_intermediate(correspondence, deleted=6, changed=0, installed=0)
    assert result is None or isinstance(result, str)


def test_split_refuses_a_ceiling_below_one():
    correspondence = correspondences(PARACETAMOL, PARACETAMOL_OME)[0]
    with pytest.raises(ValueError, match="at least one primitive"):
        split(correspondence, max_primitives=0)
