"""The ring-system signature, its corpus law, and the catalog index over it.

The load-bearing property is that ONE definition of "ring skeleton" is used on
both sides of the comparison: the corpus signature a plan is drawn from, the
template signature a plan is realized with, and the endpoint signature the arms
are scored by.  If those three drifted apart, arm C could report matching the
training distribution while realizing something else entirely, so the agreement
between a molecule's signature and a template's is tested directly rather than
assumed from the fact that both call ``minimum_cycle_basis``.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.eval.denovo_ring_marginal import (
    DEFAULT_HEAVY_ATOM_BIN_EDGES,
    RingSystemPlanPrior,
    heavy_atom_bin,
    ring_signature_census,
    ring_size_total_variation,
    ring_system_signature,
    ring_system_signature_of_smiles,
)
from compose_v4.eval.ring_calibration import ring_template_cycle_sizes


def _signature(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return ring_system_signature(mol)


# ---- Signatures ----


def test_ring_system_signature_names_systems_not_rings() -> None:
    assert _signature("c1ccccc1") == ((6,),)
    assert _signature("CCCC") == ()
    # Fused bicyclics are ONE system; two separate rings are TWO.
    assert _signature("c1ccc2ccccc2c1") == ((6, 6),)
    assert _signature("c1ccc2[nH]ccc2c1") == ((5, 6),)
    assert _signature("c1ccc(-c2ccccc2)cc1") == ((6,), (6,))
    # A biaryl bridged by a chain is still two systems.
    assert _signature("c1ccc(CCCc2ccccc2)cc1") == ((6,), (6,))
    # A spiro junction shares one atom, so its ring bonds are connected.
    assert _signature("C1CCC2(CC1)CCCC2") == ((5, 6),)


def test_ring_system_signature_is_sorted_and_hashable() -> None:
    left = _signature("c1ccc(-c2cc[nH]c2)cc1")
    assert left == tuple(sorted(left))
    assert {left: 1}[left] == 1


def test_unparseable_smiles_returns_none_rather_than_an_empty_skeleton() -> None:
    # An empty tuple is a legitimate signature (an acyclic molecule), so a
    # parse failure must be distinguishable from "no rings".
    assert ring_system_signature_of_smiles("not-a-molecule") is None
    assert ring_system_signature_of_smiles("CCO") == ()


class _Template:
    """The two attributes ``ring_template_cycle_sizes`` reads, and nothing else."""

    def __init__(self, span: int, target_bonds) -> None:
        self.span = span
        self.target_bonds = tuple(target_bonds)


@pytest.mark.parametrize(
    ("smiles", "span", "bonds"),
    [
        ("c1ccccc1", 6, ((0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1))),
        (
            "c1ccc2ccccc2c1",
            10,
            (
                (0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1),
                (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 0, 1), (3, 8, 1),
            ),
        ),
    ],
)
def test_molecule_and_template_signatures_agree(smiles: str, span: int, bonds) -> None:
    """A template's signature and its molecule's must be the SAME object.

    This is what lets a plan drawn over corpus signatures be realized by
    catalog templates: the two vocabularies coincide.  Testing it on a
    hand-built template rather than on a catalog one keeps the check
    independent of whichever checkpoint happens to be available.
    """

    molecule = _signature(smiles)
    assert len(molecule) == 1
    assert molecule[0] == ring_template_cycle_sizes(_Template(span, bonds))


# ---- Census ----


def test_census_separates_per_ring_from_per_molecule_strain() -> None:
    # One molecule carries a three-ring among three rings; two carry none.
    rows = [((3,), (6,), (6,)), ((6,),), ((6,), (6,))]
    census = ring_signature_census(rows)
    assert census["molecules"] == 3
    assert census["rings"] == 6
    assert census["ring_systems"] == 6
    assert census["ring_size_histogram"] == {"3": 1, "6": 5}
    # Per-RING strain is 1/6; per-MOLECULE prevalence is 1/3.  Quoting one for
    # the other inverted a previous de-novo reading.
    assert census["strained_ring_fraction"] == pytest.approx(1 / 6)
    assert census["fraction_with_strained_ring"] == pytest.approx(1 / 3)
    assert census["ring_system_count_fraction"]["3"] == pytest.approx(1 / 3)


def test_census_counts_a_fused_system_once_and_its_rings_separately() -> None:
    census = ring_signature_census([((5, 6),)])
    assert census["ring_systems"] == 1
    assert census["rings"] == 2
    assert census["ring_systems_per_molecule"] == 1.0
    assert census["rings_per_molecule"] == 2.0


def test_census_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="at least one molecule"):
        ring_signature_census([])


def test_ring_size_total_variation_is_symmetric_and_bounded() -> None:
    left = {"5": 0.25, "6": 0.75}
    right = {"5": 0.25, "6": 0.75}
    assert ring_size_total_variation(left, right) == pytest.approx(0.0)
    assert ring_size_total_variation(left, right) == ring_size_total_variation(
        right, left
    )
    assert ring_size_total_variation({"3": 1.0}, {"6": 1.0}) == pytest.approx(1.0)
    # A size present on only one side must still contribute.
    assert ring_size_total_variation({"6": 1.0}, {"6": 0.8, "3": 0.2}) == pytest.approx(
        0.2
    )


# ---- Bins ----


def test_heavy_atom_bins_are_contiguous_and_labelled_by_range() -> None:
    edges = DEFAULT_HEAVY_ATOM_BIN_EDGES
    assert heavy_atom_bin(10, edges=edges) == "<15"
    assert heavy_atom_bin(14, edges=edges) == "<15"
    assert heavy_atom_bin(15, edges=edges) == "15-18"
    assert heavy_atom_bin(18, edges=edges) == "15-18"
    assert heavy_atom_bin(19, edges=edges) == "19-22"
    assert heavy_atom_bin(34, edges=edges) == "31-34"
    assert heavy_atom_bin(35, edges=edges) == ">=35"
    assert heavy_atom_bin(99, edges=edges) == ">=35"


def test_bin_edges_must_increase() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        heavy_atom_bin(10, edges=(20, 10))


# ---- The prior ----


def _observations(count: int = 300):
    # Small molecules carry one ring, large ones three: the size/ring
    # correlation the conditional prior exists to preserve.
    rows = []
    for index in range(count):
        rows.append((16, ((6,),)))
        rows.append((32, ((6,), (6,), (5, 6)) if index % 2 else ((6,), (6,), (6,))))
    return rows


def test_prior_draws_only_observed_skeletons_and_respects_the_bin() -> None:
    prior = RingSystemPlanPrior.fit(_observations())
    rng = np.random.default_rng(0)
    small = {prior.sample(rng, 16) for _ in range(50)}
    large = {prior.sample(rng, 32) for _ in range(50)}
    assert small == {((6,),)}
    assert large <= {((6,), (6,), (5, 6)), ((6,), (6,), (6,))}
    assert len(large) == 2


def test_prior_is_nonparametric_so_it_cannot_invent_a_skeleton() -> None:
    prior = RingSystemPlanPrior.fit(_observations())
    observed = {signature for _heavy, signature in _observations()}
    rng = np.random.default_rng(1)
    for heavy in (16, 32):
        for _ in range(100):
            assert prior.sample(rng, heavy) in observed


def test_prior_refuses_a_bin_too_thin_to_fit() -> None:
    # A silently pooled thin bin would break the size/ring correlation without
    # any visible symptom, so fitting fails instead.
    with pytest.raises(ValueError, match="too thin"):
        RingSystemPlanPrior.fit([(16, ((6,),))] * 10)


def test_an_uncovered_size_falls_back_to_the_nearest_bin_and_says_so() -> None:
    prior = RingSystemPlanPrior.fit(_observations())
    rng = np.random.default_rng(2)
    # Nothing was observed below 15 heavy atoms.
    signature, provenance = prior.sample_with_provenance(rng, 8)
    assert provenance["requested_bin"] == "<15"
    assert provenance["drawn_bin"] == "15-18"
    assert provenance["bin_fallback"] is True
    assert signature == ((6,),)
    # A covered size must NOT report a fallback.
    _signature_value, covered = prior.sample_with_provenance(rng, 16)
    assert covered["bin_fallback"] is False


def test_prior_round_trips_through_json() -> None:
    prior = RingSystemPlanPrior.fit(_observations())
    restored = RingSystemPlanPrior.from_json(prior.to_json())
    assert restored.bin_edges == prior.bin_edges
    assert restored.tables == prior.tables
    assert restored.fitted_molecules == prior.fitted_molecules
    left = np.random.default_rng(3)
    right = np.random.default_rng(3)
    assert [prior.sample(left, 32) for _ in range(20)] == [
        restored.sample(right, 32) for _ in range(20)
    ]


def test_prior_rejects_a_foreign_payload() -> None:
    with pytest.raises(ValueError, match="not a ring-system plan prior"):
        RingSystemPlanPrior.from_json({"schema": "something.else"})


# ---- Uncertainty ----


def test_bootstrap_brackets_its_own_point_estimate() -> None:
    from compose_v4.eval.denovo_ring_marginal import bootstrap_ring_statistics

    rows = [((6,), (6,))] * 40 + [((3,), (6,))] * 10
    reference = {"6": 1.0}
    result = bootstrap_ring_statistics(rows, reference, draws=200, seed=7)
    assert result["molecules"] == 50
    strained = result["strained_ring_fraction"]
    # 10 three-rings among 100 rings.
    assert strained["point"] == pytest.approx(0.1)
    assert strained["ci95_low"] <= strained["point"] <= strained["ci95_high"]
    assert strained["stderr"] > 0.0
    variation = result["ring_size_total_variation"]
    assert variation["point"] == pytest.approx(0.1)
    assert variation["ci95_low"] <= variation["point"] <= variation["ci95_high"]


def test_bootstrap_of_a_homogeneous_sample_has_a_degenerate_interval() -> None:
    from compose_v4.eval.denovo_ring_marginal import bootstrap_ring_statistics

    result = bootstrap_ring_statistics([((6,),)] * 30, {"6": 1.0}, draws=50, seed=1)
    assert result["strained_ring_fraction"]["stderr"] == pytest.approx(0.0)
    assert result["ring_size_total_variation"]["point"] == pytest.approx(0.0)


def test_bootstrap_is_reproducible_under_its_seed() -> None:
    from compose_v4.eval.denovo_ring_marginal import bootstrap_ring_statistics

    rows = [((6,), (5,))] * 20 + [((4,),)] * 5
    left = bootstrap_ring_statistics(rows, {"6": 0.5, "5": 0.5}, draws=64, seed=3)
    right = bootstrap_ring_statistics(rows, {"6": 0.5, "5": 0.5}, draws=64, seed=3)
    assert left == right


def test_bootstrap_refuses_an_empty_sample() -> None:
    from compose_v4.eval.denovo_ring_marginal import bootstrap_ring_statistics

    with pytest.raises(ValueError, match="at least one molecule"):
        bootstrap_ring_statistics([], {"6": 1.0})


def test_proportion_stderr_matches_the_wald_form() -> None:
    from compose_v4.eval.denovo_ring_marginal import proportion_stderr

    assert proportion_stderr(14, 50) == pytest.approx(np.sqrt(0.28 * 0.72 / 50))
    assert proportion_stderr(0, 0) == 0.0
    assert proportion_stderr(50, 50) == 0.0
