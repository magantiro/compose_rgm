"""Tests for the de-novo ring/size decomposition.

The module exists to separate a ring-strain effect from a molecule-size effect.
The load-bearing test is therefore the NEGATIVE CONTROL: on data where strain has
no effect and size carries the whole signal, the fitted strain coefficient must
collapse toward zero even though the raw strained-minus-clean difference is large.
An instrument that cannot return "it was size" cannot be trusted when it returns
"it was strain".
"""

from __future__ import annotations

import numpy as np

from compose_v4.eval.denovo_ring_decomposition import (
    HEAVY_ATOM_BIN_EDGES,
    _ordinary_least_squares,
    molecule_features,
    ring_event_census,
    ring_size_distribution,
    strain_size_regression,
    strain_size_strata,
)

# Aziridine- and oxetane-bearing (strained) and plain (clean) molecules.
_STRAINED = ["C1CN1CCCC", "C1COC1CCC", "CC1CC1CCO"]
_CLEAN = ["c1ccccc1CCO", "C1CCCCC1CC", "c1ccncc1CCC"]


def _fit(strained_effect: float, size_effect: float, n: int = 200):
    """Synthetic design with a known strain/size decomposition."""

    rng = np.random.default_rng(20260921)
    # Strain is CONFOUNDED with size: strained rows are systematically larger.
    strained = np.repeat([0.0, 1.0], n // 2)
    heavy = 20.0 + 8.0 * strained + rng.normal(0.0, 1.0, size=n)
    response = 1.0 + strained_effect * strained + size_effect * heavy
    design = np.column_stack([np.ones(n), strained, heavy])
    return _ordinary_least_squares(design, response), strained, heavy, response


def test_regression_attributes_a_pure_size_effect_to_size_not_strain() -> None:
    """NEGATIVE CONTROL: strain has zero effect; size carries everything."""

    fitted, strained, _heavy, response = _fit(strained_effect=0.0, size_effect=0.1)
    assert fitted is not None
    coefficients, _errors = fitted

    # The RAW contrast is large and would read as a strong strain effect.
    raw_difference = response[strained == 1].mean() - response[strained == 0].mean()
    assert raw_difference > 0.7

    # Partialling out size collapses the strain coefficient to ~0 and recovers size.
    assert abs(coefficients[1]) < 1e-6
    assert abs(coefficients[2] - 0.1) < 1e-6


def test_regression_recovers_a_genuine_strain_effect() -> None:
    """POSITIVE CONTROL: a real strain effect survives partialling out size."""

    fitted, _strained, _heavy, _response = _fit(strained_effect=0.9, size_effect=0.1)
    assert fitted is not None
    coefficients, _errors = fitted
    assert abs(coefficients[1] - 0.9) < 1e-6
    assert abs(coefficients[2] - 0.1) < 1e-6


def test_ordinary_least_squares_returns_none_when_underdetermined() -> None:
    design = np.ones((2, 3))
    assert _ordinary_least_squares(design, np.ones(2)) is None


def test_ring_size_fraction_is_invariant_to_sample_duplication() -> None:
    """The per-ring fraction is a policy shape, so replicating the sample cannot move it."""

    single = ring_size_distribution(_STRAINED + _CLEAN)
    doubled = ring_size_distribution((_STRAINED + _CLEAN) * 3)
    assert single["strained_ring_fraction"] == doubled["strained_ring_fraction"]
    assert single["rings_per_molecule"] == doubled["rings_per_molecule"]
    # Counting denominators DO scale, which is what distinguishes them from fractions.
    assert doubled["molecules"] == 3 * single["molecules"]
    assert doubled["rings"] == 3 * single["rings"]


def test_ring_size_distribution_separates_per_ring_from_per_molecule() -> None:
    report = ring_size_distribution(_STRAINED + _CLEAN)
    assert report["molecules"] == 6
    # Every strained member carries exactly one strained ring, every clean one none.
    assert report["fraction_with_strained_ring"] == 0.5
    assert 0.0 < report["strained_ring_fraction"] <= 1.0
    assert set(report["ring_size_histogram"]) <= {3, 4, 5, 6}


def test_empty_sample_reports_none_rather_than_zero() -> None:
    report = ring_size_distribution([])
    assert report["molecules"] == 0
    assert report["strained_ring_fraction"] is None
    assert report["fraction_with_strained_ring"] is None


def test_unparseable_and_empty_strings_are_excluded_from_denominators() -> None:
    report = ring_size_distribution(["c1ccccc1", "", "not_a_molecule", None or ""])
    assert report["molecules"] == 1


def test_molecule_features_reports_heavy_atoms_and_strain() -> None:
    strained = molecule_features("C1CN1CCCC")
    clean = molecule_features("c1ccccc1CCO")
    assert strained is not None and clean is not None
    assert strained["strained"] is True
    assert clean["strained"] is False
    assert strained["heavy_atoms"] == 7
    assert molecule_features("") is None
    assert molecule_features("not_a_molecule") is None


def test_strata_use_fixed_bin_edges_and_withhold_thin_cells() -> None:
    report = strain_size_strata(_STRAINED + _CLEAN)
    assert report["bin_edges"] == list(HEAVY_ATOM_BIN_EDGES)
    # Every molecule here is small, so cells are thin and must report None, not 0.0.
    for entry in report["strata"].values():
        for arm in ("strained", "clean"):
            if entry[arm]["n"] < 3:
                assert entry[arm]["mean_sa"] is None
                assert entry[arm]["mean_qed"] is None


def test_regression_report_carries_denominator_and_raw_contrast() -> None:
    report = strain_size_regression(_STRAINED + _CLEAN)
    assert report["n"] == 6
    assert report["strained_count"] == 3
    assert report["terms"] == ["intercept", "strained", "heavy_atoms"]
    for key in ("sa", "qed"):
        assert report[key] is not None
        assert "strained_coefficient" in report[key]
        assert "strained_stderr" in report[key]
        assert "heavy_atom_coefficient" in report[key]
        assert report[key]["raw_strained_difference"] is not None


def test_regression_abstains_on_a_tiny_sample() -> None:
    assert strain_size_regression(["c1ccccc1"])["sa"] is None


def test_ring_event_census_counts_transitions_and_flags_the_creation_assumption() -> None:
    records = [
        {"event_rules": ["atom_insert", "ring_system_grow", "ring_system_grow"]},
        {"event_rules": ["ring_system_grow", "bond_reorder"]},
    ]
    census = ring_event_census(records)
    assert census["trajectories"] == 2
    assert census["events"] == 5
    assert census["ring_forming_events"] == 3
    assert census["ring_removing_events"] == 0
    assert census["ring_forming_events_per_trajectory"] == 1.5
    # With zero removals every endpoint ring corresponds to a creation event.
    assert census["endpoint_census_tracks_creation"] is True


def test_ring_event_census_detects_removals_that_break_the_assumption() -> None:
    census = ring_event_census([{"event_rules": ["ring_system_grow", "ring_system_delete"]}])
    assert census["ring_removing_events"] == 1
    assert census["endpoint_census_tracks_creation"] is False


def test_ring_event_census_skips_records_without_recorded_rules() -> None:
    census = ring_event_census([{"smiles": "c1ccccc1"}, {"event_rules": ["ring_system_grow"]}])
    assert census["trajectories"] == 1
    assert census["ring_forming_events"] == 1
