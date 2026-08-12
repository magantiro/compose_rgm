"""The frozen Claim-2 metric suite, including the falsifiability gate.

These run on real RDKit molecules but never touch the model, the checkpoint or
Modal, so the whole mobility--fidelity instrument is pinned before any
container-hour is spent.
"""

from __future__ import annotations

import math

import pytest

from compose_v4.experiments.claim2_trajectory_metrics import (
    BANNED_STATISTICS,
    DESCRIPTOR_NAMES,
    DOMINATED_BY,
    DOMINATES,
    INCOMPARABLE,
    METRIC_REGISTRY,
    UNRESOLVED,
    DescriptorEnvelope,
    FrontierPoint,
    MetricError,
    MetricSpec,
    assert_not_banned,
    calibrate_envelope,
    descriptor_vector,
    family_coverage,
    family_entropy,
    frontier_verdict,
    heavy_atom_count,
    multi_family_trajectory,
    paired_bootstrap_interval,
    resolved,
    ring_system_count,
    tanimoto_distance,
    trajectory_health,
    validate_metric_registry,
)

BENZAMIDE = "CNC(=O)c1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
BIPHENYL = "c1ccc(-c2ccccc2)cc1"
ETHANOL = "CCO"


# ---- descriptors and the envelope -----------------------------------------


def test_descriptor_vector_is_the_nine_frozen_coordinates():
    vector = descriptor_vector(BENZAMIDE)
    assert vector is not None
    assert len(vector) == len(DESCRIPTOR_NAMES) == 9
    assert all(math.isfinite(value) for value in vector)


def test_descriptor_vector_returns_none_rather_than_raising_on_a_bad_key():
    """An unparseable committed state is a finding, not a crash."""
    assert descriptor_vector("not a molecule") is None


def test_envelope_refuses_a_sample_too_small_to_calibrate():
    with pytest.raises(MetricError, match="real sample"):
        calibrate_envelope(
            [descriptor_vector(BENZAMIDE)] * 10,
            calibration_source="test",
            status="TEST",
        )


def _toy_envelope(**overrides) -> DescriptorEnvelope:
    payload = {
        "names": DESCRIPTOR_NAMES,
        "lower": (0.0,) * 9,
        "upper": (10.0,) * 9,
        "mean": (5.0,) * 9,
        "stdev": (1.0,) * 9,
        "calibration_source": "test",
        "calibration_count": 1000,
        "status": "TEST",
    }
    payload.update(overrides)
    return DescriptorEnvelope(**payload)


def test_envelope_membership_is_coordinate_wise_and_all_must_hold():
    envelope = _toy_envelope()
    assert envelope.contains((5.0,) * 9)
    assert not envelope.contains((5.0,) * 8 + (11.0,))
    membership = envelope.coordinate_membership((5.0,) * 8 + (11.0,))
    assert membership[DESCRIPTOR_NAMES[8]] is False
    assert membership[DESCRIPTOR_NAMES[0]] is True


def test_envelope_refuses_a_degenerate_standardization():
    with pytest.raises(MetricError, match="degenerate coordinate"):
        _toy_envelope(stdev=(1.0,) * 8 + (0.0,))


def test_envelope_refuses_an_inverted_interval():
    with pytest.raises(MetricError, match="inverted"):
        _toy_envelope(lower=(11.0,) + (0.0,) * 8)


def test_envelope_drift_is_zero_for_a_state_equal_to_its_source():
    envelope = _toy_envelope()
    vector = descriptor_vector(BENZAMIDE)
    assert envelope.drift(vector, vector) == pytest.approx(0.0)


def test_envelope_round_trips_through_json():
    envelope = _toy_envelope()
    assert DescriptorEnvelope.from_json(envelope.to_json()) == envelope


def test_calibrated_envelope_contains_most_of_its_own_calibration_sample():
    """A 0.5%/99.5% box must not exclude the population it was fitted on."""
    smiles = [BENZAMIDE, NAPHTHALENE, BIPHENYL, ETHANOL, "CCCCO", "c1ccncc1", "CC(=O)O"]
    vectors = [descriptor_vector(s) for s in smiles] * 30
    envelope = calibrate_envelope(vectors, calibration_source="test", status="TEST")
    inside = sum(1 for vector in vectors if envelope.contains(vector))
    assert inside == len(vectors)


# ---- structural mobility --------------------------------------------------


def test_tanimoto_distance_to_self_is_zero():
    assert tanimoto_distance(BENZAMIDE, BENZAMIDE) == pytest.approx(0.0)


def test_tanimoto_distance_is_positive_between_different_molecules():
    assert 0.0 < tanimoto_distance(BENZAMIDE, NAPHTHALENE) <= 1.0


def test_tanimoto_distance_is_none_when_a_key_does_not_parse():
    assert tanimoto_distance(BENZAMIDE, "@@@") is None


def test_ring_system_count_separates_fused_from_isolated_rings():
    """Two rings is not two ring systems if they are fused.

    This is why ring COUNT alone cannot describe topological movement.
    """
    assert ring_system_count(NAPHTHALENE) == 1   # two fused rings, one system
    assert ring_system_count(BIPHENYL) == 2      # two isolated rings
    assert ring_system_count(ETHANOL) == 0


def test_heavy_atom_count_matches_rdkit():
    assert heavy_atom_count(ETHANOL) == 3
    assert heavy_atom_count("@@@") is None


# ---- trajectory health ----------------------------------------------------


def test_health_of_a_clean_forward_trajectory():
    health = trajectory_health(["a", "b", "c", "d", "e", "f", "g"], horizon=6)
    assert health.committed_edits == 6
    assert health.distinct_states == 7
    assert health.unique_state_fraction == pytest.approx(1.0)
    assert health.immediate_reversal_count == 0
    assert health.revisit_count == 0
    assert health.reached_horizon
    assert not health.dead_end


def test_health_detects_an_immediate_reversal():
    """a -> b -> a is one undo, and also a two-cycle."""
    health = trajectory_health(["a", "b", "a", "c"], horizon=6)
    assert health.immediate_reversal_count == 1
    assert health.two_cycle_count == 1
    assert health.revisit_count == 1


def test_health_detects_a_nonadjacent_revisit_that_is_not_a_reversal():
    """a -> b -> c -> a revisits without ever undoing a single step."""
    health = trajectory_health(["a", "b", "c", "a"], horizon=6)
    assert health.immediate_reversal_count == 0
    assert health.revisit_count == 1
    assert health.unique_state_fraction == pytest.approx(0.75)


def test_health_marks_an_early_dead_end():
    health = trajectory_health(["a", "b"], horizon=6)
    assert health.dead_end
    assert not health.reached_horizon


def test_health_of_a_source_only_trajectory_is_a_dead_end_not_a_crash():
    health = trajectory_health(["a"], horizon=6)
    assert health.committed_edits == 0
    assert health.dead_end
    assert health.immediate_reversal_rate == 0.0


def test_health_refuses_an_empty_trajectory():
    with pytest.raises(MetricError, match="at least its source"):
        trajectory_health([], horizon=6)


# ---- operator families ----------------------------------------------------


def test_family_entropy_is_zero_under_operator_collapse():
    """The stop-rule case: one family for every committed edit."""
    assert family_entropy([["atom_delete"]] * 6) == pytest.approx(0.0)


def test_family_entropy_is_maximal_for_balanced_families():
    entropy = family_entropy([["a"], ["b"], ["c"], ["d"]])
    assert entropy == pytest.approx(math.log(4))


def test_family_entropy_splits_an_ambiguous_step_fractionally():
    """A successor reached by two families contributes 1/2 to each."""
    entropy = family_entropy([["a", "b"]])
    assert entropy == pytest.approx(math.log(2))


def test_family_coverage_and_multi_family_flag():
    assert family_coverage([["a"], ["a", "b"]]) == {"a", "b"}
    assert multi_family_trajectory([["a"], ["b"]])
    assert not multi_family_trajectory([["a"], ["a"]])


# ---- the frontier ---------------------------------------------------------


def _point(arm: str, mobility: float, fidelity: float) -> FrontierPoint:
    return FrontierPoint(
        arm=arm,
        mobility=mobility,
        fidelity=fidelity,
        source_count=40,
        trajectory_count=200,
        state_count=1200,
    )


def test_frontier_reports_domination_when_both_axes_favour_the_reference():
    verdict = frontier_verdict(
        _point("r_theta", 0.6, 0.9),
        _point("uniform_canonical", 0.4, 0.6),
        mobility_resolved=True,
        fidelity_resolved=True,
    )
    assert verdict == DOMINATES


def test_frontier_can_report_the_reference_being_dominated():
    """The verdict must be able to come out against R_theta, or it is not a test."""
    verdict = frontier_verdict(
        _point("r_theta", 0.2, 0.5),
        _point("uniform_canonical", 0.5, 0.8),
        mobility_resolved=True,
        fidelity_resolved=True,
    )
    assert verdict == DOMINATED_BY


def test_frontier_reports_incomparable_when_mobility_is_bought_with_fidelity():
    """A real and publishable outcome; it must not be scalarized away."""
    verdict = frontier_verdict(
        _point("r_theta", 0.7, 0.5),
        _point("uniform_canonical", 0.4, 0.8),
        mobility_resolved=True,
        fidelity_resolved=True,
    )
    assert verdict == INCOMPARABLE


def test_frontier_is_unresolved_when_no_axis_separates():
    """An underpowered panel must not produce a direction from point estimates."""
    verdict = frontier_verdict(
        _point("r_theta", 0.61, 0.91),
        _point("uniform_canonical", 0.60, 0.90),
        mobility_resolved=False,
        fidelity_resolved=False,
    )
    assert verdict == UNRESOLVED


def test_frontier_uses_only_resolved_axes():
    """An unresolved axis contributes no direction, even with a visible gap."""
    verdict = frontier_verdict(
        _point("r_theta", 0.9, 0.1),
        _point("uniform_canonical", 0.4, 0.9),
        mobility_resolved=True,
        fidelity_resolved=False,
    )
    assert verdict == DOMINATES


# ---- the falsifiability gate ----------------------------------------------


def test_every_declared_metric_states_what_would_falsify_it():
    validate_metric_registry()


def test_registry_refuses_a_metric_with_no_false_hypothesis_value():
    """The gate the workstream brief requires, made executable."""
    with pytest.raises(MetricError, match="falsifying observation"):
        validate_metric_registry(
            [
                MetricSpec(
                    name="looks_impressive",
                    definition="whatever we measured",
                    unit="none",
                    falsifying_observation="",
                )
            ]
        )


def test_registry_refuses_a_duplicate_metric_name():
    spec = METRIC_REGISTRY[0]
    with pytest.raises(MetricError, match="declared twice"):
        validate_metric_registry([spec, spec])


def test_banned_statistics_cannot_be_reported():
    """Gibbs' inequality fixes this comparison's sign before any data exists."""
    with pytest.raises(MetricError, match="may not be reported"):
        assert_not_banned("reference_log_likelihood_of_own_trajectories")


def test_banned_statistics_each_carry_a_stated_reason():
    for name, reason in BANNED_STATISTICS.items():
        assert len(reason) > 40, f"{name} must say WHY it is unfalsifiable"


def test_a_banned_name_may_not_enter_the_registry():
    with pytest.raises(MetricError, match="banned list"):
        validate_metric_registry(
            [
                MetricSpec(
                    name="validity_rate_by_arm",
                    definition="fraction of valid states",
                    unit="fraction",
                    falsifying_observation="an invalid state appears",
                )
            ]
        )


# ---- paired resampling ----------------------------------------------------


def test_paired_bootstrap_resolves_a_consistent_difference():
    left = {f"s{i}": 0.6 for i in range(40)}
    right = {f"s{i}": 0.4 for i in range(40)}
    interval = paired_bootstrap_interval(left, right, resamples=400)
    assert interval[0] == pytest.approx(0.2)
    assert resolved(interval)


def test_paired_bootstrap_does_not_resolve_a_symmetric_difference():
    left = {f"s{i}": float(i % 2) for i in range(40)}
    right = {f"s{i}": float((i + 1) % 2) for i in range(40)}
    assert not resolved(paired_bootstrap_interval(left, right, resamples=400))


def test_paired_bootstrap_pairs_on_shared_sources_only():
    interval = paired_bootstrap_interval({"a": 1.0, "b": 2.0}, {"b": 1.0}, resamples=50)
    assert interval[0] == pytest.approx(1.0)


def test_paired_bootstrap_refuses_disjoint_source_sets():
    with pytest.raises(MetricError, match="measured under both arms"):
        paired_bootstrap_interval({"a": 1.0}, {"b": 1.0}, resamples=10)
