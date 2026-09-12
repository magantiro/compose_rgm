"""Chemistry features and strict chronological supervision, no external data."""

import copy

import numpy as np
import pytest

from compose_v4.control.docking_value import (
    LEGACY_RECIPE,
    DockingValue,
    chronological_check,
    graph_kernel,
    identity,
    molecular_features,
    snapshot_equivalence,
)


def fixture_archive():
    return [{"smiles": "C" * i, "ds": -float(i), "round": 1} for i in range(1, 17)]


def test_snapshot_roundtrip_and_smiles_serialization_invariance():
    model = DockingValue.fit(fixture_archive(), before_round=2, source_sha256="a" * 64)
    restored = DockingValue.from_payload(model.payload)
    assert molecular_features("OCC") == molecular_features("CCO")
    assert restored.predict(["OCC", "CCO"])[0] == restored.predict(["OCC", "CCO"])[1]
    np.testing.assert_array_equal(model.predict(["CCO"]), restored.predict(["CCO"]))
    assert model.desirability("CCO", False) == 0
    assert 0 < model.desirability("CCO", True) <= 1


def test_legacy_snapshot_last_bit_drift_has_an_authenticated_global_bound():
    current = DockingValue.fit(
        fixture_archive(), before_round=2, source_sha256="a" * 64
    )
    legacy = copy.deepcopy(current.payload)
    legacy["recipe"] = LEGACY_RECIPE
    legacy["coefficients"][0] = float(np.nextafter(legacy["coefficients"][0], np.inf))
    body = {k: v for k, v in legacy.items() if k != "snapshot_sha256"}
    legacy["snapshot_sha256"] = identity(body)
    audit = snapshot_equivalence(current.payload, legacy, probe_smiles=("CCO", "C1CC1"))
    assert audit["equivalent"] is True
    assert audit["max_abs_coefficient_difference"] <= 1e-14
    assert (
        audit["max_abs_probe_prediction_difference"]
        <= audit["global_prediction_difference_bound"] + np.finfo(np.float64).eps
    )

    material = copy.deepcopy(legacy)
    material["coefficients"][0] += 1e-8
    body = {k: v for k, v in material.items() if k != "snapshot_sha256"}
    material["snapshot_sha256"] = identity(body)
    with pytest.raises(ValueError, match="drift exceeds"):
        snapshot_equivalence(current.payload, material)

    altered_labels = copy.deepcopy(legacy)
    altered_labels["best"] -= 0.1
    body = {k: v for k, v in altered_labels.items() if k != "snapshot_sha256"}
    altered_labels["snapshot_sha256"] = identity(body)
    with pytest.raises(ValueError, match="outside coefficient"):
        snapshot_equivalence(current.payload, altered_labels)


def test_kernel_is_normalized_and_distinguishes_graph_connectivity():
    features = [molecular_features(s)[1] for s in ("CCCC", "CC(C)C", "C1CCC1")]
    kernel = graph_kernel(features, features)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(kernel.diagonal(), 1)
    assert np.linalg.eigvalsh(kernel).min() > -1e-12
    assert kernel[0, 1] < 1 and kernel[1, 2] < 1


def test_bit_kernel_matches_independent_set_cardinalities_exactly():
    rng = np.random.default_rng(11)
    rows = [
        tuple(tuple(map(int, rng.integers(0, 2048, size=70))) for _ in range(2))
        for _ in range(18)
    ] + [((), ()), ((2, 2, 3), (3, 3))]
    expected = np.empty((len(rows), len(rows)))
    for i, left in enumerate(rows):
        for j, right in enumerate(rows):
            values = []
            for a, b in zip(left, right, strict=True):
                a, b = set(a), set(b)
                values.append(len(a & b) / len(a | b) if a or b else 1.0)
            expected[i, j] = sum(values) / 2
    np.testing.assert_array_equal(graph_kernel(rows, rows), expected)
    assert graph_kernel([], rows).shape == (0, len(rows))
    assert graph_kernel(rows, []).shape == (len(rows), 0)
    with pytest.raises(ValueError, match="declared width"):
        graph_kernel([((2048,), ())], rows)


def test_no_future_labels_duplicates_nonfinite_or_tampered_snapshot():
    archive = fixture_archive()
    with pytest.raises(ValueError, match="outside"):
        DockingValue.fit(archive, before_round=1, source_sha256="a" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        DockingValue.fit(archive + [archive[0]], before_round=2, source_sha256="a" * 64)
    bad = copy.deepcopy(archive)
    bad[0]["ds"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        DockingValue.fit(bad, before_round=2, source_sha256="a" * 64)
    model = DockingValue.fit(archive, before_round=2, source_sha256="a" * 64)
    bad = copy.deepcopy(model.payload)
    bad["coefficients"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        DockingValue.from_payload(bad)


def test_missing_oracle_label_is_recorded_not_replaced():
    model = DockingValue.fit(
        fixture_archive() + [{"smiles": "O", "ds": None, "round": 1}],
        before_round=2,
        source_sha256="a" * 64,
    )
    assert len(model.payload["training_rows"]) == 16
    assert model.payload["excluded"][0]["reason"] == "missing_oracle_label"


def test_final_validation_round_rejects_leakage_and_malformed_labels():
    with pytest.raises(ValueError, match="duplicate"):
        chronological_check(
            fixture_archive() + [{"smiles": "CC", "ds": -4.0, "round": 2}],
            source_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="malformed"):
        chronological_check(
            fixture_archive() + [{"smiles": "O", "ds": True, "round": 2}],
            source_sha256="a" * 64,
        )
