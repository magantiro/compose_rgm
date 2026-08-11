"""Regression protection for the frozen DRD2 oracle.

The reference scores in the manifest were produced by the ORIGINAL sklearn
estimator, in a throwaway environment, at extraction time.  Holding the numpy
reimplementation to them here means the upstream pickle never has to be loaded
again for the oracle to stay trustworthy: any drift in the fingerprint, the
kernel, the Platt orientation, or the libsvm coupling shows up as a test
failure rather than as quietly different DRD2 numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.drd2_oracle import (
    DRD2Oracle,
    _binary_probability_coupling,
    oracle_fingerprint,
    tanimoto_to,
)

ORACLE_DIR = Path(__file__).resolve().parents[1] / "artifacts/oracles/drd2_svm_v1"
MANIFEST = ORACLE_DIR / "drd2_oracle_manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="frozen DRD2 oracle artifact not present")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def oracle() -> DRD2Oracle:
    return DRD2Oracle.from_manifest(MANIFEST)


def test_reproduces_the_original_estimator(oracle, manifest) -> None:
    reference = manifest["reference_scores"]
    fingerprints = np.vstack([oracle_fingerprint(r["smiles"]) for r in reference])
    expected = np.array([r["predict_proba_class1"] for r in reference])
    assert np.abs(oracle.probabilities(fingerprints) - expected).max() < 1e-9


def test_platt_orientation_is_pinned_not_inferred(manifest) -> None:
    # An inverted oracle returns entirely plausible probabilities while
    # rewarding the wrong molecules, so the orientation is recorded and the
    # constructor refuses to guess one.
    assert manifest["platt_orientation"] in ("direct", "flipped")
    with pytest.raises(ValueError):
        DRD2Oracle(ORACLE_DIR / "drd2_svm_parameters.npz",
                   platt_orientation="whichever")


def test_flipping_the_orientation_breaks_parity(oracle, manifest) -> None:
    # Guards against the parity test passing for a degenerate reason: if both
    # orientations agreed, the panel would not be discriminating.
    reference = manifest["reference_scores"]
    fingerprints = np.vstack([oracle_fingerprint(r["smiles"]) for r in reference])
    expected = np.array([r["predict_proba_class1"] for r in reference])
    other = "direct" if oracle.platt_orientation == "flipped" else "flipped"
    inverted = DRD2Oracle(ORACLE_DIR / "drd2_svm_parameters.npz",
                          platt_orientation=other)
    assert np.abs(inverted.probabilities(fingerprints) - expected).max() > 0.5


def test_coupling_early_stops_like_libsvm() -> None:
    # The exact fixed point of the coupling is the sigmoid itself; libsvm stops
    # at max_error < 0.005/k and so leaves a residual. Reproducing that residual
    # is the whole reason the iteration is not short-circuited -- if this ever
    # collapses to the sigmoid, borderline molecules at the 0.5 success
    # threshold would be scored differently from every published DRD2 number.
    sigmoid = np.array([0.5, 0.9, 0.1, 0.999, 0.001])
    first, second = _binary_probability_coupling(sigmoid)
    assert np.allclose(first + second, 1.0)
    residual = np.abs(first - sigmoid).max()
    assert residual > 0.0, "coupling collapsed to the exact sigmoid"
    assert residual < 5e-3, "coupling drifted further than libsvm's tolerance"


def test_discriminates_actives_from_the_source_panel(oracle, manifest) -> None:
    # A scorer returning a constant would pass a parity check on inactives
    # alone, so the recorded panels are asserted to separate.
    parity = manifest["parity"]
    assert parity["source_panel"]["fraction_above_0.5"] == 0.0
    assert parity["source_panel"]["max"] < 0.05
    if parity.get("active_panel"):
        assert parity["active_panel"]["mean"] > 0.5


def test_similarity_uses_a_different_fingerprint_from_the_oracle() -> None:
    # The benchmark scores activity with count-based FCFP6 and constrains
    # similarity with ECFP4 bits; conflating them would silently redefine the
    # constrained task.
    assert tanimoto_to("CCO", "CCO") == pytest.approx(1.0)
    assert 0.0 <= tanimoto_to("CCO", "c1ccccc1") <= 1.0
    assert tanimoto_to("CCO", "not-a-molecule") == 0.0
    counts = oracle_fingerprint("CCO")
    assert counts is not None and counts.shape == (2048,)
    assert counts.sum() > 0 and counts.max() >= 1
    assert oracle_fingerprint("not-a-molecule") is None


def test_unparseable_molecules_score_zero_without_raising(oracle) -> None:
    scores = oracle.score_many(["CCO", "not-a-molecule", "c1ccccc1"])
    assert scores.shape == (3,)
    assert scores[1] == 0.0
    assert scores[0] > 0.0
