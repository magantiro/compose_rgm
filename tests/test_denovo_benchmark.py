"""Tests for the published de-novo benchmark metrics.

These pin the definitions rather than the values COMPOSE happens to produce:
each test constructs a sample whose correct answer is known by hand, so a
change to a threshold, a denominator or a canonicalization rule fails here.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.eval.denovo_benchmark import (
    QED_THRESHOLD,
    SA_THRESHOLD,
    aggregate_seed_metrics,
    denovo_benchmark_metrics,
)

# A small hand-checked panel of real, drug-like molecules.
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
_IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
_BENZENE = "c1ccccc1"


def _qed_sa(smiles: str) -> tuple[float, float]:
    mol = Chem.MolFromSmiles(smiles)
    return float(QED.qed(mol)), float(sascorer.calculateScore(mol))


def test_validity_denominator_is_attempts_not_emissions():
    # Two real molecules, one unparseable string, one empty emission.
    report = denovo_benchmark_metrics([_ASPIRIN, _CAFFEINE, "not-a-molecule", ""])
    assert report["attempted"] == 4
    assert report["valid"] == 2
    # Dropping the empty string would give 2/3; the contract says 2/4.
    assert report["validity"] == 0.5


def test_uniqueness_is_over_valid_not_over_attempts():
    report = denovo_benchmark_metrics([_ASPIRIN, _ASPIRIN, _CAFFEINE, "bad"])
    assert report["valid"] == 3
    assert report["unique"] == 2
    assert report["uniqueness"] == 2 / 3
    assert report["validity"] == 3 / 4


def test_uniqueness_uses_canonical_form_not_string_identity():
    # Same molecule written two ways; must collapse to one unique molecule.
    report = denovo_benchmark_metrics(["C1=CC=CC=C1", "c1ccccc1"])
    assert report["valid"] == 2
    assert report["unique"] == 1
    assert report["uniqueness"] == 0.5


def test_quality_is_the_published_conjunction():
    # Benzene has QED well below 0.6, so it must fail the conjunction while
    # still counting as valid and unique.  Aspirin is included as a measured
    # reminder that the 0.6 bar rejects real marketed drugs: its QED is 0.550.
    benzene_qed, _ = _qed_sa(_BENZENE)
    assert benzene_qed < QED_THRESHOLD
    aspirin_qed, _ = _qed_sa(_ASPIRIN)
    assert aspirin_qed < QED_THRESHOLD

    ibuprofen_qed, ibuprofen_sa = _qed_sa(_IBUPROFEN)
    assert ibuprofen_qed >= QED_THRESHOLD and ibuprofen_sa <= SA_THRESHOLD

    report = denovo_benchmark_metrics([_IBUPROFEN, _BENZENE])
    assert report["high_quality"] == 1
    assert report["quality"] == 0.5
    assert report["quality_given_valid_unique"] == 0.5


def test_quality_counts_a_duplicate_only_once():
    # Three emissions, two of them the same high-quality molecule.
    report = denovo_benchmark_metrics([_IBUPROFEN, _IBUPROFEN, _BENZENE])
    assert report["attempted"] == 3
    assert report["high_quality"] == 1
    # Denominator is attempts, so the duplicate depresses quality.
    assert report["quality"] == 1 / 3
    # Conditional denominator is the 2 unique molecules.
    assert report["quality_given_valid_unique"] == 0.5


def test_quality_never_exceeds_validity_times_uniqueness():
    sample = [_ASPIRIN, _CAFFEINE, _IBUPROFEN, _BENZENE, _ASPIRIN, "bad", ""]
    report = denovo_benchmark_metrics(sample)
    product = float(report["validity"]) * float(report["uniqueness"])
    assert float(report["quality"]) <= product + 1e-12
    assert 0.0 <= float(report["quality_given_valid_unique"]) <= 1.0


def test_thresholds_are_inclusive():
    # Construct the predicate directly at the boundary to pin inclusivity.
    from compose_v4.eval.denovo_benchmark import QED_THRESHOLD as q
    from compose_v4.eval.denovo_benchmark import SA_THRESHOLD as s

    assert q == 0.6
    assert s == 4.0
    # A molecule exactly at the threshold must PASS under >= / <=.
    assert (0.6 >= q) and (4.0 <= s)


def test_diversity_of_identical_molecules_is_zero_and_is_a_distance():
    # All duplicates collapse to one unique molecule -> no pairs -> 0.0.
    report = denovo_benchmark_metrics([_ASPIRIN, _ASPIRIN, _ASPIRIN])
    assert report["unique"] == 1
    assert report["diversity"] == 0.0


def test_diversity_is_distance_not_similarity():
    # Three structurally different molecules must have high pairwise DISTANCE.
    report = denovo_benchmark_metrics([_ASPIRIN, _CAFFEINE, _IBUPROFEN])
    assert report["unique"] == 3
    diversity = float(report["diversity"])
    assert 0.0 < diversity < 1.0
    # These three share very little Morgan substructure; distance must dominate.
    assert diversity > 0.5


def test_diversity_matches_an_independent_pairwise_computation():
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    sample = [_ASPIRIN, _CAFFEINE, _IBUPROFEN, _BENZENE]
    report = denovo_benchmark_metrics(sample)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    canonical = sorted({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in sample})
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in canonical]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    expected = 1.0 - float(np.mean(sims))
    assert abs(float(report["diversity"]) - expected) < 1e-12


def test_moses_intdiv1_includes_the_diagonal():
    sample = [_ASPIRIN, _CAFFEINE, _IBUPROFEN]
    report = denovo_benchmark_metrics(sample)
    # Including n ones on the diagonal pulls IntDiv1 BELOW the off-diagonal
    # distance for any set of dissimilar molecules.
    assert float(report["diversity_moses_intdiv1"]) < float(report["diversity"])


def test_novelty_is_reported_only_when_train_is_supplied():
    plain = denovo_benchmark_metrics([_ASPIRIN, _CAFFEINE])
    assert "novelty_unique_set" not in plain

    train = {Chem.MolToSmiles(Chem.MolFromSmiles(_ASPIRIN))}
    with_train = denovo_benchmark_metrics([_ASPIRIN, _CAFFEINE], train_canonical=train)
    assert with_train["novelty_unique_set"] == 0.5


def test_attempted_override_is_validated():
    import pytest

    with pytest.raises(ValueError):
        denovo_benchmark_metrics([_ASPIRIN, _CAFFEINE], attempted=1)


def test_empty_sample_does_not_raise():
    report = denovo_benchmark_metrics([])
    assert report["attempted"] == 0
    assert report["validity"] == 0.0
    assert report["quality"] == 0.0
    assert report["diversity"] == 0.0


def test_aggregate_reports_mean_and_spread():
    seeds = [
        {"validity": 1.0, "uniqueness": 0.98, "quality": 0.50, "diversity": 0.86,
         "quality_given_valid_unique": 0.51, "diversity_moses_intdiv1": 0.85,
         "mean_qed": 0.5, "mean_sa": 3.0},
        {"validity": 1.0, "uniqueness": 0.96, "quality": 0.54, "diversity": 0.88,
         "quality_given_valid_unique": 0.56, "diversity_moses_intdiv1": 0.87,
         "mean_qed": 0.5, "mean_sa": 3.0},
    ]
    summary = aggregate_seed_metrics(seeds)
    assert summary["seeds"] == 2
    assert abs(float(summary["quality_mean"]) - 0.52) < 1e-12
    assert summary["validity_std"] == 0.0
    assert float(summary["quality_std"]) > 0.0


def test_aggregate_single_seed_has_undefined_spread():
    summary = aggregate_seed_metrics([{"validity": 1.0, "quality": 0.5}])
    assert summary["seeds"] == 1
    assert summary["validity_std"] is None
