from __future__ import annotations

import numpy as np

from compose_v4.eval.molecular_quality import (
    _frechet_activation_components,
    molecular_quality_report,
)


def test_molecular_quality_report_covers_descriptors_rings_and_scaffolds() -> None:
    report = molecular_quality_report(
        ("CCO", "c1ccccc1", "C1CC1", "not-smiles"),
        reference_smiles=("CC", "CCC", "c1ccccc1", "C1CCCCC1"),
        train_smiles=("CCO", "c1ccccc1"),
    )
    assert report["valid_fraction"] == 0.75
    assert report["unique_fraction"] == 1.0
    assert "qed" in report["descriptor_distributions"]
    assert 0.0 <= report["ring_systems"]["ring_size_total_variation"] <= 1.0
    assert 0.0 <= report["scaffolds"]["novel_scaffold_fraction"] <= 1.0
    assert 0.0 <= report["similarity"]["internal_diversity"] <= 1.0


def test_frechet_components_separate_mean_and_covariance_shift() -> None:
    # These centered point clouds have covariance I and 4I exactly (ddof=1).
    base = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    ) * np.sqrt(1.5)
    report = _frechet_activation_components(2.0 * base, base)

    assert np.isclose(report["mean_component"], 0.0)
    assert np.isclose(report["covariance_component"], 2.0)
    assert np.isclose(report["total"], 2.0)
    assert np.isclose(report["covariance_trace_ratio"], 4.0)
