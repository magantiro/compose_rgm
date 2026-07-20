from __future__ import annotations

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.experiments.exact_rdkit_conditioning import (
    GRIDDD_ZINC_REFERENCE_THRESHOLDS,
    ExactRDKitTargetResponseEvaluator,
    TargetedCandidateRecord,
    exact_rdkit_crippen_logp,
    exact_rdkit_molecular_weight,
    exact_rdkit_qed,
    standard_exact_rdkit_target_specs,
)


def _state(smiles: str):
    return smiles_to_molecular_graph(smiles)


def test_exact_rdkit_oracles_and_frozen_griddd_zinc_references() -> None:
    ethanol = _state("CCO")
    assert exact_rdkit_molecular_weight(ethanol) == pytest.approx(46.069, abs=1e-3)
    assert exact_rdkit_crippen_logp(ethanol) == pytest.approx(-0.0014, abs=1e-4)
    assert exact_rdkit_qed(ethanol) == pytest.approx(0.4068, abs=1e-4)
    assert GRIDDD_ZINC_REFERENCE_THRESHOLDS == {
        "qed": {"mae": 0.04, "validity_fraction": 0.872},
        "molecular_weight": {"mae": 4.89, "validity_fraction": 0.842},
        "crippen_logp": {"mae": 0.19, "validity_fraction": 0.879},
    }


def test_target_priority_is_qed_then_mw_then_logp() -> None:
    specs = standard_exact_rdkit_target_specs()
    assert [item.property_name for item in specs] == [
        "qed",
        "molecular_weight",
        "crippen_logp",
    ]
    assert [item.priority for item in specs] == [1, 2, 3]


def test_mw_target_response_reports_flexible_size_and_all_attempt_validity() -> None:
    spec = standard_exact_rdkit_target_specs()[1]
    ethanol = _state("CCO")
    decane = _state("CCCCCCCCCC")
    report = ExactRDKitTargetResponseEvaluator(spec).evaluate(
        (
            TargetedCandidateRecord(46.069, ethanol, True, True),
            TargetedCandidateRecord(46.069, None, False, False),
            TargetedCandidateRecord(142.286, decane, True, True),
        )
    )
    assert report["attempts"] == 3
    assert report["valid_candidates"] == 2
    assert report["validity_fraction"] == pytest.approx(2 / 3)
    assert report["mean_absolute_error_valid_candidates"] < 0.01
    assert report["target_response"]["flexible_size_proof_required"] is True
    assert (
        report["target_response"]["mean_prediction_monotonic_non_decreasing"]
        is True
    )
    assert (
        report["target_response"][
            "mean_heavy_atom_count_monotonic_non_decreasing"
        ]
        is True
    )
    assert report["griddd_zinc_reported_reference"] == {
        "mae": 4.89,
        "validity_fraction": 0.842,
        "direct_comparison_claim_authorized": False,
    }
