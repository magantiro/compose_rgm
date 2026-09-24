"""Keep analysis denominators and metric reconciliation fail-closed."""

import pytest

from tools import inspect_fragment_program_pilot as inspection


def test_quality_counts_unique_passes_over_all_attempts(monkeypatch):
    monkeypatch.setattr(inspection.QED, "qed", lambda _: 0.7)
    monkeypatch.setattr(inspection.sascorer, "calculateScore", lambda _: 3.9)
    result = inspection.describe(
        [{"committed_smiles": "CCO"}, {"committed_smiles": "CCO"}, {"committed_smiles": None}],
        {"quality": 100 / 3},
    )
    assert result["attempts"] == 3
    assert result["unique_joint_pass"] == 1
    assert result["groups"] == {"both_pass": 2, "no_output": 1}
    assert result["molecules"][1]["duplicate"]


def test_diagnostic_must_reproduce_official_quality(monkeypatch):
    monkeypatch.setattr(inspection.QED, "qed", lambda _: 0.7)
    monkeypatch.setattr(inspection.sascorer, "calculateScore", lambda _: 3.9)
    with pytest.raises(ValueError, match="does not reproduce"):
        inspection.describe([{"committed_smiles": "CCO"}], {"quality": 0})


def test_disconnected_commit_is_an_error():
    with pytest.raises(ValueError, match="invalid/disconnected"):
        inspection.describe([{"committed_smiles": "CC.O"}], {"quality": 0})
