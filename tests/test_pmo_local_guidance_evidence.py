import pytest

from tools.pmo_local_guidance_evidence import ranking_evidence


def test_saved_rank_uses_canonical_ties_and_reports_absence():
    selection = {"pool": ["CO", "CC", "CN"], "predictions": [0.6, 0.6, 0.7]}
    assert ranking_evidence(selection, "CC")["rank"] == 2
    assert ranking_evidence(selection, "CO")["rank"] == 3
    absent = ranking_evidence(selection, "CCC")
    assert not absent["present"]
    assert absent["rank"] is absent["prediction"] is None


def test_saved_rank_rejects_inconsistent_or_aliased_pools():
    with pytest.raises(ValueError, match="unequal"):
        ranking_evidence({"pool": ["CC"], "predictions": []}, "CC")
    with pytest.raises(ValueError, match="duplicate"):
        ranking_evidence({"pool": ["CC", "CC"], "predictions": [0.1, 0.2]}, "CC")
