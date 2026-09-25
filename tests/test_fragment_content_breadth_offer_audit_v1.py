"""Behavior checks for the post-lock decoration offer audit."""

from __future__ import annotations

import pytest
from audit_fragment_content_breadth_offers_v1 import PILOT, audit, descriptor


def test_descriptor_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        descriptor("not_a_molecule")


def test_descriptor_reports_quality_components() -> None:
    result = descriptor("CCO")
    assert result["heavy_atoms"] == 3
    assert result["quality_pass"] == (result["qed_pass"] and result["sa_pass"])


@pytest.mark.skipif(
    not (PILOT / "summary.json").exists(),
    reason="sealed matched decoration pilot is not present",
)
def test_sealed_audit_reconciles_all_attempts() -> None:
    result = audit()
    assert len(result["prompts"]) == 20
    assert all(row["attempts"] == 20 for row in result["prompts"])
    assert {row["arm"] for row in result["prompts"]} == {
        "frozen",
        "uniform_within_cell",
    }
