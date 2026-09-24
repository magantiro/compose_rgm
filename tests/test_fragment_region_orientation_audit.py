"""Training-only region orientation audit checks."""

from pathlib import Path

import pytest
from audit_fragment_region_orientation import audit

CATALOG = Path("diagnostics/fragment_training_region_catalog_v1/catalog.json")


@pytest.mark.skipif(not CATALOG.exists(), reason="sealed training region catalog absent")
def test_orientation_audit_keeps_source_balance_and_catalog_identity():
    result = audit(CATALOG)
    assert result["one_boundary_entries"] <= result["catalog_entries"]
    assert result["smaller_side_instances"] > 0
    assert result["larger_side_instances"] > 0
    assert sum(result["source_balanced_smaller_side_size_probability"].values()) == pytest.approx(1)
    assert not result["qed_sa_used"]
    assert not result["benchmark_prompts_used"]
