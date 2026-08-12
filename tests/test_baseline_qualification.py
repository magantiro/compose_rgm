from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.baseline_qualification import (
    NOVELTY_BEARING_AXES,
    BaselineQualificationError,
    load_qualification_registry,
    validate_qualification_registry,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_DIR = ROOT / "docs" / "workstreams" / "baseline-qualification"
REGISTRY_PATH = QUALIFICATION_DIR / "comparator_registry_v3.json"
RENDERER = ROOT / "scripts" / "render_comparator_registry.py"
CLAIM_REGISTRY_PATH = ROOT / "configs" / "comparator_registry_v3.json"


def _registry() -> dict:
    return load_qualification_registry(REGISTRY_PATH)


def test_registry_loads_and_is_design_only() -> None:
    registry = _registry()
    assert registry["artifact_status"] == "DESIGN_ONLY"
    assert registry["held_out_data_opened"] is False
    assert registry["smoke_plan"]["executed"] is False


def test_every_capability_cell_carries_a_primary_source() -> None:
    registry = _registry()
    for method in registry["methods"]:
        for axis, cell in method["capabilities"].items():
            assert cell["evidence"].strip(), f"{method['id']}.{axis} has no evidence"


def test_unsourced_capability_assertion_is_rejected() -> None:
    registry = copy.deepcopy(_registry())
    registry["methods"][0]["capabilities"]["source_conditioning"] = {
        "verdict": "YES",
        "evidence": "",
    }
    with pytest.raises(BaselineQualificationError, match="no primary source"):
        validate_qualification_registry(registry)


def test_native_capability_must_be_escalated() -> None:
    """A baseline that natively does a COMPOSE claim cannot be buried."""
    registry = copy.deepcopy(_registry())
    natively_supported = [
        (method["id"], axis)
        for method in registry["methods"]
        for axis in NOVELTY_BEARING_AXES
        if method["capabilities"][axis]["verdict"] == "YES"
    ]
    assert natively_supported, (
        "the registry records no baseline natively supporting a COMPOSE "
        "capability; if that is genuinely true it is a strong claim and this "
        "test should be replaced by one asserting the absence deliberately"
    )
    registry["compose_claims_a_baseline_does_natively"] = []
    with pytest.raises(BaselineQualificationError, match="not escalated"):
        validate_qualification_registry(registry)


def test_partial_novelty_capability_must_appear_as_a_near_miss() -> None:
    registry = copy.deepcopy(_registry())
    registry["near_misses"] = []
    registry["compose_claims_a_baseline_does_natively"] = [
        row
        for row in registry["compose_claims_a_baseline_does_natively"]
        if row["method"] == "REINVENT"
    ]
    with pytest.raises(BaselineQualificationError, match="near_misses"):
        validate_qualification_registry(registry)


def test_escalation_must_state_what_compose_still_has() -> None:
    """Half a finding is the dangerous half."""
    registry = copy.deepcopy(_registry())
    registry["compose_claims_a_baseline_does_natively"][0].pop("what_compose_still_has")
    with pytest.raises(BaselineQualificationError, match="what COMPOSE still"):
        validate_qualification_registry(registry)


def test_aggregate_verdict_cannot_be_softer_than_its_rows() -> None:
    registry = copy.deepcopy(_registry())
    method = next(row for row in registry["methods"] if row["method_verdict"] == "MUST_RUN")
    method["method_verdict"] = "CONTEXT_ONLY"
    with pytest.raises(BaselineQualificationError, match="aggregate verdict disagrees"):
        validate_qualification_registry(registry)


def test_must_run_methods_declare_where_the_comparison_is_unfair() -> None:
    registry = _registry()
    for method in registry["methods"]:
        if method["method_verdict"] != "MUST_RUN":
            continue
        contract = method["fairness_contract"]
        assert contract["matched_quantity"].strip()
        assert contract["not_fair_because"].strip()


def test_markdown_rendering_is_in_sync_with_the_registry() -> None:
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_claim_scoped_registry_agrees_with_the_qualification_verdicts() -> None:
    """The two registries must not drift: configs/ carries the verdict only."""
    qualification = _registry()
    verdicts = {row["id"]: row["method_verdict"] for row in qualification["methods"]}
    claim_registry = json.loads(CLAIM_REGISTRY_PATH.read_text())
    externals = [
        row for row in claim_registry["comparators"] if row.get("comparator_type") == "external"
    ]
    assert externals, "claim-scoped registry lost its external comparators"
    for row in externals:
        method_id = row["id"]
        assert method_id in verdicts, f"{method_id} has no qualification record"
        assert row["required_status"] == verdicts[method_id].lower(), (
            f"{method_id} status disagrees between the claim registry and the "
            "qualification registry"
        )
        assert row["qualification_record"] == (
            "docs/workstreams/baseline-qualification/comparator_registry_v3.json"
        )
