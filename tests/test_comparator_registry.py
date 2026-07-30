from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.experiments.comparator_registry import (
    ComparatorRegistryError,
    load_comparator_registry,
    validate_comparator_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "comparator_registry_v1.json"


def test_comparator_registry_is_claim_complete_and_well_formed() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    assert registry["status"] == "PREIMPLEMENTATION"
    assert len(registry["external_small_molecule_baselines"]) == 6


def test_registry_rejects_missing_same_base_pareto_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    pareto = next(
        row
        for row in broken["claim_comparisons"]
        if row["claim_id"] == "pareto_and_dynamic_lead_optimization"
    )
    pareto["required_same_base"].remove("nsga2_over_compose_successors")
    with pytest.raises(ComparatorRegistryError, match="same-base Pareto"):
        validate_comparator_registry(broken)


def test_registry_rejects_external_baseline_without_primary_source() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    broken["external_small_molecule_baselines"][0]["source"] = "not-a-url"
    with pytest.raises(ComparatorRegistryError, match="primary HTTPS source"):
        validate_comparator_registry(broken)
