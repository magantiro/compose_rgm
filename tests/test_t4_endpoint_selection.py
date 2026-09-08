"""Endpoint screening does not change trajectory support or lose rejected records."""

import copy

import pytest
import torch
from test_t4_matched_pilot import population_fixture

from compose_v4.experiments.t4_endpoint_selection import (
    LEGACY_RANK_ALL,
    T4_FEASIBLE_ONLY,
    annotate_endpoints,
    feasible_endpoint,
)
from tools.t4_ring_quality_audit import benchmark_limits, ring_descriptors


def test_screen_preserves_records_and_generic_and_legacy_behavior():
    pool = [
        {"option": "generic", "qed": 0.8, "sa": 3.0, "sim": 0.6, "v": 0.0},
        {"option": "build_ring_system", "qed": 0.8, "sa": 4.2, "sim": 0.6, "v": 0.05},
    ]
    saved = copy.deepcopy(pool)
    screened = annotate_endpoints(pool, T4_FEASIBLE_ONLY)
    assert [r["oracle_eligible"] for r in screened] == [True, False]
    assert len(screened) == 2 and pool == saved
    assert annotate_endpoints(pool, LEGACY_RANK_ALL) is pool
    with pytest.raises(ValueError, match="unknown T4"):
        annotate_endpoints(pool, "silent_new_policy")


@pytest.mark.parametrize(
    "field,value",
    [("v", float("nan")), ("sa", float("inf")), ("sim", None), ("qed", True), ("v", -0.1)],
)
def test_missing_nonfinite_or_invalid_property_fails_closed(field, value):
    candidate = {"qed": 0.8, "sa": 3.0, "sim": 0.6, "v": 0.0, field: value}
    with pytest.raises(ValueError, match="endpoint"):
        feasible_endpoint(candidate)


@pytest.mark.parametrize(
    "smiles,bridgeheads,spiro",
    [
        ("c1ccc2ccccc2c1", 0, 0),
        ("c1ccccc1-c1ccccc1", 0, 0),
        ("C1CC2CCC1C2", 2, 0),
        ("C1CCC2(CC1)CCCC2", 0, 1),
    ],
)
def test_ring_descriptors_distinguish_fusion_from_bridge_and_spiro(smiles, bridgeheads, spiro):
    result = ring_descriptors(smiles)
    assert result["rdkit_bridgehead_atoms"] == bridgeheads
    assert result["rdkit_spiro_atoms"] == spiro


def test_existing_limits_are_read_without_new_thresholds():
    assert benchmark_limits() == {"qed_min": 0.6, "sa_max": 4.0}


@torch.enable_grad()
def test_real_selector_keeps_ineligible_products_but_never_docks_them(monkeypatch, tmp_path):
    from compose_v4.control import region, region_selector

    prepare = population_fixture(monkeypatch, tmp_path)
    largest = max(region.enumerate_regions("c1ccccc1"), key=lambda r: r.size)
    monkeypatch.setattr(region_selector, "sample_region", lambda *_a, **_kw: (largest, None))
    task = {
        "smiles": "c1ccccc1",
        "delta": 1.0,
        "target": "parp1",
        "budget": 20,
        "seed_rng": 1000,
        "lineages": 1,
        "regions_per_lineage": 1,
        "workers": 1,
        "particles_per_region": 1,
        "include_fused": True,
        "prepare_only": True,
        "primitive_guidance": "reference",
    }
    units = []
    legacy = prepare(task, units.append)
    strict = prepare(
        {**task, "endpoint_selection_policy": T4_FEASIBLE_ONLY}, parent_cache={0: units[0]}
    )
    assert len(legacy["take"]) == len(strict["pool"]) == 1
    assert strict["pool"][0]["sim"] < task["delta"]
    assert strict["pool"][0]["oracle_eligible"] is False
    assert strict["take"] == [] and strict["oracle_calls"] == 0
    assert strict["work"] == legacy["work"] and strict["bundles"] == legacy["bundles"]
