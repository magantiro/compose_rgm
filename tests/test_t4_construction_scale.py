"""Guards for the construction-scale diagnostic.

The diagnostic's conclusion is that the docking value sits at a program size the
production sampler does not reach. That conclusion is only worth anything if the
banding, the builder rule and the composition law mean what they say, and if the
production sampler really is left alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    synthesize_dynamic_program,
)
from compose_v4.experiments.t4_construction_scale import (
    DEFAULT_BUILDERS,
    MAX_BLOCKS,
    MAX_PRIMITIVES,
    SCHEMA_VERSION,
    band_of,
    builder_families,
    compose_deep,
    eligibility_by_size,
    family_profile,
    root_construction_density,
    route_decomposition,
    rule_of_three_upper_bound,
    score_by_band,
    upper_bound_95,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_construction_scale_v1.json"
ARTIFACT = ROOT / "diagnostics/t4_proposal_prior/construction_scale_v1"


def _source(cell="parp1_0"):
    units = {
        u["cell"]: u for u in unseal(ROOT / "configs/t4_objective_reset_runtime_v1.json")["units"]
    }
    return decode_state(units[cell]["source_state"])


def _record(**overrides):
    row = {
        "cell": "parp1_0",
        "primitive_count": 4,
        "changed_slot_count": 2,
        "net_created": 1,
        "net_deleted": 0,
        "score_mean": -9.0,
        "construction_families": ["append_ring"],
        "input_state_sha256": "root",
        "endpoint_sha256": "e0",
    }
    row.update(overrides)
    return row


# ---- Banding ----


def test_bands_cover_their_edges_without_overlap():
    assert band_of(1) == "1-2" and band_of(2) == "1-2"
    assert band_of(3) == "3-5" and band_of(9) == "6-9"
    assert band_of(15) == "15-22" and band_of(22) == "15-22"
    assert band_of(23) == "23-32" and band_of(32) == "23-32"


# ---- Family profile and the builder rule ----


def test_the_profile_uses_only_single_family_records():
    records = [_record(construction_families=["append_ring"], net_created=6) for _ in range(30)]
    records += [_record(construction_families=["append_ring", "cycle_open"], net_created=0)] * 50
    profile = family_profile(records)
    assert set(profile) == {"append_ring"}
    assert profile["append_ring"]["median_created"] == 6


def test_a_family_below_the_minimum_is_not_profiled():
    assert family_profile([_record()] * 5) == {}


def test_a_builder_creates_at_least_as_much_as_it_disturbs():
    profile = {
        "append_ring": {"median_created": 6, "median_changed_originals": 1},
        "bond_reroute": {"median_created": 0, "median_changed_originals": 3},
        "even": {"median_created": 2, "median_changed_originals": 2},
        "creates_nothing": {"median_created": 0, "median_changed_originals": 0},
    }
    assert builder_families(profile) == ("append_ring", "even")


def test_the_builder_set_can_be_restricted_to_one_samplers_families():
    profile = {
        "append_ring": {"median_created": 6, "median_changed_originals": 1},
        "pendant_benzene": {"median_created": 8, "median_changed_originals": 1},
    }
    assert builder_families(profile, universe=GENERIC_MODULES) == ("append_ring",)


# ---- Route decomposition ----


def test_an_ineligible_interior_is_counted_per_route_not_per_prefix():
    descriptors = [
        {
            "dependency_regions": 2,
            "primitives": 17,
            "region_primitives": [8, 9],
            "created_handles": 9,
            "reused_created_handles": 7,
            "ineligible_internal_prefixes": 11,
        },
        {
            "dependency_regions": 1,
            "primitives": 5,
            "region_primitives": [5],
            "created_handles": 2,
            "reused_created_handles": 0,
            "ineligible_internal_prefixes": 0,
        },
    ]
    d = route_decomposition(descriptors)
    assert d["routes"] == 2
    assert d["routes_with_an_ineligible_interior"] == 1
    assert d["routes_reusing_a_created_handle"] == 1
    assert d["regions_per_route"] == {1: 1, 2: 1}
    assert d["primitives_per_region"]["median"] == 8.0


# ---- Where the value is ----


def test_scores_are_banded_only_within_the_named_cells():
    records = [
        _record(primitive_count=1, score_mean=-8.0),
        _record(primitive_count=20, score_mean=-14.0),
        _record(cell="braf_1", primitive_count=20, score_mean=-99.0),
    ]
    banded = score_by_band(records, ("parp1_0",))
    assert set(banded) == {"1-2", "15-22"}
    assert banded["15-22"]["best"] == -14.0
    assert banded["1-2"]["n"] == 1


def test_root_density_counts_only_constructions_applied_to_the_root():
    records = [
        _record(primitive_count=20, input_state_sha256="root", endpoint_sha256="a"),
        _record(primitive_count=20, input_state_sha256="root", endpoint_sha256="b"),
        _record(primitive_count=2, input_state_sha256="root", endpoint_sha256="c"),
        _record(primitive_count=20, input_state_sha256="descendant", endpoint_sha256="d"),
    ]
    density = root_construction_density(records, {"parp1_0": "root"})["parp1_0"]
    assert density["root_applied"] == 3
    assert density["at_or_above_threshold"] == 2
    assert density["distinct_endpoints"] == 2
    assert density["share"] == pytest.approx(2 / 3)


# ---- The composition law ----


def test_composition_depth_must_fit_the_declared_block_support():
    rng = np.random.default_rng(0)
    for bad in (0, MAX_BLOCKS + 1, 99):
        with pytest.raises(ValueError, match="declared"):
            compose_deep(_source(), rng, bad, constructive=False)


def test_a_composed_program_stays_inside_the_declared_support():
    rng = np.random.default_rng(np.random.SeedSequence([20260916, 1]))
    trace, families = compose_deep(_source(), rng, MAX_BLOCKS, constructive=True)
    assert 1 <= len(families) <= MAX_BLOCKS
    assert len(trace["actions"]) <= MAX_PRIMITIVES
    assert all(f in GENERIC_MODULES for f in families)


def test_the_constructive_order_prefers_builders():
    rng = np.random.default_rng(np.random.SeedSequence([20260916, 2]))
    _, families = compose_deep(_source(), rng, 4, constructive=True)
    assert families[0] in DEFAULT_BUILDERS


def test_the_builder_set_is_a_parameter_not_module_state():
    """A diagnostic that mutated a global would repeat the autograd-leak defect."""
    assert not hasattr(compose_deep, "builders")
    rng = np.random.default_rng(np.random.SeedSequence([20260916, 4]))
    _, families = compose_deep(_source(), rng, 3, constructive=True, builders=("cycle_open",))
    assert families[0] == "cycle_open"
    assert DEFAULT_BUILDERS == ("append_ring", "fuse_ring", "functionalize", "segment_grow")


def test_the_production_shallow_sampler_is_left_alone():
    """v0's own three-module cap must still refuse; its lane is not modified here."""
    with pytest.raises(ValueError, match="one to three generic modules"):
        synthesize_dynamic_program(_source(), np.random.default_rng(0), max_modules=8)


def test_eligibility_by_size_makes_no_oracle_call():
    curve = eligibility_by_size(
        _source(),
        lambda candidate: {"oracle_eligible": False, "endpoint_exclusion_reasons": ["synthetic"]},
        np.random.default_rng(np.random.SeedSequence([20260916, 3])),
        attempts=4,
        constructive=True,
    )
    assert curve["new_oracle_calls"] == 0
    assert all(band["eligible"] == 0 for band in curve["bands"].values())
    assert curve["rejections"] == {"synthetic": sum(b["n"] for b in curve["bands"].values())}


# ---- Statistics ----


def test_rule_of_three():
    assert rule_of_three_upper_bound(211) == pytest.approx(0.01422, abs=1e-5)
    with pytest.raises(ValueError):
        rule_of_three_upper_bound(0)


def test_the_binomial_bound_reduces_to_the_rule_of_three_at_zero_successes():
    for trials in (100, 211, 249):
        assert upper_bound_95(0, trials) == pytest.approx(
            rule_of_three_upper_bound(trials), rel=0.02
        )


def test_one_success_is_not_reported_as_zero():
    """1/249 is a rate with a bound, not an absence; both must survive the artifact."""
    assert upper_bound_95(1, 249) > upper_bound_95(0, 249)
    assert upper_bound_95(1, 249) == pytest.approx(0.0189, abs=5e-4)
    with pytest.raises(ValueError):
        upper_bound_95(2, 1)


# ---- Published artifact ----


def test_the_published_diagnostic_is_zero_oracle_and_bound_to_its_contract():
    payload = unseal(ARTIFACT / "result.json")
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["new_oracle_calls"] == 0 and payload["new_labels"] == 0
    assert all(c["new_oracle_calls"] == 0 for c in payload["eligibility_by_size"].values())


def test_the_published_baseline_matches_its_own_curves():
    payload = unseal(ARTIFACT / "result.json")
    trials = successes = 0
    for curve in payload["eligibility_by_size"].values():
        for key, band in curve["bands"].items():
            if key in ("15-22", "23-32"):
                trials += band["n"]
                successes += band["eligible"]
    assert payload["baseline"]["trials"] == trials
    assert payload["baseline"]["successes"] == successes
    assert payload["baseline"]["upper_bound_95"] == pytest.approx(upper_bound_95(successes, trials))


def test_the_contract_freezes_a_prospective_gate_and_discloses_the_post_hoc_parts():
    contract = unseal(CONTRACT)
    gate = contract["prospective_lane_gate"]
    assert gate["pass"] and gate["kill"] and gate["minimum_attempts"] >= 200
    assert "ACCESS" in gate["claim_boundary"]
    assert "post_hoc_disclosure" in contract
    assert contract["declared_support"]["max_blocks_per_program"] == MAX_BLOCKS
