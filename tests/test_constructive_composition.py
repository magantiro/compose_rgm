"""Guards for the deep constructive proposal law.

The law exists because eligibility is free and docking is not: it generates in volume
and filters locally, so the only thing that must never slip is that filtering stays
exact, the declared support stays enforced, and the pool it hands on is deduplicated
by molecule rather than by program.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.constructive_composition import (
    DEFAULT_BUILDERS,
    MAX_BLOCKS,
    MAX_PRIMITIVES,
    compose,
    family_order,
    prospect,
)
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    synthesize_dynamic_program,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def _source(cell="parp1_0"):
    from compose_v4.rewrite.trace_shard import decode_state

    units = {
        u["cell"]: u for u in unseal(ROOT / "configs/t4_objective_reset_runtime_v1.json")["units"]
    }
    return decode_state(units[cell]["source_state"])


def _always(eligible: bool, reason="synthetic_refusal"):
    def evaluate(candidate):
        assert "smiles" in candidate
        return {
            "oracle_eligible": eligible,
            "endpoint_exclusion_reasons": [] if eligible else [reason],
        }

    return evaluate


# ---- Family order ----


def test_the_constructive_order_is_a_permutation_not_a_restriction():
    rng = np.random.default_rng(0)
    plain = family_order(rng, DEFAULT_BUILDERS, constructive=False)
    biased = family_order(rng, DEFAULT_BUILDERS, constructive=True)
    assert sorted(plain) == sorted(biased) == sorted(GENERIC_MODULES)


def test_builders_come_first_only_when_asked():
    rng = np.random.default_rng(1)
    biased = family_order(rng, DEFAULT_BUILDERS, constructive=True)
    assert set(biased[: len(DEFAULT_BUILDERS)]) == set(DEFAULT_BUILDERS)


def test_an_empty_builder_set_leaves_the_order_alone():
    rng = np.random.default_rng(np.random.SeedSequence([7]))
    a = family_order(np.random.default_rng(np.random.SeedSequence([7])), (), constructive=True)
    b = family_order(rng, (), constructive=False)
    assert a == b


# ---- Composition ----


def test_depth_must_fit_the_declared_block_support():
    rng = np.random.default_rng(0)
    for bad in (0, -1, MAX_BLOCKS + 1, 2.5):
        with pytest.raises(ValueError, match="declared"):
            compose(_source(), rng, bad)


def test_a_composed_program_stays_inside_the_declared_support():
    rng = np.random.default_rng(np.random.SeedSequence([20260916, 11]))
    trace, families = compose(_source(), rng, MAX_BLOCKS)
    assert 1 <= len(families) <= MAX_BLOCKS
    assert len(trace["actions"]) <= MAX_PRIMITIVES
    assert all(f in GENERIC_MODULES for f in families)


def test_v0_is_not_modified_by_this_module():
    """v0 keeps its own three-module cap; this law is additive, not a patch."""
    with pytest.raises(ValueError, match="one to three generic modules"):
        synthesize_dynamic_program(_source(), np.random.default_rng(0), max_modules=MAX_BLOCKS)


# ---- Prospecting ----


def test_prospecting_validates_its_budget_and_depths():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive attempt budget"):
        prospect(_source(), _always(True), rng, attempts=0)
    with pytest.raises(ValueError, match="declared"):
        prospect(_source(), _always(True), rng, attempts=4, depths=(9,))


def test_prospecting_makes_no_oracle_call_and_reports_its_realized_attempts():
    result = prospect(
        _source(), _always(False), np.random.default_rng(np.random.SeedSequence([3])), attempts=6
    )
    assert result["new_oracle_calls"] == 0
    assert result["attempted"] == 6 and result["attempt_budget"] == 6
    assert result["pool"] == [] and result["eligible_endpoints"] == 0
    assert result["attempts_per_eligible_endpoint"] is None


def test_an_ineligible_endpoint_records_why():
    result = prospect(
        _source(),
        _always(False, "qed_not_strictly_above_0.6"),
        np.random.default_rng(np.random.SeedSequence([4])),
        attempts=6,
        minimum_primitives=1,
    )
    assert result["rejections"].get("qed_not_strictly_above_0.6", 0) > 0


def test_the_size_floor_is_applied_before_the_gate_is_consulted():
    calls = []

    def evaluate(candidate):
        calls.append(candidate)
        return {"oracle_eligible": True}

    result = prospect(
        _source(),
        evaluate,
        np.random.default_rng(np.random.SeedSequence([5])),
        attempts=8,
        minimum_primitives=MAX_PRIMITIVES + 1,
    )
    assert result["pool"] == []
    assert result["below_size_floor"] > 0
    assert calls == [], "the gate should not be consulted for programs below the floor"


def test_the_pool_is_deduplicated_by_molecule_not_by_program():
    result = prospect(
        _source(),
        _always(True),
        np.random.default_rng(np.random.SeedSequence([6])),
        attempts=25,
        minimum_primitives=1,
    )
    endpoints = [row["endpoint"] for row in result["pool"]]
    assert len(endpoints) == len(set(endpoints))
    assert result["eligible_endpoints"] == len(result["pool"])
    assert all(row["primitives"] >= 1 for row in result["pool"])


def test_a_wall_budget_stops_early_and_says_so():
    result = prospect(
        _source(),
        _always(True),
        np.random.default_rng(np.random.SeedSequence([8])),
        attempts=10_000,
        minimum_primitives=1,
        wall_seconds=0.0,
    )
    assert result["attempted"] < result["attempt_budget"]


def test_every_pooled_row_carries_the_evidence_a_later_stage_needs():
    result = prospect(
        _source(),
        _always(True),
        np.random.default_rng(np.random.SeedSequence([9])),
        attempts=12,
        minimum_primitives=1,
    )
    assert result["pool"], "expected at least one composed program from a real source"
    row = result["pool"][0]
    assert {"endpoint", "primitives", "families", "trace", "changed_originals", "created"} <= set(
        row
    )
    assert row["trace"]["endpoint"] == row["endpoint"]
