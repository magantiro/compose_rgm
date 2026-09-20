"""Focused invariants for the frozen route-free v4 support gate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_feasibility_headroom_composer_v4 import (
    FreeFeasibilitySpec,
    _proposal_rank,
    allocate_feasibility_headroom_plans,
    propose_feasibility_headroom_plan,
)
from compose_v4.experiments.t4_nodistill_feasibility_headroom_gate_v4 import (
    allocated_plan_names,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(index: int):
    registry = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    return pad_molecular_graph(smiles_to_molecular_graph(registry[index]["smiles"]), 48)


def test_frozen_contract_and_runtime_boundary_have_no_identity_or_teacher_input():
    contract, payload_identity = load_contract(ROOT)
    assert payload_identity == (
        "bede1c39cbd39fdea8c3a5981192bac98597b839bdb6bae1d0e1e886188816ae"
    )
    assert contract["status"] == "FROZEN_ZERO_ORACLE_SUPPORT_GATE"
    assert tuple(inspect.signature(propose_feasibility_headroom_plan).parameters) == (
        "source",
        "seed",
        "feasibility",
        "plan_name",
        "attempts_per_plan",
        "candidate_quota_per_plan",
        "maximum_primitives",
        "maximum_blocks",
    )
    forbidden = {"target", "cell", "teacher", "route", "template", "objective"}
    assert not forbidden.intersection(
        inspect.signature(propose_feasibility_headroom_plan).parameters
    )


def test_continuous_allocator_preserves_total_budget_and_exploration_floor():
    _features, headrooms, plans, allocation = allocate_feasibility_headroom_plans(
        _source(12),
        feasibility=FreeFeasibilitySpec(0.6),
        attempts_per_plan=128,
        candidate_quota_per_plan=8,
    )
    assert len(plans) >= 12
    assert sum(plan.attempt_budget for plan in plans) == 128 * len(plans)
    assert sum(plan.candidate_quota for plan in plans) == 8 * len(plans)
    assert all(plan.attempt_budget >= 1 and plan.candidate_quota >= 1 for plan in plans)
    assert len({plan.attempt_budget for plan in plans}) > 1
    assert len({plan.compatibility for plan in plans}) > 1
    assert headrooms["normalized_allocation_headrooms"][0] == 0.4
    assert allocation["allocation_policy"]["uniform_exploration_fraction"] == 0.25


def test_capacity_abstention_is_generic_and_deterministic():
    assert allocated_plan_names(repository_root=ROOT, cell_key="braf_0_d06") == ()
    first = allocated_plan_names(repository_root=ROOT, cell_key="5ht1b_0_d04")
    second = allocated_plan_names(repository_root=ROOT, cell_key="5ht1b_0_d04")
    assert first == second
    assert all(
        name.startswith(("grow_then_close__", "replace_then_ring__")) for name in first
    )


def test_rank_is_continuous_and_does_not_promote_binary_feasibility():
    plan = {"desired_delta_heavy_atoms": [3, 7]}
    better_continuous = SimpleNamespace(
        endpoint_key="a",
        metadata={
            "macro_plan": plan,
            "observed_macro_fields": {"delta_heavy_atoms": 5},
            "endpoint_headrooms": {
                "connected": False,
                "structurally_valid": False,
                "binary_free_feasible": False,
                "continuous_total_deficit": 0.1,
                "continuous_minimum_margin": -0.1,
            },
        },
    )
    worse_continuous = SimpleNamespace(
        endpoint_key="b",
        metadata={
            "macro_plan": plan,
            "observed_macro_fields": {"delta_heavy_atoms": 5},
            "endpoint_headrooms": {
                "connected": True,
                "structurally_valid": True,
                "binary_free_feasible": True,
                "continuous_total_deficit": 0.2,
                "continuous_minimum_margin": -0.2,
            },
        },
    )
    assert _proposal_rank(better_continuous) < _proposal_rank(worse_continuous)


def test_free_thresholds_fail_closed_outside_frozen_support():
    for kwargs in (
        {"similarity_minimum": -0.1},
        {"similarity_minimum": 1.1},
        {"similarity_minimum": 0.4, "heavy_atom_maximum": 41},
    ):
        try:
            FreeFeasibilitySpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v4 support unexpectedly accepted")
