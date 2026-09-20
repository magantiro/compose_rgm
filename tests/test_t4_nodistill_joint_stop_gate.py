"""Focused zero-oracle checks for the standalone NoDistill support gate."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from compose_v4.control.nodistill_joint_stop import (
    JointStopConfig,
    _retain,
    generate_candidate_lock,
    strata_census,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_nodistill_joint_stop_gate import load_contract

ROOT = Path(__file__).resolve().parents[1]


def _record(index: int, *, strata: tuple[str, str, str, str, str]) -> dict:
    fields = (
        "scale",
        "delta_heavy",
        "delta_cycle",
        "retained_interfaces",
        "rewrite_mode",
    )
    return {
        "candidate_id": str(index),
        "endpoint": "C" * (index + 1),
        "constituents": [
            {
                "family": "functionalize",
                "local_primitives": 1,
                "local_delta_heavy": 1,
                "local_delta_cycle": 0,
            },
            {
                "family": "functionalize",
                "local_primitives": 1,
                "local_delta_heavy": 1,
                "local_delta_cycle": 0,
            },
        ],
        "features": {
            "depth": 2,
            "peak_heavy": 10,
            "interface_continuity": 0,
            "distinct_rewrite_modes": 1,
            "distinct_interfaces": 1,
            "primitives": 2,
            "blocks": 2,
            "strata": dict(zip(fields, strata, strict=True)),
        },
    }


def test_coverage_retention_reserves_joint_strata_that_marginal_collapses():
    common = ("small", "grow", "stable", "one_boundary", "grow")
    records = [_record(index, strata=common) for index in range(4)]
    records.extend(
        [
            _record(
                4,
                strata=(
                    "medium",
                    "major_shrink",
                    "decrease",
                    "mixed",
                    "mixed",
                ),
            ),
            _record(
                5,
                strata=("large", "stable", "increase", "two_boundary", "mixed"),
            ),
        ]
    )

    marginal = _retain(records, "independent_marginal", 3)
    proposed = _retain(records, "coverage_joint_stop", 3)

    assert len({tuple(r["features"]["strata"].values()) for r in marginal}) == 1
    assert len({tuple(r["features"]["strata"].values()) for r in proposed}) == 3


def test_graph_only_runtime_is_deterministic_complete_and_budget_preserving():
    source = production_state_from_smiles("C1CCCCC1", max_atoms=48)
    config = JointStopConfig(
        max_depth=2,
        beam_width=4,
        lock_width=6,
        candidates_per_family=2,
        expansions_per_parent=6,
    )

    first = generate_candidate_lock(source, arm="coverage_joint_stop", config=config)
    second = generate_candidate_lock(source, arm="coverage_joint_stop", config=config)

    assert first == second
    assert first["candidates"]
    assert all(
        row["features"]["depth"] == 2 and row["stop"] for row in first["candidates"]
    )
    assert all(row["exact_execution"] for row in first["candidates"])
    assert first["support"] == {
        "max_primitives": 32,
        "max_blocks": 8,
        "max_active_atoms": 40,
        "depths": [2, 2],
    }
    assert strata_census(first)["joint_strata"] >= 2


def test_runtime_boundary_accepts_no_task_teacher_route_or_objective_input():
    parameters = tuple(inspect.signature(generate_candidate_lock).parameters)
    assert parameters == ("source", "arm", "config")
    runtime = generate_candidate_lock(
        production_state_from_smiles("CCO", max_atoms=48),
        arm="independent_marginal",
        config=JointStopConfig(
            max_depth=2,
            beam_width=2,
            lock_width=2,
            candidates_per_family=1,
            expansions_per_parent=3,
        ),
    )["runtime_inputs"]
    assert runtime["source_graph"] is True
    assert all(
        value is False for key, value in runtime.items() if key != "source_graph"
    )


def test_support_expansion_fails_closed_and_contract_is_self_hashed():
    for field, value in (
        ("max_primitives", 33),
        ("max_blocks", 9),
        ("max_active_atoms", 41),
    ):
        with pytest.raises(ValueError, match="may not change"):
            JointStopConfig(**{field: value})

    contract = load_contract(ROOT)
    assert len(contract["cases"]) == 2
    assert contract["acceptance"]["negative_result_is_authoritative"] is True
    assert contract["scope"]["production_integration"] is False
    assert (
        contract["information_regime"]["candidate_lock_before_teacher_comparison"]
        is True
    )
