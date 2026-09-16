from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.hybrid_top3_v0_controller import freeze_cell_schedule
from compose_v4.control.target_conditioned_utility_selector import (
    TARGETS,
    MeasuredEndpoint,
    UtilityRanker,
    endpoint_features,
    fixed_structural_graph_pair_score,
    fixed_structural_score,
    graph_pair_features,
)


def _ranker(*, conditioned: bool = False) -> UtilityRanker:
    width = len(graph_pair_features("CC", "CCC"))
    return UtilityRanker(
        tuple(np.zeros(width)),
        tuple(np.ones(width)),
        tuple(np.zeros(width)),
        tuple(tuple(np.zeros(width)) for _ in TARGETS),
        TARGETS,
        conditioned,
        ("fa7", "jak2", "parp1"),
        "fixture-training",
    )


def _row(rank: int, smiles: str) -> dict:
    candidate_id = identity({"rank": rank, "smiles": smiles})
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "source_case_id": "a" * 64,
        "rank": rank,
        "cell": "jak2_1",
        "target": "jak2",
        "fold": 1,
        "generator_revision": "baseline",
        "arm_id": "baseline_learned",
        "policy_id": "balanced_joint_autoregressive",
        "source_canonical_smiles": "CC",
        "eligible": True,
        "eligibility_exclusion_reasons": [],
        "final_exclusion_reasons": [],
        "exact_action_replay": True,
        "realization": {
            "status": "realized",
            "endpoint_matches_bound_target": True,
            "primitive_teacher_actions_used": 0,
        },
    }


def test_unscored_graph_pair_interface_is_decision_equivalent() -> None:
    row = MeasuredEndpoint(
        row_id="row",
        artifact="fixture",
        target="jak2",
        cell="jak2_1",
        source_idx=1,
        delta=0.4,
        oracle_protocol="fixture",
        docking_seed=1,
        docking_score=-1.0,
        source_smiles="CC",
        endpoint_smiles="CCC",
        source_state_sha256="a" * 64,
        endpoint_state_sha256="b" * 64,
        scaffold="C",
        lineage_ids=("fixture",),
        provenance={},
    )
    ranker = _ranker()

    assert np.array_equal(endpoint_features(row), graph_pair_features("CC", "CCC"))
    assert fixed_structural_score(row) == fixed_structural_graph_pair_score("CC", "CCC")
    assert ranker.score(row) == ranker.score_graph_pair("CC", "CCC", target="jak2")


def test_freeze_cell_schedule_uses_three_distinct_endpoints_then_v0() -> None:
    rows = [_row(1, "CCC"), _row(2, "CCCC"), _row(3, "CCN"), _row(4, "CCO")]
    rows[1]["final_exclusion_reasons"] = ["eligible_not_selected_by_prior_panel"]
    schedule = freeze_cell_schedule(
        cell="jak2_1",
        candidate_rows=rows,
        ranker=_ranker(),
        checkpoint_fold=1,
        report_calls=[1, 5, 10, 20, 50, 100],
    )

    assert schedule["status"] == "three_macro_calls_then_dynamic_v0"
    assert [row["selection_role"] for row in schedule["initial_macro_calls"]] == [
        "fixed_target_blind_structural",
        "target_blind_utility",
        "target_blind_utility",
    ]
    assert len({row["canonical_smiles"] for row in schedule["initial_macro_calls"]}) == 3
    assert schedule["dynamic_v0_start_call"] == 4
    assert schedule["dynamic_v0_calls_through_100"] == 97


def test_freeze_cell_schedule_fails_closed_to_v0_when_macro_support_is_short() -> None:
    schedule = freeze_cell_schedule(
        cell="jak2_1",
        candidate_rows=[],
        ranker=_ranker(),
        checkpoint_fold=1,
        report_calls=[1, 5, 10, 20, 50, 100],
    )

    assert schedule["status"] == "macro_support_abstention_dynamic_v0_from_call_1"
    assert schedule["initial_macro_calls"] == []
    assert schedule["dynamic_v0_start_call"] == 1
    assert schedule["dynamic_v0_calls_through_100"] == 100


def test_freeze_cell_schedule_rejects_target_conditioned_checkpoint() -> None:
    with pytest.raises(ValueError, match="target-blind"):
        freeze_cell_schedule(
            cell="jak2_1",
            candidate_rows=[_row(1, "CCC"), _row(2, "CCCC"), _row(3, "CCN")],
            ranker=_ranker(conditioned=True),
            checkpoint_fold=1,
            report_calls=[1, 5, 10, 20, 50, 100],
        )


def test_contract_is_self_hashed_and_zero_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / "configs/t4_hybrid_top3_v0_controller_v1.json").read_text())
    assert document["contract_sha256"] == identity(document["payload"])
    assert document["payload"]["authority"] == {
        "oracle_calls_authorized": 0,
        "docking_calls_authorized": 0,
        "modal_launches_authorized": 0,
        "live_run_access_authorized": False,
        "model_fitting_authorized": False,
        "candidate_generation_authorized": False,
    }


def test_sealed_lock_records_supported_cells_and_fail_closed_abstentions() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "diagnostics/t4_hybrid_top3_v0_controller/attempt_1/result.json").read_text()
    )

    assert result["payload_sha256"] == identity(result["payload"])
    assert result["payload"]["summary"] == {
        "abstention_cells": ["5ht1b_0", "braf_1", "fa7_0"],
        "all_five_cells_have_three_eligible_independent_candidates": False,
        "declared_cells": 5,
        "frozen_initial_macro_calls": 6,
        "macro_support_abstention_cells": 3,
        "supported_cells": ["jak2_1", "parp1_0"],
        "three_candidate_macro_supported_cells": 2,
        "total_calls_if_full_five_cell_run_authorized": 500,
        "unchanged_dynamic_v0_calls_if_full_five_cell_run_authorized": 494,
    }
    assert result["payload"]["new_oracle_calls"] == 0
    assert result["payload"]["new_docking_calls"] == 0
