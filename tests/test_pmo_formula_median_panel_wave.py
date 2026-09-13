from pathlib import Path

import pytest

from compose_v4.experiments.pmo_formula_median_panel_wave import (
    build_curriculum,
    load_contract,
)
from compose_v4.experiments.pmo_target_program_wave import score_values

ROOT = Path(__file__).resolve().parents[1]


def test_contract_freezes_two_tasks_and_twenty_two_queries():
    contract = load_contract(ROOT)

    assert tuple(sorted(contract["tasks"])) == ("isomers_c7h8n2o2", "median1")
    assert contract["oracle"]["total_query_ceiling"] == 22
    assert all(len(row["candidates"]) == 10 for row in contract["tasks"].values())
    assert contract["information_regime"]["task_formula_used_for_candidate_design"]
    assert contract["information_regime"]["held_out_claim"] is False


def test_real_formula_and_median_programs_replay_from_frozen_task_roots():
    curriculum = build_curriculum(ROOT, load_contract(ROOT))

    assert curriculum["structural_gate"]["passed"] is True
    assert curriculum["structural_gate"]["programs_replayed"] == 20
    assert tuple(sorted(curriculum["sources"])) == (
        "isomers_c7h8n2o2",
        "median1",
    )


def test_official_metric_finishes_with_last_top_ten_value():
    result = score_values([0.0] + [1.0] * 10, load_contract(ROOT))

    assert result["oracle_calls"] == 11
    assert result["final_top10"] == 1.0
    assert result["auc_top10_official_10k"] == pytest.approx(0.99945)
