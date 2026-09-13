import json
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_target_program_wave import (
    build_task_programs,
    enumerate_target_neighborhood,
    load_contract,
    score_values,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def test_contract_freezes_five_tasks_and_eighty_queries():
    contract = load_contract(ROOT)

    assert len(contract["tasks"]) == 5
    assert contract["oracle"]["total_query_ceiling"] == 80
    assert contract["information_regime"]["candidate_injection"] is False
    assert contract["information_regime"]["held_out_claim"] is False


def test_unscored_neighborhood_is_unique_and_keeps_exact_target_first():
    contract = load_contract(ROOT)
    target = contract["tasks"]["albuterol_similarity"]["target_smiles"]
    rows = enumerate_target_neighborhood(target)

    assert rows[0]["candidate_id"] == "exact_target"
    assert len(rows) >= 15
    assert len({row["smiles"] for row in rows}) == len(rows)


def test_real_albuterol_programs_replay_from_the_declared_unrelated_root():
    contract = load_contract(ROOT)
    source = json.loads((ROOT / contract["source_root"]["path"]).read_text())["result"]
    row = build_task_programs(
        source["source_state"],
        canonical_state_key(decode_state(source["source_state"])),
        "albuterol_similarity",
        contract["tasks"]["albuterol_similarity"]["target_smiles"],
        contract,
    )

    assert row["base_route"]["primitive_steps"] <= 96
    assert len(row["programs"]) == 15
    assert len({program["receipt"]["endpoint"] for program in row["programs"]}) == 15


def test_official_metric_finishes_with_the_last_top_ten_value():
    contract = load_contract(ROOT)
    result = score_values([0.1] + [1.0] * 15, contract)

    assert result["oracle_calls"] == 16
    assert result["best_score"] == 1.0
    assert result["final_top10"] == 1.0
    assert result["auc_top10_official_10k"] == pytest.approx(0.9992)
