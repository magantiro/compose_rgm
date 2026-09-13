from pathlib import Path

import pytest

from compose_v4.experiments.pmo_ivg_oracle_parity import (
    _source_task_rows,
    build_query_lock,
    load_contract,
    score_values,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_freezes_eleven_tasks_and_all_parity_calls():
    contract = load_contract(ROOT)

    assert len(contract["tasks"]) == 11
    assert sum(row["queries"] for row in contract["tasks"].values()) == 150
    assert contract["oracle"]["authoritative_query_ceiling"] == 150
    assert contract["oracle"]["precontract_diagnostic_calls"] == 1
    assert contract["oracle"]["total_new_parity_environment_calls"] == 151
    assert contract["information_regime"]["candidate_reselection"] is False
    assert contract["information_regime"]["general_pmo_claim"] is False


def test_query_lock_preserves_all_source_rows_without_selection():
    lock = build_query_lock(ROOT, load_contract(ROOT))

    assert lock["task_count"] == 11
    assert lock["authoritative_query_count"] == 150
    assert lock["new_oracle_calls"] == 0
    assert len(lock["tasks"]["perindopril_mpo"]["queries"]) == 15
    assert len(lock["tasks"]["albuterol_similarity"]["queries"]) == 16
    assert len(lock["tasks"]["isomers_c7h8n2o2"]["queries"]) == 11
    assert all(
        row["source_query_index"] == index
        for task in lock["tasks"].values()
        for index, row in enumerate(task["queries"])
    )


def test_source_normalization_supports_direct_perindopril_rows():
    payload = {
        "contract": {"task": "perindopril_mpo"},
        "auc_top10_official_10k": 0.5,
        "rows": [
            {
                "query_index": 0,
                "role": "root",
                "source_id": "root/0",
                "smiles": "CCO",
                "score": 0.25,
            }
        ],
    }

    rows, auc = _source_task_rows(payload, "perindopril_mpo")

    assert auc == 0.5
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["source_query"]["role"] == "root"
    assert rows[0]["prior_protocol_score"] == 0.25


def test_official_metric_finishes_with_locked_top_ten_value():
    summary = score_values([0.0] + [1.0] * 10, load_contract(ROOT))

    assert summary["oracle_calls"] == 11
    assert summary["best_score"] == 1.0
    assert summary["final_top10"] == 1.0
    assert summary["auc_top10_official_10k"] == pytest.approx(0.99945)
