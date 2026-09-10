import copy

import pytest

from compose_v4.experiments.pmo_sibling_audit import MODELS, audit_rows


def row(query, parent, score, cutoff=20):
    # Synthetic accounting fixture, not invented molecular evidence.
    return {
        "id": str(query),
        "query": query,
        "parent_id": parent,
        "parent_index": 0,
        "parent_query": 1,
        "parent_score": 0.3,
        "parent_smiles": "fixture",
        "fit_through_query": cutoff,
        "score": score,
        "cycle_rank_increase": query % 2 == 0,
        "predictions": {m: 0.3 for m in MODELS},
    }


def test_exact_parent_and_snapshot_grouping_with_honest_ties():
    rows = [row(21, "a", 0.2), row(22, "a", 0.8), row(23, "b", 0.9), row(31, "a", 0.1, 30)]
    rows[0]["predictions"]["edit_delta"] = 0.1
    rows[1]["predictions"]["edit_delta"] = 0.9
    report = audit_rows(rows)
    assert report["primary"]["groups"] == 1
    assert report["primary"]["ranking_pairs"] == 1
    assert {r["id"] for r in report["excluded"]} == {"23", "31"}
    baseline = report["primary"]["metrics"]["parent_copy"]
    assert baseline["concordance"] == 0.5
    assert baseline["decisive_coverage"] == 0
    assert baseline["decisive_accuracy"] is None
    assert baseline["mean_selected_score"] == 0.5
    assert baseline["mean_regret"] == pytest.approx(0.3)
    assert baseline["mean_gain_vs_uniform"] == 0
    assert report["primary"]["metrics"]["edit_delta"]["concordance"] == 1
    changed = copy.deepcopy(rows)
    changed[-1]["score"] = 1.0
    assert audit_rows(changed)["groups"] == report["groups"]
    rows[-1]["fit_through_query"] = rows[-1]["query"]
    with pytest.raises(ValueError, match="nonchronological"):
        audit_rows(rows)


def test_true_ties_and_no_sibling_coverage_are_explicit():
    report = audit_rows([row(21, "a", 0.4), row(22, "a", 0.4)])
    assert report["primary"]["equal_score_groups"] == 1
    assert report["primary"]["ranking_pairs"] == 0
    assert report["primary"]["metrics"]["edit_delta"]["concordance"] is None
    assert report["primary"]["metrics"]["edit_delta"]["mean_regret"] == 0
    empty = audit_rows([row(21, "a", 0.4)])["primary"]
    assert empty["groups"] == 0
    assert empty["metrics"]["edit_delta"]["mean_selected_score"] is None
