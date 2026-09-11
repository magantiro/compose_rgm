import pytest

from tools.pmo_program_ranking import ranking_metrics, reconcile


def edge(parent, product, score=0.4, origin=1):
    return {
        "parent_smiles": parent,
        "parent_score": 0.5,
        "smiles": product,
        "score": score,
        "option": "donor_transplant",
        "origin": origin,
    }


def test_split_excludes_either_training_endpoint_and_preserves_origins():
    split = reconcile(
        [edge("A", "B"), edge("A", "B", origin=2)], [edge("C", "B"), edge("A", "D"), edge("C", "D")]
    )
    assert len(split["train"]) == 1
    assert split["train"][0]["origins"] == [1, 2]
    assert len(split["excluded_validation"]) == 2
    assert [(r["parent_smiles"], r["smiles"]) for r in split["validation"]] == [("C", "D")]
    with pytest.raises(ValueError, match="conflicting"):
        reconcile([edge("A", "B")], [edge("C", "B", 0.9)])


def test_rank_precision_coverage_and_choice_gain_are_distinct():
    rows = [
        {**edge("A", "B", 0.8), "utility": 0.2, "parent_utility": 0},
        {**edge("A", "C", 0.2), "utility": 0.1, "parent_utility": 0},
        {**edge("D", "E", 0.6), "utility": -0.2, "parent_utility": 0},
    ]
    result = ranking_metrics(rows)
    assert result["parents"] == 2
    assert result["multi_candidate_parents"] == 1
    assert result["pair_preference_precision"] == 1
    assert result["positive_precision"] == 0.5
    assert result["positive_recall"] == 0.5
    assert result["mean_choice_gain_over_uniform"] == pytest.approx(0.3)


def test_batched_choice_matches_existing_policy_and_never_reads_labels():
    import numpy as np

    from compose_v4.control.branch_policy import BranchPolicy
    from compose_v4.control.program_selection import choose_pools

    observations = [edge("CCO", "CCN", 0.2), edge("CCO", "CCC", 0.8)]
    model = BranchPolicy.fit(observations, source_sha256="a" * 64)
    pool = {"candidates": observations, "reference": [0.3, 0.7]}
    expected, _ = model.distribution(observations, pool["reference"], guided=True)
    result = choose_pools([pool, {"candidates": [], "reference": []}], model, seed=37)
    np.testing.assert_allclose(result[0]["probabilities"], expected, atol=1e-12)
    assert result[1]["status"] == "empty_pool"
    observations[0]["score"], observations[1]["score"] = 1, 0
    assert choose_pools([pool, {"candidates": [], "reference": []}], model, seed=37) == result
