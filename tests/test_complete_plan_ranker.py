import pytest

from compose_v4.control.complete_plan_ranker import fit, predict, validate
from compose_v4.experiments.pmo_complete_plan_ranker import (
    _bootstrap_mean,
    select_by_pool,
)


def test_complete_plan_ranker_round_trip_and_unseen_prediction():
    examples = [
        {"product_smiles": "CC", "parent_score": 0.2, "score": 0.3},
        {"product_smiles": "CCC", "parent_score": 0.3, "score": 0.25},
        {"product_smiles": "CCO", "parent_score": 0.1, "score": 0.4},
    ]
    model = fit(examples, data_sha256="a" * 64)
    validate(model)
    predictions = predict(
        model,
        [
            {
                "candidate_id": "candidate",
                "worker_id": "worker",
                "smiles": "CCN",
                "source": "CN",
                "parent_score": 0.4,
                "phase": 1,
                "slot": 0,
                "role": "actor_top",
                "release": 0.2,
                "primitive_steps": 3,
            }
        ],
    )
    assert len(predictions) == 1
    assert 0 <= predictions[0]["predicted_score"] <= 1
    assert -1 <= predictions[0]["predicted_delta"] <= 1
    assert 0 <= predictions[0]["max_training_kernel"] <= 1


def test_pool_selection_keeps_one_highest_prediction_per_pool():
    rows = [
        {
            "source": "CC",
            "worker_id": "worker",
            "phase": 1,
            "slot": 0,
            "role": role,
            "predicted_score": score,
            "smiles": smiles,
        }
        for role, score, smiles in (
            ("actor_top", 0.3, "CCC"),
            ("uniform_hash", 0.5, "CCCC"),
            ("largest_release", 0.4, "CCCCC"),
        )
    ]
    assert select_by_pool(rows)[0]["role"] == "uniform_hash"


def test_choice_bootstrap_is_deterministic_and_uses_parent_units():
    first = _bootstrap_mean([0.05, 0.10, 0.15], seed=7, draws=256)
    second = _bootstrap_mean([0.05, 0.10, 0.15], seed=7, draws=256)
    assert first == second
    assert first["mean"] == pytest.approx(0.10)
    assert first["unit"] == "exact paid parent structure"
    assert first["ci95"][0] > 0
