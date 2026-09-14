from __future__ import annotations

import numpy as np

from compose_v4.control.dynamic_program_synthesis_v1 import GENERIC_MODULES
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_route_policy_comparison import (
    Candidate,
    ComparisonConfig,
    candidate_features,
    complete_attempt_rows,
    fit_contrastive_ranker,
    predeclared_source_folds,
)
from compose_v4.rewrite.trace_shard import encode_state


def test_predeclared_folds_are_source_disjoint_and_balanced():
    metadata = {
        f"source-{protein}-{seed}": {
            "target": protein,
            "source_idx": seed,
        }
        for protein in ("a", "b", "c", "d", "e")
        for seed in range(3)
    }
    folds = predeclared_source_folds(metadata)
    assert len(folds) == 3
    assert all(len(row["test_sources"]) == 5 for row in folds)
    assert all(
        not set(row["train_sources"]) & set(row["test_sources"]) for row in folds
    )
    assert sorted(source for row in folds for source in row["test_sources"]) == sorted(
        metadata
    )


def test_candidate_features_are_address_free_fixed_shape():
    source = production_state_from_smiles("CCO", max_atoms=40)
    endpoint = production_state_from_smiles("CCN", max_atoms=40)
    action = {
        "executor_rule": "atom_restate_semantic",
        "payload": {
            "v": 2,
            "atom_type": 2,
            "formal_charge": 0,
            "implicit_h_count": 2,
        },
    }
    vector = candidate_features(
        source,
        endpoint,
        (action,),
        {"heteroatom_substitute": 1.0},
        module_count=1,
        blocks=1,
    )
    assert vector.shape == (3 * 517 + 18 + len(GENERIC_MODULES) + 7,)
    assert np.isfinite(vector).all()


def test_contrastive_ranker_is_deterministic_and_payload_has_no_teacher_rows():
    config = ComparisonConfig(ranker_updates=10)
    positives = [
        ("source-a", "route-a", np.asarray([1.0, 0.0], dtype=np.float32)),
        ("source-b", "route-b", np.asarray([0.8, 0.1], dtype=np.float32)),
    ]
    negatives = {
        "source-a": [np.asarray([0.0, 1.0], dtype=np.float32)],
        "source-b": [np.asarray([0.1, 0.8], dtype=np.float32)],
    }
    first, _ = fit_contrastive_ranker(positives, negatives, config)
    second, _ = fit_contrastive_ranker(positives, negatives, config)
    assert first == second
    assert first.score(np.asarray([1.0, 0.0])) > first.score(np.asarray([0.0, 1.0]))
    payload = first.checkpoint()
    assert payload["runtime_teacher_rows"] == 0
    assert payload["runtime_executable_rows"] == 0
    assert "route_id" not in payload
    assert "endpoint" not in payload


def test_failed_attempts_remain_visible_but_cannot_carry_endpoint():
    endpoint = encode_state(production_state_from_smiles("CCN", max_atoms=40))
    complete = Candidate(
        "complete",
        "complete",
        endpoint,
        (
            {
                "executor_rule": "atom_restate_semantic",
                "payload": {
                    "v": 2,
                    "atom_type": 2,
                    "formal_charge": 0,
                    "implicit_h_count": 2,
                },
            },
        ),
        ("heteroatom_substitute",),
        1,
        1,
    )
    rejected = Candidate("rejected", "rejected", None, (), (), 0, 0, "no binding")
    attempts = complete_attempt_rows([rejected, complete], {"complete": 2.0})
    assert [row["status"] for row in attempts] == ["complete", "rejected"]
    assert attempts[1]["endpoint_state"] is None
