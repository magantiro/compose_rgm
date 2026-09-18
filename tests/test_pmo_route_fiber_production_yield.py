import gzip
import json
from pathlib import Path

import numpy as np

from compose_v4.experiments.pmo_route_fiber_production_yield import (
    CORPUS,
    RUNTIME_FORBIDDEN_KEYS,
    TASK_FOLDS,
    _recursive_keys,
    contract_envelope,
    fit_route_transition_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _corpus():
    with gzip.open(ROOT / CORPUS, "rt") as handle:
        return json.load(handle)["payload"]


def test_route_transition_checkpoint_is_split_clean_and_sanitized():
    result = fit_route_transition_checkpoint(_corpus())
    payload = result["payload"]
    assert payload["new_oracle_calls"] == 0
    assert len(payload["folds"]) == 3
    assert not (RUNTIME_FORBIDDEN_KEYS & _recursive_keys(payload))
    for fold in payload["folds"]:
        assert fold["fold_index"] in (0, 1, 2)
        assert fold["checkpoint_id"]
        for transition in fold["transitions"].values():
            values = np.asarray(transition["probabilities"])
            assert np.all(values > 0)
            assert np.isclose(values.sum(), 1.0)
    for fold in result["training_audit"].values():
        assert fold["lineage_overlap"] == 0


def test_pilot_tasks_map_to_wholly_held_family_folds():
    corpus = _corpus()
    for task, fold in TASK_FOLDS.items():
        rows = [
            route
            for route in corpus["routes"]
            if task in {member["task"] for member in route["members"]}
        ]
        assert rows
        assert {row["test_fold"] for row in rows} == {fold}


def test_contract_matches_two_attempt_additive_comparison_without_scoring():
    contract = contract_envelope()["payload"]
    assert contract["arms"] == {
        "old_v0": ["v0", "v0"],
        "additive_route": ["same_first_v0", "route_transition_prior"],
    }
    assert contract["attempts_per_source_arm"] == 2
    assert contract["sources"] == 16
    assert contract["oracle_calls_authorized"] == 0
    assert contract["scored_launch_authorized"] is False
