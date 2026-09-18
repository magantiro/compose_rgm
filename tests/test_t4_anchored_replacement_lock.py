from __future__ import annotations

import pytest

from compose_v4.experiments.t4_anchored_replacement_lock import (
    build_exact_assessment_and_lock,
    unique_lane_candidates,
    verify_lock,
)
from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256

ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
BASIN = "O=C(CC1Nc2ccccc2-c2ccnc3[nH]cc1c23)N1CCNCC1"
OUTSIDE = "CCO"


def candidate(smiles: str) -> dict:
    return {
        "smiles": smiles,
        "similarity": 0.63,
        "qed": 0.65,
        "sa": 3.5,
        "heavy": 27,
        "program_families": ["substituent_delete", "construct_substituted_ring"],
        "created": 6,
        "deleted": 2,
        "regions": 1,
    }


def test_lane_candidates_deduplicate_identical_records():
    row = candidate(BASIN)
    attempts = [
        {"lane": "anchored_replacement", "error": None, "candidates": [row]},
        {"lane": "anchored_replacement", "error": None, "candidates": [row]},
        {"lane": "shallow", "error": None, "candidates": [candidate(OUTSIDE)]},
    ]
    assert unique_lane_candidates(attempts, lane="anchored_replacement") == [row]


def test_lane_candidates_allow_two_programs_for_the_same_endpoint():
    first = candidate(BASIN)
    second = {**first, "families": ["scale"]}
    attempts = [
        {"lane": "anchored_replacement", "error": None, "candidates": [first]},
        {"lane": "anchored_replacement", "error": None, "candidates": [second]},
    ]
    rows = unique_lane_candidates(attempts, lane="anchored_replacement")
    assert len(rows) == 1 and rows[0]["smiles"] == BASIN


def test_exact_historical_hits_are_free_and_excluded_from_lock():
    records = [
        {
            "cell": "jak2_1",
            "endpoint_sha256": endpoint_sha256(BASIN),
            "score_mean": -10.2,
            "record_id": "r",
            "receipt_ids": ["q"],
            "protocol": "p",
            "arms": ["full146"],
        }
    ]
    result = build_exact_assessment_and_lock(
        [candidate(BASIN), candidate(OUTSIDE)],
        records,
        cell="jak2_1",
        delta=0.6,
        root_smiles=ROOT,
        docking_seed=7,
        input_sha256={"input": "abc"},
    )
    assert result["eligible_anchored_basin_endpoints"] == 1
    assert result["exact_historical_rediscoveries"] == 1
    assert result["historical_hits"][0]["historical_score"] == -10.2
    assert result["prospective_novel_endpoints"] == 0
    assert result["lock"]["charged_calls"] == 0
    verify_lock(result["lock"])


def test_novel_basin_candidate_is_locked_without_a_score():
    result = build_exact_assessment_and_lock(
        [candidate(BASIN)],
        [],
        cell="jak2_1",
        delta=0.6,
        root_smiles=ROOT,
        docking_seed=7,
        input_sha256={},
    )
    assert result["exact_historical_rediscoveries"] == 0
    assert result["lock"]["charged_calls"] == 1
    assert "score" not in result["lock"]["selection"][0]
    verify_lock(result["lock"])


def test_modified_lock_fails_closed():
    result = build_exact_assessment_and_lock(
        [candidate(BASIN)],
        [],
        cell="jak2_1",
        delta=0.6,
        root_smiles=ROOT,
        docking_seed=7,
        input_sha256={},
    )
    result["lock"]["selection"][0]["endpoint"] = "tampered"
    with pytest.raises(ValueError, match="modified"):
        verify_lock(result["lock"])
