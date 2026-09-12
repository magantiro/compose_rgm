"""Protect target identity and failure accounting when admitting measured programs."""

from copy import deepcopy

import pytest

from tools.review_t4_program_curriculum import validate_outcomes


def fixture():
    candidate = {
        "candidate_id": "locked-candidate",
        "cell": "jak2_1",
        "target": "jak2",
        "role": "candidate",
        "smiles": "CC",
        "oracle_protocol": "qualified-jak2",
    }
    row = {**candidate, "index": 0, "docking_seed": 1701, "ds": -8.0, "failure": None}
    return {"take": [candidate], "docking_seed": 1701}, {"rows": [row], "new_oracle_calls": 1}


def test_qualified_scores_and_explicit_failures_are_both_preserved():
    lock, result = fixture()
    validate_outcomes(lock, result)
    failed = deepcopy(result)
    failed["rows"][0].update(ds=None, failure="oracle_failed")
    validate_outcomes(lock, failed)
    assert failed["new_oracle_calls"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "parp1"),
        ("oracle_protocol", "other"),
        ("candidate_id", "other"),
        ("docking_seed", 7),
        ("ds", float("nan")),
        ("failure", "oracle_failed"),
    ],
)
def test_different_domains_or_inconsistent_outcomes_cannot_enter_replay(field, value):
    lock, result = fixture()
    result["rows"][0][field] = value
    with pytest.raises(ValueError):
        validate_outcomes(lock, result)


def test_missing_outcome_cannot_disappear_from_call_count():
    lock, result = fixture()
    result["rows"] = []
    with pytest.raises(ValueError, match="missing or extra"):
        validate_outcomes(lock, result)
