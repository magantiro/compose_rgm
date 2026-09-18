from copy import deepcopy

import pytest

from compose_v4.experiments.t4_anchored_transfer_lock import (
    build_transfer_lock,
    verify_transfer_lock,
)
from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256


def _candidate(smiles, similarity):
    return {
        "smiles": smiles,
        "similarity": similarity,
        "qed": 0.7,
        "sa": 3.0,
        "heavy": 5,
        "program_families": ["substituent_delete", "construct_substituted_ring"],
        "families": ["base"],
        "created": 5,
        "deleted": 2,
        "regions": 1,
    }


def test_transfer_lock_excludes_history_and_is_deterministic():
    candidates = [
        _candidate("CCNCC", 0.61),
        _candidate("CCOCC", 0.62),
        _candidate("c1ccccc1", 0.63),
        _candidate("C1CCNCC1", 0.64),
    ]
    historical = [
        {
            "cell": "jak2_0",
            "endpoint_sha256": endpoint_sha256("CCNCC"),
            "score_mean": -8.3,
            "record_id": "known",
        }
    ]
    kwargs = {
        "cell": "jak2_0",
        "delta": 0.6,
        "root_smiles": "CC",
        "docking_seed": 20260919,
        "selection_limit": 2,
        "input_sha256": {"a": "b"},
    }
    first = build_transfer_lock(candidates, historical, **kwargs)
    second = build_transfer_lock(reversed(candidates), historical, **kwargs)
    assert first == second
    assert first["exact_historical_rediscoveries"] == 1
    assert first["best_historical_rediscovery"] == -8.3
    assert first["lock"]["charged_calls"] == 2
    assert "CCNCC" not in [row["endpoint"] for row in first["lock"]["selection"]]
    verify_transfer_lock(first["lock"])


def test_transfer_lock_detects_mutation():
    assessment = build_transfer_lock(
        [_candidate("CCNCC", 0.61)],
        [],
        cell="jak2_0",
        delta=0.6,
        root_smiles="CC",
        docking_seed=1,
        selection_limit=1,
        input_sha256={},
    )
    changed = deepcopy(assessment["lock"])
    changed["docking_seed"] = 2
    with pytest.raises(ValueError, match="modified"):
        verify_transfer_lock(changed)
