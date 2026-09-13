from __future__ import annotations

import pytest

from compose_v4.experiments import t4_frozen_program_rescue as rescue


def started(index: int = 7) -> dict:
    return {
        "query_index": index,
        "round_index": 2,
        "candidate_index": 3,
        "candidate_id": "candidate",
        "endpoint": "CCO",
        "target": "jak2",
        "docking_seed": 1703,
        "started_at_utc": "2026-09-13T00:00:00+00:00",
    }


def test_tombstone_preserves_query_and_records_no_score() -> None:
    reservation = started()
    tombstone = rescue.make_ambiguous_tombstone(
        reservation, reconciled_at_utc="2026-09-13T12:00:00+00:00"
    )

    rescue.validate_tombstone(tombstone, reservation)
    assert tombstone["score"] is None
    assert tombstone["oracle_call_charged"] is True
    assert tombstone["observation_recovered"] is False
    assert tombstone["failure"] == rescue.TOMBSTONE_FAILURE


def test_tombstone_rejects_changed_candidate_identity() -> None:
    reservation = started()
    tombstone = rescue.make_ambiguous_tombstone(
        reservation, reconciled_at_utc="2026-09-13T12:00:00+00:00"
    )
    tombstone["endpoint"] = "CCC"

    with pytest.raises(ValueError, match="reservation field"):
        rescue.validate_tombstone(tombstone, reservation)


def test_query_paths_require_one_final_ambiguity() -> None:
    assert rescue.validate_query_paths(set(range(8)), set(range(7))) == 7
    with pytest.raises(ValueError, match="exactly one final"):
        rescue.validate_query_paths(set(range(8)), set(range(6)))
    with pytest.raises(ValueError, match="not contiguous"):
        rescue.validate_query_paths({0, 1, 3}, {0, 1})


def test_locked_round_binds_the_exact_candidate() -> None:
    reservation = started()
    candidates = [{"candidate_id": str(index), "endpoint": "C"} for index in range(4)]
    candidates[3] = {"candidate_id": "candidate", "endpoint": "CCO"}
    rescue.validate_locked_round(reservation, {"candidates": candidates})

    candidates[3]["endpoint"] = "CCC"
    with pytest.raises(ValueError, match="locked candidate"):
        rescue.validate_locked_round(reservation, {"candidates": candidates})
