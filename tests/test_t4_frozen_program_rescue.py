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


def test_status_reader_allows_plain_progress_only() -> None:
    progress = b'{"queries":8,"unit_id":"braf_2_r0"}'
    assert (
        rescue.read_status_bytes(progress, source="unit/progress.json")["queries"] == 8
    )
    with pytest.raises(ValueError, match="unsealed non-progress"):
        rescue.read_status_bytes(progress, source="unit/result.json")


def test_query_paths_require_one_final_ambiguity() -> None:
    assert rescue.validate_query_paths(set(range(8)), set(range(7))) == 7
    with pytest.raises(ValueError, match="exactly one final"):
        rescue.validate_query_paths(set(range(8)), set(range(6)))
    with pytest.raises(ValueError, match="not contiguous"):
        rescue.validate_query_paths({0, 1, 3}, {0, 1})


def test_relaunch_query_paths_require_complete_contiguous_pairs() -> None:
    assert rescue.validate_relaunch_query_paths(set(range(8)), set(range(8))) == 8
    with pytest.raises(ValueError, match="ambiguous"):
        rescue.validate_relaunch_query_paths(set(range(8)), set(range(7)))
    with pytest.raises(ValueError, match="not contiguous"):
        rescue.validate_relaunch_query_paths({0, 1, 3}, {0, 1, 3})


def test_only_exact_tombstone_compatibility_failure_is_relaunchable() -> None:
    failure = {
        "schema_version": "t4_frozen_program_unit_failure_v1",
        "status": "failed",
        "unit": {"unit_id": "braf_0_r1"},
        "error": rescue.TOMBSTONE_COMPATIBILITY_ERROR,
        "automatic_retry": False,
    }
    rescue.validate_tombstone_compatibility_failure(failure, unit_id="braf_0_r1")
    failure["error"] = "ValueError('another failure')"
    with pytest.raises(ValueError, match="unexpected first-relaunch"):
        rescue.validate_tombstone_compatibility_failure(failure, unit_id="braf_0_r1")


def test_locked_round_binds_the_exact_candidate() -> None:
    reservation = started()
    candidates = [{"candidate_id": str(index), "endpoint": "C"} for index in range(4)]
    candidates[3] = {"candidate_id": "candidate", "endpoint": "CCO"}
    rescue.validate_locked_round(reservation, {"candidates": candidates})

    candidates[3]["endpoint"] = "CCC"
    with pytest.raises(ValueError, match="locked candidate"):
        rescue.validate_locked_round(reservation, {"candidates": candidates})


def test_missing_checkpoint_is_allowed_only_for_bootstrap_round_zero() -> None:
    assert (
        rescue.validate_resume_checkpoint(
            None, completed_results=3, started_round=0, bootstrap=True
        )
        == 0
    )
    with pytest.raises(ValueError, match="non-bootstrap"):
        rescue.validate_resume_checkpoint(
            None, completed_results=3, started_round=1, bootstrap=True
        )
    with pytest.raises(ValueError, match="non-bootstrap"):
        rescue.validate_resume_checkpoint(
            None, completed_results=3, started_round=0, bootstrap=False
        )


def test_checkpoint_cannot_lead_the_durable_results() -> None:
    checkpoint = {"query_count": 4, "curve": [{}, {}, {}, {}], "next_round": 0}
    assert (
        rescue.validate_resume_checkpoint(
            checkpoint, completed_results=4, started_round=0, bootstrap=True
        )
        == 4
    )
    with pytest.raises(ValueError, match="cannot resume safely"):
        rescue.validate_resume_checkpoint(
            checkpoint, completed_results=3, started_round=0, bootstrap=True
        )
