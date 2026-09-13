from __future__ import annotations

from pathlib import Path

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


def test_only_exact_repaired_input_override_is_accepted() -> None:
    frozen = "a" * 64
    repaired = "b" * 64
    override = {
        "path": rescue.REPAIRED_INPUT_PATH,
        "frozen_sha256": frozen,
        "repaired_sha256": repaired,
        "reason": rescue.REPAIRED_INPUT_REASON,
    }
    rescue.validate_repaired_input_override(
        override, frozen_sha256=frozen, repaired_sha256=repaired
    )

    override["path"] = "src/compose_v4/control/edit_program_policy.py"
    with pytest.raises(ValueError, match="changed scope"):
        rescue.validate_repaired_input_override(
            override, frozen_sha256=frozen, repaired_sha256=repaired
        )


def test_only_exact_prequery_identity_error_is_relaunchable() -> None:
    frozen = "a" * 64
    repaired = "b" * 64
    message = (
        f"input identity mismatch: /root/compose/{rescue.REPAIRED_INPUT_PATH}: "
        f"expected {frozen}, got {repaired}"
    )
    rescue.validate_relaunch_prequery_identity_error(
        message, frozen_sha256=frozen, repaired_sha256=repaired
    )

    with pytest.raises(ValueError, match="unexpected repaired-relaunch"):
        rescue.validate_relaunch_prequery_identity_error(
            message + " extra", frozen_sha256=frozen, repaired_sha256=repaired
        )


def test_repaired_input_verifier_overrides_only_the_sealed_digest(tmp_path) -> None:
    frozen = "a" * 64
    repaired = "b" * 64
    override = {
        "path": rescue.REPAIRED_INPUT_PATH,
        "frozen_sha256": frozen,
        "repaired_sha256": repaired,
        "reason": rescue.REPAIRED_INPUT_REASON,
    }
    observed = []
    repair = tmp_path / rescue.REPAIRED_INPUT_PATH
    ordinary = tmp_path / "src/compose_v4/control/edit_program.py"

    rescue.verify_with_repaired_input(
        repair,
        frozen,
        root=tmp_path,
        override=override,
        verify=lambda path, digest: observed.append((path, digest)),
    )
    rescue.verify_with_repaired_input(
        ordinary,
        "c" * 64,
        root=tmp_path,
        override=override,
        verify=lambda path, digest: observed.append((path, digest)),
    )

    assert observed == [(repair, repaired), (ordinary, "c" * 64)]
    with pytest.raises(ValueError, match="unexpected digest"):
        rescue.verify_with_repaired_input(
            repair,
            "d" * 64,
            root=tmp_path,
            override=override,
            verify=lambda _path, _digest: None,
        )


def test_complete_repaired_input_set_is_exact(tmp_path) -> None:
    frozen = {path: f"{index + 1:064x}" for index, path in enumerate(rescue.REPAIRED_INPUT_REASONS)}
    repaired = {
        path: f"{index + 11:064x}"
        for index, path in enumerate(rescue.REPAIRED_INPUT_REASONS)
    }
    overrides = [
        {
            "path": path,
            "frozen_sha256": frozen[path],
            "repaired_sha256": repaired[path],
            "reason": reason,
        }
        for path, reason in sorted(rescue.REPAIRED_INPUT_REASONS.items())
    ]
    rescue.validate_repaired_input_overrides(
        overrides, frozen_sha256=frozen, repaired_sha256=repaired
    )

    observed = []
    for row in overrides:
        path = tmp_path / row["path"]
        rescue.verify_with_repaired_inputs(
            path,
            row["frozen_sha256"],
            root=tmp_path,
            overrides=overrides,
            verify=lambda checked, digest: observed.append((checked, digest)),
        )
    assert observed == [
        (tmp_path / row["path"], row["repaired_sha256"]) for row in overrides
    ]

    external = Path("/opt/dock/qvina02")
    rescue.verify_with_repaired_inputs(
        external,
        "f" * 64,
        root=tmp_path,
        overrides=overrides,
        verify=lambda checked, digest: observed.append((checked, digest)),
    )
    assert observed[-1] == (external, "f" * 64)

    with pytest.raises(ValueError, match="override set changed"):
        rescue.validate_repaired_input_overrides(
            overrides[:-1], frozen_sha256=frozen, repaired_sha256=repaired
        )


def test_general_prequery_error_rejects_unapproved_path() -> None:
    with pytest.raises(ValueError, match="unexpected T4 rescue input path"):
        rescue.validate_prequery_input_identity_error(
            "unused",
            path="src/compose_v4/control/unapproved.py",
            frozen_sha256="a" * 64,
            repaired_sha256="b" * 64,
        )


def test_runtime_path_failure_must_match_exactly() -> None:
    rescue.validate_runtime_path_compatibility_error(
        rescue.RUNTIME_PATH_COMPATIBILITY_ERROR
    )
    with pytest.raises(ValueError, match="unexpected T4 rescue runtime-preflight"):
        rescue.validate_runtime_path_compatibility_error("another failure")


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
