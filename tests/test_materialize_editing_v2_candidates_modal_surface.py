"""Static and pure-contract tests for the Editing V2 candidate Modal job."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_apps.materialize_editing_v2_candidates_app import (
    RUN_SCHEMA,
    build_candidate_run_request,
)

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "modal_apps" / "materialize_editing_v2_candidates_app.py"
SHA = "a" * 64


def _request(**updates: str) -> dict[str, object]:
    arguments = {
        "commit": "1" * 40,
        "overlay_completion_path": "/artifacts/editing_v2/upstream/OVERLAY_COMPLETION.json",
        "overlay_completion_file_sha256": SHA,
        "source_binding_registry_file_sha256": SHA,
        "source_manifest_file_sha256": SHA,
        "source_manifest_sha256": SHA,
        "routing_policy_file_sha256": SHA,
        "routing_policy_sha256": SHA,
        "corpus_contract_file_sha256": SHA,
        "corpus_contract_sha256": SHA,
        "materializer_source_sha256": SHA,
        "launcher_source_sha256": SHA,
    }
    arguments.update(updates)
    return build_candidate_run_request(**arguments)


def test_candidate_request_is_deterministic_and_nonauthorizing() -> None:
    first = _request()
    second = _request()

    assert first == second
    assert first["schema"] == RUN_SCHEMA
    assert first["training_authorized"] is False
    assert len(first["run_identity_sha256"]) == 64


@pytest.mark.parametrize(
    "updates",
    [
        {"commit": "short"},
        {"overlay_completion_file_sha256": "bad"},
        {"overlay_completion_path": "/tmp/OVERLAY_COMPLETION.json"},
    ],
)
def test_candidate_request_rejects_unbound_or_escaping_identity(
    updates: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        _request(**updates)


def test_modal_job_builds_manifest_from_completion_without_arbitrary_manifest_input() -> None:
    source = APP_SOURCE.read_text()

    assert "build_packed_candidate_source_manifest_from_overlay_completion" in source
    assert "expected_overlay_completion_file_sha256" in source
    assert "--source-manifest" not in source
    assert 'training_authorized": False' in source
    assert "candidate materialization failed:" in source
