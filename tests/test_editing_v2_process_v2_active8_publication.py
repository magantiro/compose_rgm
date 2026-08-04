"""Immutable and race-safe publication for one Process-V2 Active8 task."""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import compose_v4.data.editing_v2_process_v2_active8_map as map_module
from compose_v4.data.editing_v2_process_v2_active8_map import (
    RECEIPT_FILENAME,
    ROWS_FILENAME,
    SUMMARY_FILENAME,
    TRANSITIONS_FILENAME,
    ProcessV2Active8MapError,
    read_task_rows,
    read_task_transitions,
    validate_process_v2_active8_task_result,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_RECEIPT_FIELDS,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import task_output_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

from test_editing_v2_process_v2_active8_pipeline import _Stage  # noqa: E402

_PUBLISHED_NAMES = (
    ROWS_FILENAME,
    TRANSITIONS_FILENAME,
    RECEIPT_FILENAME,
    SUMMARY_FILENAME,
)


@pytest.fixture(scope="module")
def publication_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("active8-publication")
    stage = _Stage(root, prefix="/artifacts/active8_publication_source")
    stage.run()
    task = stage.plan["tasks"][0]
    source = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
    receipt = json.loads((source / RECEIPT_FILENAME).read_bytes())
    summary = json.loads((source / SUMMARY_FILENAME).read_bytes())
    return {
        "root": root,
        "rows": read_task_rows(source),
        "transitions": read_task_transitions(source),
        "receipt_body": {
            key: receipt[key]
            for key in ACTIVE8_RECEIPT_FIELDS
            if key not in {"receipt_sha256", "decision_shard_sha256"}
        },
        "summary_body": {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "summary_sha256",
                "receipt_sha256",
                "rows_file_sha256",
                "rows_stream_sha256",
                "transitions_stream_sha256",
            }
        },
    }


def _publish(output: Path, inputs: dict[str, Any], *, summary_body=None):
    return map_module._publish(
        output,
        rows=inputs["rows"],
        transitions=inputs["transitions"],
        receipt_body=inputs["receipt_body"],
        summary_body=summary_body or inputs["summary_body"],
    )


def _snapshot(output: Path) -> dict[str, bytes]:
    return {name: (output / name).read_bytes() for name in _PUBLISHED_NAMES}


def _attempts(output: Path) -> list[Path]:
    return sorted(output.parent.glob(f".{output.name}.attempt-*"))


def test_an_identical_retry_never_replaces_the_committed_directory(
    publication_inputs: dict[str, Any],
) -> None:
    output = publication_inputs["root"] / "identical-retry"
    first = _publish(output, publication_inputs)
    before = _snapshot(output)
    inode = output.stat().st_ino

    second = _publish(output, publication_inputs)

    assert second == first
    assert output.stat().st_ino == inode
    assert _snapshot(output) == before
    assert _attempts(output) == []
    validate_process_v2_active8_task_result(output)


def test_a_preexisting_legacy_staging_directory_is_never_deleted(
    publication_inputs: dict[str, Any],
) -> None:
    output = publication_inputs["root"] / "legacy-staging"
    legacy_staging = output.parent / f".{output.name}.staging"
    legacy_staging.mkdir()
    unknown = legacy_staging / "UNKNOWN.partial"
    unknown.write_bytes(b"preserve")

    _publish(output, publication_inputs)

    assert unknown.read_bytes() == b"preserve"
    validate_process_v2_active8_task_result(output)


def test_two_identical_concurrent_publishers_share_one_committed_winner(
    publication_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = publication_inputs["root"] / "same-race"
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    first_checks: set[int] = set()
    original_lexists = os.path.lexists

    def synchronized_first_check(path) -> bool:
        exists = original_lexists(path)
        if Path(path) == output:
            thread_id = threading.get_ident()
            with lock:
                first = thread_id not in first_checks
                first_checks.add(thread_id)
            if first:
                barrier.wait(timeout=10)
        return exists

    monkeypatch.setattr(map_module.os.path, "lexists", synchronized_first_check)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_publish, output, publication_inputs) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]

    assert results[0] == results[1]
    assert len(first_checks) == 2
    assert _attempts(output) == []
    validate_process_v2_active8_task_result(output)


def test_a_valid_but_divergent_retry_is_an_immutable_collision(
    publication_inputs: dict[str, Any],
) -> None:
    output = publication_inputs["root"] / "divergent-retry"
    _publish(output, publication_inputs)
    before = _snapshot(output)
    divergent = {
        **publication_inputs["summary_body"],
        "run_identity_sha256": "f" * 64,
    }

    with pytest.raises(ProcessV2Active8MapError, match="immutable.*collision"):
        _publish(output, publication_inputs, summary_body=divergent)

    assert _snapshot(output) == before
    assert _attempts(output) == []
    validate_process_v2_active8_task_result(output)


def test_an_invalid_existing_target_is_preserved_and_refused(
    publication_inputs: dict[str, Any],
) -> None:
    output = publication_inputs["root"] / "partial-target"
    output.mkdir()
    unknown = output / "UNKNOWN.partial"
    unknown.write_bytes(b"do not delete me")

    with pytest.raises(ProcessV2Active8MapError, match="refusing to overwrite"):
        _publish(output, publication_inputs)

    assert unknown.read_bytes() == b"do not delete me"
    assert {path.name for path in output.iterdir()} == {unknown.name}
    assert _attempts(output) == []


def test_a_symlinked_published_member_is_not_a_committed_task(
    publication_inputs: dict[str, Any],
) -> None:
    output = publication_inputs["root"] / "symlink-member"
    _publish(output, publication_inputs)
    rows = output / ROWS_FILENAME
    external = publication_inputs["root"] / "external-rows.gz"
    external.write_bytes(rows.read_bytes())
    rows.unlink()
    rows.symlink_to(external)

    with pytest.raises(ProcessV2Active8MapError, match="inventory"):
        validate_process_v2_active8_task_result(output)
