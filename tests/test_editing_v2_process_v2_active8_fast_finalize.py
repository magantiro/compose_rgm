"""Fast Active8 finalization authenticates bytes without replaying chemistry."""

from __future__ import annotations

from pathlib import Path

import pytest

import test_editing_v2_process_v2_active8_parallel_sentinel as sentinel_fixture
import test_editing_v2_process_v2_active8_pipeline as pipeline_fixture
from compose_v4.data.editing_v2_process_v2_active8_fast_finalize import (
    FAST_FINALIZATION_RECEIPT_FILENAME,
    ProcessV2Active8FastFinalizeError,
    authenticate_fast_finalization_publication,
    fast_finalize_process_v2_active8,
    validate_fast_finalization_receipt,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    finalize_process_v2_active8_reduction,
    task_output_path,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import IMPLEMENTATION_FILES
from compose_v4.data.editing_v2_process_v2_active8_map import ROWS_FILENAME
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_fast_publisher_is_downstream_of_the_frozen_active8_answer_identity() -> None:
    assert (
        "src/compose_v4/data/editing_v2_process_v2_active8_fast_finalize.py"
        not in IMPLEMENTATION_FILES
    )


def _publisher_revision() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "image_revision_sha256": "3" * 64,
        "serialized_sources": {
            "src/compose_v4/data/editing_v2_process_v2_active8_fast_finalize.py": (
                "4" * 64
            )
        },
    }


def _run_root(stage) -> Path:
    return stage.artifact_root / str(stage.plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    )


def test_fast_finalizer_is_byte_identical_without_replaying_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "equivalence",
        prefix="/artifacts/active8_fast_finalize_equivalence",
    )
    stage.run()
    prepared, results = sentinel_fixture._run_parallel(
        stage,
        max_pairs=1,
        publish=True,
    )
    expected = finalize_process_v2_active8_reduction(
        stage.plan,
        prepared,
        results,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publish=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fast finalization replayed decoded Active8 evidence")

    monkeypatch.setattr(
        "compose_v4.data.editing_v2_process_v2_active8_reduce._derive_reduction_inputs",
        forbidden,
    )
    monkeypatch.setattr(
        "compose_v4.data.editing_v2_process_v2_active8_map.read_task_rows",
        forbidden,
    )
    monkeypatch.setattr(
        "compose_v4.data.editing_v2_process_v2_active8_map.read_task_transitions",
        forbidden,
    )
    observed, receipt = fast_finalize_process_v2_active8(
        stage.plan,
        prepared,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publisher_revision=_publisher_revision(),
        hash_workers=4,
        publish=True,
    )
    assert observed == expected
    assert validate_fast_finalization_receipt(receipt) == receipt
    assert receipt["completion_sha256"] == observed["completion_sha256"]
    assert receipt["verified_task_count"] == len(stage.plan["tasks"])
    assert receipt["verified_sentinel_partition_count"] == len(results)
    run_root = _run_root(stage)
    assert (run_root / COMPLETION_FILENAME).read_bytes() == canonical_bytes(observed) + b"\n"
    assert (run_root / FAST_FINALIZATION_RECEIPT_FILENAME).read_bytes() == (
        canonical_bytes(receipt) + b"\n"
    )
    assert authenticate_fast_finalization_publication(run_root) == (observed, receipt)

    retried, retried_receipt = fast_finalize_process_v2_active8(
        stage.plan,
        prepared,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publisher_revision=_publisher_revision(),
        hash_workers=4,
        publish=True,
    )
    assert retried == observed
    assert retried_receipt == receipt


def test_fast_finalizer_refuses_changed_task_payload_before_publication(
    tmp_path: Path,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "tampered",
        prefix="/artifacts/active8_fast_finalize_tampered",
    )
    stage.run()
    prepared, _results = sentinel_fixture._run_parallel(
        stage,
        max_pairs=1,
        publish=True,
    )
    first_task = stage.plan["tasks"][0]
    rows_path = (
        task_output_path(stage.plan, first_task, artifact_root=stage.artifact_root)
        / ROWS_FILENAME
    )
    rows_path.write_bytes(rows_path.read_bytes() + b" ")

    with pytest.raises(
        ProcessV2Active8FastFinalizeError,
        match="rows payload changed",
    ):
        fast_finalize_process_v2_active8(
            stage.plan,
            prepared,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
            publisher_revision=_publisher_revision(),
            hash_workers=2,
            publish=True,
        )
    run_root = _run_root(stage)
    assert not (run_root / COMPLETION_FILENAME).exists()
    assert not (run_root / FAST_FINALIZATION_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("hash_workers", [0, 65, True])
def test_fast_finalizer_refuses_invalid_hash_concurrency(
    tmp_path: Path,
    hash_workers: int,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / f"workers-{hash_workers}",
        prefix=f"/artifacts/active8_fast_finalize_workers_{hash_workers}",
    )
    stage.run()
    prepared, _results = sentinel_fixture._run_parallel(
        stage,
        max_pairs=1,
        publish=True,
    )
    with pytest.raises(ProcessV2Active8FastFinalizeError, match="hash_workers"):
        fast_finalize_process_v2_active8(
            stage.plan,
            prepared,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
            publisher_revision=_publisher_revision(),
            hash_workers=hash_workers,
            publish=False,
        )
