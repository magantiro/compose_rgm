"""Thin Modal semantic T1 launcher contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modal_apps import build_editing_v2_semantic_t1_successor_cache_app as cache_app
from modal_apps import run_editing_v2_semantic_t1_capacity_app as launcher

ROOT = Path(__file__).resolve().parents[1]


def _source_revision() -> dict[str, object]:
    hashes = cache_app._serialized_source_hashes(ROOT)
    base_body = {
        "schema": cache_app.SOURCE_REVISION_SCHEMA,
        "schema_version": cache_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": cache_app._sha(hashes),
    }
    base = {**base_body, "source_revision_sha256": cache_app._sha(base_body)}
    body = {
        "schema": launcher.SOURCE_REVISION_SCHEMA,
        "schema_version": launcher.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": base["commit"],
        "tree": base["tree"],
        "worktree_clean": True,
        "base_serialized_source_revision": base,
        "launcher_relative_path": launcher.LAUNCHER_SOURCE,
        "launcher_file_sha256": launcher._file_sha(ROOT / launcher.LAUNCHER_SOURCE),
    }
    return {**body, "source_revision_sha256": launcher._sha(body)}


def test_runner_revision_separately_binds_base_sources_and_launcher() -> None:
    revision = _source_revision()
    assert launcher._validate_runner_source_revision(revision, remote_root=ROOT) == revision
    tampered = dict(revision)
    tampered["launcher_file_sha256"] = "0" * 64
    tampered_body = dict(tampered)
    tampered_body.pop("source_revision_sha256")
    tampered["source_revision_sha256"] = launcher._sha(tampered_body)
    with pytest.raises(RuntimeError, match="source revision changed"):
        launcher._validate_runner_source_revision(tampered, remote_root=ROOT)


def test_failure_receipt_is_nonauthorizing_and_binds_recovery_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "step_0010.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"recoverable-state")
    receipt = launcher._failure_receipt(
        provenance={"runner_source_revision_sha256": "a" * 64},
        error=RuntimeError("finite-gradient gate failed"),
        run_root=tmp_path,
        failure_stage="capacity_runtime",
        request_identity_sha256="b" * 64,
        request_bindings={"prepared_input": "/artifacts/prepared.json"},
        recovery_commit_succeeded=True,
    )
    assert receipt["bounded_p50_authorized"] is False
    assert receipt["scientific_result_published"] is False
    assert receipt["failure_category"] == "RUNTIME_FAILURE"
    assert receipt["recovery_checkpoints"] == [
        {
            "relative_path": "checkpoints/step_0010.pt",
            "file_sha256": hashlib.sha256(b"recoverable-state").hexdigest(),
            "file_bytes": len(b"recoverable-state"),
        }
    ]


@pytest.mark.parametrize(
    ("stage", "message", "expected_category"),
    [
        (
            "prepared_input_authentication",
            "physical input mismatch",
            "PRERUN_INPUT_OR_PROVENANCE_FAILURE",
        ),
        ("capacity_runtime", "CUDA out of memory", "CUDA_OUT_OF_MEMORY"),
        (
            "completion_publication",
            "immutable completion differs",
            "ARTIFACT_PUBLICATION_FAILURE",
        ),
    ],
)
def test_launcher_failure_paths_commit_recovery_then_publish_receipt(
    tmp_path: Path,
    stage: str,
    message: str,
    expected_category: str,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "step_0050.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"recoverable")
    commits: list[int] = []
    failure, path = launcher._publish_failure_receipt(
        loaded=None,
        run_root=tmp_path,
        provenance=None,
        error=RuntimeError(message),
        failure_stage=stage,
        request_identity_sha256="c" * 64,
        request_bindings={"prepared_completion": "/artifacts/prepared-complete.json"},
        commit=lambda: commits.append(len(commits)),
    )
    assert commits == [0, 1]
    assert path.is_file()
    assert failure["failure_category"] == expected_category
    assert failure["pre_receipt_recovery_commit_succeeded"] is True
    assert failure["bounded_p50_authorized"] is False
    assert (
        failure["recovery_checkpoints"][0]["file_sha256"]
        == hashlib.sha256(b"recoverable").hexdigest()
    )


def test_prepared_completion_authenticates_physical_full_partition_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", tmp_path)
    artifact_path = tmp_path / "prepared" / "SEMANTIC_T1_PREPARED_INPUTS.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b'{"exact":"full-partition"}\n')
    source_revision = {
        "worktree_clean": True,
        "source_revision_sha256": "a" * 64,
    }
    artifact = {
        "artifact_sha256": "b" * 64,
        "implementation_sha256": "c" * 64,
        "panel_artifact_sha256": "d" * 64,
        "cache_completion_sha256": "e" * 64,
        "cache_manifest_sha256": "f" * 64,
        "entry_count": 17,
        "panel_entry_inventory_sha256": "1" * 64,
        "entry_inventory_sha256": "2" * 64,
    }
    completion = launcher._prepared_completion(
        artifact_path=artifact_path,
        artifact=artifact,
        source_revision=source_revision,
        panel_completion_sha256="3" * 64,
        unique_exact_source_state_count=11,
    )
    completion_path = artifact_path.parent / launcher.PREPARED_COMPLETION_FILENAME
    completion_path.write_bytes(launcher._canonical_bytes(completion, newline=True))
    loaded, opened_path = launcher._load_prepared_completion(completion_path)
    assert opened_path == artifact_path
    assert (
        loaded["prepared_input_file_sha256"]
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    assert loaded["compiler_implementation_sha256"] == "c" * 64
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="physical bytes disagree"):
        launcher._load_prepared_completion(completion_path)
