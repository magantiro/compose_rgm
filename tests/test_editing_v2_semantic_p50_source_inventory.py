"""Immutable publication and full-lineage loading for semantic P50 sources."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SEMANTIC_P50_SOURCE_INVENTORY_FILENAME,
    SemanticP50SourceInventoryError,
    load_semantic_p50_source_inventory,
    publish_semantic_p50_source_inventory,
)
from tests import test_editing_v2_semantic_active8_decision_source as source_fixtures


@pytest.fixture(scope="module")
def completed_semantic_source(tmp_path_factory: pytest.TempPathFactory):
    return source_fixtures._build_completed_source(
        tmp_path_factory.mktemp("semantic-p50-source") / "artifacts",
        full_train_active8=True,
        inject_exclusion=False,
    )


def _patch_and_resolve(completed, monkeypatch):
    source_fixtures._patch_migration_resolver(
        monkeypatch,
        completed["inventory"],
    )
    return source_fixtures._resolve(completed)


def _load(published, completed):
    return load_semantic_p50_source_inventory(
        published.path,
        expected_binding=published.binding,
        migration_completion_path=completed["inventory"].migration_completion_path,
        chunk_cache_plan_path=completed["cache_plan_path"],
        chunk_cache_global_completion_path=completed["cache_completion_path"],
        decision_plan_path=completed["decision_plan_path"],
        decision_completion_path=completed["decision_completion_path"],
        artifact_root=completed["root"],
        repo_root=Path.cwd(),
    )


def test_publish_is_canonical_idempotent_and_reopens_complete_lineage(
    completed_semantic_source,
    monkeypatch,
) -> None:
    completed = completed_semantic_source
    index = _patch_and_resolve(completed, monkeypatch)
    published = publish_semantic_p50_source_inventory(
        index,
        decision_plan_path=completed["decision_plan_path"],
        output_directory=completed["root"] / "p50-source",
    )
    assert published.path.name == SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    assert published.path.read_bytes() == (
        json.dumps(
            index.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert published.binding.source_inventory_sha256 == index.inventory_sha256
    assert published.binding.process_identity_sha256 == index.process_identity_sha256
    assert published.binding.model_runtime_identity_sha256 == index.model_runtime_identity_sha256
    assert published.binding.active8_policy_sha256 == index.policy_sha256
    assert len(published.binding.operator_capability_fingerprint) == 16

    repeated = publish_semantic_p50_source_inventory(
        index,
        decision_plan_path=completed["decision_plan_path"],
        output_directory=published.path.parent,
    )
    assert repeated == published

    verified = _load(published, completed)
    assert verified.binding == published.binding
    assert verified.index.identity_payload() == index.identity_payload()
    assert verified.index.training_authorized is False
    assert verified.index.bounded_p50_authorized is False


def test_publication_refuses_an_existing_different_artifact(
    completed_semantic_source,
    monkeypatch,
    tmp_path: Path,
) -> None:
    index = _patch_and_resolve(completed_semantic_source, monkeypatch)
    target = tmp_path / SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    target.write_text("{}\n")
    with pytest.raises(
        SemanticP50SourceInventoryError,
        match="immutable semantic P50 source-inventory collision",
    ):
        publish_semantic_p50_source_inventory(
            index,
            decision_plan_path=completed_semantic_source["decision_plan_path"],
            output_directory=tmp_path,
        )


def test_loader_rejects_wrong_operator_identity_after_full_reresolution(
    completed_semantic_source,
    monkeypatch,
) -> None:
    completed = completed_semantic_source
    index = _patch_and_resolve(completed, monkeypatch)
    published = publish_semantic_p50_source_inventory(
        index,
        decision_plan_path=completed["decision_plan_path"],
        output_directory=completed["root"] / "p50-source-operator-mismatch",
    )
    mismatched = replace(
        published.binding,
        operator_capability_fingerprint="0" * 16,
    )
    with pytest.raises(
        SemanticP50SourceInventoryError,
        match="physical, process, model, or operator binding disagrees",
    ):
        load_semantic_p50_source_inventory(
            published.path,
            expected_binding=mismatched,
            migration_completion_path=completed["inventory"].migration_completion_path,
            chunk_cache_plan_path=completed["cache_plan_path"],
            chunk_cache_global_completion_path=completed["cache_completion_path"],
            decision_plan_path=completed["decision_plan_path"],
            decision_completion_path=completed["decision_completion_path"],
            artifact_root=completed["root"],
            repo_root=Path.cwd(),
        )


def test_loader_rejects_a_substituted_lineage_artifact(
    completed_semantic_source,
    monkeypatch,
) -> None:
    completed = completed_semantic_source
    index = _patch_and_resolve(completed, monkeypatch)
    published = publish_semantic_p50_source_inventory(
        index,
        decision_plan_path=completed["decision_plan_path"],
        output_directory=completed["root"] / "p50-source-lineage-mismatch",
    )
    with pytest.raises(
        SemanticP50SourceInventoryError,
        match="failed complete physical revalidation",
    ):
        load_semantic_p50_source_inventory(
            published.path,
            expected_binding=published.binding,
            migration_completion_path=completed["inventory"].migration_completion_path,
            chunk_cache_plan_path=completed["cache_plan_path"],
            chunk_cache_global_completion_path=completed["cache_completion_path"],
            decision_plan_path=completed["decision_plan_path"],
            decision_completion_path=completed["decision_plan_path"],
            artifact_root=completed["root"],
            repo_root=Path.cwd(),
        )
