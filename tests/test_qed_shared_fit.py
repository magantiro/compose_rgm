"""Source and artifact gates for the shared-reference QED value fit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compose_v4.experiments.hphi_region_features import input_dim
from compose_v4.experiments.qed_shared_fit import (
    QEDFitConfig,
    fit_value_head,
    inspect_feature_corpus,
    training_normalization,
)
from tools import fit_qed_shared_value as fit_cli
from tools.fit_qed_shared_value import input_shard_manifest

REFERENCE = {"checkpoint_sha256": "fragment-checkpoint", "time": 0.5}
SPLIT = "source-split"


def _write_shard(
    folder: Path,
    role: str,
    index: int,
    input_index: int,
    features: np.ndarray,
    labels: np.ndarray,
    region_indices: np.ndarray | None = None,
    next_row_indices: np.ndarray | None = None,
    next_terminal_targets: np.ndarray | None = None,
    *,
    reference: dict = REFERENCE,
) -> Path:
    path = folder / f"{role}_{index:04d}.npz"
    metadata = {
        "schema_version": "compose.qed.shared_features.v4",
        "role": role,
        "source_index": index,
        "source_input_row_index": input_index,
        "source_split_sha256": SPLIT,
        "reference": reference,
        "budget_max": 1,
        "feature_schema": "region_features_v1",
        "target_semantics": "terminal_region",
        "builder_sha256": "builder",
        "goal_regions": [[0.9, 0.4]],
        "examples": len(labels),
        "positive_labels": int(labels.sum()),
        "region_examples": [len(labels)],
        "region_positive_labels": [int(labels.sum())],
    }
    if region_indices is None:
        region_indices = np.zeros(len(labels), dtype=np.int16)
    if next_row_indices is None:
        next_row_indices = np.full(len(labels), -1, dtype=np.int32)
    if next_terminal_targets is None:
        next_terminal_targets = labels.astype(np.int8)
        next_terminal_targets[next_row_indices >= 0] = -1
    metadata["bellman_pairs"] = int(np.sum((next_row_indices >= 0) | (next_terminal_targets >= 0)))
    metadata["bellman_boundary_pairs"] = int(np.sum(next_terminal_targets >= 0))
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        region_indices=region_indices,
        next_row_indices=next_row_indices,
        next_terminal_targets=next_terminal_targets,
        metadata=np.str_(json.dumps(metadata)),
    )
    return path


def _corpus(tmp_path: Path) -> tuple:
    width = input_dim(1)
    train = np.zeros((4, width), dtype=np.float32)
    train[:, 0] = np.asarray([0, 1, 0, 1])
    validation = np.zeros((2, width), dtype=np.float32)
    validation[:, 0] = 100
    _write_shard(tmp_path, "train", 0, 17, train, np.asarray([0, 1, 0, 1], np.float32))
    _write_shard(tmp_path, "validation", 0, 0, validation, np.asarray([0, 1], np.float32))
    return inspect_feature_corpus(
        tmp_path,
        reference_identity=REFERENCE,
        source_split_sha256=SPLIT,
        train_input_indices=(17,),
        validation_count=1,
        budget_max=1,
        builder_sha256="builder",
        goal_regions=((0.9, 0.4),),
    )


def test_inventory_and_normalization_are_source_disjoint(tmp_path: Path) -> None:
    shards = _corpus(tmp_path)
    mean, scale = training_normalization(shards, input_dim(1))
    assert mean[0] == 0.5
    assert scale[0] == 0.5
    assert mean[1] == 0.0
    assert scale[1] == 1.0


def test_input_shard_manifest_uses_portable_hash_bound_paths(tmp_path: Path) -> None:
    shards = _corpus(tmp_path)
    records = input_shard_manifest(shards, tmp_path)
    assert {record["path"] for record in records} == {
        "train_0000.npz",
        "validation_0000.npz",
    }
    assert all(record["sha256"] == shard.sha256 for record, shard in zip(records, shards))
    with pytest.raises(ValueError):
        input_shard_manifest(shards, tmp_path / "other")


def test_incomplete_or_wrong_reference_corpus_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        inspect_feature_corpus(
            tmp_path,
            reference_identity=REFERENCE,
            source_split_sha256=SPLIT,
            train_input_indices=(17,),
            validation_count=1,
            budget_max=1,
            builder_sha256="builder",
            goal_regions=((0.9, 0.4),),
        )
    _corpus(tmp_path)
    width = input_dim(1)
    _write_shard(
        tmp_path,
        "validation",
        0,
        0,
        np.zeros((2, width), np.float32),
        np.asarray([0, 1], np.float32),
        reference={"checkpoint_sha256": "other", "time": 0.5},
    )
    with pytest.raises(ValueError, match="reference"):
        inspect_feature_corpus(
            tmp_path,
            reference_identity=REFERENCE,
            source_split_sha256=SPLIT,
            train_input_indices=(17,),
            validation_count=1,
            budget_max=1,
            builder_sha256="builder",
            goal_regions=((0.9, 0.4),),
        )


def test_changed_shard_fails_before_fit(tmp_path: Path) -> None:
    shards = _corpus(tmp_path)
    _write_shard(
        tmp_path,
        "train",
        0,
        17,
        np.ones((2, input_dim(1)), np.float32),
        np.asarray([0, 1], np.float32),
    )
    with pytest.raises(ValueError, match="changed after inventory"):
        training_normalization(shards, input_dim(1))


def test_builder_and_region_grid_are_locked(tmp_path: Path) -> None:
    _corpus(tmp_path)
    common = {
        "reference_identity": REFERENCE,
        "source_split_sha256": SPLIT,
        "train_input_indices": (17,),
        "validation_count": 1,
        "budget_max": 1,
    }
    with pytest.raises(ValueError, match="feature builder"):
        inspect_feature_corpus(tmp_path, **common, builder_sha256="changed")
    with pytest.raises(ValueError, match="goal grid"):
        inspect_feature_corpus(tmp_path, **common, goal_regions=((0.8, 0.4),))


def test_out_of_grid_region_index_fails(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_shard(
        tmp_path,
        "validation",
        0,
        0,
        np.zeros((2, input_dim(1)), np.float32),
        np.asarray([0, 1], np.float32),
        region_indices=np.asarray([0, 1], np.int16),
    )
    with pytest.raises(ValueError, match="invalid goal indices"):
        inspect_feature_corpus(
            tmp_path,
            reference_identity=REFERENCE,
            source_split_sha256=SPLIT,
            train_input_indices=(17,),
            validation_count=1,
            budget_max=1,
            builder_sha256="builder",
            goal_regions=((0.9, 0.4),),
        )


def test_synthetic_fit_returns_source_level_validation(tmp_path: Path) -> None:
    shards = _corpus(tmp_path)
    model, mean, scale, report = fit_value_head(
        shards,
        QEDFitConfig(budget_max=1, epochs=2, patience=1, batch_size=2, seed=3),
        guidance_region_index=0,
    )
    assert mean.shape == scale.shape == (input_dim(1),)
    assert report["selected_epoch"] in (0, 1)
    assert math.isfinite(report["selected_validation_source_mean_brier"])
    assert report["guidance_train_positive_labels"] == 2
    assert report["guidance_validation_positive_labels"] == 1
    with torch.no_grad():
        assert model(torch.zeros((1, input_dim(1)))).shape == (1, 1)


def test_bellman_pairs_are_counted_and_invalid_links_fail(tmp_path: Path) -> None:
    _corpus(tmp_path)
    width = input_dim(1)
    _write_shard(
        tmp_path,
        "train",
        0,
        17,
        np.zeros((4, width), np.float32),
        np.asarray([0, 1, 0, 1], np.float32),
        next_row_indices=np.asarray([1, -1, -1, -1], np.int32),
        next_terminal_targets=np.asarray([-1, 1, 0, 1], np.int8),
    )
    refreshed = inspect_feature_corpus(
        tmp_path,
        reference_identity=REFERENCE,
        source_split_sha256=SPLIT,
        train_input_indices=(17,),
        validation_count=1,
        budget_max=1,
        goal_regions=((0.9, 0.4),),
    )
    assert next(shard.bellman_pairs for shard in refreshed if shard.role == "train") == 4
    _, _, _, report = fit_value_head(
        refreshed,
        QEDFitConfig(budget_max=1, epochs=1, patience=1, batch_size=2, seed=3),
        guidance_region_index=0,
    )
    assert report["train_bellman_pairs"] == 4
    _write_shard(
        tmp_path,
        "train",
        0,
        17,
        np.zeros((4, width), np.float32),
        np.asarray([0, 1, 0, 1], np.float32),
        next_row_indices=np.asarray([0, -1, -1, -1], np.int32),
    )
    with pytest.raises(ValueError, match="Bellman successor links"):
        inspect_feature_corpus(
            tmp_path,
            reference_identity=REFERENCE,
            source_split_sha256=SPLIT,
            train_input_indices=(17,),
            validation_count=1,
            budget_max=1,
            goal_regions=((0.9, 0.4),),
        )


def test_all_negative_target_rows_do_not_qualify_guidance(tmp_path: Path) -> None:
    width = input_dim(1)
    for role, input_index in (("train", 17), ("validation", 0)):
        _write_shard(
            tmp_path,
            role,
            0,
            input_index,
            np.zeros((2, width), np.float32),
            np.zeros(2, np.float32),
        )
    shards = inspect_feature_corpus(
        tmp_path,
        reference_identity=REFERENCE,
        source_split_sha256=SPLIT,
        train_input_indices=(17,),
        validation_count=1,
        budget_max=1,
        goal_regions=((0.9, 0.4),),
    )
    _, _, _, report = fit_value_head(
        shards,
        QEDFitConfig(budget_max=1, epochs=1, patience=1, batch_size=2, seed=3),
        guidance_region_index=0,
    )
    assert report["guidance_validation_positive_labels"] == 0
    assert report["qualified_for_guidance"] is False


def test_fit_revision_requires_a_full_matching_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        fit_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    monkeypatch.setenv("COMPOSE_SOURCE_REVISION", commit)
    assert fit_cli.revision() == commit
    monkeypatch.setenv("COMPOSE_SOURCE_REVISION", "short")
    with pytest.raises(ValueError, match="full Git SHA"):
        fit_cli.revision()
