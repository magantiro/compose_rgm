from __future__ import annotations

import argparse

import pytest

from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.rewrite.kernel import canonical_state_key
from scripts.train_tracelet_cnof_gate import (
    TRANSPORT_SUPPORT_PROJECTION_VERSION,
    _compile_tracelet_proposal_partition_shards,
    _compile_tree_transport_partition_shards,
    _filter_records_by_finalized_manifest,
    _load_sharded_path_manifest,
    _path_record_support_key,
    _save_sharded_path_manifest,
    _sharded_path_cache_root,
)


def _arguments(tmp_path, *, require_path_cache: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        path_cache=tmp_path / "compiled_paths.pt",
        path_shard_size=2,
        tree_couplings_per_target=2,
        max_atoms=8,
        tree_transport="size_matched_graft",
        path_checkpoint_interval=2,
        require_path_cache=require_path_cache,
    )


def _record_identity(records) -> tuple:
    return tuple(
        (
            record.target_key,
            canonical_state_key(record.path.trace.source),
            tuple((step.rule_name, repr(step.action)) for step in record.path.trace.steps),
            tuple(canonical_state_key(state) for state in record.path.iter_states()),
        )
        for record in records
    )


def test_tree_transport_shards_resume_without_rewriting_completed_shards(tmp_path) -> None:
    args = _arguments(tmp_path)
    signature = {"format_version": 3, "split": ("CCC", "CCO", "CCN")}
    prior = DegreeBoundedCarbonTreePrior.from_size_counts({3: 3})

    first = _compile_tree_transport_partition_shards(
        partition="train",
        smiles=("CCC", "CCO", "CCN"),
        base_seed=19,
        args=args,
        source_prior=prior,
        ring_catalog=None,
        signature=signature,
        executor=None,
    )
    shard_root = _sharded_path_cache_root(args.path_cache) / "transport"
    shard_paths = sorted(shard_root.glob("train-*.pt"))
    mtimes = tuple(path.stat().st_mtime_ns for path in shard_paths)

    second = _compile_tree_transport_partition_shards(
        partition="train",
        smiles=("CCC", "CCO", "CCN"),
        base_seed=19,
        args=args,
        source_prior=prior,
        ring_catalog=None,
        signature=signature,
        executor=None,
    )

    assert len(first) == len(second) == 6
    assert _record_identity(first) == _record_identity(second)
    assert tuple(path.stat().st_mtime_ns for path in shard_paths) == mtimes


def test_ring_proposal_shards_resume_without_recompiling(tmp_path) -> None:
    args = _arguments(tmp_path)
    signature = {"format_version": 4, "split": ("CCC", "CCO", "CCN")}
    first = _compile_tracelet_proposal_partition_shards(
        partition="train",
        smiles=("CCC", "CCO", "CCN"),
        args=args,
        signature=signature,
        executor=None,
    )
    shard_root = _sharded_path_cache_root(args.path_cache) / "proposal"
    shard_paths = sorted(shard_root.glob("train-*.pt"))
    mtimes = tuple(path.stat().st_mtime_ns for path in shard_paths)

    second = _compile_tracelet_proposal_partition_shards(
        partition="train",
        smiles=("CCC", "CCO", "CCN"),
        args=args,
        signature=signature,
        executor=None,
    )

    assert _record_identity(first) == _record_identity(second)
    assert tuple(path.stat().st_mtime_ns for path in shard_paths) == mtimes


def test_sharded_manifest_round_trip_and_required_missing_shard(tmp_path) -> None:
    args = _arguments(tmp_path)
    signature = {"format_version": 3, "split": ("CCC",)}
    supported = {"train": ("CCC",), "validation": ("CCO",), "test": ("CCN",)}
    _save_sharded_path_manifest(
        args.path_cache,
        signature=signature,
        ring_catalog=None,
        supported_smiles=supported,
        complete=False,
    )

    payload = _load_sharded_path_manifest(args.path_cache, signature=signature)
    assert payload is not None
    assert payload["supported_smiles"] == supported
    assert payload["complete"] is False
    assert payload["transport_support_projection_version"] is None

    _save_sharded_path_manifest(
        args.path_cache,
        signature=signature,
        ring_catalog=None,
        supported_smiles=supported,
        complete=True,
        supported_record_keys={
            "train": (("train-target", "train-source"),),
            "validation": (("validation-target", "validation-source"),),
            "test": (("test-target", "test-source"),),
        },
    )
    finalized = _load_sharded_path_manifest(args.path_cache, signature=signature)
    assert finalized is not None
    assert finalized["transport_support_projection_version"] == TRANSPORT_SUPPORT_PROJECTION_VERSION
    assert finalized["supported_record_keys"]["train"] == (("train-target", "train-source"),)

    required_args = _arguments(tmp_path, require_path_cache=True)
    prior = DegreeBoundedCarbonTreePrior.from_size_counts({3: 1})
    try:
        _compile_tree_transport_partition_shards(
            partition="train",
            smiles=("CCC",),
            base_seed=23,
            args=required_args,
            source_prior=prior,
            ring_catalog=None,
            signature=signature,
            executor=None,
        )
    except FileNotFoundError as error:
        assert "required compiled path shard" in str(error)
    else:
        raise AssertionError("required cache accepted a missing transport shard")


def test_finalized_manifest_projection_filters_and_audits_couplings(tmp_path) -> None:
    args = _arguments(tmp_path)
    signature = {"format_version": 5, "split": ("CCC", "CCO", "CCN")}
    prior = DegreeBoundedCarbonTreePrior.from_size_counts({3: 3})
    records = _compile_tree_transport_partition_shards(
        partition="train",
        smiles=("CCC", "CCO", "CCN"),
        base_seed=29,
        args=args,
        source_prior=prior,
        ring_catalog=None,
        signature=signature,
        executor=None,
    )

    retained = _filter_records_by_finalized_manifest(
        records,
        ("CCC", "CCN"),
        n_slots=args.max_atoms,
        couplings_per_target=args.tree_couplings_per_target,
        partition="train",
    )
    assert len(retained) == 4
    assert len({record.target_key for record in retained}) == 2

    all_record_keys = tuple(_path_record_support_key(record) for record in records)
    excluded_key = all_record_keys[-1]
    exact_supported_keys = tuple(key for key in all_record_keys if key != excluded_key)
    exact_retained = _filter_records_by_finalized_manifest(
        records,
        ("CCC", "CCO", "CCN"),
        n_slots=args.max_atoms,
        couplings_per_target=args.tree_couplings_per_target,
        partition="train",
        supported_record_keys=exact_supported_keys,
    )
    assert len(exact_retained) < len(records)
    assert all(_path_record_support_key(record) != excluded_key for record in exact_retained)

    with pytest.raises(ValueError, match="endpoint audit failed"):
        _filter_records_by_finalized_manifest(
            records[:-1],
            ("CCC", "CCO", "CCN"),
            n_slots=args.max_atoms,
            couplings_per_target=args.tree_couplings_per_target,
            partition="train",
        )
