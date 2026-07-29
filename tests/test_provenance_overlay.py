"""Overlays add missing provenance without touching immutable shards; contracts gate on completeness.

The distinction under test: a throughput benchmark whose checkpoint is discarded can run on a corpus with
incomplete codec/schema provenance, but a number that goes in a paper cannot. The training launcher must
refuse exactly what the benchmark tolerates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.provenance_overlay import (  # noqa: E402
    BENCHMARK_CONTRACT,
    SCIENTIFIC_TRAINING_CONTRACT,
    ContractViolation,
    build_overlay,
    check_contract,
    effective_provenance,
    load_overlay,
    overlay_path_for,
    upgrade_implementation_hash,
)

BENCH_FIELDS = {
    "operator_registry_hash": "9197401e8dc3a7ae",
    "scheduler_type": "PowerSurvivalScheduler",
    "scheduler_power": 1.0,
    "progress_sampler_version": 1,
    "time_sampling_implementation_hash": "fdbf297e5545522e",
}
SCI_FIELDS = {
    "codec_implementation_hash": "baaa75367f25a8c6",
    "trace_schema_version": 2,
    "packed_store_schema_version": 1,
    "tensorization_implementation_hash": "abc1234567890def",
}


def _shard(tmp_path, name="shard_0000.jsonl.gz", provenance=None):
    shard = tmp_path / name
    shard.write_bytes(b"packed-bytes")
    manifest = tmp_path / (name.replace(".gz", "") + ".manifest.json")
    manifest.write_text(json.dumps({"entries": 5, "provenance": provenance or dict(BENCH_FIELDS)}))
    return shard, manifest


def test_overlay_binds_shard_and_manifest(tmp_path):
    shard, manifest = _shard(tmp_path)
    overlay = build_overlay(shard, manifest, fields=SCI_FIELDS, packer_commit="abc1234",
                            certification={"replay_verified": True, "sentinel": 8})
    assert overlay["packed_shard_content_sha256"] == hashlib.sha256(b"packed-bytes").hexdigest()
    assert overlay["upgrade_implementation_hash"] == upgrade_implementation_hash()
    assert overlay["certification"]["replay_verified"] is True


def test_overlay_requires_all_scientific_fields(tmp_path):
    shard, manifest = _shard(tmp_path)
    partial = dict(SCI_FIELDS)
    partial.pop("codec_implementation_hash")
    with pytest.raises(ContractViolation, match="missing required fields"):
        build_overlay(shard, manifest, fields=partial, packer_commit="abc", certification={})


def test_overlay_is_invalidated_by_a_changed_shard(tmp_path):
    """An overlay is a claim about specific bytes. Change the bytes and the claim is void."""
    shard, manifest = _shard(tmp_path)
    overlay_path_for(shard).write_text(json.dumps(
        build_overlay(shard, manifest, fields=SCI_FIELDS, packer_commit="abc", certification={})))
    assert load_overlay(shard, manifest) is not None
    shard.write_bytes(b"different-bytes")
    assert load_overlay(shard, manifest) is None


def test_overlay_is_invalidated_by_a_changed_manifest(tmp_path):
    shard, manifest = _shard(tmp_path)
    overlay_path_for(shard).write_text(json.dumps(
        build_overlay(shard, manifest, fields=SCI_FIELDS, packer_commit="abc", certification={})))
    manifest.write_text(json.dumps({"entries": 6, "provenance": dict(BENCH_FIELDS)}))
    assert load_overlay(shard, manifest) is None


def test_overlay_may_not_contradict_the_manifest(tmp_path):
    """An overlay ADDS. Disagreement about the same artifact is a contradiction, not a precedence rule."""
    shard, manifest = _shard(
        tmp_path, provenance={**BENCH_FIELDS, "codec_implementation_hash": "DIFFERENT"}
    )
    overlay = build_overlay(shard, manifest, fields=SCI_FIELDS, packer_commit="abc", certification={})
    with pytest.raises(ContractViolation, match="overlay contradicts the manifest"):
        effective_provenance(json.loads(manifest.read_text()), overlay)


def test_benchmark_contract_passes_without_codec_provenance():
    """Exactly the current corruption/cycle situation: enough to time, not enough to publish."""
    layers = {"general_corruption": dict(BENCH_FIELDS), "cycle_operations": dict(BENCH_FIELDS),
              "mmp_analogue": {**BENCH_FIELDS, **SCI_FIELDS}}
    assert check_contract(layers, level=BENCHMARK_CONTRACT)["level"] == BENCHMARK_CONTRACT


def test_scientific_contract_refuses_the_same_corpus():
    layers = {"general_corruption": dict(BENCH_FIELDS), "cycle_operations": dict(BENCH_FIELDS),
              "mmp_analogue": {**BENCH_FIELDS, **SCI_FIELDS}}
    with pytest.raises(ContractViolation, match="SCIENTIFIC_TRAINING_CONTRACT not satisfied"):
        check_contract(layers, level=SCIENTIFIC_TRAINING_CONTRACT)


def test_scientific_contract_passes_once_overlays_supply_the_fields():
    full = {**BENCH_FIELDS, **SCI_FIELDS}
    layers = {"general_corruption": dict(full), "cycle_operations": dict(full),
              "mmp_analogue": dict(full)}
    result = check_contract(layers, level=SCIENTIFIC_TRAINING_CONTRACT)
    assert set(result["layers"]) == {"general_corruption", "cycle_operations", "mmp_analogue"}


def test_contract_refuses_layers_that_disagree():
    """Possessing a field is not enough -- the layers must agree on it."""
    full = {**BENCH_FIELDS, **SCI_FIELDS}
    drifted = {**full, "codec_implementation_hash": "OTHER"}
    layers = {"general_corruption": dict(full), "cycle_operations": dict(drifted),
              "mmp_analogue": dict(full)}
    with pytest.raises(ContractViolation, match="layers disagree"):
        check_contract(layers, level=SCIENTIFIC_TRAINING_CONTRACT)


def test_unknown_contract_level_is_rejected():
    with pytest.raises(ValueError, match="unknown contract level"):
        check_contract({"a": dict(BENCH_FIELDS)}, level="SOMETHING_ELSE")
