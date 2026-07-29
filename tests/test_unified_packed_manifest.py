"""The unified manifest must refuse an incoherent three-layer corpus.

The three layers are built by different Modal apps at different commits. Each being internally valid does
not make them interchangeable: a layer built under a different codec, operator registry or sampling law
would train fine and mean something else. Agreement is asserted here, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_unified_packed_manifest import UnifiedManifestError, build  # noqa: E402

CONTRACT = {
    "codec_implementation_hash": "baaa75367f25a8c6",
    "operator_registry_hash": "9197401e8dc3a7ae",
    "trace_schema_version": 2,
    "packed_store_schema_version": 1,
    "scheduler_type": "PowerSurvivalScheduler",
    "scheduler_power": 1.0,
    "progress_sampler_version": 1,
    "time_sampling_implementation_hash": "fdbf297e5545522e",
}


def _shard(directory: Path, name: str, *, entries=10, states=40, contract=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"")
    (directory / name).with_suffix(".manifest.json").write_text(
        json.dumps({
            "entries": entries, "states": states,
            "provenance": dict(contract or CONTRACT),
            "sampler_contract": dict(contract or CONTRACT),
        })
    )


def _corpus(tmp_path, *, mmp_contract=None, mmp_complete=True, audit_complete=True):
    packed, mmp = tmp_path / "packed", tmp_path / "mmp"
    for layer in ("corruption", "cycle_ops"):
        for partition in ("train", "validation", "test"):
            _shard(packed / layer / partition, "shard_0000.jsonl.gz")
    for partition in ("train", "validation", "test"):
        _shard(mmp / partition, "shard_0000.jsonl.gz", contract=mmp_contract)
    packed.mkdir(parents=True, exist_ok=True)
    (packed / "PACK_COMPLETE.json").write_text(
        json.dumps({"PACK_COMPLETE": audit_complete, "expected_shards": 6,
                    "totals": {"entries": 60}, "verify_fraction": 0.02, "authorization": {}})
    )
    (mmp / "MMP_PACK_COMPLETE.json").write_text(
        json.dumps({"MMP_PACK_COMPLETE": mmp_complete, "expected_shards": 3, "pool_records": 363456,
                    "packed_entries": 30, "pool_sha256": "abc", "duplicate_row_ids": 0,
                    "cross_partition_source_leaks": 0, "cross_partition_scaffold_leaks": 0,
                    "verify_fraction": 0.02, "authorization": {}})
    )
    return packed, mmp


def test_coherent_corpus_builds(tmp_path):
    packed, mmp = _corpus(tmp_path)
    manifest = build(packed, mmp)
    assert manifest["layer_weights"] == {
        "general_corruption": 0.40, "cycle_operations": 0.25, "mmp_analogue": 0.35
    }
    assert manifest["totals"]["shards"] == 9
    assert manifest["totals"]["entries"] == 90
    assert len(manifest["manifest_checksum"]) == 16


def test_missing_mmp_completion_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path, mmp_complete=False)
    with pytest.raises(UnifiedManifestError, match="MMP_PACK_COMPLETE is not true"):
        build(packed, mmp)


def test_missing_audit_completion_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path, audit_complete=False)
    with pytest.raises(UnifiedManifestError, match="PACK_COMPLETE is not true"):
        build(packed, mmp)


@pytest.mark.parametrize(
    "key", ["codec_implementation_hash", "operator_registry_hash", "scheduler_power",
            "progress_sampler_version", "time_sampling_implementation_hash"]
)
def test_cross_layer_contract_disagreement_is_refused(tmp_path, key):
    """A layer built under a different law must not be silently unioned with the others."""
    drifted = dict(CONTRACT)
    drifted[key] = "DRIFTED" if isinstance(drifted[key], str) else 99
    packed, mmp = _corpus(tmp_path, mmp_contract=drifted)
    with pytest.raises(UnifiedManifestError, match=f"disagree on {key}"):
        build(packed, mmp)


def test_empty_layer_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path)
    for partition in ("train", "validation", "test"):
        for f in (mmp / partition).glob("*"):
            f.unlink()
    with pytest.raises(UnifiedManifestError, match="contributed no packed shards"):
        build(packed, mmp)


def test_shard_without_manifest_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path)
    (mmp / "train" / "shard_0001.jsonl.gz").write_bytes(b"")
    with pytest.raises(UnifiedManifestError, match="has no manifest"):
        build(packed, mmp)
