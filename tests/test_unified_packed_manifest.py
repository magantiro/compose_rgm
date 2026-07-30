"""The unified manifest must refuse an incoherent three-layer corpus.

The three layers are built by different Modal apps at different commits. Each being internally valid does
not make them interchangeable: a layer built under a different codec, operator registry or sampling law
would train fine and mean something else. Agreement is asserted here, not assumed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_unified_packed_manifest import (  # noqa: E402
    UnifiedManifestError,
    build,
    validate,
)

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shard(
    directory: Path,
    name: str,
    *,
    entries=10,
    states=40,
    contract=None,
    shared_molecule: str | None = None,
    omit_isolation: bool = False,
):
    directory.mkdir(parents=True, exist_ok=True)
    shard = directory / name
    partition = directory.name
    layer = directory.parent.name
    with gzip.open(shard, "wt") as handle:
        for index in range(entries):
            metadata = {}
            if not omit_isolation:
                metadata["partition_isolation"] = {
                    "schema": "compose.data.partition_isolation",
                    "schema_version": 1,
                    "molecule_ids": [
                        (
                            shared_molecule
                            if index == 0 and shared_molecule is not None
                            else f"{partition}:{layer}:molecule:{index}"
                        )
                    ],
                    "scaffold_ids": [
                        f"{partition}:{layer}:scaffold:{index}"
                    ],
                    "source_group_id": (
                        f"{partition}:{layer}:source-group:{index}"
                    ),
                }
            handle.write(json.dumps({"trace": {"metadata": metadata}}) + "\n")
    shard.with_suffix(".manifest.json").write_text(
        json.dumps({
            "entries": entries, "states": states,
            "provenance": dict(contract or CONTRACT),
            "sampler_contract": dict(contract or CONTRACT),
        })
    )


def _shard_artifact(
    root: Path,
    shard: Path,
    *,
    layer: str | None,
) -> dict:
    manifest = shard.with_suffix(".manifest.json")
    payload = json.loads(manifest.read_text())
    result = {
        "partition": shard.parent.name,
        "shard": shard.name,
        "packed_sha256": _sha256(shard),
        "manifest_sha256": _sha256(manifest),
        "entries": payload["entries"],
        "states": payload["states"],
    }
    if layer is not None:
        result["layer"] = layer
    return result


def _corpus(
    tmp_path,
    *,
    mmp_contract=None,
    mmp_complete=True,
    audit_complete=True,
    overlap=False,
    omit_isolation=False,
):
    packed, mmp = tmp_path / "packed", tmp_path / "mmp"
    for layer in ("corruption", "cycle_ops"):
        for partition in ("train", "validation", "test"):
            _shard(
                packed / layer / partition,
                "shard_0000.jsonl.gz",
                shared_molecule=(
                    "shared-cross-partition-molecule"
                    if overlap
                    and layer == "corruption"
                    and partition == "train"
                    else None
                ),
                omit_isolation=(
                    omit_isolation
                    and layer == "corruption"
                    and partition == "train"
                ),
            )
    for partition in ("train", "validation", "test"):
        _shard(
            mmp / partition,
            "shard_0000.jsonl.gz",
            contract=mmp_contract,
            shared_molecule=(
                "shared-cross-partition-molecule"
                if overlap and partition == "validation"
                else None
            ),
        )
    packed.mkdir(parents=True, exist_ok=True)
    audit_artifacts = sorted(
        [
            _shard_artifact(
                packed,
                shard,
                layer=shard.parents[1].name,
            )
            for shard in packed.glob("*/*/*.jsonl.gz")
        ],
        key=lambda item: (
            item["layer"],
            item["partition"],
            item["shard"],
        ),
    )
    mmp_artifacts = sorted(
        [
            _shard_artifact(mmp, shard, layer=None)
            for shard in mmp.glob("*/*.jsonl.gz")
        ],
        key=lambda item: (item["partition"], item["shard"]),
    )
    (packed / "PACK_COMPLETE.json").write_text(
        json.dumps({"PACK_COMPLETE": audit_complete, "expected_shards": 6,
                    "totals": {"entries": 60}, "verify_fraction": 0.02,
                    "authorization": {}, "shard_artifacts": audit_artifacts})
    )
    (mmp / "MMP_PACK_COMPLETE.json").write_text(
        json.dumps({"MMP_PACK_COMPLETE": mmp_complete, "expected_shards": 3, "pool_records": 363456,
                    "packed_entries": 30, "pool_sha256": "abc", "duplicate_row_ids": 0,
                    "cross_partition_source_leaks": 0, "cross_partition_scaffold_leaks": 0,
                    "verify_fraction": 0.02, "authorization": {},
                    "shard_artifacts": mmp_artifacts})
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
    assert len(manifest["packed_corpus_inventory_sha256"]) == 64
    assert manifest["partition_isolation"]["status"] == "PASS"
    assert manifest["partition_isolation"][
        "cross_partition_overlap_counts"
    ] == {
        "molecule_ids": 0,
        "scaffold_ids": 0,
        "source_group_ids": 0,
    }


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


def test_completion_without_byte_inventory_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path)
    marker = packed / "PACK_COMPLETE.json"
    payload = json.loads(marker.read_text())
    del payload["shard_artifacts"]
    marker.write_text(json.dumps(payload))
    with pytest.raises(
        UnifiedManifestError,
        match="PACK_COMPLETE lacks shard_artifacts",
    ):
        build(packed, mmp)


def test_missing_explicit_partition_identifiers_block_manifest(tmp_path):
    packed, mmp = _corpus(tmp_path, omit_isolation=True)
    with pytest.raises(
        UnifiedManifestError,
        match="will not infer partition identities",
    ):
        build(packed, mmp)


def test_all_packed_producers_emit_the_explicit_isolation_envelope():
    compiler = (REPO / "scripts" / "compile_corruption_shard.py").read_text()
    audit_packer = (
        REPO / "modal_apps" / "pack_trace_states_app.py"
    ).read_text()
    mmp_packer = (
        REPO / "modal_apps" / "pack_mmp_pool_app.py"
    ).read_text()

    assert '"partition_isolation"' in compiler
    assert '"source_group_id"' in compiler
    assert '"shard_artifacts"' in audit_packer
    assert '"packed_sha256"' in audit_packer
    assert '"manifest_sha256"' in audit_packer
    assert '"partition_isolation"' in mmp_packer
    assert 'record.get("_source_scaffold"' in mmp_packer


def test_cross_layer_partition_overlap_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path, overlap=True)
    with pytest.raises(
        UnifiedManifestError,
        match="cross-layer partition isolation failed",
    ):
        build(packed, mmp)


def test_validator_recomputes_every_referenced_byte(tmp_path):
    packed, mmp = _corpus(tmp_path)
    path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    path.write_text(json.dumps(build(packed, mmp), indent=2, sort_keys=True))
    assert validate(path, packed_root=packed, mmp_root=mmp)[
        "partition_isolation"
    ]["status"] == "PASS"

    shard = packed / "corruption" / "train" / "shard_0000.jsonl.gz"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(UnifiedManifestError):
        validate(path, packed_root=packed, mmp_root=mmp)


def test_manifest_sidecar_tamper_is_refused(tmp_path):
    packed, mmp = _corpus(tmp_path)
    sidecar = (
        packed
        / "cycle_ops"
        / "validation"
        / "shard_0000.jsonl.manifest.json"
    )
    payload = json.loads(sidecar.read_text())
    payload["states"] += 1
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(
        UnifiedManifestError,
        match="shard_artifacts disagree",
    ):
        build(packed, mmp)


def _with_overlay(shard_dir: Path, name: str, fields: dict):
    """Write a VALID overlay next to a shard, as apply_provenance_overlays_app does."""
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from compose_v4.data.provenance_overlay import build_overlay, overlay_path_for

    shard = shard_dir / name
    manifest = shard_dir / (name.replace(".gz", "") + ".manifest.json")
    overlay = build_overlay(shard, manifest, fields=fields, packer_commit="abc1234",
                            certification={"replay_verified": True, "sentinel_entries_replayed": 4})
    overlay_path_for(shard).write_text(json.dumps(overlay, indent=2, sort_keys=True))


REPO = Path(__file__).resolve().parent.parent

SCI = {
    "codec_implementation_hash": "baaa75367f25a8c6",
    "trace_schema_version": 2,
    "packed_store_schema_version": 1,
    "tensorization_implementation_hash": "7b0c88b166828f44",
}


def test_without_overlays_only_the_benchmark_contract_passes(tmp_path):
    """The corpus as built: enough to time, not enough to publish."""
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    manifest = build(packed, mmp)
    assert manifest["contract_levels"]["BENCHMARK_CONTRACT"] == "PASS"
    assert manifest["contract_levels"]["SCIENTIFIC_TRAINING_CONTRACT"].startswith("FAIL")
    assert manifest["overlays_present"]["general_corruption"] == 0


def test_overlays_lift_the_corpus_to_the_scientific_contract(tmp_path):
    """An overlay must be able to satisfy the scientific contract WITHOUT the shard being rewritten."""
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    before = (packed / "corruption" / "train" / "shard_0000.jsonl.gz").read_bytes()
    for layer in ("corruption", "cycle_ops"):
        for partition in ("train", "validation", "test"):
            _with_overlay(packed / layer / partition, "shard_0000.jsonl.gz", SCI)
    manifest = build(packed, mmp)
    assert manifest["contract_levels"]["SCIENTIFIC_TRAINING_CONTRACT"] == "PASS"
    assert manifest["overlays_present"]["general_corruption"] == 3
    # the immutable shard must be untouched
    assert (packed / "corruption" / "train" / "shard_0000.jsonl.gz").read_bytes() == before


def test_a_tampered_shard_invalidates_its_overlay_and_drops_the_contract(tmp_path):
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    for layer in ("corruption", "cycle_ops"):
        for partition in ("train", "validation", "test"):
            _with_overlay(packed / layer / partition, "shard_0000.jsonl.gz", SCI)
    assert build(packed, mmp)["contract_levels"]["SCIENTIFIC_TRAINING_CONTRACT"] == "PASS"
    (packed / "corruption" / "train" / "shard_0000.jsonl.gz").write_bytes(b"tampered")
    with pytest.raises(UnifiedManifestError):
        build(packed, mmp)


def _overlay_file(tmp_path, exclusions=(), counts=None):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from compose_v4.data.representability_overlay import build_overlay as _bo

    overlay = _bo(list(exclusions), counts=counts or {"general_corruption": {"accepted": 5}},
                  enumerator_hash="enum1", packed_manifest_hashes={})
    path = tmp_path / "REPRESENTABILITY_OVERLAY.json"
    path.write_text(json.dumps(overlay))
    return path, overlay


def test_manifest_carries_the_overlay_and_effective_counts(tmp_path):
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    path, overlay = _overlay_file(
        tmp_path,
        exclusions=[{"layer": "general_corruption", "partition": "train", "trace_key": "t1"}],
        counts={"general_corruption": {"checked": 5, "accepted": 4, "excluded": 1}},
    )
    manifest = build(packed, mmp, path)
    assert manifest["representability_overlay"]["exclusions"] == 1
    assert manifest["representability_overlay"]["effective_corpus_checksum"] == (
        overlay["effective_corpus_checksum"]
    )
    assert manifest["effective_counts"] == {"general_corruption": 4}


def test_manifest_identity_changes_with_the_effective_corpus(tmp_path):
    """A different effective corpus must NOT reuse the pre-overlay manifest checksum."""
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    without = build(packed, mmp)["manifest_checksum"]

    path_a, _ = _overlay_file(tmp_path / "a", exclusions=[
        {"layer": "general_corruption", "partition": "train", "trace_key": "t1"}])
    path_b, _ = _overlay_file(tmp_path / "b", exclusions=[
        {"layer": "general_corruption", "partition": "train", "trace_key": "t2"}])
    with_a = build(packed, mmp, path_a)["manifest_checksum"]
    with_b = build(packed, mmp, path_b)["manifest_checksum"]

    assert with_a != without, "manifest identity must move once an overlay applies"
    assert with_a != with_b, "different exclusions must give different manifest identities"


def test_manifest_without_overlay_reports_none(tmp_path):
    packed, mmp = _corpus(tmp_path, mmp_contract={**CONTRACT, **SCI})
    manifest = build(packed, mmp)
    assert manifest["representability_overlay"] is None
    assert manifest["effective_counts"] is None
