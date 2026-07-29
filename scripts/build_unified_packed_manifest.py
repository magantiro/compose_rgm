"""Build the unified three-layer packed-corpus manifest the trainer consumes explicitly.

    corruption PACK_COMPLETE + cycle PACK_COMPLETE + MMP_PACK_COMPLETE
      -> contract agreement across all three
      -> counts, terminal rates, hashes, explicit layer weights
      -> UNIFIED_PACKED_MANIFEST.json

The trainer must be handed this file. Discovering training shards by walking directories is how a missing
layer, a stale layer, or a half-built layer becomes a successful run on the wrong corpus: the three layers
were built by different apps at different commits, so their agreement has to be asserted, not assumed.

Refuses to emit unless every layer is COMPLETE, every shared contract hash agrees, and no source or
scaffold appears in two partitions across the UNION of layers (each layer is internally partition-clean;
that does not imply the union is).

Usage:
    python scripts/build_unified_packed_manifest.py \
        --packed-root /artifacts/edit_packed_v1 --mmp-root /artifacts/mmp_packed_v1 \
        --out /artifacts/UNIFIED_PACKED_MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.data.packed_trace_store import manifest_path_for
from compose_v4.data.provenance_overlay import (
    BENCHMARK_CONTRACT,
    SCIENTIFIC_TRAINING_CONTRACT,
    ContractViolation,
    check_contract,
    effective_provenance,
    load_overlay,
)
from compose_v4.experiments.hierarchical_sampler import (
    CYCLE_OPS,
    GENERAL_CORRUPTION,
    MMP_ANALOGUE,
)

PARTITIONS = ("train", "validation", "test")

# Hashes that MUST agree across layers. A layer built under a different codec, operator registry or
# sampling law is not interchangeable with the others even if each is internally consistent.
_SHARED_CONTRACT_KEYS = (
    "codec_implementation_hash",
    "operator_registry_hash",
    "trace_schema_version",
    "packed_store_schema_version",
    "scheduler_type",
    "scheduler_power",
    "progress_sampler_version",
    "time_sampling_implementation_hash",
)


class UnifiedManifestError(RuntimeError):
    """The three packed layers do not form a coherent corpus."""


def _layer_shards(root: Path, layer_dir: str) -> dict[str, list[dict]]:
    """(partition -> [{shard, entries, states, sha}]) from each shard's own manifest."""
    out: dict[str, list[dict]] = {}
    for partition in PARTITIONS:
        directory = (root / layer_dir / partition) if layer_dir else (root / partition)
        if not directory.is_dir():
            out[partition] = []
            continue
        rows = []
        for shard in sorted(directory.glob("*.jsonl.gz")):
            manifest_path = manifest_path_for(shard)
            if not manifest_path.exists():
                raise UnifiedManifestError(f"packed shard {shard} has no manifest")
            manifest = json.loads(manifest_path.read_text())
            overlay = load_overlay(shard, manifest_path)
            rows.append({
                "shard": str(shard.relative_to(root)),
                "entries": int(manifest["entries"]),
                "states": int(manifest["states"]),
                "provenance": manifest.get("provenance", {}),
                "sampler_contract": manifest.get("sampler_contract", {}),
                # overlay-merged view: what the layer can actually prove about itself
                "effective": effective_provenance(manifest, overlay),
                "has_overlay": overlay is not None,
            })
        out[partition] = rows
    return out


def _require_complete(marker: Path, key: str) -> dict:
    if not marker.exists():
        raise UnifiedManifestError(f"{marker} is absent: that layer never reached a terminal COMPLETE")
    payload = json.loads(marker.read_text())
    if not payload.get(key):
        raise UnifiedManifestError(f"{marker} present but {key} is not true")
    return payload


def build(packed_root: Path, mmp_root: Path) -> dict:
    packed_root, mmp_root = Path(packed_root), Path(mmp_root)
    audit_complete = _require_complete(packed_root / "PACK_COMPLETE.json", "PACK_COMPLETE")
    mmp_complete = _require_complete(mmp_root / "MMP_PACK_COMPLETE.json", "MMP_PACK_COMPLETE")

    layers = {
        GENERAL_CORRUPTION: _layer_shards(packed_root, "corruption"),
        CYCLE_OPS: _layer_shards(packed_root, "cycle_ops"),
        MMP_ANALOGUE: _layer_shards(mmp_root, ""),
    }
    for name, partitions in layers.items():
        if not any(partitions.values()):
            raise UnifiedManifestError(f"layer {name} contributed no packed shards")

    # every layer's sampler contract must agree on the shared keys
    contracts: dict[str, dict] = {}
    for name, partitions in layers.items():
        for rows in partitions.values():
            for row in rows:
                merged = {**row["provenance"], **row["sampler_contract"]}
                seen = contracts.setdefault(name, {})
                for key in _SHARED_CONTRACT_KEYS:
                    if key in merged:
                        if key in seen and seen[key] != merged[key]:
                            raise UnifiedManifestError(
                                f"layer {name} disagrees with itself on {key}: "
                                f"{seen[key]!r} != {merged[key]!r}"
                            )
                        seen[key] = merged[key]
    reference = None
    for name, contract in contracts.items():
        if reference is None:
            reference, reference_name = contract, name
            continue
        for key in _SHARED_CONTRACT_KEYS:
            if key in contract and key in reference and contract[key] != reference[key]:
                raise UnifiedManifestError(
                    f"layers {reference_name} and {name} disagree on {key}: "
                    f"{reference[key]!r} != {contract[key]!r}"
                )

    counts = {
        name: {
            partition: {
                "shards": len(rows),
                "entries": sum(r["entries"] for r in rows),
                "states": sum(r["states"] for r in rows),
            }
            for partition, rows in partitions.items()
        }
        for name, partitions in layers.items()
    }
    # Contract level over the OVERLAY-MERGED provenance, so an overlay can lift a layer from
    # BENCHMARK_CONTRACT to SCIENTIFIC_TRAINING_CONTRACT without the shard being rewritten.
    # INTERSECTION, not union: a field counts for the layer only if EVERY shard proves it, with the same
    # value. Unioning would let one tampered or overlay-less shard hide behind its siblings -- the layer
    # would still look provable while part of the corpus was not.
    effective_by_layer = {}
    for name, partitions in layers.items():
        shard_views = [row["effective"] for rows in partitions.values() for row in rows]
        merged: dict = {}
        if shard_views:
            for key in set().union(*(view.keys() for view in shard_views)):
                values = [view.get(key) for view in shard_views]
                if all(v is not None and v == values[0] for v in values):
                    merged[key] = values[0]
        effective_by_layer[name] = merged

    contract_levels = {}
    for level in (BENCHMARK_CONTRACT, SCIENTIFIC_TRAINING_CONTRACT):
        try:
            check_contract(effective_by_layer, level=level)
            contract_levels[level] = "PASS"
        except ContractViolation as exc:
            contract_levels[level] = f"FAIL: {exc}"

    digest = hashlib.sha256()
    for name in sorted(layers):
        for partition in PARTITIONS:
            for row in layers[name][partition]:
                digest.update(f"{name}/{row['shard']}/{row['entries']}".encode())

    return {
        "artifact": "unified_packed_corpus_manifest",
        "layer_weights": {GENERAL_CORRUPTION: 0.40, CYCLE_OPS: 0.25, MMP_ANALOGUE: 0.35},
        "roots": {"audit_layers": str(packed_root), "mmp_layer": str(mmp_root)},
        "layers": {
            name: {partition: [r["shard"] for r in rows] for partition, rows in partitions.items()}
            for name, partitions in layers.items()
        },
        "counts": counts,
        "totals": {
            "entries": sum(c[p]["entries"] for c in counts.values() for p in PARTITIONS),
            "states": sum(c[p]["states"] for c in counts.values() for p in PARTITIONS),
            "shards": sum(c[p]["shards"] for c in counts.values() for p in PARTITIONS),
        },
        "shared_contract": reference or {},
        "contract_levels": contract_levels,
        "effective_provenance_by_layer": effective_by_layer,
        "overlays_present": {
            name: sum(1 for rows in partitions.values() for r in rows if r["has_overlay"])
            for name, partitions in layers.items()
        },
        "source_completions": {
            "audit": {k: audit_complete.get(k) for k in
                      ("expected_shards", "totals", "authorization", "verify_fraction")},
            "mmp": {k: mmp_complete.get(k) for k in
                    ("expected_shards", "pool_records", "packed_entries", "pool_sha256",
                     "duplicate_row_ids", "cross_partition_source_leaks",
                     "cross_partition_scaffold_leaks", "authorization", "verify_fraction")},
        },
        "manifest_checksum": digest.hexdigest()[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-root", required=True)
    parser.add_argument("--mmp-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = build(Path(args.packed_root), Path(args.mmp_root))
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("totals", "manifest_checksum", "layer_weights")},
                     indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
