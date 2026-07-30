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
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from compose_v4.data.packed_trace_store import manifest_path_for
from compose_v4.data.provenance_overlay import (
    BENCHMARK_CONTRACT,
    SCIENTIFIC_TRAINING_CONTRACT,
    ContractViolation,
    check_contract,
    effective_provenance,
    load_overlay,
    overlay_path_for,
)
from compose_v4.experiments.hierarchical_sampler import (
    CYCLE_OPS,
    GENERAL_CORRUPTION,
    MMP_ANALOGUE,
)

PARTITIONS = ("train", "validation", "test")
PARTITION_ISOLATION_SCHEMA = "compose.data.partition_isolation"
PARTITION_ISOLATION_SCHEMA_VERSION = 1
PACKED_CORPUS_INVENTORY_SCHEMA = "compose.data.unified_packed_corpus_inventory"
PACKED_CORPUS_INVENTORY_SCHEMA_VERSION = 1
_AUDIT_LAYER_DIRECTORIES = {
    GENERAL_CORRUPTION: "corruption",
    CYCLE_OPS: "cycle_ops",
}

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


def _stable_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explicit_partition_identifiers(
    trace: object,
    *,
    shard: Path,
    entry_index: int,
) -> dict[str, object]:
    if not isinstance(trace, dict):
        raise UnifiedManifestError(
            f"{shard} entry {entry_index} has no trace object"
        )
    metadata = trace.get("metadata")
    isolation = (
        metadata.get("partition_isolation")
        if isinstance(metadata, dict)
        else None
    )
    required_fields = {
        "schema",
        "schema_version",
        "molecule_ids",
        "scaffold_ids",
        "source_group_id",
    }
    if not isinstance(isolation, dict) or set(isolation) != required_fields:
        raise UnifiedManifestError(
            f"{shard} entry {entry_index} lacks the explicit "
            "metadata.partition_isolation envelope; rebuild the source trace "
            "and repack it with molecule_ids, scaffold_ids, and "
            "source_group_id. The unified manifest will not infer partition "
            "identities from SMILES or paths."
        )
    if (
        isolation["schema"] != PARTITION_ISOLATION_SCHEMA
        or isolation["schema_version"] != PARTITION_ISOLATION_SCHEMA_VERSION
    ):
        raise UnifiedManifestError(
            f"{shard} entry {entry_index} has an unsupported partition-isolation schema"
        )

    def identifiers(field: str) -> tuple[str, ...]:
        value = isolation[field]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            raise UnifiedManifestError(
                f"{shard} entry {entry_index} partition-isolation {field} "
                "must be a nonempty unique string list"
            )
        return tuple(value)

    source_group_id = isolation["source_group_id"]
    if not isinstance(source_group_id, str) or not source_group_id:
        raise UnifiedManifestError(
            f"{shard} entry {entry_index} partition-isolation "
            "source_group_id must be nonempty"
        )
    return {
        "molecule_ids": identifiers("molecule_ids"),
        "scaffold_ids": identifiers("scaffold_ids"),
        "source_group_id": source_group_id,
    }


def _scan_partition_identifiers(
    shard: Path,
    *,
    expected_entries: int,
) -> dict[str, object]:
    molecules: set[str] = set()
    scaffolds: set[str] = set()
    source_groups: set[str] = set()
    entries = 0
    try:
        with gzip.open(shard, "rt") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    packed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise UnifiedManifestError(
                        f"{shard} line {line_number} is not valid packed JSON"
                    ) from exc
                identifiers = _explicit_partition_identifiers(
                    packed.get("trace") if isinstance(packed, dict) else None,
                    shard=shard,
                    entry_index=entries,
                )
                molecules.update(identifiers["molecule_ids"])
                scaffolds.update(identifiers["scaffold_ids"])
                source_groups.add(str(identifiers["source_group_id"]))
                entries += 1
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise UnifiedManifestError(
            f"packed shard {shard} cannot be decoded exactly"
        ) from exc
    if entries != expected_entries:
        raise UnifiedManifestError(
            f"packed shard {shard} contains {entries} entries but its manifest "
            f"declares {expected_entries}"
        )
    payload = {
        "molecule_ids": sorted(molecules),
        "scaffold_ids": sorted(scaffolds),
        "source_group_ids": sorted(source_groups),
    }
    return {
        **payload,
        "identifier_inventory_sha256": _stable_sha256(payload),
    }


def _layer_shards(root: Path, layer_dir: str) -> dict[str, list[dict]]:
    """Read and physically address every packed shard, manifest, and overlay."""

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
            identifiers = _scan_partition_identifiers(
                shard,
                expected_entries=int(manifest["entries"]),
            )
            overlay_path = overlay_path_for(shard)
            rows.append({
                "shard": str(shard.relative_to(root)),
                "entries": int(manifest["entries"]),
                "states": int(manifest["states"]),
                "packed_sha256": _file_sha256(shard),
                "manifest_sha256": _file_sha256(manifest_path),
                "overlay_sha256": (
                    _file_sha256(overlay_path) if overlay_path.is_file() else None
                ),
                "identifier_inventory_sha256": identifiers[
                    "identifier_inventory_sha256"
                ],
                "identifier_counts": {
                    "molecules": len(identifiers["molecule_ids"]),
                    "scaffolds": len(identifiers["scaffold_ids"]),
                    "source_groups": len(identifiers["source_group_ids"]),
                },
                "provenance": manifest.get("provenance", {}),
                "sampler_contract": manifest.get("sampler_contract", {}),
                # overlay-merged view: what the layer can actually prove about itself
                "effective": effective_provenance(manifest, overlay),
                "has_overlay": overlay is not None,
                "_identifiers": identifiers,
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


def _public_shard_inventory(
    layers: dict[str, dict[str, list[dict]]],
) -> list[dict[str, object]]:
    return [
        {
            "layer": layer,
            "partition": partition,
            **{
                key: value
                for key, value in row.items()
                if key
                in {
                    "shard",
                    "entries",
                    "states",
                    "packed_sha256",
                    "manifest_sha256",
                    "overlay_sha256",
                    "identifier_inventory_sha256",
                    "identifier_counts",
                }
            },
        }
        for layer in sorted(layers)
        for partition in PARTITIONS
        for row in layers[layer][partition]
    ]


def _require_completion_shard_inventory(
    *,
    audit_complete: dict[str, Any],
    mmp_complete: dict[str, Any],
    shard_inventory: list[dict[str, object]],
) -> None:
    audit_expected = sorted(
        (
            {
                "layer": _AUDIT_LAYER_DIRECTORIES[str(row["layer"])],
                "partition": row["partition"],
                "shard": Path(str(row["shard"])).name,
                "packed_sha256": row["packed_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "entries": row["entries"],
                "states": row["states"],
            }
            for row in shard_inventory
            if row["layer"] in _AUDIT_LAYER_DIRECTORIES
        ),
        key=lambda item: (str(item["layer"]), str(item["partition"]), str(item["shard"])),
    )
    mmp_expected = sorted(
        (
            {
                "partition": row["partition"],
                "shard": Path(str(row["shard"])).name,
                "packed_sha256": row["packed_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "entries": row["entries"],
                "states": row["states"],
            }
            for row in shard_inventory
            if row["layer"] == MMP_ANALOGUE
        ),
        key=lambda item: (str(item["partition"]), str(item["shard"])),
    )
    for name, completion, expected in (
        ("PACK_COMPLETE", audit_complete, audit_expected),
        ("MMP_PACK_COMPLETE", mmp_complete, mmp_expected),
    ):
        observed = completion.get("shard_artifacts")
        if not isinstance(observed, list):
            raise UnifiedManifestError(
                f"{name} lacks shard_artifacts with packed and manifest SHA-256 "
                "identities; rebuild/repack before creating a scientific manifest"
            )
        if sorted(
            observed,
            key=lambda item: (
                str(item.get("layer", "")),
                str(item.get("partition", "")),
                str(item.get("shard", "")),
            ),
        ) != expected:
            raise UnifiedManifestError(
                f"{name} shard_artifacts disagree with the referenced packed bytes"
            )


def _partition_isolation_report(
    layers: dict[str, dict[str, list[dict]]],
) -> dict[str, object]:
    by_partition = {
        partition: {
            "molecule_ids": set(),
            "scaffold_ids": set(),
            "source_group_ids": set(),
        }
        for partition in PARTITIONS
    }
    for partitions in layers.values():
        for partition, rows in partitions.items():
            for row in rows:
                identifiers = row["_identifiers"]
                for field in by_partition[partition]:
                    by_partition[partition][field].update(identifiers[field])

    overlap_counts = {
        "molecule_ids": 0,
        "scaffold_ids": 0,
        "source_group_ids": 0,
    }
    overlap_details = []
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1:]:
            for field in overlap_counts:
                overlap = by_partition[left][field] & by_partition[right][field]
                overlap_counts[field] += len(overlap)
                if overlap:
                    overlap_details.append(
                        f"{left}/{right} share {len(overlap)} {field}"
                    )
    if overlap_details:
        raise UnifiedManifestError(
            "cross-layer partition isolation failed: "
            + "; ".join(overlap_details)
        )

    partition_payload = {}
    for partition in PARTITIONS:
        identifiers = {
            field: sorted(values)
            for field, values in by_partition[partition].items()
        }
        partition_payload[partition] = {
            "identifier_inventory_sha256": _stable_sha256(identifiers),
            "counts": {
                "molecules": len(identifiers["molecule_ids"]),
                "scaffolds": len(identifiers["scaffold_ids"]),
                "source_groups": len(identifiers["source_group_ids"]),
            },
        }
    return {
        "schema": PARTITION_ISOLATION_SCHEMA,
        "schema_version": PARTITION_ISOLATION_SCHEMA_VERSION,
        "status": "PASS",
        "cross_partition_overlap_counts": overlap_counts,
        "by_partition": partition_payload,
    }


def build(packed_root: Path, mmp_root: Path, overlay_path: Path | None = None) -> dict:
    packed_root, mmp_root = Path(packed_root), Path(mmp_root)
    # The representability overlay changes the EFFECTIVE corpus, so the manifest must carry it and derive
    # a distinct identity. Reusing the pre-overlay checksum would claim a corpus that is not being trained
    # on -- the exact dishonesty the overlay exists to prevent.
    representability = None
    if overlay_path is not None:
        from compose_v4.data.representability_overlay import load_overlay as _load_representability

        representability = _load_representability(Path(overlay_path))
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
    shard_inventory = _public_shard_inventory(layers)
    _require_completion_shard_inventory(
        audit_complete=audit_complete,
        mmp_complete=mmp_complete,
        shard_inventory=shard_inventory,
    )
    partition_isolation = _partition_isolation_report(layers)

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

    completion_identities = {
        "audit_pack_complete_sha256": _file_sha256(
            packed_root / "PACK_COMPLETE.json"
        ),
        "mmp_pack_complete_sha256": _file_sha256(
            mmp_root / "MMP_PACK_COMPLETE.json"
        ),
    }
    inventory_identity_payload = {
        "schema": PACKED_CORPUS_INVENTORY_SCHEMA,
        "schema_version": PACKED_CORPUS_INVENTORY_SCHEMA_VERSION,
        "completion_identities": completion_identities,
        "representability_overlay_sha256": (
            _file_sha256(Path(overlay_path)) if overlay_path is not None else None
        ),
        "shards": shard_inventory,
        "partition_isolation": partition_isolation,
    }
    packed_corpus_inventory_sha256 = _stable_sha256(
        inventory_identity_payload
    )

    result = {
        "artifact": "unified_packed_corpus_manifest",
        "schema_version": 2,
        "layer_weights": {GENERAL_CORRUPTION: 0.40, CYCLE_OPS: 0.25, MMP_ANALOGUE: 0.35},
        "roots": {"audit_layers": str(packed_root), "mmp_layer": str(mmp_root)},
        "layers": {
            name: {partition: [r["shard"] for r in rows] for partition, rows in partitions.items()}
            for name, partitions in layers.items()
        },
        "counts": counts,
        "shard_inventory": shard_inventory,
        "completion_identities": completion_identities,
        "partition_isolation": partition_isolation,
        "packed_corpus_inventory_schema": PACKED_CORPUS_INVENTORY_SCHEMA,
        "packed_corpus_inventory_schema_version": (
            PACKED_CORPUS_INVENTORY_SCHEMA_VERSION
        ),
        "packed_corpus_inventory_sha256": packed_corpus_inventory_sha256,
        "totals": {
            "entries": sum(c[p]["entries"] for c in counts.values() for p in PARTITIONS),
            "states": sum(c[p]["states"] for c in counts.values() for p in PARTITIONS),
            "shards": sum(c[p]["shards"] for c in counts.values() for p in PARTITIONS),
        },
        "representability_overlay": (
            None if representability is None else {
                "representability_filter": representability["representability_filter"],
                "filter_implementation_hash": representability["filter_implementation_hash"],
                "candidate_enumerator_hash": representability["candidate_enumerator_hash"],
                "effective_corpus_checksum": representability["effective_corpus_checksum"],
                "exclusions": len(representability["exclusions"]),
                "counts": representability["counts"],
            }
        ),
        "effective_counts": (
            None if representability is None else {
                layer: values["accepted"] for layer, values in representability["counts"].items()
            }
        ),
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
        "manifest_checksum": packed_corpus_inventory_sha256[:16],
    }
    result["manifest_payload_sha256"] = _stable_sha256(result)
    return result


def validate(
    manifest_path: Path,
    *,
    packed_root: Path,
    mmp_root: Path,
    overlay_path: Path | None = None,
) -> dict:
    """Rebuild the complete inventory and require byte-identical manifest claims."""

    path = Path(manifest_path)
    if not path.is_file():
        raise UnifiedManifestError(f"unified packed manifest is absent: {path}")
    try:
        observed = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnifiedManifestError(
            f"unified packed manifest is not valid JSON: {path}"
        ) from exc
    expected = build(
        Path(packed_root),
        Path(mmp_root),
        Path(overlay_path) if overlay_path is not None else None,
    )
    if observed != expected:
        raise UnifiedManifestError(
            "unified packed manifest disagrees with current packed bytes, "
            "completion inventories, or partition-isolation evidence"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-root", required=True)
    parser.add_argument("--mmp-root", required=True)
    parser.add_argument("--overlay", default=None,
                        help="REPRESENTABILITY_OVERLAY.json; folds the effective corpus into "
                             "the manifest identity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = build(Path(args.packed_root), Path(args.mmp_root),
                     Path(args.overlay) if args.overlay else None)
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("totals", "manifest_checksum", "layer_weights")},
                     indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
