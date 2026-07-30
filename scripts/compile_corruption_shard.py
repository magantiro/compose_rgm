#!/usr/bin/env python3
"""Compile one deterministic shard of precompiled corruption traces (V2 exact-state records).

This is the unit the parallel Modal fan-out maps over. It is deliberately a pure function of
(sources, partition, shard index, seed): no global state, no RNG carried across shards, so N containers
produce byte-identical output to one container doing the same strides.

Order of operations is load-bearing:

    scaffold-keyed partition  ->  select this shard's sources  ->  generate trajectories
                              ->  encode EXACT slot-addressed state  ->  validate by replay  ->  write

Partition assignment happens BEFORE any trajectory is generated and is keyed by Bemis-Murcko scaffold, so
every trajectory derived from any molecule sharing a scaffold is confined to one partition. Assigning after
generation -- or by molecule rather than scaffold -- leaks an edit neighbourhood across the split.

Uses only the production executor, the production corruption generator, and RewriteActionCodecV2. There is
deliberately no precompile-only executor or simplified action representation.

Usage:
  python scripts/compile_corruption_shard.py --sources corpus.smiles --partition train \
      --shard-index 0 --n-shards 1 --max-sources 200 --out-dir /tmp/corruption_shards
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, classify_smiles  # noqa: E402
from compose_v4.data.scaffold_partition import (  # noqa: E402
    assign_partitions,
    partitioner_provenance,
    verify_partition_disjointness,
)
from compose_v4.rewrite.trace_shard import (  # noqa: E402
    TraceShardError,
    decode_trace_record,
    encode_trace_record,
    shard_manifest,
    write_shard,
)

N_SLOTS = 40
SOURCE_SEED_SCHEMA = "compose.corruption.source_seed.v1"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def select_shard_sources(
    smiles_list: list[str],
    *,
    partition: str,
    shard_index: int,
    n_shards: int,
    max_sources: int | None = None,
) -> tuple[list[str], dict[str, str], dict]:
    """Scope-filter, scaffold-partition, then take this shard's deterministic stride.

    Returns ``(sources, scaffold_by_smiles, stats)``. The stride is taken AFTER partitioning so shards
    never straddle partitions.
    """
    eligible = [s for s in smiles_list if classify_smiles(s, BROAD_ORGANIC_V1)[0]]
    part_by, scaf_by, assign_stats = assign_partitions(eligible)
    verify_partition_disjointness(part_by, scaf_by)      # raises on any leak
    in_partition = sorted(s for s, p in part_by.items() if p == partition)
    if max_sources is not None:
        in_partition = in_partition[:max_sources]
    mine = [s for i, s in enumerate(in_partition) if i % n_shards == shard_index]
    stats = {
        "input_smiles": len(smiles_list),
        "scope_eligible": len(eligible),
        "in_partition": len(in_partition),
        "this_shard_sources": len(mine),
        **assign_stats,
    }
    return mine, scaf_by, stats


def _build_layer_records(layer: str, source: str, *, seed: int, catalog, depth_max: int,
                         couplings_per_target: int, max_bonds_per_molecule: int):
    """Dispatch to the production generator for this layer. Both return (records, n_attempted).

    The two compositional cycle families (cycle_close/cycle_open) come from build_cycle_op_records, NOT
    from the corruption generator -- so a corruption-only precompile would scale seven families while
    leaving the two DEFINING RingCore families on a tiny in-memory dataset. Compiling both layers here, from
    the SAME scaffold-partitioned sources through the SAME codec and validation, keeps them first-class.
    """
    if layer == "corruption":
        from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records

        return build_corrupted_prior_records(
            [source], n_slots=N_SLOTS, depth_max=depth_max, seed=seed,
            catalog=catalog, vocabulary=ORGANIC_VOCABULARY,
            couplings_per_target=couplings_per_target,
        )
    if layer == "cycle_ops":
        from compose_v4.experiments.cycle_op_prior import build_cycle_op_records

        return build_cycle_op_records(
            (source,), n_slots=N_SLOTS, seed=seed,
            max_bonds_per_molecule=max_bonds_per_molecule,
        )
    raise ValueError(f"unknown layer {layer!r}")


def deterministic_source_seed(seed: int, source: str) -> int:
    """Derive a byte-stable per-source seed independent of Python hash randomization."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    payload = f"{SOURCE_SEED_SCHEMA}\0{seed}\0{source}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31 - 1)


def compile_shard(
    sources: list[str],
    scaffold_by_smiles: dict[str, str],
    *,
    partition: str,
    shard_index: int,
    seed: int,
    layer: str = "corruption",
    depth_max: int = 5,
    couplings_per_target: int = 2,
    max_bonds_per_molecule: int = 4,
    catalog=None,
) -> tuple[list[dict], dict]:
    """Generate + encode + validate traces for this shard's sources, for one layer."""
    if catalog is None:
        from warmstart_dry_run import build_production_ring_catalog

        catalog = build_production_ring_catalog(N_SLOTS)

    records: list[dict] = []
    rejections: Counter = Counter()
    families: Counter = Counter()
    started = time.perf_counter()

    # Deterministic per-source seed so a source's trajectories are reproducible independent of shard layout.
    for source in sources:
        source_seed = deterministic_source_seed(seed, source)
        try:
            built, _log = _build_layer_records(
                layer, source, seed=source_seed, catalog=catalog, depth_max=depth_max,
                couplings_per_target=couplings_per_target,
                max_bonds_per_molecule=max_bonds_per_molecule,
            )
        except Exception as exc:  # noqa: BLE001
            rejections[f"generation:{type(exc).__name__}"] += 1
            continue
        if not built:
            rejections["generation:no_records"] += 1
            continue
        for j, rec in enumerate(built):
            trace = rec.path.trace
            if not trace.steps:
                rejections["empty_trace"] += 1
                continue
            try:
                record = encode_trace_record(
                    trace, n_slots=N_SLOTS, seed=source_seed,
                    trace_id=f"{layer}-{partition}-s{shard_index:04d}-{len(records):06d}",
                    partition=partition,
                    layer=layer,
                    source_scaffold=scaffold_by_smiles.get(source, ""),
                    extra={
                        "origin_smiles": source,
                        "partition_isolation": {
                            "schema": "compose.data.partition_isolation",
                            "schema_version": 1,
                            "molecule_ids": [source],
                            "scaffold_ids": [
                                scaffold_by_smiles.get(source, "")
                            ],
                            "source_group_id": source,
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- codec rejection is a real rejection reason
                rejections[f"encode:{type(exc).__name__}"] += 1
                continue
            # validate immediately: decode + full executor replay + canonical key assertions
            try:
                decode_trace_record(record, validate=True)
            except TraceShardError as exc:
                rejections[f"replay:{str(exc)[:40]}"] += 1
                continue
            families.update(record["family_histogram"])
            records.append(record)

    stats = {
        "layer": layer,
        "sources": len(sources),
        "accepted_traces": len(records),
        "accepted_transitions": sum(r["path_length"] for r in records),
        "unique_sources_with_output": len({r["metadata"]["origin_smiles"] for r in records}),
        "unique_scaffolds": len({r["source_scaffold"] for r in records}),
        "family_histogram": dict(families.most_common()),
        "seconds": round(time.perf_counter() - started, 2),
    }
    return records, {"stats": stats, "rejections": dict(rejections)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="newline SMILES file")
    parser.add_argument("--partition", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--layer", default="corruption", choices=("corruption", "cycle_ops"))
    parser.add_argument("--max-bonds-per-molecule", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--depth-max", type=int, default=5)
    parser.add_argument("--couplings-per-target", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    smiles = args.sources.read_text().split()
    sources, scaf_by, sel_stats = select_shard_sources(
        smiles, partition=args.partition, shard_index=args.shard_index,
        n_shards=args.n_shards, max_sources=args.max_sources,
    )
    print(json.dumps({"phase": "sources_selected", **sel_stats}, sort_keys=True), flush=True)

    records, report = compile_shard(
        sources, scaf_by, partition=args.partition, shard_index=args.shard_index,
        seed=args.seed, layer=args.layer, depth_max=args.depth_max,
        couplings_per_target=args.couplings_per_target,
        max_bonds_per_molecule=args.max_bonds_per_molecule,
    )
    shard_path = args.out_dir / args.layer / args.partition / f"shard_{args.shard_index:04d}.jsonl.gz"
    shard_stats = write_shard(shard_path, records)

    try:
        import ring_core_identity as rci

        op_hash = rci.recompute_operator_registry_hash()
        cap_hash = rci.CAPABILITY_HASH
    except Exception:  # noqa: BLE001
        op_hash = cap_hash = "unavailable"
    try:
        contract = (REPO / "diagnostics/coherence/program_contract_fingerprint.txt").read_text().strip()
    except OSError:
        contract = "unavailable"

    manifest = shard_manifest(
        shard_stats=shard_stats,
        partition=args.partition,
        counts={**report["stats"], **sel_stats},
        rejections=report["rejections"],
        compiler_commit=_git_commit(),
        program_contract_fingerprint=contract,
        operator_registry_hash=op_hash,
        capability_hash=cap_hash,
        extra={
            "shard_index": args.shard_index, "n_shards": args.n_shards, "seed": args.seed,
            # pin the partitioning semantics so a molecule cannot silently change partition later
            "layer": args.layer,
            "partitioner": partitioner_provenance(),
            "depth_max": args.depth_max, "couplings_per_target": args.couplings_per_target,
        },
    )
    manifest_path = shard_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase": "shard_written", "path": str(shard_path),
                      "manifest": str(manifest_path), **report["stats"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
