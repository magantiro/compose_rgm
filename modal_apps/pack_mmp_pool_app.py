"""Pack the MMP analogue pool: partition once -> parallel pack -> reduce -> MMP_PACK_COMPLETE.

    authoritative pool (363,456 records)
      -> ONE streaming partition pass (scaffold-keyed, exact reconciliation)
      -> N parallel packers (one per raw shard, atomic finalize)
      -> reducer (coverage, checksums, overlap, replay audit)
      -> MMP_PACK_COMPLETE

Why: corruption and cycle_ops are packed, but MMP still reconstructed every trace at load
(``rewrite_trace_from_record`` + ``TraceProgressCTMC``), measured at 13.74 ms per row scanned -- 83 min per
partition, and the loader scanned the whole pool once per partition. That dominated everything else and
would have burned ~2.8 h of A100 time before the first optimizer step.

Sharding arithmetic is MEASURED, not assumed (see _PACK_RATE): 86.4 records/s single core, 11.57 ms/record,
306 gz bytes/record -> 1.17 h serial, ~0.11 GB packed. Shards are duration-targeted so validation and test
do not become a long tail behind train.

The pool stays authoritative; this store is a deterministic derivative bound to the pool's content hash and
the full contract. A changed pool invalidates it rather than silently pairing stale states with new rows.

    modal run --detach modal_apps/pack_mmp_pool_app.py --commit <sha> --gate-sha <sha16>
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "scipy==1.13.1", "networkx==3.3", "rdkit==2024.3.5")
    .env({
        "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
)

app = modal.App("compose-v4-pack-mmp-pool")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

POOL_PATH = "/artifacts/edit_mining_full_broad_40/edit_pool_full.jsonl"
EXPECTED_POOL_RECORDS = 363_456
PARTITIONS = ("train", "validation", "test")
LAYER = "mmp_analogue"

# ---- measured sharding arithmetic (scripts-derived, persisted into the manifest) --------------------
_PACK_RATE = {
    "records_per_second_single_core": 86.4,
    "pack_ms_per_record": 11.57,
    "partition_ms_per_record": 0.36,
    "gz_bytes_per_record": 306,
    "measured_on": "600-record stride sample of the real pool, mean K 6.08 vs pool 5.895",
}
_TARGET_SHARD_SECONDS = 240        # ~4 min: well above container startup, so overhead stays <25%
_CONTAINER_MARGIN = 5
_VERIFY_FRACTION = 0.02
_SENTINEL_ENTRIES = 8
# v1: partition by target scaffold only (LEAKED: 1,325 sources spanned partitions)
# v2: require BOTH endpoints to map to the same partition; drop straddling pairs, counted
_MMP_PARTITION_RULE_VERSION = 2


def pair_rule_implementation_hash() -> str:
    """Hash of ``pair_partition``'s SOURCE.

    A hand-incremented version only invalidates a cache when someone remembers to bump it. Hashing the
    function body means an edit to the rule invalidates every derived artifact whether or not the version
    was touched.
    """
    import hashlib
    import inspect

    return hashlib.sha256(inspect.getsource(pair_partition).encode()).hexdigest()[:16]


def shards_for(record_count: int) -> int:
    """Duration-targeted shard count. Small partitions get 1 shard, not a proportional sliver."""
    if record_count <= 0:
        return 0
    seconds = record_count * _PACK_RATE["pack_ms_per_record"] / 1000.0
    return max(1, round(seconds / _TARGET_SHARD_SECONDS) or 1)


def pair_partition(source_smiles: str, target_smiles: str) -> tuple[str | None, str, str]:
    """Partition for one MMP pair, or None when its endpoints disagree.

    Returns ``(partition_or_None, source_scaffold, target_scaffold)``. An MMP pair has TWO endpoints;
    partitioning on the target alone let one source molecule reach several partitions through
    differently-scaffolded targets (measured: 1,325 sources spanned partitions). Both endpoints must map
    to the same partition, which makes molecule-level leakage impossible by construction.
    """
    from compose_v4.data.scaffold_partition import murcko_scaffold, partition_for_scaffold

    source = (source_smiles or "").strip()
    target = (target_smiles or "").strip()
    if not source or not target:
        return None, "", ""
    source_scaffold = murcko_scaffold(source)
    target_scaffold = murcko_scaffold(target)
    if not source_scaffold or not target_scaffold:
        return None, source_scaffold or "", target_scaffold or ""
    source_partition = partition_for_scaffold(source_scaffold, salt="ringcore-v1")
    target_partition = partition_for_scaffold(target_scaffold, salt="ringcore-v1")
    if source_partition != target_partition:
        return None, source_scaffold, target_scaffold
    return source_partition, source_scaffold, target_scaffold


def _status(subdir: str, status: str, **fields) -> dict:
    """RUNNING | PARTIAL | FAILED | COMPLETE. Only the reducer may declare COMPLETE."""
    payload = {"status": status, **fields}
    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "mmp_pack_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({"phase": "status", **{k: v for k, v in payload.items()
                                            if k not in ("shards", "missing", "stale")}},
                     sort_keys=True)[:900], flush=True)
    return payload


def _contract() -> dict:
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        PACKED_STORE_SCHEMA_VERSION,
        sampler_contract,
    )
    from compose_v4.data.scaffold_partition import partitioner_provenance
    from compose_v4.rewrite.action_codec import codec_implementation_hash
    from compose_v4.rewrite.trace_shard import TRACE_SCHEMA_VERSION
    from ring_core_identity import recompute_operator_registry_hash

    return {
        "packed_store_schema_version": PACKED_STORE_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "codec_implementation_hash": codec_implementation_hash(),
        "operator_registry_hash": recompute_operator_registry_hash(),
        "partitioner": partitioner_provenance(),
        **sampler_contract(),
    }


def _pool_sha256() -> str:
    """Streaming content hash of the authoritative pool, used to decide partition reuse."""
    import hashlib

    digest = hashlib.sha256()
    with open(POOL_PATH, "rb") as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=8.0, memory=32768, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume})
def partition_pool(subdir: str) -> dict:
    """Stream the pool EXACTLY ONCE: hash it, assign partitions, and write raw intermediate shards.

    Letting every packing worker scan all 363,456 records would multiply an 83-minute scan by the worker
    count. Partition assignment happens here, once, and every record is accounted for -- reconciliation to
    the expected total is required, and any rejection is counted with a reason rather than dropped.
    """
    import hashlib
    import sys
    import time

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    artifact_volume.reload()
    started = time.time()
    out_root = Path("/artifacts") / subdir / "_raw"
    out_root.mkdir(parents=True, exist_ok=True)

    # Reuse a prior partitioning when the pool and the partitioner are unchanged. The scan is cheap
    # (measured 127 s for 363,456 records) but it is pure overhead on a rebuild, and skipping it keeps
    # raw shard membership byte-identical -- which is what lets packed shards be reused too.
    existing_manifest = Path("/artifacts") / subdir / "partition_manifest.json"
    if existing_manifest.exists():
        try:
            previous = json.loads(existing_manifest.read_text())
        except json.JSONDecodeError:
            previous = None
        if previous:
            current_pool_sha = _pool_sha256()
            if (
                previous.get("pool_sha256") == current_pool_sha
                and previous.get("contract", {}).get("partitioner") == _contract()["partitioner"]
                # the app's OWN pairing rule is not covered by the partitioner provenance hash;
                # without this, a rebuild would silently reuse a partitioning built under the old rule
                and previous.get("mmp_partition_rule_version") == _MMP_PARTITION_RULE_VERSION
                and previous.get("pair_rule_implementation_hash") == pair_rule_implementation_hash()
                and previous.get("scanned") == EXPECTED_POOL_RECORDS
                and all(
                    (out_root / e["partition"] / e["shard"]).exists() for e in previous.get("shards", [])
                )
            ):
                print(json.dumps({"phase": "partition_reuse", "pool_sha256": current_pool_sha[:16],
                                  "shards": previous["expected_shards"]}), flush=True)
                return previous

    digest = hashlib.sha256()
    rows_by_partition: dict[str, list] = {p: [] for p in PARTITIONS}
    rejects: dict[str, int] = {}
    dropped_row_ids: list[int] = []
    molecules_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    dropped_bias: dict[str, dict] = {
        k: {} for k in ("path_length", "direction", "atom_count_delta",
                        "cycle_rank_delta", "operator_signature")
    }

    def _bump(counter: dict, key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    sources_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    scaffolds_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    scanned = 0

    with open(POOL_PATH, "rb") as raw:
        for line in raw:
            digest.update(line)
            text = line.strip()
            if not text:
                continue
            scanned += 1
            record = json.loads(text)
            target = (record.get("target_smiles") or "").strip()
            if not target:
                rejects["empty_target_smiles"] = rejects.get("empty_target_smiles", 0) + 1
                continue
            partition, source_scaffold, scaffold = pair_partition(
                record.get("source_smiles", ""), target
            )
            if partition is None:
                reason = ("unassignable_scaffold" if not (source_scaffold and scaffold)
                          else "endpoints_straddle_partitions")
                rejects[reason] = rejects.get(reason, 0) + 1
                dropped_row_ids.append(scanned - 1)
                if reason == "endpoints_straddle_partitions":
                    # Descriptive bias audit ONLY -- it must never feed back into the split rule. Its job
                    # is to show the holdout correction did not quietly delete one transformation class.
                    _bump(dropped_bias["path_length"], str(record.get("path_length")))
                    _bump(dropped_bias["direction"], str(record.get("direction") or "unknown"))
                    _bump(dropped_bias["atom_count_delta"], str(record.get("atom_count_delta")))
                    _bump(dropped_bias["cycle_rank_delta"], str(record.get("cycle_rank_delta")))
                    _bump(dropped_bias["operator_signature"],
                          ",".join(sorted((record.get("operator_histogram") or {}))))
                continue
            record["_row_id"] = scanned - 1          # exact original row index
            record["_scaffold"] = scaffold
            record["_source_scaffold"] = source_scaffold
            rows_by_partition[partition].append(record)
            sources_by_partition[partition].add(record.get("source_smiles", ""))
            scaffolds_by_partition[partition].add(scaffold)
            # BOTH molecules and BOTH scaffolds belong to the partition -- tracking only the source would
            # miss exactly the leak this rule exists to prevent.
            molecules_by_partition[partition].add(record.get("source_smiles", ""))
            molecules_by_partition[partition].add(target)
            scaffolds_by_partition[partition].add(source_scaffold)

    pool_sha256 = digest.hexdigest()
    if scanned != EXPECTED_POOL_RECORDS:
        return _status(subdir, "FAILED", reason=f"pool has {scanned} records, expected "
                       f"{EXPECTED_POOL_RECORDS}", pool_sha256=pool_sha256)

    # deterministic shard assignment inside each partition, duration-targeted
    plan: list[dict] = []
    for partition in PARTITIONS:
        records = rows_by_partition[partition]
        n_shards = shards_for(len(records))
        for shard_index in range(n_shards):
            slice_rows = records[shard_index::n_shards]     # stride keeps path-length mix even
            name = f"shard_{shard_index:04d}.jsonl"
            directory = out_root / partition
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in slice_rows)
            )
            plan.append({
                "partition": partition, "shard": name, "records": len(slice_rows),
                "row_ids_sha256": hashlib.sha256(
                    ",".join(str(r["_row_id"]) for r in slice_rows).encode()
                ).hexdigest()[:16],
            })

    # Zero overlap must hold at partition time. Discovering it in the reducer means the whole packing
    # fan-out was already paid for -- which is exactly what happened on the v1 rule.
    molecule_overlap, scaffold_overlap = {}, {}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            shared_mols = molecules_by_partition[a] & molecules_by_partition[b]
            shared_scaffolds = scaffolds_by_partition[a] & scaffolds_by_partition[b]
            if shared_mols:
                molecule_overlap[f"{a}|{b}"] = len(shared_mols)
            if shared_scaffolds:
                scaffold_overlap[f"{a}|{b}"] = len(shared_scaffolds)
    if molecule_overlap or scaffold_overlap:
        return _status(subdir, "FAILED", reason="partitions overlap after the pair rule",
                       molecule_overlap=molecule_overlap, scaffold_overlap=scaffold_overlap)

    kept = sum(len(v) for v in rows_by_partition.values())
    if kept + sum(rejects.values()) != scanned:
        return _status(subdir, "FAILED", reason="record reconciliation failed",
                       scanned=scanned, kept=kept, rejects=rejects)

    manifest = {
        "pool_path": POOL_PATH,
        "pool_sha256": pool_sha256,
        "scanned": scanned,
        "kept": kept,
        "rejects": rejects,
        "accepted_records": kept,
        "endpoints_straddle_partitions": rejects.get("endpoints_straddle_partitions", 0),
        "dropped_row_ids_sha256": hashlib.sha256(
            ",".join(str(r) for r in sorted(dropped_row_ids)).encode()
        ).hexdigest()[:16],
        "dropped_row_ids_count": len(dropped_row_ids),
        "unique_molecules_by_partition": {p: len(molecules_by_partition[p]) for p in PARTITIONS},
        "molecule_overlap": molecule_overlap,
        "scaffold_overlap": scaffold_overlap,
        "dropped_bias": dropped_bias,
        "mmp_partition_rule_version": _MMP_PARTITION_RULE_VERSION,
        "pair_rule_implementation_hash": pair_rule_implementation_hash(),
        "expected_shards": len(plan),
        "shards": plan,
        "counts_by_partition": {p: len(rows_by_partition[p]) for p in PARTITIONS},
        "unique_sources_by_partition": {p: len(sources_by_partition[p]) for p in PARTITIONS},
        "unique_scaffolds_by_partition": {p: len(scaffolds_by_partition[p]) for p in PARTITIONS},
        "sharding_arithmetic": {
            **_PACK_RATE,
            "target_shard_seconds": _TARGET_SHARD_SECONDS,
            "shards": len(plan),
            "max_containers": len(plan) + _CONTAINER_MARGIN,
            "expected_critical_path_seconds": _TARGET_SHARD_SECONDS,
        },
        "contract": _contract(),
        "seconds": round(time.time() - started, 1),
    }
    (Path("/artifacts") / subdir / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    artifact_volume.commit()
    print(json.dumps({"phase": "partitioned", "scanned": scanned, "kept": kept,
                      "shards": len(plan), "rejects": rejects,
                      "counts": manifest["counts_by_partition"]}, sort_keys=True), flush=True)
    return manifest


@app.function(image=image, cpu=4.0, memory=16384, timeout=4 * 3600,
              max_containers=64, volumes={"/artifacts": artifact_volume})
def pack_mmp_shard(subdir: str, partition: str, name: str, pool_sha256: str,
                   row_ids_sha256: str) -> dict:
    """Pack one raw shard. Atomic finalize; reuse only on a full contract + identity match."""
    import sys
    import time

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        assert_closed_form_applies,
        build_packed_entry,
        manifest_path_for,
        write_packed_shard,
    )
    from compose_v4.experiments.analogue_prior import rewrite_trace_from_record
    from compose_v4.rewrite.progress import TraceProgressCTMC
    from compose_v4.rewrite.trace_shard import encode_trace_record

    assert_closed_form_applies()
    artifact_volume.reload()
    contract = _contract()
    provenance = {
        "layer": LAYER,
        "source_pool_sha256": pool_sha256,
        "row_ids_sha256": row_ids_sha256,
        "partition": partition,
        **contract,
    }
    packed_name = name + ".gz"
    dest = Path("/artifacts") / subdir / partition / packed_name

    manifest_path = manifest_path_for(dest)
    if dest.exists() and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        stored = existing.get("provenance", {})
        if (
            stored.get("source_pool_sha256") == pool_sha256
            and stored.get("row_ids_sha256") == row_ids_sha256
            and all(stored.get(k) == v for k, v in contract.items())
        ):
            print(json.dumps({"phase": "reuse", "shard": f"{partition}/{packed_name}"}), flush=True)
            return {"partition": partition, "shard": packed_name, "reused": True,
                    "entries": existing.get("entries", 0)}

    started = time.time()
    raw = Path("/artifacts") / subdir / "_raw" / partition / name
    entries, unbuildable = [], 0
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            trace = rewrite_trace_from_record(record)
        except Exception:  # noqa: BLE001 -- counted, never silently dropped
            unbuildable += 1
            continue
        # Normalize the V1 pool row to the V2 trace schema. The packed loader decodes actions with the
        # V2 codec, so storing raw V1 rows would fail at read time; normalizing here also gives the MMP
        # layer the same schema and provenance as corruption/cycle_ops -- ONE loader path, not two.
        v2 = encode_trace_record(
            trace,
            n_slots=int(record.get("n_slots", 40)),
            seed=0,
            trace_id=f"mmp-{record['_row_id']}",
            partition=partition,
            layer=LAYER,
            direction=str(record.get("direction", "")),
            source_scaffold=str(record.get("_scaffold", "")),
            extra={
                "row_id": int(record["_row_id"]),
                "scaffold": str(record.get("_scaffold", "")),
                "source_smiles": str(record.get("source_smiles", "")),
                "target_smiles": str(record.get("target_smiles", "")),
            },
        )
        entries.append(build_packed_entry(v2, TraceProgressCTMC(trace)))

    # Write to a unique temp path, then rename: a killed container must never leave a half shard that a
    # later reducer could mistake for complete.
    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
    manifest = write_packed_shard(tmp, entries, provenance=provenance)
    manifest_path_for(tmp).replace(manifest_path)
    tmp.replace(dest)
    artifact_volume.commit()
    result = {"partition": partition, "shard": packed_name, "reused": False,
              "entries": manifest["entries"], "states": manifest["states"],
              "unbuildable": unbuildable, "seconds": round(time.time() - started, 1)}
    print(json.dumps({"phase": "packed", **result}, sort_keys=True), flush=True)
    return result


@app.function(image=image, cpu=16.0, memory=65536, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume})
def reduce_mmp(subdir: str, build_meta: dict) -> dict:
    """Refuse MMP_PACK_COMPLETE unless coverage, reconciliation, overlap and replay all pass."""
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        manifest_path_for,
        read_packed_shard,
        sentinel_replay_check,
    )

    artifact_volume.reload()
    root = Path("/artifacts") / subdir
    plan = json.loads((root / "partition_manifest.json").read_text())
    contract = _contract()

    missing, stale, totals = [], [], {"entries": 0, "states": 0}
    by_partition: dict[str, int] = {}
    for entry in plan["shards"]:
        partition, name = entry["partition"], entry["shard"] + ".gz"
        shard = root / partition / name
        manifest_path = manifest_path_for(shard)
        if not shard.exists() or not manifest_path.exists():
            missing.append(f"{partition}/{name}")
            continue
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("provenance", {})
        if (
            stored.get("source_pool_sha256") != plan["pool_sha256"]
            or stored.get("row_ids_sha256") != entry["row_ids_sha256"]
            or any(stored.get(k) != v for k, v in contract.items())
        ):
            stale.append(f"{partition}/{name}")
            continue
        totals["entries"] += manifest["entries"]
        totals["states"] += manifest["states"]
        by_partition[partition] = by_partition.get(partition, 0) + manifest["entries"]

    # an unexpected packed shard means the derivative drifted from its plan
    expected_names = {(e["partition"], e["shard"] + ".gz") for e in plan["shards"]}
    unexpected = [
        f"{p}/{f.name}"
        for p in PARTITIONS
        for f in sorted((root / p).glob("*.jsonl.gz")) if (root / p).is_dir()
        if (p, f.name) not in expected_names
    ]
    if missing or stale or unexpected:
        return _status(subdir, "PARTIAL", missing=missing, stale=stale,
                       unexpected=unexpected, totals=totals)

    # exact reconciliation + zero duplicate row ids + zero cross-partition source/scaffold leakage
    seen_rows: set[int] = set()
    duplicate_rows = 0
    sources: dict[str, str] = {}
    scaffolds: dict[str, str] = {}
    leak_source, leak_scaffold = 0, 0
    path_lengths: dict[int, int] = {}
    for entry in plan["shards"]:
        partition, name = entry["partition"], entry["shard"] + ".gz"
        for trace, packed in read_packed_shard(root / partition / name):
            meta = trace.metadata or {}
            row_id = meta.get("row_id")
            if row_id is not None:
                if row_id in seen_rows:
                    duplicate_rows += 1
                seen_rows.add(row_id)
            source = meta.get("source_smiles")
            if source is not None:
                if sources.setdefault(source, partition) != partition:
                    leak_source += 1
            scaffold = meta.get("scaffold")
            if scaffold is not None:
                if scaffolds.setdefault(scaffold, partition) != partition:
                    leak_scaffold += 1
            path_lengths[packed.path_length] = path_lengths.get(packed.path_length, 0) + 1

    problems = []
    if totals["entries"] != plan["kept"]:
        problems.append(f"packed {totals['entries']} entries but partitioning kept {plan['kept']}")
    if plan["kept"] + sum(plan["rejects"].values()) != plan["scanned"]:
        problems.append("accepted + rejects does not reconcile to the pool record count")
    if duplicate_rows:
        problems.append(f"{duplicate_rows} duplicate row ids")
    if leak_source:
        problems.append(f"{leak_source} sources span partitions")
    if leak_scaffold:
        problems.append(f"{leak_scaffold} scaffolds span partitions")
    if problems:
        return _status(subdir, "FAILED", reason="; ".join(problems), totals=totals)

    audited = 0
    try:
        for entry in plan["shards"][: min(len(plan["shards"]), 8)]:
            for _ in read_packed_shard(root / entry["partition"] / (entry["shard"] + ".gz"),
                                       verify_fraction=_VERIFY_FRACTION):
                audited += 1
    except Exception as exc:  # noqa: BLE001
        return _status(subdir, "FAILED", reason=f"replay audit failed: {exc}", totals=totals)

    sentinels = []
    for partition in ("train", "validation"):
        shard = next((e for e in plan["shards"] if e["partition"] == partition), None)
        if shard:
            sentinels.append(sentinel_replay_check(
                root / partition / (shard["shard"] + ".gz"), entries=_SENTINEL_ENTRIES
            ))

    payload = {
        "MMP_PACK_COMPLETE": True,
        "layer": LAYER,
        "pool_sha256": plan["pool_sha256"],
        "pool_records": plan["scanned"],
        "packed_entries": totals["entries"],
        "packed_states": totals["states"],
        "rejects": plan["rejects"],
        "accepted_records": plan["accepted_records"],
        "endpoints_straddle_partitions": plan["endpoints_straddle_partitions"],
        "dropped_row_ids_sha256": plan["dropped_row_ids_sha256"],
        "dropped_row_ids_count": plan["dropped_row_ids_count"],
        "unique_molecules_by_partition": plan["unique_molecules_by_partition"],
        "partition_time_molecule_overlap": plan["molecule_overlap"],
        "partition_time_scaffold_overlap": plan["scaffold_overlap"],
        "dropped_bias": plan["dropped_bias"],
        "mmp_partition_rule_version": plan["mmp_partition_rule_version"],
        "pair_rule_implementation_hash": plan["pair_rule_implementation_hash"],
        "entries_by_partition": by_partition,
        "expected_shards": plan["expected_shards"],
        "unique_row_ids": len(seen_rows),
        "duplicate_row_ids": duplicate_rows,
        "cross_partition_source_leaks": leak_source,
        "cross_partition_scaffold_leaks": leak_scaffold,
        "path_length_histogram": {str(k): v for k, v in sorted(path_lengths.items())},
        "verify_fraction": _VERIFY_FRACTION,
        "audited_entries": audited,
        "sentinels": sentinels,
        "contract": contract,
        "sharding_arithmetic": plan["sharding_arithmetic"],
        **build_meta,
    }
    (root / "MMP_PACK_COMPLETE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    _status(subdir, "COMPLETE", packed_entries=totals["entries"],
            expected_shards=plan["expected_shards"])
    return payload


@app.function(image=image, cpu=2.0, timeout=12 * 3600, volumes={"/artifacts": artifact_volume})
def driver(subdir: str, commit: str, gate_sha: str) -> dict:
    """Remote orchestration so --detach survives a client disconnect."""
    artifact_volume.reload()
    build_meta = {"authorization": {"launch_commit": commit, "gate_sha256": gate_sha}}
    _status(subdir, "RUNNING", **build_meta)

    plan = partition_pool.remote(subdir)
    if plan.get("status") == "FAILED":
        return plan
    tasks = [
        (subdir, e["partition"], e["shard"], plan["pool_sha256"], e["row_ids_sha256"])
        for e in plan["shards"]
    ]
    print(json.dumps({"phase": "map_start", "tasks": len(tasks),
                      "max_containers": len(tasks) + _CONTAINER_MARGIN}), flush=True)
    results = list(pack_mmp_shard.starmap(tasks))
    print(json.dumps({"phase": "map_done", "shards": len(results),
                      "reused": sum(1 for r in results if r.get("reused"))}), flush=True)
    return reduce_mmp.remote(subdir, build_meta)


@app.local_entrypoint()
def main(subdir: str = "mmp_packed_v1", commit: str = "", gate_sha: str = ""):
    call = driver.spawn(subdir, commit, gate_sha)
    print(json.dumps({
        "phase": "launched", "driver_call_id": call.object_id, "subdir": subdir,
        "commit": commit, "gate_sha256": gate_sha,
        "poll": f"modal volume get compose-v4-artifacts {subdir}/mmp_pack_status.json -",
    }, indent=2, sort_keys=True))
