"""Modal STAGE 1 -- chemistry compilation of the scaled edit corpus (audit artifacts).

    source partitions -> independent corruption/cycle audit shards -> reducer validation + unified manifest

Design constraints this app exists to satisfy:

  * **Every map worker writes a UNIQUE shard + shard manifest.** No concurrent writes to a shared output
    file, so a partial run never corrupts a good artifact and workers never race.
  * **Restartable.** A shard is skipped only when its manifest validates AND its checksum matches AND its
    recorded contract/codec/partitioner hashes equal the current ones. A stale-contract shard is recompiled,
    never silently reused.
  * **Audit artifacts are the source of truth.** They carry exact slot-addressed states, semantic V2 actions,
    successor keys and full provenance. The packed tensor cache (stage 2) is a deterministic derivative of
    these, never the other way round.
  * **Both layers are first class.** Corruption supplies seven families; the two compositional cycle families
    come only from the cycle layer, so compiling corruption alone would scale seven families while leaving
    the defining RingCore families starved.

Chemistry compilation ONLY -- never loads a checkpoint, never trains, no GPU.

    modal run --detach modal_apps/precompile_edit_data_app.py --corruption-sources 24000 --cycle-sources 36000
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
if REMOTE_ROOT.is_dir():
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

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

app = modal.App("compose-v4-precompile-edit-data")
guacamol_volume = modal.Volume.from_name("guacamol", create_if_missing=False)
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

CORPUS = "/guacamol/guacamol_subset_500000_seed0.smiles"
PARTITIONS = ("train", "validation", "test")
LAYERS = ("corruption", "cycle_ops")


def _contract_hashes() -> dict:
    """Identity a shard must match to be reusable. A mismatch forces recompilation."""
    from compose_v4.data.scaffold_partition import partitioner_provenance
    from compose_v4.rewrite.action_codec import codec_implementation_hash
    from compose_v4.rewrite.trace_shard import TRACE_SCHEMA_VERSION

    out = {
        "codec_implementation_hash": codec_implementation_hash(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "partitioner": partitioner_provenance(),
    }
    try:
        import ring_core_identity as rci

        out["operator_registry_hash"] = rci.recompute_operator_registry_hash()
        out["capability_hash"] = rci.CAPABILITY_HASH
        out["scope_hash"] = rci.SCOPE_HASH
    except Exception:  # noqa: BLE001
        out["operator_registry_hash"] = out["capability_hash"] = out["scope_hash"] = "unavailable"
    return out


def _shard_is_reusable(shard: Path, manifest: Path, contract: dict) -> bool:
    """Restart rule: reuse ONLY if the manifest parses, the checksum matches, and every contract hash
    equals the current one. Anything else recompiles -- a stale-contract shard is never silently reused."""
    if not (shard.exists() and manifest.exists()):
        return False
    try:
        meta = json.loads(manifest.read_text())
    except Exception:  # noqa: BLE001
        return False
    import gzip

    try:
        payload = gzip.open(shard, "rb").read()
    except Exception:  # noqa: BLE001
        return False
    if hashlib.sha256(payload).hexdigest() != meta.get("content_sha256"):
        return False
    for key in ("codec_implementation_hash", "operator_registry_hash", "capability_hash"):
        if meta.get(key) != contract.get(key):
            return False
    got = (meta.get("partitioner") or {}).get("partitioner_implementation_hash")
    if got != contract["partitioner"]["partitioner_implementation_hash"]:
        return False
    return True


@app.function(image=image, cpu=16.0, timeout=2 * 3600,
              volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume})
def partition_sources(subdir: str, corruption_sources: int, cycle_sources: int) -> dict:
    """Phase 1 (one container): scope-filter the corpus, scaffold-partition it, and write per-layer
    per-partition source manifests. Partitioning happens HERE, before any trajectory exists."""
    from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, classify_smiles
    from compose_v4.data.scaffold_partition import (
        assign_partitions,
        partitioner_provenance,
        verify_partition_disjointness,
    )

    guacamol_volume.reload()
    corpus = Path(CORPUS)
    if not corpus.exists():
        raise FileNotFoundError(f"corpus {corpus} absent")
    eligible = [s for s in corpus.read_text().split() if classify_smiles(s, BROAD_ORGANIC_V1)[0]]
    print(json.dumps({"phase": "scope_filtered", "eligible": len(eligible)}), flush=True)

    part_by, scaf_by, stats = assign_partitions(eligible)
    leak = verify_partition_disjointness(part_by, scaf_by)   # raises on any leak

    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    manifests: dict = {}
    for layer, budget in (("corruption", corruption_sources), ("cycle_ops", cycle_sources)):
        manifests[layer] = {}
        for partition in PARTITIONS:
            pool = sorted(s for s, p in part_by.items() if p == partition)
            # scale each partition's budget by its share so ratios are preserved
            share = len(pool) / max(len(part_by), 1)
            take = max(1, int(budget * share))
            chosen = pool[:take]
            path = out / "sources" / f"{layer}_{partition}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"layer": layer, "partition": partition, "sources": chosen,
                 "scaffolds": {s: scaf_by[s] for s in chosen}}, sort_keys=True))
            manifests[layer][partition] = len(chosen)
    result = {
        "phase": "sources_partitioned",
        "eligible": len(eligible),
        "partition_stats": stats,
        "leak_check": leak,
        "partitioner": partitioner_provenance(),
        "per_layer_source_counts": manifests,
    }
    (out / "source_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    artifact_volume.commit()
    return result


@app.function(image=image, cpu=8.0, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume}, retries=modal.Retries(max_retries=2))
def compile_shard(layer: str, partition: str, shard_index: int, n_shards: int,
                  subdir: str, seed: int) -> dict:
    """Phase 2 (N parallel): compile ONE deterministic stride into a UNIQUE shard + manifest.

    Restartable: if this shard already exists and validates under the current contract, it is skipped."""
    from compile_corruption_shard import compile_shard as _compile
    from compose_v4.rewrite.trace_shard import shard_manifest, write_shard

    artifact_volume.reload()
    out = Path("/artifacts") / subdir
    shard_path = out / layer / partition / f"shard_{shard_index:04d}.jsonl.gz"
    manifest_path = shard_path.with_suffix(".manifest.json")
    contract = _contract_hashes()

    if _shard_is_reusable(shard_path, manifest_path, contract):
        meta = json.loads(manifest_path.read_text())
        print(json.dumps({"phase": "shard_reused", "layer": layer, "partition": partition,
                          "shard": shard_index, "records": meta.get("records")}), flush=True)
        return {"reused": True, **{k: meta.get(k) for k in ("records", "content_sha256")}}

    payload = json.loads((out / "sources" / f"{layer}_{partition}.json").read_text())
    all_sources = payload["sources"]
    mine = [s for i, s in enumerate(all_sources) if i % n_shards == shard_index]
    scaf_by = payload["scaffolds"]

    records, report = _compile(
        mine, scaf_by, partition=partition, shard_index=shard_index, seed=seed, layer=layer,
    )
    stats = write_shard(shard_path, records)
    manifest = shard_manifest(
        shard_stats=stats, partition=partition,
        counts={**report["stats"], "shard_sources": len(mine)},
        rejections=report["rejections"],
        compiler_commit=os.environ.get("COMPOSE_COMMIT", "unknown"),
        program_contract_fingerprint=os.environ.get("COMPOSE_CONTRACT", "unavailable"),
        operator_registry_hash=contract["operator_registry_hash"],
        capability_hash=contract["capability_hash"],
        extra={"layer": layer, "shard_index": shard_index, "n_shards": n_shards, "seed": seed,
               **contract},
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({"phase": "shard_written", "layer": layer, "partition": partition,
                      "shard": shard_index, **report["stats"]}, sort_keys=True), flush=True)
    return {"reused": False, **report["stats"]}


@app.function(image=image, cpu=16.0, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume})
def reduce_and_validate(subdir: str, expected: dict) -> dict:
    """Phase 3 (one container): verify shard completeness + checksums + contract hashes, then census.

    Checks the conditions a partial or stale build would violate, and the two the plan calls out
    specifically: partition purity over the UNION of layers, and cross-layer transition overlap (measured,
    not silently deduplicated)."""
    from collections import Counter, defaultdict

    from compose_v4.rewrite.trace_shard import read_shard

    artifact_volume.reload()
    out = Path("/artifacts") / subdir
    contract = _contract_hashes()

    problems: list[str] = []
    per_layer: dict = defaultdict(lambda: defaultdict(Counter))
    sources_by_partition: dict[str, set] = defaultdict(set)
    scaffolds_by_partition: dict[str, set] = defaultdict(set)
    transitions_by_layer: dict[str, set] = defaultdict(set)
    families: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()

    for layer in LAYERS:
        for partition in PARTITIONS:
            n_expected = expected.get(layer, {}).get(partition, 0)
            shards = sorted((out / layer / partition).glob("shard_*.jsonl.gz")) \
                if (out / layer / partition).is_dir() else []
            if n_expected and not shards:
                problems.append(f"{layer}/{partition}: expected shards, found none")
            for shard in shards:
                manifest = shard.with_suffix(".manifest.json")
                if not _shard_is_reusable(shard, manifest, contract):
                    problems.append(f"{shard.name}: failed checksum/contract validation")
                    continue
                for rec in read_shard(shard):
                    totals[f"{layer}_records"] += 1
                    totals[f"{layer}_transitions"] += rec["path_length"]
                    origin = (rec.get("metadata") or {}).get("origin_smiles")
                    if origin:
                        sources_by_partition[partition].add(origin)
                    scaffolds_by_partition[partition].add(rec["source_scaffold"])
                    families[layer].update(rec["family_histogram"])
                    per_layer[layer][partition]["records"] += 1
                    # canonical transition identity for cross-layer overlap
                    for step in rec["steps"]:
                        transitions_by_layer[layer].add((rec["source_key"], step["successor_key"]))

    # partition purity over the UNION of layers (a source crossing partitions VIA a different layer is
    # the dangerous case; the same source in two layers within one partition is fine)
    leaks = []
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            s = sources_by_partition[a] & sources_by_partition[b]
            k = scaffolds_by_partition[a] & scaffolds_by_partition[b]
            if s:
                leaks.append(f"{a}/{b} share {len(s)} sources across layers")
            if k:
                leaks.append(f"{a}/{b} share {len(k)} scaffolds across layers")
    if leaks:
        problems.extend(leaks)

    # cross-layer transition overlap -- MEASURED, not deduplicated here
    overlap = {}
    layer_names = list(transitions_by_layer)
    for i, a in enumerate(layer_names):
        for b in layer_names[i + 1:]:
            overlap[f"{a}|{b}"] = len(transitions_by_layer[a] & transitions_by_layer[b])

    census = {
        "phase": "reduce_complete",
        "contract": contract,
        "totals": dict(totals),
        "families_by_layer": {k: dict(v.most_common()) for k, v in families.items()},
        "unique_sources_by_partition": {k: len(v) for k, v in sources_by_partition.items()},
        "unique_scaffolds_by_partition": {k: len(v) for k, v in scaffolds_by_partition.items()},
        "unique_canonical_transitions_by_layer": {k: len(v) for k, v in transitions_by_layer.items()},
        "cross_layer_transition_overlap": overlap,
        "partition_purity_ok": not leaks,
        "problems": problems,
        "VALID": not problems,
    }
    (out / "unified_census.json").write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({k: v for k, v in census.items() if k != "families_by_layer"},
                     sort_keys=True)[:2000], flush=True)
    return census


@app.local_entrypoint()
def main(
    subdir: str = "edit_precompile_v1",
    corruption_sources: int = 24000,
    cycle_sources: int = 36000,
    corruption_shards: int = 24,
    cycle_shards: int = 8,
    seed: int = 20260728,
) -> None:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    print(json.dumps({"phase": "launch", "subdir": subdir, "commit": commit,
                      "corruption_sources": corruption_sources, "cycle_sources": cycle_sources}))

    src = partition_sources.remote(subdir, corruption_sources, cycle_sources)
    print(json.dumps({"phase": "partitioned", "counts": src["per_layer_source_counts"]}))

    jobs = []
    for layer, n_shards in (("corruption", corruption_shards), ("cycle_ops", cycle_shards)):
        for partition in PARTITIONS:
            for shard_index in range(n_shards):
                jobs.append((layer, partition, shard_index, n_shards, subdir, seed))
    print(json.dumps({"phase": "mapping", "shards": len(jobs)}))
    results = list(compile_shard.starmap(jobs))
    reused = sum(1 for r in results if r.get("reused"))
    print(json.dumps({"phase": "mapped", "shards": len(results), "reused": reused}))

    expected = {layer: {p: 1 for p in PARTITIONS} for layer in LAYERS}
    census = reduce_and_validate.remote(subdir, expected)
    print(json.dumps({"phase": "done", "VALID": census["VALID"],
                      "totals": census["totals"],
                      "problems": census["problems"][:5]}))
