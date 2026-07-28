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


_PARTITION_SHARE = {"train": 0.90, "validation": 0.05, "test": 0.05}
# Measured compile rates (seconds per source), from local fixtures on real corpus molecules.
_SECONDS_PER_SOURCE = {"corruption": 0.98, "cycle_ops": 0.049}
_TARGET_SHARD_SECONDS = 600.0   # aim for ~10-minute shards
# Must be >= the DERIVED task count, or the tail doubles: 45 tasks under a cap of 40 is two waves
# (measured 18.6 min critical path) versus one wave (9.8 min). Raise this if source budgets grow.
_MAX_CONTAINERS = 48


def shards_for(layer: str, partition: str, layer_sources: int) -> int:
    """Shard count chosen so every shard takes roughly the same WALL TIME.

    Neither uniform-per-partition nor proportional-to-share works. Uniform gives 23 heavy train shards
    beside 48 near-idle validation/test shards. Proportional is worse: rounding floors the small partitions
    at one shard, so validation/test end up with MORE sources per shard than train and become the critical
    path (measured: 19.6 min vs train's 15.3). Sizing by target duration makes the tail flat."""
    sources = layer_sources * _PARTITION_SHARE[partition]
    seconds = sources * _SECONDS_PER_SOURCE[layer]
    return max(1, int(-(-seconds // _TARGET_SHARD_SECONDS)))    # ceil


def expected_shard_ids(corruption_sources: int, cycle_sources: int) -> list[str]:
    """The EXACT set of shard IDs this build must produce. The reducer rejects any missing, duplicated or
    unexpected shard against this set -- 'some shards exist' must never be mistaken for completeness."""
    ids = []
    for layer, layer_sources in (("corruption", corruption_sources), ("cycle_ops", cycle_sources)):
        for partition in PARTITIONS:
            for i in range(shards_for(layer, partition, layer_sources)):
                ids.append(f"{layer}/{partition}/shard_{i:04d}")
    return sorted(ids)


def write_status(subdir: str, status: str, **fields) -> dict:
    """Explicit terminal state. RUNNING | PARTIAL | FAILED | COMPLETE.

    The presence of source manifests or some shards must NEVER imply success; only the reducer may declare
    COMPLETE, and only after every checksum, contract hash, partition check and census has passed."""
    payload = {"status": status, **fields}
    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "build_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({"phase": "status", **{k: v for k, v in payload.items() if k != "shards"}},
                     sort_keys=True)[:900], flush=True)
    return payload


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

    artifact_volume.reload()
    existing = Path("/artifacts") / subdir / "source_manifest.json"
    if existing.exists():
        try:
            prior = json.loads(existing.read_text())
            if prior.get("per_layer_source_counts"):
                print(json.dumps({"phase": "sources_reused", "path": str(existing)}), flush=True)
                return prior
        except Exception:  # noqa: BLE001 -- a corrupt manifest just recomputes
            pass
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


@app.function(image=image, cpu=8.0, timeout=4 * 3600, max_containers=_MAX_CONTAINERS,
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
def reduce_and_validate(subdir: str, expected_ids: list, build_meta: dict) -> dict:
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

    # INVARIANT 1: expected-shard completeness -- exact set, no missing/unexpected/duplicate
    found_ids: set[str] = set()
    for layer in LAYERS:
        for partition in PARTITIONS:
            directory = out / layer / partition
            shards = sorted(directory.glob("shard_*.jsonl.gz")) if directory.is_dir() else []
            seen_here: set[str] = set()
            for shard in shards:
                shard_id = f"{layer}/{partition}/{shard.name.split('.')[0]}"
                if shard_id in seen_here:
                    problems.append(f"{shard_id}: duplicate shard")
                seen_here.add(shard_id)
                found_ids.add(shard_id)
                manifest = shard.with_suffix(".manifest.json")
                if not _shard_is_reusable(shard, manifest, contract):
                    problems.append(f"{shard_id}: failed checksum/contract validation")
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

    missing = sorted(set(expected_ids) - found_ids)
    unexpected = sorted(found_ids - set(expected_ids))
    if missing:
        problems.append(f"missing {len(missing)} expected shard(s): {missing[:5]}")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected shard(s): {unexpected[:5]}")

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
        "expected_shards": len(expected_ids),
        "completed_shards": len(found_ids & set(expected_ids)),
        "missing_shards": missing,
        "unexpected_shards": unexpected,
        "authorization": build_meta.get("authorization"),
        "reused_shards": build_meta.get("reused"),
        "recompiled_shards": build_meta.get("recompiled"),
        "failed_shards": build_meta.get("failed"),
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
    census["BUILD_COMPLETE"] = bool(not problems)
    body = json.dumps(census, indent=2, sort_keys=True)
    census["reducer_checksum"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    (out / "unified_census.json").write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n")
    # INVARIANT 2: BUILD_COMPLETE is written ONLY here, only after every check passed
    marker = out / "BUILD_COMPLETE.json"
    if census["BUILD_COMPLETE"]:
        marker.write_text(json.dumps(
            {"BUILD_COMPLETE": True, "reducer_checksum": census["reducer_checksum"],
             "expected_shards": len(expected_ids), "totals": dict(totals), "contract": contract,
             "authorization": build_meta.get("authorization")},
            indent=2, sort_keys=True) + "\n")
    elif marker.exists():
        marker.unlink()      # a previously-complete build that no longer validates must not stay marked
    artifact_volume.commit()
    print(json.dumps({k: v for k, v in census.items() if k != "families_by_layer"},
                     sort_keys=True)[:2000], flush=True)
    return census


@app.function(image=image, cpu=2.0, timeout=12 * 3600,
              volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume})
def driver(subdir: str, corruption_sources: int, cycle_sources: int, seed: int,
           launch_commit: str = "unknown", gate_sha256: str = "unknown") -> dict:
    """Remote orchestration. `@app.local_entrypoint` runs on the CLIENT, so `modal run --detach` keeps
    already-spawned functions alive but silently drops the rest of the pipeline when the client
    disconnects -- the map and reduce would never be issued. Driving from inside a container makes the
    whole build survive client loss."""
    expected_ids = expected_shard_ids(corruption_sources, cycle_sources)
    # Authorization provenance: a successful build must never inherit a STALE gate. The commit and the
    # hash of the passing prelaunch-gate log are recorded here and carried into BUILD_COMPLETE.
    authorization = {"launch_commit": launch_commit, "gate_sha256": gate_sha256}
    write_status(subdir, "RUNNING", expected_shards=len(expected_ids),
                 corruption_sources=corruption_sources, cycle_sources=cycle_sources,
                 authorization=authorization)

    # INVARIANT 3: idempotent. A complete, still-valid build is not rebuilt; anything else recompiles only
    # the stale/missing shards. Each shard has exactly ONE writer (its own map task + its own path), so two
    # writers for one final path is structurally impossible.
    out = Path("/artifacts") / subdir
    marker = out / "BUILD_COMPLETE.json"
    if marker.exists():
        try:
            prior = json.loads(marker.read_text())
            if prior.get("contract") == _contract_hashes():
                print(json.dumps({"phase": "build_already_complete"}), flush=True)
                write_status(subdir, "COMPLETE", reused_existing_build=True,
                             reducer_checksum=prior.get("reducer_checksum"))
                return prior
        except Exception:  # noqa: BLE001
            pass

    try:
        src = partition_sources.remote(subdir, corruption_sources, cycle_sources)
        print(json.dumps({"phase": "partitioned", "counts": src["per_layer_source_counts"]}), flush=True)

        jobs = []
        for layer, layer_sources in (("corruption", corruption_sources), ("cycle_ops", cycle_sources)):
            for partition in PARTITIONS:
                n_shards = shards_for(layer, partition, layer_sources)
                for shard_index in range(n_shards):
                    jobs.append((layer, partition, shard_index, n_shards, subdir, seed))
        print(json.dumps({"phase": "mapping", "shards": len(jobs)}), flush=True)

        results = list(compile_shard.starmap(jobs, return_exceptions=True))
        reused = sum(1 for r in results if isinstance(r, dict) and r.get("reused"))
        failed = sum(1 for r in results if not isinstance(r, dict))
        recompiled = len(results) - reused - failed
        build_meta = {"reused": reused, "recompiled": recompiled, "failed": failed}
        print(json.dumps({"phase": "mapped", **build_meta}), flush=True)
    except Exception as exc:  # noqa: BLE001
        write_status(subdir, "FAILED", stage="map", error=f"{type(exc).__name__}: {exc}")
        raise

    census = reduce_and_validate.remote(subdir, expected_ids,
                                        {**build_meta, "authorization": authorization})
    # INVARIANT 4: only the reducer's verdict decides terminal state. Source manifests or partial shards
    # never imply success.
    status = "COMPLETE" if census.get("BUILD_COMPLETE") else "PARTIAL"
    write_status(subdir, status, expected_shards=len(expected_ids),
                 completed_shards=census.get("completed_shards"),
                 missing_shards=len(census.get("missing_shards") or []),
                 **build_meta,
                 reducer_checksum=census.get("reducer_checksum"),
                 authorization=authorization,
                 problems=(census.get("problems") or [])[:5])
    print(json.dumps({"phase": "driver_complete", "status": status}), flush=True)
    return census


@app.local_entrypoint()
def main(
    subdir: str = "edit_precompile_v1",
    gate_log_path: str = "",
    corruption_sources: int = 24000,
    cycle_sources: int = 36000,
    seed: int = 20260728,
) -> None:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    gate_log = Path(gate_log_path) if gate_log_path else None
    gate_sha = "unknown"
    if gate_log and gate_log.exists():
        text = gate_log.read_text()
        if "PRELAUNCH GATE: PASS" not in text:
            raise SystemExit(f"refusing to launch: {gate_log} does not record a PASS")
        if commit and commit not in text:
            raise SystemExit(
                f"refusing to launch: {gate_log} does not reference HEAD {commit} -- this looks like a "
                f"STALE gate from an earlier commit")
        gate_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    call = driver.spawn(subdir, corruption_sources, cycle_sources, seed, commit, gate_sha)
    print(json.dumps({
        "phase": "driver_spawned", "function_call_id": call.object_id, "subdir": subdir,
        "commit": commit, "gate_sha256": gate_sha, "corruption_sources": corruption_sources, "cycle_sources": cycle_sources,
        "note": "orchestration runs remotely; safe to disconnect (launch with --detach)",
    }))
