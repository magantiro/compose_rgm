"""Modal: full-scale broad-organic edit-trace mining via a map -> global-group -> PARALLEL-compile -> reduce
fan-out.

The global grouping over the full GuacaMol TRAIN partition yields hundreds of thousands of MMP one-cut
pairs; compiling them serially in one container would take hours. This app keeps the full pool and
parallelizes the compile across containers:

  0. storage_preflight (ONE tiny container, V2 only): exercise create, identical reuse, and
                  different-content refusal with the exact immutable-write primitive on the mounted
                  artifact volume, in a separate content-addressed scratch namespace.
  1. mine_pairs   (ONE container): scan+split the corpus, map the full TRAIN partition, GLOBAL-group the MMP
                  pairs, characterize corruption; write pairs.jsonl + mine_meta.json to the artifact volume.
  2. compile_shard (N logical tasks, at most 5 containers concurrently on Volume v1): each reads a
                  deterministic stride of pairs.jsonl, compiles + executor-replays them (both directions),
                  writes compiled/shard_XXXX.jsonl.
  3. reduce_pool  (ONE container): read every compiled shard, global dedup + cap, write the final pool +
                  summary (census + corruption + provenance + mixture stats). V2 refuses missing,
                  overlapping or stale shards and sorts by the original global pair index before caps.
  4. V2 freeze   (inside the reducer, only with --publish-v2): copy the validated pool into an immutable
                  content-addressed namespace and publish MMP_POOL_COMPLETE.json for the packer.

DATA MINING + COMPILATION ONLY -- never loads a checkpoint, never trains. Launch from a clean committed
worktree at the launch tag (the prelaunch gate is the guard).

    modal run modal_apps/mine_edit_traces_app.py \
      --n-compile-shards 20 --publish-v2 \
      --expected-corpus-sha256 <full-lowercase-sha256>

The app module is named *_app to avoid colliding with the mine_edit_traces scripts module it imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
if REMOTE_ROOT.is_dir():  # inside the Modal container; the thin local launcher never imports the miner
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

app = modal.App("compose-v4-edit-trace-mining")
guacamol_volume = modal.Volume.from_name("guacamol", create_if_missing=False)
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)
_LEGACY_DEFAULT_SUBDIR = "edit_mining_full_broad"
_V2_RUN_PREFIX = "edit_mining_v2_run_"
MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5


def v2_run_subdir(
    config_dict: dict,
    *,
    source_commit: str,
    expected_corpus_sha256: str,
    n_compile_shards: int,
) -> str:
    """Execution namespace; frozen pool publication gets a second content-addressed identity."""
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("V2 mining requires the full lowercase 40-character source commit")
    if (
        len(expected_corpus_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_corpus_sha256)
    ):
        raise ValueError("V2 mining requires the caller's full lowercase corpus SHA-256")
    if n_compile_shards <= 0:
        raise ValueError("n_compile_shards must be positive")
    payload = {
        "schema": "compose.mmp_mining_run.v2",
        "config": {key: value for key, value in config_dict.items() if key != "out_dir"},
        "source_commit": source_commit,
        "expected_corpus_sha256": expected_corpus_sha256,
        "n_compile_shards": n_compile_shards,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{_V2_RUN_PREFIX}{identity[:20]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=1.0,
    timeout=5 * 60,
    volumes={"/artifacts": artifact_volume},
)
def storage_preflight() -> dict:
    """Fail before corpus scanning if the mounted Volume cannot publish immutable bytes."""

    from compose_v4.data.mmp_pool_freezer import run_write_once_storage_preflight

    artifact_volume.reload()
    result = run_write_once_storage_preflight("/artifacts")
    artifact_volume.commit()
    return {
        **result,
        "modal_volume_contract": {
            "generation": "v1",
            "max_concurrent_small_commit_writers": (
                MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS
            ),
            "logical_compile_shards_are_queued": True,
        },
    }


@app.function(image=image, cpu=16.0, timeout=3 * 3600,
              volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume})
def mine_pairs(config_dict: dict, source_commit: str, subdir: str, publish_v2: bool) -> dict:
    """Phase 1: scan+split + map(full train) + global-group MMP pairs + corruption char. Write pairs.jsonl
    + mine_meta.json to the volume; return the metadata (census/corruption/provenance/timing + n_pairs)."""
    from compose_v4.data.mmp_pool_freezer import (
        canonical_sha256,
        partition_independent_mining_config,
        write_bytes_if_absent,
    )
    from mine_edit_traces import MiningConfig, compiler_identity_v2, mine_mmp_pairs

    guacamol_volume.reload()
    config = MiningConfig(**config_dict)
    corpus_file = Path(config.corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"corpus {corpus_file} absent; /guacamol has "
                                f"{sorted(p.name for p in Path('/guacamol').iterdir())}")
    corpus_sha = _file_sha256(corpus_file)
    if config.corpus_sha256 and config.corpus_sha256 != corpus_sha:
        raise ValueError(
            f"corpus SHA-256 mismatch: expected {config.corpus_sha256}, observed {corpus_sha}"
        )
    if publish_v2 and not config.corpus_sha256:
        raise ValueError("V2 publication requires a caller-frozen corpus SHA-256")
    config = replace(config, corpus_sha256=corpus_sha)

    out = Path("/artifacts") / subdir
    pairs_path = out / "pairs.jsonl"
    meta_path = out / "mine_meta.json"
    if (
        publish_v2
        and out.is_dir()
        and any(out.iterdir())
        and not (pairs_path.exists() or meta_path.exists())
    ):
        raise ValueError("V2 mining namespace contains unclaimed artifacts; refusing adoption")
    if publish_v2 and (pairs_path.exists() or meta_path.exists()):
        if not (pairs_path.is_file() and meta_path.is_file()):
            raise ValueError("V2 pair-mining stage is partial; refusing adoption or overwrite")
        prior = json.loads(meta_path.read_text())
        scientific_config = partition_independent_mining_config(asdict(config))
        expected_pairs = prior.get("pairs", {})
        if (
            prior.get("config") != asdict(config)
            or prior.get("partition_independent_mining_config") != scientific_config
            or prior.get("mining_config_sha256") != canonical_sha256(scientific_config)
            or prior.get("provenance", {}).get("commit") != source_commit
            or expected_pairs.get("path") != f"/artifacts/{subdir}/pairs.jsonl"
            or expected_pairs.get("sha256") != _file_sha256(pairs_path)
            or expected_pairs.get("records")
            != sum(1 for line in pairs_path.open("rb") if line.strip())
        ):
            raise ValueError("existing V2 pair-mining identity disagrees; refusing re-blessing")
        if prior.get("provenance", {}).get("compiler_identity_v2") != compiler_identity_v2():
            raise ValueError("existing V2 pair-mining compiler identity is stale")
        print(json.dumps({
            "phase": "pair_mining_reuse",
            "pairs": expected_pairs["records"],
            "subdir": subdir,
        }, sort_keys=True), flush=True)
        return prior

    result = mine_mmp_pairs(config)
    result["provenance"]["commit"] = source_commit
    result["provenance"]["corpus_sha256"] = corpus_sha

    out.mkdir(parents=True, exist_ok=True)
    pairs = result.pop("pairs")
    pairs_bytes = b"".join(
        json.dumps(pair, separators=(",", ":")).encode() + b"\n"
        for pair in pairs
    )
    if publish_v2:
        write_bytes_if_absent(pairs_path, pairs_bytes)
    else:
        pairs_path.write_bytes(pairs_bytes)
    result["n_pairs"] = len(pairs)
    result["pairs"] = {
        "path": f"/artifacts/{subdir}/pairs.jsonl",
        "sha256": hashlib.sha256(pairs_bytes).hexdigest(),
        "records": len(pairs),
    }
    scientific_config = partition_independent_mining_config(result["config"])
    result["partition_independent_mining_config"] = scientific_config
    result["mining_config_sha256"] = canonical_sha256(scientific_config)
    meta_bytes = json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    if publish_v2:
        write_bytes_if_absent(out / "mine_meta.json", meta_bytes)
    else:
        (out / "mine_meta.json").write_bytes(meta_bytes)
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=8.0,
    timeout=3 * 3600,
    max_containers=MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
    volumes={"/artifacts": artifact_volume},
)
def compile_shard(
    shard_index: int,
    n_shards: int,
    config_dict: dict,
    subdir: str,
    publish_v2: bool,
) -> dict:
    """Phase 2 (parallel): compile the shard_index-th deterministic stride of pairs.jsonl, both directions,
    and write compiled/shard_XXXX.jsonl. Returns the record count."""
    from compose_v4.data.mmp_pool_freezer import (
        build_compile_shard_manifest,
        canonical_sha256,
        compile_manifest_artifact_path,
        compile_manifest_file_path,
        expected_pair_indices,
        partition_independent_mining_config,
        write_bytes_if_absent,
    )
    from mine_edit_traces import MiningConfig, compile_mmp, compiler_identity_v2

    artifact_volume.reload()
    config = MiningConfig(**config_dict)
    out = Path("/artifacts") / subdir
    pairs_path = out / "pairs.jsonl"
    with pairs_path.open() as handle:
        all_pairs = [json.loads(line) for line in handle if line.strip()]
    global_indices = list(expected_pair_indices(len(all_pairs), n_shards, shard_index))
    mine = [tuple(all_pairs[index]) for index in global_indices]
    cdir = out / "compiled"
    cdir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_index:04d}.jsonl"
    shard = cdir / name
    manifest_path = compile_manifest_file_path(shard)
    artifact_path = f"/artifacts/{subdir}/compiled/{name}"
    scientific_config = partition_independent_mining_config(config_dict)
    compiler_identity = compiler_identity_v2()
    pairs_sha256 = _file_sha256(pairs_path)
    if publish_v2 and (shard.exists() or manifest_path.exists()):
        if not (shard.is_file() and manifest_path.is_file()):
            raise ValueError(f"V2 compile shard {shard_index} is partial; refusing overwrite")
        expected_manifest = build_compile_shard_manifest(
            shard_path=shard,
            shard_artifact_path=artifact_path,
            shard_index=shard_index,
            n_shards=n_shards,
            total_pairs=len(all_pairs),
            source_pairs_path=f"/artifacts/{subdir}/pairs.jsonl",
            source_pairs_sha256=pairs_sha256,
            mining_config_sha256=canonical_sha256(scientific_config),
            compiler_identity=compiler_identity,
        )
        observed_manifest = json.loads(manifest_path.read_text())
        if observed_manifest != expected_manifest:
            raise ValueError(
                f"V2 compile shard {shard_index} disagrees with its frozen inputs"
            )
        manifest_bytes = manifest_path.read_bytes()
        print(json.dumps({
            "phase": "compile_shard_reuse",
            "shard_index": shard_index,
            "records": expected_manifest["records"],
        }, sort_keys=True), flush=True)
        return {
            "shard_index": shard_index,
            "pairs": expected_manifest["assigned_pairs"]["count"],
            "records": expected_manifest["records"],
            "shard_path": artifact_path,
            "manifest_path": compile_manifest_artifact_path(artifact_path),
            "content_sha256": expected_manifest["content_sha256"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }

    records = compile_mmp(mine, config, pair_indices=global_indices)
    content = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )
    if publish_v2:
        write_bytes_if_absent(shard, content)
    else:
        shard.write_bytes(content)
    # Build after the immutable bytes exist so the manifest always authenticates the actual shard.
    manifest = build_compile_shard_manifest(
        shard_path=shard,
        shard_artifact_path=artifact_path,
        shard_index=shard_index,
        n_shards=n_shards,
        total_pairs=len(all_pairs),
        source_pairs_path=f"/artifacts/{subdir}/pairs.jsonl",
        source_pairs_sha256=pairs_sha256,
        mining_config_sha256=canonical_sha256(scientific_config),
        compiler_identity=compiler_identity,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    if publish_v2:
        write_bytes_if_absent(manifest_path, manifest_bytes)
    else:
        manifest_path.write_bytes(manifest_bytes)
    artifact_volume.commit()
    print(f"[compile] shard {shard_index}/{n_shards}: {len(mine)} pairs -> {len(records)} records",
          flush=True)
    return {
        "shard_index": shard_index,
        "pairs": len(mine),
        "records": len(records),
        "shard_path": artifact_path,
        "manifest_path": compile_manifest_artifact_path(artifact_path),
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


@app.function(image=image, cpu=16.0, timeout=3 * 3600,
              volumes={"/artifacts": artifact_volume})
def reduce_pool(
    config_dict: dict,
    subdir: str,
    meta: dict,
    n_compile_shards: int,
    publish_v2: bool,
) -> dict:
    """Phase 3: require exact shard coverage, globally order, reduce, and optionally freeze V2."""
    from compose_v4.data.mmp_pool_freezer import (
        REDUCTION_SCHEMA,
        canonical_sha256,
        file_sha256,
        freeze_reduced_mmp_pool,
        partition_independent_mining_config,
        validate_compile_inventory,
        validate_compiler_identity,
        write_bytes_if_absent,
    )
    from mine_edit_traces import (
        MiningConfig,
        compiler_identity_v2,
        dedup_and_cap,
    )

    artifact_volume.reload()
    config = MiningConfig(**config_dict)
    out = Path("/artifacts") / subdir
    pairs_path = out / "pairs.jsonl"
    mine_meta_path = out / "mine_meta.json"
    pairs_sha256 = file_sha256(pairs_path)
    pair_count = sum(1 for line in pairs_path.open("rb") if line.strip())
    if meta.get("pairs") != {
        "path": f"/artifacts/{subdir}/pairs.jsonl",
        "sha256": pairs_sha256,
        "records": pair_count,
    }:
        raise ValueError("mine metadata disagrees with pairs bytes/count")
    scientific_config = partition_independent_mining_config(config_dict)
    mining_config_sha256 = canonical_sha256(scientific_config)
    if (
        meta.get("partition_independent_mining_config") != scientific_config
        or meta.get("mining_config_sha256") != mining_config_sha256
    ):
        raise ValueError("mine metadata disagrees with partition-independent configuration")
    compiler_identity = validate_compiler_identity(
        meta.get("provenance", {}).get("compiler_identity_v2", {})
    )
    if compiler_identity != compiler_identity_v2():
        raise ValueError("compiler implementation changed between pair mining and reduction")
    inventory = validate_compile_inventory(
        artifact_root="/artifacts",
        source_run_root=f"/artifacts/{subdir}",
        expected_shards=n_compile_shards,
        total_pairs=pair_count,
        source_pairs_path=f"/artifacts/{subdir}/pairs.jsonl",
        source_pairs_sha256=pairs_sha256,
        mining_config_sha256=mining_config_sha256,
        compiler_identity=compiler_identity,
    )
    records = list(inventory.rows)
    pool, cap_report = dedup_and_cap(records, config)

    pool_bytes = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in pool
    )
    pool_path = out / "edit_pool_full.jsonl"
    if publish_v2:
        write_bytes_if_absent(pool_path, pool_bytes)
    else:
        pool_path.write_bytes(pool_bytes)
    summary_artifact_path = f"/artifacts/{subdir}/mining_summary_full.json"
    corpus_provenance = meta.get("provenance", {})
    source_inputs = {
        "corpus": {
            "id": str(corpus_provenance.get("corpus_id", "")),
            "path": str(config_dict.get("corpus_path", "")),
            "sha256": str(corpus_provenance.get("corpus_sha256", "")),
            "scope_name": str(corpus_provenance.get("corpus_scope", "")),
            "scope_hash": str(corpus_provenance.get("corpus_scope_hash", "")),
        },
        "pairs": dict(meta["pairs"]),
        "mine_meta": {
            "path": f"/artifacts/{subdir}/mine_meta.json",
            "sha256": file_sha256(mine_meta_path),
        },
    }
    summary = {
        "schema": REDUCTION_SCHEMA,
        "REDUCTION_COMPLETE": True,
        "source_run_root": f"/artifacts/{subdir}",
        "summary_artifact_path": summary_artifact_path,
        "config": meta.get("config"),
        "partition_independent_mining_config": scientific_config,
        "mining_config_sha256": mining_config_sha256,
        "provenance": meta.get("provenance"),
        "compiler_identity": compiler_identity,
        "source_inputs": source_inputs,
        "corpus": meta.get("corpus"),
        "corruption": meta.get("corruption"),
        "timing_seconds": meta.get("timing_seconds"),
        "mmp_pairs_grouped": meta.get("n_pairs"),
        "records_compiled_pre_dedup": len(records),
        "compile": {
            "expected_shards": n_compile_shards,
            "records": inventory.record_count,
            "records_sha256": inventory.records_sha256,
            "shards": list(inventory.shards),
            "coverage": {
                "rule": "global_pair_index_mod_n_shards",
                "total_pairs": pair_count,
                "unique_assigned_pairs": pair_count,
                "missing_pairs": 0,
                "overlapping_pairs": 0,
            },
        },
        "global_reduce": cap_report,
        "pool_size": len(pool),
        "pool": {
            "path": f"/artifacts/{subdir}/edit_pool_full.jsonl",
            "sha256": hashlib.sha256(pool_bytes).hexdigest(),
            "records": len(pool),
        },
    }
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
    summary_path = out / "mining_summary_full.json"
    if publish_v2:
        write_bytes_if_absent(summary_path, summary_bytes)
    else:
        summary_path.write_bytes(summary_bytes)
    frozen = None
    if publish_v2:
        frozen = freeze_reduced_mmp_pool(
            summary_path,
            artifact_root="/artifacts",
        )
    artifact_volume.commit()
    return {
        **summary,
        "frozen_pool": (
            None if frozen is None else {
                key: frozen[key]
                for key in (
                    "freeze_identity",
                    "namespace",
                    "pool_path",
                    "pool_sha256",
                    "pool_records",
                    "contract_path",
                    "contract_sha256",
                )
            }
        ),
    }


@app.local_entrypoint()
def main(
    corpus: str = "guacamol_subset_500000_seed0.smiles",
    train_size: int = 400_000,
    validation_size: int = 10_000,
    test_size: int = 10_000,
    max_atoms: int = 40,
    split_workers: int = 12,
    corruption_sample_size: int = 500,
    n_compile_shards: int = 20,
    subdir: str = _LEGACY_DEFAULT_SUBDIR,
    expected_corpus_sha256: str = "",
    publish_v2: bool = False,
) -> None:
    # Thin launcher: no miner import (rdkit/etc. live only in the container). n_shards=1 => the miner maps
    # the FULL train partition (global grouping); the compile is fanned out over n_compile_shards.
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    code_dirty = [ln for ln in dirty.splitlines()
                  if ln[3:].startswith(("src/", "scripts/", "modal_apps/", "tests/"))]
    if code_dirty:
        raise SystemExit(f"REFUSING to launch from a dirty code tree; run the prelaunch gate. {code_dirty}")

    config_dict = {
        "corpus_id": Path(corpus).stem,
        "corpus_path": f"/guacamol/{corpus}",
        "corpus_sha256": expected_corpus_sha256 or None,
        "train_size": train_size, "validation_size": validation_size, "test_size": test_size,
        "max_atoms": max_atoms, "n_shards": 1, "shard_index": 0,
        "split_workers": split_workers, "scaffold_k": 0,
        "corruption_sample_size": corruption_sample_size,
    }
    if publish_v2:
        if subdir != _LEGACY_DEFAULT_SUBDIR:
            raise SystemExit("V2 publication computes its own run namespace; do not pass --subdir")
        try:
            subdir = v2_run_subdir(
                config_dict,
                source_commit=commit,
                expected_corpus_sha256=expected_corpus_sha256,
                n_compile_shards=n_compile_shards,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    config_dict["out_dir"] = f"/artifacts/{subdir}"
    if publish_v2:
        print("[0/3] validating immutable writes on the artifact Volume ...")
        storage = storage_preflight.remote()
        if storage.get("status") != "PASS":
            raise SystemExit(f"artifact Volume storage preflight failed: {storage}")
        volume_contract = storage.get("modal_volume_contract") or {}
        if (
            volume_contract.get("generation") != "v1"
            or volume_contract.get("max_concurrent_small_commit_writers")
            != MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS
        ):
            raise SystemExit(
                f"artifact Volume concurrency contract is absent or stale: {storage}"
            )
        print(
            "      -> PASS "
            f"(identity {storage['preflight_identity'][:16]}, "
            f"created={storage['probe_created_this_invocation']}, "
            f"max concurrent writers={MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS})"
        )
    print(f"[1/3] mining pairs from {corpus} at commit {commit} ...")
    meta = mine_pairs.remote(config_dict, commit, subdir, publish_v2)
    n_pairs = meta["n_pairs"]
    print(f"      -> {n_pairs} MMP pairs; corpus retained {meta['corpus']['census']['retained']}")
    resolved_config = meta.get("config")
    if not isinstance(resolved_config, dict):
        raise SystemExit("pair-mining metadata did not return the fully resolved MiningConfig")

    print(f"[2/3] compiling over {n_compile_shards} parallel containers ...")
    compile_results = list(compile_shard.starmap(
        [
            (i, n_compile_shards, resolved_config, subdir, publish_v2)
            for i in range(n_compile_shards)
        ]
    ))
    print(f"      -> {sum(result['records'] for result in compile_results)} records "
          f"across {n_compile_shards} shards")

    print("[3/3] reducing ...")
    summary = reduce_pool.remote(
        resolved_config,
        subdir,
        meta,
        n_compile_shards,
        publish_v2,
    )
    print(json.dumps({"pool_size": summary["pool_size"],
                      "mmp_pairs_grouped": summary["mmp_pairs_grouped"],
                      "records_compiled_pre_dedup": summary["records_compiled_pre_dedup"],
                      "global_reduce": summary["global_reduce"],
                      "timing_seconds": summary["timing_seconds"],
                      "frozen_pool": summary["frozen_pool"]}, indent=2))
    print(f"\nFULL MINING COMPLETE -> /artifacts/{subdir}/edit_pool_full.jsonl "
          f"+ mining_summary_full.json")
