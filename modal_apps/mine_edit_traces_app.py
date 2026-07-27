"""Modal: full-scale broad-organic edit-trace mining via a map -> global-group -> PARALLEL-compile -> reduce
fan-out.

The global grouping over the full GuacaMol TRAIN partition yields hundreds of thousands of MMP one-cut
pairs; compiling them serially in one container would take hours. This app keeps the full pool and
parallelizes the compile across containers:

  1. mine_pairs   (ONE container): scan+split the corpus, map the full TRAIN partition, GLOBAL-group the MMP
                  pairs, characterize corruption; write pairs.jsonl + mine_meta.json to the artifact volume.
  2. compile_shard (N containers, parallel): each reads a deterministic stride of pairs.jsonl, compiles +
                  executor-replays them (both directions), writes compiled/shard_XXXX.jsonl.
  3. reduce_pool  (ONE container): read every compiled shard, global dedup + cap, write the final pool +
                  summary (census + corruption + provenance + mixture stats).

DATA MINING + COMPILATION ONLY -- never loads a checkpoint, never trains. Launch from a clean committed
worktree at the launch tag (the prelaunch gate is the guard).

    modal run modal_apps/mine_edit_traces_app.py --n-compile-shards 20

The app module is named *_app to avoid colliding with the mine_edit_traces scripts module it imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=16.0, timeout=3 * 3600,
              volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume})
def mine_pairs(config_dict: dict, source_commit: str, subdir: str) -> dict:
    """Phase 1: scan+split + map(full train) + global-group MMP pairs + corruption char. Write pairs.jsonl
    + mine_meta.json to the volume; return the metadata (census/corruption/provenance/timing + n_pairs)."""
    from mine_edit_traces import MiningConfig, mine_mmp_pairs

    guacamol_volume.reload()
    config = MiningConfig(**config_dict)
    corpus_file = Path(config.corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"corpus {corpus_file} absent; /guacamol has "
                                f"{sorted(p.name for p in Path('/guacamol').iterdir())}")
    corpus_sha = _file_sha256(corpus_file)
    config = replace(config, corpus_sha256=corpus_sha)

    result = mine_mmp_pairs(config)
    result["provenance"]["commit"] = source_commit
    result["provenance"]["corpus_sha256"] = corpus_sha

    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    pairs = result.pop("pairs")
    with (out / "pairs.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    result["n_pairs"] = len(pairs)
    (out / "mine_meta.json").write_text(json.dumps(result, indent=2) + "\n")
    artifact_volume.commit()
    return result


@app.function(image=image, cpu=8.0, timeout=3 * 3600,
              volumes={"/artifacts": artifact_volume})
def compile_shard(shard_index: int, n_shards: int, config_dict: dict, subdir: str) -> int:
    """Phase 2 (parallel): compile the shard_index-th deterministic stride of pairs.jsonl, both directions,
    and write compiled/shard_XXXX.jsonl. Returns the record count."""
    from mine_edit_traces import MiningConfig, compile_mmp

    artifact_volume.reload()
    config = MiningConfig(**config_dict)
    out = Path("/artifacts") / subdir
    with (out / "pairs.jsonl").open() as handle:
        all_pairs = [json.loads(line) for line in handle if line.strip()]
    mine = [tuple(p) for i, p in enumerate(all_pairs) if i % n_shards == shard_index]
    records = compile_mmp(mine, config)

    cdir = out / "compiled"
    cdir.mkdir(parents=True, exist_ok=True)
    with (cdir / f"shard_{shard_index:04d}.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    artifact_volume.commit()
    print(f"[compile] shard {shard_index}/{n_shards}: {len(mine)} pairs -> {len(records)} records",
          flush=True)
    return len(records)


@app.function(image=image, cpu=16.0, timeout=3 * 3600,
              volumes={"/artifacts": artifact_volume})
def reduce_pool(config_dict: dict, subdir: str, meta: dict) -> dict:
    """Phase 3: read every compiled shard, global dedup + cap, write the final pool + summary."""
    from mine_edit_traces import MiningConfig, dedup_and_cap

    artifact_volume.reload()
    config = MiningConfig(**config_dict)
    out = Path("/artifacts") / subdir
    records: list[dict] = []
    for shard_file in sorted((out / "compiled").glob("shard_*.jsonl")):
        with shard_file.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    pool, cap_report = dedup_and_cap(records, config)

    with (out / "edit_pool_full.jsonl").open("w") as handle:
        for record in pool:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "config": meta.get("config"),
        "provenance": meta.get("provenance"),
        "corpus": meta.get("corpus"),
        "corruption": meta.get("corruption"),
        "timing_seconds": meta.get("timing_seconds"),
        "mmp_pairs_grouped": meta.get("n_pairs"),
        "records_compiled_pre_dedup": len(records),
        "global_reduce": cap_report,
        "pool_size": len(pool),
    }
    (out / "mining_summary_full.json").write_text(json.dumps(summary, indent=2) + "\n")
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    corpus: str = "guacamol_subset_500000_seed0.smiles",
    train_size: int = 400_000,
    validation_size: int = 10_000,
    test_size: int = 10_000,
    max_atoms: int = 48,
    split_workers: int = 12,
    corruption_sample_size: int = 500,
    n_compile_shards: int = 20,
    subdir: str = "edit_mining_full_broad",
) -> None:
    # Thin launcher: no miner import (rdkit/etc. live only in the container). n_shards=1 => the miner maps
    # the FULL train partition (global grouping); the compile is fanned out over n_compile_shards.
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
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
        "train_size": train_size, "validation_size": validation_size, "test_size": test_size,
        "max_atoms": max_atoms, "n_shards": 1, "shard_index": 0,
        "split_workers": split_workers, "scaffold_k": 0,
        "corruption_sample_size": corruption_sample_size, "out_dir": f"/artifacts/{subdir}",
    }
    print(f"[1/3] mining pairs from {corpus} at commit {commit} ...")
    meta = mine_pairs.remote(config_dict, commit, subdir)
    n_pairs = meta["n_pairs"]
    print(f"      -> {n_pairs} MMP pairs; corpus retained {meta['corpus']['census']['retained']}")

    print(f"[2/3] compiling over {n_compile_shards} parallel containers ...")
    counts = list(compile_shard.starmap(
        [(i, n_compile_shards, config_dict, subdir) for i in range(n_compile_shards)]))
    print(f"      -> {sum(counts)} records across {n_compile_shards} shards")

    print("[3/3] reducing ...")
    summary = reduce_pool.remote(config_dict, subdir, meta)
    print(json.dumps({"pool_size": summary["pool_size"],
                      "mmp_pairs_grouped": summary["mmp_pairs_grouped"],
                      "records_compiled_pre_dedup": summary["records_compiled_pre_dedup"],
                      "global_reduce": summary["global_reduce"],
                      "timing_seconds": summary["timing_seconds"]}, indent=2))
    print(f"\nFULL MINING COMPLETE -> /artifacts/{subdir}/edit_pool_full.jsonl "
          f"+ mining_summary_full.json")
