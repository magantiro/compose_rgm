"""Modal: shardable global-grouping edit-trace mining for the B-edit training pool.

This app mines the MMP + scaffold-NN pools and characterizes the corruption layer for the B-edit training
mixture (see ``scripts/mine_edit_traces.py`` for the map -> global-group -> compile -> global-reduce
design). It is DATA MINING + COMPILATION ONLY -- it never loads a checkpoint and never trains.

Gated use (do not skip a step):
  1. local fixture (``scripts/mine_edit_traces.py`` on a small local corpus) -- DONE before this app runs;
  2. ONE deterministic validation shard here (default ~20k of the TRAIN partition), from a clean committed
     checkout, reporting per-molecule yield + measured per-phase timing + the projected full-run cost;
  3. STOP for explicit authorization; only then the full sweep (all shards) below.

Validation shard:
    modal run modal_apps/mine_edit_traces.py \
        --corpus guacamol_subset_500000_seed0.smiles --n-shards 24 --shard-index 0

The shard job does the ONE-TIME full-corpus scan+split (deterministic, the same ``load_cnof_corpus_split``
the trainer uses -> TRAIN-only, no val/test leakage), then maps+groups+compiles a single stride shard and
writes the pool + summary to the artifact volume. Grouping a single shard is a LOWER BOUND on pool yield:
the full run merges every shard's emissions before the global core/scaffold grouping, so cross-shard pairs
that a single shard cannot see are recovered there.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
if REMOTE_ROOT.is_dir():  # inside the Modal container
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
else:  # local launcher: the @app.local_entrypoint imports the miner (MiningConfig) here
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
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


@app.function(image=image, cpu=16.0, timeout=6 * 3600,
              volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume})
def mine_shard(config_dict: dict, source_commit: str, out_subdir: str) -> dict:
    """Mine ONE shard: pin the corpus SHA-256 (computed where the corpus lives), run the timed pipeline,
    write pool + summary to the artifact volume, and return the summary (without the bulky pool)."""
    from mine_edit_traces import MiningConfig, _write_outputs, run_pipeline

    guacamol_volume.reload()
    config = MiningConfig(**config_dict)
    corpus_file = Path(config.corpus_path)
    if not corpus_file.exists():
        available = sorted(p.name for p in Path("/guacamol").iterdir())
        raise FileNotFoundError(f"corpus {corpus_file} absent; /guacamol has {available}")

    corpus_sha = _file_sha256(corpus_file)
    config = replace(config, corpus_sha256=corpus_sha, out_dir=f"/artifacts/{out_subdir}")

    t0 = time.time()
    summary = run_pipeline(config)
    summary["wall_seconds_total"] = round(time.time() - t0, 2)
    summary["provenance"]["commit"] = source_commit  # git is absent on the remote image
    summary["provenance"]["corpus_sha256"] = corpus_sha
    summary["source_commit"] = source_commit

    pool = summary["pool"]
    _write_outputs(dict(summary), config)  # pops "pool", writes pool jsonl + summary json
    artifact_volume.commit()

    summary.pop("pool", None)
    summary["pool_size"] = len(pool)
    return summary


def _project_full_run(summary: dict, n_shards: int) -> dict:
    """Project the full-corpus (all-shards) cost from the single validation shard's measured timing.
    The corpus scan+split is ONE-TIME; map/group/compile scales ~linearly with shard count. Corruption
    is characterized on a fixed sample (constant), so it does NOT scale with shards."""
    timing = summary["timing_seconds"]
    scan = timing["corpus_scan_split_one_time"]
    per_shard = timing["shard_map_group_compile"]
    projected_seconds = scan + per_shard * n_shards
    shard_pool = summary["pool_size"]
    return {
        "n_shards": n_shards,
        "one_time_scan_seconds": scan,
        "per_shard_map_compile_seconds": per_shard,
        "projected_wall_seconds_serial": round(projected_seconds, 1),
        "projected_wall_hours_serial": round(projected_seconds / 3600, 2),
        "projected_cpu_hours": round((scan + per_shard * n_shards) / 3600, 2),
        "shard_pool_records": shard_pool,
        "projected_pool_lower_bound": shard_pool * n_shards,
        "projection_note": (
            "pool is a LOWER BOUND: per-shard grouping misses cross-shard core/scaffold pairs that the "
            "full global reduce recovers, so the real full-run pool exceeds shard_pool * n_shards; "
            "wall-clock scales sublinearly under Modal fan-out (per-shard time is the map/compile work)."
        ),
    }


@app.local_entrypoint()
def main(
    corpus: str = "guacamol_subset_500000_seed0.smiles",
    n_shards: int = 20,
    shard_index: int = 0,
    train_size: int = 400_000,
    validation_size: int = 10_000,
    test_size: int = 10_000,
    max_atoms: int = 48,
    split_workers: int = 12,
    corruption_sample_size: int = 2_000,
    out_subdir: str = "edit_mining_validation",
) -> None:
    from mine_edit_traces import MiningConfig
    from dataclasses import asdict

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    code_dirty = [ln for ln in dirty.splitlines()
                  if ln[3:].startswith(("src/", "scripts/", "modal_apps/", "tests/"))]
    if code_dirty:
        raise SystemExit(f"REFUSING to launch from a dirty code tree; run the prelaunch gate. {code_dirty}")

    config = MiningConfig(
        corpus_id=Path(corpus).stem,
        corpus_path=f"/guacamol/{corpus}",
        train_size=train_size, validation_size=validation_size, test_size=test_size,
        max_atoms=max_atoms, n_shards=n_shards, shard_index=shard_index,
        split_workers=split_workers,
        corruption_sample_size=corruption_sample_size, out_dir=f"/artifacts/{out_subdir}",
    )
    print(f"launching validation shard {shard_index}/{n_shards} of {corpus} at commit {commit}")
    summary = mine_shard.remote(asdict(config), commit, out_subdir)

    summary["projected_full_run"] = _project_full_run(summary, n_shards)
    print(json.dumps(summary, indent=2))
    print("\nVALIDATION SHARD COMPLETE. STOP for explicit authorization before the full "
          f"{n_shards}-shard run. Summary written to /artifacts/{out_subdir}.")
