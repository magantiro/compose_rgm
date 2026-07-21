"""Staged Modal run: compile teacher paths on CPU, then train + sample on H100.

Stage 1 (CPU-only, 64 cores): compile teacher paths for the lipid corpus and write
the sharded path cache (--compile-paths-only). No GPU -> cheap, and it's the
embarrassingly-parallel bottleneck, so we throw cores at it.
Stage 2 (H100): load the cache (--require-path-cache, zero recompile) -> train ->
sample lipids. The H100 only spins up for actual training, never idles on compile.

    modal run modal_apps/lipid_smoke.py --train-size 100000 --steps 2000 --rollout 128
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
IGNORE = ("**/__pycache__/**", "**/*.pyc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "scipy==1.13.1",
                 "networkx==3.3", "rdkit==2024.3.5", "fcd-torch==1.0.7")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src"), "PYTHONUNBUFFERED": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True, ignore=IGNORE)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True, ignore=IGNORE)
    .add_local_dir(ROOT / "recipes", str(REMOTE_ROOT / "recipes"), copy=True, ignore=IGNORE)
    .add_local_file(ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/region_conditioned_prior_v1.json",
                    "/root/compose_v4/region_prior.json", copy=True)
)

app = modal.App("compose-v4-lipid-smoke")
guacamol = modal.Volume.from_name("guacamol", create_if_missing=False)
artifacts = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)
VOLS = {"/guacamol": guacamol, "/artifacts": artifacts}
LIPID_CORPUS = "/guacamol/lipid_corpus_cnof_v1.smiles"
LIPID_REF = "/guacamol/lipid_heldout_ref_5000.smiles"


def _base_recipe(train_size, steps, rollout, batch_size, label, source_prior=None):
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.recipe import load_tracelet_recipe
    recipe = load_tracelet_recipe(REMOTE_ROOT / "recipes" / "lipid_unconditional_cnof_v1_smoke.json")
    run_dir = Path("/artifacts") / label
    run_dir.mkdir(parents=True, exist_ok=True)
    if source_prior:
        recipe["arguments"]["source_prior"] = source_prior
    # identical path-determining args in BOTH stages so the path-cache signature matches
    recipe["arguments"].update({
        "train_size": train_size, "steps": steps, "schedule_steps": steps,
        "warmup_steps": max(1, min(50, steps // 5 or 1)),
        "evaluation_every": max(1, min(200, steps)),
        "early_stopping_patience": 100000,  # effectively disabled -- train the full step budget
        "fast_split": train_size <= 8000,  # <=8k: quick first-N split (real R0 lipids); larger: random scan over 429k
        "path_workers": 64, "corpus_workers": 64,
        "rollout_samples": rollout, "batch_size": batch_size,
        "output": str(run_dir / "metrics.json"), "checkpoint": str(run_dir / "checkpoint.pt"),
        "path_cache": str(run_dir / "compiled_paths.pt"), "rollout_cache": str(run_dir / "rollouts.pt"),
        "evaluation_cache_dir": "/artifacts/_shared/evaluation_batches",
        "training_support_cache_dir": str(run_dir / "training_support"),
        "training_support_shard_size": 16000,
    })
    return recipe, run_dir


def _argv(recipe, corpus=LIPID_CORPUS):
    from compose_v4.experiments.recipe import build_tracelet_recipe_argv
    return list(build_tracelet_recipe_argv(recipe, smiles_file=Path(corpus),
                                           quality_reference_file=Path(LIPID_REF)))


def _run_trainer(argv, tag="trainer", commit_every=90):
    """Run the trainer as a subprocess and commit the volume every `commit_every`s.

    The commit runs in THIS (parent) process while training proceeds in the
    subprocess, so it never pauses training -- it just flushes finished shards and
    the interim `checkpoint.best_so_far.pt` to durable storage as they appear, so a
    crash keeps progress AND a separate sampler can read the latest checkpoint."""
    import subprocess, time
    proc = subprocess.Popen(
        ["python", str(REMOTE_ROOT / "scripts" / "train_tracelet_cnof_gate.py"), *argv])
    while proc.poll() is None:
        time.sleep(commit_every)
        try:
            artifacts.commit()
            print(f"  [commit] flushed volume ({tag})", flush=True)
        except Exception as exc:  # best-effort; never kill the run over a commit
            print(f"  [commit] skipped: {type(exc).__name__}", flush=True)
    if proc.returncode != 0:
        artifacts.commit()
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    artifacts.commit()


def _extract_smiles(run_dir, name="rollouts.pt"):
    import torch
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    out = []
    try:
        payload = torch.load(run_dir / name, weights_only=False)
        for r in payload["rollouts"]:
            state = getattr(r, "final_state", None) or getattr(r, "state", None) or r
            smi = molecular_graph_to_smiles(state)
            if smi:
                out.append(smi)
    except Exception as exc:
        out = [f"<extraction failed: {type(exc).__name__}: {exc}>"]
    return out


def _lipid_quality(samples):
    """Mid-training quality eval on generated SMILES: validity, region balance
    (the 'heads look like tails' diagnostic), linker chemistry, physchem."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.lipids.region_labels import lipid_region_labels, REGION_NAMES
    from compose_v4.lipids.head_region import basic_nitrogens
    ester = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
    amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
    valid = [m for m in (Chem.MolFromSmiles(s) for s in samples if isinstance(s, str)) if m is not None]
    n, nv = len(samples), len(valid)
    if not valid:
        return {"n": n, "validity": 0.0}
    reg = {name: 0 for name in REGION_NAMES.values()}
    hasE = hasA = hasN = 0
    mw, logp = [], []
    for m in valid:
        labs = lipid_region_labels(m)
        for code, name in REGION_NAMES.items():
            reg[name] += labs.count(code)
        hasE += bool(m.HasSubstructMatch(ester))
        hasA += bool(m.HasSubstructMatch(amide))
        hasN += 1 if basic_nitrogens(m) else 0
        mw.append(Descriptors.MolWt(m))
        logp.append(Crippen.MolLogP(m))
    tot = sum(reg.values()) or 1
    med = lambda x: round(sorted(x)[len(x) // 2], 1)
    return {
        "n": n, "validity": round(nv / n, 3),
        "region_pct": {k: round(100 * v / tot, 1) for k, v in reg.items()},
        "pct_ester": round(100 * hasE / nv, 1), "pct_amide": round(100 * hasA / nv, 1),
        "pct_basic_amine_head": round(100 * hasN / nv, 1),
        "mw_median": med(mw), "logp_median": med(logp),
    }


@app.function(image=image, cpu=64.0, timeout=10800, volumes=VOLS)
def compile_paths(train_size, steps, rollout, batch_size, label, corpus=LIPID_CORPUS, source_prior=None):
    """Compile teacher paths on CPU and persist the sharded cache to the durable
    artifacts volume. Commits INCREMENTALLY (every 90s) so a mid-compile crash
    keeps every shard finished so far -- the paths are the expensive part, and
    they're worth saving even if the container dies before the full corpus is done."""
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    recipe, _ = _base_recipe(train_size, steps, rollout, batch_size, label, source_prior)
    recipe["arguments"]["device"] = "cpu"
    print(f"STAGE 1 COMPILE: {train_size} lipids from {corpus} on 64 CPUs (incremental commits) ...", flush=True)
    _run_trainer(_argv(recipe, corpus) + ["--compile-paths-only"], tag=f"compile:{label}")
    return {"compiled": True}


@app.function(image=image, cpu=64.0, timeout=10800, volumes=VOLS)
def compile_shard(train_size, steps, rollout, batch_size, label, corpus, source_prior,
                  shard_count, shard_index):
    """Distributed-compile worker: compile ONLY this worker's train transport shards
    (disjoint by shard_index) into the shared volume, then exit. A finalize pass
    (plain compile_paths) resumes all shards + does val/test + the manifest."""
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    recipe, _ = _base_recipe(train_size, steps, rollout, batch_size, label, source_prior)
    recipe["arguments"]["device"] = "cpu"
    extra = ["--compile-paths-only", "--compile-shard-count", str(shard_count),
             "--compile-shard-index", str(shard_index), "--compile-train-shards-only"]
    print(f"COMPILE SHARD {shard_index}/{shard_count}: train shards on 64 CPUs ...", flush=True)
    _run_trainer(_argv(recipe, corpus) + extra, tag=f"cshard:{label}:{shard_index}")
    return {"shard": shard_index}


@app.function(image=image, gpu="H100", cpu=8.0, timeout=14400, volumes=VOLS)
def train_and_sample(train_size, steps, rollout, batch_size, label, region_aware=True, corpus=LIPID_CORPUS, source_prior=None):
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    recipe, run_dir = _base_recipe(train_size, steps, rollout, batch_size, label, source_prior)
    recipe["arguments"].update({"device": "cuda", "data_workers": 8, "fiber_workers": 8, "rollout_workers": 8})
    extra = ["--require-path-cache"]
    if region_aware:
        extra += ["--region-aware", "--region-prior-table", "/root/compose_v4/region_prior.json"]
    print(f"STAGE 2 TRAIN on H100 (region_aware={region_aware}, loading cached paths) ...", flush=True)
    _run_trainer(_argv(recipe, corpus) + extra, tag=f"train:{label}")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    samples = _extract_smiles(run_dir)
    return {"n_samples": len(samples), "samples": samples[:60],
            "generated_validity": metrics.get("generated_nonnull_smiles"),
            "rollout": metrics.get("rollout")}


@app.function(image=image, gpu="H100", cpu=8.0, timeout=3600, volumes=VOLS)
def sample_checkpoint(label, n=64, train_size=38000, steps=4000,
                      corpus=LIPID_CORPUS, region_aware=True, source_prior=None):
    """Off-training-GPU peek: load the interim best checkpoint a running train job
    has committed and roll out N lipids. Runs on its OWN container, so it never
    slows the training job. Writes to peek_* paths so it can't clobber the real
    training outputs sharing the run dir."""
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    artifacts.reload()  # pull the latest committed checkpoint
    recipe, run_dir = _base_recipe(train_size, steps, n, 16, label, source_prior)
    recipe["arguments"].update({
        "device": "cuda", "rollout_workers": 8, "rollout_samples": n,
        "output": str(run_dir / "peek_metrics.json"),
        "rollout_cache": str(run_dir / "peek_rollouts.pt"),
        "checkpoint": str(run_dir / "peek_checkpoint.pt"),
    })
    best = run_dir / "checkpoint.best_so_far.pt"
    if not best.exists():
        return {"error": f"no interim checkpoint committed yet under {label} "
                          f"(train run may not have hit its first eval)"}
    extra = ["--require-path-cache", "--load-checkpoint", str(best)]
    if region_aware:
        extra += ["--region-aware", "--region-prior-table", "/root/compose_v4/region_prior.json"]
    print(f"PEEK: sampling {n} lipids from {best.name} (evaluation-only, off training GPU) ...", flush=True)
    _run_trainer(_argv(recipe, corpus) + extra, tag=f"peek:{label}")
    samples = _extract_smiles(run_dir, "peek_rollouts.pt")
    return {"n_samples": len(samples), "samples": samples[:n], "quality": _lipid_quality(samples)}


@app.local_entrypoint()
def peek(label: str, n: int = 64, train_size: int = 38000, steps: int = 4000,
         corpus: str = LIPID_CORPUS, region_aware: bool = True, source_prior: str = "") -> None:
    out = sample_checkpoint.remote(label, n, train_size, steps, corpus, region_aware, source_prior or None)
    if "error" in out:
        print("PEEK:", out["error"])
        return
    print(f"=== {out['n_samples']} lipids from the interim checkpoint of {label} ===")
    print("mid-training quality:", json.dumps(out.get("quality", {}), indent=2))
    for s in out["samples"]:
        print(" ", s)


@app.local_entrypoint()
def main(train_size: int = 4000, steps: int = 4000, rollout: int = 128,
         batch_size: int = 16, label: str = "lipid_staged_v1", region_aware: bool = True,
         compile_only: bool = False, corpus: str = LIPID_CORPUS, source_prior: str = "",
         n_workers: int = 1) -> None:
    if n_workers > 1:
        print(f"=== STAGE 1 (distributed): {n_workers} CPU compile-workers over train shards ===", flush=True)
        handles = [compile_shard.spawn(train_size, steps, rollout, batch_size, label, corpus,
                                       source_prior or None, n_workers, k) for k in range(n_workers)]
        for h in handles:
            h.get()
        print("=== STAGE 1b: finalize (resume all shards + val/test + manifest) ===", flush=True)
        compile_paths.remote(train_size, steps, rollout, batch_size, label, corpus, source_prior or None)
    else:
        print(f"=== STAGE 1: compile {train_size} lipid paths from {corpus} on CPU (no GPU) ===", flush=True)
        compile_paths.remote(train_size, steps, rollout, batch_size, label, corpus, source_prior or None)
    if compile_only:
        print(f"=== compile-only: {train_size}-lipid path cache is now cached under label={label}; "
              f"future runs reuse it via --require-path-cache ===", flush=True)
        return
    print(f"=== STAGE 2: train (region_aware={region_aware}) + sample on H100 ===", flush=True)
    out = train_and_sample.remote(train_size, steps, rollout, batch_size, label, region_aware, corpus, source_prior or None)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=2))
    print("\n=== generated lipid SMILES ===")
    for s in out["samples"]:
        print(" ", s)
