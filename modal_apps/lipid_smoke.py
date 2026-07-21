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


def _base_recipe(train_size, steps, rollout, batch_size, label):
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.recipe import load_tracelet_recipe
    recipe = load_tracelet_recipe(REMOTE_ROOT / "recipes" / "lipid_unconditional_cnof_v1_smoke.json")
    run_dir = Path("/artifacts") / label
    run_dir.mkdir(parents=True, exist_ok=True)
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


def _argv(recipe):
    from compose_v4.experiments.recipe import build_tracelet_recipe_argv
    return list(build_tracelet_recipe_argv(recipe, smiles_file=Path(LIPID_CORPUS),
                                           quality_reference_file=Path(LIPID_REF)))


def _run_trainer(argv):
    import subprocess
    subprocess.run(["python", str(REMOTE_ROOT / "scripts" / "train_tracelet_cnof_gate.py"), *argv],
                   check=True)
    artifacts.commit()


@app.function(image=image, cpu=64.0, timeout=10800, volumes=VOLS)
def compile_paths(train_size, steps, rollout, batch_size, label):
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    recipe, _ = _base_recipe(train_size, steps, rollout, batch_size, label)
    recipe["arguments"]["device"] = "cpu"
    print(f"STAGE 1 COMPILE: {train_size} lipids on 64 CPUs ...", flush=True)
    _run_trainer(_argv(recipe) + ["--compile-paths-only"])
    return {"compiled": True}


@app.function(image=image, gpu="H100", cpu=8.0, timeout=14400, volumes=VOLS)
def train_and_sample(train_size, steps, rollout, batch_size, label, region_aware=True):
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    recipe, run_dir = _base_recipe(train_size, steps, rollout, batch_size, label)
    recipe["arguments"].update({"device": "cuda", "data_workers": 8, "fiber_workers": 8, "rollout_workers": 8})
    extra = ["--require-path-cache"]
    if region_aware:
        extra += ["--region-aware", "--region-prior-table", "/root/compose_v4/region_prior.json"]
    print(f"STAGE 2 TRAIN on H100 (region_aware={region_aware}, loading cached paths) ...", flush=True)
    _run_trainer(_argv(recipe) + extra)

    import torch
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    metrics = json.loads((run_dir / "metrics.json").read_text())
    samples: list[str] = []
    try:
        payload = torch.load(run_dir / "rollouts.pt", weights_only=False)
        for r in payload["rollouts"]:
            state = getattr(r, "final_state", None) or getattr(r, "state", None) or r
            smi = molecular_graph_to_smiles(state)
            if smi:
                samples.append(smi)
    except Exception as exc:
        samples = [f"<extraction failed: {type(exc).__name__}: {exc}>"]
    return {"n_samples": len(samples), "samples": samples[:60],
            "generated_validity": metrics.get("generated_nonnull_smiles"),
            "rollout": metrics.get("rollout")}


@app.local_entrypoint()
def main(train_size: int = 4000, steps: int = 4000, rollout: int = 128,
         batch_size: int = 16, label: str = "lipid_staged_v1", region_aware: bool = True,
         compile_only: bool = False) -> None:
    print(f"=== STAGE 1: compile {train_size} lipid paths on CPU (no GPU) ===", flush=True)
    compile_paths.remote(train_size, steps, rollout, batch_size, label)
    if compile_only:
        print(f"=== compile-only: {train_size}-lipid path cache is now cached under label={label}; "
              f"future runs reuse it via --require-path-cache ===", flush=True)
        return
    print(f"=== STAGE 2: train (region_aware={region_aware}) + sample on H100 ===", flush=True)
    out = train_and_sample.remote(train_size, steps, rollout, batch_size, label, region_aware)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=2))
    print("\n=== generated lipid SMILES ===")
    for s in out["samples"]:
        print(" ", s)
