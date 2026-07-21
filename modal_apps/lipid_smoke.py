"""Standalone Modal GPU run: train the unconditional LIPID generator + sample lipids.

Runs the core trainer end-to-end (compile paths -> support -> train -> rollout) on
the LIPID corpus (/guacamol/lipid_corpus_cnof_v1.smiles, NOT guacamol drug-like),
with the Lineage-B lipid recipe at max_atoms=96, then extracts the generated lipid
SMILES. Reuses the launcher's image + recipe->argv conversion so args match exactly.

    modal run --detach modal_apps/lipid_smoke.py --train-size 60 --steps 8 --rollout 8   # cheap validation
    modal run --detach modal_apps/lipid_smoke.py --train-size 2500 --steps 400 --rollout 96  # real-ish
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
)

app = modal.App("compose-v4-lipid-smoke")
guacamol = modal.Volume.from_name("guacamol", create_if_missing=False)
artifacts = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

LIPID_CORPUS = "/guacamol/lipid_corpus_cnof_v1.smiles"
LIPID_REF = "/guacamol/lipid_heldout_ref_5000.smiles"


@app.function(image=image, gpu="A100", timeout=7200,
              volumes={"/guacamol": guacamol, "/artifacts": artifacts})
def train_and_sample(train_size: int, steps: int, rollout: int, batch_size: int, label: str) -> dict:
    import subprocess
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.recipe import build_tracelet_recipe_argv, load_tracelet_recipe

    recipe = load_tracelet_recipe(REMOTE_ROOT / "recipes" / "lipid_unconditional_cnof_v1_smoke.json")
    run_dir = Path("/artifacts") / label
    run_dir.mkdir(parents=True, exist_ok=True)
    recipe["arguments"].update({
        "train_size": train_size, "steps": steps, "schedule_steps": steps,
        "rollout_samples": rollout, "batch_size": batch_size, "device": "cuda",
        "output": str(run_dir / "metrics.json"), "checkpoint": str(run_dir / "checkpoint.pt"),
        "path_cache": str(run_dir / "compiled_paths.pt"), "rollout_cache": str(run_dir / "rollouts.pt"),
        "evaluation_cache_dir": "/artifacts/_shared/evaluation_batches",
        "training_support_cache_dir": str(run_dir / "training_support"),
        "training_support_shard_size": 16000,
    })
    argv = build_tracelet_recipe_argv(recipe, smiles_file=Path(LIPID_CORPUS),
                                      quality_reference_file=Path(LIPID_REF))
    print("RUNNING:", " ".join(argv[:6]), "...", flush=True)
    subprocess.run(["python", str(REMOTE_ROOT / "scripts" / "train_tracelet_cnof_gate.py"), *argv],
                   check=True)
    artifacts.commit()

    # extract generated lipid SMILES from the rollout cache
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
    except Exception as exc:  # keep metrics even if extraction shape differs
        samples = [f"<extraction failed: {type(exc).__name__}: {exc}>"]
    return {"metrics_keys": sorted(metrics)[:40], "n_samples": len(samples), "samples": samples[:40],
            "generated_validity": metrics.get("generated_nonnull_smiles"),
            "rollout": metrics.get("rollout")}


@app.local_entrypoint()
def main(train_size: int = 60, steps: int = 8, rollout: int = 8, batch_size: int = 16,
         label: str = "lipid_smoke_live") -> None:
    out = train_and_sample.remote(train_size, steps, rollout, batch_size, label)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=2))
    print("\n=== generated lipid SMILES ===")
    for s in out["samples"]:
        print(" ", s)
