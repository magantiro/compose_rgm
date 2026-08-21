"""GATE 0 for the GenMol Table-4 lead-optimization benchmark: can we reproduce
their docking numbers at all?

WHY THIS EXISTS. Table 4 is scored by QuickVina docking score. Unlike QED or
Tanimoto, a docking score is not a property of a molecule; it is a property of a
molecule PLUS a receptor preparation, a search box, an exhaustiveness setting and
a 3D embedding. Quoting our -10.6 against their -10.6 is meaningless unless the
two are the same measurement. So before any optimization run, we reproduce the
one thing they published that we can check independently: the docking scores of
the 15 seed molecules.

WHAT IS FIXED HERE, AND FROM WHERE.
  * qvina02 binary, and the five prepared receptors, are taken from the MOOD
    repository (Lee et al. 2023), which is the docking setup GenMol follows.
  * Box centres and sizes are copied from GenMol's own scripts/exps/lead/
    docking/docking.py, which is config-identical to MOOD's.
  * exhaustiveness=1, num_modes=10, obabel --gen3D, exactly as both use.
  * The 15 seed SMILES and their published DS come from GenMol's
    scripts/exps/lead/docking/actives.csv.

THE NOISE FLOOR IS PART OF THE MEASUREMENT. exhaustiveness=1 is the fastest and
noisiest QuickVina setting and neither repository passes --seed, so the score is
a random variable. We therefore dock each seed REPLICATES times. Two numbers
come out: whether our mean agrees with their published value, and how wide the
per-molecule spread is. The second is what tells us later how large a docking
margin has to be before it means anything, and it is not recoverable after the
fact. Reporting a win of 0.2 kcal/mol on a metric whose own replicate spread is
0.8 would be noise dressed as a result.

CPU only. No GPU, no oracle budget, no COMPOSE model involved.
"""

from __future__ import annotations

import json
import time
from typing import Any

import modal

MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
GENMOL = ("https://raw.githubusercontent.com/NVIDIA-BioNeMo/genmol/main/"
          "scripts/exps/lead/docking")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openbabel", "curl", "ca-certificates")
    .pip_install("rdkit==2024.3.5", "numpy==1.26.4")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        *[f"curl -sSL -o /opt/dock/receptors/{t}.pdbqt {MOOD}/receptors/{t}.pdbqt"
          for t in ("parp1", "fa7", "5ht1b", "braf", "jak2")],
        f"curl -sSL -o /opt/dock/actives.csv {GENMOL}/actives.csv",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App("genmol-t4-gate0")

#: Copied verbatim from GenMol scripts/exps/lead/docking/docking.py.
BOXES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "fa7":   ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
    "parp1": ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
    "5ht1b": ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
    "jak2":  ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
    "braf":  ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106)),
}
EXHAUSTIVENESS, NUM_MODES, CPU_DOCK = 1, 10, 4
REPLICATES = 5


def _dock_once(smiles: str, target: str, tag: str) -> float | None:
    """One obabel --gen3D + qvina02 pass. Returns best score, or None on failure."""
    import os
    import subprocess

    d = f"/tmp/{tag}"
    os.makedirs(d, exist_ok=True)
    mol, lig, out = f"{d}/l.mol", f"{d}/l.pdbqt", f"{d}/o.pdbqt"
    for p in (mol, lig, out):
        if os.path.exists(p):
            os.remove(p)
    try:
        subprocess.run(["obabel", f"-:{smiles}", "--gen3D", "-O", mol],
                       capture_output=True, timeout=120, check=True)
        subprocess.run(["obabel", mol, "-O", lig],
                       capture_output=True, timeout=60, check=True)
    except Exception:
        return None
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    cmd = ["/opt/dock/qvina02",
           "--receptor", f"/opt/dock/receptors/{target}.pdbqt",
           "--ligand", lig, "--out", out,
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--cpu", str(CPU_DOCK), "--num_modes", str(NUM_MODES),
           "--exhaustiveness", str(EXHAUSTIVENESS)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
    except Exception:
        return None
    # Best mode is the first REMARK VINA RESULT line of the output pose file.
    try:
        for line in open(out):
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
    except Exception:
        return None
    return None


@app.function(image=image, cpu=(4.0, 4.0), memory=4096, timeout=60 * 60,
              max_containers=20)
def dock_seed(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    scores = [_dock_once(task["smiles"], task["target"], f"r{task['idx']}_{r}")
              for r in range(REPLICATES)]
    return {**task, "scores": scores,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=2 * 60 * 60)
def drive() -> dict[str, Any]:
    import csv

    import numpy as np

    rows = list(csv.DictReader(open("/opt/dock/actives.csv")))
    tasks = [{"idx": i, "target": r["target"], "smiles": r["smiles"],
              "published_ds": float(r["DS"]), "qed": float(r["QED"]),
              "sa": float(r["SA"])} for i, r in enumerate(rows)]
    print(f"{len(tasks)} seeds x {REPLICATES} replicates, "
          f"exhaustiveness={EXHAUSTIVENESS}\n", flush=True)

    out = [r for r in dock_seed.map(tasks, order_outputs=True,
                                    return_exceptions=True,
                                    wrap_returned_exceptions=False)
           if isinstance(r, dict)]

    print(f"{'target':8s}{'published':>11}{'ours mean':>11}{'sd':>7}"
          f"{'min':>8}{'delta':>8}{'ok':>4}  smiles", flush=True)
    deltas, spreads = [], []
    for r in sorted(out, key=lambda x: x["idx"]):
        s = [v for v in r["scores"] if v is not None]
        r["n_ok"] = len(s)
        if not s:
            print(f"  {r['target']:8s}  ALL REPLICATES FAILED", flush=True)
            continue
        # Published DS is stored positive; qvina reports negative. Compare
        # magnitudes, and use the BEST (most negative) as the comparable
        # quantity since a docking score is a minimum over sampled poses.
        mean, sd, best = float(np.mean(s)), float(np.std(s, ddof=1) if len(s) > 1 else 0.0), float(np.min(s))
        r.update(mean=mean, sd=sd, best=best)
        delta = abs(best) - r["published_ds"]
        r["delta_best_vs_published"] = delta
        deltas.append(delta)
        spreads.append(max(s) - min(s))
        print(f"  {r['target']:8s}{-r['published_ds']:>11.1f}{mean:>11.2f}"
              f"{sd:>7.2f}{best:>8.2f}{delta:>+8.2f}{r['n_ok']:>4}  "
              f"{r['smiles'][:42]}", flush=True)

    summary = {
        "n_seeds": len(out), "replicates": REPLICATES,
        "exhaustiveness": EXHAUSTIVENESS,
        "mean_abs_delta": float(np.mean(np.abs(deltas))) if deltas else None,
        "median_abs_delta": float(np.median(np.abs(deltas))) if deltas else None,
        "max_abs_delta": float(np.max(np.abs(deltas))) if deltas else None,
        "mean_replicate_spread": float(np.mean(spreads)) if spreads else None,
        "max_replicate_spread": float(np.max(spreads)) if spreads else None,
        "rows": out,
    }
    print(f"\n  agreement with published : mean |delta| "
          f"{summary['mean_abs_delta']:.2f}  median "
          f"{summary['median_abs_delta']:.2f}  max {summary['max_abs_delta']:.2f}")
    print(f"  docking noise floor      : mean replicate spread "
          f"{summary['mean_replicate_spread']:.2f}  max "
          f"{summary['max_replicate_spread']:.2f}")
    print("\n  GATE: agreement must be comparable to the noise floor. If mean")
    print("  |delta| is much larger than the replicate spread, our stack is not")
    print("  measuring their quantity and Table 4 cannot be entered.")
    return summary


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path
    o = drive.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/genmol_t4_gate0.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
