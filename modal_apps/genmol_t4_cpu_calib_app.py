"""CPU calibration for the T4 docking stack: cheapest setting that PRESERVES Gate 0.

Gate 0's argument for a legitimate head-to-head was that we use GenMol's receptor
preparation, box, binary, exhaustiveness and modes. `--cpu` is part of that
invocation. Dropping it from 4 to 1 to save money is therefore a change to the
measurement, not just to the bill, and has to be validated rather than assumed.

So this measures BOTH axes at cpu in {1, 2, 4}, with the CONTAINER sized to match
the flag so the cost comparison is honest:

    throughput   seconds per docking, and $ per docking at that allocation
    parity       mean / bias / spread against the 15 published seed scores

DECISION RULE, fixed before the run: take the cheapest setting whose score
distribution stays compatible with the Gate 0 calibration (mean |delta| ~0.34,
bias ~-0.02, median replicate spread ~0.70). If 1 core changes the distribution
materially, keep the matched configuration and pay for it. Benchmark fidelity is
not traded for compute.

The stake: at 4 cores the matched 90-run official evaluation is ~$18; at 1 core
it is ~$6. Same experiment either way, if and only if the scores agree.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
GENMOL = ("https://raw.githubusercontent.com/NVIDIA-BioNeMo/genmol/main/"
          "scripts/exps/lead/docking")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openbabel", "curl", "ca-certificates")
    .pip_install("numpy==1.26.4")
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

app = modal.App("genmol-t4-cpu-calib")

BOXES = {
    "fa7":   ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
    "parp1": ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
    "5ht1b": ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
    "jak2":  ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
    "braf":  ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106)),
}
REPLICATES = 3
CORE_H, GIB_H = 0.04716, 0.007992


def _run(smiles: str, target: str, ncpu: int, tag: str):
    """One dock. Returns (score, seconds) with the docking timed in isolation."""
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
        return None, 0.0
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    t0 = time.perf_counter()
    try:
        subprocess.run(
            ["/opt/dock/qvina02", "--receptor", f"/opt/dock/receptors/{target}.pdbqt",
             "--ligand", lig, "--out", out,
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
             "--cpu", str(ncpu), "--num_modes", "10", "--exhaustiveness", "1"],
            capture_output=True, timeout=600, check=True)
    except Exception:
        return None, time.perf_counter() - t0
    dt = time.perf_counter() - t0
    try:
        for line in open(out):
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3]), dt
    except Exception:
        pass
    return None, dt


def _sweep(ncpu: int) -> dict[str, Any]:
    import csv
    rows = list(csv.DictReader(open("/opt/dock/actives.csv")))
    out = []
    for i, r in enumerate(rows):
        for k in range(REPLICATES):
            sc, dt = _run(r["smiles"], r["target"], ncpu, f"c{ncpu}_{i}_{k}")
            out.append({"idx": i, "target": r["target"],
                        "published_ds": float(r["DS"]), "score": sc,
                        "seconds": round(dt, 2)})
    return {"ncpu": ncpu, "rows": out}


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=4 * 60 * 60)
def sweep1():
    return _sweep(1)


@app.function(image=image, cpu=(2.0, 2.0), memory=4096, timeout=4 * 60 * 60)
def sweep2():
    return _sweep(2)


@app.function(image=image, cpu=(4.0, 4.0), memory=4096, timeout=4 * 60 * 60)
def sweep4():
    return _sweep(4)


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    res = [sweep1.remote(), sweep2.remote(), sweep4.remote()]
    print(f"15 seeds x {REPLICATES} replicates at cpu in 1/2/4, container matched\n")
    print(f"{'cpu':>4}{'s/dock':>9}{'$/dock':>10}{'$ per 90k':>11}"
          f"{'mean|d|':>10}{'bias':>8}{'med spread':>12}{'fail':>6}")
    summary = []
    for r in res:
        n = r["ncpu"]
        ok = [x for x in r["rows"] if x["score"] is not None]
        sec = float(np.mean([x["seconds"] for x in ok]))
        rate = n * CORE_H + 4 * GIB_H            # container $/hour at this size
        dps = sec / 3600 * rate
        by: dict[int, list] = {}
        for x in ok:
            by.setdefault(x["idx"], []).append(x)
        d = [abs(float(np.mean([y["score"] for y in v])) + v[0]["published_ds"])
             for v in by.values()]
        bias = float(np.mean([float(np.mean([y["score"] for y in v]))
                              + v[0]["published_ds"] for v in by.values()]))
        spread = float(np.median([max(y["score"] for y in v)
                                  - min(y["score"] for y in v) for v in by.values()]))
        summary.append({"ncpu": n, "s_per_dock": sec, "usd_per_dock": dps,
                        "usd_90k": dps * 90000, "mean_abs_delta": float(np.mean(d)),
                        "bias": bias, "median_spread": spread,
                        "n_fail": len(r["rows"]) - len(ok)})
        print(f"{n:>4}{sec:>9.2f}{dps:>10.5f}{dps*90000:>11.2f}"
              f"{np.mean(d):>10.2f}{bias:>+8.2f}{spread:>12.2f}"
              f"{len(r['rows'])-len(ok):>6}")
    print("\n  GATE 0 reference: mean|d| 0.34   bias -0.02   median spread 0.70")
    print("  DECISION: cheapest cpu whose parity stays compatible with that.")
    print("  If 1 core shifts the distribution, keep the matched config and pay.")
    p = Path(__file__).resolve().parents[1] / "diagnostics/genmol_t4_cpu_calib.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"summary": summary, "sweeps": res}, indent=1))
    print(f"\nwrote {p}")
