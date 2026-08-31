"""Gate 4: the SMALL docking diagnostic. Runs only after Gates 1-3 pass.

Docks the frontier that the zero-oracle particle controller produced. The
question is narrow and worth stating exactly:

    among molecules that are FEASIBLE and DISPLACED, found with zero oracle
    calls, what do they actually dock at?

This is a diagnostic, NOT a T4 benchmark run. A real T4 number requires docking
inside the search loop under the counted-call budget with S_dock feeding back.
Nothing here is a benchmark result and it must not be reported as one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _dock_many,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-gate4")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 6, max_containers=40,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def dock_frontier(job: dict[str, Any]) -> dict[str, Any]:
    import time
    smis = job["smiles"][: int(job["budget"])]
    t0 = time.time()
    ds = _dock_many(smis, job["target"], f"g4_{job['tag']}_{job['idx']}",
                    workers=4, cpu_per_dock=1)
    pairs = [(s, d) for s, d in zip(smis, ds) if d is not None and d != 0.0]
    pairs.sort(key=lambda p: p[1])
    return {"target": job["target"], "idx": job["idx"], "delta": job["delta"],
            "beta": job["beta"], "n_docked": len(pairs),
            "n_requested": len(smis),
            "best_ds": pairs[0][1] if pairs else None,
            "best_smi": pairs[0][0] if pairs else None,
            "top": pairs[:20], "wall_s": time.time() - t0}


@app.local_entrypoint()
def main(tag: str = "barG", budget: int = 120, out: str = "",
         only_beta: float = 4.0):
    import glob
    vol = Path("/tmp/g4in")
    jobs = []
    for f in sorted(glob.glob(str(vol / "**" / f"{tag}_*.json"), recursive=True)):
        d = json.loads(Path(f).read_text())
        if float(d.get("beta", -1)) != float(only_beta):
            continue
        delta = float(d["delta"])
        # dock only FEASIBLE, DISPLACED frontier states, most displaced first
        cands = [s for s, m in d.get("frontier_top", [])
                 if m.get("v") == 0.0 and abs(m.get("dh", 0)) >= 4]
        if not cands:
            continue
        jobs.append({"smiles": cands, "target": d["target"], "idx": d["idx"],
                     "delta": delta, "beta": d["beta"], "budget": budget,
                     "tag": tag})
    print(f"  {len(jobs)} cells to dock, <= {budget} calls each "
          f"(DIAGNOSTIC, not a T4 benchmark run)", flush=True)
    got = []
    for r in dock_frontier.map(jobs, order_outputs=False,
                               return_exceptions=True,
                               wrap_returned_exceptions=False):
        if isinstance(r, dict):
            got.append(r)
            print(f"  {r['target']} s{r['idx']} d{r['delta']} "
                  f"docked {r['n_docked']}/{r['n_requested']} "
                  f"best {r['best_ds']}", flush=True)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
    if out:
        Path(ROOT / out).write_text(json.dumps(got, indent=2))
        print(f"wrote {ROOT / out}")


@app.local_entrypoint()
def calibrate(out: str = "diagnostics/t4_oracle_calibration.json", top: int = 8):
    """Dock the PUBLISHED winners with OUR pipeline and compare to THEIR reported
    scores. This validates our oracle; it does not target their chemistry.

    If our pipeline systematically scores their molecules above their reported
    numbers, part of the apparent search gap is docking protocol (receptor prep,
    box, exhaustiveness) rather than search quality, and every cross-method
    comparison in this project has to be restated accordingly.
    """
    w = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())
    jobs = []
    for k, v in w.items():
        rows = (v.get("winners") or [])[:top]
        if not rows:
            continue
        jobs.append({"smiles": [r["smiles"] for r in rows],
                     "reported": [r["ds"] for r in rows],
                     "target": v["target"], "idx": v["seed_idx"],
                     "delta": v["delta"], "beta": -1.0,
                     "budget": len(rows), "tag": "calib"})
    print(f"  calibrating on {len(jobs)} cells, "
          f"{sum(len(j['smiles']) for j in jobs)} published molecules", flush=True)
    got = []
    for j, r in zip(jobs, dock_frontier.map(jobs, order_outputs=True,
                                            return_exceptions=True,
                                            wrap_returned_exceptions=False)):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
            continue
        ours = dict(r.get("top") or [])
        pairs = [(s, rep, ours.get(s)) for s, rep in
                 zip(j["smiles"], j["reported"]) if ours.get(s) is not None]
        if not pairs:
            print(f"  {j['target']} s{j['idx']}: NONE re-docked", flush=True)
            continue
        deltas = [o - rep for _, rep, o in pairs]
        rec = {"target": j["target"], "idx": j["idx"], "delta": j["delta"],
               "n": len(pairs),
               "mean_delta": sum(deltas) / len(deltas),
               "their_best": min(p[1] for p in pairs),
               "our_best_on_theirs": min(p[2] for p in pairs),
               "pairs": [{"smiles": s, "reported": rep, "ours": o}
                         for s, rep, o in pairs]}
        got.append(rec)
        print(f"  {j['target']} s{j['idx']} d{j['delta']}: their best "
              f"{rec['their_best']}, OUR pipeline on the SAME molecules "
              f"{rec['our_best_on_theirs']}  (mean shift "
              f"{rec['mean_delta']:+.2f} kcal/mol over {rec['n']})", flush=True)
    Path(ROOT / out).write_text(json.dumps(got, indent=2))
    print(f"wrote {ROOT / out}")
