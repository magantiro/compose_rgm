"""Receding-horizon A/B: does extra horizon convert into benchmark hit-rate?

The contact diagnostic showed more horizon produces more target contact. This
asks the question the benchmark actually scores: does it raise source success@4?

H=40 is a RECEDING-HORIZON controller, not a broken one. h_phi's budget is a
25-slot one-hot, so b_eff = min(24, b_actual): a rolling 24-edit lookahead while
far from termination, then the calibrated countdown once 24 actually remain.
Nothing was retrained.

Paired by construction: seed_for keys on (arm, source, candidate) and not on
horizon, so candidate k has the same seed in both arms. Exact McNemar on the
discordant pairs, since the arms share sources and concordant ones carry no
information about a difference.
"""
from __future__ import annotations
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any
import modal
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-recede-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
HS = (24, 40)
KS = (1, 2, 3, 4)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    per, strat = defaultdict(dict), {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        strat[r["index"]] = r["stratum"]
        per[r["index"]][r["horizon"]] = r["arms"]["restart"]
    rows = {i: v for i, v in per.items() if all(h in v for h in HS)}
    print(f"{len(rows)} sources with both horizons "
          f"({len(per)-len(rows)} incomplete)\n")
    by = defaultdict(list)
    for i in rows:
        by[strat[i]].append(i)

    def solved(i, H, k):
        return any(c["success"] for c in rows[i][H]["candidates"][:k])

    out: dict[str, Any] = {"n": len(rows), "cumulative": {}, "paired": {}}
    print(f"{'scope':<11}{'H':>4}" + "".join(f"{'k='+str(k):>6}" for k in KS)
          + f"{'contact runs':>14}{'work':>12}")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        for H in HS:
            cum = [sum(1 for i in ids if solved(i, H, k)) for k in KS]
            runs = [c for i in ids for c in rows[i][H]["candidates"]]
            cr = sum(1 for c in runs if c.get("contact", 0) > 0)
            wk = sum(rows[i][H]["work_transitions"] for i in ids)
            out["cumulative"].setdefault(scope, {})[H] = cum + [len(ids)]
            print(f"  {scope:<9}{H:>4}" + "".join(f"{c:>6}" for c in cum)
                  + f"{cr:>8}/{len(runs):<5}{wk:>12,}   /{len(ids)}")
        print()

    print("PAIRED at k=4 (exact McNemar)")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        w = [i for i in ids if solved(i, 40, 4) and not solved(i, 24, 4)]
        l = [i for i in ids if solved(i, 24, 4) and not solved(i, 40, 4)]
        out["paired"][scope] = {"h40_only": w, "h24_only": l, "p": mcnemar(len(w), len(l))}
        print(f"  {scope:<10} H40-only {len(w):>2}  H24-only {len(l):>2}"
              f"  p={mcnemar(len(w), len(l)):.4f}  won:{w[:9]} lost:{l[:9]}")

    print(f"\n{'H':>4}{'successes':>11}{'median QED':>12}{'median sim':>12}")
    for H in HS:
        succ = [c for i in rows for c in rows[i][H]["candidates"] if c["success"]]
        q = sorted(c["terminal_qed"] for c in succ)
        sm = sorted(c["terminal_sim"] for c in succ)
        med = lambda v: v[len(v)//2] if v else float("nan")
        out.setdefault("endpoints", {})[H] = {"n": len(succ), "median_qed": med(q),
                                              "median_sim": med(sm)}
        print(f"  {H:>2}{len(succ):>11}{med(q):>12.3f}{med(sm):>12.3f}")
    print("\n  Endpoint QED/sim guard against 'more edits = drift': similarity is")
    print("  measured to x0 and the region needs sim >= 0.40 regardless of H.")
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_recede_v5") -> None:
    o = read.remote(out_dir)
    Path("docs/RECEDING_HORIZON_AB.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/RECEDING_HORIZON_AB.json")
