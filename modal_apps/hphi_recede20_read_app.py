"""H24 vs H40 at 20 candidates -- GrIDDD's exact output budget.

STATISTICS. Repeated candidate seeds estimate each source's success PROBABILITY
more precisely; they do NOT create more sources. So the unit of generalisation
stays the 64 molecules, and concatenating 64 x 5 source-batch pairs into a
McNemar with n=320 would be a fake sample. Instead:

    S_H(x) = (4-candidate batches succeeding out of 5) for source x
    compare S_40(x) - S_24(x) across the 64 sources
    paired BOOTSTRAP over SOURCES for the interval

Also reports the five independent success@4 batches, because "H40 beats H24 in
most independent seed batches" is the structural consistency check that matters
more than any single p-value on a development panel.

Candidates 0-3 share seeds with the v5 run, so their success@4 is a free
reproducibility check.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import modal
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-recede20-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
HS = (24, 40)
KS = (1, 2, 3, 4, 8, 12, 16, 20)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=25 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    import numpy as np
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    per, strat = defaultdict(dict), {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        strat[r["index"]] = r["stratum"]
        per[r["index"]][r["horizon"]] = r["arms"]["restart"]["candidates"]
    rows = {i: v for i, v in per.items()
            if all(h in v and len(v[h]) >= 20 for h in HS)}
    print(f"{len(rows)} sources with 20 candidates at both horizons "
          f"({len(per)-len(rows)} incomplete)\n")
    by = defaultdict(list)
    for i in rows:
        by[strat[i]].append(i)
    out: dict[str, Any] = {"n": len(rows)}

    def sat(i, H, k):
        return any(c["success"] for c in rows[i][H][:k])

    print(f"{'scope':<11}{'H':>4}" + "".join(f"{'@'+str(k):>6}" for k in KS))
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        for H in HS:
            cum = [sum(1 for i in ids if sat(i, H, k)) for k in KS]
            out.setdefault("cumulative", {}).setdefault(scope, {})[H] = cum
            print(f"  {scope:<9}{H:>4}" + "".join(f"{c:>6}" for c in cum)
                  + f"   /{len(ids)}")
        print()

    # five INDEPENDENT 4-candidate batches
    print("independent success@4 batches (5 disjoint seed sets)")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("marginal", "hard") if by.get(s)]):
        for H in HS:
            b = [sum(1 for i in ids
                     if any(c["success"] for c in rows[i][H][4*j:4*j+4]))
                 for j in range(5)]
            out.setdefault("batches", {}).setdefault(scope, {})[H] = b
            print(f"  {scope:<9}H={H:<3} {b}   /{len(ids)}")
        print()

    # per-source success probability over the five batches, paired bootstrap
    print("per-source batch success probability S_H(x), paired over sources")
    rng = np.random.default_rng(11)
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("marginal", "hard") if by.get(s)]):
        S = {H: np.array([
            sum(1 for j in range(5)
                if any(c["success"] for c in rows[i][H][4*j:4*j+4])) / 5.0
            for i in ids]) for H in HS}
        diff = S[40] - S[24]
        boots = np.array([
            diff[rng.integers(0, len(diff), len(diff))].mean()
            for _ in range(20000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.setdefault("per_source", {})[scope] = {
            "mean_S24": float(S[24].mean()), "mean_S40": float(S[40].mean()),
            "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "sources_better": int((diff > 0).sum()),
            "sources_worse": int((diff < 0).sum())}
        print(f"  {scope:<9} S24={S[24].mean():.3f}  S40={S[40].mean():.3f}  "
              f"diff={diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"better/worse {int((diff>0).sum())}/{int((diff<0).sum())}  "
              f"/{len(ids)}")

    print("\n  CI excluding zero is the structural read; the bootstrap resamples")
    print("  SOURCES, so it generalises over molecules rather than over seeds.")
    for H in HS:
        succ = [c for i in rows for c in rows[i][H] if c["success"]]
        q = sorted(c["terminal_qed"] for c in succ)
        sm = sorted(c["terminal_sim"] for c in succ)
        m = lambda v: v[len(v)//2] if v else float("nan")
        print(f"  H={H}: {len(succ)} successful candidates, median QED "
              f"{m(q):.3f}, median sim {m(sm):.3f}")
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_recede20_v6") -> None:
    o = read.remote(out_dir)
    Path("docs/RECEDE20_AB.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/RECEDE20_AB.json")
