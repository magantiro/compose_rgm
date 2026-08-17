"""Restart vs forced-diverse branch points, 64 sources. One question, powered.

This tests the SINGLE ingredient that differed between the positive v1 and the
negative v2: v1 required each candidate to launch from a DISTINCT archived
state, v2 allowed repeats and collapsed onto one basin. So the claim under test
is about covering distinct search basins, not about intermediates being
intrinsically valuable.

Powered deliberately. v1 was 12 sources with McNemar p ~ 0.125 -- an effect the
sample could not carry, and v2 duly reversed it. 64 sources with paired
source-level outcomes is what settles it, and the exact McNemar p is reported
rather than a difference of proportions, because the arms share sources.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any
import modal
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-diverse-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
ARMS = ("restart", "diverse")
KS = (1, 2, 3, 4)


def mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial on the DISCORDANT pairs only.

    Concordant sources carry no information about a difference, so pooling them
    would understate the evidence; discordant pairs are the whole test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    per: dict[int, dict] = defaultdict(dict)
    strat: dict[int, str] = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        strat[r["index"]] = r["stratum"]
        for arm, v in r["arms"].items():
            per[r["index"]][arm] = v
    rows = {i: v for i, v in per.items() if len(v) == len(ARMS)}
    print(f"{len(rows)} sources with BOTH arms "
          f"({len(per) - len(rows)} incomplete, excluded)\n")

    def solved(i, arm, k):
        return any(c["success"] for c in rows[i][arm]["candidates"][:k])

    out: dict[str, Any] = {"n": len(rows), "cumulative": {}, "paired": {}}
    by = defaultdict(list)
    for i in rows:
        by[strat[i]].append(i)

    print(f"{'scope':<12}{'arm':<10}" + "".join(f"{'k='+str(k):>6}" for k in KS))
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        for arm in ARMS:
            cum = [sum(1 for i in ids if solved(i, arm, k)) for k in KS]
            out["cumulative"].setdefault(scope, {})[arm] = cum + [len(ids)]
            print(f"  {scope:<10}{arm:<10}" + "".join(f"{c:>6}" for c in cum)
                  + f"   / {len(ids)}")
        print()

    print("PAIRED at k=4 (exact McNemar on discordant pairs)")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        win = [i for i in ids if solved(i, "diverse", 4) and not solved(i, "restart", 4)]
        los = [i for i in ids if solved(i, "restart", 4) and not solved(i, "diverse", 4)]
        p = mcnemar_two_sided(len(win), len(los))
        out["paired"][scope] = {"diverse_only": win, "restart_only": los, "p": p}
        print(f"  {scope:<10} diverse-only {len(win):>2}   restart-only {len(los):>2}"
              f"   p={p:.4f}   won:{win[:8]} lost:{los[:8]}")

    dec = [i for i in rows if strat[i] in ("marginal", "hard")]
    w = sum(1 for i in dec if solved(i, "diverse", 4) and not solved(i, "restart", 4))
    l = sum(1 for i in dec if solved(i, "restart", 4) and not solved(i, "diverse", 4))
    print(f"\n  DECISION STRATA (marginal+hard, n={len(dec)}): "
          f"diverse-only {w}, restart-only {l}, p={mcnemar_two_sided(w, l):.4f}")

    print(f"\n{'arm':<10}{'work':>12}{'vs restart':>12}{'diversity':>11}")
    bw = sum(rows[i]["restart"]["work_transitions"] for i in rows) or 1
    for arm in ARMS:
        wk = sum(rows[i][arm]["work_transitions"] for i in rows)
        dv = sum(rows[i][arm]["distinct_returned"] for i in rows) / len(rows)
        print(f"  {arm:<8}{wk:>12,}{wk/bw*100:>11.0f}%{dv:>11.2f}")
        out.setdefault("work", {})[arm] = wk
        out.setdefault("diversity", {})[arm] = dv

    bl = [b for i in rows for b in rows[i]["diverse"]["branches"]]
    dh = Counter(b["depth"] for b in bl)
    print(f"\n  diverse branches {len(bl)}   from x0 "
          f"{sum(1 for b in bl if b.get('from_source'))}")
    print("  depth histogram " + " ".join(f"d{k}:{v}" for k, v in sorted(dh.items())[:14]))
    out["branch_depths"] = {str(k): v for k, v in dh.items()}
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_diverse_v3") -> None:
    o = read.remote(out_dir)
    Path("docs/DIVERSE_64_RESULT.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/DIVERSE_64_RESULT.json")
