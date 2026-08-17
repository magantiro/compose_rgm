"""Three-arm mechanism test: restart vs random reuse vs future-aware reuse.

    restart   every candidate from x0
    random    branch from a UNIFORMLY chosen reusable archived state
    hphi      branch from the reusable archived state maximising h_phi(x, H-d)

The decomposition is the point. hphi > random > restart would mean BOTH that
remembering valid intermediates helps AND that h_phi identifies which ones are
worth continuing from. random ~ hphi > restart would mean the win is simply "do
not throw away explored molecular states". hphi > random ~ restart would make it
specifically about future-aware frontier selection.

Reported BY STRATUM as well as overall, because a gain on sources that already
succeed is nearly worthless and pooling lets them flatter a null elsewhere.

Rung 3 bounds how a null on HARD may be read: 37 of 37 hard-source failures had
ZERO particles enter the region under this controller and horizon, so if those
sources are unreachable within H=24 no policy reaches them. A null on MARGINAL
is the strong evidence, since those sources are demonstrably reachable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-archive-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
ARMS = ("restart", "random", "hphi")
KS = (1, 2, 3, 4, 5, 8)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    rows = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    rows = [r for r in rows if len(r.get("arms", {})) == len(ARMS)]
    print(f"{len(rows)} complete sources\n")
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)

    def solved_by(r, arm, k):
        return any(c["success"] for c in r["arms"][arm]["candidates"][:k])

    out: dict[str, Any] = {"n_sources": len(rows), "cumulative": {},
                           "paired": {}, "work": {}, "diversity": {},
                           "branches": {}}

    print(f"{'scope':<12}{'arm':<9}" + "".join(f"{'k=' + str(k):>6}" for k in KS))
    for scope, rs in ([("ALL", rows)]
                      + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                         if by.get(s)]):
        for arm in ARMS:
            cum = [sum(1 for r in rs if solved_by(r, arm, k)) for k in KS]
            out["cumulative"].setdefault(scope, {})[arm] = cum
            print(f"  {scope:<10}{arm:<9}"
                  + "".join(f"{c:>6}" for c in cum) + f"   / {len(rs)}")
        print()

    print("paired at k=8 (same source, both arms)")
    for a, b in (("hphi", "restart"), ("random", "restart"), ("hphi", "random")):
        wins = [r["index"] for r in rows
                if solved_by(r, a, 8) and not solved_by(r, b, 8)]
        loss = [r["index"] for r in rows
                if solved_by(r, b, 8) and not solved_by(r, a, 8)]
        out["paired"][f"{a}_vs_{b}"] = {"wins": wins, "losses": loss}
        print(f"  {a:>7} vs {b:<8} {len(wins):>2} wins {len(loss):>2} losses"
              f"   won:{wins} lost:{loss}")

    print(f"\n{'arm':<9}{'work':>12}{'vs restart':>12}{'diversity':>11}")
    base_w = sum(r["arms"]["restart"]["work_transitions"] for r in rows) or 1
    for arm in ARMS:
        w = sum(r["arms"][arm]["work_transitions"] for r in rows)
        dv = sum(r["arms"][arm]["distinct_returned"]
                 for r in rows) / max(len(rows), 1)
        out["work"][arm] = w
        out["diversity"][arm] = dv
        print(f"  {arm:<7}{w:>12,}{w / base_w * 100:>11.0f}%{dv:>11.2f}")

    print(f"\n{'arm':<9}{'branches':>10}{'from x0':>10}{'pct x0':>9}"
          f"   depth histogram")
    for arm in ("random", "hphi"):
        bl = [b for r in rows for b in r["arms"][arm]["branches"]]
        x0 = sum(1 for b in bl if b.get("from_source"))
        depths = Counter(b["depth"] for b in bl)
        top = " ".join(f"d{k}:{v}" for k, v in sorted(depths.items())[:12])
        out["branches"][arm] = {"n": len(bl), "from_x0": x0,
                                "depths": {str(k): v for k, v in depths.items()}}
        print(f"  {arm:<7}{len(bl):>10}{x0:>10}"
              f"{x0 / max(len(bl), 1) * 100:>8.0f}%   {top}")
    print("\n  'from x0' is the adaptive test: a policy that never FORGETS should")
    print("  sometimes CHOOSE to restart, rather than be forced to branch.")
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_archive_v2") -> None:
    o = read.remote(out_dir)
    Path("docs/ARCHIVE_3ARM_RESULT.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/ARCHIVE_3ARM_RESULT.json")
