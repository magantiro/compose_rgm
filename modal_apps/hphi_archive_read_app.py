"""Read the archive A/B. Reported BY STRATUM, never pooled.

Pooling would let the four reliable sources -- which already succeed under both
arms -- flatter a null on the strata that actually carry the hypothesis. The
decision rule was fixed before the data: kill or redesign unless the archive
beats four independent restarts on MARGINAL plus HARD.

Rung 3 also constrains how a null on HARD may be read: 37 of 37 hard-source
failures had ZERO particles enter the region under this controller and horizon,
so if those sources are unreachable within H=24 then no search policy reaches
them and the archive losing there says nothing about archives. A null on
MARGINAL is the strong evidence, because those sources are demonstrably
reachable.
"""

from __future__ import annotations

import json
from collections import defaultdict
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


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    rows = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    print(f"{len(rows)} sources\n")

    order = ["reliable", "marginal", "hard"]
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)

    print(f"{'stratum':<11}{'n':>3}{'base':>7}{'arch':>7}"
          f"{'base work':>11}{'arch work':>11}{'base div':>10}{'arch div':>10}")
    out: dict[str, Any] = {"strata": {}}
    for st in order:
        rs = by.get(st, [])
        if not rs:
            continue
        b = sum(r["arms"]["baseline"]["success"] for r in rs)
        a = sum(r["arms"]["archive"]["success"] for r in rs)
        bw = sum(r["arms"]["baseline"]["work_transitions"] for r in rs)
        aw = sum(r["arms"]["archive"]["work_transitions"] for r in rs)
        bd = sum(r["arms"]["baseline"]["distinct_returned"] for r in rs) / len(rs)
        ad = sum(r["arms"]["archive"]["distinct_returned"] for r in rs) / len(rs)
        out["strata"][st] = {"n": len(rs), "baseline": b, "archive": a,
                             "baseline_work": bw, "archive_work": aw,
                             "baseline_diversity": bd, "archive_diversity": ad,
                             "rescued": [r["index"] for r in rs
                                         if r["arms"]["archive"]["success"]
                                         and not r["arms"]["baseline"]["success"]],
                             "lost": [r["index"] for r in rs
                                      if r["arms"]["baseline"]["success"]
                                      and not r["arms"]["archive"]["success"]]}
        print(f"  {st:<9}{len(rs):>3}{b:>7}{a:>7}{bw:>11,}{aw:>11,}"
              f"{bd:>10.2f}{ad:>10.2f}")

    mh = [st for st in ("marginal", "hard") if st in out["strata"]]
    b_mh = sum(out["strata"][s]["baseline"] for s in mh)
    a_mh = sum(out["strata"][s]["archive"] for s in mh)
    n_mh = sum(out["strata"][s]["n"] for s in mh)
    print(f"\n  DECISION STRATA (marginal + hard), n={n_mh}")
    print(f"    baseline {b_mh}/{n_mh}    archive {a_mh}/{n_mh}")
    resc = sorted(sum((out["strata"][s]["rescued"] for s in mh), []))
    lost = sorted(sum((out["strata"][s]["lost"] for s in mh), []))
    print(f"    rescued by archive: {resc}")
    print(f"    lost by archive:    {lost}")

    print(f"\n  success by candidate index (cumulative, all {len(rows)} sources)")
    for arm in ("baseline", "archive"):
        cum = []
        for k in range(4):
            cum.append(sum(
                1 for r in rows
                if any(c["success"] for c in r["arms"][arm]["candidates"][:k + 1])))
        print(f"    {arm:<9} {cum}")

    tw_b = sum(r["arms"]["baseline"]["work_transitions"] for r in rows)
    tw_a = sum(r["arms"]["archive"]["work_transitions"] for r in rows)
    print(f"\n  total work  baseline {tw_b:,}   archive {tw_a:,}"
          f"   ({tw_a/max(tw_b,1)*100:.0f}% of baseline)")
    print("\n  An archive win on LESS work is the strong form; on more work it")
    print("  would need the extra search accounted for before it counts.")
    out["totals"] = {"baseline_work": tw_b, "archive_work": tw_a,
                     "n_sources": len(rows)}
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_archive_v1") -> None:
    o = read.remote(out_dir)
    Path("docs/ARCHIVE_AB_RESULT.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/ARCHIVE_AB_RESULT.json")
