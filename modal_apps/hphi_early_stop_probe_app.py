"""Is `success` equivalent to `some particle reached B`, and what would STOP save?

The claim, derived from the frozen controller rather than from data: the exact
terminal potential is h_0(x) = 1[x in B], so at the final step every particle
outside the region takes log-weight -inf, and dead-end absorptions were already
-inf. The terminal measure therefore has support ONLY on particles that reached
B, and sampling from it returns a B-molecule whenever any particle got there.

    success  <=>  at least one particle reached B

If that holds, terminating the whole run at the FIRST target hit preserves the
binary benchmark event exactly, while changing which molecule is returned (first
hitter instead of a sampled hitter) and therefore the secondary terminal QED and
similarity.

A proof read off the source is exactly the kind of thing that has been wrong
before tonight, so it is checked against all 109 banked records, and the saving
is measured rather than assumed:

  1. EQUIVALENCE  success == (any transition with h_y_bm1 == 1.0). In-region is
     identified by h == 1.0 exactly, which the frozen boundary check already
     guarantees and the reference audit independently verified.
  2. SAVING       transitions actually executed, versus transitions up to and
     including the first in-region hit. Reported over successful records, where
     STOP can fire at all, and over the whole cohort.

Read-only. Persists nothing and changes no frozen artifact.
"""

from __future__ import annotations

import json
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
app = modal.App("hphi-early-stop-probe")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir / "replicates"
    files = sorted(d.glob("*.json"))
    print(f"{len(files)} banked records in {out_dir}\n")

    rows, mismatches = [], []
    for f in files:
        body = json.loads(f.read_text())
        rec = body["record"]
        tr = rec["transitions"]
        # In-region is h == 1.0 EXACTLY -- the frozen boundary, not a threshold.
        hits = [t for t in tr if t.get("h_y_bm1") == 1.0]
        reached = bool(hits)
        success = bool(rec.get("success"))
        if reached != success:
            mismatches.append({"file": f.name, "success": success,
                               "reached": reached, "n_hits": len(hits)})

        # Transitions up to and including the first in-region hit. The record
        # lists transitions in execution order within a step, so ordering by
        # (step, index-in-list) is the order the controller actually ran them.
        n_total = len(tr)
        n_to_first = n_total
        if hits:
            first = min(t["step"] for t in hits)
            n_to_first = sum(1 for t in tr if t["step"] <= first)
        rows.append({"file": f.name, "success": success, "reached": reached,
                     "n_transitions": n_total, "n_to_first_hit": n_to_first,
                     "first_hit_step": (min(t["step"] for t in hits)
                                        if hits else None),
                     "horizon": rec.get("horizon")})

    ok = not mismatches
    succ = [r for r in rows if r["success"]]
    tot_all = sum(r["n_transitions"] for r in rows)
    stop_all = sum(r["n_to_first_hit"] for r in rows)
    tot_s = sum(r["n_transitions"] for r in succ)
    stop_s = sum(r["n_to_first_hit"] for r in succ)

    print(f"EQUIVALENCE  success == (some particle reached B): "
          f"{'HOLDS' if ok else 'VIOLATED'} on {len(rows)} records")
    for m in mismatches[:5]:
        print(f"   MISMATCH {m}")
    if succ:
        steps = sorted(r["first_hit_step"] for r in succ)
        med = steps[len(steps) // 2]
        print(f"\nfirst hit step on successful records: min {steps[0]}, "
              f"median {med}, max {steps[-1]}  (horizon {rows[0]['horizon']})")
        print(f"\nSUCCESSFUL RECORDS ({len(succ)})")
        print(f"  transitions executed   {tot_s:>8}")
        print(f"  with STOP at first hit {stop_s:>8}"
              f"   saving {(1-stop_s/max(tot_s,1))*100:5.1f}%")
    print(f"\nWHOLE COHORT ({len(rows)})")
    print(f"  transitions executed   {tot_all:>8}")
    print(f"  with STOP at first hit {stop_all:>8}"
          f"   saving {(1-stop_all/max(tot_all,1))*100:5.1f}%")
    print("\nSTOP cannot help a failed run: with no hit there is nothing to "
          "stop at, so\nthe cohort figure is diluted by failures and the "
          "successful-record figure is\nthe one that describes the mechanism.")

    return {"equivalence_holds": ok, "mismatches": mismatches,
            "n_records": len(rows), "n_success": len(succ),
            "cohort_saving": 1 - stop_all / max(tot_all, 1),
            "success_saving": 1 - stop_s / max(tot_s, 1),
            "rows": rows}


@app.local_entrypoint()
def main(out_dir: str = "hphi_smc_64") -> None:
    out = probe.remote(out_dir)
    Path("docs/EARLY_STOP_PROBE.json").write_text(json.dumps(out, indent=1))
    print(f"\nequivalence {'HOLDS' if out['equivalence_holds'] else 'VIOLATED'}"
          f"   saving on successful records "
          f"{out['success_saving']*100:.1f}%")
