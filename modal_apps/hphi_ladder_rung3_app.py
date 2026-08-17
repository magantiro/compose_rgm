"""Read ladder rung 3: did a third attempt convert, and is the remainder a hard core?

Rung 1 solved 19/64 (29.7%). Rung 2 ran only the 45 unsolved and converted 7
(15.6%), giving 26/64 = 40.6%, with 38 sources still 0-of-2 and NO source 2-of-2.
The banked reading called that MIDDLING: "neither the near-zero that would prove
reachability-limited nor the high rate that would prove attempts multiply."

Rung 3 runs the 38 remaining sources once more. The count matters less than the
shape, so this reports three things:

  1. CONVERSION      how many of 38 converted, and the rate against 29.7% and
                     15.6%. A rate holding near 15% means attempts keep paying
                     and the 20-attempt projection's upper end is live; a rate
                     collapsing toward zero means the remainder is a hard core
                     and no attempt budget will reach it.

  2. REACHABILITY    of the failures, how many never had a SINGLE particle enter
                     the region across the whole run. That is the
                     reachability-limited signature -- the process cannot get
                     there within the frozen horizon -- as distinct from a
                     search that looked and lost. Extinction with 0 of 32
                     absorbed is the strong form.

  3. PROXIMITY       terminal QED and similarity of the failures, to separate
                     "far from the region on both axes" from "close on one and
                     blocked on the other". A cohort sitting just under the QED
                     threshold at adequate similarity is a different problem
                     from one that never approached either.

Read-only. Reports; it does not amend the ladder artifact.
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
app = modal.App("hphi-ladder-rung3")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
REGION = (0.90, 0.40)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str, targets: list[int], replicate: int) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir / "replicates"

    rows = []
    for idx in targets:
        f = d / f"{idx:03d}_smc_{replicate:02d}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())["record"]
        tr = rec["transitions"]
        # In-region is h == 1.0 EXACTLY: the frozen terminal boundary.
        ever_in_region = any(t.get("h_y_bm1") == 1.0 for t in tr)
        rows.append({
            "index": idx,
            "success": bool(rec.get("success")),
            "ever_in_region": ever_in_region,
            "n_absorbed": int(rec.get("n_absorbed", 0)),
            "terminal_qed": rec.get("terminal_qed"),
            "terminal_sim": rec.get("terminal_sim"),
            "n_transitions": len(tr),
            "n_particles": rec.get("n_particles"),
        })

    got = len(rows)
    conv = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    never = [r for r in fail if not r["ever_in_region"]]
    zero_abs = [r for r in fail if r["n_absorbed"] == 0]

    print(f"rung 3: {got}/{len(targets)} of the 0-of-2 sources reported\n")
    print(f"  CONVERSION   {len(conv)}/{got} = "
          f"{len(conv)/max(got,1)*100:.1f}%")
    print(f"     rung 1 29.7% (19/64)   rung 2 15.6% (7/45)")
    if conv:
        print(f"     converted: {sorted(r['index'] for r in conv)}")

    print(f"\n  REACHABILITY  of {len(fail)} failures:")
    print(f"     never had ANY particle enter the region : {len(never)} "
          f"({len(never)/max(len(fail),1)*100:.0f}%)")
    print(f"     zero of N particles absorbed at all     : {len(zero_abs)} "
          f"({len(zero_abs)/max(len(fail),1)*100:.0f}%)")

    if fail:
        q = sorted(r["terminal_qed"] for r in fail if r["terminal_qed"] is not None)
        s = sorted(r["terminal_sim"] for r in fail if r["terminal_sim"] is not None)
        med = lambda v: v[len(v) // 2] if v else float("nan")  # noqa: E731
        print(f"\n  PROXIMITY of failures (region needs QED>={REGION[0]}, "
              f"sim>={REGION[1]}):")
        print(f"     terminal QED  min {q[0]:.3f}  median {med(q):.3f}  "
              f"max {q[-1]:.3f}")
        print(f"     terminal sim  min {s[0]:.3f}  median {med(s):.3f}  "
              f"max {s[-1]:.3f}")
        near = [r for r in fail if (r["terminal_qed"] or 0) >= 0.85]
        print(f"     failures with terminal QED >= 0.85: {len(near)}")
        print("\n     NOTE: terminal QED of a failed run is the molecule the "
              "sampler happened to\n     return, which under the extinction "
              "rule is the SOURCE itself. It bounds\n     nothing about how "
              "close the run came; the reachability counts above do.")

    cumulative = 26 + len(conv)
    print(f"\n  CUMULATIVE COVERAGE  {cumulative}/64 = "
          f"{cumulative/64*100:.1f}%   (was 40.6% after two attempts)")
    return {"n_reported": got, "n_targets": len(targets),
            "conversions": sorted(r["index"] for r in conv),
            "conversion_rate": len(conv) / max(got, 1),
            "n_failures": len(fail), "never_in_region": len(never),
            "zero_absorbed": len(zero_abs),
            "cumulative_solved": cumulative, "rows": rows}


@app.local_entrypoint()
def main(out_dir: str = "hphi_smc_64", replicate: int = 2) -> None:
    targets = [0, 1, 2, 4, 6, 8, 12, 13, 16, 18, 20, 21, 23, 25, 30, 31, 32, 33,
               34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 55, 56, 57, 58,
               59, 61, 62, 63]
    out = read.remote(out_dir, targets, replicate)
    Path("docs/HPHI_LADDER_RUNG3.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote docs/HPHI_LADDER_RUNG3.json")
