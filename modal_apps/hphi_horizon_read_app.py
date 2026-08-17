"""Horizon diagnostic readout: does more budget produce target CONTACT?

Primary readout is CONTACT -- any particle entering the region -- not source
success, because contact is the reachability signal while success additionally
depends on the terminal sampling draw. Both are exact regardless of the twist,
since the terminal boundary h_0 = 1[x in B] does not involve h_phi.

READ THE ASYMMETRY. h_phi's budget is a 25-slot one-hot capped at 24, so the
extended arms present budget = 24 for their first H - 24 steps and run a
MISCALIBRATED twist. A rise with H is therefore conclusive -- it happened
despite a handicapped controller -- while a flat result cannot separate "horizon
does not matter" from "h_phi cannot steer past its training range".
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
app = modal.App("hphi-horizon-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
HS = (24, 32, 40)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    per = defaultdict(dict)
    strat = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        strat[r["index"]] = r["stratum"]
        per[r["index"]][r["horizon"]] = r["arms"]["restart"]
    rows = {i: v for i, v in per.items() if all(h in v for h in HS)}
    print(f"{len(rows)} sources with all three horizons "
          f"({len(per)-len(rows)} incomplete, excluded)\n")

    by = defaultdict(list)
    for i in rows:
        by[strat[i]].append(i)

    out: dict[str, Any] = {"n": len(rows), "by_stratum": {}}
    print(f"{'scope':<11}{'H':>5}{'contact runs':>14}{'contact particles':>19}"
          f"{'sources w/ contact':>20}{'success':>9}{'work':>12}")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("marginal", "hard") if by.get(s)]):
        for H in HS:
            runs = [c for i in ids for c in rows[i][H]["candidates"]]
            cr = sum(1 for c in runs if c.get("contact", 0) > 0)
            cp = sum(c.get("contact", 0) for c in runs)
            sc = sum(1 for i in ids if rows[i][H]["success"])
            wk = sum(rows[i][H]["work_transitions"] for i in ids)
            src_contact = sum(
                1 for i in ids
                if any(c.get("contact", 0) > 0 for c in rows[i][H]["candidates"]))
            out["by_stratum"].setdefault(scope, {})[H] = {
                "contact_runs": cr, "n_runs": len(runs),
                "contact_particles": cp, "sources_with_contact": src_contact,
                "n_sources": len(ids), "success": sc, "work": wk}
            print(f"  {scope:<9}{H:>5}{cr:>8}/{len(runs):<5}{cp:>19}"
                  f"{src_contact:>14}/{len(ids):<5}{sc:>9}{wk:>12,}")
        print()
    print("  CONTACT is the reachability signal and is exact. A rise with H is")
    print("  conclusive; a flat result is confounded by the clamped twist.")
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_horizon_v4") -> None:
    o = read.remote(out_dir)
    Path("docs/HORIZON_DIAGNOSTIC.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/HORIZON_DIAGNOSTIC.json")
