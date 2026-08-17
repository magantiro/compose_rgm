"""Slack-stratified fixed-schedule WE vs ordinary SMC, paired, H=24.

Fixes both failures found today: resampling fires on a FIXED schedule at
remaining budget 12/8/4, because 108 of 111 hard runs never crossed ESS < N/2;
and strata are formed on TARGET SLACK, because mining gave slack AUC 0.812 at
12 steps out against h_phi's 0.556.

Primary readout is marginal+hard. Reliable sources are checked only to confirm
the new resampler does not BREAK what already works -- a gain there would not
count, and a loss there would.

Gate, fixed before reading: no clear directional improvement on marginal+hard
means rare-event controller work stops. No bin or cadence tuning.
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
app = modal.App("hphi-slackwe-read")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
KS = (1, 2, 3, 4)


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(out_dir: str) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / out_dir
    per, strat = defaultdict(dict), {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        strat[r["index"]] = r["stratum"]
        per[r["index"]]["we" if r.get("we") else "smc"] = r["arms"]["restart"]
    rows = {i: v for i, v in per.items() if len(v) == 2}
    print(f"{len(rows)} sources with both arms ({len(per)-len(rows)} partial)\n")
    by = defaultdict(list)
    for i in rows:
        by[strat[i]].append(i)

    def sat(i, a, k):
        return any(c["success"] for c in rows[i][a]["candidates"][:k])

    out: dict[str, Any] = {"n": len(rows)}
    print(f"{'scope':<11}{'arm':<5}" + "".join(f"{'@'+str(k):>5}" for k in KS)
          + f"{'contact runs':>14}{'we events':>11}{'work':>11}")
    for scope, ids in ([("ALL", list(rows))]
                       + [(s, by[s]) for s in ("reliable", "marginal", "hard")
                          if by.get(s)]):
        for a in ("smc", "we"):
            cum = [sum(1 for i in ids if sat(i, a, k)) for k in KS]
            runs = [c for i in ids for c in rows[i][a]["candidates"]]
            cr = sum(1 for c in runs if c.get("contact", 0) > 0)
            ev = sum(c.get("we_events", 0) for c in runs)
            wk = sum(rows[i][a]["work_transitions"] for i in ids)
            out.setdefault("cum", {}).setdefault(scope, {})[a] = cum + [len(ids)]
            print(f"  {scope:<9}{a:<5}" + "".join(f"{c:>5}" for c in cum)
                  + f"{cr:>8}/{len(runs):<5}{ev:>11}{wk:>11,}  /{len(ids)}")
        print()

    dec = [i for i in rows if strat[i] in ("marginal", "hard")]
    w = [i for i in dec if sat(i, "we", 4) and not sat(i, "smc", 4)]
    l = [i for i in dec if sat(i, "smc", 4) and not sat(i, "we", 4)]
    cw = sum(1 for i in dec for c in rows[i]["we"]["candidates"] if c.get("contact", 0) > 0)
    cs = sum(1 for i in dec for c in rows[i]["smc"]["candidates"] if c.get("contact", 0) > 0)
    print(f"  DECISION (marginal+hard, n={len(dec)}): WE-only {len(w)} {w}, "
          f"SMC-only {len(l)} {l}")
    print(f"  contact runs on decision strata: SMC {cs}, WE {cw}")
    out["decision"] = {"we_only": w, "smc_only": l,
                       "contact_smc": cs, "contact_we": cw, "n": len(dec)}
    rel = [i for i in rows if strat[i] == "reliable"]
    broke = [i for i in rel if sat(i, "smc", 4) and not sat(i, "we", 4)]
    print(f"  reliable sources BROKEN by WE: {len(broke)} {broke}")
    out["reliable_broken"] = broke
    print("\n  Gate: no clear directional gain on marginal+hard -> stop "
          "rare-event work.\n  Do not tune bins or cadence.")
    return out


@app.local_entrypoint()
def main(out_dir: str = "hphi_slackwe_v9") -> None:
    o = read.remote(out_dir)
    Path("docs/SLACK_WE_RESULT.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/SLACK_WE_RESULT.json")
