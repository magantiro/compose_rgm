"""How often does the ESS trigger actually fire, by stratum?

WE acts ONLY at resampling events. If the hard sources -- the population WE is
meant to rescue -- rarely or never cross ESS < N/2, then WE cannot act on them
at all, and the approach is structurally unable to help where it is needed.
Checked against the banked 109-record ladder cohort before spending on efficacy.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import modal
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image
image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("rs-check")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"

@app.function(image=image, cpu=(0.5,0.5), memory=2048, timeout=20*60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def go(strata: dict):
    # docs/ is NOT mounted into the image -- only src and configs -- so the
    # strata are computed locally and passed in, the same fix the panel file
    # needed earlier.
    artifact_volume.reload()
    strat = lambda i: strata.get(str(i), "hard")
    agg = defaultdict(lambda: {"runs":0,"steps":0,"resamples":0,"runs_with_zero":0,"succ":0})
    d = Path(RUN_ROOT)/"hphi_smc_64"/"replicates"
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text()); rec = doc["record"]
        s = strat(doc["index"]); a = agg[s]
        nres = sum(1 for e in rec["sync"] if e["resampled"])
        a["runs"] += 1; a["steps"] += len(rec["sync"]); a["resamples"] += nres
        a["runs_with_zero"] += int(nres == 0); a["succ"] += int(bool(rec.get("success")))
    print(f"{'stratum':<11}{'runs':>6}{'steps':>7}{'resamples':>11}{'per run':>9}"
          f"{'runs w/ ZERO':>14}{'success':>9}")
    for s in ("reliable","marginal","hard"):
        a = agg.get(s)
        if not a: continue
        print(f"  {s:<9}{a['runs']:>6}{a['steps']:>7}{a['resamples']:>11}"
              f"{a['resamples']/max(a['runs'],1):>9.1f}"
              f"{a['runs_with_zero']:>8}/{a['runs']:<5}{a['succ']:>9}")
    print("\n  WE acts ONLY at resampling events. A stratum whose runs mostly have")
    print("  ZERO resamples is one WE structurally cannot touch.")
    return {k: dict(v) for k, v in agg.items()}

@app.local_entrypoint()
def main():
    root = Path(__file__).resolve().parents[1]
    g = json.loads((root/"docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root/"docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root/"docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strata = {str(i): ("reliable" if i in rel else
                       "marginal" if i in marg else "hard") for i in range(64)}
    o = go.remote(strata)
    Path("docs/RESAMPLE_RATE_BY_STRATUM.json").write_text(json.dumps(o, indent=1))
