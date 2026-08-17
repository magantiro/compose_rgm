"""Read the N sweep: does a smaller particle budget keep the successes?

Amendment `docs/AMENDMENT_SMC_EFFICIENCY.md`. The endpoint is the SMALLEST
configuration that does not lose against N = 32, never the best-performing one,
and the decision rule is fixed here before the arms are read.

The comparison is paired by construction: every arm runs replicate 0 on all 64
development sources with the same seed as the banked N = 32 baseline, so a
source contributes the same seed to every arm and the arms differ only in
particle count.

Three quantities, in the order they should be read:

  1. SUCCESS      per-arm source success against 19/64 at N = 32, plus the
                  paired breakdown -- retained, lost, and gained. Gains are
                  reported but NEVER used to prefer an arm: the protocol's own
                  warning is that 64 noisy binary source outcomes must not drive
                  selection, and that warning does not stop applying just
                  because the change is in our favour.

  2. COMPUTE      particle transitions and fresh law enumerations per arm, which
                  is what the particle count is being reduced to buy.

  3. EFFICIENCY   successes per million transitions. This is descriptive. It is
                  NOT the decision rule, because an arm that loses half the
                  successes at a tenth of the cost would win on it while failing
                  the actual objective, which is to preserve success.

Read-only.
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
app = modal.App("hphi-n-sweep")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(image=image, cpu=(0.5, 0.5), memory=2048, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def read(arms: dict[str, str], replicate: int) -> dict[str, Any]:
    artifact_volume.reload()

    def load(out_dir: str) -> dict[int, dict]:
        d = Path(RUN_ROOT) / out_dir / "replicates"
        got: dict[int, dict] = {}
        if not d.exists():
            return got
        for f in sorted(d.glob(f"*_smc_{replicate:02d}.json")):
            body = json.loads(f.read_text())
            rec = body["record"]
            got[int(body["index"])] = {
                "success": bool(rec.get("success")),
                "n_transitions": len(rec["transitions"]),
                "n_particles": rec.get("n_particles"),
                "seed": rec.get("seed"),
            }
        return got

    data = {name: load(d) for name, d in arms.items()}
    base_name = "N=32"
    base = data.get(base_name, {})

    print(f"{'arm':<8}{'sources':>9}{'success':>9}{'rate':>8}"
          f"{'transitions':>13}{'succ/Mtrans':>13}")
    summary = {}
    for name, got in data.items():
        if not got:
            print(f"  {name:<6}{'(no records yet)':>40}")
            continue
        succ = sum(1 for v in got.values() if v["success"])
        tr = sum(v["n_transitions"] for v in got.values())
        summary[name] = {"n": len(got), "success": succ, "transitions": tr}
        print(f"  {name:<6}{len(got):>9}{succ:>9}{succ/len(got)*100:>7.1f}%"
              f"{tr:>13,}{succ/max(tr,1)*1e6:>13.1f}")

    print("\nPAIRED against the banked N=32 baseline (same sources, same seeds):")
    for name, got in data.items():
        if name == base_name or not got or not base:
            continue
        shared = sorted(set(got) & set(base))
        retained = [i for i in shared if base[i]["success"] and got[i]["success"]]
        lost = [i for i in shared if base[i]["success"] and not got[i]["success"]]
        gained = [i for i in shared if not base[i]["success"] and got[i]["success"]]
        n32 = sum(1 for i in shared if base[i]["success"])
        # Seed pairing is the premise of the comparison; verify it rather than
        # assume it. A mismatch would mean the arms are not paired at all.
        seed_ok = all(got[i]["seed"] == base[i]["seed"] for i in shared)
        print(f"\n  {name} on {len(shared)} shared sources "
              f"(seeds paired: {'YES' if seed_ok else 'NO -- COMPARISON INVALID'})")
        print(f"     retained {len(retained)}/{n32} of the N=32 successes")
        print(f"     lost     {len(lost)}  {lost if lost else ''}")
        print(f"     gained   {len(gained)}  {gained if gained else ''} "
              f"(reported, NOT used to prefer an arm)")
        if name in summary and base_name in summary:
            saving = 1 - summary[name]["transitions"] / max(
                summary[base_name]["transitions"], 1)
            print(f"     compute  {saving*100:.1f}% fewer transitions")
        summary.setdefault(name, {}).update(
            {"retained": len(retained), "lost": lost, "gained": gained,
             "n32_successes": n32, "seeds_paired": seed_ok})

    print("\nDECISION RULE (fixed in the amendment before reading): choose the "
          "SMALLEST N whose\nsuccess count is not meaningfully below N=32 on "
          "the paired comparison. An arm is\nnever preferred for scoring "
          "higher -- 64 binary outcomes cannot support that.")
    return {"summary": summary, "arms": {k: len(v) for k, v in data.items()}}


@app.local_entrypoint()
def main(replicate: int = 0) -> None:
    arms = {"N=32": "hphi_smc_64", "N=8": "hphi_smc_n08", "N=4": "hphi_smc_n04"}
    out = read.remote(arms, replicate)
    Path("docs/HPHI_N_SWEEP.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/HPHI_N_SWEEP.json")
