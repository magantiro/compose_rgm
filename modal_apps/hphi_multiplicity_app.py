"""How much particle multiplicity is there to exploit? A measurement, not a fix.

Multiplicity-aware sampling groups identical particle states so that state-level
preparation -- encode, context, family probabilities -- happens once per UNIQUE
state rather than once per particle. The ideal reduction in that work is exactly
N/K, with K the number of distinct canonical states alive at a step. Nothing
here changes the law: m clones drawing independently from one distribution do
not need the distribution rebuilt m times.

The ceiling is entirely an empirical question, and the banked traces already
answer it: every transition records its step, particle and pre-transition state
x. So K/N is countable with no model, no sampling and no new runs.

WHAT THIS CANNOT SEE. The traces do not record which FAMILY each draw selected,
so the finer (state, family) grouping cannot be counted here -- only bounded by
distinct_families <= min(multiplicity, n_families). Measuring it for real needs
the family logits, i.e. a model run.

The prior is that this splits by stratum: hard runs almost never resample (108
of 111 never did), so they should sit near K = N and gain nothing, while runs
that collapse onto a basin should gain a lot. A number that is good only where
the controller already succeeds is not worth implementing.

CPU ONLY, read-only.
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
app = modal.App("hphi-multiplicity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(image=image, cpu=(2.0, 2.0), memory=8192, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def measure(strata: dict) -> dict[str, Any]:
    artifact_volume.reload()
    d = Path(RUN_ROOT) / "hphi_smc_64" / "replicates"
    files = sorted(d.glob("*.json"))
    print(f"{len(files)} banked runs\n", flush=True)

    # per stratum: totals of N (particle draws) and K (unique states)
    tot: dict = defaultdict(lambda: {"N": 0, "K": 0, "steps": 0})
    bydepth: dict = defaultdict(lambda: defaultdict(lambda: {"N": 0, "K": 0}))
    mult_hist: dict = defaultdict(int)
    # cross-attempt reuse, per source
    seen_by_src: dict = defaultdict(set)
    attempt_order: dict = defaultdict(list)
    reuse = {"states": 0, "already_seen": 0}

    for f in files:
        doc = json.loads(f.read_text())
        idx, rec = doc["index"], doc["record"]
        st = strata.get(str(idx), "hard")
        tr = rec.get("transitions") or []
        if not tr:
            continue
        attempt_order[idx].append(f.name)
        bystep: dict = defaultdict(list)
        for t in tr:
            bystep[t["step"]].append(t["x"])
        for step, xs in bystep.items():
            N, K = len(xs), len(set(xs))
            tot[st]["N"] += N; tot[st]["K"] += K; tot[st]["steps"] += 1
            tot["ALL"]["N"] += N; tot["ALL"]["K"] += K; tot["ALL"]["steps"] += 1
            b = min(step // 4 * 4, 20)
            bydepth[st][b]["N"] += N; bydepth[st][b]["K"] += K
            bydepth["ALL"][b]["N"] += N; bydepth["ALL"][b]["K"] += K
            cnt: dict = defaultdict(int)
            for x in xs:
                cnt[x] += 1
            for m in cnt.values():
                mult_hist[m] += 1
        # cross-attempt: unique states this run visited vs earlier runs of src
        uniq = {t["x"] for t in tr}
        prev = seen_by_src[idx]
        reuse["states"] += len(uniq)
        reuse["already_seen"] += len(uniq & prev)
        seen_by_src[idx] |= uniq

    print(f"{'stratum':<10}{'steps':>8}{'draws N':>10}{'unique K':>10}"
          f"{'K/N':>8}{'ideal N/K':>11}")
    out: dict[str, Any] = {"by_stratum": {}, "by_depth": {}, "multiplicity": {}}
    for st in ("ALL", "reliable", "marginal", "hard"):
        a = tot.get(st)
        if not a or not a["N"]:
            continue
        kn = a["K"] / a["N"]
        out["by_stratum"][st] = {"steps": a["steps"], "N": a["N"], "K": a["K"],
                                 "K_over_N": kn, "ideal_speedup": 1 / kn}
        print(f"{st:<10}{a['steps']:>8}{a['N']:>10,}{a['K']:>10,}"
              f"{kn:>8.3f}{1/kn:>11.2f}x")

    print(f"\n{'depth':<8}" + "".join(f"{s:>12}" for s in
                                      ("ALL", "reliable", "marginal", "hard")))
    depths = sorted({b for s in bydepth for b in bydepth[s]})
    for b in depths:
        row = f"{b:>3}-{b+3:<4}"
        rec_: dict = {}
        for s in ("ALL", "reliable", "marginal", "hard"):
            v = bydepth.get(s, {}).get(b)
            if v and v["N"]:
                rec_[s] = v["K"] / v["N"]
                row += f"{v['K']/v['N']:>12.3f}"
            else:
                row += f"{'-':>12}"
        out["by_depth"][b] = rec_
        print(row)

    tot_groups = sum(mult_hist.values())
    print(f"\nmultiplicity of a state within a step (share of the "
          f"{tot_groups:,} groups):")
    for m in sorted(mult_hist):
        if mult_hist[m] / tot_groups < 0.005 and m > 4:
            continue
        print(f"  m={m:<3} {mult_hist[m]/tot_groups*100:6.2f}%")
    out["multiplicity"] = {str(k): v for k, v in sorted(mult_hist.items())}

    r = reuse["already_seen"] / max(reuse["states"], 1)
    out["cross_attempt_reuse"] = {"unique_states": reuse["states"],
                                  "already_seen_in_earlier_attempt":
                                      reuse["already_seen"], "frac": r}
    print(f"\ncross-attempt: {reuse['already_seen']:,} of {reuse['states']:,} "
          f"unique states ({r*100:.1f}%) were already visited by an earlier "
          f"attempt on the SAME source")
    print("\nN/K is the ceiling on STATE-LEVEL preparation work only. It does "
          "not bound total runtime:\nper-particle work that survives grouping "
          "(the draw itself, h_phi on each distinct successor,\nQED/Tanimoto on "
          "each new molecule) is untouched by it.")
    return out


@app.local_entrypoint()
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    g = json.loads((root / "docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root / "docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root / "docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strata = {str(i): ("reliable" if i in rel else
                       "marginal" if i in marg else "hard") for i in range(64)}
    o = measure.remote(strata)
    Path("docs/PARTICLE_MULTIPLICITY.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/PARTICLE_MULTIPLICITY.json")
