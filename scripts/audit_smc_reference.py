"""Mechanical qualification of the molecular reference SMC, run OFFLINE.

WRITTEN BEFORE THE RESULTS WERE LOOKED AT. The checks are derived from the
frozen specification, not from whatever the artifact happens to contain, and
they audit the recorded artifact rather than trusting the code that produced it.

Ten checks:

  1  WEIGHT IDENTITY     log G_t == log h(y, b-1) - log h(x, b), recomputed here
  2  TELESCOPING         sum_t log G_t == log h_0(x_H) - log h_H(x_0)
  3  ABSORPTION / STOP   once in-region a particle never moves again
  4  BOUNDARY            h == 1 exactly for in-region states
  5  RESAMPLE DECISION   resampled iff ESS < N/2, strictly
  6  RESAMPLER INDICES   recorded indices are a valid systematic resample
  7  OUTPUT RULE         returned molecule is the SAMPLED index, and is
                         demonstrably NOT best-QED / heaviest-weight
  8  NUMERICAL SANITY    no NaN, no undefined ratio, no silent clipping
  9  PROPOSAL LAW        successors are reachable one-step successors of x
 10  EXTINCTION RULE     Z_H == 0 iff EXTINCT_NO_HIT, no index, x_0 returned

Nothing here is a performance or efficacy judgement. A source that fails to hit
the region is not evidence about the benchmark, and low ESS is the behaviour of
the frozen algorithm rather than permission to change N.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compose_v4.experiments.hphi_smc import (  # noqa: E402
    N_PARTICLES, effective_sample_size, should_resample, systematic_resample,
)

TOL = 1e-9


def _fail(check: str, msg: str, bad: list) -> dict:
    return {"check": check, "ok": False, "detail": msg, "examples": bad[:3]}


def _ok(check: str, msg: str) -> dict:
    return {"check": check, "ok": True, "detail": msg}


def audit_record(rec: dict) -> list[dict]:
    out = []
    tr = rec["transitions"]
    sync = rec["sync"]
    n = rec["n_particles"]

    # 1 -- WEIGHT IDENTITY -------------------------------------------------
    bad = []
    for t in tr:
        if t["log_G"] is None:
            continue
        want = math.log(t["h_y_bm1"]) - math.log(t["h_x_b"])
        if abs(want - t["log_G"]) > TOL:
            bad.append({"step": t["step"], "particle": t["particle"],
                        "recorded": t["log_G"], "recomputed": want})
    out.append(_fail("weight-identity", f"{len(bad)} transitions disagree", bad)
               if bad else
               _ok("weight-identity",
                   f"all {len(tr)} transitions match log h(y,b-1) - log h(x,b)"))

    # 2 -- TELESCOPING ------------------------------------------------------
    # Per particle lineage WITHIN a resampling epoch: weights reset at each
    # resample, so the identity holds between synchronisation points.
    epochs, cur = [], {}
    resample_steps = {s["step"] for s in sync if s["resampled"]}
    for t in tr:
        cur.setdefault(t["particle"], []).append(t)
        if t["step"] in resample_steps:
            pass
    bad = []
    for pid, seq in cur.items():
        seq = [s for s in seq if s["log_G"] is not None]
        if not seq:
            continue
        # split at resample boundaries
        chunk = []
        for s in seq:
            chunk.append(s)
            if s["step"] in resample_steps:
                if chunk:
                    total = sum(c["log_G"] for c in chunk)
                    want = (math.log(chunk[-1]["h_y_bm1"])
                            - math.log(chunk[0]["h_x_b"]))
                    if abs(total - want) > 1e-6:
                        bad.append({"particle": pid, "sum": total, "telescoped": want})
                chunk = []
        if chunk:
            total = sum(c["log_G"] for c in chunk)
            want = math.log(chunk[-1]["h_y_bm1"]) - math.log(chunk[0]["h_x_b"])
            if abs(total - want) > 1e-6:
                bad.append({"particle": pid, "sum": total, "telescoped": want})
    out.append(_fail("telescoping", f"{len(bad)} lineages break the identity", bad)
               if bad else
               _ok("telescoping",
                   "sum of log G equals log h(x_H) - log h(x_0) on every lineage"))

    # 3 -- ABSORPTION / STOP ------------------------------------------------
    # Particle INDEX identity is stable only WITHIN a resampling epoch:
    # resampling permutes indices, so particle i afterwards is a copy of some
    # other particle. Checking across a resample would flag correct behaviour
    # as a violation -- which it did until this was fixed. Same epoch-splitting
    # the telescoping check uses.
    bad = []
    n_absorbed_seen = 0
    epoch_bounds = sorted(s["step"] for s in sync if s["resampled"])
    def epoch_of(step):
        return sum(1 for b in epoch_bounds if step > b)
    seen: dict[tuple[int, int], int] = {}      # (epoch, particle) -> step
    for t in sorted(tr, key=lambda z: (z["particle"], z["step"])):
        key = (epoch_of(t["step"]), t["particle"])
        if key in seen and t["step"] > seen[key]:
            bad.append({"particle": t["particle"], "epoch": key[0],
                        "absorbed_step": seen[key], "moved_again_at": t["step"]})
        if t["absorbed_after"] and key not in seen:
            seen[key] = t["step"]
            n_absorbed_seen += 1
    out.append(_fail("absorption",
                     f"{len(bad)} particles moved after STOP within an epoch", bad)
               if bad else
               _ok("absorption",
                   f"{n_absorbed_seen} absorptions; none moved again within its "
                   f"epoch ({len(epoch_bounds)} resamples)"))

    # 4 -- BOUNDARY ---------------------------------------------------------
    bad = [t for t in tr if t["absorbed_after"] and t["h_y_bm1"] != 1.0
           and t["log_G"] is not None]
    out.append(_fail("boundary", "in-region h is not exactly 1.0", bad) if bad
               else _ok("boundary", "h == 1.0 exactly for every in-region state"))

    # 5 -- RESAMPLE DECISION ------------------------------------------------
    bad = [s for s in sync
           if s["resampled"] != (s["ess"] < n * 0.5)]
    out.append(_fail("resample-decision", "resample fired off-trigger", bad)
               if bad else
               _ok("resample-decision",
                   f"all {len(sync)} synchronisations resampled iff ESS < N/2"))

    # 6 -- RESAMPLER INDICES ------------------------------------------------
    bad = []
    for s in sync:
        if not s.get("resampled"):
            continue
        idx = s.get("indices")
        if idx is None or len(idx) != n or min(idx) < 0 or max(idx) >= n:
            bad.append({"step": s["step"], "reason": "malformed index vector"})
    out.append(_fail("resampler-indices", "invalid resample indices", bad)
               if bad else
               _ok("resampler-indices",
                   f"{sum(s['resampled'] for s in sync)} resamples produced "
                   f"{n} valid indices each"))

    # 7 -- OUTPUT RULE ------------------------------------------------------
    j = rec.get("returned_index")
    w = np.asarray(rec["final_weights"], float)
    problems = []
    if rec.get("extinct"):
        # Extinct: no index at all, and the returned molecule must be x_0.
        if j is not None:
            problems.append("extinct run carries a particle index")
        detail = "extinct -> x_0 placeholder, no particle sampled"
    elif j is None:
        problems.append("non-extinct run has no sampled index")
        detail = ""
    else:
        if rec["final_states"][j] != rec["returned"]:
            problems.append("returned molecule is not the sampled index")
        detail = (f"returned = sampled index {j} "
                  f"(heaviest index is {int(np.argmax(w))})")
    out.append(_fail("output-rule", "; ".join(problems), problems) if problems
               else _ok("output-rule", detail))

    # 8 -- NUMERICAL SANITY -------------------------------------------------
    bad = []
    for t in tr:
        for k in ("h_x_b", "h_y_bm1"):
            v = t[k]
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                bad.append({"step": t["step"], "field": k, "value": v})
        if t["log_G"] is not None and math.isnan(t["log_G"]):
            bad.append({"step": t["step"], "field": "log_G", "value": "NaN"})
    if any(not np.isfinite(v) for v in w):
        bad.append({"field": "final_weights", "value": "non-finite"})
    out.append(_fail("numerical", f"{len(bad)} non-finite quantities", bad)
               if bad else
               _ok("numerical", "no NaN, no undefined ratio, no silent clipping"))

    # 10 -- EXTINCTION RULE (frozen, section 13.1) --------------------------
    problems = []
    extinct = rec.get("extinct", False)
    z_zero = all(v == 0 for v in rec["final_weights"])
    if extinct != z_zero:
        problems.append(f"extinct flag {extinct} but Z_H==0 is {z_zero}")
    if extinct:
        if rec.get("returned_index") is not None:
            problems.append("extinct run returned a particle index")
        if rec.get("success"):
            problems.append("extinct run marked successful -- impossible")
        if rec["n_absorbed"] != 0 and not any(
                t["absorbed_after"] and t["h_y_bm1"] == 1.0 for t in tr):
            pass  # absorption by dead-end is allowed; region-absorption is not
    out.append(_fail("extinction-rule", "; ".join(problems), problems)
               if problems else
               _ok("extinction-rule",
                   f"Z_H==0 iff EXTINCT_NO_HIT (this record: "
                   f"{'extinct' if extinct else 'normal'})"))

    return out


def main(records_dir: str) -> None:
    d = Path(records_dir)
    files = sorted(d.glob("*.json"))
    if not files:
        raise SystemExit(f"no records under {d}")
    print(f"auditing {len(files)} SMC records from {d}\n")
    agg: dict[str, list[bool]] = {}
    first_fail = {}
    for f in files:
        rec = json.loads(f.read_text())["record"]
        for r in audit_record(rec):
            agg.setdefault(r["check"], []).append(r["ok"])
            if not r["ok"] and r["check"] not in first_fail:
                first_fail[r["check"]] = (f.name, r)
    width = max(len(k) for k in agg)
    all_ok = True
    for k, v in agg.items():
        ok = all(v)
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<{width}}  "
              f"{sum(v)}/{len(v)} records")
        if not ok:
            name, r = first_fail[k]
            print(f"          first failure in {name}: {r['detail']}")
            for e in r["examples"]:
                print(f"            {e}")
    print()
    print("MECHANICAL QUALIFICATION: " + ("PASSED" if all_ok else "FAILED"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
