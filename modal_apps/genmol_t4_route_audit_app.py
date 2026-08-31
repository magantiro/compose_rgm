"""Native COMPOSE route audit: the actual operator script to a dev landmark.

Answers, per step of a legal edit route from seed to a strong endpoint:

  * which operator FAMILY the step requires,
  * that mark's RANK and probability mass under the frozen R_theta,
  * whether it survives our APPLY_CAP truncation,
  * what action DOMINATES it instead,
  * QED / SA / similarity along the route.

That separates four candidate failure modes which no docking experiment can
distinguish: missing SUPPORT, reference-law BIAS, wrong operator-family
ALLOCATION, and long-horizon CREDIT.

DEVELOPMENT-LANDMARK USE ONLY. The landmark is a guide for this audit and never
enters the production optimiser, which sees no competitor molecule.

Guidance is SHARED FINGERPRINT BITS WITH THE TARGET, not Tanimoto. Tanimoto
divides by |A union B|, so every intermediate growth step looks like a
regression -- that flaw is what pinned the earlier bidirectional beam at 0.55
while it was supposed to be growing.  Shared-bit count rises monotonically as
target environments are built, which is the behaviour a route guide needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, TIME_POINT, QED_MIN, SA_MAX,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-route-audit")

MAX_STEPS = 34          # >= the 23-24 operations required, with headroom


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 4, max_containers=8, scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def route(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys, time, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig, rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, tgt, delta = job["seed"], job["target"], float(job["delta"])

    def bits(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None, None
        return set(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).GetOnBits()), m

    B_t, m_t = bits(tgt)
    B_0, m_0 = bits(seed)
    tgt_canon = Chem.MolToSmiles(m_t)
    cur = seed
    steps = []
    fam = collections.Counter()
    t0 = time.time()
    for step in range(int(job.get("max_steps", MAX_STEPS))):
        B_c, m_c = bits(cur)
        if B_c is None:
            break
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception as e:
            steps.append({"step": step, "ERROR": f"law failed: {type(e).__name__}"})
            break
        pr = np.array([m.probability for m in law.marks], float)
        order = np.argsort(-pr)
        rank_of = {int(j): r for r, j in enumerate(order)}
        best = None
        for j in order:
            mk = law.marks[int(j)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if not y or y == cur:
                continue
            B_y, m_y = bits(y)
            if B_y is None:
                continue
            shared = len(B_y & B_t)
            if best is None or shared > best[0]:
                best = (shared, y, mk.executor_rule_name, int(j), float(pr[j]), m_y)
        if best is None:
            steps.append({"step": step, "ERROR": "no applicable successor"})
            break
        shared, y, rule, j, p, m_y = best
        cur_shared = len(B_c & B_t)
        top_rule = law.marks[int(order[0])].executor_rule_name
        rk = rank_of[j]
        try:
            q = float(QED.qed(m_y)); sa = float(sascorer.calculateScore(m_y))
        except Exception:
            q = sa = float("nan")
        sim0 = float(DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(m_0, 2, nBits=2048),
            AllChem.GetMorganFingerprintAsBitVect(m_y, 2, nBits=2048)))
        fam[rule] += 1
        steps.append({
            "step": step + 1, "rule": rule, "rank": rk, "prob": p,
            "n_marks": len(law.marks),
            "SURVIVES_APPLY_CAP": bool(rk < APPLY_CAP),
            "dominated_by": top_rule, "top_prob": float(pr[order[0]]),
            "shared_bits": shared, "shared_gain": shared - cur_shared,
            "heavy": m_y.GetNumHeavyAtoms(), "rings": rdMD.CalcNumRings(m_y),
            "qed": round(q, 3), "sa": round(sa, 2), "sim_to_seed": round(sim0, 3),
            "feasible": bool(q >= QED_MIN and sa <= SA_MAX and sim0 >= delta),
        })
        print(f"  [{job['cell']}] step {step+1:>2}: {rule:22s} rank {rk:>4}/"
              f"{len(law.marks):<4} p={p:.2e} cap={'Y' if rk < APPLY_CAP else 'N'} "
              f"shared {cur_shared}->{shared}  heavy {m_y.GetNumHeavyAtoms()} "
              f"rings {rdMD.CalcNumRings(m_y)} sim {sim0:.3f} "
              f"(dominated by {top_rule})", flush=True)
        cur = y
        if Chem.MolToSmiles(m_y) == tgt_canon:
            print(f"  [{job['cell']}] *** REACHED THE LANDMARK in {step+1} steps ***",
                  flush=True)
            break
    out = {"cell": job["cell"], "seed": seed, "target": tgt,
           "reached": Chem.MolToSmiles(Chem.MolFromSmiles(cur)) == tgt_canon,
           "n_steps": len(steps), "family_counts": dict(fam),
           "final_shared": len(bits(cur)[0] & B_t) if bits(cur)[0] else 0,
           "target_bits": len(B_t), "steps": steps, "wall_s": time.time() - t0}
    try:
        d = ARTIFACT_ROOT / "t4_route_audit"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['cell']}.json").write_text(json.dumps(out))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)
    return out


@app.local_entrypoint()
def main(cells: str = "parp1,5ht1b", max_steps: int = MAX_STEPS, out: str = ""):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    win = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())
    want = [c.strip() for c in cells.split(",")]
    jobs = []
    for k, v in win.items():
        if v["target"] not in want:
            continue
        b = sorted(v.get("winners", []), key=lambda r: r["ds"])[:1]
        if not b:
            continue
        jobs.append({"cell": f"{v['target']}_s{v['seed_idx']}_d{v['delta']}",
                     "seed": seeds[v["seed_idx"]]["smiles"],
                     "target": b[0]["smiles"], "delta": v["delta"],
                     "max_steps": max_steps})
    print(f"  route audit on {len(jobs)} development landmarks, "
          f"{max_steps} steps max, ZERO oracle calls", flush=True)
    got = []
    for r in route.map(jobs, order_outputs=False, return_exceptions=True,
                       wrap_returned_exceptions=False):
        if isinstance(r, dict):
            got.append(r)
            print(f"\n  {r['cell']}: reached={r['reached']} steps={r['n_steps']} "
                  f"shared {r['final_shared']}/{r['target_bits']} "
                  f"families={r['family_counts']}", flush=True)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
    if out:
        Path(ROOT / out).write_text(json.dumps(got, indent=2))
        print(f"wrote {ROOT / out}")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 5, max_containers=16, scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def segment_probe(job: dict[str, Any]) -> dict[str, Any]:
    """Does temporal abstraction cross the measured plateau?

    Same total EDIT budget for every k. At each boundary we sample N candidate
    segments of k reference-process edits and keep the best by shared structure
    with the landmark. k = 1 reduces to the greedy audit that cycled.

    Landmark used as a DIAGNOSTIC endpoint only; nothing here reaches the
    production optimiser.
    """
    import os, sys, time, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, tgt = job["seed"], job["target"]
    k = int(job["k"]); budget = int(job["budget"]); nsamp = int(job["nsamp"])
    rng = np.random.default_rng(int(job["seed_rng"]))

    def bits(s):
        m = Chem.MolFromSmiles(s)
        return (set(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                    .GetOnBits()), m) if m else (None, None)
    B_t, _ = bits(tgt)

    def step(smi):
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            return None, None
        pr = np.array([m.probability for m in law.marks], float)
        ix = np.argsort(-pr)[:APPLY_CAP]
        w = pr[ix] / pr[ix].sum()
        for _ in range(6):
            j = int(ix[int(rng.choice(len(ix), p=w))])
            mk = law.marks[j]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if y and y != smi:
                return y, mk.executor_rule_name
        return None, None

    cur = seed
    spent = 0
    trace = []
    fam = collections.Counter()
    uniq = {seed}
    t0 = time.time()
    while spent < budget:
        kk = min(k, budget - spent)
        best = None
        for _ in range(nsamp):
            s = cur; rules = []
            for _h in range(kk):
                y, r_ = step(s)
                if y is None:
                    break
                s = y; rules.append(r_); uniq.add(y)
            if s == cur:
                continue
            B_s, m_s = bits(s)
            if B_s is None:
                continue
            sh = len(B_s & B_t)
            if best is None or sh > best[0]:
                best = (sh, s, rules, m_s)
        spent += kk
        if best is None:
            trace.append({"spent": spent, "shared": None}); continue
        sh, s, rules, m_s = best
        fam.update(rules)
        cur = s
        trace.append({"spent": spent, "shared": sh,
                      "heavy": m_s.GetNumHeavyAtoms(),
                      "rings": rdMD.CalcNumRings(m_s)})
        print(f"  [{job['cell']} k={k}] edits {spent:>3}/{budget}: shared {sh} "
              f"heavy {m_s.GetNumHeavyAtoms()} rings {rdMD.CalcNumRings(m_s)} "
              f"{time.time()-t0:.0f}s", flush=True)
    sh_series = [x["shared"] for x in trace if x["shared"] is not None]
    gains = [b - a for a, b in zip(sh_series, sh_series[1:])]
    ins = fam.get("atom_insert", 0); dele = fam.get("atom_delete", 0)
    out = {"cell": job["cell"], "k": k, "budget": budget,
           "shared_series": sh_series,
           "final_shared": sh_series[-1] if sh_series else 0,
           "target_bits": len(B_t),
           "n_nonpositive_gain": sum(1 for g in gains if g <= 0),
           "n_boundaries": len(gains),
           "insert_delete_cancellation": round(min(ins, dele) / max(ins + dele, 1), 3),
           "n_unique_states": len(uniq),
           "family_counts": dict(fam), "wall_s": time.time() - t0}
    try:
        d = ARTIFACT_ROOT / "t4_segment_probe"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['cell']}_k{k}.json").write_text(json.dumps(out))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)
    return out


@app.local_entrypoint()
def segments(cells: str = "parp1,5ht1b", ks: str = "1,2,4,8",
             budget: int = 32, nsamp: int = 8, out: str = ""):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    win = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())
    want = [c.strip() for c in cells.split(",")]
    jobs = []
    for kk, v in win.items():
        if v["target"] not in want:
            continue
        b = sorted(v.get("winners", []), key=lambda r: r["ds"])[:1]
        if not b:
            continue
        for k in [int(x) for x in ks.split(",")]:
            jobs.append({"cell": f"{v['target']}_s{v['seed_idx']}",
                         "seed": seeds[v["seed_idx"]]["smiles"],
                         "target": b[0]["smiles"], "k": k, "budget": budget,
                         "nsamp": nsamp, "seed_rng": 20260824 + k})
    print(f"  segment gate: {len(jobs)} runs, k in {ks}, EQUAL budget "
          f"{budget} edits, {nsamp} samples/boundary, ZERO oracle calls",
          flush=True)
    got = []
    for r in segment_probe.map(jobs, order_outputs=False, return_exceptions=True,
                               wrap_returned_exceptions=False):
        if isinstance(r, dict):
            got.append(r)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
    print(f"\n{'cell':16s} {'k':>3s} {'final_shared':>12s} {'stalled':>9s} "
          f"{'ins/del cancel':>14s} {'uniq':>6s}")
    for r in sorted(got, key=lambda x: (x["cell"], x["k"])):
        print(f"{r['cell']:16s} {r['k']:>3d} "
              f"{r['final_shared']:>5d}/{r['target_bits']:<6d} "
              f"{r['n_nonpositive_gain']:>4d}/{r['n_boundaries']:<4d} "
              f"{r['insert_delete_cancellation']:>14.3f} {r['n_unique_states']:>6d}")
    if out:
        Path(ROOT / out).write_text(json.dumps(got, indent=2))
        print(f"wrote {ROOT / out}")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 30, scaledown_window=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def seg_diag(smi: str, mode: str = "grow", hops: int = 8) -> dict:
    """WHY does a segment terminate after one hop? Walk it and report per hop:
    law size, per-family availability, how many successors survive props(), and
    the exact failure point."""
    import os, sys, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig, Descriptors
    from rdkit.Chem import rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    FAM = {"grow": ("atom_insert",), "shrink": ("atom_delete",), "mixed": ()}
    fams = FAM.get(mode, ())
    rng = np.random.default_rng(7)
    cur = smi; out = []
    for hop in range(hops):
        rec = {"hop": hop, "smi_len": len(cur)}
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception as e:
            rec["FAIL"] = f"law: {type(e).__name__}: {str(e)[:90]}"
            out.append(rec); break
        pr = np.array([m.probability for m in law.marks], float)
        order = np.argsort(-pr)
        byfam = collections.Counter(m.executor_rule_name for m in law.marks)
        rec["n_marks"] = len(law.marks); rec["by_family"] = dict(byfam)
        cand = [int(j) for j in order
                if (not fams) or law.marks[int(j)].executor_rule_name in fams]
        rec["n_in_family"] = len(cand)
        ok = 0; applied = 0; badprops = 0
        pick = None
        for j in cand[:96]:
            mk = law.marks[j]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if not y or y == cur:
                continue
            applied += 1
            m = Chem.MolFromSmiles(y)
            if m is None:
                badprops += 1; continue
            try:
                QED.qed(m); sascorer.calculateScore(m)
                rdMD.CalcNumHBD(m); rdMD.CalcNumRotatableBonds(m)
                Descriptors.MolLogP(m)
            except Exception:
                badprops += 1; continue
            ok += 1
            if pick is None:
                pick = y
        rec.update({"applied_ok": applied, "props_failed": badprops,
                    "survivors": ok})
        out.append(rec)
        if pick is None:
            rec["FAIL"] = "no successor survived"
            break
        cur = pick
    return {"mode": mode, "trace": out, "final": cur}


@app.local_entrypoint()
def diag(cell: int = 0, mode: str = "grow", hops: int = 8):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    r = seg_diag.remote(seeds[cell]["smiles"], mode, hops)
    print(f"\n=== segment diagnostic: cell {cell} mode={mode} ===")
    for h in r["trace"]:
        print(f"  hop {h['hop']}: marks={h.get('n_marks')} "
              f"in_family={h.get('n_in_family')} applied={h.get('applied_ok')} "
              f"props_failed={h.get('props_failed')} survivors={h.get('survivors')}"
              + (f"   <<< {h['FAIL']}" if h.get('FAIL') else ""))
        if h.get("by_family"):
            print(f"          families: {h['by_family']}")
