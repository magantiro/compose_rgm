"""Does scaffold_hop's plateau come from myopia, or from the reachable geometry?

scaffold_hop was the only PMO task that stalled: top-10 went 0.421 -> 0.485 over
2000 calls, and deeper edits bought nothing despite 400 distinct molecules and
depth 17. Unlike T4's fa7, nothing collapsed, so this is not the lineage
pathology.

The discriminating question, and the ONLY thing that would earn an h-controller
for PMO:

    are there locally mediocre successors with substantially better DOWNSTREAM
    scaffold_hop potential?

For a frozen sample of plateau states, enumerate immediate successors, record
f(y), then run bounded R_theta continuations to depth 2 and 4 and record

    V_L(y) = max over sampled continuations of f(X_L)

and the top-k mean as a robustness check. If ranking by f(y) and ranking by
V_L(y) disagree materially, future value carries information the immediate
score does not, and h is earned. If V_L tops out around 0.48 like f does, the
limit is the reachable support geometry and no controller fixes it.

The pilot did not persist its population, so this regenerates the frontier with
the identical frozen search before probing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

# Build the image here rather than importing it from pmo_pilot_app. Importing a
# sibling app module means that module must also be shipped inside the image,
# which is the exact ModuleNotFoundError this app just crash-looped on and which
# the selection diagnostic hit earlier. Self-contained is safer.
image = (
    _opt_image
    .pip_install("PyTDC==0.3.6", "requests", "fuzzywuzzy", "seaborn", "networkx")
    .add_local_file(ROOT / "modal_apps/genmol_t4_opt_app.py",
                    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)
    .add_local_file(ROOT / "docs/PMO_INIT_BANK.json",
                    str(REMOTE_ROOT / "docs/PMO_INIT_BANK.json"), copy=True)
)

app = modal.App("pmo-scaffold-probe")


def _shim():
    """PyTDC 0.3.6 imports rdkit.six, removed from RDKit years ago."""
    import sys, types
    six = types.ModuleType("rdkit.six")
    six.iteritems = lambda d, **k: iter(d.items())
    six.itervalues = lambda d, **k: iter(d.values())
    six.iterkeys = lambda d, **k: iter(d.keys())
    six.string_types = (str,)
    sys.modules["rdkit.six"] = six
    import rdkit
    rdkit.six = six

WARM_CALLS = 1500      # reach the plateau with the same frozen search
# Sized against measured cost, not guessed. 40x24x8 was 46,080 fiber
# enumerations and ran for hours without finishing one state. 15x12x4 gives 180
# successor observations at 9% of the cost, which is ample to detect whether
# immediate score and downstream value disagree.
N_STATES   = 15        # plateau states probed
N_SUCC     = 12        # immediate successors probed per state
ROLLOUTS   = 4         # R_theta continuations per successor per depth
DEPTHS     = (2, 4)


@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024),
              timeout=8 * 60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe() -> dict[str, Any]:
    import sys, os
    import numpy as np
    _shim(); os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from tdc import Oracle
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rng = np.random.default_rng(20260821)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    o = Oracle(name="scaffold_hop")
    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))
    t0 = time.time()

    F: dict[str, float] = {}
    def f(s):
        if s not in F:
            try:
                F[s] = float(o(s))
            except Exception:
                F[s] = 0.0
        return F[s]

    TT: dict[str, dict] = {}
    def fiber(smi):
        e = TT.setdefault(smi, {})
        if "f" in e:
            return e["f"]
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            e["f"] = {}; return e["f"]
        if not law.marks:
            e["f"] = {}; return e["f"]
        pr = np.array([m.probability for m in law.marks], float)
        out = {}
        for i in np.argsort(-pr)[:APPLY_CAP]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi and y not in out:
                out[y] = float(pr[int(i)])
        e["f"] = out
        return out

    # ---- 1. regenerate the plateau with the same frozen search ----
    init = json.loads((REMOTE_ROOT / "docs/PMO_INIT_BANK.json").read_text())["smiles"]
    pop = {}
    for s in init:
        c = canon(s)
        if c and c not in pop:
            pop[c] = f(c)
    calls = len(pop)
    while calls < WARM_CALLS:
        ranked = sorted(pop, key=lambda k: -pop[k])
        parents = ranked[:6] + list(rng.choice(ranked[6:], size=min(6, max(0, len(ranked)-6)),
                                               replace=False)) if len(ranked) > 6 else ranked
        cand = {}
        for p in parents:
            for y in fiber(p):
                if y not in pop and y not in cand:
                    cand[y] = 1.0
        if not cand:
            break
        ks = list(cand)[:40]
        for y in ks:
            pop[y] = f(y); calls += 1
            if calls >= WARM_CALLS:
                break
        if len(pop) > 400:
            pop = dict(sorted(pop.items(), key=lambda kv: -kv[1])[:400])
    plateau_top10 = float(np.mean(sorted(pop.values(), reverse=True)[:10]))
    print(f"  warmed to {calls} calls, top10 {plateau_top10:.4f}", flush=True)

    # ---- 2. probe the frontier ----
    frontier = sorted(pop, key=lambda k: -pop[k])[:N_STATES]
    rows = []
    for si, x in enumerate(frontier):
        fb = fiber(x)
        if not fb:
            continue
        ys = sorted(fb, key=lambda y: -fb[y])[:N_SUCC]
        for y in ys:
            fy = f(y)
            rec = {"state": si, "f_immediate": fy}
            for L in DEPTHS:
                vals = []
                for _ in range(ROLLOUTS):
                    cur = y
                    for _d in range(L):
                        g = fiber(cur)
                        if not g:
                            break
                        kk = list(g); w = np.array([g[k] for k in kk]); w /= w.sum()
                        cur = kk[int(rng.choice(len(kk), p=w))]
                    vals.append(f(cur))
                rec[f"V{L}_max"] = float(max(vals))
                rec[f"V{L}_top3"] = float(np.mean(sorted(vals, reverse=True)[:3]))
            rows.append(rec)
        if si % 3 == 0:
            print(f"  probed {si+1}/{len(frontier)} states, {len(rows)} successors, "
                  f"{len(F)} oracle values", flush=True)

    # ---- 3. does future value reorder the successors? ----
    out = {"warm_calls": calls, "plateau_top10": plateau_top10,
           "n_states": len(frontier), "n_rows": len(rows),
           "oracle_values_computed": len(F), "rows": rows}
    fi = np.array([r["f_immediate"] for r in rows])
    for L in DEPTHS:
        v = np.array([r[f"V{L}_max"] for r in rows])
        # Spearman without scipy
        def rank(a):
            o_ = np.argsort(a); r_ = np.empty_like(o_, dtype=float); r_[o_] = np.arange(len(a)); return r_
        rf, rv = rank(fi), rank(v)
        rho = float(np.corrcoef(rf, rv)[0, 1]) if len(fi) > 2 else float("nan")
        # the discriminating count: locally worse, downstream better
        inv = 0
        for i in range(len(fi)):
            for j in range(len(fi)):
                if fi[i] > fi[j] and v[i] < v[j] - 0.02:
                    inv += 1
        tot = max(1, len(fi) * (len(fi) - 1))
        out[f"depth{L}"] = {
            "spearman_immediate_vs_future": rho,
            "max_V": float(v.max()), "mean_V": float(v.mean()),
            "max_immediate": float(fi.max()),
            "future_exceeds_immediate_best": bool(v.max() > fi.max() + 0.02),
            "misranked_pair_fraction": inv / tot}
        print(f"  depth {L}: spearman {rho:.3f}  max_V {v.max():.4f} vs max_f {fi.max():.4f}  "
              f"misranked pairs {inv/tot:.3f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    try:
        d = Path("/artifacts/pmo_scaffold_probe"); d.mkdir(parents=True, exist_ok=True)
        (d / "probe.json").write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed: {e}", flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    o = probe.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/pmo_scaffold_probe.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
