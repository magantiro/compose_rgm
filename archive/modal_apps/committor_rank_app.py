"""Held-out plateau-ranking test for the fitted structural committor.

The committor is only useful if it orders successors where the greedy
connectivity potential cannot. That is exactly the plateau: states whose
admissible successors all leave the connectivity deficit d_old unchanged, so
psi gives them identical scores and the walk has no gradient to follow.

The test asks one question on regions the fit never saw:

    at a plateau, does h_phi rank a successor that a short goal search can
    still complete ABOVE one that it cannot?

Ground truth comes from running the SAME frozen search (depth, beam and
ordering unchanged) forward from each successor, so the label is "reaches a
structural terminal within the remaining horizon", not an opinion.

Held out at the MOLECULE level: the fit trains on dev seeds 6..22, this runs on
0..6. A region here can never be a region there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image
from modal_apps.committor_bellman_app import (
    CANONICAL_SLOTS, OUT_DIR, TIME_POINT, _build_net, _runtime,
    enumerate_training_regions,
)

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("committor-rank")
MEM_MIB = int(3 * 1024)


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def rank_region(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    import torch
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import Region
    from compose_v4.control import region_rewrite as RR
    from compose_v4.gates.med_chem_gate import is_valid

    artifact_volume.reload()
    ckpt = torch.load(f"{OUT_DIR}/committor_bellman_v1.pt", map_location="cpu")
    n_feat, max_b = int(ckpt["n_features"]), int(ckpt["max_b"])
    net = _build_net(torch, n_feat)          # same constructor as the fit
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="rank", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    cache: dict = {}
    horizon = int(job.get("horizon", 5))

    def enum_fn(st):
        k = canonical_state_key(st)
        if k not in cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def step(st, lin, j):
        fams, acts, _ = enum_fn(st)
        try:
            y = system.apply(st, fams[j], acts[j])
        except Exception:
            return None, None
        if y is None:
            return None, None
        k = canonical_state_key(y)
        if not k or not is_valid(k):
            return None, None
        if not RR.context_preserved(st0, y, ctx.frozen, ctx.terminal_context_slots):
            return None, None
        if not RR.graph_connected(y):
            return None, None
        return y, lin.observe(fams[j], acts[j])

    def h_phi(st, lin, budget):
        sh = RR.structural_features_shared(st, ctx, lin, old_ids)
        f = RR.features_from_shared(sh, int(budget), "establishment")
        return float(torch.sigmoid(
            net(torch.tensor([f], dtype=torch.float32))).item())

    def completes(st, lin, depth):
        """Frozen forward search: can a terminal still be reached in `depth`?"""
        frontier = [(st, lin)]
        for _ in range(depth):
            nxt = []
            for s_, l_ in frontier:
                fams, acts, probs = enum_fn(s_)
                idx, _w = RR.admissible_indices(fams, acts, ctx)
                per: dict = {}
                for j in sorted(idx, key=lambda k_: -float(probs[k_])):
                    f = fams[j]
                    if per.get(f, 0) >= 4:
                        continue
                    y, l2 = step(s_, l_, j)
                    if y is None:
                        continue
                    per[f] = per.get(f, 0) + 1
                    if RR.establishment_terminal(y, ctx, l2, old_ids):
                        return True
                    nxt.append((y, l2))
            frontier = nxt[:40]
            if not frontier:
                return False
        return False

    # walk the frozen directed proposal until it hits a plateau
    pairs, t0 = [], time.time()
    st, lin = st0, lin0
    for t in range(horizon):
        fams, acts, probs = enum_fn(st)
        idx, _w = RR.admissible_indices(fams, acts, ctx)
        if not idx:
            break
        base = RR.old_dependence(st, ctx, lin, old_ids)
        succ = []
        for j in idx:
            y, l2 = step(st, lin, j)
            if y is None:
                continue
            succ.append((j, y, l2, RR.old_dependence(y, ctx, l2, old_ids)))
        if not succ:
            break
        flat = [s for s in succ if s[3] >= base]     # no connectivity progress
        remaining = horizon - t - 1
        if len(flat) >= 2 and remaining >= 2:
            labelled = [(s, completes(s[1], s[2], remaining)) for s in flat]
            good = [s for s, ok in labelled if ok]
            bad = [s for s, ok in labelled if not ok]
            if good and bad:
                for g in good:
                    for b in bad:
                        pairs.append({
                            "t": t, "n_flat": len(flat),
                            "h_good": h_phi(g[1], g[2], remaining),
                            "h_bad": h_phi(b[1], b[2], remaining),
                            "fam_good": fams[g[0]], "fam_bad": fams[b[0]]})
                break
        nxt = min(succ, key=lambda s: (s[3], -float(probs[s[0]])))
        st, lin = nxt[1], nxt[2]

    n_ok = sum(1 for p in pairs if p["h_good"] > p["h_bad"])
    print(f"[rank] {job['case_name']} pairs={len(pairs)} correct={n_ok} "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"case": job["case_name"], "smiles": smi,
            "saturated": bool(rec.get("saturated")), "pairs": pairs,
            "n_pairs": len(pairs), "n_correct": n_ok, "sec": time.time() - t0}


@app.local_entrypoint()
def main(per_molecule: int = 6, test_lo: int = 0, test_hi: int = 6):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    test = [s["smiles"] for s in dv[test_lo:test_hi]]
    train = {s["smiles"] for s in dv[6:22]}
    assert not (set(test) & train), "held-out molecules overlap the fit set"
    regions = list_test_regions.remote(test, per_molecule=per_molecule)
    print(f"held-out molecules={len(test)} regions={len(regions)}")
    jobs = [{"region": r, "case_name": f"r{i}"} for i, r in enumerate(regions)]
    out = [o for o in rank_region.map(jobs) if o]

    tot = sum(o["n_pairs"] for o in out)
    cor = sum(o["n_correct"] for o in out)
    reg = sum(1 for o in out if o["n_pairs"] > 0)
    print(f"\nregions with a plateau pair={reg}/{len(out)}  pairs={tot}  "
          f"correct={cor} ({cor / max(1, tot):.1%})")
    if tot:
        import statistics as st_
        gaps = [p["h_good"] - p["h_bad"] for o in out for p in o["pairs"]]
        print(f"  median h_good - h_bad = {st_.median(gaps):+.4f}")
        print(f"  molecules contributing = {len({o['smiles'] for o in out if o['n_pairs']})}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_rank.json").write_text(json.dumps(
        {"n_pairs": tot, "n_correct": cor,
         "per_region": [{k: o[k] for k in
                         ("case", "smiles", "saturated", "n_pairs", "n_correct")}
                        for o in out]}))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def list_test_regions(smiles_list: list, per_molecule: int = 6) -> list:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    return enumerate_training_regions(smiles_list, per_molecule=per_molecule)
