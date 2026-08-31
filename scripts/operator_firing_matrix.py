"""Operator-firing matrix: is the useful chemistry unavailable, suppressed by
R_theta, drowned in junk, or available-and-clean but never sequenced?

For each state and each executor rule family f, report
    N_f              legal actions in the family
    P_f              R_theta probability mass on the family (full law)
    best_rank        rank of the family's highest-probability action
    in_cap           actions surviving the global top-APPLY_CAP truncation
    clean_frac       fraction of the family's actions whose product passes the
                     calibrated med-chem gate
    clean_mass       probability mass on gate-clean products only

Runs entirely on CPU against the local frozen runtime. No docking, no search.
"""
from __future__ import annotations
import collections, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from compose_v4.gates.med_chem_gate import is_valid

APPLY_CAP, CANONICAL_SLOTS, TIME_POINT = 300, 48, 0.5


def _law(model, system, smi):
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    pr = np.array([m.probability for m in law.marks], float)
    order = list(np.argsort(-pr))
    rank_of = {int(j): r for r, j in enumerate(order)}
    return st, law, pr, order, rank_of, canonical_state_key, system


def matrix_for(model, system, smi, label):
    st, law, pr, order, rank_of, ckey, system = _law(model, system, smi)
    tot = float(pr.sum())
    fam = collections.defaultdict(lambda: dict(
        n=0, mass=0.0, best_rank=10**9, in_cap=0, clean_n=0, clean_mass=0.0,
        applied=0))
    products = {}
    for j in range(len(law.marks)):
        mk = law.marks[j]; f = mk.executor_rule_name; p = float(pr[j]); r = rank_of[j]
        d = fam[f]
        d["n"] += 1; d["mass"] += p; d["best_rank"] = min(d["best_rank"], r)
        if r < APPLY_CAP: d["in_cap"] += 1
        try:
            y = ckey(system.apply(st, f, mk.action))
        except Exception:
            y = None
        if y:
            d["applied"] += 1
            products[j] = y
            if is_valid(y):
                d["clean_n"] += 1; d["clean_mass"] += p
    rows = []
    for f, d in sorted(fam.items(), key=lambda kv: -kv[1]["mass"]):
        rows.append(dict(family=f, n=d["n"], mass=round(d["mass"]/tot, 6),
                         best_rank=d["best_rank"], in_cap=d["in_cap"],
                         applied=d["applied"],
                         clean_frac=(round(d["clean_n"]/d["applied"], 3) if d["applied"] else None),
                         clean_mass=round(d["clean_mass"]/tot, 6)))
    return dict(label=label, smi=smi, n_marks=len(law.marks), total_mass=tot,
                rows=rows), products, rank_of, pr, law


def required_action(products, rank_of, pr, law, target_smi):
    """If the exact next route state is reachable in one edit, report its rank."""
    t = Chem.MolFromSmiles(target_smi)
    if t is None: return None
    tc = Chem.MolToSmiles(t)
    hits = []
    for j, y in products.items():
        ym = Chem.MolFromSmiles(y)
        if ym is not None and Chem.MolToSmiles(ym) == tc:
            hits.append((rank_of[j], float(pr[j]), law.marks[j].executor_rule_name))
    if not hits: return None
    r, p, f = min(hits)
    return dict(rank=r, prob=p, family=f, in_cap=r < APPLY_CAP, clean=is_valid(target_smi))


def main():
    from pareto_local_runtime import build_local_model
    from compose_v4.experiments.production_successor_kernel import _default_rewrite_system
    t0 = time.time(); model = build_local_model()
    system = _default_rewrite_system(model)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    seeds = {s["target"] + "_s" + str(s["idx"]): s["smiles"]
             for s in json.loads((REPO / "docs/GENMOL_T4_SEEDS.json").read_text())}
    route = json.loads((REPO / "diagnostics/parp1_witness_route.json").read_text())
    fs = route["forward_states"]

    states = [("parp1_s0 SEED (route[0])", fs[0], fs[1]),
              ("parp1_s0 route[6] opened", fs[6], fs[7]),
              ("parp1_s0 route[12] pre-close", fs[12], fs[13]),
              ("5ht1b_s7 SEED", seeds["5ht1b_s7"], None),
              ("braf_s11 SEED", seeds["braf_s11"], None),
              ("braf_s10 SEED", seeds["braf_s10"], None)]

    out = []
    for label, smi, nxt in states:
        t = time.time()
        res, products, rank_of, pr, law = matrix_for(model, system, smi, label)
        res["required"] = required_action(products, rank_of, pr, law, nxt) if nxt else None
        res["seconds"] = round(time.time() - t, 1)
        out.append(res)
        print(f"\n=== {label}   ({res['n_marks']} marks, {res['seconds']}s) ===", flush=True)
        print(f"  {'family':22s} {'N':>4s} {'mass':>9s} {'rank':>5s} {'inCap':>5s} "
              f"{'clean%':>7s} {'cleanMass':>10s}")
        for r in res["rows"]:
            cf = "n/a" if r["clean_frac"] is None else f"{100*r['clean_frac']:.0f}%"
            print(f"  {r['family']:22s} {r['n']:4d} {r['mass']:9.5f} {r['best_rank']:5d} "
                  f"{r['in_cap']:5d} {cf:>7s} {r['clean_mass']:10.5f}")
        if res["required"]:
            q = res["required"]
            print(f"  ROUTE-REQUIRED ACTION: {q['family']} rank={q['rank']} "
                  f"p={q['prob']:.3e} in_cap={q['in_cap']} clean={q['clean']}")
        elif nxt:
            print("  ROUTE-REQUIRED ACTION: not reachable in one edit from this state")
    (REPO / "diagnostics/operator_firing_matrix.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote diagnostics/operator_firing_matrix.json ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
