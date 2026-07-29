#!/usr/bin/env python3
"""Scaled hero: multi-objective lead editing, learned vs uniform proposal (fair).

The exact-graph diagnostic proved the OPPORTUNITY (greedy has a structural ceiling
below the reachable Pareto frontier) but could not decide EFFICIENCY. This runs on
real ZINC-derived leads with the LEARNED model and asks the memo's central question:
does the learned edit prior (Q_theta) explore a better Pareto set per oracle call
than the SAME archive search with a UNIFORM-legal proposal (and than unguided
generate-then-filter)?

Task = maximize (QED, Tanimoto-similarity-to-lead), both in [0,1]. This is the
standard "improve drug-likeness while staying close to the lead" objective, written
as its full Pareto FRONTIER instead of a hard similarity gate. Two design reasons,
both fixing earlier flawed versions:
  - GARBAGE-RESISTANT: a wandering uniform walk produces low-QED, low-similarity
    junk that is dominated -- so it cannot "win" hypervolume by reaching extremes
    (the failure mode when maximizing raw logP), and NO hard constraint is needed.
  - FAIR / MATCHED BUDGET: with no hard gate, every valid successor is scored, so
    both proposals reach the same oracle budget (an earlier hard Tanimoto>=0.4 gate
    made the learned proposal reject ~99% of its structural edits and never reach
    budget -- an invalid comparison).

Three arms, matched by distinct molecules SCORED (oracle calls), each archiving the
nondominated set of everything it evaluates, same steered controller, ONLY the
proposal differs:
  - gen_filter : learned proposal, NO steering (base CTMC + filter).
  - learned    : learned proposal (Q_theta), steered (rotate omega, accept detours).
  - uniform    : uniform-legal proposal, steered identically -- the ablation isolating Q_theta.
Reported: feasible hypervolume (exact 2D) vs oracle calls; best QED and best QED at
similarity>=0.4 (literature-comparable); nondominated-set size; mean similarity.
Late editing time (t~=horizon; t=0 destroys the lead). Validity 100% by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.rewrite.fiber import ActionFiberSpec, _candidate_actions  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from griddd_value_guided_smc_controller import _load_base_sampler  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
LEADS = str(_ROOT / "configs" / "benchmarks" / "cnof_leads.json")
OUTDIAG = _ROOT / "diagnostics" / "reachability"
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def nondominated(pts):
    P = np.asarray(pts)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        if ((P >= P[i]).all(1) & (P > P[i]).any(1)).any():
            keep[i] = False
    return P[keep]


def hypervolume_2d(pts, ref=(0.0, 0.0)):
    if len(pts) == 0:
        return 0.0
    P = nondominated(pts)
    P = P[np.argsort(-P[:, 0])]
    hv, y_prev = 0.0, ref[1]
    for x, y in P:
        if x <= ref[0] or y <= y_prev:
            continue
        hv += (x - ref[0]) * (y - y_prev)
        y_prev = y
    return hv


def simplex_dirs(divisions=8):
    return [np.array([t, divisions - t], float) / divisions for t in range(divisions + 1)]


def propose(node, mode, sampler, rewrite, spec, t, rng):
    if mode == "learned":
        mark = sampler.sample_rewrite_mark(node, t, rng)
        if mark.action is None:
            return None
        try:
            return rewrite.apply(node, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001
            return None
    cands = list(_candidate_actions(node, spec))          # uniform-legal (apply one, ~30 ms)
    if not cands:
        return None
    cur_key = canonical_state_key(node)
    for _ in range(6):
        rule_name, action = cands[int(rng.integers(len(cands)))]
        try:
            succ = rewrite.apply(node, rule_name, action)
        except Exception:  # noqa: BLE001
            continue
        if canonical_state_key(succ) != cur_key:
            return succ
    return None


def search(start, lead_fp, mode, steer, budget, sampler, rewrite, spec, rng, weights, cfg):
    """Archive-aware walk over (QED, Tanimoto-to-lead). No hard constraint: every valid
    molecule is scored, so both proposals reach the same oracle budget."""
    cache: dict[str, np.ndarray | None] = {}

    def evaluate(node):
        key = canonical_state_key(node)
        if key not in cache:
            m = Chem.MolFromSmiles(molecular_graph_to_smiles(node) or "")
            cache[key] = None if m is None else np.array(
                [QED.qed(m), DataStructs.TanimotoSimilarity(morgan(m), lead_fp)])
        return cache[key]

    cur = start
    u_cur = evaluate(cur)
    archive: list[np.ndarray] = [] if u_cur is None else [u_cur]
    curve, step, prev = [], 0, 0
    n_scored = 0
    while n_scored < budget and step < budget * 8:
        step += 1
        t = rng.uniform(cfg["t_lo"], cfg["t_hi"])
        succ = propose(cur, mode, sampler, rewrite, spec, t, rng)
        if succ is None:
            cur, u_cur = start, evaluate(start)
            continue
        v = evaluate(succ)
        if v is None:
            continue
        archive.append(v)
        if steer:
            w = weights[step % len(weights)]
            temp = cfg["t0"] * (cfg["t1"] / cfg["t0"]) ** min(step / cfg["anneal"], 1.0)
            du = float(w @ v) - (float(w @ u_cur) if u_cur is not None else -1e9)
            if du >= 0 or rng.random() < np.exp(du / temp):
                cur, u_cur = succ, v
        else:
            cur, u_cur = succ, v
        n_scored = sum(1 for x in cache.values() if x is not None)
        if n_scored >= prev + cfg["checkpoint"]:
            prev = n_scored
            curve.append((n_scored, hypervolume_2d(archive)))
    curve.append((n_scored, hypervolume_2d(archive)))
    arch = np.array(archive)
    qed_at_sim = arch[arch[:, 1] >= 0.4, 0]
    return curve, {
        "final_hv": hypervolume_2d(archive),
        "n_scored": n_scored, "n_steps": step,
        "n_nondominated": int(len(nondominated(archive))),
        "best_qed": float(arch[:, 0].max()),
        "best_qed_at_sim40": float(qed_at_sim.max()) if len(qed_at_sim) else 0.0,
        "mean_sim": float(arch[:, 1].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-leads", type=int, default=8)
    p.add_argument("--budget", type=int, default=150, help="distinct molecules scored / arm / lead")
    p.add_argument("--divisions", type=int, default=8)
    p.add_argument("--t-lo", type=float, default=10.0)
    p.add_argument("--t-hi", type=float, default=16.0)
    p.add_argument("--t0", type=float, default=0.12)
    p.add_argument("--t1", type=float, default=0.02)
    p.add_argument("--anneal", type=int, default=400)
    p.add_argument("--checkpoint", type=int, default=25)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    sampler, _ = _load_base_sampler(CKPT, 0.0)
    rewrite = de_novo_rewrite_system()
    spec = ActionFiberSpec.neutral_cnof()
    leads = json.load(open(LEADS))[: args.n_leads]
    weights = simplex_dirs(args.divisions)
    cfg = {"t_lo": args.t_lo, "t_hi": args.t_hi, "t0": args.t0, "t1": args.t1,
           "anneal": args.anneal, "checkpoint": args.checkpoint}
    arms = {"gen_filter": ("learned", False), "learned": ("learned", True),
            "uniform": ("uniform", True)}

    rows = {a: [] for a in arms}
    for i, (_, smi, _) in enumerate(leads):
        lead_mol = Chem.MolFromSmiles(smi)
        if lead_mol is None:
            continue
        lead_fp = morgan(lead_mol)
        start = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        line = {}
        for a, (mode, steer) in arms.items():
            hv_i = []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * i + s)
                curve, summ = search(start, lead_fp, mode, steer, args.budget,
                                     sampler, rewrite, spec, rng, weights, cfg)
                rows[a].append({"lead": i, "seed": s, "curve": curve, **summ})
                hv_i.append(summ["final_hv"])
            line[a] = float(np.mean(hv_i))
        print(f"lead {i} {smi[:30]:<32} HV: " + "  ".join(f"{a}={line[a]:.3f}" for a in arms), flush=True)

    summary = {"experiment": "pareto_editing_hero_qed_sim", "n_leads": len(leads),
               "budget": args.budget, "seeds": args.seeds, "arms": {}}
    print("\n=== mean over leads x seeds ===")
    for a in arms:
        R = rows[a]
        s = {"mean_final_hv": float(np.mean([r["final_hv"] for r in R])),
             "std_final_hv": float(np.std([r["final_hv"] for r in R])),
             "mean_best_qed": float(np.mean([r["best_qed"] for r in R])),
             "mean_best_qed_at_sim40": float(np.mean([r["best_qed_at_sim40"] for r in R])),
             "mean_nondominated": float(np.mean([r["n_nondominated"] for r in R])),
             "mean_sim": float(np.mean([r["mean_sim"] for r in R])),
             "mean_steps_per_scored": float(np.mean([r["n_steps"] / max(r["n_scored"], 1) for r in R])),
             "rows": R}
        summary["arms"][a] = s
        print(f"  {a:<11} HV={s['mean_final_hv']:.4f}+/-{s['std_final_hv']:.4f}  "
              f"bestQED={s['mean_best_qed']:.3f}  QED@sim.4={s['mean_best_qed_at_sim40']:.3f}  "
              f"|ND|={s['mean_nondominated']:.1f}  sim={s['mean_sim']:.3f}  "
              f"steps/scored={s['mean_steps_per_scored']:.1f}")
    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / "pareto_editing_hero.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
