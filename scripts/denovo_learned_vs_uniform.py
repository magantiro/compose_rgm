#!/usr/bin/env python3
"""De-novo diagnostic: is the learned base valuable in its NATIVE regime?

The editing hero found uniform-legal >= the learned proposal when EDITING a lead --
but a lead is an out-of-distribution starting state for a de-novo generator, so that
result cannot tell "the base is undertrained" apart from "we used a generator as an
editor". This isolates the base in the regime it was TRAINED for: generation from the
carbon-tree seed, time sweeping 0->horizon. Prediction if the base is a competent
generator (regardless of editing): the learned proposal should CRUSH uniform-legal,
because uniform growth from methane produces valid-but-junk molecules while the
learned rates grow drug-like ones. If learned ~ uniform even here, the base itself is
the weak link (undertraining / architecture), not just the editing regime.

Arms (matched by distinct molecules scored), maximizing QED, archiving everything:
  - learned : Q_theta proposal, steered toward QED (its native generative regime).
  - uniform : uniform-legal proposal, steered identically -- the ablation.
  - base    : Q_theta proposal, UNSTEERED (the base model's own de-novo samples).
Reports best QED, mean top-10 QED, %(QED>0.6), and mean SA (synthetic accessibility)
of the top molecules. Every state is a valid molecule (validity 100% for all arms).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer  # noqa: E402  RDKit contrib synthetic-accessibility score

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.rewrite.fiber import ActionFiberSpec  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from griddd_value_guided_smc_controller import _load_base_sampler  # noqa: E402
from pareto_editing_hero import propose  # noqa: E402  (learned / uniform-legal proposal)

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
OUTDIAG = _ROOT / "diagnostics" / "reachability"
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"


def denovo_search(start, mode, steer, budget, sampler, rewrite, spec, rng, horizon, t0, t1, anneal):
    cache: dict[str, float | None] = {}

    def score(node):
        key = canonical_state_key(node)
        if key not in cache:
            m = Chem.MolFromSmiles(molecular_graph_to_smiles(node) or "")
            cache[key] = None if m is None else float(QED.qed(m))
        return cache[key]

    cur = start
    q_cur = score(cur)
    best_smiles: dict[str, float] = {}
    step = 0
    n = 0
    while n < budget and step < budget * 8:
        step += 1
        t = min(step * horizon / budget, horizon)          # sweep 0 -> horizon (generative time)
        succ = propose(cur, mode, sampler, rewrite, spec, t, rng)
        if succ is None:
            cur, q_cur = start, score(start)
            continue
        q = score(succ)
        if q is None:
            continue
        best_smiles[molecular_graph_to_smiles(succ)] = q
        if steer:
            temp = t0 * (t1 / t0) ** min(step / anneal, 1.0)
            if q >= (q_cur or -1) or rng.random() < np.exp((q - (q_cur or -1)) / temp):
                cur, q_cur = succ, q
        else:
            cur, q_cur = succ, q
        n = sum(1 for v in cache.values() if v is not None)
    qs = np.array(sorted(best_smiles.values(), reverse=True))
    top = sorted(best_smiles.items(), key=lambda kv: kv[1], reverse=True)[:10]
    sas = []
    for smi, _ in top:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            sc = sascorer.calculateScore(m)
        except Exception:  # noqa: BLE001
            sc = None
        if sc is not None:
            sas.append(float(sc))
    return {
        "n_scored": int(n), "n_steps": int(step),
        "best_qed": float(qs.max()) if len(qs) else 0.0,
        "mean_top10_qed": float(qs[:10].mean()) if len(qs) else 0.0,
        "frac_qed_over_0.6": float((qs > 0.6).mean()) if len(qs) else 0.0,
        "mean_sa_top10": float(np.mean(sas)) if sas else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=8)
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--seed-smiles", default="C")
    p.add_argument("--horizon", type=float, default=16.0)
    p.add_argument("--t0", type=float, default=0.12)
    p.add_argument("--t1", type=float, default=0.02)
    p.add_argument("--anneal", type=int, default=400)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    sampler, _ = _load_base_sampler(CKPT, 0.0)
    rewrite = de_novo_rewrite_system()
    spec = ActionFiberSpec.neutral_cnof()
    start = pad_molecular_graph(smiles_to_molecular_graph(args.seed_smiles), 40)
    arms = {"learned": ("learned", True), "uniform": ("uniform", True), "base": ("learned", False)}

    rows = {a: [] for a in arms}
    for r in range(args.n_runs):
        for a, (mode, steer) in arms.items():
            rng = np.random.default_rng(r)
            rows[a].append(denovo_search(start, mode, steer, args.budget, sampler, rewrite, spec,
                                         rng, args.horizon, args.t0, args.t1, args.anneal))
        print(f"run {r}: " + "  ".join(
            f"{a} bestQED={rows[a][-1]['best_qed']:.3f}" for a in arms), flush=True)

    summary = {"experiment": "denovo_learned_vs_uniform", "n_runs": args.n_runs,
               "budget": args.budget, "seed": args.seed_smiles, "arms": {}}
    print("\n=== mean over runs (de-novo from seed) ===")
    for a in arms:
        R = rows[a]
        s = {k: float(np.mean([row[k] for row in R]))
             for k in ("best_qed", "mean_top10_qed", "frac_qed_over_0.6", "mean_sa_top10")}
        summary["arms"][a] = {**s, "rows": R}
        print(f"  {a:<9} bestQED={s['best_qed']:.3f}  top10QED={s['mean_top10_qed']:.3f}  "
              f"%QED>0.6={s['frac_qed_over_0.6']:.2f}  SA(top10)={s['mean_sa_top10']:.2f}")
    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / "denovo_learned_vs_uniform.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
