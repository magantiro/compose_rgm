#!/usr/bin/env python3
"""B: controller composition + the learned-vs-uniform ablation (the actual thesis test).

Controller-vs-no-controller only echoes Prop. closure (guidance must help). The
question a reviewer actually asks is whether the LEARNED quotient rate Q_theta
carries weight, i.e. does it beat a UNIFORM-over-the-legal-fiber proposal under the
SAME controller. The SMC harness (value_guided_smc) is identical; we only swap the
base sampler. Both arms: same de-novo carbon-tree seeds, same objectives, same
budget, same controller, multiple seeds. This is the de-novo leading indicator for
whether the source-conditioned edit flow (Stage B) is worth training.

Objectives (2, garbage-resistant, conflicting): QED and synthesizability (-SA),
both maximized, normalized to [0,1]. Metric: feasible hypervolume of the
nondominated set each arm reaches (population dumped by the controller across a
small omega sweep), reported per seed with spread. Every committed state is a
valid molecule, so validity is 100% by construction for both arms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer  # noqa: E402

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles  # noqa: E402
from compose_v4.rewrite.fiber import ActionFiberSpec, _candidate_actions  # noqa: E402
from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: E402
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint  # noqa: E402
from griddd_value_guided_smc_controller import _load_base_sampler, value_guided_smc  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
OUTDIAG = _ROOT / "diagnostics" / "composition"
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"

UMark = namedtuple("UMark", ["rule_name", "action"])


class UniformLegalSampler:
    """Q_theta stand-in: propose a mark uniformly from the legal fiber (ignores time)."""

    def __init__(self, spec):
        self.spec = spec

    def sample_rewrite_mark(self, node, t, rng):
        cands = _candidate_actions(node, self.spec)
        cands = cands if isinstance(cands, list) else list(cands)
        if not cands:
            return UMark(None, None)
        rule_name, action = cands[int(rng.integers(len(cands)))]
        return UMark(rule_name, action)


def safe_sa(m):
    try:
        s = sascorer.calculateScore(m)
        return float(s) if s is not None else 10.0
    except Exception:  # noqa: BLE001
        return 10.0


def objectives(mol):
    """(QED, -SA) normalized to [0,1], both maximized."""
    return np.array([QED.qed(mol), float(np.clip((10.0 - safe_sa(mol)) / 9.0, 0.0, 1.0))])


def nondominated(pts):
    P = np.asarray(pts)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i] and ((P >= P[i]).all(1) & (P > P[i]).any(1)).any():
            keep[i] = False
    return P[keep]


def hypervolume_2d(pts, ref=(0.0, 0.0)):
    if len(pts) == 0:
        return 0.0
    P = nondominated(pts)
    P = P[np.argsort(-P[:, 0])]
    hv, y_prev = 0.0, ref[1]
    for x, y in P:
        if x > ref[0] and y > y_prev:
            hv += (x - ref[0]) * (y - y_prev)
            y_prev = y
    return hv


def arm_seed_front(sampler, source_prior, rewrite, seed_int, omegas, cfg):
    """Union the dumped population across a small omega sweep from one de-novo seed;
    score both objectives; return the nondominated set (objective vectors)."""
    pts = []
    for j, w in enumerate(omegas):
        start = source_prior.sample(np.random.default_rng(1000 + seed_int), n_slots=40)

        def oracle(state, w=w):
            m = Chem.MolFromSmiles(molecular_graph_to_smiles(state) or "")
            return float(np.asarray(w) @ objectives(m)) if m is not None else -100.0

        r = value_guided_smc(start, sampler=sampler, qed_oracle=oracle, rewrite=rewrite,
                             seed=seed_int * 17 + j, similarity_minimum=0.0, budget=cfg["budget"],
                             n_particles=cfg["particles"], max_steps=cfg["max_steps"], time_step=0.1,
                             horizon=16.0, alpha_start=0.20, alpha_end=0.03, feasible_attempts=8,
                             dump_population=True)
        for smi in (r.get("population_smiles") or []):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                pts.append(objectives(m))
    return nondominated(pts) if pts else np.zeros((0, 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=80)
    p.add_argument("--particles", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    learned, _ = _load_base_sampler(CKPT, 0.0)
    _, ck = load_factorized_rollout_checkpoint(CKPT)
    source_prior = ck["tree_source_prior"]
    rewrite = de_novo_rewrite_system()
    spec = ActionFiberSpec.neutral_cnof()
    omegas = [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 1.0])]
    cfg = {"budget": args.budget, "particles": args.particles, "max_steps": args.max_steps}
    arms = {"learned": learned, "uniform": UniformLegalSampler(spec)}

    result = {"experiment": "composition_learned_vs_uniform", "objectives": ["QED", "-SA"],
              "seeds": args.seeds, "budget": args.budget, "arms": {}}
    for name, sampler in arms.items():
        hvs, bestq, bests = [], [], []
        for s in range(args.seeds):
            front = arm_seed_front(sampler, source_prior, rewrite, s, omegas, cfg)
            hvs.append(hypervolume_2d(front))
            bestq.append(float(front[:, 0].max()) if len(front) else 0.0)
            bests.append(float(front[:, 1].max()) if len(front) else 0.0)
            print(f"  {name} seed {s}: HV={hvs[-1]:.3f} bestQED={bestq[-1]:.3f} "
                  f"best(-SA)={bests[-1]:.3f} |front|={len(front)}", flush=True)
        result["arms"][name] = {"mean_hv": float(np.mean(hvs)), "std_hv": float(np.std(hvs)),
                                "mean_best_qed": float(np.mean(bestq)),
                                "mean_best_negsa": float(np.mean(bests)), "hv_per_seed": hvs}
    lr, un = result["arms"]["learned"], result["arms"]["uniform"]
    result["learned_minus_uniform_hv"] = lr["mean_hv"] - un["mean_hv"]
    print("\n=== learned vs uniform (feasible HV over QED x -SA) ===")
    for n in arms:
        a = result["arms"][n]
        print(f"  {n:<9} HV={a['mean_hv']:.4f} +/- {a['std_hv']:.4f}  bestQED={a['mean_best_qed']:.3f}")
    print(f"  learned - uniform HV = {result['learned_minus_uniform_hv']:+.4f}  "
          f"({'learned carries weight' if result['learned_minus_uniform_hv']>0.01 else 'RED FLAG: learned ~ uniform'})")
    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / "composition_learned_vs_uniform.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
