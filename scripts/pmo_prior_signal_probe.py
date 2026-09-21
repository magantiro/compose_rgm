"""Does the learned law know which legal edits preserve chemistry?

This is the decisive diagnostic behind the A/B: for one source state and one
family, enumerate EVERY legal candidate, take the model's probability for each
and the QED/SA change each one causes, and correlate them.  It isolates the
question 'does this checkpoint carry the signal' from 'is the guidance strength
right', so a negative A/B can be attributed rather than merely reported.

Zero oracle calls.  No property value is ever read by the prior itself -- QED
here is the MEASUREMENT AXIS, computed after the fact, never an input.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.current_state_edits import ENUMERATORS
from compose_v4.control.learned_successor_prior import LearnedSuccessorPrior
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")
PARENTS = Path("/Users/rmaganti/compose_pmo_replay_data/production_parents_v1.json")
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
SCOPE = "3721d69851110fdd"


def spearman(x, y) -> float:
    """Rank correlation, computed without scipy so the pinned env suffices."""
    n = len(x)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(list(x)), rank(list(y))
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT, expected_scope_hash=SCOPE)
    model.eval()
    prior = LearnedSuccessorPrior(model, cache_entries=4096)

    payload = json.loads(PARENTS.read_text())
    rows = sorted([r for t in payload for r in payload[t]["parents"]],
                  key=lambda x: (-x["uses"], x["key"]))[: args.sample]

    per_family: dict[str, list] = {}
    pooled_top_vs_rest: dict[str, list] = {}
    for row in rows:
        source = decode_state(row["state"])
        psmi = molecular_graph_to_smiles(source)
        pmol = Chem.MolFromSmiles(psmi) if psmi else None
        if pmol is None:
            continue
        pq, ps = float(QED.qed(pmol)), float(sascorer.calculateScore(pmol))
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 5:
                continue
            weights = prior.weights(source, family=family, actions=actions)
            dq, ds, ws = [], [], []
            for action, w in zip(actions, weights, strict=True):
                try:
                    child, _ = execute_program(source, [encode_action(family, action)])
                    smi = molecular_graph_to_smiles(child)
                    mol = Chem.MolFromSmiles(smi) if smi else None
                except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                    continue
                if mol is None:
                    continue
                dq.append(float(QED.qed(mol)) - pq)
                ds.append(float(sascorer.calculateScore(mol)) - ps)
                ws.append(float(w))
            if len(dq) < 5:
                continue
            per_family.setdefault(family, []).append({
                "rho_weight_dqed": spearman(ws, dq),
                "rho_weight_dsa": spearman(ws, ds),
                "n": len(dq),
            })
            # Does the prior's own top pick beat the family average?
            top = int(np.argmax(ws))
            pooled_top_vs_rest.setdefault(family, []).append({
                "top_dqed": dq[top], "mean_dqed": statistics.fmean(dq),
                "top_dsa": ds[top], "mean_dsa": statistics.fmean(ds),
                "best_possible_dqed": max(dq),
            })

    out = {
        "schema_version": "pmo_prior_signal_probe_v1",
        "oracle_calls_spent": 0,
        "parents_sampled": len(rows),
        "reading": (
            "rho_weight_dqed > 0 means the learned law puts more mass on edits that "
            "preserve QED. rho ~ 0 means the checkpoint carries no chemistry-preservation "
            "signal at this call site, and no guidance strength can create one."
        ),
        "by_family": {},
        "prior_statistics": prior.statistics,
    }
    for family, entries in sorted(per_family.items()):
        rq = [e["rho_weight_dqed"] for e in entries if e["rho_weight_dqed"] == e["rho_weight_dqed"]]
        rs = [e["rho_weight_dsa"] for e in entries if e["rho_weight_dsa"] == e["rho_weight_dsa"]]
        tv = pooled_top_vs_rest[family]
        n = len(rq)
        out["by_family"][family] = {
            "states": n,
            "mean_rho_weight_dqed": statistics.fmean(rq) if rq else None,
            "se_rho_weight_dqed": (statistics.pstdev(rq) / (n ** 0.5)) if n > 1 else None,
            "mean_rho_weight_dsa": statistics.fmean(rs) if rs else None,
            "pct_states_rho_positive": 100.0 * sum(1 for r in rq if r > 0) / n if n else None,
            "mean_top_pick_dqed": statistics.fmean(e["top_dqed"] for e in tv),
            "mean_family_dqed": statistics.fmean(e["mean_dqed"] for e in tv),
            "mean_best_possible_dqed": statistics.fmean(e["best_possible_dqed"] for e in tv),
            "mean_top_pick_dsa": statistics.fmean(e["top_dsa"] for e in tv),
            "mean_family_dsa": statistics.fmean(e["mean_dsa"] for e in tv),
        }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"{'family':24s} {'states':>7s} {'rho(w,dQED)':>12s} {'+-se':>7s} {'%rho>0':>7s} "
          f"{'topdQED':>8s} {'famdQED':>8s} {'bestdQED':>9s}")
    for family, m in out["by_family"].items():
        print(f"{family:24s} {m['states']:7d} {m['mean_rho_weight_dqed']:12.4f} "
              f"{(m['se_rho_weight_dqed'] or 0):7.4f} {m['pct_states_rho_positive']:6.1f}% "
              f"{m['mean_top_pick_dqed']:8.4f} {m['mean_family_dqed']:8.4f} "
              f"{m['mean_best_possible_dqed']:9.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
