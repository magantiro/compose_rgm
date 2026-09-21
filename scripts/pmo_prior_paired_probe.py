"""Paired A-vs-B comparison of the PMO successor draw.

The marginal A/B table compares two distributions; this compares the SAME
decision.  For each (parent, family, draw) both arms see an identical candidate
list and an identical RNG stream, so the difference in outcome is attributable
to the selection rule alone and parent/family variance cancels.  It also reports
the DISAGREEMENT RATE, without which a null result cannot be read: if the two
arms pick the same candidate almost always, a null says the guidance is too weak,
not that the prior lacks chemistry knowledge.

Zero oracle calls.
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
from compose_v4.control.current_state_edits import ENUMERATORS, current_state_program
from compose_v4.control.edit_program import execute_bound_program
from compose_v4.control.learned_successor_prior import LearnedSuccessorPrior
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")
PARENTS = Path("/Users/rmaganti/compose_pmo_replay_data/production_parents_v1.json")
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
SCOPE = "3721d69851110fdd"


def _chem(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return float(QED.qed(mol)), float(sascorer.calculateScore(mol)), int(mol.GetNumHeavyAtoms())


def _outcome(source, rng, family, prior):
    kwargs = {"successor_prior": prior} if prior is not None else {}
    try:
        program, binding, _detail = current_state_program(
            source, rng, family=family, **kwargs
        )
        endpoint, _r = execute_bound_program(source, program, tuple(binding))
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
        return None
    smiles = molecular_graph_to_smiles(endpoint)
    chem = _chem(smiles)
    return None if chem is None else (smiles, *chem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--settings", default="0.05:1.0,0.05:0.25")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT, expected_scope_hash=SCOPE)
    model.eval()
    prior = LearnedSuccessorPrior(model, cache_entries=4096)
    settings = [(float(s.split(":")[0]), float(s.split(":")[1])) for s in args.settings.split(",")]

    payload = json.loads(PARENTS.read_text())
    rows = sorted([r for t in payload for r in payload[t]["parents"]],
                  key=lambda x: (-x["uses"], x["key"]))[: args.sample]

    pairs: dict[str, list] = {f"{f}:{t}": [] for f, t in settings}
    families = sorted(ENUMERATORS)
    for row in rows:
        source = decode_state(row["state"])
        base = _chem(molecular_graph_to_smiles(source))
        if base is None:
            continue
        pq, ps, _ph = base
        for family in families:
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 3:
                continue
            for draw in range(args.draws):
                seed = [args.seed, draw, int(row["key"][:8], 16), families.index(family)]
                a = _outcome(source, np.random.default_rng(seed), family, None)
                if a is None:
                    continue
                for f, t in settings:
                    prior.floor, prior.temperature = f, t
                    b = _outcome(source, np.random.default_rng(seed), family, prior)
                    if b is None:
                        continue
                    pairs[f"{f}:{t}"].append({
                        "family": family, "parent_heavy": base[2],
                        "same": a[0] == b[0],
                        "dq_a": a[1] - pq, "dq_b": b[1] - pq,
                        "ds_a": a[2] - ps, "ds_b": b[2] - ps,
                    })

    out = {
        "schema_version": "pmo_prior_paired_probe_v1",
        "oracle_calls_spent": 0,
        "settings": {"sample": args.sample, "draws": args.draws, "seed": args.seed},
        "prior_statistics": prior.statistics,
        "arms": {},
    }
    for name, rec in pairs.items():
        if not rec:
            continue
        diff = [r["dq_b"] - r["dq_a"] for r in rec]
        sdiff = [r["ds_b"] - r["ds_a"] for r in rec]
        changed = [r for r in rec if not r["same"]]
        n = len(diff)
        entry = {
            "n_paired_decisions": n,
            "disagreement_rate_pct": 100.0 * len(changed) / n,
            "mean_dQED_A": statistics.fmean(r["dq_a"] for r in rec),
            "mean_dQED_B": statistics.fmean(r["dq_b"] for r in rec),
            "paired_mean_delta_dQED": statistics.fmean(diff),
            "paired_se_delta_dQED": statistics.pstdev(diff) / (n ** 0.5) if n > 1 else 0.0,
            "pct_losing_QED_A": 100.0 * sum(1 for r in rec if r["dq_a"] < 0) / n,
            "pct_losing_QED_B": 100.0 * sum(1 for r in rec if r["dq_b"] < 0) / n,
            "mean_dSA_A": statistics.fmean(r["ds_a"] for r in rec),
            "mean_dSA_B": statistics.fmean(r["ds_b"] for r in rec),
            "paired_mean_delta_dSA": statistics.fmean(sdiff),
            "paired_se_delta_dSA": statistics.pstdev(sdiff) / (n ** 0.5) if n > 1 else 0.0,
        }
        if changed:
            cd = [r["dq_b"] - r["dq_a"] for r in changed]
            entry["on_disagreements_only"] = {
                "n": len(cd),
                "paired_mean_delta_dQED": statistics.fmean(cd),
                "paired_se_delta_dQED": statistics.pstdev(cd) / (len(cd) ** 0.5) if len(cd) > 1 else 0.0,
                "paired_mean_delta_dSA": statistics.fmean(
                    r["ds_b"] - r["ds_a"] for r in changed),
            }
        entry["by_family"] = {}
        for fam in families:
            sub = [r for r in rec if r["family"] == fam]
            if len(sub) < 5:
                continue
            d = [r["dq_b"] - r["dq_a"] for r in sub]
            entry["by_family"][fam] = {
                "n": len(sub),
                "disagreement_rate_pct": 100.0 * sum(1 for r in sub if not r["same"]) / len(sub),
                "paired_mean_delta_dQED": statistics.fmean(d),
                "paired_se_delta_dQED": statistics.pstdev(d) / (len(d) ** 0.5),
            }
        out["arms"][name] = entry

    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"{'arm':12s} {'n':>6s} {'disagree%':>10s} {'dQED_A':>8s} {'dQED_B':>8s} "
          f"{'paired':>9s} {'+-se':>7s} {'%loseA':>7s} {'%loseB':>7s} {'pairSA':>8s}")
    for name, m in out["arms"].items():
        print(f"{name:12s} {m['n_paired_decisions']:6d} {m['disagreement_rate_pct']:9.1f}% "
              f"{m['mean_dQED_A']:8.4f} {m['mean_dQED_B']:8.4f} "
              f"{m['paired_mean_delta_dQED']:+9.4f} {m['paired_se_delta_dQED']:7.4f} "
              f"{m['pct_losing_QED_A']:6.1f}% {m['pct_losing_QED_B']:6.1f}% "
              f"{m['paired_mean_delta_dSA']:+8.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
