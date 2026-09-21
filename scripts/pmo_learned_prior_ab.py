"""Matched A/B of the PMO step-level successor draw: uniform vs learned prior.

Zero oracle calls.  Arm A is the production draw; arm B is the same production
function with a learned prior attached.  Both arms are driven through the LIVE
`current_state_program`, never a transcription, so a drift in production code
changes both arms rather than silently invalidating the comparison.

Matching: same parent, same family, same candidate list, seed-matched RNG per
(parent, family, draw).  Only the selection rule differs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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


def chem(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "qed": float(QED.qed(mol)),
        "sa": float(sascorer.calculateScore(mol)),
        "heavy": int(mol.GetNumHeavyAtoms()),
    }


def endpoint_of(source, program, binding) -> str | None:
    """The endpoint SMILES of one accepted proposal.

    `current_state_program` returns the program and its input binding, not the
    successor, so the endpoint is realized the same way the production
    controller realizes it: by executing the bound program.
    """
    try:
        endpoint, _receipt = execute_bound_program(source, program, tuple(binding))
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
        return None
    return molecular_graph_to_smiles(endpoint)


def run(sample: int, draws: int, seed: int, settings: list) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _meta = load_factorized_rollout_checkpoint(CHECKPOINT, expected_scope_hash=SCOPE)
    model.eval()
    # ONE prior instance across every setting, so the marked-law cache is shared
    # and a setting sweep costs no extra model forwards.  floor/temperature are
    # the only fields mutated between settings; the law does not depend on them.
    prior = LearnedSuccessorPrior(model, cache_entries=4096)
    arms = ["A"] + [f"B:{f}:{t}" for f, t in settings]

    payload = json.loads(PARENTS.read_text())
    rows = []
    for task in sorted(payload):
        for row in payload[task]["parents"]:
            rows.append((task, row))
    rows.sort(key=lambda r: (-r[1]["uses"], r[1]["key"]))
    rows = rows[:sample]

    records: dict[str, list] = {a: [] for a in arms}
    covered: dict[str, set] = {a: set() for a in arms}
    endpoints: dict[str, set] = {a: set() for a in arms}
    attempts = {a: 0 for a in arms}
    executed = {a: 0 for a in arms}
    permutation_checks = 0

    for task, row in rows:
        source = decode_state(row["state"])
        parent_smiles = molecular_graph_to_smiles(source)
        parent = chem(parent_smiles) if parent_smiles else None
        if parent is None:
            continue
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if not actions:
                continue
            # P5: B's ordering must be a permutation of A's candidate list.
            probe_rng = np.random.default_rng(seed)
            ordered = prior.order(source, probe_rng, family=family, actions=actions)
            assert len(ordered) == len(actions), "prior changed the candidate count"
            assert {id(a) for a in ordered} == {id(a) for a in actions}, (
                "prior is filtering: the ordered list is not a permutation"
            )
            permutation_checks += 1

            for draw in range(draws):
                # Both arms get an IDENTICAL fresh stream.  `family` is passed
                # explicitly, so no RNG is consumed before the selection and the
                # two arms differ only in how they consume the same stream.
                draw_seed = [seed, draw, int(row["key"][:8], 16), sorted(ENUMERATORS).index(family)]
                plan = [("A", None, None, None)]
                for f, t in settings:
                    plan.append((f"B:{f}:{t}", prior, f, t))
                for arm, use_prior, f, t in plan:
                    if use_prior is not None:
                        prior.floor, prior.temperature = f, t
                    rng = np.random.default_rng(draw_seed)
                    attempts[arm] += 1
                    try:
                        program, binding, _detail = current_state_program(
                            source, rng, family=family,
                            **({"successor_prior": use_prior} if use_prior is not None else {}),
                        )
                    except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                        continue
                    smiles = endpoint_of(source, program, binding)
                    if smiles is None:
                        continue
                    child = chem(smiles)
                    if child is None:
                        continue
                    executed[arm] += 1
                    covered[arm].add(row["key"])
                    endpoints[arm].add(smiles)
                    records[arm].append({
                        "task": task, "family": family, "parent_heavy": parent["heavy"],
                        "d_qed": child["qed"] - parent["qed"],
                        "d_sa": child["sa"] - parent["sa"],
                        "d_heavy": child["heavy"] - parent["heavy"],
                        "qed": child["qed"], "sa": child["sa"], "heavy": child["heavy"],
                        "parent_qed": parent["qed"],
                    })
    return {
        "records": records, "attempts": attempts, "executed": executed,
        "covered": {k: len(v) for k, v in covered.items()},
        "endpoints": {k: len(v) for k, v in endpoints.items()},
        "permutation_checks": permutation_checks,
        "prior_statistics": prior.statistics,
        "parents_used": len(rows),
    }


def summarize(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    q = [r["d_qed"] for r in rows]
    s = [r["d_sa"] for r in rows]
    aq = [r["qed"] for r in rows]
    asa = [r["sa"] for r in rows]
    hv = [r["heavy"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "mean_d_qed": statistics.fmean(q),
        "se_d_qed": (statistics.pstdev(q) / (n ** 0.5)) if n > 1 else 0.0,
        "median_d_qed": statistics.median(q),
        "pct_losing_qed": 100.0 * sum(1 for x in q if x < 0) / n,
        "mean_d_sa": statistics.fmean(s),
        "se_d_sa": (statistics.pstdev(s) / (n ** 0.5)) if n > 1 else 0.0,
        "median_qed": statistics.median(aq),
        "pct_qed_ge_0_6": 100.0 * sum(1 for x in aq if x >= 0.6) / n,
        "median_sa": statistics.median(asa),
        "pct_sa_le_4": 100.0 * sum(1 for x in asa if x <= 4.0) / n,
        "median_heavy": statistics.median(hv),
        "qed_quartiles": [round(x, 4) for x in statistics.quantiles(aq, n=4)] if n > 3 else None,
        "sa_quartiles": [round(x, 4) for x in statistics.quantiles(asa, n=4)] if n > 3 else None,
        "heavy_quartiles": [round(x, 2) for x in statistics.quantiles(hv, n=4)] if n > 3 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--settings", default="0.05:1.0,0.05:0.5,0.05:0.25,0.02:0.15",
                    help="comma list of floor:temperature arms")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    settings = [(float(x.split(":")[0]), float(x.split(":")[1]))
                for x in args.settings.split(",")]
    result = run(args.sample, args.draws, args.seed, settings)
    arms = ["A"] + [f"B:{f}:{t}" for f, t in settings]
    rec = result.pop("records")
    out = {
        "schema_version": "pmo_learned_prior_ab_v1",
        "oracle_calls_spent": 0,
        "settings": vars(args) | {"out": str(args.out)},
        "overall": {arm: summarize(rec[arm]) for arm in arms},
        "by_parent_heavy_bin": {},
        "by_family": {},
        "by_task": {},
        **result,
    }
    def binof(h):
        return "<10" if h < 10 else "10-19" if h < 20 else "20-29" if h < 30 else ">=30"
    for b in ["<10", "10-19", "20-29", ">=30"]:
        out["by_parent_heavy_bin"][b] = {
            arm: summarize([r for r in rec[arm] if binof(r["parent_heavy"]) == b])
            for arm in arms}
    for fam in sorted(ENUMERATORS):
        out["by_family"][fam] = {
            arm: summarize([r for r in rec[arm] if r["family"] == fam]) for arm in arms}
    for task in sorted({r["task"] for r in rec["A"]}):
        out["by_task"][task] = {
            arm: summarize([r for r in rec[arm] if r["task"] == task]) for arm in arms}
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    o = out["overall"]
    print(f"parents {result['parents_used']}  perm-checks {result['permutation_checks']}")
    print(f"{'arm':14s} {'n':>6s} {'meandQED':>10s} {'+-se':>7s} {'%lose':>7s} {'medQED':>7s} "
          f"{'%QED>=.6':>9s} {'meandSA':>8s} {'medSA':>7s} {'medHv':>6s} {'distinct':>9s}")
    for arm in arms:
        m = o[arm]
        if not m.get("n"):
            continue
        print(f"{arm:14s} {m['n']:6d} {m['mean_d_qed']:10.4f} {m['se_d_qed']:7.4f} "
              f"{m['pct_losing_qed']:6.1f}% {m['median_qed']:7.4f} {m['pct_qed_ge_0_6']:8.1f}% "
              f"{m['mean_d_sa']:8.4f} {m['median_sa']:7.3f} {m['median_heavy']:6.1f} "
              f"{out['endpoints'][arm]:9d}")
    print("cost:", json.dumps(out["prior_statistics"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
