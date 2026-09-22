"""Can donor recombination over CURRENT-RUN SCORED molecules source discovery goals?

Zero oracle calls.  Every molecule and every score is REPLAYED from a completed charged
run's ledger; nothing new is scored, and no prescreen bank is read.

WHY THIS IS NOT THE COMPILER THAT WAS ALREADY REFUTED
-----------------------------------------------------
`ad071cb2` measured `compile_source_to_target` -- the only compiler accepting an ARBITRARY
pair -- and found it is `delete_to_null_then_construct_v1`: retained_fraction 0.000 on 4 of
4 drug-like pairs, every route through the formal null state, even for a pair differing by
one methyl.  It concluded that a scaffold-preserving bridge for arbitrary pairs does not
exist today.

That conclusion is correct for ARBITRARY pairs and does not cover this path.
`compose_v4.control.donor_program.compile_transplant` does not accept an arbitrary pair:
it CONSTRUCTS its target from an explicit retained/added split (source minus one pendant,
plus one donor pendant), and then asserts after replay that the retained slots' atom
identity, formal charge and connectivity are UNCHANGED -- raising if they are not.  So it
cannot demolish to null; the retained region is preserved by construction.

WHAT IS ACTUALLY AT STAKE, AND THE ARM STRUCTURE
------------------------------------------------
Preservation is guaranteed; the retained FRACTION is not.  It is decided entirely by which
cut is drawn, and `donor_memory.RECIPE` ships `cut_distribution:
uniform_oriented_single_bridge` -- a uniform draw over every oriented single-bond bridge.
On a drug-like molecule most of those bridges sit near a leaf, but the ones that do not
amputate most of the molecule, so a uniform draw spends much of its mass on transplants
that keep almost nothing.  This is the same shape as the T4 region-law defect, where an
unconditioned bounded draw was the binding constraint rather than the executor.

So the two arms differ in the CUT DRAW and in nothing else -- same pairs, same pair order,
same donor pool, same compiler:

  uniform      every oriented single-bond bridge equally likely   (the shipped recipe)
  retentive    the same support, weighted toward cuts that RELEASE LITTLE of the source

`retentive` is a RE-WEIGHTING and never a filter: every cut the uniform arm can draw keeps
positive probability, so the arms share a support and a difference cannot be an artifact of
one arm reaching molecules the other cannot.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

from compose_v4.control.donor_program import PendantCut, compile_transplant, pendant_cuts
from compose_v4.experiments.editing_v2_evaluation_semantics import production_state_from_smiles

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

#: A COMPLETED charged run's ledger. Molecules and scores are replayed, never re-scored.
#: Deliberately NOT `diagnostics/pmo_banks_all.json`, whose scores come from ~249,455
#: UNCOUNTED prescreen calls and are therefore inadmissible as current-run feedback.
LEDGER = ROOT / "diagnostics/pmo_3x250_autopsy_v1.json"
LEDGER_PATH = ("celecoxib_distance", "all_rows")
SMILES_KEY, SCORE_KEY = "endpoint", "charged_score"

import sys

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer


def ledger_rows() -> list[dict]:
    payload = json.loads(LEDGER.read_text())
    for key in LEDGER_PATH:
        payload = payload[key]
    rows = [r for r in payload if r.get(SCORE_KEY) is not None and r.get(SMILES_KEY)]
    rows.sort(key=lambda r: (-r[SCORE_KEY], r[SMILES_KEY]))
    return rows


def cut_weights(cuts: tuple[PendantCut, ...], *, arm: str, n_real: int) -> np.ndarray:
    """Uniform, or tilted toward cuts that release little of the source.

    The retentive tilt is `exp(-released_fraction / temperature)` with a positive floor, so
    it RE-RANKS the shared support rather than removing anything from it -- the arms differ
    in probability, never in reach.
    """
    if arm == "uniform":
        return np.ones(len(cuts), dtype=float) / max(len(cuts), 1)
    released = np.asarray([len(cut.component) / n_real for cut in cuts], dtype=float)
    weights = np.maximum(0.05, np.exp(-released / 0.25))
    return weights / weights.sum()


def descriptors(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"qed": float(QED.qed(mol)), "sa": float(sascorer.calculateScore(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms())}


def run_arm(states: dict[str, object], pairs: list[tuple[str, str]], *, arm: str,
            seed: int, draws_per_pair: int) -> dict:
    rng = np.random.default_rng(seed)
    rows, statuses = [], {}
    for source_smiles, donor_smiles in pairs:
        source, donor = states[source_smiles], states[donor_smiles]
        source_cuts, donor_cuts = pendant_cuts(source), pendant_cuts(donor)
        if not source_cuts or not donor_cuts:
            statuses["no_cut"] = statuses.get("no_cut", 0) + 1
            continue
        source_p = cut_weights(source_cuts, arm=arm, n_real=source.n_real_atoms)
        donor_p = cut_weights(donor_cuts, arm=arm, n_real=donor.n_real_atoms)
        for _ in range(draws_per_pair):
            a = source_cuts[int(rng.choice(len(source_cuts), p=source_p))]
            b = donor_cuts[int(rng.choice(len(donor_cuts), p=donor_p))]
            result = compile_transplant(source, donor, a, b)
            status = result.get("status")
            statuses[status] = statuses.get(status, 0) + 1
            if status != "compiled":
                continue
            rows.append({
                "source": source_smiles,
                "donor": donor_smiles,
                "smiles": result["smiles"],
                "retained_fraction": 1.0 - result["released_fraction"],
                "removed_atoms": result["removed_atoms"],
                "added_atoms": result["added_atoms"],
                "steps": len(result.get("actions", ())),
                "descriptors": descriptors(result["smiles"]),
            })
    return {"arm": arm, "attempts": sum(statuses.values()), "statuses": statuses, "rows": rows}


def summarize(arm: dict, scored: set[str]) -> dict:
    rows = arm["rows"]
    retained = [r["retained_fraction"] for r in rows]
    steps = [r["steps"] for r in rows]
    qed = [r["descriptors"]["qed"] for r in rows if r["descriptors"]]
    sa = [r["descriptors"]["sa"] for r in rows if r["descriptors"]]
    novel = [r for r in rows if r["smiles"] not in scored]

    def stat(values: list[float]) -> dict | None:
        if not values:
            return None
        return {"min": min(values), "median": statistics.median(values), "max": max(values),
                "mean": statistics.fmean(values)}

    return {
        "arm": arm["arm"],
        "attempts": arm["attempts"],
        "compiled": len(rows),
        "compiled_rate": len(rows) / arm["attempts"] if arm["attempts"] else 0.0,
        "statuses": arm["statuses"],
        "distinct_products": len({r["smiles"] for r in rows}),
        "novel_vs_scored_set": len({r["smiles"] for r in novel}),
        "retained_fraction": stat(retained),
        # The share of compiled products that keep most of the parent. A transplant that
        # keeps a tenth of its source is a different molecule, not a refinement of one.
        "retained_at_least_half": sum(1 for v in retained if v >= 0.5),
        "steps": stat([float(s) for s in steps]),
        "qed": stat(qed),
        "sa": stat(sa),
        # DESCRIPTIVE ONLY. QED and SA are not the PMO objective and are not consulted by
        # anything here; they are reported so "the products are molecules" is a number
        # rather than an impression.
        "descriptor_note": "descriptive chemistry sanity, NOT an objective and not used for selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donors", type=int, default=14,
                        help="top-scoring ledger molecules used as both sources and donors")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--draws-per-pair", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = ledger_rows()
    scored = {r[SMILES_KEY] for r in rows}
    pool = rows[: args.donors]
    states, unbuildable = {}, []
    for row in pool:
        try:
            states[row[SMILES_KEY]] = production_state_from_smiles(row[SMILES_KEY], max_atoms=48)
        except Exception as error:  # noqa: BLE001 - an unbuildable ledger row IS a finding
            unbuildable.append({"smiles": row[SMILES_KEY], "error": type(error).__name__})

    keys = sorted(states)
    rng = np.random.default_rng(args.seed)
    pairs: list[tuple[str, str]] = []
    while len(pairs) < args.pairs and len(keys) > 1:
        i, j = rng.choice(len(keys), size=2, replace=False)
        pairs.append((keys[int(i)], keys[int(j)]))

    started = time.time()
    arms = {
        arm: summarize(
            run_arm(states, pairs, arm=arm, seed=args.seed, draws_per_pair=args.draws_per_pair),
            scored,
        )
        for arm in ("uniform", "retentive")
    }
    elapsed = time.time() - started

    for name, arm in arms.items():
        retained = arm["retained_fraction"]
        print(f"  {name:<10} attempts={arm['attempts']:<4} compiled={arm['compiled']:<4} "
              f"({arm['compiled_rate']:.1%})  novel={arm['novel_vs_scored_set']:<4} "
              f"retained median={retained['median']:.3f}  >=0.5: {arm['retained_at_least_half']}"
              if retained else f"  {name}: no compiled rows")
    print(f"\nledger {LEDGER.name}: {len(rows)} charged rows, pool {len(pool)}, "
          f"states {len(states)}, unbuildable {len(unbuildable)}")
    print(f"oracle calls 0; {elapsed:.1f}s")

    report = {
        "schema_version": "pmo_donor_transplant_feasibility_v1",
        "oracle_calls": 0,
        "information_regime": (
            "molecules and scores REPLAYED from a completed charged run's ledger; no new "
            "scoring, no prescreen bank, no declared target"
        ),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "ledger_rows": len(rows),
        "pool_size": len(pool),
        "states_built": len(states),
        "unbuildable_ledger_rows": unbuildable,
        "pairs": len(pairs),
        "draws_per_pair": args.draws_per_pair,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 2),
        "arms": arms,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
