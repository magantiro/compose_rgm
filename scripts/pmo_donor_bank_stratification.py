"""Does stratifying the donor bank change the material, and change the products?

Zero oracle calls.  Every molecule and every score is REPLAYED from a completed charged
run's ledger; nothing is re-scored, and the prescreen bank (~249,455 UNCOUNTED calls) is
never read.  PMO production kernel: rdkit 2023.9.6.

THE QUESTION
------------
The superseded donor pool was `DONOR_POOL_SIZE = 24` -- the 24 best-scoring molecules of
the run.  This repository measured blind PMO search leaving the drug-like manifold on
almost every task, so the top of the score ranking is the top of the drift.  The bank
replaces that ranking with three disjoint strata (elite / promising / diverse) drawn
stratum-first.

So: on a real counted population, does the stratified bank OFFER different material, and
does the difference survive into the compiled transplants?

THE ARMS DIFFER IN THE DONOR POOL AND IN NOTHING ELSE
------------------------------------------------------
Same parents in the same order, same seeds, same cut law (the retentive production
default), same compiler.  The ONLY difference is which molecules are offered as donors
and with what probability:

  top_n        the superseded pool: 24 best-scoring buildable molecules, drawn uniformly
  stratified   the bank at capacity 300, drawn stratum-first by declared mass

WHAT THIS IS NOT
----------------
Not a scored result.  One task, one ledger, offline compilation, replayed scores.  The
QED and SA figures are a DESCRIPTIVE census so "the products are molecules" is a number
rather than an impression; they are not the PMO objective and nothing here consults them.

A STATED LIMITATION, not discovered afterwards.  The ledger artifact records only the
best-so-far improvement chain, so the PROMISING stratum is exercised by the handful of
real counted parent->child edges that chain contains.  Its selection logic is exercised
deterministically in `tests/test_pmo_donor_scored_bank.py` instead.  The real-ledger
comparison below is therefore predominantly elite-versus-diverse, and it says so in the
report rather than letting a thin stratum look like a measured one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

from compose_v4.control.pmo_credit import basin_label
from compose_v4.control.pmo_donor_channel import (
    DonorStratum,
    donor_region_law,
    donor_transplant_draw,
)
from compose_v4.control.pmo_online_memory import STRATA, ScoredMoleculeBank
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

#: A COMPLETED charged run's ledger. Deliberately NOT `diagnostics/pmo_banks_all.json`,
#: whose scores come from ~249,455 UNCOUNTED prescreen calls.
LEDGER = ROOT / "diagnostics/pmo_3x250_autopsy_v1.json"
TASK = "celecoxib_rediscovery"

#: The pool this bank supersedes.
SUPERSEDED_TOP_N = 24


def counted_population() -> tuple[list[dict], list[dict]]:
    """``(scored rows, real counted parent->child edges)`` from the ledger."""
    payload = json.loads(LEDGER.read_text())
    rows = [
        row
        for row in payload["celecoxib_distance"]["all_rows"]
        if row.get("charged_score") is not None and row.get("endpoint")
    ]
    rows.sort(key=lambda r: (-r["charged_score"], r["endpoint"]))
    scored = {row["endpoint"]: row["charged_score"] for row in rows}
    edges = [
        {
            "parent": step["parent_endpoint"],
            "parent_score": scored[step["parent_endpoint"]],
            "child_score": step["score"],
        }
        for step in payload["tasks"][TASK]["best_so_far_lineage"]["improvements"]
        if step.get("parent_endpoint") and step["parent_endpoint"] in scored
    ]
    return rows, edges


def build_bank(rows: list[dict], edges: list[dict]) -> ScoredMoleculeBank:
    """The bank, fed exactly as the controller feeds it: counted score, basin, lineage."""
    bank = ScoredMoleculeBank()
    for row in rows:
        try:
            basin = basin_label(row["endpoint"])
        except (ValueError, TypeError):
            basin = None
        bank.observe(endpoint=row["endpoint"], score=row["charged_score"], basin=basin)
    for edge in edges:
        bank.observe_lineage(
            parent_endpoint=edge["parent"],
            parent_score=edge["parent_score"],
            child_score=edge["child_score"],
        )
    return bank


def descriptors(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "qed": float(QED.qed(mol)),
        "sa": float(sascorer.calculateScore(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def stat(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def census(endpoints: list[str]) -> dict:
    """Descriptive chemistry of a POOL. Never consulted by any draw."""
    rows = [d for d in (descriptors(e) for e in endpoints) if d]
    basins = set()
    for endpoint in endpoints:
        basin = safe_basin(endpoint)
        if basin is not None:
            basins.add(basin)
    return {
        "molecules": len(endpoints),
        "distinct_basins": len(basins),
        "qed": stat([r["qed"] for r in rows]),
        "sa": stat([r["sa"] for r in rows]),
        "heavy_atoms": stat([float(r["heavy_atoms"]) for r in rows]),
        "qed_at_least_0.6": sum(1 for r in rows if r["qed"] >= 0.6),
    }


def run_arm(
    name: str,
    strata: list[DonorStratum],
    parents: list[tuple[str, object]],
    *,
    seed: int,
    draws_per_parent: int,
) -> dict:
    rng = np.random.default_rng(seed)
    rows, statuses, by_stratum = [], {}, {}
    for _, source in parents:
        for _ in range(draws_per_parent):
            proposal, attempt_census = donor_transplant_draw(
                source, strata, rng, law=donor_region_law, max_attempts=4
            )
            for status, count in attempt_census.items():
                statuses[status] = statuses.get(status, 0) + count
            if proposal is None:
                continue
            by_stratum[proposal.donor_stratum] = by_stratum.get(proposal.donor_stratum, 0) + 1
            rows.append(
                {
                    "donor": proposal.donor,
                    "donor_stratum": proposal.donor_stratum,
                    "smiles": proposal.endpoint,
                    "retained_fraction": proposal.retained_fraction,
                    "removed_atoms": proposal.removed_atoms,
                    "added_atoms": proposal.added_atoms,
                    "steps": proposal.primitive_steps,
                    "descriptors": descriptors(proposal.endpoint),
                }
            )
    return {"arm": name, "rows": rows, "statuses": statuses, "realized_strata": by_stratum}


def safe_basin(smiles: str) -> str | None:
    try:
        return basin_label(smiles)
    except (ValueError, TypeError):
        return None


def reach(rows: list[dict], population_basins: set[str]) -> dict:
    """Structural REACH of a set of products: basins touched, and basins that are NEW.

    "New" means absent from the basins of the counted population the donors came from.
    Under the owner's future-value framing a donor earns its place by unlocking a
    structurally novel region, so this -- not resemblance to any known endpoint -- is the
    structural half of the question.  The VALUE half needs a scored run and is not
    measured here.
    """
    basins = {b for b in (safe_basin(r["smiles"]) for r in rows) if b is not None}
    return {
        "compiled": len(rows),
        "distinct_product_basins": len(basins),
        "product_basins_absent_from_the_counted_population": len(
            basins - population_basins
        ),
    }


def summarize(
    arm: dict, scored: set[str], attempts: int, population_basins: set[str]
) -> dict:
    rows = arm["rows"]
    products = [r for r in rows if r["descriptors"]]
    by_stratum = {}
    for name in sorted({r["donor_stratum"] for r in rows}):
        member = [r for r in rows if r["donor_stratum"] == name]
        by_stratum[name] = {
            **reach(member, population_basins),
            "qed": stat([r["descriptors"]["qed"] for r in member if r["descriptors"]]),
            "steps": stat([float(r["steps"]) for r in member]),
            "retained_fraction": stat([r["retained_fraction"] for r in member]),
        }
    return {
        "by_donor_stratum": by_stratum,
        **reach(rows, population_basins),
        "arm": arm["arm"],
        "attempts": attempts,
        "compiled": len(rows),
        "compiled_rate": len(rows) / attempts if attempts else 0.0,
        "statuses": arm["statuses"],
        "realized_strata": arm["realized_strata"],
        "distinct_products": len({r["smiles"] for r in rows}),
        "novel_vs_counted_population": len({r["smiles"] for r in rows} - scored),
        "distinct_donors_used": len({r["donor"] for r in rows}),
        "retained_fraction": stat([r["retained_fraction"] for r in rows]),
        "steps": stat([float(r["steps"]) for r in rows]),
        "qed": stat([r["descriptors"]["qed"] for r in products]),
        "sa": stat([r["descriptors"]["sa"] for r in products]),
        "product_qed_at_least_0.6": sum(1 for r in products if r["descriptors"]["qed"] >= 0.6),
        "descriptor_note": (
            "descriptive chemistry sanity, NOT an objective and not used for selection"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=60)
    parser.add_argument("--draws-per-parent", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    began = time.perf_counter()
    rows, edges = counted_population()
    bank = build_bank(rows, edges)
    selection = bank.strata(capacity=args.capacity)
    weights = bank.weights(selection)
    top_n = [row["endpoint"] for row in rows[:SUPERSEDED_TOP_N]]

    # Build 48-slot states once, for the union of everything either arm needs.
    wanted = sorted({e for members in selection.values() for e in members} | set(top_n))
    states, unbuildable = {}, []
    for endpoint in wanted:
        try:
            states[endpoint] = production_state_from_smiles(endpoint, max_atoms=48)
        except Exception as error:  # noqa: BLE001 - a census of what cannot be built
            unbuildable.append({"endpoint": endpoint, "error": str(error)[:120]})

    stratified = [
        DonorStratum(
            name=name,
            weight=weights.get(name, 0.0),
            members=tuple(
                (e, states[e]) for e in selection[name] if e in states
            ),
        )
        for name in STRATA
    ]
    # The superseded arm, expressed in the same shape so the ONLY difference is the pool:
    # one stratum, all the mass, the 24 best scorers, drawn uniformly inside it.
    baseline = [
        DonorStratum(
            name="top_n",
            weight=1.0,
            members=tuple((e, states[e]) for e in top_n if e in states),
        )
    ]

    # Parents are the SAME list, in the same order, for both arms: a parent drawn from a
    # pool would confound the donor question with a parent question.
    buildable = [e for e in (row["endpoint"] for row in rows) if e in states]
    parent_rng = np.random.default_rng(args.seed + 1)
    chosen = parent_rng.choice(
        len(buildable), size=min(args.parents, len(buildable)), replace=False
    )
    parents = [(buildable[int(i)], states[buildable[int(i)]]) for i in sorted(chosen)]
    attempts = len(parents) * args.draws_per_parent

    scored_set = {row["endpoint"] for row in rows}
    population_basins = {
        b for b in (safe_basin(row["endpoint"]) for row in rows) if b is not None
    }
    arms = {}
    for name, strata in (("top_n", baseline), ("stratified", stratified)):
        arm = run_arm(
            name, strata, parents, seed=args.seed, draws_per_parent=args.draws_per_parent
        )
        arms[name] = summarize(arm, scored_set, attempts, population_basins)

    report = {
        "schema_version": "pmo_donor_bank_stratification_v1",
        "oracle_calls": 0,
        "information_regime": (
            "molecules, scores and parent->child edges REPLAYED from a completed charged "
            "run's ledger; no new scoring, no prescreen bank, no declared target"
        ),
        "kernel": {"rdkit": Chem.rdBase.rdkitVersion},
        "ledger": LEDGER.name,
        "task": TASK,
        "counted_molecules": len(rows),
        "counted_population_basins": len(population_basins),
        "real_counted_lineage_edges": len(edges),
        "promising_stratum_is_thinly_exercised": len(edges) < 10,
        "promising_note": (
            "the ledger artifact records only the best-so-far improvement chain, so the "
            "promising stratum rests on the few real counted edges it contains; its "
            "selection logic is exercised in tests/test_pmo_donor_scored_bank.py"
        ),
        "seed": args.seed,
        "parents": len(parents),
        "draws_per_parent": args.draws_per_parent,
        "capacity": args.capacity,
        "unbuildable_states": unbuildable,
        "pools": {
            "top_n": census(top_n),
            "stratified": {
                "total": census([e for members in selection.values() for e in members]),
                "weights": weights,
                **{name: census(selection[name]) for name in STRATA},
            },
        },
        "top_n_is_contained_in_the_stratified_bank": set(top_n).issubset(
            {e for members in selection.values() for e in members}
        ),
        "arms": arms,
        "elapsed_seconds": round(time.perf_counter() - began, 2),
    }
    print(
        f"{report['counted_molecules']} counted molecules, {len(edges)} real lineage edges; "
        f"pools top_n={len(top_n)} stratified="
        f"{sum(len(m) for m in selection.values())}"
    )
    for name, arm in arms.items():
        print(
            f"  {name:<11} compiled {arm['compiled']}/{attempts} "
            f"retained_med {arm['retained_fraction']['median']:.3f} "
            f"steps_med {arm['steps']['median']:.0f} "
            f"donors {arm['distinct_donors_used']} "
            f"basins {arm['distinct_product_basins']} "
            f"new {arm['product_basins_absent_from_the_counted_population']}"
            if arm["compiled"]
            else f"  {name:<11} compiled 0/{attempts}"
        )
    print(f"oracle calls 0; {report['elapsed_seconds']}s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
