#!/usr/bin/env python
"""Does Task 3 actually pose a five-objective problem?

The question matters before any policy work: hypervolume on this task correlates
0.99 with the single best JNK3 score, which is consistent either with "the other
four objectives are real but easy" or with "the benchmark is a one-objective
search wearing a five-objective costume". Those call for different responses,
and the difference is measurable from runs already banked -- no new oracle calls.

Three measurements, all on the durable ledgers:

1. HOW MUCH OF THE HYPERVOLUME IS ONE MOLECULE. HV of the whole front against HV
   of the single best-JNK3 molecule alone. If one molecule explains nearly all
   of it, the front is decoration.

2. WHAT EACH AXIS IS WORTH. For each objective, recompute HV with that objective
   set to the best value the run achieved -- i.e. "suppose this one were free".
   The drop from the true HV is what that axis is actually costing the optimiser.
   An axis worth ~0 is not part of the problem.

3. HOW MANY MOLECULES CARRY THE FRONT. Marginal contribution of each front
   member (HV(front) - HV(front minus it)). A genuine multi-objective problem has
   many molecules each holding a piece; a disguised single-objective one has one.

4. WHERE THE CONFLICT ACTUALLY IS. JNK3 and GSK3B are both kinases, so "maximise
   JNK3, minimise GSK3B" is a selectivity requirement and may be a real conflict
   rather than two independent axes. This measures the attainable JNK3 at each
   GSK3B ceiling, pooled over every molecule ever evaluated -- which is a
   trade-off curve if there is one, and a flat line if there is not.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.benchmark.molleo_task3 import (  # noqa: E402
    _pareto_mask,
    hypervolume_qmc,
)

NAMES = ("qed", "jnk3", "sa", "gsk3b", "drd2")
#: Cheaper than the reporting default; the quantities here are ratios and
#: differences, not headline numbers.
LOG2 = 17


def load(path: Path) -> np.ndarray:
    values = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                values.append(json.loads(line)["v"])
    return np.asarray(values, dtype=float)


def analyse(points: np.ndarray) -> dict:
    front = points[_pareto_mask(points)]
    total = hypervolume_qmc(front, log2_samples=LOG2)

    best_jnk3 = front[np.argmax(front[:, 1])]
    single = hypervolume_qmc(best_jnk3[None, :], log2_samples=LOG2)

    freed = {}
    for axis, name in enumerate(NAMES):
        lifted = front.copy()
        lifted[:, axis] = front[:, axis].max()
        freed[name] = hypervolume_qmc(lifted, log2_samples=LOG2) - total

    # Marginal contribution per front member. The leave-one-out sweep is
    # O(front) hypervolumes, so it is capped -- and the cap is small, because
    # this quantity is a shape check, not a headline.
    marginal = []
    if len(front) <= 250:
        for i in range(len(front)):
            rest = np.delete(front, i, axis=0)
            marginal.append(total - hypervolume_qmc(rest, log2_samples=14))
    return {
        "evaluated": int(len(points)),
        "front_size": int(len(front)),
        "hypervolume": total,
        "hv_of_best_jnk3_alone": single,
        "fraction_from_one_molecule": single / total if total else 0.0,
        "hv_gain_if_axis_were_free": freed,
        "front_members_worth_over_1pct": (
            int(sum(1 for m in marginal if m > 0.01 * total)) if marginal else None),
        "front_members_worth_over_0.1pct": (
            int(sum(1 for m in marginal if m > 0.001 * total)) if marginal else None),
    }


def selectivity(points: np.ndarray) -> dict:
    """Attainable JNK3 at each GSK3B ceiling, and how the two co-vary.

    Columns 3 and 4 are ALREADY transformed to higher-is-better, so the raw
    activity that has to be kept LOW is `1 - column`.
    """

    jnk3 = points[:, 1]
    gsk3b_raw = 1.0 - points[:, 3]
    drd2_raw = 1.0 - points[:, 4]
    frontier = {}
    for ceiling in (0.02, 0.05, 0.1, 0.2, 0.5, 1.01):
        mask = gsk3b_raw <= ceiling
        frontier[str(ceiling)] = {
            "molecules": int(mask.sum()),
            "best_jnk3": float(jnk3[mask].max()) if mask.any() else 0.0,
        }
    bands = {}
    for low, high in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)):
        mask = (jnk3 >= low) & (jnk3 < high)
        if mask.sum() < 20:
            continue
        bands[f"[{low},{high})"] = {
            "molecules": int(mask.sum()),
            "mean_gsk3b_activity": float(gsk3b_raw[mask].mean()),
            "fraction_gsk3b_over_0.3": float((gsk3b_raw[mask] > 0.3).mean()),
        }
    return {
        "jnk3_at_gsk3b_ceiling": frontier,
        "gsk3b_activity_by_jnk3_band": bands,
        "corr_jnk3_gsk3b": float(np.corrcoef(jnk3, gsk3b_raw)[0, 1]),
        "corr_jnk3_drd2": float(np.corrcoef(jnk3, drd2_raw)[0, 1]),
        "corr_jnk3_qed": float(np.corrcoef(jnk3, points[:, 0])[0, 1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/task3_dev"))
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_objective_structure.json"))
    args = parser.parse_args()

    report: dict = {"log2_samples": LOG2, "runs": {}}
    print(f"{'run':<44}{'front':>7}{'HV':>9}{'1 mol':>9}{'share':>8}")
    for path in sorted(args.runs.glob("*/evaluations.jsonl")):
        name = path.parent.name
        points = load(path)
        if len(points) < 1000:
            continue
        result = analyse(points)
        report["runs"][name] = result
        print(f"{name:<44}{result['front_size']:>7}{result['hypervolume']:>9.4f}"
              f"{result['hv_of_best_jnk3_alone']:>9.4f}"
              f"{result['fraction_from_one_molecule']:>8.1%}")

    if report["runs"]:
        shares = [r["fraction_from_one_molecule"] for r in report["runs"].values()]
        report["fraction_from_one_molecule_mean"] = statistics.fmean(shares)
        print(f"\none molecule explains {statistics.fmean(shares):.1%} of "
              f"hypervolume on average")
        print("\nhypervolume gained if an axis were free (mean over runs):")
        for name in NAMES:
            gains = [r["hv_gain_if_axis_were_free"][name]
                     for r in report["runs"].values()]
            report.setdefault("mean_gain_if_free", {})[name] = statistics.fmean(gains)
            print(f"  {name:<7}{statistics.fmean(gains):+.4f}")
        carried = [r["front_members_worth_over_0.1pct"]
                   for r in report["runs"].values()
                   if r["front_members_worth_over_0.1pct"] is not None]
        if carried:
            report["front_members_worth_over_0.1pct_mean"] = statistics.fmean(carried)
            print(f"\nfront members individually worth >0.1% of HV: "
                  f"mean {statistics.fmean(carried):.1f}")

    pooled = np.vstack([load(p) for p in sorted(args.runs.glob("*/evaluations.jsonl"))
                        if len(load(p)) >= 1000])
    report["selectivity"] = selectivity(pooled)
    report["selectivity"]["pooled_molecules"] = int(len(pooled))
    print(f"\nSELECTIVITY, pooled over {len(pooled):,} molecules "
          f"(corr jnk3/gsk3b = {report['selectivity']['corr_jnk3_gsk3b']:+.3f}):")
    for ceiling, row in report["selectivity"]["jnk3_at_gsk3b_ceiling"].items():
        print(f"  gsk3b <= {ceiling:<5} n={row['molecules']:>7,}   "
              f"best jnk3 = {row['best_jnk3']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
