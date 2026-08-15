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

    # Marginal contribution per front member, on the fronts small enough to
    # afford it; the leave-one-out sweep is O(front) hypervolumes.
    marginal = []
    if len(front) <= 400:
        for i in range(len(front)):
            rest = np.delete(front, i, axis=0)
            marginal.append(total - hypervolume_qmc(rest, log2_samples=LOG2))
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
