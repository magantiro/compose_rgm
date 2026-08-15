#!/usr/bin/env python
"""Run the adaptive Pareto navigator on DEVELOPMENT seeds.

⚠️ THE NUMBERS THIS PRODUCES ARE NOT COMPOSE RESULTS.

The outer algorithm here -- archive maintenance, hypervolume-aware region
targeting, start-state selection, STOP, metered candidate selection -- is the
real thing. The inner `navigate()` is a DEVELOPMENT STAND-IN built from
Graph-GA's edit operators, because the COMPOSE-native five-objective expansion
policy is deliberately not chosen yet.

So what this measures is exactly one thing: whether the OUTER algorithm is worth
having, holding the inner primitive dumb. That is a useful question on its own --
if archive-driven region targeting cannot beat a scalarized GA even in principle,
no navigator will save it -- but a result from it must never be reported as
COMPOSE performance.

    python scripts/run_task3_dev_policy.py --seed 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.baselines.task3_graph_ga import size_prior  # noqa: E402
from compose_v4.benchmark.init_sets import (  # noqa: E402
    development_init_set,
    init_provenance,
)
from compose_v4.benchmark.task3_run import Task3Run  # noqa: E402
from compose_v4.policy.task3 import (  # noqa: E402
    AdaptiveParetoNavigatorPolicy,
    MarginalGainRegions,
    NavigationBudget,
    NearestRealizedStart,
    RandomEditNavigator,
    RandomRegion,
    RegionOriginStart,
)

OBJECTIVE_NAMES = ("qed", "jnk3", "sa", "gsk3b", "drd2")
REGION_SELECTORS = {"marginal-gain": MarginalGainRegions, "random": RandomRegion}
START_SELECTORS = {"region-origin": RegionOriginStart,
                   "nearest-realized": NearestRealizedStart}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--regions", default="marginal-gain",
                        choices=sorted(REGION_SELECTORS))
    parser.add_argument("--starts", default="region-origin",
                        choices=sorted(START_SELECTORS))
    parser.add_argument("--steps", type=int, default=8,
                        help="edits per navigation")
    parser.add_argument("--candidates", type=int, default=4,
                        help="molecules charged per iteration")
    parser.add_argument("--out", type=Path, default=Path("runs/task3_dev"))
    args = parser.parse_args()

    initial = development_init_set(args.seed)
    average_size, size_stdev = size_prior(initial)
    policy = AdaptiveParetoNavigatorPolicy(
        navigator=RandomEditNavigator(average_size=average_size,
                                      size_stdev=size_stdev),
        regions=REGION_SELECTORS[args.regions](),
        starts=START_SELECTORS[args.starts](),
        navigation_budget=NavigationBudget(steps=args.steps),
        candidates_per_iteration=args.candidates,
    )
    tag = f"adaptive-{args.regions}-{args.starts}"
    root = args.out / f"{tag}_seed{args.seed}"
    started = time.time()
    run = Task3Run.open(root, seed=args.seed, budget=args.budget, policy=tag)
    if run.resumed is not None:
        print(f"resuming at {run.spent:,}/{args.budget:,} spent")
    try:
        stats = policy.run(run, initial)
        hv = run.hypervolume()
        values = np.array(list(run.meter.evaluated().values()))
        best = {name: float(values[:, i].max())
                for i, name in enumerate(OBJECTIVE_NAMES)}
        summary = {
            "policy": tag, "seed": args.seed, "budget": args.budget,
            "spent": run.spent, "steps": run.step, "hypervolume": hv,
            "best_per_objective_normalised": best,
            "seconds": time.time() - started, "semantics": "strict-10k",
            "navigator": policy.navigator.name,
            "navigator_is_a_stand_in": True,
            "init": init_provenance(), **stats,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"\n{tag} seed {args.seed}: spent {run.spent:,}, HV {hv:.6f}, "
              f"{summary['seconds']:.1f}s")
        print("per-objective best: "
              + ", ".join(f"{n}={best[n]:.3f}" for n in OBJECTIVE_NAMES))
        print(f"iterations {stats['iterations']}, repeats {stats['repeats']}, "
              f"front {stats['front_size']}, "
              f"mean region gain {stats['mean_region_gain']:.4f}")
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
