#!/usr/bin/env python
"""Tiny paired charged A/B over CACHED R_theta fibers. DEVELOPMENT-ONLY.

Not Task 3 performance: the initialization is the seeded fixture, which
deliberately contains JNK3 actives whose labels earlier runs paid for.

Runs locally, because the expensive part -- R_theta expansion at ~12.5 s -- is
already cached and the oracle costs ~1.6 ms a molecule. That is what makes the
target cadence possible: an idea can go from offline smoke to a charged paired
result in minutes instead of a Modal round trip.

WHAT IS HELD IDENTICAL: the frozen seeded archive, the surrogate class and its
data, the cached fibers, the additional metered budget, the candidates per
decision, the seed, and the restriction of start selection to states whose
fibers are cached. That last one is a real simplification and it applies to BOTH
arms, so it cannot favour either.

VARIED: only the selection policy.

WHAT IS READ, IN THIS ORDER (the middle one is new, and it is the one that was
missing last time -- it measures whether the mechanism ENGAGED at all):
  1 realized HV lift
  2 fraction of decisions where the arms actually chose different molecules
  3 whether the lift is broad or carried by one lucky molecule
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.benchmark.molleo_task3 import _pareto_mask, hypervolume_qmc  # noqa: E402
from compose_v4.benchmark.oracles.task3 import navigation_lockout  # noqa: E402
from compose_v4.benchmark.task3_run import Task3Run  # noqa: E402
from compose_v4.policy.task3.archive import ParetoArchive  # noqa: E402
from compose_v4.policy.task3.steering import (  # noqa: E402
    FixedScalarization,
    ReachableHVI,
)
from compose_v4.policy.task3.surrogate import TanimotoKNN  # noqa: E402

FIXTURE = Path("artifacts/benchmarks/task3_mechanism_archives_v1/archives.json")


def load_fibers(directory: Path) -> dict[str, list[str]]:
    fibers: dict[str, list[str]] = {}
    for path in sorted(directory.glob("fibers_*.json")):
        payload = json.loads(path.read_text())
        for start, rows in payload["fibers"].items():
            fibers[start] = [row[0] for row in rows]
    return fibers


def restricted_start(steering, archive: ParetoArchive, target, rng,
                     available: set[str], exhausted: set[str]) -> str | None:
    """The arm's own choice, restricted to states we can actually expand.

    Identical machinery for both arms: each still selects by ITS OWN rule, and
    the only shared constraint is that the state must have a cached fiber and
    must not be spent.
    """

    eligible = available - exhausted
    if not eligible:
        return None
    with archive.restricted_to(eligible):
        return steering.start(archive, target, rng)


def run_arm(steering, seeded, fibers, *, seed: int, budget: int, candidates: int,
            root: Path) -> dict:
    archive = ParetoArchive()
    archive.add_many(seeded)
    surrogate = TanimotoKNN()
    surrogate.update(list(seeded), list(seeded.values()))

    run = Task3Run.open(root, seed=seed, budget=budget, policy=steering.name)
    if run.resumed is not None:
        already = run.meter.evaluated()
        archive.add_many(already)
        surrogate.update(list(already), list(already.values()))

    available = set(fibers) & set(archive.values)
    exhausted: set[str] = set()
    picks: list[tuple[str, ...]] = []
    started = time.time()
    try:
        while run.remaining > 0:
            target = steering.target(archive, run.rng)
            if target is None:
                break
            with navigation_lockout():
                proposed: list[str] = []
                for _ in range(12):
                    start = restricted_start(steering, archive, target, run.rng,
                                             available, exhausted)
                    if start is None:
                        break
                    proposed = [s for s in fibers[start]
                                if s not in archive.values]
                    if proposed:
                        break
                    exhausted.add(start)
                if not proposed:
                    break
                predicted, spread = surrogate.predict_with_spread(proposed)
                try:
                    ranks = steering.rank(predicted, target, spread)
                except TypeError:   # arms that take no uncertainty signal
                    ranks = steering.rank(predicted, target)
                order = np.argsort(-ranks)
            chosen = run.affordable([proposed[int(i)] for i in order[:candidates]])
            if not chosen:
                break
            picks.append(tuple(chosen))
            values = run.evaluate(chosen)
            for smiles, value in zip(chosen, values):
                archive.add(smiles, value)
            surrogate.update(chosen, list(values))
            run.step += 1
            if run.step % 10 == 0:
                run.archive = [s for s, _ in archive.front()]
                run.checkpoint(policy_state={"steering": steering.name})
        run.checkpoint(policy_state={"steering": steering.name, "done": True})

        new = np.asarray(list(run.meter.evaluated().values()))
        base_pts = np.asarray(list(seeded.values()))
        base = hypervolume_qmc(base_pts, log2_samples=17)
        after = hypervolume_qmc(np.vstack([base_pts, new]), log2_samples=17)
        drop = int(np.argmax(new[:, 1]))
        without = hypervolume_qmc(
            np.vstack([base_pts, np.delete(new, drop, axis=0)]), log2_samples=17)
        combined = np.vstack([base_pts, new])
        mask = _pareto_mask(combined)
        return {
            "steering": steering.name, "seed": seed, "charged": run.spent,
            "decisions": len(picks), "exhausted_states": len(exhausted),
            "hv_base": base, "hv_after": after, "lift": after - base,
            "lift_without_best_point": without - base,
            "share_from_best_point": ((after - without) / (after - base)
                                      if after > base else 0.0),
            "new_on_front": int(mask[len(base_pts):].sum()),
            "seconds": time.time() - started,
            "picks": [list(p) for p in picks],
        }
    finally:
        run.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fibers", type=Path, default=Path("/tmp/fiber_cache/seed100"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--optimism", type=float, default=0.5,
                        help="kappa. Chosen offline as the SMALLEST value that "
                             "engages the mechanism (88%% at 0.5, 16%% at 0), on "
                             "an engagement criterion independent of any HV "
                             "outcome -- so it is not tuned to the result.")
    parser.add_argument("--out", type=Path, default=Path("runs/task3_reachable"))
    parser.add_argument("--report", type=Path,
                        default=Path("diagnostics/task3_reachable_ab.json"))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    fibers = load_fibers(args.fibers)
    if not fibers:
        raise SystemExit(f"no cached fibers under {args.fibers}")
    report: dict = {
        "STATUS": "DEVELOPMENT-ONLY -- NOT TASK 3 PERFORMANCE (seeded init)",
        "cached_fibers": len(fibers), "budget_per_arm": args.budget,
        "optimism": args.optimism, "arms": {}}

    for seed in args.seeds:
        seeded = {s: tuple(v) for s, v in
                  fixture["archives"][str(seed)]["molecules"].items()}
        print(f"\n=== seed {seed} ===")
        for steering in (FixedScalarization(),
                         ReachableHVI(optimism=args.optimism)):
            result = run_arm(steering, seeded, fibers, seed=seed,
                             budget=args.budget, candidates=args.candidates,
                             root=args.out / f"{steering.name}_seed{seed}")
            report["arms"].setdefault(steering.name, []).append(result)
            print(f"  {steering.name:<22} charged {result['charged']:>4}  "
                  f"lift {result['lift']:+.4f}  "
                  f"(w/o best point {result['lift_without_best_point']:+.4f}, "
                  f"share {result['share_from_best_point']:.0%})  "
                  f"new on front {result['new_on_front']}")

    print("\n--- READ IN ORDER (development-only) ---")
    names = list(report["arms"])
    for name in names:
        lifts = [r["lift"] for r in report["arms"][name]]
        print(f"1 HV lift   {name:<22} mean {statistics.fmean(lifts):+.4f}")
    if len(names) == 2:
        a, b = (report["arms"][n] for n in names)
        divergences = []
        for ra, rb in zip(a, b):
            pa = [set(p) for p in ra["picks"]]
            pb = [set(p) for p in rb["picks"]]
            n = min(len(pa), len(pb))
            if n:
                divergences.append(statistics.fmean(
                    1.0 - len(pa[i] & pb[i]) / max(len(pa[i]), 1) for i in range(n)))
        if divergences:
            report["decision_divergence"] = statistics.fmean(divergences)
            print(f"2 divergence  arms chose differently on "
                  f"{statistics.fmean(divergences):.1%} of decisions")
    for name in names:
        shares = [r["share_from_best_point"] for r in report["arms"][name]]
        print(f"3 breadth   {name:<22} mean share from one molecule "
              f"{statistics.fmean(shares):.0%}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
