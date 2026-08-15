#!/usr/bin/env python
"""DEVELOPMENT-ONLY MECHANISM TEST -- NOT TASK 3 PERFORMANCE.

    Conditional on having a minimally informative archive, does archive-aware
    region targeting outperform fixed scalarization at filling high-JNK3 /
    low-GSK3B Pareto regions?

⚠️ THE INITIALIZATION IS SEEDED AND THEREFORE DEPARTS FROM THE BENCHMARK.
MOLLEO Task 3 specifies 120 random ZINC-250k molecules. This starts from a
frozen archive deliberately containing JNK3 actives, whose labels were paid for
by earlier development runs and are GIVEN here rather than charged. No number
produced by this script is Task 3 performance, and it must never be reported as
such. It exists to answer one causal question cheaply, and to answer it before
anyone spends thousands of calls finding out the navigator was not useful.

WHY SEEDED. `scripts/task3_surrogate_signal.py` measured that at 120-500 random
paid labels a surrogate ranks JNK3 at rho 0.28 and GSK3B at 0.22-0.32, because
such a sample contains between zero and four actives. Both arms would be driving
without headlights, and a null result would say nothing about steering. Seeding
separates the two problems the task actually poses:

    Phase A   find the first informative actives
    Phase B   navigate the tradeoff once informative labels exist

This tests Phase B only. Phase A is its own question and is not addressed here.

WHAT IS HELD IDENTICAL. Both arms get the same frozen seeded archive, the same
surrogate class fit on the same data, the same expansion operator, the same
additional metered budget, the same candidates-per-iteration, and the same seed.
The ONLY difference is how purpose is specified: a fixed weighted sum, or an
adaptive aspiration at the currently undercovered part of the front.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.baselines.task3_graph_ga import size_prior  # noqa: E402
from compose_v4.benchmark.molleo_task3 import hypervolume_qmc  # noqa: E402
from compose_v4.benchmark.oracles.task3 import navigation_lockout  # noqa: E402
from compose_v4.benchmark.task3_run import Task3Run  # noqa: E402
from compose_v4.policy.task3 import ParetoArchive, RandomEditNavigator  # noqa: E402
from compose_v4.policy.task3.interfaces import NavigationBudget  # noqa: E402
from compose_v4.policy.task3.steering import (  # noqa: E402
    AdaptiveRegion,
    FixedScalarization,
)
from compose_v4.policy.task3.surrogate import TanimotoKNN  # noqa: E402

NAMES = ("qed", "jnk3", "sa", "gsk3b", "drd2")
#: "Selective" = real JNK3 activity without the GSK3B cross-reactivity that
#: comes with it. Both thresholds are on the RAW activities.
SELECTIVE_JNK3 = 0.4
SELECTIVE_GSK3B = 0.1


def build_seeded_archive(ledgers: list[Path], size: int, actives: int,
                         seed: int) -> dict[str, tuple[float, ...]]:
    """A frozen, minimally informative archive: mostly ordinary, a few actives.

    Drawn deterministically from labels earlier runs already paid for, so both
    arms start from exactly the same evidence and neither is charged for it.
    """

    pool: dict[str, tuple[float, ...]] = {}
    for path in ledgers:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    pool.setdefault(record["smiles"], tuple(record["v"]))
    items = sorted(pool.items())
    rng = np.random.default_rng(seed)
    active = [(s, v) for s, v in items if v[1] >= SELECTIVE_JNK3]
    ordinary = [(s, v) for s, v in items if v[1] < SELECTIVE_JNK3]
    if len(active) < actives:
        raise SystemExit(f"only {len(active)} actives available, need {actives}")
    chosen = [active[i] for i in rng.choice(len(active), actives, replace=False)]
    chosen += [ordinary[i] for i in
               rng.choice(len(ordinary), size - actives, replace=False)]
    return dict(sorted(chosen))


def selective(values: np.ndarray) -> np.ndarray:
    """Rows that are JNK3-active AND GSK3B-quiet. Columns 3/4 are 1 - activity."""

    return (values[:, 1] >= SELECTIVE_JNK3) & ((1.0 - values[:, 3]) <= SELECTIVE_GSK3B)


def run_arm(steering, seeded: dict[str, tuple[float, ...]], *, seed: int,
            budget: int, root: Path, navigator, candidates: int,
            proposals: int) -> dict:
    """One arm. Identical in every respect except `steering`."""

    archive = ParetoArchive()
    archive.add_many(seeded)
    surrogate = TanimotoKNN()
    surrogate.update(list(seeded), list(seeded.values()))

    run = Task3Run.open(root, seed=seed, budget=budget, policy=steering.name)
    started = time.time()
    attacks = []
    try:
        while run.remaining > 0:
            target = steering.target(archive, run.rng)
            if target is None:
                break
            start = steering.start(archive, target, run.rng)
            if start is None:
                break
            # The navigator proposes; it cannot evaluate. The lockout makes that
            # structural rather than a convention.
            with navigation_lockout():
                proposed: list[str] = []
                for _ in range(proposals):
                    trajectory = navigator.navigate(
                        start, target, NavigationBudget(steps=4), run.rng,
                        evidence=archive)
                    proposed.extend(trajectory.states[1:])
                proposed = [s for s in dict.fromkeys(proposed)
                            if s not in archive.values]
                if not proposed:
                    continue
                # The surrogate ranks; both arms use the same one, differing
                # only in what "good" means.
                predicted = surrogate.predict(proposed)
                order = np.argsort(-steering.rank(predicted, target))
            chosen = [proposed[int(i)] for i in order[:candidates]]
            chosen = run.affordable(chosen)
            if not chosen:
                break
            before = archive.gain_of(target.as_array())
            values = run.evaluate(chosen)
            for smiles, value in zip(chosen, values):
                archive.add(smiles, value)
            surrogate.update(chosen, list(values))
            # Did attacking this region actually improve THIS region?
            attacks.append({"gain_before": before,
                            "gain_after": archive.gain_of(target.as_array()),
                            "target": list(target.target)})
            run.step += 1
            if run.step % 20 == 0:
                run.archive = [s for s, _ in archive.front()]
                run.checkpoint(policy_state={"steering": steering.name})
        run.checkpoint(policy_state={"steering": steering.name, "done": True})

        evaluated = np.asarray(list(run.meter.evaluated().values()))
        seeded_values = np.asarray(list(seeded.values()))
        combined = np.vstack([seeded_values, evaluated]) if len(evaluated) else seeded_values
        closed = [a for a in attacks if a["gain_after"] < a["gain_before"] - 1e-12]
        return {
            "steering": steering.name,
            "charged": int(run.spent),
            "iterations": len(attacks),
            "seconds": time.time() - started,
            "hv_seeded_only": hypervolume_qmc(seeded_values, log2_samples=17),
            "hv_after": hypervolume_qmc(combined, log2_samples=17),
            "front_size": len(archive.front()),
            "new_selective": int(selective(evaluated).sum()) if len(evaluated) else 0,
            "best_jnk3_new": float(evaluated[:, 1].max()) if len(evaluated) else 0.0,
            "attacks": len(attacks),
            "attacks_that_improved_their_target": len(closed),
        }
    finally:
        run.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledgers", type=Path, nargs="+",
                        default=sorted(Path("runs/task3_dev").glob(
                            "graph-ga_seed10*/evaluations.jsonl")))
    parser.add_argument("--archive-size", type=int, default=400)
    parser.add_argument("--archive-actives", type=int, default=20)
    parser.add_argument("--budget", type=int, default=300,
                        help="ADDITIONAL strict calls per arm")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--proposals", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=Path, default=Path("runs/task3_mechanism"))
    parser.add_argument("--report", type=Path,
                        default=Path("diagnostics/task3_seeded_mechanism_test.json"))
    args = parser.parse_args()

    report: dict = {
        "STATUS": "DEVELOPMENT-ONLY MECHANISM TEST -- NOT TASK 3 PERFORMANCE",
        "question": ("conditional on a minimally informative archive, does "
                     "archive-aware region targeting outperform fixed "
                     "scalarization at filling high-JNK3 / low-GSK3B regions?"),
        "initialization": ("SEEDED, and therefore NOT the benchmark's random-120. "
                           "Seed labels were paid for by earlier runs and are "
                           "given, not charged."),
        "surrogate_caveat": ("Morgan-bit k-NN predicting Morgan-RF oracles is an "
                             "OPTIMISTIC control, not evidence about an R_theta "
                             "representation controller"),
        "expansion_operator": ("shared Graph-GA edit operators (stand-in), held "
                               "IDENTICAL across arms; R_theta expansion is the "
                               "next rung, not this one"),
        "held_identical": ["seeded archive", "surrogate", "expansion operator",
                           "additional budget", "candidates per iteration", "seed"],
        "varied": "how purpose is specified (fixed scalar sum vs adaptive region)",
        "attacks_metric_caveat": (
            "'attacks_that_improved_their_target' is only meaningful for the "
            "ADAPTIVE arm. The fixed arm's target is the constant (1,1,1,1,1) "
            "corner, which dominates every probe, so its 'gain' is just the "
            "uncovered fraction and 'improved' degenerates to 'hypervolume went "
            "up at all'. Do not read the two columns as like for like."),
        "arms": {},
    }

    for seed in args.seeds:
        seeded = build_seeded_archive(list(args.ledgers), args.archive_size,
                                      args.archive_actives, seed)
        digest = hashlib.sha256("\n".join(sorted(seeded)).encode()).hexdigest()
        average_size, size_stdev = size_prior(list(seeded))
        print(f"\n=== seed {seed}: archive {len(seeded)} molecules "
              f"({args.archive_actives} actives), sha {digest[:12]} ===")
        for steering in (FixedScalarization(), AdaptiveRegion()):
            navigator = RandomEditNavigator(average_size=average_size,
                                            size_stdev=size_stdev)
            result = run_arm(
                steering, seeded, seed=seed, budget=args.budget,
                root=args.out / f"{steering.name}_seed{seed}",
                navigator=navigator, candidates=args.candidates,
                proposals=args.proposals)
            result["archive_sha256"] = digest
            report["arms"].setdefault(steering.name, []).append(result)
            print(f"  {steering.name:<22} charged {result['charged']:>4}  "
                  f"HV {result['hv_seeded_only']:.4f} -> {result['hv_after']:.4f}  "
                  f"new selective {result['new_selective']:>3}  "
                  f"best jnk3 {result['best_jnk3_new']:.2f}  "
                  f"targets improved {result['attacks_that_improved_their_target']}"
                  f"/{result['attacks']}")

    print("\n--- summary (development-only; NOT Task 3 performance) ---")
    for name, rows in report["arms"].items():
        lifts = [r["hv_after"] - r["hv_seeded_only"] for r in rows]
        report.setdefault("summary", {})[name] = {
            "hv_lift_mean": statistics.fmean(lifts),
            "hv_lift": lifts,
            "new_selective_total": sum(r["new_selective"] for r in rows),
            "best_jnk3": max(r["best_jnk3_new"] for r in rows),
        }
        print(f"{name:<22} mean HV lift {statistics.fmean(lifts):+.4f}   "
              f"new selective {sum(r['new_selective'] for r in rows)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
