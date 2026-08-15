#!/usr/bin/env python
"""Run a baseline on DEVELOPMENT seeds under strict-10k semantics.

This is not the official benchmark and cannot become it by accident: it reads
development initialization sets only, and the official ones are behind a latch.

    random-zinc   the FLOOR. Uniform sampling from ZINC-250k. A learned policy
                  that cannot beat 10,000 random drug-like molecules has
                  demonstrated nothing, and without the floor written down it is
                  easy to mistake a respectable-looking hypervolume for a result.

    graph-ga      the REFERENCE. Upstream Jensen GB-GA operators, MOLLEO's own
                  hyperparameters and scalarization, driven under our strict
                  meter. This is the comparator the plan names.

Both are fully resumable: kill either one and run the same command again.

    python scripts/run_task3_dev_baseline.py --policy graph-ga --seed 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.baselines import task3_graph_ga as ga  # noqa: E402
from compose_v4.benchmark.init_sets import (  # noqa: E402
    DEFAULT_INIT_DIR,
    development_init_set,
    init_provenance,
)
from compose_v4.benchmark.task3_run import Task3Run  # noqa: E402

DEFAULT_ZINC = Path("local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv")
OBJECTIVE_NAMES = ("qed", "jnk3", "sa", "gsk3b", "drd2")


def development_pool(zinc: Path) -> list[str]:
    """Canonical ZINC-250k with every official starting molecule removed.

    A development baseline must not be able to draw a molecule that an official
    run starts from, or "development" and "test" stop being disjoint.
    """

    from compose_v4.benchmark.oracles import canonical

    official = json.loads(
        (DEFAULT_INIT_DIR / "official_init_sets.json").read_text())
    reserved = {smiles for molecules in official["sets"].values()
                for smiles in molecules}
    with open(zinc, newline="") as handle:
        raw = [row["smiles"].strip() for row in csv.DictReader(handle)]
    pool: dict[str, None] = {}
    for smiles in raw:
        key = canonical(smiles)
        if key is not None and key not in reserved:
            pool.setdefault(key, None)
    return list(pool)


def run_random_zinc(run: Task3Run, args) -> dict:
    pool = development_pool(args.zinc)
    print(f"development pool: {len(pool):,} molecules (official sets removed)")
    while run.remaining > 0:
        size = min(args.generation, run.remaining)
        index = run.rng.choice(len(pool), size=size, replace=False)
        proposal = run.affordable([pool[int(i)] for i in index])
        if not proposal:
            break
        run.evaluate(proposal)
        run.step += 1
        if run.step % 20 == 0:
            run.checkpoint(policy_state={"phase": "sampling"})
            print(f"  step {run.step}: {run.spent:,}/{args.budget:,}", flush=True)
    run.checkpoint(policy_state={"phase": "finished"})
    return {}


def run_graph_ga(run: Task3Run, args) -> dict:
    resumed = (run.resumed.policy_state if run.resumed else None) or {}
    if "population" in resumed:
        state = ga.GraphGAState.from_dict(resumed)
        ga.restore(state)                     # the size-prior globals do not persist
        print(f"resumed a population of {len(state.population)}")
    else:
        state = ga.initialize(run, development_init_set(args.seed))
        run.step = 1
        print(f"initialized: {run.spent:,} spent, size prior "
              f"{state.average_size:.1f} +/- {state.size_stdev:.1f} atoms")
        run.archive = list(state.population)
        run.checkpoint(policy_state=state.as_dict())

    stalled = 0
    while run.remaining > 0:
        before = run.spent
        state = ga.step(run, state)
        run.step += 1
        run.archive = list(state.population)
        run.checkpoint(policy_state=state.as_dict())
        if run.spent == before:
            # Reproduction can fail entirely at zero oracle cost; without this
            # the loop would spin forever against an unspent budget.
            stalled += 1
            if stalled >= 5:
                print("  reproduction stalled five generations running; stopping")
                break
        else:
            stalled = 0
        if run.step % 10 == 0:
            print(f"  generation {run.step}: {run.spent:,}/{args.budget:,}, "
                  f"best fitness {state.scores[0]:.4f}", flush=True)
    return {"failed_reproductions": state.failed_reproductions,
            "best_fitness": state.scores[0] if state.scores else None,
            "average_size": state.average_size, "size_stdev": state.size_stdev}


def summarize(run: Task3Run, args, root: Path, started: float, extra: dict) -> None:
    hv = run.hypervolume()
    values = np.array(list(run.meter.evaluated().values()))
    best = {name: float(values[:, i].max()) for i, name in enumerate(OBJECTIVE_NAMES)}
    summary = {
        "policy": args.policy, "seed": args.seed, "budget": args.budget,
        "spent": run.spent, "steps": run.step, "hypervolume": hv,
        "best_per_objective_normalised": best,
        "seconds": time.time() - started, "semantics": "strict-10k",
        "init": init_provenance(), **extra,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\n{args.policy} seed {args.seed}: spent {run.spent:,}, HV {hv:.6f}, "
          f"{summary['seconds']:.1f}s")
    print("per-objective best (normalised, higher better): "
          + ", ".join(f"{name}={best[name]:.3f}" for name in OBJECTIVE_NAMES))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="random-zinc",
                        choices=["random-zinc", "graph-ga"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--generation", type=int, default=120,
                        help="molecules proposed per step (random-zinc only)")
    parser.add_argument("--zinc", type=Path, default=DEFAULT_ZINC)
    parser.add_argument("--out", type=Path, default=Path("runs/task3_dev"))
    args = parser.parse_args()

    root = args.out / f"{args.policy}_seed{args.seed}"
    started = time.time()
    run = Task3Run.open(root, seed=args.seed, budget=args.budget,
                        policy=args.policy)
    if run.resumed is not None:
        print(f"resuming at {run.spent:,}/{args.budget:,} spent, step {run.step} "
              f"({run.resumed.uncheckpointed} evaluations arrived after the last "
              f"checkpoint)")
    try:
        if args.policy == "graph-ga":
            extra = run_graph_ga(run, args)
        else:
            if run.step == 0:
                # The initialization set is charged like everything else: 120
                # molecules receiving the objective vector is 120 budget units.
                initial = development_init_set(args.seed)
                run.evaluate(initial)
                run.archive = list(initial)
                run.step = 1
                run.checkpoint(policy_state={"phase": "initialized"})
                print(f"initialized: {run.spent:,} spent")
            extra = run_random_zinc(run, args)
        summarize(run, args, root, started, extra)
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
