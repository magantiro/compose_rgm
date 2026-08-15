#!/usr/bin/env python
"""Run a baseline on DEVELOPMENT seeds under strict-10k semantics.

This is not the official benchmark and cannot become it by accident: it reads
development initialization sets only, and the official ones are behind a latch.

The first policy here is deliberately the dumbest one that is still a real
method -- uniform sampling from ZINC-250k. It exists to establish the FLOOR. A
learned policy that cannot beat "draw 10,000 random drug-like molecules" has not
demonstrated anything, and without the floor on record it is easy to mistake a
respectable-looking hypervolume for a result.

    python scripts/run_task3_dev_baseline.py --policy random-zinc --seed 100
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

from compose_v4.benchmark.init_sets import (  # noqa: E402
    DEFAULT_INIT_DIR,
    development_init_set,
    init_provenance,
)
from compose_v4.benchmark.task3_run import Task3Run  # noqa: E402

DEFAULT_ZINC = Path("local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="random-zinc", choices=["random-zinc"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--generation", type=int, default=120,
                        help="molecules proposed per step")
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

    pool = development_pool(args.zinc)
    print(f"development pool: {len(pool):,} molecules (official sets removed)")

    try:
        # The initialization set is charged like everything else -- it is 120
        # molecules receiving the objective vector, which is 120 budget units.
        if run.step == 0:
            initial = development_init_set(args.seed)
            run.evaluate(initial)
            run.archive = list(initial)
            run.step = 1
            run.checkpoint(policy_state={"phase": "initialized"})
            print(f"initialized: {run.spent:,} spent")

        while run.remaining > 0:
            size = min(args.generation, run.remaining)
            index = run.rng.choice(len(pool), size=size, replace=False)
            proposal = [pool[int(i)] for i in index]
            proposal = run.affordable(proposal)
            if not proposal:
                break
            run.evaluate(proposal)
            run.step += 1
            if run.step % 10 == 0:
                run.checkpoint(policy_state={"phase": "sampling"})
                print(f"  step {run.step}: {run.spent:,}/{args.budget:,}", flush=True)
        run.checkpoint(policy_state={"phase": "finished"})

        hv = run.hypervolume()
        elapsed = time.time() - started
        summary = {
            "policy": args.policy, "seed": args.seed, "budget": args.budget,
            "spent": run.spent, "steps": run.step, "hypervolume": hv,
            "seconds": elapsed, "semantics": "strict-10k",
            "init": init_provenance(),
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"\n{args.policy} seed {args.seed}: spent {run.spent:,}, "
              f"HV {hv:.6f}, {elapsed:.1f}s")

        values = np.array(list(run.meter.evaluated().values()))
        print("per-objective best (normalised, higher better): "
              + ", ".join(f"{name}={values[:, i].max():.3f}" for i, name in
                          enumerate(("qed", "jnk3", "sa", "gsk3b", "drd2"))))
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
