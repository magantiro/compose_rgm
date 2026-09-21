"""Run the OFFICIAL GuacaMol goal-directed suite against COMPOSE.

The scoring path is entirely upstream: this driver constructs the official
benchmark objects and calls the official ``GoalDirectedBenchmark.assess_model``.
``guacamol.assess_goal_directed_generation`` is exactly a loop over that same
call, so running benchmarks one at a time is the identical computation with
incremental, resumable output -- which matters because a single benchmark can
take a long time and a crash must not discard finished ones.

Usage
-----
    PYTHONPATH=<guacamol-pkg>:.:src:scripts python scripts/run_guacamol_goal_directed.py \
        --checkpoint /path/to/lineageB/checkpoint.best_so_far.pt \
        --output diagnostics/guacamol_goal_directed_v1.json \
        --samples 64 --benchmarks all
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from compose_v4.experiments.guacamol_compat import (
    fcd_guard_was_touched,
    install_guacamol_compat,
)
from compose_v4.experiments.guacamol_goal_directed import (
    ComposeGoalDirectedEngine,
    GuacamolRunConfig,
    load_lineage_b,
)


def _environment() -> dict[str, Any]:
    """Record the chemistry kernel: canonical SMILES differ between rdkit
    releases, so a score is only interpretable beside the version that made it."""

    import numpy
    import rdkit
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def _atomic_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="v2", choices=("v1", "v2", "trivial"))
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="'all' or a comma-separated list of official benchmark names",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--proposals-per-event", type=int, default=4)
    parser.add_argument("--guidance-strength", type=float, default=8.0)
    parser.add_argument("--max-guided-events", type=int, default=12)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="stop starting new benchmarks once exceeded (finished ones are kept)",
    )
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)

    applied = install_guacamol_compat()

    import guacamol
    from guacamol.benchmark_suites import goal_directed_benchmark_suite
    from guacamol.goal_directed_generator import GoalDirectedGenerator

    suite = goal_directed_benchmark_suite(version_name=args.suite)
    if args.benchmarks != "all":
        wanted = {name.strip() for name in args.benchmarks.split(",") if name.strip()}
        unknown = wanted - {b.name for b in suite}
        if unknown:
            raise SystemExit(f"unknown benchmark names: {sorted(unknown)}")
        suite = [b for b in suite if b.name in wanted]

    model, source_prior = load_lineage_b(args.checkpoint)
    config = GuacamolRunConfig(
        checkpoint=args.checkpoint,
        samples=args.samples,
        proposals_per_event=args.proposals_per_event,
        guidance_strength=args.guidance_strength,
        max_guided_events=args.max_guided_events,
        max_events=args.max_events,
        n_slots=args.max_atoms,
        workers=args.workers,
        seed=args.seed,
    )

    def progress(message: str) -> None:
        print(json.dumps({"phase": "benchmark_done", "detail": message}), flush=True)

    engine = ComposeGoalDirectedEngine(model, source_prior, config, progress=progress)

    class ComposeGoalDirectedGenerator(GoalDirectedGenerator):
        """The official interface, implemented over COMPOSE."""

        def generate_optimized_molecules(
            self, scoring_function, number_molecules, starting_population=None
        ):
            return engine.generate_optimized_molecules(
                scoring_function, number_molecules, starting_population
            )

    generator = ComposeGoalDirectedGenerator()

    payload: dict[str, Any] = {
        "format": "compose_guacamol_goal_directed_v1",
        "guacamol_version": guacamol.__version__,
        "benchmark_suite_version": args.suite,
        "scored_by": "official guacamol GoalDirectedBenchmark.assess_model",
        "compat_shims_applied": applied,
        "environment": _environment(),
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "samples_per_benchmark": args.samples,
            "proposals_per_event": args.proposals_per_event,
            "guidance_strength": args.guidance_strength,
            "max_guided_events": args.max_guided_events,
            "max_events": args.max_events,
            "max_atoms": args.max_atoms,
            "workers": args.workers,
            "seed": args.seed,
        },
        "results": [],
        "compose_reports": [],
        "not_run": [],
    }

    began = time.monotonic()
    for index, benchmark in enumerate(suite, start=1):
        elapsed = time.monotonic() - began
        if args.time_budget_seconds is not None and elapsed > args.time_budget_seconds:
            payload["not_run"].append(
                {"benchmark": benchmark.name, "reason": "time budget exhausted"}
            )
            _atomic_write(payload, args.output)
            continue
        print(
            json.dumps(
                {
                    "phase": "benchmark_start",
                    "index": index,
                    "total": len(suite),
                    "benchmark": benchmark.name,
                }
            ),
            flush=True,
        )
        engine.set_benchmark_name(benchmark.name)
        try:
            result = benchmark.assess_model(generator)
        except Exception as error:  # noqa: BLE001 - record and continue the suite
            payload["not_run"].append(
                {
                    "benchmark": benchmark.name,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            _atomic_write(payload, args.output)
            continue
        payload["results"].append(
            {
                "benchmark_name": result.benchmark_name,
                "score": result.score,
                "execution_time": result.execution_time,
                "number_scoring_function_calls": result.number_scoring_function_calls,
                "metadata": {
                    key: value
                    for key, value in result.metadata.items()
                    if not key.startswith("internal_similarity_histogram")
                },
                "top_molecules": [
                    {"smiles": smiles, "score": score}
                    for smiles, score in result.optimized_molecules[:10]
                ],
            }
        )
        payload["compose_reports"] = [r.as_dict() for r in engine.reports]
        payload["fcd_guard_touched"] = fcd_guard_was_touched()
        _atomic_write(payload, args.output)

    payload["fcd_guard_touched"] = fcd_guard_was_touched()
    scores = [entry["score"] for entry in payload["results"]]
    payload["summary"] = {
        "benchmarks_run": len(scores),
        "benchmarks_in_suite": len(suite),
        "mean_score_over_run_benchmarks": (sum(scores) / len(scores)) if scores else None,
        "total_scoring_function_calls": sum(
            entry["number_scoring_function_calls"] for entry in payload["results"]
        ),
    }
    _atomic_write(payload, args.output)
    print(json.dumps({"phase": "done", "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
