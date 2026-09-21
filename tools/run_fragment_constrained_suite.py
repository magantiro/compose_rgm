#!/usr/bin/env python3
"""Run the public fragment-constrained benchmark under the OFFICIAL protocol.

Protocol, taken from the released InVirtuoGen evaluator rather than from a paper
description (``in_virtuo_gen/evaluation/downstream.py``):

* 100 generations per drug per task (``--num_samples_eval``, forced to 100).
* Metrics per drug from ``train_utils.metrics.evaluate_smiles``.
* A task row is the UNWEIGHTED MEAN over that task's 10 drugs.
* Reported as mean +/- standard deviation over ``--num_seeds`` (default 3) runs.
* ``scaffold_morphing`` reuses the linker prompts and the linker results, which
  is what the executable upstream code does.

COMPOSE emits the failure placeholder for an attempt that produced nothing
admissible, so the official function counts it invalid.  Official validity is
therefore COMPOSE's STRICT benchmark validity: chemically valid, connected, AND
prompt-satisfying.  Chemical validity of committed endpoints is recorded
separately, because committed states are valid by construction and that is a
different claim from satisfying the prompt.

Zero oracle calls.  QED and SA are the benchmark's own quality diagnostic and are
computed inside the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from compose_v4.benchmark.fragment_conditioned_sampler import (
    FragmentConditioningError,
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
OFFICIAL_SAMPLES_PER_PROMPT = 100
OFFICIAL_SEEDS = 3


def _kernel_provenance() -> dict[str, str]:
    import rdkit
    import torch

    return {
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_task(
    model,
    system,
    prompts,
    task: FragmentTask,
    *,
    seeds: int,
    samples: int,
    config: SamplerConfig,
    verbose: bool = True,
    seed_list: list[int] | None = None,
    drugs: list[str] | None = None,
) -> dict:
    task_prompts = [p for p in prompts if p.task is task]
    if drugs:
        wanted = {d.upper() for d in drugs}
        task_prompts = [p for p in task_prompts if p.drug_name.upper() in wanted]
    per_seed_rows: list[dict[str, float]] = []
    per_drug_detail: dict[str, list[dict]] = defaultdict(list)
    build_failures: list[dict[str, str]] = []

    for seed in (seed_list if seed_list is not None else range(seeds)):
        drug_metrics: list[dict[str, float]] = []
        for prompt in task_prompts:
            try:
                context = build_prompt_context(prompt, config=config)
            except FragmentConditioningError as exc:
                build_failures.append(
                    {"drug": prompt.drug_name, "seed": seed, "error": str(exc)}
                )
                continue

            rng = np.random.default_rng(
                abs(hash((prompt.drug_name, task.value, seed))) % (2**32)
            )
            receipt = SamplingReceipt()
            emitted: list[str] = []
            started = time.time()
            for _ in range(samples):
                out = sample_completion(
                    model, system, context, rng, config=config, receipt=receipt
                )
                emitted.append(out if out else FAILED_SAMPLE_PLACEHOLDER)
            elapsed = time.time() - started

            metrics = official_prompt_metrics(emitted, expected_samples=samples)
            drug_metrics.append(metrics)

            committed = receipt.committed_endpoints
            per_drug_detail[prompt.drug_name].append(
                {
                    "seed": seed,
                    "official": metrics,
                    "attempts": samples,
                    "committed_endpoints": len(committed),
                    "committed_chemically_valid": _chemically_valid(committed),
                    "constraint_failures": receipt.constraint_failures,
                    "lock_rejections": receipt.lock_rejections,
                    "budget_exhausted": receipt.budget_exhausted,
                    "executor_refusals": receipt.executor_refusals,
                    "mean_events": float(np.mean(receipt.events)) if receipt.events else 0.0,
                    "families": dict(receipt.families),
                    "seconds": round(elapsed, 2),
                    "example_completions": [s for s in emitted if s][:5],
                }
            )
            if verbose:
                print(
                    f"  [{task.value} seed={seed}] {prompt.drug_name:14s} "
                    f"val={metrics['validity']:6.2f} uniq={metrics['uniqueness']:6.2f} "
                    f"qual={metrics['quality']:6.2f} div={metrics['diversity']:.3f} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

        if drug_metrics:
            per_seed_rows.append(
                {
                    key: float(np.mean([m[key] for m in drug_metrics]))
                    for key in ("validity", "uniqueness", "quality", "diversity")
                }
            )

    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in per_seed_rows])),
            "std": float(np.std([r[key] for r in per_seed_rows])),
        }
        for key in ("validity", "uniqueness", "quality", "diversity")
    } if per_seed_rows else {}

    return {
        "task": task.value,
        "prompts_declared": len(task_prompts),
        "prompts_scored": len(task_prompts) - len({f["drug"] for f in build_failures}),
        "samples_per_prompt": samples,
        "seeds": seeds,
        "per_seed": per_seed_rows,
        "summary": summary,
        "build_failures": build_failures,
        "per_drug": {k: v for k, v in per_drug_detail.items()},
    }


def _chemically_valid(smiles_list) -> int:
    from rdkit import Chem

    total = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None and len(Chem.GetMolFrags(mol)) == 1:
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task", action="append", default=None, help="repeatable; default = all"
    )
    parser.add_argument("--seeds", type=int, default=OFFICIAL_SEEDS)
    parser.add_argument("--seed-list", type=int, action="append", default=None,
                        help="explicit seed ids for sharding; default = range(--seeds)")
    parser.add_argument("--drug", action="append", default=None,
                        help="repeatable drug filter for sharding")
    parser.add_argument("--samples", type=int, default=OFFICIAL_SAMPLES_PER_PROMPT)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    prompts = load_genmol_prompts(MANIFEST)
    config = SamplerConfig(
        max_events=args.max_events, operational_horizon=args.operational_horizon
    )

    selected = (
        [FragmentTask(t) for t in args.task]
        if args.task
        else list(FragmentTask)
    )

    results = {}
    for task in selected:
        print(f"\n=== {task.value} ===", flush=True)
        results[task.value] = run_task(
            model,
            system,
            prompts,
            task,
            seeds=args.seeds,
            samples=args.samples,
            config=config,
            seed_list=args.seed_list,
            drugs=args.drug,
        )

    payload = {
        "schema": "compose_fragment_official_suite_v1",
        "protocol": {
            "source": "in_virtuo_gen/evaluation/downstream.py @ b50bb3ae",
            "samples_per_prompt": args.samples,
            "seeds": args.seeds,
            "task_row": "unweighted mean over the task's drugs",
            "scaffold_morphing": "upstream copies the linker result",
        },
        "kernel": _kernel_provenance(),
        "checkpoint": {
            "path": str(args.checkpoint),
            "completed_steps": meta.get("completed_steps"),
            "corpus_scope_hash": meta.get("corpus_scope_hash"),
            "atom_vocabulary": "ORGANIC" if len(model.atom_vocabulary.classes) == 15 else "CNOF",
        },
        "sampler": {
            "n_slots": config.n_slots,
            "max_events": config.max_events,
            "operational_horizon": config.operational_horizon,
            "mark_attempts_per_event": config.mark_attempts_per_event,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
