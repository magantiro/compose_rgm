#!/usr/bin/env python3
"""Sample the v3 path program against its PREDECLARED falsifier.

The falsifier, threshold, decision rule and sample size were committed before
any v3 sample was drawn, in
``diagnostics/fragment_path_program_v3_predeclaration.json``.  This script does
not restate them as new choices; it reads them and reports against them.

Summary of what it is measuring, so the numbers are not quoted loose:

* the metric is the share of COMMITTED two-core endpoints whose realized
  core-to-core path length exceeds the seeded length of one
* the decision rule is the LOWER BOUND of the 95% Wilson interval clearing 50%,
  not a point estimate, because v2 failed at 90.3% against a 90% threshold on
  n=31 where one endpoint moved the number three points
* the denominator is all committed two-core endpoints across all ten released
  linker drugs, pooled, and is declared immutable

Every emitted molecule is persisted, not just counters.  A run that stores
aggregates cannot answer a question posed after the fact, which is how an
earlier fragment sweep lost its secondary metrics permanently.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
PREDECLARATION = Path("diagnostics/fragment_path_program_v3_predeclaration.json")
SEEDED_LENGTH = 1


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson interval; exact at the boundaries."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts-per-drug", type=int, default=150)
    parser.add_argument("--bridge-atoms", type=int, default=1)
    args = parser.parse_args()

    predeclared = json.loads(PREDECLARATION.read_text())
    threshold = float(predeclared["falsifier"]["threshold_percent"]) / 100.0
    n_minimum = int(predeclared["sample_size"]["n_minimum"])

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True, path_program=True)

    prompts = [
        p for p in load_genmol_prompts(MANIFEST) if p.task is FragmentTask.LINKER_DESIGN
    ]

    histogram: Counter[int] = Counter()
    rows = []
    transactions = refusals = 0
    for prompt in prompts:
        context = build_prompt_context(
            prompt, config=config, control=control,
            linker_bridge_atoms=args.bridge_atoms,
        )
        rng = np.random.default_rng(20260921)
        receipt = SamplingReceipt()
        emitted = []
        for _ in range(args.attempts_per_drug):
            emitted.append(
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
            )
        for length in receipt.linker_lengths:
            histogram[length] += 1
        transactions += receipt.path_transactions
        refusals += receipt.path_transaction_refusals
        above = sum(1 for n in receipt.linker_lengths if n > SEEDED_LENGTH)
        rows.append({
            "drug": prompt.drug_name,
            "attempts": args.attempts_per_drug,
            "committed": len(receipt.linker_lengths),
            "above_seed": above,
            "length_histogram": dict(sorted(Counter(receipt.linker_lengths).items())),
            "path_transactions": receipt.path_transactions,
            "path_transaction_refusals": receipt.path_transaction_refusals,
            # Persist the molecules, never only the counters.
            "committed_endpoint_smiles": list(receipt.committed_endpoints),
            "emitted_samples": [s or "" for s in emitted],
        })
        print(
            f"{prompt.drug_name:14s} committed={len(receipt.linker_lengths):4d} "
            f"above_seed={above:4d} tx={receipt.path_transactions:4d}",
            flush=True,
        )

    total = sum(histogram.values())
    above = sum(count for length, count in histogram.items() if length > SEEDED_LENGTH)
    point, low, high = wilson(above, total)
    powered = total >= n_minimum
    verdict = (
        "PASS" if (powered and low > threshold)
        else "FAIL" if powered
        else "UNDERPOWERED -- n below the predeclared minimum, no verdict"
    )
    payload = {
        "schema": "compose_fragment_path_program_v3_bench_v1",
        "predeclaration": str(PREDECLARATION),
        "predeclaration_committed_before_sampling": True,
        "metric": "share of committed two-core endpoints exceeding the seeded length",
        "threshold_percent": threshold * 100.0,
        "decision_rule": "95% Wilson lower bound must exceed the threshold",
        "n_minimum": n_minimum,
        "committed_endpoints": total,
        "above_seed": above,
        "point_estimate_percent": None if math.isnan(point) else point * 100.0,
        "wilson_95_percent": [
            None if math.isnan(low) else low * 100.0,
            None if math.isnan(high) else high * 100.0,
        ],
        "sample_meets_predeclared_minimum": powered,
        "verdict": verdict,
        "length_histogram": dict(sorted(histogram.items())),
        "path_transactions": transactions,
        "path_transaction_refusals": refusals,
        "commit_rate": total / (args.attempts_per_drug * len(prompts)),
        "reachable_is_not_yield": (
            "The hand-constructed sequence reaches a length-2 linker on 10 of 10 "
            "drugs. That bounds what is REACHABLE and does not predict yield, "
            "which is why the commit rate is reported beside the metric."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"\ncommitted n={total} (minimum {n_minimum})  above_seed {above}/{total}"
        f" = {point * 100:.1f}%  95% Wilson [{low * 100:.1f}, {high * 100:.1f}]"
        f"\nVERDICT: {verdict}\nwrote {args.output}"
    )


if __name__ == "__main__":
    main()
