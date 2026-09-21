#!/usr/bin/env python3
"""Re-run one shard deterministically and write ALL 100 emitted strings.

The sweep stores metrics and five example completions, not the full emission
list, because 120 shards of 100 SMILES is bulk nobody reads.  But a
cross-kernel check needs the molecules themselves: the question "do the
reported metrics move under the pinned RDKit" is only answerable by scoring
the SAME strings twice.  The sweep's seeding is deterministic, so this
reproduces the identical emission list without re-deriving it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_fragment_constrained_suite import MANIFEST, prompt_rng_seed

from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import FAILED_SAMPLE_PLACEHOLDER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import rdkit
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    task = FragmentTask(args.task)
    prompt = next(
        p
        for p in load_genmol_prompts(MANIFEST)
        if p.drug_name == args.drug and p.task is task
    )
    config = SamplerConfig()
    context = build_prompt_context(prompt, config=config)
    rng = np.random.default_rng(prompt_rng_seed(args.drug, task.value, args.seed))

    receipt = SamplingReceipt()
    emitted = []
    for _ in range(args.samples):
        out = sample_completion(model, system, context, rng, config=config, receipt=receipt)
        emitted.append(out if out else FAILED_SAMPLE_PLACEHOLDER)

    payload = {
        "schema": "compose_fragment_emissions_v1",
        "task": task.value,
        "drug": args.drug,
        "seed": args.seed,
        "rng_seed": prompt_rng_seed(args.drug, task.value, args.seed),
        "generating_rdkit": rdkit.__version__,
        "prompt_fragments": list(prompt.fragments),
        "original_smiles": prompt.original_smiles,
        "attempts": args.samples,
        "emitted": emitted,
        "committed_endpoints": list(receipt.committed_endpoints),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output} ({sum(1 for e in emitted if e)}/{args.samples} non-empty)")


if __name__ == "__main__":
    raise SystemExit(main())
