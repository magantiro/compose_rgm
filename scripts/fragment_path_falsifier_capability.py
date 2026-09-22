#!/usr/bin/env python3
"""Ask whether the v3 predeclared falsifier is CAPABLE of returning FAIL.

The v3 falsifier is ``share of COMMITTED two-core endpoints whose realized
core-to-core path length exceeds the seeded length of one``, with an immutable
denominator.  The v3 bench returned 185 of 185 = 100.0%.  A metric that reads
100.0% on its first and only sample is exactly the shape that has to be
interrogated before it is believed, because a comparison whose expectation is
determined by the code under test cannot fail.

The structural worry is specific.  ``path_permits`` refuses every event that
does not lengthen the path while the path is short of this trajectory's target
AND a transaction site exists; the sites are available at the start state on all
ten released linker prompts (pinned by a test); a single ordinary event has never
been observed to lengthen the path; and a trajectory that admits ZERO events
returns ``None`` before it reaches the commit path.  If that chain holds, then a
trajectory whose composite transaction never fires does not commit at the seeded
length -- it does not commit at all -- so the v2 failure mode (endpoints piling
up AT the seed) is not counted as a failure, it is removed from the denominator.

Three arms settle it.  All three use the same prompts, the same seed and the
same budget; only the named condition differs.

* ``v3``          the mechanism as shipped, with a FRESH RECEIPT PER ATTEMPT so
                  the joint distribution of (committed, transactions fired,
                  realized length) is observed per trajectory rather than pooled
* ``no_transaction``  the gate is left ON and the composite transaction is
                  stubbed to always decline.  This is the mechanism BROKEN.  If
                  the falsifier can fail, this is where it fails.
* ``gate_off``    ``path_program=False``: the same broken mechanism, with the
                  per-event gate removed, which is the v2/baseline denominator.

The prediction under the structural worry is that ``no_transaction`` commits
almost nothing -- so the falsifier reports UNDERPOWERED rather than FAIL -- while
``gate_off`` commits freely at the seeded length and would report FAIL.  Same
broken mechanism, two different verdicts, the difference being only whether the
gate filtered the denominator.

Nothing here reads a drug name or a task label to make a decision; the drug names
are report labels only.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.benchmark import fragment_conditioned_sampler as sampler_module
from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
SEEDED_LENGTH = 1
SEED = 20260921


def _environment() -> dict:
    import numpy
    import rdkit

    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def _run_arm(
    name: str,
    model,
    system,
    prompts,
    *,
    config: SamplerConfig,
    control: AttachmentControlConfig,
    attempts: int,
    stub_transaction: bool,
) -> dict:
    """Run one arm, recording per-TRAJECTORY outcomes rather than pooled counters."""
    original = sampler_module._attempt_path_transaction
    if stub_transaction:
        # Wrap the module GLOBAL the sampler resolves at call time; never a
        # transcription of the sampler loop.
        sampler_module._attempt_path_transaction = lambda *a, **k: None
    try:
        rows = []
        trajectories = []
        for prompt in prompts:
            context = build_prompt_context(
                prompt, config=config, control=control, linker_bridge_atoms=1
            )
            rng = np.random.default_rng(SEED)
            per_drug = []
            for _ in range(attempts):
                receipt = SamplingReceipt()
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
                committed = len(receipt.linker_lengths) > 0
                per_drug.append({
                    "committed": committed,
                    "length": receipt.linker_lengths[0] if committed else None,
                    "transactions": receipt.path_transactions,
                    "transaction_refusals": receipt.path_transaction_refusals,
                    "events": receipt.events[0] if receipt.events else 0,
                })
            trajectories.extend(per_drug)
            committed_rows = [t for t in per_drug if t["committed"]]
            rows.append({
                "drug": prompt.drug_name,
                "attempts": attempts,
                "committed": len(committed_rows),
                "above_seed": sum(
                    1 for t in committed_rows if t["length"] > SEEDED_LENGTH
                ),
                "transactions": sum(t["transactions"] for t in per_drug),
            })
            print(
                f"  {name:16s} {prompt.drug_name:14s} "
                f"committed={len(committed_rows):4d}/{attempts} "
                f"above_seed={sum(1 for t in committed_rows if t['length'] > SEEDED_LENGTH):4d}",
                flush=True,
            )
    finally:
        sampler_module._attempt_path_transaction = original

    committed = [t for t in trajectories if t["committed"]]
    above = [t for t in committed if t["length"] > SEEDED_LENGTH]
    # The joint fact the structural worry is about: did any trajectory reach the
    # commit path WITHOUT the composite transaction ever firing?
    committed_without_transaction = [t for t in committed if t["transactions"] == 0]
    return {
        "arm": name,
        "attempts": len(trajectories),
        "committed": len(committed),
        "commit_rate": len(committed) / len(trajectories) if trajectories else 0.0,
        "above_seed": len(above),
        "metric_percent": (
            100.0 * len(above) / len(committed) if committed else None
        ),
        "length_histogram": dict(
            sorted(Counter(t["length"] for t in committed).items())
        ),
        "committed_without_transaction": len(committed_without_transaction),
        "committed_without_transaction_lengths": dict(
            sorted(Counter(t["length"] for t in committed_without_transaction).items())
        ),
        "total_transactions": sum(t["transactions"] for t in trajectories),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts-per-drug", type=int, default=40)
    parser.add_argument("--mark-attempts-per-event", type=int, default=24)
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig(mark_attempts_per_event=args.mark_attempts_per_event)
    prompts = [
        p for p in load_genmol_prompts(MANIFEST) if p.task is FragmentTask.LINKER_DESIGN
    ]

    on = AttachmentControlConfig(enabled=True, path_program=True)
    off = AttachmentControlConfig(enabled=True, path_program=False)

    started = time.time()
    arms = {}
    for name, control, stub in (
        ("v3", on, False),
        ("no_transaction", on, True),
        ("gate_off", off, True),
    ):
        print(f"-- arm {name} --", flush=True)
        arms[name] = _run_arm(
            name, model, system, prompts,
            config=config, control=control,
            attempts=args.attempts_per_drug, stub_transaction=stub,
        )

    v3 = arms["v3"]
    broken_gated = arms["no_transaction"]
    broken_ungated = arms["gate_off"]
    # The falsifier can only return a verdict at all once the committed sample
    # reaches its predeclared minimum of 150.  Whether the BROKEN mechanism can
    # ever reach that denominator is the whole question.
    denominator_is_conditioned = (
        v3["committed_without_transaction"] == 0
        and broken_gated["commit_rate"] < 0.02 <= broken_ungated["commit_rate"]
    )
    payload = {
        "schema": "compose_fragment_path_falsifier_capability_v1",
        "question": (
            "can the v3 predeclared falsifier return FAIL, or does the per-event "
            "gate remove its failure mode from its own denominator"
        ),
        "environment": _environment(),
        "seed": SEED,
        "mark_attempts_per_event": args.mark_attempts_per_event,
        "arms": arms,
        "denominator_is_conditioned_on_the_mechanism_succeeding": (
            denominator_is_conditioned
        ),
        "reading": (
            "A committed endpoint in the v3 arm with zero fired transactions is "
            "the only way the metric can fall below 100%. The broken-mechanism "
            "arms differ ONLY in whether the per-event gate is active, so any "
            "difference in commit rate between them is the gate converting "
            "seed-length COMMITS into NON-commits."
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    for name, arm in arms.items():
        print(
            f"{name:16s} commit_rate={arm['commit_rate']:.4f} "
            f"committed={arm['committed']:4d} "
            f"metric={arm['metric_percent']} "
            f"committed_without_tx={arm['committed_without_transaction']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
