#!/usr/bin/env python3
"""Is a genuine linker reachable once the constructed join is released?

The v2 invalidation established that linker design was not being measured: the
adapter joins the two retained cores with a DIRECT BOND so the executor has a
connected state to start from, the region lock then pins that bond, and every
emitted "linker" is zero atoms long.  The benchmark's own endpoint test cannot
see it, because each core satisfies the other core's attachment requirement.

Attachment control changes two things that bear on it.  The join pair is
released from the region lock, so the bond CAN be displaced; and coverage
counts only NON-LOCKED neighbours, so the join does not count and both
declared sites read as unsatisfied, which is what staging then tries to fix.

This probe asks whether that is enough.  It reports, per drug:

    free valence at each declared site in the constructed start state
    whether any single legal event can increase coverage at all
    what the sampler actually reaches in N attempts

The first two are structural and need no model.  A site with no free valence
cannot accept a bond, and if BOTH sites are saturated then no first event can
make progress, so the trajectory dies at event zero however good the proposal
distribution is.  That is a START-STATE property, not a proposal-coverage one,
and the distinction is the whole point of the probe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compose_v4.benchmark.fragment_attachment_control import (
    AttachmentControlConfig,
    AttachmentController,
)
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")


def structural_verdict(context, controller) -> dict:
    """What the START STATE allows, before any model is consulted."""
    state = context.start_state
    free = {
        int(slot): int(state.implicit_h_counts[slot])
        for slot in context.attachment.interfaces
    }
    open_sites = [slot for slot, h in free.items() if h > 0]
    return {
        "free_valence_by_interface": free,
        "interfaces_with_free_valence": len(open_sites),
        "cores_directly_bonded_at_start": not controller.cores_are_separated(state),
        "join_released_from_lock": sorted(
            sorted(pair) for pair in context.attachment.released_pairs
        ),
        # With every declared site saturated by the constructed join, no atom
        # can be attached at a declared site, and growth anywhere else is
        # outside the declared interfaces, so no first event can increase
        # coverage.  Staging then refuses every candidate and the trajectory
        # ends with zero events.
        "first_event_can_increase_coverage": bool(open_sites),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--drug", action="append", default=None)
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    prompts = load_genmol_prompts(MANIFEST)
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True)

    rows = []
    for prompt in prompts:
        if prompt.task is not FragmentTask.LINKER_DESIGN:
            continue
        if args.drug and prompt.drug_name.upper() not in {d.upper() for d in args.drug}:
            continue
        context = build_prompt_context(prompt, config=config, control=control)
        controller = AttachmentController(
            context.attachment, context.locked_slots, control
        )
        verdict = structural_verdict(context, controller)

        rng = np.random.default_rng(20260921)
        receipt = SamplingReceipt()
        emitted = []
        started = time.time()
        for _ in range(args.samples):
            emitted.append(
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
            )
        rows.append(
            {
                "drug": prompt.drug_name,
                "start_smiles": context.start_smiles,
                "structural": verdict,
                "attempts": args.samples,
                "committed_endpoints": len(receipt.committed_endpoints),
                "task_success": sum(1 for x in emitted if x),
                "separation_failures": receipt.separation_failures,
                "constraint_failures": receipt.constraint_failures,
                "staging_rejections": receipt.staging_rejections,
                "interface_rejections": receipt.interface_rejections,
                "lock_rejections": receipt.lock_rejections,
                "budget_exhausted": receipt.budget_exhausted,
                "mean_events": float(np.mean(receipt.events)) if receipt.events else 0.0,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"{prompt.drug_name:14s} open_sites={verdict['interfaces_with_free_valence']} "
            f"committed={rows[-1]['committed_endpoints']:3d} "
            f"task_success={rows[-1]['task_success']:3d}/{args.samples} "
            f"sep_fail={rows[-1]['separation_failures']:3d} "
            f"events={rows[-1]['mean_events']:.2f}",
            flush=True,
        )

    both_saturated = [r["drug"] for r in rows
                      if not r["structural"]["first_event_can_increase_coverage"]]
    reached = [r["drug"] for r in rows if r["task_success"] > 0]
    payload = {
        "schema": "compose_fragment_linker_reachability_v1",
        "question": (
            "does releasing the constructed join and steering growth to the "
            "declared interfaces make a genuine (non-zero-atom) linker reachable?"
        ),
        "verdict": "REACHABLE" if reached else "NOT REACHABLE UNDER THIS START STATE",
        "drugs_with_every_declared_site_saturated_by_the_join": both_saturated,
        "drugs_reaching_a_separated_linker": reached,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nverdict: {payload['verdict']}\nwrote {args.output}")


if __name__ == "__main__":
    main()
