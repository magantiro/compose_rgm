#!/usr/bin/env python3
"""Does the fragment path execute transitions its proposal law never ranked?

A sibling agent found on the T4 path that ``expand`` gates a molecule the
proposal law never ranked (recovered fraction 0.0000), while the PMO path scores
the proposal directly (1.0000).  The fragment path has a specific reason to be
exposed: the attachment controller's ``redirect`` REPLACES the action after the
model has sampled it, keeping the prior's payload but moving the anchor.  The
executed transition is therefore not the one the model drew, and it may be one
the model would never draw.

That matters twice over.  It decides whether the two-interface path program can
be scored at all, and it is a live question about the attachment rows already
measured.

Method -- instrument by WRAPPING, never by transcribing.  The probe drives the
production ``sample_completion`` and wraps ``AttachmentController.redirect`` so
the ORIGINAL method still decides; it only records what went in and what came
out.  For every ACCEPTED event whose action was redirected, it then asks the
model for ``support_draws`` fresh marks at that same state and checks whether
the redirected transition appears among them.  A transcribed sampler could not
fail usefully here, and the production redirect is the only thing that decides.

``recovered_fraction`` is the share of redirected accepted events whose executed
transition the model does in fact propose.  1.0 means the controller only ever
re-ranks inside the proposal law's own support; 0.0 means the fragment path has
the T4 exposure and a program built on redirection would be scoring something
the generator never proposed.

The comparison is on the SUCCESSOR, not on the action object: two different
actions that produce the same committed molecule are the same transition as far
as anything downstream can tell, and the benchmark scores molecules.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")


def _successor_key(system, state, rule_name, action) -> str | None:
    """The committed molecule a transition produces, or None if it cannot run."""
    try:
        successor = system.apply(state, rule_name, action)
    except Exception:  # noqa: BLE001
        return None
    return molecular_graph_to_smiles(successor) or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--support-draws", type=int, default=256)
    parser.add_argument("--drug", action="append", default=None)
    parser.add_argument(
        "--task",
        default="scaffold_decoration",
        help="decoration declares the most interfaces, so it redirects most",
    )
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True)
    task = FragmentTask(args.task)

    prompts = [p for p in load_genmol_prompts(MANIFEST) if p.task is task]
    if args.drug:
        wanted = {d.upper() for d in args.drug}
        prompts = [p for p in prompts if p.drug_name.upper() in wanted]

    rows = []
    totals = Counter()
    original_redirect = AttachmentController.redirect

    for prompt in prompts:
        context = build_prompt_context(prompt, config=config, control=control)
        # Every redirect the production controller performs, in order.
        observed: list[dict] = []

        def _wrapped(self, rule_name, action, state, _observed=observed):
            # The ORIGINAL method decides; this only records.
            result = original_redirect(self, rule_name, action, state)
            if result is not action:
                _observed.append(
                    {
                        "rule_name": rule_name,
                        "state": state,
                        "redirected_action": result,
                        "original_action": action,
                    }
                )
            return result

        AttachmentController.redirect = _wrapped
        try:
            rng = np.random.default_rng(20260921)
            receipt = SamplingReceipt()
            for _ in range(args.samples):
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
        finally:
            AttachmentController.redirect = original_redirect

        # For each redirected transition, is its SUCCESSOR one the model proposes?
        recovered = 0
        checked = 0
        for record in observed:
            state = record["state"]
            target = _successor_key(
                system, state, record["rule_name"], record["redirected_action"]
            )
            if target is None:
                continue  # not executable; it cannot have been an accepted event
            checked += 1
            support_rng = np.random.default_rng(777)
            found = False
            for _ in range(args.support_draws):
                try:
                    mark = model.sample_rewrite_mark(state, 0.0, support_rng)
                except Exception:  # noqa: BLE001, S112 -- a refused draw is not a match
                    continue
                if _successor_key(system, state, mark.rule_name, mark.action) == target:
                    found = True
                    break
            recovered += int(found)

        totals["redirects_observed"] += len(observed)
        totals["executable_redirects"] += checked
        totals["recovered"] += recovered
        rows.append(
            {
                "drug": prompt.drug_name,
                "declared_interfaces": len(context.attachment.interfaces),
                "attempts": args.samples,
                "redirects_observed": len(observed),
                "executable_redirects": checked,
                "recovered_by_model_proposal": recovered,
                "recovered_fraction": (recovered / checked) if checked else None,
            }
        )
        print(
            f"{prompt.drug_name:14s} redirects={len(observed):4d} "
            f"executable={checked:4d} recovered={recovered:4d} "
            f"frac={(recovered / checked) if checked else float('nan'):.4f}",
            flush=True,
        )

    overall = (
        totals["recovered"] / totals["executable_redirects"]
        if totals["executable_redirects"]
        else None
    )
    payload = {
        "schema": "compose_fragment_proposal_scoring_probe_v1",
        "question": (
            "does the fragment path execute transitions its proposal law never "
            "ranked, the way the T4 path does?"
        ),
        "task": args.task,
        "support_draws_per_redirect": args.support_draws,
        "comparison": "successor molecule, not action identity",
        "totals": dict(totals),
        "recovered_fraction_overall": overall,
        "verdict": (
            "UNSCORED_PROPOSALS" if overall is not None and overall < 0.5
            else "PROPOSAL_IS_SCORED" if overall is not None
            else "NO_REDIRECTS_OBSERVED"
        ),
        "rows": rows,
        "note": (
            "recovered_fraction is a LOWER BOUND: a redirected transition the "
            "model proposes with low probability may not appear in a finite "
            "sample, so a low value needs the draw count stated beside it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\noverall recovered_fraction: {overall}\nwrote {args.output}")


if __name__ == "__main__":
    main()
