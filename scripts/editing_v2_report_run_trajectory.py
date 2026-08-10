#!/usr/bin/env python
"""Turn a run checkpoint's trajectory into the four-view report.

One vote, three diagnostics. The selection column is the reference-law-weighted
NLL; family/cell are gates; support bands are the generalization regimes; the
zero-mass strata and the external 16-shard cohort are reported and never vote.

Reads the checkpoint rather than the logs, so the numbers are the ones the run
actually selected on rather than whatever was printed alongside them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trajectory = payload["trajectory"]
    if not trajectory:
        print("no evaluations recorded yet")
        return 0

    print(f"identity {payload['identity_sha256'][:12]}   "
          f"steps {payload['completed_steps']:,}   "
          f"selected step {payload['selected_step']:,}   "
          f"resumes {payload['resume_count']}")
    baseline = payload.get("collapse_baseline") or {}
    print(f"collapse baseline over {len(baseline)} capabilities\n")

    print(f"{'step':>7} {'ref-law':>9} {'native':>8} {'elig':>5} "
          f"{'cohort':>8}  collapsed")
    for item in trajectory:
        weighted = item.get("reference_law_weighted_mean_nll")
        cohort = item.get("external_cohort_mean_nll")
        print(f"{item['step']:7,} "
              f"{weighted if weighted is None else round(weighted, 4)!s:>9} "
              f"{item['panel_native_mean_nll']:8.4f} "
              f"{str(item.get('eligible', '?')):>5} "
              f"{'-' if cohort is None else f'{cohort:.4f}':>8}  "
              f"{','.join(item.get('collapsed_capabilities') or []) or '-'}")

    last = trajectory[-1]
    for title, key in (("per-family NLL", "by_family_mean_nll"),
                       ("per-cell NLL", "by_cell_mean_nll"),
                       ("support band NLL", "by_support_band_mean_nll"),
                       ("zero-mass strata (diagnostic)", "zero_mass_stratum_mean_nll")):
        values = last.get(key) or {}
        if not values:
            continue
        print(f"\n{title} at step {last['step']:,}")
        first = trajectory[0].get(key) or {}
        for name in sorted(values, key=lambda n: -values[n]):
            start = first.get(name)
            delta = "" if start is None else f"   {values[name] - start:+.3f} from first"
            print(f"  {name[:58]:58} {values[name]:8.4f}{delta}")

    if args.out:
        args.out.write_text(json.dumps({
            "completed_steps": payload["completed_steps"],
            "selected_step": payload["selected_step"],
            "selected_criterion": payload["selected_criterion"],
            "identity": payload["identity"],
            "collapse_baseline": baseline,
            "trajectory": trajectory,
        }, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
