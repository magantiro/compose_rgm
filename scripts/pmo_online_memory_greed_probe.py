"""How concentrated is the memory's region law? Zero oracle calls.

A support floor guarantees every drawable region keeps POSITIVE probability,
which is a statement about support, not about exploration: a floor of 0.05
against a top weight of e^9 leaves the floored regions at ~1e-6 and the law is
greedy in everything but name. This reports the realized concentration so that
claim is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pmo_online_memory_gate import TASKS, load_task, warm_memory

from compose_v4.rewrite.trace_shard import decode_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/rmaganti/compose_pmo_data/rounds")
    ap.add_argument("--parents", type=int, default=40)
    ap.add_argument("--out", default="diagnostics/pmo_online_memory_greed_v1.json")
    args = ap.parse_args()
    out: dict = {"schema": "pmo_online_memory_greed_v1", "tasks": {}}
    for task in TASKS:
        rounds = load_task(Path(args.root), task)
        memory, _ = warm_memory(rounds)
        law = memory.region_law()
        entries = rounds[-1]["entries"]
        ratios, tops, effective, counts = [], [], [], []
        for entry_id in sorted(entries)[: args.parents]:
            graph = decode_state(entries[entry_id]["trace"]["states"][-1])
            regions = law.regions(graph)
            if len(regions) < 2:
                continue
            weights = law.weights(graph, regions)
            probability = weights / weights.sum()
            ratios.append(float(weights.max() / weights.min()))
            tops.append(float(probability.max()))
            # Effective support size: exp(Shannon entropy). A uniform law over
            # n regions gives exactly n; a law that always picks one gives 1.
            entropy = -float(np.sum(probability * np.log(probability + 1e-300)))
            effective.append(float(np.exp(entropy)))
            counts.append(len(regions))
        out["tasks"][task] = {
            "parents": len(ratios),
            "median_regions": float(np.median(counts)),
            "median_max_over_min_weight": float(np.median(ratios)),
            "median_top_region_probability": float(np.median(tops)),
            "median_effective_support_size": float(np.median(effective)),
            "median_effective_fraction_of_support": float(
                np.median(np.array(effective) / np.array(counts))
            ),
        }
        print(f"[{task}] {out['tasks'][task]}", flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
