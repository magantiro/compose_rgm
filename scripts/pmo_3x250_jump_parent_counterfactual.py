"""Was the joint-jump lane failing on its plans, or on the parents it was aimed at?

The binder probe measured that 72-82% of jump bindings die at role step 0 and that no
beam width rescues them, and separately that the median jump PARENT carried only 8 heavy
atoms.  Those two facts have different fixes, so they have to be separated: a 28-primitive
role sequence mined from ~26-heavy-atom drug-like routes may be unbindable on an
8-heavy-atom fragment for want of atoms, quite apart from how exact the role descriptors
are.

This driver replays ``bind_joint_plan`` -- the production function, unmodified -- over a
stratified set of parents drawn from the run's own archive: the highest-scoring entries,
the largest entries, and the size band the lane actually sampled.  Zero oracle calls: the
binder is pure graph chemistry and every score quoted was charged by the original run.

If elite/large parents bind at a materially higher rate, the lane's failure is a PARENT
SELECTION defect and is cheap to fix.  If they do not, the failure is in the exact-role
binding contract and the plan bank, which is a compiler change.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np

from compose_v4.control.pmo_joint_dependency_jump import bind_joint_plan
from compose_v4.rewrite.trace_shard import decode_state

CHECKPOINT = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def load_plans() -> list[dict]:
    envelope = json.loads(Path(CHECKPOINT).read_text())
    return envelope["payload"]["checkpoints"]["shared_all_routes"]["plan_latents"]


def load_entries(task_dir: Path) -> dict:
    entries = {}
    for folder in sorted(glob.glob(str(task_dir / "campaign" / "round_*"))):
        snapshot = json.loads(Path(folder, "complete.json").read_text())["snapshot"]
        entries.update(snapshot["entries"])
    return entries


def load_scores(task_dir: Path) -> dict[str, float]:
    scores = {}
    for path in sorted((task_dir / "oracle").glob("query_*/result.json")):
        record = json.loads(path.read_text())
        scores[record["endpoint"]] = float(record["score"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parents-per-stratum", type=int, default=8)
    parser.add_argument("--plans", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260920)
    arguments = parser.parse_args()

    plans = load_plans()
    rng = random.Random(arguments.seed)
    plan_sample = sorted(
        rng.sample(plans, min(arguments.plans, len(plans))),
        key=lambda plan: plan["plan_id"],
    )

    task_dir = Path(arguments.task_dir)
    entries = load_entries(task_dir)
    scores = load_scores(task_dir)

    rows = []
    for key, entry in entries.items():
        state = decode_state(entry["trace"]["states"][-1])
        rows.append({
            "entry_id": key,
            "endpoint": entry["endpoint"],
            "heavy_atoms": int(state.n_real_atoms),
            "score": scores.get(entry["endpoint"]),
            "state": state,
        })
    scored = [r for r in rows if r["score"] is not None]

    strata = {
        "top_scoring": sorted(scored, key=lambda r: -r["score"])[:arguments.parents_per_stratum],
        "largest": sorted(rows, key=lambda r: -r["heavy_atoms"])[:arguments.parents_per_stratum],
        "small_like_the_lane_sampled": sorted(
            [r for r in rows if r["heavy_atoms"] <= 10],
            key=lambda r: r["entry_id"])[:arguments.parents_per_stratum],
    }

    results = {}
    for name, parents in strata.items():
        cells = []
        for parent in parents:
            bound_at_4 = 0
            bound_at_8 = 0
            for plan in plan_sample:
                if bind_joint_plan(parent["state"], plan, beam_width=4):
                    bound_at_4 += 1
                    bound_at_8 += 1
                    continue
                if bind_joint_plan(parent["state"], plan, beam_width=8):
                    bound_at_8 += 1
            cells.append({
                "entry_id": parent["entry_id"][:12],
                "endpoint": parent["endpoint"],
                "heavy_atoms": parent["heavy_atoms"],
                "score": parent["score"],
                "plans_tried": len(plan_sample),
                "plans_bound_beam4": bound_at_4,
                "plans_bound_beam8": bound_at_8,
            })
        attempts = len(cells) * len(plan_sample)
        results[name] = {
            "parents": len(cells),
            "plans_per_parent": len(plan_sample),
            "attempts": attempts,
            "bound_beam4": sum(c["plans_bound_beam4"] for c in cells),
            "bound_beam8": sum(c["plans_bound_beam8"] for c in cells),
            "bind_rate_beam4": (
                sum(c["plans_bound_beam4"] for c in cells) / attempts if attempts else 0.0),
            "bind_rate_beam8": (
                sum(c["plans_bound_beam8"] for c in cells) / attempts if attempts else 0.0),
            "median_parent_heavy_atoms": float(np.median([c["heavy_atoms"] for c in cells])),
            "median_parent_score": float(np.median(
                [c["score"] for c in cells if c["score"] is not None])) if any(
                    c["score"] is not None for c in cells) else None,
            "cells": cells,
        }

    payload = {
        "schema_version": "pmo_3x250_jump_parent_counterfactual_v1",
        "oracle_calls_made_by_this_analysis": 0,
        "task_dir": str(task_dir),
        "plan_sample_size": len(plan_sample),
        "plan_primitive_median": float(np.median(
            [p["primitive_count"] for p in plan_sample])),
        "archive_heavy_atom_distribution": {
            "median": float(np.median([r["heavy_atoms"] for r in rows])),
            "p90": float(np.percentile([r["heavy_atoms"] for r in rows], 90)),
            "max": int(max(r["heavy_atoms"] for r in rows)),
            "share_at_or_below_10": float(np.mean(
                [r["heavy_atoms"] <= 10 for r in rows])),
        },
        "strata": results,
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    summary = {
        name: {k: v for k, v in block.items() if k != "cells"}
        for name, block in results.items()
    }
    print(json.dumps({"archive": payload["archive_heavy_atom_distribution"],
                      "strata": summary}, indent=2))


if __name__ == "__main__":
    main()
