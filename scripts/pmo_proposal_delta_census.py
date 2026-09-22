#!/usr/bin/env python
"""PART 2 -- what the CURRENT blind controller actually proposes, in the same
global representation as the productive-basin atlas, plus a task-independent NULL
and the proposal funnel that point 6 asks about.

FOUR populations, one instrument:

* ``PROPOSED_CHARGED``    eligible proposals that reached a charged oracle call.
* ``PROPOSED_DISCARDED``  eligible proposals -- a complete program executed and
  produced a scoreable molecule -- that the per-round allocator dropped before
  any oracle call.  This is the population point 6 is about: a realized
  structural option killed by selection, not by the executor.
* ``PROPOSED_REJECTED``   attempts whose program never executed.  Carries the
  executor's own reason, so "never proposed" and "proposed and refused" stay
  separate.
* ``NULL_RANDOM_PAIR``    ordered pairs of distinct task-independent init-bank
  molecules.  The control that decides whether "productive transitions are large
  multi-region replacements" is a statement about productivity or merely about
  any two different drug-like molecules.

TWO INDEPENDENT INSTRUMENTS.  Production recorded ``actual_changes`` on every
executed attempt; this script recomputes an MCS global delta on the charged
pairs, where the parent is recoverable from the committed trajectory.  The
agreement between them on the overlap is what licenses reading the production
record on the attempts whose parent is not recoverable.

ZERO ORACLE CALLS.  Seeds are BLAKE2b digests, never ``hash()``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compose_v4.experiments.global_delta_census import (
    assign_template,
    compute_global_delta,
    stable_seed,
)

RUNS = "/Users/rmaganti/compose_pmo_atlas_runs/test_c_blind"
ATLAS = "diagnostics/pmo_atlas_v1"
OUT = "diagnostics/pmo_global_delta_census_v1"


def parent_map() -> dict[tuple[str, str], str]:
    """(task, endpoint) -> parent endpoint, from the committed blind trajectories."""
    mapping: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(os.path.join(ATLAS, "test_c_blind_*.json"))):
        payload = json.load(open(path))["payload"]
        for run in payload["runs"]:
            for row in run["trajectory"]:
                if row.get("parent_endpoint"):
                    mapping[(run["task"], row["endpoint"])] = row["parent_endpoint"]
    return mapping


def production_shape(changes: dict | None) -> dict:
    """The production-recorded global shape of one executed proposal."""
    if not changes:
        return {}
    components = changes.get("source_induced_components") or []
    sizes = sorted((len(c) for c in components), reverse=True)
    return {
        "prod_touched_source_atoms": len(changes.get("changed_original_slots") or []),
        "prod_changed_site_count": changes.get("changed_site_count"),
        "prod_touched_region_sizes": sizes,
        "prod_largest_touched_region": sizes[0] if sizes else 0,
        "prod_deleted_original_atoms": changes.get("deleted_original_atoms"),
        "prod_surviving_new_atoms": changes.get("surviving_new_atoms"),
        "prod_structural_scale": (changes.get("deleted_original_atoms") or 0)
        + (changes.get("surviving_new_atoms") or 0),
        "prod_multi_region": (changes.get("changed_site_count") or 0) >= 2,
    }


def iter_attempts():
    for directory in sorted(os.listdir(RUNS)):
        task = directory.replace("__blind_250", "")
        pattern = os.path.join(RUNS, directory, "campaign", "round_*", "pending.json")
        for path in sorted(glob.glob(pattern)):
            batch = json.load(open(path))["batch"]
            charged_endpoints = {c["endpoint"] for c in batch["candidates"]}
            round_index = int(os.path.basename(os.path.dirname(path)).split("_")[1])
            for attempt in batch["attempts"]:
                status = attempt.get("status")
                endpoint = attempt.get("endpoint")
                if status == "eligible":
                    pool = (
                        "PROPOSED_CHARGED"
                        if endpoint in charged_endpoints
                        else "PROPOSED_DISCARDED"
                    )
                elif status == "execution_rejected":
                    pool = "PROPOSED_REJECTED"
                else:
                    pool = "PROPOSED_DUPLICATE"
                composition = (attempt.get("metadata") or {}).get(
                    "dynamic_generic_composition"
                ) or {}
                yield {
                    "pool": pool,
                    "task": task,
                    "round": round_index,
                    "channel": attempt.get("planner_channel") or attempt.get("channel"),
                    "status": status,
                    "reason": attempt.get("reason"),
                    "endpoint": endpoint,
                    "requested_modules": composition.get("requested_module_count"),
                    "completed_modules": composition.get("completed_module_count"),
                    "module_families": [m.get("family") for m in composition.get("modules") or []],
                    "primitive_edits": sum(
                        m.get("primitive_edits") or 0 for m in composition.get("modules") or []
                    )
                    or None,
                    "source_heavy_atoms": composition.get("source_heavy_atoms"),
                    "intermediate_task_evaluations": composition.get(
                        "intermediate_task_evaluations"
                    ),
                    "actual_changes": attempt.get("actual_changes"),
                }


def null_pairs(count: int) -> list[tuple[str, str]]:
    bank = json.load(open("docs/PMO_INIT_BANK.json"))["smiles"]
    rng = random.Random(stable_seed("pmo_global_delta_census", "null_random_pair", "v1"))
    seen: set[tuple[int, int]] = set()
    pairs = []
    while len(pairs) < count:
        a, b = rng.sample(range(len(bank)), 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((bank[a], bank[b]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-pairs", type=int, default=400)
    parser.add_argument("--out", default=os.path.join(OUT, "proposal_delta_census_v1.json"))
    args = parser.parse_args()

    parents = parent_map()
    attempts = list(iter_attempts())
    print(f"[strata] proposal attempts: {len(attempts)}", flush=True)
    for key in ("pool", "channel", "status"):
        print(f"  by {key}:", dict(collections.Counter(a[key] for a in attempts)))
    print("  by task:", dict(collections.Counter(a["task"] for a in attempts)))

    records = []
    resolved = 0
    for index, attempt in enumerate(attempts):
        row = {k: v for k, v in attempt.items() if k != "actual_changes"}
        row.update(production_shape(attempt["actual_changes"]))
        parent = parents.get((attempt["task"], attempt["endpoint"])) if attempt["endpoint"] else None
        row["parent_resolved"] = parent is not None
        if parent:
            resolved += 1
            delta = compute_global_delta(parent, attempt["endpoint"])
            row["source"] = parent
            row["delta"] = delta.as_dict()
            row["template"] = assign_template(delta)
        records.append(row)
        if (index + 1) % 1000 == 0:
            print(f"    ... {index + 1}/{len(attempts)}  (mcs on {resolved})", flush=True)
    print(f"[strata] parents recoverable from committed trajectory: {resolved}", flush=True)

    print("[strata] NULL_RANDOM_PAIR", flush=True)
    for index, (source, target) in enumerate(null_pairs(args.null_pairs)):
        delta = compute_global_delta(source, target)
        records.append(
            {
                "pool": "NULL_RANDOM_PAIR",
                "task": "TASK_INDEPENDENT",
                "channel": None,
                "status": "null",
                "parent_resolved": True,
                "source": source,
                "endpoint": target,
                "delta": delta.as_dict(),
                "template": assign_template(delta),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"    ... {index + 1}/{args.null_pairs}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "schema_version": "pmo_proposal_delta_census_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "new_oracle_calls": 0,
        "kernel": "rdkit 2023.09.6 (PMO production)",
        "source_of_proposals": RUNS,
        "records": records,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}  ({len(records)} rows)")


if __name__ == "__main__":
    main()
