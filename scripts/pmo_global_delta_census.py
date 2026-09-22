#!/usr/bin/env python
"""PART 1 -- the productive-basin global delta atlas.

Assembles every (blind / current-search state -> productive-region endpoint)
pair that can be built from committed PMO atlas evidence, and characterises each
one GLOBALLY: what a complete COMPOSE program would have to accomplish, treated
as one macro option.  No primitive sequence is decomposed anywhere.

PRODUCTIVITY IS MEASURED, NOT ASSUMED.  Both ends of a POOL_1 pair carry a
``top_ten_new_mean`` from a 64-call frozen-local-controller probe:
Test B measured it from the teacher spine, Test C measured it from states the
blind run actually reached.  A pair is admitted only when the target's measured
V_local genuinely exceeds the source's.

ZERO ORACLE CALLS.  Every score used here was already charged and committed.

DEVELOPMENT_INFORMED_DIAGNOSTIC -- see the emitted artifact's regime block.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compose_v4.experiments.global_delta_census import (  # noqa: E402
    TEMPLATES,
    assign_template,
    compute_global_delta,
)

ATLAS = "diagnostics/pmo_atlas_v1"
OUT = "diagnostics/pmo_global_delta_census_v1"

#: A target region counts as productive at or above this fraction of the task's
#: own anchor productivity.  Anchors differ 0.32 (median1) to 0.99 (qed), so an
#: absolute bar would silently drop whole tasks.
PRODUCTIVE_FRACTION_OF_ANCHOR = 0.85
#: and the lift must be real, not a rounding difference.
MIN_VLOCAL_LIFT = 0.15


def _load(name: str) -> dict:
    with open(os.path.join(ATLAS, name)) as handle:
        return json.load(handle)["payload"]


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def collect_test_b() -> list[dict]:
    payload = _load("test_b_local_lift.json")
    return [
        {
            "task": run["task"],
            "smiles": run["seed_smiles"],
            "v_local": run["top_ten_new_mean"],
            "label": run["checkpoint_label"],
            "heavy": run["seed_heavy_atoms"],
            "origin": "teacher_spine",
        }
        for run in payload["runs"]
    ]


def collect_test_c_entry() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(ATLAS, "test_c_entry_*.json"))):
        payload = json.load(open(path))["payload"]
        for probe in payload["probes"]:
            rows.append(
                {
                    "task": probe["task"],
                    "smiles": probe["seed_smiles"],
                    "v_local": probe["top_ten_new_mean"],
                    "label": "blind_" + str((probe.get("rung") or {}).get("rung_reached")),
                    "heavy": probe["seed_heavy_atoms"],
                    "origin": "blind_reached",
                    "charged_call": probe.get("seed_charged_call"),
                }
            )
    return rows


def collect_blind_best() -> list[dict]:
    """Best-scoring molecule each blind run actually reached (no V_local probe)."""
    best: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(ATLAS, "test_c_blind_*.json"))):
        payload = json.load(open(path))["payload"]
        for run in payload["runs"]:
            top = max(run["trajectory"], key=lambda row: row["score"])
            best[run["task"]] = {
                "task": run["task"],
                "smiles": top["endpoint"],
                "score": top["score"],
                "heavy": top["heavy_atoms"],
                "origin": "blind_best_scored",
            }
    return list(best.values())


def collect_test_a() -> list[dict]:
    payload = _load("test_a_transport.json")
    rows = []
    for attempt in payload["attempts"]:
        if attempt.get("status") != "witness_found":
            continue
        if attempt.get("source_pool") == "recorded_source":
            continue
        rows.append(
            {
                "task": attempt["task"],
                "source": attempt["source_smiles"],
                "target": attempt["destination_smiles"],
                "source_pool": attempt["source_pool"],
                "witness_steps": attempt.get("witness_steps"),
            }
        )
    return rows


def build_pools() -> dict:
    test_b = collect_test_b()
    entry = collect_test_c_entry()
    anchors = {row["task"]: row["v_local"] for row in test_b if row["label"] == "anchor"}

    productive = [
        row
        for row in test_b
        if anchors.get(row["task"]) and row["v_local"] >= PRODUCTIVE_FRACTION_OF_ANCHOR * anchors[row["task"]]
    ]

    pool1 = []
    for source in entry:
        for target in productive:
            if source["task"] != target["task"]:
                continue
            lift = target["v_local"] - source["v_local"]
            if lift < MIN_VLOCAL_LIFT:
                continue
            pool1.append(
                {
                    "pool": "POOL_1_MEASURED_VLOCAL",
                    "task": source["task"],
                    "source": source["smiles"],
                    "target": target["smiles"],
                    "source_v_local": source["v_local"],
                    "target_v_local": target["v_local"],
                    "v_local_lift": lift,
                    "source_label": source["label"],
                    "target_label": target["label"],
                    "target_fraction_of_anchor": target["v_local"] / anchors[target["task"]],
                }
            )

    pool2 = []
    for source in collect_blind_best():
        for target in productive:
            if source["task"] != target["task"] or target["label"] != "anchor":
                continue
            pool2.append(
                {
                    "pool": "POOL_2_BLIND_BEST_TO_ANCHOR",
                    "task": source["task"],
                    "source": source["smiles"],
                    "target": target["smiles"],
                    "source_score": source["score"],
                    "target_v_local": target["v_local"],
                }
            )

    pool3 = [
        {
            "pool": "POOL_3_CONSTRUCTIBLE_TRANSFER",
            "task": row["task"],
            "source": row["source"],
            "target": row["target"],
            "source_pool": row["source_pool"],
            "witness_steps": row["witness_steps"],
        }
        for row in collect_test_a()
    ]
    return {"POOL_1_MEASURED_VLOCAL": pool1, "POOL_2_BLIND_BEST_TO_ANCHOR": pool2,
            "POOL_3_CONSTRUCTIBLE_TRANSFER": pool3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(OUT, "productive_basin_atlas_v1.json"))
    args = parser.parse_args()

    pools = build_pools()
    records = []
    for pool_name, pairs in pools.items():
        print(f"[strata] {pool_name}: {len(pairs)} pairs", flush=True)
        by_task = collections.Counter(pair["task"] for pair in pairs)
        for task, count in sorted(by_task.items()):
            print(f"    {task:28s} {count}")
        for index, pair in enumerate(pairs):
            delta = compute_global_delta(pair["source"], pair["target"])
            row = dict(pair)
            row["delta"] = delta.as_dict()
            row["template"] = assign_template(delta)
            records.append(row)
            if (index + 1) % 25 == 0:
                print(f"    ... {index + 1}/{len(pairs)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "schema_version": "pmo_global_delta_census_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "information_regime_statement": (
            "Answer-known, panel-informed development evidence. Targets are published task "
            "answers or teacher-spine checkpoints. Nothing here may become a prior, library, "
            "initialization or runtime policy for a scored no-prescreen run. The product is a "
            "task-independent structural grammar; the task labels exist only to stratify."
        ),
        "new_oracle_calls": 0,
        "kernel": "rdkit 2023.09.6 (PMO production)",
        "productive_region_definition": (
            "Target measured V_local (top-ten mean of DISTINCT new molecules after a 64-call "
            f"frozen-local-controller probe) at or above {PRODUCTIVE_FRACTION_OF_ANCHOR} of the "
            f"task anchor, with a lift over the source of at least {MIN_VLOCAL_LIFT}."
        ),
        "inputs": {
            os.path.join(ATLAS, name): _sha256(os.path.join(ATLAS, name))
            for name in sorted(os.listdir(ATLAS))
            if name.endswith(".json")
        },
        "excluded_sources": {
            "diagnostics/ivg_winner_paths": (
                "competitor-derived (InVirtuoGen results), target-informed, non-commercially "
                "licensed, and T4 DOCKING cells rather than PMO tasks. Not used."
            )
        },
        "templates": list(TEMPLATES),
        "records": records,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}  ({len(records)} pairs)")


if __name__ == "__main__":
    main()
