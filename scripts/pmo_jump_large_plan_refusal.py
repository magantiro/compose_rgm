"""WHY do the LARGE plans refuse even under a role-based descriptor?

Role-based rebinding lifts step-0 feasibility to ~100% and still binds NOTHING on plans
above 20 primitives.  A step-0 fix cannot explain a refusal that happens at depth, so this
records the refusal the search computes under the ROLE-based specification, stratified by
plan scale.  Zero oracle calls.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import rdkit

from compose_v4.control import pmo_realization as PR
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_jump_binding_autopsy import (
    classify,
    diagnose_pair,
    role_based_specification,
)
from compose_v4.rewrite.trace_shard import decode_state

JUMP = "joint_dependency_region_jump"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def band(count: int) -> str:
    if count < 12:
        return "below_12"
    if count <= 18:
        return "12_18_target"
    if count <= 20:
        return "19_20"
    return "above_20"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--per-band", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    snapshot = json.loads(Path(args.result).read_text())["campaign"]["snapshot"]
    payload = json.loads((repo / CHECKPOINTS).read_text())["payload"]["checkpoints"][
        "shared_all_routes"
    ]
    if identity(payload) != snapshot["pmo_population"]["jump_checkpoint_id"]:
        raise SystemExit("jump checkpoint is not the run's checkpoint")
    plans = {plan["plan_id"]: plan for plan in payload["plan_latents"]}
    entries = snapshot["entries"]
    attempts = [
        attempt
        for round_row in snapshot["history"]
        for attempt in round_row["batch"]["attempts"]
        if attempt.get("planner_channel") == JUMP
    ]
    # Stratify, and SHUFFLE within each stratum: attempts are emitted in round and plan
    # order, so taking the first N of a band samples one corner of the corpus.
    by_band: dict[str, list] = {}
    for attempt in attempts:
        by_band.setdefault(band(plans[attempt["plan_id"]]["primitive_count"]), []).append(attempt)
    rng = random.Random(args.seed)
    sample = []
    realized = {}
    for name, rows in by_band.items():
        picked = rows[:] if len(rows) <= args.per_band else rng.sample(rows, args.per_band)
        realized[name] = {"available": len(rows), "sampled": len(picked)}
        sample.extend((name, attempt) for attempt in picked)

    spec = role_based_specification()
    out_rows = []
    for name, attempt in sample:
        source = decode_state(entries[attempt["entry_id"]]["trace"]["states"][-1])
        plan = plans[attempt["plan_id"]]
        observed = diagnose_pair(source, plan, specification=spec)
        out_rows.append(
            {
                "band": name,
                "plan_id": attempt["plan_id"],
                "entry_id": attempt["entry_id"],
                "plan_primitive_count": int(plan["primitive_count"]),
                "mechanism": classify(observed),
                "outcome": observed["outcome"],
                "bindings": observed["bindings"],
                "root_refusal": observed["root_refusal"],
                "first_refusal_step": observed["first_refusal_step"],
                "first_refusal_reason": observed["first_refusal_reason"],
                "max_depth_reached": observed["max_depth_reached"],
                "nodes_expanded": observed["nodes_expanded"],
                "seconds": observed["seconds"],
            }
        )
        print(f"{len(out_rows)}/{len(sample)} {name}", flush=True)

    summary = {
        "schema_version": "pmo_jump_large_plan_refusal_v1",
        "answer_known_offline_diagnostic": True,
        "new_oracle_calls": 0,
        "rdkit_version": rdkit.__version__,
        "specification": {"name": spec.name, "compared_fields": list(spec.compared_fields)},
        "budget": {
            "node_budget": PR.PRODUCTION_NODE_BUDGET,
            "seconds_cap": PR.PRODUCTION_SECONDS_CAP,
        },
        "strata": realized,
        "by_band": {
            name: {
                "n": len([r for r in out_rows if r["band"] == name]),
                "bound": len([r for r in out_rows if r["band"] == name and r["bindings"]]),
                "mechanism": dict(
                    Counter(r["mechanism"] for r in out_rows if r["band"] == name).most_common()
                ),
                "first_refusal_reason": dict(
                    Counter(
                        str(r["first_refusal_reason"])
                        for r in out_rows
                        if r["band"] == name
                    ).most_common()
                ),
                "max_depth_reached_median": sorted(
                    r["max_depth_reached"] for r in out_rows if r["band"] == name
                )[max(0, len([r for r in out_rows if r["band"] == name]) // 2)],
                "reached_full_depth": len(
                    [
                        r
                        for r in out_rows
                        if r["band"] == name
                        and r["max_depth_reached"] >= r["plan_primitive_count"]
                    ]
                ),
            }
            for name in sorted(by_band)
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "large_plan_refusal_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (out / "large_plan_refusal_rows_v1.json").write_text(
        json.dumps({"rows": out_rows}, indent=1, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
