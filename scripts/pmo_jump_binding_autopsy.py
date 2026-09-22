"""Reproduce and decompose every jump-lane binding failure of the arm-B celecoxib run.

ZERO oracle calls.  Reads a completed run artifact and the jump checkpoint it pins,
re-runs the PRODUCTION binder on the same (parent, plan) pairs, and records the refusal
the search itself computed.  Answer-known material is not consulted: the target structure
never enters, and the parents are the run's own archived molecules.

Usage:
    python scripts/pmo_jump_binding_autopsy.py --result <result.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import rdkit

from compose_v4.control import pmo_realization as PR
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_jump_binding_autopsy import (
    PROJECTABLE_FIELDS,
    SCHEMA,
    SPEC_V1_FIELDS,
    classify,
    diagnose_pair,
    field_projection_lift,
    role_based_specification,
    step0_feasible_under,
)
from compose_v4.rewrite.trace_shard import decode_state

JUMP = "joint_dependency_region_jump"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def load_pairs(result_path: Path, repo: Path):
    snapshot = json.loads(result_path.read_text())["campaign"]["snapshot"]
    envelope = json.loads((repo / CHECKPOINTS).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    checkpoint_id = identity(payload)
    expected = snapshot["pmo_population"]["jump_checkpoint_id"]
    if checkpoint_id != expected:
        raise SystemExit(
            f"jump checkpoint identity {checkpoint_id} is not the run's {expected}"
        )
    plans = {plan["plan_id"]: plan for plan in payload["plan_latents"]}
    entries = snapshot["entries"]
    attempts = [
        attempt
        for round_row in snapshot["history"]
        for attempt in round_row["batch"]["attempts"]
        if attempt.get("planner_channel") == JUMP
    ]
    return snapshot, plans, entries, attempts, checkpoint_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--control-sample", type=int, default=40)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot, plans, entries, attempts, checkpoint_id = load_pairs(Path(args.result), repo)

    role_spec = role_based_specification()
    began = time.perf_counter()
    rows = []
    control_disagreements = []
    step0_control_disagreements = 0

    for index, attempt in enumerate(attempts):
        entry = entries[attempt["entry_id"]]
        source = decode_state(entry["trace"]["states"][-1])
        plan = plans[attempt["plan_id"]]

        diagnosis = diagnose_pair(source, plan)

        # CONTROL 1 -- the recorder must not change the production decision.  Re-run a
        # sample WITHOUT the wrapper and require the same outcome.
        if index < args.control_sample:
            clean = PR.realize_plan_binding(
                source,
                plan,
                specification=PR.PRODUCTION_SPECIFICATION,
                node_budget=PR.PRODUCTION_NODE_BUDGET,
                seconds_cap=PR.PRODUCTION_SECONDS_CAP,
                max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
            )
            if clean["outcome"] != diagnosis["outcome"] or len(clean["bindings"]) != diagnosis[
                "bindings"
            ]:
                control_disagreements.append(
                    {
                        "index": index,
                        "wrapped": diagnosis["outcome"],
                        "clean": clean["outcome"],
                    }
                )

        # CONTROL 2 -- the production identity predicate and the all-fields descriptor
        # predicate must agree at step 0.
        exact0 = step0_feasible_under(source, plan, PR.SPEC_V1_EXACT)
        fields0 = step0_feasible_under(source, plan, SPEC_V1_FIELDS)
        if exact0 != fields0:
            step0_control_disagreements += 1

        projection = field_projection_lift(source, plan)
        role0 = step0_feasible_under(source, plan, role_spec)

        rows.append(
            {
                "index": index,
                "plan_id": attempt["plan_id"],
                "entry_id": attempt["entry_id"],
                "mode": attempt.get("mode"),
                "parent_measured_score": attempt.get("parent_measured_score"),
                "recorded_reason": attempt.get("reason"),
                "parent_endpoint": entry.get("endpoint"),
                "parent_heavy_atoms": int(source.n_real_atoms),
                "parent_slots": int(source.n_atoms),
                "plan_primitive_count": int(plan["primitive_count"]),
                "plan_component_count": int(plan.get("component_count", 1)),
                "plan_role0_rule": str(plan["roles"][0]["executor_rule"]),
                "mechanism": classify(diagnosis),
                "step0_feasible_v1": projection["baseline_feasible"],
                "step0_feasible_role_based": role0,
                "step0_single_drop": projection["single_drop_feasible"],
                **{k: v for k, v in diagnosis.items() if k != "refusal_counts"},
                "refusal_counts": diagnosis["refusal_counts"],
            }
        )

    elapsed = time.perf_counter() - began

    # ---- reductions ----
    recorded = Counter(
        row["recorded_reason"].rsplit("(", 1)[-1].rstrip(")") for row in rows
    )
    reproduced = Counter(row["outcome"] for row in rows)
    mechanisms = Counter(row["mechanism"] for row in rows)
    root_reasons = Counter(str(row["root_refusal"]) for row in rows)
    first_reasons = Counter(str(row["first_refusal_reason"]) for row in rows)
    first_steps = Counter(row["first_refusal_step"] for row in rows)

    n = len(rows)
    base_feasible = sum(1 for row in rows if row["step0_feasible_v1"])
    role_feasible = sum(1 for row in rows if row["step0_feasible_role_based"])
    infeasible = [row for row in rows if not row["step0_feasible_v1"]]

    per_field = {}
    for field in PROJECTABLE_FIELDS:
        gained = [row for row in infeasible if row["step0_single_drop"][field]]
        per_field[field] = {
            "step0_feasible_when_dropped": sum(
                1 for row in rows if row["step0_single_drop"][field]
            ),
            "pairs_unblocked_alone": len(gained),
            "fraction_of_step0_failures_unblocked_alone": (
                len(gained) / len(infeasible) if infeasible else None
            ),
            "lost_mass_share_of_all_pairs": len(gained) / n,
        }
    # Sole-cause attribution: failures that EXACTLY ONE field unblocks.
    sole = Counter()
    unblocked_by_any = 0
    for row in infeasible:
        hits = [f for f in PROJECTABLE_FIELDS if row["step0_single_drop"][f]]
        if hits:
            unblocked_by_any += 1
        if len(hits) == 1:
            sole[hits[0]] += 1

    summary = {
        "schema_version": SCHEMA,
        "answer_known_offline_diagnostic": True,
        "new_oracle_calls": 0,
        "rdkit_version": rdkit.__version__,
        "jump_checkpoint_identity": checkpoint_id,
        "run_id": snapshot.get("snapshot_id"),
        "pairs": n,
        "distinct_plans": len({row["plan_id"] for row in rows}),
        "distinct_parents": len({row["entry_id"] for row in rows}),
        "production_budget": {
            "specification": PR.PRODUCTION_SPECIFICATION.name,
            "node_budget": PR.PRODUCTION_NODE_BUDGET,
            "seconds_cap": PR.PRODUCTION_SECONDS_CAP,
            "max_realizations": PR.PRODUCTION_MAX_REALIZATIONS,
        },
        "controls": {
            "wrapper_changed_a_decision": len(control_disagreements),
            "wrapper_control_sample": min(args.control_sample, n),
            "step0_exact_vs_fields_disagreements": step0_control_disagreements,
            "disagreement_rows": control_disagreements,
        },
        "recorded_outcome_in_artifact": dict(recorded),
        "reproduced_outcome": dict(reproduced),
        "outcome_reproduction_exact": dict(recorded) == dict(reproduced),
        "mechanism_distribution": dict(mechanisms.most_common()),
        "root_refusal_distribution": dict(root_reasons.most_common()),
        "first_refusal_distribution": dict(first_reasons.most_common()),
        "first_refusal_step_distribution": {
            str(k): v for k, v in sorted(first_steps.items(), key=lambda kv: (kv[0] is None, kv[0]))
        },
        "nodes_expanded": {
            "zero": sum(1 for row in rows if row["nodes_expanded"] == 0),
            "median": statistics.median(row["nodes_expanded"] for row in rows),
            "max": max(row["nodes_expanded"] for row in rows),
        },
        "max_depth_reached": {
            "median": statistics.median(row["max_depth_reached"] for row in rows),
            "max": max(row["max_depth_reached"] for row in rows),
            "reached_full_depth": sum(
                1
                for row in rows
                if row["max_depth_reached"] >= row["plan_primitive_count"]
            ),
        },
        "step0_feasibility": {
            "v1_production": base_feasible,
            "v1_production_fraction": base_feasible / n,
            "role_based": role_feasible,
            "role_based_fraction": role_feasible / n,
            "note": "NECESSARY, NOT SUFFICIENT -- a step-0 number is not a binding rate",
        },
        "field_projection": per_field,
        "step0_failures": len(infeasible),
        "step0_failures_unblocked_by_some_single_field": unblocked_by_any,
        "step0_failures_unblocked_by_exactly_one_field": dict(sole.most_common()),
        "seconds": elapsed,
    }

    (out / "autopsy_summary_v1.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "autopsy_rows_v1.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA + "_rows",
                "answer_known_offline_diagnostic": True,
                "rows": rows,
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
