"""Measured step-0 AND full-search feasibility of a ROLE-based rebinding, same 357 pairs.

ZERO oracle calls.  Arms differ ONLY in the acceptance specification; parent, plan,
budget, enumerator and executor are byte-identical across arms.

The role-based specification is CONTENT-IDENTICAL to ``pmo_realization.SPEC_R2``, which
already exists and was measured WORSE than the production predicate on 36 matched
TEACHER pairs (a plan bound back near its own source).  This probe asks a different
question on a different population: the PRODUCTION parents, where the production
predicate binds almost nothing.  The two results are not in conflict and neither
supersedes the other.

Program SHAPE is recorded beside the binding rate, because a rate that rises while the
realized scale and retained fraction do not has not repaired the located defect -- that
is the falsifier pinned in ``configs/pmo_population_controller_v1.json``.

Usage:
    python scripts/pmo_jump_role_rebinding_probe.py --result <result.json> --out <dir>
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
from compose_v4.experiments.pmo_jump_binding_autopsy import role_based_specification
from compose_v4.rewrite.trace_shard import decode_state

JUMP = "joint_dependency_region_jump"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def arm(source, plan, spec):
    began = time.perf_counter()
    result = PR.realize_plan_binding(
        source,
        plan,
        specification=spec,
        node_budget=PR.PRODUCTION_NODE_BUDGET,
        seconds_cap=PR.PRODUCTION_SECONDS_CAP,
        max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
    )
    rows = []
    for row in result["bindings"]:
        endpoint = decode_state(row["endpoint_state"])
        rows.append(
            {
                "endpoint_key": row["endpoint_key"],
                "primitive_count": int(row["primitive_count"]),
                "component_count": int(row["component_count"]),
                "created_dependency_edges": int(row["created_dependency_edges"]),
                "exact_replay": bool(row["exact_replay"]),
                "retained_fraction": float(PR.retained_fraction(source, endpoint)),
                "endpoint_heavy_atoms": int(endpoint.n_real_atoms),
                "heavy_atom_delta": int(endpoint.n_real_atoms) - int(source.n_real_atoms),
            }
        )
    return {
        "outcome": result["outcome"],
        "bindings": len(rows),
        "nodes_expanded": int(result["nodes_expanded"]),
        "successors_enumerated": int(result["successors_enumerated"]),
        "max_depth_reached": int(result["max_depth_reached"]),
        "seconds": time.perf_counter() - began,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(Path(args.result).read_text())["campaign"]["snapshot"]
    envelope = json.loads((repo / CHECKPOINTS).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    checkpoint_id = identity(payload)
    if checkpoint_id != snapshot["pmo_population"]["jump_checkpoint_id"]:
        raise SystemExit("jump checkpoint is not the run's checkpoint")
    plans = {plan["plan_id"]: plan for plan in payload["plan_latents"]}
    entries = snapshot["entries"]
    attempts = [
        attempt
        for round_row in snapshot["history"]
        for attempt in round_row["batch"]["attempts"]
        if attempt.get("planner_channel") == JUMP
    ]
    if args.limit:
        attempts = attempts[: args.limit]

    role_spec = role_based_specification()
    arms = {"v1_exact_production": PR.SPEC_V1_EXACT, "role_based": role_spec}
    rows = []
    began = time.perf_counter()
    for index, attempt in enumerate(attempts):
        entry = entries[attempt["entry_id"]]
        source = decode_state(entry["trace"]["states"][-1])
        plan = plans[attempt["plan_id"]]
        row = {
            "index": index,
            "plan_id": attempt["plan_id"],
            "entry_id": attempt["entry_id"],
            "parent_endpoint": entry.get("endpoint"),
            "parent_measured_score": attempt.get("parent_measured_score"),
            "parent_heavy_atoms": int(source.n_real_atoms),
            "plan_primitive_count": int(plan["primitive_count"]),
            "arms": {name: arm(source, plan, spec) for name, spec in arms.items()},
        }
        rows.append(row)
        if index % 25 == 0:
            print(f"{index}/{len(attempts)} {time.perf_counter()-began:.0f}s", flush=True)

    def reduce_arm(name):
        sub = [row["arms"][name] for row in rows]
        bound = [row for row in sub if row["bindings"]]
        prog = [r for row in bound for r in row["rows"]]
        return {
            "pairs": len(sub),
            "pairs_bound": len(bound),
            "pairs_bound_fraction": len(bound) / len(sub),
            "outcomes": dict(Counter(row["outcome"] for row in sub).most_common()),
            "distinct_endpoints": len({r["endpoint_key"] for r in prog}),
            "realizations": len(prog),
            "seconds_total": sum(row["seconds"] for row in sub),
            "shape": None
            if not prog
            else {
                "primitive_count_median": statistics.median(r["primitive_count"] for r in prog),
                "primitive_count_min": min(r["primitive_count"] for r in prog),
                "primitive_count_max": max(r["primitive_count"] for r in prog),
                "retained_fraction_median": statistics.median(
                    r["retained_fraction"] for r in prog
                ),
                "retained_fraction_min": min(r["retained_fraction"] for r in prog),
                "retained_fraction_max": max(r["retained_fraction"] for r in prog),
                "purely_additive_fraction": sum(
                    1 for r in prog if r["retained_fraction"] >= 0.9999
                )
                / len(prog),
                "heavy_atom_delta_median": statistics.median(
                    r["heavy_atom_delta"] for r in prog
                ),
                "heavy_atom_delta_min": min(r["heavy_atom_delta"] for r in prog),
                "heavy_atom_delta_max": max(r["heavy_atom_delta"] for r in prog),
                "exact_replay_all": all(r["exact_replay"] for r in prog),
            },
        }

    summary = {
        "schema_version": "pmo_jump_role_rebinding_probe_v1",
        "answer_known_offline_diagnostic": True,
        "new_oracle_calls": 0,
        "rdkit_version": rdkit.__version__,
        "jump_checkpoint_identity": checkpoint_id,
        "pairs": len(rows),
        "role_based_specification": {
            "name": role_spec.name,
            "relaxed": sorted(role_spec.relaxed),
            "compared_fields": list(role_spec.compared_fields),
            "content_identical_to_SPEC_R2": role_spec.relaxed == PR.SPEC_R2.relaxed,
        },
        "budget": {
            "node_budget": PR.PRODUCTION_NODE_BUDGET,
            "seconds_cap": PR.PRODUCTION_SECONDS_CAP,
            "max_realizations": PR.PRODUCTION_MAX_REALIZATIONS,
        },
        "arms": {name: reduce_arm(name) for name in arms},
        "seconds": time.perf_counter() - began,
    }
    (out / "role_rebinding_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    # Persist the MOLECULES, not just the counters.
    (out / "role_rebinding_rows_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "pmo_jump_role_rebinding_probe_v1_rows",
                "answer_known_offline_diagnostic": True,
                "rows": [
                    {
                        **{k: v for k, v in row.items() if k != "arms"},
                        "arms": {
                            name: {k: v for k, v in value.items()}
                            for name, value in row["arms"].items()
                        },
                    }
                    for row in rows
                ],
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
