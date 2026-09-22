"""In the 12-18 primitive band, is role-based rebinding COMPATIBILITY-limited or BUDGET-limited?

The stratified refusal census found that band's searches overwhelmingly hit a cap rather
than a constraint, which makes its measured binding rate a LOWER BOUND.  This raises the
budget and reports what the extra compute buys, and which cap was binding.  Zero oracle
calls.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import rdkit

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control import pmo_realization as PR
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_jump_binding_autopsy import (
    role_based_specification,
    structural_delta,
)
from compose_v4.rewrite.trace_shard import decode_state

JUMP = "joint_dependency_region_jump"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--sample", type=int, default=16)
    parser.add_argument("--node-budget", type=int, default=512)
    parser.add_argument("--seconds-cap", type=float, default=180.0)
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
        and 12 <= plans[attempt["plan_id"]]["primitive_count"] <= 18
    ]
    rng = random.Random(args.seed)
    sample = rng.sample(attempts, min(args.sample, len(attempts)))
    spec = role_based_specification()
    rows = []
    for index, attempt in enumerate(sample):
        source = decode_state(entries[attempt["entry_id"]]["trace"]["states"][-1])
        plan = plans[attempt["plan_id"]]
        parent_smiles = molecular_graph_to_smiles(source)
        arms = {}
        for name, node_budget, seconds in (
            ("production", PR.PRODUCTION_NODE_BUDGET, PR.PRODUCTION_SECONDS_CAP),
            ("raised", args.node_budget, args.seconds_cap),
        ):
            result = PR.realize(
                source,
                plan,
                spec,
                node_budget=node_budget,
                seconds_cap=seconds,
                collect_all=True,
                max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
            )
            shapes = []
            for row in result["realizations"]:
                endpoint = decode_state(row["endpoint_state"])
                delta = structural_delta(
                    parent_smiles, molecular_graph_to_smiles(endpoint), timeout=10
                ) or {}
                shapes.append(
                    {
                        "endpoint_smiles": molecular_graph_to_smiles(endpoint),
                        "largest_changed_region": delta.get("largest_changed_region"),
                        "n_changed_regions": delta.get("n_changed_regions"),
                        "retained_fraction_mcs": delta.get("retained_fraction_mcs"),
                        "d_heavy": delta.get("d_heavy"),
                        "d_rings": delta.get("d_rings"),
                        "realized_primitive_count": int(row["primitive_count"]),
                    }
                )
            arms[name] = {
                "outcome": result["outcome"],
                "budget_reason": result.get("budget_reason"),
                "bindings": len(result["realizations"]),
                "nodes_expanded": int(result["nodes_expanded"]),
                "max_depth_reached": int(result["max_depth_reached"]),
                "seconds": float(result["seconds"]),
                "shapes": shapes,
            }
        rows.append(
            {
                "plan_id": attempt["plan_id"],
                "entry_id": attempt["entry_id"],
                "parent_smiles": parent_smiles,
                "plan_primitive_count": int(plan["primitive_count"]),
                "arms": arms,
            }
        )
        print(f"{index+1}/{len(sample)}", flush=True)

    def reduce(name):
        sub = [row["arms"][name] for row in rows]
        shapes = [s for arm in sub for s in arm["shapes"]]
        regions = [s["largest_changed_region"] for s in shapes if s["largest_changed_region"]]
        retained = [
            s["retained_fraction_mcs"] for s in shapes if s["retained_fraction_mcs"] is not None
        ]
        return {
            "pairs": len(sub),
            "pairs_bound": sum(1 for arm in sub if arm["bindings"]),
            "pairs_bound_fraction": sum(1 for arm in sub if arm["bindings"]) / len(sub),
            "outcomes": dict(Counter(arm["outcome"] for arm in sub).most_common()),
            "budget_reason": dict(
                Counter(str(arm["budget_reason"]) for arm in sub).most_common()
            ),
            "realizations": len(shapes),
            "largest_changed_region_median": (
                sorted(regions)[len(regions) // 2] if regions else None
            ),
            "largest_changed_region_max": max(regions) if regions else None,
            "retained_fraction_mcs_median": (
                sorted(retained)[len(retained) // 2] if retained else None
            ),
            "seconds_total": sum(arm["seconds"] for arm in sub),
        }

    summary = {
        "schema_version": "pmo_jump_target_band_budget_v1",
        "answer_known_offline_diagnostic": True,
        "new_oracle_calls": 0,
        "rdkit_version": rdkit.__version__,
        "band": "12_to_18_primitives",
        "sampled": len(rows),
        "available": len(attempts),
        "raised_budget": {"node_budget": args.node_budget, "seconds_cap": args.seconds_cap},
        "arms": {name: reduce(name) for name in ("production", "raised")},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "target_band_budget_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (out / "target_band_budget_rows_v1.json").write_text(
        json.dumps({"rows": rows}, indent=1, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
