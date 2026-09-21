#!/usr/bin/env python3
"""Does jump role-0 feasibility depend on the PARENT rather than on the plan bank?

The 3x250 autopsy measured 33.1% step-0 feasibility over (plan, parent) pairs drawn
from a LIVE campaign.  The teacher-root audit measures it over the same task-blind
bank on the teacher ROOTS.  If the two disagree, "role 0 demands descriptors no atom
carries" is a statement about which parents the campaign reaches, not a fixed
property of the bank -- a different column to fix.

This driver settles that inside ONE process, on ONE bank: it samples parent states
the proposer actually worked from in a free-scorer campaign and re-runs the identical
role-0 probe used for the roots.

No oracle is constructed and no call is charged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pmo_teacher_capability_jump_lane_v1 import load_plans, probe

from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.rewrite.trace_shard import decode_state


def campaign_parents(campaign: Path) -> list[dict]:
    """Distinct parent states the proposer was driven from, in round order."""
    parents, seen = [], set()
    for pending in sorted(campaign.glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        round_name = pending.parent.name
        for pool in ("proposal_pool", "eligible_pool"):
            for candidate in batch.get(pool, {}).get("candidates", []):
                state = candidate.get("source_state")
                if state is None:
                    continue
                key = json.dumps(state, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                parents.append({"round": round_name, "state": state})
    return parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=8)
    options = parser.parse_args()

    plans = load_plans()
    parents = campaign_parents(Path(options.campaign))
    # Take the LATEST parents: those are the ones the campaign drifted to.
    chosen = parents[-options.sample :] if options.sample else parents

    rows = []
    for index, parent in enumerate(chosen):
        graph = decode_state(parent["state"])
        feasible = 0
        successors = 0
        for plan in plans:
            result = probe(graph, plan, roles=1, drop=())
            feasible += result["bindings"] > 0
            successors += result["successors_enumerated"]
        rows.append(
            {
                "round": parent["round"],
                "smiles": molecular_graph_to_smiles(graph),
                "heavy_atoms": int(is_element(graph.atom_types).sum()),
                "plans": len(plans),
                "role0_feasible": feasible,
                "role0_feasible_rate": feasible / len(plans),
                "successors_enumerated": successors,
            }
        )
        print(
            f"parent {index} ({rows[-1]['round']}, {rows[-1]['heavy_atoms']} heavy):"
            f" role0 {feasible}/{len(plans)}",
            flush=True,
        )

    payload = {
        "schema_version": "pmo_teacher_capability_parent_drift_v1",
        "charged_oracle_calls": 0,
        "campaign": options.campaign,
        "distinct_parents_in_campaign": len(parents),
        "sampled": len(rows),
        "plan_bank": len(plans),
        "summary": {
            "role0_feasible_rate_min": min(row["role0_feasible_rate"] for row in rows),
            "role0_feasible_rate_median": sorted(
                row["role0_feasible_rate"] for row in rows
            )[len(rows) // 2],
            "role0_feasible_rate_max": max(row["role0_feasible_rate"] for row in rows),
            "successors_enumerated": sum(row["successors_enumerated"] for row in rows),
        },
        "parents": rows,
    }
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {options.out}")


if __name__ == "__main__":
    main()
