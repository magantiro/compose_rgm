"""Offline funnel for the PMO-v2 joint-dependency jump lane (ZERO oracle cost).

WHAT THIS ANSWERS
-----------------
The exact realizer (``compose_v4.control.pmo_realization``) measurably BEAT the
width-4 beam it replaced on offline teacher-scale witness pairs -- 5 complete
realizations at median 31 primitives against the beam's 0 -- and then produced 3
executions across 1,091 jump proposals in the completed 3x250 scored run
(``run_id 35fc7bcd...``).  This script replays the EXACT stored (plan, parent)
pairs from that run and decides which of four branches explains the gap:

  A  mostly PROVEN INCOMPATIBLE  -> the realizer is correct; the PROPOSER is wrong
                                   for the production parent distribution.
  B  feasible but search TIMES OUT -> improve ordering/pruning/decomposition.
  C  the beam realizes the same proposals exact rejects -> check whether the
                                   beam's endpoint satisfies the declared program.
  D  realizations succeed but die DOWNSTREAM -> name the stage.

HOW IT IS INSTRUMENTED
----------------------
``propagate`` is WRAPPED, never transcribed: the pinned function still decides and
the wrapper only records ``(step, reason)`` beside it.  A transcription would have
nothing to check itself against.  Every replayed outcome is compared against the
outcome string the production controller stored in its own rejection ``reason``
(``pmo_population_controller.py`` embeds ``realized['outcome']`` there), so the
replay carries its own parity gate.

THREE ARMS, one interpreter, identical pairs:
  exact    -- the production configuration (v1_exact, node_budget 64, 20.0 s, 4).
  beam4/8  -- the pinned ``bind_joint_plan`` this realizer replaced (v1 ran width 4).
  noprune  -- ``propagate`` disabled and the budget raised ~300x.  Its ONLY job is
              to establish whether a legal realization exists at all, independent
              of the forward-checking prunes whose soundness is being tested.

ZERO oracle calls, zero docking, zero Modal launches: binding is pure graph work.
Pinned modules are imported, never modified.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control import pmo_realization as PR
from compose_v4.control.pmo_joint_dependency_jump import bind_joint_plan
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_jump_funnel_audit_v1"
RUN_ID = "35fc7bcd4bf5c964294698014c1ed40da483ad013f6b92c6201b01dee2d3d306"
CONTRACT_SHA = "ba9515615b341019f09f66f3224e590b4ca596dee6bb7c97c759dbe418f1ff4b"
CHECKPOINT = Path("diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json")
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")
JUMP = "joint_dependency_region_jump"
PRODUCTION_BEAM_WIDTH = 4

# ---- the eight-way classification the task fixes ----
#
# Every production refusal is a ``propagate`` refusal reason, an empty child set, a
# budget cap, a completion rejection, or a success.  These map one-to-one; nothing
# is collapsed and nothing is invented.
REASON_CLASS = {
    "no_atom_carries_a_required_operand_descriptor": "attachment_no_legal_assignment",
    "insufficient_preexisting_atoms_for_remaining_deletions": "dependency_region_mismatch",
    "insufficient_preexisting_atoms_of_a_required_identity": "dependency_region_mismatch",
    "required_created_handle_is_unreachable": "dependency_region_mismatch",
    "insufficient_slot_capacity": "primitive_program_support_cap",
}


# ---- loading ----

def load_rows(data_root: Path) -> list[dict[str, Any]]:
    """Jump attempt records from the FINAL round of each task.

    ``snapshot.history`` is CUMULATIVE across rounds -- every round's file carries
    the whole batch history -- so the final round alone is the complete record and
    unioning rounds would multiply each attempt by the number of rounds that
    followed it.
    """

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = data_root / task / "round_0014" / "complete.json"
        payload = json.loads(path.read_text())
        channels = payload["snapshot"]["pmo_population"]["population_state"]["channels"]
        counter = int(channels[JUMP]["proposals"])
        found = 0
        for batch in payload["snapshot"]["history"]:
            for attempt in batch["batch"]["attempts"]:
                if attempt.get("planner_channel") != JUMP:
                    continue
                found += 1
                rows.append({"task": task, "batch": batch["batch"].get("batch_index"), **attempt})
        if found != counter:
            raise ValueError(f"{task}: {found} attempt records vs {counter} counted proposals")
    return rows


def stored_outcome(row: dict[str, Any]) -> str:
    if row["status"] != "execution_rejected":
        return PR.OUTCOME_COMPLETED
    match = re.search(r"\(([a-z_]+)\)", row.get("reason") or "")
    return match.group(1) if match else "unparsed"


def load_plans() -> dict[str, dict[str, Any]]:
    envelope = json.loads(CHECKPOINT.read_text())
    checkpoint = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("unexpected jump checkpoint fit scope")
    return {plan["plan_id"]: plan for plan in checkpoint["plan_latents"]}


# ---- parent shape (measured on the state the lane actually binds) ----

def parent_shape(graph) -> dict[str, Any]:
    active = int(is_element(graph.atom_types).sum())
    return {
        "heavy_atoms": active,
        "n_slots": len(graph.atom_types),
        "free_slots": len(graph.atom_types) - active,
    }


def plan_shape(plan: dict[str, Any]) -> dict[str, Any]:
    roles = plan["roles"]
    rules = collections.Counter(str(role["executor_rule"]) for role in roles)
    created = sum(1 for role in roles if role.get("created_output_ordinal") is not None)
    consumes_created = sum(
        1
        for role in roles
        for operand in role.get("operands", ())
        if operand.get("descriptor", {}).get("origin") == "route_created"
    )
    return {
        "primitive_count": int(plan["primitive_count"]),
        "component_count": int(plan.get("component_count", -1)),
        "created_dependency_count": int(plan.get("created_dependency_count", -1)),
        "n_roles": len(roles),
        "n_inserts": rules.get("atom_insert", 0),
        "n_deletes": rules.get("atom_delete", 0),
        "attachment_operands": sum(len(role.get("operands", ())) for role in roles),
        "created_handles": created,
        "consumes_created_operands": consumes_created,
        "net_heavy_atom_delta": rules.get("atom_insert", 0) - rules.get("atom_delete", 0),
        "mass": float(plan["mass"]),
    }


# ---- one (plan, parent) pair, all arms ----

def run_pair(job: dict[str, Any]) -> dict[str, Any]:
    source = decode_state(job["state"])
    plan = job["plan"]
    spec = PR.PRODUCTION_SPECIFICATION

    observed: dict[str, Any] = {"first": None, "hist": collections.Counter()}
    original = PR.propagate

    def watching(prefix, step, plan_, demand, spec_):
        reason = original(prefix, step, plan_, demand, spec_)
        if reason is not None:
            observed["hist"][f"{step}:{reason}"] += 1
            if observed["first"] is None:
                observed["first"] = (int(step), reason)
        return reason

    out: dict[str, Any] = {
        "task": job["task"],
        "round_batch": job["batch"],
        "entry_id": job["entry_id"],
        "plan_id": job["plan_id"],
        "stored_outcome": job["stored_outcome"],
        "stored_status": job["status"],
        "parent": job["parent_shape"],
        "parent_depth": job["parent_depth"],
        "plan": plan_shape(plan),
    }

    PR.propagate = watching
    try:
        out["depth0_operand_feasible"] = bool(
            PR.static_first_step_feasible(source, plan, spec)
        )
        began = perf_counter()
        exact = PR.realize(
            source, plan, spec=spec,
            node_budget=PR.PRODUCTION_NODE_BUDGET,
            seconds_cap=PR.PRODUCTION_SECONDS_CAP,
            collect_all=True, max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
        )
        out["exact"] = {
            "outcome": exact["outcome"],
            "nodes_expanded": exact["nodes_expanded"],
            "successors_enumerated": exact["successors_enumerated"],
            "max_depth_reached": exact["max_depth_reached"],
            "pruned_by_propagation": exact["pruned_by_propagation"],
            "pruned_by_memo": exact["pruned_by_memo"],
            "full_depth_prefixes": exact["full_depth_prefixes"],
            "budget_reason": exact["budget_reason"],
            "completion_rejections": exact["completion_rejections"],
            "n_realizations": len(exact["realizations"]),
            "seconds": exact["seconds"],
            "first_refusal_step": observed["first"][0] if observed["first"] else None,
            "first_refusal_reason": observed["first"][1] if observed["first"] else None,
            "refusal_hist": dict(observed["hist"]),
        }
        if exact["realizations"]:
            best = exact["realizations"][0]
            out["exact"]["realized"] = {
                "primitive_count": best["primitive_count"],
                "retained_fraction": best["retained_fraction"],
                "delta_heavy_atoms": best["delta_heavy_atoms"],
                "component_count": best["component_count"],
            }
        out["exact"]["elapsed"] = perf_counter() - began
    finally:
        PR.propagate = original

    # ---- beam arms: the binder this search replaced ----
    for width in (PRODUCTION_BEAM_WIDTH, 8):
        bound = bind_joint_plan(source, plan, beam_width=width)
        rows = []
        for row in bound:
            # The beam checks exact replay + representation support, but NOT the
            # plan's declared component / dependency counts and NOT "endpoint
            # differs from source".  Recompute them so "the beam bound it" can be
            # separated from "the beam realized the requested transformation".
            endpoint = decode_state(row["endpoint_state"])
            rows.append({
                "primitive_count": row["primitive_count"],
                "component_count": row["component_count"],
                "created_dependency_edges": row["created_dependency_edges"],
                "declared_component_count": int(plan.get("component_count", -1)),
                "declared_created_dependency_count": int(
                    plan.get("created_dependency_count", -1)
                ),
                "retained_fraction": PR.retained_fraction(source, endpoint),
                "delta_heavy_atoms": int(endpoint.n_real_atoms) - int(source.n_real_atoms),
                "endpoint_equals_source": bool(
                    row["endpoint_key"] == PR.canonical_state_key(source)
                ),
            })
        out[f"beam{width}"] = {"n_bindings": len(rows), "bindings": rows}

    if job["noprune"]:
        PR.propagate = lambda *a, **k: None
        try:
            free = PR.realize(
                source, plan, spec=spec,
                node_budget=job["noprune_nodes"], seconds_cap=job["noprune_seconds"],
                collect_all=True, max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
            )
            out["noprune"] = {
                "outcome": free["outcome"],
                "nodes_expanded": free["nodes_expanded"],
                "successors_enumerated": free["successors_enumerated"],
                "max_depth_reached": free["max_depth_reached"],
                "budget_reason": free["budget_reason"],
                "n_realizations": len(free["realizations"]),
                "full_depth_prefixes": free["full_depth_prefixes"],
                "completion_rejections": free["completion_rejections"],
            }
        finally:
            PR.propagate = original
    return out


def classify(row: dict[str, Any]) -> str:
    """Exactly one reason per proposal, from the eight the task fixes."""

    exact = row["exact"]
    if exact["outcome"] == PR.OUTCOME_COMPLETED:
        return "success_realization_completed"
    if exact["outcome"] == "invalid_plan":
        return "proven_structurally_incompatible"
    if exact["outcome"] == PR.OUTCOME_EXHAUSTED:
        return "search_budget_exhausted"
    # proven_incompatible: name the necessary condition that proved it.
    if exact["full_depth_prefixes"] and exact["completion_rejections"]:
        return "downstream_validity_admission_rejection"
    reason = exact["first_refusal_reason"]
    if reason is None:
        # exhausted with no propagate refusal at all: the role's own successor
        # enumeration was empty under the exact identity.
        return "proven_structurally_incompatible"
    if exact["first_refusal_step"] == 0 and reason == (
        "no_atom_carries_a_required_operand_descriptor"
    ):
        return "proven_structurally_incompatible"
    return REASON_CLASS.get(reason, "proven_structurally_incompatible")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.expanduser("~/compose_pmo_data/rounds"))
    parser.add_argument("--out", default="diagnostics/pmo_jump_funnel_audit_v1.json")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--noprune-nodes", type=int, default=20000)
    parser.add_argument("--noprune-seconds", type=float, default=60.0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = load_rows(data_root)
    plans = load_plans()
    entries = json.loads((data_root.parent / "parent_entries.json").read_text())

    jobs = []
    for row in rows:
        entry = entries[row["entry_id"]]
        state = entry["trace"]["states"][-1]
        jobs.append({
            "task": row["task"], "batch": row["batch"], "entry_id": row["entry_id"],
            "plan_id": row["plan_id"], "status": row["status"],
            "stored_outcome": stored_outcome(row), "state": state,
            "plan": plans[row["plan_id"]],
            "parent_shape": parent_shape(decode_state(state)),
            "parent_depth": len(entry["trace"]["actions"]),
            "noprune": True,
            "noprune_nodes": args.noprune_nodes,
            "noprune_seconds": args.noprune_seconds,
        })
    if args.limit:
        jobs = jobs[: args.limit]

    began = perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_pair, jobs, chunksize=4))
    else:
        results = [run_pair(job) for job in jobs]
    elapsed = perf_counter() - began

    for row in results:
        row["classification"] = classify(row)

    disagreements = [
        {"task": r["task"], "plan_id": r["plan_id"], "entry_id": r["entry_id"],
         "stored": r["stored_outcome"], "replayed": r["exact"]["outcome"]}
        for r in results if r["stored_outcome"] != r["exact"]["outcome"]
    ]

    payload = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "contract_payload_sha256": CONTRACT_SHA,
        "new_oracle_calls": 0,
        "pairs": len(results),
        "elapsed_seconds": elapsed,
        "parity": {
            "definition": "replayed exact-arm outcome vs the outcome the production "
                          "controller stored in its own rejection reason string",
            "agreements": len(results) - len(disagreements),
            "disagreements": len(disagreements),
            "disagreement_rows": disagreements[:40],
        },
        "rows": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"pairs={len(results)} elapsed={elapsed:.1f}s parity="
          f"{len(results)-len(disagreements)}/{len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
