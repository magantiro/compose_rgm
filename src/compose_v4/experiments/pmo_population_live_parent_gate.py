"""Zero-oracle live-parent support gate for PMO-v1.

The gate samples non-root descendants from the fixed objective-blind PMO
initialization lock using a deterministic score-blind current-state policy, then
binds a prospectively frozen scale-diverse subset of generic joint plans.  It
does not inspect task labels, scores, teacher endpoints, or winners.  Scoring
is prohibited until the resulting receipt passes the non-root support floor.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from compose_v4.control.current_state_edits import current_state_program
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.pmo_joint_dependency_jump import bind_joint_plan
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_population_v1 import (
    CHECKPOINTS,
    INITIALIZATION,
    configuration,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "pmo_population_live_parent_gate_v1"
FAMILIES = (
    "atom_restate_semantic",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
)
MIN_ROUTE_SCALE_PRIMITIVES = 8
MIN_SUPPORTED_PARENTS = 4
PLAN_SAMPLE_PER_BAND = 4


def _band(count: int) -> str:
    return "small" if count <= 7 else "medium" if count <= 15 else "large"


def select_plan_sample(checkpoint: dict) -> list[dict]:
    """Freeze a task-blind mass-ranked sample before any parent is examined."""

    grouped = {band: [] for band in ("small", "medium", "large")}
    for plan in checkpoint["plan_latents"]:
        grouped[_band(int(plan["primitive_count"]))].append(plan)
    selected = []
    for values in grouped.values():
        rows = sorted(values, key=lambda row: (-float(row["mass"]), row["plan_id"]))
        selected.extend(rows[:PLAN_SAMPLE_PER_BAND])
    return selected


def _sample_descendant(source, index: int):
    rng = np.random.default_rng(np.random.SeedSequence([20260920, 911, index]))
    family = FAMILIES[index % len(FAMILIES)]
    try:
        program, binding, metadata = current_state_program(source, rng, family=family)
        _, trace = execute_program_graph(
            source, compile_program_graph(program), binding, max_primitives=32, max_blocks=8
        )
    except (ValueError, RuntimeError) as error:
        return {"status": "descendant_rejected", "family": family, "reason": str(error)}
    return {
        "status": "descendant_exact",
        "family": family,
        "source_state": encode_state(source),
        "program": program.payload(),
        "assignment": list(binding),
        "trace": trace,
        "metadata": metadata,
    }


def _bind_parent_plans(payload: tuple[int, dict, list[list[dict]]]) -> tuple[int, dict, list[dict]]:
    """Independent deterministic parent shard for the exhaustive coverage audit."""

    index, descendant, plan_batches = payload
    child = decode_state(descendant["trace"]["states"][-1])
    attempts = successes = 0
    lengths, rows = [], []
    for batch_index, plans in enumerate(plan_batches):
        for plan in plans:
            attempts += 1
            bound = bind_joint_plan(child, plan, beam_width=1)
            exact = [row for row in bound if row.get("exact_replay")]
            successes += len(exact)
            lengths.extend(row["primitive_count"] for row in exact)
            rows.extend(
                {
                    "parent_index": index,
                    "batch_index": batch_index,
                    "plan_id": plan["plan_id"],
                    "primitive_count": row["primitive_count"],
                    "endpoint_key": row["endpoint_key"],
                    "exact_replay": True,
                }
                for row in exact
            )
    return (
        index,
        {
            "plans_attempted": attempts,
            "exact_bindings": successes,
            "primitive_lengths": sorted(lengths),
            "route_scale_bindings": sum(length >= MIN_ROUTE_SCALE_PRIMITIVES for length in lengths),
        },
        rows,
    )


def run_gate(
    root: Path, *, output: Path, exhaustive: bool = False, scheduler: bool = False
) -> dict:
    init = json.loads((root / INITIALIZATION).read_text())
    checkpoint_envelope = json.loads((root / CHECKPOINTS).read_text())
    checkpoint = checkpoint_envelope["payload"]["checkpoints"]["shared_all_routes"]
    plans = list(checkpoint["plan_latents"]) if exhaustive else select_plan_sample(checkpoint)
    if not plans or {_band(int(row["primitive_count"])) for row in plans} != {
        "small",
        "medium",
        "large",
    }:
        raise ValueError("joint plan sample is missing an available scale band")
    scheduled_plan_batches = None
    parents, plan_rows = [], []
    for index, locked in enumerate(init["candidates"]):
        source = decode_state(locked["state"])
        descendant = _sample_descendant(source, index)
        parent = {
            "index": index,
            "source_id": locked["source_id"],
            "endpoint": locked["endpoint"],
            "descendant": descendant,
        }
        parents.append(parent)
        if descendant["status"] != "descendant_exact":
            continue
    if scheduler:
        controller = PmoPopulationController(
            configuration(),
            source_group="pmo-live-parent-gate",
            oracle_protocol="zero-oracle-support-gate",
            hierarchy=None,
            jump_checkpoint=checkpoint,
        )
        plan_order = controller._jump_plan_order()
        plan_batches = []
        for batch in range(4):
            offset = (batch * 32) % len(plan_order)
            plan_batches.append(
                plan_order[offset : offset + 32]
                if offset + 32 <= len(plan_order)
                else plan_order[offset:] + plan_order[: (offset + 32) % len(plan_order)]
            )
        scheduled_plan_batches = [[row["plan_id"] for row in batch] for batch in plan_batches]
        if {row["plan_id"] for batch in plan_batches for row in batch} != {
            row["plan_id"] for row in checkpoint["plan_latents"]
        }:
            raise ValueError("scheduler chunks omit a joint plan latent")
        work = [
            (index, parent["descendant"], plan_batches)
            for index, parent in enumerate(parents)
            if parent["descendant"]["status"] == "descendant_exact"
        ]
        with ProcessPoolExecutor(max_workers=min(4, max(1, len(work)))) as pool:
            shards = list(pool.map(_bind_parent_plans, work))
        shard_by_index = {index: (summary, rows) for index, summary, rows in shards}
        for index, parent in enumerate(parents):
            if index in shard_by_index:
                summary, rows = shard_by_index[index]
                parent["joint_binding"] = summary
                plan_rows.extend(rows)
    elif exhaustive:
        work = [
            (index, parent["descendant"], [plans])
            for index, parent in enumerate(parents)
            if parent["descendant"]["status"] == "descendant_exact"
        ]
        with ProcessPoolExecutor(max_workers=min(4, max(1, len(work)))) as pool:
            shards = list(pool.map(_bind_parent_plans, work))
        shard_by_index = {index: (summary, rows) for index, summary, rows in shards}
        for index, parent in enumerate(parents):
            if index in shard_by_index:
                summary, rows = shard_by_index[index]
                parent["joint_binding"] = summary
                plan_rows.extend(rows)
    else:
        for index, parent in enumerate(parents):
            if parent["descendant"]["status"] != "descendant_exact":
                continue
            child = decode_state(parent["descendant"]["trace"]["states"][-1])
            attempts = successes = 0
            lengths, rows = [], []
            for plan in plans:
                attempts += 1
                exact = [
                    row
                    for row in bind_joint_plan(child, plan, beam_width=8)
                    if row.get("exact_replay")
                ]
                successes += len(exact)
                lengths.extend(row["primitive_count"] for row in exact)
                rows.extend(
                    {
                        "parent_index": index,
                        "plan_id": plan["plan_id"],
                        "primitive_count": row["primitive_count"],
                        "endpoint_key": row["endpoint_key"],
                        "exact_replay": True,
                    }
                    for row in exact
                )
            parent["joint_binding"] = {
                "plans_attempted": attempts,
                "exact_bindings": successes,
                "primitive_lengths": sorted(lengths),
                "route_scale_bindings": sum(
                    length >= MIN_ROUTE_SCALE_PRIMITIVES for length in lengths
                ),
            }
            plan_rows.extend(rows)
    supported = {
        index
        for index in range(len(init["candidates"]))
        if any(
            row["parent_index"] == index and row["primitive_count"] >= MIN_ROUTE_SCALE_PRIMITIVES
            for row in plan_rows
        )
    }
    scheduler_batches = scheduled_plan_batches
    payload = {
        "schema_version": SCHEMA,
        "initialization_sha256": sha256_file(root / INITIALIZATION),
        "initialization_lock_sha256": init["lock_sha256"],
        "checkpoint_sha256": sha256_file(root / CHECKPOINTS),
        "checkpoint_payload_sha256": identity(checkpoint),
        "plan_sample": {
            "rule": "scheduler four 32-plan chunks over all plan latents"
            if scheduler
            else "all plan latents"
            if exhaustive
            else "top task-blind plan mass within each small/medium/large band",
            "plan_ids": [row["plan_id"] for row in plans],
            "scheduler_batches": scheduler_batches,
            "counts_by_band": {
                band: sum(_band(int(row["primitive_count"])) == band for row in plans)
                for band in ("small", "medium", "large")
            },
        },
        "parents": parents,
        "support": {
            "parent_count": len(parents),
            "descendant_exact_count": sum(
                row["descendant"]["status"] == "descendant_exact" for row in parents
            ),
            "route_scale_exact_count": len(
                [row for row in plan_rows if row["primitive_count"] >= MIN_ROUTE_SCALE_PRIMITIVES]
            ),
            "supported_parent_indices": sorted(supported),
            "supported_parent_count": len(supported),
            "unique_route_scale_endpoints": len(
                {
                    row["endpoint_key"]
                    for row in plan_rows
                    if row["primitive_count"] >= MIN_ROUTE_SCALE_PRIMITIVES
                }
            ),
            "minimum_route_scale_primitives": MIN_ROUTE_SCALE_PRIMITIVES,
            "required_supported_parents": MIN_SUPPORTED_PARENTS,
        },
        "oracle_calls": 0,
        "scored_launch_authorized": False,
    }
    result = {"payload": payload, "payload_sha256": identity(payload)}
    publish_json(output, result)
    return result


__all__ = ["run_gate", "select_plan_sample"]
