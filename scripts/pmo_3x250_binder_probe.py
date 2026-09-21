"""Decompose joint-jump binding failures in the PMO 3x250 run (offline, zero oracle calls).

96-99% of ``joint_dependency_region_jump`` attempts in the run failed with ONE reason
string -- ``joint plan has no legal binding on this parent`` -- but the controller raises
that string for two structurally different events, because it only checks whether
``bind_joint_plan`` returned an empty list:

  A. NO COMPATIBLE REALIZATION -- the beam emptied at some role step, i.e. no legal
     successor of the current graph carries the role the plan demands next;
  B. BINDING DISCARDED -- the whole role sequence bound, but every surviving beam prefix
     was then rejected by the dependency-region post-filter (``exact_replay`` /
     ``complete_representation_supported``).

A third possibility is beam truncation: a realization exists but the ranking -- which is a
CONTENT HASH of (plan_id, step, candidate), carrying no chemical preference -- dropped it
before the last step.  The controller calls ``bind_joint_plan(..., beam_width=4)`` while
the function's own default is 8, so this is testable by replay.

The step walk below uses the production primitives ``enumerate_role_successors`` and
``action_role_supervision`` rather than a transcription of the beam body, and every probed
pair is cross-checked against the real ``bind_joint_plan``: if the walk reports the beam
died before the last step, the real function MUST return empty, and if the real function
returns a binding, the walk MUST report survival.  Both directions are asserted, so a
divergence is reported as a probe defect instead of being read as chemistry.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision
from compose_v4.control.pmo_joint_dependency_jump import (
    bind_joint_plan,
    enumerate_role_successors,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CHECKPOINT = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
CONTROLLER_BEAM_WIDTH = 4  # pmo_population_controller._generate_jump_pool
FUNCTION_DEFAULT_BEAM_WIDTH = 8  # bind_joint_plan signature


def load_plans() -> dict[str, dict]:
    envelope = json.loads(Path(CHECKPOINT).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    return {plan["plan_id"]: plan for plan in payload["plan_latents"]}


def beam_walk(source, plan: dict, *, beam_width: int) -> dict:
    """How far the role sequence binds, and how wide the frontier is at each step.

    Mirrors ``bind_joint_plan``'s frontier exactly but records per-step survival.  It stops
    at the last role; the dependency-region post-filter is deliberately NOT applied here,
    because separating it from the walk is the entire purpose of this probe.
    """
    prefixes = [{
        "graph": source,
        "states": (encode_state(source),),
        "actions": (),
        "created": (),
        "next_ordinal": 0,
    }]
    widths = []
    for step, desired in enumerate(plan["roles"]):
        desired_identity = identity(desired)
        advanced: dict[tuple[str, str], dict] = {}
        for prefix in prefixes:
            prefix_created = {
                slot: (ordinal, made_at) for slot, ordinal, made_at in prefix["created"]
            }
            for candidate in enumerate_role_successors(
                prefix["graph"], desired, created=prefix_created, step=step
            ):
                created = dict(prefix_created)
                observed, next_ordinal = action_role_supervision(
                    prefix["graph"], candidate.action_record, created, step,
                    prefix["next_ordinal"],
                )
                if identity(observed) != desired_identity:
                    continue
                child = {
                    "graph": candidate.successor,
                    "states": (*prefix["states"], encode_state(candidate.successor)),
                    "actions": (*prefix["actions"], candidate.action_record),
                    "created": tuple(sorted(
                        (slot, ordinal, made_at)
                        for slot, (ordinal, made_at) in created.items())),
                    "next_ordinal": next_ordinal,
                }
                advanced.setdefault((candidate.successor_key, identity(child["actions"])), child)
        widths.append(len(advanced))
        if not advanced:
            return {
                "survived_all_steps": False,
                "died_at_step": step,
                "plan_length": len(plan["roles"]),
                "frontier_widths": widths,
                "fraction_of_plan_bound": step / len(plan["roles"]),
            }
        ranked = sorted(advanced.items(), key=lambda row: identity({
            "plan_id": plan["plan_id"], "step": step,
            "candidate": row[0], "beam_stream": 20260919,
        }))
        prefixes = [prefix for _, prefix in ranked[:beam_width]]
    return {
        "survived_all_steps": True,
        "died_at_step": None,
        "plan_length": len(plan["roles"]),
        "frontier_widths": widths,
        "fraction_of_plan_bound": 1.0,
    }


def _role_digest(role: dict) -> dict:
    """What the plan demanded at the step the beam died, in readable terms."""
    operands = role.get("operands") or []
    descriptors = []
    for operand in operands:
        descriptor = operand.get("descriptor") or {}
        descriptors.append({
            "role": operand.get("role"),
            "atom_type": descriptor.get("atom_type"),
            "degree": descriptor.get("degree"),
            "implicit_hydrogens": descriptor.get("implicit_hydrogens"),
            "formal_charge": descriptor.get("formal_charge"),
            "origin": descriptor.get("origin"),
        })
    return {
        "executor_rule": role.get("executor_rule"),
        "operand_count": len(operands),
        "operands": descriptors,
        "requires_created_handle": bool(role.get("created_handle_dependencies")),
    }


def collect_failures(task_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    entries, attempts = {}, []
    folders = sorted(glob.glob(str(task_dir / "campaign" / "round_*")))
    for folder in folders:
        snapshot = json.loads(Path(folder, "complete.json").read_text())["snapshot"]
        entries.update(snapshot["entries"])
        for attempt in json.loads(Path(folder, "pending.json").read_text())["batch"]["attempts"]:
            if attempt.get("planner_channel") != "joint_dependency_region_jump":
                continue
            attempts.append({"round": Path(folder).name, **attempt})
    return attempts, entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--wider-beams", default="8,16")
    arguments = parser.parse_args()

    plans = load_plans()
    attempts, entries = collect_failures(Path(arguments.task_dir))
    failures = [a for a in attempts if a["status"] == "execution_rejected"]
    other_reason = [
        a for a in failures if a.get("reason") != "joint plan has no legal binding on this parent"
    ]

    resolvable = [a for a in failures if a.get("entry_id") in entries and a.get("plan_id") in plans]
    rng = random.Random(arguments.seed)
    sample = rng.sample(resolvable, min(arguments.sample, len(resolvable)))
    wider = [int(x) for x in arguments.wider_beams.split(",") if x.strip()]

    rows, defects = [], []
    for attempt in sample:
        plan = plans[attempt["plan_id"]]
        parent = entries[attempt["entry_id"]]
        source = decode_state(parent["trace"]["states"][-1])
        walk = beam_walk(source, plan, beam_width=CONTROLLER_BEAM_WIDTH)
        real = bind_joint_plan(source, plan, beam_width=CONTROLLER_BEAM_WIDTH)
        # Both directions of the cross-check.
        if not walk["survived_all_steps"] and real:
            defects.append({"attempt": attempt["attempt"], "issue": "walk died but real bound"})
        if walk["survived_all_steps"] and not real:
            cause = "binding_discarded_by_dependency_region_filter"
        elif not walk["survived_all_steps"]:
            cause = "no_compatible_realization"
        else:
            cause = "bound_but_controller_recorded_failure"
        wider_results = {}
        if cause != "bound_but_controller_recorded_failure":
            for width in wider:
                wider_results[str(width)] = bool(bind_joint_plan(source, plan, beam_width=width))
        rows.append({
            "round": attempt["round"],
            "attempt": attempt["attempt"],
            "plan_id": attempt["plan_id"][:16],
            "plan_primitives": plan["primitive_count"],
            "plan_components": plan["component_count"],
            "parent_heavy_atoms": int(source.n_real_atoms),
            "parent_score": attempt.get("parent_measured_score"),
            "cause": cause,
            "died_at_step": walk["died_at_step"],
            "fraction_of_plan_bound": walk["fraction_of_plan_bound"],
            "frontier_width_at_death": (
                None if walk["died_at_step"] is None
                else (walk["frontier_widths"][walk["died_at_step"] - 1]
                      if walk["died_at_step"] else None)
            ),
            "max_frontier_width": max(walk["frontier_widths"]) if walk["frontier_widths"] else 0,
            "binds_at_wider_beam": wider_results,
            "blocking_role": (
                None if walk["died_at_step"] is None
                else _role_digest(plan["roles"][walk["died_at_step"]])
            ),
        })

    causes = Counter(row["cause"] for row in rows)
    died = [row for row in rows if row["cause"] == "no_compatible_realization"]
    by_step = Counter(row["died_at_step"] for row in died)
    rescued = {
        str(width): sum(1 for row in rows if row["binds_at_wider_beam"].get(str(width)))
        for width in wider
    }

    payload = {
        "schema_version": "pmo_3x250_binder_probe_v1",
        "oracle_calls_made_by_this_analysis": 0,
        "task_dir": str(arguments.task_dir),
        "controller_beam_width": CONTROLLER_BEAM_WIDTH,
        "function_default_beam_width": FUNCTION_DEFAULT_BEAM_WIDTH,
        "jump_attempts_total": len(attempts),
        "jump_attempts_failed": len(failures),
        "jump_failures_with_a_different_reason_string": len(other_reason),
        "jump_failures_resolvable_for_replay": len(resolvable),
        "sampled": len(rows),
        "probe_defects": defects,
        "cause_decomposition": dict(causes),
        "death_step_histogram": dict(sorted(
            (k, v) for k, v in by_step.items() if k is not None)),
        "died_at_step_0_share": (
            by_step.get(0, 0) / len(died) if died else None
        ),
        "median_fraction_of_plan_bound_before_death": (
            sorted(row["fraction_of_plan_bound"] for row in died)[len(died) // 2] if died else None
        ),
        "rescued_by_wider_beam": rescued,
        "blocking_rule_histogram": dict(sorted(Counter(
            row["blocking_role"]["executor_rule"] for row in rows
            if row.get("blocking_role")).items())),
        "blocking_role_needs_created_handle": sum(
            1 for row in rows
            if row.get("blocking_role") and row["blocking_role"]["requires_created_handle"]),
        "rows": rows,
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
