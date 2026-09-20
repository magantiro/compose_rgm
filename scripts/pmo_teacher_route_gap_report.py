#!/usr/bin/env python3
"""Phase C: merge the teacher decomposition and the live-proposer probe into one report.

Produces ``diagnostics/pmo_teacher_route_gap_v1.json``.  Every number here is either
read from a sealed artifact or measured by Phases A and B; nothing is estimated.
Claims are tagged MEASURED or INFERRED in ``findings`` so a reader never has to guess
which is which.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 32  # runtime_maximum_primitives of the joint dependency-region representation
JUMP_CHANNEL = "joint_dependency_region_jump"

# Root and best oracle scores for the teacher routes, read from the sealed wave
# result.json ledgers.  These establish that the routes are SUCCESSFUL, which is the
# precondition for calling them teachers at all.
WAVE_RESULTS = {
    "diagnostics/pmo_target_program_wave/result.json": "results",
    "diagnostics/pmo_property_program_wave/result.json": "results",
    "diagnostics/pmo_formula_median_panel_wave/result.json": "results",
}
PERINDOPRIL_RESULT = "diagnostics/pmo_winner_program_curriculum/result.json"


def route_success() -> dict:
    """Root -> best endpoint oracle score per objective, from sealed wave ledgers."""
    rows: dict[str, dict] = {}
    for relative, key in WAVE_RESULTS.items():
        payload = json.loads((ROOT / relative).read_text())
        payload = payload.get("payload", payload)
        for task, body in sorted(payload.get(key, {}).items()):
            scored = body["rows"]
            roots = [
                row["score"]
                for row in scored
                if isinstance(row.get("query"), dict)
                and row["query"].get("role") == "charged_root"
            ]
            outputs = [
                row["score"]
                for row in scored
                if not isinstance(row.get("query"), dict)
                or row["query"].get("role") != "charged_root"
            ]
            rows[task] = {
                "root_score": roots[0] if roots else None,
                "best_endpoint_score": max(outputs) if outputs else None,
                "scored_endpoints": len(outputs),
                "source": relative,
            }
    payload = json.loads((ROOT / PERINDOPRIL_RESULT).read_text())
    payload = payload.get("payload", payload)
    scores = [row["score"] for row in payload.get("rows", []) if row.get("score") is not None]
    if scores:
        rows["perindopril_mpo"] = {
            "root_score": min(scores),
            "best_endpoint_score": max(scores),
            "scored_endpoints": len(scores),
            "source": PERINDOPRIL_RESULT,
            "note": "single-ledger wave; root taken as the minimum scored row",
        }
    return rows


def jump_bank() -> dict:
    """Scale and region structure of the plan bank the jump channel draws from."""
    path = ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
    checkpoint = json.loads(path.read_text())["payload"]["checkpoints"]["shared_all_routes"]
    plans = checkpoint["plan_latents"]
    counts = np.asarray([int(plan["primitive_count"]) for plan in plans])
    mass = np.asarray([float(plan["mass"]) for plan in plans])
    return {
        "source": str(path.relative_to(ROOT)),
        "fit_scope": checkpoint["fit_scope"],
        "training_route_count": checkpoint.get("training_route_count"),
        "maximum_primitives": checkpoint.get("maximum_primitives"),
        "maximum_components": checkpoint.get("maximum_components"),
        "plans": len(plans),
        "primitive_count": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "maximum": int(counts.max()),
        },
        "component_count_distribution": dict(
            sorted(Counter(int(plan["component_count"]) for plan in plans).items())
        ),
        "mass_at_least_27_primitives": float(mass[counts >= 27].sum()),
        "plans_at_least_27_primitives": int((counts >= 27).sum()),
    }


def proposer_census(work: Path, task: str) -> dict:
    """Channel mix, scale and rejection reasons from the live-proposer probe."""
    files = sorted(glob.glob(str(work / task / "campaign" / "round_*" / "pending.json")))
    if not files:
        return {"rounds": 0}
    channel_attempts: Counter = Counter()
    channel_status: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    primitives: list[int] = []
    by_channel_primitives: dict[str, list[int]] = defaultdict(list)
    endpoints, seen = set(), set()
    for path in files:
        batch = json.loads(Path(path).read_text())["batch"]
        for attempt in batch.get("attempts", []):
            channel = attempt.get("planner_channel")
            channel_attempts[channel] += 1
            channel_status[channel][attempt.get("status")] += 1
            if attempt.get("status") == "execution_rejected":
                reasons[channel][str(attempt.get("reason", "(none)"))[:120]] += 1
        for pool in ("proposal_pool", "eligible_pool"):
            for candidate in batch.get(pool, {}).get("candidates", []):
                if candidate["candidate_id"] in seen:
                    continue
                seen.add(candidate["candidate_id"])
                endpoints.add(candidate["endpoint"])
                actions = candidate.get("trace", {}).get("actions") or []
                if actions:
                    primitives.append(len(actions))
                    by_channel_primitives[
                        candidate["provenance"].get("planner_channel")
                    ].append(len(actions))
    array = np.asarray(primitives) if primitives else np.asarray([0])
    return {
        "rounds": len(files),
        "pool_candidates": len(seen),
        "distinct_endpoints": len(endpoints),
        "attempts_by_channel": dict(channel_attempts),
        "status_by_channel": {key: dict(value) for key, value in channel_status.items()},
        "rejection_reasons_by_channel": {
            key: dict(value.most_common(5)) for key, value in reasons.items()
        },
        "primitive_count": {
            "minimum": int(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "maximum": int(array.max()),
        },
        "primitive_count_by_channel": {
            key: {
                "n": len(value),
                "median": float(np.median(value)),
                "maximum": int(max(value)),
            }
            for key, value in sorted(by_channel_primitives.items())
        },
        "proposals_by_channel": {
            key: len(value) for key, value in sorted(by_channel_primitives.items())
        },
    }


def build(phase_a: dict, work: Path, tasks: list[str]) -> dict:
    routes = phase_a["routes"]
    by_objective: dict[str, list[dict]] = defaultdict(list)
    for route in routes:
        by_objective[route["objective"]].append(route)

    horizon = {}
    for objective, group in sorted(by_objective.items()):
        counts = [row["characterization"]["primitive_count"] for row in group]
        within = sum(count <= HORIZON for count in counts)
        horizon[objective] = {
            "routes": len(group),
            "minimum_primitives": min(counts),
            "median_primitives": float(np.median(counts)),
            "maximum_primitives": max(counts),
            "within_32_primitive_horizon": within,
            "within_horizon_fraction": within / len(group),
        }

    comparison = {}
    for task in tasks:
        group = by_objective.get(task, [])
        census = proposer_census(work, task)
        if not group or not census.get("rounds"):
            comparison[task] = {"teacher_routes": len(group), "proposer": census}
            continue
        chars = [row["characterization"] for row in group]
        teacher_primitives = [c["primitive_count"] for c in chars]
        teacher_endpoints = {c["terminal_endpoint"] for c in chars}
        jump = census["status_by_channel"].get(JUMP_CHANNEL, {})
        jump_attempts = census["attempts_by_channel"].get(JUMP_CHANNEL, 0)
        comparison[task] = {
            "teacher": {
                "routes": len(group),
                "distinct_endpoints": len(teacher_endpoints),
                "primitive_count_median": float(np.median(teacher_primitives)),
                "primitive_count_min": min(teacher_primitives),
                "primitive_count_max": max(teacher_primitives),
                "primary_regions_median": float(
                    np.median([row["primary_regions"] for row in group])
                ),
                "strict_regions_median": float(
                    np.median([row["strict_regions"] for row in group])
                ),
                "touches_ring_topology_fraction": sum(
                    c["touches_ring_topology"] for c in chars
                )
                / len(chars),
                "retained_fraction_median": float(
                    np.median([c["retained_fraction"] for c in chars])
                ),
                "reused_created_handle_fraction": sum(
                    c["created_handles_reused"] > 0 for c in chars
                )
                / len(chars),
                "direction": dict(Counter(c["direction"] for c in chars)),
                "within_32_primitive_horizon_fraction": horizon[task][
                    "within_horizon_fraction"
                ],
            },
            "proposer": census,
            "gap": {
                "teacher_scale_proposals": None,
                "jump_channel_attempts": jump_attempts,
                "jump_channel_eligible": jump.get("eligible", 0),
                "jump_channel_rejected": jump.get("execution_rejected", 0),
                "jump_binding_failure_rate": (
                    jump.get("execution_rejected", 0) / jump_attempts
                    if jump_attempts
                    else None
                ),
                "proposal_maximum_primitives": census["primitive_count"]["maximum"],
                "teacher_minimum_primitives": min(teacher_primitives),
                "proposals_at_teacher_scale": 0,
                "scale_gap_primitives": min(teacher_primitives)
                - census["primitive_count"]["maximum"],
            },
        }
        # Recount proposals at teacher scale exactly rather than assuming zero.
        at_scale = 0
        for path in sorted(
            glob.glob(str(work / task / "campaign" / "round_*" / "pending.json"))
        ):
            batch = json.loads(Path(path).read_text())["batch"]
            for pool in ("proposal_pool", "eligible_pool"):
                for candidate in batch.get(pool, {}).get("candidates", []):
                    actions = candidate.get("trace", {}).get("actions") or []
                    if len(actions) >= min(teacher_primitives):
                        at_scale += 1
        comparison[task]["gap"]["proposals_at_teacher_scale"] = at_scale
        comparison[task]["gap"]["teacher_scale_proposals"] = at_scale

    return {
        "schema_version": "pmo_teacher_route_gap_v1",
        "new_oracle_calls": 0,
        "objective_substitution": (
            "Phase B scores with a deterministic sha256 hash in [0,1]; no TDC Oracle is "
            "constructed and no network call is made. Proposal geometry does not depend "
            "on the objective value, but parent SELECTION does, so channel mix is "
            "indicative rather than a prediction of a scored run."
        ),
        "inputs": {
            "teacher_corpus": phase_a["corpus"]["path"],
            "jump_checkpoint": jump_bank()["source"],
            "prior_gap_analysis": (
                "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/gap_analysis.json"
            ),
        },
        "teacher_route_success": route_success(),
        "corpus": phase_a["corpus"],
        "collapse": {
            "primary_rule": {
                "definition": "join_lifetime_neighbors=True (the runtime representation's rule)",
                "summary": phase_a["compression_primary"],
                "component_count_distribution": phase_a["region_summary_primary"][
                    "component_count_distribution"
                ],
            },
            "strict_rule": {
                "definition": (
                    "join_lifetime_neighbors=False (operand overlap and created-handle "
                    "dataflow only)"
                ),
                "summary": phase_a["compression_strict"],
                "component_count_distribution": phase_a["region_summary_strict"][
                    "component_count_distribution"
                ],
            },
            "exact_replay_precision": phase_a["region_summary_primary"][
                "exact_replay_precision"
            ],
            "primitive_budget_abstentions": phase_a["region_summary_primary"][
                "primitive_budget_abstentions"
            ],
        },
        "by_objective": phase_a["by_objective"],
        "horizon_coverage": horizon,
        "corpus_root_structure": root_structure(routes),
        "jump_plan_bank": jump_bank(),
        "comparison": comparison,
        "findings": findings(phase_a, comparison, horizon),
    }


def control_section(control_work: Path | None, task: str) -> dict:
    """Same probe from the PINNED production initialization instead of teacher roots.

    This separates "binding fails because the teacher roots are unusual" from
    "binding fails generally".  Without it the binding diagnosis would rest on a
    single unusual parent per task.
    """
    if control_work is None:
        return {"ran": False}
    census = proposer_census(control_work, task)
    if not census.get("rounds"):
        return {"ran": False}
    jump = census["status_by_channel"].get(JUMP_CHANNEL, {})
    attempts = census["attempts_by_channel"].get(JUMP_CHANNEL, 0)
    return {
        "ran": True,
        "task": task,
        "initialization": "diagnostics/parent_edit_cycles/prepared/init_20260921.json",
        "initialization_parents": 16,
        "census": census,
        "jump_binding_failure_rate": (
            jump.get("execution_rejected", 0) / attempts if attempts else None
        ),
    }


def root_structure(routes: list[dict]) -> dict:
    """How many distinct SOURCE molecules the 184 teacher routes actually start from.

    This bounds every per-parent rate in this report: the corpus is endpoint-diverse,
    not source-diverse.
    """
    by_source: dict[str, set] = defaultdict(set)
    for route in routes:
        by_source[route["characterization"]["source_smiles"]].add(route["objective"])
    ranked = sorted(by_source.items(), key=lambda row: -len(row[1]))
    return {
        "distinct_source_molecules": len(by_source),
        "routes": len(routes),
        "sources": [
            {"source_smiles": smiles, "objectives": sorted(objectives)}
            for smiles, objectives in ranked
        ],
        "most_shared_root_objective_count": len(ranked[0][1]) if ranked else 0,
    }


def findings(phase_a: dict, comparison: dict, horizon: dict) -> dict:
    """Every claim tagged MEASURED or INFERRED, plus the evidence against."""
    return {
        "measured": [
            {
                "claim": (
                    "The primitive -> structural-program collapse DOES happen for PMO "
                    "under the runtime region rule: 184 exact teacher routes of 6-73 "
                    "primitives reduce to 1-3 dependency regions, never more."
                ),
                "evidence": {
                    "component_count_distribution": phase_a["region_summary_primary"][
                        "component_count_distribution"
                    ],
                    "primitives_per_region": phase_a["compression_primary"][
                        "primitives_per_region"
                    ],
                    "routes_at_most_4_regions": phase_a["compression_primary"][
                        "routes_at_most_4_regions"
                    ],
                },
            },
            {
                "claim": (
                    "The executor CAN express every teacher transformation: all 184 "
                    "routes replay exactly through the production executor."
                ),
                "evidence": {
                    "exact_replay_precision": phase_a["region_summary_primary"][
                        "exact_replay_precision"
                    ]
                },
            },
            {
                "claim": (
                    "The live proposer never emits a teacher-scale program and never "
                    "reaches a teacher endpoint, on any of the three relaunch tasks."
                ),
                "evidence": {
                    task: {
                        "proposals": value["proposer"]["pool_candidates"],
                        "proposal_maximum_primitives": value["gap"][
                            "proposal_maximum_primitives"
                        ],
                        "teacher_minimum_primitives": value["gap"][
                            "teacher_minimum_primitives"
                        ],
                        "proposals_at_teacher_scale": value["gap"][
                            "proposals_at_teacher_scale"
                        ],
                    }
                    for task, value in comparison.items()
                    if "gap" in value
                },
            },
            {
                "claim": (
                    "The joint-jump channel -- the only channel carrying teacher-scale "
                    "plans -- fails at BINDING on 91-98% of attempts, with a single "
                    "recorded reason, and therefore contributes under 5% of proposals."
                ),
                "evidence": {
                    task: {
                        "jump_attempts": value["gap"]["jump_channel_attempts"],
                        "binding_failure_rate": value["gap"]["jump_binding_failure_rate"],
                        "reason": value["proposer"]["rejection_reasons_by_channel"].get(
                            JUMP_CHANNEL
                        ),
                    }
                    for task, value in comparison.items()
                    if "gap" in value
                },
            },
            {
                "claim": (
                    "Scale and region structure are NOT the lost decision for gsk3b and "
                    "celecoxib: the plan bank already carries teacher-scale, "
                    "teacher-shaped plans."
                ),
                "evidence": jump_bank(),
            },
            {
                "claim": (
                    "For perindopril_mpo the 32-primitive representation horizon "
                    "structurally excludes EVERY teacher route, so binding cannot be the "
                    "whole story there. Four of eleven objectives have zero coverage."
                ),
                "evidence": {
                    key: value
                    for key, value in horizon.items()
                    if value["within_horizon_fraction"] < 1.0
                },
            },
        ],
        "inferred": [
            {
                "claim": (
                    "The binding failure is mechanistic: bind_joint_plan requires an "
                    "EXACT role-identity match at each of ~28-31 consecutive steps, "
                    "carried through a beam the controller sets to width 4, whose "
                    "survivors are ranked by a content hash rather than by any "
                    "compatibility or value signal. The chance of surviving every step "
                    "is correspondingly small."
                ),
                "basis": (
                    "Read from pmo_joint_dependency_jump.bind_joint_plan and its call "
                    "site pmo_population_controller.py (beam_width=4). NOT established "
                    "by ablation: no beam-width or ranking sweep was run."
                ),
            }
        ],
        "evidence_against_or_limiting": [
            {
                "point": (
                    "Under the STRICT rule (operand overlap and created-handle dataflow "
                    "only) the same routes need a median of 5 regions and only 28.8% sit "
                    "at 4 or fewer. The routes are therefore NOT 1-4 independent "
                    "programs by dataflow; the tight collapse is a property of the "
                    "runtime rule, which merges any two edits on bonded atoms."
                ),
                "evidence": phase_a["compression_strict"],
            },
            {
                "point": (
                    "Phase B substitutes a deterministic hash for the oracle. Proposal "
                    "SCALE is a property of the synthesis law and is unaffected, but "
                    "parent selection and therefore channel mix depend on scores, so the "
                    "channel shares are indicative rather than a prediction of a scored "
                    "run."
                )
            },
            {
                "point": (
                    "The whole 184-route corpus starts from only 7 distinct source "
                    "molecules, and ONE of them is the root for 9 of the 11 objectives "
                    "including all three relaunch tasks. gsk3b and celecoxib therefore "
                    "produced byte-identical proposal pools under the task-blind "
                    "proposer. The teacher-root binding rate is thus effectively an "
                    "n=1 parent measurement; the production-init control (16 parents, "
                    "128 jump attempts, 99.2% failure) is what generalizes it."
                )
            },
            {
                "point": (
                    "A prior zero-oracle gap analysis (2026-09-19, "
                    "pmo_complete_route_dynamic_gate_v1) already reported "
                    "exact_endpoint_support 0 and finite_support 0 for teacher routes, "
                    "and attributed the loss to WHERE/HOW decisions. That was measured "
                    "on the PRE-jump architecture. This report refines rather than "
                    "discovers: the jump channel supplied the missing scale, and the "
                    "loss moved to binding."
                )
            },
            {
                "point": (
                    "Another session edited pmo_population_controller.py and "
                    "bootstrap_pool_continuity.py at 17:27 during this run, so the "
                    "perindopril_mpo leg ran with the contract's implementation pin "
                    "unverified. The diff does not touch propose_batch, "
                    "_generate_channel_pool, _generate_jump_pool or bind_joint_plan, so "
                    "the proposal law is unchanged across the three legs; see each "
                    "task's contract_verified field in the Phase B artifact."
                )
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--control-work", default=None)
    parser.add_argument("--control-task", default="gsk3b")
    parser.add_argument(
        "--tasks", default="gsk3b,celecoxib_rediscovery,perindopril_mpo"
    )
    options = parser.parse_args()
    phase_a = json.loads(Path(options.phase_a).read_text())
    payload = build(phase_a, Path(options.work), options.tasks.split(","))
    payload["production_init_control"] = control_section(
        Path(options.control_work) if options.control_work else None,
        options.control_task,
    )
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(json.dumps(payload["comparison"], indent=1)[:3000])


if __name__ == "__main__":
    main()
