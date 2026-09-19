"""Run the zero-oracle 5HT1B-0 binding-particle joint proposal gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import rdkit

from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_policy import materialize_template
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    propose_virtual_joint_region_paths,
)
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from tools.t4_virtual_joint_5ht1b0_support_gate import (
    CHECKPOINT,
    FORENSICS,
    RECEIPTS,
    ROOT,
    ROUTE_DIRECTORY,
    ROUTE_IDS,
    SHARED_LIBRARY,
    SOURCE_AUDIT,
    SOURCE_REGISTRY,
    T4_CONTRACT,
    _constituent_key,
    _load_expert,
    _route_inputs,
    _sha256,
)

SCHEMA = "t4_virtual_joint_5ht1b0_binding_particle_gate_v1"
RUNTIME_SCHEMA = "t4_virtual_joint_5ht1b0_binding_particle_runtime_v1"
PRIOR_RESULT = (
    ROOT / "diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_1/result.json"
)
PARTICLE_INDICES = tuple(range(8))
ALLOCATION = "template_binding_particle"
EXPECTED_BUDGETS = {
    "max_depth": 4,
    "beam_width": 64,
    "expansion_width": 48,
    "max_bindings_per_template": 8,
    "max_binding_visits": 16_384,
    "max_planning_expansions": 4_096,
    "max_targets": 128,
    "max_realization_attempts": 128,
    "max_realization_expansions": 65_536,
    "maximum_expansions_per_realization": 4_000,
    "maximum_primitives": 32,
}
CODE_INPUTS = (
    "src/compose_v4/control/complete_region_program.py",
    "src/compose_v4/control/route_distilled_goal_expert.py",
    "src/compose_v4/control/structural_subgoal.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/structural_subgoal_realizer.py",
    "src/compose_v4/control/virtual_joint_region_proposer.py",
    "src/compose_v4/experiments/route_proposal_quality.py",
    "tools/t4_structural_subgoal_audit.py",
    "tools/t4_structural_subgoal_policy.py",
    "tools/t4_virtual_joint_5ht1b0_support_gate.py",
    "tools/t4_virtual_joint_5ht1b0_particle_gate.py",
)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sealed(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_sha256 != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload, str(payload_sha256)


def _teacher_constituents(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for route in routes:
        for region_index, (template, binding) in enumerate(
            zip(route["templates"], route["bindings"], strict=True)
        ):
            patch = materialize_template(template, route["source"], binding)
            rows.append(
                {
                    "route_id": route["route_id"],
                    "region_index": region_index,
                    "template_id": template.template_id,
                    "constituent_key": _constituent_key(template, binding, patch),
                }
            )
    if len(rows) != 9:
        raise RuntimeError("strong-route constituent census changed")
    return rows


def _compact_bound_census(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: row.get(key)
            for key in (
                "rank",
                "global_rank",
                "within_template_rank",
                "selection_rank",
                "constituent_key",
                "template_id",
                "scale",
                "selected_for_expansion",
            )
        }
        for row in telemetry["bound_constituent_ranks"]
    ]


def _teacher_coverage(
    teacher_constituents: list[dict[str, Any]], telemetry: dict[str, Any]
) -> dict[str, Any]:
    bound_rows = {
        row["constituent_key"]: row for row in telemetry["bound_constituent_ranks"]
    }
    rows = []
    for teacher in teacher_constituents:
        bound = bound_rows.get(teacher["constituent_key"])
        rows.append(
            {
                **teacher,
                "bound": bound is not None,
                "global_rank": None if bound is None else int(bound["global_rank"]),
                "within_template_rank": (
                    None if bound is None else int(bound["within_template_rank"])
                ),
                "selected_for_expansion": bool(
                    bound and bound["selected_for_expansion"]
                ),
                "selection_rank": (None if bound is None else bound["selection_rank"]),
                "scale": None if bound is None else bound["scale"],
            }
        )
    route_rows = []
    for route_id in ROUTE_IDS:
        selected = [row for row in rows if row["route_id"] == route_id]
        route_rows.append(
            {
                "route_id": route_id,
                "bound_constituents": sum(row["bound"] for row in selected),
                "selected_constituents": sum(
                    row["selected_for_expansion"] for row in selected
                ),
                "all_three_selected": all(
                    row["selected_for_expansion"] for row in selected
                ),
            }
        )
    return {
        "constituents": rows,
        "bound_numerator": sum(row["bound"] for row in rows),
        "bound_denominator": len(rows),
        "selected_numerator": sum(row["selected_for_expansion"] for row in rows),
        "selected_denominator": len(rows),
        "routes": route_rows,
    }


def _recovery(
    routes: list[dict[str, Any]], proposals: list[Any]
) -> list[dict[str, Any]]:
    rows = []
    for route in routes:
        exact_rank = next(
            (
                rank
                for rank, proposal in enumerate(proposals, 1)
                if proposal.endpoint_key == route["endpoint_key"]
            ),
            None,
        )
        equivalent_rank = next(
            (
                rank
                for rank, proposal in enumerate(proposals, 1)
                if transformation_equivalent(
                    route["source"], proposal.endpoint, route["endpoint"]
                )
            ),
            None,
        )
        rows.append(
            {
                "route_id": route["route_id"],
                "exact_endpoint_recovered": exact_rank is not None,
                "exact_endpoint_rank": exact_rank,
                "transformation_equivalent_recovered": equivalent_rank is not None,
                "transformation_equivalent_rank": equivalent_rank,
            }
        )
    return rows


def _depth_counts(proposals: list[Any]) -> dict[str, int]:
    counts = Counter(proposal.depth for proposal in proposals)
    return {str(depth): counts[depth] for depth in range(1, 5)}


def _diversity(proposals: list[Any]) -> dict[str, Any]:
    endpoint_keys = [proposal.endpoint_key for proposal in proposals]
    program_ids = {proposal.program_id for proposal in proposals}
    combinations = {
        tuple(step.constituent_key for step in proposal.steps) for proposal in proposals
    }
    primitive_counts = Counter(proposal.primitive_count for proposal in proposals)
    return {
        "unique_endpoints": len(set(endpoint_keys)),
        "unique_program_ids": len(program_ids),
        "unique_constituent_combinations": len(combinations),
        "depth_counts": _depth_counts(proposals),
        "primitive_count_distribution": {
            str(key): primitive_counts[key] for key in sorted(primitive_counts)
        },
        "endpoint_set_sha256": identity(sorted(endpoint_keys)),
        "program_set_sha256": identity(sorted(program_ids)),
        "constituent_combination_set_sha256": identity(sorted(combinations)),
    }


def _budget_accounting(
    telemetry: dict[str, Any], budgets: VirtualJointRegionBudgets
) -> dict[str, Any]:
    return {
        "binding_visits": {
            "used": int(telemetry["binding_visits"]),
            "limit": budgets.max_binding_visits,
            "exhausted": bool(telemetry.get("binding_visit_budget_exhausted", 0)),
        },
        "planning_expansions": {
            "used": int(telemetry["planning_expansions"]),
            "limit": budgets.max_planning_expansions,
            "exhausted": bool(telemetry.get("planning_expansion_budget_exhausted", 0)),
        },
        "retained_targets": {
            "used": int(telemetry["unique_valid_targets"]),
            "limit": budgets.max_targets,
            "exhausted": bool(telemetry.get("target_budget_exhausted", 0)),
        },
        "realization_attempts": {
            "used": int(telemetry["realization_attempts"]),
            "limit": budgets.max_realization_attempts,
            "exhausted": bool(telemetry.get("realization_attempt_budget_exhausted", 0)),
        },
        "realizer_expansions": {
            "used": int(telemetry["realizer_expansions"]),
            "limit": budgets.max_realization_expansions,
            "exhausted": bool(
                telemetry.get("realization_expansion_budget_exhausted", 0)
            ),
        },
        "per_realization_expansion_cap": {
            "limit": budgets.maximum_expansions_per_realization,
            "abstentions_at_cap": int(
                telemetry.get("per_realization_expansion_cap_abstentions", 0)
            ),
        },
    }


def _stop_counts(telemetry: dict[str, Any]) -> dict[str, Any]:
    return {
        str(depth): {
            "valid": int(telemetry.get(f"valid_stop_targets_depth_{depth}", 0)),
            "invalid": int(telemetry.get(f"invalid_stop_targets_depth_{depth}", 0)),
            "retained": int(telemetry.get(f"retained_targets_depth_{depth}", 0)),
        }
        for depth in range(2, 5)
    }


def _abstentions(telemetry: dict[str, Any]) -> dict[str, Any]:
    statuses = {
        key.removeprefix("realization_status:"): int(value)
        for key, value in telemetry.items()
        if key.startswith("realization_status:")
    }
    return {
        "final_target_abstentions": int(telemetry["final_target_abstentions"]),
        "compiler_abstentions": int(telemetry["compiler_abstentions"]),
        "realization_status_counts": dict(sorted(statuses.items())),
    }


def _union_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    proposal = row["proposal"]
    return (
        -proposal.log_probability,
        proposal.endpoint_key,
        tuple(step.constituent_key for step in proposal.steps),
        proposal.program_id,
        row["particle_index"],
        row["particle_rank"],
    )


def _deduplicate_union(particle_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_endpoint: dict[str, dict[str, Any]] = {}
    for particle in particle_batches:
        for particle_rank, proposal in enumerate(particle["batch"].proposals, 1):
            row = {
                "particle_index": particle["particle_index"],
                "particle_rank": particle_rank,
                "proposal": proposal,
            }
            previous = by_endpoint.get(proposal.endpoint_key)
            if previous is None or _union_rank_key(row) < _union_rank_key(previous):
                by_endpoint[proposal.endpoint_key] = row
    return sorted(by_endpoint.values(), key=_union_rank_key)


def _union_diversity(
    particle_batches: list[dict[str, Any]], union_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    endpoint_particles: dict[str, set[int]] = defaultdict(set)
    total_occurrences = 0
    new_endpoint_contribution = {}
    seen = set()
    for particle in particle_batches:
        new_count = 0
        for proposal in particle["batch"].proposals:
            total_occurrences += 1
            endpoint_particles[proposal.endpoint_key].add(particle["particle_index"])
            if proposal.endpoint_key not in seen:
                seen.add(proposal.endpoint_key)
                new_count += 1
        new_endpoint_contribution[str(particle["particle_index"])] = new_count
    occurrence_histogram = Counter(len(rows) for rows in endpoint_particles.values())
    proposals = [row["proposal"] for row in union_rows]
    return {
        **_diversity(proposals),
        "total_particle_endpoint_occurrences": total_occurrences,
        "duplicate_endpoint_occurrences": total_occurrences - len(proposals),
        "endpoint_particle_count_distribution": {
            str(key): occurrence_histogram[key] for key in sorted(occurrence_histogram)
        },
        "new_unique_endpoint_contribution_by_particle": new_endpoint_contribution,
    }


def _material_inputs() -> dict[str, str]:
    paths = (
        CHECKPOINT,
        SOURCE_AUDIT,
        FORENSICS,
        SHARED_LIBRARY,
        SOURCE_REGISTRY,
        T4_CONTRACT,
        PRIOR_RESULT,
        *(ROUTE_DIRECTORY / f"{route_id}.json" for route_id in ROUTE_IDS),
        *(RECEIPTS[route_id] for route_id in ROUTE_IDS),
    )
    return {_relative(path): _sha256(path) for path in paths}


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    overall_started = perf_counter()
    budgets = VirtualJointRegionBudgets()
    if asdict(budgets) != EXPECTED_BUDGETS:
        raise RuntimeError("default virtual-joint budgets changed")
    routes, route_context = _route_inputs()
    expert, checkpoint_payload_sha256 = _load_expert()
    source = route_context["source"]

    particle_batches = []
    particle_runtimes = []
    for particle_index in PARTICLE_INDICES:
        started = perf_counter()
        batch = propose_virtual_joint_region_paths(
            source,
            expert,
            budgets=budgets,
            constituent_allocation=ALLOCATION,
            binding_particle_index=particle_index,
        )
        particle_runtimes.append(
            {
                "particle_index": particle_index,
                "proposal_wall_seconds": perf_counter() - started,
            }
        )
        if (
            batch.telemetry["task_cell_route_or_endpoint_input_used"]
            or batch.telemetry["primitive_teacher_actions_used"] != 0
        ):
            raise RuntimeError("particle proposal used forbidden teacher or task input")
        particle_batches.append({"particle_index": particle_index, "batch": batch})

    posthoc_started = perf_counter()
    teachers = _teacher_constituents(routes)
    particle_reports = []
    for particle in particle_batches:
        particle_index = particle["particle_index"]
        batch = particle["batch"]
        telemetry = batch.telemetry
        compact_census = _compact_bound_census(telemetry)
        teacher_coverage = _teacher_coverage(teachers, telemetry)
        recovery = _recovery(routes, list(batch.proposals))
        particle_reports.append(
            {
                "particle_index": particle_index,
                "teacher_constituent_support": teacher_coverage,
                "recovery": recovery,
                "valid_stop_targets_depth_2_to_4": _stop_counts(telemetry),
                "exact_realization_precision": {
                    "numerator": int(
                        telemetry["exact_realization_precision_numerator"]
                    ),
                    "denominator": int(
                        telemetry["exact_realization_precision_denominator"]
                    ),
                },
                "abstentions": _abstentions(telemetry),
                "budgets": _budget_accounting(telemetry, budgets),
                "proposal_diversity": _diversity(list(batch.proposals)),
                "bound_constituents": int(telemetry["bound_constituents"]),
                "selected_constituents": int(telemetry["selected_constituents"]),
                "bound_templates": int(telemetry["bound_template_group_count"]),
                "scale_census": telemetry["scale_census"],
                "bound_constituent_census_sha256": identity(compact_census),
            }
        )

    union_rows = _deduplicate_union(particle_batches)
    union_proposals = [row["proposal"] for row in union_rows]
    union_recovery = _recovery(routes, union_proposals)
    exact_precision_numerator = sum(
        report["exact_realization_precision"]["numerator"]
        for report in particle_reports
    )
    exact_precision_denominator = sum(
        report["exact_realization_precision"]["denominator"]
        for report in particle_reports
    )
    selected_particles = {
        teacher["constituent_key"]: [
            report["particle_index"]
            for report in particle_reports
            if next(
                row
                for row in report["teacher_constituent_support"]["constituents"]
                if row["constituent_key"] == teacher["constituent_key"]
            )["selected_for_expansion"]
        ]
        for teacher in teachers
    }
    bound_particles = {
        teacher["constituent_key"]: [
            report["particle_index"]
            for report in particle_reports
            if next(
                row
                for row in report["teacher_constituent_support"]["constituents"]
                if row["constituent_key"] == teacher["constituent_key"]
            )["bound"]
        ]
        for teacher in teachers
    }
    route_particles = {
        route_id: [
            report["particle_index"]
            for report in particle_reports
            if next(
                row
                for row in report["teacher_constituent_support"]["routes"]
                if row["route_id"] == route_id
            )["all_three_selected"]
        ]
        for route_id in ROUTE_IDS
    }
    valid_stop_occurrences = {
        str(depth): {
            "valid": sum(
                report["valid_stop_targets_depth_2_to_4"][str(depth)]["valid"]
                for report in particle_reports
            ),
            "retained": sum(
                report["valid_stop_targets_depth_2_to_4"][str(depth)]["retained"]
                for report in particle_reports
            ),
            "particles_with_valid_stops": sum(
                report["valid_stop_targets_depth_2_to_4"][str(depth)]["valid"] > 0
                for report in particle_reports
            ),
            "unique_committed_union_endpoints": sum(
                proposal.depth == depth for proposal in union_proposals
            ),
        }
        for depth in range(2, 5)
    }
    aggregate_abstentions = {
        "final_target_abstentions": sum(
            report["abstentions"]["final_target_abstentions"]
            for report in particle_reports
        ),
        "compiler_abstentions": sum(
            report["abstentions"]["compiler_abstentions"] for report in particle_reports
        ),
        "realization_status_counts": dict(
            sorted(
                sum(
                    (
                        Counter(report["abstentions"]["realization_status_counts"])
                        for report in particle_reports
                    ),
                    Counter(),
                ).items()
            )
        ),
    }
    aggregate_budget_usage = {
        name: {
            "used": sum(report["budgets"][name]["used"] for report in particle_reports),
            "total_eight_call_limit": sum(
                report["budgets"][name]["limit"] for report in particle_reports
            ),
            "particles_exhausted": sum(
                report["budgets"][name]["exhausted"] for report in particle_reports
            ),
        }
        for name in (
            "binding_visits",
            "planning_expansions",
            "retained_targets",
            "realization_attempts",
            "realizer_expansions",
        )
    }
    union_signature = [
        {
            "endpoint_sha256": identity(row["proposal"].endpoint_key),
            "particle_index": row["particle_index"],
            "particle_rank": row["particle_rank"],
            "program_id": row["proposal"].program_id,
            "constituent_keys": [
                step.constituent_key for step in row["proposal"].steps
            ],
            "actions_sha256": identity(list(row["proposal"].actions)),
        }
        for row in union_rows
    ]
    prior_result, prior_payload_sha256 = _sealed(PRIOR_RESULT)
    public_routes = [
        {
            "route_id": route["route_id"],
            "reported_external_docking_score": route["reported_external_docking_score"],
            "reported_run_seed": route["reported_run_seed"],
            "teacher_primitive_count": route["teacher_primitive_count"],
            "teacher_receipt_sha256": route["teacher_receipt_sha256"],
            "endpoint_sha256": identity(route["endpoint_key"]),
            "runtime_artifact_sha256": route["runtime_artifact_sha256"],
            "runtime_artifact_payload_sha256": route["runtime_artifact_payload_sha256"],
        }
        for route in routes
    ]
    payload = {
        "schema_version": SCHEMA,
        "evidence": (
            "computed zero-oracle autonomous binding-particle joint proposal gate"
        ),
        "scientific_problem": (
            "test whether deterministic within-template binding particles expose and "
            "autonomously recover the three strong 5HT1B-0 delta=0.4 routes"
        ),
        "central_claim_under_test": (
            "eight fixed binding particles expand binding-alternative coverage without "
            "widening any per-call budget or injecting known endpoints"
        ),
        "claim_boundary": (
            "teacher route and constituent identities are used only after all autonomous "
            "particle calls return; no docking utility or prospective optimization is measured"
        ),
        "code_revision": _revision(),
        "implementation_commit_under_test": (
            "5f9d08203016b7d978fe604f0aa30db09d5361b3"
        ),
        "configuration": {
            "constituent_allocation": ALLOCATION,
            "particle_indices": list(PARTICLE_INDICES),
            "joint_budgets_per_particle": asdict(budgets),
            "union_deduplication": (
                "canonical endpoint; retain minimum proposal rank key then sort by the "
                "same key with particle index and particle rank as final tie breakers"
            ),
            "transformation_equivalence_radius": 2,
            "budget_widening": False,
            "runtime_threads": 1,
        },
        "determinism": {
            "proposal_rng": "none",
            "seed": None,
            "particle_order": list(PARTICLE_INDICES),
            "serialization": "sorted-key compact JSON with SHA-256 payload identity",
            "wall_runtime_excluded_from_deterministic_result": True,
        },
        "runtime_reporting": {
            "artifact_name": "runtime.json",
            "role": "non-authoritative operational telemetry excluded from result identity",
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rdkit": rdkit.__version__,
            "platform_system": platform.system(),
            "machine": platform.machine(),
            "precision": "native CPU graph execution; no learned numeric inference",
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
        "source": {
            "source_group": SOURCE_AUDIT.parent.name,
            "source_state_sha256": route_context["source_state_sha256"],
        },
        "routes": public_routes,
        "particles": particle_reports,
        "union": {
            "teacher_constituent_support": {
                "bound_numerator": sum(bool(rows) for rows in bound_particles.values()),
                "bound_denominator": len(teachers),
                "selected_numerator": sum(
                    bool(rows) for rows in selected_particles.values()
                ),
                "selected_denominator": len(teachers),
                "selected_particles_by_constituent": [
                    {
                        **teacher,
                        "bound_particle_indices": bound_particles[
                            teacher["constituent_key"]
                        ],
                        "particle_indices": selected_particles[
                            teacher["constituent_key"]
                        ],
                    }
                    for teacher in teachers
                ],
                "particles_with_all_three_constituents_by_route": [
                    {
                        "route_id": route_id,
                        "particle_indices": route_particles[route_id],
                    }
                    for route_id in ROUTE_IDS
                ],
            },
            "recovery": union_recovery,
            "valid_stop_targets_depth_2_to_4": valid_stop_occurrences,
            "exact_realization_precision_over_particle_occurrences": {
                "numerator": exact_precision_numerator,
                "denominator": exact_precision_denominator,
            },
            "abstentions_over_particle_calls": aggregate_abstentions,
            "budgets_over_eight_independent_calls": aggregate_budget_usage,
            "proposal_diversity": _union_diversity(particle_batches, union_rows),
            "union_proposal_set_sha256": identity(union_signature),
        },
        "comparison": {
            "attempt_1_path": _relative(PRIOR_RESULT),
            "attempt_1_sha256": _sha256(PRIOR_RESULT),
            "attempt_1_payload_sha256": prior_payload_sha256,
            "attempt_1_autonomous_exact_recovery": prior_result["autonomous_joint"][
                "exact_endpoint_recovery_numerator"
            ],
            "attempt_1_autonomous_transformation_recovery": prior_result[
                "autonomous_joint"
            ]["transformation_equivalent_recovery_numerator"],
        },
        "gate": {
            "all_teacher_constituents_bound_every_particle": all(
                len(rows) == len(PARTICLE_INDICES) for rows in bound_particles.values()
            ),
            "all_teacher_constituents_selected_across_particles": all(
                selected_particles.values()
            ),
            "every_route_has_a_particle_with_all_three_constituents": all(
                route_particles.values()
            ),
            "exact_recovery_all_routes": all(
                row["exact_endpoint_recovered"] for row in union_recovery
            ),
            "transformation_recovery_all_routes": all(
                row["transformation_equivalent_recovered"] for row in union_recovery
            ),
            "exact_realization_precision_one": (
                exact_precision_numerator == exact_precision_denominator
            ),
            "fixed_default_budgets_every_particle": True,
            "zero_teacher_primitive_actions": True,
            "zero_oracle_execution": True,
        },
        "limitations": [
            "The three known endpoints are retrospective Full-146 development routes.",
            "Particle union increases total work through eight independent fixed-budget calls.",
            "Teacher identity joins are post-hoc support diagnostics, not runtime inputs.",
            "Operational wall time is machine-dependent and stored in a separate sidecar.",
            "No docking value, eligibility gain or prospective optimization claim is tested.",
        ],
        "material_inputs_sha256": _material_inputs(),
        "material_payload_sha256": {
            _relative(CHECKPOINT): checkpoint_payload_sha256,
            _relative(FORENSICS): route_context["forensic_payload_sha256"],
            _relative(PRIOR_RESULT): prior_payload_sha256,
        },
        "implementation_inputs_sha256": {
            path: _sha256(ROOT / path) for path in CODE_INPUTS
        },
    }
    posthoc_seconds = perf_counter() - posthoc_started
    runtime_payload = {
        "schema_version": RUNTIME_SCHEMA,
        "evidence": "operational non-deterministic wall-clock telemetry",
        "result_payload_sha256": identity(payload),
        "clock": "time.perf_counter",
        "particle_runtimes": particle_runtimes,
        "union_and_posthoc_analysis_wall_seconds": posthoc_seconds,
        "end_to_end_wall_seconds": perf_counter() - overall_started,
        "deterministic_scientific_result": False,
    }
    return payload, runtime_payload


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite particle-gate artifact: {path}")
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    runtime_output = args.runtime_output.resolve()
    if output == runtime_output:
        raise ValueError("result and runtime outputs must differ")
    if output.exists() or runtime_output.exists():
        raise ValueError("refusing to overwrite particle-gate artifacts")
    payload, runtime_payload = run()
    _publish(output, payload)
    _publish(runtime_output, runtime_payload)
    print(
        json.dumps(
            {
                "result": {"path": str(output), "payload_sha256": identity(payload)},
                "runtime": {
                    "path": str(runtime_output),
                    "payload_sha256": identity(runtime_payload),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
