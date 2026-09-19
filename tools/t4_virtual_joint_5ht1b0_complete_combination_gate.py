"""Run the bounded 5HT1B-0 depth-three complete-combination gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import rdkit

from compose_v4.chem.molecular_graph import is_rdkit_valid
from compose_v4.chem.state import is_connected_or_null
from compose_v4.control.docking_value import identity
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    propose_virtual_joint_region_paths,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_virtual_joint_5ht1b0_particle3_combination_census import (
    DIAGNOSTIC_CODE_INPUTS,
    IMPLEMENTATION_CODE_INPUTS,
)
from tools.t4_virtual_joint_5ht1b0_particle3_combination_census import (
    _material_inputs as _census_material_inputs,
)
from tools.t4_virtual_joint_5ht1b0_particle_gate import (
    EXPECTED_BUDGETS,
    _abstentions,
    _budget_accounting,
    _compact_bound_census,
    _diversity,
    _recovery,
    _sealed,
)
from tools.t4_virtual_joint_5ht1b0_support_gate import (
    CHECKPOINT,
    ROOT,
    SOURCE_GROUP,
    _load_expert,
    _route_inputs,
    _sha256,
)

SCHEMA = "t4_virtual_joint_5ht1b0_complete_combination_gate_v1"
RUNTIME_SCHEMA = "t4_virtual_joint_5ht1b0_complete_combination_runtime_v1"
PROPOSER_COMMIT = "2a3062f7a918c5debf7bbaa1f28a28db8a00ee22"
CENSUS_COMMIT = "ad52c0bf92ec7b97b8ac324586b5389dc4b44b45"
PROPOSER_PATH = "src/compose_v4/control/virtual_joint_region_proposer.py"
CENSUS_PATH = (
    "diagnostics/t4_virtual_joint_5ht1b0_support_gate/"
    "particle_3_depth3_combination_census_v1/result.json"
)
CENSUS_RESULT = ROOT / CENSUS_PATH
ATTEMPT_2 = (
    ROOT / "diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_2/result.json"
)
SOURCE_ARTIFACT = (
    ROOT / "diagnostics/t4_shared_program_controller/attempt_2/5ht1b_0.json"
)
BINDING_PARTICLE_INDEX = 3
DEPTH = 3
ALLOCATION = "template_binding_particle"
PLANNER = "complete_combination_particle"
EXPECTED_PARTICLE_COUNT = 5


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _git_blob_sha256(commit: str, path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def _source_input() -> tuple[Any, str]:
    """Load only the exact benchmark source, without teacher route identities."""

    artifact = json.loads(SOURCE_ARTIFACT.read_text())
    if artifact.get("source_group") != SOURCE_GROUP:
        raise ValueError("source artifact group changed")
    encoded_sources = [
        row.get("source_state") for row in artifact.get("candidates", [])
    ]
    if not encoded_sources or any(not isinstance(row, dict) for row in encoded_sources):
        raise ValueError("source artifact has no exact candidate source states")
    sources = [decode_state(row) for row in encoded_sources]
    source_keys = {canonical_state_key(row) for row in sources}
    if len(source_keys) != 1:
        raise ValueError("source artifact does not contain one exact source")
    source = sources[0]
    return source, identity(encode_state(source))


def _required_particle_count(budgets: VirtualJointRegionBudgets) -> tuple[int, int]:
    total = math.comb(budgets.expansion_width, DEPTH)
    return total, max(1, math.ceil(total / budgets.max_planning_expansions))


def _run_shard(particle_index: int) -> dict[str, Any]:
    """Run one task-blind proposal shard in an isolated local process."""

    budgets = VirtualJointRegionBudgets()
    source, source_state_sha256 = _source_input()
    expert, checkpoint_payload_sha256 = _load_expert()
    started = perf_counter()
    batch = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=budgets,
        constituent_allocation=ALLOCATION,
        binding_particle_index=BINDING_PARTICLE_INDEX,
        combination_planner=PLANNER,
        combination_particle_depth=DEPTH,
        combination_particle_index=particle_index,
        combination_particle_count=EXPECTED_PARTICLE_COUNT,
    )
    return {
        "particle_index": particle_index,
        "batch": batch,
        "source_state_sha256": source_state_sha256,
        "checkpoint_payload_sha256": checkpoint_payload_sha256,
        "proposal_wall_seconds": perf_counter() - started,
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


def _deduplicate_union(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_endpoint: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for particle_rank, proposal in enumerate(shard["batch"].proposals, 1):
            row = {
                "particle_index": shard["particle_index"],
                "particle_rank": particle_rank,
                "proposal": proposal,
            }
            previous = by_endpoint.get(proposal.endpoint_key)
            if previous is None or _union_rank_key(row) < _union_rank_key(previous):
                by_endpoint[proposal.endpoint_key] = row
    return sorted(by_endpoint.values(), key=_union_rank_key)


def _capacity_abstention_reasons(telemetry: dict[str, Any]) -> list[str]:
    reasons = []
    flag_names = (
        "combination_particle_planning_capacity_abstention",
        "combination_particle_target_capacity_abstention",
        "realization_attempt_budget_exhausted",
        "realization_expansion_budget_exhausted",
    )
    reasons.extend(name for name in flag_names if telemetry.get(name, 0))
    if telemetry.get("combination_particle_unvalidated_combinations", 0):
        reasons.append("unvalidated_combinations")
    if telemetry.get("combination_particle_unretained_valid_targets", 0):
        reasons.append("unretained_valid_targets")
    if telemetry.get("combination_particle_unattempted_targets", 0):
        reasons.append("unattempted_targets")
    if telemetry.get("per_realization_expansion_cap_abstentions", 0):
        reasons.append("per_realization_expansion_cap_abstentions")
    return sorted(set(reasons))


def _endpoint_validity(source: Any, proposals: list[Any]) -> dict[str, int]:
    valid = sum(
        proposal.endpoint.n_atoms == 48
        and proposal.endpoint.n_real_atoms <= 40
        and is_rdkit_valid(proposal.endpoint)
        and is_connected_or_null(proposal.endpoint)
        and charge_policy_preserved(source, proposal.endpoint)
        and canonical_state_key(proposal.endpoint) == proposal.endpoint_key
        for proposal in proposals
    )
    return {"numerator": valid, "denominator": len(proposals)}


def _primitive_distribution(proposals: list[Any]) -> dict[str, Any]:
    counts = Counter(row.primitive_count for row in proposals)
    values = [row.primitive_count for row in proposals]
    return {
        "distribution": {str(key): counts[key] for key in sorted(counts)},
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "total": sum(values),
    }


def _shard_report(shard: dict[str, Any], budgets: VirtualJointRegionBudgets) -> dict:
    batch = shard["batch"]
    telemetry = batch.telemetry
    compact_census = _compact_bound_census(telemetry)
    reasons = _capacity_abstention_reasons(telemetry)
    return {
        "particle_index": shard["particle_index"],
        "combination_range": {
            "start_inclusive": int(telemetry["combination_particle_shard_start"]),
            "stop_exclusive": int(telemetry["combination_particle_shard_stop"]),
            "assigned": int(telemetry["combination_particle_assigned_combinations"]),
        },
        "valid_stop_occurrences": int(telemetry["valid_stop_targets_depth_3"]),
        "unique_valid_targets": int(
            telemetry["combination_particle_unique_valid_targets"]
        ),
        "committed_proposals": len(batch.proposals),
        "exact_realization_precision": {
            "numerator": int(telemetry["exact_realization_precision_numerator"]),
            "denominator": int(telemetry["exact_realization_precision_denominator"]),
        },
        "realization_success_yield": {
            "numerator": len(batch.proposals),
            "denominator": int(telemetry["realization_attempts"]),
        },
        "endpoint_validity": _endpoint_validity(
            _source_input()[0], list(batch.proposals)
        ),
        "budgets": _budget_accounting(telemetry, budgets),
        "capacity_abstention": bool(reasons),
        "capacity_abstention_reasons": reasons,
        "abstentions": _abstentions(telemetry),
        "primitive_counts": _primitive_distribution(list(batch.proposals)),
        "proposal_diversity": _diversity(list(batch.proposals)),
        "bound_constituent_census_sha256": identity(compact_census),
        "sanitized_binding_rows": all(
            "binding" not in row for row in telemetry["bound_constituent_ranks"]
        ),
    }


def _proposal_signature(union_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "endpoint_sha256": identity(row["proposal"].endpoint_key),
            "particle_index": row["particle_index"],
            "particle_rank": row["particle_rank"],
            "program_id": row["proposal"].program_id,
            "constituent_keys": [
                step.constituent_key for step in row["proposal"].steps
            ],
            "actions_sha256": identity(list(row["proposal"].actions)),
            "primitive_count": row["proposal"].primitive_count,
        }
        for row in union_rows
    ]


def run(*, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    overall_started = perf_counter()
    budgets = VirtualJointRegionBudgets()
    if asdict(budgets) != EXPECTED_BUDGETS:
        raise RuntimeError("default VirtualJointRegionBudgets changed")
    total_combinations, particle_count = _required_particle_count(budgets)
    if particle_count != EXPECTED_PARTICLE_COUNT or total_combinations != 17_296:
        raise RuntimeError("derived complete-combination particle census changed")
    if _sha256(ROOT / PROPOSER_PATH) != _git_blob_sha256(
        PROPOSER_COMMIT, PROPOSER_PATH
    ):
        raise RuntimeError("loaded proposer does not match committed proposer")
    if _sha256(CENSUS_RESULT) != _git_blob_sha256(CENSUS_COMMIT, CENSUS_PATH):
        raise RuntimeError("census artifact does not match committed census")
    census, census_payload_sha256 = _sealed(CENSUS_RESULT)
    if census["census"]["combination_count"] != total_combinations:
        raise RuntimeError("committed census combination count changed")

    proposal_started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        shards = list(executor.map(_run_shard, range(particle_count)))
    proposal_seconds = perf_counter() - proposal_started
    shards.sort(key=lambda row: row["particle_index"])
    if [row["particle_index"] for row in shards] != list(range(particle_count)):
        raise RuntimeError("a declared complete-combination shard is missing")
    source_hashes = {row["source_state_sha256"] for row in shards}
    checkpoint_hashes = {row["checkpoint_payload_sha256"] for row in shards}
    if len(source_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise RuntimeError("proposal shards disagree on source or checkpoint identity")
    for shard in shards:
        telemetry = shard["batch"].telemetry
        if (
            telemetry["task_cell_route_or_endpoint_input_used"]
            or telemetry["primitive_teacher_actions_used"] != 0
            or telemetry["combination_particle_count"] != particle_count
            or telemetry["combination_particle_depth"] != DEPTH
            or telemetry["binding_particle_index"] != BINDING_PARTICLE_INDEX
        ):
            raise RuntimeError("proposal shard violated frozen task-blind inputs")

    posthoc_started = perf_counter()
    routes, route_context = _route_inputs()
    if route_context["source_state_sha256"] != next(iter(source_hashes)):
        raise RuntimeError("post-hoc teacher source differs from proposal source")
    source, _ = _source_input()
    union_rows = _deduplicate_union(shards)
    union_proposals = [row["proposal"] for row in union_rows]
    shard_reports = [_shard_report(row, budgets) for row in shards]
    recovery = _recovery(routes, union_proposals)
    raw_ranges = [report["combination_range"] for report in shard_reports]
    range_coverage = (
        raw_ranges[0]["start_inclusive"] == 0
        and raw_ranges[-1]["stop_exclusive"] == total_combinations
        and all(
            left["stop_exclusive"] == right["start_inclusive"]
            for left, right in pairwise(raw_ranges)
        )
        and sum(row["assigned"] for row in raw_ranges) == total_combinations
    )
    all_occurrences = [
        proposal for shard in shards for proposal in shard["batch"].proposals
    ]
    exact_numerator = sum(
        report["exact_realization_precision"]["numerator"] for report in shard_reports
    )
    exact_denominator = sum(
        report["exact_realization_precision"]["denominator"] for report in shard_reports
    )
    endpoint_validity = _endpoint_validity(source, union_proposals)
    capacity_abstentions = [
        {
            "particle_index": report["particle_index"],
            "reasons": report["capacity_abstention_reasons"],
        }
        for report in shard_reports
        if report["capacity_abstention"]
    ]
    unique_programs = len({row.program_id for row in union_proposals})
    union_signature = _proposal_signature(union_rows)
    posthoc_seconds = perf_counter() - posthoc_started

    proposer_inputs = {
        path: _sha256(ROOT / path) for path in IMPLEMENTATION_CODE_INPUTS
    }
    payload = {
        "schema_version": SCHEMA,
        "scientific_problem": (
            "test whether complete bounded enumeration of binding particle 3 "
            "recovers valid executable 5HT1B-0 depth-three combinations"
        ),
        "primary_model_output": (
            "deterministically unioned exact complete programs and endpoints from "
            "five task-blind complete-combination proposal shards"
        ),
        "central_claim_under_test": (
            "the attempt_2 miss was caused by incremental marginal-beam pruning, "
            "and the unchanged task-blind support can recover the known combinations "
            "within five unchanged-budget complete-combination particles"
        ),
        "validation_setting": (
            "retrospective zero-oracle 5HT1B-0 binding-particle-3 support gate"
        ),
        "baseline": (
            "sealed attempt_2 incremental beam and committed exhaustive census"
        ),
        "declared_support": (
            "exact 48-slot, at most 40-active-atom, charge-preserving graphs; "
            "binding particle 3; depth 3; at most 32 realized primitives"
        ),
        "evidence": "computed zero-oracle complete-combination particle gate",
        "claim_boundary": (
            "teacher identities are joined only after all five proposal calls; "
            "this measures retrospective structural support, not docking utility"
        ),
        "code_revision": _revision(),
        "proposer_commit": PROPOSER_COMMIT,
        "census_commit": CENSUS_COMMIT,
        "configuration": {
            "binding_particle_index": BINDING_PARTICLE_INDEX,
            "constituent_allocation": ALLOCATION,
            "combination_planner": PLANNER,
            "combination_depth": DEPTH,
            "selected_constituents": budgets.expansion_width,
            "total_combinations_formula": "C(48,3)",
            "total_combinations": total_combinations,
            "particle_count_formula": "ceil(C(48,3)/4096)",
            "derived_minimum_particle_count": particle_count,
            "particle_indices": list(range(particle_count)),
            "joint_budgets_per_shard": asdict(budgets),
            "budget_widening": False,
            "global_marginal_truncation": False,
            "union_deduplication": (
                "canonical endpoint; retain minimum deterministic proposal rank key"
            ),
            "transformation_equivalence_radius": 2,
            "proposal_worker_processes": workers,
        },
        "source": {
            "source_group": SOURCE_GROUP,
            "source_state_sha256": next(iter(source_hashes)),
            "source_artifact": _relative(SOURCE_ARTIFACT),
        },
        "shards": shard_reports,
        "union": {
            "raw_combination_range_coverage": range_coverage,
            "valid_stop_occurrences": sum(
                row["valid_stop_occurrences"] for row in shard_reports
            ),
            "unique_valid_targets_before_realization": sum(
                row["unique_valid_targets"] for row in shard_reports
            ),
            "total_committed_proposal_occurrences": len(all_occurrences),
            "duplicate_committed_endpoint_occurrences": len(all_occurrences)
            - len(union_proposals),
            "unique_endpoints": len(union_proposals),
            "unique_programs": unique_programs,
            "proposal_diversity": _diversity(union_proposals),
            "primitive_counts": _primitive_distribution(union_proposals),
            "endpoint_validity": endpoint_validity,
            "exact_realization_precision": {
                "numerator": exact_numerator,
                "denominator": exact_denominator,
            },
            "capacity_abstentions": capacity_abstentions,
            "teacher_recovery_posthoc": recovery,
            "union_proposal_set_sha256": identity(union_signature),
        },
        "gate": {
            "all_declared_shards_completed": len(shards) == particle_count,
            "complete_raw_combination_coverage": range_coverage,
            "no_shard_capacity_abstention": not capacity_abstentions,
            "exact_realization_precision_one": exact_numerator == exact_denominator,
            "all_committed_endpoints_valid": endpoint_validity["numerator"]
            == endpoint_validity["denominator"],
            "exact_endpoint_recovery_all_three": all(
                row["exact_endpoint_recovered"] for row in recovery
            ),
            "radius_2_transformation_recovery_all_three": all(
                row["transformation_equivalent_recovered"] for row in recovery
            ),
            "zero_teacher_actions_in_proposals": True,
            "zero_oracle_execution": True,
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
        "determinism": {
            "proposal_rng": "none",
            "seed": None,
            "shard_order": list(range(particle_count)),
            "serialization": "sorted-key compact JSON with SHA-256 payload identity",
            "wall_runtime_excluded_from_deterministic_result": True,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rdkit": rdkit.__version__,
            "platform_system": platform.system(),
            "machine": platform.machine(),
            "precision": "native CPU graph execution; no learned numeric inference",
        },
        "limitations": [
            "The three teacher identities are retrospective Full-146 development evidence.",
            "This gate measures structural support and exact execution, not docking value.",
            "Only binding particle 3 and depth-three combinations are in scope.",
            "Five independent calls increase total work without widening a per-call budget.",
        ],
        "material_inputs_sha256": {
            **_census_material_inputs(),
            _relative(SOURCE_ARTIFACT): _sha256(SOURCE_ARTIFACT),
            _relative(CENSUS_RESULT): _sha256(CENSUS_RESULT),
            _relative(ATTEMPT_2): _sha256(ATTEMPT_2),
        },
        "material_payload_sha256": {
            _relative(CHECKPOINT): next(iter(checkpoint_hashes)),
            _relative(CENSUS_RESULT): census_payload_sha256,
            "posthoc_forensics": route_context["forensic_payload_sha256"],
        },
        "implementation_inputs_sha256": {
            **proposer_inputs,
            "committed_proposer_blob": _git_blob_sha256(PROPOSER_COMMIT, PROPOSER_PATH),
            "committed_census_blob": _git_blob_sha256(CENSUS_COMMIT, CENSUS_PATH),
            **{path: _sha256(ROOT / path) for path in DIAGNOSTIC_CODE_INPUTS},
            _relative(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        },
    }
    payload["gate"]["passed"] = all(payload["gate"].values())
    runtime_payload = {
        "schema_version": RUNTIME_SCHEMA,
        "evidence": "non-authoritative operational wall-clock telemetry",
        "result_payload_sha256": identity(payload),
        "clock": "time.perf_counter",
        "worker_processes": workers,
        "shards": [
            {
                "particle_index": row["particle_index"],
                "proposal_wall_seconds": row["proposal_wall_seconds"],
            }
            for row in shards
        ],
        "parallel_proposal_wall_seconds": proposal_seconds,
        "union_and_posthoc_analysis_wall_seconds": posthoc_seconds,
        "end_to_end_wall_seconds": perf_counter() - overall_started,
        "deterministic_scientific_result": False,
    }
    return payload, runtime_payload


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite gate artifact: {path}")
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=EXPECTED_PARTICLE_COUNT)
    args = parser.parse_args()
    if not 1 <= args.workers <= EXPECTED_PARTICLE_COUNT:
        raise ValueError("workers must be within 1..5")
    payload, runtime = run(workers=args.workers)
    _publish(args.output_directory / "result.json", payload)
    _publish(args.output_directory / "runtime.json", runtime)
    print(
        json.dumps(
            {
                "gate_passed": payload["gate"]["passed"],
                "output_directory": str(args.output_directory),
                "payload_sha256": identity(payload),
                "runtime_payload_sha256": identity(runtime),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
