"""Deterministic complete-combination particles for the route proposal lane.

This module is an additive producer for the existing ``route_complete_region``
expert.  It does not define a fourth proposal expert and it consumes no
controller random state.  Every parent receives the same fixed schedule: four
binding particles, complete depths one through three, and disjoint combination
shards containing at most 4,096 raw combinations.

Particle receipts are reduced all-or-nothing.  A missing, corrupt, or
operationally failed required receipt causes particle augmentation to abstain
and returns an unchanged copy of the independently valid legacy route pool.
Explicit scientific abstentions inside structurally complete receipts remain
reported and do not invalidate exact committed endpoints.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal_policy import (
    structural_rewrite_event_count,
)
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    VirtualJointRegionProposal,
    propose_virtual_joint_region_paths,
)
from compose_v4.rewrite.trace_shard import encode_state

SCHEMA_VERSION = "route_complete_region_particles_v1"
JOB_SCHEMA_VERSION = "route_complete_region_particle_job_v1"
RECEIPT_SCHEMA_VERSION = "route_complete_region_particle_receipt_v1"
REDUCTION_SCHEMA_VERSION = "route_complete_region_particle_reduction_v1"

ROUTE_EXPERT = "route_complete_region"
SELECTED_CONSTITUENT_COUNT = 48
MAX_COMBINATIONS_PER_JOB = 4_096
BINDING_PARTICLE_INDICES = tuple(range(4))
SUPPORTED_COMPLETE_DEPTHS = (1, 2, 3)
DEPTH_4_COMPLETE_COVERAGE = False
COMBINATION_COUNTS = {
    depth: math.comb(SELECTED_CONSTITUENT_COUNT, depth) for depth in range(1, 5)
}
REQUIRED_SHARD_COUNTS = {
    depth: math.ceil(count / MAX_COMBINATIONS_PER_JOB)
    for depth, count in COMBINATION_COUNTS.items()
}
SCHEDULED_SHARD_COUNTS = {1: 1, 2: 1, 3: 5}

_COMPLETE_STATUSES = {
    "complete",
    "complete_with_realization_abstentions",
    "empty_supported_depth",
    "planning_capacity_abstention",
    "target_capacity_abstention",
}
_OPERATIONAL_FAILURE_STATUSES = {"failed", "missing_at_deadline"}
PARTICLE_RECEIPT_STATUSES = _COMPLETE_STATUSES | _OPERATIONAL_FAILURE_STATUSES

_SAFE_TELEMETRY_EXACT = {
    "combination_planner_mode",
    "combination_particle_total_combinations",
    "combination_particle_shard_start",
    "combination_particle_shard_stop",
    "combination_particle_assigned_combinations",
    "combination_particle_required_minimum_count",
    "combination_particle_raw_union_covers_requested_depth",
    "combination_particle_coverage_requires_all_particles",
    "combination_particle_telemetry_sanitized",
    "combination_particle_empty_depth_abstention",
    "combination_particle_planning_capacity_abstention",
    "combination_particle_unvalidated_combinations",
    "combination_particle_target_capacity_abstention",
    "combination_particle_unretained_valid_targets",
    "combination_particle_unique_valid_targets",
    "combination_particle_target_capacity",
    "combination_particle_realization_coverage_numerator",
    "combination_particle_realization_coverage_denominator",
    "combination_particle_unattempted_targets",
    "planning_expansions",
    "planning_expansion_budget_exhausted",
    "stop_target_attempts",
    "final_target_abstentions",
    "canonical_target_aliases",
    "realization_attempts",
    "realization_attempt_budget_exhausted",
    "realization_expansion_budget_exhausted",
    "realizer_expansions",
    "successful_realized_targets",
    "compiler_abstentions",
    "per_realization_expansion_cap_abstentions",
    "realized_primitives",
    "unique_valid_targets",
    "unique_committed_endpoints",
    "exact_realization_precision_numerator",
    "exact_realization_precision_denominator",
    "max_depth",
    "max_planning_expansions",
    "max_targets",
    "max_realization_attempts",
    "max_realization_expansions",
    "maximum_expansions_per_realization",
    "maximum_primitives",
    "virtual_prefix_commits",
    "partial_endpoint_evaluations",
}
_SAFE_TELEMETRY_PATTERN = re.compile(
    r"^(?:planning_prefixes|invalid_stop_targets|valid_stop_targets|"
    r"retained_targets)_depth_[1-4]$|^realization_status:[a-z0-9_:-]+$"
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CompleteCombinationJobSpec:
    """One immutable, target-independent combination-particle job."""

    binding_particle: int
    depth: int
    shard: int
    shard_count: int
    total_combinations: int
    shard_start: int
    shard_stop: int
    max_combinations_per_job: int = MAX_COMBINATIONS_PER_JOB
    schema_version: str = JOB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOB_SCHEMA_VERSION:
            raise ValueError("complete-combination job schema changed")
        if self.binding_particle not in BINDING_PARTICLE_INDICES:
            raise ValueError("complete-combination binding particle is outside 0..3")
        if self.depth not in SUPPORTED_COMPLETE_DEPTHS:
            raise ValueError("complete-combination jobs support depths 1..3 only")
        expected_shards = SCHEDULED_SHARD_COUNTS[self.depth]
        expected_total = COMBINATION_COUNTS[self.depth]
        if (
            self.shard_count != expected_shards
            or self.total_combinations != expected_total
        ):
            raise ValueError("complete-combination job changed its frozen census")
        if type(self.shard) is not int or not 0 <= self.shard < self.shard_count:
            raise ValueError("complete-combination shard index is out of range")
        expected_start = expected_total * self.shard // self.shard_count
        expected_stop = expected_total * (self.shard + 1) // self.shard_count
        if (self.shard_start, self.shard_stop) != (expected_start, expected_stop):
            raise ValueError("complete-combination job changed its disjoint range")
        if self.max_combinations_per_job != MAX_COMBINATIONS_PER_JOB:
            raise ValueError("complete-combination per-job capacity changed")
        if self.shard_stop - self.shard_start > self.max_combinations_per_job:
            raise ValueError("complete-combination job exceeds its frozen capacity")

    @property
    def job_id(self) -> str:
        return (
            f"route-complete-region-b{self.binding_particle}-d{self.depth}-"
            f"s{self.shard:02d}-of-{self.shard_count:02d}"
        )

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "job_id": self.job_id}


def frozen_particle_budgets() -> VirtualJointRegionBudgets:
    """Return the explicit production-aligned, per-job work limits."""

    return VirtualJointRegionBudgets(
        max_depth=4,
        beam_width=64,
        expansion_width=SELECTED_CONSTITUENT_COUNT,
        max_bindings_per_template=len(BINDING_PARTICLE_INDICES),
        max_binding_visits=16_384,
        max_planning_expansions=MAX_COMBINATIONS_PER_JOB,
        max_targets=96,
        max_realization_attempts=96,
        max_realization_expansions=65_536,
        maximum_expansions_per_realization=4_000,
        maximum_primitives=32,
    )


def complete_combination_job_specs() -> tuple[CompleteCombinationJobSpec, ...]:
    """Return the same 28 jobs for every parent, target, cell, and delta."""

    jobs = []
    for binding_particle in BINDING_PARTICLE_INDICES:
        for depth in SUPPORTED_COMPLETE_DEPTHS:
            shard_count = SCHEDULED_SHARD_COUNTS[depth]
            total = COMBINATION_COUNTS[depth]
            for shard in range(shard_count):
                jobs.append(
                    CompleteCombinationJobSpec(
                        binding_particle=binding_particle,
                        depth=depth,
                        shard=shard,
                        shard_count=shard_count,
                        total_combinations=total,
                        shard_start=total * shard // shard_count,
                        shard_stop=total * (shard + 1) // shard_count,
                    )
                )
    return tuple(jobs)


def _rewrite_scale(events: int) -> str:
    if events <= 3:
        return "local"
    if events <= 11:
        return "medium"
    return "large"


def _primitive_band(primitives: int) -> str:
    if primitives <= 3:
        return "small"
    if primitives <= 11:
        return "medium"
    return "large"


def _particle_origin(
    proposal: VirtualJointRegionProposal,
    spec: CompleteCombinationJobSpec,
    local_rank: int,
) -> dict[str, Any]:
    return {
        "job_id": spec.job_id,
        "binding_particle": spec.binding_particle,
        "depth": spec.depth,
        "shard": spec.shard,
        "shard_count": spec.shard_count,
        "local_rank": local_rank,
        "program_id": proposal.program_id,
    }


def route_record_from_virtual_proposal(
    proposal: VirtualJointRegionProposal,
    spec: CompleteCombinationJobSpec,
    local_rank: int,
) -> dict[str, Any]:
    """Map one committed virtual proposal onto the legacy route-row schema."""

    if type(local_rank) is not int or local_rank < 1:
        raise ValueError("particle-local route rank must be a positive integer")
    if proposal.depth != spec.depth:
        raise ValueError("virtual proposal depth differs from its particle job")
    smiles = molecular_graph_to_smiles(proposal.endpoint)
    if not smiles:
        raise ValueError("virtual proposal endpoint has no canonical SMILES")
    rewrite_events = sum(
        structural_rewrite_event_count(template) for template in proposal.templates
    )
    primitive_count = int(proposal.primitive_count)
    if not 0 <= primitive_count <= frozen_particle_budgets().maximum_primitives:
        raise ValueError("virtual proposal primitive count is outside support")
    origin = _particle_origin(proposal, spec, local_rank)
    return {
        "smiles": smiles,
        "proposal_lane": ROUTE_EXPERT,
        "proposal_experts": [ROUTE_EXPERT],
        "families": [ROUTE_EXPERT],
        "program_families": [ROUTE_EXPERT],
        "regions": proposal.depth,
        "created": sum(len(step.patch.output_atoms) for step in proposal.steps),
        "deleted": sum(
            atom is None for step in proposal.steps for atom in step.patch.target_atoms
        ),
        "route_prior_score": float(proposal.log_probability),
        "route_proposal_rank": local_rank,
        "route_program_id": proposal.program_id,
        "route_template_ids": [step.template_id for step in proposal.steps],
        "rewrite_events": rewrite_events,
        "rewrite_scale": _rewrite_scale(rewrite_events),
        "realized_primitives": primitive_count,
        "realized_primitive_band": _primitive_band(primitive_count),
        "compiler_strategy": proposal.compiler_strategy,
        "route_generation_modes": ["complete_combination_particle"],
        "route_particle_local_rank": local_rank,
        "route_particle_origins": [origin],
    }


def _safe_scalar(value: Any) -> bool:
    return value is None or type(value) in {bool, int, float, str}


def sanitize_particle_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only aggregate scalar work, abstention, and precision telemetry.

    Raw bindings, slots, actions, source states, endpoints, task/cell identities,
    and teacher information are intentionally not recursively copied.
    """

    if not isinstance(telemetry, Mapping):
        raise TypeError("particle telemetry must be a mapping")
    sanitized: dict[str, Any] = {}
    for raw_key, value in telemetry.items():
        key = str(raw_key)
        if key not in _SAFE_TELEMETRY_EXACT and not _SAFE_TELEMETRY_PATTERN.fullmatch(
            key
        ):
            continue
        if not _safe_scalar(value):
            raise ValueError(f"allowed particle telemetry field {key!r} is not scalar")
        sanitized[key] = value
    return dict(sorted(sanitized.items()))


def _receipt_status(telemetry: Mapping[str, Any]) -> str:
    if telemetry.get("combination_particle_empty_depth_abstention"):
        return "empty_supported_depth"
    if telemetry.get("combination_particle_planning_capacity_abstention"):
        return "planning_capacity_abstention"
    if telemetry.get("combination_particle_target_capacity_abstention"):
        return "target_capacity_abstention"
    realization_abstention_fields = (
        "compiler_abstentions",
        "combination_particle_unattempted_targets",
        "realization_attempt_budget_exhausted",
        "realization_expansion_budget_exhausted",
        "per_realization_expansion_cap_abstentions",
    )
    if any(telemetry.get(field) for field in realization_abstention_fields):
        return "complete_with_realization_abstentions"
    return "complete"


def _candidate_set_sha256(records: list[dict[str, Any]]) -> str:
    return identity(records)


def run_complete_combination_job(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    spec: CompleteCombinationJobSpec,
) -> dict[str, Any]:
    """Run one exact, task-independent particle and return a self-hashed receipt."""

    expected = {row.job_id: row for row in complete_combination_job_specs()}
    if (
        not isinstance(spec, CompleteCombinationJobSpec)
        or expected.get(spec.job_id) != spec
    ):
        raise ValueError("particle job is not in the frozen 28-job schedule")
    budgets = frozen_particle_budgets()
    batch = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=budgets,
        constituent_allocation="template_binding_particle",
        binding_particle_index=spec.binding_particle,
        combination_planner="complete_combination_particle",
        combination_particle_depth=spec.depth,
        combination_particle_index=spec.shard,
        combination_particle_count=spec.shard_count,
    )
    raw = batch.telemetry
    safety_values = (
        raw.get("task_cell_route_or_endpoint_input_used"),
        raw.get("primitive_teacher_actions_used"),
        raw.get("virtual_prefix_commits"),
        raw.get("partial_endpoint_evaluations"),
    )
    if safety_values != (False, 0, 0, 0):
        raise RuntimeError("particle proposal violated its task-blind commit boundary")
    expected_identity = (
        raw.get("binding_particle_index"),
        raw.get("combination_particle_depth"),
        raw.get("combination_particle_index"),
        raw.get("combination_particle_count"),
    )
    if expected_identity != (
        spec.binding_particle,
        spec.depth,
        spec.shard,
        spec.shard_count,
    ):
        raise RuntimeError("particle proposer returned telemetry for another job")
    records = [
        route_record_from_virtual_proposal(proposal, spec, rank)
        for rank, proposal in enumerate(batch.proposals, 1)
    ]
    sanitized = sanitize_particle_telemetry(raw)
    sanitized.update(
        {
            "candidate_count": len(records),
            "candidate_set_sha256": _candidate_set_sha256(records),
            "source_state_sha256": identity(encode_state(source)),
            "expert_training_identity_sha256": expert.training_identity,
        }
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "job": spec.payload(),
        "status": _receipt_status(raw),
        "records": records,
        "telemetry": dict(sorted(sanitized.items())),
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None


def _validate_route_record(record: Any, spec: CompleteCombinationJobSpec) -> bool:
    if not isinstance(record, dict):
        return False
    if (
        record.get("proposal_lane") != ROUTE_EXPERT
        or record.get("proposal_experts") != [ROUTE_EXPERT]
        or record.get("families") != [ROUTE_EXPERT]
        or record.get("program_families") != [ROUTE_EXPERT]
        or record.get("route_generation_modes") != ["complete_combination_particle"]
        or record.get("regions") != spec.depth
    ):
        return False
    smiles = record.get("smiles")
    origins = record.get("route_particle_origins")
    if not isinstance(smiles, str) or not smiles or not isinstance(origins, list):
        return False
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or Chem.MolToSmiles(molecule) != smiles:
        return False
    if len(origins) != 1 or not isinstance(origins[0], dict):
        return False
    origin = origins[0]
    if set(origin) != {
        "job_id",
        "binding_particle",
        "depth",
        "shard",
        "shard_count",
        "local_rank",
        "program_id",
    }:
        return False
    score = record.get("route_prior_score")
    local_rank = record.get("route_particle_local_rank")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or type(local_rank) is not int
        or local_rank < 1
        or not isinstance(record.get("route_program_id"), str)
        or not record["route_program_id"]
        or not isinstance(record.get("route_template_ids"), list)
        or any(
            not isinstance(value, str) or not value
            for value in record["route_template_ids"]
        )
    ):
        return False
    return (
        origin.get("job_id") == spec.job_id
        and origin.get("binding_particle") == spec.binding_particle
        and origin.get("depth") == spec.depth
        and origin.get("shard") == spec.shard
        and origin.get("shard_count") == spec.shard_count
        and origin.get("local_rank") == record.get("route_particle_local_rank")
        and origin.get("program_id") == record.get("route_program_id")
    )


def _validate_receipt(
    receipt: Any,
    expected: Mapping[str, CompleteCombinationJobSpec],
) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(receipt, dict) or set(receipt) != {"payload", "payload_sha256"}:
        return None, None
    payload = receipt.get("payload")
    if (
        not isinstance(payload, dict)
        or not _valid_hash(receipt.get("payload_sha256"))
        or identity(payload) != receipt["payload_sha256"]
        or set(payload) != {"schema_version", "job", "status", "records", "telemetry"}
        or payload.get("schema_version") != RECEIPT_SCHEMA_VERSION
    ):
        return None, None
    raw_job = payload.get("job")
    job_id = raw_job.get("job_id") if isinstance(raw_job, dict) else None
    spec = expected.get(str(job_id))
    if spec is None or raw_job != spec.payload():
        return str(job_id) if job_id is not None else None, None
    status = payload.get("status")
    records = payload.get("records")
    telemetry = payload.get("telemetry")
    if (
        status not in PARTICLE_RECEIPT_STATUSES
        or not isinstance(records, list)
        or not isinstance(telemetry, dict)
    ):
        return spec.job_id, None
    safe = sanitize_particle_telemetry(telemetry)
    receipt_extras = {
        key: value
        for key, value in telemetry.items()
        if key
        in {
            "candidate_count",
            "candidate_set_sha256",
            "source_state_sha256",
            "expert_training_identity_sha256",
        }
    }
    if {
        **safe,
        **receipt_extras,
    } != telemetry:
        return spec.job_id, None
    if (
        telemetry.get("candidate_count") != len(records)
        or telemetry.get("candidate_set_sha256") != _candidate_set_sha256(records)
        or not _valid_hash(telemetry.get("candidate_set_sha256"))
        or not _valid_hash(telemetry.get("source_state_sha256"))
        or not _valid_hash(telemetry.get("expert_training_identity_sha256"))
        or any(not _validate_route_record(record, spec) for record in records)
    ):
        return spec.job_id, None
    if status in _OPERATIONAL_FAILURE_STATUSES and records:
        return spec.job_id, None
    if (
        status
        in {
            "empty_supported_depth",
            "planning_capacity_abstention",
            "target_capacity_abstention",
        }
        and records
    ):
        return spec.job_id, None
    if status in _COMPLETE_STATUSES and _receipt_status(telemetry) != status:
        return spec.job_id, None
    return spec.job_id, payload


def validate_complete_combination_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_spec: CompleteCombinationJobSpec | None = None,
    source_state_sha256: str | None = None,
    expert_training_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one self-hashed particle receipt and return its payload.

    This boundary is intentionally public for durable proposal orchestration.
    It validates only the frozen particle schema and optional parent identities;
    it does not settle a partial set of receipts or alter failure semantics.
    """

    expected = {row.job_id: row for row in complete_combination_job_specs()}
    job_id, payload = _validate_receipt(receipt, expected)
    if payload is None or job_id is None:
        raise ValueError("invalid complete-combination particle receipt")
    if expected_spec is not None and (
        expected.get(expected_spec.job_id) != expected_spec
        or job_id != expected_spec.job_id
    ):
        raise ValueError("particle receipt belongs to another frozen job")
    telemetry = payload["telemetry"]
    if (
        source_state_sha256 is not None
        and telemetry["source_state_sha256"] != source_state_sha256
    ):
        raise ValueError("particle receipt source-state identity mismatch")
    if (
        expert_training_identity_sha256 is not None
        and telemetry["expert_training_identity_sha256"]
        != expert_training_identity_sha256
    ):
        raise ValueError("particle receipt training identity mismatch")
    return payload


def _legacy_only_result(
    legacy: list[dict[str, Any]],
    *,
    reason: str,
    received: int,
    invalid_receipts: list[str],
    missing_jobs: list[str],
    failed_jobs: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [dict(row) for row in legacy], {
        "schema_version": REDUCTION_SCHEMA_VERSION,
        "particle_augmentation_status": "abstained",
        "particle_augmentation_abstained": True,
        "particle_augmentation_abstention_reason": reason,
        "required_receipts": len(complete_combination_job_specs()),
        "received_receipts": received,
        "invalid_receipts": sorted(invalid_receipts),
        "missing_jobs": sorted(missing_jobs),
        "failed_jobs": sorted(failed_jobs),
        "legacy_records": len(legacy),
        "combined_records": len(legacy),
        "scheduled_complete_depths": list(SUPPORTED_COMPLETE_DEPTHS),
        "depth_4_complete_coverage": DEPTH_4_COMPLETE_COVERAGE,
        "proposal_experts": [ROUTE_EXPERT],
        "controller_rng_consumed": False,
    }


def _particle_alias_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    origin = record["route_particle_origins"][0]
    return (
        -float(record["route_prior_score"]),
        int(origin["binding_particle"]),
        int(origin["depth"]),
        int(origin["shard"]),
        int(origin["local_rank"]),
        str(record["route_program_id"]),
    )


def _origin_key(origin: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(origin["job_id"]),
        int(origin["local_rank"]),
        str(origin["program_id"]),
    )


def _combined_route_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    legacy_rank = record.get("legacy_route_proposal_rank")
    return (
        -float(record.get("route_prior_score", float("-inf"))),
        0 if legacy_rank is not None else 1,
        int(legacy_rank) if legacy_rank is not None else 1 << 30,
        str(record["smiles"]),
        str(record.get("route_program_id") or ""),
    )


def reduce_route_complete_region_pool(
    legacy: Iterable[dict[str, Any]],
    particle_receipts: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Union legacy and particle rows, or abstain atomically to legacy-only."""

    legacy_rows = [dict(row) for row in legacy]
    receipts = list(particle_receipts)
    expected_specs = complete_combination_job_specs()
    expected = {row.job_id: row for row in expected_specs}
    validated: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    duplicate: list[str] = []
    for position, receipt in enumerate(receipts):
        job_id, payload = _validate_receipt(receipt, expected)
        if payload is None or job_id is None:
            invalid.append(job_id or f"receipt-index-{position}")
            continue
        if job_id in validated:
            duplicate.append(job_id)
            continue
        validated[job_id] = payload
    missing = sorted(set(expected).difference(validated))
    failed = sorted(
        job_id
        for job_id, payload in validated.items()
        if payload["status"] in _OPERATIONAL_FAILURE_STATUSES
    )
    if invalid or duplicate:
        return _legacy_only_result(
            legacy_rows,
            reason="corrupt_required_receipt",
            received=len(receipts),
            invalid_receipts=[*invalid, *duplicate],
            missing_jobs=missing,
            failed_jobs=failed,
        )
    if missing:
        return _legacy_only_result(
            legacy_rows,
            reason="missing_required_receipt",
            received=len(receipts),
            invalid_receipts=[],
            missing_jobs=missing,
            failed_jobs=failed,
        )
    if failed:
        return _legacy_only_result(
            legacy_rows,
            reason="operationally_failed_required_receipt",
            received=len(receipts),
            invalid_receipts=[],
            missing_jobs=[],
            failed_jobs=failed,
        )

    source_hashes = {
        payload["telemetry"]["source_state_sha256"] for payload in validated.values()
    }
    expert_hashes = {
        payload["telemetry"]["expert_training_identity_sha256"]
        for payload in validated.values()
    }
    if len(source_hashes) != 1 or len(expert_hashes) != 1:
        return _legacy_only_result(
            legacy_rows,
            reason="inconsistent_required_receipt_identity",
            received=len(receipts),
            invalid_receipts=sorted(validated),
            missing_jobs=[],
            failed_jobs=[],
        )

    particle_occurrences: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for job_id in sorted(expected):
        payload = validated[job_id]
        status_counts[str(payload["status"])] += 1
        particle_occurrences.extend(dict(row) for row in payload["records"])

    particle_by_smiles: dict[str, dict[str, Any]] = {}
    origins_by_smiles: dict[str, list[dict[str, Any]]] = {}
    for row in particle_occurrences:
        smiles = str(row["smiles"])
        origins_by_smiles.setdefault(smiles, []).extend(row["route_particle_origins"])
        previous = particle_by_smiles.get(smiles)
        if previous is None or _particle_alias_key(row) < _particle_alias_key(previous):
            particle_by_smiles[smiles] = row

    legacy_by_smiles: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(legacy_rows):
        smiles = str(row.get("smiles") or "")
        if not smiles:
            raise ValueError("legacy route record has no canonical SMILES")
        previous = legacy_by_smiles.get(smiles)
        legacy_key = (int(row.get("route_proposal_rank", 1 << 30)), position)
        previous_key = None
        if previous is not None:
            previous_key = (
                int(previous.get("route_proposal_rank", 1 << 30)),
                int(previous["_legacy_position"]),
            )
        if previous is None or legacy_key < previous_key:
            legacy_by_smiles[smiles] = {**row, "_legacy_position": position}

    union: list[dict[str, Any]] = []
    all_smiles = sorted(set(legacy_by_smiles) | set(particle_by_smiles))
    overlap = 0
    for smiles in all_smiles:
        particle = particle_by_smiles.get(smiles)
        legacy_row = legacy_by_smiles.get(smiles)
        if legacy_row is not None:
            representative = dict(legacy_row)
            representative.pop("_legacy_position", None)
            representative["legacy_route_proposal_rank"] = representative.get(
                "route_proposal_rank"
            )
            modes = set(representative.get("route_generation_modes") or [])
            modes.add("legacy_complete_region")
            if particle is not None:
                overlap += 1
                modes.add("complete_combination_particle")
            representative["route_generation_modes"] = sorted(modes)
        else:
            if particle is None:
                raise AssertionError("route union lost a particle-only endpoint")
            representative = dict(particle)
        origins = origins_by_smiles.get(smiles, [])
        if origins:
            unique_origins = {_origin_key(origin): dict(origin) for origin in origins}
            representative["route_particle_origins"] = [
                unique_origins[key] for key in sorted(unique_origins)
            ]
        union.append(representative)

    union.sort(key=_combined_route_key)
    for rank, row in enumerate(union, 1):
        row["route_proposal_rank"] = rank
    telemetry = {
        "schema_version": REDUCTION_SCHEMA_VERSION,
        "particle_augmentation_status": "complete",
        "particle_augmentation_abstained": False,
        "required_receipts": len(expected_specs),
        "received_receipts": len(receipts),
        "receipt_status_counts": dict(sorted(status_counts.items())),
        "legacy_records": len(legacy_rows),
        "particle_record_occurrences": len(particle_occurrences),
        "particle_unique_endpoints": len(particle_by_smiles),
        "legacy_particle_endpoint_overlap": overlap,
        "combined_records": len(union),
        "scheduled_complete_depths": list(SUPPORTED_COMPLETE_DEPTHS),
        "depth_4_complete_coverage": DEPTH_4_COMPLETE_COVERAGE,
        "strict_complete_without_capacity_abstention": not any(
            status
            in {
                "planning_capacity_abstention",
                "target_capacity_abstention",
                "complete_with_realization_abstentions",
            }
            for status in status_counts
        ),
        "proposal_experts": [ROUTE_EXPERT],
        "controller_rng_consumed": False,
        "source_state_sha256": next(iter(source_hashes)),
        "expert_training_identity_sha256": next(iter(expert_hashes)),
        "combined_candidate_set_sha256": identity(union),
    }
    return union, telemetry


__all__ = [
    "BINDING_PARTICLE_INDICES",
    "COMBINATION_COUNTS",
    "DEPTH_4_COMPLETE_COVERAGE",
    "MAX_COMBINATIONS_PER_JOB",
    "PARTICLE_RECEIPT_STATUSES",
    "REQUIRED_SHARD_COUNTS",
    "ROUTE_EXPERT",
    "SCHEDULED_SHARD_COUNTS",
    "SCHEMA_VERSION",
    "SELECTED_CONSTITUENT_COUNT",
    "SUPPORTED_COMPLETE_DEPTHS",
    "CompleteCombinationJobSpec",
    "complete_combination_job_specs",
    "frozen_particle_budgets",
    "reduce_route_complete_region_pool",
    "route_record_from_virtual_proposal",
    "run_complete_combination_job",
    "sanitize_particle_telemetry",
    "validate_complete_combination_receipt",
]
