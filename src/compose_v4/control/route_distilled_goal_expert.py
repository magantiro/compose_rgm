"""Route-distilled complete-region proposals for objective-guided search.

The expert stores only address-free structural-delta templates and a balanced
frequency law fitted from training routes.  At runtime templates are rebound to
the current molecule, composed into one-to-four-region goals, and passed through
the same protected complete-region runtime used by the representation audit.
No source graph, endpoint, route identifier, target name, or executable teacher
program is stored in the checkpoint.
"""

from __future__ import annotations

import math
import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.control.complete_region_program import (
    execute_complete_region_program,
    program_from_structural_goal,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
    proposal_rewrite_event_count,
    proposal_rewrite_scale,
    propose_structural_goals,
)
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

CHECKPOINT_SCHEMA = "route_distilled_complete_region_expert_v1"
_PROCESS_CLEANUP_SECONDS = 1.0


class _CandidateRealizationTimeout(RuntimeError):
    """Internal control signal for one bounded candidate realization."""


def _realization_worker(
    connection: Connection,
    source: MolecularGraph,
    program: Any,
    resolved_bindings: Any,
    maximum_expansions: int,
) -> None:
    """Execute one candidate and report its result across a process boundary."""

    try:
        receipt = execute_complete_region_program(
            source,
            program,
            resolved_bindings=resolved_bindings,
            config=RealizerConfig(maximum_expansions=maximum_expansions),
        )
        connection.send(("committed_result", receipt))
    except ValueError as exc:
        connection.send(("value_error", str(exc)))
    finally:
        connection.close()


def _terminate_process(process: multiprocessing.Process) -> None:
    """Terminate and reap one worker, escalating to kill when supported."""

    if process.is_alive():
        process.terminate()
    process.join(_PROCESS_CLEANUP_SECONDS)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(_PROCESS_CLEANUP_SECONDS)
    if process.is_alive():
        raise RuntimeError("route realization worker could not be terminated")


def _execute_with_candidate_timeout(
    source: MolecularGraph,
    program: Any,
    *,
    resolved_bindings: Any,
    maximum_expansions: int,
    timeout_seconds: float,
) -> dict:
    """Execute one candidate in an independently terminable child process."""

    start_methods = multiprocessing.get_all_start_methods()
    start_method = "fork" if "fork" in start_methods else "spawn"
    context = multiprocessing.get_context(start_method)
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_realization_worker,
        args=(
            child_connection,
            source,
            program,
            resolved_bindings,
            maximum_expansions,
        ),
    )
    try:
        process.start()
        child_connection.close()
        if not parent_connection.poll(timeout_seconds):
            _terminate_process(process)
            raise _CandidateRealizationTimeout
        try:
            message_type, payload = parent_connection.recv()
        except EOFError as exc:
            process.join(_PROCESS_CLEANUP_SECONDS)
            raise RuntimeError(
                "route realization worker exited without a result "
                f"(exitcode={process.exitcode})"
            ) from exc
        process.join(_PROCESS_CLEANUP_SECONDS)
        if process.is_alive():
            _terminate_process(process)
            raise RuntimeError("route realization worker did not exit after reporting")
        if process.exitcode != 0:
            raise RuntimeError(
                "route realization worker failed after reporting "
                f"(exitcode={process.exitcode})"
            )
        if message_type == "committed_result":
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "route realization worker returned a non-dict receipt"
                )
            return payload
        if message_type == "value_error":
            raise ValueError(str(payload))
        raise RuntimeError(f"unknown route realization worker message {message_type!r}")
    finally:
        parent_connection.close()
        child_connection.close()
        if process.pid is not None:
            _terminate_process(process)
            process.close()


@dataclass(frozen=True)
class RouteDistilledGoalExpert:
    """A target-free runtime distribution over transferable complete regions."""

    templates: tuple[StructuralDeltaTemplate, ...]
    marginal: MarginalSubgoalPolicy
    training_identity: str

    def __post_init__(self) -> None:
        template_ids = tuple(row.template_id for row in self.templates)
        if (
            not self.templates
            or template_ids != tuple(sorted(template_ids))
            or len(set(template_ids)) != len(template_ids)
            or set(template_ids) != set(self.marginal.template_ids)
            or not self.training_identity
        ):
            raise ValueError("invalid route-distilled complete-region expert")

    def checkpoint(self) -> dict:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "templates": [row.payload() for row in self.templates],
            "marginal": self.marginal.checkpoint(),
            "training_identity": self.training_identity,
        }
        serialized = str(payload)
        for forbidden in ("route_id", "source_group", "endpoint", "target_name"):
            if forbidden in serialized:
                raise RuntimeError(f"runtime route expert leaked {forbidden}")
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict) -> RouteDistilledGoalExpert:
        if payload.get("schema_version") != CHECKPOINT_SCHEMA or set(payload) != {
            "schema_version",
            "templates",
            "marginal",
            "training_identity",
        }:
            raise ValueError("route-distilled complete-region checkpoint mismatch")
        return cls(
            tuple(
                StructuralDeltaTemplate.from_payload(row)
                for row in payload["templates"]
            ),
            MarginalSubgoalPolicy.from_checkpoint(payload["marginal"]),
            str(payload["training_identity"]),
        )


def make_route_expert(
    templates: tuple[StructuralDeltaTemplate, ...],
    marginal: MarginalSubgoalPolicy,
    *,
    training_evidence_identity: str,
) -> RouteDistilledGoalExpert:
    """Seal already split-first templates and their balanced marginal law."""

    ordered = tuple(sorted(templates, key=lambda row: row.template_id))
    training_identity = identity(
        {
            "schema_version": "route_distilled_complete_region_fit_identity_v1",
            "template_ids": [row.template_id for row in ordered],
            "marginal": marginal.training_identity,
            "training_evidence_identity": training_evidence_identity,
        }
    )
    return RouteDistilledGoalExpert(ordered, marginal, training_identity)


def propose_route_expert_candidates(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    *,
    pool_size: int = 64,
    realization_limit: int = 32,
    beam_width: int = 32,
    expansion_width: int = 24,
    max_bindings_per_template: int = 4,
    maximum_expansions: int = 4_000,
    scale_balanced: bool = False,
    per_candidate_timeout_seconds: float | None = None,
) -> tuple[list[dict], dict]:
    """Generate and exact-execute complete region programs on one current state.

    The proposal score is the route marginal, not docking reward.  Returned rows
    are deliberately evaluator-free so the same expert can feed FiberControl or
    another objective adapter.
    """

    if realization_limit < 1 or realization_limit > pool_size:
        raise ValueError("route expert realization limit must be within the pool")
    if per_candidate_timeout_seconds is not None:
        if (
            isinstance(per_candidate_timeout_seconds, bool)
            or not isinstance(per_candidate_timeout_seconds, (int, float))
            or not math.isfinite(float(per_candidate_timeout_seconds))
            or per_candidate_timeout_seconds <= 0
        ):
            raise ValueError(
                "route expert per-candidate timeout must be positive and finite"
            )
        per_candidate_timeout_seconds = float(per_candidate_timeout_seconds)
    goals, proposal_telemetry = propose_structural_goals(
        source,
        expert.templates,
        expert.marginal,
        ranker=None,
        pool_size=pool_size,
        beam_width=beam_width,
        expansion_width=expansion_width,
        max_bindings_per_template=max_bindings_per_template,
        scale_balanced=scale_balanced,
    )
    records = []
    status_counts: dict[str, int] = {}
    realized_primitive_band_counts = {"small": 0, "medium": 0, "large": 0}
    for proposal_rank, proposal in enumerate(goals[:realization_limit], 1):
        program = program_from_structural_goal(proposal.goal)
        try:
            if per_candidate_timeout_seconds is None:
                receipt = execute_complete_region_program(
                    source,
                    program,
                    resolved_bindings=proposal.bindings,
                    config=RealizerConfig(maximum_expansions=maximum_expansions),
                )
            else:
                receipt = _execute_with_candidate_timeout(
                    source,
                    program,
                    resolved_bindings=proposal.bindings,
                    maximum_expansions=maximum_expansions,
                    timeout_seconds=per_candidate_timeout_seconds,
                )
        except _CandidateRealizationTimeout:
            status_counts["realizer_candidate_timeout"] = (
                status_counts.get("realizer_candidate_timeout", 0) + 1
            )
            continue
        except ValueError:
            status_counts["invalid_composed_target"] = (
                status_counts.get("invalid_composed_target", 0) + 1
            )
            continue
        status = str(receipt["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "committed":
            continue
        endpoint = decode_state(receipt["committed_endpoint_state"])
        if canonical_state_key(endpoint) != canonical_state_key(proposal.endpoint):
            raise RuntimeError("complete-region runtime changed a route proposal")
        realized_primitives = int(receipt["realized_primitive_count"])
        primitive_band = (
            "small"
            if realized_primitives <= 3
            else "medium" if realized_primitives <= 11 else "large"
        )
        realized_primitive_band_counts[primitive_band] += 1
        records.append(
            {
                "smiles": molecular_graph_to_smiles(endpoint),
                "proposal_lane": "route_complete_region",
                "proposal_experts": ["route_complete_region"],
                "families": ["route_complete_region"],
                "program_families": ["route_complete_region"],
                "regions": len(proposal.templates),
                "created": sum(len(row.output_atoms) for row in proposal.templates),
                "deleted": sum(
                    atom is None
                    for row in proposal.templates
                    for atom in row.target_atoms
                ),
                "route_prior_score": float(proposal.score),
                "route_proposal_rank": proposal_rank,
                "route_program_id": program.program_id,
                "route_template_ids": [row.template_id for row in proposal.templates],
                "rewrite_events": proposal_rewrite_event_count(proposal),
                "rewrite_scale": proposal_rewrite_scale(proposal),
                "realized_primitives": realized_primitives,
                "realized_primitive_band": primitive_band,
                "compiler_strategy": receipt.get("compiler_strategy"),
            }
        )
    telemetry = {
        **proposal_telemetry,
        "realization_limit": realization_limit,
        "realization_status_counts": dict(sorted(status_counts.items())),
        "complete_programs_committed": len(records),
        "realized_primitive_band_counts": realized_primitive_band_counts,
        "exact_realization_precision_numerator": len(records),
        "exact_realization_precision_denominator": len(records),
    }
    if per_candidate_timeout_seconds is not None:
        telemetry["per_candidate_timeout_seconds"] = per_candidate_timeout_seconds
    return records, telemetry


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RouteDistilledGoalExpert",
    "make_route_expert",
    "propose_route_expert_candidates",
]
