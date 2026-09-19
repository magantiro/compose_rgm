"""Route-distilled complete-region proposals for objective-guided search.

The expert stores only address-free structural-delta templates and a balanced
frequency law fitted from training routes.  At runtime templates are rebound to
the current molecule, composed into one-to-four-region goals, and passed through
the same protected complete-region runtime used by the representation audit.
No source graph, endpoint, route identifier, target name, or executable teacher
program is stored in the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            tuple(StructuralDeltaTemplate.from_payload(row) for row in payload["templates"]),
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
) -> tuple[list[dict], dict]:
    """Generate and exact-execute complete region programs on one current state.

    The proposal score is the route marginal, not docking reward.  Returned rows
    are deliberately evaluator-free so the same expert can feed FiberControl or
    another objective adapter.
    """

    if realization_limit < 1 or realization_limit > pool_size:
        raise ValueError("route expert realization limit must be within the pool")
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
            receipt = execute_complete_region_program(
                source,
                program,
                resolved_bindings=proposal.bindings,
                config=RealizerConfig(maximum_expansions=maximum_expansions),
            )
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
            else "medium"
            if realized_primitives <= 11
            else "large"
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
                    atom is None for row in proposal.templates for atom in row.target_atoms
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
    return records, telemetry


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RouteDistilledGoalExpert",
    "make_route_expert",
    "propose_route_expert_candidates",
]
