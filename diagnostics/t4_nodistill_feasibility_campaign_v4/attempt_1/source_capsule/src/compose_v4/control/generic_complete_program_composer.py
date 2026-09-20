"""Route-free complete-program composition for COMPOSE-NoDistill.

This module composes existing generic structural operators on their exact
successors.  It contains no trajectory-derived template, fitted route weight,
task identity, endpoint, or objective value.  A requested scale is enforced
from the realized primitive count, never inferred from a route label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    changed_input_sites,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.progressive_structured_sampler import progressive_module
from compose_v4.rewrite.kernel import canonical_state_key

SCALE_BANDS = ("small", "medium", "large")

# Fixed generic operator schedules.  These are algorithmic support, not
# trajectory statistics.  Every schedule is rebound on the current exact state.
GENERIC_FAMILY_SCHEDULES: dict[str, tuple[tuple[str, ...], ...]] = {
    "small": (
        ("functionalize",),
        ("heteroatom_substitute",),
        ("carbonyl_insert",),
        ("bond_reroute",),
        ("cycle_close",),
        ("cycle_open",),
        ("ring_system_restate",),
        ("segment_grow",),
    ),
    "medium": (
        ("segment_replace",),
        ("ring_path_remodel",),
        ("construct_substituted_ring",),
        ("substituent_delete", "functionalize"),
        ("segment_grow", "cycle_close"),
        ("cycle_open", "segment_grow"),
        ("substituent_delete", "segment_grow"),
        ("segment_grow", "functionalize"),
    ),
    "large": (
        ("substituent_delete", "construct_substituted_ring"),
        ("segment_replace", "construct_substituted_ring"),
        ("ring_path_remodel", "segment_grow", "cycle_close"),
        ("substituent_delete", "segment_grow", "append_ring"),
        ("segment_grow", "append_ring", "functionalize"),
        (
            "substituent_delete",
            "construct_substituted_ring",
            "ring_path_remodel",
        ),
        ("segment_replace", "fuse_ring", "functionalize"),
        ("cycle_open", "segment_grow", "construct_substituted_ring"),
    ),
}


def primitive_scale(primitive_count: int) -> str:
    """Return the frozen task-independent scale band."""

    if type(primitive_count) is not int or not 1 <= primitive_count <= 32:
        raise ValueError("primitive count must be within the declared 1..32 support")
    if primitive_count <= 3:
        return "small"
    if primitive_count <= 11:
        return "medium"
    return "large"


@dataclass(frozen=True)
class GenericCompleteProgramProposal:
    """One exact protected program constructed without route distillation."""

    endpoint: MolecularGraph
    actions: tuple[dict[str, Any], ...]
    program: dict[str, Any]
    program_graph: dict[str, Any]
    families: tuple[str, ...]
    requested_scale: str
    realized_scale: str
    metadata: dict[str, Any]
    actual_changes: dict[str, Any]

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(self.endpoint)


@dataclass(frozen=True)
class GenericCompleteProgramBatch:
    proposals: tuple[GenericCompleteProgramProposal, ...]
    telemetry: dict[str, Any]


def _compile_schedule(
    source: MolecularGraph,
    rng: np.random.Generator,
    families: tuple[str, ...],
    *,
    maximum_primitives: int,
    maximum_blocks: int,
) -> GenericCompleteProgramProposal:
    current = source
    stages: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    preferred = frozenset()
    for family in families:
        product, stage = progressive_module(
            current,
            rng,
            family,
            preferred=preferred,
        )
        candidate_program, _ = extract_program(source, [*stages, stage])
        if (
            len(candidate_program.marks) > maximum_primitives
            or len(candidate_program.blocks) > maximum_blocks
        ):
            raise ValueError("generic complete program exceeds its work support")
        stages.append(stage)
        module_rows.append(
            {
                "family": family,
                "primitive_edits": len(stage["actions"]),
                "parameters": stage["parameters"],
            }
        )
        current = product
        preferred = v1._following_context(current, stage)

    program, assignment = extract_program(source, stages)
    graph = compile_program_graph(program)
    endpoint, receipt = execute_program_graph(
        source,
        graph,
        assignment,
        max_primitives=maximum_primitives,
        max_blocks=maximum_blocks,
    )
    if canonical_state_key(endpoint) != canonical_state_key(current):
        raise RuntimeError("generic complete composition changed on exact replay")
    actions = tuple(receipt["actions"])
    realized_scale = primitive_scale(len(actions))
    return GenericCompleteProgramProposal(
        endpoint=endpoint,
        actions=actions,
        program=program.payload(),
        program_graph=graph.payload(),
        families=families,
        requested_scale=realized_scale,
        realized_scale=realized_scale,
        metadata={
            "schema_version": "generic_complete_program_composer_v1",
            "modules": module_rows,
            "completed_module_count": len(module_rows),
            "created_handle_dependencies": len(graph.dependencies),
            "serialization_edges": len(graph.serialization_edges),
            "initial_stored_complete_routes": 0,
            "source_library_rows_loaded": 0,
            "trajectory_distilled_templates_loaded": 0,
            "intermediate_task_evaluations": 0,
            "task_oracle_information": False,
        },
        actual_changes=changed_input_sites(source, endpoint, actions),
    )


def propose_generic_complete_programs(
    source: MolecularGraph,
    *,
    seed: int,
    per_scale: int,
    attempts_per_scale: int,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> GenericCompleteProgramBatch:
    """Generate a scale-balanced lock of exact route-free complete programs.

    Candidate limits are output limits.  A scale shortfall is reported rather
    than filled from another band.  Duplicate endpoints do not consume output
    capacity.  The proposal law receives only the molecular source and a seed.
    """

    if type(seed) is not int:
        raise TypeError("generic complete-program seed must be an integer")
    if type(per_scale) is not int or per_scale < 1:
        raise ValueError("per_scale must be a positive integer")
    if type(attempts_per_scale) is not int or attempts_per_scale < per_scale:
        raise ValueError("attempts_per_scale must cover the output budget")
    if maximum_primitives != 32 or maximum_blocks != 8:
        raise ValueError("generic composer support is frozen at 32 primitives/8 blocks")

    accepted: dict[str, GenericCompleteProgramProposal] = {}
    telemetry: Counter[str] = Counter()
    telemetry_rows: dict[str, Any] = {}
    for scale_index, scale in enumerate(SCALE_BANDS):
        schedules = GENERIC_FAMILY_SCHEDULES[scale]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 911, scale_index]))
        scale_rows: dict[str, GenericCompleteProgramProposal] = {}
        failures: Counter[str] = Counter()
        schedule_attempts: Counter[str] = Counter()
        for attempt in range(attempts_per_scale):
            if len(scale_rows) >= per_scale:
                break
            schedule = schedules[attempt % len(schedules)]
            schedule_key = "+".join(schedule)
            schedule_attempts[schedule_key] += 1
            try:
                proposal = _compile_schedule(
                    source,
                    rng,
                    schedule,
                    maximum_primitives=maximum_primitives,
                    maximum_blocks=maximum_blocks,
                )
            except (ValueError, RuntimeError) as error:
                failures[f"compile:{type(error).__name__}:{error}"] += 1
                continue
            telemetry["exact_execution_attempts"] += 1
            if proposal.realized_scale != scale:
                failures[f"realized_scale:{proposal.realized_scale}"] += 1
                continue
            if proposal.endpoint_key == canonical_state_key(source):
                failures["canonical_self_event"] += 1
                continue
            if proposal.endpoint_key in accepted or proposal.endpoint_key in scale_rows:
                failures["canonical_endpoint_alias"] += 1
                continue
            scale_rows[proposal.endpoint_key] = GenericCompleteProgramProposal(
                endpoint=proposal.endpoint,
                actions=proposal.actions,
                program=proposal.program,
                program_graph=proposal.program_graph,
                families=proposal.families,
                requested_scale=scale,
                realized_scale=proposal.realized_scale,
                metadata=proposal.metadata,
                actual_changes=proposal.actual_changes,
            )
            telemetry["exact_execution_accepts"] += 1
        accepted.update(scale_rows)
        telemetry_rows[scale] = {
            "requested_candidates": per_scale,
            "accepted_candidates": len(scale_rows),
            "candidate_shortfall": per_scale - len(scale_rows),
            "attempt_limit": attempts_per_scale,
            "schedule_attempts": dict(sorted(schedule_attempts.items())),
            "failure_counts": dict(sorted(failures.items())),
        }

    proposals = tuple(
        sorted(
            accepted.values(),
            key=lambda row: (
                SCALE_BANDS.index(row.realized_scale),
                row.families,
                row.endpoint_key,
            ),
        )
    )
    return GenericCompleteProgramBatch(
        proposals=proposals,
        telemetry={
            "schema_version": "generic_complete_program_composer_telemetry_v1",
            "seed": seed,
            "per_scale": per_scale,
            "attempts_per_scale": attempts_per_scale,
            "maximum_primitives": maximum_primitives,
            "maximum_blocks": maximum_blocks,
            "scale_allocation": telemetry_rows,
            "exact_execution_attempts": telemetry["exact_execution_attempts"],
            "exact_execution_accepts": telemetry["exact_execution_accepts"],
            "unique_committed_endpoints": len(proposals),
            "family_schedule_identity": identity(GENERIC_FAMILY_SCHEDULES),
            "trajectory_distilled_templates_loaded": 0,
            "route_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_teacher_endpoint_input": False,
            "runtime_objective_input": False,
        },
    )


__all__ = [
    "GENERIC_FAMILY_SCHEDULES",
    "SCALE_BANDS",
    "GenericCompleteProgramBatch",
    "GenericCompleteProgramProposal",
    "primitive_scale",
    "propose_generic_complete_programs",
]
