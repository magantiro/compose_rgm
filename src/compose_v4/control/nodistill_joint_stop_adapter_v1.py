"""Production-shaped adapter for the route-free joint-STOP composer.

The public generation boundary accepts exactly a molecular source, a deterministic
seed, and generic work settings.  It does not accept a task or cell name, target,
Fiber threshold, objective, docking score, route, template, teacher action,
endpoint target, or fitted weight.  Endpoint admission remains the caller's job
after all optional expert pools have completed.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.nodistill_joint_stop import (
    JointStopConfig,
    generate_candidate_lock,
    strata_census,
)
from compose_v4.rewrite.kernel import canonical_state_key

SCHEMA = "nodistill_joint_stop_adapter_v1"


@dataclass(frozen=True)
class JointStopAdapterSettings:
    """Generic support and work settings for the optional joint-STOP expert."""

    min_depth: int = 2
    max_depth: int = 4
    beam_width: int = 24
    lock_width: int = 32
    candidates_per_family: int = 6
    expansions_per_parent: int = 24
    max_primitives: int = 32
    max_blocks: int = 8
    max_active_atoms: int = 40

    def __post_init__(self) -> None:
        JointStopConfig(seed=0, **asdict(self))

    def payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class JointStopCandidate:
    """One ranked complete program replayed from the supplied source."""

    endpoint: MolecularGraph
    actions: tuple[dict[str, Any], ...]
    program: dict[str, Any]
    program_graph: dict[str, Any] | None
    families: tuple[str, ...]
    actual_changes: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(self.endpoint)


@dataclass(frozen=True)
class JointStopCandidateBatch:
    """Ranked optional-expert candidates and route-free generation telemetry."""

    candidates: tuple[JointStopCandidate, ...]
    telemetry: dict[str, Any]

    @property
    def proposals(self) -> tuple[JointStopCandidate, ...]:
        """Compatibility alias for complete-program composer consumers."""

        return self.candidates


def _settings(
    value: JointStopAdapterSettings | Mapping[str, Any],
) -> JointStopAdapterSettings:
    if isinstance(value, JointStopAdapterSettings):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("joint-STOP settings must be a settings object or mapping")
    allowed = {field.name for field in fields(JointStopAdapterSettings)}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"joint-STOP settings contain forbidden fields: {unknown}")
    return JointStopAdapterSettings(**dict(value))


def _candidate(
    source: MolecularGraph,
    row: Mapping[str, Any],
    *,
    rank: int,
    config: JointStopConfig,
) -> JointStopCandidate:
    program = EditProgram.from_payload(dict(row["program"]))
    graph = compile_program_graph(program)
    endpoint, receipt = execute_program_graph(
        source,
        graph,
        tuple(row["assignment"]),
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )
    endpoint_key = canonical_state_key(endpoint)
    if (
        endpoint_key != row["endpoint"]
        or receipt["endpoint"] != row["primitive_trace_endpoint"]
        or identity(receipt) != row["trace_identity"]
    ):
        raise RuntimeError("joint-STOP candidate changed on adapter replay")
    actions = tuple(copy.deepcopy(receipt["actions"]))
    primitive_count = len(actions)
    block_count = len(program.blocks)
    features = copy.deepcopy(dict(row["features"]))
    if (
        primitive_count != features["primitives"]
        or block_count != features["blocks"]
        or not 2 <= int(features["depth"]) <= 4
    ):
        raise RuntimeError("joint-STOP candidate metadata differs from replay")
    families = tuple(str(item["family"]) for item in row["constituents"])
    axes = copy.deepcopy(dict(features["strata"]))
    plan_identity = identity({"schema_version": SCHEMA, "axes": axes})
    return JointStopCandidate(
        endpoint=endpoint,
        actions=actions,
        program=copy.deepcopy(program.payload()),
        program_graph=copy.deepcopy(graph.payload()),
        families=families,
        actual_changes=copy.deepcopy(dict(receipt["actual_changes"])),
        metadata={
            "schema_version": SCHEMA,
            "candidate_id": str(row["candidate_id"]),
            "joint_stop_rank": rank,
            "rank_features": features,
            "macro_plan": {
                "name": f"joint_stop_{plan_identity[:16]}",
                "identity": plan_identity,
                "axes": axes,
            },
            "headroom_features": {
                "source_active_atom_headroom": (
                    config.max_active_atoms - int(features["source_heavy"])
                ),
                "endpoint_active_atom_headroom": (
                    config.max_active_atoms - int(features["endpoint_heavy"])
                ),
                "peak_active_atom_headroom": (
                    config.max_active_atoms - int(features["peak_heavy"])
                ),
                "primitive_headroom": config.max_primitives - primitive_count,
                "block_headroom": config.max_blocks - block_count,
            },
            "primitive_count": primitive_count,
            "block_count": block_count,
            "completed_constituent_count": len(families),
            "stop": True,
            "exact_execution_verified": True,
            "intermediate_endpoints_queried": 0,
            "endpoint_admission_applied_during_generation": False,
            "initial_stored_complete_routes": 0,
            "route_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_requested_delta_or_fiber_input": False,
            "runtime_teacher_action_or_endpoint_input": False,
            "runtime_objective_or_docking_score_input": False,
            "runtime_absolute_route_address_input": False,
        },
    )


def propose_joint_stop_candidates(
    source: MolecularGraph,
    *,
    seed: int,
    settings: JointStopAdapterSettings | Mapping[str, Any],
) -> JointStopCandidateBatch:
    """Return ranked complete candidates from source, seed, and settings only.

    The result is pre-admission.  A production caller must canonical-deduplicate
    the optional expert union and apply the ordinary Fiber/archive admission only
    after every producer has finished.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("joint-STOP seed must be a nonnegative integer")
    normalized = _settings(settings)
    config = JointStopConfig(seed=seed, **normalized.payload())
    lock = generate_candidate_lock(
        source,
        arm="coverage_joint_stop",
        config=config,
    )
    candidates = tuple(
        _candidate(source, row, rank=rank, config=config)
        for rank, row in enumerate(lock["candidates"], 1)
    )
    if len({row.endpoint_key for row in candidates}) != len(candidates):
        raise RuntimeError("joint-STOP adapter emitted duplicate canonical endpoints")
    depth_counts = Counter(
        str(row.metadata["rank_features"]["depth"]) for row in candidates
    )
    plan_counts = Counter(row.metadata["macro_plan"]["name"] for row in candidates)
    return JointStopCandidateBatch(
        candidates=candidates,
        telemetry={
            "schema_version": SCHEMA,
            "seed": seed,
            "settings": normalized.payload(),
            "core_lock_id": lock["lock_id"],
            "candidate_order_sha256": identity(
                [row.metadata["candidate_id"] for row in candidates]
            ),
            "raw_exact_unique": len(candidates),
            "exact_replay_verified": len(candidates),
            "depth_counts": dict(sorted(depth_counts.items())),
            "multi_axis_macro_plan_count": len(plan_counts),
            "multi_axis_macro_plan_counts": dict(sorted(plan_counts.items())),
            "strata": strata_census(lock),
            "levels": copy.deepcopy(lock["levels"]),
            "rejections": copy.deepcopy(lock["rejections"]),
            "support": copy.deepcopy(lock["support"]),
            "generation_arm": "coverage_joint_stop",
            "endpoint_constraints_applied_during_generation": False,
            "intermediate_endpoints_queried": 0,
            "initial_stored_complete_routes": 0,
            "route_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "runtime_inputs": {
                "source_graph": True,
                "seed": True,
                "settings": True,
                "task_or_cell_name": False,
                "target_or_endpoint": False,
                "requested_delta_or_fiber": False,
                "objective_or_docking_score": False,
                "route_or_template_identity": False,
                "teacher_action_or_endpoint": False,
                "absolute_route_address": False,
            },
        },
    )


__all__ = [
    "JointStopAdapterSettings",
    "JointStopCandidate",
    "JointStopCandidateBatch",
    "propose_joint_stop_candidates",
]
