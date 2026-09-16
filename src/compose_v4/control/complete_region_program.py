"""Protected runtime for complete address-free dependency-region programs.

The learned/runtime decision unit is one complete structural target patch, not
one primitive executor action. A program opens with one patch, may continue
with up to three more patches, and commits only after the final STOP. Primitive
actions and intermediate molecular states remain internal to the exact
realizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    attachment_bindings,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_structural_goal,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

PROGRAM_SCHEMA = "complete_region_program_v1"
DECISION_SCHEMA = "complete_region_decision_v1"
CREATED_ROLE_SCHEMA = "relative_created_role_reference_v1"
CONTINUE = "continue"
STOP = "stop"


@dataclass(frozen=True)
class CreatedRoleReference:
    """Address-free reference to an output created by an earlier region."""

    region_index: int
    output_index: int

    def __post_init__(self) -> None:
        if self.region_index < 0 or self.output_index < 0:
            raise ValueError("created-role indices must be nonnegative")

    def payload(self) -> dict:
        return {
            "schema_version": CREATED_ROLE_SCHEMA,
            "region_index": self.region_index,
            "output_index": self.output_index,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CreatedRoleReference:
        if payload.get("schema_version") != CREATED_ROLE_SCHEMA or set(payload) != {
            "schema_version",
            "region_index",
            "output_index",
        }:
            raise ValueError("unexpected created-role reference schema")
        return cls(int(payload["region_index"]), int(payload["output_index"]))


@dataclass(frozen=True)
class CompleteRegionDecision:
    """One complete target patch and its protected-program control decision."""

    patch: StructuralSubgoal
    input_provenance: tuple[CreatedRoleReference | None, ...]
    control_after: str

    def __post_init__(self) -> None:
        if len(self.input_provenance) != len(self.patch.input_atoms):
            raise ValueError("input provenance and structural input roles disagree")
        if self.control_after not in (CONTINUE, STOP):
            raise ValueError("region control must be continue or stop")

    def payload(self) -> dict:
        return {
            "schema_version": DECISION_SCHEMA,
            "patch": self.patch.payload(),
            "input_provenance": [
                None if row is None else row.payload() for row in self.input_provenance
            ],
            "control_after": self.control_after,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CompleteRegionDecision:
        if payload.get("schema_version") != DECISION_SCHEMA or set(payload) != {
            "schema_version",
            "patch",
            "input_provenance",
            "control_after",
        }:
            raise ValueError("unexpected complete-region decision schema")
        return cls(
            StructuralSubgoal.from_payload(payload["patch"]),
            tuple(
                None if row is None else CreatedRoleReference.from_payload(row)
                for row in payload["input_provenance"]
            ),
            str(payload["control_after"]),
        )


@dataclass(frozen=True)
class CompleteRegionProgram:
    """One protected program containing one to four complete region targets."""

    decisions: tuple[CompleteRegionDecision, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.decisions) <= 4:
            raise ValueError("complete region program requires one to four decisions")
        for index, decision in enumerate(self.decisions):
            expected = STOP if index == len(self.decisions) - 1 else CONTINUE
            if decision.control_after != expected:
                raise ValueError(
                    "complete region program has an invalid CONTINUE/STOP sequence"
                )
            for reference in decision.input_provenance:
                if reference is None:
                    continue
                if reference.region_index >= index:
                    raise ValueError(
                        "a created-role reference must point to an earlier region"
                    )
                prior = self.decisions[reference.region_index].patch
                if reference.output_index >= len(prior.output_atoms):
                    raise ValueError(
                        "created-role reference exceeds the prior region outputs"
                    )

    @property
    def program_id(self) -> str:
        return identity(self.payload())

    @property
    def has_created_role_dependencies(self) -> bool:
        return any(
            reference is not None
            for decision in self.decisions
            for reference in decision.input_provenance
        )

    @property
    def goal(self) -> StructuralGoal:
        return StructuralGoal(tuple(row.patch for row in self.decisions))

    def payload(self) -> dict:
        return {
            "schema_version": PROGRAM_SCHEMA,
            "decisions": [row.payload() for row in self.decisions],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CompleteRegionProgram:
        if payload.get("schema_version") != PROGRAM_SCHEMA or set(payload) != {
            "schema_version",
            "decisions",
        }:
            raise ValueError("unexpected complete-region program schema")
        return cls(
            tuple(
                CompleteRegionDecision.from_payload(row) for row in payload["decisions"]
            )
        )


def program_from_structural_goal(goal: StructuralGoal) -> CompleteRegionProgram:
    """Lift an existing complete structural goal into deployed program control."""

    return CompleteRegionProgram(
        tuple(
            CompleteRegionDecision(
                patch=subgoal,
                input_provenance=tuple(None for _ in subgoal.input_atoms),
                control_after=STOP if index == len(goal.subgoals) - 1 else CONTINUE,
            )
            for index, subgoal in enumerate(goal.subgoals)
        )
    )


def _abstention(
    *,
    program: CompleteRegionProgram,
    reason: str,
    regions_completed: int,
    internal_primitives: int,
) -> dict:
    return {
        "status": reason,
        "program_id": program.program_id,
        "regions_requested": len(program.decisions),
        "regions_completed_in_protected_state": regions_completed,
        "stop_reached": False,
        "committed_endpoint_count": 0,
        "committed_endpoint_state": None,
        "realized_actions": [],
        "internal_primitive_count_before_abort": internal_primitives,
        "partial_endpoint_evaluations": 0,
        "partial_endpoint_locks": 0,
        "primitive_teacher_actions_used": 0,
    }


def _binding(
    decision: CompleteRegionDecision,
    graph: MolecularGraph,
    *,
    supplied: tuple[int, ...] | None,
    created_roles: dict[tuple[int, int], int],
) -> tuple[int, ...] | None:
    census = attachment_bindings(decision.patch, graph)
    candidates = list(census.assignments)
    for input_index, reference in enumerate(decision.input_provenance):
        if reference is None:
            continue
        required = created_roles.get((reference.region_index, reference.output_index))
        if required is None:
            return None
        candidates = [row for row in candidates if row[input_index] == required]
    if supplied is not None:
        return supplied if supplied in candidates else None
    return min(candidates) if candidates else None


def execute_complete_region_program(
    source: MolecularGraph,
    program: CompleteRegionProgram,
    *,
    resolved_bindings: tuple[tuple[int, ...], ...] | None = None,
    config: RealizerConfig | None = None,
) -> dict:
    """Execute a complete protected program and commit only at STOP.

    ``resolved_bindings`` represents the output of the ordinary address-free
    binder. When supplied for a teacher-forced diagnostic, every row is still
    required to occur in the runtime binding enumeration. It is not serialized
    into the program and it never supplies a primitive action.
    """

    if resolved_bindings is not None and len(resolved_bindings) != len(
        program.decisions
    ):
        raise ValueError("resolved binding count and region decisions disagree")
    config = RealizerConfig() if config is None else config

    # Independent region targets can be compiled together at STOP. This keeps
    # compensating edits protected within one exact target and one 32-primitive
    # budget. Cross-region created-role references use the sequential protected
    # path below because their bindings exist only after earlier realization.
    if not program.has_created_role_dependencies:
        bindings = []
        for index, decision in enumerate(program.decisions):
            supplied = None if resolved_bindings is None else resolved_bindings[index]
            assignment = _binding(decision, source, supplied=supplied, created_roles={})
            if assignment is None:
                return _abstention(
                    program=program,
                    reason="binding_abstention",
                    regions_completed=0,
                    internal_primitives=0,
                )
            bindings.append(assignment)
        realized = realize_structural_goal(
            source,
            program.goal,
            tuple(bindings),
            config=config,
        )
        if (
            realized["status"] != "realized"
            or not realized["endpoint_matches_bound_target"]
        ):
            return _abstention(
                program=program,
                reason=f"realizer_{realized['status']}",
                regions_completed=0,
                internal_primitives=len(realized["actions"]),
            )
        endpoint_state = realized["states"][-1]
        return {
            "status": "committed",
            "program_id": program.program_id,
            "execution_mode": "global_at_stop",
            "regions_requested": len(program.decisions),
            "regions_completed_in_protected_state": len(program.decisions),
            "stop_reached": True,
            "committed_endpoint_count": 1,
            "committed_endpoint_state": endpoint_state,
            "committed_endpoint_key": canonical_state_key(decode_state(endpoint_state)),
            "realized_actions": realized["actions"],
            "realized_primitive_count": len(realized["actions"]),
            "compiler_strategy": realized.get("compiler_strategy"),
            "expanded": realized["expanded"],
            "attempted": realized["attempted"],
            "partial_endpoint_evaluations": 0,
            "partial_endpoint_locks": 0,
            "primitive_teacher_actions_used": realized[
                "primitive_teacher_actions_used"
            ],
            "region_receipts": [
                {
                    "region_index": index,
                    "subgoal_id": decision.patch.subgoal_id,
                    "control_after": decision.control_after,
                    "binding_enumerated": True,
                }
                for index, decision in enumerate(program.decisions)
            ],
        }

    current = source
    created_roles: dict[tuple[int, int], int] = {}
    actions: list[dict] = []
    receipts = []
    total_expanded = total_attempted = 0
    for index, decision in enumerate(program.decisions):
        supplied = None if resolved_bindings is None else resolved_bindings[index]
        assignment = _binding(
            decision,
            current,
            supplied=supplied,
            created_roles=created_roles,
        )
        if assignment is None:
            return _abstention(
                program=program,
                reason="binding_abstention",
                regions_completed=index,
                internal_primitives=len(actions),
            )
        remaining = config.maximum_primitives - len(actions)
        if remaining < 1:
            return _abstention(
                program=program,
                reason="total_primitive_budget_abstention",
                regions_completed=index,
                internal_primitives=len(actions),
            )
        realized = realize_structural_goal(
            current,
            StructuralGoal((decision.patch,)),
            (assignment,),
            config=RealizerConfig(
                maximum_primitives=remaining,
                maximum_active_atoms=config.maximum_active_atoms,
                maximum_expansions=config.maximum_expansions,
                children_per_expansion=config.children_per_expansion,
            ),
        )
        total_expanded += int(realized["expanded"])
        total_attempted += int(realized["attempted"])
        if (
            realized["status"] != "realized"
            or not realized["endpoint_matches_bound_target"]
        ):
            return _abstention(
                program=program,
                reason=f"realizer_{realized['status']}",
                regions_completed=index,
                internal_primitives=len(actions),
            )
        output_slots = tuple(map(int, realized.get("output_role_slots", ())))
        if len(output_slots) != len(decision.patch.output_atoms):
            return _abstention(
                program=program,
                reason="output_role_receipt_abstention",
                regions_completed=index,
                internal_primitives=len(actions),
            )
        for output_index, slot in enumerate(output_slots):
            created_roles[(index, output_index)] = slot
        actions.extend(realized["actions"])
        current = decode_state(realized["states"][-1])
        receipts.append(
            {
                "region_index": index,
                "subgoal_id": decision.patch.subgoal_id,
                "control_after": decision.control_after,
                "binding_enumerated": True,
                "output_roles_created": len(output_slots),
                "primitive_count": len(realized["actions"]),
                "compiler_strategy": realized.get("compiler_strategy"),
            }
        )

    endpoint_state = realized["states"][-1]
    return {
        "status": "committed",
        "program_id": program.program_id,
        "execution_mode": "sequential_created_role_dependencies",
        "regions_requested": len(program.decisions),
        "regions_completed_in_protected_state": len(program.decisions),
        "stop_reached": True,
        "committed_endpoint_count": 1,
        "committed_endpoint_state": endpoint_state,
        "committed_endpoint_key": canonical_state_key(current),
        "realized_actions": actions,
        "realized_primitive_count": len(actions),
        "expanded": total_expanded,
        "attempted": total_attempted,
        "partial_endpoint_evaluations": 0,
        "partial_endpoint_locks": 0,
        "primitive_teacher_actions_used": 0,
        "region_receipts": receipts,
    }


__all__ = [
    "CONTINUE",
    "CREATED_ROLE_SCHEMA",
    "DECISION_SCHEMA",
    "PROGRAM_SCHEMA",
    "STOP",
    "CompleteRegionDecision",
    "CompleteRegionProgram",
    "CreatedRoleReference",
    "execute_complete_region_program",
    "program_from_structural_goal",
]
