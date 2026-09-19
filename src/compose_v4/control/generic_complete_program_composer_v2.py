"""Route-free macro plans for repeated topology-closing programs.

The plan law is task independent.  It describes desired graph-level change
ranges, then composes existing generic COMPOSE operators and accepts only exact
replays inside the declared plan.  It never reads a route, template, endpoint,
target identity, or objective value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer import (
    GenericCompleteProgramProposal,
    _compile_schedule,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class GenericTopologyMacroPlan:
    """One task-independent graph-change intention."""

    name: str
    families: tuple[str, ...]
    target_delta_heavy_atoms: tuple[int, int]
    target_delta_cycle_rank: tuple[int, int]
    retained_fraction_floor: float
    interface_count_range: tuple[int, int]
    rewrite_mode: str
    candidate_quota: int

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "families": list(self.families),
            "target_delta_heavy_atoms": list(self.target_delta_heavy_atoms),
            "target_delta_cycle_rank": list(self.target_delta_cycle_rank),
            "retained_fraction_floor": self.retained_fraction_floor,
            "interface_count_range": list(self.interface_count_range),
            "rewrite_mode": self.rewrite_mode,
            "candidate_quota": self.candidate_quota,
        }


TOPOLOGY_MACRO_PLANS: tuple[GenericTopologyMacroPlan, ...] = (
    GenericTopologyMacroPlan(
        name="construct_two_connected_rings",
        families=("construct_substituted_ring", "construct_substituted_ring"),
        target_delta_heavy_atoms=(6, 16),
        target_delta_cycle_rank=(2, 2),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 8),
        rewrite_mode="construct_connected_topology",
        candidate_quota=8,
    ),
    GenericTopologyMacroPlan(
        name="construct_three_connected_rings",
        families=(
            "construct_substituted_ring",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        target_delta_heavy_atoms=(9, 22),
        target_delta_cycle_rank=(3, 3),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 10),
        rewrite_mode="construct_connected_topology",
        candidate_quota=4,
    ),
    GenericTopologyMacroPlan(
        name="grow_then_construct_two_rings",
        families=(
            "segment_grow",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        target_delta_heavy_atoms=(7, 20),
        target_delta_cycle_rank=(2, 2),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 10),
        rewrite_mode="grow_and_construct_connected_topology",
        candidate_quota=3,
    ),
    GenericTopologyMacroPlan(
        name="replace_then_construct_two_rings",
        families=(
            "segment_replace",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        target_delta_heavy_atoms=(6, 20),
        target_delta_cycle_rank=(2, 2),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 10),
        rewrite_mode="release_replace_and_construct_connected_topology",
        candidate_quota=3,
    ),
    GenericTopologyMacroPlan(
        name="open_then_construct_three_rings",
        families=(
            "cycle_open",
            "construct_substituted_ring",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        target_delta_heavy_atoms=(9, 22),
        target_delta_cycle_rank=(2, 2),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 10),
        rewrite_mode="open_and_reconstruct_connected_topology",
        candidate_quota=3,
    ),
    GenericTopologyMacroPlan(
        name="release_grow_then_construct_two_rings",
        families=(
            "substituent_delete",
            "segment_grow",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        target_delta_heavy_atoms=(6, 20),
        target_delta_cycle_rank=(2, 2),
        retained_fraction_floor=0.45,
        interface_count_range=(1, 10),
        rewrite_mode="release_grow_and_construct_connected_topology",
        candidate_quota=3,
    ),
)


@dataclass(frozen=True)
class GenericTopologyMacroBatch:
    proposals: tuple[GenericCompleteProgramProposal, ...]
    telemetry: dict[str, Any]


def _created_attachment_interface_count(
    source: MolecularGraph, actions: tuple[dict[str, Any], ...]
) -> int:
    original = {int(slot) for slot in np.flatnonzero(is_element(source.atom_types))}
    created: set[int] = set()
    deleted: set[int] = set()
    interface: set[int] = set()
    for action in actions:
        rule = action["executor_rule"]
        payload = action["payload"]
        if rule == "atom_delete":
            deleted.add(int(payload["v"]))
        elif rule == "atom_insert":
            slot = int(payload["slot"])
            created.add(slot)
            for neighbor, _order in payload["neighbors"]:
                neighbor = int(neighbor)
                if neighbor in original and neighbor not in created and neighbor not in deleted:
                    interface.add(neighbor)
        elif rule == "bond_insert":
            left, right = int(payload["u"]), int(payload["v"])
            if left in created and right in original - deleted:
                interface.add(right)
            if right in created and left in original - deleted:
                interface.add(left)
    return len(interface)


def propose_generic_topology_macro_programs(
    source: MolecularGraph,
    *,
    seed: int,
    attempts_per_plan: int,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> GenericTopologyMacroBatch:
    """Compile the fixed route-free macro-plan allocation on one source."""

    if type(seed) is not int:
        raise TypeError("generic topology-macro seed must be an integer")
    if type(attempts_per_plan) is not int or attempts_per_plan < 1:
        raise ValueError("attempts_per_plan must be positive")
    if maximum_primitives != 32 or maximum_blocks != 8:
        raise ValueError("generic topology-macro support is frozen at 32 primitives/8 blocks")

    source_topology = topology(source)
    accepted: dict[str, GenericCompleteProgramProposal] = {}
    telemetry_rows: dict[str, Any] = {}
    for plan_index, plan in enumerate(TOPOLOGY_MACRO_PLANS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2207, plan_index]))
        plan_rows: dict[str, GenericCompleteProgramProposal] = {}
        failures: Counter[str] = Counter()
        for _attempt in range(attempts_per_plan):
            if len(plan_rows) >= plan.candidate_quota:
                break
            try:
                proposal = _compile_schedule(
                    source,
                    rng,
                    plan.families,
                    maximum_primitives=maximum_primitives,
                    maximum_blocks=maximum_blocks,
                )
            except (ValueError, RuntimeError) as error:
                failures[f"compile:{type(error).__name__}:{error}"] += 1
                continue
            endpoint_topology = topology(proposal.endpoint)
            delta_heavy = endpoint_topology["n_heavy"] - source_topology["n_heavy"]
            delta_cycle = endpoint_topology["cycle_rank"] - source_topology["cycle_rank"]
            retained_fraction = 1.0 - (
                int(proposal.actual_changes["deleted_original_atoms"]) / source_topology["n_heavy"]
            )
            interface_count = _created_attachment_interface_count(source, proposal.actions)
            if proposal.realized_scale != "large":
                failures[f"primitive_scale:{proposal.realized_scale}"] += 1
                continue
            if (
                not plan.target_delta_heavy_atoms[0]
                <= delta_heavy
                <= plan.target_delta_heavy_atoms[1]
            ):
                failures["delta_heavy_atoms"] += 1
                continue
            if (
                not plan.target_delta_cycle_rank[0]
                <= delta_cycle
                <= plan.target_delta_cycle_rank[1]
            ):
                failures["delta_cycle_rank"] += 1
                continue
            if retained_fraction < plan.retained_fraction_floor:
                failures["retained_fraction"] += 1
                continue
            if (
                not plan.interface_count_range[0]
                <= interface_count
                <= plan.interface_count_range[1]
            ):
                failures["interface_count"] += 1
                continue
            key = canonical_state_key(proposal.endpoint)
            if key in accepted or key in plan_rows:
                failures["canonical_endpoint_alias"] += 1
                continue
            plan_rows[key] = GenericCompleteProgramProposal(
                endpoint=proposal.endpoint,
                actions=proposal.actions,
                program=proposal.program,
                program_graph=proposal.program_graph,
                families=proposal.families,
                requested_scale="large",
                realized_scale="large",
                metadata={
                    **proposal.metadata,
                    "schema_version": "generic_topology_macro_composer_v2",
                    "macro_plan": plan.payload(),
                    "observed_macro_fields": {
                        "delta_heavy_atoms": delta_heavy,
                        "delta_cycle_rank": delta_cycle,
                        "retained_fraction": retained_fraction,
                        "created_attachment_interface_count": interface_count,
                    },
                },
                actual_changes=proposal.actual_changes,
            )
        accepted.update(plan_rows)
        telemetry_rows[plan.name] = {
            "requested_candidates": plan.candidate_quota,
            "accepted_candidates": len(plan_rows),
            "candidate_shortfall": plan.candidate_quota - len(plan_rows),
            "attempt_limit": attempts_per_plan,
            "failure_counts": dict(sorted(failures.items())),
        }

    proposals = tuple(
        sorted(
            accepted.values(),
            key=lambda row: (
                row.metadata["macro_plan"]["name"],
                row.endpoint_key,
            ),
        )
    )
    return GenericTopologyMacroBatch(
        proposals=proposals,
        telemetry={
            "schema_version": "generic_topology_macro_composer_telemetry_v2",
            "seed": seed,
            "attempts_per_plan": attempts_per_plan,
            "maximum_primitives": maximum_primitives,
            "maximum_blocks": maximum_blocks,
            "plans": telemetry_rows,
            "macro_plan_identity": identity([plan.payload() for plan in TOPOLOGY_MACRO_PLANS]),
            "unique_committed_endpoints": len(proposals),
            "trajectory_distilled_templates_loaded": 0,
            "route_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_teacher_endpoint_input": False,
            "runtime_objective_input": False,
        },
    )


__all__ = [
    "TOPOLOGY_MACRO_PLANS",
    "GenericTopologyMacroBatch",
    "GenericTopologyMacroPlan",
    "propose_generic_topology_macro_programs",
]
