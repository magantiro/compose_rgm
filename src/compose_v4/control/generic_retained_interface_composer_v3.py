"""Route-free state-conditioned retained-interface macro composition.

The allocator consumes only the current molecular graph, a deterministic seed,
and frozen representation/work limits.  It allocates a balanced grid of generic
growth particles across heavy-atom and cycle-rank bands, binds their first module
to structural retained-interface roles, and rebinds every later module on the
exact current successor.  No task, Fiber threshold, objective, route, template,
teacher endpoint, or fitted weight enters this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    changed_input_sites,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.generic_complete_program_composer import (
    GenericCompleteProgramProposal,
    primitive_scale,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.control.progressive_structured_sampler import progressive_module
from compose_v4.rewrite.kernel import canonical_state_key

HEAVY_GROWTH_BANDS: dict[str, tuple[int, int]] = {
    "small": (3, 6),
    "medium": (7, 11),
    "large": (12, 16),
}
CYCLE_GAIN_GRID = (1, 2, 3)
MACRO_MODES = ("grow", "replace", "open_grow")
RETAINED_FRACTION_FLOOR = 0.8
INTERFACE_COUNT_RANGE = (1, 4)


@dataclass(frozen=True)
class RetainedInterfaceSourceFeatures:
    """Task-independent graph state used by the allocator."""

    n_heavy: int
    available_heavy_capacity: int
    cycle_rank: int
    ring_systems: int
    interface_roles: tuple[tuple[str, tuple[int, ...]], ...]
    replaceable_pendant_boundaries: int
    remaining_primitive_budget: int
    remaining_block_budget: int

    def payload(self) -> dict[str, Any]:
        return {
            "n_heavy": self.n_heavy,
            "available_heavy_capacity": self.available_heavy_capacity,
            "cycle_rank": self.cycle_rank,
            "ring_systems": self.ring_systems,
            "interface_roles": {
                role: list(slots) for role, slots in self.interface_roles
            },
            "replaceable_pendant_boundaries": self.replaceable_pendant_boundaries,
            "remaining_primitive_budget": self.remaining_primitive_budget,
            "remaining_block_budget": self.remaining_block_budget,
        }


@dataclass(frozen=True)
class RetainedInterfaceMacroPlan:
    """One graph-derived allocation of a predeclared generic grid particle."""

    name: str
    particle_index: int
    desired_heavy_band: str
    desired_delta_heavy_atoms: tuple[int, int]
    desired_delta_cycle_rank: int
    mode: str
    families: tuple[str, ...]
    initial_interface_role: str
    initial_preferred_slots: tuple[int, ...]
    retained_fraction_floor: float
    interface_count_range: tuple[int, int]
    candidate_quota: int

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "particle_index": self.particle_index,
            "desired_heavy_band": self.desired_heavy_band,
            "desired_delta_heavy_atoms": list(self.desired_delta_heavy_atoms),
            "desired_delta_cycle_rank": self.desired_delta_cycle_rank,
            "mode": self.mode,
            "families": list(self.families),
            "initial_interface_role": self.initial_interface_role,
            "initial_preferred_slot_count": len(self.initial_preferred_slots),
            "retained_fraction_floor": self.retained_fraction_floor,
            "interface_count_range": list(self.interface_count_range),
            "candidate_quota": self.candidate_quota,
        }


@dataclass(frozen=True)
class RetainedInterfaceMacroBatch:
    proposals: tuple[GenericCompleteProgramProposal, ...]
    telemetry: dict[str, Any]


def _source_graph(source: MolecularGraph) -> nx.Graph:
    live = {int(slot) for slot in np.flatnonzero(is_element(source.atom_types))}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(live))
    graph.add_edges_from(
        (left, right)
        for left in sorted(live)
        for right in sorted(live)
        if left < right and int(source.bonds[left, right]) > 0
    )
    return graph


def retained_interface_source_features(
    source: MolecularGraph,
    *,
    maximum_heavy_atoms: int = 40,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> RetainedInterfaceSourceFeatures:
    """Factor generic retained-interface roles and remaining support from a graph."""

    observed = topology(source)
    graph = _source_graph(source)
    cycle_slots = {slot for cycle in nx.cycle_basis(graph) for slot in cycle}
    role_slots: dict[str, list[int]] = {
        "ring": [],
        "linker": [],
        "terminal": [],
        "junction": [],
    }
    for slot in sorted(graph.nodes):
        degree = int(graph.degree[slot])
        if slot in cycle_slots:
            role = "ring"
        elif degree <= 1:
            role = "terminal"
        elif degree == 2:
            role = "linker"
        else:
            role = "junction"
        role_slots[role].append(slot)

    pendant_boundaries = 0
    for left, right in nx.bridges(graph):
        reduced = graph.copy()
        reduced.remove_edge(left, right)
        components = [len(component) for component in nx.connected_components(reduced)]
        if components and min(components) <= 8:
            pendant_boundaries += 1

    return RetainedInterfaceSourceFeatures(
        n_heavy=int(observed["n_heavy"]),
        available_heavy_capacity=max(0, maximum_heavy_atoms - int(observed["n_heavy"])),
        cycle_rank=int(observed["cycle_rank"]),
        ring_systems=int(observed["n_ring_systems"]),
        interface_roles=tuple(
            (role, tuple(role_slots[role])) for role in sorted(role_slots)
        ),
        replaceable_pendant_boundaries=pendant_boundaries,
        remaining_primitive_budget=maximum_primitives,
        remaining_block_budget=maximum_blocks,
    )


def _families_for_particle(
    *, heavy_band: str, cycle_gain: int, mode: str
) -> tuple[str, ...]:
    growth_prefix = {
        "small": (),
        "medium": ("segment_grow",),
        "large": ("segment_grow", "segment_grow"),
    }[heavy_band]
    closures = ("construct_substituted_ring",) * cycle_gain
    if mode == "grow":
        return (*growth_prefix, *closures)
    if mode == "replace":
        return (
            "segment_replace",
            *(growth_prefix[1:] if growth_prefix else ()),
            *closures,
        )
    if mode == "open_grow":
        return (
            "cycle_open",
            *growth_prefix,
            "construct_substituted_ring",
            *closures,
        )
    raise ValueError(f"unknown retained-interface macro mode: {mode}")


def allocate_retained_interface_macro_plans(
    source: MolecularGraph,
    *,
    candidate_quota_per_particle: int,
    maximum_heavy_atoms: int = 40,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> tuple[
    RetainedInterfaceSourceFeatures, tuple[RetainedInterfaceMacroPlan, ...], dict
]:
    """Allocate the fixed balanced particle grid from current graph features only."""

    if (
        type(candidate_quota_per_particle) is not int
        or candidate_quota_per_particle < 1
    ):
        raise ValueError("candidate quota per particle must be positive")
    if maximum_heavy_atoms != 40 or maximum_primitives != 32 or maximum_blocks != 8:
        raise ValueError("v3 support is frozen at 40 atoms/32 primitives/8 blocks")
    features = retained_interface_source_features(
        source,
        maximum_heavy_atoms=maximum_heavy_atoms,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
    )
    role_map = dict(features.interface_roles)
    role_preferences = {
        "grow": ("ring", "linker", "terminal", "junction"),
        "replace": ("terminal", "linker", "junction", "ring"),
        "open_grow": ("ring", "junction", "linker", "terminal"),
    }
    plans = []
    skips: Counter[str] = Counter()
    for particle_index, (heavy_band, cycle_gain, mode) in enumerate(
        product(HEAVY_GROWTH_BANDS, CYCLE_GAIN_GRID, MACRO_MODES)
    ):
        low, high = HEAVY_GROWTH_BANDS[heavy_band]
        if features.available_heavy_capacity < low:
            skips[f"capacity:{heavy_band}"] += 1
            continue
        if mode == "replace" and features.replaceable_pendant_boundaries == 0:
            skips["mode:replace_unavailable"] += 1
            continue
        if mode == "open_grow" and features.cycle_rank == 0:
            skips["mode:open_grow_unavailable"] += 1
            continue
        families = _families_for_particle(
            heavy_band=heavy_band, cycle_gain=cycle_gain, mode=mode
        )
        if len(families) > features.remaining_block_budget:
            skips["block_budget"] += 1
            continue
        available_roles = [role for role in role_preferences[mode] if role_map[role]]
        if not available_roles:
            skips[f"mode:{mode}_no_retained_interface"] += 1
            continue
        role = available_roles[particle_index % len(available_roles)]
        target = (low, min(high, features.available_heavy_capacity))
        plans.append(
            RetainedInterfaceMacroPlan(
                name=f"{mode}_{heavy_band}_h_c{cycle_gain}_{role}",
                particle_index=particle_index,
                desired_heavy_band=heavy_band,
                desired_delta_heavy_atoms=target,
                desired_delta_cycle_rank=cycle_gain,
                mode=mode,
                families=families,
                initial_interface_role=role,
                initial_preferred_slots=role_map[role],
                retained_fraction_floor=RETAINED_FRACTION_FLOOR,
                interface_count_range=INTERFACE_COUNT_RANGE,
                candidate_quota=candidate_quota_per_particle,
            )
        )
    return (
        features,
        tuple(plans),
        {
            "predeclared_particle_count": (
                len(HEAVY_GROWTH_BANDS) * len(CYCLE_GAIN_GRID) * len(MACRO_MODES)
            ),
            "allocated_particle_count": len(plans),
            "skipped_particle_count": (
                len(HEAVY_GROWTH_BANDS) * len(CYCLE_GAIN_GRID) * len(MACRO_MODES)
                - len(plans)
            ),
            "skip_counts": dict(sorted(skips.items())),
        },
    )


def _compile_state_conditioned_schedule(
    source: MolecularGraph,
    rng: np.random.Generator,
    plan: RetainedInterfaceMacroPlan,
    *,
    maximum_primitives: int,
    maximum_blocks: int,
) -> GenericCompleteProgramProposal:
    current = source
    stages: list[dict[str, Any]] = []
    modules: list[dict[str, Any]] = []
    preferred = frozenset(plan.initial_preferred_slots)
    for family in plan.families:
        product_graph, stage = progressive_module(
            current, rng, family, preferred=preferred
        )
        candidate_program, _ = extract_program(source, [*stages, stage])
        if len(candidate_program.marks) > maximum_primitives:
            raise ValueError("retained-interface program exceeds primitive support")
        if len(candidate_program.blocks) > maximum_blocks:
            raise ValueError("retained-interface program exceeds block support")
        stages.append(stage)
        modules.append(
            {
                "family": family,
                "primitive_edits": len(stage["actions"]),
                "parameters": stage["parameters"],
            }
        )
        current = product_graph
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
        raise RuntimeError("retained-interface composition changed on exact replay")
    actions = tuple(receipt["actions"])
    realized_scale = primitive_scale(len(actions))
    return GenericCompleteProgramProposal(
        endpoint=endpoint,
        actions=actions,
        program=program.payload(),
        program_graph=graph.payload(),
        families=plan.families,
        requested_scale=realized_scale,
        realized_scale=realized_scale,
        metadata={
            "schema_version": "generic_retained_interface_composer_v3",
            "modules": modules,
            "completed_module_count": len(modules),
            "created_handle_dependencies": len(graph.dependencies),
            "serialization_edges": len(graph.serialization_edges),
            "initial_stored_complete_routes": 0,
            "source_library_rows_loaded": 0,
            "trajectory_distilled_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "intermediate_task_evaluations": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_requested_delta_input": False,
            "runtime_fiber_input": False,
            "runtime_teacher_endpoint_input": False,
            "runtime_objective_input": False,
        },
        actual_changes=changed_input_sites(source, endpoint, actions),
    )


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
                if (
                    neighbor in original
                    and neighbor not in created
                    and neighbor not in deleted
                ):
                    interface.add(neighbor)
        elif rule == "bond_insert":
            left, right = int(payload["u"]), int(payload["v"])
            if left in created and right in original - deleted:
                interface.add(right)
            if right in created and left in original - deleted:
                interface.add(left)
    return len(interface)


def propose_generic_retained_interface_programs(
    source: MolecularGraph,
    *,
    seed: int,
    attempts_per_particle: int,
    candidate_quota_per_particle: int,
    maximum_heavy_atoms: int = 40,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> RetainedInterfaceMacroBatch:
    """Compile a source-state-conditioned, balanced route-free macro grid."""

    if type(seed) is not int:
        raise TypeError("retained-interface composer seed must be an integer")
    if type(attempts_per_particle) is not int or attempts_per_particle < 1:
        raise ValueError("attempts per particle must be positive")
    features, plans, allocation = allocate_retained_interface_macro_plans(
        source,
        candidate_quota_per_particle=candidate_quota_per_particle,
        maximum_heavy_atoms=maximum_heavy_atoms,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
    )
    return _propose_allocated_plans(
        source,
        seed=seed,
        attempts_per_particle=attempts_per_particle,
        candidate_quota_per_particle=candidate_quota_per_particle,
        maximum_heavy_atoms=maximum_heavy_atoms,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
        features=features,
        plans=plans,
        allocation=allocation,
    )


def _propose_allocated_plans(
    source: MolecularGraph,
    *,
    seed: int,
    attempts_per_particle: int,
    candidate_quota_per_particle: int,
    maximum_heavy_atoms: int,
    maximum_primitives: int,
    maximum_blocks: int,
    features: RetainedInterfaceSourceFeatures,
    plans: tuple[RetainedInterfaceMacroPlan, ...],
    allocation: dict[str, Any],
) -> RetainedInterfaceMacroBatch:
    source_topology = topology(source)
    source_key = canonical_state_key(source)
    accepted: dict[str, GenericCompleteProgramProposal] = {}
    telemetry_rows: dict[str, Any] = {}
    raw_compile_successes = 0
    for plan in plans:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 3301, plan.particle_index])
        )
        plan_rows: dict[str, GenericCompleteProgramProposal] = {}
        failures: Counter[str] = Counter()
        for _attempt in range(attempts_per_particle):
            if len(plan_rows) >= plan.candidate_quota:
                break
            try:
                proposal = _compile_state_conditioned_schedule(
                    source,
                    rng,
                    plan,
                    maximum_primitives=maximum_primitives,
                    maximum_blocks=maximum_blocks,
                )
            except (ValueError, RuntimeError) as error:
                failures[f"compile:{type(error).__name__}:{error}"] += 1
                continue
            raw_compile_successes += 1
            endpoint_topology = topology(proposal.endpoint)
            delta_heavy = int(endpoint_topology["n_heavy"] - source_topology["n_heavy"])
            delta_cycle = int(
                endpoint_topology["cycle_rank"] - source_topology["cycle_rank"]
            )
            retained_fraction = 1.0 - (
                int(proposal.actual_changes["deleted_original_atoms"])
                / max(1, int(source_topology["n_heavy"]))
            )
            interface_count = _created_attachment_interface_count(
                source, proposal.actions
            )
            primitive_count = len(proposal.actions)
            block_count = len(proposal.program["blocks"])
            if int(endpoint_topology["n_heavy"]) > maximum_heavy_atoms:
                failures["heavy_atom_ceiling"] += 1
                continue
            if not (
                plan.desired_delta_heavy_atoms[0]
                <= delta_heavy
                <= plan.desired_delta_heavy_atoms[1]
            ):
                failures["desired_delta_heavy_atoms"] += 1
                continue
            if delta_cycle != plan.desired_delta_cycle_rank:
                failures["desired_delta_cycle_rank"] += 1
                continue
            if retained_fraction < plan.retained_fraction_floor:
                failures["retained_fraction"] += 1
                continue
            if not (
                plan.interface_count_range[0]
                <= interface_count
                <= plan.interface_count_range[1]
            ):
                failures["created_attachment_interface_count"] += 1
                continue
            if primitive_count > maximum_primitives or block_count > maximum_blocks:
                raise RuntimeError("exact proposal escaped frozen work support")
            key = canonical_state_key(proposal.endpoint)
            if key == source_key:
                failures["canonical_self_event"] += 1
                continue
            if key in accepted or key in plan_rows:
                failures["canonical_endpoint_alias"] += 1
                continue
            plan_rows[key] = GenericCompleteProgramProposal(
                endpoint=proposal.endpoint,
                actions=proposal.actions,
                program=proposal.program,
                program_graph=proposal.program_graph,
                families=proposal.families,
                requested_scale=proposal.requested_scale,
                realized_scale=proposal.realized_scale,
                metadata={
                    **proposal.metadata,
                    "macro_plan": plan.payload(),
                    "state_allocator_inputs": features.payload(),
                    "observed_macro_fields": {
                        "delta_heavy_atoms": delta_heavy,
                        "delta_cycle_rank": delta_cycle,
                        "retained_fraction": retained_fraction,
                        "created_attachment_interface_count": interface_count,
                        "primitive_count": primitive_count,
                        "block_count": block_count,
                    },
                },
                actual_changes=proposal.actual_changes,
            )
        accepted.update(plan_rows)
        telemetry_rows[plan.name] = {
            "particle": plan.payload(),
            "requested_candidates": plan.candidate_quota,
            "accepted_candidates": len(plan_rows),
            "candidate_shortfall": plan.candidate_quota - len(plan_rows),
            "attempt_limit": attempts_per_particle,
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
    grid_payload = {
        "heavy_growth_bands": {
            name: list(bounds) for name, bounds in HEAVY_GROWTH_BANDS.items()
        },
        "cycle_gain_grid": list(CYCLE_GAIN_GRID),
        "macro_modes": list(MACRO_MODES),
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "interface_count_range": list(INTERFACE_COUNT_RANGE),
    }
    return RetainedInterfaceMacroBatch(
        proposals=proposals,
        telemetry={
            "schema_version": "generic_retained_interface_composer_telemetry_v3",
            "seed": seed,
            "attempts_per_particle": attempts_per_particle,
            "candidate_quota_per_particle": candidate_quota_per_particle,
            "maximum_heavy_atoms": maximum_heavy_atoms,
            "maximum_primitives": maximum_primitives,
            "maximum_blocks": maximum_blocks,
            "source_features": features.payload(),
            "allocation": allocation,
            "particles": telemetry_rows,
            "raw_compile_successes": raw_compile_successes,
            "exact_unique_candidates": len(proposals),
            "balanced_grid_identity": identity(grid_payload),
            "allocated_plan_identity": identity([plan.payload() for plan in plans]),
            "trajectory_distilled_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_requested_delta_input": False,
            "runtime_fiber_input": False,
            "runtime_teacher_endpoint_input": False,
            "runtime_objective_input": False,
        },
    )


def propose_generic_retained_interface_particle(
    source: MolecularGraph,
    *,
    seed: int,
    particle_name: str,
    attempts_per_particle: int,
    candidate_quota_per_particle: int,
    maximum_heavy_atoms: int = 40,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> RetainedInterfaceMacroBatch:
    """Compile one deterministic particle for resumable process-level sharding."""

    if type(seed) is not int:
        raise TypeError("retained-interface composer seed must be an integer")
    if type(particle_name) is not str or not particle_name:
        raise ValueError("particle name must be nonempty")
    if type(attempts_per_particle) is not int or attempts_per_particle < 1:
        raise ValueError("attempts per particle must be positive")
    features, plans, allocation = allocate_retained_interface_macro_plans(
        source,
        candidate_quota_per_particle=candidate_quota_per_particle,
        maximum_heavy_atoms=maximum_heavy_atoms,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
    )
    selected = tuple(plan for plan in plans if plan.name == particle_name)
    if len(selected) != 1:
        raise ValueError(
            f"particle is not allocated exactly once on this source: {particle_name}"
        )
    return _propose_allocated_plans(
        source,
        seed=seed,
        attempts_per_particle=attempts_per_particle,
        candidate_quota_per_particle=candidate_quota_per_particle,
        maximum_heavy_atoms=maximum_heavy_atoms,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
        features=features,
        plans=selected,
        allocation=allocation,
    )


__all__ = [
    "CYCLE_GAIN_GRID",
    "HEAVY_GROWTH_BANDS",
    "INTERFACE_COUNT_RANGE",
    "MACRO_MODES",
    "RETAINED_FRACTION_FLOOR",
    "RetainedInterfaceMacroBatch",
    "RetainedInterfaceMacroPlan",
    "RetainedInterfaceSourceFeatures",
    "allocate_retained_interface_macro_plans",
    "propose_generic_retained_interface_particle",
    "propose_generic_retained_interface_programs",
    "retained_interface_source_features",
]
