"""Generic retained-interface composition with continuous free-feasibility headroom.

This module accepts a current molecular graph, deterministic seed, generic work
settings, and numeric benchmark-adapter thresholds.  It has no target, cell,
teacher, route, template, objective, or fitted input.  Exact candidates are
oversampled and ranked only by continuous free endpoint margins and generic
structural-plan margins.  Binary endpoint admission remains outside generation.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from math import exp, floor
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import QED, rdFingerprintGenerator

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer import (
    GenericCompleteProgramProposal,
)
from compose_v4.control.generic_retained_interface_composer_v3 import (
    RetainedInterfaceMacroPlan,
    RetainedInterfaceSourceFeatures,
    _compile_state_conditioned_schedule,
    _created_attachment_interface_count,
    retained_interface_source_features,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid
from compose_v4.rewrite.kernel import canonical_state_key

SCHEMA_VERSION = "generic_feasibility_headroom_composer_v4"
INTERFACE_ROLES = ("terminal", "linker", "ring", "junction")

MODE_SPECS: dict[str, dict[str, Any]] = {
    "direct_ring_growth": {
        "families": ("construct_substituted_ring",),
        "delta_heavy": (3, 7),
        "delta_cycle": 1,
    },
    "grow_then_close": {
        "families": ("segment_grow", "cycle_close"),
        "delta_heavy": (1, 7),
        "delta_cycle": 1,
    },
    "grow_then_ring": {
        "families": ("segment_grow", "construct_substituted_ring"),
        "delta_heavy": (4, 12),
        "delta_cycle": 1,
    },
    "replace_then_ring": {
        "families": ("segment_replace", "construct_substituted_ring"),
        "delta_heavy": (1, 12),
        "delta_cycle": 1,
    },
    "double_ring_growth": {
        "families": ("construct_substituted_ring", "construct_substituted_ring"),
        "delta_heavy": (6, 16),
        "delta_cycle": 2,
    },
    "grow_then_double_ring": {
        "families": (
            "segment_grow",
            "construct_substituted_ring",
            "construct_substituted_ring",
        ),
        "delta_heavy": (7, 18),
        "delta_cycle": 2,
    },
}


@dataclass(frozen=True)
class FreeFeasibilitySpec:
    """Allowed generic numeric endpoint constraints, never task identity."""

    similarity_minimum: float
    qed_minimum: float = 0.6
    sa_maximum: float = 4.0
    heavy_atom_maximum: int = 40

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.similarity_minimum) <= 1.0:
            raise ValueError("similarity minimum must be within zero and one")
        if not 0.0 <= float(self.qed_minimum) <= 1.0:
            raise ValueError("QED minimum must be within zero and one")
        if float(self.sa_maximum) <= 0.0:
            raise ValueError("SA maximum must be positive")
        if self.heavy_atom_maximum != 40:
            raise ValueError("v4 feasibility support is frozen at 40 heavy atoms")

    @property
    def retained_fraction_target(self) -> float:
        return 0.75 + 0.25 * float(self.similarity_minimum)

    def payload(self) -> dict[str, Any]:
        return {
            "similarity_minimum": float(self.similarity_minimum),
            "qed_minimum": float(self.qed_minimum),
            "sa_maximum": float(self.sa_maximum),
            "heavy_atom_maximum": self.heavy_atom_maximum,
            "retained_fraction_target": self.retained_fraction_target,
        }


@dataclass(frozen=True)
class FeasibilityHeadroomPlan:
    name: str
    particle_index: int
    mode: str
    interface_role: str
    families: tuple[str, ...]
    desired_delta_heavy_atoms: tuple[int, int]
    desired_delta_cycle_rank: int
    desired_retained_fraction: float
    desired_interface_count: tuple[int, int]
    allocation_headroom_score: float
    available_cycle_capacity: int
    load_vector: tuple[float, ...]
    compatibility: float
    allocation_weight: float
    preferred_slots: tuple[int, ...]
    attempt_budget: int
    candidate_quota: int

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "particle_index": self.particle_index,
            "mode": self.mode,
            "interface_role": self.interface_role,
            "families": list(self.families),
            "desired_delta_heavy_atoms": list(self.desired_delta_heavy_atoms),
            "desired_delta_cycle_rank": self.desired_delta_cycle_rank,
            "desired_retained_fraction": self.desired_retained_fraction,
            "desired_interface_count": list(self.desired_interface_count),
            "allocation_headroom_score": self.allocation_headroom_score,
            "available_cycle_capacity": self.available_cycle_capacity,
            "load_vector": list(self.load_vector),
            "compatibility": self.compatibility,
            "allocation_weight": self.allocation_weight,
            "preferred_slot_count": len(self.preferred_slots),
            "attempt_budget": self.attempt_budget,
            "candidate_quota": self.candidate_quota,
        }


@dataclass(frozen=True)
class FeasibilityHeadroomBatch:
    proposals: tuple[GenericCompleteProgramProposal, ...]
    telemetry: dict[str, Any]


def _source_free_features(
    source: MolecularGraph, feasibility: FreeFeasibilitySpec
) -> dict[str, Any]:
    smiles = molecular_graph_to_smiles(source)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("headroom composer received an invalid source molecule")
    qed = float(QED.qed(molecule))
    sa = float(sascorer.calculateScore(molecule))
    source_topology = topology(source)
    return {
        "current_similarity": 1.0,
        "current_similarity_headroom": 1.0 - feasibility.similarity_minimum,
        "current_qed": qed,
        "current_qed_margin": qed - feasibility.qed_minimum,
        "current_sa": sa,
        "current_sa_margin": feasibility.sa_maximum - sa,
        "heavy_atom_capacity": feasibility.heavy_atom_maximum
        - int(source_topology["n_heavy"]),
        "current_cycle_rank": int(source_topology["cycle_rank"]),
        "current_ring_systems": int(source_topology["n_ring_systems"]),
        "desired_retained_fraction": feasibility.retained_fraction_target,
        "desired_interface_count": [1, 3],
    }


def allocate_feasibility_headroom_plans(
    source: MolecularGraph,
    *,
    feasibility: FreeFeasibilitySpec,
    attempts_per_plan: int,
    candidate_quota_per_plan: int,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> tuple[
    RetainedInterfaceSourceFeatures,
    dict[str, Any],
    tuple[FeasibilityHeadroomPlan, ...],
    dict[str, Any],
]:
    """Allocate the frozen mode-role grid from continuous current-state margins."""

    if type(attempts_per_plan) is not int or attempts_per_plan < 1:
        raise ValueError("attempts per v4 plan must be positive")
    if type(candidate_quota_per_plan) is not int or candidate_quota_per_plan < 1:
        raise ValueError("candidate quota per v4 plan must be positive")
    if maximum_primitives != 32 or maximum_blocks != 8:
        raise ValueError("v4 work support is frozen at 32 primitives/8 blocks")
    graph_features = retained_interface_source_features(
        source,
        maximum_heavy_atoms=feasibility.heavy_atom_maximum,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
    )
    source_features = _source_free_features(source, feasibility)
    roles = dict(graph_features.interface_roles)
    # A continuous, task-independent work allocator.  It changes work allocation,
    # never support or endpoint admission.  Every feasible plan retains an explicit
    # exploration floor, while compatibility receives the remaining fixed budget.
    normalized_headrooms = (
        float(source_features["current_similarity_headroom"]),
        max(0.0, min(1.0, float(source_features["current_qed_margin"]))),
        max(0.0, min(1.0, float(source_features["current_sa_margin"]))),
        max(
            0.0,
            min(
                1.0,
                float(graph_features.available_heavy_capacity)
                / float(feasibility.heavy_atom_maximum),
            ),
        ),
    )
    available_cycle_capacity = min(
        2,
        graph_features.remaining_block_budget,
        graph_features.remaining_primitive_budget // 3,
    )
    role_availability = sum(bool(roles[role]) for role in INTERFACE_ROLES) / len(
        INTERFACE_ROLES
    )
    topology_headroom = 0.5 * (available_cycle_capacity / 2.0) + 0.5 * role_availability
    normalized_headrooms = (*normalized_headrooms, topology_headroom)
    allocation_headroom_score = sum(normalized_headrooms) / len(normalized_headrooms)
    source_features["normalized_allocation_headrooms"] = list(normalized_headrooms)
    source_features["allocation_headroom_score"] = allocation_headroom_score
    source_features["available_cycle_capacity"] = available_cycle_capacity
    plan_specs = []
    skipped: Counter[str] = Counter()
    for particle_index, (mode, role) in enumerate(
        (mode, role) for mode in MODE_SPECS for role in INTERFACE_ROLES
    ):
        spec = MODE_SPECS[mode]
        low, high = spec["delta_heavy"]
        if graph_features.available_heavy_capacity < low:
            skipped[f"capacity:{mode}"] += 1
            continue
        if not roles[role]:
            skipped[f"interface_role:{role}"] += 1
            continue
        families = tuple(spec["families"])
        if len(families) > maximum_blocks:
            skipped[f"block_budget:{mode}"] += 1
            continue
        heavy_load = high / 18.0
        replacement_load = float(mode.startswith("replace"))
        role_load = {
            "terminal": 0.20,
            "linker": 0.35,
            "ring": 0.50,
            "junction": 0.75,
        }[role]
        cycle_load = int(spec["delta_cycle"]) / 2.0
        load_vector = (
            min(1.0, 0.25 + 0.60 * heavy_load + 0.10 * replacement_load),
            min(1.0, 0.10 + 0.45 * heavy_load + 0.15 * cycle_load),
            min(1.0, 0.10 + 0.55 * heavy_load + 0.20 * replacement_load),
            min(1.0, high / feasibility.heavy_atom_maximum),
            min(1.0, 0.55 * cycle_load + 0.45 * role_load),
        )
        taus = (0.20, 0.20, 0.25, 0.20, 0.25)
        compatibility = 1.0
        for headroom, load, tau in zip(normalized_headrooms, load_vector, taus):
            value = max(-40.0, min(40.0, (headroom - load) / tau))
            compatibility *= 1.0 / (1.0 + exp(-value))
        plan_specs.append(
            {
                "name": f"{mode}__{role}",
                "particle_index": particle_index,
                "mode": mode,
                "interface_role": role,
                "families": families,
                "desired_delta_heavy_atoms": (
                    low,
                    min(high, graph_features.available_heavy_capacity),
                ),
                "desired_delta_cycle_rank": int(spec["delta_cycle"]),
                "desired_retained_fraction": feasibility.retained_fraction_target,
                "desired_interface_count": (1, 3),
                "allocation_headroom_score": allocation_headroom_score,
                "available_cycle_capacity": available_cycle_capacity,
                "load_vector": load_vector,
                "compatibility": compatibility,
                "preferred_slots": roles[role],
            }
        )

    def largest_remainder(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
        if not weights:
            return ()
        if total < len(weights):
            raise ValueError("v4 allocation budget cannot satisfy exploration floor")
        remaining = total - len(weights)
        exact = tuple(remaining * weight for weight in weights)
        values = [1 + floor(value) for value in exact]
        residual = total - sum(values)
        order = sorted(
            range(len(weights)), key=lambda index: (-(exact[index] % 1.0), index)
        )
        for index in order[:residual]:
            values[index] += 1
        return tuple(values)

    compatibility_sum = sum(row["compatibility"] for row in plan_specs)
    count = len(plan_specs)
    epsilon = 0.25
    weights = tuple(
        (
            epsilon / count + (1.0 - epsilon) * row["compatibility"] / compatibility_sum
            if compatibility_sum > 0
            else 1.0 / count
        )
        for row in plan_specs
    )
    attempt_budgets = largest_remainder(attempts_per_plan * count, weights)
    quota_budgets = largest_remainder(candidate_quota_per_plan * count, weights)
    plans = tuple(
        FeasibilityHeadroomPlan(
            **row,
            allocation_weight=weights[index],
            attempt_budget=attempt_budgets[index],
            candidate_quota=quota_budgets[index],
        )
        for index, row in enumerate(plan_specs)
    )
    return (
        graph_features,
        source_features,
        tuple(plans),
        {
            "predeclared_plan_count": len(MODE_SPECS) * len(INTERFACE_ROLES),
            "allocated_plan_count": len(plans),
            "skipped_plan_count": len(MODE_SPECS) * len(INTERFACE_ROLES) - len(plans),
            "skip_counts": dict(sorted(skipped.items())),
            "allocation_policy": {
                "name": "continuous_compatibility_largest_remainder",
                "headroom_vector": list(normalized_headrooms),
                "sigmoid_taus": [0.20, 0.20, 0.25, 0.20, 0.25],
                "uniform_exploration_fraction": epsilon,
                "attempt_budget_total": sum(attempt_budgets),
                "candidate_quota_total": sum(quota_budgets),
                "minimum_attempts_per_plan": min(attempt_budgets, default=0),
                "minimum_quota_per_plan": min(quota_budgets, default=0),
            },
        },
    )


def _v3_plan(plan: FeasibilityHeadroomPlan) -> RetainedInterfaceMacroPlan:
    return RetainedInterfaceMacroPlan(
        name=plan.name,
        particle_index=plan.particle_index,
        desired_heavy_band="headroom_allocated",
        desired_delta_heavy_atoms=plan.desired_delta_heavy_atoms,
        desired_delta_cycle_rank=plan.desired_delta_cycle_rank,
        mode=plan.mode,
        families=plan.families,
        initial_interface_role=plan.interface_role,
        initial_preferred_slots=plan.preferred_slots,
        retained_fraction_floor=plan.desired_retained_fraction,
        interface_count_range=plan.desired_interface_count,
        candidate_quota=plan.candidate_quota,
    )


def _endpoint_headrooms(
    *,
    source_fingerprint,
    endpoint: MolecularGraph,
    feasibility: FreeFeasibilitySpec,
    retained_fraction: float,
    interface_count: int,
    delta_cycle_rank: int,
    desired_cycle_rank: int,
    available_cycle_capacity: int,
) -> dict[str, Any]:
    smiles = molecular_graph_to_smiles(endpoint)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("exact v4 proposal did not convert to a molecule")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    similarity = float(
        DataStructs.TanimotoSimilarity(
            source_fingerprint, generator.GetFingerprint(molecule)
        )
    )
    qed = float(QED.qed(molecule))
    sa = float(sascorer.calculateScore(molecule))
    interface_margin = min(interface_count - 1, 3 - interface_count)
    values = {
        "similarity": similarity,
        "qed": qed,
        "sa": sa,
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "structurally_valid": bool(structurally_valid(smiles)),
        "connected": "." not in smiles,
        "endpoint_similarity_margin": similarity - feasibility.similarity_minimum,
        "endpoint_qed_margin": qed - feasibility.qed_minimum,
        "endpoint_sa_margin": feasibility.sa_maximum - sa,
        "heavy_atom_margin": feasibility.heavy_atom_maximum
        - int(molecule.GetNumHeavyAtoms()),
        "retained_fraction_margin": retained_fraction
        - feasibility.retained_fraction_target,
        "interface_count_margin": interface_margin,
        "cycle_capacity": available_cycle_capacity - desired_cycle_rank,
    }
    continuous = (
        values["endpoint_similarity_margin"],
        values["endpoint_qed_margin"],
        values["endpoint_sa_margin"] / feasibility.sa_maximum,
        values["heavy_atom_margin"] / feasibility.heavy_atom_maximum,
        values["retained_fraction_margin"],
        float(values["interface_count_margin"]),
        float(values["cycle_capacity"]),
    )
    values["continuous_minimum_margin"] = min(continuous)
    values["continuous_total_deficit"] = sum(max(0.0, -value) for value in continuous)
    values["binary_free_feasible"] = bool(
        values["connected"]
        and values["structurally_valid"]
        and values["endpoint_similarity_margin"] >= 0
        and values["endpoint_qed_margin"] >= 0
        and values["endpoint_sa_margin"] >= 0
        and values["heavy_atom_margin"] >= 0
    )
    return values


def free_endpoint_headrooms(
    source: MolecularGraph,
    endpoint: MolecularGraph,
    *,
    feasibility: FreeFeasibilitySpec,
    retained_fraction: float,
    interface_count: int,
    desired_cycle_rank: int,
    available_cycle_capacity: int,
) -> dict[str, Any]:
    """Return the common continuous endpoint schema without applying admission."""

    source_molecule = Chem.MolFromSmiles(molecular_graph_to_smiles(source))
    if source_molecule is None:
        raise ValueError("headroom composer received an invalid source molecule")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source_fingerprint = generator.GetFingerprint(source_molecule)
    source_topology = topology(source)
    endpoint_topology = topology(endpoint)
    return _endpoint_headrooms(
        source_fingerprint=source_fingerprint,
        endpoint=endpoint,
        feasibility=feasibility,
        retained_fraction=retained_fraction,
        interface_count=interface_count,
        delta_cycle_rank=int(
            endpoint_topology["cycle_rank"] - source_topology["cycle_rank"]
        ),
        desired_cycle_rank=desired_cycle_rank,
        available_cycle_capacity=available_cycle_capacity,
    )


def _proposal_rank(proposal: GenericCompleteProgramProposal) -> tuple:
    headrooms = proposal.metadata["endpoint_headrooms"]
    plan = proposal.metadata["macro_plan"]
    delta_heavy = proposal.metadata["observed_macro_fields"]["delta_heavy_atoms"]
    desired_midpoint = sum(plan["desired_delta_heavy_atoms"]) / 2.0
    return (
        float(headrooms["continuous_total_deficit"]),
        -float(headrooms["continuous_minimum_margin"]),
        abs(float(delta_heavy) - desired_midpoint),
        proposal.endpoint_key,
    )


def _compile_plan(
    source: MolecularGraph,
    *,
    seed: int,
    feasibility: FreeFeasibilitySpec,
    plan: FeasibilityHeadroomPlan,
    graph_features: RetainedInterfaceSourceFeatures,
    source_features: dict[str, Any],
    maximum_primitives: int,
    maximum_blocks: int,
) -> FeasibilityHeadroomBatch:
    source_smiles = molecular_graph_to_smiles(source)
    source_molecule = Chem.MolFromSmiles(source_smiles)
    if source_molecule is None:
        raise ValueError("headroom composer received an invalid source molecule")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source_fingerprint = generator.GetFingerprint(source_molecule)
    source_topology = topology(source)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 4401, plan.particle_index])
    )
    candidates: dict[str, GenericCompleteProgramProposal] = {}
    failures: Counter[str] = Counter()
    raw_compile_successes = 0
    v3_plan = _v3_plan(plan)
    for _attempt in range(plan.attempt_budget):
        try:
            proposal = _compile_state_conditioned_schedule(
                source,
                rng,
                v3_plan,
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
        interface_count = _created_attachment_interface_count(source, proposal.actions)
        if int(endpoint_topology["n_heavy"]) > feasibility.heavy_atom_maximum:
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
        key = canonical_state_key(proposal.endpoint)
        if key == canonical_state_key(source):
            failures["canonical_self_event"] += 1
            continue
        headrooms = _endpoint_headrooms(
            source_fingerprint=source_fingerprint,
            endpoint=proposal.endpoint,
            feasibility=feasibility,
            retained_fraction=retained_fraction,
            interface_count=interface_count,
            delta_cycle_rank=delta_cycle,
            desired_cycle_rank=plan.desired_delta_cycle_rank,
            available_cycle_capacity=plan.available_cycle_capacity,
        )
        enriched = GenericCompleteProgramProposal(
            endpoint=proposal.endpoint,
            actions=proposal.actions,
            program=proposal.program,
            program_graph=proposal.program_graph,
            families=proposal.families,
            requested_scale=proposal.requested_scale,
            realized_scale=proposal.realized_scale,
            metadata={
                **proposal.metadata,
                "schema_version": SCHEMA_VERSION,
                "macro_plan": {**plan.payload(), "identity": identity(plan.payload())},
                "state_allocator_inputs": {
                    "graph_features": graph_features.payload(),
                    "free_headroom_features": source_features,
                    "feasibility_thresholds": feasibility.payload(),
                },
                "observed_macro_fields": {
                    "delta_heavy_atoms": delta_heavy,
                    "delta_cycle_rank": delta_cycle,
                    "retained_fraction": retained_fraction,
                    "created_attachment_interface_count": interface_count,
                    "primitive_count": len(proposal.actions),
                    "block_count": len(proposal.program["blocks"]),
                },
                "endpoint_headrooms": headrooms,
                "endpoint_admission_applied_during_generation": False,
                "runtime_generic_constraint_thresholds_input": True,
                "runtime_task_cell_or_target_input": False,
                "runtime_teacher_action_endpoint_or_proximity_input": False,
                "runtime_route_or_template_input": False,
                "runtime_objective_or_docking_score_input": False,
            },
            actual_changes=proposal.actual_changes,
        )
        incumbent = candidates.get(key)
        if incumbent is None or _proposal_rank(enriched) < _proposal_rank(incumbent):
            candidates[key] = enriched
    ranked = tuple(
        sorted(candidates.values(), key=_proposal_rank)[: plan.candidate_quota]
    )
    return FeasibilityHeadroomBatch(
        proposals=ranked,
        telemetry={
            "schema_version": "generic_feasibility_headroom_plan_telemetry_v4",
            "seed": seed,
            "plan": plan.payload(),
            "plan_identity": identity(plan.payload()),
            "attempt_budget": plan.attempt_budget,
            "raw_compile_successes": raw_compile_successes,
            "support_matching_unique": len(candidates),
            "retained_candidates": len(ranked),
            "retained_binary_free_feasible": sum(
                bool(row.metadata["endpoint_headrooms"]["binary_free_feasible"])
                for row in ranked
            ),
            "failure_counts": dict(sorted(failures.items())),
            "endpoint_constraints_applied_during_generation": False,
            "headroom_ranking_applied_after_exact_generation": True,
            "route_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_teacher_action_endpoint_or_proximity_input": False,
            "runtime_objective_or_docking_score_input": False,
        },
    )


def propose_feasibility_headroom_plan(
    source: MolecularGraph,
    *,
    seed: int,
    feasibility: FreeFeasibilitySpec,
    plan_name: str,
    attempts_per_plan: int,
    candidate_quota_per_plan: int,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> FeasibilityHeadroomBatch:
    """Compile one durable source-only plan shard."""

    if type(seed) is not int:
        raise TypeError("v4 seed must be an integer")
    graph_features, source_features, plans, _allocation = (
        allocate_feasibility_headroom_plans(
            source,
            feasibility=feasibility,
            attempts_per_plan=attempts_per_plan,
            candidate_quota_per_plan=candidate_quota_per_plan,
            maximum_primitives=maximum_primitives,
            maximum_blocks=maximum_blocks,
        )
    )
    selected = tuple(plan for plan in plans if plan.name == plan_name)
    if len(selected) != 1:
        raise ValueError(f"v4 plan is not allocated exactly once: {plan_name}")
    return _compile_plan(
        source,
        seed=seed,
        feasibility=feasibility,
        plan=selected[0],
        graph_features=graph_features,
        source_features=source_features,
        maximum_primitives=maximum_primitives,
        maximum_blocks=maximum_blocks,
    )


def propose_feasibility_headroom_programs(
    source: MolecularGraph,
    *,
    seed: int,
    feasibility: FreeFeasibilitySpec,
    attempts_per_plan: int,
    candidate_quota_per_plan: int,
    maximum_primitives: int = 32,
    maximum_blocks: int = 8,
) -> FeasibilityHeadroomBatch:
    """Compile the full fixed mode-role grid and preserve plan-ranked order."""

    graph_features, source_features, plans, allocation = (
        allocate_feasibility_headroom_plans(
            source,
            feasibility=feasibility,
            attempts_per_plan=attempts_per_plan,
            candidate_quota_per_plan=candidate_quota_per_plan,
            maximum_primitives=maximum_primitives,
            maximum_blocks=maximum_blocks,
        )
    )
    rows = []
    telemetry = {}
    for plan in plans:
        batch = _compile_plan(
            source,
            seed=seed,
            feasibility=feasibility,
            plan=plan,
            graph_features=graph_features,
            source_features=source_features,
            maximum_primitives=maximum_primitives,
            maximum_blocks=maximum_blocks,
        )
        rows.extend(batch.proposals)
        telemetry[plan.name] = batch.telemetry
    return FeasibilityHeadroomBatch(
        proposals=tuple(rows),
        telemetry={
            "schema_version": "generic_feasibility_headroom_composer_telemetry_v4",
            "seed": seed,
            "feasibility": feasibility.payload(),
            "graph_features": graph_features.payload(),
            "source_headroom_features": source_features,
            "allocation": allocation,
            "plans": telemetry,
            "candidate_count_before_cross_plan_deduplication": len(rows),
            "endpoint_constraints_applied_during_generation": False,
            "headroom_ranking_applied_after_exact_generation": True,
            "route_templates_loaded": 0,
            "route_weights_loaded": 0,
            "fitted_weights_loaded": 0,
            "runtime_task_cell_or_target_input": False,
            "runtime_teacher_action_endpoint_or_proximity_input": False,
            "runtime_objective_or_docking_score_input": False,
        },
    )


__all__ = [
    "INTERFACE_ROLES",
    "MODE_SPECS",
    "FeasibilityHeadroomBatch",
    "FeasibilityHeadroomPlan",
    "FreeFeasibilitySpec",
    "allocate_feasibility_headroom_plans",
    "free_endpoint_headrooms",
    "propose_feasibility_headroom_plan",
    "propose_feasibility_headroom_programs",
]
