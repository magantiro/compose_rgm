"""Independent complete oracle for bounded aromatic-cycle-open audits.

The component-factored prototype solves each resonance-invariant aromatic
component with a purpose-built recursive enumerator.  This module provides a
structurally independent reference: one mixed-integer feasibility problem over
all semantic aromatic edges in the exact persistent-slot state.  Repeated
no-good constraints enumerate every binary single/double assignment whose
double-bond incidence matches the fixed-charge, fixed-hydrogen source.

Every candidate is rechecked with the production validity, connectivity, and
canonical molecular-identity predicates.  Solver output is rounded only after
binary closeness and the integer incidence equations have been verified.  An
explicit assignment cap fails loudly; it never truncates the represented
fiber.

This is validation infrastructure.  It does not alter production actions,
executors, legal fibers, corpora, checkpoints, or training authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    AromaticCycleOpenAliasOverflow,
    KekuleAliasEnumeration,
)
from compose_v4.rewrite.kernel import canonical_state_key

GLOBAL_MILP_ORACLE_STATUS = "NON_AUTHORIZING_COMPLETE_GLOBAL_KEKULE_MILP_ORACLE"


class GlobalKekuleMilpError(RuntimeError):
    """The independent bounded MILP oracle could not prove a complete result."""


@dataclass(frozen=True)
class GlobalKekuleMilpDiagnostics:
    """Deterministic work evidence for one completed global enumeration."""

    semantic_aromatic_edge_count: int
    constrained_vertex_count: int
    feasible_assignment_count: int
    preserving_alias_count: int
    solver_call_count: int


@dataclass(frozen=True)
class GlobalKekuleMilpResult:
    """Complete preserving aliases plus independently recorded solver work."""

    enumeration: KekuleAliasEnumeration
    diagnostics: GlobalKekuleMilpDiagnostics


def _semantic_aromatic_edges(state: MolecularGraph) -> tuple[tuple[int, int], ...]:
    perceived = resonance_invariant_bond_classes(state)
    real_slots = tuple(
        int(slot) for slot in np.flatnonzero(is_element(state.atom_types))
    )
    return tuple(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    )


def _exact_state_key(state: MolecularGraph) -> tuple:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def _state_with_global_orders(
    state: MolecularGraph,
    edges: tuple[tuple[int, int], ...],
    orders: tuple[int, ...],
) -> MolecularGraph:
    if len(edges) != len(orders):
        raise ValueError("global aromatic edge and bond-order lengths differ")
    bonds = state.bonds.copy()
    for (left, right), order in zip(edges, orders, strict=True):
        bonds[left, right] = bonds[right, left] = int(order)
    return MolecularGraph(
        state.atom_types.copy(),
        state.formal_charges.copy(),
        state.implicit_h_counts.copy(),
        bonds,
    )


def _incidence_problem(
    state: MolecularGraph,
    edges: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    vertices = tuple(sorted({vertex for edge in edges for vertex in edge}))
    offsets = {vertex: index for index, vertex in enumerate(vertices)}
    incidence = np.zeros((len(vertices), len(edges)), dtype=np.int64)
    demand = np.zeros(len(vertices), dtype=np.int64)
    for edge_index, (left, right) in enumerate(edges):
        incidence[offsets[left], edge_index] = 1
        incidence[offsets[right], edge_index] = 1
        if int(state.bonds[left, right]) == BOND_DOUBLE:
            demand[offsets[left]] += 1
            demand[offsets[right]] += 1
    return incidence, demand


def _verified_binary_solution(
    values: np.ndarray, incidence: np.ndarray, demand: np.ndarray
) -> np.ndarray:
    if values.shape != (incidence.shape[1],) or not np.isfinite(values).all():
        raise GlobalKekuleMilpError(
            "MILP returned malformed or nonfinite decision values"
        )
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=1e-7, rtol=0.0):
        raise GlobalKekuleMilpError("MILP returned a nonintegral decision vector")
    if np.any((rounded < 0) | (rounded > 1)):
        raise GlobalKekuleMilpError("MILP returned a value outside the binary domain")
    if not np.array_equal(incidence @ rounded, demand):
        raise GlobalKekuleMilpError(
            "rounded MILP solution violates the exact incidence equations"
        )
    return rounded


def enumerate_global_milp_kekule_aliases(
    state: MolecularGraph,
    *,
    maximum_assignments: int = 4096,
) -> GlobalKekuleMilpResult:
    """Enumerate every bounded slot-labeled fixed-charge/H Kekule lowering.

    A binary variable denotes whether one semantic aromatic edge is double.
    Its per-vertex incidence is fixed to that of the valid source lowering.
    This is the same mathematical fixed-charge/H condition used by the
    component formulation, solved here as one global MILP without aromatic
    component decomposition.
    """

    if type(maximum_assignments) is not int or maximum_assignments <= 0:
        raise ValueError("maximum_assignments must be a positive integer")
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("global Kekule enumeration requires a valid connected source")

    source_key = canonical_state_key(state)
    edges = _semantic_aromatic_edges(state)
    if not edges:
        enumeration = KekuleAliasEnumeration(
            source_key=source_key,
            aromatic_edges=(),
            raw_structure_count=0,
            aliases=(state,),
        )
        diagnostics = GlobalKekuleMilpDiagnostics(
            semantic_aromatic_edge_count=0,
            constrained_vertex_count=0,
            feasible_assignment_count=0,
            preserving_alias_count=1,
            solver_call_count=0,
        )
        return GlobalKekuleMilpResult(enumeration=enumeration, diagnostics=diagnostics)

    source_orders = tuple(int(state.bonds[edge]) for edge in edges)
    if any(order not in {BOND_SINGLE, BOND_DOUBLE} for order in source_orders):
        enumeration = KekuleAliasEnumeration(
            source_key=source_key,
            aromatic_edges=edges,
            raw_structure_count=0,
            aliases=(),
        )
        diagnostics = GlobalKekuleMilpDiagnostics(
            semantic_aromatic_edge_count=len(edges),
            constrained_vertex_count=len({vertex for edge in edges for vertex in edge}),
            feasible_assignment_count=0,
            preserving_alias_count=0,
            solver_call_count=0,
        )
        return GlobalKekuleMilpResult(enumeration=enumeration, diagnostics=diagnostics)

    incidence, demand = _incidence_problem(state, edges)
    constraints: list[LinearConstraint] = [LinearConstraint(incidence, demand, demand)]
    objective = np.zeros(len(edges), dtype=np.float64)
    integrality = np.ones(len(edges), dtype=np.int8)
    bounds = Bounds(np.zeros(len(edges)), np.ones(len(edges)))
    assignments: set[tuple[int, ...]] = set()
    aliases_by_key: dict[tuple, MolecularGraph] = {}
    solver_calls = 0

    while True:
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"presolve": True},
        )
        solver_calls += 1
        if result.status == 2:
            break
        if not result.success or result.x is None:
            raise GlobalKekuleMilpError(
                f"MILP did not prove feasibility or exhaustion: status={result.status}, "
                f"message={result.message!r}"
            )
        binary = _verified_binary_solution(result.x, incidence, demand)
        assignment = tuple(int(value) for value in binary)
        if assignment in assignments:
            raise GlobalKekuleMilpError("MILP repeated a blocked assignment")
        assignments.add(assignment)
        if len(assignments) > maximum_assignments:
            raise AromaticCycleOpenAliasOverflow(
                maximum_aliases=maximum_assignments,
                observed_structures=len(assignments),
            )

        orders = tuple(BOND_DOUBLE if value else BOND_SINGLE for value in assignment)
        alias = _state_with_global_orders(state, edges, orders)
        if (
            is_valid_state(alias)
            and is_connected_or_null(alias)
            and canonical_state_key(alias) == source_key
        ):
            aliases_by_key.setdefault(_exact_state_key(alias), alias)

        # Exclude exactly this binary vector.  For ones S and zeros Z,
        # sum_{i in S}(1-x_i) + sum_{i in Z}x_i >= 1.
        no_good = np.where(binary == 1, -1.0, 1.0)
        lower = 1 - int(binary.sum())
        constraints.append(LinearConstraint(no_good, lower, np.inf))

    aliases = tuple(aliases_by_key[key] for key in sorted(aliases_by_key))
    enumeration = KekuleAliasEnumeration(
        source_key=source_key,
        aromatic_edges=edges,
        raw_structure_count=len(assignments),
        aliases=aliases,
    )
    diagnostics = GlobalKekuleMilpDiagnostics(
        semantic_aromatic_edge_count=len(edges),
        constrained_vertex_count=incidence.shape[0],
        feasible_assignment_count=len(assignments),
        preserving_alias_count=len(aliases),
        solver_call_count=solver_calls,
    )
    return GlobalKekuleMilpResult(enumeration=enumeration, diagnostics=diagnostics)


__all__ = [
    "GLOBAL_MILP_ORACLE_STATUS",
    "GlobalKekuleMilpDiagnostics",
    "GlobalKekuleMilpError",
    "GlobalKekuleMilpResult",
    "enumerate_global_milp_kekule_aliases",
]
