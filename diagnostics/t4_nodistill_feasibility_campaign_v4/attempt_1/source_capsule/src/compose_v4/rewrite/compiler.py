"""Compile complete molecules into legal micro-rewrite traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import empty_molecular_graph, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem, default_rewrite_system
from compose_v4.rewrite.operators import (
    AtomInsert,
    BondInsert,
    BondReorder,
    MICRO_BOND_CLASSES,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace, execute_trace, invert_trace


class TraceCompilationError(ValueError):
    """Raised when no validity-closed micro trace can realize a target."""


@dataclass(frozen=True)
class Traversal:
    root: int
    order: tuple[int, ...]
    parent: dict[int, int | None]


def _bond_cost(order: int) -> int:
    return int(BOND_CLASS_TO_H_CHANGE[int(order)])


def _real_slots(target: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))


def _connected(target: MolecularGraph, real_slots: tuple[int, ...]) -> bool:
    if not real_slots:
        return True
    real = set(real_slots)
    seen = {real_slots[0]}
    stack = [real_slots[0]]
    while stack:
        v = stack.pop()
        for u in np.flatnonzero(target.bonds[v] != 0):
            u = int(u)
            if u in real and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == real


def _depth_first_traversal(
    target: MolecularGraph,
    root: int,
    rng: np.random.Generator | None = None,
) -> Traversal:
    parent: dict[int, int | None] = {root: None}
    order: list[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        neighbors = [
            int(u)
            for u in np.flatnonzero(target.bonds[v] != 0)
            if int(u) not in parent
        ]
        if rng is None:
            neighbors.sort(
                key=lambda u: (_bond_cost(int(target.bonds[v, u])), -u)
            )
        else:
            rng.shuffle(neighbors)
        for u in neighbors:
            if u in parent:
                continue
            parent[u] = v
            stack.append(u)
    return Traversal(root=root, order=tuple(order), parent=parent)


def _same_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def _compile_with_traversal(
    target: MolecularGraph,
    traversal: Traversal,
    runtime: RewriteSystem,
    rng: np.random.Generator | None = None,
    defer_bond_orders: bool = False,
) -> RewriteTrace:
    source = empty_molecular_graph(target.n_atoms)
    state = source
    steps: list[RewriteStep] = []
    tree_edges: set[tuple[int, int]] = set()

    for v in traversal.order:
        parent = traversal.parent[v]
        neighbors: tuple[tuple[int, int], ...]
        if parent is None:
            neighbors = ()
        else:
            order = 1 if defer_bond_orders else int(target.bonds[v, parent])
            neighbors = ((parent, order),)
            tree_edges.add(tuple(sorted((v, parent))))

        target_neighbors = tuple(int(u) for u in np.flatnonzero(target.bonds[v] != 0))
        if defer_bond_orders:
            saturated_h = int(target.implicit_h_counts[v]) + sum(
                _bond_cost(int(target.bonds[v, u])) - 1 for u in target_neighbors
            )
            missing_bond_load = sum(
                1 for u in target_neighbors if parent is None or u != parent
            )
            insertion_h = saturated_h + missing_bond_load
        else:
            missing_bond_load = sum(
                _bond_cost(int(target.bonds[v, u]))
                for u in target_neighbors
                if parent is None or u != parent
            )
            insertion_h = int(target.implicit_h_counts[v]) + missing_bond_load
        if insertion_h > MAX_H_COUNT:
            raise TraceCompilationError(
                f"slot {v} requires transient H={insertion_h} under root "
                f"{traversal.root}"
            )
        step = RewriteStep(
            "atom_insert",
            AtomInsert(
                slot=v,
                atom_type=int(target.atom_types[v]),
                formal_charge=int(target.formal_charges[v]),
                implicit_h_count=insertion_h,
                neighbors=neighbors,
            ),
        )
        state = runtime.apply(state, step.rule_name, step.action)
        steps.append(step)

    chords = [
        (a, b)
        for a in traversal.order
        for b in traversal.order
        if b > a
        and int(target.bonds[a, b]) != 0
        and tuple(sorted((a, b))) not in tree_edges
    ]
    if rng is not None:
        rng.shuffle(chords)
    for a, b in chords:
        order = 1 if defer_bond_orders else int(target.bonds[a, b])
        if order == BOND_AROMATIC or order not in MICRO_BOND_CLASSES:
            raise TraceCompilationError(
                "target was not normalized to the micro bond vocabulary"
            )
        step = RewriteStep("bond_insert", BondInsert(a, b, order))
        state = runtime.apply(state, step.rule_name, step.action)
        steps.append(step)

    bond_order_steps = 0
    if defer_bond_orders:
        multiple_bonds = [
            (a, b)
            for a in traversal.order
            for b in traversal.order
            if b > a and int(target.bonds[a, b]) > 1
        ]
        if rng is not None:
            rng.shuffle(multiple_bonds)
        for a, b in multiple_bonds:
            step = RewriteStep(
                "bond_reorder",
                BondReorder(a, b, int(target.bonds[a, b])),
            )
            state = runtime.apply(state, step.rule_name, step.action)
            steps.append(step)
            bond_order_steps += 1

    if not _same_state(state, target):
        raise TraceCompilationError("compiled trace did not reconstruct the exact target")
    return RewriteTrace(
        source=source,
        target=target,
        steps=tuple(steps),
        metadata={
            "compiler": "dfs_spanning_tree_then_chords_v1",
            "root": traversal.root,
            "atom_steps": len(traversal.order),
            "bond_steps": len(chords),
            "bond_order_steps": bond_order_steps,
            "defer_bond_orders": defer_bond_orders,
            "randomized": rng is not None,
        },
    )


def compile_null_to_target(
    target: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int | None = None,
    defer_bond_orders: bool = False,
) -> RewriteTrace:
    """Compile a connected target into an exact validity-closed construction."""

    if not is_valid_state(target):
        raise TraceCompilationError("target state is invalid")
    real_slots = _real_slots(target)
    if not real_slots:
        source = empty_molecular_graph(target.n_atoms)
        return RewriteTrace(source, target, (), {"compiler": "identity_null"})
    if not _connected(target, real_slots):
        raise TraceCompilationError("the base compiler currently requires a connected target")
    if np.any(target.bonds == BOND_AROMATIC):
        raise TraceCompilationError("target contains non-Kekulized aromatic bonds")

    runtime = system or default_rewrite_system()
    failures: list[str] = []
    deterministic_roots = sorted(
        real_slots,
        key=lambda v: (
            int(target.implicit_h_counts[v])
            + sum(_bond_cost(int(x)) for x in target.bonds[v] if int(x) != 0),
            v,
        ),
    )
    if rng is None:
        roots = deterministic_roots
    else:
        attempts = max_attempts or max(4 * len(real_slots), 16)
        if attempts <= 0:
            raise ValueError("max_attempts must be positive")
        roots = []
        while len(roots) < attempts:
            cycle = list(real_slots)
            rng.shuffle(cycle)
            roots.extend(cycle)
        roots = roots[:attempts]

    for root in roots:
        traversal = _depth_first_traversal(target, root, rng=rng)
        if len(traversal.order) != len(real_slots):
            continue
        try:
            return _compile_with_traversal(
                target,
                traversal,
                runtime,
                rng=rng,
                defer_bond_orders=defer_bond_orders,
            )
        except (InvalidRewrite, TraceCompilationError) as exc:
            failures.append(f"root {root}: {exc}")
    detail = "; ".join(failures[:5])
    raise TraceCompilationError(f"no legal construction trace found ({detail})")


def compile_source_to_target(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    rng: np.random.Generator | None = None,
    defer_bond_orders: bool = False,
) -> RewriteTrace:
    """Compile a validity-closed baseline bridge between arbitrary molecules.

    The baseline deletes the source through the exact inverse of a legal source
    construction, reaches the formal null state, and then constructs the
    target. It is intentionally not an edit-minimal alignment. Task-specific
    bridges and scaffold-preserving traces can replace it without changing the
    generator or executor interfaces.
    """

    if not is_valid_state(source) or not is_valid_state(target):
        raise TraceCompilationError("source and target must both be valid states")
    n_slots = max(source.n_atoms, target.n_atoms)
    padded_source = pad_molecular_graph(source, n_slots)
    padded_target = pad_molecular_graph(target, n_slots)
    runtime = system or default_rewrite_system()
    source_construction = compile_null_to_target(
        padded_source,
        system=runtime,
        rng=rng,
        defer_bond_orders=defer_bond_orders,
    )
    target_construction = compile_null_to_target(
        padded_target,
        system=runtime,
        rng=rng,
        defer_bond_orders=defer_bond_orders,
    )
    deletion = invert_trace(
        source_construction.source,
        source_construction.steps,
        system=runtime,
    )
    steps = (*deletion, *target_construction.steps)
    endpoint = execute_trace(padded_source, steps, system=runtime)
    if not _same_state(endpoint, padded_target):
        raise TraceCompilationError("source-target bridge did not reconstruct target")
    return RewriteTrace(
        source=padded_source,
        target=padded_target,
        steps=tuple(steps),
        metadata={
            "compiler": "delete_to_null_then_construct_v1",
            "source_delete_steps": len(deletion),
            "target_construct_steps": len(target_construction.steps),
            "randomized": rng is not None,
        },
    )
