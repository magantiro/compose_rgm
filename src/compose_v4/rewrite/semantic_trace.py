"""Process-specific inversion for the frozen Active8 Editing-V2 runtime.

The historical generic trace API intentionally keeps its original raw-action
semantics. This module returns only actions accepted by ActionCodecV4 and never
assumes one-step inverse closure. Unsupported inverses fail with a typed error.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    ORGANIC_VOCABULARY,
    MolecularGraph,
)
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
    resolve_cycle_open_edge,
)
from compose_v4.rewrite.trace import RewriteStep, execute_trace
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate


class SemanticInverseUnavailable(ValueError):
    """A legal Active8 step has no admitted one-step Active8 inverse."""


def _atom_class_index(state: MolecularGraph, vertex: int) -> int | None:
    bond_order_sum = sum(
        int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex]
    )
    return ORGANIC_VOCABULARY.class_index(
        int(state.atom_types[vertex]),
        bond_order_sum,
        int(state.implicit_h_counts[vertex]),
        int(state.formal_charges[vertex]),
    )


def inverse_semantic_step(
    source: MolecularGraph,
    step: RewriteStep,
    *,
    system: RewriteSystem | None = None,
) -> RewriteStep:
    """Return and verify one canonical molecular inverse under Active8."""

    runtime = system or editing_v2_semantic_rewrite_system()
    action = step.action
    successor = runtime.apply(source, step.rule_name, action)

    if step.rule_name == "atom_insert":
        inverse = RewriteStep("atom_delete", AtomDelete(int(action.slot)))
    elif step.rule_name == "atom_delete":
        vertex = int(action.v)
        neighbors = tuple(
            (int(neighbor), int(source.bonds[vertex, neighbor]))
            for neighbor in np.flatnonzero(source.bonds[vertex] != 0)
        )
        if len(neighbors) > 1:
            raise SemanticInverseUnavailable(
                "atom deletion has no one-step inverse under zero/one-neighbor birth"
            )
        inverse = RewriteStep(
            "atom_insert",
            AtomInsert(
                slot=vertex,
                atom_type=int(source.atom_types[vertex]),
                formal_charge=int(source.formal_charges[vertex]),
                implicit_h_count=int(source.implicit_h_counts[vertex]),
                neighbors=neighbors,
            ),
        )
    elif step.rule_name == "atom_restate_semantic":
        vertex = int(action.v)
        class_index = _atom_class_index(source, vertex)
        if class_index is None:
            raise SemanticInverseUnavailable(
                "source atom has no broad-organic class for semantic inversion"
            )
        inverse = RewriteStep(
            "atom_restate_semantic",
            SemanticAtomRestate(vertex, class_index),
        )
    elif step.rule_name == "bond_reorder":
        inverse = RewriteStep(
            "bond_reorder",
            BondReorder(
                int(action.a),
                int(action.b),
                int(source.bonds[int(action.a), int(action.b)]),
            ),
        )
    elif step.rule_name == "bond_reroute":
        inverse = RewriteStep(
            "bond_reroute",
            BondReroute(
                a=int(action.u),
                b=int(action.v),
                u=int(action.a),
                v=int(action.b),
                new_order=int(source.bonds[int(action.a), int(action.b)]),
            ),
        )
    elif step.rule_name == "cycle_close":
        inverse = RewriteStep(
            "cycle_open",
            CycleOpenEdge(int(action.a), int(action.b)),
        )
    elif step.rule_name == "cycle_open":
        resolution = resolve_cycle_open_edge(source, action)
        if (
            resolution is None
            or not resolution.admitted
            or resolution.inverse_bond_order is None
        ):
            raise SemanticInverseUnavailable(
                "cycle-open resolution does not declare an inverse bond order"
            )
        inverse = RewriteStep(
            "cycle_close",
            CycleCloseEdge(
                int(action.a),
                int(action.b),
                int(resolution.inverse_bond_order),
            ),
        )
    elif step.rule_name == "ring_system_restate":
        inverse = RewriteStep(
            "ring_system_restate",
            RingSystemRestate(
                tuple(
                    BondOrderChange(
                        int(change.a),
                        int(change.b),
                        int(source.bonds[int(change.a), int(change.b)]),
                    )
                    for change in action.changes
                )
            ),
        )
    else:
        raise SemanticInverseUnavailable(
            f"rule {step.rule_name!r} is outside frozen Active8"
        )

    try:
        restored = runtime.apply(successor, inverse.rule_name, inverse.action)
    except ValueError as error:
        raise SemanticInverseUnavailable(
            f"inverse for {step.rule_name!r} is not admitted by Active8"
        ) from error
    if canonical_state_key(restored) != canonical_state_key(source):
        raise SemanticInverseUnavailable(
            f"inverse for {step.rule_name!r} does not restore the source molecule"
        )
    return inverse


def invert_semantic_trace(
    source: MolecularGraph,
    steps: tuple[RewriteStep, ...],
    *,
    system: RewriteSystem | None = None,
) -> tuple[RewriteStep, ...]:
    """Invert an Active8 trace without mutating generic historical semantics."""

    runtime = system or editing_v2_semantic_rewrite_system()
    _, states = execute_trace(source, steps, system=runtime, return_states=True)
    inverses = tuple(
        inverse_semantic_step(states[index], step, system=runtime)
        for index, step in enumerate(steps)
    )
    return tuple(reversed(inverses))


__all__ = [
    "SemanticInverseUnavailable",
    "inverse_semantic_step",
    "invert_semantic_trace",
]
