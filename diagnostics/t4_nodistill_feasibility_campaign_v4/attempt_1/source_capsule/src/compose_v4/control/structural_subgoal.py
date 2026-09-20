"""Address-free structural graph subgoals and exact bound target construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    NULL_IDX,
    MolecularGraph,
    is_element,
    is_rdkit_valid,
)
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
    trace_structure,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "structural_subgoal_v1"
GOAL_SCHEMA = "structural_goal_v1"

Token = tuple[str, int]
AtomSignature = tuple[int, int, int, int]


@dataclass(frozen=True)
class StructuralSubgoal:
    """One local target graph, expressed over source roles and new output roles."""

    input_atoms: tuple[AtomSignature, ...]
    input_bonds: tuple[tuple[int, ...], ...]
    environments: tuple[str, ...]
    target_atoms: tuple[AtomSignature | None, ...]
    output_atoms: tuple[AtomSignature, ...]
    target_bonds: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        n_input = len(self.input_atoms)
        n_target = n_input + len(self.output_atoms)
        if not n_input:
            raise ValueError("a structural subgoal needs at least one source role")
        if len(self.environments) != n_input or len(self.target_atoms) != n_input:
            raise ValueError("structural subgoal input dimensions disagree")
        if len(self.input_bonds) != n_input or any(len(row) != n_input for row in self.input_bonds):
            raise ValueError("structural subgoal input bond matrix is malformed")
        if len(self.target_bonds) != n_target or any(
            len(row) != n_target for row in self.target_bonds
        ):
            raise ValueError("structural subgoal target bond matrix is malformed")
        for matrix in (self.input_bonds, self.target_bonds):
            if any(
                matrix[i][j] != matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix))
            ):
                raise ValueError("structural subgoal bond matrix must be symmetric")
            if any(matrix[i][i] for i in range(len(matrix))):
                raise ValueError("structural subgoal bond matrix diagonal must be zero")
        if (
            all(
                before == after
                for before, after in zip(self.input_atoms, self.target_atoms, strict=True)
            )
            and not self.output_atoms
            and self.input_bonds == self.target_bonds
        ):
            raise ValueError("structural subgoal cannot be a self event")

    @property
    def subgoal_id(self) -> str:
        return identity(self.payload())

    def payload(self) -> dict:
        return {"schema_version": SCHEMA, **asdict(self)}

    @classmethod
    def from_payload(cls, payload: dict) -> StructuralSubgoal:
        if payload.get("schema_version") != SCHEMA or set(payload) != {
            "schema_version",
            "input_atoms",
            "input_bonds",
            "environments",
            "target_atoms",
            "output_atoms",
            "target_bonds",
        }:
            raise ValueError("unexpected structural-subgoal schema or fields")
        return cls(
            input_atoms=tuple(tuple(row) for row in payload["input_atoms"]),
            input_bonds=tuple(tuple(row) for row in payload["input_bonds"]),
            environments=tuple(payload["environments"]),
            target_atoms=tuple(
                None if row is None else tuple(row) for row in payload["target_atoms"]
            ),
            output_atoms=tuple(tuple(row) for row in payload["output_atoms"]),
            target_bonds=tuple(tuple(row) for row in payload["target_bonds"]),
        )


@dataclass(frozen=True)
class StructuralGoal:
    """One complete high-level proposal containing independent target patches."""

    subgoals: tuple[StructuralSubgoal, ...]

    def __post_init__(self) -> None:
        if not self.subgoals or len(self.subgoals) > 4:
            raise ValueError("structural goal requires one to four subgoals")

    @property
    def goal_id(self) -> str:
        return identity(self.payload())

    def payload(self) -> dict:
        return {
            "schema_version": GOAL_SCHEMA,
            "subgoals": [row.payload() for row in self.subgoals],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> StructuralGoal:
        if payload.get("schema_version") != GOAL_SCHEMA or set(payload) != {
            "schema_version",
            "subgoals",
        }:
            raise ValueError("unexpected structural-goal schema or fields")
        return cls(tuple(StructuralSubgoal.from_payload(row) for row in payload["subgoals"]))


@dataclass(frozen=True)
class BindingCensus:
    assignments: tuple[tuple[int, ...], ...]
    visits: int
    truncated: bool


def _atom_key(graph: MolecularGraph, slot: int) -> tuple:
    """Permutation-invariant local key used only to order equivalent role rows."""

    real = tuple(int(value) for value in np.flatnonzero(is_element(graph.atom_types)))
    labels = {
        value: identity(
            (
                int(graph.atom_types[value]),
                int(graph.formal_charges[value]),
                int(graph.implicit_h_counts[value]),
            )
        )
        for value in real
    }
    for _ in range(len(real)):
        labels = {
            value: identity(
                (
                    labels[value],
                    sorted(
                        (int(graph.bonds[value, other]), labels[other])
                        for other in real
                        if graph.bonds[value, other]
                    ),
                )
            )
            for value in real
        }
    return labels[slot], atom_signature(graph, slot), environment(graph, slot)


def _bond(graph: MolecularGraph, left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 0
    return int(graph.bonds[left, right])


def _signature(graph: MolecularGraph, slot: int | None) -> AtomSignature | None:
    return None if slot is None else atom_signature(graph, slot)


def _extract_component(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    structure: dict,
    primitive_indices: tuple[int, ...],
) -> tuple[StructuralSubgoal, tuple[int, ...], tuple[int, ...]]:
    initial_token_slots: dict[Token, int] = structure["initial_token_slots"]
    final_token_slots = {token: slot for slot, token in structure["final_active_slots"].items()}
    relevant = set().union(*(structure["footprints"][index] for index in primitive_indices))
    source_tokens = {token for token in relevant if token[0] == "source"}
    created_tokens = {
        token for token in relevant if token[0] == "created" and token in final_token_slots
    }

    initial_slot_tokens = {slot: token for token, slot in initial_token_slots.items()}
    for token in tuple(source_tokens):
        slot = initial_token_slots[token]
        for neighbor in np.flatnonzero(source.bonds[slot]):
            other = initial_slot_tokens.get(int(neighbor))
            if other is not None:
                source_tokens.add(other)
    for token in tuple(relevant):
        slot = final_token_slots.get(token)
        if slot is None:
            continue
        final_slot_tokens = {
            active_slot: active_token
            for active_slot, active_token in structure["final_active_slots"].items()
        }
        for neighbor in np.flatnonzero(target.bonds[slot]):
            other = final_slot_tokens.get(int(neighbor))
            if other is not None and other[0] == "source":
                source_tokens.add(other)

    # Exact execution can re-kekulize an aromatic system beyond the explicit
    # primitive footprint.  Treat the connected changed-bond component as part
    # of the same structural patch so the target is a complete graph delta,
    # rather than a mixture of source and target Kekule assignments.
    changed_bond_neighbors: dict[Token, set[Token]] = defaultdict(set)
    retained_source_tokens = tuple(
        token for token in initial_token_slots if token in final_token_slots
    )
    for left_index, left_token in enumerate(retained_source_tokens):
        source_left = initial_token_slots[left_token]
        target_left = final_token_slots[left_token]
        for right_token in retained_source_tokens[left_index + 1 :]:
            source_right = initial_token_slots[right_token]
            target_right = final_token_slots[right_token]
            if int(source.bonds[source_left, source_right]) == int(
                target.bonds[target_left, target_right]
            ):
                continue
            changed_bond_neighbors[left_token].add(right_token)
            changed_bond_neighbors[right_token].add(left_token)
    frontier = list(source_tokens)
    while frontier:
        token = frontier.pop()
        for neighbor in changed_bond_neighbors.get(token, ()):
            if neighbor not in source_tokens:
                source_tokens.add(neighbor)
                frontier.append(neighbor)

    source_order = tuple(
        sorted(
            source_tokens,
            key=lambda token: (
                _atom_key(source, initial_token_slots[token]),
                _signature(target, final_token_slots.get(token)),
                token[1],
            ),
        )
    )
    output_order = tuple(
        sorted(
            created_tokens,
            key=lambda token: (
                _atom_key(target, final_token_slots[token]),
                token[1],
            ),
        )
    )
    input_slots = tuple(initial_token_slots[token] for token in source_order)
    target_input_slots = tuple(final_token_slots.get(token) for token in source_order)
    output_slots = tuple(final_token_slots[token] for token in output_order)
    target_slots = (*target_input_slots, *output_slots)
    subgoal = StructuralSubgoal(
        input_atoms=tuple(atom_signature(source, slot) for slot in input_slots),
        input_bonds=tuple(
            tuple(int(source.bonds[left, right]) for right in input_slots) for left in input_slots
        ),
        environments=tuple(environment(source, slot) for slot in input_slots),
        target_atoms=tuple(_signature(target, slot) for slot in target_input_slots),
        output_atoms=tuple(atom_signature(target, slot) for slot in output_slots),
        target_bonds=tuple(
            tuple(_bond(target, left, right) for right in target_slots) for left in target_slots
        ),
    )
    # Target output slots are returned only to support exact training-data
    # audits.  They are deliberately absent from the serialized subgoal.
    return subgoal, input_slots, output_slots


def extract_structural_goal(
    states: tuple[dict, ...], actions: tuple[dict, ...]
) -> tuple[StructuralGoal, tuple[tuple[int, ...], ...], dict]:
    """Extract WHAT changed without retaining the teacher primitive realization."""

    if len(states) != len(actions) + 1 or not actions:
        raise ValueError("structural goal extraction requires one complete trace")
    source, target = decode_state(states[0]), decode_state(states[-1])
    config = DependencyRegionConfig(
        runtime_maximum_primitives=32,
        runtime_maximum_components=4,
        join_lifetime_neighbors=False,
    )
    regions = dependency_region_program(states, actions, config)
    if not regions["complete_representation_supported"]:
        raise ValueError(f"route exceeds structural-goal support: {regions['abstention_reason']}")
    structure = trace_structure(states, actions)
    rows = [
        _extract_component(
            source,
            target,
            structure=structure,
            primitive_indices=tuple(component["primitive_indices"]),
        )
        for component in regions["components"]
    ]
    goal = StructuralGoal(tuple(row[0] for row in rows))
    bindings = tuple(row[1] for row in rows)
    return goal, bindings, regions


def attachment_bindings(
    subgoal: StructuralSubgoal,
    graph: MolecularGraph,
    *,
    max_bindings: int = 128,
    max_visits: int = 16384,
) -> BindingCensus:
    """Enumerate exact address-free source-role bindings for one subgoal."""

    if min(max_bindings, max_visits) < 1:
        raise ValueError("binding limits must be positive")
    real = tuple(int(value) for value in np.flatnonzero(is_element(graph.atom_types)))
    signatures = {slot: atom_signature(graph, slot) for slot in real}
    contexts = {slot: environment(graph, slot) for slot in real}
    candidates = [
        [slot for slot in real if signatures[slot] == signature and contexts[slot] == context]
        for signature, context in zip(subgoal.input_atoms, subgoal.environments, strict=True)
    ]
    order = sorted(range(len(candidates)), key=lambda index: (len(candidates[index]), index))
    assigned: dict[int, int] = {}
    rows: list[tuple[int, ...]] = []
    visits = 0
    truncated = False

    def visit(depth: int) -> None:
        nonlocal visits, truncated
        if len(rows) >= max_bindings or visits >= max_visits:
            truncated = True
            return
        visits += 1
        if depth == len(order):
            rows.append(tuple(assigned[index] for index in range(len(order))))
            return
        index = order[depth]
        for slot in candidates[index]:
            if slot in assigned.values():
                continue
            if any(
                int(graph.bonds[slot, other_slot]) != subgoal.input_bonds[index][other_index]
                for other_index, other_slot in assigned.items()
            ):
                continue
            assigned[index] = slot
            visit(depth + 1)
            del assigned[index]
            if truncated:
                return

    visit(0)
    return BindingCensus(tuple(rows), visits, truncated)


def instantiate_goal(
    source: MolecularGraph,
    goal: StructuralGoal,
    bindings: tuple[tuple[int, ...], ...],
    *,
    prefer_initially_empty_output_slots: bool = False,
) -> tuple[MolecularGraph, dict]:
    """Construct the complete bound target graph without a primitive teacher trace."""

    if len(bindings) != len(goal.subgoals):
        raise ValueError("goal and binding counts disagree")
    initially_empty = tuple(int(slot) for slot in np.flatnonzero(source.atom_types == NULL_IDX))
    atom_types = source.atom_types.copy()
    charges = source.formal_charges.copy()
    hydrogens = source.implicit_h_counts.copy()
    bonds = source.bonds.copy()
    targets: dict[int, AtomSignature | None] = {}
    for subgoal, assignment in zip(goal.subgoals, bindings, strict=True):
        if len(assignment) != len(subgoal.input_atoms) or len(set(assignment)) != len(assignment):
            raise ValueError("subgoal binding is malformed")
        for slot, before, after in zip(
            assignment, subgoal.input_atoms, subgoal.target_atoms, strict=True
        ):
            if atom_signature(source, slot) != before:
                raise ValueError("subgoal binding violates its source atom role")
            if slot in targets and targets[slot] != after:
                raise ValueError("overlapping subgoals request inconsistent atom targets")
            targets[slot] = after

    for slot, target in targets.items():
        if target is None:
            atom_types[slot] = NULL_IDX
            charges[slot] = 0
            hydrogens[slot] = 0
            bonds[slot, :] = 0
            bonds[:, slot] = 0

    allocated_output_slots: set[int] = set()
    output_slots: list[tuple[int, ...]] = []
    for subgoal in goal.subgoals:
        available = [
            int(slot)
            for slot in np.flatnonzero(atom_types == NULL_IDX)
            if slot not in allocated_output_slots
        ]
        if prefer_initially_empty_output_slots:
            initially_available = [slot for slot in initially_empty if slot in available]
            newly_freed = [slot for slot in available if slot not in initially_empty]
            available = [*initially_available, *newly_freed]
        if len(available) < len(subgoal.output_atoms):
            raise ValueError("structural goal exceeds persistent-slot capacity")
        selected = tuple(available[: len(subgoal.output_atoms)])
        output_slots.append(selected)
        allocated_output_slots.update(selected)
        for slot, signature in zip(selected, subgoal.output_atoms, strict=True):
            atom_types[slot], charges[slot], hydrogens[slot] = signature[:3]

    for slot, target in targets.items():
        if target is not None:
            atom_types[slot], charges[slot], hydrogens[slot] = target[:3]

    for subgoal, assignment, outputs in zip(goal.subgoals, bindings, output_slots, strict=True):
        local_slots = (*assignment, *outputs)
        local_targets = (*subgoal.target_atoms, *subgoal.output_atoms)
        for left_index, left in enumerate(local_slots):
            # A created output may legitimately reuse the persistent slot of a
            # deleted source role.  Role identity remains typed even though the
            # coordinate is reused, so the deleted role must not write bonds.
            if local_targets[left_index] is None:
                continue
            for right_index in range(left_index + 1, len(local_slots)):
                right = local_slots[right_index]
                # Deletion already cleared every incident bond.  Do not write
                # through the deleted role again because its persistent slot
                # may now hold a created role from another subgoal.
                if local_targets[right_index] is None:
                    continue
                order = subgoal.target_bonds[left_index][right_index]
                bonds[left, right] = bonds[right, left] = order

    product = MolecularGraph(atom_types, charges, hydrogens, bonds)
    for subgoal, assignment, outputs in zip(goal.subgoals, bindings, output_slots, strict=True):
        for slot, expected in zip(
            (*assignment, *outputs),
            (*subgoal.target_atoms, *subgoal.output_atoms),
            strict=True,
        ):
            if expected is not None and atom_signature(product, slot) != expected:
                raise ValueError("instantiated target does not satisfy an atom role")
    if product.n_real_atoms > 40 or not is_rdkit_valid(product):
        raise ValueError("instantiated structural goal is outside molecular support")
    if canonical_state_key(product) == canonical_state_key(source):
        raise ValueError("instantiated structural goal is a canonical self event")
    return product, {
        "schema_version": "bound_structural_goal_receipt_v1",
        "goal_id": goal.goal_id,
        "bindings": [list(row) for row in bindings],
        "output_slots": [list(row) for row in output_slots],
        "output_slot_policy": (
            "initially_empty_then_freed"
            if prefer_initially_empty_output_slots
            else "lowest_available_after_deletion"
        ),
        "endpoint": canonical_state_key(product),
        "primitive_teacher_actions_used": 0,
    }


__all__ = [
    "GOAL_SCHEMA",
    "SCHEMA",
    "BindingCensus",
    "StructuralGoal",
    "StructuralSubgoal",
    "attachment_bindings",
    "extract_structural_goal",
    "instantiate_goal",
]
