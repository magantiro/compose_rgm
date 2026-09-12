"""Joint multi-site proposals, explicit dataflow and conservative serial scheduling.

This is a proposal/compiler layer, not a new molecular transition or an exact
acceleration of R_theta. Disjoint operand footprints are not a chemical safety
certificate. Every scheduled primitive is revalidated on its actual predecessor.
No intermediate endpoint utility participates in execution or scheduling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramBlock,
    _json,
    _slots,
    atom_signature,
    environment,
    execute_bound_program,
)

Handle = tuple[str, int]


def _handle(ref: dict) -> Handle:
    return next(iter(ref.items()))


def _references(record: dict) -> frozenset[Handle]:
    references = set()

    def collect(ref):
        references.add(_handle(ref))
        return ref

    _slots(record, collect)
    return frozenset(references)


@dataclass(frozen=True)
class BlockFootprint:
    reads: tuple[Handle, ...]
    writes: tuple[Handle, ...]
    creates: tuple[Handle, ...]
    deletes: tuple[Handle, ...]
    minimum_delta: int
    peak_delta: int
    final_delta: int


@dataclass(frozen=True)
class ProgramGraph:
    program: EditProgram
    footprints: tuple[BlockFootprint, ...]
    dependencies: tuple[tuple[int, int], ...]
    conflicts: tuple[tuple[int, int], ...]
    serialization_edges: tuple[tuple[int, int], ...]

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(set(self.dependencies) | set(self.serialization_edges)))

    def payload(self) -> dict:
        return {
            "schema_version": "multi_site_program_graph_v1",
            "program": self.program.payload(),
            "footprints": [asdict(value) for value in self.footprints],
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "serialization_edges": self.serialization_edges,
            "footprint_scope": "explicit atom operands, not exhaustive chemical guard reads",
            "global_revalidation": ["connectivity", "valence", "aromaticity", "capacity", "charge"],
            "task_interactions": "not inferred from execution compatibility",
        }


def compile_program_graph(
    program: EditProgram, *, dependencies: tuple[tuple[int, int], ...] = ()
) -> ProgramGraph:
    """Separate created-atom requirements from overlap serialization constraints.

    Overlap retains the observed program order rather than excluding either edit.
    Additional caller-declared dependencies may constrain initially separate branches.
    Global executor guards remain necessary even when no pairwise conflict is found.
    """
    footprints, owner, required = [], {}, set(dependencies)
    start = 0
    for index, block in enumerate(program.blocks):
        reads, writes, creates, deletes = set(), set(), set(), set()
        delta = low = peak = 0
        for text in program.marks[start : block.stop]:
            record = json.loads(text)
            refs = set(_references(record))
            if record["executor_rule"] == "atom_insert":
                created = _handle(record["payload"]["slot"])
                creates.add(created)
                owner[created] = index
                refs.remove(created)
                delta += 1
            if record["executor_rule"] == "atom_delete":
                deletes.add(_handle(record["payload"]["v"]))
                delta -= 1
            for ref in refs:
                if ref[0] == "created" and owner[ref] != index:
                    required.add((owner[ref], index))
            # All explicit operands are conservatively treated as writes, including
            # anchors whose hydrogen counts change. This is not a full guard model.
            reads.update(refs - creates)
            writes.update(refs)
            low, peak = min(low, delta), max(peak, delta)
        writes.update(creates)
        footprints.append(
            BlockFootprint(
                tuple(sorted(reads)),
                tuple(sorted(writes)),
                tuple(sorted(creates)),
                tuple(sorted(deletes)),
                low,
                peak,
                delta,
            )
        )
        start = block.stop
    n = len(footprints)
    if any(
        len(edge) != 2
        or any(type(i) is not int or not 0 <= i < n for i in edge)
        or edge[0] == edge[1]
        for edge in required
    ):
        raise ValueError("program dependency must link two distinct existing blocks")
    conflicts = tuple(
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if set(footprints[i].writes) & (set(footprints[j].reads) | set(footprints[j].writes))
        or set(footprints[j].writes) & set(footprints[i].reads)
    )
    serial = tuple(edge for edge in conflicts if edge not in required)
    result = ProgramGraph(program, tuple(footprints), tuple(sorted(required)), conflicts, serial)
    # Structural acyclicity only; resource checks require the bound source.
    _topological(result, None, None)
    return result


def _topological(graph, initial_atoms, priority):
    n = len(graph.program.blocks)
    preference = tuple(range(n)) if priority is None else tuple(priority)
    if len(preference) != n or set(preference) != set(range(n)):
        raise ValueError("schedule priority must be a permutation of block indices")
    if initial_atoms is not None and (
        type(initial_atoms) is not int or not 1 <= initial_atoms <= 40
    ):
        raise ValueError("source is outside the frozen 1..40 active-atom support")
    pending, finished, order, timeline = set(range(n)), set(), [], []
    atoms = initial_atoms
    while pending:
        ready = [
            i
            for i in preference
            if i in pending and all(a in finished for a, b in graph.edges if b == i)
        ]
        if not ready:
            raise ValueError("program dependency/serialization graph contains a cycle")
        feasible = [
            i
            for i in ready
            if atoms is None
            or (
                1 <= atoms + graph.footprints[i].minimum_delta
                and atoms + graph.footprints[i].peak_delta <= 40
            )
        ]
        if not feasible:
            raise ValueError("no ready complete block fits the frozen peak-capacity limit")
        chosen = feasible[0]
        if atoms is not None:
            footprint = graph.footprints[chosen]
            timeline.append(
                {
                    "block": chosen,
                    "start": atoms,
                    "peak": atoms + footprint.peak_delta,
                    "end": atoms + footprint.final_delta,
                }
            )
            atoms += footprint.final_delta
        pending.remove(chosen)
        finished.add(chosen)
        order.append(chosen)
    return tuple(order), timeline


def scheduled_program(graph: ProgramGraph, initial_atoms: int, *, priority=None):
    """Deterministic complete-block schedule with globally fresh symbolic handles.

    The bounded greedy resource schedule may abstain even when another ordering
    works. It never expands the atom ceiling or emits a partially completed endpoint.
    """
    order, timeline = _topological(graph, initial_atoms, priority)
    starts = (0, *(b.stop for b in graph.program.blocks[:-1]))
    renamed, marks, blocks = {}, [], []
    for index in order:
        block = graph.program.blocks[index]
        for text in graph.program.marks[starts[index] : block.stop]:
            record = json.loads(text)
            if record["executor_rule"] == "atom_insert":
                renamed[_handle(record["payload"]["slot"])] = {"created": len(renamed)}

            def translate(ref):
                return ref if "input" in ref else renamed[_handle(ref)]

            marks.append(_json(_slots(record, translate)))
        blocks.append(ProgramBlock(block.label, len(marks)))
    original = graph.program
    program = EditProgram(
        original.input_atoms,
        original.input_bonds,
        original.environments,
        tuple(marks),
        tuple(blocks),
    )
    return program, order, timeline


def execute_program_graph(
    source,
    graph: ProgramGraph,
    assignment,
    *,
    priority=None,
    max_primitives=24,
    max_blocks=5,
    mutable_slots=None,
):
    program, order, timeline = scheduled_program(graph, source.n_real_atoms, priority=priority)
    product, receipt = execute_bound_program(
        source,
        program,
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
        mutable_slots=mutable_slots,
    )
    return product, {
        **receipt,
        "graph_id": identity(graph.payload()),
        "selected_program_id": graph.program.program_id,
        "block_order": order,
        "capacity_timeline": timeline,
        "scheduling": "deterministic serial; executor revalidation after every primitive",
        "proposal_law": "new optimization proposal; no reference likelihood claimed",
        "actual_changes": changed_input_sites(source, product, receipt["actions"]),
    }


def changed_input_sites(source, product, actions) -> dict:
    """Net changed original-atom environments, not the intended mutable region.

    Sites are connected components induced by those atoms in the ORIGINAL graph.
    This reproducible descriptor is not a certificate of chemical independence.
    Persistent-slot reuse is tracked so a birth never masquerades as a saved atom.
    """
    tokens = {i: ("input", i) for i in range(source.n_atoms) if is_element(source.atom_types[i])}
    original = dict(tokens)
    born = 0
    for record in actions:
        if record["executor_rule"] == "atom_insert":
            tokens[record["payload"]["slot"]] = ("created", born)
            born += 1
        elif record["executor_rule"] == "atom_delete":
            del tokens[record["payload"]["v"]]
    if set(tokens) != {i for i in range(product.n_atoms) if is_element(product.atom_types[i])}:
        raise ValueError("action lineage does not match the final exact state")
    survivors = {ref[1]: slot for slot, ref in tokens.items() if ref[0] == "input"}

    def context(molecule, slot, labels):
        return (
            atom_signature(molecule, slot),
            tuple(
                sorted(
                    (labels[i], int(molecule.bonds[slot, i]))
                    for i in labels
                    if molecule.bonds[slot, i]
                )
            ),
        )

    changed = {
        slot
        for slot in original
        if slot not in survivors
        or context(source, slot, original) != context(product, survivors[slot], tokens)
    }
    pending, components = set(changed), []
    while pending:
        frontier, component = [min(pending)], set()
        while frontier:
            slot = frontier.pop()
            if slot in component:
                continue
            component.add(slot)
            frontier.extend(i for i in pending - component if source.bonds[slot, i])
        pending.difference_update(component)
        components.append(sorted(component))
    return {
        "changed_original_slots": sorted(changed),
        "source_induced_components": components,
        "changed_site_count": len(components),
        "surviving_new_atoms": sum(ref[0] == "created" for ref in tokens.values()),
        "deleted_original_atoms": len(original) - len(survivors),
        "definition": "net original atom attributes or neighbor identities/bonds changed",
    }


def combine_bound_programs(
    source: MolecularGraph,
    parts: tuple[tuple[EditProgram, tuple[int, ...]], ...],
    *,
    max_primitives: int = 24,
    max_blocks: int = 5,
) -> tuple[EditProgram, tuple[int, ...]]:
    """Plan two to four site-bound programs together, without intermediate scoring.

    Sites may overlap or be disconnected. Dependencies within a selected program
    survive; shared input atoms are unified, not copied. Global compatibility is
    decided only by subsequent executor replay, never by these binding checks.
    """
    if not 2 <= len(parts) <= 4:
        raise ValueError("multi-site channel requires two to four component programs")
    if (
        sum(len(p.marks) for p, _ in parts) > max_primitives
        or sum(len(p.blocks) for p, _ in parts) > max_blocks
    ):
        raise ValueError("joint program exceeds its declared primitive/block budget")
    inputs, marks, blocks, next_created = [], [], [], 0
    for component, (program, binding) in enumerate(parts):
        if len(binding) != len(program.input_atoms) or len(set(binding)) != len(binding):
            raise ValueError("component requires an injective complete input binding")
        for i, slot in enumerate(binding):
            if atom_signature(source, slot)[:2] != program.input_atoms[i][:2] or any(
                int(source.bonds[slot, other]) != program.input_bonds[i][j]
                for j, other in enumerate(binding)
            ):
                raise ValueError("component input binding is incompatible with the parent")
            if slot not in inputs:
                inputs.append(slot)
        created_map = {}
        offset = len(marks)
        for text in program.marks:
            record = json.loads(text)
            if record["executor_rule"] == "atom_insert":
                created_map[_handle(record["payload"]["slot"])] = {"created": next_created}
                next_created += 1

            def translate(ref, binding=binding, created_map=created_map):
                return (
                    {"input": inputs.index(binding[ref["input"]])}
                    if "input" in ref
                    else created_map[_handle(ref)]
                )

            marks.append(_json(_slots(record, translate)))
        blocks.extend(
            ProgramBlock(f"{component}:{block.label}", offset + block.stop)
            for block in program.blocks
        )
    combined = EditProgram(
        tuple(atom_signature(source, i) for i in inputs),
        tuple(tuple(int(source.bonds[i, j]) for j in inputs) for i in inputs),
        tuple(environment(source, i) for i in inputs),
        tuple(marks),
        tuple(blocks),
    )
    return combined, tuple(inputs)
