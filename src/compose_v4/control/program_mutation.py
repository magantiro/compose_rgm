"""Variable complete programs, preserving symbolic dependencies rather than endpoints.

Each edit changes a construction parameter or a dependency-closed branch. A
returned symbolic program is only a proposal: the production executor must still
validate its complete trace on the actual bound source.
"""

from __future__ import annotations

import json
from dataclasses import replace

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, ORGANIC_VOCABULARY
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramBlock,
    _json,
    _slots,
    attachment_bindings,
)
from compose_v4.control.edit_program_graph import (
    _references,
    combine_bound_programs,
    compile_program_graph,
)

PARAMETER_MOVES = ("created_atom_type", "extend_segment", "contract_segment")


def _rebuild(program, records_by_block):
    """Rename newly introduced handles throughout every dependent operation."""
    renamed, marks, blocks = {}, [], []
    for block, records in zip(program.blocks, records_by_block, strict=True):
        for record in records:
            if record["executor_rule"] == "atom_insert":
                old = tuple(record["payload"]["slot"].items())
                if old in renamed:
                    raise ValueError("program mutation reuses a created handle")
                renamed[old] = {"created": len(renamed)}

            def translate(ref):
                if "input" in ref:
                    return ref
                key = tuple(ref.items())
                if key not in renamed:
                    raise ValueError("mutation left a dangling dependent atom reference")
                return renamed[key]

            marks.append(_json(_slots(record, translate)))
        if records:
            blocks.append(ProgramBlock(block.label, len(marks)))
    return replace(program, marks=tuple(marks), blocks=tuple(blocks))


def parameter_choices(program: EditProgram, move: str):
    """Finite per-parent mutation census; atom classes come from the registry."""
    if move not in PARAMETER_MOVES:
        raise ValueError(f"unknown program parameter move: {move}")
    records = [json.loads(t) for t in program.marks]
    choices = []
    for index, record in enumerate(records):
        if record["executor_rule"] != "atom_insert":
            continue
        payload = record["payload"]
        neighbors = payload["neighbors"]
        if len(neighbors) != 1 or payload["formal_charge"] != 0:
            continue
        if move == "created_atom_type":
            for class_index, (element, _) in enumerate(ORGANIC_VOCABULARY.classes):
                h = ORGANIC_VOCABULARY.h_count(class_index, neighbors[0][1], 0)
                if h is not None and (element, h) != (
                    payload["atom_type"],
                    payload["implicit_h_count"],
                ):
                    choices.append((index, class_index))
        elif move == "extend_segment" and neighbors[0][1] == 1:
            choices.append((index, None))
        elif move == "contract_segment" and neighbors[0][1] == 1:
            ref = next(iter(payload["slot"].items()))
            uses = [j for j, r in enumerate(records) if j != index and ref in _references(r)]
            # Only a created chain atom used once as a later one-neighbor birth
            # anchor may disappear. Additional closure/refinement uses of THAT
            # atom prevent its contraction; untouched chain segments may shrink.
            if len(uses) == 1:
                child = records[uses[0]]
                if child["executor_rule"] == "atom_insert" and child["payload"]["neighbors"] == [
                    [payload["slot"], 1]
                ]:
                    choices.append((index, uses[0]))
    return tuple(choices)


def mutate_parameter(program: EditProgram, move: str, choice) -> EditProgram:
    if choice not in parameter_choices(program, move):
        raise ValueError("parameter mutation is outside its declared conditional choices")
    records = [json.loads(t) for t in program.marks]
    index, parameter = choice
    selected = records[index]
    blocks, start = [], 0
    for block in program.blocks:
        block_records = []
        for i in range(start, block.stop):
            record = records[i]
            if i == index:
                if move == "created_atom_type":
                    record["payload"]["atom_type"] = ORGANIC_VOCABULARY.element_of(parameter)
                    record["payload"]["implicit_h_count"] = ORGANIC_VOCABULARY.h_count(
                        parameter, record["payload"]["neighbors"][0][1], 0
                    )
                elif move == "extend_segment":
                    # A fresh one-neighbor birth, not a multi-neighbor insertion.
                    anchor = record["payload"]["neighbors"][0][0]
                    fresh = {"created": len(records) + 1}
                    extension = json.loads(_json(record))
                    extension["payload"].update(
                        slot=fresh,
                        atom_type=ELEMENT_TO_IDX["C"],
                        formal_charge=0,
                        implicit_h_count=3,
                        neighbors=[[anchor, 1]],
                    )
                    block_records.append(extension)
                    record["payload"]["neighbors"] = [[fresh, 1]]
                else:
                    continue
            if move == "contract_segment" and i == parameter:
                record["payload"]["neighbors"] = selected["payload"]["neighbors"]
            block_records.append(record)
        blocks.append(block_records)
        start = block.stop
    return _rebuild(program, blocks)


def branch_components(program: EditProgram) -> tuple[tuple[int, ...], ...]:
    """Dependency/operand-connected components, not a chemical independence proof."""
    graph = compile_program_graph(program)
    pending, components = set(range(len(program.blocks))), []
    while pending:
        component, frontier = set(), [min(pending)]
        while frontier:
            index = frontier.pop()
            if index in component:
                continue
            component.add(index)
            frontier.extend(b if a == index else a for a, b in graph.edges if index in (a, b))
        pending.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def select_branch(program: EditProgram, indices) -> tuple[EditProgram, tuple[int, ...]]:
    indices = tuple(sorted(set(indices)))
    if not indices or any(type(i) is not int or not 0 <= i < len(program.blocks) for i in indices):
        raise ValueError("branch requires existing block indices")
    records, labels, roots, created = [], [], [], {}
    starts = (0, *(b.stop for b in program.blocks[:-1]))
    for index in indices:
        for text in program.marks[starts[index] : program.blocks[index].stop]:
            record = json.loads(text)
            if record["executor_rule"] == "atom_insert":
                created[tuple(record["payload"]["slot"].items())] = {"created": len(created)}

            def translate(ref):
                if "input" in ref:
                    if ref["input"] not in roots:
                        roots.append(ref["input"])
                    return {"input": roots.index(ref["input"])}
                key = tuple(ref.items())
                if key not in created:
                    raise ValueError("branch is not dependency-closed; a created atom is missing")
                return created[key]

            records.append(_json(_slots(record, translate)))
        labels.append(ProgramBlock(program.blocks[index].label, len(records)))
    return EditProgram(
        tuple(program.input_atoms[i] for i in roots),
        tuple(tuple(program.input_bonds[i][j] for j in roots) for i in roots),
        tuple(program.environments[i] for i in roots),
        tuple(records),
        tuple(labels),
    ), tuple(roots)


def replace_branch(
    source, program, assignment, removed, donor, donor_assignment, *, max_primitives, max_blocks
):
    components = branch_components(program)
    removed = tuple(sorted(removed))
    if removed not in components:
        raise ValueError("replacement must name one complete independent program branch")
    retained = sorted(set(range(len(program.blocks))) - set(removed))
    if not retained:
        return donor, tuple(donor_assignment)
    keep, roots = select_branch(program, retained)
    return combine_bound_programs(
        source,
        ((keep, tuple(assignment[i] for i in roots)), (donor, tuple(donor_assignment))),
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )


def mutate_attachment(source, program, assignment, rng, *, max_changed=2, max_bindings=64):
    census = attachment_bindings(program, source, max_bindings=max_bindings)
    choices = [
        (a, d)
        for a, d in zip(census.assignments, census.context_distances, strict=True)
        if 1 <= sum(x != y for x, y in zip(a, assignment, strict=True)) <= max_changed
    ]
    if not choices:
        raise ValueError("no alternative context-compatible bounded attachment assignment")
    import numpy as np

    distances = np.asarray([d for _, d in choices], dtype=float)
    weights = np.exp(-(distances - distances.min()))
    weights = 0.1 / len(weights) + 0.9 * weights / weights.sum()
    at = int(rng.choice(len(choices), p=weights))
    return choices[at][0], {
        "bindings": len(census.assignments),
        "truncated": census.truncated,
        "selected_probability": float(weights[at]),
    }
