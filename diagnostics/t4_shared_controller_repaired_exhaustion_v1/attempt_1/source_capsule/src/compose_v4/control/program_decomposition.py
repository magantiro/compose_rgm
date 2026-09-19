"""Conservative, replay-verified decomposition of opaque construction programs.

Operand/dependency components propose branch boundaries, not chemical safety.
Each component must execute alone and the recombined construction must retain
the original endpoint. Global application conditions can still cause abstention.
"""

from dataclasses import replace

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.edit_program import ProgramBlock
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.program_mutation import branch_components, select_branch
from compose_v4.rewrite.trace_shard import decode_state


def _observed_components(atomized, trace):
    """Merge operand components using actual replay writes, including neighbors.

    Deleting an atom changes its neighbor's valence/hydrogens even though the
    explicit mark mentions only the deleted atom. Adjacent deletions therefore
    belong together. Atom tokens, not recycled physical slots, track overlap.
    """
    graph = decode_state(trace["states"][0])
    tokens = {i: ("original", i) for i in range(graph.n_atoms) if is_element(graph.atom_types[i])}
    writes = []
    for i, action in enumerate(trace["actions"]):
        following = decode_state(trace["states"][i + 1])
        after_tokens = dict(tokens)
        if action["executor_rule"] == "atom_delete":
            del after_tokens[action["payload"]["v"]]
        elif action["executor_rule"] == "atom_insert":
            after_tokens[action["payload"]["slot"]] = ("birth", i)
        changed = set(
            np.flatnonzero(
                (graph.atom_types != following.atom_types)
                | (graph.formal_charges != following.formal_charges)
                | (graph.implicit_h_counts != following.implicit_h_counts)
                | np.any(graph.bonds != following.bonds, axis=1)
            )
        )
        writes.append(
            {
                mapping[slot]
                for mapping in (tokens, after_tokens)
                for slot in changed
                if slot in mapping
            }
        )
        graph, tokens = following, after_tokens
    groups = [set(c) for c in branch_components(atomized)]
    for i in range(len(writes)):
        for j in range(i):
            if writes[i] & writes[j]:
                touching = [g for g in groups if i in g or j in g]
                union = set().union(*touching)
                groups = [g for g in groups if not g & union] + [union]
    return tuple(sorted((tuple(sorted(g)) for g in groups), key=lambda g: g[0]))


def verified_branches(source, program, assignment, *, max_primitives=32, max_branches=4):
    original = compile_program_graph(program)
    _, reference = execute_program_graph(
        source,
        original,
        assignment,
        max_primitives=max_primitives,
        max_blocks=len(program.blocks),
    )
    atomized = replace(
        program,
        blocks=tuple(ProgramBlock(f"operation:{i}", i + 1) for i in range(len(program.marks))),
    )
    components = _observed_components(atomized, reference)
    report = {
        "original_program_id": program.program_id,
        "primitive_components": [list(c) for c in components],
        "reference_endpoint": reference["endpoint"],
        "status": "not_decomposed",
        "boundary_rule": "operand/dependency components merged by exact replay write overlap",
    }
    if not 2 <= len(components) <= max_branches:
        return (
            program,
            tuple(assignment),
            {**report, "reason": "single_component_or_exceeds_declared_branch_limit"},
        )
    parts, receipts = [], []
    try:
        for i, component in enumerate(components):
            branch, roots = select_branch(atomized, component)
            branch = replace(
                branch, blocks=(ProgramBlock(f"dependency_branch:{i}", len(branch.marks)),)
            )
            anchors = tuple(assignment[j] for j in roots)
            _, receipt = execute_program_graph(
                source,
                compile_program_graph(branch),
                anchors,
                max_primitives=max_primitives,
                max_blocks=1,
            )
            parts.append((branch, anchors))
            receipts.append({"program_id": branch.program_id, "endpoint": receipt["endpoint"]})
        combined, anchors = combine_bound_programs(
            source,
            tuple(parts),
            max_primitives=max_primitives,
            max_blocks=max_branches,
        )
        _, receipt = execute_program_graph(
            source,
            compile_program_graph(combined),
            anchors,
            max_primitives=max_primitives,
            max_blocks=max_branches,
        )
    except ValueError as error:
        return (
            program,
            tuple(assignment),
            {**report, "reason": "independent_or_joint_replay_failed", "detail": str(error)},
        )
    if receipt["endpoint"] != reference["endpoint"]:
        return program, tuple(assignment), {**report, "reason": "joint_endpoint_changed"}
    return (
        combined,
        anchors,
        {
            **report,
            "status": "verified",
            "branches": receipts,
            "decomposed_program_id": combined.program_id,
            "scope": "these bindings and this verified schedule; not universal commutativity",
        },
    )
