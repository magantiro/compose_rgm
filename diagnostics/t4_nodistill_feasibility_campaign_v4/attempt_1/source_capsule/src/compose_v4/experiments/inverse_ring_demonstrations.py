"""Derive executable whole-ring demonstrations from exact molecular endpoints.

This is development-data preparation, never a target-free proposal sampler.
Targets are inspected to identify detachable peripheral cycles. Every removal
uses the production executor and every reconstruction uses the existing
parameterized ring compiler. Failed cuts are retained with reason codes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from compose_v4.control.macro_engine import ELEMENT_CODE
from compose_v4.control.ring_program import RingSpec, construction_branches, real_slots
from compose_v4.experiments.whole_ring_plan import (
    RingRequest,
    compile_ring,
    execute_program,
    fresh_slot,
)
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.operators import AtomDelete, AtomInsert
from compose_v4.rewrite.trace_shard import encode_state


def peripheral_cycles(graph) -> tuple[tuple[int, ...], ...]:
    """All chordless five/six cycles in persistent slots, in deterministic order.

    Cycle size here is a construction-option parameter, not a validity filter
    on the broader generator. No canonical-SMILES atom order is used.
    """
    real = real_slots(graph)
    adjacency = {i: set(map(int, np.flatnonzero(graph.bonds[i]))) & real for i in real}
    found = set()
    for root in sorted(real):
        stack = [(root,)]
        while stack:
            path = stack.pop()
            if (
                len(path) in (5, 6)
                and root in adjacency[path[-1]]
                and path[1] < path[-1]
                and all(len(adjacency[slot] & set(path)) == 2 for slot in path)
            ):
                found.add(path)
            if len(path) < 6:
                for neighbor in sorted(adjacency[path[-1]] - set(path), reverse=True):
                    if neighbor > root:
                        stack.append((*path, neighbor))
    return tuple(sorted(found))


def _cuts(graph, cycle) -> Iterator[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Candidate structural cuts; descriptor/executor checks decide feasibility."""
    members = set(cycle)
    external = {slot: set(map(int, np.flatnonzero(graph.bonds[slot]))) - members for slot in cycle}
    outside_edges = [(slot, neighbor) for slot in cycle for neighbor in sorted(external[slot])]
    for start, anchor in outside_edges:
        decorations = tuple(
            sorted(
                neighbor
                for edge in outside_edges
                if edge != (start, anchor)
                for neighbor in (edge[1],)
            )
        )
        if any(np.count_nonzero(graph.bonds[slot]) != 1 for slot in decorations):
            continue
        position = cycle.index(start)
        for direction in (1, -1):
            path = tuple(cycle[(position + direction * i) % len(cycle)] for i in range(len(cycle)))
            yield "pendant", (anchor,), path, decorations
    for position in range(len(cycle)):
        for direction in (1, -1):
            ordered = tuple(
                cycle[(position + direction * i) % len(cycle)] for i in range(len(cycle))
            )
            anchors, path = (ordered[0], ordered[-1]), ordered[1:-1]
            decorations = tuple(sorted({neighbor for slot in path for neighbor in external[slot]}))
            if all(np.count_nonzero(graph.bonds[slot]) == 1 for slot in decorations):
                yield "fused", anchors, path, decorations


def inverse_ring_demonstrations(graph) -> dict:
    """Return verified precursor -> ring programs and all attempted-cut failures.

    A precursor can be much closer to a winner than a benchmark seed. Recovery
    from such a precursor is deliberately labeled inverse-derived reconstruction,
    not autonomous seed-to-winner discovery. No probabilities or task labels are
    inferred from the existence of a program.
    """
    if graph.n_atoms != 48 or not 1 <= graph.n_real_atoms <= 40:
        raise ValueError("inverse demonstration requires a supported exact 48-slot state")
    target = canonical_state_key(graph)
    cycles = peripheral_cycles(graph)
    examples, attempts = [], []
    seen = {}
    elements = {ELEMENT_CODE[element]: element for element in ("C", "N", "O")}
    for cycle in cycles:
        if any(int(graph.atom_types[slot]) not in elements for slot in cycle):
            attempts.append({"cycle": cycle, "status": "outside_ring_descriptor_composition"})
            continue
        counts = tuple(
            sum(int(graph.atom_types[slot]) == ELEMENT_CODE[e] for slot in cycle)
            for e in ("C", "N", "O")
        )
        cuts = tuple(_cuts(graph, cycle))
        if not cuts:
            attempts.append({"cycle": cycle, "status": "no_clean_peripheral_cut"})
        for topology, anchors, path, decorations in cuts:
            attempt = {
                "cycle": cycle,
                "topology": topology,
                "anchors": anchors,
                "path": path,
                "terminal_decorations": decorations,
            }
            deletion = [encode_action("atom_delete", AtomDelete(slot)) for slot in reversed(path)]
            try:
                decoration_deletion = [
                    encode_action("atom_delete", AtomDelete(slot)) for slot in decorations
                ]
                skeleton, strip = execute_program(graph, decoration_deletion)
                precursor, removal = execute_program(skeleton, deletion)
                reverse = {
                    "actions": strip["actions"] + removal["actions"],
                    "states": strip["states"] + removal["states"][1:],
                    "endpoint": removal["endpoint"],
                    "primitive_edits": len(decorations) + len(path),
                }
            except (ValueError, InvalidRewrite) as error:
                attempts.append({**attempt, "status": "reverse_rejected", "reason": str(error)})
                continue
            successes, failures = [], []
            for electronic in ("aromatic", "nonaromatic"):
                spec = RingSpec(topology, len(cycle), counts, electronic)
                branches = [
                    branch
                    for branch in construction_branches(precursor, real_slots(precursor), spec)
                    if branch[0] == anchors
                ]
                for branch in branches:
                    request = RingRequest(
                        spec,
                        anchors,
                        tuple(elements[int(graph.atom_types[slot])] for slot in path),
                        branch[1],
                    )
                    try:
                        product, forward = compile_ring(precursor, request)
                    except (ValueError, InvalidRewrite) as error:
                        failures.append({"request": request.payload(), "reason": str(error)})
                        continue
                    if canonical_state_key(product) != canonical_state_key(skeleton):
                        failures.append(
                            {"request": request.payload(), "reason": "different_canonical_endpoint"}
                        )
                        continue
                    # Put stripped terminal decorations back with ordinary
                    # one-neighbor births. Transport only the known new-ring
                    # slot correspondence; never rebuild state from SMILES.
                    mapping = dict(zip(path, forward["progress"]["path"], strict=True))
                    decorated = product
                    decoration_actions = []
                    try:
                        for slot in decorations:
                            old_anchor = int(np.flatnonzero(graph.bonds[slot])[0])
                            mark = AtomInsert(
                                fresh_slot(decorated),
                                int(graph.atom_types[slot]),
                                int(graph.formal_charges[slot]),
                                int(graph.implicit_h_counts[slot]),
                                (
                                    (
                                        mapping.get(old_anchor, old_anchor),
                                        int(graph.bonds[slot, old_anchor]),
                                    ),
                                ),
                            )
                            record = encode_action("atom_insert", mark)
                            decorated, _ = execute_program(decorated, [record])
                            decoration_actions.append(record)
                        restored, redecoration = execute_program(product, decoration_actions)
                    except (ValueError, InvalidRewrite) as error:
                        failures.append(
                            {"request": request.payload(), "reason": f"redecoration: {error}"}
                        )
                        continue
                    if canonical_state_key(restored) != target:
                        failures.append(
                            {
                                "request": request.payload(),
                                "reason": "redecoration_endpoint_mismatch",
                            }
                        )
                        continue
                    # compile_ring already checks each descriptor against the
                    # existing support contract. Independently replay its saved
                    # trace to bind the exact persistent-slot outputs as well.
                    replayed, replay = execute_program(precursor, forward["actions"])
                    if replay["states"] != forward["states"] or canonical_state_key(
                        replayed
                    ) != canonical_state_key(skeleton):
                        raise RuntimeError("compiled demonstration failed independent exact replay")
                    identity = (canonical_state_key(precursor), spec.option, target)
                    successes.append({"option": spec.option, "precursor": identity[0]})
                    if identity in seen:
                        examples[seen[identity]]["descriptor_aliases"].append(request.payload())
                        continue
                    seen[identity] = len(examples)
                    examples.append(
                        {
                            "option": spec.option,
                            "precursor": identity[0],
                            "target": target,
                            "precursor_state": encode_state(precursor),
                            "reverse": reverse,
                            "forward": forward,
                            "redecoration": redecoration,
                            "ring_endpoint": canonical_state_key(product),
                            "origin_cycle": cycle,
                            "descriptor_aliases": [request.payload()],
                            "evidence": "inverse-derived, exact-replayed option demonstration",
                        }
                    )
            attempts.append(
                {
                    **attempt,
                    "status": "reconstructed" if successes else "no_exact_forward_reconstruction",
                    "successful_descriptors": successes,
                    "forward_failures": failures,
                }
            )
    return {
        "target": target,
        "cycles": cycles,
        "examples": examples,
        "attempts": attempts,
        "support": "existing unrefined pendant/fused CNO five/six ring options only; generic unchanged",
    }
