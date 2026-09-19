"""Answer-known program compilation, not a learned law or blind proposal channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import ALLOWED_VALENCES, NULL_IDX, MolecularGraph
from compose_v4.control.macro_engine import ELEMENT_CODE
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.ring_program import (
    RingProgress,
    RingSpec,
    completed_construction,
    construction_branches,
    construction_indices,
    real_slots,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.action_codec_v4 import decode_action, encode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, CycleCloseEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state


@dataclass(frozen=True)
class RingRequest:
    """All ring choices fixed before execution; new atoms in growth order."""

    spec: RingSpec
    anchors: tuple[int, ...]
    elements: tuple[str, ...]
    pattern: tuple[int, ...]

    def __post_init__(self):
        if self.spec.refine:
            raise ValueError("this diagnostic compiles construction, not unspecified refinement")
        if len(self.anchors) != (2 if self.spec.topology == "fused" else 1):
            raise ValueError("ring attachment arity disagrees with topology")
        if len(set(self.anchors)) != len(self.anchors) or any(
            type(i) is not int or i < 0 for i in self.anchors
        ):
            raise ValueError("anchors require distinct nonnegative persistent slots")
        if len(self.elements) != self.spec.growth or any(
            e not in ("C", "N", "O") for e in self.elements
        ):
            raise ValueError("new ring elements must match the C/N/O growth length")
        if len(self.pattern) != self.spec.size or any(
            type(b) is not int or b not in (1, 2) for b in self.pattern
        ):
            raise ValueError("ring pattern must specify every cycle bond")

    def payload(self):
        return {
            "option": self.spec.option,
            "anchors": list(self.anchors),
            "elements": list(self.elements),
            "pattern": list(self.pattern),
        }


def fresh_slot(graph: MolecularGraph) -> int:
    empty = np.flatnonzero(graph.atom_types == NULL_IDX)
    if not len(empty) or graph.n_real_atoms >= 40:
        raise ValueError("no room inside the frozen 40-active/48-slot support")
    return int(empty[0])


def execute_program(graph: MolecularGraph, actions: list[dict]) -> tuple[MolecularGraph, dict]:
    """Execute the entire candidate privately; return only a complete receipt."""
    if graph.n_atoms != 48 or not 1 <= graph.n_real_atoms <= 40:
        raise ValueError("expected an exact supported 48-slot source")
    system = editing_v2_semantic_rewrite_system()
    current, states = graph, [encode_state(graph)]
    for index, record in enumerate(actions):
        family, action = decode_action(record)
        product = system.apply(current, family, action)
        if not 1 <= product.n_real_atoms <= 40 or not charge_policy_preserved(current, product):
            raise ValueError(f"step {index}: size or charge policy failed")
        if canonical_state_key(current) == canonical_state_key(product):
            raise ValueError(f"step {index}: program contains a canonical self-event")
        current = product
        states.append(encode_state(current))
    return current, {
        "actions": actions,
        "states": states,
        "endpoint": canonical_state_key(current),
        "primitive_edits": len(actions),
    }


def compile_ring(graph: MolecularGraph, request: RingRequest) -> tuple[MolecularGraph, dict]:
    """Linear descriptor execution with existing contracts, no global fiber search.

    This establishes executor/descriptor support only. It does not assign an
    R_theta probability or pretend that a forced descriptor is a sampled law.
    """
    if graph.n_atoms != 48 or not 1 <= graph.n_real_atoms <= 40:
        raise ValueError("expected an exact supported 48-slot source")
    spec = request.spec
    branch = (request.anchors, request.pattern)
    if branch not in construction_branches(graph, real_slots(graph), spec):
        raise ValueError("complete ring descriptor is not applicable at the supplied anchors")
    system = editing_v2_semantic_rewrite_system()
    origin, current, progress = graph, graph, RingProgress()
    states, actions = [encode_state(graph)], []
    for step in range(spec.horizon):
        if step < spec.growth:
            position = step + int(spec.topology == "fused")
            order = 1 if spec.topology == "pendant" and step == 0 else request.pattern[position - 1]
            element = request.elements[step]
            anchor = progress.path[-1] if progress.path else request.anchors[0]
            slot = fresh_slot(current)
            valences = ALLOWED_VALENCES[element]
            if len(valences) != 1:
                raise ValueError("the ring request must select an unambiguous neutral valence")
            action = AtomInsert(
                slot, ELEMENT_CODE[element], 0, valences[0] - order, ((anchor, order),)
            )
            family = "atom_insert"
            following = RingProgress(request.anchors, request.pattern, progress.path + (slot,))
        else:
            endpoint = request.anchors[1] if spec.topology == "fused" else progress.path[0]
            a, b = sorted((progress.path[-1], endpoint))
            order = request.pattern[-2] if spec.topology == "fused" else request.pattern[-1]
            family, action, following = "cycle_close", CycleCloseEdge(a, b, order), progress
        # A singleton descriptor is a membership check, NOT a reference law.
        if construction_indices(
            (family,), (action,), (1.0,), current, spec, progress, branch, step
        ) != [0]:
            raise ValueError(
                f"step {step}: requested mark fails the existing ring descriptor contract"
            )
        product = system.apply(current, family, action)
        if not charge_policy_preserved(current, product):
            raise ValueError(f"step {step}: ring program changed the charge policy")
        following.validate(product, origin, real_slots(product), step + 1, spec)
        if canonical_state_key(current) == canonical_state_key(product):
            raise ValueError(f"step {step}: ring program contains a canonical self-event")
        actions.append(encode_action(family, action))
        states.append(encode_state(product))
        current, progress = product, following
    if not completed_construction(origin, current, progress, spec):
        raise ValueError("completed ring does not satisfy its existing construction contract")
    return current, {
        "request": request.payload(),
        "actions": actions,
        "states": states,
        "progress": progress.payload(),
        "primitive_edits": len(actions),
        "endpoint": canonical_state_key(current),
        "existing_ring_contract": True,
        "learned_law_support": "not_evaluated",
    }


def verify_trace(source: dict, stages: list[dict], target: str) -> dict:
    """Independent full replay checks every persisted state, not just the endpoint."""
    current, total = decode_state(source), 0
    for stage in stages:
        if exact_graph_key(current) != exact_graph_key(decode_state(stage["states"][0])):
            raise ValueError("program boundary loses exact persistent-slot state")
        endpoint, receipt = execute_program(current, stage["actions"])
        if receipt["states"] != stage["states"] or receipt["endpoint"] != stage["endpoint"]:
            raise ValueError("saved program states disagree with exact replay")
        current = endpoint
        total += len(stage["actions"])
    if canonical_state_key(current) != target:
        raise ValueError("complete program plan did not recover the exact 2D winner")
    return {
        "exact_endpoint": True,
        "exact_states": True,
        "primitive_edits": total,
        "execution_programs": len(stages),
    }


def first_winner_plan(path: dict, *, electronic_preparation: bool = False) -> list[dict]:
    """The declared answer-known structural plan, with persistent-slot continuity.

    Only this diagnostic recipe uses a winner witness. The ring compiler above
    takes a standalone descriptor and never reads a target or witness.
    """
    source = path["source_state"]
    current = decode_state(source)
    stages = []

    def append(name, result, **metadata):
        nonlocal current
        current, receipt = result
        stages.append({"name": name, **metadata, **receipt})

    append(
        "remodel_linker",
        execute_program(current, path["actions"][:4]),
        interface="diagnostic primitive program; shrink x3, grow x1",
    )
    _, link = decode_action(path["actions"][3])
    request = RingRequest(
        RingSpec("pendant", 6, (6, 0, 0), "aromatic"), (link.slot,), ("C",) * 6, (1, 2, 1, 2, 1, 2)
    )
    append(
        "pendant_benzene",
        compile_ring(current, request),
        interface="existing parameterized ring option",
    )
    pendant = stages[-1]["progress"]["path"]
    anchors = (pendant[2], pendant[3])
    shared = int(current.bonds[anchors])
    request = RingRequest(
        RingSpec("fused", 6, (6, 0, 0), "nonaromatic"), anchors, ("C",) * 4, (1, 1, 1, 1, 1, shared)
    )
    append(
        "fused_six_ring",
        compile_ring(current, request),
        interface="existing parameterized ring option",
    )
    carbon = stages[-1]["progress"]["path"][-1]
    oxygen = AtomInsert(fresh_slot(current), ELEMENT_CODE["O"], 0, 0, ((carbon, 2),))
    append(
        "ring_carbonyl",
        execute_program(current, [encode_action("atom_insert", oxygen)]),
        interface="one ordinary grow primitive, not the halogen decorate option",
    )
    start = 14 if electronic_preparation else 16
    append(
        "core_carbonyl_insertion",
        execute_program(current, path["actions"][start:20]),
        interface="diagnostic core program; not the existing restricted expand_ring option",
    )
    return stages
