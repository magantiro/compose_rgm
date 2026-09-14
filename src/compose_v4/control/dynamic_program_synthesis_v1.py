"""Focused Dynamic-v1 program modules diagnosed from three T4 development cells.

This module is deliberately separate from ``dynamic_program_synthesis`` so the
running Dynamic-v0 diagnostic remains byte-identical.  It contains no target
name, endpoint, score, source atom index or stored complete route.  Answer-known
routes may supply request values to the public deterministic compilers in an
offline reachability audit; the stochastic proposer must construct those
values from the current graph.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import permutations, product
from time import perf_counter

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    MolecularGraph,
    is_element,
)
from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    FRESH_SYNTHESIS_PROBABILITY,
    MODULE_COUNT_PROBABILITIES,
    NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES,
    NEAR_CAPACITY_MODULE_WEIGHTS,
    DynamicProgramOptimizer,
    _pendant_fragments,
    _stage,
    compile_generic_module,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES as V0_GENERIC_MODULES,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.ring_program import (
    RingSpec,
    construction_branches,
    default_ring_options,
    residual,
)
from compose_v4.experiments.whole_ring_plan import (
    RingRequest,
    compile_ring,
    execute_program,
    fresh_slot,
)
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    CycleCloseEdge,
    CycleOpenEdge,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

SCHEMA = "dynamic_generic_program_synthesis_v2"
GENERIC_MODULES = (
    *V0_GENERIC_MODULES,
    "ring_path_remodel",
    "construct_substituted_ring",
)
MAX_RING_PATH_INTERNAL_ATOMS = 4
MAX_RING_PATH_REPLACEMENT_ATOMS = 4
MAX_RING_SUBSTITUENT_ATOMS = 6
MAX_CONTEXT_CANDIDATES = 16
MAX_CONTEXT_TRIALS = 32
MAX_PANEL_CACHE_ENTRIES = 128
CONTEXT_EXPLORATION = 0.2
V1_COMPOSITION_PATTERNS = (
    ("substituent_delete", "functionalize"),
    ("substituent_delete", "construct_substituted_ring"),
    ("construct_substituted_ring", "ring_path_remodel"),
    (
        "substituent_delete",
        "construct_substituted_ring",
        "ring_path_remodel",
    ),
)
V1_PATTERN_PROBABILITY = 0.5
V1_CAPABILITY_PROBABILITY = 0.35
GENERIC_SUBSTITUENT_TEMPLATES = (
    "none",
    "methyl",
    "hydroxy",
    "amino",
    "carboxyl",
    "methyl_carboxyl",
)


@dataclass(frozen=True)
class CycleOpenRequest:
    """One explicit generic ring-opening instance for a forced support test."""

    a: int
    b: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.a, self.b)):
            raise ValueError("cycle-open slots must be nonnegative integers")
        if self.a >= self.b:
            raise ValueError("cycle-open slots must be canonically ordered")


@dataclass(frozen=True)
class PendantDeleteRequest:
    """Delete one bounded fragment attached through one retained boundary."""

    fragment_slots: tuple[int, ...]
    retained_anchor: int

    def __post_init__(self):
        if (
            not self.fragment_slots
            or len(self.fragment_slots) > 8
            or len(set(self.fragment_slots)) != len(self.fragment_slots)
            or any(type(slot) is not int or slot < 0 for slot in self.fragment_slots)
            or type(self.retained_anchor) is not int
            or self.retained_anchor < 0
            or self.retained_anchor in self.fragment_slots
        ):
            raise ValueError("malformed bounded pendant-deletion request")


@dataclass(frozen=True)
class FunctionalizeRequest:
    """Attach one neutral supported atom to a hydrogen-bearing current atom."""

    anchor: int
    element: str
    order: int = 1

    def __post_init__(self):
        if type(self.anchor) is not int or self.anchor < 0:
            raise ValueError("functionalization anchor must be a nonnegative slot")
        if self.element not in ("C", "N", "O", "F") or self.order not in (1, 2):
            raise ValueError("unsupported functionalization element or bond order")


@dataclass(frozen=True)
class CarbonylInsertRequest:
    """Insert one neutral double-bonded oxygen at a compatible carbon."""

    anchor: int

    def __post_init__(self):
        if type(self.anchor) is not int or self.anchor < 0:
            raise ValueError("carbonyl anchor must be a nonnegative slot")


@dataclass(frozen=True)
class RingSystemRestateRequest:
    """One supported joint bond-order restatement on a current ring system."""

    changes: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        if not self.changes or any(
            len(change) != 3
            or any(type(value) is not int for value in change)
            or change[0] < 0
            or change[0] >= change[1]
            or change[2] not in (1, 2)
            for change in self.changes
        ):
            raise ValueError("ring-system restatement changes are malformed")


@dataclass(frozen=True)
class RingPathRemodelRequest:
    """Replace one bounded cyclic path while retaining its boundary atoms."""

    path: tuple[int, ...]
    replacement_elements: tuple[str, ...] = ()
    replacement_bond_orders: tuple[int, ...] = (1,)

    def __post_init__(self):
        internal = len(self.path) - 2
        if (
            not 1 <= internal <= MAX_RING_PATH_INTERNAL_ATOMS
            or len(set(self.path)) != len(self.path)
            or any(type(slot) is not int or slot < 0 for slot in self.path)
        ):
            raise ValueError(
                "ring path requires two boundaries and one to four internal atoms"
            )
        if (
            len(self.replacement_elements) > MAX_RING_PATH_REPLACEMENT_ATOMS
            or any(
                element not in ("C", "N", "O") for element in self.replacement_elements
            )
            or len(self.replacement_bond_orders) != len(self.replacement_elements) + 1
            or any(order not in (1, 2) for order in self.replacement_bond_orders)
        ):
            raise ValueError("ring-path replacement atoms or bond orders are malformed")


@dataclass(frozen=True)
class RingSubstituentAtom:
    """One topologically ordered atom in a branch attached to a new ring."""

    parent_kind: str
    parent_index: int
    element: str
    order: int = 1

    def __post_init__(self):
        if self.parent_kind not in ("ring", "substituent"):
            raise ValueError(
                "substituent parent must be a ring or earlier substituent atom"
            )
        if type(self.parent_index) is not int or self.parent_index < 0:
            raise ValueError("substituent parent index must be nonnegative")
        if self.element not in ("C", "N", "O", "F") or self.order not in (1, 2):
            raise ValueError("unsupported substituent atom or bond order")


@dataclass(frozen=True)
class SubstitutedRingRequest:
    """Joint ring and bounded branch specification, fixed before execution."""

    ring: RingRequest
    substituents: tuple[RingSubstituentAtom, ...] = ()

    def __post_init__(self):
        if len(self.substituents) > MAX_RING_SUBSTITUENT_ATOMS:
            raise ValueError(
                "substituted ring exceeds its bounded branch-atom allowance"
            )
        for index, atom in enumerate(self.substituents):
            if atom.parent_kind == "ring" and atom.parent_index >= self.ring.spec.size:
                raise ValueError(
                    "substituent ring position is outside the constructed cycle"
                )
            if atom.parent_kind == "substituent" and atom.parent_index >= index:
                raise ValueError(
                    "substituent atoms may refer only to earlier branch atoms"
                )

    def payload(self) -> dict:
        return {
            "ring": self.ring.payload(),
            "substituents": [asdict(atom) for atom in self.substituents],
        }


@dataclass(frozen=True)
class BoundModuleCandidate:
    """One exactly executed binding option with a non-oracle structural rank."""

    product: MolecularGraph
    stage: dict
    structural_rank: tuple


def _real_slots(graph: MolecularGraph) -> set[int]:
    return {int(slot) for slot in np.flatnonzero(is_element(graph.atom_types))}


def _connected_without(
    graph: MolecularGraph, start: int, goal: int, excluded: set[int]
) -> bool:
    seen, pending = {start}, [start]
    while pending:
        slot = pending.pop()
        if slot == goal:
            return True
        for neighbor in np.flatnonzero(graph.bonds[slot]):
            neighbor = int(neighbor)
            if neighbor not in excluded and neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def _one_boundary_fragments(
    graph: MolecularGraph, *, maximum: int = 8
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return bridge and pendant-ring fragments separated by one retained atom.

    Removing a retained articulation atom exposes every connected lobe attached
    only through it.  This includes ordinary bridge-separated substituents and
    bounded cyclic lobes that the v0 single-edge cutter cannot expose.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError("fragment size bound must be a positive integer")
    real = _real_slots(graph)
    adjacency = {
        slot: {int(neighbor) for neighbor in np.flatnonzero(graph.bonds[slot])}
        for slot in real
    }
    fragments = set(_pendant_fragments(graph, maximum=maximum))
    for anchor in sorted(real):
        remaining = real - {anchor}
        while remaining:
            start = min(remaining)
            component, pending = {start}, [start]
            while pending:
                slot = pending.pop()
                for neighbor in adjacency[slot] & remaining - component:
                    component.add(neighbor)
                    pending.append(neighbor)
            remaining -= component
            boundary = {
                neighbor
                for slot in component
                for neighbor in adjacency[slot]
                if neighbor not in component
            }
            if boundary == {anchor} and 1 <= len(component) <= maximum:
                fragments.add((tuple(sorted(component)), anchor))
    return tuple(sorted(fragments, key=lambda row: (len(row[0]), row)))


def _nearby_anchors(graph: MolecularGraph, roots, *, radius: int = 2) -> frozenset[int]:
    """Bounded graph-context expansion used only for structural binding rank."""
    real = _real_slots(graph)
    frontier = set(roots) & real
    seen = set(frontier)
    for _ in range(radius):
        following = {
            int(neighbor)
            for slot in frontier
            for neighbor in np.flatnonzero(graph.bonds[slot])
            if int(neighbor) in real and int(neighbor) not in seen
        }
        seen.update(following)
        frontier = following
    return frozenset(seen)


def _neutral_insert(
    graph: MolecularGraph, anchor: int, element: str, order: int
) -> dict:
    if anchor not in _real_slots(graph):
        raise ValueError("atom insertion anchor is absent")
    valences = ALLOWED_VALENCES[element]
    if len(valences) != 1 or valences[0] < order:
        raise ValueError("atom insertion needs one compatible neutral valence")
    action = AtomInsert(
        fresh_slot(graph),
        ELEMENT_TO_IDX[element],
        0,
        valences[0] - order,
        ((anchor, order),),
    )
    return encode_action("atom_insert", action)


def _single_action_stage(
    source: MolecularGraph, label: str, record: dict, parameters: dict
):
    product, receipt = execute_program(source, [record])
    return product, _stage(label, receipt, parameters)


def compile_cycle_open(source: MolecularGraph, request: CycleOpenRequest):
    return _single_action_stage(
        source,
        "cycle_open",
        encode_action("cycle_open", CycleOpenEdge(request.a, request.b)),
        asdict(request),
    )


def _delete_order(
    source: MolecularGraph,
    remaining: frozenset[int],
    *,
    work: list[int] | None = None,
    max_expansions: int = 64,
) -> tuple[dict, ...] | None:
    """Deterministic bounded backtracking through exact legal atom deletions."""
    if work is None:
        work = [0]
    if work[0] >= max_expansions:
        return None
    if not remaining:
        return ()
    for slot in sorted(remaining):
        work[0] += 1
        if work[0] > max_expansions:
            return None
        record = encode_action("atom_delete", AtomDelete(slot))
        try:
            product, _ = execute_program(source, [record])
        except ValueError:
            continue
        suffix = _delete_order(
            product,
            remaining - {slot},
            work=work,
            max_expansions=max_expansions,
        )
        if suffix is not None:
            return (record, *suffix)
    return None


def compile_pendant_delete(source: MolecularGraph, request: PendantDeleteRequest):
    real = _real_slots(source)
    fragment = frozenset(request.fragment_slots)
    if not fragment < real or request.retained_anchor not in real:
        raise ValueError("pendant-deletion slots are absent from the source")
    boundary = {
        int(neighbor)
        for slot in fragment
        for neighbor in np.flatnonzero(source.bonds[slot])
        if int(neighbor) not in fragment
    }
    if boundary != {request.retained_anchor}:
        raise ValueError("fragment is not attached only through its retained anchor")
    actions = _delete_order(source, fragment)
    if actions is None:
        raise ValueError("pendant fragment has no exact legal deletion order")
    product, receipt = execute_program(source, list(actions))
    return product, _stage(
        "substituent_delete",
        receipt,
        {
            "fragment_slots": list(request.fragment_slots),
            "retained_anchor": request.retained_anchor,
            "binding_source": "explicit generic request",
        },
    )


def compile_functionalize(source: MolecularGraph, request: FunctionalizeRequest):
    return _single_action_stage(
        source,
        "functionalize",
        _neutral_insert(source, request.anchor, request.element, request.order),
        asdict(request),
    )


def compile_carbonyl_insert(source: MolecularGraph, request: CarbonylInsertRequest):
    if (
        request.anchor not in _real_slots(source)
        or int(source.atom_types[request.anchor]) != ELEMENT_TO_IDX["C"]
    ):
        raise ValueError("carbonyl insertion requires a current carbon anchor")
    return _single_action_stage(
        source,
        "carbonyl_insert",
        _neutral_insert(source, request.anchor, "O", 2),
        asdict(request),
    )


def compile_ring_system_restate(
    source: MolecularGraph, request: RingSystemRestateRequest
):
    action = RingSystemRestate(
        tuple(BondOrderChange(*change) for change in request.changes)
    )
    return _single_action_stage(
        source,
        "ring_system_restate",
        encode_action("ring_system_restate", action),
        {"changes": [list(change) for change in request.changes]},
    )


def compile_ring_path_remodel(source: MolecularGraph, request: RingPathRemodelRequest):
    real = _real_slots(source)
    if not set(request.path) <= real:
        raise ValueError("ring-remodel path contains an absent atom")
    if any(
        not source.bonds[a, b]
        for a, b in zip(request.path, request.path[1:], strict=False)
    ):
        raise ValueError("ring-remodel path is not an existing bonded path")
    left, right = request.path[0], request.path[-1]
    internal = frozenset(request.path[1:-1])
    if not _connected_without(source, left, right, set(internal)):
        raise ValueError("ring-remodel path is not cyclic between retained boundaries")
    delete_actions = _delete_order(source, internal)
    if delete_actions is None:
        raise ValueError("ring-remodel path has no exact legal deletion order")
    current, _ = execute_program(source, list(delete_actions))
    actions = list(delete_actions)
    anchor = left
    created = []
    for element, order in zip(
        request.replacement_elements,
        request.replacement_bond_orders[:-1],
        strict=True,
    ):
        record = _neutral_insert(current, anchor, element, order)
        current, _ = execute_program(current, [record])
        actions.append(record)
        anchor = int(record["payload"]["slot"])
        created.append(anchor)
    a, b = sorted((anchor, right))
    close = encode_action(
        "cycle_close", CycleCloseEdge(a, b, request.replacement_bond_orders[-1])
    )
    actions.append(close)
    product, receipt = execute_program(source, actions)
    return product, _stage(
        "ring_path_remodel",
        receipt,
        {
            "path": list(request.path),
            "retained_boundaries": [left, right],
            "removed_internal_atoms": len(internal),
            "replacement_elements": list(request.replacement_elements),
            "replacement_bond_orders": list(request.replacement_bond_orders),
            "created_slots": created,
        },
    )


def compile_substituted_ring(source: MolecularGraph, request: SubstitutedRingRequest):
    ring_product, ring_receipt = compile_ring(source, request.ring)
    path = tuple(ring_receipt["progress"]["path"])
    ring_slots = (
        (request.ring.anchors[0], *path, request.ring.anchors[1])
        if request.ring.spec.topology == "fused"
        else path
    )
    if len(ring_slots) != request.ring.spec.size:
        raise RuntimeError("constructed ring slot accounting changed")
    current = ring_product
    actions = list(ring_receipt["actions"])
    branch_slots = []
    for atom in request.substituents:
        anchor = (
            ring_slots[atom.parent_index]
            if atom.parent_kind == "ring"
            else branch_slots[atom.parent_index]
        )
        record = _neutral_insert(current, anchor, atom.element, atom.order)
        current, _ = execute_program(current, [record])
        actions.append(record)
        branch_slots.append(int(record["payload"]["slot"]))
    product, receipt = execute_program(source, actions)
    return product, _stage(
        "construct_substituted_ring",
        receipt,
        {
            **request.payload(),
            "ring_slots": list(ring_slots),
            "branch_slots": branch_slots,
            "protected_complete_module": True,
        },
    )


def compile_forced_module_sequence(
    source: MolecularGraph,
    requests,
    *,
    max_primitives: int = 32,
    max_blocks: int = 8,
):
    """Compile supplied generic parameters; never infer them from a teacher route."""
    if not 1 <= len(requests) <= 3:
        raise ValueError("Dynamic-v1 forced support test requires one to three modules")
    dispatch = {
        CycleOpenRequest: compile_cycle_open,
        PendantDeleteRequest: compile_pendant_delete,
        FunctionalizeRequest: compile_functionalize,
        CarbonylInsertRequest: compile_carbonyl_insert,
        RingSystemRestateRequest: compile_ring_system_restate,
        RingPathRemodelRequest: compile_ring_path_remodel,
        SubstitutedRingRequest: compile_substituted_ring,
    }
    current, stages, modules = source, [], []
    for request in requests:
        compiler = dispatch.get(type(request))
        if compiler is None:
            raise ValueError(
                f"unknown Dynamic-v1 forced request: {type(request).__name__}"
            )
        current, stage = compiler(current, request)
        stages.append(stage)
        modules.append(stage["name"])
    program, assignment = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("Dynamic-v1 forced composition changed on exact replay")
    return (
        source,
        program,
        assignment,
        trace,
        {
            "schema_version": SCHEMA,
            "modules": modules,
            "module_count": len(modules),
            "primitive_edits": len(program.marks),
            "blocks": len(program.blocks),
            "intermediate_task_evaluations": 0,
            "teacher_parameters_deployed": False,
        },
    )


def enumerate_functionalizations(
    source: MolecularGraph,
    *,
    preferred_anchors: frozenset[int] = frozenset(),
    max_candidates: int = MAX_CONTEXT_CANDIDATES,
) -> tuple[BoundModuleCandidate, ...]:
    """Deterministic bounded binding panel with no task-oracle information."""
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("context candidate cap must be a positive integer")
    anchors = sorted(
        [
            int(slot)
            for slot in np.flatnonzero(is_element(source.atom_types))
            if int(source.implicit_h_counts[slot]) >= 1
        ],
        key=lambda slot: (slot not in preferred_anchors, slot),
    )
    candidates = []
    for anchor, element in product(anchors, ("C", "N", "O", "F")):
        try:
            product_graph, stage = compile_functionalize(
                source, FunctionalizeRequest(anchor, element)
            )
        except ValueError:
            continue
        rank = (
            int(anchor not in preferred_anchors),
            abs(product_graph.n_real_atoms - source.n_real_atoms),
            canonical_state_key(product_graph),
        )
        candidates.append(BoundModuleCandidate(product_graph, stage, rank))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda candidate: candidate.structural_rank)
    return tuple(candidates[:max_candidates])


def enumerate_pendant_deletions(
    source: MolecularGraph,
    *,
    preferred_anchors: frozenset[int] = frozenset(),
    max_candidates: int = MAX_CONTEXT_CANDIDATES,
) -> tuple[BoundModuleCandidate, ...]:
    """Enumerate exact one-boundary deletions before stochastic selection."""
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("context candidate cap must be a positive integer")
    candidates = []
    near_capacity = source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
    fragments = sorted(
        _one_boundary_fragments(source, maximum=8),
        key=lambda row: (
            row[1] not in preferred_anchors,
            -len(row[0]) if near_capacity else len(row[0]),
            row,
        ),
    )
    for fragment, anchor in fragments:
        try:
            product_graph, stage = compile_pendant_delete(
                source, PendantDeleteRequest(tuple(fragment), anchor)
            )
        except ValueError:
            continue
        rank = (
            int(anchor not in preferred_anchors),
            -len(fragment) if near_capacity else len(fragment),
            anchor,
            canonical_state_key(product_graph),
        )
        candidates.append(BoundModuleCandidate(product_graph, stage, rank))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda candidate: candidate.structural_rank)
    return tuple(candidates[:max_candidates])


def enumerate_ring_path_remodels(
    source: MolecularGraph,
    *,
    preferred_slots: frozenset[int] = frozenset(),
    max_candidates: int = MAX_CONTEXT_CANDIDATES,
) -> tuple[BoundModuleCandidate, ...]:
    """Enumerate bounded cyclic paths and exact contraction/replacement products."""
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("context candidate cap must be a positive integer")
    graph = nx.Graph()
    graph.add_nodes_from(sorted(_real_slots(source)))
    graph.add_edges_from(
        (a, b)
        for a in sorted(_real_slots(source))
        for b in sorted(_real_slots(source))
        if a < b and source.bonds[a, b]
    )
    requests = set()
    for cycle in nx.cycle_basis(graph):
        size = len(cycle)
        for start in range(size):
            for internal in range(1, min(MAX_RING_PATH_INTERNAL_ATOMS, size - 2) + 1):
                path = tuple(
                    cycle[(start + offset) % size] for offset in range(internal + 2)
                )
                path = min(path, tuple(reversed(path)))
                requests.add(RingPathRemodelRequest(path))
                for element in ("C", "N", "O"):
                    requests.add(RingPathRemodelRequest(path, (element,), (1, 1)))
    candidates, seen = [], set()
    for request in sorted(
        requests,
        key=lambda row: (
            len(row.replacement_elements),
            len(row.path),
            row.path,
            row.replacement_elements,
        ),
    ):
        try:
            product_graph, stage = compile_ring_path_remodel(source, request)
        except ValueError:
            continue
        endpoint = canonical_state_key(product_graph)
        if endpoint in seen:
            continue
        seen.add(endpoint)
        rank = (
            int(not (set(request.path) & preferred_slots)),
            len(request.replacement_elements),
            -len(request.path),
            endpoint,
        )
        candidates.append(BoundModuleCandidate(product_graph, stage, rank))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda candidate: candidate.structural_rank)
    return tuple(candidates[:max_candidates])


def _ring_specs() -> tuple[RingSpec, ...]:
    specs = []
    for option in default_ring_options():
        _, topology, size, counts, electronic, refine = option.split(":")
        specs.append(
            RingSpec(
                topology,
                int(size),
                tuple(map(int, counts.split(","))),
                electronic,
                int(refine),
            )
        )
    specs.extend(two_nitrogen_saturated_ring_specs())
    return tuple(specs)


def _substituent_atoms(
    template: str, size: int, rng
) -> tuple[RingSubstituentAtom, ...]:
    if template not in GENERIC_SUBSTITUENT_TEMPLATES:
        raise ValueError("unknown generic substituent template")
    if template == "none":
        return ()
    first = int(rng.integers(size))
    if template == "methyl":
        return (RingSubstituentAtom("ring", first, "C"),)
    if template == "hydroxy":
        return (RingSubstituentAtom("ring", first, "O"),)
    if template == "amino":
        return (RingSubstituentAtom("ring", first, "N"),)
    if template == "carboxyl":
        return (
            RingSubstituentAtom("ring", first, "C"),
            RingSubstituentAtom("substituent", 0, "O", 2),
            RingSubstituentAtom("substituent", 0, "O"),
        )
    second_choices = [position for position in range(size) if position != first]
    second = second_choices[int(rng.integers(len(second_choices)))]
    return (
        RingSubstituentAtom("ring", first, "C"),
        RingSubstituentAtom("ring", second, "C"),
        RingSubstituentAtom("substituent", 1, "O", 2),
        RingSubstituentAtom("substituent", 1, "O"),
    )


def enumerate_substituted_rings(
    source: MolecularGraph,
    rng,
    *,
    preferred_anchors: frozenset[int] = frozenset(),
    max_candidates: int = MAX_CONTEXT_CANDIDATES,
    max_trials: int = MAX_CONTEXT_TRIALS,
) -> tuple[BoundModuleCandidate, ...]:
    """Sample a bounded exact panel from a generic ring/branch parameter product.

    The panel is deterministic for the supplied RNG state.  It contains no
    target-specific parameter or score.  Context only changes the probability
    assigned to currently compatible anchors.
    """
    if any(
        type(value) is not int or value < 1 for value in (max_candidates, max_trials)
    ):
        raise ValueError(
            "ring binding candidate and trial caps must be positive integers"
        )
    specs = _ring_specs()
    candidates, seen = [], set()
    for _ in range(max_trials):
        spec = specs[int(rng.integers(len(specs)))]
        branches = construction_branches(source, _real_slots(source), spec)
        if not branches:
            continue
        contextual = np.asarray(
            [
                4.0 if set(anchors) & preferred_anchors else 1.0
                for anchors, _ in branches
            ],
            dtype=float,
        )
        branch_index = int(rng.choice(len(branches), p=contextual / contextual.sum()))
        anchors, pattern = branches[branch_index]
        counts = residual(spec, source, anchors, ())
        if any(value < 0 for value in counts) or sum(counts) != spec.growth:
            continue
        pool = []
        for element, count in zip(("C", "N", "O"), counts, strict=True):
            pool.extend([element] * count)
        orders = sorted(set(permutations(pool)))
        if not orders:
            continue
        elements = orders[int(rng.integers(len(orders)))]
        template = GENERIC_SUBSTITUENT_TEMPLATES[
            int(rng.integers(len(GENERIC_SUBSTITUENT_TEMPLATES)))
        ]
        request = SubstitutedRingRequest(
            RingRequest(spec, tuple(anchors), tuple(elements), tuple(pattern)),
            _substituent_atoms(template, spec.size, rng),
        )
        try:
            product_graph, stage = compile_substituted_ring(source, request)
        except ValueError:
            continue
        endpoint = canonical_state_key(product_graph)
        if endpoint in seen:
            continue
        seen.add(endpoint)
        rank = (
            int(not (set(anchors) & preferred_anchors)),
            abs(product_graph.n_real_atoms - source.n_real_atoms),
            len(request.substituents),
            endpoint,
        )
        candidates.append(BoundModuleCandidate(product_graph, stage, rank))
    candidates.sort(key=lambda candidate: candidate.structural_rank)
    return tuple(candidates[:max_candidates])


def _choose_bound_candidate(candidates, rng, *, family: str):
    if not candidates:
        raise ValueError(f"{family} has no exact context-compatible binding")
    ranks = np.arange(len(candidates), dtype=float)
    contextual = np.exp(-ranks / max(1.0, len(candidates) / 4))
    weights = CONTEXT_EXPLORATION / len(candidates) + (
        (1 - CONTEXT_EXPLORATION) * contextual / contextual.sum()
    )
    selected = int(rng.choice(len(candidates), p=weights))
    candidate = candidates[selected]
    stage = {
        **candidate.stage,
        "parameters": {
            **candidate.stage["parameters"],
            "binding_panel": {
                "compatible_candidates": len(candidates),
                "selected_rank": selected,
                "selected_probability": float(weights[selected]),
                "exploration_floor": CONTEXT_EXPLORATION,
                "task_oracle_information": False,
            },
        },
    }
    return candidate.product, stage


def candidate_program(
    source: MolecularGraph,
    candidate: BoundModuleCandidate,
    *,
    max_primitives: int = 32,
    max_blocks: int = 8,
):
    """Promote one enumerated structural option only after exact replay."""
    program, assignment = extract_program(source, [candidate.stage])
    product_graph, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product_graph) != canonical_state_key(candidate.product):
        raise RuntimeError("enumerated Dynamic-v1 candidate changed on exact replay")
    return source, program, assignment, trace


def two_nitrogen_saturated_ring_specs() -> tuple[RingSpec, ...]:
    """Shared parameter menu extension; no target or route frequencies."""
    return tuple(
        RingSpec(topology, size, (size - 2, 2, 0), "saturated")
        for topology in ("pendant", "fused")
        for size in (5, 6)
    )


def _weighted_v1_family_order(rng, *, near_capacity: bool):
    remaining, ordered = list(GENERIC_MODULES), []
    while remaining:
        weights = np.asarray(
            [
                (
                    NEAR_CAPACITY_MODULE_WEIGHTS.get(family, 2.0)
                    if near_capacity
                    else 1.0
                )
                for family in remaining
            ],
            dtype=float,
        )
        index = int(rng.choice(len(remaining), p=weights / weights.sum()))
        ordered.append(remaining.pop(index))
    return ordered


def compile_generic_module_v1(
    source: MolecularGraph,
    rng,
    family: str,
    *,
    preferred_anchors: frozenset[int] = frozenset(),
    panel_cache: dict | None = None,
):
    """Bind one v1 family through a bounded executor-verified context panel."""
    if family not in GENERIC_MODULES:
        raise ValueError(f"unknown Dynamic-v1 generic module: {family}")
    state_id = identity(encode_state(source))
    context = tuple(sorted(preferred_anchors))

    def cached(key, prepare):
        if panel_cache is None:
            return prepare()
        if key not in panel_cache:
            if len(panel_cache) >= MAX_PANEL_CACHE_ENTRIES:
                del panel_cache[next(iter(panel_cache))]
            panel_cache[key] = prepare()
        return panel_cache[key]

    if family == "substituent_delete":
        return _choose_bound_candidate(
            cached(
                (family, state_id, context),
                lambda: enumerate_pendant_deletions(
                    source,
                    preferred_anchors=preferred_anchors,
                    max_candidates=MAX_CONTEXT_CANDIDATES,
                ),
            ),
            rng,
            family=family,
        )
    if family == "functionalize":
        return _choose_bound_candidate(
            cached(
                (family, state_id, context),
                lambda: enumerate_functionalizations(
                    source,
                    preferred_anchors=preferred_anchors,
                    max_candidates=MAX_CONTEXT_CANDIDATES,
                ),
            ),
            rng,
            family=family,
        )
    if family == "ring_path_remodel":
        return _choose_bound_candidate(
            cached(
                (family, state_id, context),
                lambda: enumerate_ring_path_remodels(
                    source,
                    preferred_slots=preferred_anchors,
                    max_candidates=MAX_CONTEXT_CANDIDATES,
                ),
            ),
            rng,
            family=family,
        )
    if family == "construct_substituted_ring":
        panel_id = int(rng.integers(4))
        key = (family, state_id, context, panel_id)

        def ring_panel():
            seed = int(identity(key)[:16], 16)
            return enumerate_substituted_rings(
                source,
                np.random.default_rng(seed),
                preferred_anchors=preferred_anchors,
                max_candidates=MAX_CONTEXT_CANDIDATES,
                max_trials=MAX_CONTEXT_TRIALS,
            )

        return _choose_bound_candidate(
            cached(key, ring_panel),
            rng,
            family=family,
        )
    return compile_generic_module(source, rng, family)


def _following_context(product_graph: MolecularGraph, stage: dict) -> frozenset[int]:
    parameters = stage["parameters"]
    roots = set()
    if "retained_anchor" in parameters:
        roots.add(parameters["retained_anchor"])
    roots.update(parameters.get("retained_boundaries", ()))
    roots.update(parameters.get("ring_slots", ()))
    roots.update(parameters.get("branch_slots", ()))
    if "anchor" in parameters:
        roots.add(parameters["anchor"])
    return _nearby_anchors(product_graph, roots, radius=2) if roots else frozenset()


def synthesize_dynamic_program_v1(
    source: MolecularGraph,
    rng,
    *,
    max_modules: int = 3,
    max_primitives: int = 32,
    max_blocks: int = 8,
    panel_cache: dict | None = None,
):
    """Construct one protected v1 program with bounded contextual rebinding."""
    if not 1 <= max_modules <= 3:
        raise ValueError("Dynamic-v1 supports one to three generic modules")
    if rng.random() >= V1_CAPABILITY_PROBABILITY:
        prior = synthesize_dynamic_program(
            source,
            rng,
            max_modules=max_modules,
            max_primitives=max_primitives,
            max_blocks=max_blocks,
        )
        prior[4]["v1_selection"] = "preserved_dynamic_v0_channel"
        return prior
    near_capacity = source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
    patterned = bool(rng.random() < V1_PATTERN_PROBABILITY)
    if patterned:
        patterns = [row for row in V1_COMPOSITION_PATTERNS if len(row) <= max_modules]
        families = patterns[int(rng.integers(len(patterns)))]
        requested = len(families)
    else:
        probabilities = (
            NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
            if near_capacity
            else MODULE_COUNT_PROBABILITIES
        )
        requested = int(
            rng.choice(
                np.arange(1, max_modules + 1),
                p=np.asarray(probabilities[:max_modules])
                / sum(probabilities[:max_modules]),
            )
        )
        families = None

    current, stages, selected, failures = source, [], [], Counter()
    preferred = frozenset()
    for module_index in range(requested):
        order = (
            (families[module_index],)
            if families is not None
            else _weighted_v1_family_order(rng, near_capacity=near_capacity)
        )
        accepted = None
        for family in order:
            try:
                product_graph, stage = compile_generic_module_v1(
                    current,
                    rng,
                    family,
                    preferred_anchors=preferred,
                    panel_cache=panel_cache,
                )
                program, assignment = extract_program(source, [*stages, stage])
            except ValueError as error:
                failures[f"{family}:{error!s}"] += 1
                continue
            if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
                failures[f"{family}:work_limit"] += 1
                continue
            accepted = family, product_graph, stage, program, assignment
            break
        if accepted is None:
            if families is not None or not stages:
                raise ValueError(f"no v1 module executed: {dict(failures)}")
            break
        family, current, stage, program, assignment = accepted
        stages.append(stage)
        selected.append(
            {
                "index": module_index,
                "family": family,
                "parameters": stage["parameters"],
                "primitive_edits": len(stage["actions"]),
                "intermediate_endpoint": stage["endpoint"],
            }
        )
        preferred = _following_context(current, stage)

    program, assignment = extract_program(source, stages)
    product_graph, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product_graph) != canonical_state_key(current):
        raise RuntimeError("Dynamic-v1 composition changed on exact replay")
    metadata = {
        "schema_version": SCHEMA,
        "initial_stored_complete_routes": 0,
        "source_library_rows_loaded": 0,
        "requested_module_count": requested,
        "completed_module_count": len(stages),
        "selection": "generic_pattern" if patterned else "free_generic_composition",
        "selected_pattern": None if families is None else list(families),
        "capacity_aware": near_capacity,
        "source_heavy_atoms": source.n_real_atoms,
        "modules": selected,
        "intermediate_task_evaluations": 0,
        "module_failure_counts": dict(failures),
    }
    return source, program, assignment, trace, metadata


def initial_dynamic_program_batch_v1(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    broad_sampler=None,
):
    """Cold-start a v1 unit from an empty route archive and no target route."""
    if entries:
        raise ValueError("Dynamic-v1 initialization requires an empty route archive")
    if broad_sampler is not None:
        raise ValueError("Dynamic-v1 initialization does not use reference inference")
    began = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 73]))
    attempts, candidates = [], []
    seen = {canonical_state_key(source)}
    panel_cache = {}
    attempt_limit = (
        CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS + CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
        if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
        else config.attempts_per_batch
    )
    for index in range(attempt_limit):
        if (
            len(candidates) >= config.candidates_per_batch
            or perf_counter() - began >= config.wall_seconds
        ):
            break
        try:
            if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD:
                families = (
                    ("substituent_delete",)
                    if index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
                    else ("substituent_delete", "substituent_delete")
                )
                _, program, binding, trace, metadata = synthesize_named_module_sequence(
                    source,
                    rng,
                    families,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
                metadata["v1_selection"] = "preserved_dynamic_v0_capacity_bootstrap"
            else:
                _, program, binding, trace, metadata = synthesize_dynamic_program_v1(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                    panel_cache=panel_cache,
                )
        except ValueError as error:
            attempts.append(
                {"attempt": index, "status": "execution_rejected", "reason": str(error)}
            )
            continue
        endpoint = trace["endpoint"]
        properties = eligibility({"smiles": endpoint})
        status = (
            "duplicate"
            if endpoint in seen
            else (
                "eligible"
                if properties.get("oracle_eligible") is True
                else "ineligible"
            )
        )
        attempt = {
            "attempt": index,
            "channel": "dynamic_v1_generic_composition",
            "endpoint": endpoint,
            "status": status,
            "properties": properties,
            "metadata": metadata,
            "actual_changes": trace["actual_changes"],
        }
        attempts.append(attempt)
        if status != "eligible":
            continue
        seen.add(endpoint)
        candidate = {
            "source_group": source_group,
            "oracle_protocol": oracle_protocol,
            "source_state": encode_state(source),
            "program": program.payload(),
            "assignment": list(binding),
            "trace": trace,
            "endpoint": endpoint,
            "provenance": attempt,
            "score": None,
        }
        candidates.append({**candidate, "candidate_id": identity(candidate)})
    body = {
        "schema_version": "initial_dynamic_generic_program_batch_v2",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "candidates": candidates,
        "attempts": attempts,
        "rng_state_after_preparation": rng.bit_generator.state,
    }
    return {
        **body,
        "batch_id": identity(body),
        "proposal_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
    }


class DynamicV1ProgramOptimizer(DynamicProgramOptimizer):
    """Reuse scored self-discovered routes while proposing fresh v1 programs."""

    def _mutate(self, entry):
        if self.rng.random() >= FRESH_SYNTHESIS_PROBABILITY:
            return ProgramOptimizer._mutate(self, entry)
        source = decode_state(entry["trace"]["states"][-1])
        if not hasattr(self, "_dynamic_v1_panel_cache"):
            self._dynamic_v1_panel_cache = {}
        _, program, binding, _, metadata = synthesize_dynamic_program_v1(
            source,
            self.rng,
            max_modules=3,
            max_primitives=self.config.max_primitives,
            max_blocks=self.config.max_blocks,
            panel_cache=self._dynamic_v1_panel_cache,
        )
        return (
            source,
            program,
            binding,
            {
                "dynamic_v1_generic_composition": metadata,
                **self._continuation_lineage(entry),
            },
        )
