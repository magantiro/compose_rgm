"""Route-free synthesis of bounded executable molecular edit programs.

The sampler owns no stored molecule, endpoint, or complete program.  It binds
generic parameterized modules to the current exact graph, executes the whole
composition privately, and exposes only the completed endpoint to a task
evaluator.  Successfully measured programs may subsequently enter the ordinary
online archive; every independent run starts with that archive empty.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from time import perf_counter

import numpy as np

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
)
from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.current_state_edits import current_state_program
from compose_v4.control.docking_value import identity
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
from compose_v4.rewrite.operators import AtomDelete, AtomInsert, CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "dynamic_generic_program_synthesis_v1"
GENERIC_MODULES = (
    "segment_grow",
    "segment_shrink",
    "segment_replace",
    "substituent_delete",
    "append_ring",
    "fuse_ring",
    "functionalize",
    "carbonyl_insert",
    "heteroatom_substitute",
    "bond_reroute",
    "cycle_open",
    "cycle_close",
    "ring_system_restate",
)
MODULE_COUNT_PROBABILITIES = (0.4, 0.4, 0.2)
NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES = (0.15, 0.6, 0.25)
# The three-module ceiling was a v0 design choice, not a property of the compiler. It
# binds hard: payoff from a benchmark root sits at 15-22 primitives and Dynamic proposes
# a median of 5.5, so the ceiling is one candidate explanation for why autonomous runs
# never reach the productive band. Counts beyond the declared table decay geometrically
# rather than being uniform, so raising the cap widens the tail without inverting the
# preference for small coordinated programs.
MAX_GENERIC_MODULES = 8
EXTENDED_MODULE_DECAY = 0.5


def module_count_distribution(table, max_modules: int) -> np.ndarray:
    """Probabilities over 1..max_modules, extending the declared table if needed."""
    if max_modules <= len(table):
        weights = np.asarray(table[:max_modules], dtype=float)
    else:
        tail = [table[-1] * EXTENDED_MODULE_DECAY ** (i + 1)
                for i in range(max_modules - len(table))]
        weights = np.asarray([*table, *tail], dtype=float)
    return weights / weights.sum()
FRESH_SYNTHESIS_PROBABILITY = 0.5
ONLINE_COMPOSITION_PROBABILITY = 0.25
MAX_SEGMENT_LENGTH = 8
CAPACITY_AWARE_THRESHOLD = 36
CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS = 16
CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS = 32
NEAR_CAPACITY_MODULE_WEIGHTS = {
    "segment_grow": 0.5,
    "segment_shrink": 2.0,
    "segment_replace": 2.0,
    "substituent_delete": 6.0,
    "append_ring": 0.5,
    "fuse_ring": 0.5,
    "functionalize": 0.5,
    "carbonyl_insert": 1.0,
    "heteroatom_substitute": 2.0,
    "bond_reroute": 1.0,
    "cycle_open": 1.0,
    "cycle_close": 0.5,
    "ring_system_restate": 1.0,
}


def _stage(name: str, receipt: dict, parameters: dict) -> dict:
    return {"name": name, "parameters": parameters, **receipt}


def _execute_actions(source: MolecularGraph, name: str, actions, parameters):
    product, receipt = execute_program(source, list(actions))
    return product, _stage(name, receipt, parameters)


def _grow_actions(source, rng, *, length, elements, anchor=None):
    current, actions = source, []
    anchors = [
        int(i)
        for i in np.flatnonzero(is_element(current.atom_types))
        if int(current.implicit_h_counts[i]) >= 1
    ]
    if anchor is not None:
        anchors = [anchor] if anchor in anchors else []
    if not anchors:
        raise ValueError("segment growth has no hydrogen-bearing anchor")
    at = anchors[int(rng.integers(len(anchors)))]
    chosen = []
    for _ in range(length):
        element = elements[int(rng.integers(len(elements)))]
        valences = ALLOWED_VALENCES[element]
        if len(valences) != 1:
            raise ValueError("dynamic growth requires an unambiguous neutral valence")
        action = AtomInsert(
            fresh_slot(current),
            ELEMENT_TO_IDX[element],
            0,
            valences[0] - 1,
            ((at, 1),),
        )
        record = encode_action("atom_insert", action)
        current, _ = execute_program(current, [record])
        actions.append(record)
        at = action.slot
        chosen.append(element)
    return actions, at, chosen


def _terminal_shrink(source, rng, *, requested_length):
    current, actions, path, anchor = source, [], [], None
    preferred = None
    for _ in range(requested_length):
        real = [int(i) for i in np.flatnonzero(is_element(current.atom_types))]
        terminal = [i for i in real if int(np.count_nonzero(current.bonds[i])) == 1]
        if preferred in terminal:
            candidates = [preferred]
        else:
            candidates = terminal
        if not candidates or len(real) <= 1:
            break
        slot = candidates[int(rng.integers(len(candidates)))]
        neighbors = [int(i) for i in np.flatnonzero(current.bonds[slot])]
        next_slot = neighbors[0]
        record = encode_action("atom_delete", AtomDelete(slot))
        try:
            following, _ = execute_program(current, [record])
        except ValueError:
            preferred = None
            continue
        actions.append(record)
        path.append(slot)
        current = following
        anchor = next_slot
        preferred = next_slot
    if not actions:
        raise ValueError("segment shrink found no legal terminal deletion")
    return actions, current, anchor, path


def _component_without_edge(graph, start, cut):
    seen, stack = {start}, [start]
    while stack:
        at = stack.pop()
        for neighbor in np.flatnonzero(graph.bonds[at]):
            neighbor = int(neighbor)
            if frozenset((at, neighbor)) == cut or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


def _pendant_fragments(graph, *, maximum):
    real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
    fragments = []
    for a in sorted(real):
        for b in sorted(real):
            if a >= b or not graph.bonds[a, b]:
                continue
            cut = frozenset((a, b))
            left = _component_without_edge(graph, a, cut)
            if b in left:
                continue
            right = real - left
            for fragment, anchor in ((left, b), (right, a)):
                if 1 <= len(fragment) <= maximum and len(fragment) < len(real):
                    fragments.append((tuple(sorted(fragment)), anchor))
    return tuple(sorted(set(fragments), key=lambda row: (len(row[0]), row)))


def _delete_pendant_fragment(source, rng, *, maximum=MAX_SEGMENT_LENGTH, law=None):
    """Excise one bridge-separated substituent, chosen by the region-draw law.

    ``law is None`` is the historical behaviour and is preserved exactly: uniform
    over fragments of at most ``maximum`` atoms, ordered by the same single
    ``rng.permutation`` call, so every existing caller stays byte-identical.
    Passing a :class:`~compose_v4.control.bridge_region_law.BridgeRegionLaw`
    replaces only WHICH region is tried first; the deletion schedule, the
    executor calls and the acceptance rule below are untouched.
    """

    if law is None:
        choices = _pendant_fragments(source, maximum=maximum)
        if not choices:
            raise ValueError("no bounded pendant fragment is connected by a bridge")
        ordered = [choices[int(raw)] for raw in rng.permutation(len(choices))]
    else:
        regions = law.order(source, rng)
        if not regions:
            raise ValueError("no bounded pendant fragment is connected by a bridge")
        ordered = [(region.fragment, int(region.anchor)) for region in regions]
    failures = []
    for fragment, anchor in ordered:
        current, actions, remaining = source, [], set(fragment)
        try:
            while True:
                internal_edges = [
                    CycleOpenEdge(a, b)
                    for a in sorted(remaining)
                    for b in sorted(remaining)
                    if a < b and current.bonds[a, b]
                ]
                opened = False
                for raw_edge in rng.permutation(len(internal_edges)):
                    record = encode_action("cycle_open", internal_edges[int(raw_edge)])
                    try:
                        following, _ = execute_program(current, [record])
                    except ValueError:
                        continue
                    current = following
                    actions.append(record)
                    opened = True
                    break
                if not opened:
                    break
            while remaining:
                leaves = [
                    slot
                    for slot in sorted(remaining)
                    if int(np.count_nonzero(current.bonds[slot])) == 1
                ]
                if not leaves:
                    raise ValueError(
                        "pendant fragment does not admit leaf deletion order"
                    )
                slot = leaves[int(rng.integers(len(leaves)))]
                record = encode_action("atom_delete", AtomDelete(slot))
                current, _ = execute_program(current, [record])
                actions.append(record)
                remaining.remove(slot)
        except ValueError as error:
            failures.append(str(error))
            continue
        return actions, current, anchor, list(fragment)
    raise ValueError(f"no pendant fragment deleted exactly: {dict(Counter(failures))}")


def _ring_module(source, rng, *, topology, label, locus=None):
    """Build one ring of ``topology`` on ``source``.

    ``locus`` restricts which atoms may carry the new ring.  ``None`` means every
    real atom, which is the historical behaviour and keeps every existing caller
    byte-identical.  Passing a narrower set is what lets a region replacement
    rebuild AT the anchor it just excised from, reusing this builder rather than
    reimplementing ring chemistry beside it.
    """
    specs = [
        RingSpec(*_ring_option_parts(option))
        for option in default_ring_options()
        if f"construct:{topology}:" in option
    ]
    allowed = (
        frozenset(int(i) for i in np.flatnonzero(is_element(source.atom_types)))
        if locus is None
        else frozenset(int(i) for i in locus)
    )
    for raw_spec in rng.permutation(len(specs)):
        spec = specs[int(raw_spec)]
        branches = construction_branches(source, allowed, spec)
        if not branches:
            continue
        for raw_branch in rng.permutation(len(branches)):
            anchors, pattern = branches[int(raw_branch)]
            counts = residual(spec, source, anchors, ())
            if any(value < 0 for value in counts) or sum(counts) != spec.growth:
                continue
            pool = []
            for atom_type, count in zip(
                (ELEMENT_TO_IDX["C"], ELEMENT_TO_IDX["N"], ELEMENT_TO_IDX["O"]),
                counts,
                strict=True,
            ):
                pool.extend([IDX_TO_ELEMENT[atom_type]] * count)
            element_orders = sorted(set(permutations(pool)))
            for raw_elements in rng.permutation(len(element_orders))[:32]:
                elements = element_orders[int(raw_elements)]
                request = RingRequest(
                    spec, tuple(anchors), tuple(elements), tuple(pattern)
                )
                try:
                    product, receipt = compile_ring(source, request)
                except ValueError:
                    continue
                return product, _stage(
                    label,
                    receipt,
                    {
                        "topology": topology,
                        "size": spec.size,
                        "counts": list(spec.counts),
                        "electronic": spec.electronic,
                        "anchors": list(anchors),
                        "elements": list(elements),
                        "pattern": list(pattern),
                    },
                )
    raise ValueError(f"no exact-executable generic {topology} ring construction")


def _ring_option_parts(option):
    _, topology, size, counts, electronic, refine = option.split(":")
    return (
        topology,
        int(size),
        tuple(map(int, counts.split(","))),
        electronic,
        int(refine),
    )


def _weighted_module_order(rng, *, near_capacity):
    remaining, order = list(GENERIC_MODULES), []
    while remaining:
        weights = np.asarray(
            [
                NEAR_CAPACITY_MODULE_WEIGHTS[family] if near_capacity else 1.0
                for family in remaining
            ],
            dtype=float,
        )
        at = int(rng.choice(len(remaining), p=weights / weights.sum()))
        order.append(remaining.pop(at))
    return order


def _local_module(source, rng, *, family, label):
    program, binding, detail = current_state_program(source, rng, family=family)
    detail = {
        key: value
        for key, value in detail.items()
        if key != "enumeration_and_execution_seconds"
    }
    product, receipt = execute_program_graph(
        source,
        compile_program_graph(program),
        binding,
        max_primitives=32,
        max_blocks=8,
    )
    return product, _stage(
        label,
        {
            "actions": receipt["actions"],
            "states": receipt["states"],
            "endpoint": receipt["endpoint"],
            "primitive_edits": len(receipt["actions"]),
        },
        detail,
    )


def compile_generic_module(
    source: MolecularGraph, rng, family: str, *, region_law=None
):
    """Bind one generic module to the supplied exact state and execute it.

    ``region_law`` is threaded to the two modules that excise a bridge-separated
    substituent (``substituent_delete`` and the delete half of
    ``segment_replace``).  ``None`` keeps v1's uniform bounded law.
    """
    if family not in GENERIC_MODULES:
        raise ValueError(f"unknown dynamic generic module: {family}")
    if family == "segment_grow":
        capacity = min(MAX_SEGMENT_LENGTH, 40 - source.n_real_atoms)
        if capacity < 1:
            raise ValueError("segment growth has no remaining heavy-atom capacity")
        length = int(rng.integers(1, capacity + 1))
        actions, _, elements = _grow_actions(
            source, rng, length=length, elements=("C", "N", "O")
        )
        return _execute_actions(
            source, family, actions, {"length": length, "elements": elements}
        )
    if family == "functionalize":
        actions, _, elements = _grow_actions(
            source, rng, length=1, elements=("C", "N", "O", "F")
        )
        return _execute_actions(source, family, actions, {"element": elements[0]})
    if family == "segment_shrink":
        requested = int(rng.integers(1, MAX_SEGMENT_LENGTH + 1))
        actions, _, _, path = _terminal_shrink(source, rng, requested_length=requested)
        return _execute_actions(
            source,
            family,
            actions,
            {
                "requested_length": requested,
                "actual_length": len(actions),
                "path": path,
            },
        )
    if family == "segment_replace":
        delete_actions, contracted, anchor, path = _delete_pendant_fragment(
            source, rng, law=region_law
        )
        # A replacement may be as large as what it replaced.  v1 re-capped the
        # regrowth at MAX_SEGMENT_LENGTH regardless of how much was excised, so an
        # uncapped region draw was undone on the way back up: delete twenty atoms,
        # refill eight, and the endpoint carries only eight new ones.  Measured, the
        # realized largest changed region sat at the cap whatever the law offered.
        # With a law supplied the ceiling becomes the excision itself; with no law
        # the expression is byte-identical to v1, so every existing caller is
        # unaffected.
        removed = source.n_real_atoms - contracted.n_real_atoms
        ceiling = MAX_SEGMENT_LENGTH if region_law is None else max(removed, MAX_SEGMENT_LENGTH)
        capacity = min(ceiling, 40 - contracted.n_real_atoms)
        if capacity < 1:
            raise ValueError("segment replacement has no insertion capacity")
        growth = int(rng.integers(1, capacity + 1))
        grow_actions, _, elements = _grow_actions(
            contracted,
            rng,
            length=growth,
            elements=("C", "N", "O"),
            anchor=anchor,
        )
        return _execute_actions(
            source,
            family,
            [*delete_actions, *grow_actions],
            {
                "deleted_path": path,
                "deleted_atoms": len(path),
                "cycle_openings": len(delete_actions) - len(path),
                "inserted_atoms": growth,
                "elements": elements,
                "retained_anchor": anchor,
            },
        )
    if family == "substituent_delete":
        actions, _, anchor, path = _delete_pendant_fragment(
            source, rng, law=region_law
        )
        return _execute_actions(
            source,
            family,
            actions,
            {
                "fragment_slots": path,
                "deleted_atoms": len(path),
                "retained_anchor": anchor,
                "cycle_openings": len(actions) - len(path),
            },
        )
    if family == "append_ring":
        return _ring_module(source, rng, topology="pendant", label=family)
    if family == "fuse_ring":
        return _ring_module(source, rng, topology="fused", label=family)
    if family == "carbonyl_insert":
        anchors = [
            int(i)
            for i in np.flatnonzero(is_element(source.atom_types))
            if int(source.atom_types[i]) == ELEMENT_TO_IDX["C"]
            and int(source.implicit_h_counts[i]) >= 2
        ]
        if not anchors:
            raise ValueError("carbonyl insertion has no compatible carbon anchor")
        anchor = anchors[int(rng.integers(len(anchors)))]
        action = AtomInsert(
            fresh_slot(source), ELEMENT_TO_IDX["O"], 0, 0, ((anchor, 2),)
        )
        return _execute_actions(
            source,
            family,
            [encode_action("atom_insert", action)],
            {"anchor": anchor},
        )
    families = {
        "heteroatom_substitute": "atom_restate_semantic",
        "bond_reroute": "bond_reroute",
        "cycle_open": "cycle_open",
        "cycle_close": "cycle_close",
        "ring_system_restate": "ring_system_restate",
    }
    return _local_module(source, rng, family=families[family], label=family)


def synthesize_dynamic_program(
    source: MolecularGraph,
    rng,
    *,
    max_modules: int = 3,
    max_primitives: int = 32,
    max_blocks: int = 8,
    region_law=None,
):
    """Construct one complete K-module program without any task evaluation."""
    if not 1 <= max_modules <= MAX_GENERIC_MODULES:
        raise ValueError(f"generic module count must be 1 to {MAX_GENERIC_MODULES}")
    near_capacity = source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
    count_probabilities = (
        NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
        if near_capacity
        else MODULE_COUNT_PROBABILITIES
    )
    count = int(
        rng.choice(
            np.arange(1, max_modules + 1),
            p=module_count_distribution(count_probabilities, max_modules),
        )
    )
    current, stages, selected, failures = source, [], [], Counter()
    for module_index in range(count):
        accepted = None
        for family in _weighted_module_order(rng, near_capacity=near_capacity):
            try:
                product, stage = compile_generic_module(
                    current, rng, family, region_law=region_law
                )
            except ValueError as error:
                failures[f"{family}:{error!s}"] += 1
                continue
            candidate_stages = [*stages, stage]
            try:
                program, assignment = extract_program(source, candidate_stages)
            except ValueError as error:
                failures[f"{family}:extract:{error!s}"] += 1
                continue
            if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
                failures[f"{family}:work_limit"] += 1
                continue
            accepted = product, stage, program, assignment
            break
        if accepted is None:
            if not stages:
                raise ValueError(f"no generic module executed: {dict(failures)}")
            break
        current, stage, program, assignment = accepted
        stages.append(stage)
        selected.append(
            {
                "index": module_index,
                "family": stage["name"],
                "parameters": stage["parameters"],
                "primitive_edits": len(stage["actions"]),
                "intermediate_endpoint": stage["endpoint"],
            }
        )
    program, assignment = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("dynamic composition replay changed its exact endpoint")
    metadata = {
        "schema_version": SCHEMA,
        "initial_stored_complete_routes": 0,
        "source_library_rows_loaded": 0,
        "requested_module_count": count,
        "completed_module_count": len(stages),
        "capacity_aware": near_capacity,
        "source_heavy_atoms": source.n_real_atoms,
        "modules": selected,
        "intermediate_task_evaluations": 0,
        "module_failure_counts": dict(failures),
    }
    return source, program, assignment, trace, metadata


def synthesize_named_module_sequence(
    source: MolecularGraph,
    rng,
    families,
    *,
    max_primitives=32,
    max_blocks=8,
    region_law=None,
):
    """Compile a prospectively chosen generic module sequence on exact states."""
    if not 1 <= len(families) <= 3 or any(
        family not in GENERIC_MODULES for family in families
    ):
        raise ValueError("named dynamic sequence requires one to three generic modules")
    current, stages, selected = source, [], []
    for index, family in enumerate(families):
        current, stage = compile_generic_module(
            current, rng, family, region_law=region_law
        )
        stages.append(stage)
        selected.append(
            {
                "index": index,
                "family": family,
                "parameters": stage["parameters"],
                "primitive_edits": len(stage["actions"]),
                "intermediate_endpoint": stage["endpoint"],
            }
        )
    program, assignment = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("named dynamic sequence changed on exact replay")
    return (
        source,
        program,
        assignment,
        trace,
        {
            "schema_version": SCHEMA,
            "initial_stored_complete_routes": 0,
            "source_library_rows_loaded": 0,
            "requested_module_count": len(families),
            "completed_module_count": len(families),
            "capacity_aware": source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD,
            "source_heavy_atoms": source.n_real_atoms,
            "modules": selected,
            "intermediate_task_evaluations": 0,
            "module_failure_counts": {},
            "selection": (
                "capacity_bootstrap_pair"
                if tuple(families) == ("substituent_delete", "substituent_delete")
                else "declared_generic_sequence"
            ),
        },
    )


def initial_dynamic_program_batch(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    broad_sampler=None,
):
    """Cold-start a unit from an empty route archive via dynamic synthesis."""
    if entries:
        raise ValueError("dynamic-only initialization requires an empty route archive")
    if broad_sampler is not None:
        raise ValueError("dynamic-only initialization does not use reference inference")
    began = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 71]))
    attempts, candidates = [], []
    seen = {canonical_state_key(source)}
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
            if (
                source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
                and index
                < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS + CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
            ):
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
            else:
                _, program, binding, trace, metadata = synthesize_dynamic_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
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
            "channel": "dynamic_generic_composition",
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
        "schema_version": "initial_dynamic_generic_program_batch_v1",
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


class DynamicProgramOptimizer(ProgramOptimizer):
    """Reuse only measured self-discovered routes, mixed with fresh synthesis."""

    def _mutate(self, entry):
        if self.rng.random() >= FRESH_SYNTHESIS_PROBABILITY:
            return super()._mutate(entry)
        source = decode_state(entry["trace"]["states"][-1])
        _, program, binding, _, metadata = synthesize_dynamic_program(
            source,
            self.rng,
            max_modules=3,
            max_primitives=self.config.max_primitives,
            max_blocks=self.config.max_blocks,
        )
        return (
            source,
            program,
            binding,
            {
                "dynamic_generic_composition": metadata,
                **self._continuation_lineage(entry),
            },
        )
