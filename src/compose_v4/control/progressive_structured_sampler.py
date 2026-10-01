"""Progressive generic protected programs; no teacher routes or endpoint inputs.

This is a proposal-support repair, not a learned prior. Every ring panel consumes
the caller's persistent RNG instead of selecting from four forever-fixed panels.
Compilation and primitive legality are the unchanged v1/exact implementations.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import numpy as np

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, is_element
from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.rewrite.kernel import canonical_state_key

ANCHORED_REPLACEMENT_MAX_TRIALS = 32


def _is_carbonyl_anchor(graph, slot: int) -> bool:
    """Return whether ``slot`` is a carbon double bonded to an oxygen."""
    if int(graph.atom_types[slot]) != ELEMENT_TO_IDX["C"]:
        return False
    return any(
        int(graph.bonds[slot, neighbor]) == 2
        and int(graph.atom_types[neighbor]) == ELEMENT_TO_IDX["O"]
        for neighbor in np.flatnonzero(graph.bonds[slot])
    )


def _anchored_deletion_weights(source, candidates) -> np.ndarray:
    """Generic structural prior for a replaceable one-boundary substituent."""
    weights = []
    for candidate in candidates:
        parameters = candidate.stage["parameters"]
        anchor = int(parameters["retained_anchor"])
        fragment = tuple(map(int, parameters["fragment_slots"]))
        hetero = any(
            is_element(source.atom_types[slot])
            and int(source.atom_types[slot]) != ELEMENT_TO_IDX["C"]
            for slot in fragment
        )
        weight = 1.0
        weight *= 4.0 if _is_carbonyl_anchor(source, anchor) else 1.0
        weight *= 2.0 if hetero else 1.0
        weight /= max(1, len(fragment))
        weights.append(weight)
    normalized = np.asarray(weights, dtype=float)
    return normalized / normalized.sum()


def _sample_anchored_diamine_ring(source, rng, anchor: int):
    """Sample one bounded saturated two-N ring at a retained anchor.

    The family is a transferable structural option rather than a molecule
    template. Its C/N counts, size, atom order and branches are sampled afresh.
    """
    sizes = (5, 6)
    size = sizes[int(rng.integers(len(sizes)))]
    spec = v1.RingSpec("pendant", size, (size - 2, 2, 0), "saturated")
    branches = tuple(
        branch
        for branch in v1.construction_branches(source, {anchor}, spec)
        if branch[0] == (anchor,)
    )
    if not branches:
        raise ValueError("retained anchor cannot support a saturated two-N ring")
    anchors, pattern = branches[int(rng.integers(len(branches)))]
    orders = sorted(set(permutations(("C",) * (size - 2) + ("N", "N"))))
    elements = orders[int(rng.integers(len(orders)))]
    template = v1.GENERIC_SUBSTITUENT_TEMPLATES[
        int(rng.integers(len(v1.GENERIC_SUBSTITUENT_TEMPLATES)))
    ]
    request = v1.SubstitutedRingRequest(
        v1.RingRequest(spec, tuple(anchors), tuple(elements), tuple(pattern)),
        v1._substituent_atoms(template, size, rng),
    )
    return v1.compile_substituted_ring(source, request)


def synthesize_anchored_replacement_program(
    source,
    rng,
    *,
    max_primitives=32,
    max_blocks=8,
    max_trials=ANCHORED_REPLACEMENT_MAX_TRIALS,
):
    """Couple deletion and construction at one retained structural role.

    This changes proposal scheduling only. Both modules and every primitive are
    compiled and replayed by the existing Dynamic-v1/exact executor. Prefixes
    are protected and no benchmark property or oracle value is evaluated inside.
    """
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError("anchored-replacement trials must be a positive integer")
    deletions = v1.enumerate_pendant_deletions(source, max_candidates=v1.MAX_CONTEXT_CANDIDATES)
    if not deletions:
        raise ValueError("anchored replacement has no one-boundary deletion")
    probabilities = _anchored_deletion_weights(source, deletions)
    failures = Counter()
    for _ in range(max_trials):
        selected = int(rng.choice(len(deletions), p=probabilities))
        deletion = deletions[selected]
        deleted = deletion.product
        anchor = int(deletion.stage["parameters"]["retained_anchor"])
        try:
            product, ring_stage = _sample_anchored_diamine_ring(deleted, rng, anchor)
            stages = [deletion.stage, ring_stage]
            program, binding = extract_program(source, stages)
            if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
                failures["work_limit"] += 1
                continue
            replayed, trace = execute_program_graph(
                source,
                compile_program_graph(program),
                binding,
                max_primitives=max_primitives,
                max_blocks=max_blocks,
            )
        except (ValueError, RuntimeError) as error:
            failures[str(error)] += 1
            continue
        if canonical_state_key(replayed) != canonical_state_key(product):
            raise RuntimeError("anchored replacement changed during exact replay")
        return (
            source,
            program,
            binding,
            trace,
            {
                "schema_version": "anchored_replacement_program_v1",
                "modules": [
                    {
                        "family": "substituent_delete",
                        "parameters": deletion.stage["parameters"],
                        "primitive_edits": len(deletion.stage["actions"]),
                        "intermediate_endpoint": deletion.stage["endpoint"],
                    },
                    {
                        "family": "construct_substituted_ring",
                        "parameters": ring_stage["parameters"],
                        "primitive_edits": len(ring_stage["actions"]),
                        "intermediate_endpoint": ring_stage["endpoint"],
                    },
                ],
                "selection": "generic_retained_anchor_replacement",
                "selected_deletion_rank": selected,
                "selected_deletion_probability": float(probabilities[selected]),
                "completed_module_count": 2,
                "initial_stored_complete_routes": 0,
                "source_library_rows_loaded": 0,
                "intermediate_task_evaluations": 0,
                "task_oracle_information": False,
                "failed_trials_before_success": int(sum(failures.values())),
            },
        )
    raise ValueError(f"anchored replacement exhausted {max_trials} trials: {dict(failures)}")


def progressive_module(source, rng, family, *, preferred=frozenset()):
    """Sample a fresh bounded conditional panel without changing operator support."""
    if family == "construct_substituted_ring":
        panel = v1.enumerate_substituted_rings(
            source,
            rng,
            preferred_anchors=preferred,
            max_candidates=v1.MAX_CONTEXT_CANDIDATES,
            max_trials=v1.MAX_CONTEXT_TRIALS,
        )
        return v1._choose_bound_candidate(panel, rng, family=family)
    return v1.compile_generic_module_v1(
        source,
        rng,
        family,
        preferred_anchors=preferred,
        panel_cache=None,
    )


def synthesize_progressive_program(source, rng, *, max_primitives=32, max_blocks=8):
    """Compose 1–3 generic modules; evaluate no task criteria at internal prefixes."""
    near_capacity = source.n_real_atoms >= v1.CAPACITY_AWARE_THRESHOLD
    patterned = rng.random() < v1.V1_PATTERN_PROBABILITY
    if patterned:
        families = v1.V1_COMPOSITION_PATTERNS[int(rng.integers(len(v1.V1_COMPOSITION_PATTERNS)))]
        count = len(families)
    else:
        families = None
        probabilities = (
            v1.NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
            if near_capacity
            else v1.MODULE_COUNT_PROBABILITIES
        )
        count = int(rng.choice(np.arange(1, 4), p=probabilities))
    current, stages, modules, failures = source, [], [], Counter()
    preferred = frozenset()
    for index in range(count):
        order = (
            (families[index],)
            if families
            else v1._weighted_v1_family_order(rng, near_capacity=near_capacity)
        )
        accepted = None
        for family in order:
            try:
                product, stage = progressive_module(current, rng, family, preferred=preferred)
                program, binding = extract_program(source, [*stages, stage])
            except ValueError as error:
                failures[f"{family}:{error}"] += 1
                continue
            if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
                failures[f"{family}:work_limit"] += 1
                continue
            accepted = product, stage, family
            break
        if accepted is None:
            if families or not stages:
                raise ValueError(f"progressive program could not complete: {dict(failures)}")
            break
        current, stage, family = accepted
        stages.append(stage)
        modules.append(
            {
                "family": family,
                "parameters": stage["parameters"],
                "primitive_edits": len(stage["actions"]),
                "intermediate_endpoint": stage["endpoint"],
            }
        )
        preferred = v1._following_context(current, stage)
    program, binding = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        binding,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("progressive structured program changed during exact replay")
    return (
        source,
        program,
        binding,
        trace,
        {
            "schema_version": "progressive_structured_program_v1",
            "modules": modules,
            "selection": "generic_pattern" if patterned else "free_generic_composition",
            "selected_pattern": None if families is None else list(families),
            "requested_module_count": count,
            "completed_module_count": len(stages),
            "module_failure_counts": dict(failures),
            "initial_stored_complete_routes": 0,
            "source_library_rows_loaded": 0,
            "intermediate_task_evaluations": 0,
        },
    )


def synthesize_lane(source, rng, lane, config):
    if lane == "shallow":
        return synthesize_dynamic_program(
            source,
            rng,
            max_modules=3,
            max_primitives=config.max_primitives,
            max_blocks=config.max_blocks,
        )
    if lane == "structured":
        return synthesize_progressive_program(
            source, rng, max_primitives=config.max_primitives, max_blocks=config.max_blocks
        )
    if lane == "anchored_replacement":
        return synthesize_anchored_replacement_program(
            source, rng, max_primitives=config.max_primitives, max_blocks=config.max_blocks
        )
    raise ValueError(f"unknown progressive proposal lane: {lane}")
