"""Progressive generic protected programs; no teacher routes or endpoint inputs.

This is a proposal-support repair, not a learned prior. Every ring panel consumes
the caller's persistent RNG instead of selecting from four forever-fixed panels.
Compilation and primitive legality are the unchanged v1/exact implementations.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.rewrite.kernel import canonical_state_key


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
    raise ValueError(f"unknown progressive proposal lane: {lane}")
