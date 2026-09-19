"""Explicit input-atom editing channel through existing production enumerators.

This optional optimization proposal neither defines reference probabilities nor
replaces the broad channel. All candidates pass the unchanged semantic executor.
"""

from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import extract_program
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.factorized_fiber import enumerate_pendant_graft_actions
from compose_v4.rewrite.operators import (
    enumerate_cycle_close_edges,
    enumerate_cycle_open_edges,
    enumerate_semantic_atom_restates,
)
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions

ENUMERATORS = {
    "atom_restate_semantic": enumerate_semantic_atom_restates,
    "bond_reroute": enumerate_pendant_graft_actions,
    "cycle_close": enumerate_cycle_close_edges,
    "cycle_open": enumerate_cycle_open_edges,
    "ring_system_restate": enumerate_ring_system_restate_actions,
}


def current_state_program(source, rng, *, family=None, work_cache=None):
    """One sampled legal operation; absent/invalid choices are explicit failures.

    Select the family before enumerating, so a large restatement frontier does
    not suppress topology choices. No task score is evaluated inside this path.
    """
    start = perf_counter()
    families = tuple(ENUMERATORS)
    family = families[int(rng.integers(len(families)))] if family is None else family
    if family not in ENUMERATORS:
        raise ValueError(f"unknown explicit current-state edit family: {family}")
    actions = (
        ENUMERATORS[family](source)
        if work_cache is None
        else work_cache.get(
            "current_state_actions",
            (identity(encode_state(source)), family),
            lambda: ENUMERATORS[family](source),
        )
    )
    if not actions:
        raise ValueError(f"current-state family has no applicable action: {family}")
    action = actions[int(rng.integers(len(actions)))]
    _, receipt = execute_program(source, [encode_action(family, action)])
    program, binding = extract_program(source, [{"name": f"current:{family}", **receipt}])
    return (
        program,
        binding,
        {
            "family": family,
            "enumerated_actions": len(actions),
            "enumeration_and_execution_seconds": perf_counter() - start,
            "conditional_action_probability": 1 / len(actions),
            "source": "saved exact current state, not reconstruction from SMILES",
        },
    )
