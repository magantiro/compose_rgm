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


def current_state_program(source, rng, *, family=None, work_cache=None, successor_prior=None):
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
    if successor_prior is None:
        # Historical uniform draw.  Absence of a prior is the only byte-identical
        # "off": a uniform prior object would reproduce this support but not these
        # draws, because it consumes a different number of RNG values.
        action = actions[int(rng.integers(len(actions)))]
        conditional_probability = 1 / len(actions)
    else:
        ordered = successor_prior.order(source, rng, family=family, actions=actions)
        action = ordered[0]
        conditional_probability = successor_prior.probability_of(
            source, family=family, actions=actions, action=action
        )
    _, receipt = execute_program(source, [encode_action(family, action)])
    program, binding = extract_program(source, [{"name": f"current:{family}", **receipt}])
    return (
        program,
        binding,
        {
            "family": family,
            "enumerated_actions": len(actions),
            "enumeration_and_execution_seconds": perf_counter() - start,
            "conditional_action_probability": conditional_probability,
            "source": "saved exact current state, not reconstruction from SMILES",
            # ABSENT means the uniform v1 draw. Tagging the uniform case too would put a
            # new key in every stage dict, which changes stage CONTENT and therefore the
            # program -- measured: it perturbed 1 of 120 seeded programs while every
            # RNG-consuming path was byte-identical. A prior-off arm must be byte-identical,
            # so the tag is only emitted when a prior actually ran.
            **({} if successor_prior is None
               else {"proposal_law": "learned editing model over legal actions, floored"}),
        },
    )
