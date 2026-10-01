"""Constraint binding for the actual three-lane T4 program proposer.

This wrapper imports the production constructors. It contains no copied chemical
constructor, reward gate, learned-head override or drug-specific branch. A
complete proposed program is rebound before exact execution. Locked core edits
are inadmissible even if an unconstrained T4 lane proposed them successfully.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    ProgramConstraint,
    admit_program_stages,
    candidate_stages,
    propose_region_completion,
    verify_prompt_endpoint,
)
from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.edit_program import attachment_bindings
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.progressive_structured_sampler import synthesize_lane
from compose_v4.rewrite.trace_shard import decode_state

T4_LANES = ("shallow", "structured", "anchored_replacement")


@dataclass(frozen=True)
class T4ProgramLimits:
    max_primitives: int = 32
    max_blocks: int = 8
    max_bindings: int = 32
    max_binding_visits: int = 4096

    def __post_init__(self):
        if not 1 <= self.max_primitives <= 32 or not 1 <= self.max_blocks <= 8:
            raise ValueError("fragment T4 programs must retain 32-primitive/eight-block support")
        if min(self.max_bindings, self.max_binding_visits) < 1:
            raise ValueError("binding work caps must be positive")


class ConstrainedProgramAbstention(ValueError):
    def __init__(self, reason: str, receipt: dict):
        super().__init__(reason)
        self.receipt = receipt


DEFAULT_LIMITS = T4ProgramLimits()


def bind_complete_program(source, program, constraint, rng, *, limits=DEFAULT_LIMITS):
    """Bind one complete production program inside the legal input interface.

    Shared constructor programs conservatively treat every explicit operand as
    a write. Accordingly an original-atom role may bind only to an interface or
    an unlocked atom, never an internal protected atom. Full pathwise chemistry
    checks remain necessary; the binding mask is not itself a validity proof.
    """
    real = frozenset(int(i) for i in np.flatnonzero(is_element(source.atom_types)))
    mutable = (real - frozenset(constraint.locked_slots)) | frozenset(constraint.interfaces)
    census = attachment_bindings(
        program,
        source,
        mutable_slots=mutable,
        max_bindings=limits.max_bindings,
        max_visits=limits.max_binding_visits,
    )
    graph = compile_program_graph(program)
    receipt = {
        "program_id": program.program_id,
        "program_blocks": len(program.blocks),
        "program_primitives": len(program.marks),
        "block_labels": [b.label for b in program.blocks],
        "dependencies": list(graph.dependencies),
        "conflicts": list(graph.conflicts),
        "serialization_edges": list(graph.serialization_edges),
        "binding_assignments": len(census.assignments),
        "binding_visits": census.visits,
        "binding_truncated": census.truncated,
        "attempted_bindings": 0,
    }
    failures = Counter()
    lock = constraint.lock(source)
    for index in rng.permutation(len(census.assignments)):
        assignment = census.assignments[int(index)]
        receipt["attempted_bindings"] += 1
        try:
            endpoint, trace = execute_program_graph(
                source,
                graph,
                assignment,
                max_primitives=limits.max_primitives,
                max_blocks=limits.max_blocks,
                mutable_slots=mutable,
            )
        except ValueError as error:
            failures[f"executor:{error}"] += 1
            continue
        if any(not lock.permits(decode_state(state)) for state in trace["states"]):
            failures["pathwise_fragment_lock"] += 1
            continue
        if not constraint.complete(endpoint):
            failures["required_interfaces_incomplete"] += 1
            continue
        receipt.update(
            selected_binding=list(assignment),
            binding_failures=dict(sorted(failures.items())),
        )
        ends = [b.stop for b in program.blocks]
        lengths = [end - start for start, end in zip([0, *ends[:-1]], ends, strict=True)]
        receipt["program_block_lengths"] = [lengths[i] for i in trace["block_order"]]
        receipt["program_block_labels"] = [program.blocks[i].label for i in trace["block_order"]]
        return CompleteProgram(endpoint, trace, receipt)
    receipt["binding_failures"] = dict(sorted(failures.items()))
    raise ConstrainedProgramAbstention("complete T4 program has no admissible binding", receipt)


def propose_t4_completion(context, rng, lane, *, limits=DEFAULT_LIMITS):
    """Invoke an unchanged T4 lane, constrain its complete program, then verify."""
    if lane not in T4_LANES:
        raise ValueError(f"unknown T4 proposal lane: {lane!r}")
    source = context.start_state
    try:
        _, program, _, _, metadata = synthesize_lane(source, rng, lane, limits)
    except ValueError as error:
        raise ConstrainedProgramAbstention(
            f"T4 constructor abstained: {error}", {"lane": lane, "constructor_completed": False}
        ) from error
    try:
        result = bind_complete_program(
            source, program, ProgramConstraint.from_context(context), rng, limits=limits
        )
    except ConstrainedProgramAbstention as error:
        raise ConstrainedProgramAbstention(
            str(error),
            {**error.receipt, "lane": lane, "constructor_completed": True, "metadata": metadata},
        ) from error
    verify_prompt_endpoint(context, result)
    return CompleteProgram(
        result.endpoint,
        result.trace,
        {
            **result.provenance,
            "lane": lane,
            "constructor_completed": True,
            "constructor": f"progressive_structured_sampler.synthesize_lane:{lane}",
            "metadata": metadata,
            "frozen_model_scored": False,
            "qed_sa_guidance": False,
        },
    )


def refine_with_t4(context, seed, rng, lane):
    """Refine editable generated content through a native complete T4 program.

    The seed is a privately compiled proposal, not an accepted output or an
    additional generation attempt. Its primitives count against the same full
    program budget, and no prefix is selected using QED/SA or a target score.
    """
    if lane not in T4_LANES:
        raise ValueError(f"unknown T4 proposal lane: {lane!r}")
    remaining = 32 - len(seed.trace["actions"])
    remaining_blocks = 8 - len(seed.provenance["program_block_lengths"])
    if remaining < 1 or remaining_blocks < 1:
        raise ConstrainedProgramAbstention(
            "seed leaves no primitive/block capacity for T4 refinement",
            {"lane": lane, "seed_primitives": len(seed.trace["actions"])},
        )
    limits = T4ProgramLimits(max_primitives=remaining, max_blocks=remaining_blocks)
    try:
        _, program, _, _, metadata = synthesize_lane(seed.endpoint, rng, lane, limits)
    except ValueError as error:
        raise ConstrainedProgramAbstention(
            f"T4 constructor abstained: {error}", {"lane": lane, "constructor_completed": False}
        ) from error
    constraint = ProgramConstraint.from_context(context)
    try:
        refinement = bind_complete_program(seed.endpoint, program, constraint, rng, limits=limits)
    except ConstrainedProgramAbstention as error:
        raise ConstrainedProgramAbstention(
            str(error), {**error.receipt, "lane": lane, "metadata": metadata}
        ) from error
    if refinement.smiles == seed.smiles:
        raise ConstrainedProgramAbstention(
            "T4 refinement is a net self-event", refinement.provenance
        )
    candidate = admit_program_stages(
        context.start_state,
        candidate_stages(seed) + candidate_stages(refinement),
        constraint,
        provenance={
            **seed.provenance,
            "lane": f"training_regions+t4_{lane}",
            "t4_lane": lane,
            "t4_metadata": metadata,
            "t4_binding_receipt": refinement.provenance,
            "seed_endpoint": seed.smiles,
            "seed_primitives": len(seed.trace["actions"]),
            "t4_primitives": len(refinement.trace["actions"]),
            "t4_changed_endpoint": True,
        },
    )
    verify_prompt_endpoint(context, candidate)
    return candidate


def propose_program_panel_member(context, entries, rng, draw):
    """Frozen four-lane allocation: region, then each of the three T4 lanes."""
    if type(draw) is not int or draw < 0:
        raise ValueError("candidate draw must be a nonnegative integer")
    seed = propose_region_completion(context, entries, rng)
    choice = draw % 4
    return seed if choice == 0 else refine_with_t4(context, seed, rng, T4_LANES[choice - 1])
