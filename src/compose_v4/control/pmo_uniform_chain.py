"""The length-matched uniform legal-chain arm of the PMO proposal ablation.

WHAT THIS IS
------------
Arm B of `structured programs vs uniform legal local editing under a matched
planned editing allowance`.  A proposal is K successive primitive edits, each
drawn UNIFORMLY over the distinct canonical successors legal at the state the
previous edit produced.  The fiber is RECOMPUTED after every intermediate; a
chain is never produced by shuffling or replaying a compiled program.

THE UNIFORMITY DEFINITION, STATED ONCE AND FROZEN
-------------------------------------------------
Uniform over LEGAL NATIVE MARKS THAT EXECUTE -- not over distinct canonical
successors.  A mark is drawn by `pmo_legal_mark_sampler.LegalMarkSampler`, which
samples each rule's own tentative set in proportion to its size and rejects what
the production admission predicate refuses; the drawn mark is then executed and
rejected again if the executor refuses it.  Conditional on acceptance this is
exactly uniform over the executable legal marks.

MARK MULTIPLICITY IS DISCLOSED: several distinct marks can land on one molecule,
so a molecule reachable by many marks is drawn more often than one reachable by
few.  Measured on a 40-heavy-atom PMO parent: 965 candidate marks against 885
distinct canonical successors.

PRE-LAUNCH CHANGE, RECORDED.  This was first pinned as uniform over DISTINCT
CANONICAL SUCCESSORS.  That law requires deduplicating every successor, which
means applying and canonicalizing all ~1,000 candidates at every step: measured
17.55 s per step against 448 ms for the law used here.  The change was made
BEFORE any arm-B run began and on cost-and-naturalness grounds -- "pick a legal
edit at random" rather than "pick a legal destination at random" -- never after
seeing an outcome.

WHERE THE LENGTH COMES FROM
---------------------------
K is the realized primitive count of the program the UNCHANGED arm-A sampler
produced for this same (parent, channel) draw.  The structured program is then
discarded.  That matches the planned allowance per proposal using the existing
configuration, and it is outcome-independent: no score, endpoint or benchmark
answer is consulted.  Measured arm-A distribution: mean 6.8, median 4, p90 17,
max 32 primitives.

FAILURE IS DECLARED, NOT HIDDEN
-------------------------------
If a state offers no legal successor the chain stops there, keeps the steps it
executed, and records `halt_reason` with `planned_length` and `completed_length`.
There is NO restart-until-success and NO fallback to a structured program: a
chain that executes zero steps raises, exactly as an arm-A module that compiles
nothing raises, so the lane counts it as a rejection in the ordinary way.

This module reads no oracle, no target, no archive score and no learned model.
"""

from __future__ import annotations

import os
from time import perf_counter as _perf_counter

from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.pmo_legal_mark_sampler import LegalMarkSampler
from compose_v4.experiments.whole_ring_plan import execute_program

#: Environment switch.  ABSENT is the byte-identical arm A: no call site consults
#: this module unless the flag is set, so arm A's RNG consumption is untouched.
ENV_FLAG = "PMO_UNIFORM_CHAIN"

STAGE_NAME = "uniform_legal_chain"

#: Live counters for streaming diagnostics.  The campaign driver snapshots and resets
#: these each round, so a long run reports chain health as it goes rather than only in
#: a post-hoc artifact.  Counting here -- at the one place a chain is built -- means the
#: numbers cannot drift from what actually ran.
STATS = {
    "chains": 0,
    "planned_total": 0,
    "completed_total": 0,
    "terminated_early": 0,
    "sampler_attempts": 0,
    "sampler_rejections": 0,
    "seconds": 0.0,
    "halt_reasons": {},
    "raised_no_step": 0,
}


def snapshot_stats(reset: bool = True) -> dict:
    """Return a copy of the live chain counters, optionally zeroing them."""
    import copy

    out = copy.deepcopy(STATS)
    chains = max(1, out["chains"])
    attempts = max(1, out["sampler_attempts"])
    out["mean_planned"] = out["planned_total"] / chains
    out["mean_completed"] = out["completed_total"] / chains
    out["accept_rate"] = 1.0 - out["sampler_rejections"] / attempts
    out["seconds_per_chain"] = out["seconds"] / chains
    if reset:
        STATS.update(
            chains=0, planned_total=0, completed_total=0, terminated_early=0,
            sampler_attempts=0, sampler_rejections=0, seconds=0.0,
            halt_reasons={}, raised_no_step=0,
        )
    return out


def chain_arm_enabled() -> bool:
    """Whether this process runs arm B."""
    return bool(os.environ.get(ENV_FLAG))


def uniform_legal_chain(
    source,
    rng,
    *,
    length: int,
    max_primitives: int = 32,
    max_blocks: int = 8,
):
    """Draw `length` successive uniform legal primitive edits from `source`.

    Returns ``(source, program, assignment, trace, metadata)`` with the same shape
    the arm-A synthesizer returns, so every downstream consumer -- candidate
    record, `population_features`, the fitted program value, credit allocation,
    dedup and the ledger -- is reached unchanged.
    """
    if length < 1:
        raise ValueError("uniform legal chain needs a positive planned length")
    _began = _perf_counter()
    from compose_v4.experiments.pmo_legal_action_policy import (
        editing_v2_semantic_rewrite_system,
    )
    from compose_v4.rewrite.action_codec_v4 import encode_action
    from compose_v4.rewrite.kernel import InvalidRewrite

    system = editing_v2_semantic_rewrite_system()
    planned = int(min(length, max_primitives))
    current, actions, halt = source, [], None
    attempts = rejections = 0
    for _ in range(planned):
        sampler = LegalMarkSampler(current)
        executed = None
        # Draw uniformly, then execute.  A mark the EXECUTOR refuses is rejected and
        # redrawn, which keeps the law uniform over marks that actually execute --
        # the production enumerator applies the same try/except.
        for _try in range(64):
            drawn = sampler.draw(rng)
            if drawn is None:
                break
            rule, action = drawn
            try:
                successor = system.apply(current, rule, action)
            except (InvalidRewrite, ValueError):
                rejections += 1
                continue
            executed = (rule, action, successor)
            break
        attempts += sampler.attempts
        rejections += sampler.rejections
        if executed is None:
            halt = "no_executable_legal_mark"
            break
        rule, action, successor = executed
        actions.append(encode_action(rule, action))
        current = successor
    if not actions:
        STATS["raised_no_step"] += 1
        raise ValueError("uniform legal chain executed no step")

    # Replay through the production executor so the stage carries exactly the
    # state/endpoint encoding `extract_program` validates against, rather than a
    # transcription of it that could drift.
    _, receipt = execute_program(source, actions)
    stage = {
        "name": STAGE_NAME,
        "actions": receipt["actions"],
        "states": receipt["states"],
        "endpoint": receipt["endpoint"],
    }
    program, assignment = extract_program(source, [stage])
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    metadata = {
        "schema_version": "pmo_uniform_legal_chain_v1",
        "arm": "uniform_legal_chain",
        "uniformity": "executable_legal_native_marks",
        "sampler": "rejection over each rule's production tentative set",
        "planned_length": int(length),
        "requested_length_after_work_limit": planned,
        "completed_length": len(actions),
        "terminated_early": bool(halt is not None),
        "halt_reason": halt,
        "fiber_recomputed_every_step": True,
        "sampler_attempts": attempts,
        "sampler_rejections": rejections,
        "intermediate_task_evaluations": 0,
        "source_heavy_atoms": int(source.n_real_atoms),
        "endpoint_heavy_atoms": int(product.n_real_atoms),
    }
    STATS["chains"] += 1
    STATS["planned_total"] += int(length)
    STATS["completed_total"] += len(actions)
    STATS["terminated_early"] += int(halt is not None)
    STATS["sampler_attempts"] += attempts
    STATS["sampler_rejections"] += rejections
    STATS["seconds"] += _perf_counter() - _began
    if halt:
        STATS["halt_reasons"][halt] = STATS["halt_reasons"].get(halt, 0) + 1
    return source, program, assignment, trace, metadata
