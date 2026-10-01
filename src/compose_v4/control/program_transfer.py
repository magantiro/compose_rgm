"""Cold-start the shared program optimizer without borrowing another task's labels."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.edit_program_policy import ProgramEntry, propose_programs
from compose_v4.control.molecular_task_search import dispatch_complete_proposal
from compose_v4.control.program_mutation import (
    PARAMETER_MOVES,
    branch_components,
    mutate_attachment,
    mutate_parameter,
    parameter_choices,
    select_branch,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state


def shared_program_library(records):
    """Extract program logic and closed branches, never endpoints or oracle labels."""
    programs, sources = {}, {}
    for row in records:
        full = EditProgram.from_payload(row["program"])
        variants = [full]
        variants.extend(select_branch(full, branch)[0] for branch in branch_components(full))
        for program in variants:
            key = program.program_id
            programs[key] = program
            sources.setdefault(key, set()).add(row["source_group"])
    return tuple(ProgramEntry(programs[k], tuple(sorted(sources[k]))) for k in sorted(programs))


def initial_program_batch(
    source, entries, config, *, source_group, oracle_protocol, eligibility, broad_sampler=None
):
    """Lock the first new-target endpoints before any measured archive exists.

    The broad callback returns (program, assignment, metadata), rooted at this
    exact source. Production requires that callback; local absence is explicit.
    Subsequent genuinely scored records enter ProgramOptimizer unchanged.
    """
    if config.require_broad_runtime and broad_sampler is None:
        raise ValueError("production transfer requires its authenticated broad callback")
    if not source_group or not oracle_protocol:
        raise ValueError("transfer requires source and oracle identities")
    started = perf_counter()
    # One shared, bounded binding census per source, not per proposal attempt.
    proposals = propose_programs(
        source,
        entries,
        seed=config.seed,
        count=2 * config.attempts_per_batch,
        max_bindings=config.max_bindings,
        ranked_count=min(
            config.attempts_per_batch,
            4 * config.cold_start_retrieval_candidates,
        ),
    )
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    attempts, candidates, seen = [], [], {canonical_state_key(source)}
    retrievals, retrieval_cursor = 0, 0
    for index in range(config.attempts_per_batch):
        if (
            len(candidates) >= config.candidates_per_batch
            or perf_counter() - started >= config.wall_seconds
        ):
            break
        chosen = {}
        ranked = proposals.get("ranked_draws", ())
        direct_retrieval = (
            retrievals < config.cold_start_retrieval_candidates
            and retrieval_cursor < len(ranked)
        )

        if direct_retrieval:
            draw = ranked[retrieval_cursor]
            retrieval_cursor += 1
            chosen["channel"] = "retrieval"
            program = entries[draw["program_index"]].program
            binding = tuple(draw["assignment"])
            metadata = {
                "draws": [draw],
                "mutations": [],
                "direct_retrieval": True,
            }

        def program_sampler(channel, index=index, chosen=chosen):
            chosen["channel"] = channel
            if not proposals["draws"]:
                raise ValueError("no context-compatible shared program binding")
            draw = proposals["draws"][2 * index]
            program = entries[draw["program_index"]].program
            binding = tuple(draw["assignment"])
            metadata = {"draws": [draw], "mutations": []}
            if channel == "recombination":
                other = proposals["draws"][2 * index + 1]
                donor = entries[other["program_index"]].program
                program, binding = combine_bound_programs(
                    source,
                    ((program, binding), (donor, tuple(other["assignment"]))),
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
                metadata["draws"].append(other)
            else:
                for _ in range(1 + int(rng.random() < config.double_mutation_probability)):
                    move = ("attachment", *PARAMETER_MOVES)[int(rng.integers(4))]
                    if move == "attachment":
                        binding, detail = mutate_attachment(
                            source, program, binding, rng, max_bindings=config.max_bindings
                        )
                    else:
                        choices = parameter_choices(program, move)
                        if not choices:
                            raise ValueError(f"no conditional choices for {move}")
                        choice = choices[int(rng.integers(len(choices)))]
                        program = mutate_parameter(program, move, choice)
                        detail = {"choice": choice, "choices": len(choices)}
                    metadata["mutations"].append({"kind": move, **detail})
            return program, binding, metadata

        def reference_sampler(chosen=chosen):
            chosen["channel"] = "broad"
            if broad_sampler is None:
                raise ValueError("broad_runtime_unavailable_in_local_development")
            return broad_sampler(rng)

        try:
            if direct_retrieval:
                channel = "retrieval"
            else:
                channel, (program, binding, metadata) = dispatch_complete_proposal(
                    rng,
                    program_sampler=program_sampler,
                    broad_sampler=reference_sampler,
                    probabilities=config.channel_probabilities,
                    proposal_mode=config.proposal_mode,
                )
            _, trace = execute_program_graph(
                source,
                compile_program_graph(program),
                binding,
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
            )
        except ValueError as error:
            attempts.append(
                {"attempt": index, **chosen, "status": "execution_rejected", "reason": str(error)}
            )
            continue
        endpoint = trace["endpoint"]
        properties = eligibility({"smiles": endpoint})
        if type(properties.get("oracle_eligible")) is not bool:
            raise ValueError("transfer endpoint evaluator lacks explicit eligibility")
        status = (
            "duplicate"
            if endpoint in seen
            else "eligible"
            if properties["oracle_eligible"]
            else "ineligible"
        )
        attempt = {
            "attempt": index,
            "channel": channel,
            "metadata": metadata,
            "endpoint": endpoint,
            "status": status,
            "properties": properties,
            "actual_changes": trace["actual_changes"],
        }
        attempts.append(attempt)
        if status != "eligible":
            continue
        seen.add(endpoint)
        if channel == "retrieval":
            retrievals += 1
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
        "schema_version": "initial_transferred_program_batch_v1",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "library_id": identity(
            [{"program": e.program.payload(), "sources": e.source_groups} for e in entries]
        ),
        "binding_census": {
            k: v for k, v in proposals.items() if k not in ("draws", "ranked_draws")
        },
        "candidates": candidates,
        "attempts": attempts,
        "rng_state_after_preparation": rng.bit_generator.state,
    }
    if config.cold_start_retrieval_candidates:
        body["direct_retrieval"] = {
            "candidate_target": config.cold_start_retrieval_candidates,
            "candidates_admitted": retrievals,
            "priority_attempts": retrieval_cursor,
        }
    return {
        **body,
        "batch_id": identity(body),
        "proposal_seconds": perf_counter() - started,
        "new_oracle_calls": 0,
    }
