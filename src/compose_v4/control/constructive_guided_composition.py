"""Route-distilled guidance over the existing generic constructive compiler.

This is an additive proposal expert.  It does not replace Dynamic-v0, change the
executor, retrieve a route, or inspect a target.  At each construction step it asks the
unchanged generic compiler for a bounded candidate set, scores the realized generic
``(WHERE, HOW)`` decisions with a frozen route-distilled prior, and chooses one while
retaining an explicit exploration floor.

The distinction from post-hoc reranking is important: guidance changes which stage is
committed before the rest of the complete program is generated.  The resulting program
is still exact-executed and task eligibility is checked only on its completed endpoint.
"""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

from compose_v4.control.constructive_composition import (
    DEFAULT_BUILDERS,
    MAX_BLOCKS,
    MAX_PRIMITIVES,
    family_order,
)
from compose_v4.control.constructive_decision import stage_decision
from compose_v4.control.constructive_features import mode_identity, mode_matrix, site_features
from compose_v4.control.constructive_policy import pair_scores
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import compile_generic_module
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

SCHEMA_VERSION = "constructive_guided_composition_v1"


def score_stage(model, vocabulary, source, stage) -> tuple[float | None, dict | None]:
    """Score a compiled stage under the same decision law used for route training."""
    state = encode_state(source)
    decision = stage_decision(state, stage.get("actions") or [])
    if decision is None:
        return None, None
    sites, slots = site_features(state)
    mode_lookup = {mode_identity(mode): index for index, mode in enumerate(vocabulary)}
    key = mode_identity(decision["mode"])
    if decision["site"] not in slots or key not in mode_lookup:
        return None, decision
    scores = pair_scores(model["theta"], model["shape"], sites, mode_matrix(vocabulary))
    return float(scores[slots.index(decision["site"]), mode_lookup[key]]), decision


def _candidate_stages(
    source,
    root,
    previous_stages,
    rng,
    model,
    vocabulary,
    *,
    attempts: int,
    builders,
):
    """Compile a diverse bounded stage pool and retain only program-supported choices."""
    if attempts < 1:
        raise ValueError("guided stage generation needs a positive attempt budget")
    order, candidates, failures = [], {}, Counter()
    while len(order) < attempts:
        order.extend(family_order(rng, builders, constructive=True))
    for family in order[:attempts]:
        try:
            product, stage = compile_generic_module(source, rng, family)
            program, _ = extract_program(root, [*previous_stages, stage])
        except ValueError as error:
            failures[f"{family}:{error!s}"] += 1
            continue
        if len(program.marks) > MAX_PRIMITIVES or len(program.blocks) > MAX_BLOCKS:
            failures[f"{family}:work_limit"] += 1
            continue
        score, decision = score_stage(model, vocabulary, source, stage)
        key = canonical_state_key(product)
        candidate = {
            "product": product,
            "stage": stage,
            "family": family,
            "prior_score": score,
            "decision": decision,
        }
        held = candidates.get(key)
        if held is None or (
            score is not None and (held["prior_score"] is None or score > held["prior_score"])
        ):
            candidates[key] = candidate
    return list(candidates.values()), failures


def _choose(candidates, rng, *, exploration: float, temperature: float):
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be between zero and one")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not candidates:
        raise ValueError("the guided compiler produced no supported stage")
    scored = [candidate for candidate in candidates if candidate["prior_score"] is not None]
    if not scored or rng.random() < exploration:
        chosen = candidates[int(rng.integers(len(candidates)))]
        selection = "exploration"
    else:
        values = np.asarray([candidate["prior_score"] for candidate in scored], dtype=float)
        weights = np.exp((values - values.max()) / temperature)
        chosen = scored[int(rng.choice(len(scored), p=weights / weights.sum()))]
        selection = "route_distilled_prior"
    ranked = sorted(
        (candidate for candidate in candidates if candidate["prior_score"] is not None),
        key=lambda candidate: -candidate["prior_score"],
    )
    rank = next((index for index, candidate in enumerate(ranked) if candidate is chosen), None)
    return chosen, selection, rank


def compose_guided(
    source,
    rng,
    modules: int,
    model,
    vocabulary,
    *,
    candidates_per_step: int = 64,
    exploration: float = 0.10,
    temperature: float = 1.0,
    builders=DEFAULT_BUILDERS,
):
    """Generate one complete exact program under route-distilled stage guidance."""
    if not isinstance(modules, int) or not 1 <= modules <= MAX_BLOCKS:
        raise ValueError(f"composition depth must fit the declared {MAX_BLOCKS}-block support")
    current, stages, selections, failures = source, [], [], Counter()
    for index in range(modules):
        candidates, rejected = _candidate_stages(
            current,
            source,
            stages,
            rng,
            model,
            vocabulary,
            attempts=candidates_per_step,
            builders=tuple(builders),
        )
        failures.update(rejected)
        if not candidates:
            break
        chosen, selection, rank = _choose(
            candidates, rng, exploration=exploration, temperature=temperature
        )
        current = chosen["product"]
        stages.append(chosen["stage"])
        selections.append(
            {
                "index": index,
                "family": chosen["family"],
                "selection": selection,
                "prior_score": chosen["prior_score"],
                "prior_rank_in_realized_pool": rank,
                "realized_pool": len(candidates),
                "decision": chosen["decision"],
                "primitive_edits": len(chosen["stage"].get("actions") or []),
            }
        )
    if not stages:
        raise ValueError("no guided constructive stage executed")
    program, binding = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        binding,
        max_primitives=MAX_PRIMITIVES,
        max_blocks=MAX_BLOCKS,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("guided constructive composition changed on exact replay")
    return trace, {
        "schema_version": SCHEMA_VERSION,
        "requested_modules": modules,
        "completed_modules": len(stages),
        "candidates_per_step": candidates_per_step,
        "exploration": exploration,
        "temperature": temperature,
        "selections": selections,
        "failure_counts": dict(failures),
        "initial_stored_complete_routes": 0,
        "source_library_rows_loaded": 0,
    }


def prospect_guided(
    source,
    eligibility,
    rng,
    model,
    vocabulary,
    *,
    attempts: int,
    minimum_primitives: int = 15,
    depths=(5, 6, 7, 8),
    candidates_per_step: int = 64,
    exploration: float = 0.10,
    temperature: float = 1.0,
    builders=DEFAULT_BUILDERS,
    wall_seconds: float | None = None,
    progress=None,
):
    """Generate, exact-filter and deduplicate guided candidates without an oracle."""
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("guided prospecting needs a positive attempt budget")
    if not depths or any(not 1 <= depth <= MAX_BLOCKS for depth in depths):
        raise ValueError(f"every depth must fit the declared {MAX_BLOCKS}-block support")
    began = perf_counter()
    kept, sizes, rejections = {}, Counter(), Counter()
    realized = failed = below_floor = 0
    for index in range(attempts):
        if wall_seconds is not None and perf_counter() - began >= wall_seconds:
            break
        realized += 1
        if progress is not None:
            progress({"attempts": realized, "eligible_endpoints": len(kept)})
        depth = depths[int(rng.integers(len(depths)))]
        try:
            trace, metadata = compose_guided(
                source,
                rng,
                depth,
                model,
                vocabulary,
                candidates_per_step=candidates_per_step,
                exploration=exploration,
                temperature=temperature,
                builders=builders,
            )
        except (ValueError, RuntimeError):
            failed += 1
            continue
        primitives = len(trace["actions"])
        sizes[primitives] += 1
        if primitives < minimum_primitives:
            below_floor += 1
            continue
        properties = eligibility({"smiles": trace["endpoint"]})
        if properties.get("oracle_eligible") is not True:
            for reason in properties.get("endpoint_exclusion_reasons") or []:
                rejections[reason] += 1
            continue
        kept.setdefault(
            trace["endpoint"],
            {
                "endpoint": trace["endpoint"],
                "primitives": primitives,
                "properties": properties,
                "changed_originals": len(trace["actual_changes"]["changed_original_slots"]),
                "created": trace["actual_changes"]["surviving_new_atoms"],
                "trace": trace,
                "guidance": metadata,
                "found_at_attempt": index,
            },
        )
    return {
        "schema_version": "constructive_guided_prospect_v1",
        "pool": sorted(kept.values(), key=lambda row: (-row["primitives"], row["endpoint"])),
        "attempted": realized,
        "attempt_budget": attempts,
        "build_failures": failed,
        "below_size_floor": below_floor,
        "eligible_endpoints": len(kept),
        "eligible_per_attempt": len(kept) / realized if realized else 0.0,
        "attempts_per_eligible_endpoint": realized / len(kept) if kept else None,
        "minimum_primitives": minimum_primitives,
        "size_histogram": dict(sorted(sizes.items())),
        "rejections": dict(rejections.most_common()),
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
        "pool_id": identity(sorted(kept)),
    }
