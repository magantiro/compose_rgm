#!/usr/bin/env python3
"""Conditional decomposition of where PMO's proposal law loses a teacher program.

The sibling archaeology (``pmo_teacher_route_gap_v1``) compares the MARGINAL
distribution of teacher routes against the marginal distribution of proposals.  That
answers "the teacher uses larger edits" and stops there.  This script asks the
CONDITIONAL question: for one teacher program pi with parent state x0, replay x0
through the production proposal law and locate which factor of

    P(pi) = P(R) * P(s, m | R) * P(H, alpha, D | R, s, m) * P(execute) * P(select)

collapses first.  R is the released/retained region, s the scale, m the mode
(grow / prune / replace / remodel / restate_only), H the ring-topology change, alpha
the retained interface the new material attaches to, D the dependency structure.

Every route is then classified into exactly one failure mode:
  (a) zero_support        - the runtime cannot express the program at all
  (b) tiny_probability    - expressible and generated, at negligible mass or rank
  (c) wrong_region_scale  - the proposer commits to a different region / scale / mode
  (d) realization_failure - region, scale and mode are right, realization is not
  (e) never_evaluated     - a matching candidate is in the pool and is never selected

ZERO ORACLE CALLS.  Eligibility for a PMO task is ``Chem.MolFromSmiles is not None``
(``ProgramTask.endpoint_evaluator``), which is free; no PyTDC ``Oracle`` is ever
constructed and no scored value is required anywhere in this file.

Nothing under ``src/``, ``modal_apps/`` or ``configs/`` is imported for mutation; the
production classes are used exactly as the campaign uses them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.control.dependency_region_program import trace_structure
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.control.pmo_population_controller import (
    JUMP_CHANNEL as JUMP_CHANNEL_NAME,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments import pmo_population_v1 as population
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CORPUS = (
    ROOT
    / "diagnostics/pmo_dependency_region_program_v2/attempt_1"
    / "training_dependency_region_corpus.json.gz"
)
SCHEMA = "pmo_route_counterfactual_v1"

# Match thresholds.  Reported with a sensitivity sweep so no verdict rests on one cut.
REGION_TAU = 0.5
INTERFACE_TAU = 0.5


# ---- corpus ----------------------------------------------------------------


def load_routes() -> list[dict]:
    with gzip.open(CORPUS) as handle:
        payload = json.load(handle)["payload"]
    if payload["schema_version"] != "pmo_dependency_region_training_corpus_v2":
        raise ValueError("unexpected PMO dependency-region corpus schema")
    return payload["routes"]


def route_task(route: dict) -> str:
    tasks = sorted({member["task"] for member in route["members"]})
    return tasks[0] if len(tasks) == 1 else "|".join(tasks)


# ---- factor extraction -----------------------------------------------------


def _ring_count(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetRingInfo().NumRings()


def factors(source_state: dict, states, actions, endpoint: str) -> dict:
    """The (R, s, m, H, alpha, D) coordinates of one program, from its exact trace.

    Slot indices are the SOURCE state's own slots, so a teacher and a proposal built
    from the same parent are directly comparable coordinate by coordinate.
    """
    structure = trace_structure(tuple(states), tuple(actions))
    initial = set(structure["initial_token_slots"])
    surviving = {t for t in structure["final_active_slots"].values() if t[0] == "source"}

    touched = {t[1] for footprint in structure["footprints"] for t in footprint if t[0] == "source"}
    released = {t[1] for t in initial - surviving}
    retained = {t[1] for t in surviving}
    # alpha: retained source atoms the program still edits -- the interface the new
    # material is attached to, as distinct from the material it replaces.
    interface = touched & retained

    source = decode_state(source_state)
    final = decode_state(states[-1])
    source_atoms = int(is_element(source.atom_types).sum())
    final_atoms = int(is_element(final.atom_types).sum())
    rules = Counter(str(action["executor_rule"]) for action in actions)
    inserted, deleted = rules.get("atom_insert", 0), rules.get("atom_delete", 0)
    retained_fraction = len(surviving) / len(initial) if initial else 0.0

    if retained_fraction < 0.5:
        mode = "remodel"
    elif inserted and deleted:
        mode = "replace"
    elif inserted:
        mode = "grow"
    elif deleted:
        mode = "prune"
    else:
        mode = "restate_only"

    source_rings = _ring_count(molecular_graph_to_smiles(source))
    final_rings = _ring_count(endpoint)
    delta_rings = (
        None if source_rings is None or final_rings is None else final_rings - source_rings
    )

    return {
        "primitive_count": len(actions),
        # R
        "touched_slots": sorted(touched),
        "released_slots": sorted(released),
        "interface_slots": sorted(interface),
        "source_slot_count": len(initial),
        "retained_fraction": retained_fraction,
        # s
        "released_count": len(released),
        "scale_bucket": scale_bucket(len(released)),
        "primitive_bucket": scale_bucket(len(actions)),
        # m
        "mode": mode,
        # H
        "delta_rings": delta_rings,
        "ring_direction": None if delta_rings is None else int(np.sign(delta_rings)),
        "touches_ring_topology": bool(
            rules.get("cycle_open", 0)
            or rules.get("cycle_close", 0)
            or rules.get("ring_system_restate", 0)
        ),
        # D
        "created_handles": len(structure["created_handles"]),
        "created_handles_reused": sum(
            bool(handle["consumers"]) for handle in structure["created_handles"]
        ),
        "created_dependency_edges": len(structure["dependency_edges"]),
        # descriptive
        "delta_heavy_atoms": final_atoms - source_atoms,
        "rule_counts": dict(sorted(rules.items())),
        "endpoint": endpoint,
    }


def scale_bucket(value: int) -> str:
    for upper, label in ((0, "0"), (2, "1-2"), (5, "3-5"), (10, "6-10"), (16, "11-16"), (32, "17-32")):
        if value <= upper:
            return label
    return ">32"


def jaccard(left, right) -> float:
    left, right = set(left), set(right)
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


# ---- phase 0: support ------------------------------------------------------


def support_audit(route: dict, config) -> dict:
    """Can the production program machinery express this teacher program at all?

    This mirrors ``PmoPopulationController._joint_program`` exactly: one stage
    carrying the whole action list, compiled and executed under the SAME
    ``max_primitives`` / ``max_blocks`` the live channels use.  The source is decoded
    from the corpus payload, so it carries production slot padding -- re-parsing the
    SMILES would silently delete the whole ``atom_insert`` family.
    """
    source = decode_state(route["source_state"])
    states, actions = route["states"], route["actions"]
    stage = {
        "name": "teacher_program",
        "actions": list(actions),
        "states": list(states),
        # The stage endpoint is a canonical state KEY, exactly as
        # ``PmoPopulationController._joint_program`` passes ``bound["endpoint_key"]``.
        "endpoint": canonical_state_key(decode_state(states[-1])),
    }
    detail: dict = {
        "primitive_count": len(actions),
        "runtime_max_primitives": config.max_primitives,
        "runtime_max_blocks": config.max_blocks,
        "source_padded_slots": len(source.atom_types),
        "source_heavy_atoms": int(is_element(source.atom_types).sum()),
    }
    try:
        program, binding = extract_program(source, [stage])
        graph = compile_program_graph(program)
        profile = program_size_profile(graph, source.n_real_atoms)
        _, trace = execute_program_graph(
            source,
            graph,
            binding,
            max_primitives=config.max_primitives,
            max_blocks=config.max_blocks,
        )
        exact = trace["states"] == list(states)
        detail.update(
            {
                "expressible_under_runtime_caps": True,
                "blocks": len(program.blocks),
                "peak_heavy_atoms": profile["peak_heavy_atoms"],
                "replay_exact": bool(exact),
                "reason": None if exact else "replay_diverged",
            }
        )
        detail["expressible_under_runtime_caps"] = bool(exact)
    except Exception as failure:  # noqa: BLE001 - any refusal is the measurement
        detail.update(
            {
                "expressible_under_runtime_caps": False,
                "reason": f"{type(failure).__name__}: {failure}",
            }
        )
    # Could a multi-round campaign reach it as a chain of <= cap programs?
    segments = math.ceil(len(actions) / config.max_primitives)
    boundaries = [
        index * config.max_primitives for index in range(1, segments)
    ]
    eligible_boundaries = 0
    for index in boundaries:
        try:
            smiles = molecular_graph_to_smiles(decode_state(states[index]))
            eligible_boundaries += Chem.MolFromSmiles(smiles) is not None
        except (ValueError, KeyError, RuntimeError):
            # An unreadable boundary state simply is not an eligible round endpoint.
            continue
    detail.update(
        {
            "minimum_rounds_at_cap": segments,
            "cap_boundary_states": len(boundaries),
            "cap_boundary_states_eligible": eligible_boundaries,
            "multi_round_chain_possible": len(boundaries) == eligible_boundaries,
        }
    )
    return detail


def phase_support(out: Path) -> dict:
    config = population.configuration(20260920)
    routes = load_routes()
    rows = []
    for route in routes:
        audit = support_audit(route, config)
        teacher = factors(
            route["source_state"], route["states"], route["actions"], route["terminal_endpoint"]
        )
        rows.append(
            {
                "trace_identity": route["trace_identity"],
                "task": route_task(route),
                "task_family": route["task_family"],
                "test_fold": route["test_fold"],
                "support": audit,
                "teacher_factors": teacher,
                "corpus_flags": {
                    key: route["dependency_region_program"][key]
                    for key in (
                        "complete_representation_supported",
                        "runtime_length_supported",
                        "component_budget_supported",
                        "component_count",
                        "abstention_reason",
                    )
                },
            }
        )
    payload = {
        "schema_version": f"{SCHEMA}_support",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "routes": len(rows),
        "runtime": {
            "max_primitives": config.max_primitives,
            "max_blocks": config.max_blocks,
            "attempts_per_batch": config.attempts_per_batch,
            "candidates_per_batch": config.candidates_per_batch,
            "wall_seconds": config.wall_seconds,
        },
        "expressible": sum(r["support"]["expressible_under_runtime_caps"] for r in rows),
        "inexpressible": sum(not r["support"]["expressible_under_runtime_caps"] for r in rows),
        "reasons": dict(
            Counter(
                r["support"]["reason"]
                for r in rows
                if not r["support"]["expressible_under_runtime_caps"]
            )
        ),
        "multi_round_chain_possible": sum(r["support"]["multi_round_chain_possible"] for r in rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    return payload


# ---- parents ----------------------------------------------------------------


def parent_key(source_state: dict) -> str:
    return identity(source_state)


def group_by_parent(rows: list[dict], routes: dict[str, dict]) -> dict[str, list[dict]]:
    """Group teacher routes by the parent state they are conditioned on.

    The corpus looks like 184 independent programs and is not: they share a handful of
    source molecules, one of which carries most of the corpus.  Generating a proposal
    pool per ROUTE would regenerate the identical pool many times over (same source,
    same seeds, same config) and would present correlated samples as independent
    evidence.  One pool per distinct parent is both cheaper and the honest unit.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[parent_key(routes[row["trace_identity"]]["source_state"])].append(row)
    return grouped


# ---- phase 1: conditional proposal pool at each parent ---------------------


def allocate_pool(controller: PmoPopulationController, pool: list[dict]) -> tuple[set[str], dict]:
    """Run the REAL PMO allocator over one conditional pool.

    ``_allocate`` is called on a controller with an empty archive, which is the
    cold-start regime ``PopulationCredit`` has no cells for, so the role it reports is
    the production early-round role rather than a reconstruction.
    """
    augmented = [controller._augment(row) for row in pool]
    selected, allocation = controller._allocate(augmented)
    return {row["candidate_id"] for row in selected}, {
        "augmented": augmented,
        "allocation": allocation,
    }


def build_parent_pool(
    *,
    source_state: dict,
    task_name: str,
    replays: int,
    seed_block: int,
    base_seed: int,
    checkpoint: dict,
    wall_seconds: float,
) -> dict:
    """Draw the production proposal law conditioned on one parent state.

    ``initial_dynamic_program_batch_v21`` is the ONLY production path that proposes
    from an un-edited scored parent: ``run_program_campaign`` routes a bootstrap round
    -- round 0, plus every later round that passes ``initial_parent_fraction`` (0.2 in
    the PMO recipe) -- through it, while every other round proposes from archive
    entries instead.  Conditioning on a teacher's own parent therefore means this
    function, not ``propose_batch``, whose parents are constructed endpoints and cannot
    be pinned to an arbitrary state.  The jump channel is measured separately.
    """
    source = decode_state(source_state)
    task = ProgramTask(task_name, identity({"probe": SCHEMA, "task": task_name}), "pmo")
    eligibility = task.endpoint_evaluator()
    group = identity({"probe": SCHEMA, "parent": parent_key(source_state)})

    proposals: list[dict] = []
    attempt_status: Counter = Counter()
    pool_sizes, selected_counts, production_counts, elapsed = [], [], [], []
    for index in range(replays):
        # ``wall_seconds`` is a COMPUTE budget, not part of the proposal law: the law is
        # the RNG-driven attempt sequence and the pool is capped at
        # CHANNEL_CANDIDATE_LIMIT (16) per channel.  Measured on an unloaded machine the
        # candidate cap binds and the pool is exactly 32; on a loaded machine the 45 s
        # wall binds first and pool size becomes a function of machine load, which would
        # make the scale coordinate unreproducible.  Raising it restores that behaviour.
        seed = base_seed + 1013 * (seed_block * replays + index)
        config = replace(population.configuration(seed), seed=seed, wall_seconds=wall_seconds)
        began = perf_counter()
        batch = initial_dynamic_program_batch_v21(
            source,
            (),
            config,
            source_group=group,
            oracle_protocol=task.oracle_protocol,
            eligibility=eligibility,
        )
        elapsed.append(perf_counter() - began)
        for attempt in batch["attempts"]:
            attempt_status[str(attempt.get("status"))] += 1
        pool = batch["proposal_pool"]["candidates"]
        pool_sizes.append(len(pool))
        # Two different allocators act on a pool like this one.  A BOOTSTRAP round --
        # the only round that proposes from an un-edited parent -- selects with
        # ``arbitrate_candidates`` inside this batch; every later round selects with
        # ``PmoPopulationController._allocate``.  Recording both keeps the
        # proposal-versus-allocation verdict from resting on a choice of allocator.
        production_selected = {row["endpoint"] for row in batch["candidates"]}
        production_counts.append(len(batch["candidates"]))
        controller = PmoPopulationController(
            config,
            source_group=group,
            oracle_protocol=task.oracle_protocol,
            hierarchy=None,
            jump_checkpoint=checkpoint,
        )
        chosen_ids, detail = allocate_pool(controller, pool)
        selected_counts.append(len(chosen_ids))
        for candidate, augmented in zip(pool, detail["augmented"], strict=True):
            if candidate["source_state"] != source_state:
                raise ValueError("conditional replay drifted off the teacher parent state")
            trace = candidate["trace"]
            proposals.append(
                {
                    "replay": seed_block * replays + index,
                    "planner_channel": candidate["provenance"].get("planner_channel"),
                    "selected": augmented["candidate_id"] in chosen_ids,
                    "selected_by_production_bootstrap": candidate["endpoint"]
                    in production_selected,
                    "selection_role": detail["allocation"]
                    .get("selection_role_by_candidate", {})
                    .get(augmented["candidate_id"]),
                    "coordinates": factors(
                        candidate["source_state"],
                        trace["states"],
                        trace["actions"],
                        candidate["endpoint"],
                    ),
                }
            )
    return {
        "parent_key": parent_key(source_state),
        "parent_smiles": molecular_graph_to_smiles(source),
        "parent_heavy_atoms": int(is_element(source.atom_types).sum()),
        "seed_block": seed_block,
        "replays": replays,
        "probe_wall_seconds": wall_seconds,
        "pool_sizes": pool_sizes,
        "selected_counts": selected_counts,
        "production_bootstrap_selected_counts": production_counts,
        "proposal_seconds": elapsed,
        "attempt_status": dict(attempt_status),
        "proposals": proposals,
    }


def parent_jobs(support: dict, routes: dict[str, dict], blocks: int) -> list[tuple[str, int]]:
    """One job per (parent, seed block); blocks give a parent more independent draws."""
    grouped = group_by_parent(support["rows"], routes)
    jobs = []
    for key in sorted(grouped, key=lambda k: (-len(grouped[k]), k)):
        for block in range(blocks):
            jobs.append((key, block))
    return jobs


def phase_replay(options) -> None:
    support = json.loads(Path(options.support).read_text())
    routes = {route["trace_identity"]: route for route in load_routes()}
    grouped = group_by_parent(support["rows"], routes)
    jobs = parent_jobs(support, routes, options.blocks)
    mine = [job for index, job in enumerate(jobs) if index % options.shards == options.shard]
    checkpoint = json.loads((ROOT / population.CHECKPOINTS).read_text())["payload"][
        "checkpoints"
    ]["shared_all_routes"]
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for key, block in mine:
        members = grouped[key]
        began = perf_counter()
        results.append(
            build_parent_pool(
                source_state=routes[members[0]["trace_identity"]]["source_state"],
                task_name=members[0]["task"],
                replays=options.replays,
                seed_block=block,
                base_seed=options.base_seed,
                checkpoint=checkpoint,
                wall_seconds=options.wall_seconds,
            )
        )
        print(
            f"shard {options.shard}: parent {key[:12]} block {block} "
            f"({len(members)} routes) {perf_counter() - began:.1f}s",
            flush=True,
        )
        # Written after every job: a probe on a shared machine may be cut short, and a
        # partial pool is still a valid sample of the proposal law.
        out.write_text(
            json.dumps(
                {
                    "schema_version": f"{SCHEMA}_pool_shard",
                    "shard": options.shard,
                    "shards": options.shards,
                    "base_seed": options.base_seed,
                    "pools": results,
                },
                sort_keys=True,
            )
            + "\n"
        )


# ---- phase 1b: the jump channel, conditioned on the same parents -----------


def jump_pool_for_parent(
    *, source_state: dict, task_name: str, offsets: int, base_seed: int, checkpoint: dict
) -> dict:
    """Drive ``_generate_jump_pool`` from a teacher parent state.

    The cold-start pool has two channels; the live ``propose_batch`` has three, and the
    third -- the joint dependency-region jump -- carries plan latents fit on complete
    routes, so it is the channel most likely to emit a teacher-scale edit.  Omitting it
    would understate the proposer, so it is measured here.

    ``_generate_jump_pool`` reads only ``entry["trace"]["states"][-1]`` from its parent,
    which makes a parent pinned to the teacher's own state an exact conditional
    injection rather than a reimplementation.  ``self.batches`` is the plan-offset axis,
    so sweeping it sweeps the sealed checkpoint's plan catalog.
    """
    source = decode_state(source_state)
    task = ProgramTask(task_name, identity({"probe": SCHEMA, "task": task_name}), "pmo")
    eligibility = task.endpoint_evaluator()
    group = identity({"probe": SCHEMA, "parent": parent_key(source_state)})
    # A ROOT entry: the teacher parent is a seed state, not a constructed endpoint, so it
    # carries zero construction ancestry and an empty mark list.  ``_generate_jump_pool``
    # reads the current state from ``trace.states[-1]``; ``_continuation_lineage`` reads
    # the lineage fields.  Both are supplied here rather than faked downstream.
    entry = {
        "entry_id": "teacher_parent",
        "trace": {"states": [source_state]},
        "source_state": source_state,
        "endpoint": molecular_graph_to_smiles(source),
        "program": {"marks": []},
        # A seed state inherits no static-control preference; ``_candidate`` copies this
        # through as ``inherited_static_score``.
        "static_score": None,
    }
    parent = {
        "entry_id": "teacher_parent",
        "parent_probability": 1.0,
        "parent_measured_score": 0.0,
    }

    proposals, attempt_status, plans = [], Counter(), Counter()
    for offset in range(offsets):
        seed = base_seed + 1013 * offset
        config = replace(population.configuration(seed), seed=seed, wall_seconds=600.0)
        controller = PmoPopulationController(
            config,
            source_group=group,
            oracle_protocol=task.oracle_protocol,
            hierarchy=None,
            jump_checkpoint=checkpoint,
        )
        controller.batches = offset
        attempts, candidates, _ = controller._generate_jump_pool(
            eligibility, set(), [(entry, parent)]
        )
        for attempt in attempts:
            attempt_status[str(attempt.get("status"))] += 1
        for candidate in candidates:
            trace = candidate["trace"]
            plans[
                candidate["provenance"]["metadata"]["joint_dependency_region_jump"]["plan_id"]
            ] += 1
            proposals.append(
                {
                    "replay": offset,
                    "planner_channel": JUMP_CHANNEL_NAME,
                    "selected": None,
                    "selected_by_production_bootstrap": None,
                    "selection_role": None,
                    "coordinates": factors(
                        candidate["source_state"],
                        trace["states"],
                        trace["actions"],
                        candidate["endpoint"],
                    ),
                }
            )
    return {
        "parent_key": parent_key(source_state),
        "parent_smiles": molecular_graph_to_smiles(source),
        "offsets": offsets,
        "attempt_status": dict(attempt_status),
        "distinct_plans_realized": len(plans),
        "proposals": proposals,
    }


def phase_jump(options) -> None:
    support = json.loads(Path(options.support).read_text())
    routes = {route["trace_identity"]: route for route in load_routes()}
    grouped = group_by_parent(support["rows"], routes)
    keys = sorted(grouped, key=lambda k: (-len(grouped[k]), k))
    mine = [key for index, key in enumerate(keys) if index % options.shards == options.shard]
    checkpoint = json.loads((ROOT / population.CHECKPOINTS).read_text())["payload"][
        "checkpoints"
    ]["shared_all_routes"]
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for key in mine:
        members = grouped[key]
        results.append(
            jump_pool_for_parent(
                source_state=routes[members[0]["trace_identity"]]["source_state"],
                task_name=members[0]["task"],
                offsets=options.replays,
                base_seed=options.base_seed,
                checkpoint=checkpoint,
            )
        )
        print(f"jump shard {options.shard}: parent {key[:12]}", flush=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": f"{SCHEMA}_jump_shard",
                    "shard": options.shard,
                    "pools": results,
                },
                sort_keys=True,
            )
            + "\n"
        )


# ---- phase 2: reduce and classify ------------------------------------------

MODES = (
    "a_zero_support",
    "b_tiny_probability",
    "c_wrong_region_scale",
    "d_realization_failure",
    "e_never_evaluated",
)


def match_indicators(teacher: dict, proposal: dict) -> dict:
    """Coordinate-by-coordinate comparison of one proposal against one teacher program.

    Each indicator is an UPPER BOUND on reproducing the teacher: a proposal that lands
    in the teacher's (R, s, m, H, alpha, D) cell is teacher-LIKE, not the teacher, so a
    factor that is already zero on the cell is zero on the program too.
    """
    # The region is the RELEASED set, which is what a replace/remodel teacher is about.
    # A teacher that releases nothing has an empty released set, and
    # Jaccard(empty, empty) == 1 would hand every non-deleting proposal a free region
    # hit; those teachers are compared on the TOUCHED set instead.
    if teacher["released_slots"]:
        region_released = jaccard(teacher["released_slots"], proposal["released_slots"])
    else:
        region_released = jaccard(teacher["touched_slots"], proposal["touched_slots"])
    interface = jaccard(teacher["interface_slots"], proposal["interface_slots"])
    region_hit = region_released >= REGION_TAU
    scale_hit = teacher["scale_bucket"] == proposal["scale_bucket"]
    mode_hit = teacher["mode"] == proposal["mode"]
    topology_hit = teacher["ring_direction"] == proposal["ring_direction"]
    dependency_hit = (teacher["created_handles_reused"] > 0) == (
        proposal["created_handles_reused"] > 0
    )
    realization_hit = interface >= INTERFACE_TAU and topology_hit and dependency_hit
    return {
        "region_jaccard_released": region_released,
        "region_jaccard_touched": jaccard(
            teacher["touched_slots"], proposal["touched_slots"]
        ),
        "interface_jaccard": interface,
        "region_hit": bool(region_hit),
        "scale_hit": bool(scale_hit),
        "mode_hit": bool(mode_hit),
        "topology_hit": bool(topology_hit),
        "dependency_hit": bool(dependency_hit),
        "realization_hit": bool(realization_hit),
        "cell_hit": bool(region_hit and scale_hit and mode_hit and realization_hit),
        "endpoint_exact": proposal["endpoint"] == teacher["endpoint"],
    }


def upper_bound(successes: int, trials: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound; the honest reading of a zero count."""
    if trials == 0:
        return 1.0
    if successes == 0:
        return 1.0 - 0.05 ** (1.0 / trials)
    return min(1.0, (successes + 1.96 * math.sqrt(successes)) / trials)


def chain(proposals: list[dict], *, region_tau: float, interface_tau: float) -> dict:
    total = len(proposals)
    region = [p for p in proposals if p["region_jaccard_released"] >= region_tau]
    region_scale_mode = [p for p in region if p["scale_hit"] and p["mode_hit"]]
    cell = [
        p
        for p in region_scale_mode
        if p["interface_jaccard"] >= interface_tau and p["topology_hit"] and p["dependency_hit"]
    ]
    selected_cell = [p for p in cell if p["selected"]]
    production_cell = [p for p in cell if p.get("selected_by_production_bootstrap")]
    return {
        "proposals": total,
        "n_region": len(region),
        "n_region_scale_mode": len(region_scale_mode),
        "n_cell": len(cell),
        "n_cell_selected": len(selected_cell),
        "p_region": len(region) / total if total else 0.0,
        "p_region_upper95": upper_bound(len(region), total),
        "p_scale_mode_given_region": (
            len(region_scale_mode) / len(region) if region else 0.0
        ),
        "p_scale_mode_given_region_upper95": upper_bound(len(region_scale_mode), len(region)),
        "p_realization_given_rsm": (
            len(cell) / len(region_scale_mode) if region_scale_mode else 0.0
        ),
        "p_realization_given_rsm_upper95": upper_bound(len(cell), len(region_scale_mode)),
        "n_cell_selected_production": len(production_cell),
        "p_select_given_cell_production": len(production_cell) / len(cell) if cell else 0.0,
        "p_select_given_cell": len(selected_cell) / len(cell) if cell else 0.0,
        "p_select_given_cell_upper95": upper_bound(len(selected_cell), len(cell)),
        "p_cell_joint": len(cell) / total if total else 0.0,
    }


def classify(route: dict, link: dict) -> dict:
    if not route["expressible_under_runtime_caps"]:
        return {
            "mode": "a_zero_support",
            "decision": "runtime program caps",
            "detail": route["support_reason"],
        }
    if link["n_region"] == 0:
        return {
            "mode": "c_wrong_region_scale",
            "decision": "region R never proposed",
            "detail": f"p_region <= {link['p_region_upper95']:.2e} over {link['proposals']}",
        }
    if link["n_region_scale_mode"] == 0:
        return {
            "mode": "c_wrong_region_scale",
            "decision": "scale/mode wrong given the region",
            "detail": (
                f"p(s,m|R) <= {link['p_scale_mode_given_region_upper95']:.2e} "
                f"over {link['n_region']} region hits"
            ),
        }
    if link["n_cell"] == 0:
        return {
            "mode": "d_realization_failure",
            "decision": "topology/attachment/dependency not realized",
            "detail": (
                f"p(H,alpha,D|R,s,m) <= {link['p_realization_given_rsm_upper95']:.2e} "
                f"over {link['n_region_scale_mode']}"
            ),
        }
    if link["n_cell_selected"] == 0 and link["n_cell_selected_production"] == 0:
        return {
            "mode": "e_never_evaluated",
            "decision": "neither allocator ever selects the matching candidate",
            "detail": (
                f"{link['n_cell']} cell-matching candidates generated, 0 selected by "
                f"either allocator; p_select <= {link['p_select_given_cell_upper95']:.2e}"
            ),
        }
    return {
        "mode": "b_tiny_probability",
        "decision": "reachable cell, negligible joint mass",
        "detail": f"p_cell_joint = {link['p_cell_joint']:.2e}",
    }


def _load_pools(folder: Path, pattern: str) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Merge every shard's pools into one proposal list per parent, plus its census."""
    pools: dict[str, list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for path in sorted(folder.glob(pattern)):
        payload = json.loads(path.read_text())
        for pool in payload["pools"]:
            pools[pool["parent_key"]].extend(pool["proposals"])
            record = meta.setdefault(
                pool["parent_key"],
                {
                    "parent_smiles": pool["parent_smiles"],
                    "attempt_status": Counter(),
                    "pool_sizes": [],
                    "selected_counts": [],
                    "production_bootstrap_selected_counts": [],
                },
            )
            record["attempt_status"].update(pool.get("attempt_status", {}))
            for key in (
                "pool_sizes",
                "selected_counts",
                "production_bootstrap_selected_counts",
            ):
                record[key].extend(pool.get(key, []))
    return pools, meta


def phase_reduce(options) -> dict:
    support = json.loads(Path(options.support).read_text())
    folder = Path(options.shard_dir)
    pools, meta = _load_pools(folder, "shard_*.json")
    if not pools:
        raise ValueError("no conditional proposal pools found")
    jump_pools, _ = _load_pools(folder, "jump_*.json")
    routes = {route["trace_identity"]: route for route in load_routes()}

    rows, skipped = [], 0
    for record in support["rows"]:
        key = parent_key(routes[record["trace_identity"]]["source_state"])
        proposals = pools.get(key)
        if not proposals:
            skipped += 1
            continue
        teacher = record["teacher_factors"]
        described = [
            {**match_indicators(teacher, item["coordinates"]), **item} for item in proposals
        ]
        order = sorted(
            range(len(described)),
            key=lambda i: (-described[i]["region_jaccard_released"], i),
        )
        for rank, position in enumerate(order):
            described[position]["region_rank"] = rank

        link = chain(described, region_tau=REGION_TAU, interface_tau=INTERFACE_TAU)
        info = meta[key]
        attempts = info["attempt_status"]
        total_attempts = sum(attempts.values())
        rejected = attempts.get("execution_rejected", 0)
        link["p_execute"] = (
            (total_attempts - rejected) / total_attempts if total_attempts else 0.0
        )
        pool_total = sum(info["pool_sizes"])
        link["p_select_base_rate"] = (
            sum(info["selected_counts"]) / pool_total if pool_total else 0.0
        )
        link["p_select_base_rate_production"] = (
            sum(info["production_bootstrap_selected_counts"]) / pool_total
            if pool_total
            else 0.0
        )

        # Partial progress: one proposal cannot BE a 31-primitive teacher program, but it
        # can be a legal first instalment of one.  A proposal whose released set is a
        # nonempty SUBSET of the teacher's is on the teacher's path without being the
        # whole journey.  Without this a multi-round-reachable teacher is
        # indistinguishable from one the proposer never heads toward at all.
        teacher_released = set(teacher["released_slots"])
        teacher_touched = set(teacher["touched_slots"])
        subset_rows, precision_rows, touch_rows = [], [], []
        for item in described:
            released = set(item["coordinates"]["released_slots"])
            touched = set(item["coordinates"]["touched_slots"])
            if released:
                subset_rows.append(released <= teacher_released)
                precision_rows.append(len(released & teacher_released) / len(released))
            touch_rows.append(
                len(touched & teacher_touched) / len(touched) if touched else 0.0
            )
        link["n_releasing"] = len(subset_rows)
        link["p_releasing"] = len(subset_rows) / link["proposals"] if link["proposals"] else 0.0
        link["n_region_subset"] = sum(subset_rows)
        link["p_region_subset"] = (
            sum(subset_rows) / link["proposals"] if link["proposals"] else 0.0
        )
        link["median_release_precision"] = (
            float(np.median(precision_rows)) if precision_rows else None
        )
        link["median_touch_precision"] = float(np.median(touch_rows)) if touch_rows else None

        jump = jump_pools.get(key) or []
        jump_described = [
            {**match_indicators(teacher, item["coordinates"]), **item} for item in jump
        ]
        jump_link = (
            chain(jump_described, region_tau=REGION_TAU, interface_tau=INTERFACE_TAU)
            if jump_described
            else None
        )

        verdict = classify(
            {
                "expressible_under_runtime_caps": record["support"][
                    "expressible_under_runtime_caps"
                ],
                "support_reason": record["support"]["reason"],
            },
            link,
        )
        cell_ranks = [item["region_rank"] for item in described if item["cell_hit"]]
        best = max((item["region_jaccard_released"] for item in described), default=0.0)
        rows.append(
            {
                "trace_identity": record["trace_identity"],
                "task": record["task"],
                "task_family": record["task_family"],
                "parent_key": key,
                "parent_smiles": info["parent_smiles"],
                "expressible_under_runtime_caps": record["support"][
                    "expressible_under_runtime_caps"
                ],
                "teacher": {
                    field: teacher[field]
                    for field in (
                        "primitive_count",
                        "released_count",
                        "scale_bucket",
                        "mode",
                        "retained_fraction",
                        "delta_rings",
                        "delta_heavy_atoms",
                        "created_handles_reused",
                        "source_slot_count",
                    )
                },
                "chain": link,
                "jump_channel_chain": jump_link,
                "verdict": verdict,
                "sensitivity": {
                    f"region_tau={rt}|interface_tau={it}": {
                        field: value
                        for field, value in chain(
                            described, region_tau=rt, interface_tau=it
                        ).items()
                        if field
                        in ("n_region", "n_region_scale_mode", "n_cell", "n_cell_selected")
                    }
                    for rt in (0.25, 0.5, 0.75)
                    for it in (0.25, 0.5, 0.75)
                },
                "best_region_jaccard": best,
                "cell_hit_best_rank": min(cell_ranks) if cell_ranks else None,
                "exact_endpoint_hits": sum(item["endpoint_exact"] for item in described),
            }
        )

    by_mode = Counter(row["verdict"]["mode"] for row in rows)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_task[row["task"]][row["verdict"]["mode"]] += 1

    proposal_modes: Counter = Counter()
    proposal_released, proposal_primitives = [], []
    for key, proposals in pools.items():
        for item in proposals:
            proposal_modes[item["coordinates"]["mode"]] += 1
            proposal_released.append(item["coordinates"]["released_count"])
            proposal_primitives.append(item["coordinates"]["primitive_count"])

    headline = {
        "routes_classified": len(rows),
        "routes_without_a_pool": skipped,
        "distinct_parents": len(pools),
        "proposals_per_parent": {key: len(value) for key, value in sorted(pools.items())},
        "failure_mode_distribution": dict(by_mode),
        "failure_mode_fraction": {
            mode: by_mode.get(mode, 0) / len(rows) for mode in MODES
        },
        "corpus_support": {
            "routes": support["routes"],
            "expressible": support["expressible"],
            "inexpressible": support["inexpressible"],
            "reasons": support["reasons"],
            "multi_round_chain_possible": support["multi_round_chain_possible"],
        },
        "exact_endpoint_hits_total": sum(row["exact_endpoint_hits"] for row in rows),
        "teacher_primitive_median": float(
            np.median([row["teacher"]["primitive_count"] for row in rows])
        ),
        "proposal_primitive_median": float(np.median(proposal_primitives)),
        "teacher_released_median": float(
            np.median([row["teacher"]["released_count"] for row in rows])
        ),
        "proposal_released_median": float(np.median(proposal_released)),
        "proposal_release_nothing_fraction": float(
            np.mean([value == 0 for value in proposal_released])
        ),
        "teacher_mode_census": dict(Counter(row["teacher"]["mode"] for row in rows)),
        "proposal_mode_census": dict(proposal_modes),
        "median_p_region": float(np.median([row["chain"]["p_region"] for row in rows])),
        "median_p_scale_mode_given_region": float(
            np.median([row["chain"]["p_scale_mode_given_region"] for row in rows])
        ),
        "median_p_execute": float(np.median([row["chain"]["p_execute"] for row in rows])),
        "median_p_select_base_rate": float(
            np.median([row["chain"]["p_select_base_rate"] for row in rows])
        ),
        "median_p_select_base_rate_production": float(
            np.median([row["chain"]["p_select_base_rate_production"] for row in rows])
        ),
        "median_p_region_subset": float(
            np.median([row["chain"]["p_region_subset"] for row in rows])
        ),
        "routes_with_zero_region_subset": sum(
            row["chain"]["n_region_subset"] == 0 for row in rows
        ),
        "jump_channel_measured": bool(jump_pools),
    }
    payload = {
        "schema_version": SCHEMA,
        "accounting": "replay-only probe; zero oracle calls, nothing launched",
        "region_tau": REGION_TAU,
        "interface_tau": INTERFACE_TAU,
        "headline": headline,
        "by_task": {task: dict(counts) for task, counts in sorted(by_task.items())},
        "routes": rows,
    }
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["support", "replay", "jump", "reduce"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--support")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260920)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--shard-dir")
    options = parser.parse_args()
    if options.phase == "support":
        payload = phase_support(Path(options.out))
        print(
            json.dumps(
                {k: v for k, v in payload.items() if k not in ("rows",)},
                sort_keys=True,
                indent=1,
            )
        )
    elif options.phase == "replay":
        phase_replay(options)
    elif options.phase == "jump":
        phase_jump(options)
    else:
        payload = phase_reduce(options)
        print(json.dumps(payload["headline"], sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
