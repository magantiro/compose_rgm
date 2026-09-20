#!/usr/bin/env python3
"""PMO-v2 binder failure DECOMPOSITION: why a joint plan does not realize on a parent.

The PMO-v1 jump channel fails 96.7% of attempts with one reason string,
``joint plan has no legal binding on this parent``.  That string conflates three
outcomes that have completely different repairs:

  (a) NO_COMPATIBLE_BINDING -- no role-consistent realization of this plan exists on
      this parent at all.  Proven, not assumed: an exhaustive referee search with the
      beam removed also fails.  Exact teacher replay does NOT bear on this; it only
      shows those transformations execute on their own recorded states.
  (b) DISCARDED_BY_SEARCH -- a realization exists and the bounded beam threw it away.
      Measured as: referee succeeds where the bounded beam failed.
  (c) BUDGET_EXHAUSTED -- the referee hit a frontier/time cap without deciding.

This decomposition is the deliverable.  The beam-width sweep and the ranking A/B are
INSTRUMENTS for sizing (b) only.  They are an offline diagnostic and explicitly not a
production tuning campaign: primitive autoregression plus beam is the legacy
implementation under investigation, not the intended architecture (source-conditioned
structural/dependency-region program policy -> constrained realization -> exact
execution -> FiberControl).  Nothing here is wired into production.

The pinned modules ``pmo_joint_dependency_jump`` and ``pmo_population_controller`` are
imported read-only.  All chemistry -- successor enumeration, role supervision, program
extraction, executor replay -- is the pinned implementation in every arm, so an arm
differs from v1 only in the search policy under test.

Zero oracle calls.  Binding is pure graph work.

Invariants maintained:
  * ``bind_instrumented(..., ranking="hash")`` reproduces the pinned ``bind_joint_plan``
    endpoint set exactly at the same beam width (``verify_equivalence``); the sweep is
    meaningless without this.
  * the referee dedupes on exact (state, created-map, next-ordinal), which is the full
    determinant of a prefix's future, so collapsing it preserves the EXISTENCE answer.
  * every reported realization is replayed through the pinned ``execute_program``.
  * complete programs only.  A partial/prefix binding is NOT reported as a realization:
    an executable fragment that does not meet the plan's own completion conditions is a
    different, unspecified transformation, not a smaller version of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision, atom_role
from compose_v4.control.pmo_joint_dependency_jump import (
    bind_joint_plan,
    enumerate_role_successors,
)
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    MAXIMUM_COMPONENTS,
    MAXIMUM_PRIMITIVES,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
PRODUCTION_INIT = ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
TEACHER_CORPUS = ROOT / (
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)
SCHEMA = "pmo_binder_repair_v1"

# The pinned controller call site: pmo_population_controller.py:275.
PRODUCTION_BEAM_WIDTH = 4


# ---- inputs ----


def load_plans() -> list[dict[str, Any]]:
    """The 95 shared_all_routes plan latents the production controller binds."""

    envelope = json.loads(CHECKPOINT.read_text())
    checkpoint = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("unexpected jump checkpoint fit scope")
    return list(checkpoint["plan_latents"])


def load_production_parents() -> list[dict[str, Any]]:
    """The pinned 16-molecule production initialization (read-only)."""

    body = json.loads(PRODUCTION_INIT.read_text())
    return [
        {
            "parent_id": f"production_init:{index}",
            "stratum": "production_init",
            "smiles": row["endpoint"],
            "state": row["state"],
        }
        for index, row in enumerate(body["candidates"])
    ]


def load_teacher_root_parents() -> list[dict[str, Any]]:
    """The distinct teacher route source molecules (7 of them across 184 routes)."""

    import gzip

    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    if corpus["schema_version"] != "pmo_dependency_region_training_corpus_v2":
        raise ValueError("unexpected PMO dependency-region corpus schema")
    parents, seen = [], set()
    for route in corpus["routes"]:
        state = route["states"][0]
        key = json.dumps(state, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        graph = decode_state(state)
        parents.append(
            {
                "parent_id": f"teacher_root:{len(parents)}",
                "stratum": "teacher_root",
                "smiles": canonical_state_key(graph),
                "state": state,
            }
        )
    return parents


def load_plan_source_witnesses() -> dict[str, list[dict[str, Any]]]:
    """Map each plan_id to the teacher route sources that GENERATED it.

    This is the positive control the decomposition needs.  For such a pair the teacher
    route is itself a witness that a complete role-consistent realization exists on
    that parent, so the referee must answer ``binding_exists``.  Any failure of the
    bounded beam there is therefore DISCARDED_BY_SEARCH by construction, with no
    appeal to the possibility that the parent was simply incompatible.
    """

    import gzip

    from compose_v4.control.pmo_joint_dependency_jump import _generic_role_sequence

    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    witnesses: dict[str, list[dict[str, Any]]] = {}
    for index, route in enumerate(corpus["routes"]):
        if not route["dependency_region_program"].get("complete_representation_supported"):
            continue
        roles = _generic_role_sequence(route)
        plan_id = identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        witnesses.setdefault(plan_id, []).append(
            {
                "parent_id": f"teacher_witness_route:{index}",
                "stratum": "teacher_witness",
                "state": route["states"][0],
                "route_primitive_count": len(roles),
            }
        )
    return witnesses


# ---- prefix state ----


@dataclass(frozen=True)
class _Prefix:
    graph: Any
    states: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    created: tuple[tuple[int, int, int], ...]
    next_ordinal: int


def _created_dict(prefix: _Prefix) -> dict[int, tuple[int, int]]:
    return {slot: (ordinal, step) for slot, ordinal, step in prefix.created}


def _future_signature(prefix: _Prefix) -> str:
    """Everything a prefix's continuations can depend on.

    The binder's next step reads only the graph, the created-handle map and the next
    ordinal.  Two prefixes agreeing on all three have identical continuations, so the
    referee may keep one without changing whether a complete binding EXISTS.  Note the
    graph is compared by its exact encoded slots, not by a canonical key: the created
    map is keyed by slot address, so isomorphic graphs with different layouts are not
    interchangeable here.
    """

    return identity(
        {
            "state": prefix.states[-1],
            "created": list(prefix.created),
            "next_ordinal": prefix.next_ordinal,
        }
    )


def _hash_rank_key(plan: dict[str, Any], step: int, candidate_key: Any) -> str:
    """Byte-for-byte the pinned v1 survivor ordering (pmo_joint_dependency_jump.py:359)."""

    return identity(
        {
            "plan_id": plan["plan_id"],
            "step": step,
            "candidate": candidate_key,
            "beam_stream": 20260919,
        }
    )


def _operand_descriptors(role: dict[str, Any]) -> list[dict[str, Any]]:
    return [row["descriptor"] for row in role.get("operands", ())]


def _lookahead_matches(
    graph,
    created: dict[int, tuple[int, int]],
    step: int,
    descriptors: list[dict[str, Any]],
) -> int:
    """Exact one-step-lookahead necessary condition for extending a prefix.

    Returns the minimum, over the next role's operand descriptors, of the number of
    active atoms whose address-free descriptor equals it.  A zero means the prefix is
    provably dead at the next step: the binder computes exactly this descriptor for
    whatever operand it picks, so no enumeration can produce a role match.  This is a
    necessary, not sufficient, condition -- it ignores the action parameters and the
    executor's own legality.
    """

    if not descriptors:
        return 1
    available = []
    for slot in np.flatnonzero(is_element(graph.atom_types)):
        try:
            available.append(atom_role(graph, int(slot), created, step))
        except ValueError:
            continue
    return min(sum(1 for found in available if found == wanted) for wanted in descriptors)


def static_first_step_feasible(source, plan: dict[str, Any]) -> bool:
    """Zero-search NO_COMPATIBLE_BINDING certificate at depth 0.

    If the parent carries no atom matching some operand descriptor of role 0, no beam
    width and no ranking can bind this plan.  Cheap enough to run on every pair.
    """

    roles = plan["roles"]
    if not roles:
        return False
    return _lookahead_matches(source, {}, 0, _operand_descriptors(roles[0])) > 0


# ---- role-specification attribution (diagnostic, NOT a proposed relaxation) ----

# The fields ``atom_role`` pins on every operand of every step.  A v1 role match
# requires ALL of them to be equal, at all ~28-31 steps, simultaneously.
ROLE_DESCRIPTOR_FIELDS = (
    "atom_type",
    "formal_charge",
    "implicit_hydrogens",
    "degree",
    "bond_class_histogram",
    "neighbor_element_histogram",
    "origin",
    "created_ordinal",
    "creation_lag",
)

# Progressively coarser views, used only to ATTRIBUTE the blocking constraint.
ATTRIBUTION_VIEWS: dict[str, tuple[str, ...]] = {
    "v1_full": ROLE_DESCRIPTOR_FIELDS,
    "drop_neighbor_element_histogram": tuple(
        f for f in ROLE_DESCRIPTOR_FIELDS if f != "neighbor_element_histogram"
    ),
    "drop_implicit_hydrogens": tuple(
        f for f in ROLE_DESCRIPTOR_FIELDS if f != "implicit_hydrogens"
    ),
    "drop_bond_class_histogram": tuple(
        f for f in ROLE_DESCRIPTOR_FIELDS if f != "bond_class_histogram"
    ),
    "drop_both_histograms": tuple(
        f
        for f in ROLE_DESCRIPTOR_FIELDS
        if f not in ("bond_class_histogram", "neighbor_element_histogram")
    ),
    "element_and_degree_only": ("atom_type", "degree", "origin"),
    "element_only": ("atom_type", "origin"),
}


def _project(descriptor: dict[str, Any], fields: tuple[str, ...]) -> tuple:
    return tuple(json.dumps(descriptor.get(name), sort_keys=True) for name in fields)


def first_step_feasible_under(
    source, plan: dict[str, Any], fields: tuple[str, ...]
) -> bool:
    """Step-0 operand availability when operands are compared on ``fields`` only.

    DIAGNOSTIC ONLY.  Coarsening the comparison does not define a legal weaker binding:
    a match under a projected view realizes a DIFFERENT transformation from the one the
    plan specifies.  This measures which pinned field is responsible for the depth-0
    refusal, so a repair can target the representation rather than the search.
    """

    roles = plan["roles"]
    if not roles:
        return False
    descriptors = _operand_descriptors(roles[0])
    if not descriptors:
        return True
    available = []
    for slot in np.flatnonzero(is_element(source.atom_types)):
        try:
            available.append(_project(atom_role(source, int(slot), {}, 0), fields))
        except ValueError:
            continue
    pool = set(available)
    return all(_project(wanted, fields) in pool for wanted in descriptors)


def plan_specification_census(plan: dict[str, Any]) -> dict[str, Any]:
    """How much step-ordering and created-handle structure the plan pins."""

    roles = plan["roles"]
    operand_total = sum(len(role.get("operands", ())) for role in roles)
    created_operands = sum(
        1
        for role in roles
        for row in role.get("operands", ())
        if row["descriptor"]["origin"] == "route_created"
    )
    lags = [
        int(row["descriptor"]["creation_lag"])
        for role in roles
        for row in role.get("operands", ())
        if row["descriptor"]["origin"] == "route_created"
    ]
    return {
        "plan_id": plan["plan_id"],
        "primitive_count": len(roles),
        "operand_total": operand_total,
        "created_origin_operands": created_operands,
        "created_origin_fraction": created_operands / operand_total if operand_total else 0.0,
        "creation_lag_nonzero": sum(1 for value in lags if value > 0),
        "creation_lag_max": max(lags) if lags else 0,
    }


# ---- expansion shared by every arm ----


def _expand(prefix: _Prefix, desired: dict[str, Any], desired_identity: str, step: int):
    """All role-consistent children of one prefix, using the PINNED enumerator."""

    prefix_created = _created_dict(prefix)
    enumerated = 0
    children: list[tuple[tuple[str, str], _Prefix]] = []
    for candidate in enumerate_role_successors(
        prefix.graph, desired, created=prefix_created, step=step
    ):
        enumerated += 1
        created = dict(prefix_created)
        observed, next_ordinal = action_role_supervision(
            prefix.graph, candidate.action_record, created, step, prefix.next_ordinal
        )
        if identity(observed) != desired_identity:
            continue
        child = _Prefix(
            graph=candidate.successor,
            states=(*prefix.states, encode_state(candidate.successor)),
            actions=(*prefix.actions, candidate.action_record),
            created=tuple(
                sorted((slot, ordinal, made) for slot, (ordinal, made) in created.items())
            ),
            next_ordinal=next_ordinal,
        )
        children.append(((candidate.successor_key, identity(child.actions)), child))
    return enumerated, children


def _finalize(source, prefix: _Prefix, plan: dict[str, Any]) -> dict[str, Any] | None:
    """Pinned program extraction + executor replay on a COMPLETE binding."""

    if len(prefix.actions) != len(plan["roles"]):
        raise ValueError("only complete bindings may be finalized")
    program = dependency_region_program(
        list(prefix.states),
        list(prefix.actions),
        config=DependencyRegionConfig(
            runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
            runtime_maximum_components=MAXIMUM_COMPONENTS,
        ),
    )
    if not program["exact_replay"] or not program["complete_representation_supported"]:
        return None
    endpoint, receipt = execute_program(source, list(prefix.actions))
    if receipt["states"] != list(prefix.states):
        raise RuntimeError("joint plan exact replay changed")
    endpoint_key = canonical_state_key(endpoint)
    if endpoint_key == canonical_state_key(source):
        return None
    return {
        "plan_id": plan["plan_id"],
        "endpoint_key": endpoint_key,
        "primitive_count": len(prefix.actions),
        "plan_primitive_count": int(plan["primitive_count"]),
        "component_count": program["component_count"],
        "created_dependency_edges": len(program["created_dependency_edges"]),
        "retained_fraction": retained_fraction(source, endpoint),
        "delta_heavy_atoms": int(endpoint.n_real_atoms) - int(source.n_real_atoms),
        "exact_replay": True,
    }


def retained_fraction(source, endpoint) -> float:
    """Fraction of the parent's heavy atoms surviving, by the controller's own rule."""

    from compose_v4.control.pmo_population_controller import _retained_fraction

    return float(_retained_fraction(source, endpoint))


# ---- bounded beam (the legacy policy under investigation) ----


def bind_instrumented(
    source,
    plan: dict[str, Any],
    *,
    beam_width: int,
    ranking: str = "hash",
    seconds_cap: float | None = None,
) -> dict[str, Any]:
    """Bind one plan under a bounded beam, with per-step instrumentation.

    ``ranking='hash'`` reproduces v1 exactly.  ``ranking='compatibility'`` orders
    survivors by the one-step-lookahead match count before truncation, breaking ties on
    the v1 hash so the arm stays deterministic.  Only COMPLETE bindings are returned.
    """

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    began = perf_counter()
    prefixes = [
        _Prefix(graph=source, states=(encode_state(source),), actions=(), created=(),
                next_ordinal=0)
    ]
    roles = plan["roles"]
    steps: list[dict[str, Any]] = []
    died_at: int | None = None
    capped_at: int | None = None
    enumerated_total = 0
    for step, desired in enumerate(roles):
        if seconds_cap is not None and perf_counter() - began >= seconds_cap:
            # The production controller also runs under a wall budget
            # (PmoPopulationController.config.wall_seconds), so an arm that cannot
            # finish in time is a real operating condition -- reported as its own
            # outcome, never silently as "no binding".
            capped_at = step
            break
        desired_identity = identity(desired)
        advanced: dict[tuple[str, str], _Prefix] = {}
        enumerated = matched = 0
        for prefix in prefixes:
            count, children = _expand(prefix, desired, desired_identity, step)
            enumerated += count
            matched += len(children)
            for key, child in children:
                advanced.setdefault(key, child)
        enumerated_total += enumerated
        next_descriptors = _operand_descriptors(roles[step + 1]) if step + 1 < len(roles) else []
        if ranking == "hash":
            ranked = sorted(advanced.items(), key=lambda row: _hash_rank_key(plan, step, row[0]))
            live_lookahead = None
        elif ranking == "compatibility":
            scored = []
            for key, child in advanced.items():
                score = _lookahead_matches(
                    child.graph, _created_dict(child), step + 1, next_descriptors
                )
                scored.append((-score, _hash_rank_key(plan, step, key), key, child))
            scored.sort(key=lambda row: (row[0], row[1]))
            ranked = [(key, child) for _, _, key, child in scored]
            live_lookahead = sum(1 for row in scored if -row[0] > 0)
        else:
            raise ValueError(f"unknown ranking: {ranking}")
        steps.append(
            {
                "step": step,
                "prefixes_in": len(prefixes),
                "successors_enumerated": enumerated,
                "role_matches": matched,
                "distinct_children": len(advanced),
                "children_with_live_lookahead": live_lookahead,
                "truncated_away": max(0, len(ranked) - beam_width),
            }
        )
        survivors = [child for _, child in ranked[:beam_width]]
        if not survivors:
            died_at = step
            break
        prefixes = survivors
    elapsed = perf_counter() - began
    complete: list[dict[str, Any]] = []
    if died_at is None and capped_at is None:
        for prefix in prefixes:
            row = _finalize(source, prefix, plan)
            if row is not None:
                complete.append(row)
    return {
        "plan_id": plan["plan_id"],
        "plan_primitive_count": int(plan["primitive_count"]),
        "beam_width": beam_width,
        "ranking": ranking,
        "died_at_step": died_at,
        "capped_at_step": capped_at,
        "depth_reached": len(roles) if died_at is None and capped_at is None
        else (died_at if died_at is not None else capped_at),
        "plan_length": len(roles),
        "successors_enumerated": enumerated_total,
        "seconds": elapsed,
        "complete": complete,
        "bound": bool(complete),
        "steps": steps,
    }


# ---- exhaustive referee: the (a) / (b) / (c) authority ----

OUTCOME_NO_BINDING = "no_compatible_binding"
OUTCOME_EXISTS = "binding_exists"
OUTCOME_UNDECIDED = "budget_exhausted"
# Role-consistent prefixes reached full depth, but none survived dependency-region
# program extraction / executor replay.  This is NOT role incompatibility: the parent
# admits the transformation and the representation refuses it, which is a different
# defect with a different repair.  Reported separately so the two are never conflated.
OUTCOME_PROGRAM_REJECTED = "full_depth_but_program_rejected"


def referee_binding_exists(
    source,
    plan: dict[str, Any],
    *,
    frontier_cap: int = 4096,
    seconds_cap: float = 120.0,
    prefilter: bool = True,
) -> dict[str, Any]:
    """Decide whether ANY role-consistent complete realization exists on this parent.

    Breadth-first over the full role-match tree with the beam removed, deduplicated on
    ``_future_signature``.  A clean failure is a proof of NO_COMPATIBLE_BINDING; hitting
    a cap is reported as BUDGET_EXHAUSTED and never counted as either of the others.
    """

    began = perf_counter()
    prefixes = [
        _Prefix(graph=source, states=(encode_state(source),), actions=(), created=(),
                next_ordinal=0)
    ]
    roles = plan["roles"]
    enumerated_total = 0
    peak_frontier = 1
    capped_at: int | None = None
    died_at: int | None = None
    prefiltered = 0
    for step, desired in enumerate(roles):
        desired_identity = identity(desired)
        descriptors = _operand_descriptors(desired)
        advanced: dict[str, _Prefix] = {}
        for prefix in prefixes:
            if perf_counter() - began >= seconds_cap:
                capped_at = step
                break
            if prefilter and not _lookahead_matches(
                prefix.graph, _created_dict(prefix), step, descriptors
            ):
                # Necessary condition: the binder computes exactly this descriptor for
                # whichever operand it selects, so a prefix carrying no matching atom has
                # no role-consistent child.  Skipping it cannot change the EXISTENCE
                # answer, only the cost of reaching it.
                prefiltered += 1
                continue
            count, children = _expand(prefix, desired, desired_identity, step)
            enumerated_total += count
            for _, child in children:
                advanced.setdefault(_future_signature(child), child)
            if len(advanced) > frontier_cap:
                capped_at = step
                break
        if capped_at is not None:
            return {
                "outcome": OUTCOME_UNDECIDED,
                "capped_at_step": capped_at,
                "peak_frontier": max(peak_frontier, len(advanced)),
                "successors_enumerated": enumerated_total,
                "prefixes_prefiltered": prefiltered,
                "seconds": perf_counter() - began,
                "depth_reached": step,
                "plan_length": len(roles),
                "witness": None,
            }
        peak_frontier = max(peak_frontier, len(advanced))
        if not advanced:
            died_at = step
            break
        prefixes = list(advanced.values())
    if died_at is not None:
        return {
            "outcome": OUTCOME_NO_BINDING,
            "capped_at_step": None,
            "peak_frontier": peak_frontier,
            "successors_enumerated": enumerated_total,
            "prefixes_prefiltered": prefiltered,
            "seconds": perf_counter() - began,
            "depth_reached": died_at,
            "plan_length": len(roles),
            "witness": None,
        }
    witness = None
    for prefix in prefixes:
        witness = _finalize(source, prefix, plan)
        if witness is not None:
            break
    return {
        "outcome": OUTCOME_EXISTS if witness is not None else OUTCOME_PROGRAM_REJECTED,
        "full_depth_prefixes": len(prefixes),
        "capped_at_step": None,
        "peak_frontier": peak_frontier,
        "successors_enumerated": enumerated_total,
        "prefixes_prefiltered": prefiltered,
        "seconds": perf_counter() - began,
        "depth_reached": len(roles),
        "plan_length": len(roles),
        "witness": witness,
        "note": None
        if witness is not None
        else "role-consistent prefixes reached full depth but none survived program "
        "extraction / executor replay; note the referee dedupes on _future_signature, "
        "so a merged-away sibling with a different action history is not re-tried here",
    }


def verify_equivalence(source, plan: dict[str, Any], *, beam_width: int) -> dict[str, Any]:
    """Prove the instrumented binder reproduces the PINNED function, not a copy of it."""

    live = bind_joint_plan(source, plan, beam_width=beam_width)
    probe = bind_instrumented(source, plan, beam_width=beam_width, ranking="hash")
    live_keys = sorted(row["endpoint_key"] for row in live)
    probe_keys = sorted(row["endpoint_key"] for row in probe["complete"])
    return {
        "beam_width": beam_width,
        "live_bindings": len(live),
        "probe_bindings": len(probe["complete"]),
        "endpoint_sets_identical": live_keys == probe_keys,
    }


__all__ = [
    "ATTRIBUTION_VIEWS",
    "OUTCOME_EXISTS",
    "OUTCOME_NO_BINDING",
    "OUTCOME_PROGRAM_REJECTED",
    "OUTCOME_UNDECIDED",
    "PRODUCTION_BEAM_WIDTH",
    "ROLE_DESCRIPTOR_FIELDS",
    "SCHEMA",
    "bind_instrumented",
    "first_step_feasible_under",
    "load_plan_source_witnesses",
    "load_plans",
    "load_production_parents",
    "load_teacher_root_parents",
    "plan_specification_census",
    "referee_binding_exists",
    "retained_fraction",
    "static_first_step_feasible",
    "verify_equivalence",
]
