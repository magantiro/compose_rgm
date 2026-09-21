"""Constraint-propagating realizer for joint role plans (production PMO binder).

WHAT THIS IS
------------
Given a parent molecule and one joint role plan -- a sequence of ~16-31 typed roles,
each naming an executor rule, its parameters, and a per-operand descriptor -- this
module finds a legal executable program on that parent that realizes EVERY role, in
order, with the plan's declared dataflow intact.  ``bind_realized_plan`` is the
production entry point and returns rows shaped exactly like
``pmo_joint_dependency_jump.bind_joint_plan``, so the controller consumes them
unchanged.

It replaces a width-4 beam.  The beam discarded prefixes for being ranked low and
could not recover; this search discards nothing for ranking -- it orders children by
one-step operand availability and keeps all of them reachable until an explicit budget
is spent.  A budget hit is reported as its own outcome and never as incompatibility.

WHY, MEASURED
-------------
``diagnostics/pmo_realization_repair_v1.json`` (36 matched teacher-scale (plan, parent)
pairs, three arms, one interpreter, identical budgets):

    arm                                completed  at_scale  in_band  prim_med  retained_med
    v1_exact (this search, NO relax)       5          5        3       31.0      0.630
    R1 (drop environment + lag)            1          1        0       31.0      0.963
    R2 (R1 + capacity)                     3          3        3       28.0      0.852

The repair is the SEARCH ARCHITECTURE, not semantic relaxation: relaxation measured
WORSE at teacher scale.  Production therefore runs the PINNED predicate
``SPEC_V1_EXACT`` -- byte-for-byte the role-identity equality
``pmo_joint_dependency_jump.py:344`` applies -- and the relaxed specifications below
are retained only so that result stays reproducible.

The defect being repaired is visible in ``retained_fraction``.  The seven prior arms
produced ZERO programs at teacher scale and retained 1.000 on every program: they were
purely additive, removing nothing from the parent.  This search produced three programs
simultaneously at teacher scale (median 31 primitives) and inside the teacher retained
band 0.52-0.96 (median 0.630).  A binding-rate gain WITHOUT that shape is not this
repair; see the falsifier pinned in ``configs/pmo_population_controller_v1.json``.

ROLE SPECIFICATIONS: WHAT MAY BE RELAXED
----------------------------------------
The role is split into three parts with different authority:

  SEMANTIC CORE -- never relaxed.  ``executor_rule``, ``model_family`` and the full
    ``parameters`` dict (inserted element/charge/H-count, attachment bond classes,
    target restate class, closure bond class, ring restate bond classes).  This is WHAT
    the operation does.  Relaxing any of it would realize a different transformation.
    The operand's own ``atom_type`` and ``formal_charge`` are held here too: attaching
    to a carbon is not the same operation as attaching to an oxygen, and charge state
    is chemically load-bearing (charge is preserved, never optimized, in this repo).

  DATAFLOW -- never relaxed.  ``origin`` (preexisting vs route_created) and, for a
    created operand, ``created_ordinal``.  This is the edge that makes the plan a
    PROGRAM rather than N unrelated edits: "act on the atom that step j created".  An
    operand declared preexisting must bind to a parent atom, never to the realizer's
    own product.

  INCIDENTAL / CAPACITY -- relaxable by named component, one at a time, and NOT used in
    production: ``neighbor_element_histogram``, ``creation_lag``, ``degree``,
    ``implicit_hydrogens``, ``bond_class_histogram``.

DECLARED COMPLETION CONDITIONS
------------------------------
A realization must satisfy the plan's OWN declared conditions, which the beam never
checked: ``component_count``, ``created_dependency_count``, exact replay, complete
representation support, and an endpoint that differs from the source.  An executable
prefix that misses them is a different, unspecified transformation and is never
reported as a smaller version of the requested one.

Forward constraint propagation over the remaining plan supplies the necessary
conditions that make the search finish: element/charge budget for future deletions net
of future insertions, created-handle availability, free-slot capacity for insertions,
and a one-step operand-availability condition.

FOUR OUTCOMES, never collapsed:
  ``completed_realization``     a complete program meeting every declared condition.
  ``proven_incompatible``       the search space was EXHAUSTED with no cap hit.
  ``invalid_plan``              the plan's own declared structure is unrealizable
                                independent of any parent.
  ``search_budget_exhausted``   a node or wall cap was hit; nothing is proven.

All chemistry -- successor enumeration, role supervision, program extraction, executor
replay -- is the PINNED implementation.  This module changes the ACCEPTANCE PREDICATE
and the SEARCH, never the rewrite system.  Zero oracle calls; binding is pure graph
work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision, atom_role
from compose_v4.control.pmo_joint_dependency_jump import enumerate_role_successors
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

SCHEMA = "pmo_realization_repair_v1"

# ---- four outcomes ----

OUTCOME_COMPLETED = "completed_realization"
OUTCOME_INCOMPATIBLE = "proven_incompatible"
OUTCOME_INVALID_PLAN = "invalid_plan"
OUTCOME_EXHAUSTED = "search_budget_exhausted"

OUTCOMES = (OUTCOME_COMPLETED, OUTCOME_INCOMPATIBLE, OUTCOME_INVALID_PLAN, OUTCOME_EXHAUSTED)


# ---- descriptor specifications ----

# Held in every specification: the operand identity the edit semantics depend on, and
# the dataflow edge that makes the plan a program.
SEMANTIC_OPERAND_FIELDS = ("atom_type", "formal_charge")
DATAFLOW_OPERAND_FIELDS = ("origin", "created_ordinal")

# Relaxable components, named so a result can be attributed to one of them.
RELAXABLE_COMPONENTS = (
    "neighbor_element_histogram",
    "creation_lag",
    "implicit_hydrogens",
    "degree",
    "bond_class_histogram",
)


@dataclass(frozen=True)
class RoleSpecification:
    """Which descriptor components a role match compares."""

    name: str
    relaxed: frozenset[str]
    # v1 compares the whole role object by hash; that path is kept verbatim so the
    # control arm is provably the pinned predicate rather than a transcription of it.
    exact_role_identity: bool = False

    @property
    def compared_fields(self) -> tuple[str, ...]:
        keep = [
            *SEMANTIC_OPERAND_FIELDS,
            *DATAFLOW_OPERAND_FIELDS,
            "implicit_hydrogens",
            "degree",
            "bond_class_histogram",
            "neighbor_element_histogram",
            "creation_lag",
        ]
        return tuple(name for name in keep if name not in self.relaxed)


SPEC_V1_EXACT = RoleSpecification("v1_exact", frozenset(), exact_role_identity=True)
# R1: drop the teacher's local chemistry and the teacher's schedule.  Everything a
# capacity argument would touch (degree, H count, bond classes) is still held, so R1
# isolates the two components the located defect implicates.
SPEC_R1 = RoleSpecification(
    "R1_drop_environment_and_lag",
    frozenset({"neighbor_element_histogram", "creation_lag"}),
)
# R2: R1 plus the capacity quantities, deferring to the executor's own legality.
SPEC_R2 = RoleSpecification(
    "R2_plus_capacity",
    frozenset(
        {
            "neighbor_element_histogram",
            "creation_lag",
            "implicit_hydrogens",
            "degree",
            "bond_class_histogram",
        }
    ),
)

SPECIFICATIONS = {spec.name: spec for spec in (SPEC_V1_EXACT, SPEC_R1, SPEC_R2)}


def _descriptor_key(descriptor: dict[str, Any], spec: RoleSpecification) -> tuple:
    return tuple(
        json.dumps(descriptor.get(name), sort_keys=True) for name in spec.compared_fields
    )


def role_match(
    desired: dict[str, Any],
    observed: dict[str, Any],
    spec: RoleSpecification,
    *,
    desired_identity: str | None = None,
) -> bool:
    """Does ``observed`` satisfy the requirement ``desired`` expresses, under ``spec``?

    Under ``SPEC_V1_EXACT`` this is byte-for-byte the pinned predicate
    (``pmo_joint_dependency_jump.py:344``): whole-role hash equality.
    """

    if spec.exact_role_identity:
        return identity(observed) == (desired_identity or identity(desired))
    # Semantic core: never relaxed.
    if str(observed["executor_rule"]) != str(desired["executor_rule"]):
        return False
    if str(observed["model_family"]) != str(desired["model_family"]):
        return False
    if observed["parameters"] != desired["parameters"]:
        return False
    want = desired.get("operands", ())
    have = observed.get("operands", ())
    if len(want) != len(have):
        return False
    for wanted, found in zip(want, have, strict=True):
        if str(wanted["role"]) != str(found["role"]):
            return False
        if _descriptor_key(wanted["descriptor"], spec) != _descriptor_key(
            found["descriptor"], spec
        ):
            return False
    return True


# ---- plan-level static validity (outcome: invalid_plan) ----


def plan_validity(plan: dict[str, Any]) -> dict[str, Any]:
    """Is the plan's own declared structure realizable, independent of any parent?

    These are defects of the LATENT, not of the parent, and must never be reported as
    incompatibility: no parent could repair them.
    """

    roles = list(plan["roles"])
    reasons: list[str] = []
    if not roles:
        reasons.append("empty_role_sequence")
    if len(roles) > MAXIMUM_PRIMITIVES:
        reasons.append("declared_primitive_count_exceeds_runtime_maximum")
    if int(plan.get("component_count", 1)) > MAXIMUM_COMPONENTS:
        reasons.append("declared_component_count_exceeds_runtime_maximum")
    # Every created operand must reference an ordinal some EARLIER role emits.
    emitted: set[int] = set()
    for step, role in enumerate(roles):
        for row in role.get("operands", ()):
            descriptor = row["descriptor"]
            if descriptor.get("origin") != "route_created":
                continue
            ordinal = descriptor.get("created_ordinal")
            if ordinal is None:
                reasons.append(f"step_{step}_created_operand_without_ordinal")
            elif int(ordinal) not in emitted:
                reasons.append(f"step_{step}_references_unemitted_ordinal_{ordinal}")
        output = role.get("created_output_ordinal")
        if output is not None:
            emitted.add(int(output))
    # The declared dependency edge count must match the roles' own dependency lists.
    declared = int(plan.get("created_dependency_count", 0))
    counted = sum(len(role.get("created_handle_dependencies", ())) for role in roles)
    if declared != counted:
        reasons.append("declared_created_dependency_count_disagrees_with_roles")
    return {"valid": not reasons, "reasons": sorted(set(reasons))}


# ---- prefix ----


@dataclass(frozen=True)
class _Prefix:
    graph: Any
    actions: tuple[dict[str, Any], ...]
    created: tuple[tuple[int, int, int], ...]
    next_ordinal: int

    def created_dict(self) -> dict[int, tuple[int, int]]:
        return {slot: (ordinal, step) for slot, ordinal, step in self.created}


def _graph_signature(graph) -> bytes:
    """Exact slot-level graph identity.

    Compared by encoded slots rather than by a canonical key: the created map is keyed
    by slot address, so two isomorphic graphs with different layouts are NOT
    interchangeable for the continuation.
    """

    return (
        np.ascontiguousarray(graph.atom_types).tobytes()
        + b"|"
        + np.ascontiguousarray(graph.formal_charges).tobytes()
        + b"|"
        + np.ascontiguousarray(graph.implicit_h_counts).tobytes()
        + b"|"
        + np.ascontiguousarray(graph.bonds).tobytes()
    )


def _future_signature(prefix: _Prefix) -> bytes:
    """Everything the continuation can depend on: graph, created map, next ordinal."""

    return (
        _graph_signature(prefix.graph)
        + b"#"
        + repr(prefix.created).encode()
        + b"#"
        + str(prefix.next_ordinal).encode()
    )


# ---- constraint propagation over the remaining plan ----


@dataclass
class _PlanDemand:
    """Static, parent-independent demand profile of the plan suffix at each step."""

    # remaining_deletes[step][(atom_type, formal_charge)] -> count of atom_delete roles
    # at or after ``step`` whose operand carries that identity.
    remaining_deletes: list[dict[tuple[int, int], int]] = field(default_factory=list)
    # remaining_inserts[step][(atom_type, formal_charge)] -> count of atom_insert roles
    # at or after ``step`` that CREATE an atom of that identity.
    remaining_inserts: list[dict[tuple[int, int], int]] = field(default_factory=list)
    # remaining_insert_total[step] -> number of insertions still to perform (slot need).
    remaining_insert_total: list[int] = field(default_factory=list)
    # remaining_preexisting_deletes[step] -> element-agnostic count of deletions at or
    # after ``step`` that consume a parent atom.
    remaining_preexisting_deletes: list[int] = field(default_factory=list)
    # remaining_restates[step] -> atom_restate_semantic roles at or after ``step``.
    # While any remain, an atom's element can still change, so the per-identity
    # deletion budget is not a sound necessary condition.
    remaining_restates: list[int] = field(default_factory=list)
    # required_ordinals[step] -> created ordinals that some role at/after step needs.
    required_ordinals: list[frozenset[int]] = field(default_factory=list)


def plan_demand(plan: dict[str, Any]) -> _PlanDemand:
    roles = list(plan["roles"])
    n = len(roles)
    demand = _PlanDemand(
        remaining_deletes=[{} for _ in range(n + 1)],
        remaining_inserts=[{} for _ in range(n + 1)],
        remaining_insert_total=[0] * (n + 1),
        remaining_preexisting_deletes=[0] * (n + 1),
        remaining_restates=[0] * (n + 1),
        required_ordinals=[frozenset() for _ in range(n + 1)],
    )
    for step in range(n - 1, -1, -1):
        role = roles[step]
        deletes = dict(demand.remaining_deletes[step + 1])
        inserts = dict(demand.remaining_inserts[step + 1])
        total = demand.remaining_insert_total[step + 1]
        preexisting_deletes = demand.remaining_preexisting_deletes[step + 1]
        restates = demand.remaining_restates[step + 1]
        ordinals = set(demand.required_ordinals[step + 1])
        rule = str(role["executor_rule"])
        if rule == "atom_restate_semantic":
            restates += 1
        if rule == "atom_delete":
            descriptor = role["operands"][0]["descriptor"]
            # Only a PREEXISTING deletion consumes a parent atom; a deletion of a
            # route-created atom consumes the realizer's own product instead.
            if descriptor.get("origin") == "preexisting":
                key = (int(descriptor["atom_type"]), int(descriptor["formal_charge"]))
                deletes[key] = deletes.get(key, 0) + 1
                preexisting_deletes += 1
        elif rule == "atom_insert":
            key = (
                int(role["parameters"]["atom_type"]),
                int(role["parameters"]["formal_charge"]),
            )
            inserts[key] = inserts.get(key, 0) + 1
            total += 1
        for row in role.get("operands", ()):
            descriptor = row["descriptor"]
            if descriptor.get("origin") == "route_created":
                ordinal = descriptor.get("created_ordinal")
                if ordinal is not None:
                    ordinals.add(int(ordinal))
        demand.remaining_deletes[step] = deletes
        demand.remaining_inserts[step] = inserts
        demand.remaining_insert_total[step] = total
        demand.remaining_preexisting_deletes[step] = preexisting_deletes
        demand.remaining_restates[step] = restates
        demand.required_ordinals[step] = frozenset(ordinals)
    return demand


def _atom_identity_census(graph, created: dict[int, tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Count PREEXISTING active atoms by (atom_type, formal_charge)."""

    census: dict[tuple[int, int], int] = {}
    types = graph.atom_types
    charges = graph.formal_charges
    for slot in np.flatnonzero(is_element(types)):
        slot = int(slot)
        if slot in created:
            continue
        key = (int(types[slot]), int(charges[slot]))
        census[key] = census.get(key, 0) + 1
    return census


def _free_slots(graph) -> int:
    return int(graph.n_atoms) - int(np.count_nonzero(is_element(graph.atom_types)))


def propagate(
    prefix: _Prefix,
    step: int,
    plan: dict[str, Any],
    demand: _PlanDemand,
    spec: RoleSpecification,
) -> str | None:
    """Forward-check the plan suffix against the current graph.

    Returns a refusal reason when the prefix is PROVABLY dead, else ``None``.  Every
    check here is a necessary condition, so pruning on it cannot change whether a
    complete realization exists -- only the cost of finding out.
    """

    roles = plan["roles"]
    if step >= len(roles):
        return None
    created = prefix.created_dict()
    # (1) resource limit: enough free slots for every remaining insertion.
    #     Deletions free slots, so only the running maximum matters; the cheap
    #     necessary condition is that total remaining insertions cannot exceed free
    #     slots plus the atoms still to be deleted.
    remaining_deletes_any = sum(
        1
        for later in range(step, len(roles))
        if str(roles[later]["executor_rule"]) == "atom_delete"
    )
    if demand.remaining_insert_total[step] > _free_slots(prefix.graph) + remaining_deletes_any:
        return "insufficient_slot_capacity"
    # (2) element/charge budget: preexisting atoms the suffix still has to delete must
    #     exist.  Nothing ever creates a PREEXISTING atom -- an insertion produces a
    #     route_created one, which a preexisting-origin deletion may not consume -- so
    #     the element-agnostic count is an unconditional necessary condition.
    #
    #     The per-(element, charge) refinement is NOT unconditionally sound:
    #     ``atom_restate_semantic`` changes an atom's element in place, so an atom that
    #     is a carbon now may be the nitrogen a later deletion wants.  It is therefore
    #     applied only when no restatement remains in the suffix.  (This is the prune
    #     that first reported a teacher-witness pair -- one whose own route is a witness
    #     that a realization exists -- as proven_incompatible.)
    census = _atom_identity_census(prefix.graph, created)
    if sum(census.values()) < demand.remaining_preexisting_deletes[step]:
        return "insufficient_preexisting_atoms_for_remaining_deletions"
    if not demand.remaining_restates[step]:
        for key, needed in demand.remaining_deletes[step].items():
            if census.get(key, 0) < needed:
                return "insufficient_preexisting_atoms_of_a_required_identity"
    # (3) created-handle availability: an ordinal some later role requires must either
    #     already be live or still be emitted by a later insertion.
    live = {ordinal for _, (ordinal, _) in created.items()}
    future_outputs = {
        int(roles[later]["created_output_ordinal"])
        for later in range(step, len(roles))
        if roles[later].get("created_output_ordinal") is not None
    }
    if not demand.required_ordinals[step] <= (live | future_outputs):
        return "required_created_handle_is_unreachable"
    # (4) one-step operand availability: the binder computes exactly this descriptor for
    #     whichever operand it selects, so a prefix carrying no matching atom has no
    #     role-consistent child at this step.
    descriptors = [row["descriptor"] for row in roles[step].get("operands", ())]
    if descriptors:
        available = []
        for slot in np.flatnonzero(is_element(prefix.graph.atom_types)):
            try:
                available.append(
                    _descriptor_key(atom_role(prefix.graph, int(slot), created, step), spec)
                )
            except ValueError:
                continue
        pool = set(available)
        for wanted in descriptors:
            if _descriptor_key(wanted, spec) not in pool:
                return "no_atom_carries_a_required_operand_descriptor"
    return None


def operand_availability(
    graph, created: dict[int, tuple[int, int]], step: int, role: dict[str, Any],
    spec: RoleSpecification,
) -> int:
    """Minimum operand-match count over the role's operands (a lookahead score)."""

    descriptors = [row["descriptor"] for row in role.get("operands", ())]
    if not descriptors:
        return 1
    available = []
    for slot in np.flatnonzero(is_element(graph.atom_types)):
        try:
            available.append(_descriptor_key(atom_role(graph, int(slot), created, step), spec))
        except ValueError:
            continue
    return min(sum(1 for found in available if found == _descriptor_key(w, spec))
               for w in descriptors)


# ---- expansion (pinned chemistry) ----


def expand(
    prefix: _Prefix,
    desired: dict[str, Any],
    step: int,
    spec: RoleSpecification,
    *,
    desired_identity: str | None = None,
) -> tuple[int, list[_Prefix]]:
    """Every role-consistent child of one prefix, using the PINNED enumerator.

    ``cycle_close`` is enumerated with ``created=None`` under a relaxed specification.
    That is still the pinned enumerator: with ``created`` supplied it applies an EXACT
    descriptor filter internally (pmo_joint_dependency_jump.py:251), which would
    re-impose the very constraint under test.  With ``created=None`` it returns the full
    executor-verified closure fiber, a strict SUPERSET of the filtered set, so the
    relaxation remains purely a widening of the acceptance predicate.
    """

    prefix_created = prefix.created_dict()
    rule = str(desired["executor_rule"])
    enumerator_created = (
        None if (rule == "cycle_close" and not spec.exact_role_identity) else prefix_created
    )
    children: list[_Prefix] = []
    enumerated = 0
    for candidate in enumerate_role_successors(
        prefix.graph, desired, created=enumerator_created, step=step
    ):
        enumerated += 1
        created = dict(prefix_created)
        observed, next_ordinal = action_role_supervision(
            prefix.graph, candidate.action_record, created, step, prefix.next_ordinal
        )
        if not role_match(desired, observed, spec, desired_identity=desired_identity):
            continue
        children.append(
            _Prefix(
                graph=candidate.successor,
                actions=(*prefix.actions, candidate.action_record),
                created=tuple(
                    sorted((slot, ordinal, made) for slot, (ordinal, made) in created.items())
                ),
                next_ordinal=next_ordinal,
            )
        )
    return enumerated, children




# ---- declared completion conditions ----


def retained_fraction(source, endpoint) -> float:
    """Fraction of the parent's heavy atoms surviving, by the controller's own rule."""

    from compose_v4.control.pmo_population_controller import _retained_fraction

    return float(_retained_fraction(source, endpoint))


def finalize(
    source,
    prefix: _Prefix,
    plan: dict[str, Any],
    *,
    enforce_declared: bool = True,
) -> dict[str, Any] | None:
    """Pinned program extraction + executor replay + the plan's DECLARED conditions.

    A realization must satisfy the plan's own completion conditions.  An executable
    prefix that does not is a different, unspecified transformation and is never
    reported as a smaller version of the requested one -- which is why this function
    refuses anything whose length is not the plan's length.

    An accepted row carries ``actions``, ``states`` and ``endpoint_state`` so that it is
    consumable by the controller without re-execution.  ``states`` is the executor
    receipt's own state list -- the SAME ``encode_state`` encoding ``bind_joint_plan``
    emits, and the same list ``execute_program`` will produce again on replay -- so the
    controller's ``trace["states"] == bound["states"]`` assertion compares like with
    like.  ``states[0]`` is the source, so the list is one longer than ``actions``.
    """

    if len(prefix.actions) != len(plan["roles"]):
        raise ValueError("only complete bindings may be finalized")
    endpoint, receipt = execute_program(source, list(prefix.actions))
    states = list(receipt["states"])
    program = dependency_region_program(
        states,
        list(prefix.actions),
        config=DependencyRegionConfig(
            runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
            runtime_maximum_components=MAXIMUM_COMPONENTS,
        ),
    )
    rejected: list[str] = []
    if not program["exact_replay"]:
        rejected.append("exact_replay_failed")
    if not program["complete_representation_supported"]:
        rejected.append("complete_representation_unsupported")
    endpoint_key = canonical_state_key(endpoint)
    if endpoint_key == canonical_state_key(source):
        rejected.append("endpoint_equals_source")
    declared_components = int(plan.get("component_count", -1))
    realized_components = int(program["component_count"])
    declared_edges = int(plan.get("created_dependency_count", -1))
    realized_edges = len(program["created_dependency_edges"])
    if enforce_declared:
        if declared_components >= 0 and realized_components != declared_components:
            rejected.append("declared_component_count_not_realized")
        if declared_edges >= 0 and realized_edges != declared_edges:
            rejected.append("declared_created_dependency_count_not_realized")
    if rejected:
        return {"accepted": False, "rejected_because": sorted(set(rejected))}
    return {
        "accepted": True,
        "plan_id": plan["plan_id"],
        "endpoint_key": endpoint_key,
        "primitive_count": len(prefix.actions),
        "plan_primitive_count": int(plan["primitive_count"]),
        "component_count": realized_components,
        "declared_component_count": declared_components,
        "created_dependency_edges": realized_edges,
        "declared_created_dependency_count": declared_edges,
        "retained_fraction": retained_fraction(source, endpoint),
        "delta_heavy_atoms": int(endpoint.n_real_atoms) - int(source.n_real_atoms),
        "actions": list(prefix.actions),
        "states": states,
        "endpoint_state": states[-1],
        "exact_replay": True,
    }


# ---- the realizer ----


def realize(
    source,
    plan: dict[str, Any],
    spec: RoleSpecification,
    *,
    node_budget: int = 40_000,
    seconds_cap: float = 60.0,
    enforce_declared: bool = True,
    collect_all: bool = False,
    max_realizations: int | None = None,
) -> dict[str, Any]:
    """Depth-first, constraint-propagating, COMPLETE search for a realization.

    Depth-first rather than a frontier: the question is EXISTENCE, so a witness should
    be reached in O(depth) memory, and a dead-signature memo makes the search complete
    rather than truncated.  This is explicitly NOT a beam -- nothing is discarded for
    being ranked low; children are ORDERED by operand availability at the next step and
    all of them remain reachable until the budget is spent.  Hitting the budget is
    reported as its own outcome and never as incompatibility.

    ``collect_all`` keeps searching after the first realization instead of returning it.
    ``max_realizations`` then stops the search once that many have been accepted, which
    is how the production binder asks for SEVERAL realizations without paying for an
    exhaustive enumeration: the controller ranks the returned rows by retained fraction,
    so one row would silently collapse that ranking, and an uncapped collection would
    spend the whole node budget on a plan whose first realization already arrived.
    ``max_realizations`` is ignored unless ``collect_all`` is set.
    """

    if max_realizations is not None and max_realizations < 1:
        raise ValueError("max_realizations must be positive")

    began = perf_counter()
    validity = plan_validity(plan)
    if not validity["valid"]:
        return {
            "outcome": OUTCOME_INVALID_PLAN,
            "plan_id": plan["plan_id"],
            "plan_primitive_count": int(plan["primitive_count"]),
            "specification": spec.name,
            "invalid_reasons": validity["reasons"],
            "nodes_expanded": 0,
            "successors_enumerated": 0,
            "max_depth_reached": 0,
            "seconds": perf_counter() - began,
            "realizations": [],
            "completion_rejections": [],
        }
    roles = list(plan["roles"])
    demand = plan_demand(plan)
    identities = [identity(role) for role in roles] if spec.exact_role_identity else [None] * len(roles)

    dead: set[bytes] = set()
    realizations: list[dict[str, Any]] = []
    completion_rejections: list[str] = []
    stats = {
        "nodes_expanded": 0,
        "successors_enumerated": 0,
        "max_depth_reached": 0,
        "pruned_by_propagation": 0,
        "pruned_by_memo": 0,
        "full_depth_prefixes": 0,
    }
    exhausted = {"hit": False, "reason": None}

    def over_budget() -> bool:
        if stats["nodes_expanded"] >= node_budget:
            exhausted["hit"] = True
            exhausted["reason"] = "node_budget"
            return True
        if perf_counter() - began >= seconds_cap:
            exhausted["hit"] = True
            exhausted["reason"] = "seconds_cap"
            return True
        return False

    def search(prefix: _Prefix, step: int) -> bool:
        """Return True when a realization was found and the caller may stop."""

        stats["max_depth_reached"] = max(stats["max_depth_reached"], step)
        if step == len(roles):
            stats["full_depth_prefixes"] += 1
            row = finalize(source, prefix, plan, enforce_declared=enforce_declared)
            if row is not None and row.get("accepted"):
                realizations.append(row)
                if not collect_all:
                    return True
                return (
                    max_realizations is not None
                    and len(realizations) >= max_realizations
                )
            if row is not None:
                completion_rejections.extend(row.get("rejected_because", ()))
            return False
        if over_budget():
            return False
        signature = _future_signature(prefix) + b"@" + str(step).encode()
        if signature in dead:
            stats["pruned_by_memo"] += 1
            return False
        refusal = propagate(prefix, step, plan, demand, spec)
        if refusal is not None:
            stats["pruned_by_propagation"] += 1
            dead.add(signature)
            return False
        stats["nodes_expanded"] += 1
        enumerated, children = expand(
            prefix, roles[step], step, spec, desired_identity=identities[step]
        )
        stats["successors_enumerated"] += enumerated
        if not children:
            dead.add(signature)
            return False
        # Order children by how well they support the NEXT role.  Ordering only; no
        # child is removed.
        if step + 1 < len(roles):
            scored = []
            for child in children:
                score = operand_availability(
                    child.graph, child.created_dict(), step + 1, roles[step + 1], spec
                )
                scored.append((-score, _future_signature(child), child))
            scored.sort(key=lambda row: (row[0], row[1]))
            ordered = [child for _, _, child in scored]
        else:
            ordered = sorted(children, key=lambda child: _future_signature(child))
        found_any = False
        for child in ordered:
            if over_budget():
                return found_any
            if search(child, step + 1):
                return True
            found_any = found_any or bool(realizations)
        if not exhausted["hit"]:
            dead.add(signature)
        return False

    root = _Prefix(graph=source, actions=(), created=(), next_ordinal=0)
    search(root, 0)
    seconds = perf_counter() - began
    if realizations:
        outcome = OUTCOME_COMPLETED
    elif exhausted["hit"]:
        outcome = OUTCOME_EXHAUSTED
    else:
        outcome = OUTCOME_INCOMPATIBLE
    return {
        "outcome": outcome,
        "plan_id": plan["plan_id"],
        "plan_primitive_count": int(plan["primitive_count"]),
        "specification": spec.name,
        "invalid_reasons": [],
        "budget_reason": exhausted["reason"],
        "seconds": seconds,
        "realizations": realizations,
        # A full-depth prefix refused by the DECLARED completion conditions is not
        # incompatibility; it is recorded so the two are never conflated.
        "completion_rejections": sorted(set(completion_rejections)),
        "full_depth_but_completion_rejected": bool(
            stats["full_depth_prefixes"] and not realizations
        ),
        **stats,
    }


# ---- production binder adapter ----

# The predicate production runs.  Held at the PINNED v1 role identity: the matched
# 36-pair comparison measured both relaxations WORSE at teacher scale, so relaxing here
# would trade the property being repaired for a binding-rate number.
PRODUCTION_SPECIFICATION = SPEC_V1_EXACT

# Chosen from the measured cost/yield curve; see the module docstring and
# ``diagnostics/pmo_realization_binder_cost_v1.json``.  Every realization the matched
# comparison found needed 51-55 expanded nodes, so a budget below ~55 would discard the
# entire measured benefit while still paying most of the cost.
PRODUCTION_NODE_BUDGET = 64

# Per-attempt wall cap.  The jump lane's own budget is ``config.wall_seconds``; this
# bounds ONE attempt so a single hopeless plan cannot consume the lane.  The caller
# passes the lane's remaining time as well, and the smaller of the two binds.
#
# MEASURED, single process, all 48 matched pairs at this exact configuration
# (``diagnostics/pmo_realization_binder_cost_v1.json``): a 12 s cap LOSES 3 of the 6
# realizations the comparison found, because their first realization arrives at node
# 51-55 of 64.  A 20 s cap keeps 6 of 6 and costs 6.88 s per attempt on average, since
# 28 of 48 attempts are refused by constraint propagation in under a second.  The
# separation is what makes the cut safe: every SUCCESS finishes within 16.92 s while
# the unproductive searches run 18-64 s.
#
# The cap is SOFT by about 4% (measured max 20.77 s): ``over_budget`` is checked
# between expansions, so the executor replay and program extraction of an in-flight
# node complete past it.
PRODUCTION_SECONDS_CAP = 20.0

# The controller ranks bindings by retained fraction against a random retention target
# and takes the closest.  One row would make that target inert, so several are
# requested; the cap stops an easy plan from spending its whole node budget enumerating
# realizations the ranking will not use.  Measured at this budget: all 6 productive
# pairs return more than one row and 5 of 6 return more than one DISTINCT endpoint, so
# the ranking has something to rank -- matching the width-4 beam's own maximum of 4.
PRODUCTION_MAX_REALIZATIONS = 4

# Keys a binding row must carry for the controller to consume it unchanged.
BINDING_ROW_FIELDS = (
    "plan_id",
    "endpoint_key",
    "endpoint_state",
    "primitive_count",
    "component_count",
    "created_dependency_edges",
    "actions",
    "states",
    "exact_replay",
)


def realize_plan_binding(
    source,
    plan: dict[str, Any],
    *,
    specification: RoleSpecification = PRODUCTION_SPECIFICATION,
    node_budget: int = PRODUCTION_NODE_BUDGET,
    seconds_cap: float = PRODUCTION_SECONDS_CAP,
    max_realizations: int = PRODUCTION_MAX_REALIZATIONS,
) -> dict[str, Any]:
    """``realize`` plus binding rows, keeping the four outcomes intact.

    Returned under ``"bindings"`` are rows shaped like ``bind_joint_plan``'s; the search
    ``outcome`` is preserved alongside them so a caller can distinguish a plan PROVEN
    incompatible with this parent from one whose search merely ran out of budget.
    Collapsing those two into one "no binding" string is the reporting defect this
    module exists to stop repeating.
    """

    result = realize(
        source,
        plan,
        specification,
        node_budget=node_budget,
        seconds_cap=seconds_cap,
        collect_all=True,
        max_realizations=max_realizations,
    )
    rows = [_binding_row(row) for row in result["realizations"]]
    # Total, content-addressed order: the controller breaks ranking ties on
    # ``endpoint_key`` alone, so two realizations reaching the same endpoint by
    # different action sequences would otherwise be separated by search order.
    rows.sort(key=lambda row: (row["endpoint_key"], identity(row["actions"])))
    result["bindings"] = rows
    return result


def _binding_row(row: dict[str, Any]) -> dict[str, Any]:
    """One accepted realization as a ``bind_joint_plan``-shaped row.

    A SUPERSET: the realizer's own evidence (retained fraction, declared-vs-realized
    component and dependency counts, heavy-atom delta) rides along so a run artifact can
    be read without re-deriving it.  ``accepted`` is dropped -- it is finalize's internal
    discriminator, and a production row that carries it invites a caller to branch on a
    field that is True by construction.
    """

    out = {key: value for key, value in row.items() if key != "accepted"}
    missing = [key for key in BINDING_ROW_FIELDS if key not in out]
    if missing:
        raise RuntimeError(f"realization row is missing binder fields: {sorted(missing)}")
    if len(out["states"]) != len(out["actions"]) + 1:
        raise RuntimeError("binding states must be one longer than actions")
    if out["states"][-1] != out["endpoint_state"]:
        raise RuntimeError("endpoint_state must be the last recorded state")
    return out


def bind_realized_plan(
    source,
    plan: dict[str, Any],
    *,
    specification: RoleSpecification = PRODUCTION_SPECIFICATION,
    node_budget: int = PRODUCTION_NODE_BUDGET,
    seconds_cap: float = PRODUCTION_SECONDS_CAP,
    max_realizations: int = PRODUCTION_MAX_REALIZATIONS,
) -> list[dict[str, Any]]:
    """Drop-in replacement for ``bind_joint_plan`` backed by the complete search.

    Returns the same row shape, so the controller's retained-fraction ranking, program
    extraction and replay assertion are untouched.  An empty list means "no realization
    found under this budget" and, exactly as with the beam, is the caller's signal that
    the plan did not bind on this parent.
    """

    return realize_plan_binding(
        source,
        plan,
        specification=specification,
        node_budget=node_budget,
        seconds_cap=seconds_cap,
        max_realizations=max_realizations,
    )["bindings"]


def static_first_step_feasible(source, plan: dict[str, Any], spec: RoleSpecification) -> bool:
    """Zero-search depth-0 certificate under ``spec``.

    If no active atom of the parent carries some operand descriptor of role 0 under the
    comparison ``spec`` makes, no search of any shape can realize the plan.
    """

    roles = plan["roles"]
    if not roles:
        return False
    return operand_availability(source, {}, 0, roles[0], spec) > 0


# ---- is the teacher's own action inside the runtime fiber? ----


def teacher_step_gaps(
    roles: list[dict[str, Any]],
    states: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Steps of a teacher route whose own action the RUNTIME cannot re-offer.

    A plan bound back onto the source that generated it is treated elsewhere as a pair
    where a complete role-consistent realization exists BY CONSTRUCTION.  That premise
    holds only if every teacher action is re-enumerable by ``enumerate_role_successors``
    at the state it was taken from.  It is not guaranteed: the runtime fiber applies
    validity gates (connectivity, charge policy, aromaticity) that the corpus generator
    did not necessarily apply, so a teacher action can fall outside it.

    A non-empty result means the plan is unrealizable on its OWN source, and any
    ``proven_incompatible`` verdict there is CORRECT rather than a search defect.
    """

    from compose_v4.rewrite.trace_shard import decode_state

    created: dict[int, tuple[int, int]] = {}
    next_ordinal = 0
    gaps: list[dict[str, Any]] = []
    for step, (state, record) in enumerate(zip(states, records, strict=True)):
        graph = decode_state(state)
        desired = roles[step]
        desired_identity = identity(desired)
        found = False
        enumerated = 0
        for candidate in enumerate_role_successors(
            graph, desired, created=dict(created), step=step
        ):
            enumerated += 1
            observed, _ = action_role_supervision(
                graph, candidate.action_record, dict(created), step, next_ordinal
            )
            if identity(observed) == desired_identity:
                found = True
                break
        if not found:
            gaps.append(
                {
                    "step": step,
                    "executor_rule": str(desired["executor_rule"]),
                    "enumerated": enumerated,
                }
            )
        _, next_ordinal = action_role_supervision(graph, record, created, step, next_ordinal)
    return gaps


__all__ = [
    "BINDING_ROW_FIELDS",
    "OUTCOMES",
    "OUTCOME_COMPLETED",
    "OUTCOME_EXHAUSTED",
    "OUTCOME_INCOMPATIBLE",
    "OUTCOME_INVALID_PLAN",
    "PRODUCTION_MAX_REALIZATIONS",
    "PRODUCTION_NODE_BUDGET",
    "PRODUCTION_SECONDS_CAP",
    "PRODUCTION_SPECIFICATION",
    "RELAXABLE_COMPONENTS",
    "SCHEMA",
    "SPECIFICATIONS",
    "SPEC_R1",
    "SPEC_R2",
    "SPEC_V1_EXACT",
    "RoleSpecification",
    "bind_realized_plan",
    "expand",
    "finalize",
    "plan_demand",
    "plan_validity",
    "propagate",
    "realize",
    "realize_plan_binding",
    "retained_fraction",
    "role_match",
    "static_first_step_feasible",
    "teacher_step_gaps",
]
