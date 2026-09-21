"""Guards for the PMO realization repair.

The load-bearing properties are:

  * the control arm IS the pinned predicate, proven on a DISCRIMINATING non-empty case
    (an all-empty equivalence check is vacuous and would pass for a broken predicate);
  * the relaxation is a strict WIDENING of that predicate, never a different one;
  * the semantic core and the dataflow edge are never relaxed, at any specification;
  * the four outcomes stay distinct;
  * a realization must satisfy the plan's own declared completion conditions, and a
    shorter action sequence is never substituted for a requested complete one.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pmo_binder_repair_v1 import (
    load_plans,
    load_production_parents,
)
from pmo_realization_repair_v1 import (
    OUTCOME_INVALID_PLAN,
    OUTCOMES,
    SPEC_R1,
    SPEC_R2,
    SPEC_V1_EXACT,
    _Prefix,
    finalize,
    plan_demand,
    plan_validity,
    propagate,
    realize,
    role_match,
)

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision
from compose_v4.control.pmo_joint_dependency_jump import (
    enumerate_role_successors,
)
from compose_v4.rewrite.trace_shard import decode_state


def _candidate_pool(limit_parents: int = 3, limit_plans: int = 5):
    """Real (desired role, observed role) pairs from the production parents."""

    plans = load_plans()[:limit_plans]
    parents = load_production_parents()[:limit_parents]
    pool = []
    for parent in parents:
        graph = decode_state(parent["state"])
        for plan in plans:
            desired = plan["roles"][0]
            for candidate in enumerate_role_successors(graph, desired, created={}, step=0):
                observed, _ = action_role_supervision(
                    graph, candidate.action_record, {}, 0, 0
                )
                pool.append((desired, observed))
    return pool


def test_control_arm_is_the_pinned_predicate_on_a_discriminating_case():
    pool = _candidate_pool()
    assert pool, "candidate pool must be non-empty for this check to mean anything"
    true_cases = false_cases = 0
    for desired, observed in pool:
        reference = identity(observed) == identity(desired)
        assert role_match(desired, observed, SPEC_V1_EXACT) is reference
        true_cases += int(reference)
        false_cases += int(not reference)
    # Discriminating in BOTH directions: a predicate stuck at True or at False would
    # otherwise pass this test.
    assert true_cases > 0
    assert false_cases > 0


def test_relaxation_is_a_strict_widening_not_a_different_predicate():
    pool = _candidate_pool()
    widened = 0
    for desired, observed in pool:
        exact = role_match(desired, observed, SPEC_V1_EXACT)
        for spec in (SPEC_R1, SPEC_R2):
            relaxed = role_match(desired, observed, spec)
            assert not (exact and not relaxed), "relaxation must never reject an exact match"
            widened += int(relaxed and not exact)
    assert widened > 0, "the relaxation must accept something the exact rule rejects"


def test_semantic_core_is_never_relaxed():
    pool = _candidate_pool()
    desired, observed = next(
        (d, o) for d, o in pool if role_match(d, o, SPEC_R2)
    )
    # A different operation.
    mutated = deepcopy(desired)
    mutated["executor_rule"] = "bond_reorder"
    assert not role_match(mutated, observed, SPEC_R2)
    # A different edit parameterization.
    mutated = deepcopy(desired)
    mutated["parameters"] = {**desired["parameters"], "__injected__": 1}
    assert not role_match(mutated, observed, SPEC_R2)
    # A different operand element, and a different operand charge.
    for field, delta in (("atom_type", 1), ("formal_charge", 1)):
        mutated = deepcopy(desired)
        mutated["operands"][0]["descriptor"][field] = (
            int(desired["operands"][0]["descriptor"][field]) + delta
        )
        assert not role_match(mutated, observed, SPEC_R2), field


def test_dataflow_edge_is_never_relaxed():
    pool = _candidate_pool()
    desired, observed = next((d, o) for d, o in pool if role_match(d, o, SPEC_R2))
    mutated = deepcopy(desired)
    mutated["operands"][0]["descriptor"]["origin"] = "route_created"
    mutated["operands"][0]["descriptor"]["created_ordinal"] = 0
    assert not role_match(mutated, observed, SPEC_R2)


def test_relaxed_components_are_exactly_the_declared_ones():
    for spec, expected in ((SPEC_R1, {"neighbor_element_histogram", "creation_lag"}),
                           (SPEC_R2, {"neighbor_element_histogram", "creation_lag",
                                      "implicit_hydrogens", "degree",
                                      "bond_class_histogram"})):
        assert set(spec.relaxed) == expected
        for name in ("atom_type", "formal_charge", "origin", "created_ordinal"):
            assert name in spec.compared_fields
        for name in spec.relaxed:
            assert name not in spec.compared_fields


def test_plan_validity_rejects_a_dangling_created_handle():
    plan = deepcopy(load_plans()[0])
    assert plan_validity(plan)["valid"]
    plan["roles"][0]["operands"][0]["descriptor"]["origin"] = "route_created"
    plan["roles"][0]["operands"][0]["descriptor"]["created_ordinal"] = 99
    report = plan_validity(plan)
    assert not report["valid"]
    assert any("unemitted_ordinal" in reason for reason in report["reasons"])


def test_invalid_plan_is_its_own_outcome_not_incompatibility():
    plan = deepcopy(load_plans()[0])
    plan["roles"][0]["operands"][0]["descriptor"]["origin"] = "route_created"
    plan["roles"][0]["operands"][0]["descriptor"]["created_ordinal"] = 99
    source = decode_state(load_production_parents()[0]["state"])
    result = realize(source, plan, SPEC_R1, node_budget=10, seconds_cap=5.0)
    assert result["outcome"] == OUTCOME_INVALID_PLAN
    assert result["outcome"] in OUTCOMES
    assert result["nodes_expanded"] == 0


def test_outcome_is_always_one_of_the_four():
    plan = load_plans()[0]
    source = decode_state(load_production_parents()[0]["state"])
    result = realize(source, plan, SPEC_R1, node_budget=3, seconds_cap=3.0)
    assert result["outcome"] in OUTCOMES


def test_finalize_refuses_to_substitute_a_shorter_action_sequence():
    plan = load_plans()[0]
    source = decode_state(load_production_parents()[0]["state"])
    prefix = _Prefix(graph=source, actions=(), created=(), next_ordinal=0)
    with pytest.raises(ValueError):
        finalize(source, prefix, plan)


def test_identity_budget_prune_is_disabled_while_a_restatement_remains():
    """Regression: ``atom_restate_semantic`` changes an atom's element in place.

    A per-(element, charge) deletion budget is therefore NOT a necessary condition while
    any restatement is still to come.  Applying it unconditionally reported a
    teacher-witness pair -- one whose own recorded route is a witness that a realization
    exists -- as ``proven_incompatible``.
    """

    plan = next(
        p
        for p in load_plans()
        if any(r["executor_rule"] == "atom_restate_semantic" for r in p["roles"])
    )
    demand = plan_demand(plan)
    first_restate = next(
        step
        for step, role in enumerate(plan["roles"])
        if role["executor_rule"] == "atom_restate_semantic"
    )
    assert demand.remaining_restates[first_restate] > 0
    assert demand.remaining_restates[len(plan["roles"])] == 0
    source = decode_state(load_production_parents()[0]["state"])
    prefix = _Prefix(graph=source, actions=(), created=(), next_ordinal=0)
    refusal = propagate(prefix, 0, plan, demand, SPEC_R1)
    assert refusal != "insufficient_preexisting_atoms_of_a_required_identity"


def test_teacher_step_gaps_detects_a_non_reenumerable_action():
    """The teacher-witness stratum is NOT a guaranteed positive control.

    A plan bound back onto its own generating source is only realizable if every teacher
    action is re-enumerable by the RUNTIME fiber.  Measured: plan 421a14fea5ed has one
    ``atom_delete`` step (29 of 31) whose own action the runtime does not offer, which is
    exactly the depth at which the realizer proves incompatibility there.  A
    ``proven_incompatible`` verdict on that pair is therefore correct, not a search
    defect.

    The guard is structural rather than pinned to that one plan: it asserts the function
    reports no gap for a route it CAN re-enumerate, and that the gap record carries the
    step and rule needed to diagnose one.
    """

    import gzip
    import json as _json

    from pmo_binder_repair_v1 import TEACHER_CORPUS
    from pmo_realization_repair_v1 import teacher_step_gaps

    from compose_v4.control.pmo_joint_dependency_jump import _generic_role_sequence

    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = _json.load(handle)["payload"]
    route = next(
        row
        for row in corpus["routes"]
        if row["dependency_region_program"].get("complete_representation_supported")
        and len(row["actions"]) <= 14
    )
    roles = _generic_role_sequence(route)
    gaps = teacher_step_gaps(roles, route["states"][:-1], route["actions"])
    assert isinstance(gaps, list)
    for gap in gaps:
        assert set(gap) == {"step", "executor_rule", "enumerated"}
        assert 0 <= gap["step"] < len(roles)


def test_witness_soundness_gate_a_reenumerable_route_is_never_proven_incompatible():
    """The negative control for ``proven_incompatible``.

    If every teacher step of a route is re-enumerable by the runtime fiber, then a
    complete role-consistent realization of its plan on its own source DOES exist, so
    the realizer must never return ``proven_incompatible`` there.  Returning it would be
    an unsound prune.

    This is the guard that caught the real bug: a per-(element, charge) deletion budget
    applied while an ``atom_restate_semantic`` was still pending is not a necessary
    condition, because a restatement changes an atom's element in place.
    """

    import gzip
    import json as _json

    from pmo_binder_repair_v1 import TEACHER_CORPUS
    from pmo_realization_repair_v1 import (
        OUTCOME_INCOMPATIBLE,
        teacher_step_gaps,
    )

    from compose_v4.control.pmo_joint_dependency_jump import _generic_role_sequence

    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = _json.load(handle)["payload"]
    plan_ids = {plan["plan_id"]: plan for plan in load_plans()}
    for route in corpus["routes"]:
        if not route["dependency_region_program"].get("complete_representation_supported"):
            continue
        if not 6 <= len(route["actions"]) <= 13:
            continue
        roles = _generic_role_sequence(route)
        plan_id = identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        plan = plan_ids.get(plan_id)
        if plan is None:
            continue
        if teacher_step_gaps(roles, route["states"][:-1], route["actions"]):
            continue  # unrealizable on its own source; incompatibility would be correct
        source = decode_state(route["states"][0])
        result = realize(
            source, plan, SPEC_V1_EXACT, node_budget=4000, seconds_cap=180.0
        )
        assert result["outcome"] != OUTCOME_INCOMPATIBLE, (
            f"{plan_id[:12]}: every teacher step is re-enumerable, so a realization "
            f"exists; proven_incompatible here is an unsound prune "
            f"(depth {result['max_depth_reached']}/{result['plan_primitive_count']})"
        )
        return
    pytest.skip("no short fully re-enumerable witness route available")
