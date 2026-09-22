"""Guards for the jump-lane binding autopsy.

The module OBSERVES the production binder; every guard here therefore checks that it
observes without deciding, and that the labels it produces are the ones the production
search actually justified.  Fixture pairs are REAL (parent, plan) pairs drawn from the
arm-B celecoxib run and are chosen to cover every mechanism label, so each guard is
exercised where it BINDS rather than only where it is vacuous.

Expectations in the fixture were recorded under a NON-BINDING seconds cap, so they
depend only on ``node_budget`` and the deterministic depth-first order.  A guard that
asserted a wall-clock-dependent outcome would flake on a slower machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control import pmo_realization as realization
from compose_v4.experiments.pmo_jump_binding_autopsy import (
    PROJECTABLE_FIELDS,
    REFUSAL_MECHANISM,
    SPEC_V1_FIELDS,
    classify,
    diagnose_pair,
    field_projection_lift,
    first_refusal,
    recording_propagate,
    role_based_specification,
    root_refusal,
    step0_feasible_under,
)
from compose_v4.rewrite.trace_shard import decode_state

_FIXTURE = Path(__file__).parent / "fixtures" / "pmo_jump_autopsy_pairs_v1.json"
_CHECKPOINT = Path(__file__).resolve().parents[1] / (
    "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
)
# Large enough that the clock never fires, which is what makes every expectation below
# a function of the node budget alone.
_NON_BINDING_CAP = 600.0


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _plans() -> dict:
    payload = json.loads(_CHECKPOINT.read_text())["payload"]["checkpoints"]["shared_all_routes"]
    return {plan["plan_id"]: plan for plan in payload["plan_latents"]}


def _pairs():
    plans = _plans()
    for row in _fixture()["pairs"]:
        yield row, decode_state(row["source_state"]), plans[row["plan_id"]]


def test_fixture_covers_both_step_zero_verdicts_and_several_mechanisms():
    """A fixture that never reaches a refusal cannot test a refusal."""

    rows = _fixture()["pairs"]
    verdicts = {row["expected_step0_feasible_v1"] for row in rows}
    assert verdicts == {True, False}, "fixture must contain step-0 feasible AND infeasible pairs"
    mechanisms = {row["expected_mechanism"] for row in rows}
    assert "bound" in mechanisms
    assert any(name.endswith("@root") for name in mechanisms)
    assert any(name.endswith("@depth") for name in mechanisms)


def test_recording_propagate_restores_the_production_function():
    original = realization.propagate
    with recording_propagate():
        assert realization.propagate is not original
    assert realization.propagate is original


def test_recording_propagate_restores_the_production_function_after_an_exception():
    original = realization.propagate
    with pytest.raises(RuntimeError), recording_propagate():
        raise RuntimeError("boom")
    assert realization.propagate is original


def test_the_recorder_never_changes_the_production_decision():
    """The wrapper observes; the production function decides.

    Compared against an INDEPENDENT unwrapped call rather than against a recomputation
    of the wrapper's own output, which could not fail.
    """

    for row, source, plan in _pairs():
        observed = diagnose_pair(source, plan, seconds_cap=_NON_BINDING_CAP)
        clean = realization.realize_plan_binding(
            source,
            plan,
            specification=realization.PRODUCTION_SPECIFICATION,
            node_budget=realization.PRODUCTION_NODE_BUDGET,
            seconds_cap=_NON_BINDING_CAP,
            max_realizations=realization.PRODUCTION_MAX_REALIZATIONS,
        )
        assert observed["outcome"] == clean["outcome"], row["plan_id"]
        assert observed["bindings"] == len(clean["bindings"]), row["plan_id"]
        # Outcome alone is NOT enough: propagate's checks are necessary conditions, so a
        # wrapper that discarded them would leave the outcome intact and only make the
        # search more expensive.  The cost is what exposes that, and it is compared
        # against an independent unwrapped run rather than recomputed from the wrapper.
        assert observed["nodes_expanded"] == clean["nodes_expanded"], row["plan_id"]
        assert observed["successors_enumerated"] == clean["successors_enumerated"], (
            row["plan_id"]
        )


def test_production_identity_and_all_fields_specifications_agree_at_step_zero():
    """``SPEC_V1_FIELDS`` stands in for the production predicate at step 0.

    The projection diagnostic needs a specification whose fields can be removed one at a
    time; it is only a valid stand-in if it reproduces the production step-0 verdict.
    """

    for row, source, plan in _pairs():
        exact = step0_feasible_under(source, plan, realization.SPEC_V1_EXACT)
        fields = step0_feasible_under(source, plan, SPEC_V1_FIELDS)
        assert exact == fields == row["expected_step0_feasible_v1"], row["plan_id"]


def test_dropping_a_field_can_only_add_step_zero_feasibility():
    """Monotonicity.  Binds: the fixture carries step-0 INFEASIBLE pairs."""

    infeasible = 0
    for _row, source, plan in _pairs():
        projection = field_projection_lift(source, plan)
        if not projection["baseline_feasible"]:
            infeasible += 1
            continue
        for field, feasible in projection["single_drop_feasible"].items():
            assert feasible, f"dropping {field} removed a step-0 match"
    assert infeasible > 0, "monotonicity was never tested against an infeasible pair"


def test_field_projection_sweeps_every_descriptor_component_production_defines():
    """Derived from the PRODUCTION constants, never from the sweep's own list.

    Asserting the sweep covers ``PROJECTABLE_FIELDS`` cannot fail, because the sweep is
    built from that same constant -- both sides move together.  The expectation here is
    the descriptor components ``pmo_realization`` itself declares, so shortening the
    sweep turns this red.
    """

    expected = set(realization.RELAXABLE_COMPONENTS) | set(
        realization.SEMANTIC_OPERAND_FIELDS
    ) | set(realization.DATAFLOW_OPERAND_FIELDS)
    assert set(PROJECTABLE_FIELDS) == expected, (
        "the projection sweep must cover every descriptor component production declares"
    )
    _row, source, plan = next(iter(_pairs()))
    projection = field_projection_lift(source, plan)
    assert set(projection["single_drop_feasible"]) == expected


def test_role_based_specification_keeps_the_semantic_core_and_the_dataflow_edge():
    spec = role_based_specification()
    compared = set(spec.compared_fields)
    assert {"atom_type", "formal_charge"} <= compared, "semantic core must never be projected"
    assert {"origin", "created_ordinal"} <= compared, "dataflow edge must never be projected"
    assert "neighbor_element_histogram" not in compared
    assert spec.relaxed == realization.SPEC_R2.relaxed, (
        "the role-based view is content-identical to the pre-existing R2 arm; a result "
        "must be attributable to that arm rather than appear to be a new mechanism"
    )


def test_a_budget_hit_is_never_classified_as_incompatibility():
    """Binds: the budget is forced low enough that the search cannot finish."""

    forced = 0
    for _row, source, plan in _pairs():
        observed = diagnose_pair(source, plan, node_budget=1, seconds_cap=_NON_BINDING_CAP)
        if observed["outcome"] != realization.OUTCOME_EXHAUSTED:
            continue
        forced += 1
        assert classify(observed) == "search_budget_exhausted"
    assert forced > 0, "no fixture pair was driven into a budget hit; the guard is vacuous"


def test_an_invalid_plan_is_never_classified_as_incompatibility():
    _row, source, plan = next(iter(_pairs()))
    broken = json.loads(json.dumps(plan))
    broken["roles"][0].setdefault("operands", [])
    broken["roles"][0]["operands"] = [
        {
            "role": "target",
            "descriptor": {"origin": "route_created", "created_ordinal": 99, "atom_type": 2},
        }
    ]
    observed = diagnose_pair(source, broken, seconds_cap=_NON_BINDING_CAP)
    assert observed["outcome"] == realization.OUTCOME_INVALID_PLAN
    assert classify(observed) == "invalid_plan"


def test_root_and_depth_refusals_are_distinguished_on_real_pairs():
    roots = depths = 0
    for row, source, plan in _pairs():
        observed = diagnose_pair(source, plan, seconds_cap=_NON_BINDING_CAP)
        label = classify(observed)
        assert label == row["expected_mechanism"], row["plan_id"]
        if label.endswith("@root"):
            roots += 1
            assert observed["root_refused"] is True
            assert observed["nodes_expanded"] == 0, "a root refusal expands nothing"
        if label.endswith("@depth"):
            depths += 1
            assert observed["root_refused"] is False
            assert observed["nodes_expanded"] > 0, "a depth refusal must have expanded a node"
    assert roots > 0 and depths > 0


def test_every_refusal_string_the_production_search_can_return_has_a_mechanism_name():
    """Derived from the production source, so a new refusal cannot go unlabelled."""

    source = Path(realization.__file__).read_text()
    body = source.split("def propagate(", 1)[1].split("\ndef ", 1)[0]
    returned = {
        line.split('return "', 1)[1].split('"', 1)[0]
        for line in body.splitlines()
        if 'return "' in line
    }
    assert returned, "no refusal strings were recovered from propagate; the parse is wrong"
    assert returned <= set(REFUSAL_MECHANISM), (
        f"unlabelled propagate refusals: {sorted(returned - set(REFUSAL_MECHANISM))}"
    )


def test_first_and_root_refusal_readers_agree_with_the_log():
    log = [
        {"step": 0, "reason": None},
        {"step": 1, "reason": "insufficient_slot_capacity"},
        {"step": 1, "reason": None},
    ]
    assert root_refusal(log) is None
    assert first_refusal(log) == {"step": 1, "reason": "insufficient_slot_capacity"}
    assert root_refusal([{"step": 0, "reason": "insufficient_slot_capacity"}]) == (
        "insufficient_slot_capacity"
    )
    assert first_refusal([{"step": 0, "reason": None}]) is None


def test_structural_delta_reports_the_region_shape_on_hand_checkable_pairs():
    """Pinned on pairs whose answer can be read off the molecules.

    ``retained_fraction_mcs`` is the MCS core over the SOURCE's heavy atoms and is NOT the
    controller's ``_retained_fraction``; the distinct name is the guard against quoting
    them interchangeably.
    """

    from compose_v4.experiments.pmo_jump_binding_autopsy import structural_delta

    # One methyl added to benzene: one changed atom, one region, whole ring retained.
    methylation = structural_delta("c1ccccc1", "Cc1ccccc1")
    assert methylation["d_heavy"] == 1
    assert methylation["d_rings"] == 0
    assert methylation["n_changed_regions"] == 1
    assert methylation["largest_changed_region"] == 1
    assert methylation["retained_fraction_mcs"] == 1.0

    # A large connected substituent replacing a small one: one big changed region.
    replacement = structural_delta(
        "CCOc1ccc(CN2CCCC2)cc1", "CCOc1ccc(CNCC(=O)NC2CCCCC2)cc1"
    )
    assert replacement["n_changed_regions"] == 1
    assert replacement["largest_changed_region"] == 11
    assert replacement["retained_fraction_mcs"] < 1.0

    # Unparseable input is refused rather than silently scored.
    assert structural_delta("c1ccccc1", "not a molecule") is None


def test_structural_delta_separates_one_large_region_from_scattered_small_ones():
    """The discriminator the census rests on must actually discriminate.

    Two endpoints with the SAME number of changed atoms, differing only in whether those
    atoms form one connected region or several.  A measure that could not tell them apart
    would report the generic sampler and the required macro as identical.
    """

    from compose_v4.experiments.pmo_jump_binding_autopsy import structural_delta

    scattered = structural_delta("c1ccccc1", "Cc1cc(C)cc(C)c1")
    coherent = structural_delta("c1ccccc1", "CCCc1ccccc1")
    assert scattered["total_changed"] == coherent["total_changed"] == 3
    assert scattered["n_changed_regions"] == 3
    assert scattered["largest_changed_region"] == 1
    assert coherent["n_changed_regions"] == 1
    assert coherent["largest_changed_region"] == 3
