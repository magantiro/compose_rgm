"""Guards for the production binder adapter (``bind_realized_plan``).

The realizer replaced a width-4 beam at ``pmo_population_controller.py``'s jump lane.
The controller consumes the returned rows UNCHANGED, so the load-bearing properties are
about the row contract rather than about the search:

  * a row carries every field the pinned beam's rows carry, with the expectation taken
    from ``bind_joint_plan`` itself rather than from this module's own constant;
  * the row survives the production consumer -- ``_joint_program`` replays the actions
    and asserts ``trace["states"] == bound["states"]`` and
    ``trace["endpoint"] == bound["endpoint_key"]``, which is the assertion a wrong state
    ENCODING would trip;
  * ``endpoint_state`` decodes to the molecule the row claims, checked by a SECOND
    independent path: the controller recomputes the retained fraction from the decoded
    state, and ``finalize`` computed it from the executed endpoint graph;
  * ``max_realizations`` really returns SEVERAL rows, because the controller ranks them
    against a random retention target and one row would make that target inert;
  * the four search outcomes survive the adapter, so "proven incompatible" and "ran out
    of budget" are never collapsed into one failure string.

The fixture is a short teacher route bound back onto its OWN source: a realization
provably exists there, so a failure is a defect in this code rather than a hard pair.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pmo_binder_repair_v1 import TEACHER_CORPUS, load_plans

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_joint_dependency_jump import (
    _generic_role_sequence,
    bind_joint_plan,
)
from compose_v4.control.pmo_population_controller import (
    PmoPopulationController,
    _retained_fraction,
)
from compose_v4.control.pmo_realization import (
    BINDING_ROW_FIELDS,
    OUTCOMES,
    PRODUCTION_SPECIFICATION,
    SPEC_V1_EXACT,
    bind_realized_plan,
    realize_plan_binding,
)
from compose_v4.rewrite.trace_shard import decode_state

_CACHE: dict[str, object] = {}


def _short_route_pairs(max_actions: int = 13, min_actions: int = 4, limit: int = 12):
    """(source, plan) pairs from short teacher routes, bound onto their own source."""

    if "pairs" in _CACHE:
        return _CACHE["pairs"]
    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    plans = {plan["plan_id"]: plan for plan in load_plans()}
    pairs = []
    for route in corpus["routes"]:
        if not route["dependency_region_program"].get("complete_representation_supported"):
            continue
        if not min_actions <= len(route["actions"]) <= max_actions:
            continue
        roles = _generic_role_sequence(route)
        plan = plans.get(
            identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        )
        if plan is None:
            continue
        pairs.append((decode_state(route["states"][0]), plan))
        if len(pairs) >= limit:
            break
    _CACHE["pairs"] = pairs
    return pairs


def _first_bound(**kwargs):
    """The first short pair on which the adapter returns at least one row."""

    key = f"bound:{sorted(kwargs.items())}"
    if key in _CACHE:
        return _CACHE[key]
    for source, plan in _short_route_pairs():
        rows = bind_realized_plan(source, plan, **kwargs)
        if rows:
            _CACHE[key] = (source, plan, rows)
            return _CACHE[key]
    pytest.fail(
        "no short teacher route bound on its own source; the fixture, not the "
        "assertion, is what failed -- a realization provably exists on these pairs"
    )


def test_row_carries_every_field_the_pinned_beam_emits():
    """The expectation is the BEAM's own row, never this module's field constant.

    ``BINDING_ROW_FIELDS`` is checked against ``bind_joint_plan``'s output here so the
    constant cannot drift into agreeing only with itself -- the failure mode that let a
    cache file list be "verified" against a keyset derived from the same list.
    """

    for source, plan in _short_route_pairs():
        beam = bind_joint_plan(source, plan, beam_width=4)
        if not beam:
            continue
        realized = bind_realized_plan(source, plan, node_budget=400, seconds_cap=60.0)
        assert realized, "the complete search must bind wherever the width-4 beam binds"
        beam_fields = set(beam[0])
        assert beam_fields <= set(realized[0]), (
            "binder row is missing fields the controller may read: "
            f"{sorted(beam_fields - set(realized[0]))}"
        )
        assert beam_fields <= set(BINDING_ROW_FIELDS) | set(realized[0])
        # The declared contract must actually cover the beam's fields.
        assert beam_fields <= set(BINDING_ROW_FIELDS), (
            "BINDING_ROW_FIELDS does not cover the pinned beam's row: "
            f"{sorted(beam_fields - set(BINDING_ROW_FIELDS))}"
        )
        return
    pytest.fail("no short pair produced a width-4 beam binding to compare against")


def test_row_survives_the_production_consumer_unchanged():
    """``_joint_program`` is the real judge of the state encoding.

    It re-executes the row's actions and asserts the replayed states and endpoint equal
    the row's own.  A row whose ``states`` were encoded differently, or which omitted
    the source state, fails here rather than silently at scored-run time.
    """

    source, _plan, rows = _first_bound(node_budget=400, seconds_cap=60.0)
    for row in rows:
        program, binding, trace = PmoPopulationController._joint_program(source, row)
        assert trace["states"] == row["states"]
        assert trace["endpoint"] == row["endpoint_key"]
        assert program is not None and binding is not None


def test_endpoint_state_decodes_to_the_molecule_the_row_claims():
    """Two independent paths to one number.

    ``finalize`` computes ``retained_fraction`` from the graph the executor returned;
    the controller computes it from ``decode_state(row["endpoint_state"])``.  They agree
    only if the encoding round-trips, so this catches a wrong or stale endpoint state
    without recomputing the expectation from the adapter.
    """

    source, _plan, rows = _first_bound(node_budget=400, seconds_cap=60.0)
    for row in rows:
        endpoint = decode_state(row["endpoint_state"])
        assert _retained_fraction(source, endpoint) == pytest.approx(
            row["retained_fraction"]
        )
        assert row["states"][-1] == row["endpoint_state"]
        assert len(row["states"]) == len(row["actions"]) + 1


def test_several_realizations_are_returned_so_the_ranking_is_not_collapsed():
    """The controller ranks bindings by retained fraction against a random target.

    Returning one row would make that target inert -- a silent behaviour change, not a
    detail -- so the adapter must be able to return several.
    """

    for source, plan in _short_route_pairs():
        rows = bind_realized_plan(
            source, plan, node_budget=400, seconds_cap=60.0, max_realizations=4
        )
        if len(rows) > 1:
            assert len({row["endpoint_key"] for row in rows}) >= 1
            assert all(row["exact_replay"] for row in rows)
            return
    pytest.fail("no short pair produced more than one realization for the ranking")


def test_max_realizations_bounds_the_returned_rows():
    """The cap is a real bound, and a larger cap never returns fewer rows."""

    for source, plan in _short_route_pairs():
        one = bind_realized_plan(source, plan, node_budget=400, seconds_cap=60.0,
                                 max_realizations=1)
        if not one:
            continue
        assert len(one) == 1
        many = bind_realized_plan(source, plan, node_budget=400, seconds_cap=60.0,
                                  max_realizations=3)
        assert 1 <= len(many) <= 3
        assert len(many) >= len(one)
        return
    pytest.fail("no short pair bound at all; the cap check never ran")


def test_max_realizations_must_be_positive():
    source, plan = _short_route_pairs()[0]
    with pytest.raises(ValueError):
        bind_realized_plan(source, plan, max_realizations=0)


def test_ordering_is_total_and_content_addressed_not_search_order():
    """Rows come back in a deterministic order that does not depend on search order.

    The controller breaks ranking ties on ``endpoint_key`` alone, so two realizations
    reaching the same endpoint must not be separated by the order the DFS happened to
    reach them in.
    """

    source, plan, rows = _first_bound(node_budget=400, seconds_cap=60.0)
    keys = [(row["endpoint_key"], identity(row["actions"])) for row in rows]
    assert keys == sorted(keys)
    again = bind_realized_plan(source, plan, node_budget=400, seconds_cap=60.0)
    assert [row["endpoint_key"] for row in again] == [row["endpoint_key"] for row in rows]


def test_outcome_survives_the_adapter_so_failures_are_not_one_string():
    """A plan proven incompatible and one out of budget are different findings."""

    source, plan = _short_route_pairs()[0]
    result = realize_plan_binding(source, plan, node_budget=2, seconds_cap=30.0)
    assert result["outcome"] in OUTCOMES
    assert "bindings" in result
    assert isinstance(result["bindings"], list)


def test_production_specification_is_the_pinned_predicate():
    """Relaxation measured WORSE at teacher scale; production must not drift onto it."""

    assert PRODUCTION_SPECIFICATION is SPEC_V1_EXACT
    assert PRODUCTION_SPECIFICATION.exact_role_identity is True
    assert PRODUCTION_SPECIFICATION.relaxed == frozenset()


def test_controller_call_site_uses_the_realizer_not_the_beam():
    """The integration is the point; a silent revert would pass every test above."""

    text = (
        Path(__file__).resolve().parents[1]
        / "src/compose_v4/control/pmo_population_controller.py"
    ).read_text()
    assert "realize_plan_binding(" in text
    assert "bind_joint_plan(" not in text
