"""Guards for the PMO-v2 binder failure decomposition probe.

The probe re-implements the v1 binding loop so it can be instrumented.  A
re-implementation is only evidence about production if it reproduces production, so
the first test drives the PINNED ``bind_joint_plan`` and requires an identical endpoint
set -- including on a pair where the binder actually returns bindings, since an
equivalence test whose cases are all empty cannot fail usefully.

The remaining tests pin the soundness properties the decomposition rests on:
the referee's prefilter may not change an outcome, the static certificate must be a
genuine necessary condition, the attribution views must be monotone, and no partial
binding may ever be reported as a realization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pmo_binder_repair_v1 import (
    ATTRIBUTION_VIEWS,
    OUTCOME_UNDECIDED,
    bind_instrumented,
    first_step_feasible_under,
    load_plans,
    load_production_parents,
    load_teacher_root_parents,
    referee_binding_exists,
    static_first_step_feasible,
    verify_equivalence,
)

from compose_v4.rewrite.trace_shard import decode_state


def _plans():
    return {plan["plan_id"]: plan for plan in load_plans()}


def _parents():
    rows = load_production_parents() + load_teacher_root_parents()
    return {row["parent_id"]: row for row in rows}


def _graph(parent_id):
    return decode_state(_parents()[parent_id]["state"])


# A pair the probe must reproduce on, chosen because the pinned binder returns a
# NON-EMPTY result there: an all-empty equivalence check is vacuous.
DISCRIMINATING_PARENT = "teacher_root:3"
DISCRIMINATING_PLAN_PRIMITIVES = 15


def _discriminating_plan():
    for plan in sorted(load_plans(), key=lambda row: row["primitive_count"]):
        if int(plan["primitive_count"]) == DISCRIMINATING_PLAN_PRIMITIVES:
            graph = _graph(DISCRIMINATING_PARENT)
            if bind_instrumented(graph, plan, beam_width=4)["complete"]:
                return plan
    return None


def test_probe_reproduces_pinned_binder_on_a_nonempty_case():
    plan = _discriminating_plan()
    assert plan is not None, "no discriminating plan found; equivalence test would be vacuous"
    graph = _graph(DISCRIMINATING_PARENT)
    for width in (4, 8):
        result = verify_equivalence(graph, plan, beam_width=width)
        assert result["endpoint_sets_identical"], (width, result)
        assert result["live_bindings"] > 0, "case must be non-empty to discriminate"


def test_static_certificate_is_a_necessary_condition():
    """If the zero-search certificate refuses, no beam width may bind -- at step 0."""

    plans = list(_plans().values())
    parents = _parents()
    checked = 0
    for parent_id in ("production_init:0", "production_init:1", "teacher_root:0"):
        graph = decode_state(parents[parent_id]["state"])
        for plan in plans:
            if static_first_step_feasible(graph, plan):
                continue
            result = bind_instrumented(graph, plan, beam_width=64)
            assert result["died_at_step"] == 0, (parent_id, plan["plan_id"])
            assert not result["complete"]
            checked += 1
            if checked >= 12:
                return
    assert checked, "no infeasible pair found; the certificate was never exercised"


def test_referee_prefilter_does_not_change_the_outcome():
    """The prefilter is a necessary condition, so it may only change COST."""

    plans = sorted(_plans().values(), key=lambda row: row["primitive_count"])
    graph = _graph("production_init:0")
    compared = 0
    for plan in plans[:20]:
        slow = referee_binding_exists(graph, plan, seconds_cap=20.0, prefilter=False)
        fast = referee_binding_exists(graph, plan, seconds_cap=20.0, prefilter=True)
        if OUTCOME_UNDECIDED in (slow["outcome"], fast["outcome"]):
            continue
        assert slow["outcome"] == fast["outcome"], plan["plan_id"]
        assert slow["depth_reached"] == fast["depth_reached"], plan["plan_id"]
        compared += 1
        if compared >= 5:
            break
    assert compared, "no decided pair; prefilter soundness was never exercised"


def test_attribution_views_are_monotone():
    """Dropping a pinned field can only ever admit more parents, never fewer."""

    graph = _graph("production_init:0")
    plans = sorted(_plans().values(), key=lambda row: row["primitive_count"])[:30]
    full = ATTRIBUTION_VIEWS["v1_full"]
    for name, fields in ATTRIBUTION_VIEWS.items():
        if name == "v1_full" or not set(fields) <= set(full):
            continue
        for plan in plans:
            if first_step_feasible_under(graph, plan, full):
                assert first_step_feasible_under(graph, plan, fields), (name, plan["plan_id"])


def test_only_complete_programs_are_reported():
    """A realization must consume every role of the plan; no prefix substitutes."""

    plans = sorted(_plans().values(), key=lambda row: row["primitive_count"])
    graph = _graph("teacher_root:3")
    seen = 0
    for plan in plans[:25]:
        result = bind_instrumented(graph, plan, beam_width=4)
        for entry in result["complete"]:
            assert entry["primitive_count"] == result["plan_length"]
            assert entry["exact_replay"] is True
            seen += 1
        if result["died_at_step"] is not None:
            assert not result["complete"]
    assert seen, "no complete program produced; completeness was never exercised"


@pytest.mark.parametrize("width", [0, -1])
def test_beam_width_must_be_positive(width):
    plan = next(iter(_plans().values()))
    with pytest.raises(ValueError):
        bind_instrumented(_graph("production_init:0"), plan, beam_width=width)
