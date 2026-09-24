"""The app hop: does the CAMPAIGN's own round loop reach the routing and the ladder?

WHY THIS FILE EXISTS SEPARATELY FROM `test_t4_unified_controller.py`
----------------------------------------------------------------------
`tests/test_t4_unified_controller.py` proves the library hops: a contract that
names neither object is refused, a probe router installed into
`routed_support_expansion` is consulted, and a terminal record this module did
not produce cannot publish candidate exhaustion.  Every one of those can pass
while the Modal campaign app never calls any of it -- which is precisely the
inert-mechanism failure being fixed, and the reason two call sites threading one
parameter need two guards rather than one.

This file guards the APP.  It does it two ways, because neither alone is enough:

BY EXECUTION.  `modal` is stubbed in `sys.modules` -- the network and the
orchestration, never the resolver -- so `modal_apps.t4_unified_controller_app`
imports in the pinned chemistry environment, and the app's OWN `expand_support`
is driven on real T4 sources with an injected dispatcher.  What is verified is
the function `run_cell` calls, not a transcription of it.

BY CALL SITE.  The remaining guards read the app's AST and require the load-
bearing calls to be present in the functions that must make them, with an
explicit FLOOR on how many functions were inspected so the check cannot go blind
if the app is restructured.  `inspect.signature` would pass a keyword accepted
and dropped one hop later; a call-site check will not.

Every guard here is paired with a mutation in
`scripts/t4_unified_wiring_mutations.py` that must turn it red.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from compose_v4.control.frozen_proposal_escalation import FROZEN_LADDER_SHA256
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_unified_controller import (
    ROUTED_RUNG_ZERO_LANE,
    ROUTER_SHA256,
    UnifiedExpansionNotConsumed,
    assert_unified_expansion_is_consumed,
)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "modal_apps/t4_unified_controller_app.py"

#: The cells the two routed kernels are measured on. `fa7_0` is neutral, so the
#: region kernel keeps its support; `5ht1b_2` carries a formal charge, on which
#: the executor's charge policy refuses region excisions.
REGION_CELL = ("fa7", "fa7_0")
STATE_AWARE_CELL = ("5ht1b", "5ht1b_2")


# ---- Importing the app without Modal ----


class _Chainable:
    """A stand-in for a Modal builder: every attribute returns something chainable.

    The app's image is a long `.pip_install(...).add_local_dir(...)` chain
    evaluated at import, and none of it is under test here. Stubbing the
    ORCHESTRATION is the analogue of stubbing the network rather than the
    resolver: the app's own logic still runs, unmodified.
    """

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self


class _StubFunction:
    """What `@app.function(...)` returns: callable, mappable, remotable."""

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "stub")
        self.__doc__ = fn.__doc__

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def map(self, tasks, **kwargs):  # pragma: no cover - never reached in these tests
        raise AssertionError("a test reached the real Modal fan-out")

    def remote(self, *args, **kwargs):  # pragma: no cover - same
        raise AssertionError("a test reached the real Modal fan-out")


class _StubApp:
    def __init__(self, name):
        self.name = name

    def function(self, *args, **kwargs):
        return _StubFunction

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


def _import_app():
    """Import the campaign app with `modal` stubbed, leaving `sys.modules` clean."""

    if "modal_apps.t4_unified_controller_app" in sys.modules:
        return sys.modules["modal_apps.t4_unified_controller_app"]
    stub = types.ModuleType("modal")
    stub.Image = _Chainable()
    stub.Volume = _Chainable()
    stub.Secret = _Chainable()
    stub.Retries = _Chainable()
    stub.App = _StubApp
    saved = sys.modules.get("modal")
    sys.modules["modal"] = stub
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import modal_apps.t4_unified_controller_app as app_module

        return app_module
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved


def _contract(protein: str) -> dict:
    return unseal(ROOT / f"configs/t4_unified_controller_{protein}_d06_v1.json")


def _cell_row(protein: str, cell: str) -> tuple[dict, dict]:
    payload = _contract(protein)
    return payload, next(row for row in payload["cells"] if row["cell"] == cell)


def _quiet_fan_out(seen: list[dict]):
    """A dispatcher that records what the app asked for and returns nothing.

    Returning no records keeps the ladder running to its declared end, which is
    what makes the rung sequence observable: a dispatcher that returned an
    eligible endpoint would stop at `reached_target` and hide the later rungs.
    """

    def _dispatch(requests):
        requests = list(requests)
        seen.extend(requests)
        # The app's `_collect` zips answers against requests positionally with
        # `strict=True`, so a dispatcher returns answers only, in order.
        return [
            {
                "expert": request["expert"],
                "records": [],
                "eligible_records_returned": 0,
                "telemetry": {},
                "status": "complete",
            }
            for request in requests
        ]

    return _dispatch


# ---- BY EXECUTION ----


@pytest.mark.parametrize(
    ("protein", "cell", "kernel"),
    [(*REGION_CELL, "region"), (*STATE_AWARE_CELL, "state_aware")],
)
def test_the_apps_own_expansion_routes_each_parent_and_dispatches_that_lane(
    protein, cell, kernel
):
    """The app's `expand_support`, driven on a real source, routes and dispatches.

    This is the guard that cannot be satisfied by a library test: it drives the
    function `run_cell` calls, with the app's own closures, and checks that the
    lane the router chose is the lane the fan-out was actually asked for.
    """

    app_module = _import_app()
    payload, row = _cell_row(protein, cell)
    seen: list[dict] = []
    outcome, record = app_module.expand_support(
        {"run_id": "test", "cell": cell},
        payload,
        row,
        parents=[row["smiles"]],
        archive={row["smiles"]: 0.0},
        round_index=1,
        fan_out=_quiet_fan_out(seen),
    )

    assert record["routed_kernels"] == [kernel]
    assert record["routed_lanes"] == [ROUTED_RUNG_ZERO_LANE[kernel]]
    assert record["router_source"] == "contract"
    assert record["ladder_source"] == "contract"
    assert record["ladder"]["policy_sha256"] == FROZEN_LADDER_SHA256
    assert record["router"]["policy_sha256"] == ROUTER_SHA256

    # The rung-0 request the app actually issued carries the ROUTED lane. A
    # record naming one lane while the fan-out ran another is the exact shape a
    # library-only test cannot see.
    rung_zero = [
        request
        for request in seen
        if request["expert"] in set(ROUTED_RUNG_ZERO_LANE.values())
    ]
    assert rung_zero, "the app dispatched no rung-0 request"
    assert {request["expert"] for request in rung_zero} == {
        ROUTED_RUNG_ZERO_LANE[kernel]
    }

    # With an empty dispatcher the frozen ladder runs to its declared end.
    assert outcome.stop_reason == "ladder_exhausted"
    assert outcome.attempts == 3
    assert outcome.draws_spent == 6720
    # And the terminal guard accepts a record built this way.
    assert_unified_expansion_is_consumed(record, outcome)


def test_the_routing_probe_reaches_the_router_before_any_dispatch():
    """Hop 2 on the APP's path: the probe raises before a fan-out is reached.

    The dispatcher handed in RAISES `AssertionError` if called. A wiring that
    consulted the router only after scheduling work would surface that instead
    of the probe's own exception, so this distinguishes "consulted" from
    "consulted early enough to matter".
    """

    from compose_v4.experiments.t4_unified_controller import (
        assert_state_routing_is_consumed,
    )

    app_module = _import_app()
    payload, row = _cell_row(*REGION_CELL)

    def _refuse(_requests):
        raise AssertionError("the probe reached a proposal fan-out")

    def _probe_route(probe_router, attempt):
        return app_module.expand_support(
            {"run_id": "test", "cell": row["cell"]},
            payload,
            row,
            parents=[row["smiles"]],
            archive={row["smiles"]: 0.0},
            round_index=1,
            router=probe_router,
            fan_out=_refuse,
        )

    assert assert_state_routing_is_consumed(_probe_route) >= 1


def test_a_probe_record_can_never_be_published_as_a_campaign_result():
    """An injected router marks the record, and the terminal guard refuses it.

    Without this, the consumption probe itself would be a way to manufacture a
    publishable exhaustion record that never ran the contract's routing.
    """

    app_module = _import_app()
    payload, row = _cell_row(*REGION_CELL)

    class _Stand:
        policy = "t4_unified_routing_v1"
        policy_sha256 = ROUTER_SHA256

        def decision(self, parent_smiles, **kwargs):
            return {
                "activated_kernel": "region",
                "support_ramp": 1.0,
                "applicability": {},
                "search": {},
            }

        @staticmethod
        def lane_for(kernel):
            return ROUTED_RUNG_ZERO_LANE[kernel]

        def as_record(self):
            return {"policy": self.policy, "policy_sha256": self.policy_sha256}

    seen: list[dict] = []
    outcome, record = app_module.expand_support(
        {"run_id": "test", "cell": row["cell"]},
        payload,
        row,
        parents=[row["smiles"]],
        archive={row["smiles"]: 0.0},
        round_index=1,
        router=_Stand(),
        fan_out=_quiet_fan_out(seen),
    )
    assert record["router_source"] == "injected"
    with pytest.raises(UnifiedExpansionNotConsumed):
        assert_unified_expansion_is_consumed(record, outcome)


# ---- BY CALL SITE ----


def _functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(APP.read_text())
    found = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # A FLOOR, so a restructure that removed the functions this file names would
    # fail here rather than silently inspecting nothing.
    assert len(found) >= 12, f"only {len(found)} functions found in {APP.name}"
    return found


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


@pytest.mark.parametrize(
    ("function", "required"),
    [
        ("_validate_task", "resolve_unified_controller"),
        ("proposal_worker", "proposal_unit"),
        ("expand_support", "routed_support_expansion"),
        ("run_cell", "expand_support"),
        ("run_cell", "assert_state_routing_is_consumed"),
        ("run_cell", "assert_unified_expansion_is_consumed"),
        ("run_cell", "normalize_expansion_records"),
    ],
)
def test_the_load_bearing_call_is_made_by_the_function_that_must_make_it(
    function, required
):
    """Each hop is asserted at its OWN call site, never at a shared one.

    Two call sites threading one parameter can leave a single consultation test
    green when the parameter is dropped at one of them, so a mutation removing
    any one of these must turn exactly this row red.
    """

    node = _functions()[function]
    assert required in _calls(node), (
        f"{function}() no longer calls {required}(); the hop it guards is open"
    )


@pytest.mark.parametrize(
    "forbidden",
    ["resolve_support_expansion", "run_support_expansion"],
)
def test_the_engine_never_resolves_an_unfrozen_ladder(forbidden):
    """A contract that can carry a ladder number can carry a per-target one.

    `resolve_support_expansion` reads the ladder FROM the contract and
    `run_support_expansion` would let the app drive the stopping rule itself,
    bypassing the routing. The frozen panel goes through
    `resolve_frozen_escalation` and `routed_support_expansion` or it is not one
    controller.
    """

    assert forbidden not in _calls(ast.parse(APP.read_text()))


def test_candidate_exhaustion_is_published_only_behind_the_terminal_guard():
    """The status string may appear only in a function that runs the guard first.

    This is what stops a round loop from quietly returning to the bare hard stop:
    the terminal record cannot be written without a routed expansion having
    produced it.
    """

    tree = ast.parse(APP.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        writes_status = any(
            isinstance(child, ast.Constant) and child.value == "candidate_exhaustion"
            for child in ast.walk(node)
        )
        if writes_status and "assert_unified_expansion_is_consumed" not in _calls(node):
            offenders.append(node.name)
    assert not offenders, (
        f"{offenders} publish candidate_exhaustion without running the terminal guard"
    )


def test_every_wrapper_points_at_an_existing_unified_contract_and_the_engine():
    """A wrapper that sets a contract nobody sealed is a launch that cannot start."""

    wrappers = sorted(
        path
        for path in (ROOT / "modal_apps").glob("t4_unified_controller_*_app.py")
        if path != APP
    )
    assert wrappers, "no unified-controller wrappers found"
    named = set()
    for wrapper in wrappers:
        text = wrapper.read_text()
        assert "from modal_apps.t4_unified_controller_app import" in text, wrapper.name
        contracts = [
            part
            for part in text.replace('"', " ").replace("'", " ").split()
            if part.startswith("configs/t4_unified_controller_")
        ]
        assert contracts, f"{wrapper.name} names no unified contract"
        for relative in contracts:
            assert (ROOT / relative).exists(), f"{wrapper.name} -> missing {relative}"
        named.update(contracts)

    # The count used to be pinned at five. That is an inventory literal: it fails
    # whenever an arm is legitimately added and it never checks the property that
    # matters. The bijection does, in BOTH directions -- a wrapper naming a
    # contract nobody sealed is a launch that cannot start, and a sealed contract
    # with no wrapper is an arm nobody can launch.
    sealed = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "configs").glob("t4_unified_controller_*_v1.json")
    }
    assert named == sealed, (
        f"wrappers and sealed contracts disagree; "
        f"contracts with no wrapper: {sorted(sealed - named)}; "
        f"wrappers naming an unsealed contract: {sorted(named - sealed)}"
    )


def test_every_contract_refuses_to_launch_without_the_owner():
    """The scored panel is a 22,500-call decision, and it is not ours to take."""

    import json

    checked = 0
    for path in sorted((ROOT / "configs").glob("t4_unified_controller_*_v1.json")):
        payload = json.loads(path.read_text())["payload"]
        assert payload["scored_launch_authorized"] is False, path.name
        assert payload["modal_launch_authorized"] is False, path.name
        # The threshold is part of the arm's identity, so a filename that
        # disagrees with the executable field would launch the wrong experiment.
        expected = 0.4 if "_d04" in path.name else 0.6
        assert payload["delta"] == expected, (
            f"{path.name} declares delta={payload['delta']} but its name says "
            f"{expected}"
        )
        checked += 1
    assert checked >= 5, f"expected at least the five base arms, checked {checked}"
