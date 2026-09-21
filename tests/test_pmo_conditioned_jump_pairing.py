"""Guards for support-conditioned (parent x plan) pairing in the PMO jump lane.

The repair replaces ``P(plan) * P(parent)`` -- two counters indexed independently --
with ``P(plan | parent, plan in F(parent))``.  Every test here is written so that
removing a piece of the production wiring turns it RED; a guard that cannot fail when
the production path is broken would repeat the inert-repair defect it exists to stop.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control import pmo_population_controller as PPC
from compose_v4.control import pmo_realization as PR
from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_population_controller import (
    JUMP_PAIRING_INDEPENDENT,
    JUMP_PAIRING_SUPPORT_CONDITIONED,
    PRODUCTION_JUMP_PAIRING,
    PmoPopulationController,
    supported_plan_index,
)
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

# Production PMO archive states carry 48 slots; a tight graph would delete the whole
# atom_insert family from the legal support and silently understate every certificate.
PARENT_SLOTS = 48
PARENT_SMILES = "CC(=O)Nc1ccc(O)cc1"
TINY_SMILES = "C"


class _ProbeReached(BaseException):
    """Deliberately NOT a ValueError/RuntimeError.

    ``_generate_jump_pool`` catches both per attempt, so either would be swallowed by
    the very path the probe is trying to observe.
    """


def _parent(smiles: str = PARENT_SMILES):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), PARENT_SLOTS)


def _entry(smiles: str = PARENT_SMILES, entry_id: str = "fixture-parent"):
    state = encode_state(_parent(smiles))
    entry = {
        "entry_id": entry_id,
        "trace": {"states": [state]},
        "source_state": state,
        "endpoint": smiles,
        "program": {"marks": []},
        "static_score": None,
    }
    parent = {
        "entry_id": entry_id,
        "parent_probability": 1.0,
        "parent_measured_score": 0.0,
    }
    return entry, parent


def _controller(seed=23, wall_seconds=5.0):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=32,
        candidates_per_batch=8,
        wall_seconds=wall_seconds,
        parent_allocation="niche_score",
    )
    return PmoPopulationController(
        config,
        source_group="fixture-source",
        oracle_protocol="fixture-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
    )


def _eligibility(row):
    return {**row, "oracle_eligible": True}


# ---- the certificate itself ----


def test_certificate_refusal_is_a_necessary_condition_on_the_real_catalog():
    """A refused plan must have NO realization -- checked against the real searcher."""

    source = _parent()
    spec = PR.PRODUCTION_SPECIFICATION
    refused = []
    for plan in CHECKPOINT["plan_latents"]:
        if not PR.plan_parent_support(source, plan, spec)["supported"]:
            refused.append(plan)
    assert refused, "fixture parent must refuse some plans for this guard to mean anything"
    for plan in refused[:20]:
        result = PR.realize(
            source,
            plan,
            spec,
            node_budget=PR.PRODUCTION_NODE_BUDGET,
            seconds_cap=PR.PRODUCTION_SECONDS_CAP,
            collect_all=True,
            max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
        )
        assert not result["realizations"], plan["plan_id"]
        assert result["outcome"] != PR.OUTCOME_COMPLETED


def test_certificate_is_necessary_not_sufficient_and_says_which_stage_refused():
    source = _parent(TINY_SMILES)
    spec = PR.PRODUCTION_SPECIFICATION
    stages = {
        PR.plan_parent_support(source, plan, spec)["stage"]
        for plan in CHECKPOINT["plan_latents"]
    }
    # A one-atom parent cannot support the teacher-scale catalog, and the refusal must
    # be ATTRIBUTED rather than reported as one undifferentiated string.
    assert stages <= {None, *PR.SUPPORT_STAGES}
    assert stages & set(PR.SUPPORT_STAGES)


def test_certificate_is_reward_blind_and_task_blind():
    """The only inputs are the parent graph, the plan latent and the spec."""

    import inspect

    signature = inspect.signature(PR.plan_parent_support)
    assert list(signature.parameters) == ["source", "plan", "spec"]


# ---- the sampler ----


def test_selected_plan_is_always_one_the_certificate_admits():
    source = _parent()
    plans = list(CHECKPOINT["plan_latents"])
    seen = 0
    for position in range(40):
        index, census = supported_plan_index(source, plans, position)
        if index is None:
            continue
        seen += 1
        assert PR.plan_parent_support(source, plans[index])["supported"]
        assert census["supported"] >= 1
    assert seen, "fixture parent must support at least one plan"


def test_sampler_indexes_the_feasible_SUBSEQUENCE_in_order_not_a_reranking():
    """Conditioning restricts the intended order; it does not re-rank it."""

    plans = list(CHECKPOINT["plan_latents"])
    feasible = {plans[i]["plan_id"] for i in (3, 11, 29, 47)}

    def stub(source, plan, spec):
        supported = plan["plan_id"] in feasible
        return {
            "supported": supported,
            "stage": None if supported else PR.SUPPORT_STAGE_ROOT_PROPAGATION,
            "reason": None if supported else "stub",
            "specification": spec.name,
        }

    picks = [
        supported_plan_index(_parent(), plans, position, support=stub)[0]
        for position in range(8)
    ]
    # Positions walk the feasible sub-sequence IN ITS OWN ORDER and wrap.  A "first
    # feasible at or after position" rule, or any compatibility ranking, breaks this.
    assert picks == [3, 11, 29, 47, 3, 11, 29, 47]


def test_sampler_abstains_rather_than_returning_an_unsupported_plan():
    def refuse_everything(source, plan, spec):
        return {
            "supported": False,
            "stage": PR.SUPPORT_STAGE_DEPTH0,
            "reason": "stub",
            "specification": spec.name,
        }

    index, census = supported_plan_index(
        _parent(), list(CHECKPOINT["plan_latents"]), 0, support=refuse_everything
    )
    assert index is None
    assert census["supported"] == 0
    assert census["plans_scanned"] == len(CHECKPOINT["plan_latents"])


def test_sampler_memo_is_keyed_on_the_plan_and_reused():
    plans = list(CHECKPOINT["plan_latents"])
    calls = {"n": 0}

    def counting(source, plan, spec):
        calls["n"] += 1
        return PR.plan_parent_support(source, plan, spec)

    memo: dict = {}
    source = _parent()
    supported_plan_index(source, plans, 0, support=counting, memo=memo)
    first = calls["n"]
    supported_plan_index(source, plans, 1, support=counting, memo=memo)
    assert first == len(plans)
    assert calls["n"] == first, "a memoized second pass must evaluate no certificate"


def test_sampler_default_support_is_the_pinned_certificate():
    import inspect

    default = inspect.signature(supported_plan_index).parameters["support"].default
    assert default is PR.plan_parent_support


# ---- production wiring: the repair must not be inert ----


def test_production_pairing_is_the_conditioned_policy():
    assert PRODUCTION_JUMP_PAIRING == JUMP_PAIRING_SUPPORT_CONDITIONED
    assert _controller().jump_pairing == JUMP_PAIRING_SUPPORT_CONDITIONED


def test_generate_jump_pool_reaches_the_support_sampler(monkeypatch):
    """The consultation guard.

    Dropping the conditioning branch from ``_generate_jump_pool`` -- which is exactly
    how a validated repair becomes inert -- turns this RED.
    """

    def probe(*args, **kwargs):
        raise _ProbeReached

    monkeypatch.setattr(PPC, "supported_plan_index", probe)
    controller = _controller()
    with pytest.raises(_ProbeReached):
        controller._generate_jump_pool(_eligibility, set(), [_entry()])


def test_abstention_returns_the_opportunity_and_fabricates_no_plan(monkeypatch):
    monkeypatch.setattr(
        PPC,
        "supported_plan_index",
        lambda *args, **kwargs: (None, {"supported": 0, "refusals_by_reason": {}}),
    )
    controller = _controller()
    attempts, candidates, _ = controller._generate_jump_pool(_eligibility, set(), [_entry()])
    counts = controller.population_state["channels"][PPC.JUMP_CHANNEL]
    assert candidates == []
    assert attempts and all(row["status"] == "support_abstained" for row in attempts)
    assert all(row["plan_id"] is None for row in attempts)
    assert all("endpoint" not in row for row in attempts)
    # An abstention proposed nothing and refused nothing: it must not be laundered
    # into either counter, or the funnel becomes unreadable again.
    assert counts["proposals"] == 0
    assert counts["execution_rejections"] == 0
    assert counts["support_abstentions"] == len(attempts)


def test_independent_policy_reproduces_the_legacy_pairing(monkeypatch):
    """The control arm must be the LEGACY rule, and must not consult the certificate."""

    def probe(*args, **kwargs):
        raise _ProbeReached

    monkeypatch.setattr(PPC, "supported_plan_index", probe)
    seen: list[str] = []

    def capture(source, plan, **kwargs):
        seen.append(plan["plan_id"])
        return {"outcome": PR.OUTCOME_INCOMPATIBLE, "bindings": []}

    monkeypatch.setattr(PPC, "realize_plan_binding", capture)
    controller = _controller(wall_seconds=600.0)
    controller.jump_pairing = JUMP_PAIRING_INDEPENDENT
    controller._generate_jump_pool(_eligibility, set(), [_entry()])
    order = [row["plan_id"] for row in controller._jump_plan_order()]
    assert seen == [order[index % len(order)] for index in range(len(seen))]


# ---- the starved-cap defect ----


def test_jump_lane_searches_only_at_the_honest_seconds_cap(monkeypatch):
    """A cap below ``PRODUCTION_SECONDS_CAP`` manufactures false exhaustion.

    ``wall_seconds`` is deliberately far below the cap here: the superseded
    ``min(PRODUCTION_SECONDS_CAP, wall_seconds - elapsed)`` form would hand the search
    a cap of at most 2.0 s, so restoring it turns this RED.
    """

    caps: list[float] = []

    def capture(source, plan, **kwargs):
        caps.append(float(kwargs["seconds_cap"]))
        return {"outcome": PR.OUTCOME_INCOMPATIBLE, "bindings": []}

    monkeypatch.setattr(PPC, "realize_plan_binding", capture)
    controller = _controller(wall_seconds=2.0)
    controller._generate_jump_pool(_eligibility, set(), [_entry()])
    assert caps, "the lane must run at least one honest attempt"
    assert all(cap == PR.PRODUCTION_SECONDS_CAP for cap in caps), caps


def test_lane_stops_rather_than_starving_later_attempts(monkeypatch):
    """With no honest cap left, the lane STOPS; it does not search under a short cap."""

    def slow(source, plan, **kwargs):
        # 30 s of a 45 s lane budget: the lane is NOT out of time (the pre-existing
        # wall guard does not fire), but the 15 s left is short of the 20 s honest
        # cap.  That is exactly the state the superseded form searched under.
        controller.jump_lane_clock += 30.0
        return {"outcome": PR.OUTCOME_INCOMPATIBLE, "bindings": []}

    monkeypatch.setattr(PPC, "realize_plan_binding", slow)
    real_counter = PPC.perf_counter
    controller = _controller(wall_seconds=45.0)
    controller.jump_lane_clock = 0.0
    monkeypatch.setattr(PPC, "perf_counter", lambda: real_counter() + controller.jump_lane_clock)
    attempts, _, _ = controller._generate_jump_pool(_eligibility, set(), [_entry()])
    searched = [row for row in attempts if row["status"] != "support_abstained"]
    assert len(searched) == 1
    counts = controller.population_state["channels"][PPC.JUMP_CHANNEL]
    assert counts["honest_cap_lane_stops"] == 1
