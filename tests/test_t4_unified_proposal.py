"""Guards for the ONE per-expert proposal unit the unified T4 controller runs.

Until this module existed the same five lanes were written twice -- once inside the
Modal campaign app's `proposal_worker` and once inside a zero-oracle gate harness
that reproduced the app's call shape by hand.  Two copies of a proposal law are two
proposal laws, and the resemblance between them was maintained by reading rather
than by construction.  What is tested here is therefore the SINGLE definition: the
lane vocabulary, the authorization that keeps a rung-0 lane out of an ordinary
round, and the record shape that decides whether a successful rung-0 stage survives
the round path or kills it.

TWO GUARDS ARE FIRES-ONLY-ON-SUCCESS SHAPES, which is why they are driven through
the REAL production functions rather than asserted on a stub:

* `t4_integrated_route_fiber._experts` validates a record's `proposal_lane` against
  the frozen expert vocabulary and RAISES on an unknown one, so a rung-0 record
  carrying its own stage name as a lane would kill `attach_features` at precisely
  the moment rung 0 first produced an eligible endpoint. A stubbed `{"smiles": ...}`
  record cannot catch that, so a REAL fallback record is pushed through the REAL
  feature path.
* `proposal_experts: []` is FALSY, so `_experts` falls through to `proposal_lane`;
  the empty list only survives because the lane is `None`. The two fields are
  therefore asserted together, not separately.

ZERO oracle calls, zero docking, zero Modal.  CPU only.  The one real end-to-end
draw runs at an explicit `draws` override, because the shallow lane costs roughly a
second per draw on a drug-like parent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.frozen_proposal_escalation import frozen_escalation_block
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_integrated_route_fiber import EXPERTS
from compose_v4.experiments.t4_support_expansion import (
    ESCALATABLE_LANES,
    SupportExpansionContractError,
)
from compose_v4.experiments.t4_unified_controller import (
    DECLARED_REGION_LAW,
    ROUTED_RUNG_ZERO_LANE,
    StateRoutingContractError,
    state_routing_block,
)
from compose_v4.experiments.t4_unified_proposal import (
    LANES,
    ROUND_ONE_LANES,
    RUNG_ZERO_LANES,
    SCHEMA_VERSION,
    ProposalLaneError,
    proposal_unit,
)

ROOT = Path(__file__).resolve().parents[1]
FA7 = ROOT / "configs/t4_held_target_distilled_fa7_d06_250.json"

REGION_LANE = ROUTED_RUNG_ZERO_LANE["region"]
STATE_AWARE_LANE = ROUTED_RUNG_ZERO_LANE["state_aware"]

#: A tiny neutral parent for the authorization tests. The authorization is checked
#: before any chemistry, so the molecule is irrelevant to what they assert -- and a
#: cheap one keeps them cheap under a mutation that makes the check a no-op and lets
#: the real lane run.
CHEAP = "CCOCC"


def _payload() -> dict:
    from compose_v4.experiments.t4_matched_pilot import unseal

    return unseal(FA7)


def _contract(payload: dict | None = None, **overrides) -> dict:
    payload = payload if payload is not None else _payload()
    contract = dict(payload)
    contract["state_routing"] = state_routing_block()
    contract["support_expansion"] = frozen_escalation_block()
    contract["proposal"] = {
        **payload["proposal"],
        "shallow": {**payload["proposal"]["shallow"], "region_law": DECLARED_REGION_LAW},
    }
    contract.update(overrides)
    return contract


def _sources() -> dict[str, str]:
    return {row["cell"]: row["smiles"] for row in _payload()["cells"]}


def _task(expert: str, parent: str, **overrides) -> dict:
    task = {
        "expert": expert,
        "parent": parent,
        "parent_score": -7.5,
        "original_seed": parent,
        "proposal_seed": 20260922,
    }
    task.update(overrides)
    return task


def _fiber(parent: str, contract: dict) -> Fiber:
    return Fiber(parent, contract["delta"], support=contract["support"])


# ---- The lane vocabulary -------------------------------------------------


def test_the_round_one_lanes_are_the_frozen_expert_vocabulary():
    """Read from `t4_integrated_route_fiber`, never restated here: a constant
    written twice is a constant that drifts."""

    assert set(ROUND_ONE_LANES) == set(EXPERTS)
    assert len(ROUND_ONE_LANES) == len(EXPERTS) == 3


def test_the_escalatable_lanes_are_the_round_one_lanes_with_a_draw_budget():
    """`route_complete_region` is a beam over a fitted expert with no draw
    parameter, so the ladder cannot escalate it and it is deliberately absent."""

    assert set(ESCALATABLE_LANES) < set(ROUND_ONE_LANES)
    assert "route_complete_region" not in ESCALATABLE_LANES


def test_the_rung_zero_lanes_are_exactly_the_two_routed_ones():
    assert set(RUNG_ZERO_LANES) == set(ROUTED_RUNG_ZERO_LANE.values())
    assert set(RUNG_ZERO_LANES).isdisjoint(ROUND_ONE_LANES)
    assert set(LANES) == set(ROUND_ONE_LANES) | set(RUNG_ZERO_LANES)


@pytest.mark.parametrize(
    "expert", ["", "unknown", "generic_topology_macro_v2", "local", "region"]
)
def test_an_unknown_lane_is_refused(expert):
    contract = _contract()
    with pytest.raises(ProposalLaneError, match="unknown proposal lane"):
        proposal_unit(contract, _task(expert, CHEAP), fiber=object())


# ---- A rung-0 lane is unreachable from an unauthorized contract ----------


@pytest.mark.parametrize("lane", [REGION_LANE, STATE_AWARE_LANE])
def test_a_rung_zero_lane_is_refused_without_a_ladder_that_authorizes_the_stage(lane):
    """Running a rung-0 lane from an ordinary round -- by naming it in a proposal
    request -- would add a lane the contract never declared."""

    contract = _contract()
    contract.pop("support_expansion")
    with pytest.raises(SupportExpansionContractError):
        proposal_unit(contract, _task(lane, CHEAP), fiber=_fiber(CHEAP, contract))


@pytest.mark.parametrize("lane", [REGION_LANE, STATE_AWARE_LANE])
def test_a_rung_zero_lane_is_refused_without_a_resolvable_routing(lane):
    """The only legitimate way to reach a rung-0 lane is to have been ROUTED to it,
    so the routing must resolve even though this worker does not consult it."""

    contract = _contract()
    contract.pop("state_routing")
    with pytest.raises(StateRoutingContractError):
        proposal_unit(contract, _task(lane, CHEAP), fiber=_fiber(CHEAP, contract))


@pytest.mark.parametrize("lane", [REGION_LANE, STATE_AWARE_LANE])
def test_a_rung_zero_lane_is_refused_when_the_ladder_disables_the_stage(lane):
    """A ladder whose `zero_support_fallback` is false authorizes no rung 0, and a
    contract that merely NAMES the frozen ladder cannot turn it on per target."""

    from compose_v4.experiments.t4_support_expansion import SupportExpansionPolicy

    contract = _contract()
    disabled = SupportExpansionPolicy(
        draw_ladder=(960,),
        lanes=("shallow",),
        zero_support_fallback=False,
        stop_at_distinct_eligible=4,
        max_extra_draws_per_event=960,
        wall_seconds=60.0,
    )
    contract["support_expansion"] = {
        key: value
        for key, value in disabled.as_record().items()
        if key != "schema_version"
    }
    with pytest.raises(SupportExpansionContractError):
        proposal_unit(contract, _task(lane, CHEAP), fiber=_fiber(CHEAP, contract))


# ---- A model-backed lane needs its fitted expert -------------------------


@pytest.mark.parametrize("lane", ["route_complete_region", STATE_AWARE_LANE])
def test_a_model_backed_lane_without_its_route_expert_is_refused(lane):
    """The caller loads the checkpoint, because only the caller knows whether it
    sits in a baked image or a repository checkout -- so the unit must refuse a
    missing one loudly rather than proposing nothing and reporting a clean zero."""

    contract = _contract()
    contract["proposal"] = {
        **contract["proposal"],
        STATE_AWARE_LANE: {"shallow_draws": 1},
    }
    with pytest.raises(ProposalLaneError, match="route expert"):
        proposal_unit(
            contract,
            _task(lane, CHEAP),
            fiber=_fiber(CHEAP, contract),
            route_expert=None,
        )


# ---- The rung-0 record shape, on a REAL production run -------------------


#: The rung-0 seed these guards run under. The stage is deterministic given its
#: seed, and `fa7_0` at delta 0.6 is a hard cell: MEASURED, seeds 1, 3 and 20260922
#: return zero eligible endpoints and seed 11 returns one. A record-shape guard
#: needs a real record, so the seed is pinned to a productive one and the fixture
#: asserts the yield rather than assuming it -- a drift that empties the stage fails
#: loudly instead of leaving these tests vacuously green.
FALLBACK_SEED = 11


@pytest.fixture(scope="module")
def fa7_0_fallback():
    """ONE real `zero_support_fallback` run on `fa7_0`. Roughly 14 CPU-seconds.

    Module-scoped because every guard below needs the same real records and the
    stage is deterministic given its seed: re-running it would buy nothing.
    """

    contract = _contract()
    parent = _sources()["fa7_0"]
    fiber = _fiber(parent, contract)
    unit = proposal_unit(
        contract, _task(REGION_LANE, parent, proposal_seed=FALLBACK_SEED), fiber=fiber
    )
    return contract, parent, unit, fiber


def test_a_rung_zero_record_carries_its_stage_and_no_proposal_lane(fa7_0_fallback):
    """The two fields are asserted TOGETHER because `proposal_experts: []` is falsy:
    `_experts` falls through to `proposal_lane`, so the empty list only survives
    because the lane is `None`."""

    _, parent, unit, _ = fa7_0_fallback
    assert unit["schema_version"] == SCHEMA_VERSION
    assert unit["expert"] == REGION_LANE
    assert unit["parent"] == parent
    assert unit["proposal_seed"] == FALLBACK_SEED
    assert unit["records"], "the fallback returned nothing, so this test proves nothing"
    for record in unit["records"]:
        assert record["proposal_lane"] is None
        assert record["proposal_experts"] == []
        assert record["support_expansion_stage"] == REGION_LANE
        assert record["parent"] == parent
        assert record["smiles"] != parent
        assert record["delta"] == 0.6
    assert "zero_support_fallback_work" in unit["telemetry"]


def test_the_frozen_vocabulary_would_reject_the_stage_name_as_a_lane(fa7_0_fallback):
    """The mechanism behind the test above, stated as a fact about the REAL
    validator rather than as a claim in a docstring."""

    from compose_v4.experiments.t4_integrated_route_fiber import _experts

    _, _, unit, _ = fa7_0_fallback
    record = dict(unit["records"][0])
    assert _experts(record) == ()
    with pytest.raises(ValueError, match="unknown proposal experts"):
        _experts({**record, "proposal_lane": REGION_LANE, "proposal_experts": []})


def test_a_real_rung_zero_record_survives_the_production_feature_path(fa7_0_fallback):
    """The fires-only-on-success shape: `attach_features` runs `_experts`, so a
    record labelled with its own stage would kill the round at exactly the moment
    rung 0 first succeeded."""

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        select_batch,
    )

    _, parent, unit, fiber = fa7_0_fallback
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    candidates = attach_features(unit["records"], state, fiber)
    assert candidates, "attach_features dropped every real rung-0 record"
    expert_census(candidates)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=1.0),
        state,
        np.random.default_rng(1),
        round_index=1,
        batch=4,
        exploration=1,
        expert_floor_rounds=2,
    )
    assert selected, "select_batch could not select a real rung-0 candidate"


def test_a_stage_labelled_record_would_not_survive_it(fa7_0_fallback):
    """NEGATIVE CONTROL for the test above. Without it, a `attach_features` that had
    stopped validating lanes would let the guard pass while testing nothing."""

    from compose_v4.control.fiber_control import SearchState
    from compose_v4.experiments.t4_integrated_route_fiber import attach_features

    _, parent, unit, fiber = fa7_0_fallback
    mislabelled = [
        {**record, "proposal_lane": REGION_LANE, "proposal_experts": []}
        for record in unit["records"]
    ]
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    with pytest.raises(ValueError, match="unknown proposal experts"):
        attach_features(mislabelled, state, fiber)


# ---- One REAL end-to-end draw lane ---------------------------------------


def test_the_shallow_lane_runs_end_to_end_on_a_real_cell():
    """The production `expand`, the production fiber, a real T4 parent.

    Kept to four draws: the lane costs roughly a second per draw on a drug-like
    parent. MEASURED at this budget `fa7_0` returns zero eligible endpoints, which
    is the cell's whole problem and is why the routed expansion exists -- so what
    this establishes is that the lane RUNS and reports honestly, not that it yields.
    Any record it does return must be one the production gate admitted.
    """

    contract = _contract()
    parent = _sources()["fa7_0"]
    fiber = _fiber(parent, contract)
    unit = proposal_unit(
        contract, _task("shallow", parent, draws=4), fiber=fiber, region_law=None
    )
    assert unit["schema_version"] == SCHEMA_VERSION
    assert unit["expert"] == "shallow"
    assert unit["telemetry"]["raw_draws"] == 4
    assert unit["eligible_records_returned"] == len(unit["records"])
    assert unit["elapsed_seconds"] >= 0.0
    for record in unit["records"]:
        assert fiber.check(record["smiles"]) is not None
        assert record["proposal_lane"] == "shallow"


def test_the_draw_override_reaches_the_production_expand_call(monkeypatch):
    """The override is load-bearing rather than cosmetic: without it this lane runs
    its contract's 480 draws. A telemetry field alone would not prove it arrived, so
    the production call is observed.
    """

    import compose_v4.experiments.t4_unified_proposal as unit_module

    seen: dict = {}

    def recorder(parent, parent_score, fiber, rng, **kwargs):
        seen.update(kwargs)
        seen["parent"] = parent
        return []

    monkeypatch.setattr(unit_module, "expand", recorder)
    contract = _contract()
    proposal_unit(contract, _task("shallow", CHEAP, draws=7), fiber=object())
    assert seen["draws"] == 7
    assert seen["proposal_lane"] == "shallow"
    assert seen["horizon"] == contract["proposal"]["shallow"]["horizon"]
    assert seen["parent"] == CHEAP

    seen.clear()
    proposal_unit(contract, _task("shallow", CHEAP), fiber=object())
    assert seen["draws"] == contract["proposal"]["shallow"]["draws"]


def test_a_region_law_is_threaded_to_the_shallow_lane(monkeypatch):
    """`expand` fails closed on a law handed to any other lane, so the law must
    reach the one lane that can consume it and nothing else."""

    import compose_v4.experiments.t4_unified_proposal as unit_module

    seen: dict = {}
    monkeypatch.setattr(
        unit_module,
        "expand",
        lambda parent, score, fiber, rng, **kwargs: seen.update(kwargs) or [],
    )
    law = object()
    contract = _contract()
    proposal_unit(
        contract, _task("shallow", CHEAP, draws=1), fiber=object(), region_law=law
    )
    assert seen["region_law"] is law
