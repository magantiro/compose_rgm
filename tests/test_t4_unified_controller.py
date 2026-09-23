"""Guards for the executable contract surface of the unified T4 controller.

The invariant under test is REACHABILITY.  Two mechanisms -- the state routing and
the frozen escalation ladder -- were built, measured and tested with no production
caller, which is the inert-mechanism failure this repository has paid for seven
times.  `t4_unified_controller` is the one place a signed contract turns into those
objects, and the three hops it declares are guarded here SEPARATELY, because two
call sites threading one parameter can leave a consultation test green when the
parameter is dropped at one of them:

  hop 1  contract -> objects   `resolve_state_routing` /
                               `assert_declared_region_configuration`
  hop 2  objects -> draw site  `assert_state_routing_is_consumed`, which installs a
                               probe router and requires the production function to
                               consult it
  hop 3  draw site -> terminal `assert_unified_expansion_is_consumed`, which refuses
                               to let a cell publish candidate exhaustion without a
                               record this module produced

Every behavioural test drives the REAL `routed_support_expansion` over REAL T4
sources read from the shipped contracts, with the rung-0 and ladder fan-outs stubbed
so a unit test costs no proposal work.  What is stubbed is the caller's own fan-out,
which the production function takes as a callable for exactly that reason; the
routing, the ladder resolution, the assignment construction and the record are the
production ones.

Each guard is paired with a mutation in `scripts/t4_unified_wiring_mutations.py`
that must turn a NAMED test here red.  A test whose expectation is recomputed from
the code under test cannot fail, so the hash guard perturbs the DECLARED rule
description and requires the pinned hash to move, and the probe guard is written
behaviourally -- a closure that swallows the five exception types the draw path
catches must still observe the probe -- rather than by restating a class hierarchy.

ZERO oracle calls, zero docking, zero Modal.  CPU only.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.frozen_proposal_escalation import (
    FROZEN_LADDER_ID,
    FROZEN_LADDER_SHA256,
    frozen_escalation_block,
)
from compose_v4.control.t4_unified_routing import (
    MolecularApplicability,
    SearchProgress,
    activated_kernel,
    kernel_weights,
    molecular_applicability,
)
from compose_v4.experiments.t4_support_expansion import (
    ExpansionOutcome,
    SupportExpansionNotConsumed,
)
from compose_v4.experiments.t4_unified_controller import (
    DECLARED_REGION_LAW,
    ROUTED_RUNG_ZERO_LANE,
    ROUTER_DESCRIPTION,
    ROUTER_ID,
    ROUTER_SHA256,
    StateRoutingContractError,
    StateRoutingNotConsumed,
    UnifiedExpansionNotConsumed,
    UnifiedRouter,
    _RouterProbeConsumed,
    assert_declared_region_configuration,
    assert_state_routing_is_consumed,
    assert_unified_expansion_is_consumed,
    resolve_state_routing,
    resolve_unified_controller,
    routed_support_expansion,
    state_routing_block,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "fa7": "configs/t4_held_target_distilled_fa7_d06_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_d06_250.json",
}

#: The exception types `t4_fiber_campaign.expand` catches per draw and
#: `synthesize_dynamic_program` catches per module family. A probe raising any of
#: them would be swallowed by the very path it exists to observe.
SWALLOWED = (ValueError, RuntimeError, KeyError, IndexError, TypeError)


def _payload(name: str) -> dict:
    """The shipped contract payload, unsealed through the production checker."""

    from compose_v4.experiments.t4_matched_pilot import unseal

    return unseal(ROOT / CONTRACTS[name])


def _contract(name: str) -> dict:
    """A unified-panel contract: the shipped payload plus the three declarations."""

    payload = _payload(name)
    contract = dict(payload)
    contract["state_routing"] = state_routing_block()
    contract["support_expansion"] = frozen_escalation_block()
    contract["proposal"] = {
        **payload["proposal"],
        "shallow": {**payload["proposal"]["shallow"], "region_law": DECLARED_REGION_LAW},
    }
    return contract


def _sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in CONTRACTS:
        for row in _payload(name)["cells"]:
            out[row["cell"]] = row["smiles"]
    return out


SOURCES = _sources()


def _stub_expansion(name: str, cells, **overrides):
    """Drive the REAL `routed_support_expansion` with stubbed fan-outs.

    `rung_zero` and `escalate` are the CALLER's production fan-outs, injected as
    callables. Stubbing them is what keeps this a unit test; everything between the
    contract and the record is the production path.
    """

    seen: list = []
    kwargs = {
        "contract": _contract(name),
        "parents": [SOURCES[cell] for cell in cells],
        "rounds_completed": 1,
        "rung_zero": lambda assignments: (seen.extend(assignments) or [], {"probe": 1}),
        "escalate": lambda draws, attempt: [],
    }
    kwargs.update(overrides)
    outcome, record = routed_support_expansion(**kwargs)
    return outcome, record, seen


# ---- hop 1: the contract's routing block ---------------------------------


def test_the_exact_block_is_accepted_and_returns_the_production_router():
    router = resolve_state_routing({"state_routing": state_routing_block()})
    assert isinstance(router, UnifiedRouter)
    assert router.policy == ROUTER_ID
    assert router.policy_sha256 == ROUTER_SHA256
    assert router.as_record() == state_routing_block()


def test_an_absent_routing_block_is_refused_rather_than_defaulted():
    """There is no `None` return: a run that silently falls back to one hard-coded
    kernel while its contract says otherwise is the defect this surface stops."""

    with pytest.raises(StateRoutingContractError, match="declares no"):
        resolve_state_routing({})


@pytest.mark.parametrize("block", [[], "t4_unified_routing_v1", 3, True])
def test_a_non_mapping_routing_block_is_refused(block):
    """`None` is deliberately absent: an absent block has its own refusal message
    and its own test, and folding the two together would let either one cover the
    other."""

    with pytest.raises(StateRoutingContractError, match="must be a mapping"):
        resolve_state_routing({"state_routing": block})


def test_an_extra_key_is_refused_so_a_block_cannot_carry_a_parameter():
    """A block that can hold a routing parameter can hold a different one per
    target, and then the panel is a family of controllers rather than one."""

    block = {**state_routing_block(), "draws": 960}
    with pytest.raises(StateRoutingContractError, match="must be exactly"):
        resolve_state_routing({"state_routing": block})


@pytest.mark.parametrize("dropped", ["policy", "policy_sha256"])
def test_a_missing_key_is_refused(dropped):
    block = {k: v for k, v in state_routing_block().items() if k != dropped}
    with pytest.raises(StateRoutingContractError, match="must be exactly"):
        resolve_state_routing({"state_routing": block})


def test_a_wrong_policy_name_is_refused():
    block = {**state_routing_block(), "policy": "t4_unified_routing_v2"}
    with pytest.raises(StateRoutingContractError, match="policy must be"):
        resolve_state_routing({"state_routing": block})


def test_a_stale_policy_hash_is_refused():
    """The hash is what makes the block pin the GUARANTEE rather than the name."""

    block = {**state_routing_block(), "policy_sha256": "0" * 64}
    with pytest.raises(StateRoutingContractError, match="policy_sha256"):
        resolve_state_routing({"state_routing": block})


def test_the_whole_contract_policy_resolves_in_one_call():
    router, ladder, declaration = resolve_unified_controller(_contract("fa7"))
    assert router.policy_sha256 == ROUTER_SHA256
    assert sum(ladder.draw_ladder) == ladder.max_extra_draws_per_event
    assert declaration["region_law"] == DECLARED_REGION_LAW


# ---- hop 1: the hash pins what the routing may read ----------------------


def test_the_pinned_hash_is_the_hash_of_the_declared_rule():
    """A hand-edited constant that no longer hashes its own description is caught
    here rather than by a contract that validates against a stale pin."""

    assert ROUTER_SHA256 == identity(ROUTER_DESCRIPTION)


def test_the_declared_rule_pins_the_live_field_sets_of_both_routing_records():
    """The field sets ARE the guarantee that no target identity and no reward can
    reach the routing, so the description must carry the LIVE ones."""

    assert ROUTER_DESCRIPTION["molecular_applicability_fields"] == [
        "heavy_atoms",
        "growth_headroom",
        "net_formal_charge",
        "charged_centres",
        "total_regions",
        "executable_regions",
        "charge_refused_regions",
        "large_regions",
        "protonation_sites",
    ]
    assert ROUTER_DESCRIPTION["search_progress_fields"] == [
        "eligible_pool_size",
        "consecutive_empty_rounds",
        "rounds_completed",
    ]
    assert ROUTER_DESCRIPTION["molecular_applicability_fields"] == [
        field.name for field in fields(MolecularApplicability)
    ]
    assert ROUTER_DESCRIPTION["search_progress_fields"] == [
        field.name for field in fields(SearchProgress)
    ]
    assert ROUTER_DESCRIPTION["rung_zero_lane"] == dict(ROUTED_RUNG_ZERO_LANE)


@pytest.mark.parametrize(
    "key,perturbed",
    [
        ("molecular_applicability_fields", ["heavy_atoms"]),
        ("search_progress_fields", ["eligible_pool_size", "docking_score"]),
        (
            "rung_zero_lane",
            {"region": "zero_support_fallback", "state_aware": "zero_support_fallback"},
        ),
        ("kernels", ["local", "region"]),
        ("proposal_slots", 40),
        ("router", "t4_unified_routing_v2"),
    ],
)
def test_the_pinned_hash_moves_when_any_pinned_component_moves(key, perturbed):
    """The load-bearing half is the PRESENCE assertion: a description that stopped
    pinning a component would let that component drift under a contract that still
    validates, and this fails rather than silently perturbing an absent key."""

    assert key in ROUTER_DESCRIPTION, f"the rule description no longer pins {key!r}"
    assert ROUTER_DESCRIPTION[key] != perturbed, "the perturbation is not a perturbation"
    assert identity({**ROUTER_DESCRIPTION, key: perturbed}) != ROUTER_SHA256


# ---- hop 1: the declared region configuration ----------------------------


def test_the_declared_region_law_is_accepted_in_its_bare_string_form():
    payload = {"proposal": {"shallow": {"region_law": DECLARED_REGION_LAW}}}
    assert assert_declared_region_configuration(payload)["region_law"] == (
        DECLARED_REGION_LAW
    )


def test_the_declared_region_law_is_accepted_in_its_block_form():
    """`region_law_contract` accepts a parameterised block, so this surface must
    read the law NAME out of one rather than demanding the bare string."""

    block = {"law": DECLARED_REGION_LAW, "maximum": None, "floor": 0.05}
    payload = {"proposal": {"shallow": {"region_law": block}}}
    assert assert_declared_region_configuration(payload)["region_law"] == block


def test_an_absent_proposal_lane_is_refused():
    with pytest.raises(StateRoutingContractError, match="no proposal.shallow block"):
        assert_declared_region_configuration({"proposal": {}})
    with pytest.raises(StateRoutingContractError, match="no proposal.shallow block"):
        assert_declared_region_configuration({})


def test_an_absent_region_law_field_is_refused():
    with pytest.raises(StateRoutingContractError, match="must declare"):
        assert_declared_region_configuration(
            {"proposal": {"shallow": {"draws": 480, "horizon": 3}}}
        )


@pytest.mark.parametrize(
    "declared",
    ["uniform_bounded_v1", "free_gate_margin_v2", {"law": "uniform_bounded_v1"}, None],
)
def test_a_different_region_law_is_refused(declared):
    payload = {"proposal": {"shallow": {"region_law": declared}}}
    with pytest.raises(StateRoutingContractError):
        assert_declared_region_configuration(payload)


@pytest.mark.parametrize("override", ["proposal", "region_law", "state_routing"])
def test_a_per_cell_proposal_override_is_refused(override):
    """One controller means one configuration. A per-cell override would make the
    panel a family of controllers again, so it is made unrepresentable rather than
    merely discouraged."""

    payload = {
        "proposal": {"shallow": {"region_law": DECLARED_REGION_LAW}},
        "cells": [
            {"cell": "fa7_0", "smiles": "CCO"},
            {"cell": "fa7_1", "smiles": "CCN", override: {"anything": 1}},
        ],
    }
    with pytest.raises(StateRoutingContractError, match="fa7_1"):
        assert_declared_region_configuration(payload)


def test_a_real_shipped_contract_with_the_declaration_added_is_accepted():
    """The real panel rows carry no override, so the shipped cells pass."""

    declaration = assert_declared_region_configuration(_contract("5ht1b"))
    assert declaration["per_cell_overrides"] == []
    assert declaration["declared_for_all_applicable_states"] is True


# ---- hop 2: the routed expansion over REAL T4 sources --------------------


def test_fa7_0_routes_to_the_region_kernel_and_its_rung_zero_lane():
    """MEASURED: `fa7_0` is neutral, so the region kernel keeps its support and the
    routing allocates rung 0 to the zero-support fallback."""

    outcome, record, seen = _stub_expansion("fa7", ["fa7_0"])
    assert record["routed_kernels"] == ["region"]
    assert record["routed_lanes"] == ["zero_support_fallback"]
    assert seen == [
        {
            "parent_index": 0,
            "parent": SOURCES["fa7_0"],
            "kernel": "region",
            "lane": "zero_support_fallback",
        }
    ]
    assert outcome.fallback_ran is True


def test_5ht1b_2_routes_to_the_state_aware_kernel_and_its_rung_zero_lane():
    """MEASURED: `5ht1b_2` carries net charge +1, where the executor's charge policy
    refuses 9 of 16 region excisions, so rung 0 goes to the protonation-aware lane."""

    _, record, seen = _stub_expansion("5ht1b", ["5ht1b_2"])
    assert record["routed_kernels"] == ["state_aware"]
    assert record["routed_lanes"] == ["protonation_aware_retained_subgraph"]
    assert seen[0]["kernel"] == "state_aware"
    assert seen[0]["lane"] == "protonation_aware_retained_subgraph"


def test_every_parent_is_handed_to_rung_zero_with_its_own_route():
    """One assignment per parent, in order, each carrying the four declared keys.

    The parent list is deliberately MIXED -- a neutral source and a charged one in
    one event -- because a same-kernel list cannot distinguish "each parent is
    routed on its own chemistry" from "every parent gets one hard-coded lane". A
    fixture that never reaches the second kernel does not test the dispatch, and
    this one was vacuous until a mutation that collapsed every kernel onto the
    region lane survived it.

    `parents` is a list of SMILES and the routing reads the MOLECULE, so a parent
    need not appear in the contract's own cell rows.
    """

    parents = [SOURCES[cell] for cell in ("fa7_0", "5ht1b_2", "fa7_2")]
    _, record, seen = _stub_expansion("fa7", [], parents=parents)
    assert [item["parent_index"] for item in seen] == [0, 1, 2]
    assert [item["parent"] for item in seen] == parents
    assert [item["kernel"] for item in seen] == ["region", "state_aware", "region"]
    assert [item["lane"] for item in seen] == [
        "zero_support_fallback",
        "protonation_aware_retained_subgraph",
        "zero_support_fallback",
    ]
    for item in seen:
        assert set(item) == {"parent_index", "parent", "kernel", "lane"}
        assert item["lane"] == ROUTED_RUNG_ZERO_LANE[item["kernel"]]
    assert len({item["kernel"] for item in seen}) == 2, (
        "both kernels must appear or this test cannot see a collapsed dispatch"
    )
    assert record["routed_kernels"] == ["region", "state_aware"]
    assert [row["parent_index"] for row in record["routing"]] == [0, 1, 2]
    # The parent SMILES is deliberately absent from the recorded assignments: the
    # record is published into a round lock and carries the route, not the molecule.
    assert all("parent" not in row for row in record["rung_zero_assignments"])


def test_all_three_charged_sources_route_to_the_same_kernel():
    """MEASURED: all three 5HT1B sources are cations, so a rule that had secretly
    keyed on `5ht1b_2` -- the only one of the three that historically exhausted --
    would separate them here."""

    _, record, seen = _stub_expansion("5ht1b", ["5ht1b_0", "5ht1b_1", "5ht1b_2"])
    assert record["routed_kernels"] == ["state_aware"]
    assert {item["kernel"] for item in seen} == {"state_aware"}
    # The LANE is asserted too, not just the kernel: a `lane_for` collapsed onto one
    # lane leaves every kernel correct and hands a charged parent the excision lane
    # its executor refuses, which a kernel-only assertion cannot see.
    assert record["routed_lanes"] == ["protonation_aware_retained_subgraph"]
    assert {item["lane"] for item in seen} == {"protonation_aware_retained_subgraph"}


def test_an_empty_parent_list_is_refused():
    """An expansion with no rung-0 stage would publish a terminal record that looks
    like a completed event."""

    with pytest.raises(StateRoutingContractError, match="at least one parent"):
        routed_support_expansion(
            contract=_contract("fa7"),
            parents=[],
            rounds_completed=1,
            rung_zero=lambda assignments: ([], {}),
            escalate=lambda draws, attempt: [],
        )


def test_an_injected_router_or_ladder_is_recorded_as_injected():
    """The injection points exist only for the probes, so a probe run can never be
    published as a campaign result -- hop 3 refuses a record built from one."""

    _, record, _ = _stub_expansion("fa7", ["fa7_0"], router=UnifiedRouter())
    assert record["router_source"] == "injected"
    assert record["ladder_source"] == "contract"
    with pytest.raises(UnifiedExpansionNotConsumed, match="router_source"):
        assert_unified_expansion_is_consumed(record, ExpansionOutcome(stop_reason="x"))


# ---- hop 2: the consumption probe ----------------------------------------


def _route_with(router, attempt, *, contract=None):
    routed_support_expansion(
        contract=contract if contract is not None else _contract("fa7"),
        parents=[SOURCES["fa7_0"]],
        rounds_completed=attempt,
        rung_zero=lambda assignments: ([], {}),
        escalate=lambda draws, attempt_index: [],
        router=router,
    )


def test_a_live_wiring_is_observed_on_the_first_attempt():
    """The router is consulted before any RNG is drawn, so the check is
    deterministic: for a given code state it always passes or always fails.

    SCOPE, measured by mutation rather than assumed: the probe raises from
    `decision` AND from `lane_for`, so this observes that the injected router is
    consulted AT ALL, not that the decision call specifically survives. A path that
    skipped `decision` and still resolved a lane would pass here -- it is the
    behavioural routing tests above that refuse it.
    """

    contract = _contract("fa7")
    attempts = assert_state_routing_is_consumed(
        lambda router, attempt: _route_with(router, attempt, contract=contract)
    )
    assert attempts == 1


def test_a_path_that_ignores_the_router_it_is_handed_is_refused():
    """The inert-mechanism failure, made loud. A closure that runs the expansion
    from the contract's own router never consults the probe."""

    contract = _contract("fa7")

    def ignores(router, attempt):
        _route_with(None, attempt, contract=contract)

    with pytest.raises(StateRoutingNotConsumed, match="without the path consulting"):
        assert_state_routing_is_consumed(ignores)


def test_the_probe_survives_the_five_exception_types_the_draw_path_catches():
    """BEHAVIOURAL, and it is the guard that matters.

    `t4_fiber_campaign.expand` catches ValueError / RuntimeError / KeyError /
    IndexError / TypeError per draw and `synthesize_dynamic_program` catches
    ValueError per module family, so a probe raising any of them would be swallowed
    somewhere downstream of the very call it exists to observe. This closure
    swallows exactly those five and the probe must still be seen.
    """

    contract = _contract("fa7")

    def swallowing(router, attempt):
        try:
            _route_with(router, attempt, contract=contract)
        except SWALLOWED:  # pragma: no cover - green only if the probe is swallowed
            pass

    assert assert_state_routing_is_consumed(swallowing) == 1


def test_the_probe_exception_is_none_of_the_swallowed_types():
    """STRUCTURAL companion to the behavioural guard above."""

    for swallowed in SWALLOWED:
        assert not issubclass(_RouterProbeConsumed, swallowed), swallowed.__name__
    assert issubclass(_RouterProbeConsumed, Exception)
    assert not issubclass(StateRoutingNotConsumed, _RouterProbeConsumed)


def test_the_probe_needs_at_least_one_attempt():
    with pytest.raises(ValueError, match="at least one attempt"):
        assert_state_routing_is_consumed(lambda router, attempt: None, attempts=0)


# ---- hop 3: nothing terminal without a routed expansion ------------------


def _terminal():
    """A real routed expansion's record and outcome, from the production path."""

    outcome, record, _ = _stub_expansion("fa7", ["fa7_0"])
    return record, outcome


def test_a_real_routed_expansion_is_accepted_as_terminal():
    record, outcome = _terminal()
    assert outcome.stop_reason == "ladder_exhausted"
    assert_unified_expansion_is_consumed(record, outcome)


def test_a_missing_record_is_refused():
    _, outcome = _terminal()
    with pytest.raises(UnifiedExpansionNotConsumed, match="no routed-expansion record"):
        assert_unified_expansion_is_consumed(None, outcome)
    with pytest.raises(UnifiedExpansionNotConsumed):
        assert_unified_expansion_is_consumed({"schema_version": "x"}, None)


def test_a_foreign_schema_version_is_refused():
    record, outcome = _terminal()
    with pytest.raises(UnifiedExpansionNotConsumed, match="schema"):
        assert_unified_expansion_is_consumed(
            {**record, "schema_version": "t4_support_expansion_v1"}, outcome
        )


def test_an_injected_router_source_is_refused():
    record, outcome = _terminal()
    with pytest.raises(UnifiedExpansionNotConsumed, match="router_source"):
        assert_unified_expansion_is_consumed(
            {**record, "router_source": "injected"}, outcome
        )


def test_an_injected_ladder_source_is_refused():
    record, outcome = _terminal()
    with pytest.raises(UnifiedExpansionNotConsumed, match="ladder_source"):
        assert_unified_expansion_is_consumed(
            {**record, "ladder_source": "injected"}, outcome
        )


def test_a_ladder_that_is_not_the_frozen_one_is_refused():
    record, outcome = _terminal()
    assert record["ladder"] == {
        "policy": FROZEN_LADDER_ID,
        "policy_sha256": FROZEN_LADDER_SHA256,
    }
    stale = {"policy": FROZEN_LADDER_ID, "policy_sha256": "0" * 64}
    with pytest.raises(UnifiedExpansionNotConsumed, match="not the frozen"):
        assert_unified_expansion_is_consumed({**record, "ladder": stale}, outcome)


def test_a_router_that_is_not_the_pinned_one_is_refused():
    record, outcome = _terminal()
    stale = {"policy": ROUTER_ID, "policy_sha256": "0" * 64}
    with pytest.raises(UnifiedExpansionNotConsumed, match="ran router"):
        assert_unified_expansion_is_consumed({**record, "router": stale}, outcome)


def test_an_expansion_with_no_routing_decision_is_refused():
    record, outcome = _terminal()
    with pytest.raises(UnifiedExpansionNotConsumed, match="no routing decision"):
        assert_unified_expansion_is_consumed({**record, "routing": []}, outcome)


def test_a_decision_naming_an_unregistered_kernel_is_refused():
    record, outcome = _terminal()
    rogue = [{**record["routing"][0], "activated_kernel": "local"}]
    with pytest.raises(UnifiedExpansionNotConsumed, match="which is not one of"):
        assert_unified_expansion_is_consumed({**record, "routing": rogue}, outcome)


def test_a_decision_taken_on_a_non_empty_pool_is_refused():
    """An expansion may only run on an exhausted pool, where the ramp is exactly
    one; a ramp below it means the routing and this caller disagree about the
    search state."""

    record, outcome = _terminal()
    rogue = [{**record["routing"][0], "support_ramp": 0.0}]
    with pytest.raises(UnifiedExpansionNotConsumed, match="support_ramp"):
        assert_unified_expansion_is_consumed({**record, "routing": rogue}, outcome)


def test_the_expansions_own_stop_reason_check_is_delegated_not_duplicated():
    """DELEGATION, proven by the exception TYPE.

    `assert_support_expansion_is_consumed` raises `SupportExpansionNotConsumed`,
    which is not a `UnifiedExpansionNotConsumed`, so this test goes red the moment
    the delegation is dropped. The record is otherwise valid, and `fallback_ran` is
    True so only the stop-reason clause can refuse -- a bare `ExpansionOutcome()`
    trips two clauses at once and would survive dropping either.
    """

    record, _ = _terminal()
    with pytest.raises(SupportExpansionNotConsumed):
        assert_unified_expansion_is_consumed(
            record, ExpansionOutcome(stop_reason="not_run", fallback_ran=True)
        )


# ---- The non-trigger guarantee -------------------------------------------


def test_a_healthy_round_consults_no_alternate_kernel():
    """`support_ramp` is exactly zero while the pool is non-empty, so on any round
    that produced an eligible candidate the weights are (1, 0, 0) and the controller
    is byte-identical to the primary lane. Structural, not statistical."""

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.t4_unified_routing import PROPOSAL_SLOTS

    for cell in ("fa7_0", "5ht1b_2"):
        applicability = molecular_applicability(
            pad_molecular_graph(smiles_to_molecular_graph(SOURCES[cell]), PROPOSAL_SLOTS)
        )
        for pool in (1, 8, 64):
            healthy = SearchProgress(
                eligible_pool_size=pool, consecutive_empty_rounds=0, rounds_completed=3
            )
            assert kernel_weights(applicability, healthy) == {
                "local": 1.0,
                "region": 0.0,
                "state_aware": 0.0,
            }, cell
            assert activated_kernel(applicability, healthy) is None, cell
        empty = SearchProgress(
            eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=3
        )
        assert activated_kernel(applicability, empty) is not None, (
            f"{cell} activates no kernel on an empty pool, so the test above is vacuous"
        )


def test_the_router_refuses_a_kernel_with_no_registered_rung_zero_lane():
    with pytest.raises(StateRoutingContractError, match="no rung-0 lane"):
        UnifiedRouter().lane_for("local")


def test_the_shipped_contracts_are_read_not_transcribed():
    """A guard against this file drifting from the panel it claims to test."""

    assert set(SOURCES) == {
        "fa7_0", "fa7_1", "fa7_2", "5ht1b_0", "5ht1b_1", "5ht1b_2",
    }
    for relative in CONTRACTS.values():
        assert json.loads((ROOT / relative).read_text())["payload"]["delta"] == 0.6
