"""The executable contract surface that makes the unified T4 controller REACHABLE.

WHAT THIS MODULE IS FOR
-----------------------
Two mechanisms were built, measured and tested, and neither had a production
caller:

* :mod:`compose_v4.control.t4_unified_routing` decides, from the MOLECULAR state
  alone, which proposal kernel an exhausted round allocates compute to.
* :mod:`compose_v4.control.frozen_proposal_escalation` is the one frozen,
  target-agnostic escalation ladder an exhausted round climbs.

A mechanism with no production caller is inert, and this repository has paid for
that seven times -- the bridge region law behind an opt-in keyword, the
``exact_early_ring`` scheduler, ``allocation_priority`` with zero call sites,
``donor_program`` absent from the scored entry point's import closure, a
completion law dropped by a drop-in replacement adapter, and the two above.  The
tell in every case was available statically and cost one grep: search for the
symbol at CALL sites, never at definition sites.

This module is the one place a signed contract turns into those two objects, the
one place that refuses to proceed when a contract declares them and the runtime
cannot honour them, and the one function an empty candidate pool goes through.

THE THREE HOPS, AND WHY EACH IS GUARDED SEPARATELY
---------------------------------------------------
Two call sites threading one parameter can leave a consultation test green when
the parameter is dropped at one of them, so each hop carries its own guard:

  hop 1  contract -> objects    :func:`resolve_state_routing` and
                                ``resolve_frozen_escalation`` REFUSE a contract
                                that does not name both. A contract that carries
                                a ladder number, or a routing parameter, is
                                unrepresentable rather than merely discouraged.
  hop 2  objects -> draw site   :func:`assert_state_routing_is_consumed` installs
                                a probe router and requires the production
                                function to consult it. The probe raises
                                :class:`_RouterProbeConsumed`, which is
                                deliberately NOT a ``ValueError`` /
                                ``RuntimeError`` / ``KeyError`` / ``IndexError``
                                / ``TypeError``: ``synthesize_dynamic_program``
                                catches ``ValueError`` per module family and
                                ``t4_fiber_campaign.expand`` catches all five per
                                draw, so any of those would be swallowed by the
                                very path the probe is trying to observe.
  hop 3  draw site -> terminal  :func:`assert_unified_expansion_is_consumed`
                                refuses to let a cell publish
                                ``candidate_exhaustion`` without a record this
                                module produced, naming the frozen ladder and
                                carrying a routing decision per parent. A round
                                loop that quietly returns to the bare hard stop
                                therefore cannot end a cell.

``inspect.signature`` is not any of these.  A keyword can be accepted and dropped
one hop later: this repository has a launch receipt reading ``SPAWNED_ALL`` while
nothing ran, and a contract declaring ``charged_calls_per_task: 250`` beside a
module constant of 1000.  A signature check passes both.

WHAT THE ROUTING MAY READ, AND WHY THAT IS STRUCTURAL
------------------------------------------------------
:func:`routed_support_expansion` hands the router a PARENT SMILES and a COUNT.
``t4_unified_routing.kernel_weights`` accepts two frozen dataclasses whose fields
are enumerated in that module, and neither can carry a target name, a protein, a
cell id, a seed index, a docking score or an archive value.  Adding such a field
is the only way to smuggle one in, and ``tests/test_t4_unified_routing.py`` fails
if the field sets move.  :data:`ROUTER_SHA256` hashes those field sets, so a
contract that pins it also pins the guarantee.

WHAT IT SPENDS
--------------
CPU, never oracle calls.  The routed expansion returns proposal records that the
caller locks and docks through the unchanged round path, so charged-call
accounting, the per-query receipts, the no-retry / no-replacement / no-backfill
policy and the budget ceiling are untouched.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.frozen_proposal_escalation import (
    FROZEN_LADDER_ID,
    FROZEN_LADDER_SHA256,
    resolve_frozen_escalation,
)
from compose_v4.control.region_law_contract import FREE_GATE_MARGIN_V1
from compose_v4.control.t4_unified_routing import (
    KERNELS,
    PROPOSAL_SLOTS,
    MolecularApplicability,
    SearchProgress,
    molecular_applicability,
    routing_decision,
)
from compose_v4.control.t4_unified_routing import (
    SCHEMA_VERSION as ROUTING_SCHEMA_VERSION,
)
from compose_v4.experiments.t4_support_expansion import (
    ExpansionOutcome,
    SupportExpansionPolicy,
    assert_support_expansion_is_consumed,
    run_support_expansion,
)

SCHEMA_VERSION = "t4_unified_controller_v1"

#: The contract block that NAMES the routing. A contract names it; it never
#: describes one, for the same reason `frozen_proposal_escalation` refuses a
#: contract that carries ladder numbers: a block that can hold a parameter can
#: hold a different one per target, and then the panel is a family of
#: controllers rather than one.
ROUTING_CONTRACT_FIELD = "state_routing"

#: The name a contract uses to request the routing.
ROUTER_ID = ROUTING_SCHEMA_VERSION

#: The rung-0 proposal lane each routed kernel allocates compute to.
#:
#: ``region``      the production zero-support fallback: excise each
#:                 bridge-separated region through the executor and gate the
#:                 EXECUTED endpoint, never entering the goal-abstraction layer.
#: ``state_aware`` the protonation-aware retained-subgraph expert, which is the
#:                 kernel that keeps support where the executor's charge policy
#:                 refuses a region excision.
ROUTED_RUNG_ZERO_LANE = {
    "region": "zero_support_fallback",
    "state_aware": "protonation_aware_retained_subgraph",
}

#: The lane whose contract block must carry the declared region configuration,
#: and the field it carries. Mirrors `region_law_contract`, which owns the
#: field's semantics; this module only asserts the unified panel declares it.
REGION_CONFIGURATION_LANE = "shallow"
REGION_CONFIGURATION_FIELD = "region_law"
DECLARED_REGION_LAW = FREE_GATE_MARGIN_V1

#: Attempts the routing consumption probe makes. The check is DETERMINISTIC --
#: the router is consulted before any RNG is drawn -- so one attempt already
#: decides it; several are made only because a caller's probe closure may vary
#: the parent it routes, and a fixed count keeps the check reproducible: for a
#: given code state it always passes or always fails.
ROUTING_PROBE_ATTEMPTS = 4

#: The frozen description of the routing RULE, hashed into `ROUTER_SHA256`.
#:
#: It deliberately pins the FIELD SETS of the two dataclasses the rule reads,
#: because those field sets ARE the guarantee that no target identity and no
#: reward can reach the routing. A contract pinning this hash therefore pins the
#: guarantee, not merely the rule's name.
ROUTER_DESCRIPTION = {
    "schema_version": SCHEMA_VERSION,
    "router": ROUTER_ID,
    "kernels": list(KERNELS),
    "molecular_applicability_fields": [f.name for f in fields(MolecularApplicability)],
    "search_progress_fields": [f.name for f in fields(SearchProgress)],
    "rung_zero_lane": dict(sorted(ROUTED_RUNG_ZERO_LANE.items())),
    "proposal_slots": PROPOSAL_SLOTS,
}

#: The content hash a contract pins. A silent edit to the rule's inputs or to the
#: kernel/lane mapping invalidates every contract that names it.
ROUTER_SHA256 = identity(ROUTER_DESCRIPTION)

#: The exact block a contract must carry. Nothing else is accepted.
_REQUIRED_ROUTING_KEYS = frozenset({"policy", "policy_sha256"})


class StateRoutingContractError(ValueError):
    """The contract's state-routing block cannot be honoured as written."""


class StateRoutingNotConsumed(StateRoutingContractError):
    """The contract declares routing the running path never consults."""


class UnifiedExpansionNotConsumed(StateRoutingContractError):
    """A terminal result was reached without a routed expansion producing it."""


class _RouterProbeConsumed(Exception):
    """Signalled the instant the production path consults the probe router.

    Deliberately NOT a ``ValueError`` / ``RuntimeError`` / ``KeyError`` /
    ``IndexError`` / ``TypeError``: ``synthesize_dynamic_program`` catches
    ``ValueError`` per module family and ``t4_fiber_campaign.expand`` catches all
    five per draw, so any of those would be swallowed somewhere downstream of the
    very call the probe exists to observe. It also inherits from ``Exception``
    rather than ``BaseException`` so an ordinary ``except Exception`` in a test
    harness still shows it rather than unwinding the interpreter.
    """


# ---- The router ----------------------------------------------------------


@dataclass(frozen=True)
class UnifiedRouter:
    """The production router: molecular state in, rung-0 kernel out.

    It is a thin, frozen adapter over :mod:`compose_v4.control.t4_unified_routing`
    and holds no state of its own, so two cells routed in the same process cannot
    influence each other.

    The parent arrives as a SMILES and is padded to the T4 proposal slot capacity
    before measurement. That padding is load-bearing rather than cosmetic: a
    tight graph deletes the whole insertion family from the legal support, which
    would change every count the applicability record carries.
    ``molecular_applicability`` refuses a source that is not at
    :data:`~compose_v4.control.t4_unified_routing.PROPOSAL_SLOTS` slots rather
    than accepting one quietly.
    """

    policy: str = ROUTER_ID
    policy_sha256: str = ROUTER_SHA256

    def applicability(self, parent_smiles: str) -> MolecularApplicability:
        """Measure the parent with the PRODUCTION excision path."""

        source = pad_molecular_graph(
            smiles_to_molecular_graph(parent_smiles), PROPOSAL_SLOTS
        )
        return molecular_applicability(source)

    def decision(
        self,
        parent_smiles: str,
        *,
        eligible_pool_size: int,
        rounds_completed: int,
        consecutive_empty_rounds: int = 1,
    ) -> dict:
        """The full auditable routing record for one parent.

        ``eligible_pool_size`` is a COUNT of admitted candidates, never their
        scores, which is what makes the ramp reward-free.
        """

        search = SearchProgress(
            eligible_pool_size=int(eligible_pool_size),
            consecutive_empty_rounds=int(consecutive_empty_rounds),
            rounds_completed=int(rounds_completed),
        )
        return routing_decision(self.applicability(parent_smiles), search)

    @staticmethod
    def lane_for(kernel: str) -> str:
        """The rung-0 proposal lane a routed kernel allocates compute to."""

        try:
            return ROUTED_RUNG_ZERO_LANE[kernel]
        except KeyError:
            raise StateRoutingContractError(
                f"no rung-0 lane is registered for routed kernel {kernel!r}; "
                f"registered kernels are {sorted(ROUTED_RUNG_ZERO_LANE)}"
            ) from None

    def as_record(self) -> dict:
        return {"policy": self.policy, "policy_sha256": self.policy_sha256}


class _RouterProbe:
    """A stand-in router that records being consulted and stops the expansion.

    It reports on ``decision``, which is the single method
    :func:`routed_support_expansion` calls, and raises immediately so a live
    wiring costs one partial call rather than a full expansion event -- and, more
    importantly, so the probe can never reach a rung-0 fan-out and never spend
    the CPU a real expansion would.
    """

    policy = ROUTER_ID
    policy_sha256 = ROUTER_SHA256

    def __init__(self) -> None:
        self.consulted = 0

    def applicability(self, parent_smiles: str):
        raise _RouterProbeConsumed("applicability measurement reached the probe")

    def decision(self, parent_smiles: str, **kwargs) -> dict:
        self.consulted += 1
        raise _RouterProbeConsumed("the expansion path consulted the router")

    @staticmethod
    def lane_for(kernel: str) -> str:
        raise _RouterProbeConsumed("lane resolution reached the probe")

    def as_record(self) -> dict:
        return {"policy": self.policy, "policy_sha256": self.policy_sha256}


# ---- Reading the contract ------------------------------------------------


def state_routing_block() -> dict:
    """The ``state_routing`` block a unified-panel contract must carry."""

    return {"policy": ROUTER_ID, "policy_sha256": ROUTER_SHA256}


def state_routing_request(payload: Mapping) -> Any:
    """The raw contract block, or ``None`` when the field is absent.

    One function knows the field's path, so a future move of the key cannot
    leave a second reader addressing the old location.
    """

    return payload.get(ROUTING_CONTRACT_FIELD)


def resolve_state_routing(payload: Mapping) -> UnifiedRouter:
    """The production router, or a refusal naming what the contract did wrong.

    There is no ``None`` return. A unified-panel contract that does not name the
    routing is refused, because a run that silently falls back to a single
    hard-coded kernel while its contract says otherwise is exactly the defect
    this surface exists to stop.
    """

    block = state_routing_request(payload)
    if block is None:
        raise StateRoutingContractError(
            f"the contract declares no {ROUTING_CONTRACT_FIELD!r} block; a unified "
            "controller contract must name the state routing so an exhausted round "
            "allocates compute from the molecular state rather than from a kernel "
            "chosen by hand"
        )
    if not isinstance(block, Mapping):
        raise StateRoutingContractError(
            f"{ROUTING_CONTRACT_FIELD!r} must be a mapping, got {type(block).__name__}"
        )
    keys = frozenset(block)
    if keys != _REQUIRED_ROUTING_KEYS:
        extra = sorted(keys - _REQUIRED_ROUTING_KEYS)
        missing = sorted(_REQUIRED_ROUTING_KEYS - keys)
        raise StateRoutingContractError(
            f"{ROUTING_CONTRACT_FIELD!r} must be exactly "
            f"{sorted(_REQUIRED_ROUTING_KEYS)} (unexpected {extra}, missing "
            f"{missing}); a block that can carry a routing parameter can carry a "
            "different one per target, which is the freedom this panel exists to "
            "remove"
        )
    if block["policy"] != ROUTER_ID:
        raise StateRoutingContractError(
            f"{ROUTING_CONTRACT_FIELD}.policy must be {ROUTER_ID!r}, "
            f"got {block['policy']!r}"
        )
    if block["policy_sha256"] != ROUTER_SHA256:
        raise StateRoutingContractError(
            f"{ROUTING_CONTRACT_FIELD}.policy_sha256 pins "
            f"{block['policy_sha256']!r} but the routing rule hashes to "
            f"{ROUTER_SHA256!r}; either the rule's inputs moved or the contract was "
            "sealed against a different one"
        )
    return UnifiedRouter()


def assert_declared_region_configuration(payload: Mapping) -> dict:
    """Refuse a unified contract whose declared region configuration is absent.

    ``q_region`` is the region kernel, and the panel declared -- in writing,
    before the deciding measurement -- that it carries
    ``proposal.shallow.region_law = "free_gate_margin_v1"`` whenever that kernel
    activates.  The declaration is honoured for ALL applicable molecular states
    and never for named cells, so this also refuses a contract in which any cell
    row carries its own proposal configuration: a per-cell override would make
    the panel a family of controllers again, and the cheapest way to keep that
    honest is to make it unrepresentable.

    ``region_law_contract`` owns the field's semantics and proves the runtime
    consumes it; this function only asserts the unified panel declares it.
    """

    lane = (payload.get("proposal") or {}).get(REGION_CONFIGURATION_LANE)
    if not isinstance(lane, Mapping):
        raise StateRoutingContractError(
            f"the contract has no proposal.{REGION_CONFIGURATION_LANE} block to "
            f"carry the declared {REGION_CONFIGURATION_FIELD!r}"
        )
    declared = lane.get(REGION_CONFIGURATION_FIELD)
    name = declared.get("law") if isinstance(declared, Mapping) else declared
    if name != DECLARED_REGION_LAW:
        raise StateRoutingContractError(
            f"proposal.{REGION_CONFIGURATION_LANE}.{REGION_CONFIGURATION_FIELD} must "
            f"declare {DECLARED_REGION_LAW!r} for the unified panel, got "
            f"{declared!r}; the historical arms that produced the wins ran the "
            "shallow lane conditioned by it"
        )
    offenders = sorted(
        str(row.get("cell"))
        for row in payload.get("cells", ())
        if isinstance(row, Mapping)
        and ({"proposal", REGION_CONFIGURATION_FIELD, ROUTING_CONTRACT_FIELD} & set(row))
    )
    if offenders:
        raise StateRoutingContractError(
            f"cell rows {offenders} carry their own proposal configuration; the "
            "unified panel is one controller, so a per-cell override is refused "
            "rather than merely discouraged"
        )
    return {
        REGION_CONFIGURATION_FIELD: declared,
        "declared_for_all_applicable_states": True,
        "per_cell_overrides": [],
    }


def resolve_unified_controller(
    payload: Mapping,
) -> tuple[UnifiedRouter, SupportExpansionPolicy, dict]:
    """The whole unified-controller contract policy, in one call.

    Returns ``(router, ladder, declaration)``.  Every entry point resolves
    through here so a contract missing either half fails on the first worker that
    reads it, rather than on whichever one happens to need it first.
    """

    router = resolve_state_routing(payload)
    ladder = resolve_frozen_escalation(payload)
    declaration = assert_declared_region_configuration(payload)
    return router, ladder, declaration


# ---- The production empty-pool path --------------------------------------


def routed_support_expansion(
    *,
    contract: Mapping,
    parents: Sequence[str],
    rounds_completed: int,
    rung_zero: Callable[[Sequence[dict]], tuple[Iterable[dict], Mapping]],
    escalate: Callable[[int, int], Iterable[dict]],
    already_seen: Sequence[str] = (),
    router: Any = None,
    ladder: SupportExpansionPolicy | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[ExpansionOutcome, dict]:
    """Run the routed, frozen escalation for a round whose pool came back empty.

    This is THE production empty-pool handler.  The Modal campaign app calls it
    from the one branch that used to publish ``candidate_exhaustion`` directly,
    and the zero-oracle gate calls the same function, so the gate measures the
    code the campaign executes rather than a transcription of it.

    ``rung_zero(assignments)`` runs rung 0 over the routed lane assignments and
    returns its records plus load-independent work counters.  Each assignment is
    ``{"parent_index", "parent", "kernel", "lane"}``.  ``escalate(draws, attempt)``
    runs one ladder step over the frozen lanes.  Both are the caller's production
    fan-outs injected as callables rather than transcriptions of them, which is
    what lets one function serve a Modal fan-out and a serial local driver.

    ``router`` and ``ladder`` exist ONLY so the consumption probes can install a
    stand-in.  An injected object is recorded as such, and
    :func:`assert_unified_expansion_is_consumed` refuses a terminal result built
    from one, so a probe run can never be published as a campaign result.

    Charges nothing.
    """

    if not parents:
        raise StateRoutingContractError(
            "a routed expansion needs at least one parent to route; an empty parent "
            "list would produce an expansion with no rung-0 stage and a terminal "
            "record that looks like a completed event"
        )
    router_source = "injected" if router is not None else "contract"
    ladder_source = "injected" if ladder is not None else "contract"
    resolved_router = router if router is not None else resolve_state_routing(contract)
    resolved_ladder = ladder if ladder is not None else resolve_frozen_escalation(contract)

    assignments: list[dict] = []
    decisions: list[dict] = []
    for parent_index, parent in enumerate(parents):
        # The router is consulted HERE, before any RNG is drawn and before any
        # proposal work is done, which is where the consumption probe lands.
        decision = resolved_router.decision(
            parent,
            eligible_pool_size=0,
            rounds_completed=int(rounds_completed),
        )
        kernel = decision.get("activated_kernel")
        if kernel is None:
            # `support_ramp` is 1.0 on an empty pool and the applicability mask
            # always permits exactly one alternate kernel, so this is
            # unreachable by construction. It is a raise rather than a silent
            # skip because a parent that routes nowhere would otherwise vanish
            # from the expansion with no record of why.
            raise StateRoutingContractError(
                f"parent {parent_index} routed to no kernel on an empty pool; the "
                "applicability mask always permits exactly one, so the routing rule "
                "and this caller disagree about the search state"
            )
        lane = resolved_router.lane_for(kernel)
        assignments.append(
            {
                "parent_index": parent_index,
                "parent": parent,
                "kernel": kernel,
                "lane": lane,
            }
        )
        decisions.append({"parent_index": parent_index, **decision})

    fallback = None
    if resolved_ladder.zero_support_fallback:

        def fallback(*, _assignments=tuple(assignments)):
            produced, work = rung_zero(_assignments)
            return list(produced), dict(work)

    outcome = run_support_expansion(
        resolved_ladder,
        escalate=escalate,
        fallback=fallback,
        already_seen=already_seen,
        clock=clock,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "router": dict(resolved_router.as_record()),
        "router_source": router_source,
        "ladder": {"policy": FROZEN_LADDER_ID, "policy_sha256": FROZEN_LADDER_SHA256},
        "ladder_source": ladder_source,
        "ladder_policy": resolved_ladder.as_record(),
        "rounds_completed": int(rounds_completed),
        "routing": decisions,
        "rung_zero_assignments": [
            {k: v for k, v in item.items() if k != "parent"} for item in assignments
        ],
        "routed_kernels": sorted({item["kernel"] for item in assignments}),
        "routed_lanes": sorted({item["lane"] for item in assignments}),
        "expansion": outcome.as_record(),
    }
    return outcome, record


# ---- Proving the runtime consumes it -------------------------------------


def assert_state_routing_is_consumed(
    route: Callable[[Any, int], Any], *, attempts: int = ROUTING_PROBE_ATTEMPTS
) -> int:
    """Run the caller's own expansion path until it consults a router.

    ``route(router, attempt)`` must execute ONE production empty-pool decision
    with ``router`` installed exactly where the contract's routing goes.  The
    probe raises from ``decision`` the moment the path reaches it, so a healthy
    wiring returns before any rung-0 fan-out starts and costs no proposal work.

    Returns the number of attempts it took, which the caller should record: it is
    evidence the check RAN rather than that a field parsed.

    Raises :class:`StateRoutingNotConsumed` if ``attempts`` deterministic calls
    all complete without the router being consulted.  That is the loud failure
    the contract asks for -- the alternative is charging an oracle budget to a
    run whose declared routing was never installed.
    """

    if attempts < 1:
        raise ValueError("consumption probe needs at least one attempt")
    probe = _RouterProbe()
    for attempt in range(attempts):
        try:
            route(probe, attempt)
        except _RouterProbeConsumed:
            return attempt + 1
    raise StateRoutingNotConsumed(
        f"the contract requests {ROUTING_CONTRACT_FIELD!r} but {attempts} production "
        "empty-pool calls completed without the path consulting it; the runtime "
        "cannot honour this contract and must not spend oracle calls"
    )


def assert_unified_expansion_is_consumed(
    record: Mapping | None, outcome: ExpansionOutcome | None
) -> None:
    """Refuse a terminal result produced without the routed frozen expansion.

    This is hop 3, and it is what stops a round loop from quietly returning to
    the bare hard stop: candidate exhaustion can only be published when this
    module produced the record, the record names the frozen ladder, the routing
    was resolved from the CONTRACT rather than injected by a probe, and every
    parent carries a routing decision naming a registered kernel.

    It delegates the expansion's own stop-reason check to
    ``assert_support_expansion_is_consumed`` rather than duplicating it, so the
    two cannot drift apart.
    """

    if record is None or outcome is None:
        raise UnifiedExpansionNotConsumed(
            "the terminal path was reached with no routed-expansion record; the "
            "round loop did not go through `routed_support_expansion` and must not "
            "publish candidate exhaustion"
        )
    if record.get("schema_version") != SCHEMA_VERSION:
        raise UnifiedExpansionNotConsumed(
            f"the expansion record carries schema {record.get('schema_version')!r}, "
            f"not {SCHEMA_VERSION!r}; it was not produced by this module"
        )
    for key in ("router_source", "ladder_source"):
        if record.get(key) != "contract":
            raise UnifiedExpansionNotConsumed(
                f"{key} is {record.get(key)!r}, not 'contract'; a probe or an "
                "injected object produced this record and it must not be published "
                "as a campaign result"
            )
    ladder = record.get("ladder") or {}
    if (
        ladder.get("policy") != FROZEN_LADDER_ID
        or ladder.get("policy_sha256") != FROZEN_LADDER_SHA256
    ):
        raise UnifiedExpansionNotConsumed(
            f"the expansion ran ladder {ladder!r}, not the frozen "
            f"{FROZEN_LADDER_ID!r} at {FROZEN_LADDER_SHA256!r}"
        )
    router = record.get("router") or {}
    if (
        router.get("policy") != ROUTER_ID
        or router.get("policy_sha256") != ROUTER_SHA256
    ):
        raise UnifiedExpansionNotConsumed(
            f"the expansion ran router {router!r}, not {ROUTER_ID!r} at "
            f"{ROUTER_SHA256!r}"
        )
    decisions = record.get("routing") or []
    if not decisions:
        raise UnifiedExpansionNotConsumed(
            "the expansion record carries no routing decision; an exhausted round "
            "must record which kernel each parent was routed to"
        )
    for decision in decisions:
        kernel = decision.get("activated_kernel")
        if kernel not in ROUTED_RUNG_ZERO_LANE:
            raise UnifiedExpansionNotConsumed(
                f"parent {decision.get('parent_index')} recorded kernel {kernel!r}, "
                f"which is not one of {sorted(ROUTED_RUNG_ZERO_LANE)}"
            )
        if decision.get("support_ramp") != 1.0:
            raise UnifiedExpansionNotConsumed(
                f"parent {decision.get('parent_index')} recorded support_ramp "
                f"{decision.get('support_ramp')!r}; an expansion may only run on an "
                "exhausted pool, where the ramp is exactly one"
            )
    assert_support_expansion_is_consumed(outcome)
