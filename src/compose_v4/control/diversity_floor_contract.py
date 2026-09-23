"""The executable contract surface that selects a PMO structural-diversity floor.

WHAT THIS MODULE IS FOR
-----------------------
:mod:`compose_v4.control.structural_diversity_floor` implements the floor, and
:class:`~compose_v4.control.pmo_population_controller.PmoPopulationController`
accepts it through a ``diversity_floor=`` keyword that defaults to ``None``.
``None`` is the shipped controller verbatim, so the floor is INERT until some
caller passes one.  This module is the one place that turns a signed contract
field into a floor, and the one place that refuses to proceed when the field is
declared but the runtime cannot honour it.

THE FIELD
---------
``payload["population"]["diversity_floor"]``.

    absent                        ->  ``None``  ->  the shipped controller,
                                      byte-identically
    "archive_cluster_floor_v1"    ->  the measured floor at its declared
                                      defaults
    {"floor": "archive_cluster_floor_v1", "max_clusters": .., "attempt_share":
     .., "pool_slots": ..}

An ABSENT field is the OFF state, and OFF means ``diversity_floor=None``, not
"a floor configured to do nothing".  The distinction is load-bearing and is the
same one the region-law contract records: with ``None`` the controller builds
its parent schedules with ``attempts_per_batch`` consecutive ``_parent()``
draws; with any floor object some of those positions become cluster-restricted
draws, which consume the same generator differently.  A floor with
``attempt_share`` small enough to reserve nothing would still reserve one slot
-- ``reserved_slot_count`` floors at one -- so no "identity" floor is
registered here.  A name that looked like OFF but moved every existing run is
exactly the silent drift this field exists to prevent.

WHY THE FIELD SITS UNDER ``population`` AND NOT UNDER A PROPOSAL LANE
--------------------------------------------------------------------
The floor is a population-level policy: it reserves schedule positions across
ALL THREE proposal lanes from ONE archive partition, and it reserves
selected-batch slots in an allocation step that runs once per batch after the
lanes have merged.  A lane block cannot consume it -- no lane sees the other
lanes' schedules or the merged pool -- so a ``diversity_floor`` key parked on a
lane, or at the top level, would be read by nobody while the signed contract
claimed the floor was active.  :func:`assert_no_unconsumable_diversity_floor_request`
refuses those placements.

WHY A CONSUMPTION CHECK AND NOT A SIGNATURE CHECK
-------------------------------------------------
A keyword argument can exist and be dropped one hop later.  This repository has
paid for that repeatedly: a region law validated, tested and merged behind a
keyword no production caller passed; a completion law that reached 3.8% of
charged candidates because a drop-in replacement adapter forwarded every other
argument and not that one.  ``inspect.signature`` passes both.

:func:`assert_diversity_floor_is_consumed` therefore RUNS the caller's own
proposal path and observes BOTH hops.  The two hops are checked separately,
because they share one floor object and a probe that stopped at the first hop
would leave the second untested -- the exact near-miss the region-law wiring
hit when two call sites shared a sink.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from compose_v4.control.structural_diversity_floor import (
    ARCHIVE_CLUSTER_FLOOR_V1,
    DiversityFloorError,
    StructuralDiversityFloor,
)

SCHEMA_VERSION = "pmo_diversity_floor_contract_v1"

#: The contract key, and the surface whose block carries it.
CONTRACT_FIELD = "diversity_floor"
CONTRACT_SURFACE = "population"

REGISTERED_DIVERSITY_FLOORS = (ARCHIVE_CLUSTER_FLOOR_V1,)

#: Parameters a contract block may set, beyond naming the floor.
FLOOR_PARAMETERS = ("max_clusters", "attempt_share", "pool_slots")


class DiversityFloorContractError(ValueError):
    """The contract's diversity-floor field cannot be honoured as written."""


class DiversityFloorNotConsumed(DiversityFloorContractError):
    """The contract requested a floor the running proposal path never consults."""


class _ProbeConsumed(Exception):
    """Signalled the instant the production allocation step consults the probe.

    Deliberately NOT a ``ValueError``/``RuntimeError``/``KeyError``/
    ``IndexError``/``TypeError``: the channel pool generators catch ``ValueError``
    per attempt and the jump lane catches ``ValueError`` and ``RuntimeError``,
    so any of those could be swallowed by the very path the probe is trying to
    observe.
    """


class _ConsumptionProbe:
    """A floor that delegates to a real one and records being consulted.

    ``attempt_plan`` records and DELEGATES, because the schedule it returns has
    to be usable or the run cannot reach the second hop at all.  ``pool_plan``
    records and RAISES, so a live wiring costs one proposal batch rather than a
    full campaign round, and so a dropped second hop is distinguishable from a
    dropped first one by which counter is zero.
    """

    def __init__(self, inner: StructuralDiversityFloor) -> None:
        self.inner = inner
        self.partition_consulted = 0
        self.attempt_plan_consulted = 0
        self.pool_plan_consulted = 0

    def partition(self, endpoints: Sequence[str]):
        self.partition_consulted += 1
        return self.inner.partition(endpoints)

    @staticmethod
    def cluster_of_endpoint(partition: Sequence[Sequence[str]]) -> dict[str, int]:
        return StructuralDiversityFloor.cluster_of_endpoint(partition)

    def reserved_slot_count(self, slots: int) -> int:
        return self.inner.reserved_slot_count(slots)

    def attempt_plan(self, **kwargs):
        self.attempt_plan_consulted += 1
        return self.inner.attempt_plan(**kwargs)

    def pool_plan(self, **kwargs):
        self.pool_plan_consulted += 1
        raise _ProbeConsumed("the allocation step consulted the diversity floor")

    @property
    def pool_slots(self) -> int:
        return self.inner.pool_slots

    def payload(self) -> dict[str, object]:
        return self.inner.payload()


# ---- Reading the field ---------------------------------------------------


def diversity_floor_request(payload: Mapping[str, Any]) -> Any:
    """The raw contract request, or ``None`` when the field is absent.

    One function knows the field's path, so a future move of the key cannot
    leave a second reader addressing the old location.
    """

    surface = (payload.get(CONTRACT_SURFACE) or {}) if isinstance(payload, Mapping) else {}
    if not isinstance(surface, Mapping):
        raise DiversityFloorContractError(
            f"{CONTRACT_SURFACE} must be a mapping to carry {CONTRACT_FIELD!r}"
        )
    return surface.get(CONTRACT_FIELD)


def assert_no_unconsumable_diversity_floor_request(payload: Mapping[str, Any]) -> None:
    """Refuse a diversity-floor field parked where nothing can consume it.

    The floor is population level.  A ``diversity_floor`` key at the top of the
    contract, or inside a ``proposal`` lane block, would be read by nobody and
    would silently degrade the run to the shipped controller while the signed
    contract said otherwise.
    """

    if CONTRACT_FIELD in payload:
        raise DiversityFloorContractError(
            f"{CONTRACT_FIELD!r} is a population field; no top-level reader consumes "
            f"it. Declare it under {CONTRACT_SURFACE}.{CONTRACT_FIELD}"
        )
    proposal = payload.get("proposal") or {}
    if not isinstance(proposal, Mapping):
        return
    misplaced = sorted(
        lane
        for lane, block in proposal.items()
        if isinstance(block, Mapping) and CONTRACT_FIELD in block
    )
    if misplaced:
        raise DiversityFloorContractError(
            f"{CONTRACT_FIELD!r} declared on proposal lane(s) {misplaced}; a lane sees "
            "neither the other lanes' schedules nor the merged pool, so it cannot "
            f"consume a population floor. Declare it under {CONTRACT_SURFACE}."
            f"{CONTRACT_FIELD}"
        )


def _parameters(request: Any) -> dict:
    if isinstance(request, str):
        return {"floor": request}
    if isinstance(request, Mapping):
        if "floor" not in request:
            raise DiversityFloorContractError(
                f"{CONTRACT_FIELD!r} block must name its floor under the 'floor' key"
            )
        return dict(request)
    raise DiversityFloorContractError(
        f"{CONTRACT_FIELD!r} must be a floor name or a block naming one, got "
        f"{type(request).__name__}"
    )


def build_diversity_floor(request: Any) -> StructuralDiversityFloor:
    """Construct the floor a contract request names."""

    parameters = _parameters(request)
    name = parameters.pop("floor")
    if name not in REGISTERED_DIVERSITY_FLOORS:
        raise DiversityFloorContractError(
            f"unknown diversity floor {name!r}; registered floors are "
            f"{REGISTERED_DIVERSITY_FLOORS}"
        )
    unknown = set(parameters) - set(FLOOR_PARAMETERS)
    if unknown:
        raise DiversityFloorContractError(
            f"unknown {CONTRACT_FIELD!r} parameters {sorted(unknown)}"
        )
    try:
        return StructuralDiversityFloor(**parameters)
    except DiversityFloorError as error:
        raise DiversityFloorContractError(str(error)) from error


def resolve_diversity_floor(
    payload: Mapping[str, Any],
) -> StructuralDiversityFloor | None:
    """The contract's floor, or ``None`` when the field is absent.

    ``None`` is returned only for an ABSENT field.  A field that is present but
    unusable raises, because a run that silently degrades to the shipped
    controller while its contract says otherwise is the defect this whole
    surface exists to stop.
    """

    assert_no_unconsumable_diversity_floor_request(payload)
    request = diversity_floor_request(payload)
    if request is None:
        return None
    return build_diversity_floor(request)


# ---- Proving the runtime consumes it -------------------------------------


def assert_diversity_floor_is_consumed(
    propose: Callable[[Any], Any], *, floor: StructuralDiversityFloor
) -> dict[str, int]:
    """Run the caller's proposal path and observe BOTH floor hops.

    ``propose(floor)`` must execute ONE production proposal batch with ``floor``
    installed exactly where the contract says it goes.  The probe delegates the
    schedule hop and raises from the allocation hop, so a healthy wiring returns
    after one batch.

    Returns the consultation counts, which the caller should record: they are
    evidence the check RAN rather than that the field parsed.

    Raises :class:`DiversityFloorNotConsumed` naming WHICH hop was missed.  The
    two are reported separately on purpose: they share one floor object, and a
    check that stopped at the first hop would let the second be dropped
    silently.
    """

    probe = _ConsumptionProbe(floor)
    try:
        propose(probe)
    except _ProbeConsumed:
        pass
    if not probe.attempt_plan_consulted:
        raise DiversityFloorNotConsumed(
            f"the contract requests {CONTRACT_FIELD!r} but one production proposal "
            "batch built its parent schedules without consulting the floor; the "
            "attempt reservation is not wired and the run must not spend oracle calls"
        )
    if not probe.pool_plan_consulted:
        raise DiversityFloorNotConsumed(
            f"the contract requests {CONTRACT_FIELD!r} and the schedule hop is wired, "
            "but one production proposal batch allocated its selected slots without "
            "consulting the floor; the eligible-pool reservation is not wired and the "
            "run must not spend oracle calls"
        )
    return {
        "partition_consulted": probe.partition_consulted,
        "attempt_plan_consulted": probe.attempt_plan_consulted,
        "pool_plan_consulted": probe.pool_plan_consulted,
    }
