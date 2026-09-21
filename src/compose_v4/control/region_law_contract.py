"""The executable contract surface that selects a bridge region-draw law.

WHAT THIS MODULE IS FOR
-----------------------
:mod:`compose_v4.control.bridge_region_law` implements the repaired region-draw
law, and ``dynamic_program_synthesis`` accepts it through a ``region_law=``
keyword that defaults to ``None``.  ``None`` is v1 verbatim, so the repair is
INERT until some caller passes a law.  This module is the one place that turns a
signed contract field into that law, and the one place that refuses to proceed
when the field is declared but the runtime cannot honour it.

THE FIELD
---------
``payload["proposal"]["shallow"]["region_law"]``.  It sits under the ``shallow``
lane because that lane -- and only that lane -- reaches
``synthesize_dynamic_program``, which is the only entry point that threads a law
to the draw site.  Putting the field anywhere more general would declare a
capability over lanes that cannot consume it.

    absent            ->  ``None``  ->  v1 verbatim, byte-identical
    "free_gate_margin_v1"            ->  the measured repair, default tilt
    {"law": "free_gate_margin_v1", "maximum": null, "floor": .., "temperature": ..}

An ABSENT field is the OFF state, and OFF means ``law=None``, not "a uniform
law".  The distinction is load-bearing and measured: with ``law=None`` the draw
site consumes ``rng.permutation``; with any law object it consumes
``rng.random`` through ``BridgeRegionLaw.order``.  Those are different RNG
streams, so ``UNIFORM_BOUNDED_V1`` reproduces v1's SUPPORT but not v1's draws.
Only the absent field is byte-identical, which is why no uniform law is
registered here: a name that looked like "off" but moved every existing run
would be exactly the silent drift this field exists to prevent.

WHY A CONSUMPTION CHECK AND NOT A SIGNATURE CHECK
-------------------------------------------------
A keyword argument can exist and be dropped one hop later.  This repository has
already paid for that twice: a launch receipt that read ``SPAWNED_ALL`` while
nothing ran, and a contract declaring ``charged_calls_per_task: 250`` beside a
runtime that used a module constant of 1000.  ``inspect.signature`` would have
passed both.

:func:`assert_region_law_is_consumed` therefore RUNS the caller's own proposal
path and observes the draw site consulting the law.  The caller supplies the
``draw`` callable, so what is verified is the path the campaign will actually
execute rather than a transcription of it -- and this module never imports an
experiment module to do it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.bridge_region_law import (
    BridgeRegion,
    BridgeRegionLaw,
    FreeFeasibilityGate,
    free_gate_margin_law,
)

SCHEMA_VERSION = "t4_region_law_contract_v1"

#: The contract key, and the lane whose block carries it.
CONTRACT_FIELD = "region_law"
CONTRACT_LANE = "shallow"

#: The only registered law. Its name is the arm name the gate harness measured.
FREE_GATE_MARGIN_V1 = "free_gate_margin_v1"
REGISTERED_REGION_LAWS = (FREE_GATE_MARGIN_V1,)

#: Attempts the consumption check makes before declaring the law unreachable.
#: Seeds are fixed, so the check is deterministic: for a given code state it
#: always passes or always fails, and a failure is a real wiring defect rather
#: than an unlucky draw. The module order is a permutation over thirteen
#: families and the two that route through the law are tried whenever they
#: precede the first family that succeeds, so a live wiring is observed within
#: the first attempt or two.
CONSUMPTION_PROBE_ATTEMPTS = 16


class RegionLawContractError(ValueError):
    """The contract's region-law field cannot be honoured as written."""


class RegionLawNotConsumed(RegionLawContractError):
    """The contract requested a law the running proposal path never consults."""


class _ProbeConsumed(Exception):
    """Signalled the instant the production draw site consults the probe.

    Deliberately NOT a ``ValueError``/``RuntimeError``/``KeyError``/
    ``IndexError``/``TypeError``: ``synthesize_dynamic_program`` catches
    ``ValueError`` per module family and ``t4_fiber_campaign.expand`` catches
    all five per draw, so any of those would be swallowed by the very path the
    probe is trying to observe.
    """


class _ConsumptionProbe:
    """A stand-in law that records being consulted and stops the draw.

    It reports on ``order``, which is the single method
    ``_delete_pendant_fragment`` calls, and raises immediately so a live wiring
    costs one partial synthesis rather than a full expansion.
    """

    def __init__(self) -> None:
        self.consulted = 0

    def regions(self, graph: MolecularGraph) -> tuple[BridgeRegion, ...]:
        raise _ProbeConsumed("region enumeration reached the probe")

    def weights(self, graph: MolecularGraph, regions):
        raise _ProbeConsumed("region weighting reached the probe")

    def order(self, graph: MolecularGraph, rng) -> list[BridgeRegion]:
        self.consulted += 1
        raise _ProbeConsumed("the draw site consulted the region law")


# ---- Reading the field ---------------------------------------------------


def region_law_request(payload: dict) -> Any:
    """The raw contract request, or ``None`` when the field is absent.

    One function knows the field's path, so a future move of the key cannot
    leave a second reader addressing the old location.
    """

    lane = (payload.get("proposal") or {}).get(CONTRACT_LANE) or {}
    if not isinstance(lane, dict):
        raise RegionLawContractError(
            f"proposal.{CONTRACT_LANE} must be a mapping to carry {CONTRACT_FIELD!r}"
        )
    return lane.get(CONTRACT_FIELD)


def assert_no_unconsumable_region_law_request(payload: dict) -> None:
    """Refuse a region-law field parked on a lane that cannot consume one.

    Only the ``shallow`` lane reaches ``synthesize_dynamic_program``, so a
    ``region_law`` key under any other lane -- or at the top level of the
    contract -- would be read by nobody and would silently degrade the run to
    v1 while the signed contract said otherwise.  Every proposal worker calls
    this, including the ones whose own lane could never honour the field, so the
    refusal happens on the worker that would have ignored it.
    """

    if CONTRACT_FIELD in payload:
        raise RegionLawContractError(
            f"{CONTRACT_FIELD!r} is a lane field; no top-level reader consumes it. "
            f"Declare it under proposal.{CONTRACT_LANE}"
        )
    proposal = payload.get("proposal") or {}
    if not isinstance(proposal, dict):
        return
    misplaced = sorted(
        lane
        for lane, block in proposal.items()
        if lane != CONTRACT_LANE and isinstance(block, dict) and CONTRACT_FIELD in block
    )
    if misplaced:
        raise RegionLawContractError(
            f"{CONTRACT_FIELD!r} declared on lane(s) {misplaced} that cannot consume "
            f"it; only proposal.{CONTRACT_LANE} reaches a synthesizer that threads a "
            "region law to the draw site"
        )


def _parameters(request: Any) -> dict:
    if isinstance(request, str):
        return {"law": request}
    if isinstance(request, dict):
        if "law" not in request:
            raise RegionLawContractError(
                f"{CONTRACT_FIELD!r} block must name its law under the 'law' key"
            )
        return dict(request)
    raise RegionLawContractError(
        f"{CONTRACT_FIELD!r} must be a law name or a block naming one, got "
        f"{type(request).__name__}"
    )


def build_region_law(
    request: Any, *, delta: float, reference_smiles: str
) -> BridgeRegionLaw:
    """Construct the law a contract request names.

    ``reference_smiles`` is the similarity reference the task declares -- the
    ORIGINAL benchmark source for the cell, never the current parent -- so the
    law cannot drift its own target as a campaign walks away from the root.
    """

    parameters = _parameters(request)
    name = parameters.pop("law")
    if name not in REGISTERED_REGION_LAWS:
        raise RegionLawContractError(
            f"unknown region law {name!r}; registered laws are {REGISTERED_REGION_LAWS}"
        )
    unknown = set(parameters) - {"maximum", "floor", "temperature"}
    if unknown:
        raise RegionLawContractError(
            f"unknown {CONTRACT_FIELD!r} parameters {sorted(unknown)}"
        )
    gate = FreeFeasibilityGate(delta=float(delta))
    return free_gate_margin_law(gate, reference_smiles, **parameters)


def resolve_region_law(
    payload: dict, *, delta: float, reference_smiles: str
) -> BridgeRegionLaw | None:
    """The contract's law, or ``None`` when the field is absent.

    ``None`` is returned only for an ABSENT field. A field that is present but
    unusable raises, because a run that silently degrades to v1 while its
    contract says otherwise is the defect this whole surface exists to stop.
    """

    request = region_law_request(payload)
    if request is None:
        return None
    return build_region_law(request, delta=delta, reference_smiles=reference_smiles)


# ---- Proving the runtime consumes it -------------------------------------


def assert_region_law_is_consumed(
    draw: Callable[[Any, int], Any], *, attempts: int = CONSUMPTION_PROBE_ATTEMPTS
) -> int:
    """Run the caller's proposal path until the draw site consults a law.

    ``draw(law, seed)`` must execute ONE production proposal draw with ``law``
    installed exactly where the contract says it goes. The probe raises from
    ``order`` the moment the draw site reaches it, so a healthy wiring returns
    after a partial synthesis.

    Returns the number of attempts it took, which the caller should record: it
    is evidence the check ran rather than that the field parsed.

    Raises :class:`RegionLawNotConsumed` if ``attempts`` deterministic draws all
    complete without the law being consulted. That is the loud failure the
    contract asks for -- the alternative is charging an oracle budget to a run
    whose declared repair was never installed.
    """

    if attempts < 1:
        raise ValueError("consumption probe needs at least one attempt")
    probe = _ConsumptionProbe()
    for seed in range(attempts):
        try:
            draw(probe, seed)
        except _ProbeConsumed:
            return seed + 1
    raise RegionLawNotConsumed(
        f"the contract requests {CONTRACT_FIELD!r} but {attempts} production "
        "proposal draws completed without the draw site consulting it; the "
        "runtime cannot honour this contract and must not spend oracle calls"
    )
