"""The executable contract surface that selects a replace-completion law.

This mirrors :mod:`compose_v4.control.region_law_contract` deliberately and
closely.  That module turns a signed contract field into the law that conditions
the EXCISION half of ``segment_replace``; this one does the same for the
COMPLETION half, which was measured to be the remaining unconditioned step.

THE FIELD
---------
``payload["proposal"]["shallow"]["completion_law"]``.  It sits under ``shallow``
for the same reason the region field does: that lane, and only that lane,
reaches ``synthesize_dynamic_program``, which is the only entry point that
threads a law to the construction site.

    absent            ->  ``None``  ->  historical behaviour, byte-identical
    "free_gate_margin_v1"            ->  the measured repair, default tilt
    {"law": "free_gate_margin_v1", "candidates": .., "floor": .., "temperature": ..}

AN ABSENT FIELD IS THE OFF STATE, and OFF means ``law=None``.  This is
load-bearing exactly as it is for the region field: with ``law=None`` the
construction site draws ONE length and ONE chain, consuming one
``rng.integers`` plus the growth draws; with any law object it draws
``candidates`` of them and then consults ``order``.  Those are different RNG
streams, so even a uniform completion law would reproduce the SUPPORT but not
the DRAWS of the historical path.  No uniform law is registered here, for the
same reason none is registered there: a name that looked like "off" but moved
every existing run would be the silent drift this surface exists to prevent.

WHY A CONSUMPTION CHECK
-----------------------
A keyword can exist and be dropped one hop later, and this repository has paid
for that repeatedly -- most recently when a validated region repair sat INERT
behind a keyword no production caller passed.  ``inspect.signature`` would have
passed it.  :func:`assert_completion_law_is_consumed` therefore RUNS the
caller's own proposal path and observes the construction site consulting the
law.

The completion site is reached strictly less often than the region site: the
module order must select ``segment_replace``, the excision half must succeed,
and at least one candidate completion must execute.  The attempt budget is
larger than the region contract's for that reason alone, and the seeds are
fixed so the check stays deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compose_v4.control.bridge_region_law import (
    FreeFeasibilityGate,
    free_gate_margin_law,
)
from compose_v4.control.replace_completion_law import (
    DEFAULT_CANDIDATES,
    ReplaceCompletionLaw,
    free_gate_margin_completion_law,
)

SCHEMA_VERSION = "t4_completion_law_contract_v1"

#: The contract key, and the lane whose block carries it.
CONTRACT_FIELD = "completion_law"
CONTRACT_LANE = "shallow"

#: The only registered law. Its name is the arm name the gate harness measured.
FREE_GATE_MARGIN_V1 = "free_gate_margin_v1"
REGISTERED_COMPLETION_LAWS = (FREE_GATE_MARGIN_V1,)

#: Attempts the consumption check makes before declaring the law unreachable.
#: Larger than the region contract's budget because the completion site sits
#: strictly downstream of the excision site and of the module-order draw.
CONSUMPTION_PROBE_ATTEMPTS = 64


class CompletionLawContractError(ValueError):
    """The contract's completion-law field cannot be honoured as written."""


class CompletionLawNotConsumed(CompletionLawContractError):
    """The contract requested a law the running proposal path never consults."""


class _ProbeConsumed(Exception):
    """Signalled the instant the production construction site consults the probe.

    Deliberately NOT a ``ValueError``/``RuntimeError``/``KeyError``/
    ``IndexError``/``TypeError``: ``compile_generic_module`` and
    ``synthesize_dynamic_program`` catch ``ValueError`` per family and
    ``t4_fiber_campaign.expand`` catches all five per draw, so any of those
    would be swallowed by the very path the probe is trying to observe.
    """


class _ConsumptionProbe:
    """A stand-in law that records being consulted and stops the construction.

    ``candidates`` is deliberately 1: the probe only needs the site to reach
    ``order``, and a single candidate makes a live wiring cost one executor
    replay rather than sixteen.
    """

    candidates = 1

    def __init__(self) -> None:
        self.consulted = 0

    def weights(self, endpoints):
        raise _ProbeConsumed("completion weighting reached the probe")

    def order(self, endpoints, rng) -> list[int]:
        self.consulted += 1
        raise _ProbeConsumed("the construction site consulted the completion law")


# ---- Reading the field ---------------------------------------------------


def completion_law_request(payload: dict) -> Any:
    """The raw contract request, or ``None`` when the field is absent."""

    lane = (payload.get("proposal") or {}).get(CONTRACT_LANE) or {}
    if not isinstance(lane, dict):
        raise CompletionLawContractError(
            f"proposal.{CONTRACT_LANE} must be a mapping to carry {CONTRACT_FIELD!r}"
        )
    return lane.get(CONTRACT_FIELD)


def assert_no_unconsumable_completion_law_request(payload: dict) -> None:
    """Refuse a completion-law field parked on a lane that cannot consume one."""

    if CONTRACT_FIELD in payload:
        raise CompletionLawContractError(
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
        raise CompletionLawContractError(
            f"{CONTRACT_FIELD!r} declared on lane(s) {misplaced} that cannot consume "
            f"it; only proposal.{CONTRACT_LANE} reaches a synthesizer that threads a "
            "completion law to the construction site"
        )


def _parameters(request: Any) -> dict:
    if isinstance(request, str):
        return {"law": request}
    if isinstance(request, dict):
        if "law" not in request:
            raise CompletionLawContractError(
                f"{CONTRACT_FIELD!r} block must name its law under the 'law' key"
            )
        return dict(request)
    raise CompletionLawContractError(
        f"{CONTRACT_FIELD!r} must be a law name or a block naming one, got "
        f"{type(request).__name__}"
    )


def build_completion_law(
    request: Any, *, delta: float, reference_smiles: str
) -> ReplaceCompletionLaw:
    """Construct the law a contract request names.

    The endpoint margin is taken from the PRODUCTION region-law factory rather
    than rebuilt here, so the two halves of ``segment_replace`` are conditioned
    by byte-identical chemistry and a change to one cannot silently diverge from
    the other.
    """

    parameters = _parameters(request)
    name = parameters.pop("law")
    if name not in REGISTERED_COMPLETION_LAWS:
        raise CompletionLawContractError(
            f"unknown completion law {name!r}; registered laws are "
            f"{REGISTERED_COMPLETION_LAWS}"
        )
    unknown = set(parameters) - {"candidates", "floor", "temperature"}
    if unknown:
        raise CompletionLawContractError(
            f"unknown {CONTRACT_FIELD!r} parameters {sorted(unknown)}"
        )
    gate = FreeFeasibilityGate(delta=float(delta))
    margin = free_gate_margin_law(gate, reference_smiles).margin
    if margin is None:  # pragma: no cover - the factory always sets one
        raise CompletionLawContractError("the region-law factory returned no margin")
    return free_gate_margin_completion_law(margin, **parameters)


def resolve_completion_law(
    payload: dict, *, delta: float, reference_smiles: str
) -> ReplaceCompletionLaw | None:
    """The contract's law, or ``None`` when the field is absent.

    ``None`` is returned only for an ABSENT field. A field that is present but
    unusable raises, because a run that silently degrades while its contract
    says otherwise is the defect this surface exists to stop.
    """

    request = completion_law_request(payload)
    if request is None:
        return None
    return build_completion_law(
        request, delta=delta, reference_smiles=reference_smiles
    )


# ---- Proving the runtime consumes it -------------------------------------


def assert_completion_law_is_consumed(
    draw: Callable[[Any, int], Any], *, attempts: int = CONSUMPTION_PROBE_ATTEMPTS
) -> int:
    """Run the caller's proposal path until the construction site consults a law.

    ``draw(law, seed)`` must execute ONE production proposal draw with ``law``
    installed exactly where the contract says it goes. The probe raises from
    ``order`` the moment the construction site reaches it, so a healthy wiring
    returns after a partial synthesis.

    Returns the number of attempts it took, which the caller should record: it
    is evidence the check ran rather than that the field parsed.
    """

    if attempts < 1:
        raise ValueError("consumption probe needs at least one attempt")
    probe = _ConsumptionProbe()
    for seed in range(attempts):
        try:
            draw(probe, seed)
        except _ProbeConsumed:
            return seed + 1
    raise CompletionLawNotConsumed(
        f"the contract requests {CONTRACT_FIELD!r} but {attempts} production "
        "proposal draws completed without the construction site consulting it; "
        "the runtime cannot honour this contract and must not spend oracle calls"
    )


def completion_law_for_proposal_lane(
    payload: dict,
    *,
    lane: str,
    delta: float,
    reference_smiles: str,
    draw: Callable[[Any, int], Any],
    attempts: int = CONSUMPTION_PROBE_ATTEMPTS,
) -> tuple[ReplaceCompletionLaw | None, dict]:
    """The whole contract policy for one proposal worker, in one call."""

    assert_no_unconsumable_completion_law_request(payload)
    if lane != CONTRACT_LANE:
        return None, {}
    law = resolve_completion_law(
        payload, delta=delta, reference_smiles=reference_smiles
    )
    if law is None:
        return None, {}
    consumed = assert_completion_law_is_consumed(draw, attempts=attempts)
    return law, {
        CONTRACT_FIELD: completion_law_request(payload),
        "completion_law_consumption_attempts": consumed,
        "completion_law_candidates": int(law.candidates),
    }


#: Re-exported so a caller can name the default without importing the law module.
__DEFAULT_CANDIDATES = DEFAULT_CANDIDATES


def default_candidates() -> int:
    """The K a bare ``"free_gate_margin_v1"`` request resolves to."""

    return __DEFAULT_CANDIDATES
