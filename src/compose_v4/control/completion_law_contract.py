"""The executable contract surface that selects a completion law.

WHY A CONTRACT AND NOT A BARE KEYWORD
-------------------------------------
``compile_generic_module`` accepts a ``completion_law=`` keyword that defaults
to ``None``, and ``None`` is v1 verbatim.  A repair behind an opt-in keyword is
INERT until a production caller passes it, and this repository has paid for that
exact shape more than once: a region law that was measured, tested and merged
while no production caller passed it; ``donor_program`` absent from the scored
entry point's import closure; ``allocation_priority`` with zero call sites.

This module is the one place that turns a declared arm name into a law object,
and :func:`assert_completion_law_is_consumed` is the one place that proves the
production path reaches the draw site.  The check RUNS the caller's own proposal
closure rather than inspecting a signature, because a keyword can exist and be
dropped one hop later and ``inspect.signature`` cannot see that.

THE ARMS
--------
    absent / None          v1 verbatim, byte-identical
    "scale_only_v1"        octave-balanced scale, v1 linear C/N/O content
    "content_only_v1"      v1 bounded scale, structured component content
    "scale_content_v1"     octave-balanced scale AND component content

An ABSENT arm is the OFF state and means ``completion_law=None``, never "a law
that happens to reproduce v1".  With ``None`` the excision draw consumes
``rng.permutation``; with any region law it consumes ``rng.random`` through
``BridgeRegionLaw.order``.  Those are different RNG streams, so a law object
named "off" would move every existing run while reading as a no-op.  That is why
no such arm is registered.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from compose_v4.control.completion_component_law import (
    ComponentBank,
    CompletionLaw,
    ScaleLaw,
)

SCHEMA_VERSION = "pmo_completion_law_contract_v1"

#: The contract key and the lane whose block carries it. The shallow and
#: structured program lanes are the only ones that reach a generic module.
CONTRACT_FIELD = "completion_law"
CONTRACT_LANE = "shallow"

#: The committed objective-blind component bank.
BANK_PATH = "configs/pmo_completion_component_bank_v1.json"

SCALE_ONLY_V1 = "scale_only_v1"
CONTENT_ONLY_V1 = "content_only_v1"
SCALE_CONTENT_V1 = "scale_content_v1"
REGISTERED_COMPLETION_LAWS = (SCALE_ONLY_V1, CONTENT_ONLY_V1, SCALE_CONTENT_V1)

#: v1's bound, quoted so a counterfactual can name it.
V1_MAXIMUM = 8


class CompletionLawContractError(ValueError):
    """The field is declared but the runtime cannot honour it."""


@lru_cache(maxsize=4)
def load_bank(path: str = BANK_PATH) -> ComponentBank:
    location = Path(path)
    if not location.is_absolute():
        location = _repo_root() / path
    if not location.exists():
        raise CompletionLawContractError(f"component bank not found: {location}")
    return ComponentBank.load(location)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_completion_law(declared: Any, *, bank_path: str = BANK_PATH):
    """Turn a declared contract value into a law object, or ``None``.

    Accepts ``None`` (OFF), a registered arm name, or an already-built
    :class:`CompletionLaw`, which is what the zero-oracle harness passes.
    """

    if declared is None:
        return None
    if isinstance(declared, CompletionLaw):
        return declared
    if not isinstance(declared, str):
        raise CompletionLawContractError(
            f"completion law must be a registered name or None, not {declared!r}"
        )
    if declared not in REGISTERED_COMPLETION_LAWS:
        raise CompletionLawContractError(
            f"unregistered completion law {declared!r}; "
            f"registered: {REGISTERED_COMPLETION_LAWS}"
        )
    if declared == SCALE_ONLY_V1:
        return CompletionLaw(
            scale=ScaleLaw(maximum=None),
            bank=None,
            expand_excision=True,
            name=SCALE_ONLY_V1,
        )
    if declared == CONTENT_ONLY_V1:
        return CompletionLaw(
            scale=ScaleLaw(maximum=V1_MAXIMUM),
            bank=load_bank(bank_path),
            expand_excision=False,
            name=CONTENT_ONLY_V1,
        )
    return CompletionLaw(
        scale=ScaleLaw(maximum=None),
        bank=load_bank(bank_path),
        expand_excision=True,
        name=SCALE_CONTENT_V1,
    )


def completion_law_identity(law) -> dict:
    """A recordable description of whatever the runtime actually resolved."""

    if law is None:
        return {"arm": None, "state": "v1_verbatim"}
    return {
        "arm": law.name,
        "scale_maximum": law.scale.maximum,
        "expand_excision": bool(law.expand_excision),
        "bank_identity_sha256": None if law.bank is None else law.bank.identity_sha256,
        "bank_components": 0 if law.bank is None else len(law.bank.components),
    }


class _CompletionLawProbe(Exception):
    """Raised from inside the law to prove the production path reached it.

    Deliberately NOT a ``ValueError``/``RuntimeError``/``KeyError``/
    ``IndexError``/``TypeError``: ``compile_generic_module`` is wrapped in a
    per-family ``except ValueError`` and the proposal loops catch the rest, so
    any of those would be swallowed by the very path being observed.
    """


class _ProbeScale(ScaleLaw):
    def draw(self, rng, capacity: int) -> int:  # noqa: D102
        raise _CompletionLawProbe("completion scale law consulted")


def assert_completion_law_is_consumed(
    draw: Callable[[Any], Any],
    *,
    seeds: tuple[int, ...] = tuple(range(16)),
) -> dict:
    """Require the caller's own proposal path to consult a completion law.

    ``draw`` receives one law object and must run exactly the production
    proposal path the campaign will run.  Seeds are FIXED rather than random:
    ``_weighted_module_order`` permutes thirteen families and only two route
    through a completion, so a single draw can legitimately miss the law.  With
    fixed seeds the check always passes or always fails for a given code state,
    so a failure is a wiring defect and never an unlucky draw.
    """

    import numpy as np

    probe = CompletionLaw(
        scale=_ProbeScale(maximum=None),
        bank=None,
        expand_excision=False,
        name="consumption_probe",
    )
    for attempt, seed in enumerate(seeds, start=1):
        try:
            draw(probe, np.random.default_rng(seed))
        except _CompletionLawProbe:
            return {
                "schema_version": SCHEMA_VERSION,
                "consumed": True,
                "attempts": attempt,
                "seeds_tried": list(seeds[:attempt]),
            }
        except Exception:  # noqa: BLE001 - an ordinary refusal; try the next seed
            continue
    raise CompletionLawContractError(
        "the completion law was never consulted by the supplied proposal path; "
        "the field is declared but the runtime drops it"
    )
