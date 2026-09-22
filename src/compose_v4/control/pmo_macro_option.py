"""Bounded, explicit protection for the bridge state of a declared macro option.

A MACRO OPTION is a complete, end-to-end EXECUTED and validated transformation

    G0 -> G_bridge -> G_destination

whose total primitive count exceeds what one proposal round can realize, so it can
only be reached by passing through `G_bridge` and continuing from there in a later
round.  Executing and validating the whole thing costs ZERO oracle calls -- the
executor is the authority -- so the controller can COMMIT to the complete macro
before it asks whether the bridge deserves to stay in the population.

What this module changes, and what it deliberately does not
----------------------------------------------------------
MEASURED: `ProgramOptimizer.entries` has no eviction, pruning, capacity bound or
removal path.  Its only write site is insert-if-absent.  A bridge is therefore
never EVICTED, and nothing here claims to prevent an eviction that cannot happen.
The two real ways a bridge is pruned are both about SELECTION:

  1. EXPANSION STARVATION.  `ProgramOptimizer.selection` weights an endpoint by
     `1 / score_rank`, so a bridge that scored worse than its own parent -- which
     is the whole point of a bridge -- receives near-zero parent mass and is never
     drawn for expansion again.
  2. ALLOCATION DISCARD.  `_allocate` keeps `candidates_per_batch` of a strictly
     larger merged pool, so a continuation proposed FROM the bridge can be
     generated, ranked low because its parent scored low, and dropped before it is
     ever charged.

`protected_parent_weights` addresses (1) and `reserve_continuation_slots`
addresses (2).  Both are pure functions of their inputs so they can be tested
without a controller, and both are RE-RANKINGS: every candidate the ordinary
policy could have chosen keeps positive probability, and no state is ever removed.

Invariants maintained here
--------------------------
* Protection is BOUNDED and EXPLICIT.  `protection_rounds` is a named integer,
  recorded in the payload, validated into `[1, PROTECTION_ROUNDS_CEILING]` at
  declaration, and decremented once per advanced round.  There is no path that
  extends a window, and no blanket exemption.
* Protection is RELEASED as soon as it is pointless: the moment the destination is
  charged, or the window runs out.  An expired option can never be revived.
* Charging is unchanged.  Nothing here makes a call free, defers a reservation, or
  moves a query outside the ledger; the registry only ever OBSERVES endpoints the
  ledger has already charged.
* Every state the registry touches is an executed, committed, complete molecule.
  This is a selection policy, not a change to the validity contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compose_v4.control.docking_value import identity

SCHEMA = "pmo_macro_option_registry_v1"

# A ceiling the configuration cannot exceed, so no future caller can turn a bounded
# window into an open-ended allowance.  The MEASURED requirement on real transports is
# a median and a maximum of ONE protected round
# (diagnostics/pmo_discovery_v1/transport_ordering_policy_v1.json, sibling branch):
# a single-crossing guarantee is what the data supports.
PROTECTION_ROUNDS_CEILING = 4
DEFAULT_PROTECTION_ROUNDS = 1

# A macro option is a SHORT chain by design. The measured transport scale on real
# declared targets is 19 / 36 / 56 primitives against a measured realization ceiling of
# median 16 and max 23, so two to four legs covers the range; a longer chain would be a
# planner, which this is deliberately not.
MAX_STAGES = 4

# The MEASURED single-program realization ceiling, not the executor's configured
# `max_primitives`. Bound-program lengths observed on the live PMO path are {14, 16, 17,
# 23}: a macro longer than 23 is beyond what this controller has been measured to
# realize in one program. It is NOT a hard executor bound, and this module does not
# claim one.
MEASURED_REALIZATION_CEILING = 23

# Total parent mass guaranteed to the protected bridges as a group.  It is a FLOOR, so
# a bridge that is already winning on its own score keeps its larger share.
DEFAULT_PARENT_MASS_FLOOR = 0.25

DECLARED = "declared"
BRIDGE_CHARGED = "bridge_charged"
REACHED = "reached"
EXPIRED = "expired"

TERMINAL_STATUSES = frozenset({REACHED, EXPIRED})


# ---- Declared options ----


@dataclass(frozen=True)
class MacroOptionStage:
    """One realizable leg of a macro option, already executed and validated."""

    index: int
    parent_endpoint: str
    endpoint: str
    primitives: int

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "parent_endpoint": self.parent_endpoint,
            "endpoint": self.endpoint,
            "primitives": self.primitives,
        }


@dataclass(frozen=True)
class MacroOption:
    """A complete validated `G0 -> G_bridge -> G_destination` transformation.

    `stages` is ordered and contiguous: stage `i + 1` starts from the endpoint stage
    `i` produced.  The option is DECLARED before any of its states is charged, which
    is what makes the commitment to the complete macro meaningful rather than a
    description applied afterwards.
    """

    option_id: str
    origin_endpoint: str
    stages: tuple[MacroOptionStage, ...]
    protection_rounds: int
    declared_at_round: int
    realization_ceiling: int

    def __post_init__(self) -> None:
        if not 2 <= len(self.stages) <= MAX_STAGES:
            raise ValueError(
                f"a macro option needs 2 to {MAX_STAGES} stages -- a bridge and a destination"
            )
        if not isinstance(self.protection_rounds, int) or not (
            1 <= self.protection_rounds <= PROTECTION_ROUNDS_CEILING
        ):
            raise ValueError(
                "macro option protection must be a bounded explicit round count in "
                f"[1, {PROTECTION_ROUNDS_CEILING}]"
            )
        expected = self.origin_endpoint
        for position, stage in enumerate(self.stages):
            if stage.index != position or stage.parent_endpoint != expected:
                raise ValueError("macro option stages are not a contiguous chain")
            if stage.primitives < 1:
                raise ValueError("every macro option stage must do real work")
            expected = stage.endpoint
        if self.total_primitives <= self.realization_ceiling:
            # A macro that one proposal round could realize needs no staging and no
            # protection; declaring it would manufacture a success the mechanism did
            # not earn.
            raise ValueError(
                "a macro option must exceed the single-round realization ceiling"
            )

    @property
    def total_primitives(self) -> int:
        return sum(stage.primitives for stage in self.stages)

    @property
    def bridge_endpoints(self) -> tuple[str, ...]:
        """Every intermediate the macro must pass through, in order."""
        return tuple(stage.endpoint for stage in self.stages[:-1])

    @property
    def bridge_endpoint(self) -> str:
        """The FIRST bridge -- the canonical `G0 -> G_bridge -> G_destination` case."""
        return self.stages[0].endpoint

    @property
    def destination_endpoint(self) -> str:
        return self.stages[-1].endpoint

    @property
    def total_protected_rounds_budget(self) -> int:
        """Bounded by construction: one window per crossing, no crossing extendable."""
        return (len(self.stages) - 1) * self.protection_rounds

    def payload(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "origin_endpoint": self.origin_endpoint,
            "stages": [stage.payload() for stage in self.stages],
            "protection_rounds": self.protection_rounds,
            "declared_at_round": self.declared_at_round,
            "realization_ceiling": self.realization_ceiling,
            "total_primitives": self.total_primitives,
            "total_protected_rounds_budget": self.total_protected_rounds_budget,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MacroOption:
        return cls(
            option_id=str(payload["option_id"]),
            origin_endpoint=str(payload["origin_endpoint"]),
            stages=tuple(
                MacroOptionStage(
                    index=int(row["index"]),
                    parent_endpoint=str(row["parent_endpoint"]),
                    endpoint=str(row["endpoint"]),
                    primitives=int(row["primitives"]),
                )
                for row in payload["stages"]
            ),
            protection_rounds=int(payload["protection_rounds"]),
            declared_at_round=int(payload["declared_at_round"]),
            realization_ceiling=int(payload["realization_ceiling"]),
        )


def option_identity(origin_endpoint: str, endpoints: list[str]) -> str:
    """Content address of a macro option: its origin and the chain it commits to."""
    return identity({"origin": origin_endpoint, "chain": list(endpoints)})


# ---- Registry ----


@dataclass
class _OptionRecord:
    option: MacroOption
    # The EXECUTED construction material for every stage, kept because an option is a
    # commitment to a macro that was already validated end to end.  A resumed run that
    # held only the declaration could not continue an in-flight option, and its charged
    # bridge would be stranded: the call is spent and nothing can ever follow it.
    construction: dict[str, Any] = field(default_factory=dict)
    status: str = DECLARED
    # The LAST proposal round this option's bridge is protected in, inclusive.  An
    # explicit horizon rather than a countdown, because a countdown decremented once per
    # advanced round expires a one-round window before the round it was meant to cover:
    # the bridge is charged DURING round r, so the first round that can continue it is
    # r + 1, and a window of W covers r + 1 .. r + W.
    protected_through_round: int | None = None
    # Index of the last stage endpoint the LEDGER has charged; -1 before any. The
    # protected state is always `stages[frontier].endpoint`, so exactly one window is
    # open per option at a time and a chain costs one bounded window PER CROSSING.
    frontier: int = -1
    # The round the CURRENT window opened. Distinct from `bridge_charged_at_round`,
    # which stays pinned to the first crossing: on a chain the horizon moves with the
    # frontier, so validating it against the first crossing would reject a legitimate
    # resume of a partly walked option.
    window_opened_at_round: int | None = None
    bridge_charged_at_round: int | None = None
    resolved_at_round: int | None = None
    stage_charges: dict[str, int] = field(default_factory=dict)


class MacroOptionRegistry:
    """Declare options, protect their bridges for a bounded window, then release.

    The registry is the single authority on which endpoints are protected and for how
    long.  It OBSERVES charges; it never performs or defers one.
    """

    def __init__(self, *, parent_mass_floor: float = DEFAULT_PARENT_MASS_FLOOR) -> None:
        if not 0.0 < parent_mass_floor < 1.0:
            raise ValueError("protected parent-mass floor must lie strictly inside (0, 1)")
        self.parent_mass_floor = float(parent_mass_floor)
        self._records: dict[str, _OptionRecord] = {}
        self._round = 0

    # ---- Declaration ----

    def declare(self, option: MacroOption, construction: dict[str, Any]) -> str:
        """Register a macro whose every stage has ALREADY been executed and validated.

        `construction` carries one executed record per stage, so the declaration is a
        commitment to work the executor has already certified rather than an intention.
        """
        if option.option_id in self._records:
            raise ValueError("macro option already declared under this identity")
        stages = construction.get("stages") or []
        if len(stages) != len(option.stages):
            raise ValueError("macro option declaration lacks an executed record per stage")
        self._records[option.option_id] = _OptionRecord(
            option=option,
            construction=json.loads(json.dumps(construction)),
        )
        return option.option_id

    def construction(self, option_id: str) -> dict[str, Any]:
        return self._records[option_id].construction

    @property
    def options(self) -> dict[str, MacroOption]:
        return {key: record.option for key, record in self._records.items()}

    def status(self, option_id: str) -> str:
        return self._records[option_id].status

    # ---- Round clock ----

    def advance_round(self, round_index: int) -> dict[str, Any]:
        """Move the protection clock forward, expiring every exhausted window.

        The clock ticks on the ROUND INDEX, not on calls into this method, so a
        double invocation inside one round cannot silently shorten a window and a
        skipped round cannot silently lengthen one.
        """
        if round_index < self._round:
            raise ValueError("macro option protection clock may not run backwards")
        elapsed = round_index - self._round
        self._round = round_index
        expired = []
        for option_id, record in self._records.items():
            if record.status != BRIDGE_CHARGED:
                continue
            horizon = record.protected_through_round
            if horizon is None or round_index > horizon:
                record.status = EXPIRED
                record.resolved_at_round = round_index
                expired.append(option_id)
        return {"round": round_index, "elapsed": elapsed, "expired": expired}

    # ---- Observation of charged endpoints ----

    def note_charged(self, endpoint: str, round_index: int) -> list[str]:
        """Record that the LEDGER charged `endpoint`. Never charges anything itself.

        The frontier only ever moves FORWARD along the declared chain, so charging an
        already-passed intermediate cannot reopen a window, and charging a later stage
        out of order cannot skip the ones before it.
        """
        touched = []
        for option_id, record in self._records.items():
            if record.status in TERMINAL_STATUSES:
                continue
            option = record.option
            nxt = record.frontier + 1
            if nxt >= len(option.stages) or endpoint != option.stages[nxt].endpoint:
                continue
            record.frontier = nxt
            record.stage_charges[endpoint] = round_index
            touched.append(option_id)
            if nxt == len(option.stages) - 1:
                record.status = REACHED
                record.resolved_at_round = round_index
                record.protected_through_round = None
                record.window_opened_at_round = None
                continue
            record.status = BRIDGE_CHARGED
            if record.bridge_charged_at_round is None:
                record.bridge_charged_at_round = round_index
            record.window_opened_at_round = round_index
            # The window opens when the intermediate is actually in the population, not
            # when the option was declared: protecting a state that was never charged
            # would spend the window on nothing.
            record.protected_through_round = round_index + option.protection_rounds
        return touched

    # ---- Protection surface ----

    def protected_bridges(self) -> dict[str, str]:
        """Endpoint -> option id, for every option with an OPEN protection window."""
        return {
            record.option.stages[record.frontier].endpoint: option_id
            for option_id, record in self._records.items()
            if record.status == BRIDGE_CHARGED and record.frontier >= 0
        }

    def next_stage_index(self, option_id: str) -> int:
        """Which declared stage this option is waiting to have executed and charged."""
        return self._records[option_id].frontier + 1

    def open_options(self) -> dict[str, MacroOption]:
        return {
            option_id: record.option
            for option_id, record in self._records.items()
            if record.status == BRIDGE_CHARGED
        }

    def awaiting_bridge(self) -> dict[str, MacroOption]:
        """Options whose first leg has not been charged yet."""
        return {
            option_id: record.option
            for option_id, record in self._records.items()
            if record.status == DECLARED
        }

    def _rounds_remaining(self, record: _OptionRecord) -> int:
        """Proposal rounds still covered, INCLUDING the current one. Display only."""
        if record.status != BRIDGE_CHARGED or record.protected_through_round is None:
            return 0
        return max(0, record.protected_through_round - self._round + 1)

    # ---- Reporting ----

    def report(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return {
            "schema_version": SCHEMA,
            "round": self._round,
            "parent_mass_floor": self.parent_mass_floor,
            "declared": len(self._records),
            "status_counts": dict(sorted(counts.items())),
            "reached": sorted(
                key for key, record in self._records.items() if record.status == REACHED
            ),
            "expired_unreached": sorted(
                key for key, record in self._records.items() if record.status == EXPIRED
            ),
            "options": [
                {
                    **record.option.payload(),
                    "status": record.status,
                    "rounds_remaining": self._rounds_remaining(record),
                    "protected_through_round": record.protected_through_round,
                    "window_opened_at_round": record.window_opened_at_round,
                    "frontier": record.frontier,
                    "stages_charged": record.frontier + 1,
                    "bridge_charged_at_round": record.bridge_charged_at_round,
                    "resolved_at_round": record.resolved_at_round,
                    "stage_charges": dict(sorted(record.stage_charges.items())),
                }
                for _, record in sorted(self._records.items())
            ],
        }

    # ---- Durable state ----

    def payload(self) -> dict[str, Any]:
        """Everything a preempted run needs to resume with its windows intact.

        A resumed run that forgot an open window would strand a charged bridge
        forever: the call is already spent, and nothing would ever continue from it.
        """
        return json.loads(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "round": self._round,
                    "parent_mass_floor": self.parent_mass_floor,
                    "records": [
                        {
                            "option": record.option.payload(),
                            "construction": record.construction,
                            "status": record.status,
                            "protected_through_round": record.protected_through_round,
                            "window_opened_at_round": record.window_opened_at_round,
                            "frontier": record.frontier,
                            "bridge_charged_at_round": record.bridge_charged_at_round,
                            "resolved_at_round": record.resolved_at_round,
                            "stage_charges": record.stage_charges,
                        }
                        for _, record in sorted(self._records.items())
                    ],
                }
            )
        )

    @classmethod
    def restore(cls, payload: dict[str, Any] | None) -> MacroOptionRegistry:
        if not payload:
            return cls()
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("macro option registry payload carries a foreign schema")
        registry = cls(parent_mass_floor=float(payload["parent_mass_floor"]))
        registry._round = int(payload["round"])
        for row in payload["records"]:
            option = MacroOption.from_payload(row["option"])
            status = str(row["status"])
            if status not in (DECLARED, BRIDGE_CHARGED, REACHED, EXPIRED):
                raise ValueError("macro option payload carries an unknown status")
            horizon = row["protected_through_round"]
            opened = row["window_opened_at_round"]
            charged = row["bridge_charged_at_round"]
            # A restored window may never reach FURTHER than the declared one; that is
            # the one way a resume could silently convert a bounded protection into an
            # open-ended allowance.
            if horizon is not None and (
                opened is None or int(horizon) > int(opened) + option.protection_rounds
            ):
                raise ValueError("restored macro option window exceeds its declaration")
            registry._records[option.option_id] = _OptionRecord(
                option=option,
                construction=row.get("construction") or {},
                status=status,
                protected_through_round=None if horizon is None else int(horizon),
                frontier=int(row["frontier"]),
                window_opened_at_round=None if opened is None else int(opened),
                bridge_charged_at_round=charged,
                resolved_at_round=row["resolved_at_round"],
                stage_charges=dict(row["stage_charges"]),
            )
        return registry


# ---- Pure selection policies ----


def protected_parent_weights(
    keys: list[str],
    weights: np.ndarray,
    endpoint_of: dict[str, str],
    protected: dict[str, str] | set[str],
    *,
    floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Guarantee the protected bridges a FLOOR of total parent mass.

    This is a re-ranking, never a filter: unprotected entries keep positive mass in
    proportion to what the ordinary policy gave them.  A bridge already above the
    floor on its own score is left alone, so protection can only ever add mass to a
    state the score law was starving -- which is exactly the pruning being repaired.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(keys):
        raise ValueError("parent weights must be one-dimensional and aligned to keys")
    if not 0.0 < floor < 1.0:
        raise ValueError("protected parent-mass floor must lie strictly inside (0, 1)")
    index = [
        position
        for position, key in enumerate(keys)
        if endpoint_of.get(key) in set(protected)
    ]
    detail: dict[str, Any] = {
        "protected_entries": len(index),
        "floor": float(floor),
        "mass_before": float(weights[index].sum()) if index else 0.0,
        "applied": False,
    }
    if not index:
        detail["mass_after"] = 0.0
        return weights, detail
    current = float(weights[index].sum())
    if current >= floor or current >= 1.0:
        detail["mass_after"] = current
        return weights, detail
    lifted = weights * ((1.0 - floor) / (1.0 - current))
    if current > 0.0:
        lifted[index] = weights[index] * (floor / current)
    else:
        lifted[index] = floor / len(index)
    total = lifted.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("protected parent re-weighting produced no usable mass")
    lifted = lifted / total
    detail["applied"] = True
    detail["mass_after"] = float(lifted[index].sum())
    return lifted, detail


def reserve_continuation_slots(
    chosen: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    reserved_ids: set[str],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Guarantee an oracle slot to the declared continuation of a protected option.

    `_allocate` keeps `limit` of a strictly larger pool, so a continuation whose
    parent scored badly is generated and then dropped.  This puts the reserved
    continuations at the front and displaces from the TAIL of the ordinary choice,
    so the ordinary policy's own highest-ranked picks survive.

    The reservation is capped at `limit` and can never exceed it, so protection can
    take the round's whole batch only if the option machinery declared that many
    continuations -- and never more than the batch the caller authorized.
    """
    if limit < 0:
        raise ValueError("allocation limit may not be negative")
    chosen_ids = {row["candidate_id"] for row in chosen}
    reserved = [
        row
        for row in pool
        if row["candidate_id"] in reserved_ids and row["candidate_id"] not in chosen_ids
    ]
    detail: dict[str, Any] = {
        "reserved_available": len(reserved_ids),
        "reserved_already_chosen": len(reserved_ids & chosen_ids),
        "reserved_added": 0,
        "displaced": 0,
    }
    if not reserved or not limit:
        return chosen, detail
    reserved = reserved[:limit]
    keep = limit - len(reserved)
    merged = reserved + chosen[:keep]
    detail["reserved_added"] = len(reserved)
    detail["displaced"] = max(0, len(chosen) - keep)
    return merged, detail


__all__ = [
    "BRIDGE_CHARGED",
    "DECLARED",
    "DEFAULT_PARENT_MASS_FLOOR",
    "DEFAULT_PROTECTION_ROUNDS",
    "EXPIRED",
    "MAX_STAGES",
    "MEASURED_REALIZATION_CEILING",
    "PROTECTION_ROUNDS_CEILING",
    "REACHED",
    "SCHEMA",
    "MacroOption",
    "MacroOptionRegistry",
    "MacroOptionStage",
    "option_identity",
    "protected_parent_weights",
    "reserve_continuation_slots",
]
