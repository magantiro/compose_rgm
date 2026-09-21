"""Discovery allocation for the PMO population controller (arm C).

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------
Arm B (:mod:`compose_v4.control.pmo_online_memory`) changed which proposals are
GENERATED, and the matched 1,000-call A/B measured it winning at every
checkpoint.  It also measured the remaining gap: arm B found its best molecule
at 500 charged calls and never improved ``best`` again through 1,000, while its
top ten kept filling in.  That is the signature of a controller that exploits a
discovered basin well and discovers poorly.

This module changes where the BUDGET goes, and nothing else.  It does not touch
the proposal law, the realizer, the fiber, or the executor.  Arm C is arm B plus
this allocator.

THE DEFECT, MEASURED
--------------------
:class:`~compose_v4.control.pmo_credit.PopulationCredit` already carries the
joint cell ``(basin, parent, family, scale)``, already credits PARENT-RELATIVE
improvement rather than raw child score, and already mixes a persistent
exploration floor ``q = (1 - eps) * q_credit + eps * q_explore``.  The floor is
real.  It is simply uniform over CELLS, and a cell is not a basin.

So the allocation unit is not the diversity unit, and whichever side of the run
holds more cells captures the floor itself.  Measured on this module's own
fixture -- one basin that has opened 30 cells against three structurally
distinct basins holding one cell each, eps = 0.2:

    dominant basin   0.9174 of the budget
    three rivals     0.0826  (0.0275 each)

A floor stratified over basins instead guarantees each of the four 0.0500.
Note the defect is DIRECTIONLESS, which is why it is worth fixing at the unit
rather than by tuning eps: ``tests/test_pmo_credit_adversarial.py``
``::test_cell_proliferation_cannot_outbid_measured_credit`` records the SAME
root cause pointing the other way, with 30 never-productive fragment cells
taking budget from one cell holding 40 trials at +4.0.  Uniform-over-cells
rewards whoever proliferates; it does not reward exploration.

THE FOUR PIECES, AND WHICH ONES ALREADY EXISTED
-----------------------------------------------
1. EXPLICIT EXPLOITATION / DISCOVERY SPLIT.  :func:`discovery_quota` reserves a
   whole number of oracle slots per batch for cells whose BASIN is thin, drawn
   before the credit-weighted draw sees the room.  The reserved count is
   realized, not expected: a weight mix guarantees a share in expectation, and
   over the 14-16 slots of one PMO batch an expectation is not a floor.

2. FRONTIER CREDIT.  PMO grades a run on the mean of its top ten distinct
   scored molecules, so what an edit is WORTH is what it adds to that.
   :meth:`OnlineProposalMemory.allocation_priority` already computes exactly
   this and -- measured by grep over ``src``, ``scripts`` and ``modal_apps`` --
   had ZERO production call sites on arm B: it appears only in its own test.
   Its docstring states the gap it was built to close ("It supplies the frontier
   term that allocator currently has no way to see") and nothing consumed it.
   :class:`DiscoveryCredit` consumes it, in cell value, BESIDE the parent-
   relative term rather than instead of it -- that method's docstring is
   explicit that the two must not be collapsed, because parent-relative remains
   the right learning evidence for what an edit does.

   NOT DONE, and deliberately: the WITHIN-cell ranking still orders by predicted
   endpoint value.  ``frontier_gain`` is monotone non-decreasing in the endpoint
   value and is identically zero below the frontier, so ranking inside a cell by
   frontier gain would leave every sub-frontier candidate tied at 0.0 and hand
   the order to the candidate id.  It changes allocation ACROSS cells, where the
   values differ, and would only destroy information within one.

3. DONATION.  Already arm B's: :class:`DonorRegionMemory` observes regions of
   high-scoring in-run molecules and modulates the region draw, capped as an
   association.  Nothing is added here; arm C inherits it by requiring arm B.

4. HARD EXPLORATION FLOOR.  Two independent guarantees, because they fail
   differently.  :func:`basin_stratified_shares` makes the floor provable in
   expectation at the basin level, and :func:`discovery_quota` makes it realized
   in slot counts.  A reward that makes one early exploit arbitrarily attractive
   can drive ``q_credit`` for every rival basin to a rounding error; neither
   guarantee depends on ``q_credit``.

INFORMATION BOUNDARY
--------------------
Every quantity here is derived from COUNTED in-run observations: charged oracle
scores, the parent they came from, the program that produced them, and the
Bemis-Murcko scaffold of the endpoint.  No oracle internals, no target
structure, no hidden component score, no uncounted same-task history, and no
property is computed on an unscored candidate in order to select it.  Basin
labelling is a structural property of a molecule the controller already holds.

SEARCH SETTINGS, NOT OPTIMA
---------------------------
``DEFAULT_DISCOVERY_FRACTION``, ``DEFAULT_BASIN_TRIAL_THRESHOLD`` and
``DEFAULT_FRONTIER_WEIGHT`` are settings. No calibration or optimality is
claimed for any of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compose_v4.control.pmo_credit import (
    CREDIT_AXES,
    CreditCell,
    CreditKey,
    PopulationCredit,
)

#: Bumped from ``pmo_credit_v2``: a cell now carries frontier evidence beside its
#: parent-relative evidence. A v2 payload restored into this class would read as a
#: cell whose frontier contribution is zero rather than unmeasured, so `restore`
#: refuses it outright rather than resuming on evidence it does not have.
SCHEMA_VERSION = "pmo_discovery_credit_v1"

#: Share of each batch's credit-allocated room reserved for thin basins. A setting.
DEFAULT_DISCOVERY_FRACTION = 0.25

#: A basin at or below this many trials is DISCOVERY-eligible. Counted per basin by
#: summing the joint over every other axis, so proliferating cells inside a basin
#: cannot make it look thin.
DEFAULT_BASIN_TRIAL_THRESHOLD = 3

#: Weight on measured frontier contribution relative to measured parent-relative
#: upside in a cell's value. Both are in oracle-score units, so this is a ratio of
#: two measured quantities rather than a scale conversion. A setting.
DEFAULT_FRONTIER_WEIGHT = 1.0


@dataclass(frozen=True)
class DiscoveryConfig:
    """Settings for the discovery allocator. Every field is a search setting."""

    discovery_fraction: float = DEFAULT_DISCOVERY_FRACTION
    basin_trial_threshold: int = DEFAULT_BASIN_TRIAL_THRESHOLD
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.discovery_fraction < 1.0:
            raise ValueError("discovery fraction must lie in [0, 1)")
        if self.basin_trial_threshold < 0:
            raise ValueError("basin trial threshold must be non-negative")
        if not math.isfinite(self.frontier_weight) or self.frontier_weight < 0.0:
            raise ValueError("frontier weight must be finite and non-negative")

    def payload(self) -> dict[str, float | int]:
        return {
            "discovery_fraction": self.discovery_fraction,
            "basin_trial_threshold": self.basin_trial_threshold,
            "frontier_weight": self.frontier_weight,
        }


def basin_stratified_shares(
    keys: list[CreditKey], values: np.ndarray, *, floor: float
) -> np.ndarray:
    """``(1 - eps) * q_credit + eps * (uniform over basins, then within basin)``.

    The only difference from :meth:`PopulationCredit.allocate` is the measure the
    exploration mass is spread over. Uniform-over-cells gives a basin exploration
    mass proportional to how many cells it has opened; uniform-over-basins gives
    every basin present the same mass however many or few cells carry it.

    GUARANTEE, and it does not depend on `values`: every basin receives at least
    ``floor / n_basins``, and every cell at least ``floor / (n_basins * cells in
    its basin)``. Both survive an adversarial `values` because the floor term is
    added after the credit term is normalized, never multiplied into it.
    """
    if len(keys) != len(values):
        raise ValueError("basin-stratified shares need one value per key")
    if not keys:
        return np.zeros(0, dtype=float)
    if not 0.0 < floor < 1.0:
        raise ValueError("exploration floor must lie strictly inside (0, 1)")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("basin-stratified shares need finite values")
    clipped = np.maximum(values, 0.0)
    total = float(clipped.sum())
    credit = clipped / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))

    cells_per_basin: dict[str, int] = {}
    for key in keys:
        cells_per_basin[key.basin] = cells_per_basin.get(key.basin, 0) + 1
    n_basins = len(cells_per_basin)
    explore = np.asarray(
        [1.0 / (n_basins * cells_per_basin[key.basin]) for key in keys], dtype=float
    )
    mixed = (1.0 - floor) * credit + floor * explore
    return mixed / mixed.sum()


def discovery_quota(room: int, *, fraction: float) -> int:
    """Whole oracle slots reserved for discovery out of `room`.

    Rounded UP, so a fractional entitlement is a slot rather than a rounding loss:
    at the production batch size the exploitation side loses at most one slot and
    the discovery side is never silently zeroed. Zero room reserves nothing, and a
    zero fraction disables the reservation entirely.
    """
    if room <= 0:
        return 0
    if not 0.0 <= fraction < 1.0:
        raise ValueError("discovery fraction must lie in [0, 1)")
    if fraction == 0.0:
        return 0
    return min(room, max(1, math.ceil(fraction * room)))


class DiscoveryCredit(PopulationCredit):
    """Joint credit with frontier evidence and a basin-stratified floor.

    Subclasses rather than replaces :class:`PopulationCredit`: arm B instantiates
    the base class and is untouched by every line here.
    """

    def __init__(
        self,
        *,
        exploration_floor: float | None = None,
        prior_weight: float | None = None,
        config: DiscoveryConfig | None = None,
    ) -> None:
        base: dict[str, float] = {}
        if exploration_floor is not None:
            base["exploration_floor"] = exploration_floor
        if prior_weight is not None:
            base["prior_weight"] = prior_weight
        super().__init__(**base)
        self.config = config or DiscoveryConfig()
        #: Measured frontier contribution per cell, kept BESIDE the parent-relative
        #: evidence the base class stores. Never collapsed into it.
        self.frontier_cells: dict[CreditKey, CreditCell] = {}

    # ---- Evidence ----

    def observe_frontier(self, key: CreditKey, frontier_gain: float) -> CreditCell:
        """Record what one scored child added to the top-ten mean.

        `frontier_gain` is ``U10(A + {child}) - U10(A)``, computed by
        :class:`~compose_v4.control.pmo_online_memory.FrontierLedger` from counted
        scores only. It is zero for a child that does not reach the frontier, which
        is the correct credit: it cost a call and moved the graded quantity not at
        all.
        """
        if not isinstance(key, CreditKey):
            raise TypeError("frontier evidence must be keyed by a CreditKey")
        value = float(frontier_gain)
        if not math.isfinite(value):
            raise ValueError("frontier evidence needs a finite gain")
        if value < 0.0:
            raise ValueError("a frontier gain is non-negative by construction")
        cell = self.frontier_cells.setdefault(key, CreditCell())
        cell.trials += 1
        cell.best_improvement = max(cell.best_improvement, value)
        if value > 0.0:
            cell.improvements += 1
            cell.positive_improvement_sum += value
        return cell

    def frontier_cell(self, key: CreditKey) -> CreditCell:
        return self.frontier_cells.get(key, CreditCell())

    # ---- Basin-level counting ----

    def basin_trials(self) -> dict[str, int]:
        """Trials per basin, summed over the joint. Derived, never stored."""
        totals: dict[str, int] = {}
        for key, cell in self.cells.items():
            totals[key.basin] = totals.get(key.basin, 0) + cell.trials
        return totals

    def is_discovery_cell(self, key: CreditKey) -> bool:
        """True when `key`'s BASIN is at or below the trial threshold.

        Counted at the basin, not the cell: a heavily-worked basin cannot qualify
        as discovery by opening one fresh cell, which is exactly the move that
        captures a cell-level floor.
        """
        return self.basin_trials().get(key.basin, 0) <= self.config.basin_trial_threshold

    # ---- Value and allocation ----

    def frontier_scale(self) -> float:
        """Mean positive frontier gain per frontier-observed trial. Counted."""
        trials = sum(cell.trials for cell in self.frontier_cells.values())
        if not trials:
            return 0.0
        total = sum(cell.positive_improvement_sum for cell in self.frontier_cells.values())
        return total / trials

    def value(self, key: CreditKey) -> float:
        """Parent-relative upside, plus measured frontier contribution, plus optimism.

        The base class's expected-upside-plus-optimism term is kept verbatim, so with
        no frontier evidence anywhere this returns exactly what `PopulationCredit`
        returns and arm C's allocation is arm B's. The frontier term is added, not
        substituted: `parent_relative` remains the learning evidence for what an edit
        does, and frontier gain is what the benchmark grades.
        """
        base = super().value(key)
        if self.config.frontier_weight == 0.0:
            return base
        cell = self.frontier_cell(key)
        frontier = cell.positive_improvement_sum / cell.trials if cell.trials else 0.0
        return base + self.config.frontier_weight * frontier

    def allocate(self, keys) -> np.ndarray:
        """Budget shares with the exploration floor stratified over BASINS."""
        keys = list(keys)
        if not keys:
            return np.zeros(0, dtype=float)
        if len(set(keys)) != len(keys):
            raise ValueError("allocation keys must be distinct cells")
        values = np.asarray([max(0.0, self.value(key)) for key in keys], dtype=float)
        return basin_stratified_shares(keys, values, floor=self.exploration_floor)

    def allocation_report(self, keys) -> dict:
        keys = list(keys)
        report = super().allocation_report(keys)
        basin_counts: dict[str, int] = {}
        for key in keys:
            basin_counts[key.basin] = basin_counts.get(key.basin, 0) + 1
        report.update(
            {
                "schema_version": SCHEMA_VERSION,
                "discovery": self.config.payload(),
                "basins_present": len(basin_counts),
                "frontier_scale": self.frontier_scale(),
                # The guarantee this allocator adds, stated in the artifact so a run
                # can be audited against it without re-deriving the algebra.
                "minimum_basin_share": self.exploration_floor / len(basin_counts),
                "discovery_cells": sum(1 for key in keys if self.is_discovery_cell(key)),
            }
        )
        for row in report["cells"]:
            key = CreditKey(**{axis: row[axis] for axis in CREDIT_AXES})
            row["frontier"] = self.frontier_cell(key).payload()
            row["discovery_eligible"] = self.is_discovery_cell(key)
        return report

    # ---- Serialization ----

    def payload(self) -> dict:
        payload = super().payload()
        payload["schema_version"] = SCHEMA_VERSION
        payload["discovery"] = self.config.payload()
        payload["frontier_cells"] = [
            {**key.payload(), **self.frontier_cells[key].payload()}
            for key in sorted(self.frontier_cells)
        ]
        return payload

    @classmethod
    def restore(cls, payload: dict) -> DiscoveryCredit:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("discovery credit snapshot has an unexpected schema")
        settings = payload.get("discovery") or {}
        credit = cls(
            exploration_floor=float(payload["exploration_floor"]),
            prior_weight=float(payload["prior_weight"]),
            config=DiscoveryConfig(
                discovery_fraction=float(
                    settings.get("discovery_fraction", DEFAULT_DISCOVERY_FRACTION)
                ),
                basin_trial_threshold=int(
                    settings.get("basin_trial_threshold", DEFAULT_BASIN_TRIAL_THRESHOLD)
                ),
                frontier_weight=float(
                    settings.get("frontier_weight", DEFAULT_FRONTIER_WEIGHT)
                ),
            ),
        )
        for row in payload["cells"]:
            key = CreditKey(**{axis: row[axis] for axis in CREDIT_AXES})
            if key in credit.cells:
                raise ValueError("discovery credit snapshot repeats a cell key")
            credit.cells[key] = CreditCell.restore(row)
        for row in payload.get("frontier_cells", []):
            key = CreditKey(**{axis: row[axis] for axis in CREDIT_AXES})
            if key in credit.frontier_cells:
                raise ValueError("discovery credit snapshot repeats a frontier cell key")
            credit.frontier_cells[key] = CreditCell.restore(row)
        return credit
