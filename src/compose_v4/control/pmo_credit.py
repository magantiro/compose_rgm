"""Joint reward credit and floored allocation for the PMO population controller.

The controller in `pmo_population_controller.py` already tracks reward, but it tracks
two DISJOINT MARGINALS: `population_state["channels"][scale]` and
`population_state["parent_outcomes"][parent]`. Neither can answer the question the
allocation actually asks -- "does a jump help *from this parent, in this basin, with
this kind of program*" -- because a marginal averages that question away. This module
stores the JOINT cell `Q(basin, parent, family, scale)` and DERIVES every marginal from
it by summation, so a marginal can never disagree with the joint.

Invariants maintained here and asserted in `tests/test_pmo_credit.py`:

1. `observe` is the only writer. A cell's `trials` equals the number of `observe` calls
   routed to it, and `marginal(axis)` equals the summation of the joint over that axis.
2. Reward is the OBSERVED improvement `df = f(child) - f(parent)` in larger-is-better
   convention. `improvement()` performs the sign flip explicitly and refuses an unknown
   direction, because PMO maximizes its oracle while T4 minimizes a docking score and a
   silent sign error there is indistinguishable from a working controller.
3. `allocate` returns `q = (1 - eps) * q_credit + eps * q_explore` with `q_explore`
   UNIFORM over the live cells. Because `q_credit >= 0`, every live cell provably keeps
   at least `eps / n` of the budget forever. That is the whole point: a floor that decays
   lets one early lineage absorb the run, and the 384-call pilot could not compound
   because budget never returned to an abandoned basin.
4. `value` is expected UPSIDE, not mean signed improvement: a negative `df` contributes
   zero, because the parent stays in the archive and is never lost by trying a child.
   This matches the existing controller's `positive_improvement_sum / trials` tilt.
5. Serialization is deterministic -- cells are emitted in sorted key order -- so a
   snapshot round-trips byte-identically.

The uncertainty bonus and the exploration floor are SEARCH SETTINGS, not established
optima and not calibrated uncertainty. No confidence claim is made or implied.

BASIN LABELLING. The frozen controller records a `basin_id` built from a 64-bit prefix
of the Morgan fingerprint plus the heavy-atom count. Measured on a five-member analogue
series sharing one Bemis-Murcko scaffold, that identifier produced 5 distinct values for
5 molecules, so it is a per-molecule key and cannot group a lineage: used as a credit
key every cell would hold exactly one trial and nothing would ever be learned. This
module therefore labels a basin by its Bemis-Murcko scaffold, and gives acyclic
molecules an explicit sentinel rather than letting them share the empty-string scaffold
silently with every other acyclic molecule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# v2: `prior_weight` changed UNITS. In v1 it was an absolute oracle-improvement
# optimism bonus; it is now a dimensionless multiple of `observed_scale()`. A v1
# snapshot carries 0.25 in the old units and would be silently misread, so the schema
# version is bumped and `restore` refuses it rather than resuming on a wrong scale.
SCHEMA_VERSION = "pmo_credit_v2"

# Action scale. One composer with a scale variable, not three algorithms.
REFINE = "refine"
MEDIUM = "medium"
JUMP = "jump"
SCALES = (REFINE, MEDIUM, JUMP)

# The frozen controller's channel names, mapped onto the scale vocabulary. The channel
# names are kept as the source of truth; this is a translation, never a rename.
SCALE_BY_CHANNEL = {
    "shallow_program_channel": REFINE,
    "structured_program_channel": MEDIUM,
    "joint_dependency_region_jump": JUMP,
}

# An acyclic molecule has an EMPTY Bemis-Murcko scaffold. Bucketing those under "" would
# merge every acyclic lineage into one basin -- a key that matches nearly everything.
ACYCLIC_BASIN = "acyclic"

# Persistent floor. The brief's authorized band is 0.10-0.25; 0.20 matches the parent-level
# floor already carried by `ProgramSearchConfig.exploration`.
DEFAULT_EXPLORATION_FLOOR = 0.2

# Optimism given to a cell with no trials, in MULTIPLES of the run's own measured mean
# upside per trial -- not in absolute oracle-improvement units. A fixed 0.25 was sized
# for a docking score: against PMO's measured mean upside per trial (~0.007) it made an
# untried cell strictly outrank a cell that had already delivered a real improvement, so
# proliferating new cells beat accumulating evidence in any of them and the joint credit
# fragmented to ~1.1 trials per cell. Dimensionless here, so one value covers every
# oracle and the scale is counted rather than assumed.
DEFAULT_PRIOR_WEIGHT = 1.0

CREDIT_AXES = ("basin", "parent", "family", "scale")


def basin_label(smiles: str) -> str:
    """Bemis-Murcko scaffold of `smiles`, or the acyclic sentinel.

    Raises on unparseable input rather than returning a bucket, because an unparseable
    endpoint silently joining a basin would corrupt every cell that basin feeds.
    """
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("basin label needs a non-empty SMILES string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"basin label needs an RDKit-valid molecule: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return scaffold if scaffold else ACYCLIC_BASIN


def program_family(executor_rules) -> str:
    """Task-blind family label: the sorted DISTINCT executor rules of one program.

    Multiplicity is deliberately dropped. Two programs that both insert atoms and close a
    cycle are the same kind of transformation whether they do it twice or five times; the
    magnitude is carried by the scale axis, not by the family.
    """
    rules = sorted({str(rule) for rule in executor_rules})
    if not rules:
        raise ValueError("a complete program family needs at least one executor rule")
    return "+".join(rules)


def scale_for_channel(channel: str) -> str:
    """Translate a frozen controller channel name into the scale vocabulary."""
    try:
        return SCALE_BY_CHANNEL[str(channel)]
    except KeyError:
        raise ValueError(f"unknown proposal channel: {channel!r}") from None


def improvement(child_score: float, parent_score: float, *, direction: str) -> float:
    """Observed `df` in larger-is-better convention.

    PMO maximizes its oracle and T4 minimizes a docking score. Making the caller name the
    direction, and refusing anything else, is what keeps one sign convention across both
    outer regimes; a flipped sign here produces a controller that confidently allocates
    budget away from every improvement it finds.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"score direction must be maximize or minimize, got {direction!r}")
    child, parent = float(child_score), float(parent_score)
    if not math.isfinite(child) or not math.isfinite(parent):
        raise ValueError("improvement needs two finite scores")
    return child - parent if direction == "maximize" else parent - child


@dataclass(frozen=True, order=True)
class CreditKey:
    """One allocation cell: a basin, a parent, a program family, and an action scale."""

    basin: str
    parent: str
    family: str
    scale: str

    def __post_init__(self) -> None:
        for axis in CREDIT_AXES:
            value = getattr(self, axis)
            if not isinstance(value, str) or not value:
                raise ValueError(f"credit key needs a non-empty {axis}")
        if self.scale not in SCALES:
            raise ValueError(f"credit key scale must be one of {SCALES}, got {self.scale!r}")

    def payload(self) -> dict[str, str]:
        return {axis: getattr(self, axis) for axis in CREDIT_AXES}


@dataclass
class CreditCell:
    """Accumulated evidence for one cell. Counts only; no distribution is assumed."""

    trials: int = 0
    improvements: int = 0
    positive_improvement_sum: float = 0.0
    best_improvement: float = -math.inf

    def payload(self) -> dict[str, float | int | None]:
        best = self.best_improvement
        return {
            "trials": self.trials,
            "improvements": self.improvements,
            "positive_improvement_sum": self.positive_improvement_sum,
            "best_improvement": None if best == -math.inf else best,
        }

    @classmethod
    def restore(cls, payload: dict) -> CreditCell:
        best = payload.get("best_improvement")
        return cls(
            trials=int(payload["trials"]),
            improvements=int(payload["improvements"]),
            positive_improvement_sum=float(payload["positive_improvement_sum"]),
            best_improvement=-math.inf if best is None else float(best),
        )


def credit_key_from_candidate(candidate: dict) -> CreditKey:
    """Derive a cell key from a controller candidate record, without touching it.

    This is the integration seam with the frozen `PmoPopulationController`: it reads the
    fields that record already carries and adds nothing to it.
    """
    provenance = candidate.get("provenance") or {}
    parent = provenance.get("entry_id")
    if not parent:
        raise ValueError("candidate provenance is missing its parent entry_id")
    rules = [row["executor_rule"] for row in candidate["trace"]["actions"]]
    return CreditKey(
        basin=basin_label(candidate["endpoint"]),
        parent=str(parent),
        family=program_family(rules),
        scale=scale_for_channel(provenance.get("planner_channel", "")),
    )


class PopulationCredit:
    """Joint credit over `(basin, parent, family, scale)` with a floored allocation."""

    def __init__(
        self,
        *,
        exploration_floor: float = DEFAULT_EXPLORATION_FLOOR,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    ) -> None:
        if not 0.0 < exploration_floor < 1.0:
            raise ValueError("exploration floor must lie strictly inside (0, 1)")
        if not math.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError("prior weight must be a finite non-negative optimism bonus")
        self.exploration_floor = float(exploration_floor)
        self.prior_weight = float(prior_weight)
        self.cells: dict[CreditKey, CreditCell] = {}

    # ---- Evidence ----

    def observe(self, key: CreditKey, realized_improvement: float) -> CreditCell:
        """Record one scored outcome. The only writer of cell state."""
        if not isinstance(key, CreditKey):
            raise TypeError("credit evidence must be keyed by a CreditKey")
        value = float(realized_improvement)
        if not math.isfinite(value):
            raise ValueError("credit evidence needs a finite realized improvement")
        cell = self.cells.setdefault(key, CreditCell())
        cell.trials += 1
        cell.best_improvement = max(cell.best_improvement, value)
        if value > 0.0:
            cell.improvements += 1
            cell.positive_improvement_sum += value
        return cell

    def cell(self, key: CreditKey) -> CreditCell:
        """Evidence for one cell; an untried cell reads as empty, never as absent."""
        return self.cells.get(key, CreditCell())

    def marginal(self, axis: str) -> dict[str, CreditCell]:
        """Collapse the joint onto one axis. Derived by summation, never stored."""
        if axis not in CREDIT_AXES:
            raise ValueError(f"credit axis must be one of {CREDIT_AXES}, got {axis!r}")
        totals: dict[str, CreditCell] = {}
        for key, cell in self.cells.items():
            bucket = totals.setdefault(getattr(key, axis), CreditCell())
            bucket.trials += cell.trials
            bucket.improvements += cell.improvements
            bucket.positive_improvement_sum += cell.positive_improvement_sum
            bucket.best_improvement = max(bucket.best_improvement, cell.best_improvement)
        return dict(sorted(totals.items()))

    # ---- Value and allocation ----

    def observed_scale(self) -> float:
        """Mean positive improvement per trial, summed over the joint. Counted, not set.

        This is the unit the optimism bonus is denominated in. With no trials anywhere it
        is 0.0, which makes every untried cell's bonus identical and therefore leaves the
        cold-start allocation uniform -- exactly where it already was.
        """
        trials = sum(cell.trials for cell in self.cells.values())
        if not trials:
            return 0.0
        return sum(cell.positive_improvement_sum for cell in self.cells.values()) / trials

    def value(self, key: CreditKey) -> float:
        """Expected upside per trial, plus a scale-aware optimism bonus for thin cells.

        A negative `df` contributes zero rather than a penalty: the parent remains in the
        archive, so a failed child costs one call, not a regression. The bonus shrinks as
        `1 / sqrt(trials + 1)` and is measured in multiples of `observed_scale()`, so an
        untried cell is worth about one typical trial's upside rather than a constant
        that no measured improvement on this task could ever overtake.
        """
        cell = self.cell(key)
        upside = cell.positive_improvement_sum / cell.trials if cell.trials else 0.0
        return upside + self.prior_weight * self.observed_scale() / math.sqrt(cell.trials + 1)

    def allocate(self, keys) -> np.ndarray:
        """Budget shares over `keys`: `(1 - eps) * q_credit + eps * uniform`.

        Every returned share is at least `eps / len(keys)`, so no lineage can ever be
        starved out of the run by an early winner.
        """
        keys = list(keys)
        if not keys:
            return np.zeros(0, dtype=float)
        if len(set(keys)) != len(keys):
            raise ValueError("allocation keys must be distinct cells")
        values = np.asarray([max(0.0, self.value(key)) for key in keys], dtype=float)
        total = float(values.sum())
        credit = values / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))
        explore = np.full(len(keys), 1.0 / len(keys))
        mixed = (1.0 - self.exploration_floor) * credit + self.exploration_floor * explore
        return mixed / mixed.sum()

    def allocation_report(self, keys) -> dict:
        """Auditable record of one allocation decision."""
        keys = list(keys)
        shares = self.allocate(keys)
        return {
            "schema_version": SCHEMA_VERSION,
            "exploration_floor": self.exploration_floor,
            "prior_weight": self.prior_weight,
            "observed_scale": self.observed_scale(),
            "minimum_share": (
                self.exploration_floor / len(keys) if keys else 0.0
            ),
            "cells": [
                {
                    **key.payload(),
                    "share": float(share),
                    "value": self.value(key),
                    **self.cell(key).payload(),
                }
                for key, share in zip(keys, shares, strict=True)
            ],
        }

    # ---- Serialization ----

    def payload(self) -> dict:
        """Deterministic snapshot: cells sorted by key."""
        return {
            "schema_version": SCHEMA_VERSION,
            "exploration_floor": self.exploration_floor,
            "prior_weight": self.prior_weight,
            "cells": [
                {**key.payload(), **self.cells[key].payload()} for key in sorted(self.cells)
            ],
        }

    @classmethod
    def restore(cls, payload: dict) -> PopulationCredit:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("population credit snapshot has an unexpected schema")
        credit = cls(
            exploration_floor=float(payload["exploration_floor"]),
            prior_weight=float(payload["prior_weight"]),
        )
        for row in payload["cells"]:
            key = CreditKey(**{axis: row[axis] for axis in CREDIT_AXES})
            if key in credit.cells:
                raise ValueError("population credit snapshot repeats a cell key")
            credit.cells[key] = CreditCell.restore(row)
        return credit
