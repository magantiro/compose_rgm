"""PMO-v2 allocation prototype: count-shrunk hierarchical backoff over the v1 joint cell.

PROTOTYPE, NOT WIRED.  `PmoPopulationController` is frozen and still allocates with
`PopulationCredit.allocate`.  Nothing here is imported by the controller, by
`src/compose_v4/experiments/`, by `modal_apps/` or by any pinned config.  The class
below SUBCLASSES the frozen `PopulationCredit` so it inherits `observe`, `cell`,
`marginal` and the serialization byte-for-byte -- the evidence model is untouched and
only the ALLOCATION rule is replaced.

THE v1 DEFECT THIS REPLACES
---------------------------
`PopulationCredit.allocate` spreads budget over CELLS keyed `(basin, parent, family,
scale)` and scores each one with `value = positive_sum/trials + prior_weight/sqrt(n+1)`.
An untried cell is therefore worth a flat `0.25` no matter where it sits, and the
allocation is proportional to a SUM over cells, so a region's budget is
`(number of its cells) x (per-cell value)`.  Two measured consequences:

  * proliferation beats credit -- 30 never-rewarded cells take 0.714 of the budget
    against 0.286 for a cell with 40 trials at +4.0 (`s5b_cell_fragmentation`);
  * no generational inheritance -- `parent` is the `entry_id`, so a child re-entering
    the archive opens a NEW cell and `value(child of a 60-trial +5.0 basin)` equals
    `value(child of a barren basin)` equals `0.25` exactly (`s7`).

`PopulationCredit.marginal(axis)` already derives every marginal from the joint; no
allocation path has ever consulted it.  v2 consumes it.

THE v2 RULE
-----------
Allocation is a TOP-DOWN walk of a nested backoff chain, coarse to fine, ending at the
full cell.  The production chain is

    ()  ->  (basin,)  ->  (basin, scale)  ->  (basin, family, scale)
                                          ->  (basin, family, scale, parent)

whose last step is exactly the demanded factorisation `P(b,p,f,s) = P(b,f,s) *
P(p | b,f,s)`.  The levels above it are ordered by how fast each axis CHURNS: `parent`
is a fresh `entry_id` for every child and so is dropped first, while `basin` is a
Bemis-Murcko scaffold shared by a whole lineage and so is dropped last.  The graduated
four-level form was chosen by held-out predictive error, not by taste -- see the note on
`PRODUCTION_CHAIN` below.

Each node carries a COUNT-SHRUNK value that blends its own evidence with its parent's:

    Q~(v) = (n_v / (n_v + kappa)) * Q_raw(v) + (kappa / (n_v + kappa)) * Q~(parent(v))
    Q~(root) = (n / (n + kappa)) * Q_raw(root) + (kappa / (n + kappa)) * prior_weight

`Q_raw` is v1's expected upside, `positive_improvement_sum / trials`, so a v1 cell and a
v2 leaf with the same evidence agree in the large-`n` limit.  An untried node has
`n_v = 0` and therefore `Q~(v) = Q~(parent(v))` EXACTLY: it inherits its context instead
of being issued a fresh flat prior.  As `n_v` grows the own-estimate weight goes to 1,
so an exact joint interaction always wins in the end.

Note what happened to v1's flat prior: it is not deleted, it is DEMOTED to the root of
the backoff chain, where it decays with the evidence of the whole run rather than being
re-issued at full strength to every newly-created cell.

MASS SPLIT AT A NODE, AND THE COUNT INVARIANT
---------------------------------------------
At node `v` holding share `S(v)`, its children in the pool are partitioned into TRIED
(`n > 0`) and UNTRIED (`n == 0`).  The untried children are treated as ONE pseudo-sibling
carrying weight `Q~(v)` -- the backoff value they would all inherit -- and that single
share is then split evenly among them:

    share(t in TRIED)   = S(v) * Q~(t) / D,      D = sum_TRIED Q~(t) + Q~(v) * [UNTRIED]
    share(u in UNTRIED) = S(v) * Q~(v) / D / |UNTRIED|

This is the whole fix, and it yields the central invariant:

    EXPLOITATION-MASS COUNT INVARIANCE.  For any node `v` with at least one untried
    child, the total exploitation mass of its untried children,
    `S(v) * Q~(v) / D`, does not depend on how many untried children there are.

It holds because untried cells contribute zero trials and zero improvement sum, so they
move neither `Q~(v)`, nor `S(v)`, nor `D`.  Duplicating untried cells therefore
redistributes that region's exploitation mass among more cells; it never enlarges it.
The same statement holds at EVERY level, so it covers both failure modes at once: new
`entry_id` children under one context (the `s7`/`s9` churn) and whole new basins invented
by the jump channel (the `s5b` fragmentation sweep).

EXPLORATION IS STILL A RESERVED BUDGET
--------------------------------------
v1's floor is preserved unchanged and stays OUTSIDE the hierarchy:

    q = (1 - eps) * q_exploit + eps * uniform_over_cells

so every live cell provably keeps at least `eps / n_cells` forever, which is what keeps a
late-blooming basin discoverable.  Holding the floor mechanism identical to v1 is also
what makes the A/B honest: the only thing that differs between the arms is how
`q_exploit` is computed.  Because the floor is uniform over cells, a region's TOTAL share
still rises with its cell count through the floor term alone -- that residual is bounded
by `eps` by construction, it is deliberate exploration rather than an accident of
proliferation, and it is reported separately from the exploitation mass throughout.

WHAT v2 DOES NOT DO
-------------------
It does NOT reduce how much budget reaches untried cells, and measuring that would be the
wrong test.  Under parent churn both arms send a similar share there (measured ~0.58-0.72
of the exploitation mass, v1 and v2 alike) because ~87% of every round's pool is untried
and those cells are the children of the productive lineages.  What changes is whether the
mass is INFORMATIVE: v1 values every untried cell at the identical flat prior, so across a
round's untried cells it produces exactly ONE distinct share and a rank correlation of 0
with context productivity, by construction.  v2 produced 9 distinct shares and a rank
correlation of 0.68-0.95 on the same pools.  The defect was never that the prior mass was
large; it was that it was FLAT.

WHAT IS NOT CLAIMED
-------------------
`kappa` and the chain are SEARCH SETTINGS selected on synthetic landscapes (see
`scripts/pmo_credit_v2_audit.py`); no real PMO credit snapshot exists on disk, so nothing
here is calibrated against a scored run.  No uncertainty claim is made or implied.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from compose_v4.control.pmo_credit import (
    CREDIT_AXES,
    DEFAULT_EXPLORATION_FLOOR,
    DEFAULT_PRIOR_WEIGHT,
    CreditKey,
    PopulationCredit,
)

SCHEMA_VERSION = "pmo_credit_v2_prototype"

# Backoff chains, coarse to fine.  Every chain must start at the root `()`, end at the
# full cell, and be strictly nested so the recursion is a tree.  `PRODUCTION_CHAIN` is the
# one the audit selects; the rest exist to be swept against it.
ROOT: tuple[str, ...] = ()
FULL_CELL: tuple[str, ...] = ("basin", "family", "scale", "parent")

CHAINS: dict[str, tuple[tuple[str, ...], ...]] = {
    # basin -> context -> parent.  The demanded factorisation, plus a basin level.
    "basin_context_parent": (ROOT, ("basin",), ("basin", "family", "scale"), FULL_CELL),
    # drop family before scale on the way up
    "basin_scale_context_parent": (
        ROOT,
        ("basin",),
        ("basin", "scale"),
        ("basin", "family", "scale"),
        FULL_CELL,
    ),
    # drop scale before family on the way up
    "basin_family_context_parent": (
        ROOT,
        ("basin",),
        ("basin", "family"),
        ("basin", "family", "scale"),
        FULL_CELL,
    ),
    # basin-blind: back off to the operator context rather than to the chemistry
    "context_first": (ROOT, ("family", "scale"), ("basin", "family", "scale"), FULL_CELL),
    # single backoff: context straight to root, no basin level
    "context_only": (ROOT, ("basin", "family", "scale"), FULL_CELL),
    # degenerate control: no backoff structure at all, one flat level of cells.  This is
    # v1's partition, so it isolates the value rule from the hierarchy.
    "flat": (ROOT, FULL_CELL),
}

# SELECTED BY MEASUREMENT, not by taste.  Held-out predictive error over 10 synthetic
# landscapes (`scripts/pmo_credit_v2_audit.py::select_hierarchy`) ranks the chains
#   basin_scale_context_parent 1.6 | basin_family_context_parent 1.8
#   basin_context_parent 3.0 | context_first 3.9 | context_only 4.9 | flat 5.8
# so the graduated four-level chain beats the three-level one I first guessed, and both
# basin-blind chains lose badly.  The two leaders are within each other's spread and
# swap places with the landscape; they COINCIDE in the current controller, where
# `RULES_BY_CHANNEL` makes the family axis a bijection of the channel and hence of the
# scale, so nothing observable today distinguishes them.
PRODUCTION_CHAIN = "basin_scale_context_parent"

# Shrinkage strength, in pseudo-trials of backoff evidence: a cell needs `kappa` of its
# own trials before its own estimate carries half the weight.
#
# SELECTED BY A DECLARED RULE, applied in code in `select_kappa`: the smallest kappa whose
# single-observation weight `1/(1+kappa)` is at most 0.5, subject to a worst-case relative
# gain regret of at most 0.25.  The two measured criteria DISAGREE -- held-out predictive
# error wants kappa to track the observation noise (best 1.0 at noise 0.1, 4.0 at 1.0,
# 64.0 at 8.0), while realized gain prefers the smallest kappa swept at every noise level
# (regret 0.0 at 0.25, 0.159 at 1.0, 0.469 at 4.0).  The rule breaks the tie on the
# stability of an established lineage: one measurement of a cell must not outweigh
# everything its lineage has established.
#
# What does NOT depend on kappa: an UNTRIED cell has n = 0, so it inherits its parent
# EXACTLY for every kappa > 0.  kappa governs only how fast a MEASURED cell moves off the
# inherited value -- which is why the v1 defect is fixed regardless of this constant, and
# why getting it wrong costs accuracy rather than the invariant.
#
# It is a search setting, not a calibrated prior.  Production reward noise is unmeasured.
DEFAULT_KAPPA = 1.0


def _validate_chain(chain) -> tuple[tuple[str, ...], ...]:
    """Refuse any chain that is not a nested root-to-cell hierarchy."""
    levels = tuple(tuple(level) for level in chain)
    if len(levels) < 2:
        raise ValueError("a backoff chain needs a root and at least the full cell")
    if levels[0] != ROOT:
        raise ValueError(f"a backoff chain must start at the root (), got {levels[0]!r}")
    if set(levels[-1]) != set(CREDIT_AXES):
        raise ValueError(f"a backoff chain must end at the full cell, got {levels[-1]!r}")
    for level in levels:
        unknown = set(level) - set(CREDIT_AXES)
        if unknown:
            raise ValueError(f"backoff level names unknown axes: {sorted(unknown)}")
        if len(set(level)) != len(level):
            raise ValueError(f"backoff level repeats an axis: {level!r}")
    for coarse, fine in pairwise(levels):
        if not set(coarse) < set(fine):
            raise ValueError(f"backoff chain is not strictly nested: {coarse!r} -> {fine!r}")
    return levels


class HierarchicalCredit(PopulationCredit):
    """v1 evidence, v2 allocation: count-shrunk backoff with a preserved uniform floor.

    Everything that WRITES evidence -- `observe`, `cell`, `marginal`, `payload` -- is
    inherited from the frozen `PopulationCredit` unchanged, so an arm swap changes the
    allocation rule and nothing else.
    """

    def __init__(
        self,
        *,
        exploration_floor: float = DEFAULT_EXPLORATION_FLOOR,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        kappa: float = DEFAULT_KAPPA,
        chain: str | tuple = PRODUCTION_CHAIN,
    ) -> None:
        super().__init__(exploration_floor=exploration_floor, prior_weight=prior_weight)
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be a finite positive pseudo-trial count")
        self.kappa = float(kappa)
        self.chain_name = chain if isinstance(chain, str) else "custom"
        self.chain = _validate_chain(CHAINS[chain] if isinstance(chain, str) else chain)

    # ---- Node statistics, derived from the joint exactly as `marginal` is ----

    def _level_stats(self, level: tuple[str, ...]) -> dict[tuple[str, ...], tuple[int, float]]:
        """`(trials, positive_improvement_sum)` per node at one level of the chain.

        Summed over the whole joint, never stored, so a node can never disagree with the
        cells beneath it -- the same contract `PopulationCredit.marginal` keeps.
        """
        totals: dict[tuple[str, ...], list] = {}
        for key, cell in self.cells.items():
            node = tuple(getattr(key, axis) for axis in level)
            bucket = totals.setdefault(node, [0, 0.0])
            bucket[0] += cell.trials
            bucket[1] += cell.positive_improvement_sum
        return {node: (int(t), float(s)) for node, (t, s) in totals.items()}

    def _stats_table(self) -> list[dict[tuple[str, ...], tuple[int, float]]]:
        return [self._level_stats(level) for level in self.chain]

    def shrunk_values(self, key: CreditKey) -> list[float]:
        """`Q~` at every level of the chain for one cell, root first.

        Exposed because the inheritance story is a statement about these numbers: an
        untried leaf returns exactly its parent's value, and the gap between two leaves in
        different basins is the gap between their basins.
        """
        table = self._stats_table()
        out: list[float] = []
        backoff = self.prior_weight
        for level, stats in zip(self.chain, table, strict=True):
            node = tuple(getattr(key, axis) for axis in level)
            trials, positive = stats.get(node, (0, 0.0))
            raw = positive / trials if trials else 0.0
            weight = trials / (trials + self.kappa)
            backoff = weight * raw + (1.0 - weight) * backoff
            out.append(backoff)
        return out

    def shrunk_value(self, key: CreditKey) -> float:
        """The v2 leaf value: v1's `value` with the flat prior replaced by inheritance."""
        return self.shrunk_values(key)[-1]

    def untried_attachment(self, key: CreditKey) -> tuple[int, tuple[str, ...]] | None:
        """The node whose UNTRIED-CHILDREN POOL this cell falls into, or `None`.

        Walk the chain from the root and find the first level at which this cell's node
        carries no evidence; the level above it is the node whose pooled untried portion
        the cell is paid out of.  `None` means every ancestor including the leaf has been
        tried, so the cell is paid from the tried branch instead.

        This is the identity the count invariant is stated over.  Naming the node
        explicitly matters: "the region a proliferating cell belongs to" is a property of
        the CHAIN, not of any one axis, and defining it per-axis by hand silently measures
        a SUBSET of the pool -- which reads as a violation when it is only dilution.
        """
        table = self._stats_table()
        previous: tuple[int, tuple[str, ...]] | None = None
        for level_index, (level, stats) in enumerate(zip(self.chain, table, strict=True)):
            node = tuple(getattr(key, axis) for axis in level)
            if stats.get(node, (0, 0.0))[0] == 0:
                return previous
            previous = (level_index, node)
        return None

    # ---- Allocation ----

    def exploitation_shares(self, keys) -> np.ndarray:
        """`q_exploit`: the hierarchical mass, BEFORE the uniform exploration floor.

        Separated from `allocate` on purpose -- the count invariant is a statement about
        this quantity, and folding the floor in would hide it.
        """
        keys = list(keys)
        if not keys:
            return np.zeros(0, dtype=float)
        table = self._stats_table()

        # Q~ per node, computed top-down so each node reads its own parent's value.
        node_value: dict[tuple[int, tuple[str, ...]], float] = {}
        for level_index, (level, stats) in enumerate(zip(self.chain, table, strict=True)):
            nodes = {tuple(getattr(key, axis) for axis in level) for key in keys} | set(stats)
            for node in nodes:
                if level_index == 0:
                    parent_value = self.prior_weight
                else:
                    coarse = self.chain[level_index - 1]
                    take = [level.index(axis) for axis in coarse]
                    parent_value = node_value[(level_index - 1, tuple(node[i] for i in take))]
                trials, positive = stats.get(node, (0, 0.0))
                raw = positive / trials if trials else 0.0
                weight = trials / (trials + self.kappa)
                node_value[(level_index, node)] = weight * raw + (1.0 - weight) * parent_value

        def node_trials(level_index: int, node: tuple[str, ...]) -> int:
            return table[level_index].get(node, (0, 0.0))[0]

        share: dict[tuple[int, tuple[str, ...]], float] = {(0, ()): 1.0}
        current: dict[tuple[int, tuple[str, ...]], list[CreditKey]] = {(0, ()): keys}
        for level_index in range(len(self.chain) - 1):
            child_level = self.chain[level_index + 1]
            nxt: dict[tuple[int, tuple[str, ...]], list[CreditKey]] = {}
            for node, node_keys in current.items():
                groups: dict[tuple[int, tuple[str, ...]], list[CreditKey]] = {}
                for key in node_keys:
                    child = (level_index + 1, tuple(getattr(key, axis) for axis in child_level))
                    groups.setdefault(child, []).append(key)
                self._split(share, node, groups, node_value, node_trials)
                nxt.update(groups)
            current = nxt

        leaf_level = len(self.chain) - 1
        shares = np.asarray(
            [
                share[(leaf_level, tuple(getattr(key, axis) for axis in self.chain[-1]))]
                for key in keys
            ],
            dtype=float,
        )
        total = float(shares.sum())
        return shares / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))

    def _split(self, share, node, groups, node_value, node_trials) -> None:
        """Divide `share[node]` among its children; untried children share ONE portion."""
        available = share[node]
        tried = [child for child in groups if node_trials(*child) > 0]
        untried = [child for child in groups if node_trials(*child) == 0]
        weights = {child: max(0.0, node_value[child]) for child in tried}
        backoff = max(0.0, node_value[node]) if untried else 0.0
        denominator = sum(weights.values()) + backoff
        if denominator <= 0.0:
            # No measured upside anywhere beneath this node and no optimism left.  Fall
            # back to uniform, but STILL as one portion per tried child plus ONE shared
            # portion for the untried pool -- a plain uniform split over children would be
            # count-proportional and would reopen the proliferation hole in exactly the
            # corner where every value is zero.  Reachable only at `prior_weight == 0`,
            # and fuzzed there on purpose.
            portions = len(tried) + (1 if untried else 0)
            for child in tried:
                share[child] = available / portions
            if untried:
                for child in untried:
                    share[child] = available / portions / len(untried)
            return
        for child in tried:
            share[child] = available * weights[child] / denominator
        if untried:
            pooled = available * backoff / denominator
            for child in untried:
                share[child] = pooled / len(untried)

    def allocate(self, keys) -> np.ndarray:
        """Budget shares: `(1 - eps) * q_exploit + eps * uniform`, floor identical to v1."""
        keys = list(keys)
        if not keys:
            return np.zeros(0, dtype=float)
        if len(set(keys)) != len(keys):
            raise ValueError("allocation keys must be distinct cells")
        exploit = self.exploitation_shares(keys)
        explore = np.full(len(keys), 1.0 / len(keys))
        mixed = (1.0 - self.exploration_floor) * exploit + self.exploration_floor * explore
        return mixed / mixed.sum()

    def allocation_report(self, keys) -> dict:
        """Auditable record of one v2 decision, with the v1 value shown alongside."""
        keys = list(keys)
        shares = self.allocate(keys)
        exploit = self.exploitation_shares(keys)
        return {
            "schema_version": SCHEMA_VERSION,
            "chain": self.chain_name,
            "chain_levels": ["|".join(level) if level else "root" for level in self.chain],
            "kappa": self.kappa,
            "exploration_floor": self.exploration_floor,
            "prior_weight": self.prior_weight,
            "minimum_share": self.exploration_floor / len(keys) if keys else 0.0,
            "cells": [
                {
                    **key.payload(),
                    "share": float(share),
                    "exploitation_share": float(exploit_share),
                    "shrunk_value_v2": self.shrunk_value(key),
                    "value_v1": super(HierarchicalCredit, self).value(key),
                    **self.cell(key).payload(),
                }
                for key, share, exploit_share in zip(keys, shares, exploit, strict=True)
            ],
        }


def region_exploitation_mass(credit: HierarchicalCredit, keys, region) -> float:
    """Total `q_exploit` carried by the cells of `keys` that satisfy `region`.

    The count invariant is stated on this quantity, so the audit and the tests compute it
    the same way through one helper rather than each rebuilding the sum.
    """
    keys = list(keys)
    exploit = credit.exploitation_shares(keys)
    return float(sum(s for key, s in zip(keys, exploit, strict=True) if region(key)))
