"""Training-derived whole-completion cells with exact stochastic allocation.

This is a proposal-density model, not QED/SA guidance or an executor restriction.
All compatible region allocations have positive mass. Dynamic programming sums
allocations exactly instead of pruning or constructing a beam of candidates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointCompletionPrior:
    counts: tuple[tuple[int, int, int], ...]
    pseudocount_mass: float = 1.0

    def __post_init__(self):
        if not self.counts or self.pseudocount_mass != 1.0:
            raise ValueError("joint prior requires nonempty counts and frozen pseudocount mass one")
        seen = set()
        for atoms, rings, count in self.counts:
            if (
                any(type(v) is not int for v in (atoms, rings, count))
                or not 1 <= atoms <= 40
                or rings < 0
                or count < 1
                or (atoms, rings) in seen
            ):
                raise ValueError(f"invalid or duplicate joint prior cell: {(atoms, rings, count)}")
            seen.add((atoms, rings))

    def cell_probabilities(self, cells):
        cells = tuple(tuple(c) for c in cells)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("reachable cells must be nonempty and unique")
        if any(len(c) != 2 or not 1 <= c[0] <= 40 or c[1] < 0 for c in cells):
            raise ValueError("reachable cell outside whole-molecule descriptor support")
        lookup = {(a, r): n for a, r, n in self.counts}
        weights = np.asarray(
            [lookup.get(c, 0) + self.pseudocount_mass / len(cells) for c in cells], dtype=np.float64
        )
        return weights / weights.sum()

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != "joint_completion_structural_prior_v1":
            raise ValueError("not a joint-completion prior artifact")
        return cls(tuple(tuple(c) for c in data["counts"]), data["pseudocount_mass"])


@dataclass(frozen=True)
class _Group:
    atoms: int
    rings: int
    entries: tuple[dict[str, Any], ...]
    weights: np.ndarray
    mass: float


@dataclass(frozen=True)
class JointPlanTable:
    cells: tuple[tuple[int, int], ...]
    probabilities: np.ndarray
    suffix_mass: tuple[np.ndarray, ...]
    context_keys: tuple[str, ...]
    core_atoms: int
    core_rings: int


class JointCompletionSampler:
    """Sample a final structural cell, then a joint region plan inside its fiber."""

    def __init__(self, entries, prior: JointCompletionPrior):
        self.prior = prior
        grouped = defaultdict(lambda: defaultdict(list))
        seen = set()
        for entry in entries:
            if len(entry["contexts"]) != 1:
                continue
            key = entry["contexts"][0]
            atoms, rings, count = entry["heavy_atoms"], entry["ring_count"], entry["occurrences"]
            identity = (key, entry["rooted_smiles"])
            if (
                any(type(v) is not int for v in (atoms, rings, count))
                or not 1 <= atoms <= 40
                or rings < 0
                or count < 1
                or identity in seen
            ):
                raise ValueError(f"invalid or duplicate training region: {identity}")
            seen.add(identity)
            grouped[key][(atoms, rings)].append(entry)
        self.pools = {}
        self._tables = {}
        for key, groups in sorted(grouped.items()):
            total = sum(np.sqrt(e["occurrences"]) for group in groups.values() for e in group)
            prepared = []
            for (atoms, rings), values in sorted(groups.items()):
                values.sort(key=lambda e: e["rooted_smiles"])
                weights = np.sqrt([e["occurrences"] for e in values])
                prepared.append(
                    _Group(
                        atoms,
                        rings,
                        tuple(values),
                        weights / weights.sum(),
                        float(weights.sum() / total),
                    )
                )
            self.pools[key] = tuple(prepared)

    def plan_table(self, context_keys: tuple[str, ...], core_atoms: int, core_rings: int):
        context_keys = tuple(context_keys)
        cache_key = (context_keys, core_atoms, core_rings)
        if cache_key in self._tables:
            return self._tables[cache_key]
        if (
            not context_keys
            or type(core_atoms) is not int
            or type(core_rings) is not int
            or not 1 <= core_atoms < 40
            or core_rings < 0
        ):
            raise ValueError("invalid retained-core descriptors or empty interface list")
        if any(key not in self.pools for key in context_keys):
            raise ValueError("no observed region has a required boundary context")
        room = 40 - core_atoms
        pools = [tuple(g for g in self.pools[k] if g.atoms <= room) for k in context_keys]
        if any(not p for p in pools) or sum(min(g.atoms for g in p) for p in pools) > room:
            raise ValueError("required interfaces have no feasible joint allocation")
        max_rings = sum(max(g.rings for g in p) for p in pools)
        suffix = [
            np.zeros((room + 1, max_rings + 1), dtype=np.float64) for _ in range(len(pools) + 1)
        ]
        suffix[-1][0, 0] = 1.0
        for i in range(len(pools) - 1, -1, -1):
            for group in pools[i]:
                size, rings = group.atoms, group.rings
                suffix[i][size:, rings:] += (
                    group.mass * suffix[i + 1][: room + 1 - size, : max_rings + 1 - rings]
                )
        feasible = np.argwhere(suffix[0] > 0)
        if not len(feasible):
            raise ValueError("no positive-mass joint allocation")
        cells = tuple((core_atoms + int(a), core_rings + int(r)) for a, r in feasible)
        probabilities = self.prior.cell_probabilities(cells)
        for arr in (*suffix, probabilities):
            arr.setflags(write=False)
        table = JointPlanTable(
            cells, probabilities, tuple(suffix), context_keys, core_atoms, core_rings
        )
        if len(self._tables) == 64:
            self._tables.pop(next(iter(self._tables)))
        self._tables[cache_key] = table
        return table

    def sample(self, context_keys, core_atoms, core_rings, rng):
        table = self.plan_table(tuple(context_keys), core_atoms, core_rings)
        cell_index = int(rng.choice(len(table.cells), p=table.probabilities))
        target_atoms, target_rings = table.cells[cell_index]
        atoms, rings = target_atoms - core_atoms, target_rings - core_rings
        selected, conditional_probability = [], 1.0
        for index, key in enumerate(table.context_keys):
            groups, masses = [], []
            for group in self.pools[key]:
                if group.atoms > atoms or group.rings > rings:
                    continue
                mass = (
                    group.mass
                    * table.suffix_mass[index + 1][atoms - group.atoms, rings - group.rings]
                )
                if mass > 0:
                    groups.append(group)
                    masses.append(mass)
            weights = np.asarray(masses, dtype=np.float64)
            if not len(weights) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise RuntimeError("joint allocation lost positive suffix mass")
            weights /= weights.sum()
            choice = int(rng.choice(len(groups), p=weights))
            group = groups[choice]
            entry_index = int(rng.choice(len(group.entries), p=group.weights))
            selected.append(group.entries[entry_index])
            conditional_probability *= float(weights[choice] * group.weights[entry_index])
            atoms -= group.atoms
            rings -= group.rings
        if atoms or rings:
            raise RuntimeError("joint sampled plan does not realize its structural cell")
        return tuple(selected), {
            "schema": "joint_completion_plan_v1",
            "core_heavy_atoms": core_atoms,
            "core_rings": core_rings,
            "planned_heavy_atoms": target_atoms,
            "planned_rings": target_rings,
            "reachable_cells": len(table.cells),
            "cell_probability": float(table.probabilities[cell_index]),
            "conditional_content_probability": conditional_probability,
            "qed_sa_guidance": False,
            "all_interfaces_planned_together": True,
        }
