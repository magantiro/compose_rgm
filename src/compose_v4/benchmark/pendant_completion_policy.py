"""Joint capacity-conditioned sampling of observed training pendant regions.

The product law uses a source-balanced empirical region-size distribution for
each supplied attachment context. Exact dynamic programming conditions all
interfaces on the unchanged 40-atom support without drawing and rejecting
oversized combinations. Chemical execution remains the shared COMPOSE stack.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class _Group:
    atoms: int
    rings: int
    entries: tuple[dict, ...]
    content_probabilities: np.ndarray
    mass: float


class PendantCompletionSampler:
    def __init__(
        self,
        catalog: dict,
        *,
        content_allocation: Literal["sqrt_source_weight", "uniform_within_cell"] = (
            "sqrt_source_weight"
        ),
    ):
        if content_allocation not in ("sqrt_source_weight", "uniform_within_cell"):
            raise ValueError(f"unknown pendant content allocation: {content_allocation}")
        if catalog.get("schema") != "split_first_training_pendant_catalog_v1":
            raise ValueError("not a split-first training pendant catalog")
        if catalog.get("split", {}).get("partition") != "train":
            raise ValueError("pendant catalog is not from the train partition")
        if catalog.get("selection_uses_qed_sa") or catalog.get(
            "benchmark_prompts_used_for_selection"
        ):
            raise ValueError("pendant content must be task-target and quality blind")
        grouped = defaultdict(lambda: defaultdict(list))
        identities = set()
        for entry in catalog["entries"]:
            key = entry["context"], entry["rooted_smiles"]
            mass = entry["source_balanced_weight"]
            if (
                key in identities
                or not isinstance(mass, (int, float))
                or not isfinite(mass)
                or mass <= 0
            ):
                raise ValueError(f"duplicate or nonpositive pendant content: {key}")
            identities.add(key)
            if entry["occurrences"] != len(entry["source_rows"]) or entry["occurrences"] < 1:
                raise ValueError(f"pendant source provenance differs from occurrence count: {key}")
            molecule = Chem.MolFromSmiles(entry["rooted_smiles"])
            if molecule is None:
                raise ValueError(f"pendant region does not parse: {key}")
            dummies = [a for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]
            if len(dummies) != 1 or dummies[0].GetIsotope() != 1 or dummies[0].GetDegree() != 1:
                raise ValueError(f"pendant region lacks one valid boundary: {key}")
            if (
                molecule.GetNumHeavyAtoms() != entry["heavy_atoms"]
                or molecule.GetRingInfo().NumRings() != entry["ring_count"]
                or not 1 <= entry["heavy_atoms"] <= 40
            ):
                raise ValueError(f"pendant region metadata changed: {key}")
            grouped[entry["context"]][(entry["heavy_atoms"], entry["ring_count"])].append(entry)
        if not grouped:
            raise ValueError("pendant catalog has no train-supported boundary context")
        self.content_allocation = content_allocation
        self.pools = {}
        self._suffix_cache = {}
        for context, buckets in sorted(grouped.items()):
            total = sum(e["source_balanced_weight"] for items in buckets.values() for e in items)
            groups = []
            for (atoms, rings), items in sorted(buckets.items()):
                entries = tuple(sorted(items, key=lambda e: e["rooted_smiles"]))
                weights = np.asarray(
                    [
                        1.0
                        if content_allocation == "uniform_within_cell"
                        else sqrt(e["source_balanced_weight"])
                        for e in entries
                    ],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                weights.setflags(write=False)
                groups.append(
                    _Group(
                        atoms,
                        rings,
                        entries,
                        weights,
                        sum(e["source_balanced_weight"] for e in entries) / total,
                    )
                )
            self.pools[context] = tuple(groups)

    def _suffix(self, contexts: tuple[str, ...], capacity: int) -> tuple[np.ndarray, ...]:
        key = contexts, capacity
        if key in self._suffix_cache:
            return self._suffix_cache[key]
        if not contexts or not 1 <= capacity <= 40:
            raise ValueError("pendant plan needs interfaces and capacity within 1..40")
        if any(context not in self.pools for context in contexts):
            raise ValueError("no observed training pendant at a supplied attachment context")
        if (
            sum(min(group.atoms for group in self.pools[context]) for context in contexts)
            > capacity
        ):
            raise ValueError("declared attachment interfaces exceed atom capacity")
        suffix = [np.zeros(capacity + 1, dtype=np.float64) for _ in range(len(contexts) + 1)]
        suffix[-1][:] = 1.0
        for index in range(len(contexts) - 1, -1, -1):
            for group in self.pools[contexts[index]]:
                if group.atoms <= capacity:
                    suffix[index][group.atoms :] += (
                        group.mass * suffix[index + 1][: capacity + 1 - group.atoms]
                    )
        if not isfinite(float(suffix[0][capacity])) or suffix[0][capacity] <= 0:
            raise ValueError("training pendant law has no feasible joint completion")
        for array in suffix:
            array.setflags(write=False)
        if len(self._suffix_cache) == 64:
            self._suffix_cache.pop(next(iter(self._suffix_cache)))
        self._suffix_cache[key] = tuple(suffix)
        return tuple(suffix)

    def sample(
        self, contexts: tuple[str, ...], capacity: int, rng
    ) -> tuple[tuple[dict, ...], dict]:
        contexts = tuple(contexts)
        suffix = self._suffix(contexts, capacity)
        chosen, draws, remaining = [], [], capacity
        for index, context in enumerate(contexts):
            eligible = [
                group
                for group in self.pools[context]
                if group.atoms <= remaining and suffix[index + 1][remaining - group.atoms] > 0
            ]
            probabilities = np.asarray(
                [group.mass * suffix[index + 1][remaining - group.atoms] for group in eligible],
                dtype=np.float64,
            )
            if not len(probabilities) or probabilities.sum() <= 0:
                raise RuntimeError("positive-mass pendant suffix lost a feasible group")
            probabilities /= probabilities.sum()
            group_index = int(rng.choice(len(eligible), p=probabilities))
            group = eligible[group_index]
            content_index = int(rng.choice(len(group.entries), p=group.content_probabilities))
            entry = group.entries[content_index]
            chosen.append(entry)
            draws.append(
                {
                    "context": context,
                    "heavy_atoms": group.atoms,
                    "rings": group.rings,
                    "group_probability_given_remaining_capacity": float(probabilities[group_index]),
                    "content_probability_given_group": float(
                        group.content_probabilities[content_index]
                    ),
                    "rooted_smiles": entry["rooted_smiles"],
                    "source_rows": entry["source_rows"],
                }
            )
            remaining -= group.atoms
        return tuple(chosen), {
            "schema": "pendant_completion_plan_v1",
            "initial_atom_capacity": capacity,
            "unused_atom_capacity": remaining,
            "joint_capacity_probability": float(suffix[0][capacity]),
            "draws": draws,
            "qed_sa_guidance": False,
            "all_interfaces_planned_together": True,
        }
