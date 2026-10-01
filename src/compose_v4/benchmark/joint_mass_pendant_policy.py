"""Train-only joint mass planning for exact scaffold-decoration programs.

The official decoration prompt exposes a fixed Murcko-like scaffold and its
attachment interfaces. Training molecules provide a distribution of the TOTAL
decoration mass conditional on scaffold size and interface count. This sampler
draws that mass once, then allocates observed training pendants jointly under
the exact COMPOSE atom limit. It does not choose the shortest fragment, score
QED/SA, modify a benchmark core, or relax exact execution.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite

import numpy as np

from compose_v4.benchmark.pendant_completion_policy import PendantCompletionSampler


class JointMassPendantSampler(PendantCompletionSampler):
    """Sample a feasible whole decoration budget before its constituent pieces."""

    def __init__(self, catalog: dict, prior: dict):
        if (
            prior.get("schema") != "split_first_training_decoration_mass_prior_v1"
            or catalog.get("source_sha256") != prior.get("source_sha256")
            or catalog.get("split") != prior.get("split")
            or catalog.get("split", {}).get("partition") != "train"
            or prior.get("quality_labels_used") is not False
            or prior.get("benchmark_prompts_used_to_fit") is not False
        ):
            raise ValueError(
                "joint decoration mass prior lacks the same quality-blind train lineage"
            )
        super().__init__(catalog)
        counts = Counter()
        for core_size, interfaces, added, count in prior["counts"]:
            if (
                not 1 <= core_size <= 40
                or not 1 <= interfaces <= added <= 40 - core_size
                or not isinstance(count, int)
                or count < 1
                or (core_size, interfaces, added) in counts
            ):
                raise ValueError("joint decoration mass prior has malformed or duplicate cells")
            counts[(core_size, interfaces, added)] = count
        if not counts or sum(counts.values()) != prior["rows_with_murcko_decorations"]:
            raise ValueError("joint decoration mass prior count total changed")
        self.mass_counts = counts
        self._exact_suffix_cache: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, ...]] = {}

    def _exact_suffix(self, contexts: tuple[str, ...], capacity: int) -> tuple[np.ndarray, ...]:
        key = contexts, capacity
        if key in self._exact_suffix_cache:
            return self._exact_suffix_cache[key]
        if not contexts or not 1 <= capacity <= 40:
            raise ValueError("joint mass plan needs interfaces and capacity within 1..40")
        if any(context not in self.pools for context in contexts):
            raise ValueError("no observed training pendant at a supplied attachment context")
        suffix = [np.zeros(capacity + 1, dtype=np.float64) for _ in range(len(contexts) + 1)]
        suffix[-1][0] = 1.0
        for index in range(len(contexts) - 1, -1, -1):
            for group in self.pools[contexts[index]]:
                if group.atoms <= capacity:
                    suffix[index][group.atoms :] += (
                        group.mass * suffix[index + 1][: capacity + 1 - group.atoms]
                    )
        if not np.any(suffix[0] > 0) or not np.isfinite(suffix[0]).all():
            raise ValueError("training pendants have no feasible exact-size completion")
        for array in suffix:
            array.setflags(write=False)
        if len(self._exact_suffix_cache) == 64:
            self._exact_suffix_cache.pop(next(iter(self._exact_suffix_cache)))
        self._exact_suffix_cache[key] = tuple(suffix)
        return tuple(suffix)

    def sample(
        self, contexts: tuple[str, ...], capacity: int, rng
    ) -> tuple[tuple[dict, ...], dict]:
        contexts = tuple(contexts)
        suffix = self._exact_suffix(contexts, capacity)
        core_size = 40 - capacity
        interfaces = len(contexts)
        feasible_added = tuple(int(value) for value in np.flatnonzero(suffix[0] > 0))
        if not feasible_added or core_size < 1:
            raise ValueError("no attainable decoration mass in declared support")
        nearby = Counter()
        for (train_core, train_interfaces, added), count in self.mass_counts.items():
            if train_interfaces == interfaces and abs(train_core - core_size) <= 2:
                nearby[added] += count
        support_source = "core_plus_minus_2_and_exact_interfaces"
        if not nearby:
            support_source = "exact_interfaces_all_core_sizes"
            for (_train_core, train_interfaces, added), count in self.mass_counts.items():
                if train_interfaces == interfaces:
                    nearby[added] += count
        if not nearby:
            support_source = "uniform_pseudocount_only"
        # A unit pseudocount keeps every executor-feasible size available.
        weights = np.asarray([nearby[added] + 1 for added in feasible_added], dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise RuntimeError("training decoration-mass weights are not finite")
        weights /= weights.sum()
        size_index = int(rng.choice(len(feasible_added), p=weights))
        selected_added = feasible_added[size_index]
        remaining = selected_added
        chosen, draws = [], []
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
            if (
                not len(probabilities)
                or not isfinite(float(probabilities.sum()))
                or probabilities.sum() <= 0
            ):
                raise RuntimeError("joint-mass suffix lost a feasible group")
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
                    "group_probability_given_exact_mass": float(probabilities[group_index]),
                    "content_probability_given_group": float(
                        group.content_probabilities[content_index]
                    ),
                    "rooted_smiles": entry["rooted_smiles"],
                    "source_rows": entry["source_rows"],
                }
            )
            remaining -= group.atoms
        if remaining != 0:
            raise RuntimeError("joint decoration plan failed to realize selected total mass")
        return tuple(chosen), {
            "schema": "joint_mass_pendant_plan_v1",
            "initial_atom_capacity": capacity,
            "unused_atom_capacity": capacity - selected_added,
            "core_heavy_atoms": core_size,
            "interfaces": interfaces,
            "planned_heavy_atoms": core_size + selected_added,
            "planned_decoration_mass": selected_added,
            "reachable_masses": len(feasible_added),
            "train_context_mass_count": sum(nearby.values()),
            "train_support_source": support_source,
            "mass_probability": float(weights[size_index]),
            "draws": draws,
            "qed_sa_guidance": False,
            "all_interfaces_planned_together": True,
        }
