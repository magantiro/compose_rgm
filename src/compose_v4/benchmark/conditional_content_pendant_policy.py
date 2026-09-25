"""Train-conditioned content draws within the frozen joint decoration-size law.

The size and ring-category proposal from ``JointMassPendantSampler`` remains
unchanged. Only the rooted pendant selected within a category is conditioned on
the training molecule's Murcko core size and number of decoration interfaces.
Every globally supported pendant retains positive probability.
"""

from __future__ import annotations

from collections import Counter
from math import isclose

import numpy as np

from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler


class ConditionalContentPendantSampler(JointMassPendantSampler):
    def __init__(
        self,
        catalog: dict,
        mass_prior: dict,
        content_context_prior: dict,
        *,
        catalog_sha256: str,
    ):
        if (
            content_context_prior.get("schema")
            != "split_first_training_decoration_content_context_v1"
            or content_context_prior.get("source_sha256") != catalog.get("source_sha256")
            or content_context_prior.get("catalog_sha256") != catalog_sha256
            or content_context_prior.get("split") != catalog.get("split")
            or content_context_prior.get("quality_labels_used") is not False
            or content_context_prior.get("benchmark_prompts_used_to_fit") is not False
        ):
            raise ValueError("conditional content prior lost frozen train-only provenance")
        super().__init__(catalog, mass_prior)
        raw_contexts = content_context_prior["row_contexts"]
        if (
            len(raw_contexts) != catalog["training_molecules"]
            or [row[0] for row in raw_contexts] != catalog["accepted_source_rows"]
            or any(len(row) != 3 for row in raw_contexts)
        ):
            raise ValueError("conditional content prior changed frozen training rows")
        self.row_contexts = {
            int(row): (int(core), int(interfaces)) for row, core, interfaces in raw_contexts
        }
        self.cut_counts = Counter(
            source_row for entry in catalog["entries"] for source_row in entry["source_rows"]
        )
        if any(row not in self.row_contexts for row in self.cut_counts):
            raise ValueError("pendant catalog references an unrecognized train row")
        for entry in catalog["entries"]:
            rebuilt_mass = sum(1 / self.cut_counts[row] for row in entry["source_rows"])
            if not isclose(
                rebuilt_mass,
                entry["source_balanced_weight"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("source-balanced pendant mass cannot be reconstructed")
        self._conditioned_content_cache: dict[
            tuple[int, int, str, int, int], tuple[np.ndarray, float]
        ] = {}

    def content_weights(
        self,
        core_size: int,
        interfaces: int,
        context: str,
        atoms: int,
        rings: int,
    ) -> tuple[np.ndarray, float]:
        """Return smoothed conditional probabilities and observed training mass."""
        key = core_size, interfaces, context, atoms, rings
        if key in self._conditioned_content_cache:
            return self._conditioned_content_cache[key]
        group = next(
            (
                group
                for group in self.pools[context]
                if group.atoms == atoms and group.rings == rings
            ),
            None,
        )
        if group is None:
            raise ValueError(f"unsupported pendant group: {key}")
        global_mass = np.asarray(
            [entry["source_balanced_weight"] for entry in group.entries], dtype=np.float64
        )
        global_probability = global_mass / global_mass.sum()
        observed = np.asarray(
            [
                sum(
                    1 / self.cut_counts[row]
                    for row in entry["source_rows"]
                    if self.row_contexts[row][1] == interfaces
                    and abs(self.row_contexts[row][0] - core_size) <= 2
                )
                for entry in group.entries
            ],
            dtype=np.float64,
        )
        # One global pseudo-observation preserves the complete training-derived
        # content support. Square-root flattening matches the original catalog law.
        weights = np.sqrt(observed + global_probability)
        weights /= weights.sum()
        weights.setflags(write=False)
        result = weights, float(observed.sum())
        if len(self._conditioned_content_cache) == 256:
            self._conditioned_content_cache.pop(next(iter(self._conditioned_content_cache)))
        self._conditioned_content_cache[key] = result
        return result

    def sample(self, contexts: tuple[str, ...], capacity: int, rng):
        base_entries, receipt = super().sample(contexts, capacity, rng)
        core_size = receipt["core_heavy_atoms"]
        interfaces = receipt["interfaces"]
        chosen = []
        for base_entry, draw in zip(base_entries, receipt["draws"], strict=True):
            group = next(
                group
                for group in self.pools[draw["context"]]
                if group.atoms == draw["heavy_atoms"] and group.rings == draw["rings"]
            )
            weights, observed_mass = self.content_weights(
                core_size,
                interfaces,
                draw["context"],
                draw["heavy_atoms"],
                draw["rings"],
            )
            index = int(rng.choice(len(group.entries), p=weights))
            entry = group.entries[index]
            chosen.append(entry)
            draw.update(
                {
                    "base_rooted_smiles": base_entry["rooted_smiles"],
                    "rooted_smiles": entry["rooted_smiles"],
                    "source_rows": entry["source_rows"],
                    "content_probability_given_group": float(weights[index]),
                    "conditional_training_mass_in_group": observed_mass,
                }
            )
        receipt.update(
            {
                "schema": "conditional_content_pendant_plan_v1",
                "content_prior": "train_only_murcko_core_size_plus_minus_2_and_exact_interfaces",
                "global_pseudocount_mass": 1.0,
                "selection_uses_qed_sa": False,
            }
        )
        return tuple(chosen), receipt
