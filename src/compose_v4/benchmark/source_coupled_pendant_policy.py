"""Training-source coupling for complete scaffold-decoration programs.

The existing joint-mass law chooses the total added-atom count and each site's
size/ring category. This optional content law samples a shared training source
row for compatible categories, so pendant choices can reflect training-molecule
co-occurrence. Cuts from one molecule need not form a disjoint observed
decoration bundle; the executor still decides whether the resulting program is
legal. An independent-content lane preserves the original proposal support.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isclose

from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler


class SourceCoupledPendantSampler(JointMassPendantSampler):
    """Couple within-category pendant content through a train-only source row."""

    def __init__(self, catalog: dict, mass_prior: dict, *, coupled_probability: float = 0.5):
        if not 0.0 < coupled_probability < 1.0:
            raise ValueError("source coupling must retain both proposal lanes")
        super().__init__(catalog, mass_prior)
        source_rows = tuple(catalog["accepted_source_rows"])
        if len(source_rows) != catalog["training_molecules"] or len(set(source_rows)) != len(
            source_rows
        ):
            raise ValueError("training-source rows are missing or duplicated")
        admitted_rows = set(source_rows)
        cut_counts = Counter(row for entry in catalog["entries"] for row in entry["source_rows"])
        if not set(cut_counts) <= admitted_rows:
            raise ValueError("pendant catalog contains a source outside the training split")
        for entry in catalog["entries"]:
            expected = sum(1 / cut_counts[row] for row in entry["source_rows"])
            if not isclose(
                expected,
                entry["source_balanced_weight"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("training-source balanced pendant mass changed")
        group_rows = {}
        for context, groups in self.pools.items():
            for group in groups:
                rows = defaultdict(list)
                for entry in group.entries:
                    for row in entry["source_rows"]:
                        rows[row].append(entry)
                group_rows[(context, group.atoms, group.rings)] = {
                    row: tuple(entries) for row, entries in rows.items()
                }
        self.group_rows = group_rows
        self.coupled_probability = float(coupled_probability)

    def sample(self, contexts: tuple[str, ...], capacity: int, rng):
        base_entries, receipt = super().sample(contexts, capacity, rng)
        group_keys = tuple(
            (draw["context"], draw["heavy_atoms"], draw["rings"]) for draw in receipt["draws"]
        )
        common_rows = set(self.group_rows[group_keys[0]])
        for key in group_keys[1:]:
            common_rows.intersection_update(self.group_rows[key])
        coupled = bool(common_rows) and bool(rng.random() < self.coupled_probability)
        chosen = base_entries
        source_row = None
        if coupled:
            source_row = sorted(common_rows)[int(rng.integers(len(common_rows)))]
            chosen = tuple(
                entries[int(rng.integers(len(entries)))]
                for key in group_keys
                for entries in (self.group_rows[key][source_row],)
            )
            for draw, entry in zip(receipt["draws"], chosen, strict=True):
                draw.update(
                    {
                        "independent_rooted_smiles": draw["rooted_smiles"],
                        "rooted_smiles": entry["rooted_smiles"],
                        "source_rows": entry["source_rows"],
                        "content_probability_given_group": None,
                    }
                )
        receipt.update(
            {
                "schema": "source_coupled_pendant_plan_v1",
                "content_law": "half_independent_half_uniform_shared_training_source",
                "shared_training_rows": len(common_rows),
                "source_coupling_requested_probability": self.coupled_probability,
                "source_coupling_used": coupled,
                "shared_training_source_row": source_row,
                "selection_uses_qed_sa": False,
                "observed_decoration_bundle_claim": False,
            }
        )
        return tuple(chosen), receipt
