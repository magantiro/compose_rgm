"""An explicit winner-informed proposal prior over the caller's applicable row.

This is supervised editing frequency, not task value or a future-value model.
The reference remains visible, generic stays active, and unseen descriptors are
retained through the reference pseudo-observation and the qualified KL floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.option_demonstrations import source_weights
from compose_v4.control.option_policy import OptionDistribution, conservative_option_distribution


@dataclass(frozen=True)
class DemonstrationPrior:
    frequencies: tuple[tuple[str, float], ...]
    training_identity: str

    def __post_init__(self):
        names = [name for name, _ in self.frequencies]
        weights = np.asarray([value for _, value in self.frequencies], dtype=float)
        if (
            not self.training_identity
            or not names
            or names != sorted(set(names))
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1, atol=1e-12)
        ):
            raise ValueError(
                "demonstration prior requires sorted normalized frequencies and provenance"
            )

    @classmethod
    def fit(cls, rows, *, training_identity: str):
        if not rows or any(row["role"] != "train" for row in rows):
            raise ValueError("demonstration prior fitting accepts training-role rows only")
        counts = {}
        for row, weight in zip(
            rows, source_weights([row["source_id"] for row in rows]), strict=True
        ):
            counts[row["option"]] = counts.get(row["option"], 0.0) + float(weight)
        return cls(tuple(sorted(counts.items())), training_identity)

    @property
    def snapshot(self) -> str:
        return identity(
            {
                "schema_version": "demonstration_option_prior_v1",
                "frequencies": self.frequencies,
                "training_identity": self.training_identity,
                "reference_pseudo_observation": 1.0,
            }
        )

    def distribution(self, options, reference) -> OptionDistribution:
        names = tuple(options)
        base = np.asarray(reference, dtype=float)
        # Validate the actual applicability row with the production normalizer.
        conservative_option_distribution(names, base, np.zeros(len(names)))
        weights = dict(self.frequencies)
        posterior = base + np.asarray([weights.get(name, 0.0) for name in names])
        posterior /= posterior.sum()
        return conservative_option_distribution(names, base, np.log(posterior / base))
