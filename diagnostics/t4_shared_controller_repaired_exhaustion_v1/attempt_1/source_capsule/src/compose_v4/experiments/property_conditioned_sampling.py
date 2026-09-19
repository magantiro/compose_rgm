"""Target-conditioned ancestral sampling for the factorized rewrite model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    SampledRewriteMark,
)


@dataclass(frozen=True)
class PropertyConditionedRewriteSampler:
    """Bind one standardized property target to a trained RGM sampler.

    The wrapper changes only learned rates over the model's existing legal
    rewrite support.  It cannot introduce an invalid action or bypass the
    executor.  ``mask=False`` denotes a missing/classifier-free condition.
    """

    model: FactorizedTraceletRateModel
    property_values: tuple[float, ...]
    property_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        dimension = int(self.model.property_condition_dim)
        if dimension <= 0:
            raise ValueError("property-conditioned sampler requires a conditioned model")
        if len(self.property_values) != dimension:
            raise ValueError("property target has the wrong dimension")
        resolved_mask = self.property_mask
        if resolved_mask is None:
            resolved_mask = (True,) * dimension
            object.__setattr__(self, "property_mask", resolved_mask)
        if len(resolved_mask) != dimension:
            raise ValueError("property mask has the wrong dimension")
        observed = np.asarray(self.property_values, dtype=np.float64)[
            np.asarray(resolved_mask, dtype=np.bool_)
        ]
        if not np.isfinite(observed).all():
            raise ValueError("observed property targets must be finite")

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        assert self.property_mask is not None
        return self.model.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            property_values=self.property_values,
            property_mask=self.property_mask,
        )


__all__ = ["PropertyConditionedRewriteSampler"]
