"""Draw from a frozen reference restricted to declared native edit families.

This is a task controller. It leaves the hash-bound reference model and its
unconditioned sampling method unchanged. The restriction is applied to the
complete normalized native mark law before a mark is drawn.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from compose_v4.benchmark.uniform_native_mark_law import UniformNativeMarkLaw
from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    SampledRewriteMark,
)


@dataclass(frozen=True)
class _RestrictedModelView:
    """Supply a per-call family mask to the unchanged model sampler."""

    model: FactorizedTraceletRateModel
    disabled_sampling_rule_names: frozenset[str]

    def __getattr__(self, name: str):
        return getattr(self.model, name)


def sample_reference_mark_in_families(
    model: FactorizedTraceletRateModel | UniformNativeMarkLaw,
    state: MolecularGraph,
    time: float,
    rng: np.random.Generator,
    *,
    allowed_rule_names: Collection[str],
) -> SampledRewriteMark:
    """Sample the model's native mark law conditional on a nonempty family set.

    An empty restricted fiber returns a terminal mark without consuming random
    state. The model object and its frozen source bytes are never modified.
    """

    allowed = frozenset(allowed_rule_names)
    unknown = allowed.difference(MARK_RULE_TO_INDEX)
    if unknown:
        raise ValueError(f"unknown allowed rule names: {sorted(unknown)}")
    if not allowed:
        return SampledRewriteMark(0.0, "<TERMINAL>", None)
    if isinstance(model, UniformNativeMarkLaw):
        return model.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            property_values=None,
            property_mask=None,
            allowed_rule_names=allowed,
        )
    disabled = frozenset(getattr(model, "disabled_sampling_rule_names", ())) | (
        frozenset(MARK_RULE_TO_INDEX) - allowed
    )
    view = _RestrictedModelView(model, disabled)
    return FactorizedTraceletRateModel.sample_rewrite_mark_conditioned(
        view,
        state,
        time,
        rng,
        property_values=None,
        property_mask=None,
    )
