"""Integrated option-boundary policy and distributional SMC twist.

WHERE remains external and unchanged.  This controller chooses WHAT from the
applicable option row, while the existing option kernel executes HOW.  The
distributional value supplies only a positive intermediate twisting function;
the caller must use the exact declared terminal potential at the last boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from compose_v4.control.improvement_value import (
    MonotoneImprovementModel,
    predict_improvement,
)
from compose_v4.control.option_features import option_feature_names, structural_option_features
from compose_v4.control.option_policy import (
    AdvantageWeightedOptionActor,
    OptionDistribution,
    conservative_option_distribution,
)


@dataclass(frozen=True)
class SelectedOption:
    selected: str
    selected_index: int
    reference_probability: float
    proposal_probability: float
    distribution: OptionDistribution
    uniform_draw: float
    controller_snapshot: str


class OptionBoundaryController:
    """Frozen actor/value pair used between complete structural options."""

    def __init__(
        self,
        actor: AdvantageWeightedOptionActor,
        value: MonotoneImprovementModel,
        *,
        actor_snapshot: str,
        value_snapshot: str,
        horizons: Sequence[int],
        thresholds: Sequence[float],
        base_floor: float = 0.1,
        max_kl: float = 1.0,
        beta: float = 10.0,
    ) -> None:
        self.actor, self.value = actor.eval(), value.eval()
        self.actor_snapshot, self.value_snapshot = actor_snapshot, value_snapshot
        raw_horizons, raw_thresholds = tuple(horizons), tuple(thresholds)
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
            for item in raw_horizons
        ) or any(isinstance(item, (bool, np.bool_)) for item in raw_thresholds):
            raise ValueError("invalid option-boundary controller configuration")
        self.horizons = tuple(int(item) for item in raw_horizons)
        self.thresholds = tuple(float(item) for item in raw_thresholds)
        self.base_floor, self.max_kl, self.beta = float(base_floor), float(max_kl), float(beta)
        if (
            not actor_snapshot
            or not value_snapshot
            or len(self.horizons) != value.n_horizons
            or len(self.thresholds) != value.n_thresholds
            or actor.state_dim != value.input_dim
            or actor.option_dim != len(option_feature_names())
            or not self.horizons
            or not self.thresholds
            or any(value < 0 for value in self.horizons)
            or any(value < 0 or not math.isfinite(value) for value in self.thresholds)
            or any(b <= a for a, b in zip(self.horizons, self.horizons[1:], strict=False))
            or any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:], strict=False))
            or not 0 < self.base_floor <= 1
            or not math.isfinite(self.base_floor)
            or self.max_kl < 0
            or not math.isfinite(self.max_kl)
            or self.beta <= 0
            or not math.isfinite(self.beta)
        ):
            raise ValueError("invalid option-boundary controller configuration")
        self.snapshot = f"{actor_snapshot}:{value_snapshot}"

    def decide(self, options, reference, state_features, rng) -> SelectedOption:
        names = tuple(options)
        option_features = np.stack([structural_option_features(name) for name in names])
        state = np.asarray(state_features, dtype=np.float32)
        with torch.inference_mode():
            scores = self.actor(torch.from_numpy(state), torch.from_numpy(option_features)).numpy()
        distribution = conservative_option_distribution(
            names,
            reference,
            scores,
            base_floor=self.base_floor,
            max_kl=self.max_kl,
        )
        uniform = float(rng.random())
        index = min(
            int(np.searchsorted(np.cumsum(distribution.proposal), uniform, side="right")),
            len(names) - 1,
        )
        return SelectedOption(
            names[index],
            index,
            distribution.reference[index],
            distribution.proposal[index],
            distribution,
            uniform,
            self.snapshot,
        )

    def log_potential(
        self,
        state_features,
        *,
        remaining_options: int,
        incumbent: float,
        terminal_best: float | None = None,
    ) -> float:
        """Return an approximate intermediate or exact terminal log potential."""

        if terminal_best is not None:
            if remaining_options != 0 or not math.isfinite(terminal_best):
                raise ValueError("terminal potential requires budget zero and a finite best score")
            return self.beta * float(terminal_best)
        if remaining_options not in self.horizons or not math.isfinite(incumbent):
            raise ValueError("intermediate potential requires a registered horizon and incumbent")
        prediction = predict_improvement(
            self.value, np.asarray(state_features, dtype=np.float32)[None, :]
        )[0, self.horizons.index(remaining_options)]
        widths = np.diff(np.asarray((0.0, *self.thresholds), dtype=float))
        # Right-Riemann survival integral, truncated at the largest registered
        # improvement threshold. It is an approximate twist, not exact h.
        expected_improvement = float(widths @ prediction)
        return self.beta * (float(incumbent) + expected_improvement)
