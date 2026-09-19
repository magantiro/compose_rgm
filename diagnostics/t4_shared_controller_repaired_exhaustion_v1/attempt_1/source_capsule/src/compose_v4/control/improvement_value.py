"""Policy-congruent distributional value at completed-option boundaries.

This module predicts a survival function for the best utility improvement seen
within a remaining *option* budget.  It is deliberately not an endpoint score
model and not an exact Doob value.  Training rows identify the behavior policy
that produced their continuations, and right-censored suffixes never become
invented failures.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import nn

from compose_v4.control.docking_value import identity


@dataclass(frozen=True)
class ImprovementTrace:
    """One observed continuation from an option-boundary state.

    ``scores[0]`` is the utility at the boundary.  Later entries are utilities
    after completed options.  ``terminated`` means no continuation exists after
    the final entry; otherwise the trace is right-censored there.
    """

    source_id: str
    trace_id: str
    behavior_policy_id: str
    features: tuple[float, ...]
    scores: tuple[float, ...]
    incumbent: float
    terminated: bool
    importance_weight: float | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.scores, dtype=float)
        features = np.asarray(self.features, dtype=float)
        if not self.source_id or not self.trace_id or not self.behavior_policy_id:
            raise ValueError("trace, source and behavior-policy identities are required")
        if features.ndim != 1 or not features.size or not np.isfinite(features).all():
            raise ValueError("trace features must be a finite nonempty vector")
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("trace scores must be a finite nonempty sequence")
        if not np.isfinite(self.incumbent):
            raise ValueError("trace incumbent must be finite")
        if self.importance_weight is not None and (
            not math.isfinite(self.importance_weight) or self.importance_weight <= 0
        ):
            raise ValueError("an explicit importance weight must be finite and positive")


@dataclass(frozen=True)
class ImprovementTargets:
    """Dense target grid plus a mask distinguishing known from censored cells."""

    features: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    row_weights: np.ndarray
    source_ids: tuple[str, ...]
    trace_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    thresholds: tuple[float, ...]
    behavior_policy_id: str

    def subset(self, indices: Sequence[int]) -> ImprovementTargets:
        idx = np.asarray(indices, dtype=int)
        return ImprovementTargets(
            self.features[idx],
            self.labels[idx],
            self.mask[idx],
            self.row_weights[idx],
            tuple(self.source_ids[i] for i in idx),
            tuple(self.trace_ids[i] for i in idx),
            self.horizons,
            self.thresholds,
            self.behavior_policy_id,
        )


def _grid(horizons: Sequence[int], thresholds: Sequence[float]):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in horizons):
        raise ValueError("horizons must be strictly increasing nonnegative integers")
    h = tuple(int(v) for v in horizons)
    if any(isinstance(v, (bool, np.bool_)) for v in thresholds):
        raise ValueError("thresholds must be finite, nonnegative and strictly increasing")
    d = tuple(float(v) for v in thresholds)
    if not h or any(v < 0 for v in h) or any(b <= a for a, b in pairwise(h)):
        raise ValueError("horizons must be strictly increasing nonnegative integers")
    if not d or not np.isfinite(d).all() or d[0] < 0 or any(b <= a for a, b in pairwise(d)):
        raise ValueError("thresholds must be finite, nonnegative and strictly increasing")
    return h, d


def censored_targets(
    scores: Sequence[float],
    incumbent: float,
    horizons: Sequence[int],
    thresholds: Sequence[float],
    *,
    terminated: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct logically identified exceedance labels under right censoring.

    At an unobserved horizon, an already achieved threshold is a known success.
    An unachieved threshold is unknown, rather than a fabricated failure, unless
    the continuation is known to have terminated.
    """

    h, d = _grid(horizons, thresholds)
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("scores must be a finite nonempty sequence")
    if not math.isfinite(incumbent):
        raise ValueError("incumbent must be finite")
    available = len(values) - 1
    labels = np.zeros((len(h), len(d)), dtype=np.float32)
    mask = np.zeros_like(labels, dtype=bool)
    for i, horizon in enumerate(h):
        observed = min(horizon, available)
        improvement = float(np.max(values[: observed + 1]) - incumbent)
        achieved = np.asarray([improvement >= threshold for threshold in d], dtype=bool)
        labels[i] = achieved
        fully_observed = horizon <= available or terminated
        mask[i] = True if fully_observed else achieved
    return labels, mask


def build_targets(
    traces: Sequence[ImprovementTrace],
    horizons: Sequence[int],
    thresholds: Sequence[float],
    *,
    behavior_policy_id: str,
) -> ImprovementTargets:
    """Build targets for one declared policy, with explicit off-policy weights."""

    h, d = _grid(horizons, thresholds)
    rows = tuple(traces)
    if not rows or not behavior_policy_id:
        raise ValueError("nonempty traces and a behavior-policy identity are required")
    width = len(rows[0].features)
    labels, masks, weights = [], [], []
    for row in rows:
        if len(row.features) != width:
            raise ValueError("trace feature widths differ")
        if row.behavior_policy_id == behavior_policy_id:
            weight = 1.0 if row.importance_weight is None else row.importance_weight
        elif row.importance_weight is None:
            raise ValueError(
                "mixed behavior-policy snapshots require an explicit importance weight"
            )
        else:
            weight = row.importance_weight
        label, mask = censored_targets(row.scores, row.incumbent, h, d, terminated=row.terminated)
        labels.append(label)
        masks.append(mask)
        weights.append(float(weight))
    return ImprovementTargets(
        np.asarray([row.features for row in rows], dtype=np.float32),
        np.stack(labels),
        np.stack(masks),
        np.asarray(weights, dtype=np.float32),
        tuple(row.source_id for row in rows),
        tuple(row.trace_id for row in rows),
        h,
        d,
        behavior_policy_id,
    )


def source_group_split(
    source_ids: Sequence[str], *, calibration_fraction: float = 0.2, seed: int = 20260912
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split source groups before feature construction or fit."""

    sources = tuple(str(value) for value in source_ids)
    groups = sorted(set(sources))
    if len(groups) < 2 or not 0 < calibration_fraction < 1:
        raise ValueError("a grouped split requires at least two sources and a fraction in (0, 1)")
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).digest(),
    )
    n_calibration = min(len(groups) - 1, max(1, round(len(groups) * calibration_fraction)))
    calibration_groups = set(ranked[:n_calibration])
    train = np.asarray([i for i, value in enumerate(sources) if value not in calibration_groups])
    calibration = np.asarray([i for i, value in enumerate(sources) if value in calibration_groups])
    if not len(train) or not len(calibration):
        raise RuntimeError("grouped split produced an empty role")
    return train, calibration


class MonotoneImprovementModel(nn.Module):
    """A monotone survival surface over a fixed horizon/threshold grid."""

    def __init__(self, input_dim: int, n_horizons: int, n_thresholds: int, hidden: int = 64):
        super().__init__()
        if min(input_dim, n_horizons, n_thresholds, hidden) < 1:
            raise ValueError("model dimensions must be positive")
        self.input_dim = int(input_dim)
        self.n_horizons = int(n_horizons)
        self.n_thresholds = int(n_thresholds)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.origin = nn.Linear(hidden, 1)
        self.horizon_increments = nn.Linear(hidden, n_horizons)
        self.threshold_increments = nn.Linear(hidden, n_thresholds)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("improvement features have the wrong rank or width")
        encoded = self.encoder(features)
        horizon = torch.cumsum(torch.nn.functional.softplus(self.horizon_increments(encoded)), 1)
        threshold = torch.cumsum(
            torch.nn.functional.softplus(self.threshold_increments(encoded)), 1
        )
        return self.origin(encoded)[:, :, None] + horizon[:, :, None] - threshold[:, None, :]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


@dataclass(frozen=True)
class ImprovementFitConfig:
    hidden: int = 64
    updates: int = 400
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 20260912


def improvement_parameter_id(model: MonotoneImprovementModel) -> str:
    parameters = {
        name: value.detach().cpu().tolist() for name, value in sorted(model.state_dict().items())
    }
    return identity(
        {
            "schema_version": "monotone_improvement_parameters_v1",
            "input_dim": model.input_dim,
            "n_horizons": model.n_horizons,
            "n_thresholds": model.n_thresholds,
            "parameters": parameters,
        }
    )


def fit_improvement_model(
    targets: ImprovementTargets, config: ImprovementFitConfig | None = None
) -> tuple[MonotoneImprovementModel, dict]:
    """Fit on known cells only; censored cells carry exactly zero loss weight."""

    config = ImprovementFitConfig() if config is None else config
    if config.updates < 1 or config.hidden < 1 or config.learning_rate <= 0:
        raise ValueError("invalid improvement-model fit configuration")
    if not targets.mask.any():
        raise ValueError("no identified target cells are available")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    model = MonotoneImprovementModel(
        targets.features.shape[1], len(targets.horizons), len(targets.thresholds), config.hidden
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x = torch.from_numpy(targets.features)
    y = torch.from_numpy(targets.labels)
    known = torch.from_numpy(targets.mask)
    row_weight = torch.from_numpy(targets.row_weights)[:, None, None]
    cell_weight = known * row_weight
    history = []
    for update in range(config.updates):
        loss_cells = torch.nn.functional.binary_cross_entropy_with_logits(
            model.logits(x), y, reduction="none"
        )
        loss = (loss_cells * cell_weight).sum() / cell_weight.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite improvement-model loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update in (0, config.updates - 1):
            history.append({"update": update + 1, "loss": float(loss.detach())})
    model.eval()
    parameters = {
        name: value.detach().cpu().tolist() for name, value in sorted(model.state_dict().items())
    }
    body = {
        "schema_version": "option_improvement_fit_v1",
        "behavior_policy_id": targets.behavior_policy_id,
        "horizons": list(targets.horizons),
        "thresholds": list(targets.thresholds),
        "rows": len(targets.features),
        "identified_cells": int(targets.mask.sum()),
        "config": config.__dict__,
        "parameters": parameters,
        "history": history,
    }
    return model, {
        **body,
        "parameter_id": improvement_parameter_id(model),
        "model_id": identity(body),
    }


def predict_improvement(model: MonotoneImprovementModel, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(values)).numpy()
    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite improvement prediction")
    return prediction


def _auc(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    positive, negative = predictions[labels == 1], predictions[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def evaluate_improvement(
    targets: ImprovementTargets, prediction: np.ndarray, *, decision_threshold: float = 0.5
) -> dict:
    """Report calibration, ranking, precision and coverage on identified cells."""

    p = np.asarray(prediction, dtype=float)
    if p.shape != targets.labels.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("predictions must align with targets and lie in [0, 1]")
    known = targets.mask
    labels, scores = targets.labels[known], p[known]
    weights = np.broadcast_to(targets.row_weights[:, None, None], p.shape)[known]
    selected = scores >= decision_threshold
    return {
        "identified_cells": int(known.sum()),
        "total_cells": int(known.size),
        "target_coverage": float(known.mean()),
        "positive_rate": float(np.average(labels, weights=weights)),
        "brier": float(np.average((scores - labels) ** 2, weights=weights)),
        "ranking_auc": _auc(labels, scores),
        "decision_coverage": float(np.mean(selected)),
        "improvement_precision": (
            float(np.average(labels[selected], weights=weights[selected]))
            if selected.any()
            else None
        ),
    }
