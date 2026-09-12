"""Conservative learned policy over applicable structural options.

The actor changes only the WHAT decision.  The applicability-aware reference
distribution remains the target law for importance accounting, every option in
its support retains positive proposal probability, and ``generic`` is required
at every decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from compose_v4.control.continuation import _tilted
from compose_v4.control.docking_value import identity
from compose_v4.control.option_selector import GENERIC_OPTION


@dataclass(frozen=True)
class OptionDistribution:
    options: tuple[str, ...]
    reference: tuple[float, ...]
    proposal: tuple[float, ...]
    scores: tuple[float, ...]
    eta: float
    kl: float
    base_floor: float


def conservative_option_distribution(
    options: Sequence[str],
    reference: Sequence[float],
    scores: Sequence[float],
    *,
    base_floor: float = 0.1,
    max_kl: float = 1.0,
) -> OptionDistribution:
    """Exponentially tilt an option row under a KL budget and reference floor."""

    names = tuple(options)
    base = np.asarray(reference, dtype=float)
    values = np.asarray(scores, dtype=float)
    if not names or len(names) != len(set(names)) or GENERIC_OPTION not in names:
        raise ValueError("applicable options must be unique, nonempty and contain generic")
    if (
        base.shape != (len(names),)
        or not np.isfinite(base).all()
        or np.any(base <= 0)
        or not np.isclose(base.sum(), 1, atol=1e-12)
    ):
        raise ValueError("option reference must be aligned, positive and normalized")
    if values.shape != base.shape or not np.isfinite(values).all():
        raise ValueError("option scores must be aligned and finite")
    if not math.isfinite(base_floor) or not 0 < base_floor <= 1:
        raise ValueError("base_floor must lie in (0, 1]")
    if not math.isfinite(max_kl) or max_kl < 0:
        raise ValueError("max_kl must be finite and nonnegative")
    desirability = np.exp(np.clip(values - values.max(), -80, 0))
    proposal, eta, kl = _tilted(
        base, desirability, kappa=float(max_kl), exploration=float(base_floor)
    )
    if (
        not np.isclose(proposal.sum(), 1, atol=1e-12)
        or np.any(proposal < base_floor * base - 1e-12)
        or kl > max_kl + 1e-10
    ):
        raise RuntimeError("conservative option-policy invariant violated")
    return OptionDistribution(
        names,
        tuple(map(float, base)),
        tuple(map(float, proposal)),
        tuple(map(float, values)),
        float(eta),
        float(kl),
        float(base_floor),
    )


@dataclass(frozen=True)
class OptionActorExample:
    """One observed option choice from a declared behavior-policy snapshot."""

    source_id: str
    decision_id: str
    behavior_policy_id: str
    state_features: tuple[float, ...]
    options: tuple[str, ...]
    option_features: tuple[tuple[float, ...], ...]
    reference: tuple[float, ...]
    behavior: tuple[float, ...]
    selected: int
    advantage: float
    importance_weight: float | None = None

    def __post_init__(self) -> None:
        n = len(self.options)
        state = np.asarray(self.state_features, dtype=float)
        option = np.asarray(self.option_features, dtype=float)
        reference = np.asarray(self.reference, dtype=float)
        behavior = np.asarray(self.behavior, dtype=float)
        if not self.source_id or not self.decision_id or not self.behavior_policy_id:
            raise ValueError("actor example identities are required")
        if n < 1 or len(set(self.options)) != n or GENERIC_OPTION not in self.options:
            raise ValueError("actor options must be unique and include generic")
        if state.ndim != 1 or not state.size or not np.isfinite(state).all():
            raise ValueError("actor state features must be a finite vector")
        if option.ndim != 2 or option.shape[0] != n or not np.isfinite(option).all():
            raise ValueError("actor option features must be a finite aligned matrix")
        for name, values in (("reference", reference), ("behavior", behavior)):
            if (
                values.shape != (n,)
                or not np.isfinite(values).all()
                or np.any(values <= 0)
                or not np.isclose(values.sum(), 1, atol=1e-10)
            ):
                raise ValueError(f"actor {name} probabilities must be positive and normalized")
        if not 0 <= self.selected < n or not math.isfinite(self.advantage):
            raise ValueError("actor selected index or advantage is invalid")
        if self.importance_weight is not None and (
            not math.isfinite(self.importance_weight) or self.importance_weight <= 0
        ):
            raise ValueError("actor importance weight must be finite and positive")


class AdvantageWeightedOptionActor(nn.Module):
    """Scores options from shared state features and compositional option features."""

    def __init__(self, state_dim: int, option_dim: int, hidden: int = 64):
        super().__init__()
        if min(state_dim, option_dim, hidden) < 1:
            raise ValueError("actor dimensions must be positive")
        self.state_dim, self.option_dim = int(state_dim), int(option_dim)
        self.network = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
        if state.ndim != 1 or state.shape[0] != self.state_dim:
            raise ValueError("actor state tensor has the wrong shape")
        if option.ndim != 2 or option.shape[1] != self.option_dim:
            raise ValueError("actor option tensor has the wrong shape")
        tiled = state[None, :].expand(len(option), -1)
        return self.network(torch.cat((tiled, option), dim=1)).squeeze(-1)


@dataclass(frozen=True)
class OptionActorFitConfig:
    hidden: int = 64
    updates: int = 300
    learning_rate: float = 3e-3
    temperature: float = 0.25
    max_weight: float = 20.0
    base_floor: float = 0.1
    kl_penalty: float = 0.02
    seed: int = 20260912


def option_actor_parameter_id(actor: AdvantageWeightedOptionActor) -> str:
    parameters = {
        name: value.detach().cpu().tolist() for name, value in sorted(actor.state_dict().items())
    }
    return identity(
        {
            "schema_version": "option_actor_parameters_v1",
            "state_dim": actor.state_dim,
            "option_dim": actor.option_dim,
            "parameters": parameters,
        }
    )


def _training_distribution(logits, reference, floor):
    controlled = torch.softmax(torch.log(reference) + logits, dim=0)
    return floor * reference + (1 - floor) * controlled


def fit_option_actor(
    examples: Sequence[OptionActorExample],
    *,
    behavior_policy_id: str,
    config: OptionActorFitConfig | None = None,
) -> tuple[AdvantageWeightedOptionActor, dict]:
    """Fit an advantage-weighted actor without relabeling off-policy rows."""

    rows = tuple(examples)
    config = OptionActorFitConfig() if config is None else config
    if not rows or not behavior_policy_id:
        raise ValueError("actor fitting requires examples and a behavior-policy identity")
    if (
        config.updates < 1
        or config.hidden < 1
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
        or config.temperature <= 0
        or not math.isfinite(config.temperature)
        or not math.isfinite(config.max_weight)
        or config.max_weight < 1
        or not math.isfinite(config.base_floor)
        or not 0 < config.base_floor <= 1
        or not math.isfinite(config.kl_penalty)
        or config.kl_penalty < 0
    ):
        raise ValueError("invalid actor fit configuration")
    state_dim, option_dim = len(rows[0].state_features), len(rows[0].option_features[0])
    weights = []
    for row in rows:
        if len(row.state_features) != state_dim or any(
            len(features) != option_dim for features in row.option_features
        ):
            raise ValueError("actor feature widths differ")
        if row.behavior_policy_id == behavior_policy_id:
            correction = 1.0 if row.importance_weight is None else row.importance_weight
        elif row.importance_weight is None:
            raise ValueError(
                "mixed behavior-policy snapshots require an explicit importance weight"
            )
        else:
            correction = row.importance_weight
        # Clip before exponentiation so extreme but finite advantages cannot
        # overflow. Apply the explicit off-policy correction and then cap the
        # final learning weight at the declared maximum.
        log_cap = math.log(config.max_weight)
        advantage_weight = math.exp(min(row.advantage / config.temperature, log_cap))
        weights.append(float(min(correction * advantage_weight, config.max_weight)))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    actor = AdvantageWeightedOptionActor(state_dim, option_dim, config.hidden)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    history = []
    for update in range(config.updates):
        losses = []
        for row, weight in zip(rows, weights, strict=True):
            state = torch.tensor(row.state_features, dtype=torch.float32)
            option = torch.tensor(row.option_features, dtype=torch.float32)
            reference = torch.tensor(row.reference, dtype=torch.float32)
            q = _training_distribution(actor(state, option), reference, config.base_floor)
            kl = torch.sum(q * (torch.log(q) - torch.log(reference)))
            losses.append(-weight * torch.log(q[row.selected]) + config.kl_penalty * kl)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite option-actor loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update in (0, config.updates - 1):
            history.append({"update": update + 1, "loss": float(loss.detach())})
    actor.eval()
    parameters = {
        name: value.detach().cpu().tolist() for name, value in sorted(actor.state_dict().items())
    }
    body = {
        "schema_version": "advantage_weighted_option_actor_v1",
        "behavior_policy_id": behavior_policy_id,
        "state_dim": state_dim,
        "option_dim": option_dim,
        "config": config.__dict__,
        "parameters": parameters,
        "decisions": [row.decision_id for row in rows],
        "history": history,
    }
    return actor, {
        **body,
        "parameter_id": option_actor_parameter_id(actor),
        "policy_id": identity(body),
    }


def actor_distribution(
    actor: AdvantageWeightedOptionActor,
    example: OptionActorExample,
    *,
    base_floor: float = 0.1,
    max_kl: float = 1.0,
) -> OptionDistribution:
    with torch.inference_mode():
        scores = actor(
            torch.tensor(example.state_features, dtype=torch.float32),
            torch.tensor(example.option_features, dtype=torch.float32),
        ).numpy()
    return conservative_option_distribution(
        example.options,
        example.reference,
        scores,
        base_floor=base_floor,
        max_kl=max_kl,
    )
