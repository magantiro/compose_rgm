#!/usr/bin/env python3
"""Reward fine-tuning of the rewrite CTMC (Rung 2 of the conditional ladder).

Relative Trajectory Balance (RTB, Venkatraman et al. 2024) / DRAKES-style reward
fine-tuning: adjust a policy theta (initialized from frozen base B) so that it
samples proportional to the reward-tilted target

    pi(tau) ~ p_B(tau) * exp(r(x_T) / alpha) * 1[x in F],

i.e. diverse, reward-proportional trajectories over VALID molecules. Unlike the
inference-time twisted-SMC controller (Rung 1), this *retrains* the base and is
typically the stronger lever; it is what we escalate to if the learned twist
does not clear GrIDDD.

Why this uniquely fits Rewrite Generator Matching: the RTB constraint needs the
exact terminal reward on a real molecule and the exact trajectory log-prob under
the model. Both are available here because every state is a valid molecule and
`forward_mark_batch` scores the exact executed marks -- no surrogate over
corrupted diffusion states.

This module provides the correct, tested RTB objective and the differentiable
trajectory log-prob; the trajectory-collection loop (sample tau from theta's
sampler, score reward under the frozen oracle, enforce the hard fiber) plugs into
`train_step`. Trajectory sampling is CPU-heavy, so training runs when cores free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)


def trajectory_log_prob(
    model: FactorizedTraceletRateModel,
    states: tuple,
    times: tuple[float, ...],
    actions: tuple,
    rule_names: tuple[str, ...],
    jump_rates: tuple[float, ...],
) -> Tensor:
    """Exact differentiable log p_model(tau) = sum_t log P(mark_t | x_t).

    Uses the executed (state, action) marks -- the same objects the teacher path
    uses -- so the score is exact for the sampled trajectory, not an ELBO.
    """
    batch = prepare_factorized_mark_batch(states, times, actions, rule_names, jump_rates)
    prediction = model.forward_mark_batch(batch)
    return prediction.selected_mark_log_probability.sum()


def rtb_loss(
    log_p_theta: Tensor,
    log_p_base: Tensor,
    reward: Tensor,
    log_z: Tensor,
    alpha: float,
) -> Tensor:
    """Relative Trajectory Balance residual, squared.

    At the optimum, `log p_theta(tau) = log p_base(tau) + r(x_T)/alpha - log Z`,
    so theta samples the reward-tilted target. `log_z` is a single learnable
    scalar (the partition constant) shared across trajectories.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    residual = log_z + log_p_theta - log_p_base - reward / alpha
    return residual.pow(2)


@dataclass
class RTBConfig:
    alpha: float = 0.1
    learning_rate: float = 1e-4
    log_z_learning_rate: float = 1e-2
    # theta is KL-anchored to base implicitly by RTB (it targets p_B*exp(r/a));
    # keep updates small and warm-start from B to stay in-distribution.


class RewardFineTuner:
    """RTB fine-tuner: a policy theta (init from B) + a frozen base + logZ."""

    def __init__(
        self,
        policy: FactorizedTraceletRateModel,
        frozen_base: FactorizedTraceletRateModel,
        config: RTBConfig | None = None,
    ) -> None:
        self.policy = policy
        self.frozen_base = frozen_base.eval()
        for p in self.frozen_base.parameters():
            p.requires_grad_(False)
        self.config = config or RTBConfig()
        self.log_z = nn.Parameter(torch.zeros(()))
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": self.config.learning_rate},
                {"params": [self.log_z], "lr": self.config.log_z_learning_rate},
            ]
        )

    def train_step(self, trajectories: list[dict]) -> float:
        """One RTB update over a minibatch of sampled trajectories.

        Each trajectory dict has: states, times, actions, rule_names, jump_rates
        (the executed marks) and `reward` (terminal QED, already constrained to
        the hard fiber F by the sampler that produced tau). Trajectories are
        sampled from theta OFF-policy (no grad through sampling); RTB scores them
        under theta (grad) and the frozen base (no grad).
        """
        self.optimizer.zero_grad()
        losses = []
        for tau in trajectories:
            log_p_theta = trajectory_log_prob(
                self.policy,
                tau["states"], tau["times"], tau["actions"],
                tau["rule_names"], tau["jump_rates"],
            )
            with torch.no_grad():
                log_p_base = trajectory_log_prob(
                    self.frozen_base,
                    tau["states"], tau["times"], tau["actions"],
                    tau["rule_names"], tau["jump_rates"],
                )
            reward = torch.as_tensor(float(tau["reward"]))
            losses.append(rtb_loss(log_p_theta, log_p_base, reward, self.log_z, self.config.alpha))
        loss = torch.stack(losses).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss)
