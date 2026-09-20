"""State-dependent Generator-Matching tilt of a stochastic rewrite prior."""

from __future__ import annotations

import torch
from torch import nn

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.corpus_marginal import CorpusMarginalRateModel
from compose_v4.model.rate_model import FiberRatePrediction, WholeGraphRateModel
from compose_v4.rewrite.factorized_fiber import (
    FactorizedFiber,
    enumerate_factorized_cnof_fiber,
)
from compose_v4.rewrite.fiber import group_transition_indices_by_successor


class PriorTiltedRewriteRateModel(WholeGraphRateModel):
    """Learn a normalized state-dependent residual over a rewrite CTMC prior.

    At initialization this model is exactly the supplied corpus-marginal
    generator. Neural residuals exponentially tilt its legal marked-action
    distribution and total hazard. Because the prior hazard is zero at t=1,
    endpoint shutoff remains exact for every parameter value.
    """

    def __init__(
        self,
        prior: CorpusMarginalRateModel,
        *,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        use_rewrite_context: bool = False,
        use_topology_context: bool = False,
        max_log_hazard_tilt: float = 3.0,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_passing_steps,
            use_rewrite_context=use_rewrite_context,
            use_topology_context=use_topology_context,
        )
        if max_log_hazard_tilt <= 0.0:
            raise ValueError("max_log_hazard_tilt must be positive")
        self.prior = prior
        self.max_log_hazard_tilt = float(max_log_hazard_tilt)
        for parameter in self.rate_head.parameters():
            parameter.requires_grad_(False)
        self.log_rate_bias.requires_grad_(False)

        self.action_residual_head = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hazard_residual_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.action_residual_head[-1].weight)
        nn.init.zeros_(self.action_residual_head[-1].bias)
        nn.init.zeros_(self.hazard_residual_head[-1].weight)
        nn.init.zeros_(self.hazard_residual_head[-1].bias)

    def predict_factorized_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: FactorizedFiber | None = None,
    ) -> FiberRatePrediction:
        legal_fiber = fiber or enumerate_factorized_cnof_fiber(state)
        transitions = legal_fiber.transitions
        if not transitions:
            return FiberRatePrediction(
                transitions=(),
                marked_rates=torch.empty(0, device=self.device),
                grouped_indices={},
            )

        prior_prediction = self.prior.predict_factorized_fiber(
            state,
            time,
            fiber=legal_fiber,
        )
        prior_rates = prior_prediction.marked_rates.to(self.device)
        prior_total = prior_rates.sum()
        if float(prior_total) <= 0.0:
            return FiberRatePrediction(
                transitions=transitions,
                marked_rates=torch.zeros(len(transitions), device=self.device),
                grouped_indices=group_transition_indices_by_successor(transitions),
            )

        node_states, global_state, time_state, bonds = self._encode_graph(state, time)
        topology = self._topology_features(state)
        action_rows = torch.stack(
            [
                self._encode_action(
                    transition,
                    state,
                    node_states,
                    global_state,
                    time_state,
                    bonds,
                    topology,
                )
                for transition in transitions
            ]
        )
        action_tilt = self.action_residual_head(action_rows).squeeze(-1)
        support = prior_rates > 0.0
        tilted_logits = torch.log(prior_rates.clamp_min(1e-30)) + action_tilt
        tilted_logits = tilted_logits.masked_fill(~support, float("-inf"))
        probabilities = torch.softmax(tilted_logits, dim=0)
        prior_probabilities = prior_rates / prior_total
        action_kl = (
            probabilities[support]
            * (
                torch.log(probabilities[support].clamp_min(1e-30))
                - torch.log(prior_probabilities[support].clamp_min(1e-30))
            )
        ).sum()

        context = torch.cat((global_state, time_state), dim=-1)
        raw_hazard_tilt = self.hazard_residual_head(context).squeeze(-1)
        log_hazard_tilt = self.max_log_hazard_tilt * torch.tanh(raw_hazard_tilt)
        total_hazard = prior_total * torch.exp(log_hazard_tilt)
        rates = total_hazard * probabilities
        return FiberRatePrediction(
            transitions=transitions,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(transitions),
            action_kl_to_prior=action_kl,
            squared_log_hazard_tilt=log_hazard_tilt.square(),
        )
