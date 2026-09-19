"""State-dependent Generator-Matching residual over an empirical tracelet prior."""

from __future__ import annotations

import torch
from torch import nn

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.tracelet_corpus_marginal import (
    TraceletCorpusMarginalRateModel,
)
from compose_v4.model.rate_model import FiberRatePrediction
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.fiber import group_transition_indices_by_successor
from compose_v4.rewrite.tracelet_fiber import (
    TraceletFiber,
    enumerate_tracelet_cnof_fiber,
)


class PriorTiltedTraceletRateModel(TraceletRateModel):
    """Learn contextual action/hazard residuals from an explicit data baseline."""

    def __init__(
        self,
        prior: TraceletCorpusMarginalRateModel,
        *,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        use_rewrite_context: bool = True,
        use_topology_context: bool = True,
        use_aromatic_bond_view: bool = False,
        max_log_hazard_tilt: float = 3.0,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_passing_steps,
            use_rewrite_context=use_rewrite_context,
            use_topology_context=use_topology_context,
            use_aromatic_bond_view=use_aromatic_bond_view,
        )
        if max_log_hazard_tilt <= 0.0:
            raise ValueError("max_log_hazard_tilt must be positive")
        self.prior = prior
        self.max_log_hazard_tilt = float(max_log_hazard_tilt)
        for module in (self.rate_head, self.total_hazard_head, self.family_head):
            for parameter in module.parameters():
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

    def predict_tracelet_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: TraceletFiber | None = None,
    ) -> FiberRatePrediction:
        legal_fiber = fiber or enumerate_tracelet_cnof_fiber(state)
        transitions = legal_fiber.transitions
        if not transitions:
            return FiberRatePrediction((), torch.empty(0, device=self.device), {})

        prior_prediction = self.prior.predict_tracelet_fiber(
            state,
            time,
            fiber=legal_fiber,
        )
        prior_rates = prior_prediction.marked_rates.to(self.device)
        prior_total = prior_rates.sum()
        if float(prior_total) <= 0.0:
            return FiberRatePrediction(
                transitions,
                torch.zeros(len(transitions), device=self.device),
                group_transition_indices_by_successor(transitions),
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
