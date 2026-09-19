"""Relational bond-reroute scorer for the Editing-V2 capacity repair.

The production graft executor represents one reroute by the directed triple
``(removed_neighbor, moved_root, new_target)``.  The historical score used only
the moved and target node embeddings plus the global state.  It therefore
cannot distinguish two grafts when the endpoint nodes occupy the same neural
orbits but the new target has a different relationship to the removed
neighbor.

This subclass preserves the historical score exactly and adds a zero-initialized
residual over the three pairwise relations among the moved atom, removed
neighbor, and new target.  Zero initialization makes every pre-existing logit
identical at construction, while the residual exposes the complete addressed
reroute triple to optimization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)

RELATIONAL_BOND_REROUTE_SCORER_MODE = "reroute_triple_pair_residual_v1"


class RelationalRerouteFactorizedTraceletRateModel(FactorizedTraceletRateModel):
    """Add reroute-specific relational capacity without changing legal support."""

    bond_reroute_scorer_mode = RELATIONAL_BOND_REROUTE_SCORER_MODE

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.graft_relation_head = nn.Linear(3 * self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.graft_relation_head.weight)

    @staticmethod
    def _graft_relation_features(
        batch: FactorizedMarkBatch,
        pair: Tensor,
    ) -> Tensor:
        """Return all pairwise context for each addressed reroute triple."""

        if pair.ndim != 4:
            raise ValueError("reroute pair features must have rank four")
        batch_size, n_slots, right_slots, _hidden_dim = pair.shape
        if n_slots != right_slots or tuple(batch.graft_mask.shape) != (
            batch_size,
            n_slots,
            n_slots,
        ):
            raise ValueError("reroute pair features disagree with the candidate mask")
        removed = batch.graft_remove_neighbors.to(device=pair.device)
        mask = batch.graft_mask.to(device=pair.device)
        if tuple(removed.shape) != tuple(mask.shape):
            raise ValueError("reroute removed-neighbor coordinates have the wrong shape")
        if bool((mask & ((removed < 0) | (removed >= n_slots))).any()):
            raise ValueError("a legal reroute lacks a valid removed neighbor")

        safe_removed = removed.clamp(min=0, max=n_slots - 1)
        batch_index = torch.arange(batch_size, device=pair.device)[:, None, None]
        moved_index = torch.arange(n_slots, device=pair.device)[None, :, None]
        target_index = torch.arange(n_slots, device=pair.device)[None, None, :]
        removed_target_pair = pair[batch_index, safe_removed, target_index]
        moved_removed_pair = pair[batch_index, moved_index, safe_removed]
        return torch.cat((pair, removed_target_pair, moved_removed_pair), dim=-1)

    def _action_tables(
        self,
        batch: FactorizedMarkBatch,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        *,
        require_exact_ring_support: bool = True,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        masks, logits, action_log_z = super()._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=require_exact_ring_support,
        )
        relation_features = self._graft_relation_features(batch, pair)
        relation_residual = self.graft_relation_head(relation_features).squeeze(-1)
        logits["bond_reroute"] = logits["bond_reroute"] + relation_residual
        action_log_z = action_log_z.clone()
        reroute_logits = logits["bond_reroute"].flatten(start_dim=1)
        reroute_mask = masks["bond_reroute"].flatten(start_dim=1)
        action_log_z[:, MARK_RULE_TO_INDEX["bond_reroute"]] = torch.logsumexp(
            reroute_logits.masked_fill(~reroute_mask, float("-inf")),
            dim=-1,
        )
        return masks, logits, action_log_z


__all__ = [
    "RELATIONAL_BOND_REROUTE_SCORER_MODE",
    "RelationalRerouteFactorizedTraceletRateModel",
]
