"""Canonical-group contextual scorer for ring-system restatement.

The base semantic scorer represents a canonical ring-restatement successor by
its resonance-invariant bond-class changes.  That representation is sufficient
for ordinary aromatization and dearomatization, but two distinct executable
successors can share the same bond-class descriptor when the restatement acts
at different heteroatom or ring sites.  No optimizer can separate candidates
that enter the head as the same vector.

This score-only subclass adds a zero-initialized residual over the complete
action signature and its molecular context.  It encodes every changed edge and
target order, pools node features at all changed-bond endpoints, and pools pair
features from those endpoints to every real atom.  Raw aliases are first
deduplicated into action variants and then pooled within the canonical-successor
group.  The residual is therefore constant within each successor fiber and is
equivariant to persistent-slot relabeling.  It changes neither legal support nor
the production executor, canonicalization, or Active8 admission evidence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from compose_v4.chem.molecular_graph import BOND_CLASSES, NULL_IDX, SCAR_IDX
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedMarkBatch,
)
from compose_v4.model.relational_reroute_rate_model import (
    RelationalRerouteFactorizedTraceletRateModel,
)

CONTEXTUAL_RING_RESTATE_SCORER_MODE = (
    "canonical_group_action_site_context_residual_v1"
)


class ContextualRingRestateFactorizedTraceletRateModel(
    RelationalRerouteFactorizedTraceletRateModel
):
    """Add successor-group site context without changing executable support."""

    ring_restate_context_scorer_mode = CONTEXTUAL_RING_RESTATE_SCORER_MODE

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.ring_restate_scorer_mode != SEMANTIC_RING_RESTATE_SCORER_MODE:
            raise ValueError(
                "contextual ring-restatement scoring requires semantic successor groups"
            )
        self.ring_restate_context_head = nn.Linear(
            2 * self.hidden_dim,
            1,
            bias=False,
        )
        nn.init.zeros_(self.ring_restate_context_head.weight)

    def _ring_restate_context_features(
        self,
        batch: FactorizedMarkBatch,
        node: Tensor,
        pair: Tensor,
    ) -> Tensor:
        """Return one canonical-group site representation per raw action alias."""

        if node.ndim != 3 or pair.ndim != 4:
            raise ValueError("ring-restatement context tensors have invalid rank")
        batch_size, n_slots, hidden_dim = node.shape
        if tuple(pair.shape) != (batch_size, n_slots, n_slots, hidden_dim):
            raise ValueError("ring-restatement node and pair tensors disagree")
        group_ids = batch.ring_restate_successor_group_ids
        multiplicities = batch.ring_restate_successor_group_multiplicities
        if group_ids is None or multiplicities is None:
            raise ValueError(
                "contextual ring-restatement scoring lacks canonical group metadata"
            )
        if len(group_ids) != batch_size or len(multiplicities) != batch_size:
            raise ValueError(
                "ring-restatement group metadata does not align with the batch"
            )

        width = max(
            max((len(actions) for actions in batch.ring_restate_actions), default=0),
            1,
        )
        features = pair.new_zeros((batch_size, width, 2 * hidden_dim))
        for batch_index, actions in enumerate(batch.ring_restate_actions):
            row_group_ids = group_ids[batch_index]
            row_multiplicities = multiplicities[batch_index]
            if len(actions) != len(row_group_ids):
                raise ValueError(
                    "ring-restatement actions and successor-group IDs differ"
                )
            if tuple(
                row_group_ids.count(group_index)
                for group_index in range(len(row_multiplicities))
            ) != tuple(int(value) for value in row_multiplicities):
                raise ValueError(
                    "ring-restatement group multiplicities disagree with actions"
                )
            real_slots = torch.nonzero(
                (batch.atom_types[batch_index] != NULL_IDX)
                & (batch.atom_types[batch_index] != SCAR_IDX),
                as_tuple=False,
            ).flatten()
            if not actions:
                continue
            if not len(real_slots):
                raise ValueError("a ring-restatement action has no real source atoms")

            action_variants_by_group: list[
                set[tuple[tuple[int, int, int], ...]]
            ] = [
                set() for _ in row_multiplicities
            ]
            for action, group_index in zip(actions, row_group_ids):
                index = int(group_index)
                if not 0 <= index < len(row_multiplicities):
                    raise ValueError("ring-restatement group ID is out of range")
                signature = tuple(
                    sorted(
                        {
                            (
                                min(int(change.a), int(change.b)),
                                max(int(change.a), int(change.b)),
                                int(change.new_order),
                            )
                            for change in action.changes
                        }
                    )
                )
                if not signature or any(
                    not 0 <= slot < n_slots
                    for left, right, _ in signature
                    for slot in (left, right)
                ):
                    raise ValueError(
                        "ring-restatement action has an invalid changed-bond signature"
                    )
                action_variants_by_group[index].add(signature)

            group_features: list[Tensor] = []
            for variants in action_variants_by_group:
                if not variants:
                    raise ValueError(
                        "ring-restatement canonical group has no action variant"
                    )
                variant_features = []
                for signature in sorted(variants):
                    sites = tuple(
                        sorted(
                            {
                                slot
                                for left, right, _ in signature
                                for slot in (left, right)
                            }
                        )
                    )
                    site_index = torch.tensor(
                        sites,
                        dtype=torch.long,
                        device=pair.device,
                    )
                    node_context = node[batch_index].index_select(0, site_index).mean(
                        dim=0
                    )
                    pair_context = (
                        pair[batch_index]
                        .index_select(0, site_index)
                        .index_select(1, real_slots)
                        .mean(dim=(0, 1))
                    )
                    transition_pieces = []
                    for left, right, target_order in signature:
                        source_order = int(batch.bonds[batch_index, left, right])
                        if not (
                            0 <= source_order < BOND_CLASSES
                            and 0 <= target_order < BOND_CLASSES
                        ):
                            raise ValueError(
                                "ring-restatement target order is outside the fixed encoding"
                            )
                        transition_pieces.append(
                            pair[batch_index, left, right]
                            + self.restate_transition_embedding.weight[
                                source_order * BOND_CLASSES + target_order
                            ]
                        )
                    local_context = node_context + torch.stack(
                        transition_pieces
                    ).mean(dim=0)
                    variant_features.append(
                        torch.cat((local_context, pair_context), dim=-1)
                    )
                group_features.append(torch.stack(variant_features).mean(dim=0))
            for action_index, group_index in enumerate(row_group_ids):
                features[batch_index, action_index] = group_features[
                    int(group_index)
                ]
        return features

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
        context = self._ring_restate_context_features(batch, node, pair)
        residual = self.ring_restate_context_head(context).squeeze(-1)
        if tuple(residual.shape) != tuple(logits["ring_system_restate"].shape):
            raise RuntimeError(
                "ring-restatement context residual disagrees with action logits"
            )
        logits["ring_system_restate"] = logits["ring_system_restate"] + residual
        action_log_z = action_log_z.clone()
        family_index = MARK_RULE_TO_INDEX["ring_system_restate"]
        action_log_z[:, family_index] = torch.logsumexp(
            logits["ring_system_restate"].masked_fill(
                ~masks["ring_system_restate"],
                float("-inf"),
            ),
            dim=-1,
        )
        return masks, logits, action_log_z


__all__ = [
    "CONTEXTUAL_RING_RESTATE_SCORER_MODE",
    "ContextualRingRestateFactorizedTraceletRateModel",
]
