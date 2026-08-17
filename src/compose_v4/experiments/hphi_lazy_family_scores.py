"""Per-family RAW coordinate logits, computed one family at a time.

WHY THIS EXISTS, AND WHY IT IS A DUPLICATE
------------------------------------------
The lazy sampler draws the operator family FIRST and then builds only that
family's machinery. `_action_tables` cannot serve it: that function computes
every family in one pass, which is precisely the eager cost being removed --
cycle-close alone is 56.5% of law-build time while carrying 0.53% of realized
family mass.

The safe refactor would be to factor `_action_tables` so the eager and lazy
paths call the same per-family scorers. That is not available:
`factorized_tracelet_rate_model.py` is inside the Process-V2 identity hash, so
editing it moves `process_identity_sha256` and stops the Active8 / Gate-0 chain
from authenticating. Duplication is therefore forced by the provenance design,
not chosen.

Which makes the parity test load-bearing rather than decorative. Every function
here is checked BIT-FOR-BIT against the corresponding entry of
`_action_tables`' logits dict on real states before any of it is used
(`modal_apps/hphi_lazy_parity_app.py`). "Looks equivalent" is not a standard
these can be held to; a silent divergence in a logit changes the sampled law
without changing anything observable.

SCOPE
-----
The four families below carry 92.3% of realized family mass:

    bond_reroute    0.2749      atom_delete     0.2293
    atom_insert     0.2593      atom_restate    0.1592

Nothing here touches legality. These are the raw scores the masks gate, and the
masks provably do not modify them -- verified by reading `_action_tables`, where
every `logits[...]` assignment takes the head output directly and the admission
masks are only ANDed into `masks[...]`.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

__all__ = [
    "lazy_bond_reroute_logits",
    "lazy_atom_delete_logits",
    "lazy_atom_restate_logits",
    "lazy_grow_connected_logits",
    "LAZY_SCORERS",
]


def lazy_bond_reroute_logits(model, node: Tensor, global_state: Tensor,
                             pair: Tensor, batch) -> Tensor:
    """Graft head PLUS the relational residual. Both terms are required.

    The runtime model is `RelationalRerouteFactorizedTraceletRateModel`, whose
    `_action_tables` calls `super()` and then does

        logits["bond_reroute"] = logits["bond_reroute"] + relation_residual

    so the base-class graft head is only HALF the score. Duplicating the base
    class alone reproduced the features bit-for-bit and still missed the law by
    up to 3.5 in logit space -- on the family carrying 27.5% of all probability
    mass. The parity gate caught it; reading the base class did not, because the
    override lives in a different module.

    The pair-feature concatenation order is node-i, node-j, global. Order
    matters: the head is a learned projection of the concatenation, so
    transposing blocks yields finite, plausible, wrong logits.
    """
    n_slots = node.shape[1]
    graft_features = torch.cat(
        (
            node.unsqueeze(2).expand(-1, -1, n_slots, -1),
            node.unsqueeze(1).expand(-1, n_slots, -1, -1),
            global_state[:, None, None, :].expand(-1, n_slots, n_slots, -1),
        ),
        dim=-1,
    )
    base = model.graft_head(graft_features).squeeze(-1)
    if not hasattr(model, "graft_relation_head"):
        return base
    relation_features = model._graft_relation_features(batch, pair)
    return base + model.graft_relation_head(relation_features).squeeze(-1)


def lazy_atom_delete_logits(model, node: Tensor, global_state: Tensor,
                            pair: Tensor, batch) -> Tensor:
    """Mirror of `delete_logits = self.delete_head(node).squeeze(-1)`."""
    return model.delete_head(node).squeeze(-1)


def lazy_atom_restate_logits(model, node: Tensor, global_state: Tensor,
                             pair: Tensor, batch) -> Tensor:
    """Mirror of the restate head plus its per-class log prior.

    The prior is added with two leading unsqueezes so it broadcasts over batch
    and slot; dropping it would leave a head output that is off by a constant
    per target class, which is exactly the kind of error that survives a
    smoke test and changes every sampled restatement.
    """
    return model.restate_head(node) + model.atom_restate_log_prior.unsqueeze(
        0
    ).unsqueeze(0)


def lazy_grow_connected_logits(model, node: Tensor, global_state: Tensor,
                               pair: Tensor, batch) -> Tensor:
    """Mirror of the connected-insertion einsum, scaling and order prior.

    `grow_option.weight` is reshaped to (order, atom class, mark_dim) and
    contracted against the per-slot query. The 1/sqrt(mark_dim) scaling and the
    order/class log prior are both part of the score, not of the mask.
    """
    grow_query = model.grow_query(node)
    grow_option = model.grow_option.weight.reshape(
        3, len(model.atom_vocabulary), model.mark_dim
    )
    logits = torch.einsum("bnr,otr->bnot", grow_query, grow_option) / sqrt(
        model.mark_dim
    )
    return logits + model.connected_atom_order_log_prior.unsqueeze(0).unsqueeze(0)


#: table name -> scorer. Keyed by the `_action_tables` table name so parity can
#: be checked by direct dictionary lookup rather than by a hand-maintained map.
LAZY_SCORERS = {
    "bond_reroute": lazy_bond_reroute_logits,
    "atom_delete": lazy_atom_delete_logits,
    "atom_restate": lazy_atom_restate_logits,
    "grow_connected": lazy_grow_connected_logits,
}
