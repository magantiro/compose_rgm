"""epsilon full-support audit (recipe prerequisite): determine whether an epsilon full-support mixture
    P_eps = (1-eps) P_theta + eps P_legal
is REQUIRED, or merely an optional exploration/robustness ablation.

The model's mark law is a softmax over the dense-mask-legal marks, so every mark with a finite logit
(i.e. every mark the model deems legal and can sample) receives STRICTLY POSITIVE probability -- P_theta
already has full support over its declared mark/successor space. The only transitions with zero
probability are OUT-OF-VOCABULARY by design: families the model does not represent (bond_insert /
bond_delete) and executor-legal-but-mask-illegal marks (non-root-free grafts, ring-site restates without
heteroatom-scan). Those are model-side gating choices, not hard-gated *legal* actions that epsilon must
rescue. Conclusion (see docs/AUDIT_REPORT.md): epsilon is OPTIONAL, not a prerequisite; if used, it must
be mixed on the post-quotient fixed-step MOLECULAR kernel (which editing samples), never on a CTMC hazard
that editing discards.
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog

_SLOTS = 20


def _model():
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        TypedRingCatalog((), (), ()), hidden_dim=16, message_passing_steps=1
    ).eval()


def _state(smi):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)


def _sampleable_mark_probs(model, state, *, n=400):
    """Probability of every DISTINCT mark the model actually samples at this state (dense-mask-legal by
    construction), scored through the same forward pass the loss uses."""
    rng = np.random.default_rng(0)
    seen: dict = {}
    for _ in range(n):
        mark = model.sample_rewrite_mark(state, 0.5, rng)
        if mark.action is not None:
            seen[(mark.rule_name, repr(mark.action))] = (mark.rule_name, mark.action)
    marks = list(seen.values())
    rules = tuple(r for r, _ in marks)
    actions = tuple(a for _, a in marks)
    batch = prepare_factorized_mark_batch(
        (state,) * len(marks), (0.5,) * len(marks), actions, rules, (0.0,) * len(marks)
    ).to(model.device)
    with torch.no_grad():
        return np.exp(model.forward_mark_batch(batch).selected_mark_log_probability.detach().cpu().numpy())


def test_softmax_gives_full_support_over_sampleable_marks() -> None:
    # Every mark the model can sample gets strictly positive, finite probability => P_theta has full
    # support over its declared mark space => an epsilon full-support mixture is OPTIONAL, not required.
    model = _model()
    for smi in ("CCC=O", "CC(C)(C)C", "CCOCC", "C"):
        probs = _sampleable_mark_probs(model, _state(smi))
        assert len(probs) > 0
        assert bool(np.isfinite(probs).all()), f"{smi}: non-finite mark probability"
        assert float(probs.min()) > 0.0, f"{smi}: a sampleable mark had zero probability"
