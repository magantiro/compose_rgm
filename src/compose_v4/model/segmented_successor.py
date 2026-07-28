"""Canonical-successor aggregation over ragged candidate sets: trusted reference + vectorized path.

The molecular kernel is defined on SUCCESSORS, not marks:

    P(y | x) = sum_{a : T(x,a) ~= y} p(a | x)          [aggregate in probability space]
    log P    = logsumexp_{a in G_y(x)} log p(a | x)    [what we actually compute]

Two implementations live here on purpose:

  * ``reference_*`` -- explicit Python loops over examples and successor groups. Slow, obviously correct,
    and the arbiter of truth. It stays as the test oracle and is never the production path.
  * ``segmented_*`` -- flat ragged tensors with segmented reductions. This is what training uses.

They must agree on more than the scalar loss: a scalar can coincide while the GROUPING or the GRADIENTS are
wrong, so the equivalence panel compares successor log-probabilities, target probability, per-example NLL,
gradients w.r.t. candidate logits, and an optimizer step.

Ragged layout (all 1-D, length = total candidates across the batch):
    candidate_logits          raw scores, pre-normalization
    candidate_to_example      which example each candidate belongs to
    candidate_to_successor    GLOBAL successor-group id (candidates that execute to the same molecule)
    target_successor          [batch] the successor group the teacher produced
"""
from __future__ import annotations

import torch


def _segment_logsumexp(values: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Numerically-stable segmented log-sum-exp.

    Subtracts a per-segment max before exponentiating; empty segments stay -inf rather than NaN, which
    matters because a batch legitimately contains successor ids with no candidates in some examples.
    """
    if values.numel() == 0:
        return torch.full((n_segments,), float("-inf"), dtype=values.dtype, device=values.device)
    neg_inf = torch.finfo(values.dtype).min
    maxes = torch.full((n_segments,), neg_inf, dtype=values.dtype, device=values.device)
    maxes = maxes.scatter_reduce(0, index, values, reduce="amax", include_self=True)
    # empty segments keep the sentinel; shift them to 0 so exp() is finite, then restore -inf at the end
    empty = maxes <= neg_inf
    safe_max = torch.where(empty, torch.zeros_like(maxes), maxes)
    shifted = torch.exp(values - safe_max.index_select(0, index))
    summed = torch.zeros(n_segments, dtype=values.dtype, device=values.device)
    summed = summed.index_add(0, index, shifted)
    out = safe_max + torch.log(summed.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(empty, torch.full_like(out, float("-inf")), out)


# ---- trusted reference (loops; the oracle) ----------------------------------------------------------


def reference_successor_logprobs(
    candidate_logits: torch.Tensor,
    candidate_to_example: torch.Tensor,
    candidate_to_successor: torch.Tensor,
    n_examples: int,
    n_successors: int,
) -> torch.Tensor:
    """[n_successors] log P(successor). Explicit loops: normalize within each example, then aggregate
    each successor group in probability space via logsumexp."""
    out = torch.full((n_successors,), float("-inf"), dtype=candidate_logits.dtype)
    for example in range(n_examples):
        mask = candidate_to_example == example
        if not bool(mask.any()):
            continue
        logits = candidate_logits[mask]
        log_probs = logits - torch.logsumexp(logits, dim=0)        # normalize over this example's fiber
        groups = candidate_to_successor[mask]
        for group in torch.unique(groups):
            members = log_probs[groups == group]
            aggregated = torch.logsumexp(members, dim=0)
            prior = out[group]
            out[group] = aggregated if prior == float("-inf") else torch.logsumexp(
                torch.stack([prior, aggregated]), dim=0
            )
    return out


def reference_target_nll(
    candidate_logits: torch.Tensor,
    candidate_to_example: torch.Tensor,
    candidate_to_successor: torch.Tensor,
    target_successor: torch.Tensor,
    n_examples: int,
    n_successors: int,
) -> torch.Tensor:
    """[n_examples] per-example canonical-successor NLL, computed with loops."""
    losses = []
    for example in range(n_examples):
        mask = candidate_to_example == example
        logits = candidate_logits[mask]
        log_probs = logits - torch.logsumexp(logits, dim=0)
        groups = candidate_to_successor[mask]
        target = target_successor[example]
        members = log_probs[groups == target]
        if members.numel() == 0:
            losses.append(torch.tensor(float("inf"), dtype=candidate_logits.dtype))
        else:
            losses.append(-torch.logsumexp(members, dim=0))
    return torch.stack(losses)


# ---- vectorized production path ---------------------------------------------------------------------


def segmented_successor_logprobs(
    candidate_logits: torch.Tensor,
    candidate_to_example: torch.Tensor,
    candidate_to_successor: torch.Tensor,
    n_examples: int,
    n_successors: int,
) -> torch.Tensor:
    """[n_successors] log P(successor), with no Python loop over examples or groups."""
    example_norm = _segment_logsumexp(candidate_logits, candidate_to_example, n_examples)
    log_probs = candidate_logits - example_norm.index_select(0, candidate_to_example)
    return _segment_logsumexp(log_probs, candidate_to_successor, n_successors)


def segmented_target_nll(
    candidate_logits: torch.Tensor,
    candidate_to_example: torch.Tensor,
    candidate_to_successor: torch.Tensor,
    target_successor: torch.Tensor,
    n_examples: int,
    n_successors: int,
) -> torch.Tensor:
    """[n_examples] per-example canonical-successor NLL, fully vectorized.

    Aggregation is done per (example, successor) pair rather than per global successor: two examples can
    share a successor id, and collapsing them would leak one example's mass into another's target.
    """
    example_norm = _segment_logsumexp(candidate_logits, candidate_to_example, n_examples)
    log_probs = candidate_logits - example_norm.index_select(0, candidate_to_example)
    pair = candidate_to_example * n_successors + candidate_to_successor
    pair_logsumexp = _segment_logsumexp(log_probs, pair, n_examples * n_successors)
    target_pair = torch.arange(n_examples, device=candidate_logits.device) * n_successors + target_successor
    return -pair_logsumexp.index_select(0, target_pair)
