"""Equivalence panel: vectorized canonical-successor scoring vs the trusted loop reference.

A scalar loss can coincide while the GROUPING or the GRADIENTS are wrong, so every case compares:
  1. successor log-probabilities
  2. target successor probability
  3. per-example NLL
  4. gradients w.r.t. candidate logits
  5. one optimizer update

Deliberately difficult cases (each is a way the aggregation can silently disagree):
  * one successor reached by several symmetric marks
  * several successors with UNEQUAL alias counts
  * graft-style and non-graft aliases in the same batch
  * singleton candidate groups
  * examples with different ragged lengths in one batch
  * a target that is NOT the highest-logit candidate
  * large negative logits (numerical stability)
  * two examples SHARING a successor id (the pair-aggregation trap)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.model.segmented_successor import (  # noqa: E402
    reference_successor_logprobs,
    reference_target_nll,
    segmented_successor_logprobs,
    segmented_target_nll,
)

TOL = 1e-6


def _case(name, logits, to_example, to_successor, target, n_examples, n_successors):
    return dict(
        name=name,
        logits=torch.tensor(logits, dtype=torch.float64),
        to_example=torch.tensor(to_example, dtype=torch.long),
        to_successor=torch.tensor(to_successor, dtype=torch.long),
        target=torch.tensor(target, dtype=torch.long),
        n_examples=n_examples,
        n_successors=n_successors,
    )


PANEL = [
    # one successor reached by 3 symmetric marks; another by 1
    _case("symmetric_aliases", [0.5, 0.5, 0.5, 2.0], [0, 0, 0, 0], [0, 0, 0, 1], [0], 1, 2),
    # unequal alias counts across successors (1 vs 2 vs 4)
    _case("unequal_alias_counts",
          [1.0, 0.2, 0.3, -0.5, -0.4, -0.3, -0.2],
          [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 2, 2, 2, 2], [2], 1, 3),
    # graft-style (many aliases) and non-graft (singleton) in ONE batch
    _case("graft_and_nongraft_mixed",
          [0.7, 0.7, 0.7, 0.7, 1.5, -0.2],
          [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 2], [0, 2], 2, 3),
    # every group a singleton -> aggregation must reduce to plain log-softmax
    _case("all_singletons", [0.1, 0.9, -0.4], [0, 0, 0], [0, 1, 2], [1], 1, 3),
    # ragged: example 0 has 5 candidates, example 1 has 2
    _case("ragged_lengths",
          [0.1, 0.2, 0.3, 0.4, 0.5, -1.0, 2.0],
          [0, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 2, 3, 3], [1, 3], 2, 4),
    # target is NOT the argmax candidate
    _case("target_not_argmax", [3.0, 0.1, 0.2], [0, 0, 0], [0, 1, 1], [1], 1, 2),
    # large negative logits -- numerical stability
    _case("large_negative_logits",
          [-120.0, -119.5, -300.0, -0.5],
          [0, 0, 0, 0], [0, 0, 1, 2], [0], 1, 3),
    # TWO EXAMPLES SHARING a successor id: aggregating globally would leak mass between examples
    _case("shared_successor_id_across_examples",
          [0.3, 0.4, 1.1, 0.9],
          [0, 0, 1, 1], [0, 1, 0, 1], [0, 1], 2, 2),
]


@pytest.mark.parametrize("case", PANEL, ids=lambda c: c["name"])
def test_successor_logprobs_match(case):
    """(1) successor log-probabilities."""
    ref = reference_successor_logprobs(
        case["logits"], case["to_example"], case["to_successor"],
        case["n_examples"], case["n_successors"])
    vec = segmented_successor_logprobs(
        case["logits"], case["to_example"], case["to_successor"],
        case["n_examples"], case["n_successors"])
    finite = torch.isfinite(ref)
    assert torch.allclose(ref[finite], vec[finite], atol=TOL, rtol=TOL), f"{ref} vs {vec}"
    assert torch.equal(torch.isfinite(ref), torch.isfinite(vec)), "empty-group handling differs"


@pytest.mark.parametrize("case", PANEL, ids=lambda c: c["name"])
def test_per_example_nll_and_target_probability_match(case):
    """(2) target successor probability and (3) per-example NLL."""
    ref = reference_target_nll(
        case["logits"], case["to_example"], case["to_successor"], case["target"],
        case["n_examples"], case["n_successors"])
    vec = segmented_target_nll(
        case["logits"], case["to_example"], case["to_successor"], case["target"],
        case["n_examples"], case["n_successors"])
    assert torch.allclose(ref, vec, atol=TOL, rtol=TOL), f"{case['name']}: {ref} vs {vec}"
    # target probability itself, not just its log
    assert torch.allclose(torch.exp(-ref), torch.exp(-vec), atol=TOL, rtol=TOL)


@pytest.mark.parametrize("case", PANEL, ids=lambda c: c["name"])
def test_gradients_match(case):
    """(4) gradients w.r.t. candidate logits -- where a wrong grouping usually shows up even when the
    scalar loss happens to agree."""
    a = case["logits"].clone().requires_grad_(True)
    b = case["logits"].clone().requires_grad_(True)
    reference_target_nll(a, case["to_example"], case["to_successor"], case["target"],
                         case["n_examples"], case["n_successors"]).sum().backward()
    segmented_target_nll(b, case["to_example"], case["to_successor"], case["target"],
                         case["n_examples"], case["n_successors"]).sum().backward()
    assert a.grad is not None and b.grad is not None
    assert torch.allclose(a.grad, b.grad, atol=TOL, rtol=TOL), (
        f"{case['name']} gradient mismatch:\n  ref {a.grad}\n  vec {b.grad}")


@pytest.mark.parametrize("case", PANEL, ids=lambda c: c["name"])
def test_one_optimizer_update_matches(case):
    """(5) a single optimizer step must land on the same parameters."""
    a = case["logits"].clone().requires_grad_(True)
    b = case["logits"].clone().requires_grad_(True)
    opt_a = torch.optim.AdamW([a], lr=1e-2)
    opt_b = torch.optim.AdamW([b], lr=1e-2)
    opt_a.zero_grad(set_to_none=True)
    reference_target_nll(a, case["to_example"], case["to_successor"], case["target"],
                         case["n_examples"], case["n_successors"]).sum().backward()
    opt_a.step()
    opt_b.zero_grad(set_to_none=True)
    segmented_target_nll(b, case["to_example"], case["to_successor"], case["target"],
                         case["n_examples"], case["n_successors"]).sum().backward()
    opt_b.step()
    assert torch.allclose(a.detach(), b.detach(), atol=TOL, rtol=TOL)


def test_singletons_reduce_to_plain_log_softmax():
    """Sanity anchor: with every candidate its own successor, aggregation must be exactly log-softmax."""
    logits = torch.tensor([0.1, 0.9, -0.4], dtype=torch.float64)
    to_example = torch.zeros(3, dtype=torch.long)
    to_successor = torch.arange(3)
    vec = segmented_successor_logprobs(logits, to_example, to_successor, 1, 3)
    assert torch.allclose(vec, torch.log_softmax(logits, dim=0), atol=TOL)


def test_aggregation_strictly_exceeds_any_single_alias():
    """Aggregating aliases must ADD probability: P(successor) > max alias probability when >1 alias."""
    logits = torch.tensor([0.5, 0.5, 0.5, 2.0], dtype=torch.float64)
    to_example = torch.zeros(4, dtype=torch.long)
    to_successor = torch.tensor([0, 0, 0, 1])
    log_p = segmented_successor_logprobs(logits, to_example, to_successor, 1, 2)
    single = torch.log_softmax(logits, dim=0)
    assert log_p[0] > single[0], "three aliases must carry more mass than one of them"
    assert torch.allclose(log_p[1], single[3], atol=TOL), "a singleton group must be unchanged"


def test_probabilities_sum_to_one_per_example():
    """Successor probabilities within an example must still form a distribution."""
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    to_example = torch.zeros(5, dtype=torch.long)
    to_successor = torch.tensor([0, 0, 1, 1, 2])
    log_p = segmented_successor_logprobs(logits, to_example, to_successor, 1, 3)
    assert torch.allclose(torch.exp(log_p).sum(), torch.tensor(1.0, dtype=torch.float64), atol=TOL)


def test_panel_detects_the_naive_global_aggregation_bug():
    """Proof the panel is DISCRIMINATING, not just green.

    The obvious implementation aggregates per GLOBAL successor id. That is wrong when two examples share a
    successor id: one example's alias mass leaks into another's target. This reproduces that bug and asserts
    the panel's shared-id case rejects it -- so the case is load-bearing rather than decorative.
    """
    from compose_v4.model.segmented_successor import _segment_logsumexp

    def naive_global_nll(logits, to_example, to_successor, target, n_examples, n_successors):
        example_norm = _segment_logsumexp(logits, to_example, n_examples)
        log_probs = logits - example_norm.index_select(0, to_example)
        # BUG: aggregate by successor id ALONE, ignoring which example the candidate came from
        per_successor = _segment_logsumexp(log_probs, to_successor, n_successors)
        return -per_successor.index_select(0, target)

    case = next(c for c in PANEL if c["name"] == "shared_successor_id_across_examples")
    ref = reference_target_nll(
        case["logits"], case["to_example"], case["to_successor"], case["target"],
        case["n_examples"], case["n_successors"])
    good = segmented_target_nll(
        case["logits"], case["to_example"], case["to_successor"], case["target"],
        case["n_examples"], case["n_successors"])
    bad = naive_global_nll(
        case["logits"], case["to_example"], case["to_successor"], case["target"],
        case["n_examples"], case["n_successors"])

    assert torch.allclose(ref, good, atol=TOL), "correct implementation must match the reference"
    assert not torch.allclose(ref, bad, atol=TOL), (
        "the naive global aggregation must DISAGREE here -- if it agrees, this panel case cannot catch "
        "cross-example mass leakage and needs strengthening")
