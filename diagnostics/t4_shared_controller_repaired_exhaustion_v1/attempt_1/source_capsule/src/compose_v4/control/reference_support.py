"""Effective accessibility of an option under a reference process.

Control can only reweight what the reference already emits. If `q_ref(O|G)` is zero, or
small enough that the option would never be drawn inside a run, no reward tilt of any
strength reaches it. So before a controller is built on the admitted intervention
siblings, the question is whether any reference process can actually propose them.

Two candidate reference laws are compared, and the comparison is the point.

`q_legal` -- the uninformed law over the frozen grammar. It draws each token uniformly
from the fiber `token_domain` returns for it, given the prefix. This is a genuine
stochastic process over the legal option space rather than an informal "random mutation":
the sequence of fibers is exactly what the decoder reads, so every draw is a legal patch
and every legal patch has positive mass.

    log q_legal(O | G) = - sum_j log |D_j(prefix_j)|

`q_block` -- the closure-conditional law. It conditions on an EXISTING patch, picks one
semantic coordinate, and resamples that coordinate's minimal closure from its own fiber,
holding the compatible complement fixed. Its support is the set of patches reachable by
one intervention from the conditioning patch.

    q_block(O' | O, G) = P(coordinate) * P(value | coordinate)

The expected number of draws to encounter an option is `1 / q`. That number, not the
presence or absence of a zero, is what decides whether a reference is usable inside a
budget.
"""

from __future__ import annotations

import math

from compose_v4.control.complete_region_patch_policy import (
    SourceRegionContext,
    encode_patch_stream,
    token_domain,
)

SCHEMA_VERSION = "reference_support_v1"


def log_q_legal(subgoal, context: SourceRegionContext | None = None) -> dict:
    """Log-probability of one patch under the uniform-over-fiber grammar law.

    Walks the patch's own token stream and charges each decision the log-size of the
    fiber it was drawn from. Deterministic tokens -- a fiber of size one -- are free,
    which is what makes this a law over genuine decisions rather than over serialisation.
    """
    context = context or SourceRegionContext.from_subgoal(subgoal)
    tokens = encode_patch_stream(subgoal)
    total, decisions, widths = 0.0, 0, []
    for index, token in enumerate(tokens):
        fiber = token_domain(context, tokens[:index], kind=token.kind, factor=token.factor)
        size = len(fiber)
        if size <= 1:
            continue
        if token.value not in fiber:
            return {
                "log_probability": -math.inf,
                "outside_fiber": True,
                "kind": token.kind,
                "decisions": decisions,
            }
        total -= math.log(size)
        decisions += 1
        widths.append(size)
    return {
        "log_probability": total,
        "outside_fiber": False,
        "decisions": decisions,
        "mean_fiber_width": sum(widths) / len(widths) if widths else 1.0,
        "expected_draws": math.exp(-total) if total > -700 else math.inf,
    }


def log_q_block(*, coordinates: int, alternatives: int) -> dict:
    """Log-probability of one sibling under the closure-conditional law.

    A draw picks a coordinate uniformly among those the patch offers, then a value
    uniformly among that coordinate's alternatives. Conditioning on an existing patch is
    what keeps this tractable: the complement is not re-drawn, so its cost does not appear.
    """
    if min(coordinates, alternatives) < 1:
        raise ValueError("a closure-conditional draw needs at least one coordinate and value")
    total = -math.log(coordinates) - math.log(alternatives)
    return {
        "log_probability": total,
        "expected_draws": math.exp(-total),
        "coordinates": coordinates,
        "alternatives": alternatives,
    }


def mixture(log_route: float, log_legal: float, *, epsilon: float) -> float:
    """log of (1-eps) q_route + eps q_legal, computed stably.

    `epsilon` is a support floor, not a tuning knob: it guarantees absolute continuity so
    that a controlled law can reach any legal option. It must not be fitted against
    benchmark scores, because that would turn a reachability guarantee into a quantity
    selected on the test set.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    terms = []
    if log_route > -math.inf:
        terms.append(math.log1p(-epsilon) + log_route)
    terms.append(math.log(epsilon) + log_legal)
    high = max(terms)
    return high + math.log(sum(math.exp(t - high) for t in terms))


def accessibility(log_probability: float, *, budget: int) -> dict:
    """Would this option realistically be seen inside a budget of draws?"""
    if log_probability == -math.inf:
        return {"reachable": False, "probability_within_budget": 0.0, "expected_draws": math.inf}
    # 1 - (1-p)^budget, stable for tiny p
    probability = (
        -math.expm1(budget * math.log1p(-math.exp(log_probability)))
        if log_probability < -1e-12
        else 1.0
    )
    return {
        "reachable": True,
        "probability_within_budget": probability,
        "expected_draws": math.exp(-log_probability),
    }
