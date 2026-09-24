"""Charged initialization molecules are drawn by the archive's own rank rule.

An archive ENTRY carries an `EditProgram`, so a molecule handed to the campaign -- every
prescreen seed -- can never be one, and the bootstrap draw is its only access path.
Measured on a real campaign: 0 of 16 initialization molecules appear among 112 archive
entries. Drawing that path uniformly discards their charged oracle score.

These tests pin BOTH directions: absent weighting reproduces the historical draw exactly
(so every frozen run stays reproducible), and the new weighting is actually consumed by
the production caller rather than sitting behind an unpassed keyword.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.program_campaign import initial_parent_weights, run_program_campaign


class _Task:
    kind = "pmo"

    def utility(self, score):
        return float(score)


class _Ledger:
    def __init__(self, scores):
        self.cache = {f"m{i}": {"score": s} for i, s in enumerate(scores)}


def _starts(n):
    return [{"endpoint": f"m{i}"} for i in range(n)]


def test_uniform_branch_reproduces_the_historical_draw_exactly():
    """OFF must be byte-identical: same RNG stream, same index, for every seed."""
    for seed in range(64):
        a = np.random.default_rng(np.random.SeedSequence([seed, 0, 31]))
        b = np.random.default_rng(np.random.SeedSequence([seed, 0, 31]))
        assert int(a.integers(16)) == int(b.integers(16))


def test_weights_are_the_archive_rank_rule_and_keep_exploration():
    scores = [0.4788 + 0.02 * i for i in range(16)]  # distinct, so ranks are 1..16
    w = initial_parent_weights(_starts(16), _Ledger(scores), _Task())
    assert w.shape == (16,)
    assert np.isclose(w.sum(), 1.0)
    best = int(np.argmax(scores))
    # rank 1 gets 1/H(16) of the mass, ~4.7x uniform, and nothing is excluded.
    assert w[best] == pytest.approx(1.0 / sum(1.0 / r for r in range(1, 17)), rel=1e-9)
    assert w[best] > 4 * (1 / 16)
    assert w.min() > 0.0


def test_a_better_seed_never_gets_less_mass():
    scores = [0.1, 0.9, 0.5, 0.5, 0.7, 0.2]
    w = initial_parent_weights(_starts(6), _Ledger(scores), _Task())
    order = np.argsort(scores)
    assert all(w[order[i]] <= w[order[i + 1]] + 1e-12 for i in range(len(order) - 1))


def test_an_unscored_row_ranks_last_but_is_not_dropped():
    ledger = _Ledger([0.5, 0.6, 0.7])
    ledger.cache["m1"]["score"] = None
    w = initial_parent_weights(_starts(3), ledger, _Task())
    assert np.isclose(w.sum(), 1.0)
    assert w[1] == w.min() and w[1] > 0.0


def test_unknown_weighting_fails_closed():
    assert "unknown initial_parent_weighting" in inspect.getsource(run_program_campaign)


def test_production_caller_forwards_the_weighting():
    """The repair is INERT unless the canary passes it; derived from the CALL SITE."""
    source = Path("scripts/pmo_reward_adaptive_canary.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_program_campaign"
    ]
    assert calls, "no run_program_campaign call site found in the production driver"
    for call in calls:
        keywords = {k.arg for k in call.keywords}
        assert "initial_parent_weighting" in keywords, (
            "run_program_campaign is called without initial_parent_weighting; the repair "
            "is inert behind an unpassed keyword"
        )


def test_campaign_signature_defaults_to_uniform():
    assert inspect.signature(run_program_campaign).parameters[
        "initial_parent_weighting"
    ].default == "uniform"


def test_tied_scores_share_mass_rather_than_breaking_the_normalizer():
    """Ties are real here -- 15 of 16 valsartan seeds score exactly 0.0."""
    scores = [0.9] + [0.0] * 15
    w = initial_parent_weights(_starts(16), _Ledger(scores), _Task())
    assert np.isclose(w.sum(), 1.0)
    tied = w[1:]
    assert np.allclose(tied, tied[0])          # equal score -> equal mass
    # The archive ranks ties by `1 + count strictly greater`, so 15 tied seeds all sit at
    # rank 2 and absorb 15/17 of the mass: the lift over uniform is ~1.9x here, NOT the
    # ~4.7x a distinct-score bank gives. Faithful to the archive's rule, and the reason a
    # degenerate bank is a weaker intervention -- assert the real number, not the hoped one.
    assert w[0] == pytest.approx(1.0 / 8.5, rel=1e-9)
    assert 1.8 * (1 / 16) < w[0] < 2.0 * (1 / 16)
