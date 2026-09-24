"""Behavioral checks for the opt-in learned legal-mark proposal tilt."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.model.legal_mark_prior import LearnedLegalMarkPrior, LegalMarkTiltModel
from compose_v4.rewrite.kernel import InvalidRewrite


def test_prior_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LearnedLegalMarkPrior(candidates=1)
    with pytest.raises(ValueError, match="finite and positive"):
        LearnedLegalMarkPrior(strength=0.0)


def test_prior_uses_only_supplied_admitted_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = LearnedLegalMarkPrior(candidates=3, strength=2.0)
    seen: list[str] = []

    def score(_self, _model, _state, _time, rule, _action):
        seen.append(rule)
        return {"a": -8.0, "b": -1.0, "c": -9.0}[rule]

    monkeypatch.setattr(LearnedLegalMarkPrior, "log_probability", score)
    chosen, scores = prior.choose(
        object(),
        object(),
        0.5,
        np.random.default_rng(7),
        [("a", object()), ("b", object()), ("c", object())],
    )
    assert seen == ["a", "b", "c"]
    assert scores == (-8.0, -1.0, -9.0)
    assert chosen == 1


def test_prior_abstains_on_nonfinite_admitted_score(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = LearnedLegalMarkPrior()
    monkeypatch.setattr(
        LearnedLegalMarkPrior,
        "log_probability",
        lambda *_args: float("-inf"),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        prior.choose(object(), object(), 0.5, np.random.default_rng(1), [("a", object())])


def test_rollout_adapter_ranks_only_executable_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    marks = iter(
        (
            SampledRewriteMark(3.0, "first", 1),
            SampledRewriteMark(3.0, "refused", 2),
            SampledRewriteMark(3.0, "second", 3),
        )
    )

    class Model:
        def sample_rewrite_mark(self, _state, _time, _rng):
            return next(marks)

    class System:
        def apply(self, _state, rule, _action):
            if rule == "refused":
                raise InvalidRewrite("incompatible edit")

    def score(_self, _model, _state, _time, rule, _action):
        return {"first": -10.0, "second": -1.0}[rule]

    monkeypatch.setattr(LearnedLegalMarkPrior, "log_probability", score)
    adapter = LegalMarkTiltModel(Model(), System(), LearnedLegalMarkPrior())
    rng = np.random.default_rng(7)
    first = adapter.sample_hazard_probe(object(), 0.5, rng)
    assert adapter.offered == 0
    chosen = adapter.resample_rewrite_mark_after_event(object(), 0.5, rng, first)
    assert chosen.rule_name == "second"
    assert (adapter.offered, adapter.executable, adapter.rank_events) == (3, 2, 1)
