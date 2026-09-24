"""Optional learned-law tilt over already admitted molecular rewrite marks.

The base model already learns a legal-mark distribution.  This small development
ablation draws a bounded set of its marks and favours marks with higher learned
log probability *after* the caller has checked executor and task constraints.
It uses no QED, synthetic-accessibility score, benchmark target, or endpoint
filter.  A caller that does not opt in retains the original sampler unchanged.

This is a mark-level proposal ablation, not a representation-invariant molecular
control law.  Candidate count and extra model evaluations must be reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from compose_v4.model.factorized_tracelet_rate_model import prepare_factorized_mark_batch
from compose_v4.rewrite.kernel import InvalidRewrite


@dataclass(frozen=True)
class LearnedLegalMarkPrior:
    """Select among admitted marks using the checkpoint's own learned log law."""

    candidates: int = 3
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.candidates < 2:
            raise ValueError("prior candidate count must be at least two")
        if not math.isfinite(self.strength) or self.strength <= 0:
            raise ValueError("prior strength must be finite and positive")

    def log_probability(self, model: Any, state: Any, time: float, rule: str, action: Any) -> float:
        """Score an executed action through the model's production teacher scorer."""
        batch = prepare_factorized_mark_batch(
            (state,),
            (float(time),),
            (action,),
            (rule,),
            (1.0,),
            ring_catalog=model.ring_catalog,
            compute_ring_grow_support=model.enable_ring_grow_macro,
            compute_ring_restates=model.enable_ring_restates,
            compute_cyclic_graft=model.enable_cyclic_graft,
            compute_ring_opening=model.enable_ring_opening,
            compute_ring_system_delete=model.enable_ring_system_delete,
            editing_process_semantics=model.editing_process_semantics,
            atom_restate_action_semantics=model.atom_restate_action_semantics,
            ring_restate_scorer_mode=model.ring_restate_scorer_mode,
            cycle_close_action_semantics=model.cycle_close_action_semantics,
            cycle_open_action_semantics=model.cycle_open_action_semantics,
            atom_delete_action_semantics=model.atom_delete_action_semantics,
        )
        with torch.no_grad():
            value = float(model.forward_mark_batch(batch).selected_mark_log_probability[0])
        if not math.isfinite(value):
            raise ValueError(f"admitted {rule} mark has nonfinite learned probability")
        return value

    def choose(
        self,
        model: Any,
        state: Any,
        time: float,
        rng: np.random.Generator,
        candidates: Sequence[tuple[str, Any]],
    ) -> tuple[int, tuple[float, ...]]:
        """Choose one admitted candidate, with a stable softmax of learned scores."""
        if not candidates:
            raise ValueError("cannot rank an empty admitted-candidate set")
        scores = tuple(
            self.log_probability(model, state, time, rule, action) for rule, action in candidates
        )
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("admitted mark has nonfinite learned probability")
        if len(scores) == 1:
            return 0, scores
        shifted = self.strength * (np.asarray(scores, dtype=np.float64) - max(scores))
        weights = np.exp(shifted)
        probabilities = weights / weights.sum()
        return int(rng.choice(len(scores), p=probabilities)), scores


class LegalMarkTiltModel:
    """Opt-in rollout adapter; leave the checkpoint and executor unchanged.

    Ring-plan installation remains on the original checkpoint law so a paired
    de novo arm changes only subsequent edit selection, not the frozen plan.
    """

    def __init__(self, model: Any, system: Any, prior: LearnedLegalMarkPrior) -> None:
        self._model = model
        self._system = system
        self._prior = prior
        self.offered = 0
        self.executable = 0
        self.rank_events = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @property
    def disabled_sampling_rule_names(self) -> tuple[str, ...]:
        return getattr(self._model, "disabled_sampling_rule_names", ())

    @disabled_sampling_rule_names.setter
    def disabled_sampling_rule_names(self, value: tuple[str, ...]) -> None:
        self._model.disabled_sampling_rule_names = value

    @property
    def excluded_sampling_ring_template_indices(self) -> tuple[int, ...]:
        return getattr(self._model, "excluded_sampling_ring_template_indices", ())

    @excluded_sampling_ring_template_indices.setter
    def excluded_sampling_ring_template_indices(self, value: tuple[int, ...]) -> None:
        self._model.excluded_sampling_ring_template_indices = value

    def sample_rewrite_mark(self, state: Any, time: float, rng: np.random.Generator) -> Any:
        """Keep direct draws unchanged, including planned ring installation."""
        return self._model.sample_rewrite_mark(state, time, rng)

    def sample_hazard_probe(self, state: Any, time: float, rng: np.random.Generator) -> Any:
        """One unchanged base-law draw establishes the CTMC event hazard."""
        return self._model.sample_rewrite_mark(state, time, rng)

    def resample_rewrite_mark_after_event(
        self, state: Any, time: float, rng: np.random.Generator, first: Any
    ) -> Any:
        """Spend the extra candidate work only after an event actually occurs."""
        if first.total_hazard <= 0 or first.rule_name == "ring_system_grow":
            return first
        admissible = []
        for mark in (
            first,
            *(
                self._model.sample_rewrite_mark(state, time, rng)
                for _ in range(self._prior.candidates - 1)
            ),
        ):
            self.offered += 1
            if mark.action is None:
                continue
            if not math.isclose(mark.total_hazard, first.total_hazard, rel_tol=1e-5, abs_tol=1e-8):
                raise RuntimeError("candidate selection changed the CTMC hazard")
            try:
                self._system.apply(state, mark.rule_name, mark.action)
            except InvalidRewrite:
                continue
            self.executable += 1
            admissible.append(mark)
        if not admissible:
            if first.action is None and first.rule_name.startswith("<VIRTUAL_"):
                return first
            raise RuntimeError("learned-law candidate set contains no executable mark")
        chosen, _scores = self._prior.choose(
            self._model,
            state,
            time,
            rng,
            [(mark.rule_name, mark.action) for mark in admissible],
        )
        self.rank_events += 1
        return admissible[chosen]
