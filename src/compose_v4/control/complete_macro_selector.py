"""Source-conditioned selection over complete realized structural macros.

The selector changes only the ordering of an immutable generated candidate
pool.  It reuses the existing complete structural-goal feature map and pairwise
ranker, then applies a small fixed penalty for compiler work already observed in
the candidate's exact-realization receipt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import StructuralGoal
from compose_v4.control.structural_subgoal_policy import (
    ContextSubgoalRanker,
    fit_context_subgoal_ranker,
    minimize_subgoal,
    proposal_features,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

CHECKPOINT_SCHEMA = "t4_complete_macro_selector_v1"


def complete_macro_templates(goal_payload: dict) -> tuple:
    """Return address-free minimal templates for one complete macro goal."""

    goal = StructuralGoal.from_payload(goal_payload)
    return tuple(minimize_subgoal(subgoal)[0] for subgoal in goal.subgoals)


def complete_macro_features(source, endpoint_state: dict, goal_payload: dict) -> np.ndarray:
    """Reuse the registered source/endpoint/complete-goal feature map."""

    return proposal_features(
        source,
        decode_state(endpoint_state),
        complete_macro_templates(goal_payload),
    )


def compiler_log_work(candidate: dict) -> float:
    """Validate an exact realization receipt and summarize its observed work."""

    receipt = candidate.get("realization")
    if not isinstance(receipt, dict):
        raise TypeError("complete macro lacks an exact-realization receipt")
    if (
        receipt.get("status") != "realized"
        or receipt.get("endpoint_matches_bound_target") is not True
        or receipt.get("primitive_teacher_actions_used") != 0
    ):
        raise ValueError("complete macro is not a teacher-free exact realization")
    counts = []
    for name in ("primitive_count", "expanded", "attempted"):
        value = receipt.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid exact-realization work field: {name}")
        counts.append(value)
    if counts[0] < 1:
        raise ValueError("a realized complete macro must emit a primitive")
    return math.log1p(sum(counts))


def complete_macro_signature(candidate: dict) -> tuple[str, ...]:
    """Canonical exact patch-composition signature used by the diversity guard."""

    patch_ids = candidate.get("patch_ids")
    if (
        not isinstance(patch_ids, list)
        or not patch_ids
        or any(not isinstance(value, str) or not value for value in patch_ids)
    ):
        raise ValueError("complete macro requires nonempty patch identities")
    return tuple(sorted(patch_ids))


@dataclass(frozen=True)
class CompleteMacroSelector:
    """Pairwise complete-goal ranker with a frozen compiler-work auxiliary."""

    ranker: ContextSubgoalRanker
    compiler_cost_mean: float
    compiler_cost_scale: float
    compiler_cost_weight: float
    training_identity: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.compiler_cost_mean)
            or not math.isfinite(self.compiler_cost_scale)
            or self.compiler_cost_scale <= 0
            or not math.isfinite(self.compiler_cost_weight)
            or self.compiler_cost_weight < 0
            or not self.training_identity
        ):
            raise ValueError("invalid complete-macro selector")

    def score(self, source, candidate: dict) -> float:
        features = complete_macro_features(source, candidate["endpoint_state"], candidate["goal"])
        normalized_work = (
            compiler_log_work(candidate) - self.compiler_cost_mean
        ) / self.compiler_cost_scale
        return self.ranker.score(features) - self.compiler_cost_weight * normalized_work

    def checkpoint(self) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "ranker": self.ranker.checkpoint(),
            "compiler_cost_mean": self.compiler_cost_mean,
            "compiler_cost_scale": self.compiler_cost_scale,
            "compiler_cost_weight": self.compiler_cost_weight,
            "training_identity": self.training_identity,
            "runtime_teacher_rows": 0,
            "runtime_candidate_rows": 0,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> CompleteMacroSelector:
        expected = {
            "schema_version",
            "ranker",
            "compiler_cost_mean",
            "compiler_cost_scale",
            "compiler_cost_weight",
            "training_identity",
            "runtime_teacher_rows",
            "runtime_candidate_rows",
        }
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA
            or set(payload) != expected
            or payload.get("runtime_teacher_rows") != 0
            or payload.get("runtime_candidate_rows") != 0
        ):
            raise ValueError("complete-macro selector checkpoint schema mismatch")
        return cls(
            ContextSubgoalRanker.from_checkpoint(payload["ranker"]),
            float(payload["compiler_cost_mean"]),
            float(payload["compiler_cost_scale"]),
            float(payload["compiler_cost_weight"]),
            str(payload["training_identity"]),
        )


def _source_balanced_cost_moments(
    negatives: dict[str, list[tuple[np.ndarray, float]]],
) -> tuple[float, float]:
    if not negatives or any(not rows for rows in negatives.values()):
        raise ValueError("compiler-cost moments require negatives for every source")
    values = []
    weights = []
    for source in sorted(negatives):
        rows = negatives[source]
        for _, value in rows:
            if not math.isfinite(value):
                raise ValueError("nonfinite compiler-cost feature")
            values.append(value)
            weights.append(1 / (len(negatives) * len(rows)))
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    mean = float(np.sum(array * weight))
    scale = max(float(np.sqrt(np.sum(weight * (array - mean) ** 2))), 0.05)
    return mean, scale


def fit_complete_macro_selector(
    positives: list[tuple[str, np.ndarray]],
    negatives: dict[str, list[tuple[np.ndarray, float]]],
    *,
    updates: int,
    learning_rate: float,
    l2: float,
    seed: int,
    compiler_cost_weight: float,
) -> tuple[CompleteMacroSelector, dict]:
    """Fit one split-first selector on same-source cross-fitted hard negatives."""

    feature_negatives = {
        source: [features for features, _ in rows] for source, rows in negatives.items()
    }
    ranker, ranker_report = fit_context_subgoal_ranker(
        positives,
        feature_negatives,
        updates=updates,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    cost_mean, cost_scale = _source_balanced_cost_moments(negatives)
    training_identity = identity(
        {
            "schema_version": "t4_complete_macro_selector_training_identity_v1",
            "ranker": ranker.training_identity,
            "sources": len(negatives),
            "positives": len(positives),
            "negatives": sum(len(rows) for rows in negatives.values()),
            "compiler_cost_mean": cost_mean,
            "compiler_cost_scale": cost_scale,
            "compiler_cost_weight": compiler_cost_weight,
        }
    )
    model = CompleteMacroSelector(
        ranker,
        cost_mean,
        cost_scale,
        compiler_cost_weight,
        training_identity,
    )
    return model, {
        "ranker": ranker_report,
        "compiler_cost_mean": cost_mean,
        "compiler_cost_scale": cost_scale,
        "compiler_cost_weight": compiler_cost_weight,
    }


def rank_complete_macros(
    source, candidates: list[dict], selector: CompleteMacroSelector
) -> list[dict]:
    """Rerank a fixed pool, placing unique complete-macro signatures first."""

    if not candidates:
        raise ValueError("complete-macro ranking requires a candidate pool")
    rows = []
    seen_identities = set()
    for original_rank, candidate in enumerate(candidates, 1):
        candidate_identity = identity(candidate)
        if candidate_identity in seen_identities:
            raise ValueError("complete-macro pool contains a duplicate candidate")
        seen_identities.add(candidate_identity)
        endpoint_key = canonical_state_key(decode_state(candidate["endpoint_state"]))
        if endpoint_key == canonical_state_key(source):
            raise ValueError("complete-macro pool contains a self event")
        rows.append(
            {
                "candidate": candidate,
                "candidate_identity": candidate_identity,
                "complete_macro_signature": complete_macro_signature(candidate),
                "endpoint_key": endpoint_key,
                "original_rank": original_rank,
                "selector_score": selector.score(source, candidate),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["selector_score"],
            row["original_rank"],
            row["endpoint_key"],
            row["candidate_identity"],
        )
    )
    novel, repeated, signatures = [], [], set()
    for row in rows:
        signature = row["complete_macro_signature"]
        if signature in signatures:
            repeated.append(row)
        else:
            signatures.add(signature)
            novel.append(row)
    return [*novel, *repeated]


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CompleteMacroSelector",
    "compiler_log_work",
    "complete_macro_features",
    "complete_macro_signature",
    "complete_macro_templates",
    "fit_complete_macro_selector",
    "rank_complete_macros",
]
