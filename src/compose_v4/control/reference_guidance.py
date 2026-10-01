"""Optional reference preferences over an existing controller's candidate panel.

This module neither constructs molecules nor invents a baseline selection law.
The caller supplies its actual pre-lock probabilities. No random draws occur here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

SCORE_SEMANTICS = "mean_productive_canonical_successor_log_probability; finite_panel_preference"


@dataclass(frozen=True)
class ProgramScore:
    candidate_id: str
    value: float | None
    status: Literal["scored", "unsupported_state", "unsupported_native_mark", "missing_trace"]
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        if self.status == "scored":
            if self.value is None or not math.isfinite(self.value) or self.value > 0:
                raise ValueError("a scored successor log-probability must be finite and <= 0")
        elif self.status in ("unsupported_state", "unsupported_native_mark", "missing_trace"):
            if self.value is not None or not self.detail:
                raise ValueError("unscored programs require a reason and no numerical substitute")
        else:
            raise ValueError(f"unknown program score status: {self.status!r}")


@dataclass(frozen=True)
class ScoredPanel:
    checkpoint_sha256: str
    scores: tuple[ProgramScore, ...]
    progress: float
    score_semantics: str = SCORE_SEMANTICS

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.progress) or not 0 <= self.progress <= 1:
            raise ValueError("reference progress must be finite and in [0, 1]")
        if self.score_semantics != SCORE_SEMANTICS:
            raise ValueError("unsupported reference score semantics")


@dataclass(frozen=True)
class GuidanceConfig:
    mode: Literal["off", "shadow", "active"] = "off"
    strength: float = 0.0
    log_weight_cap: float = 1.0
    on_unsupported: Literal["baseline", "error", "preserve_mass"] = "baseline"

    def __post_init__(self) -> None:
        if self.mode not in ("off", "shadow", "active"):
            raise ValueError(f"unknown reference-guidance mode: {self.mode!r}")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("reference strength must be finite and nonnegative")
        if (self.mode == "active") != (self.strength > 0):
            raise ValueError("only active mode takes a nonzero reference strength")
        if (
            not math.isfinite(self.log_weight_cap)
            or self.log_weight_cap <= 0
            or math.exp(-self.log_weight_cap) == 0
        ):
            raise ValueError("log_weight_cap must be positive with representable exp(-cap)")
        if self.on_unsupported not in ("baseline", "error", "preserve_mass"):
            raise ValueError(f"unknown coverage policy: {self.on_unsupported!r}")


@dataclass(frozen=True)
class GuidedPanel:
    candidate_ids: tuple[str, ...]
    baseline_probabilities: tuple[float, ...]
    probabilities: tuple[float, ...]
    config: GuidanceConfig
    reference: ScoredPanel | None
    outcome: Literal["off", "shadow", "active", "active_partial", "baseline_fallback"]

    @property
    def probabilities_changed(self) -> bool:
        return self.probabilities != self.baseline_probabilities

    def receipt(self) -> dict:
        covered = (
            []
            if self.reference is None
            else [i for i, item in enumerate(self.reference.scores) if item.status == "scored"]
        )
        return {
            "schema": "compose.reference_guidance",
            "schema_version": 2,
            **asdict(self),
            "reference_evaluated": self.reference is not None,
            "probabilities_changed": self.probabilities_changed,
            "coverage": None
            if self.reference is None
            else {
                "scored_count": len(covered),
                "candidate_count": len(self.candidate_ids),
                "scored_baseline_mass": math.fsum(self.baseline_probabilities[i] for i in covered),
                "scored_guided_mass": math.fsum(self.probabilities[i] for i in covered),
            },
        }


def guide_panel(
    candidate_ids: Sequence[str],
    baseline_probabilities: Sequence[float],
    *,
    config: GuidanceConfig | None = None,
    score: Callable[[], ScoredPanel] | None = None,
) -> GuidedPanel:
    """Reweight a frozen, eligible panel without changing its membership or order.

    Off never invokes ``score``. Shadow returns the baseline bit-for-bit. The
    scorer must be deterministic and use no controller RNG. FrozenProgramReference
    implements that boundary. Unexpected scoring exceptions propagate unchanged.
    """
    config = GuidanceConfig() if config is None else config
    ids = tuple(candidate_ids)
    probabilities = tuple(baseline_probabilities)
    if (
        not ids
        or any(not isinstance(key, str) or not key for key in ids)
        or len(set(ids)) != len(ids)
        or len(probabilities) != len(ids)
    ):
        raise ValueError("panel requires unique nonempty IDs and one probability per candidate")
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("baseline probabilities must be finite and nonnegative")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            "baseline probabilities must already sum to one; no implicit normalization"
        )
    if config.mode == "off":
        return GuidedPanel(ids, probabilities, probabilities, config, None, "off")
    if score is None:
        raise ValueError(f"{config.mode} mode requires a frozen-reference scorer")
    reference = score()
    if tuple(item.candidate_id for item in reference.scores) != ids:
        raise ValueError("reference scores must match the exact candidate IDs and order")
    if config.mode == "shadow":
        return GuidedPanel(ids, probabilities, probabilities, config, reference, "shadow")
    unsupported = [item for item in reference.scores if item.status != "scored"]
    if unsupported:
        if config.on_unsupported == "error":
            reasons = "; ".join(f"{item.candidate_id}: {item.detail}" for item in unsupported)
            raise ValueError(f"reference coverage incomplete: {reasons}")
        if config.on_unsupported == "baseline":
            return GuidedPanel(
                ids, probabilities, probabilities, config, reference, "baseline_fallback"
            )
    covered = [
        i
        for i, item in enumerate(reference.scores)
        if item.status == "scored" and probabilities[i] > 0
    ]
    if unsupported and len(covered) < 2:
        return GuidedPanel(
            ids, probabilities, probabilities, config, reference, "baseline_fallback"
        )
    outcome = "active_partial" if unsupported else "active"
    if len({reference.scores[i].value for i in covered}) == 1:
        return GuidedPanel(ids, probabilities, probabilities, config, reference, outcome)
    maximum = max(float(reference.scores[i].value) for i in covered)
    # Work with conditional probabilities to avoid squaring a small group's mass.
    # Missing scores never enter this computation. Their probabilities stay exact.
    mass = math.fsum(probabilities[i] for i in covered)
    weights = {
        i: (probabilities[i] / mass)
        * math.exp(
            max(
                -config.log_weight_cap,
                min(0.0, config.strength * (float(reference.scores[i].value) - maximum)),
            )
        )
        for i in covered
    }
    total = math.fsum(weights.values())
    guided = tuple(
        mass * (weights[i] / total) if i in weights else p for i, p in enumerate(probabilities)
    )
    if any(p > 0 and q == 0 for p, q in zip(probabilities, guided)):
        raise FloatingPointError("reference reweighting underflowed positive baseline support")
    return GuidedPanel(ids, probabilities, guided, config, reference, outcome)
