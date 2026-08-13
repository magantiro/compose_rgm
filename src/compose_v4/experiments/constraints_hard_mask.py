"""Hard structural-constraint mask over the exact canonical successor fiber.

`DESIGN_ONLY`. This module implements the injection point and the census
statistics only. It contains no controller, runs nothing, and does not touch
`R_theta`.

Why a filter and not `RewriteSystem(constraints=)`
--------------------------------------------------
`RewriteSystem.apply` **raises** `InvalidRewrite` on a failing constraint, and
`production_successor_kernel.canonical_successor_result` wraps `apply` in a
`try/except` that converts any executor exception into a fatal
`ProductionSuccessorKernelError` ("the model's legal mask admitted an action
rejected by the production executor"). Installing a semantic constraint there
would crash the kernel on the first core-violating mark rather than filter it,
and would blind a guard that exists to catch genuine model/executor mask
disagreement.

So the mask filters the *enumerated* successor list and renormalizes:

    F(x)   = batch.successors
    F_C(x) = [s for s in F(x) if predicate(s.key)]
    renormalize probability over F_C(x)

Renormalization is mandatory: `validate_successor_batch` rejects a batch whose
successor probabilities do not sum to 1.

This leaves the kernel, the marked law and `R_theta` completely untouched, which
is exactly what the paper claims. The resulting arm is a `SUPPORT_ABLATION`, not
`LAW_ONLY`, and must be registered as one.

See `docs/workstreams/constraints-hard/CONSTRAINT_SEMANTICS.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "MaskedFiber",
    "apply_hard_mask",
    "retention_statistics",
]

Predicate = Callable[[str], bool]


@dataclass(frozen=True)
class MaskedFiber:
    """`F_C(x)` with the bookkeeping the census needs.

    `retained_probability` is renormalized to sum to 1 over the retained
    successors, matching the kernel's normalization convention (probabilities
    are over productive canonical successors CONDITIONED ON TAKING A JUMP).
    """

    source_key: str
    retained_keys: tuple[str, ...]
    retained_probability: tuple[float, ...]
    fiber_width: int
    """|F(x)| - the unconstrained legal fiber width."""
    retained_width: int
    """|F_C(x)| - the hard-mask retained fiber width."""
    removed_reference_mass: float
    """Probability mass under the UNMASKED law that the mask removed."""

    @property
    def is_mask_empty(self) -> bool:
        """The mask left no admissible successor. A terminal state, not an error."""
        return self.retained_width == 0

    @property
    def support_retention(self) -> float:
        """`|F_C(x)| / |F(x)|`.

        NOT sign-fixed: ranges over [0, 1]. This is the statistic that exposes a
        mask which strangles the fiber, and it is the reason the hard arm's
        100% constraint satisfaction is not the measurement.
        """
        if self.fiber_width == 0:
            return 0.0
        return self.retained_width / self.fiber_width


def apply_hard_mask(
    source_key: str,
    successor_keys: tuple[str, ...],
    successor_probability: tuple[float, ...],
    predicate: Predicate,
) -> MaskedFiber:
    """Restrict the canonical successor fiber to `C`, then renormalize.

    `successor_keys` / `successor_probability` are `batch.keys` and the matching
    `CanonicalSuccessor.probability` values. They are passed as plain tuples so
    this function - and its adversarial fixtures - need neither the kernel nor
    the checkpoint.
    """
    if len(successor_keys) != len(successor_probability):
        raise ValueError("successor keys and probabilities have different lengths")
    if len(set(successor_keys)) != len(successor_keys):
        raise ValueError(
            "successor keys are not distinct -- aliases were not merged upstream"
        )
    if source_key in successor_keys:
        raise ValueError(
            f"a successor carries the source key {source_key!r}; a self-transition "
            "is not a molecular jump"
        )

    kept: list[tuple[str, float]] = [
        (key, float(probability))
        for key, probability in zip(successor_keys, successor_probability, strict=True)
        if predicate(key)
    ]
    retained_mass = sum(probability for _, probability in kept)
    total_mass = sum(float(p) for p in successor_probability)
    removed_reference_mass = total_mass - retained_mass

    if kept and retained_mass > 0.0:
        renormalized = tuple(probability / retained_mass for _, probability in kept)
    else:
        # Mask-empty, or every retained successor carried zero mass. Either way
        # there is nothing to normalize; the state is terminal under F_C.
        kept = []
        renormalized = ()

    return MaskedFiber(
        source_key=source_key,
        retained_keys=tuple(key for key, _ in kept),
        retained_probability=renormalized,
        fiber_width=len(successor_keys),
        retained_width=len(kept),
        removed_reference_mass=removed_reference_mass,
    )


def retention_statistics(fibers: list[MaskedFiber]) -> dict[str, float | int]:
    """Census statistics over a set of masked fibers.

    Every quantity here has a genuine falsifying range. None of them is the
    hard arm's constraint-satisfaction rate, which is 1 by construction and is
    recorded as a construction check without a denominator.
    """
    if not fibers:
        return {"states": 0}
    retentions = sorted(fiber.support_retention for fiber in fibers)
    mid = len(retentions) // 2
    median = (
        retentions[mid]
        if len(retentions) % 2
        else (retentions[mid - 1] + retentions[mid]) / 2
    )
    empty = sum(1 for fiber in fibers if fiber.is_mask_empty)
    return {
        "states": len(fibers),
        "support_retention_median": median,
        "support_retention_mean": sum(retentions) / len(retentions),
        "support_retention_min": retentions[0],
        "support_retention_max": retentions[-1],
        "mask_empty_states": empty,
        "mask_empty_rate": empty / len(fibers),
        "removed_reference_mass_mean": sum(
            fiber.removed_reference_mass for fiber in fibers
        )
        / len(fibers),
    }
