"""Controller-agnostic mechanics for the pathwise and hard-support experiments.

WHY THIS MODULE EXISTS, AND WHY IT TAKES NO CONTROLLER
------------------------------------------------------
Both experiments ask the same structural question -- *does enforcing a
constraint on the SUPPORT of the controlled process beat enforcing it after the
fact?* -- and neither question is about which controller is used. Pathwise
enforces a predicate at every committed molecular state; hard support removes
infeasible transitions from the support entirely. In both, the comparison arm
applies the identical predicate only at the endpoint.

So the arms are a property of the SUPPORT, not of the controller, and this
module defines them against an abstract `TrajectoryController`. The claim-bearing
controller (region-`h_phi`) is injected later, once its weights exist and the
QED lane has finished the preregistered ladder. Nothing here freezes or presumes
a controller, and nothing here may be run to produce a claim-bearing result.

ONE ENGINE, TWO PREREGISTERED EXPERIMENTS -- DO NOT MERGE THEM
---------------------------------------------------------------
Sharing this module is an implementation decision, NOT a scientific one. Both
experiments are built on the same primitive, the restricted action set

    A_C(x) = { a in A(x) : C(T_a x) = 1 }

but they ask different questions and are reported as separate rows of the
authoritative experiment map:

  * HARD SUPPORT -- hard support vs fixed soft guidance vs post-hoc filtering.
    Estimand: does making feasibility STRUCTURAL improve useful constrained
    optimization?

  * PATHWISE -- identical endpoint requirement throughout; enforce C only at the
    end versus C(X_t) at EVERY committed state. Estimand: does ROUTE-LEVEL
    enforcement matter?

Anyone tempted to collapse these into a single result because they share code
should stop: the shared code is `A_C`, and that is all they share.

WHAT IS DELIBERATELY ABSENT
---------------------------
No `h_phi`. No five-objective policy. No Modal app. This is mechanics plus the
estimators the two preregistrations already specify, so that when the controller
does arrive the experiments are a wiring step rather than a design step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

__all__ = [
    "Enforcement",
    "StatePredicate",
    "TrajectoryController",
    "ConstrainedTrajectory",
    "run_constrained_trajectory",
    "p_hidden",
    "source_clustered_bootstrap",
    "noninferiority_verdict",
]


class Enforcement(Enum):
    """WHERE the predicate is enforced. This is the experimental arm."""

    #: Predicate enforced at EVERY committed state: violating successors are
    #: absent from the controlled support, so the trajectory can never pass
    #: through a forbidden molecule. COMPOSE-native arm for both experiments.
    PATHWISE = "pathwise"

    #: Predicate applied only to the final molecule. Intermediates are
    #: unconstrained and may violate freely. The matched restriction arm.
    ENDPOINT_ONLY = "endpoint_only"

    #: No enforcement anywhere. Reference arm for measuring how often
    #: unconstrained control wanders out of the corridor on its own.
    NONE = "none"


class StatePredicate(Protocol):
    """`True` iff the molecule satisfies the constraint (in corridor / feasible)."""

    def __call__(self, smiles: str) -> bool: ...


class TrajectoryController(Protocol):
    """Any controller. The experiments never depend on which one this is.

    Returns the next SMILES, or `None` when it cannot or chooses not to
    continue (empty fiber, or a native STOP decision). `allowed` is the support
    restriction for this step: when not `None`, the controller MUST return a
    member of it or `None`. Passing the restriction into the controller (rather
    than filtering afterwards) is what makes PATHWISE a genuine support
    restriction rather than rejection sampling on top of an unrestricted law.
    """

    def propose(
        self,
        state: str,
        source: str,
        budget_remaining: int,
        rng: np.random.Generator,
        allowed: Callable[[str], bool] | None = None,
    ) -> str | None: ...


@dataclass(frozen=True)
class ConstrainedTrajectory:
    """One realized trajectory under one enforcement arm."""

    source: str
    path: tuple[str, ...]
    enforcement: Enforcement
    #: True if the controller stopped early (native STOP or exhausted fiber).
    halted_early: bool
    #: Indices of committed states violating the predicate, source excluded.
    violations: tuple[int, ...] = field(default=())

    @property
    def terminal(self) -> str:
        return self.path[-1]

    @property
    def edits(self) -> int:
        return len(self.path) - 1

    @property
    def visited_forbidden(self) -> bool:
        """Did the realized path ever pass through a forbidden molecule?"""
        return bool(self.violations)


def run_constrained_trajectory(
    controller: TrajectoryController,
    predicate: StatePredicate,
    enforcement: Enforcement,
    source: str,
    horizon: int,
    rng: np.random.Generator,
) -> ConstrainedTrajectory:
    """Roll one trajectory under one arm. Controller-agnostic by construction.

    Under PATHWISE the predicate is handed to the controller as a support
    restriction, so forbidden molecules are never committed. Under
    ENDPOINT_ONLY and NONE the controller runs unrestricted and violations are
    merely RECORDED -- which is exactly the quantity P1 estimates.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    restrict = predicate if enforcement is Enforcement.PATHWISE else None

    path = [source]
    halted = False
    for step in range(horizon):
        nxt = controller.propose(
            path[-1], source, horizon - step - 1, rng, allowed=restrict)
        if nxt is None:
            halted = True
            break
        if restrict is not None and not restrict(nxt):
            # A conforming controller cannot do this. Fail loudly rather than
            # silently downgrading a pathwise run into an endpoint-only one.
            raise RuntimeError(
                f"controller returned {nxt!r}, which violates the support "
                f"restriction it was given; PATHWISE would be unenforced")
        path.append(nxt)

    violations = tuple(
        i for i, s in enumerate(path) if i > 0 and not predicate(s))
    return ConstrainedTrajectory(
        source=source, path=tuple(path), enforcement=enforcement,
        halted_early=halted, violations=violations)


def p_hidden(trajectories: Sequence[ConstrainedTrajectory]) -> float:
    """P1's estimand: terminal satisfies the predicate, but the path did not.

        p_hidden = P( x_H in C  AND  exists t < H : x_t not in C )

    Undefined on an empty sequence, which is an error rather than 0.0 -- a
    silent zero here would read as "no hidden-path behaviour".
    """
    if not trajectories:
        raise ValueError("p_hidden is undefined on zero trajectories")
    hidden = sum(
        1 for t in trajectories
        # Terminal conforms, yet some EARLIER committed state did not.
        if not any(v == t.edits for v in t.violations) and
        any(v < t.edits for v in t.violations)
    )
    return hidden / len(trajectories)


def source_clustered_bootstrap(
    values_by_source: dict[str, Sequence[float]],
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap resampling SOURCES, not observations. Returns (point, lo, hi).

    The source is the independent unit in both preregistrations. Resampling
    individual trajectories would treat repeated draws from one source as
    independent evidence and understate the interval.
    """
    srcs = sorted(values_by_source)
    if not srcs:
        raise ValueError("no sources to bootstrap")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    flat = np.concatenate([np.asarray(values_by_source[s], float) for s in srcs])
    point = statistic(flat)

    draws = np.empty(resamples, float)
    for r in range(resamples):
        pick = rng.integers(0, len(srcs), size=len(srcs))
        draws[r] = statistic(np.concatenate(
            [np.asarray(values_by_source[srcs[i]], float) for i in pick]))
    tail = (1.0 - confidence) / 2.0
    return point, float(np.quantile(draws, tail)), float(
        np.quantile(draws, 1.0 - tail))


def noninferiority_verdict(ci_lower: float, delta: float) -> bool:
    """P2 passes iff the CI lower bound on mean delta_V exceeds -delta.

    `delta` is a TOLERANCE on a measurement scale, never a biological margin.
    The preregistration requires the continuous estimate and its interval to be
    reported alongside this verdict, never replaced by it, and requires the
    stricter sensitivity to be reported whether or not it agrees.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    return ci_lower > -delta
