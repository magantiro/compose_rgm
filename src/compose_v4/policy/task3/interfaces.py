"""The seams of the Task 3 policy. The inner navigator is deliberately unfrozen.

    archive -> select_target_region -> select_start_state -> navigate
            -> select_for_oracle -> update_archive -> repeat

`navigate()` is a Protocol, not an implementation, because the COMPOSE-native
expansion policy for five objectives is not decided yet and must not be decided
by accident.  The QED lane is qualifying a twisted-SMC controller against a
QED/similarity region model; if that substrate matures it may be reusable here,
and if it does not, this task can use a different COMPOSE-native expansion.  The
outer algorithm is independent of that choice, so it is built now and the seam is
kept open.

THE HYPOTHESIS THIS ARCHITECTURE EXISTS TO TEST
-----------------------------------------------
A fixed scalarization is not blind to the objectives it sums -- it rewards high
JNK3 and low GSK3B alike, because both are terms in it. Its limitation is
narrower and more interesting: it COMPRESSES THE TRADEOFF INTO ONE NUMBER, and
so has no mechanism to notice that a valuable part of the current Pareto front
is MISSING and deliberately redirect search there. Adaptive region targeting has
exactly that mechanism.

So the causal comparison is between two ways of specifying purpose, holding the
surrogate and the evaluated data fixed:

    A   fixed scalarization steering
    B   adaptive region/archive steering

Anything else that differs between them is a confound, not a result.

THE ONE STRUCTURAL GUARANTEE, STATED EXACTLY
--------------------------------------------
A navigator may see EVERYTHING WE HAVE ALREADY PAID FOR -- the archive of
evaluated molecules with their objective vectors.  It is never handed the run,
the meter, or the evaluator.  So it can fit a surrogate on purchased data, which
is what every method does, and it cannot obtain the objectives of a NEW molecule
without that molecule being charged.

That boundary is the whole accounting difference with the released benchmark,
which screens its entire offspring population through unmetered evaluators and
charges only the survivors.  Here the leak is unreachable rather than merely
avoided.

WHY `evidence` EXISTS AT ALL -- IT WAS MEASURED, NOT ASSUMED
------------------------------------------------------------
A region is a point in OBJECTIVE space.  A navigator with no way to estimate
objectives cannot tell whether an edit moved toward it or away, so it cannot
steer, and region targeting collapses into "choose a good starting molecule".
That is not a hypothetical: a full 10,000-call development run with a
steering-blind stand-in navigator reached JNK3 0.26, against 0.68 for uniform
random sampling on the same seed, because it explored hard in no particular
direction.

So the seam carries the paid-for archive, and the real navigator's first job is
a five-objective notion of future value built from it.  That model is this
task's own -- the QED lane's `h_phi` is a QED/similarity region model and does
not answer this question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

#: Objective vectors are always normalised, higher-is-better, in [0, 1]^5.
N_OBJECTIVES = 5


@dataclass(frozen=True)
class Region:
    """A valuable missing piece of objective space, as an aspiration point.

    `target` is a point in [0, 1]^5 that no archive member currently dominates.
    `gain` is what reaching it would add to the hypervolume, measured on the
    same fixed probe set the metric uses -- so "valuable" means valuable in the
    benchmark's own units rather than in a proxy of our choosing.

    `origin` records which realized molecule the aspiration was built from,
    which is what makes the region reachable rather than merely desirable: the
    corner (1,1,1,1,1) always has the highest conceivable gain and is never a
    useful thing to aim at.
    """

    target: tuple[float, ...]
    gain: float
    origin: str | None = None
    note: str = ""

    def as_array(self) -> np.ndarray:
        return np.asarray(self.target, dtype=float)


@dataclass
class Trajectory:
    """A realized molecular route. Every state is a complete valid molecule.

    `states[0]` is the start. `stop` is the index the navigator PROSPECTIVELY
    chose to stop at -- a COMPOSE affordance, and the reason intermediates are
    kept: they are legitimate returnable states, not discarded scaffolding.
    """

    states: list[str] = field(default_factory=list)
    stop: int = 0
    region: Region | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def stopped_at(self) -> str | None:
        return self.states[self.stop] if self.states else None


@dataclass(frozen=True)
class NavigationBudget:
    """What a navigator may spend. NONE of it is benchmark oracle budget.

    `steps` bounds edits along one route; `expansions` bounds how much of the
    legal successor structure may be examined. These are internal resources and
    are reported separately -- the benchmark constrains oracle calls, not search.
    """

    steps: int = 8
    expansions: int = 64
    wall_seconds: float = 30.0


class Navigator(Protocol):
    """Expand from a realized state toward a region. PLUGGABLE ON PURPOSE.

    `evidence` is the archive of molecules already paid for. Read it freely;
    there is no way to add to it from here.
    """

    name: str

    def navigate(self, start: str, region: Region,
                 budget: NavigationBudget,
                 rng: np.random.Generator,
                 *, evidence: object | None = None) -> Trajectory:
        ...


class RegionSelector(Protocol):
    def select(self, archive, rng: np.random.Generator) -> Region | None:
        ...


class StartSelector(Protocol):
    def select(self, archive, region: Region,
               rng: np.random.Generator) -> str | None:
        ...


class CandidateSelector(Protocol):
    """Chooses which realized states are worth a benchmark oracle call."""

    def select(self, trajectory: Trajectory, archive,
               limit: int) -> Sequence[str]:
        ...
