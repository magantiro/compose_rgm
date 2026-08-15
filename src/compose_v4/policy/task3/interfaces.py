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

THE ONE STRUCTURAL GUARANTEE
----------------------------
A navigator is handed a start state, a target region, and a step budget.  IT IS
NEVER HANDED THE RUN, THE METER, OR THE OBJECTIVES.  The only path from a
molecule to its five objectives is `select_for_oracle` followed by
`update_archive`, which charges.

This is not a stylistic preference.  The released MOLLEO benchmark evaluates its
whole offspring population through unmetered evaluators and charges only the
survivors, which is why its effective budget is several times its nominal one.
The way to be sure we never do the same thing is to make it unreachable: a
navigator that cannot see the objectives cannot screen against them.
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
    """Expand from a realized state toward a region. PLUGGABLE ON PURPOSE."""

    name: str

    def navigate(self, start: str, region: Region,
                 budget: NavigationBudget,
                 rng: np.random.Generator) -> Trajectory:
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
