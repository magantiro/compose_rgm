"""The outer algorithm: archive -> region -> start -> navigate -> STOP -> charge.

    while budget remains:
        region     = select_target_region(archive)      # what is missing
        start      = select_start_state(archive, region) # what to attack it from
        trajectory = navigate(start, region, budget)     # PLUGGABLE
        candidate  = select_for_oracle(trajectory)       # what is worth a call
        update_archive(candidate, run.evaluate([...]))   # the ONE metered step

Every part of this is independent of how `navigate` works inside, which is why
it is built now while that choice is still open.

TWO PROPERTIES THIS LOOP HAS BY CONSTRUCTION
--------------------------------------------
1. `run.evaluate` is called in exactly one place. Nothing else in the policy can
   reach the objectives, so the number of molecules that receive the objective
   vector equals the number charged. That is the difference between our
   accounting and the released benchmark's, and it is structural rather than
   a matter of discipline.

2. The archive is a pure function of the evaluated set, so a resumed run
   rebuilds it from the durable ledger instead of from a checkpointed copy. The
   checkpoint therefore stays small (an iteration counter), and there is no way
   for a checkpointed archive to disagree with the ledger it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compose_v4.benchmark.oracles.task3 import navigation_lockout
from compose_v4.policy.task3.archive import ParetoArchive
from compose_v4.policy.task3.interfaces import (
    CandidateSelector,
    NavigationBudget,
    Navigator,
    RegionSelector,
    StartSelector,
    Trajectory,
)
from compose_v4.policy.task3.regions import (
    MarginalGainRegions,
    RegionOriginStart,
)


@dataclass
class StopStatePreferNovel:
    """Charge the STOP state, or the nearest realized state not yet paid for.

    Walking back along the trajectory costs nothing and avoids spending an
    iteration on a molecule whose objectives are already known. It never
    consults the objectives to choose -- only whether a molecule has already
    been evaluated, which is bookkeeping, not screening.
    """

    name: str = "stop-prefer-novel"

    def select(self, trajectory: Trajectory, archive: ParetoArchive,
               limit: int) -> list[str]:
        if not trajectory.states or limit <= 0:
            return []
        order = [trajectory.stop] + [i for i in range(len(trajectory.states) - 1,
                                                      -1, -1)
                                     if i != trajectory.stop]
        chosen: list[str] = []
        for index in order:
            smiles = trajectory.states[index]
            if smiles in archive.values or smiles in chosen:
                continue
            chosen.append(smiles)
            if len(chosen) >= limit:
                break
        if not chosen:
            chosen = [trajectory.states[trajectory.stop]]
        return chosen


@dataclass
class AdaptiveParetoNavigatorPolicy:
    """The COMPOSE-native outer policy. The inner navigator is injected."""

    navigator: Navigator
    regions: RegionSelector = field(default_factory=MarginalGainRegions)
    starts: StartSelector = field(default_factory=RegionOriginStart)
    candidates: CandidateSelector = field(default_factory=StopStatePreferNovel)
    navigation_budget: NavigationBudget = field(default_factory=NavigationBudget)
    candidates_per_iteration: int = 4
    checkpoint_every: int = 25
    #: Consecutive iterations that charge NOTHING before the run gives up.
    #: A navigator that keeps re-proposing molecules the archive already holds
    #: spends no budget, so `remaining` never falls and the loop would otherwise
    #: never terminate. Found by a test, not by reasoning.
    stall_limit: int = 50

    @property
    def name(self) -> str:
        return (f"adaptive-pareto[{self.regions.name}/{self.starts.name}/"
                f"{self.navigator.name}]")

    def run(self, run, initial: list[str]) -> dict:
        """Drive `run` to exhaustion. Returns diagnostics, not a result."""

        archive = ParetoArchive()
        # The ledger already holds everything ever charged, so a resumed run
        # reconstitutes the archive from it rather than from a saved copy.
        archive.add_many(run.meter.evaluated())

        if not archive.values:
            run.evaluate(initial)
            archive.add_many(run.meter.evaluated())
            run.step = 1
            run.archive = [smiles for smiles, _ in archive.front()]
            run.checkpoint(policy_state={"iteration": run.step,
                                         "policy": self.name})

        stats = {"iterations": 0, "no_region": 0, "no_start": 0,
                 "empty_trajectory": 0, "charged": 0, "repeats": 0,
                 "stalled_out": False, "region_gain": []}
        stalled = 0
        while run.remaining > 0:
            if stalled >= self.stall_limit:
                stats["stalled_out"] = True
                break
            stats["iterations"] += 1
            region = self.regions.select(archive, run.rng)
            if region is None:
                stats["no_region"] += 1
                break                      # nothing left worth aiming at
            start = self.starts.select(archive, region, run.rng)
            if start is None:
                stats["no_start"] += 1
                break
            # The archive is everything already paid for. Handing it over lets
            # a navigator learn to steer; it is still the case that nothing here
            # can evaluate a NEW molecule without charging.
            with navigation_lockout():
                trajectory = self.navigator.navigate(
                    start, region, self.navigation_budget, run.rng,
                    evidence=archive)
            if not trajectory.states:
                stats["empty_trajectory"] += 1
                stalled += 1
                continue
            limit = min(self.candidates_per_iteration, run.remaining)
            chosen = list(self.candidates.select(trajectory, archive, limit))
            chosen = run.affordable(chosen)
            if not chosen:
                stalled += 1
                continue
            before = run.spent
            values = run.evaluate(chosen)
            charged = run.spent - before
            stalled = 0 if charged else stalled + 1
            stats["charged"] += charged
            stats["repeats"] += len(chosen) - charged
            stats["region_gain"].append(region.gain)
            for smiles, value in zip(chosen, values):
                archive.add(smiles, value)

            run.step += 1
            if run.step % self.checkpoint_every == 0:
                run.archive = [smiles for smiles, _ in archive.front()]
                run.checkpoint(policy_state={"iteration": run.step,
                                             "policy": self.name})

        run.archive = [smiles for smiles, _ in archive.front()]
        run.checkpoint(policy_state={"iteration": run.step, "policy": self.name,
                                     "finished": True})
        stats["front_size"] = len(run.archive)
        stats["mean_region_gain"] = (float(np.mean(stats["region_gain"]))
                                     if stats["region_gain"] else 0.0)
        stats.pop("region_gain")
        stats["probe_coverage"] = archive.covered_fraction
        return stats
