"""Inner navigation. THE COMPOSE-NATIVE ONE IS NOT DECIDED YET, ON PURPOSE.

`navigate()` is the seam the outer algorithm is built around, and it is left
open because the five-objective COMPOSE expansion policy has not been chosen.
Deciding it here, by writing the first thing that works, is exactly how a
default becomes a commitment.

What is provided now:

  RandomEditNavigator   a DEVELOPMENT STAND-IN. Legal molecular edits, but the
                        edit operators are Graph-GA's, not COMPOSE's R_theta
                        process. It exists so the OUTER algorithm -- region
                        targeting, state reuse, STOP, archive update -- can be
                        built and tested before the inner primitive is settled.
                        A number produced with this navigator is NOT a COMPOSE
                        result and must never be reported as one.

  RThetaNavigator       the real one, blocked on runtime provenance (see below).

WHY THE R_THETA NAVIGATOR IS NOT HERE YET, AND WHAT THE BLOCKER ACTUALLY IS
----------------------------------------------------------------------------
The local Active8 copy was believed to differ from the volume's. IT DOES NOT.
`source_index_sha256` is computed over a body whose first field is
`active8_run_root` -- the absolute filesystem path
(`editing_v2_process_v2_gate_zero.py:593`). Recomputing the local index with
that ONE field set to the container path the decision was made under
(`/artifacts/editing_v2/process_v2_active8/8ecc0e5e...`) reproduces the Gate-0
PASS's `b5d042a0...` exactly. Every content field -- completion sha, sentinel
sha, contracts binding, all 270 shard entries, eligible task identities, role
census, sealed role metadata -- already agreed.

So the corpus is content-identical and no 751 MB fetch is needed. What the check
actually pins is the MOUNT POINT, in two places: the hashed index body and
`reduce_gate_zero`'s internal re-derivation. The local corpus lives at
`local_runtime/active8/...` and therefore cannot reproduce a hash of
`/artifacts/...`; it fails on location while the content is right.

That is a real property of the guard, not something to route around quietly.
Where the volume is mounted at its own path the chain validates today with no
change at all. Making it validate off-volume would mean separating IDENTITY
(which mount the decision named) from LOCATION (where the bytes are read) --
a change to shared lineage code, which belongs to whoever owns that guard and
not to this module.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compose_v4.policy.task3.interfaces import (
    NavigationBudget,
    Region,
    Trajectory,
)

_GB_GA = Path(__file__).resolve().parents[4] / "baselines" / "graph_ga" / "upstream"


class NavigatorUnavailable(RuntimeError):
    """Raised instead of silently substituting a different molecular process."""


@dataclass
class RandomEditNavigator:
    """DEVELOPMENT STAND-IN. Legal edits, but NOT the COMPOSE process.

    A short random walk of Graph-GA mutations from the start state, keeping
    every realized molecule and stopping at the end of the walk. It cannot
    steer: nothing in it consults the region, and that is exactly what makes it
    a stand-in rather than a policy.

    Note that upstream `mutate` returns the molecule UNCHANGED with probability
    1 - mutation_rate, so a walk of n steps makes about n * mutation_rate real
    edits.
    """

    name: str = "random-edit-standin"
    mutation_rate: float = 0.5
    #: Set once from the starting population, exactly as the Graph-GA baseline
    #: does; without it upstream's `mol_OK` rejects every candidate in silence.
    average_size: float = 24.0
    size_stdev: float = 4.0

    def _operators(self):
        if str(_GB_GA) not in sys.path:
            sys.path.insert(0, str(_GB_GA))
        import crossover as co
        import mutate as mu
        co.average_size = self.average_size
        co.size_stdev = self.size_stdev
        return co, mu

    def navigate(self, start: str, region: Region, budget: NavigationBudget,
                 rng: np.random.Generator, *,
                 evidence: object | None = None) -> Trajectory:
        # `evidence` is accepted and IGNORED here, which is the point: this
        # stand-in cannot steer, and the run that proved steering matters is
        # the one where it did not.
        from rdkit import Chem

        _, mu = self._operators()
        random.seed(int(rng.integers(0, 2 ** 31 - 1)))
        np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))

        deadline = time.time() + budget.wall_seconds
        current = Chem.MolFromSmiles(start)
        if current is None:
            return Trajectory(states=[], stop=0, region=region,
                              diagnostics={"error": "unparseable start"})
        states = [start]
        rejected = 0
        for _ in range(budget.steps):
            if time.time() > deadline:
                break
            candidate = mu.mutate(current, self.mutation_rate)
            if candidate is None:
                rejected += 1
                continue
            try:
                smiles = Chem.MolToSmiles(candidate)
            except Exception:  # noqa: BLE001
                rejected += 1
                continue
            if not smiles:
                rejected += 1
                continue
            states.append(smiles)
            current = candidate

        stop = self._stop(states, region)
        return Trajectory(states=states, stop=stop, region=region,
                          diagnostics={"rejected": rejected,
                                       "length": len(states)})

    def _stop(self, states: list[str], region: Region) -> int:
        """Stop at the end of the walk. Ranking states is the REAL navigator's job.

        It is tempting to pick the intermediate with the best QED, since RDKit
        gives it away for free. That temptation is the whole problem: QED IS one
        of the five objectives, and screening candidates on an objective without
        charging for it is precisely what makes the released benchmark's
        effective budget several times its nominal one. Doing it with one
        objective instead of five would be the same mistake, smaller.

        So no objective touches this decision. A genuine prospective STOP needs
        an estimate of whether the region is being approached, which is a
        five-objective value model this task has to build for itself -- and that
        belongs to the COMPOSE navigator, not to a stand-in.
        """

        return max(0, len(states) - 1)


@dataclass
class RThetaNavigator:
    """The COMPOSE-native navigator. Not implemented; blocked, not forgotten."""

    name: str = "r-theta"

    def navigate(self, start: str, region: Region, budget: NavigationBudget,
                 rng: np.random.Generator, *,
                 evidence: object | None = None) -> Trajectory:
        raise NavigatorUnavailable(
            "the frozen R_theta runtime cannot be built in this checkout. The "
            "local Active8 corpus is CONTENT-IDENTICAL to the authoritative one "
            "-- its index reproduces the Gate-0 PASS's b5d042a0... exactly once "
            "active8_run_root is set to the container path -- but the guard "
            "hashes that path, so an off-volume mirror cannot satisfy it. Run "
            "where the volume is mounted at its own path, where this works "
            "unchanged. Do not bypass the check.")
