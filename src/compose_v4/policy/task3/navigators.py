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

WHY THE R_THETA NAVIGATOR IS NOT HERE YET
------------------------------------------
Driving the frozen R_theta locally needs the volume-authoritative Active8
stream. This checkout has a copy that differs: `process_identity_sha256`,
`contracts_binding_sha256` and `gate_zero_structural_contract_sha256` all MATCH
the Gate-0 PASS, and only `source_index_sha256` differs (decision b5d042a0...
vs local ed51268a..., same 270 eligible shards). So the model and process
identity are provably the ones the decision names; what cannot be certified
locally is that the corpus STREAM is the volume's.

That guard is doing its job and is not something to route around quietly. The
options are to fetch the authoritative Active8, or to run where the volume is
mounted -- either way a decision, not a workaround.
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
            "the frozen R_theta runtime cannot be built in this checkout: the "
            "local Active8 copy's source_index_sha256 does not match the Gate-0 "
            "PASS (process identity, contracts binding and structural contract "
            "all DO match). Fetch the volume-authoritative Active8, or run where "
            "the volume is mounted. Do not bypass the check.")
