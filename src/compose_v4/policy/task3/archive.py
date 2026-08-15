"""The evaluated Pareto archive, and where its hypervolume is missing.

WHY A PROBE SET RATHER THAN REPEATED HYPERVOLUME CALLS
------------------------------------------------------
The reported metric integrates a fixed Sobol sequence over the unit box. The
policy needs the same question answered many times per run -- "how much
hypervolume would reaching HERE add?" -- and recomputing a 2^20-sample
hypervolume for every candidate region would cost more than the search.

So the archive keeps a smaller, fixed probe set from the SAME sequence and a
boolean of which probes the front already dominates. Then:

    gain(z) = |probes z dominates that the archive does not| / |probes|

which is an unbiased, deterministic, and very cheap estimate of the marginal
hypervolume of achieving z -- in the metric's own units, not a proxy of our
choosing. Adding a molecule updates the mask with one vectorised OR.

The estimate is coarser than the reported metric by construction. That is
correct: it is used to CHOOSE where to look, and a choice does not need the
precision of a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compose_v4.benchmark.molleo_task3 import _pareto_mask, hypervolume_qmc

#: Probes for the policy's internal accounting. 2^14 resolves a gain of ~6e-5,
#: far finer than any difference the policy acts on.
PROBE_LOG2 = 14


def probe_points(log2_samples: int = PROBE_LOG2, dimensions: int = 5) -> np.ndarray:
    """The same unscrambled Sobol prefix the metric integrates."""

    from scipy.stats import qmc

    engine = qmc.Sobol(d=dimensions, scramble=False)
    return engine.random(1 << log2_samples)


@dataclass
class ParetoArchive:
    """Everything evaluated, the front it implies, and its coverage mask."""

    probes: np.ndarray = field(default_factory=probe_points)
    values: dict[str, tuple[float, ...]] = field(default_factory=dict)
    _covered: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._covered is None:
            self._covered = np.zeros(len(self.probes), dtype=bool)

    def __len__(self) -> int:
        return len(self.values)

    def add(self, smiles: str, values: tuple[float, ...]) -> bool:
        """Record an evaluated molecule. Returns whether it was new."""

        if smiles in self.values:
            return False
        self.values[smiles] = tuple(float(v) for v in values)
        point = np.asarray(values, dtype=float)
        self._covered |= np.all(self.probes <= point, axis=1)
        return True

    def add_many(self, items: dict[str, tuple[float, ...]]) -> int:
        return sum(1 for smiles, values in items.items()
                   if self.add(smiles, values))

    # ---- the front -------------------------------------------------------

    def points(self) -> np.ndarray:
        if not self.values:
            return np.empty((0, 5))
        return np.asarray(list(self.values.values()), dtype=float)

    def front(self) -> list[tuple[str, tuple[float, ...]]]:
        """Non-dominated members, best-first by summed objectives."""

        if not self.values:
            return []
        keys = list(self.values)
        points = self.points()
        mask = _pareto_mask(points)
        members = [(keys[i], self.values[keys[i]]) for i in np.flatnonzero(mask)]
        members.sort(key=lambda item: sum(item[1]), reverse=True)
        return members

    # ---- coverage --------------------------------------------------------

    @property
    def covered_fraction(self) -> float:
        """Probe-set estimate of the current hypervolume."""

        return float(self._covered.mean())

    def gain_of(self, target) -> float:
        """Probe-set hypervolume that reaching `target` would ADD."""

        point = np.asarray(target, dtype=float)
        dominated = np.all(self.probes <= point, axis=1)
        return float((dominated & ~self._covered).mean())

    def gains_of(self, targets: np.ndarray) -> np.ndarray:
        """Vectorised `gain_of` for a batch of candidate targets."""

        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        out = np.empty(len(targets), dtype=float)
        # Chunked: the intermediate is (chunk, n_probes, 5) booleans, which is
        # 5 MB per 64 targets and would be gigabytes if a caller passed
        # thousands at once.
        for start in range(0, len(targets), 64):
            block = targets[start:start + 64]
            dominated = np.all(self.probes[None, :, :] <= block[:, None, :], axis=2)
            out[start:start + 64] = (dominated & ~self._covered[None, :]).mean(axis=1)
        return out

    def reported_hypervolume(self, *, log2_samples: int = 20) -> float:
        """The metric itself -- for reporting, not for steering."""

        return hypervolume_qmc(self.points(), log2_samples=log2_samples)

    # ---- persistence -----------------------------------------------------

    def as_dict(self) -> dict:
        return {"values": {k: list(v) for k, v in self.values.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> ParetoArchive:
        archive = cls()
        for smiles, values in data.get("values", {}).items():
            archive.add(smiles, tuple(values))
        return archive
