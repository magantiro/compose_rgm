"""V_z(x, M): is spending a rewrite on THIS region likely to help THIS task?

    Q(M | x, z)  ∝  mu_exec(M | x) · exp( V_z(x, M) / tau )

mu_exec already answers "is this region executable and economical". It is
task-blind by construction, so a population using it alone has no reason to pay
for an expensive global rewrite when a cheap local one is easier to execute --
which is exactly what the QED pilot showed, with six of eight accepted rewrites
landing in the smallest scope band. V_z supplies the missing term.

Deliberately small. The estimator is shrunk group means over region descriptors,
not a network:

  * region-level credit is what we are testing for; a model with enough capacity
    to fit anything would not tell us whether that credit is learnable.
  * the observations arrive one per search iteration, so early on there are tens
    of them, not thousands. Shrinkage toward the global mean is what keeps a
    cell with two observations from dominating the allocator.

V_z estimates E[delta U | accepted], NOT expected improvement per proposal. The
probability of getting a usable rewrite at all already lives in mu_exec's
feasibility term, and multiplying it in here would double-count it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _band(released_fraction: float) -> int:
    return min(int(float(released_fraction) * 5), 4)


@dataclass
class TaskValue:
    """Shrunk mean task delta per (interface, scope band).

    `strength` is in units of observations: a cell is trusted in proportion to
    how many it has, and falls back to the global mean otherwise.
    """

    strength: float = 6.0
    _cells: dict = field(default_factory=dict)
    _n: int = 0
    _sum: float = 0.0

    def observe(self, interface: str, released_fraction: float, delta: float,
                accepted: bool = True) -> None:
        """Record one (x, M) -> delta U outcome.

        Only accepted rewrites carry task information: a failed proposal says
        the region was hard to execute, which is mu_exec's business, not this
        estimator's.
        """
        if not accepted:
            return
        k = (str(interface), _band(released_fraction))
        c = self._cells.setdefault(k, [0, 0.0])
        c[0] += 1
        c[1] += float(delta)
        self._n += 1
        self._sum += float(delta)

    @property
    def global_mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    def value(self, interface: str, released_fraction: float) -> float:
        g = self.global_mean
        c = self._cells.get((str(interface), _band(released_fraction)))
        if not c:
            return g
        n, s = c
        return (s + self.strength * g) / (n + self.strength)

    def value_fn(self):
        """A callable for `rank_regions(value_fn=...)` / `sample_region`."""
        return lambda r: self.value(r.interface, r.released_fraction)

    def to_dict(self) -> dict:
        return {"strength": self.strength, "n": self._n, "sum": self._sum,
                "cells": {f"{k[0]}|{k[1]}": v for k, v in self._cells.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskValue":
        tv = cls(strength=float(d.get("strength", 6.0)))
        tv._n = int(d.get("n", 0))
        tv._sum = float(d.get("sum", 0.0))
        for k, v in (d.get("cells") or {}).items():
            iface, band = k.split("|")
            tv._cells[(iface, int(band))] = [int(v[0]), float(v[1])]
        return tv

    @classmethod
    def from_events(cls, events, strength: float = 6.0) -> "TaskValue":
        """Fit from population-search events, which already carry the tuple."""
        tv = cls(strength=strength)
        for e in events:
            if e.get("ok") and e.get("delta") is not None:
                tv.observe(e["interface"], e["r_release"], e["delta"], True)
        return tv
