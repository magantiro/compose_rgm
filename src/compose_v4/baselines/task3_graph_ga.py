"""Graph-GA on Task 3, driven under strict-10k accounting.

The genetic operators are UPSTREAM AND UNMODIFIED: `baselines/graph_ga/upstream/
crossover.py` and `mutate.py`, byte-identical to jensengroup/GB_GA @ 4b49f182
(see that directory's PROVENANCE.md).  Only the driver is ours, because the
upstream driver reads molecules from a file, scores with the paper's own
penalised-logP objective, and has no notion of a budget.

WHAT IS REPRODUCED FROM MOLLEO'S OWN GRAPH-GA BASELINE
------------------------------------------------------
* fitness = the SUM of the five transformed objectives, which is exactly what
  their `Oracle.evaluate` computes (`pareto_optimizer.py:87-96`:
  `sum(max_eva) + (1 - (sa-1)/9) + sum(1 - min_eva)`).  Note this is the same
  normalisation our benchmark applies, arrived at independently -- a useful
  check that the transforms are right.
* population 120, offspring 70, mutation rate 0.067 (their `hparams_default.yaml`).
* roulette selection on normalised fitness, and `sanitize`'s dedupe-then-truncate,
  reproduced from upstream `GB_GA.py`.

WHAT IS DELIBERATELY NOT REPRODUCED
-----------------------------------
Their Pareto path screens the whole offspring population through unmetered
evaluators and never truncates the population.  Here every molecule that
receives the objective vector is charged, and the population is truncated to
`population_size` as upstream GB-GA does.  This is the strict-10k baseline, and
it is the one a COMPOSE policy must beat.

TWO UPSTREAM PROPERTIES THAT WOULD OTHERWISE BITE
-------------------------------------------------
1. `crossover.mol_OK` reads module-level `average_size` and `size_stdev` that
   upstream sets only in its driver.  Unset, it raises `NameError`, a bare
   `except` swallows it, EVERY candidate is rejected, and the run hangs with no
   error message.  They are set here from the actual initial population, since
   they impose a soft size prior on offspring and are therefore a parameter of
   the comparison rather than a constant.
2. `reproduce()` loops until it has filled the population, and failed crossovers
   cost no oracle calls -- so an oracle budget does not bound its wall time.
   Attempts are capped per generation and the failures are counted.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_UPSTREAM = Path(__file__).resolve().parents[3] / "baselines" / "graph_ga" / "upstream"

#: MOLLEO's hparams_default.yaml for Task 3.
POPULATION_SIZE = 120
OFFSPRING_SIZE = 70
MUTATION_RATE = 0.067
#: Upstream's reproduce() can fail forever at zero oracle cost. This bounds it.
ATTEMPTS_PER_OFFSPRING = 20


def _upstream():
    """Import the vendored operators. `mutate` imports `crossover` by bare name."""

    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    import crossover as co
    import mutate as mu
    return co, mu


def size_prior(smiles: list[str]) -> tuple[float, float]:
    """Mean and stdev of heavy-atom count over the starting population.

    Upstream hardcodes the ZINC values in its driver. Deriving them from the
    actual initial population keeps the offspring size prior matched to the
    cohort the baseline was given, rather than to whatever cohort Jensen used.
    """

    from rdkit import Chem

    sizes = [Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles
             if Chem.MolFromSmiles(s) is not None]
    return float(np.mean(sizes)), float(np.std(sizes))


def fitness(values: tuple[float, ...]) -> float:
    """MOLLEO's scalarization: the plain sum of the five transformed objectives."""

    return float(sum(values))


@dataclass
class GraphGAState:
    """Everything needed to continue: the population and how it is doing."""

    population: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    average_size: float = 0.0
    size_stdev: float = 0.0
    failed_reproductions: int = 0

    def as_dict(self) -> dict:
        return {"population": self.population, "scores": self.scores,
                "average_size": self.average_size, "size_stdev": self.size_stdev,
                "failed_reproductions": self.failed_reproductions}

    @classmethod
    def from_dict(cls, data: dict) -> GraphGAState:
        return cls(population=list(data["population"]),
                   scores=[float(s) for s in data["scores"]],
                   average_size=float(data["average_size"]),
                   size_stdev=float(data["size_stdev"]),
                   failed_reproductions=int(data.get("failed_reproductions", 0)))


def mating_pool(population: list[str], scores: list[float], size: int,
                rng: np.random.Generator) -> list[str]:
    """Roulette selection, upstream's `make_mating_pool` semantics.

    Drawn through the run's own Generator rather than the global `np.random`, so
    a resumed run continues the same stream.
    """

    total = float(sum(scores))
    weights = (np.asarray(scores, dtype=float) / total if total > 0
               else np.full(len(scores), 1.0 / len(scores)))
    index = rng.choice(len(population), size=size, replace=True, p=weights)
    return [population[int(i)] for i in index]


def sanitize(population: list[str], scores: list[float],
             population_size: int, *, prune: bool = True
             ) -> tuple[list[str], list[float]]:
    """Upstream's `sanitize`: optionally dedupe, sort by score, truncate.

    Molecules are already canonical here (the meter canonicalises), so the
    dedupe is a set membership test rather than upstream's double round-trip.
    """

    pairs = list(zip(scores, population))
    if prune:
        seen: set[str] = set()
        unique = []
        for score, smiles in pairs:
            if smiles not in seen:
                seen.add(smiles)
                unique.append((score, smiles))
        pairs = unique
    pairs.sort(key=lambda item: item[0], reverse=True)
    pairs = pairs[:population_size]
    return [smiles for _, smiles in pairs], [score for score, _ in pairs]


def reproduce(pool: list[str], count: int, mutation_rate: float,
              rng: np.random.Generator) -> tuple[list[str], int]:
    """`count` offspring from the mating pool, with a bounded attempt budget.

    The upstream operators use the global `random`/`np.random`, so a seed is
    drawn from the run's Generator and pushed into them. That keeps the whole
    run reproducible from the checkpointed Generator state alone, without having
    to serialise two global RNGs.
    """

    from rdkit import Chem

    co, mu = _upstream()
    seed = int(rng.integers(0, 2 ** 31 - 1))
    random.seed(seed)
    np.random.seed(seed)

    parents = [Chem.MolFromSmiles(s) for s in pool]
    parents = [m for m in parents if m is not None]
    offspring: list[str] = []
    failures = 0
    for _ in range(count * ATTEMPTS_PER_OFFSPRING):
        if len(offspring) >= count:
            break
        child = co.crossover(random.choice(parents), random.choice(parents))
        if child is None:
            failures += 1
            continue
        child = mu.mutate(child, mutation_rate)
        if child is None:
            failures += 1
            continue
        try:
            smiles = Chem.MolToSmiles(child)
        except Exception:  # noqa: BLE001 - upstream can emit unsanitisable mols
            failures += 1
            continue
        if smiles:
            offspring.append(smiles)
    return offspring, failures


def initialize(run, initial: list[str]) -> GraphGAState:
    """Charge the starting population and set the upstream size prior."""

    co, _ = _upstream()
    values = run.evaluate(initial)
    average_size, size_stdev = size_prior(initial)
    # THE globals whose absence silently rejects every candidate.
    co.average_size = average_size
    co.size_stdev = size_stdev
    # The frozen initialization sets are already canonical, so these keys are
    # the same strings the meter charged.
    population, scores = sanitize(list(initial), [fitness(v) for v in values],
                                  POPULATION_SIZE)
    return GraphGAState(population=population, scores=scores,
                        average_size=average_size, size_stdev=size_stdev)


def restore(state: GraphGAState) -> None:
    """Re-install the size prior after a resume; the globals do not persist."""

    co, _ = _upstream()
    co.average_size = state.average_size
    co.size_stdev = state.size_stdev


def step(run, state: GraphGAState, *, offspring_size: int = OFFSPRING_SIZE,
         mutation_rate: float = MUTATION_RATE) -> GraphGAState:
    """One generation: select, reproduce, charge, truncate."""

    pool = mating_pool(state.population, state.scores, POPULATION_SIZE, run.rng)
    offspring, failures = reproduce(pool, offspring_size, mutation_rate, run.rng)
    state.failed_reproductions += failures
    if not offspring:
        return state
    offspring = run.affordable(offspring)
    if not offspring:
        return state
    values = run.evaluate(offspring)
    # The meter canonicalises; align the population with the keys it charged so
    # the dedupe and the budget agree about what a molecule is.
    from compose_v4.benchmark.oracles import canonical
    keys = [canonical(s) or s for s in offspring]
    state.population, state.scores = sanitize(
        state.population + keys,
        state.scores + [fitness(v) for v in values],
        POPULATION_SIZE)
    return state
