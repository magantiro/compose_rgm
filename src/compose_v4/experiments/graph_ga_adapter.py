"""Adapter driving the REAL upstream GB-GA under COMPOSE oracle accounting.

This runs the vendored, byte-identical upstream algorithm
(``baselines/graph_ga/upstream/``, commit ``4b49f182``, MIT) -- not a GA-shaped
reimplementation. The adapter's whole job is to supply the three things upstream
leaves to its caller, and to do so in a way that is recorded rather than
implicit:

1. **The objective.** ``GB_GA.py`` imports ``scoring_functions`` at module level
   purely for its ``calculate_scores`` map. That module also loads 750 KB of
   ZINC-derived normalization data from CWD-relative paths, which has nothing to
   do with the algorithm, so the adapter installs a shim reproducing
   ``calculate_scores`` verbatim and routes every call through an
   :class:`OracleAccountant`.

2. **The two undocumented size globals.** ``crossover.mol_OK`` reads
   ``average_size`` and ``size_stdev``; unset, it raises ``NameError`` into a
   bare ``except`` and every candidate is silently rejected. They impose a soft
   Gaussian size prior on offspring, so they are a **fairness parameter**: left
   at upstream's ZINC values (39.15 / 3.50) while running a COMPOSE panel, they
   would ask the baseline to build ZINC-sized molecules. The adapter derives
   them from the actual source panel by default and records what it used.

3. **A wall-clock guard.** ``reproduce()`` loops unbounded and failed crossovers
   cost no oracle calls, so an oracle budget does not bound runtime. GB-GA is
   the one method in the registry that can spend unbounded time at zero budget.

Nothing here modifies upstream. The vendored files stay verbatim so that a
reviewer can diff them against GitHub.
"""

from __future__ import annotations

import statistics
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem

from compose_v4.experiments.oracle_accounting import (
    BudgetExhausted,
    OracleAccountant,
)

UPSTREAM_DIR = (
    Path(__file__).resolve().parents[3] / "baselines" / "graph_ga" / "upstream"
)

#: Upstream's own ZINC-derived values, from GA_logP.py lines 19-20. Recorded so
#: a run that deliberately reproduces upstream can ask for them by name.
UPSTREAM_ZINC_SIZE_PRIOR = (39.15, 3.50)

#: GB-GA's selection operator REQUIRES a non-negative objective. Upstream:
#:
#:     def calculate_normalized_fitness(scores):
#:       sum_scores = sum(scores)
#:       normalized_fitness = [score/sum_scores for score in scores]
#:
#: which is then fed to ``np.random.choice(population, p=fitness)``. A single
#: negative score raises "probabilities are not non-negative". This is not an
#: accident: upstream's own objective clamps for exactly this reason
#: (``scoring_functions.logP_max``: ``return max(0.0, score)``).
#:
#: A COMPOSE margin objective is negative whenever the molecule sits outside the
#: box, which is most held-in molecules at step zero. So a transform must be
#: chosen, and the choice is a DEVIATION that changes selection pressure. It is
#: therefore required explicitly -- there is no default.
#: Both options are applied PER MOLECULE, never per batch. Roulette-wheel
#: fitness is recomputed over a pool that mixes generations
#: (``scores + new_scores``), so a batch-relative transform would rescale old
#: survivors against new offspring and would not be a fixed objective at all.
NONNEGATIVE_TRANSFORMS = ("upstream_clamp", "constant_shift")

#: Positive floor used when a declared ``constant_shift`` still leaves a score
#: non-positive. Small enough to be effectively zero selection weight, large
#: enough to keep the fitness sum non-zero.
_SHIFT_FLOOR = 1e-9


class GraphGAUnavailable(RuntimeError):
    """The vendored upstream GB-GA could not be loaded."""


def _install_scoring_shim() -> None:
    """Provide ``scoring_functions.calculate_scores`` with upstream semantics.

    Must run **before** ``GB_GA`` is imported: upstream does
    ``import scoring_functions as sc`` at module level.

    Upstream, verbatim::

        def calculate_scores(population,function,scoring_args):
          if 'pop' in function.__name__:
            scores = function(population,scoring_args)
          else:
            scores = [function(gene,scoring_args) for gene in population]
          return scores

    The adapter always supplies a per-molecule function, so only the second
    branch is ever taken; the first is reproduced so the shim cannot diverge if
    a future caller passes a population-level objective.
    """
    module = types.ModuleType("scoring_functions")

    def calculate_scores(population, function, scoring_args):
        if "pop" in function.__name__:
            return function(population, scoring_args)
        return [function(gene, scoring_args) for gene in population]

    module.calculate_scores = calculate_scores
    sys.modules["scoring_functions"] = module


def load_upstream(average_size: float, size_stdev: float):
    """Import the vendored upstream modules and set the required globals."""
    if not (UPSTREAM_DIR / "GB_GA.py").exists():
        raise GraphGAUnavailable(
            f"vendored GB-GA not found at {UPSTREAM_DIR}; see its PROVENANCE.md"
        )
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))

    import crossover as co

    # THE UNDOCUMENTED REQUIREMENT. Without both of these, mol_OK raises
    # NameError, the bare except swallows it, every candidate is rejected,
    # crossover always returns None, and reproduce() spins forever in silence.
    co.average_size = float(average_size)
    co.size_stdev = float(size_stdev)

    # GB_GA does `import scoring_functions as sc` at module level.
    _install_scoring_shim()
    import GB_GA as gb_ga

    return gb_ga, co


def size_prior_from_sources(smiles: list[str]) -> tuple[float, float]:
    """Derive the offspring size prior from the actual source panel.

    Upstream hardcodes ZINC's mean and standard deviation. Reusing those on a
    different panel silently changes what the baseline is asked to build, so the
    default is to measure the panel instead. A single-source panel has no
    spread, so the upstream standard deviation is retained in that case and the
    fallback is reported.
    """
    counts = []
    for smi in smiles:
        molecule = Chem.MolFromSmiles(smi)
        if molecule is not None:
            counts.append(molecule.GetNumAtoms())
    if not counts:
        raise ValueError("no parseable source molecules")
    mean = float(statistics.fmean(counts))
    stdev = float(statistics.stdev(counts)) if len(counts) > 1 else UPSTREAM_ZINC_SIZE_PRIOR[1]
    return mean, stdev


@dataclass
class GraphGAResult:
    scores: list[float]
    endpoints: list[str]
    high_scores: list[tuple[float, str]]
    generations_run: int
    counters: dict[str, int]
    accountant_manifest: dict[str, Any]
    wall_seconds: float
    budget_exhausted: bool
    wall_clock_exceeded: bool
    shift_floor_hits: int
    size_prior: dict[str, float]
    seed_file: str
    settings: dict[str, Any] = field(default_factory=dict)


def run_graph_ga(
    *,
    sources: list[str],
    objective: Callable[[str], float],
    seed_path: Path,
    budget: int,
    budget_counter: str,
    population_size: int = 20,
    generations: int = 5,
    mating_pool_size: int | None = None,
    mutation_rate: float = 0.05,
    prune_population: bool = True,
    seed: int = 20260813,
    max_wall_seconds: float = 300.0,
    size_prior: tuple[float, float] | None = None,
    nonnegative_transform: str,
    constant_shift: float = 0.0,
) -> GraphGAResult:
    """Run upstream GB-GA on a source panel under a declared oracle budget.

    Args:
        sources: seed molecules, written to ``seed_path`` one per line. Upstream
            samples the initial population from this file **with replacement**
            (``random.choice``), so a one-line file replicates one molecule.
        objective: ``canonical SMILES -> float``, the frozen COMPOSE objective.
        budget/budget_counter: passed straight to :class:`OracleAccountant`.
        max_wall_seconds: guard against ``reproduce()``'s unbounded loop, which
            costs no oracle calls and so is not bounded by ``budget``.
        size_prior: ``(average_size, size_stdev)``. Defaults to the measured
            panel; pass :data:`UPSTREAM_ZINC_SIZE_PRIOR` to reproduce upstream.
        nonnegative_transform: required. GB-GA's roulette-wheel selection
            divides by the sum of scores and feeds the result to
            ``np.random.choice(p=...)``, so a negative score raises. See
            :data:`NONNEGATIVE_TRANSFORMS`. ``upstream_clamp`` is what the
            method's own authors do (``logP_max`` returns ``max(0.0, score)``)
            and flattens everything infeasible to one value; ``constant_shift``
            adds a declared constant and preserves the gradient but changes the
            fitness ratios roulette-wheel selection acts on. Either way this is
            a recorded deviation, which is why there is no default.
        constant_shift: the constant, required when using ``constant_shift``.
    """
    if nonnegative_transform not in NONNEGATIVE_TRANSFORMS:
        raise ValueError(
            f"nonnegative_transform must be one of {NONNEGATIVE_TRANSFORMS}; "
            "GB-GA's selection cannot accept a negative objective"
        )
    if nonnegative_transform == "constant_shift" and constant_shift <= 0.0:
        raise ValueError("constant_shift must be positive when it is the transform")

    average_size, size_stdev = size_prior or size_prior_from_sources(sources)
    gb_ga, _ = load_upstream(average_size, size_stdev)

    seed_path = Path(seed_path)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text("\n".join(sources) + "\n")

    accountant = OracleAccountant(
        evaluate=objective, budget=budget, budget_counter=budget_counter
    )
    started = time.perf_counter()
    state = {"wall_exceeded": False, "shift_floor_hits": 0}

    def scored(molecule, _args) -> float:
        # GB-GA hands us an RDKit Mol. Canonicalization and all counting happen
        # inside the accountant, on SMILES, exactly as for every other method.
        if time.perf_counter() - started > max_wall_seconds:
            state["wall_exceeded"] = True
            raise BudgetExhausted("wall-clock guard tripped")
        try:
            smiles = Chem.MolToSmiles(molecule)
        except Exception:
            smiles = ""
        raw = accountant.score(smiles)
        # The accountant sees and records the UNTRANSFORMED objective; only the
        # value handed back to GB-GA's selection is made non-negative, so the
        # deviation cannot leak into the recorded oracle values.
        if nonnegative_transform == "upstream_clamp":
            return max(0.0, raw)
        shifted = raw + constant_shift
        if shifted <= 0.0:
            # The declared shift was not large enough for this molecule. Floor it
            # so selection stays defined, and count it: a high floor rate means
            # the shift is doing the clamp's job and the deviation is worse than
            # it looks.
            state["shift_floor_hits"] += 1
            return _SHIFT_FLOOR
        return shifted

    args = (
        population_size,
        str(seed_path),
        scored,
        generations,
        mating_pool_size or population_size,
        mutation_rate,
        None,          # scoring_args
        float("inf"),  # max_score: never early-stop on score
        prune_population,
        seed,
    )

    budget_exhausted = False
    try:
        scores, population, high_scores, generations_run = gb_ga.GA(args)
    except BudgetExhausted:
        # Upstream has no budget concept, so exhaustion unwinds out of GA().
        # Everything the run produced is already recorded in the accountant.
        budget_exhausted = True
        scores, population, high_scores, generations_run = [], [], [], 0

    elapsed = time.perf_counter() - started
    return GraphGAResult(
        scores=[float(s) for s in scores],
        endpoints=[Chem.MolToSmiles(m) for m in population],
        high_scores=[(float(s), smi) for s, smi in high_scores],
        generations_run=int(generations_run),
        counters=accountant.counts.as_dict(),
        accountant_manifest=accountant.manifest(),
        wall_seconds=round(elapsed, 4),
        budget_exhausted=budget_exhausted or accountant.exhausted,
        wall_clock_exceeded=state["wall_exceeded"],
        shift_floor_hits=state["shift_floor_hits"],
        size_prior={"average_size": average_size, "size_stdev": size_stdev},
        seed_file=str(seed_path),
        settings={
            "population_size": population_size,
            "generations": generations,
            "mating_pool_size": mating_pool_size or population_size,
            "mutation_rate": mutation_rate,
            "prune_population": prune_population,
            "seed": seed,
            "max_wall_seconds": max_wall_seconds,
            "upstream_commit": "4b49f1822c5190e8b2bbb8b7403eed30af9e50fd",
            "nonnegative_transform": nonnegative_transform,
            "constant_shift": constant_shift,
            "deviation_note": (
                "GB-GA's roulette-wheel selection requires a non-negative "
                "objective; upstream clamps its own objective for the same "
                "reason. The transform is applied only to the value handed to "
                "selection, never to the recorded oracle values."
            ),
        },
    )


def expected_scoring_calls(population_size: int, generations_run: int) -> int:
    """Upstream's own accounting identity: initial population plus each generation.

    The paper states "The population size is 20 and 50 generations are used
    (i.e. 1000 J(m) evaluations per run)", which is 20 x 51 = 1020 rounded down.
    """
    return population_size * (generations_run + 1)
