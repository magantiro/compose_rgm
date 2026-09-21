"""COMPOSE adapter for the OFFICIAL GuacaMol goal-directed benchmark suite.

Scope and honesty contract
--------------------------
Everything scored here is scored by the official ``guacamol`` package: this
module supplies only a ``GoalDirectedGenerator``-shaped object and never
computes a benchmark number itself.  A benchmark scored by a local
reimplementation is not a comparison, so the harness stays upstream and
COMPOSE adapts to it.

What the generator actually is
------------------------------
The base checkpoint (Lineage B) is a DE-NOVO factorized tracelet rate model: a
learned marked CTMC over molecular graphs whose committed states are complete,
supported, connected molecules.  Goal direction is applied at INFERENCE by
:class:`~compose_v4.experiments.guided_rewrite_sampling.ProposalGuidedRewriteSampler`,
which tilts the conditional mark law toward high objective score without ever
proposing an action outside the base model's legal fiber.  The prior is frozen;
no benchmark objective is trained on.

Harvesting
----------
A trajectory's endpoint is not the only molecule it produces.  Because every
committed state of the process is a complete valid molecule, each scored
proposal is itself a legitimate generated candidate.  This adapter therefore
returns the best DISTINCT scored molecules seen anywhere in the run, not only
trajectory endpoints -- which is what makes a benchmark needing 100-250 distinct
molecules reachable at a sane trajectory count.  Nothing is invented: every
returned molecule was produced by executing a legal rewrite through the
production executor and was scored by the official objective.

Two call meters, deliberately both reported
-------------------------------------------
``guacamol``'s ``ScoringFunctionWrapper.evaluations`` counts every call the
harness sees, and is the official number.  COMPOSE additionally caches by
canonical molecular identity, so ``unique_molecules_scored`` is smaller.  The
official meter is authoritative; the unique count is reported beside it because
conflating the two is how a budget claim goes wrong.

Starting populations
--------------------
``generate_optimized_molecules`` receives an optional ``starting_population``
(only ``Ranolazine MPO`` supplies one in suite v2).  Lineage B is a de-novo
model with a ``carbon_tree`` source prior and cannot be initialised from an
arbitrary molecule, so the population is RECORDED AND IGNORED rather than
quietly half-used.  :attr:`GuacamolRunReport.ignored_starting_population` makes
that explicit for any benchmark where it happened.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles

#: Slot capacity of the Lineage B editing/generation kernel.
DEFAULT_SLOTS = 40

#: ``guacamol`` returns this for a molecule it cannot parse or score.
GUACAMOL_CORRUPT_SCORE = -1.0


@dataclass
class GuacamolRunConfig:
    """Sampling configuration for one benchmark.

    Defaults follow the established controlled-rollout driver
    (``scripts/evaluate_qed_controlled_rollouts.py``) so this adapter does not
    silently introduce a second, differently-tuned inference regime.
    """

    checkpoint: Path
    samples: int = 64
    proposals_per_event: int = 4
    guidance_strength: float = 8.0
    tempering_power: float = 0.0
    guidance_start_event: int = 0
    max_guided_events: int | None = 12
    n_slots: int = DEFAULT_SLOTS
    operational_horizon: float = 16.0
    time_step: float = 0.1
    max_events: int = 128
    workers: int = 1
    seed: int = 20260920

    def __post_init__(self) -> None:
        if int(self.samples) <= 0:
            raise ValueError("samples must be positive")
        if int(self.proposals_per_event) <= 0:
            raise ValueError("proposals_per_event must be positive")
        if int(self.workers) <= 0:
            raise ValueError("workers must be positive")
        if self.max_guided_events is not None and int(self.max_guided_events) <= 0:
            raise ValueError("max_guided_events must be positive when provided")


@dataclass
class GuacamolRunReport:
    """Per-benchmark COMPOSE-side accounting, beside the official score."""

    benchmark_name: str = ""
    requested_molecules: int = 0
    returned_molecules: int = 0
    distinct_candidates: int = 0
    rollouts: int = 0
    objective_calls: int = 0
    unique_molecules_scored: int = 0
    invalid_or_unscorable: int = 0
    ignored_starting_population: int = 0
    wall_seconds: float = 0.0
    best_candidate_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class GuacamolObjectiveScorer:
    """Bridge a ``guacamol`` ``ScoringFunction`` to a COMPOSE state scorer.

    COMPOSE guidance wants ``MolecularGraph -> float``; GuacaMol objectives want
    a SMILES string.  This is the whole adapter seam on the scoring side.

    Every distinct molecule that is successfully scored is retained in
    :attr:`observed`, because each is a committed state of the process and
    therefore a valid generated candidate.

    The control potential is a positive tilt, so the corrupt score (-1.0) is
    floored at ``floor``: an unscorable proposal must not be rewarded, and a
    negative exponent tilt would distort the mark law rather than merely
    declining to favour it.
    """

    def __init__(self, scoring_function: Any, *, floor: float = 0.0) -> None:
        self.scoring_function = scoring_function
        self.floor = float(floor)
        self.observed: dict[str, float] = {}
        self.calls = 0
        self.invalid = 0

    def __call__(self, state: MolecularGraph) -> float:
        smiles = molecular_graph_to_smiles(state)
        if smiles is None:
            self.invalid += 1
            return self.floor
        self.calls += 1
        raw = float(self.scoring_function.score(smiles))
        if raw <= GUACAMOL_CORRUPT_SCORE:
            # Scored, but the objective rejected it. Not a candidate.
            self.invalid += 1
            return self.floor
        previous = self.observed.get(smiles)
        if previous is None or raw > previous:
            self.observed[smiles] = raw
        return max(self.floor, raw)


def load_lineage_b(checkpoint: Path) -> tuple[Any, Any]:
    """Load the frozen de-novo base model and its carbon-tree source prior."""

    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, payload = load_factorized_rollout_checkpoint(Path(checkpoint))
    source_prior = payload.get("tree_source_prior")
    if source_prior is None:
        raise ValueError(
            "checkpoint carries no tree_source_prior; de-novo sampling needs one"
        )
    return model, source_prior


class ComposeGoalDirectedEngine:
    """COMPOSE-side implementation of the GuacaMol generator contract.

    Kept free of any ``guacamol`` import at module scope so the repository does
    not acquire a hard dependency on the benchmark package; the official
    ``GoalDirectedGenerator`` subclass lives in the driver script.
    """

    def __init__(
        self,
        model: Any,
        source_prior: Any,
        config: GuacamolRunConfig,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.source_prior = source_prior
        self.config = config
        self.progress = progress
        self.reports: list[GuacamolRunReport] = []
        self._benchmark_name = ""

    def set_benchmark_name(self, name: str) -> None:
        """Label the next report; the interface gives no benchmark identity."""
        self._benchmark_name = str(name)

    def generate_optimized_molecules(
        self,
        scoring_function: Any,
        number_molecules: int,
        starting_population: Sequence[str] | None = None,
    ) -> list[str]:
        """Official ``GoalDirectedGenerator`` entry point."""

        from guacamol.utils.chemistry import canonicalize_list

        from compose_v4.experiments.guided_rewrite_sampling import (
            CachedStateScorer,
            ProposalGuidedRewriteSampler,
        )
        from compose_v4.experiments.parallel_tracelet_sampling import (
            sample_tracelet_ancestral_many,
        )

        cfg = self.config
        report = GuacamolRunReport(
            benchmark_name=self._benchmark_name,
            requested_molecules=int(number_molecules),
            rollouts=int(cfg.samples),
            ignored_starting_population=len(starting_population or ()),
        )
        began = time.monotonic()

        objective = GuacamolObjectiveScorer(scoring_function)
        cached = CachedStateScorer(objective)
        sampler = ProposalGuidedRewriteSampler(
            self.model,
            cached,
            proposals_per_event=int(cfg.proposals_per_event),
            guidance_strength=float(cfg.guidance_strength),
            tempering_power=float(cfg.tempering_power),
            guidance_start_event=int(cfg.guidance_start_event),
            max_guided_events=cfg.max_guided_events,
        )

        rollouts = sample_tracelet_ancestral_many(
            sampler,
            seed=int(cfg.seed),
            samples=int(cfg.samples),
            workers=int(cfg.workers),
            n_slots=int(cfg.n_slots),
            operational_horizon=float(cfg.operational_horizon),
            time_step=float(cfg.time_step),
            max_events=int(cfg.max_events),
            torch_threads_per_worker=1,
            source_prior=self.source_prior,
        )

        # Endpoints are scored too: a trajectory that never triggered guidance
        # would otherwise contribute nothing.
        candidates = dict(objective.observed)
        for rollout in rollouts:
            smiles = molecular_graph_to_smiles(rollout.final_state)
            if smiles is None or smiles in candidates:
                continue
            value = float(scoring_function.score(smiles))
            objective.calls += 1
            if value > GUACAMOL_CORRUPT_SCORE:
                candidates[smiles] = value

        # Canonicalize exactly the way the official harness will before scoring,
        # so our ranking is over the same molecular identities it will see.
        selected = _rank_distinct_candidates(
            candidates, int(number_molecules), canonicalize_list
        )

        report.objective_calls = objective.calls
        report.unique_molecules_scored = len(candidates)
        report.invalid_or_unscorable = objective.invalid
        report.distinct_candidates = len(candidates)
        report.returned_molecules = len(selected)
        report.best_candidate_score = max(candidates.values()) if candidates else None
        report.wall_seconds = time.monotonic() - began
        self.reports.append(report)
        if self.progress is not None:
            self.progress(
                f"{report.benchmark_name}: returned {report.returned_molecules}"
                f"/{report.requested_molecules} distinct from "
                f"{report.distinct_candidates} candidates "
                f"({report.objective_calls} objective calls, "
                f"{report.wall_seconds:.1f}s)"
            )
        return selected


def _rank_distinct_candidates(
    candidates: dict[str, float],
    number_molecules: int,
    canonicalize_list: Callable[..., list[str]],
) -> list[str]:
    """Best-scoring distinct molecules under the harness's own canonical form.

    The harness canonicalizes without stereocentres and de-duplicates before
    scoring, so two of our candidates can collapse into one of its molecules.
    Collapsing first means we return ``number_molecules`` entries that actually
    survive as distinct, instead of silently under-filling the request.
    """

    if not candidates:
        return []
    ordered = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    smiles = [item[0] for item in ordered]
    canonical = canonicalize_list(smiles, include_stereocenters=False)

    best: dict[str, float] = {}
    for original, canon in zip(smiles, canonical):
        if not canon:
            continue
        score = candidates[original]
        if canon not in best or score > best[canon]:
            best[canon] = score
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [entry[0] for entry in ranked[: int(number_molecules)]]
