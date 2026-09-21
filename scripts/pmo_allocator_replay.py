"""Zero-oracle replay of PMO parent/lane allocation against a completed run.

Every number here is produced by driving the PRODUCTION allocation path --
`ProgramOptimizer.restore`, `ProgramOptimizer.selection` and
`PmoPopulationController.selection` -- on an archive restored from a finished
campaign's `campaign/round_*/complete.json`.  Nothing in this module reimplements
the allocation it measures, so a drift in production shows up here as a changed
number rather than as a harness that agrees with itself.

The scorer is FROZEN: every score is read from `snapshot.observations`, which
already carries the charged oracle result.  No oracle is constructed and no call
is charged.  The score-swing probes mutate a restored in-memory copy only.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.pmo_population_controller import PmoPopulationController

REPLAY_ROOT = os.environ.get(
    "PMO_REPLAY_ROOT", os.path.expanduser("~/compose_pmo_replay_data")
)
TASKS = ("celecoxib_rediscovery", "gsk3b", "perindopril_mpo")
DRAWS = 20000


# ---- Restoration ----


def round_files(task: str) -> list[str]:
    return sorted(glob.glob(os.path.join(REPLAY_ROOT, task, "campaign", "round_*", "complete.json")))


def load_snapshot(path: str) -> dict:
    return json.load(open(path))["snapshot"]


def restore_view(snapshot: dict):
    """A real ProgramOptimizer carrying the run's archive, plus its population state.

    `PmoPopulationController.selection` calls `ProgramOptimizer.selection(self)`
    explicitly and then reads `self.population_state`, so the restored base object is
    a faithful receiver for the production method.  The controller's own `restore`
    additionally demands the jump checkpoint, which the allocation path never reads.
    """
    view = ProgramOptimizer.restore(snapshot)
    view.population_state = json.loads(json.dumps(snapshot["pmo_population"]["population_state"]))
    # Bind the controller's own scale estimator to the restored receiver, so the
    # production `selection` runs against production code end to end.
    view.observed_upside_scale = PmoPopulationController.observed_upside_scale.__get__(view)
    return view


# ---- Mass ----


def endpoint_mass(view) -> dict[str, float]:
    """Endpoint-level parent mass under the production controller selection."""
    keys, probabilities = PmoPopulationController.selection(view)
    mass: dict[str, float] = {}
    for key, probability in zip(keys, probabilities, strict=True):
        endpoint = view.entries[key]["endpoint"]
        mass[endpoint] = mass.get(endpoint, 0.0) + float(probability)
    return mass


def drawn_parents(view, *, draws: int = DRAWS, seed: int = 20260921) -> list[str]:
    """Endpoints of parents actually drawn, sampling the production law."""
    keys, probabilities = PmoPopulationController.selection(view)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(keys), size=draws, p=probabilities / probabilities.sum())
    return [view.entries[keys[int(i)]]["endpoint"] for i in picks]


# ---- Chemistry (read-only descriptors; the scorer stays frozen) ----


def heavy_atoms(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"replay descriptor needs an RDKit-valid molecule: {smiles!r}")
    return int(molecule.GetNumHeavyAtoms())


def qed_sa(smiles: str) -> tuple[float, float]:
    from rdkit.Contrib.SA_Score import sascorer

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"replay descriptor needs an RDKit-valid molecule: {smiles!r}")
    return float(QED.qed(molecule)), float(sascorer.calculateScore(molecule))


def quantiles(values) -> dict[str, float]:
    array = np.asarray(sorted(values), dtype=float)
    if not len(array):
        return {}
    return {
        "n": int(len(array)),
        "min": float(array[0]),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array[-1]),
        "mean": float(array.mean()),
    }


def concentration(mass: dict[str, float], fraction: float = 0.8) -> dict:
    ordered = sorted(mass.items(), key=lambda kv: -kv[1])
    total, running, holders = sum(mass.values()), 0.0, 0
    for _, value in ordered:
        if running >= fraction * total:
            break
        running += value
        holders += 1
    return {
        "endpoints_holding_80pct": holders,
        "archive_size": len(mass),
        "top8_mass": float(sum(v for _, v in ordered[:8])),
        "top_members": [
            {"smiles": s, "mass": float(v), "heavy_atoms": heavy_atoms(s)} for s, v in ordered[:8]
        ],
    }


# ---- Score-swing probe ----


def set_score(view, endpoint: str, score: float) -> None:
    """Rebind every observation of one endpoint on a restored in-memory copy."""
    touched = 0
    for receipt in view.observations.values():
        if receipt["endpoint"] == endpoint:
            receipt["score"] = float(score)
            touched += 1
    if not touched:
        raise ValueError(f"no observation to rebind for {endpoint!r}")
    view._niche_cache = None


def score_swing(snapshot: dict, endpoint: str, low: float, high: float) -> dict:
    """Mass of `endpoint` at a low and a high score, all else held fixed."""
    masses = {}
    for label, score in (("low", low), ("high", high)):
        view = restore_view(snapshot)
        set_score(view, endpoint, score)
        masses[label] = endpoint_mass(view).get(endpoint, 0.0)
    return {
        "endpoint": endpoint,
        "heavy_atoms": heavy_atoms(endpoint),
        "mass_at_low_score": float(masses["low"]),
        "mass_at_high_score": float(masses["high"]),
        "delta": float(masses["high"] - masses["low"]),
        "low_score": low,
        "high_score": high,
    }


def observed_scores(view) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for receipt in view.observations.values():
        per.setdefault(receipt["endpoint"], []).append(float(receipt["score"]))
    return {k: float(np.mean(v)) for k, v in per.items()}


# ---- Escape latch history (measured, per round) ----


def escape_history(task: str) -> list[dict]:
    rows = []
    for path in round_files(task):
        payload = json.load(open(path))
        state = payload["snapshot"]["pmo_population"]["population_state"]
        rows.append(
            {
                "round": int(payload["summary"]["round"]),
                "best_score": state["best_score"],
                "rounds_without_improvement": int(state["rounds_without_improvement"]),
                "escape_rounds_remaining": int(state["escape_rounds_remaining"]),
                "allocation_decisions": int(state["allocation_decisions"]),
            }
        )
    return rows


def credit_trials(task: str) -> dict:
    snapshot = load_snapshot(round_files(task)[-1])
    cells = snapshot["pmo_population"]["credit"]["cells"]
    trials = [int(cell["trials"]) for cell in cells]
    return {
        "cells": len(cells),
        "total_trials": int(sum(trials)),
        "trials_per_cell": float(sum(trials) / len(trials)) if trials else 0.0,
        "histogram": dict(sorted(Counter(trials).items())),
        "basins": len({cell["basin"] for cell in cells}),
    }


def mean_positive_improvement(task: str) -> dict:
    """Measured reward scale: mean positive parent->child improvement, from the run."""
    snapshot = load_snapshot(round_files(task)[-1])
    cells = snapshot["pmo_population"]["credit"]["cells"]
    total = sum(float(cell["positive_improvement_sum"]) for cell in cells)
    wins = sum(int(cell["improvements"]) for cell in cells)
    trials = sum(int(cell["trials"]) for cell in cells)
    return {
        "positive_improvement_sum": float(total),
        "improvements": wins,
        "trials": trials,
        "mean_positive_improvement_per_win": float(total / wins) if wins else 0.0,
        "mean_positive_improvement_per_trial": float(total / trials) if trials else 0.0,
    }


# ---- Task-level report ----


def task_report(task: str) -> dict:
    snapshot = load_snapshot(round_files(task)[-1])
    view = restore_view(snapshot)
    mass = endpoint_mass(view)
    scores = observed_scores(view)
    archive = sorted(mass)
    archive_heavy = {s: heavy_atoms(s) for s in archive}
    drawn = drawn_parents(view)
    drawn_heavy = [archive_heavy[s] for s in drawn]
    mass_weighted_quality = []
    for smiles in archive:
        quality, access = qed_sa(smiles)
        mass_weighted_quality.append((mass[smiles], quality, access, archive_heavy[smiles]))
    weights = np.asarray([row[0] for row in mass_weighted_quality])
    weights = weights / weights.sum()
    return {
        "task": task,
        "archive_size": len(archive),
        "archive_heavy_atoms": quantiles(list(archive_heavy.values())),
        "drawn_parent_heavy_atoms": quantiles(drawn_heavy),
        "drawn_under_10_heavy_fraction": float(np.mean([h < 10 for h in drawn_heavy])),
        "archive_under_10_heavy_fraction": float(
            np.mean([h < 10 for h in archive_heavy.values()])
        ),
        "concentration": concentration(mass),
        "mass_weighted_qed": float(sum(w * row[1] for w, row in zip(weights, mass_weighted_quality))),
        "mass_weighted_sa": float(sum(w * row[2] for w, row in zip(weights, mass_weighted_quality))),
        "archive_mean_qed": float(np.mean([row[1] for row in mass_weighted_quality])),
        "archive_mean_sa": float(np.mean([row[2] for row in mass_weighted_quality])),
        "escape_history": escape_history(task),
        "credit_trials": credit_trials(task),
        "reward_scale": mean_positive_improvement(task),
        "best_observed_score": float(max(scores.values())),
    }


# ---- Arms ----


def arm_view(snapshot: dict, allocation: str):
    """Restored archive under a named parent-allocation mode.

    The mode is the only thing that differs between arms: same archive, same frozen
    scores, same duplicate counts, same population state.
    """
    from dataclasses import replace as _replace

    view = restore_view(snapshot)
    view.config = _replace(view.config, parent_allocation=allocation)
    view._niche_cache = None
    return view


def arm_report(snapshot: dict, allocation: str) -> dict:
    view = arm_view(snapshot, allocation)
    mass = endpoint_mass(view)
    archive_heavy = {s: heavy_atoms(s) for s in mass}
    drawn = drawn_parents(view)
    drawn_heavy = [archive_heavy[s] for s in drawn]
    scores = observed_scores(view)
    weights = np.asarray([mass[s] for s in sorted(mass)])
    weights = weights / weights.sum()
    ordered = sorted(mass)
    quality = [qed_sa(s) for s in ordered]
    return {
        "allocation": allocation,
        "archive_heavy_atoms": quantiles(list(archive_heavy.values())),
        "drawn_parent_heavy_atoms": quantiles(drawn_heavy),
        "drawn_under_10_heavy_fraction": float(np.mean([h < 10 for h in drawn_heavy])),
        "drawn_mean_parent_score": float(np.mean([scores[s] for s in drawn])),
        "archive_mean_score": float(np.mean(list(scores.values()))),
        "best_archive_score": float(max(scores.values())),
        "concentration": concentration(mass),
        "mass_weighted_qed": float(sum(w * q for w, (q, _) in zip(weights, quality))),
        "mass_weighted_sa": float(sum(w * a for w, (_, a) in zip(weights, quality))),
        "archive_mean_qed": float(np.mean([q for q, _ in quality])),
        "archive_mean_sa": float(np.mean([a for _, a in quality])),
    }


def singleton_swing(snapshot: dict, allocation: str, endpoint: str) -> dict:
    """Mass response of one endpoint to a score swing, all else fixed."""
    from dataclasses import replace as _replace

    masses = {}
    for label, score in (("low", 0.01), ("high", 0.99)):
        view = restore_view(snapshot)
        view.config = _replace(view.config, parent_allocation=allocation)
        set_score(view, endpoint, score)
        masses[label] = endpoint_mass(view).get(endpoint, 0.0)
    return {
        "allocation": allocation,
        "endpoint": endpoint,
        "heavy_atoms": heavy_atoms(endpoint),
        "mass_at_0.01": float(masses["low"]),
        "mass_at_0.99": float(masses["high"]),
        "delta": float(masses["high"] - masses["low"]),
    }
