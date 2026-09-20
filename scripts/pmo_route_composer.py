"""Route-distilled program composer for PMO, fitted leave-one-task-out.

Model family
------------
This is the T4 composer, not a new one.  ``q_theta(Z | G)`` is
``MarginalSubgoalPolicy`` from ``compose_v4.control.structural_subgoal_policy``
-- a balanced frequency law over address-free ``StructuralDeltaTemplate``
patches plus a region-count law -- and generation is the T4
``propose_structural_goals`` beam, which rebinds each template to the runtime
graph and composes one to four regions.

The one PMO-specific addition is the shape law ``q_theta(s, m)`` over
scale ``s in {small, medium, large}`` and mode ``m in {grow, prune, replace,
remodel}``, so the composer scores ``q_theta(Z, s, m | G)``.  Scale and mode are
deterministic functions of ``Z``, so the joint factorises exactly as
``q(Z) * q(s, m)`` only under the convention recorded here: the shape law is a
*reweighting* of the template law toward teacher-shaped programs, not an
independent second factor.  ``score_proposal`` therefore returns
``log q(Z) + shape_weight * log q(s, m)`` and ``shape_weight`` is reported.

Balancing
---------
T4 balances its marginal by source molecule.  The PMO corpus has 7 distinct
source states and 11 tasks, and the transfer axis under test is the task, so
the default balance key is the TASK.  ``fit_marginal_subgoal_policy`` is reused
unchanged by passing the balance key in its ``source_group`` slot; the choice is
recorded in the fit audit and is never silent.

No oracle, no scoring, no target name enters the fitted object.  The held task
is used only to remove its rows before fitting.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from pmo_route_program_corpus import (
    MODES,
    SCALE_BANDS,
    ProgramRow,
    program_mode,
    program_scale,
    template_shape,
)

from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
    fit_marginal_subgoal_policy,
    propose_structural_goals,
)

COMPOSER_SCHEMA = "pmo_route_distilled_composer_v1"
SHAPE_SCHEMA = "pmo_program_shape_law_v1"
DEFAULT_EXPLORATION_FLOOR = 0.10
DEFAULT_SHAPE_WEIGHT = 1.0


# ---- Shape law q(s, m) ---------------------------------------------------


@dataclass(frozen=True)
class ProgramShapeLaw:
    """Balanced joint law over program scale and program mode."""

    scales: tuple[str, ...]
    modes: tuple[str, ...]
    probabilities: tuple[float, ...]
    exploration_floor: float
    training_identity: str

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=float)
        if (
            values.shape != (len(self.scales) * len(self.modes),)
            or np.any(values <= 0)
            or not np.isclose(values.sum(), 1)
            or not 0 < self.exploration_floor < 1
            or not self.training_identity
        ):
            raise ValueError("invalid program shape law")

    def _index(self, scale: str, mode: str) -> int:
        if scale not in self.scales or mode not in self.modes:
            raise KeyError(f"unknown program shape ({scale!r}, {mode!r})")
        return self.scales.index(scale) * len(self.modes) + self.modes.index(mode)

    def probability(self, scale: str, mode: str) -> float:
        return self.probabilities[self._index(scale, mode)]

    def log_probability(self, scale: str, mode: str) -> float:
        return math.log(self.probability(scale, mode))

    def cells(self) -> dict:
        return {
            f"{scale}|{mode}": self.probability(scale, mode)
            for scale in self.scales
            for mode in self.modes
        }

    def checkpoint(self) -> dict:
        return {
            "schema_version": SHAPE_SCHEMA,
            "scales": list(self.scales),
            "modes": list(self.modes),
            "probabilities": list(self.probabilities),
            "exploration_floor": self.exploration_floor,
            "training_identity": self.training_identity,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> ProgramShapeLaw:
        if payload.get("schema_version") != SHAPE_SCHEMA:
            raise ValueError("program shape law schema mismatch")
        return cls(
            tuple(payload["scales"]),
            tuple(payload["modes"]),
            tuple(map(float, payload["probabilities"])),
            float(payload["exploration_floor"]),
            str(payload["training_identity"]),
        )


def fit_program_shape_law(
    rows: list[ProgramRow], *, balance_key: str = "task", exploration_floor: float
) -> ProgramShapeLaw:
    """Fit the balanced joint (scale, mode) law over training programs."""
    if not rows or not 0 < exploration_floor < 1:
        raise ValueError("shape fitting requires rows and an exploration floor")
    grouped: dict[str, list[ProgramRow]] = defaultdict(list)
    for row in rows:
        grouped[_balance_value(row, balance_key)].append(row)
    weights = np.zeros(len(SCALE_BANDS) * len(MODES), dtype=float)
    for group in grouped.values():
        for row in group:
            index = SCALE_BANDS.index(row.scale) * len(MODES) + MODES.index(row.mode)
            weights[index] += 1 / (len(grouped) * len(group))
    empirical = weights / weights.sum()
    cells = len(weights)
    probabilities = exploration_floor / cells + (1 - exploration_floor) * empirical
    training_identity = identity(
        {
            "schema_version": "pmo_program_shape_fit_v1",
            "groups": len(grouped),
            "rows": len(rows),
            "balance_key": balance_key,
            "probabilities": probabilities.tolist(),
            "exploration_floor": exploration_floor,
        }
    )
    return ProgramShapeLaw(
        SCALE_BANDS, MODES, tuple(map(float, probabilities)), exploration_floor, training_identity
    )


def _balance_value(row: ProgramRow, balance_key: str) -> str:
    if balance_key == "task":
        return row.task
    if balance_key == "lineage":
        return row.lineage or row.task
    if balance_key == "source":
        return row.source_group
    raise ValueError(f"unknown balance key {balance_key!r}")


# ---- Composer ------------------------------------------------------------


@dataclass(frozen=True)
class PmoRouteComposer:
    """q_theta(Z, s, m | G) over complete dependency-aware PMO programs."""

    marginal: MarginalSubgoalPolicy
    shape: ProgramShapeLaw
    templates: tuple[StructuralDeltaTemplate, ...]
    held_task: str | None
    shape_weight: float
    training_identity: str

    def __post_init__(self) -> None:
        if {row.template_id for row in self.templates} != set(self.marginal.template_ids):
            raise ValueError("composer vocabulary and marginal disagree")
        if self.shape_weight < 0:
            raise ValueError("shape weight must be nonnegative")

    def score(self, templates: tuple[StructuralDeltaTemplate, ...]) -> float:
        """log q_theta(Z): the T4 balanced template and region-count law."""
        return self.marginal.score(templates)

    def program_shape(self, templates: tuple[StructuralDeltaTemplate, ...]) -> tuple[str, str]:
        """Deterministic (scale, mode) of one complete program."""
        shapes = [template_shape(row) for row in templates]
        events = sum(_events(row) for row in templates)
        return (
            program_scale(events),
            program_mode(
                sum(row["created"] for row in shapes), sum(row["released"] for row in shapes)
            ),
        )

    def score_proposal(self, templates: tuple[StructuralDeltaTemplate, ...]) -> float:
        """log q_theta(Z, s, m): template law reweighted by the shape law."""
        scale, mode = self.program_shape(templates)
        return self.score(templates) + self.shape_weight * self.shape.log_probability(scale, mode)

    def propose(self, source, **kwargs):
        """Generate bound, valid candidate programs through the T4 beam."""
        return propose_structural_goals(source, self.templates, self.marginal, ranker=None, **kwargs)

    def checkpoint(self) -> dict:
        return {
            "schema_version": COMPOSER_SCHEMA,
            "marginal": self.marginal.checkpoint(),
            "shape": self.shape.checkpoint(),
            "templates": [row.payload() for row in self.templates],
            "held_task": self.held_task,
            "shape_weight": self.shape_weight,
            "training_identity": self.training_identity,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> PmoRouteComposer:
        if payload.get("schema_version") != COMPOSER_SCHEMA:
            raise ValueError("pmo route composer schema mismatch")
        return cls(
            MarginalSubgoalPolicy.from_checkpoint(payload["marginal"]),
            ProgramShapeLaw.from_checkpoint(payload["shape"]),
            tuple(StructuralDeltaTemplate.from_payload(row) for row in payload["templates"]),
            payload["held_task"],
            float(payload["shape_weight"]),
            str(payload["training_identity"]),
        )


def _events(template: StructuralDeltaTemplate) -> int:
    from compose_v4.control.structural_subgoal_policy import structural_rewrite_event_count

    return structural_rewrite_event_count(template)


def fit_pmo_route_composer(
    rows: list[ProgramRow],
    *,
    held_task: str | None = None,
    balance_key: str = "task",
    exploration_floor: float = DEFAULT_EXPLORATION_FLOOR,
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
) -> tuple[PmoRouteComposer, dict]:
    """Fit the composer after removing one complete PMO task.

    The held task is excluded before the template vocabulary is derived, so a
    template that occurs only in the held task is absent from the runtime
    vocabulary and is unreachable by construction.  That is the property the
    expressibility measurement reports.
    """
    train = [row for row in rows if row.task != held_task] if held_task else list(rows)
    if not train:
        raise ValueError(f"holding out {held_task!r} left no training programs")
    if held_task and any(row.task == held_task for row in train):
        raise RuntimeError(f"task leakage in {held_task!r} composer")
    marginal_rows = [
        {"source_group": _balance_value(row, balance_key), "templates": row.templates}
        for row in train
    ]
    marginal = fit_marginal_subgoal_policy(marginal_rows, exploration_floor=exploration_floor)
    shape = fit_program_shape_law(
        train, balance_key=balance_key, exploration_floor=exploration_floor
    )
    lookup = {row.template_id: row for group in train for row in group.templates}
    templates = tuple(lookup[name] for name in marginal.template_ids)
    training_identity = identity(
        {
            "schema_version": "pmo_route_composer_fit_v1",
            "marginal": marginal.training_identity,
            "shape": shape.training_identity,
            "held_task": held_task,
            "balance_key": balance_key,
            "shape_weight": shape_weight,
        }
    )
    composer = PmoRouteComposer(
        marginal, shape, templates, held_task, shape_weight, training_identity
    )
    audit = {
        "held_task": held_task,
        "balance_key": balance_key,
        "corpus_programs": len(rows),
        "training_programs": len(train),
        "excluded_held_task_programs": len(rows) - len(train),
        "training_tasks": sorted({row.task for row in train}),
        "held_task_absent_from_training": held_task not in {row.task for row in train},
        "training_lineages": len({row.lineage for row in train if row.lineage}),
        "balance_groups": len({_balance_value(row, balance_key) for row in train}),
        "template_vocabulary": len(templates),
        "exploration_floor": exploration_floor,
        "shape_weight": shape_weight,
        "shape_cells": shape.cells(),
        "region_count_law": list(marginal.goal_count_probabilities),
        "training_identity": training_identity,
    }
    return composer, audit


def uniform_composer_like(composer: PmoRouteComposer) -> PmoRouteComposer:
    """The matched generic arm: same vocabulary and beam, flat frequency laws.

    This is the PMO analogue of T4's ``uniform_same_pool`` control.  It isolates
    the distillation -- the template pool, the binding mechanism, the beam and
    the budget are identical; only the learned frequencies are removed.
    """
    size = len(composer.marginal.template_ids)
    flat = MarginalSubgoalPolicy(
        composer.marginal.template_ids,
        tuple([1 / size] * size),
        (0.25, 0.25, 0.25, 0.25),
        composer.marginal.exploration_floor,
        identity({"schema": "pmo_uniform_marginal_v1", "size": size}),
    )
    cells = len(SCALE_BANDS) * len(MODES)
    flat_shape = ProgramShapeLaw(
        SCALE_BANDS,
        MODES,
        tuple([1 / cells] * cells),
        composer.shape.exploration_floor,
        identity({"schema": "pmo_uniform_shape_v1", "cells": cells}),
    )
    return PmoRouteComposer(
        flat,
        flat_shape,
        composer.templates,
        composer.held_task,
        composer.shape_weight,
        identity({"schema": "pmo_uniform_composer_v1", "base": composer.training_identity}),
    )


def teacher_template_coverage(rows: list[ProgramRow], held_task: str) -> dict:
    """Expressibility: how much of the held task's teacher vocabulary survives.

    A template absent from the training fold cannot be proposed at all, so this
    is a hard ceiling on the distilled proposer, measured before any generation.
    """
    held = [row for row in rows if row.task == held_task]
    train = [row for row in rows if row.task != held_task]
    if not held:
        raise ValueError(f"no held programs for {held_task!r}")
    train_ids = {name for row in train for name in row.template_ids}
    held_ids = Counter(name for row in held for name in row.template_ids)
    covered = {name for name in held_ids if name in train_ids}
    whole = sum(1 for row in held if set(row.template_ids) <= train_ids)
    shapes = {(row.scale, row.mode) for row in held}
    train_shapes = {(row.scale, row.mode) for row in train}
    return {
        "held_task": held_task,
        "held_programs": len(held),
        "held_distinct_templates": len(held_ids),
        "held_template_instances": int(sum(held_ids.values())),
        "covered_distinct_templates": len(covered),
        "distinct_template_coverage": len(covered) / len(held_ids),
        "covered_template_instances": int(sum(held_ids[n] for n in covered)),
        "instance_template_coverage": sum(held_ids[n] for n in covered) / sum(held_ids.values()),
        "fully_expressible_programs": whole,
        "fully_expressible_rate": whole / len(held),
        "held_shapes": sorted(f"{s}|{m}" for s, m in shapes),
        "shape_coverage": len(shapes & train_shapes) / len(shapes),
    }


def restrict_vocabulary(composer: PmoRouteComposer, limit: int) -> tuple[PmoRouteComposer, dict]:
    """Keep the ``limit`` most probable templates and renormalize.

    Generation cost is linear in the vocabulary and every template is rebound
    against the runtime graph, so a bounded run needs a bounded vocabulary.
    This is a BUDGET choice and it lowers expressibility, so the retained
    probability mass is returned and must be reported alongside any result.
    """
    if limit < 1:
        raise ValueError("vocabulary limit must be positive")
    pairs = sorted(
        zip(composer.marginal.template_ids, composer.marginal.probabilities, strict=True),
        key=lambda row: (-row[1], row[0]),
    )
    kept = pairs[:limit]
    if len(kept) == len(pairs):
        return composer, {"restricted": False, "vocabulary": len(pairs), "retained_mass": 1.0}
    names = tuple(name for name, _ in kept)
    mass = np.asarray([value for _, value in kept], dtype=float)
    retained = float(mass.sum())
    mass = mass / retained
    lookup = {row.template_id: row for row in composer.templates}
    marginal = MarginalSubgoalPolicy(
        names,
        tuple(map(float, mass)),
        composer.marginal.goal_count_probabilities,
        composer.marginal.exploration_floor,
        identity(
            {
                "schema": "pmo_restricted_marginal_v1",
                "base": composer.marginal.training_identity,
                "limit": limit,
            }
        ),
    )
    restricted = PmoRouteComposer(
        marginal,
        composer.shape,
        tuple(lookup[name] for name in names),
        composer.held_task,
        composer.shape_weight,
        identity(
            {
                "schema": "pmo_restricted_composer_v1",
                "base": composer.training_identity,
                "limit": limit,
            }
        ),
    )
    return restricted, {
        "restricted": True,
        "vocabulary": len(names),
        "full_vocabulary": len(pairs),
        "retained_mass": retained,
    }
