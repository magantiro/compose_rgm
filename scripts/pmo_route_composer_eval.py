"""Leave-one-TASK-out, zero-oracle evaluation of the PMO route composer.

PMO has many objectives, so the clean analogue of T4's held-target split is to
fit on the successful routes of every other PMO task and evaluate proposal
quality on the held-out task.  Nothing here calls an oracle, scores a molecule,
or reads a task target; the held task name is used only to remove its rows
before fitting and to select its own source graph afterwards.

Three zero-oracle quantities are reported, in increasing cost:

1. EXPRESSIBILITY -- set arithmetic, no generation.  A template absent from the
   training fold cannot be proposed, so held-task template coverage is a hard
   ceiling on the distilled proposer.
2. GENERATION RATE -- how many bound, valid, distinct complete programs the
   beam yields on the held task's own source at a fixed budget, for the
   distilled arm and for the matched uniform arm.
3. TEACHER RANK -- where a teacher-like program first appears in each arm's
   ranking.  "Teacher-like" is reported at two strengths: the exact canonical
   endpoint of a held-task teacher window, and the weaker (scale, mode) shape
   match.

The two arms share the source, the template pool, the binding mechanism, the
beam and the budget.  Only the frequency law differs, which is what isolates
the distillation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from pmo_route_composer import (
    fit_pmo_route_composer,
    restrict_vocabulary,
    teacher_template_coverage,
    uniform_composer_like,
)
from pmo_route_program_corpus import (
    ProgramRow,
    StructuralDeltaTemplate,
    program_mode,
    program_scale,
    template_shape,
)

from compose_v4.control.structural_subgoal_policy import (
    proposal_rewrite_event_count,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_route_composer_leave_task_out_v1"


@dataclass(frozen=True)
class EvalConfig:
    """Every knob that changes a reported number, in one auditable object."""

    window: int = 4
    exploration_floor: float = 0.10
    shape_weight: float = 1.0
    balance_key: str = "task"
    vocabulary_cap: int = 48
    max_bindings_per_template: int = 2
    pool_size: int = 32
    beam_width: int = 6
    expansion_width: int = 6
    scale_balanced: bool = True


def proposal_shape(proposal) -> tuple[str, str]:
    """(scale, mode) of one generated proposal, on the corpus definitions."""
    shapes = [template_shape(row) for row in proposal.templates]
    events = proposal_rewrite_event_count(proposal)
    return (
        program_scale(events),
        program_mode(
            sum(row["created"] for row in shapes), sum(row["released"] for row in shapes)
        ),
    )


def run_arm(composer, source, config: EvalConfig, *, arm: str) -> dict:
    """Generate one equal-budget candidate pool and describe it structurally."""
    began = perf_counter()
    proposals, telemetry = composer.propose(
        source,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        expansion_width=config.expansion_width,
        max_bindings_per_template=config.max_bindings_per_template,
        scale_balanced=config.scale_balanced,
    )
    seconds = perf_counter() - began
    shapes = [proposal_shape(row) for row in proposals]
    endpoints = [canonical_state_key(row.endpoint) for row in proposals]
    return {
        "arm": arm,
        "proposals": len(proposals),
        "distinct_endpoints": len(set(endpoints)),
        "seconds": seconds,
        "scale_distribution": dict(sorted(Counter(scale for scale, _ in shapes).items())),
        "mode_distribution": dict(sorted(Counter(mode for _, mode in shapes).items())),
        "region_count_distribution": dict(
            sorted(Counter(len(row.templates) for row in proposals).items())
        ),
        "rewrite_events_max": max((proposal_rewrite_event_count(r) for r in proposals), default=0),
        "rewrite_events_median": (
            sorted(proposal_rewrite_event_count(r) for r in proposals)[len(proposals) // 2]
            if proposals
            else 0
        ),
        "generation_telemetry": telemetry,
        "_shapes": shapes,
        "_endpoints": endpoints,
    }


def teacher_rank(arm_result: dict, held_rows: list[ProgramRow]) -> dict:
    """Rank of the first teacher-like proposal, at two match strengths."""
    endpoints = arm_result["_endpoints"]
    shapes = arm_result["_shapes"]
    teacher_endpoints = {row.endpoint_key for row in held_rows}
    teacher_shapes = {(row.scale, row.mode) for row in held_rows}
    exact = next((i + 1 for i, key in enumerate(endpoints) if key in teacher_endpoints), None)
    shaped = next((i + 1 for i, shape in enumerate(shapes) if shape in teacher_shapes), None)
    return {
        "teacher_endpoints": len(teacher_endpoints),
        "exact_endpoint_hits": sum(1 for key in endpoints if key in teacher_endpoints),
        "first_exact_endpoint_rank": exact,
        "shape_matched_proposals": sum(1 for shape in shapes if shape in teacher_shapes),
        "first_shape_match_rank": shaped,
        "shape_match_rate": (
            sum(1 for shape in shapes if shape in teacher_shapes) / len(shapes) if shapes else 0.0
        ),
    }


def evaluate_held_task(
    rows: list[ProgramRow], source_state: dict, held_task: str, config: EvalConfig
) -> dict:
    """One complete leave-task-out cell: fit, express, generate, rank."""
    composer, audit = fit_pmo_route_composer(
        rows,
        held_task=held_task,
        balance_key=config.balance_key,
        exploration_floor=config.exploration_floor,
        shape_weight=config.shape_weight,
    )
    coverage = teacher_template_coverage(rows, held_task)
    bounded, restriction = restrict_vocabulary(composer, config.vocabulary_cap)
    uniform = uniform_composer_like(bounded)
    # The corpus payload is already a padded production state; decoding it (not
    # re-parsing SMILES) is what keeps atom_insert inside the legal support.
    source = decode_state(source_state)
    held_rows = [row for row in rows if row.task == held_task]
    distilled = run_arm(bounded, source, config, arm="route_distilled")
    generic = run_arm(uniform, source, config, arm="uniform_same_pool")
    result = {
        "held_task": held_task,
        "fit_audit": audit,
        "expressibility": coverage,
        "vocabulary_restriction": restriction,
        "source_real_atoms": int(source.n_real_atoms),
        "source_slots": int(source.n_atoms),
        "arms": {},
        "teacher_rank": {},
    }
    for arm in (distilled, generic):
        name = arm["arm"]
        result["teacher_rank"][name] = teacher_rank(arm, held_rows)
        result["arms"][name] = {k: v for k, v in arm.items() if not k.startswith("_")}
    return result


def leave_task_out(
    rows: list[ProgramRow],
    sources: dict[str, dict],
    config: EvalConfig,
    *,
    tasks: tuple[str, ...] | None = None,
) -> dict:
    """Run every requested held-task cell and summarize."""
    all_tasks = tuple(sorted({row.task for row in rows}))
    selected = tasks or all_tasks
    cells = []
    for task in selected:
        if task not in sources:
            continue
        cells.append(evaluate_held_task(rows, sources[task], task, config))
    return {
        "schema_version": SCHEMA,
        "config": asdict(config),
        "corpus_programs": len(rows),
        "corpus_tasks": list(all_tasks),
        "evaluated_tasks": [cell["held_task"] for cell in cells],
        "cells": cells,
        "summary": _summarize(cells),
    }


def _summarize(cells: list[dict]) -> dict:
    if not cells:
        return {}

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    return {
        "mean_distinct_template_coverage": mean(
            c["expressibility"]["distinct_template_coverage"] for c in cells
        ),
        "mean_instance_template_coverage": mean(
            c["expressibility"]["instance_template_coverage"] for c in cells
        ),
        "mean_fully_expressible_rate": mean(
            c["expressibility"]["fully_expressible_rate"] for c in cells
        ),
        "mean_proposals_distilled": mean(
            c["arms"]["route_distilled"]["proposals"] for c in cells
        ),
        "mean_proposals_uniform": mean(
            c["arms"]["uniform_same_pool"]["proposals"] for c in cells
        ),
        "exact_endpoint_hits_distilled": sum(
            c["teacher_rank"]["route_distilled"]["exact_endpoint_hits"] for c in cells
        ),
        "exact_endpoint_hits_uniform": sum(
            c["teacher_rank"]["uniform_same_pool"]["exact_endpoint_hits"] for c in cells
        ),
        "mean_shape_match_rate_distilled": mean(
            c["teacher_rank"]["route_distilled"]["shape_match_rate"] for c in cells
        ),
        "mean_shape_match_rate_uniform": mean(
            c["teacher_rank"]["uniform_same_pool"]["shape_match_rate"] for c in cells
        ),
    }


def load_cached_corpus(path: Path) -> tuple[list[ProgramRow], dict[str, dict], dict]:
    """Rehydrate a cached corpus build without re-running the extractor."""
    payload = json.loads(path.read_text())
    rows = [
        ProgramRow(
            task=row["task"],
            route_id=row["route_id"],
            source_group=row["source_group"],
            lineage=row["lineage"],
            task_family=row["task_family"],
            window_index=row["window_index"],
            window_primitives=row["window_primitives"],
            templates=tuple(
                StructuralDeltaTemplate.from_payload(item) for item in row["templates"]
            ),
            region_count=row["region_count"],
            rewrite_events=row["rewrite_events"],
            scale=row["scale"],
            mode=row["mode"],
            created_atoms=row["created_atoms"],
            released_atoms=row["released_atoms"],
            restated_atoms=row["restated_atoms"],
            artifact="",
            success_basis="",
            endpoint_key=row["endpoint_key"],
        )
        for row in payload["rows"]
    ]
    sources: dict[str, dict] = {}
    for route in payload["routes"]:
        sources.setdefault(route["task"], route["source_state"])
    return rows, sources, payload["census"]
