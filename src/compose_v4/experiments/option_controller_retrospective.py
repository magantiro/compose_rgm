"""Offline gate for the option-controller value target using stored trajectories.

The first supported input is the completed PMO option-particle result.  Only the
reference arm is eligible because it records continuations from the declared
balanced option law.  Optimized arms are not silently relabeled as reference
rollouts.
"""

from __future__ import annotations

import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.improvement_value import (
    ImprovementFitConfig,
    ImprovementTrace,
    build_targets,
    censored_targets,
    evaluate_improvement,
    fit_improvement_model,
    predict_improvement,
    source_group_split,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.inference_package import software

HORIZONS = (1, 2, 4, 6)
THRESHOLDS = (0.005, 0.02, 0.05)
SEED = 20260912


def _validate(data: dict) -> dict:
    configuration = data["configuration"]
    if configuration.get("schema_version") != "option_particles_contract_v1":
        raise ValueError("retrospective requires the option-particle result schema")
    body = {key: value for key, value in configuration.items() if key != "contract_sha256"}
    if identity(body) != configuration.get("contract_sha256"):
        raise ValueError("stored option-particle contract identity mismatch")
    if "reference" not in configuration["arms"] or not data["rounds"]:
        raise ValueError("stored result lacks reference continuations")
    if len(data["rounds"]) != configuration["boundaries"]:
        raise ValueError("stored result did not reach its declared option horizon")
    return configuration


def _records(data: dict) -> tuple[list[dict], str]:
    configuration = _validate(data)
    nodes = {row["id"]: {**row, "boundary": 0} for row in data["initial_parents"]}
    for round_row in data["rounds"]:
        for row in round_row["arms"]["reference"]["proposals"]:
            if row is not None:
                nodes[row["id"]] = {**row, "boundary": round_row["boundary"]}
    initial_best = max(row["score"] for row in data["initial_parents"])
    incumbent = {0: initial_best}
    for row in data["rounds"]:
        incumbent[row["boundary"]] = row["arms"]["reference"]["best"]
    records = []
    final = data["rounds"][-1]["arms"]["reference"]["proposals"]
    for terminal in final:
        if terminal is None:
            continue
        lineage = [nodes[key] for key in terminal["chain"] if key in nodes]
        if not lineage or lineage[-1]["id"] != terminal["id"]:
            raise ValueError("terminal option ancestry is incomplete")
        boundaries = [row["boundary"] for row in lineage]
        if boundaries != list(range(boundaries[0], configuration["boundaries"] + 1)):
            raise ValueError("terminal option ancestry skips a completed boundary")
        source_id = lineage[0]["node"]["root_id"]
        for start, row in enumerate(lineage):
            records.append(
                {
                    "source_id": source_id,
                    "trace_id": f"{terminal['id']}@{row['id']}",
                    "smiles": row["smiles"],
                    "score": row["score"],
                    "scores": tuple(value["score"] for value in lineage[start:]),
                    "incumbent": incumbent[row["boundary"]],
                    "remaining": configuration["boundaries"] - row["boundary"],
                }
            )
    policy_id = identity(
        {
            "role": "balanced_option_reference_continuations",
            "contract_sha256": configuration["contract_sha256"],
            "arm": "reference",
        }
    )
    return records, policy_id


def _target_census(records: list[dict]) -> dict:
    known = positive = 0
    total = len(records) * len(HORIZONS) * len(THRESHOLDS)
    for row in records:
        label, mask = censored_targets(
            row["scores"], row["incumbent"], HORIZONS, THRESHOLDS, terminated=False
        )
        known += int(mask.sum())
        positive += int((label * mask).sum())
    return {
        "rows": len(records),
        "cells": total,
        "identified_cells": known,
        "identified_coverage": known / total if total else 0,
        "positive_identified_cells": positive,
    }


def analyze(data: dict, *, input_path: Path, repo_root: Path) -> dict:
    records, policy_id = _records(data)
    source_ids = [row["source_id"] for row in records]
    configuration = data["configuration"]
    target_census = _target_census(records)
    report = {
        "schema_version": "option_controller_retrospective_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "retrospective stored-trajectory development gate; no new oracle calls",
        "task": configuration["task"],
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=repo_root, text=True
        ),
        "software": software(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "configuration": {
            "arm": "reference",
            "behavior_policy_id": policy_id,
            "horizons": list(HORIZONS),
            "thresholds": list(THRESHOLDS),
            "seed": SEED,
            "new_oracle_calls": 0,
            "input_historical_calls": data["new_oracle_calls"],
        },
        "trajectory_census": {
            **target_census,
            "source_groups": len(set(source_ids)),
            "terminal_lineages": sum(
                row is not None for row in data["rounds"][-1]["arms"]["reference"]["proposals"]
            ),
            "option_counts": dict(
                sorted(
                    Counter(
                        row["bundle"]["option"]
                        for round_row in data["rounds"]
                        for row in round_row["arms"]["reference"]["proposals"]
                        if row is not None
                    ).items()
                )
            ),
        },
    }
    if not 0 < target_census["positive_identified_cells"] < target_census["identified_cells"]:
        return {
            **report,
            "status": "abstained_no_improvement_contrast",
            "fit": None,
            "evaluation": None,
            "decision": {
                "claim": "stored trajectories cannot identify improvement ranking or calibration",
                "reason": "the eligible reference continuations contain no positive target cells",
                "next": (
                    "retain the controller core; collect policy-congruent option continuations "
                    "with both improving and non-improving outcomes before fitting"
                ),
            },
        }
    if len(set(source_ids)) < 2:
        return {
            **report,
            "status": "abstained_insufficient_source_groups",
            "fit": None,
            "evaluation": None,
            "decision": {
                "claim": "stored trajectories cannot support source-grouped value validation",
                "reason": "all eligible option-boundary continuations descend from one source",
                "next": (
                    "retain the controller core; obtain policy-congruent continuations from at "
                    "least two additional development sources before fitting or live guidance"
                ),
            },
        }

    train_indices, calibration_indices = source_group_split(source_ids, seed=SEED)
    traces = []
    for row in records:
        features = tuple(molecule_features(row["smiles"])) + (
            float(row["score"]),
            float(row["incumbent"]),
            float(row["remaining"] / configuration["boundaries"]),
        )
        traces.append(
            ImprovementTrace(
                row["source_id"],
                row["trace_id"],
                policy_id,
                features,
                row["scores"],
                row["incumbent"],
                False,
            )
        )
    targets = build_targets(traces, HORIZONS, THRESHOLDS, behavior_policy_id=policy_id)
    model, fit = fit_improvement_model(
        targets.subset(train_indices), ImprovementFitConfig(seed=SEED)
    )
    calibration = targets.subset(calibration_indices)
    evaluation = evaluate_improvement(calibration, predict_improvement(model, calibration.features))
    return {
        **report,
        "status": "evaluated",
        "split": {
            "train_sources": sorted(set(np.asarray(source_ids)[train_indices])),
            "calibration_sources": sorted(set(np.asarray(source_ids)[calibration_indices])),
        },
        "fit": fit,
        "evaluation": evaluation,
        "decision": {
            "claim": "retrospective calibration only; no autonomous benchmark improvement",
            "next": "apply the prospective decision thresholds in OPTION_CONTROLLER_V1.md",
        },
    }
