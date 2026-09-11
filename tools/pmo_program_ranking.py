"""Chronological complete-edit ranking check using already-paid trajectories."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.branch_policy import RECIPE, BranchPolicy
from compose_v4.experiments.continuation_profile import publish_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def comparison_rows(result: dict) -> list[dict]:
    if result["status"] != "complete_development":
        raise ValueError("ranking input is not a completed development comparison")
    rows = []
    for rd in result["rounds"]:
        for arm, a in sorted(rd["arms"].items()):
            for slot, child in enumerate(a["proposals"]):
                if child is not None:
                    rows.append(
                        {
                            **{
                                k: child[k]
                                for k in ("parent_smiles", "parent_score", "smiles", "score")
                            },
                            "option": child["bundle"]["option"],
                            "origin": [result["run_id"], rd["boundary"], arm, slot],
                        }
                    )
    return rows


def reconcile(train: list[dict], validation: list[dict]) -> dict:
    """Split before features; no canonical endpoint crosses training/validation."""
    labels = {}
    datasets = []
    for raw in (train, validation):
        edges = {}
        for row in raw:
            for smi, score in (
                (row["parent_smiles"], row["parent_score"]),
                (row["smiles"], row["score"]),
            ):
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"invalid recorded score for {smi}")
                if smi in labels and abs(score - labels[smi]) > 1e-12:
                    raise ValueError(f"conflicting deterministic score for {smi}")
                labels[smi] = score
            key = row["parent_smiles"], row["smiles"]
            entry = edges.setdefault(
                key,
                {
                    **{k: row[k] for k in ("parent_smiles", "parent_score", "smiles", "score")},
                    "options": [],
                    "origins": [],
                },
            )
            entry["options"] = sorted(set(entry["options"]) | {row["option"]})
            entry["origins"].append(row["origin"])
        datasets.append([edges[k] for k in sorted(edges)])
    train, validation = datasets
    train_molecules = {row[k] for row in train for k in ("parent_smiles", "smiles")}
    kept, excluded = [], []
    for row in validation:
        overlap = sorted({row[k] for k in ("parent_smiles", "smiles")} & train_molecules)
        if overlap:
            excluded.append({**row, "reason": "training_endpoint_overlap", "overlap": overlap})
        else:
            kept.append(row)
    return {"train": train, "validation": kept, "excluded_validation": excluded}


def census(rows):
    options = defaultdict(Counter)
    for row in rows:
        gap = row["score"] - row["parent_score"]
        sign = "positive" if gap > 1e-12 else "negative" if gap < -1e-12 else "tied"
        for option in row["options"]:
            options[option][sign] += 1
    return {
        "edges": len(rows),
        "parents": len({r["parent_smiles"] for r in rows}),
        "by_option": dict(sorted(options.items())),
    }


def ranking_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_smiles"]].append(row)
    choices, pairs = [], []
    for parent, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        best = min(members, key=lambda r: (-r["utility"], r["smiles"]))
        uniform_score = float(np.mean([r["score"] for r in members]))
        choices.append(
            {
                "parent_smiles": parent,
                "candidates": len(members),
                "parent_score": members[0]["parent_score"],
                "selected_smiles": best["smiles"],
                "selected_score": best["score"],
                "uniform_expected_score": uniform_score,
                "best_available_score": max(r["score"] for r in members),
                "selected_minus_uniform": best["score"] - uniform_score,
            }
        )
        for left, right in combinations(members, 2):
            actual = left["score"] - right["score"]
            if abs(actual) <= 1e-12:
                continue
            predicted = left["utility"] - right["utility"]
            pairs.append(0.5 if abs(predicted) <= 1e-12 else float(predicted * actual > 0))
    positive = [r["score"] > r["parent_score"] + 1e-12 for r in rows]
    selected_positive = [r["utility"] > r["parent_utility"] + 1e-12 for r in rows]
    true_positive = sum(a and b for a, b in zip(positive, selected_positive, strict=True))
    return {
        "rows": len(rows),
        "parents": len(groups),
        "multi_candidate_parents": len(choices),
        "informative_pairs": len(pairs),
        "pair_preference_precision": float(np.mean(pairs)) if pairs else None,
        "observed_improvements": sum(positive),
        "predicted_improvements": sum(selected_positive),
        "positive_precision": true_positive / sum(selected_positive)
        if any(selected_positive)
        else None,
        "positive_recall": true_positive / sum(positive) if any(positive) else None,
        "mean_choice_gain_over_uniform": float(
            np.mean([r["selected_minus_uniform"] for r in choices])
        )
        if choices
        else None,
        "groups": choices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    for name in ("first", "replicate", "archive"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.probe / "batch.json",
        args.probe / "result.json",
        args.first,
        args.replicate,
        args.archive,
    ]
    material = {str(p.resolve()): sha256_file(p) for p in paths}
    batch, probe, first, replicate, archive = [json.loads(p.read_text()) for p in paths]
    parents = {i: r for i, r in enumerate(batch["parents"])}
    parity = {r["parent"]: r["score"] for r in probe["parity"]}
    train = []
    for row in probe["attempts"]:
        if row["status"] == "compiled":
            train.append(
                {
                    "parent_smiles": parents[row["parent"]]["smiles"],
                    "parent_score": parity[row["parent"]],
                    "smiles": row["smiles"],
                    "score": row["score"],
                    "option": "donor_transplant",
                    "origin": ["isolated_probe", row["index"]],
                }
            )
    train.extend(comparison_rows(first))
    split = reconcile(train, comparison_rows(replicate) + comparison_rows(archive))
    implementation = [
        "tools/pmo_program_ranking.py",
        "docs/PMO_PROGRAM_RANKING_DIAGNOSTIC.md",
        "src/compose_v4/control/branch_policy.py",
        "src/compose_v4/control/docking_value.py",
        "src/compose_v4/experiments/pmo_chronological.py",
    ]
    frozen = {
        "schema_version": "program_ranking_split_v1",
        "inputs_sha256": material,
        "implementation_sha256": {p: sha256_file(ROOT / p) for p in implementation},
        "recipe": RECIPE,
        "split": split,
        "census": {k: census(v) for k, v in split.items()},
        "role": "exposed chronological development; endpoint-disjoint, not scaffold-disjoint",
    }
    split_path = args.output / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != frozen:
        raise ValueError("frozen ranking inputs/split/recipe changed")
    publish_json(split_path, frozen)
    # No feature construction or fitting occurs until the split is published.
    started = perf_counter()
    model = BranchPolicy.fit(split["train"], source_sha256=sha256_file(split_path))
    fit_seconds = perf_counter() - started
    publish_json(args.output / "model.json", model.payload)
    rows = split["validation"]
    anchors = [{"parent_smiles": r["parent_smiles"], "smiles": r["parent_smiles"]} for r in rows]
    prediction_start = perf_counter()
    utilities = model.utilities(rows + anchors)
    predicted = [
        {**r, "utility": float(u), "parent_utility": float(p)}
        for r, u, p in zip(rows, utilities[: len(rows)], utilities[len(rows) :], strict=True)
    ]
    result = {
        "schema_version": "program_ranking_report_v1",
        "split_sha256": sha256_file(split_path),
        "model_sha256": sha256_file(args.output / "model.json"),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")},
        },
        "hardware": {
            "platform": platform.platform(),
            "workers": 1,
            "precision": "float64",
            "threads": 1,
        },
        "randomness": "deterministic model fit and canonical tie-breaking, no draws",
        "fit_seconds": fit_seconds,
        "prediction_seconds": perf_counter() - prediction_start,
        "new_oracle_calls": 0,
        "new_reference_calls": 0,
        "fit": model.payload["fit"],
        "census": frozen["census"],
        "all_options": ranking_metrics(predicted),
        "donor_options": ranking_metrics(
            [r for r in predicted if "donor_transplant" in r["options"]]
        ),
        "predictions": predicted,
        "interpretation": frozen["role"],
    }
    publish_json(args.output / "report.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("predictions", "census", "worktree_status")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
