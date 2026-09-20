"""Fit and freeze a zero-oracle ranker on locked complete-plan candidates."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compose_v4.control.complete_plan_ranker import fit, predict, validate
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_delayed_credit_audit import (
    _proposal_row as proposal_row,
)
from compose_v4.experiments.pmo_delayed_credit_audit import extract
from compose_v4.experiments.pmo_plan_pool_lock import validate_compiled_lock

BOOTSTRAP_SEED = 20261013
BOOTSTRAP_DRAWS = 4096


def select_by_pool(predictions: list[dict]) -> list[dict]:
    pools = defaultdict(list)
    for row in predictions:
        pools[row["worker_id"]].append(row)
    selected = []
    for key, rows in sorted(pools.items()):
        if len(rows) not in {2, 3}:
            raise ValueError(f"locked pool {key} has {len(rows)} compiled roles")
        selected.append(max(rows, key=lambda row: (row["predicted_score"], row["smiles"])))
    return selected


def build(
    config_path: Path,
    artifact_root: Path,
    compiled_path: Path,
    audit_path: Path,
    output: Path,
    repo_root: Path,
) -> dict:
    config = json.loads(config_path.read_text())
    rows, receipts = extract(config, artifact_root)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema_version") != "pmo_delayed_credit_audit_v2"
        or audit.get("new_oracle_calls") != 0
        or audit.get("provenance", {}).get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError("ranker requires the matching completed delayed-credit audit")
    compiled = json.loads(compiled_path.read_text())
    validate_compiled_lock(compiled)
    training_identity = identity(
        {
            "config_sha256": sha256_file(config_path),
            "inputs": [{"name": row["name"], "sha256": row["sha256"]} for row in receipts],
            "scored_occurrences": len(rows),
        }
    )
    model = fit(rows, data_sha256=training_identity)
    predictions = predict(model, compiled["candidates"])
    selected = select_by_pool(predictions)
    support = np.asarray([row["max_training_kernel"] for row in predictions])
    deltas = np.asarray([row["predicted_delta"] for row in predictions])
    report = {
        "schema_version": "pmo_complete_plan_ranker_lock_v2",
        "evidence_class": "computed_exposed_development_zero_oracle",
        "task": "perindopril_mpo",
        "new_oracle_calls": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "training_inputs": receipts,
        "delayed_credit_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "verdict": audit["verdict"],
        },
        "compiled_lock": {
            "path": str(compiled_path),
            "sha256": sha256_file(compiled_path),
            "content_sha256": compiled["content_sha256"],
        },
        "candidate_count": len(predictions),
        "pool_count": len(selected),
        "predicted_delta": {
            "min": float(deltas.min()),
            "median": float(np.median(deltas)),
            "max": float(deltas.max()),
        },
        "applicability": {
            "metric": "maximum fixed molecular kernel to a scored training product",
            "min": float(support.min()),
            "median": float(np.median(support)),
            "max": float(support.max()),
            "uncertainty_claim": False,
        },
        "selected_role_counts": dict(sorted(Counter(row["role"] for row in selected).items())),
        "predictions": predictions,
        "selected_one_per_pool": selected,
        "interpretation": (
            "Prospectively locked complete-plan endpoint ranking diagnostic. It assigns one score "
            "to each fully executed multi-primitive plan and does not claim a future-value model."
        ),
        "provenance": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
            "implementation_sha256": {
                "ranker": sha256_file(repo_root / "src/compose_v4/control/complete_plan_ranker.py"),
                "analysis": sha256_file(Path(__file__)),
                "entrypoint": sha256_file(repo_root / "tools/pmo_complete_plan_ranker.py"),
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "precision": "float64",
        },
    }
    report["lock_sha256"] = identity(report)
    publish_json(output, report)
    return report


def _safe_spearman(left, right):
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _bootstrap_mean(values, *, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a nonempty finite vector")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "unit": "exact paid parent structure",
        "seed": seed,
        "draws": draws,
    }


def evaluate_temporal_holdout(
    ranker_path: Path, holdout_path: Path, output: Path, repo_root: Path
) -> dict:
    ranker = json.loads(ranker_path.read_text())
    body = {key: value for key, value in ranker.items() if key != "lock_sha256"}
    if ranker.get("schema_version") != "pmo_complete_plan_ranker_lock_v2" or identity(
        body
    ) != ranker.get("lock_sha256"):
        raise ValueError("invalid complete-plan ranker lock")
    model = ranker["model"]
    validate(model)
    sealed = json.loads(holdout_path.read_text())
    if (
        set(sealed) != {"payload", "payload_sha256"}
        or identity(sealed["payload"]) != sealed["payload_sha256"]
    ):
        raise ValueError("invalid sealed temporal holdout")
    result = sealed["payload"]
    if result.get("status") != "complete_development" or result.get("new_oracle_calls") != 58:
        raise ValueError("unexpected complete-plan temporal holdout")
    occurrences = []
    for round_ in result["rounds"]:
        for arm, arm_result in sorted(round_["arms"].items()):
            for slot, proposal in enumerate(arm_result["proposals"]):
                row = proposal_row(
                    "plan_policy_holdout",
                    result["run_id"],
                    arm,
                    round_["boundary"],
                    slot,
                    proposal,
                )
                if row is not None:
                    occurrences.append(row)
    unique = {}
    for row in occurrences:
        previous = unique.setdefault(row["product_smiles"], row)
        if previous["score"] != row["score"]:
            raise ValueError("holdout duplicate product has conflicting scores")
        if previous["parent_smiles"] != row["parent_smiles"]:
            raise ValueError("holdout product was reached from multiple paid parents")
    training = set(model["training_products"])
    overlap = sorted(training & set(unique))
    candidates, scores = [], {}
    for index, row in enumerate(unique.values()):
        if row["product_smiles"] in training:
            continue
        candidate = {
            "candidate_id": f"holdout-{index}",
            "worker_id": row["id"],
            "smiles": row["product_smiles"],
            "source": row["parent_smiles"],
            "parent_score": row["parent_score"],
            "phase": row["boundary"],
            "slot": row["slot"],
            "role": row["arm"],
            "release": row["r_release"],
            "primitive_steps": 0,
        }
        candidates.append(candidate)
        scores[candidate["candidate_id"]] = row["score"]
    predictions = predict(model, candidates)
    rows = []
    for prediction in predictions:
        score = scores[prediction["candidate_id"]]
        rows.append(
            {
                **prediction,
                "score": score,
                "gain": score - prediction["parent_score"],
                "prediction_error": prediction["predicted_score"] - score,
            }
        )
    predicted = np.asarray([row["predicted_score"] for row in rows])
    actual = np.asarray([row["score"] for row in rows])
    predicted_delta = np.asarray([row["predicted_delta"] for row in rows])
    actual_gain = np.asarray([row["gain"] for row in rows])
    parent = np.asarray([row["parent_score"] for row in rows])
    ranked = sorted(rows, key=lambda row: (-row["predicted_score"], row["smiles"]))
    top10 = ranked[: min(10, len(ranked))]
    parent_groups = defaultdict(list)
    for row in rows:
        parent_groups[row["source"]].append(row)
    choices = []
    for group in parent_groups.values():
        if len(group) < 2:
            continue
        selected = max(group, key=lambda row: (row["predicted_score"], row["smiles"]))
        choices.append(
            {
                "selected": selected["score"],
                "oracle": max(row["score"] for row in group),
                "random": float(np.mean([row["score"] for row in group])),
            }
        )
    choice_advantage = (
        None
        if not choices
        else _bootstrap_mean([row["selected"] - row["random"] for row in choices])
    )
    report = {
        "schema_version": "pmo_complete_plan_ranker_temporal_holdout_v2",
        "evidence_class": "retrospective_temporal_holdout_exposed_development",
        "task": "perindopril_mpo",
        "new_oracle_calls": 0,
        "historical_holdout_oracle_calls": result["new_oracle_calls"],
        "ranker_lock": {"path": str(ranker_path), "sha256": sha256_file(ranker_path)},
        "holdout": {
            "path": str(holdout_path),
            "sha256": sha256_file(holdout_path),
            "run_id": result["run_id"],
        },
        "occurrences": len(occurrences),
        "distinct_products": len(unique),
        "training_product_overlap_excluded": len(overlap),
        "evaluated_products": len(rows),
        "metrics": {
            "score_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "parent_score_rmse": float(np.sqrt(np.mean((parent - actual) ** 2))),
            "score_spearman": _safe_spearman(predicted, actual),
            "gain_spearman": _safe_spearman(predicted_delta, actual_gain),
            "actual_parent_improvements": int(np.sum(actual_gain > 1e-12)),
            "predicted_parent_improvements": int(np.sum(predicted_delta > 1e-12)),
            "mean_actual_score": float(actual.mean()),
            "predicted_top10_actual_mean": float(np.mean([row["score"] for row in top10])),
            "predicted_top_actual_score": top10[0]["score"],
            "actual_best_score": float(actual.max()),
            "choice_sets": len(choices),
            "choice_selected_mean": (
                None if not choices else float(np.mean([row["selected"] for row in choices]))
            ),
            "choice_random_mean": (
                None if not choices else float(np.mean([row["random"] for row in choices]))
            ),
            "choice_oracle_mean": (
                None if not choices else float(np.mean([row["oracle"] for row in choices]))
            ),
            "choice_selected_minus_random": choice_advantage,
        },
        "top10_predicted": top10,
        "rows": rows,
        "interpretation_limit": (
            "The holdout predates the ranker fit data but remains exposed development. It tests "
            "complete-plan endpoint discrimination, not autonomous improvement or future value."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "implementation_sha256": {
            "ranker": sha256_file(repo_root / "src/compose_v4/control/complete_plan_ranker.py"),
            "analysis": sha256_file(Path(__file__)),
            "entrypoint": sha256_file(repo_root / "tools/pmo_complete_plan_ranker.py"),
        },
    }
    report["content_sha256"] = identity(report)
    publish_json(output, report)
    return report
