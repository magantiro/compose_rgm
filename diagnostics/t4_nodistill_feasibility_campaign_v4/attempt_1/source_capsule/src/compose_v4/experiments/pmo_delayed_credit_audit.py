"""Audit learnable delayed credit in scored complete-option genealogies.

The fitted quantities are witnessed behavior-policy returns. They are neither an
endpoint property oracle nor an exact reference-process backward value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_chronological import state_features

SCHEMA = "pmo_delayed_credit_audit_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_score(value, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} is not a finite PMO score in [0,1]")
    return value


def option_family(option: str) -> str:
    if not isinstance(option, str) or not option:
        raise ValueError("proposal lacks an option identity")
    prefix = "replace_region:"
    replaced = option.startswith(prefix)
    core = option[len(prefix) :] if replaced else option
    if core.startswith("construct:pendant:"):
        core = "construct_pendant"
    elif core.startswith("construct:fused:"):
        core = "construct_fused"
    else:
        core = core.split(":", 1)[0]
    return f"replace_region_{core}" if replaced else core


def _proposal_row(name, run_id, arm, boundary, slot, proposal):
    if proposal is None:
        return None
    required = {
        "id",
        "smiles",
        "parent_smiles",
        "score",
        "parent_score",
        "bundle",
        "chain",
        "node",
    }
    missing = sorted(required - proposal.keys())
    if missing:
        raise ValueError(f"{name}/{arm}/{boundary}/{slot} lacks {missing}")
    chain = proposal["chain"]
    if not isinstance(chain, list) or not chain or chain[-1] != proposal["id"]:
        raise ValueError(f"{name}/{arm}/{boundary}/{slot} has an invalid chain")
    node = proposal["node"]
    root = node.get("root_id") if isinstance(node, dict) else None
    bundle = proposal["bundle"]
    option = bundle.get("option")
    if not isinstance(root, str) or not root:
        raise ValueError(f"{name}/{arm}/{boundary}/{slot} lacks an exact root identity")
    region = bundle.get("region") or {}
    release = float(bundle.get("r_release", 0.0))
    if not math.isfinite(release) or not 0 <= release <= 1:
        raise ValueError(f"{name}/{arm}/{boundary}/{slot} has invalid release scale")
    return {
        "run_name": name,
        "run_id": run_id,
        "arm": arm,
        "boundary": int(boundary),
        "slot": int(slot),
        "id": proposal["id"],
        "root_id": root,
        "parent_smiles": proposal["parent_smiles"],
        "product_smiles": proposal["smiles"],
        "parent_score": _finite_score(proposal["parent_score"], "parent score"),
        "score": _finite_score(proposal["score"], "product score"),
        "option": option,
        "option_family": option_family(option),
        "r_release": release,
        "region_interface": str(region.get("interface", "none")),
        "region_kind": str(region.get("kind", "none")),
        "chain": chain,
    }


def extract(config: dict, artifact_root: Path) -> tuple[list[dict], list[dict]]:
    rows, receipts = [], []
    for source in config["inputs"]:
        path = artifact_root / source["local_path"]
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise ValueError(f"input hash mismatch for {path}: {actual}")
        result = json.loads(path.read_text())
        if result.get("status") not in {"complete", "complete_development"}:
            raise ValueError(f"{path} is not a complete development result")
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{path} lacks a run id")
        before = len(rows)
        for round_ in result.get("rounds", []):
            boundary = int(round_["boundary"])
            for arm, arm_result in sorted(round_["arms"].items()):
                proposals = arm_result.get("proposals")
                if proposals is None:
                    continue
                for slot, proposal in enumerate(proposals):
                    row = _proposal_row(source["name"], run_id, arm, boundary, slot, proposal)
                    if row is not None:
                        rows.append(row)
        receipts.append(
            {
                "name": source["name"],
                "run_id": run_id,
                "local_path": str(path),
                "volume_path": source["volume_path"],
                "sha256": actual,
                "scored_occurrences": len(rows) - before,
            }
        )
    if not rows:
        raise ValueError("no scored complete-option occurrences found")
    return rows, receipts


def attach_returns(rows: list[dict], horizons: list[int]) -> None:
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[(row["run_id"], row["arm"])].append(row)
    for trajectory in by_trajectory.values():
        for row in trajectory:
            descendants = [
                other
                for other in trajectory
                if other["boundary"] >= row["boundary"] and row["id"] in other["chain"]
            ]
            if not any(other is row for other in descendants):
                raise ValueError("a proposal is absent from its own witnessed lineage")
            row["returns"] = {}
            for horizon in horizons:
                eligible = [
                    other["score"]
                    for other in descendants
                    if other["boundary"] <= row["boundary"] + horizon
                ]
                witnessed = max(eligible)
                # The parent stays available in these archive/population optimizers.
                row["returns"][str(horizon)] = max(row["parent_score"], witnessed)


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def leakage_groups(rows: list[dict]) -> dict[str, str]:
    roots = sorted({row["root_id"] for row in rows})
    groups = _UnionFind(roots)
    owners = defaultdict(set)
    for row in rows:
        owners[row["parent_smiles"]].add(row["root_id"])
        owners[row["product_smiles"]].add(row["root_id"])
    for shared in owners.values():
        first, *rest = sorted(shared)
        for other in rest:
            groups.union(first, other)
    components = defaultdict(list)
    for root in roots:
        components[groups.find(root)].append(root)
    return {
        root: identity({"roots": members})[:16]
        for members in components.values()
        for root in members
    }


def _context_factor(rows: list[dict], bandwidth: float) -> np.ndarray:
    exact = np.asarray([row["option"] for row in rows])
    family = np.asarray([row["option_family"] for row in rows])
    interface = np.asarray([row["region_interface"] for row in rows])
    categorical = (
        0.5
        + 0.25 * (exact[:, None] == exact[None, :])
        + 0.15 * (family[:, None] == family[None, :])
        + 0.10 * (interface[:, None] == interface[None, :])
    )
    scale = np.asarray([row["r_release"] for row in rows])
    scale_kernel = np.exp(-0.5 * ((scale[:, None] - scale[None, :]) / bandwidth) ** 2)
    return categorical * scale_kernel


def context_kernel(rows: list[dict], bandwidth: float) -> np.ndarray:
    parents = [row["parent_smiles"] for row in rows]
    state, _ = state_features(parents)
    return state * _context_factor(rows, bandwidth)


def product_state_kernel(rows: list[dict]) -> np.ndarray:
    products = [row["product_smiles"] for row in rows]
    state, _ = state_features(products)
    return state


def realized_edit_kernel(rows: list[dict], bandwidth: float) -> np.ndarray:
    molecules = sorted(
        {row["parent_smiles"] for row in rows} | {row["product_smiles"] for row in rows}
    )
    index = {smiles: i for i, smiles in enumerate(molecules)}
    state, _ = state_features(molecules)
    parent_index = np.asarray([index[row["parent_smiles"]] for row in rows])
    product_index = np.asarray([index[row["product_smiles"]] for row in rows])
    parent = state[np.ix_(parent_index, parent_index)]
    product = state[np.ix_(product_index, product_index)]
    parent_product = state[np.ix_(parent_index, product_index)]
    product_parent = state[np.ix_(product_index, parent_index)]
    change = parent + product - parent_product - product_parent
    norms = np.sqrt(np.maximum(np.diag(change), 0))
    denominator = norms[:, None] * norms[None, :]
    normalized_change = np.divide(
        change, denominator, out=np.zeros_like(change), where=denominator > 1e-12
    )
    context = parent * _context_factor(rows, bandwidth)
    return (context + product + normalized_change) / 3


def _rmse(prediction, target) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def _rank_correlation(left, right) -> float | None:
    from scipy.stats import spearmanr

    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _cluster_interval(values_by_group, *, seed: int, draws: int = 4096):
    groups = sorted(values_by_group)
    values = np.asarray(
        [float(np.mean(values_by_group[group])) for group in groups], dtype=np.float64
    )
    if not len(values):
        return {"groups": 0, "mean": None, "ci95": [None, None], "draws": draws}
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "groups": len(values),
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "draws": draws,
    }


def _choice_metrics(rows, target, future_prediction, endpoint_prediction, leakage_group, *, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["run_id"], row["arm"], row["boundary"], row["parent_smiles"])
        groups[key].append(index)
    eligible = []
    for indices in groups.values():
        unique = {}
        for index in indices:
            unique.setdefault(rows[index]["product_smiles"], index)
        indices = list(unique.values())
        if len(indices) >= 2 and np.ptp(np.asarray(target)[indices]) > 1e-12:
            eligible.append(indices)

    def summarize(prediction):
        selected, oracle, random = [], [], []
        selected_indices = []
        for indices in eligible:
            chosen = max(indices, key=lambda i: (prediction[i], rows[i]["product_smiles"]))
            selected_indices.append(chosen)
            selected.append(target[chosen])
            oracle.append(max(target[i] for i in indices))
            random.append(float(np.mean([target[i] for i in indices])))
        return {
            "mean_selected_return": None if not selected else float(np.mean(selected)),
            "mean_oracle_return": None if not oracle else float(np.mean(oracle)),
            "mean_random_return": None if not random else float(np.mean(random)),
            "mean_regret": (
                None if not selected else float(np.mean(np.asarray(oracle) - np.asarray(selected)))
            ),
        }, selected_indices

    future_summary, future_indices = summarize(future_prediction)
    endpoint_summary, endpoint_indices = summarize(endpoint_prediction)
    paired = defaultdict(list)
    for future_index, endpoint_index in zip(future_indices, endpoint_indices, strict=True):
        if leakage_group[future_index] != leakage_group[endpoint_index]:
            raise ValueError("one choice set crosses leakage groups")
        paired[leakage_group[future_index]].append(
            float(target[future_index] - target[endpoint_index])
        )

    return {
        "choice_sets": len(eligible),
        "future_model": future_summary,
        "endpoint_model": endpoint_summary,
        "paired_future_minus_endpoint": _cluster_interval(paired, seed=seed),
    }


def _kernel_prediction(kernel, labels, train, test, ridge):
    train, test = np.asarray(train), np.asarray(test)
    if not len(train) or np.intersect1d(train, test).size:
        raise ValueError("prediction requires nonempty disjoint train/test rows")
    y = np.asarray(labels, dtype=np.float64)[train]
    mean = float(y.mean())
    coefficients = np.linalg.solve(
        kernel[np.ix_(train, train)] + np.eye(len(train)) * ridge, y - mean
    )
    return mean + kernel[np.ix_(test, train)] @ coefficients


def _evaluate_kernel(rows, config, kernel, groups, folds):
    parent = np.asarray([row["parent_score"] for row in rows])
    endpoint = np.maximum(parent, [row["score"] for row in rows])
    endpoint_gain = endpoint - parent
    endpoint_gain_prediction = np.empty(len(rows))
    future_gain_predictions = {str(h): np.empty(len(rows)) for h in config["horizons"]}
    ridge = float(config["ridge"])
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    for fold in folds:
        train = np.asarray(fold["train_indices"])
        test = np.asarray(fold["test_indices"])
        endpoint_gain_prediction[test] = _kernel_prediction(
            kernel, endpoint_gain, train, test, ridge
        )
        for horizon in config["horizons"]:
            target = np.asarray([row["returns"][str(horizon)] for row in rows])
            future_gain_predictions[str(horizon)][test] = _kernel_prediction(
                kernel, target - parent, train, test, ridge
            )
    endpoint_prediction = parent + np.clip(endpoint_gain_prediction, 0, 1 - parent)
    metrics = {}
    for horizon in config["horizons"]:
        key = str(horizon)
        target = np.asarray([row["returns"][key] for row in rows])
        future = parent + np.clip(future_gain_predictions[key], 0, 1 - parent)
        temporary = np.asarray([row["score"] < row["parent_score"] - 1e-12 for row in rows])
        recovered = temporary & (target > parent + 1e-12)
        squared_advantage = (endpoint_prediction - target) ** 2 - (future - target) ** 2
        mse_by_group = defaultdict(list)
        for index, value in enumerate(squared_advantage):
            mse_by_group[groups[index]].append(float(value))
        metrics[key] = {
            "target_mean": float(target.mean()),
            "parent_rmse": _rmse(parent, target),
            "endpoint_model_rmse": _rmse(endpoint_prediction, target),
            "future_model_rmse": _rmse(future, target),
            "endpoint_model_spearman": _rank_correlation(endpoint_prediction, target),
            "future_model_spearman": _rank_correlation(future, target),
            "endpoint_gain_spearman": _rank_correlation(
                endpoint_prediction - parent, target - parent
            ),
            "future_gain_spearman": _rank_correlation(future - parent, target - parent),
            "paired_mse_advantage_future_over_endpoint": _cluster_interval(
                mse_by_group, seed=20260926 + horizon
            ),
            "parent_improving_branches": int(np.sum(target > parent + 1e-12)),
            "temporary_loss_branches": int(temporary.sum()),
            "delayed_recoveries": int(recovered.sum()),
            "delayed_recovery_fraction_of_losses": float(recovered.sum() / temporary.sum()),
            "choice_ranking": _choice_metrics(
                rows,
                target,
                future.tolist(),
                endpoint_prediction.tolist(),
                groups,
                seed=20261000 + horizon,
            ),
        }
    return metrics


def evaluate(rows: list[dict], config: dict) -> dict:
    mapping = leakage_groups(rows)
    groups = np.asarray([mapping[row["root_id"]] for row in rows])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise ValueError("fewer than three leakage-connected root groups")
    folds = []
    for group in unique_groups:
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        folds.append(
            {
                "group": group,
                "train": len(train),
                "test": len(test),
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
            }
        )
    bandwidth = float(config["scale_bandwidth"])
    kernels = {
        "option_context": context_kernel(rows, bandwidth),
        "product_state": product_state_kernel(rows),
        "realized_edit": realized_edit_kernel(rows, bandwidth),
    }
    models = {
        name: _evaluate_kernel(rows, config, kernel, groups, folds)
        for name, kernel in kernels.items()
    }
    roots = sorted({row["root_id"] for row in rows})
    state_owners = defaultdict(set)
    for row in rows:
        state_owners[row["parent_smiles"]].add(row["root_id"])
        state_owners[row["product_smiles"]].add(row["root_id"])
    return {
        "scored_occurrences": len(rows),
        "distinct_run_arms": len({(row["run_id"], row["arm"]) for row in rows}),
        "distinct_roots": len(roots),
        "leakage_connected_groups": len(unique_groups),
        "identities_shared_across_roots": sum(len(owners) > 1 for owners in state_owners.values()),
        "option_family_counts": dict(
            sorted(
                (family, sum(row["option_family"] == family for row in rows))
                for family in {row["option_family"] for row in rows}
            )
        ),
        "folds": [{key: fold[key] for key in ("group", "train", "test")} for fold in folds],
        "models": models,
    }


def analyze(config: dict, artifact_root: Path) -> dict:
    if (
        config.get("schema_version") != "pmo_delayed_credit_audit_config_v1"
        or config.get("task") != "perindopril_mpo"
        or config.get("new_oracle_calls") != 0
    ):
        raise ValueError("delayed-credit audit configuration changed")
    horizons = config.get("horizons")
    if horizons != sorted(set(horizons)) or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("invalid option horizons")
    rows, receipts = extract(config, artifact_root)
    attach_returns(rows, horizons)
    evaluation = evaluate(rows, config)
    primary_model = "realized_edit"
    primary = evaluation["models"][primary_model][str(max(horizons))]
    future_rmse = primary["future_model_rmse"]
    endpoint_rmse = primary["endpoint_model_rmse"]
    ranking = primary["choice_ranking"]
    future_regret = ranking["future_model"]["mean_regret"]
    endpoint_regret = ranking["endpoint_model"]["mean_regret"]
    mse_interval = primary["paired_mse_advantage_future_over_endpoint"]["ci95"]
    choice_interval = ranking["paired_future_minus_endpoint"]["ci95"]
    if primary["delayed_recoveries"] < 10:
        verdict = "insufficient_natural_delayed_recovery_supervision"
    elif (
        future_rmse < endpoint_rmse
        and mse_interval[0] is not None
        and mse_interval[0] > 0
        and future_regret is not None
        and endpoint_regret is not None
        and future_regret < endpoint_regret
        and choice_interval[0] is not None
        and choice_interval[0] > 0
    ):
        verdict = "witnessed_future_return_adds_robust_out_of_root_signal"
    elif future_rmse < endpoint_rmse and (
        future_regret is None or endpoint_regret is None or future_regret < endpoint_regret
    ):
        verdict = "witnessed_future_return_lift_is_directional_not_robust"
    else:
        verdict = "witnessed_future_return_not_better_than_endpoint_signal"
    return {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": config["evidence_class"],
        "task": config["task"],
        "new_oracle_calls": 0,
        "configuration": {
            "horizons": horizons,
            "ridge": config["ridge"],
            "scale_bandwidth": config["scale_bandwidth"],
            "target": "max(parent_stop_score, witnessed_descendant_scores_within_horizon)",
            "models": ["option_context", "product_state", "realized_edit"],
            "primary_model": primary_model,
            "validation": "leave-one-leakage-connected-root-out kernel ridge",
            "endpoint_comparator": "same model and folds trained only on one-option endpoint return",
        },
        "inputs": receipts,
        "summary": evaluation,
        "verdict": verdict,
        "limitations": [
            "Returns are maxima witnessed under an exposed mixture of prior development controllers.",
            "They are not expectations under the frozen reference process and are not exact Doob h values.",
            "The public winner is not an input, but several warm starts and one prior head had winner exposure.",
            "Cross-validation withholds leakage-connected roots; it is retrospective development, not a sealed benchmark.",
            "A predictive pass would authorize a frozen controller probe, not a performance claim.",
        ],
    }


def publish(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(config_path: Path, artifact_root: Path, output: Path, repo_root: Path) -> dict:
    config = json.loads(config_path.read_text())
    result = analyze(config, artifact_root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result["provenance"] = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "code_revision": revision,
        "implementation_sha256": {
            "analysis": sha256_file(Path(__file__)),
            "entrypoint": sha256_file(repo_root / "tools/pmo_delayed_credit_audit.py"),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hardware": platform.machine(),
        "precision": "float64",
    }
    publish(output, result)
    return result
