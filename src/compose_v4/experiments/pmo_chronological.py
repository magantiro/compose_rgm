"""Offline, prefix-only PMO prediction diagnostics; not a deployed controller."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.rewrite.kernel import canonical_state_key

RECIPE = {
    "schema_version": "pmo_chronological_recipe_v1",
    "warmup_queries": 20,
    "refresh_queries": 10,
    "ridge": 1.0,
    "option_pseudocount": 2.0,
    "count_names": ["heavy", "cycle_rank", "aromatic_rings", "hbd", "hba"],
    "count_scales": [5, 1, 1, 1, 2],
    "state_kernel": "half_existing_morgan_atompair_plus_half_fixed_count_rbf",
    "context_kernel": "parent_state_times_half_one_plus_option_match_times_half_one_plus_scale_match",
    "edit_kernel": "mean_context_endpoint_normalized_state_feature_difference",
    "center_targets": "training_prefix_mean",
    "clip_endpoints": [0, 1],
    "uncertainty": "not_estimated",
    "new_oracle_calls": 0,
}
MODELS = ("parent_copy", "option_delta", "context_delta", "endpoint_score", "edit_delta")


def prepare_stream(result: dict, roots: list[dict]) -> dict:
    """Recover query order and known parent labels without re-evaluating molecules."""
    if result["status"] != "complete" or result["oracle_calls"] != 100 or len(roots) != 4:
        raise ValueError("expected complete 100-query development run and four root receipts")
    archive = {r["id"]: r for r in result["archive"]}
    for row in archive.values():
        node = decode_search_state(row["node"])
        if canonical_state_key(node.graph) != row["smiles"]:
            raise ValueError(f"exact archive graph disagrees with canonical identity: {row['id']}")
    molecules, scores, index, rows, exclusions = [], [], {}, [], []
    for i, root in enumerate(roots):
        smiles = archive[f"root_{i}"]["smiles"]
        if root["index"] != i or root["status"] != "complete" or root["smiles"] != smiles:
            raise ValueError("root receipt identity/order/status mismatch")
        if smiles in index:
            raise ValueError("duplicate initial canonical identity")
        index[smiles] = len(molecules)
        molecules.append(smiles)
        scores.append(root["score"])
    for attempt in result["attempts"]:
        edge = f"attempts/{attempt['attempt']:04}"
        if attempt["status"] != "complete" or not attempt["new_canonical"]:
            exclusions.append(
                {
                    "id": edge,
                    "reason": attempt["status"]
                    if attempt["status"] != "complete"
                    else "already_charged_canonical",
                }
            )
            continue
        parent = archive[attempt["parent"]]
        if parent["smiles"] not in index or edge not in archive:
            raise ValueError(f"missing earlier parent or novel candidate: {edge}")
        product = archive[edge]["smiles"]
        if product in index or attempt["calls"] != len(molecules) + 1:
            raise ValueError(f"novel query order/identity mismatch: {edge}")
        p, q = index[parent["smiles"]], len(molecules)
        parent_score = scores[p]
        rows.append(
            {
                "id": edge,
                "query": q + 1,
                "parent_index": p,
                "product_index": q,
                "parent_query": p + 1,
                "parent_score": parent_score,
                "score": attempt["score"],
                "option": attempt["option"],
                "scale": attempt["scale"],
            }
        )
        index[product] = q
        molecules.append(product)
        scores.append(attempt["score"])
    if (
        len(molecules) != 100
        or not np.isfinite(scores).all()
        or np.any((np.array(scores) < 0) | (np.array(scores) > 1))
    ):
        raise ValueError("incomplete/nonfinite/out-of-range query stream")
    for i, point in enumerate(result["curve"]):
        prefix = scores[: i + 1]
        if (
            abs(max(prefix) - point["best"]) > 1e-12
            or abs(sum(sorted(prefix, reverse=True)[:10]) / 10 - point["top10_mean"]) > 1e-12
        ):
            raise ValueError(f"recovered labels disagree with locked curve at query {i + 1}")
    return {
        "case": result["case"],
        "smiles": molecules,
        "scores": scores,
        "rows": rows,
        "exclusions": exclusions,
    }


def state_features(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features, counts = [], []
    for value in smiles:
        canonical, fp = molecular_features(value)
        if canonical != value:
            raise ValueError("feature input is not the locked canonical identity")
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError("invalid locked feature molecule")
        features.append(fp)
        counts.append(
            [
                mol.GetNumHeavyAtoms(),
                mol.GetNumBonds() - mol.GetNumAtoms() + 1,
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
            ]
        )
    counts = np.asarray(counts, dtype=np.float64)
    scaled = counts / np.asarray(RECIPE["count_scales"])
    squared = ((scaled[:, None] - scaled[None, :]) ** 2).sum(axis=2)
    return 0.5 * (graph_kernel(features, features) + np.exp(-0.5 * squared)), counts


def edit_kernels(state_kernel: np.ndarray, rows: list[dict]) -> dict[str, np.ndarray]:
    p = np.asarray([r["parent_index"] for r in rows])
    q = np.asarray([r["product_index"] for r in rows])
    parent, product = state_kernel[np.ix_(p, p)], state_kernel[np.ix_(q, q)]
    options = np.asarray([r["option"] for r in rows])
    scales = np.asarray([r["scale"] for r in rows])
    context = parent * (1 + (options[:, None] == options[None, :])) / 2
    context *= (1 + (scales[:, None] == scales[None, :])) / 2
    change = parent + product - state_kernel[np.ix_(p, q)] - state_kernel[np.ix_(q, p)]
    norms = np.sqrt(np.maximum(np.diag(change), 0))
    denominator = norms[:, None] * norms[None, :]
    normalized = np.divide(
        change, denominator, out=np.zeros_like(change), where=denominator > 1e-12
    )
    return {"context_delta": context, "edit_delta": (context + product + normalized) / 3}


def kernel_prediction(kernel, labels, train, test) -> tuple[np.ndarray, dict]:
    train, test = np.asarray(train), np.asarray(test)
    if not len(train) or np.intersect1d(train, test).size:
        raise ValueError("prediction requires nonempty disjoint train/test rows")
    y = np.asarray(labels, dtype=np.float64)[train]
    mean = float(y.mean())
    coefficients = np.linalg.solve(
        kernel[np.ix_(train, train)] + np.eye(len(train)) * RECIPE["ridge"], y - mean
    )
    prediction = mean + kernel[np.ix_(test, train)] @ coefficients
    snapshot = {
        "train_indices": train.tolist(),
        "mean": mean,
        "coefficients": coefficients.tolist(),
    }
    return prediction, {**snapshot, "snapshot_sha256": identity(snapshot)}


def evaluate_stream(stream: dict, state_kernel: np.ndarray, counts: np.ndarray) -> dict:
    rows = stream["rows"]
    queries = np.asarray([r["query"] for r in rows])
    parent_scores = np.asarray([r["parent_score"] for r in rows])
    delta = np.asarray([r["score"] for r in rows]) - parent_scores
    kernels = edit_kernels(state_kernel, rows)
    predictions, snapshots = [], []
    for cutoff in range(RECIPE["warmup_queries"], len(stream["scores"]), RECIPE["refresh_queries"]):
        train = np.flatnonzero(queries <= cutoff)
        test = np.flatnonzero((queries > cutoff) & (queries <= cutoff + RECIPE["refresh_queries"]))
        if not len(test):
            continue
        pooled = float(delta[train].mean())
        groups = defaultdict(list)
        for i in train:
            groups[rows[i]["option"]].append(float(delta[i]))
        values = {"parent_copy": parent_scores[test].copy()}
        values["option_delta"] = np.asarray(
            [
                rows[i]["parent_score"]
                + (sum(groups[rows[i]["option"]]) + RECIPE["option_pseudocount"] * pooled)
                / (len(groups[rows[i]["option"]]) + RECIPE["option_pseudocount"])
                for i in test
            ]
        )
        for name, kernel in kernels.items():
            estimate, snapshot = kernel_prediction(kernel, delta, train, test)
            values[name] = parent_scores[test] + estimate
            snapshots.append({"model": name, "cutoff": cutoff, **snapshot})
        endpoints = np.asarray([rows[i]["product_index"] for i in test])
        values["endpoint_score"], snapshot = kernel_prediction(
            state_kernel, stream["scores"], np.arange(cutoff), endpoints
        )
        snapshots.append({"model": "endpoint_score", "cutoff": cutoff, **snapshot})
        for offset, i in enumerate(test):
            r = rows[i]
            if r["parent_query"] >= r["query"]:
                raise ValueError("candidate uses an unavailable parent label")
            predictions.append(
                {
                    **r,
                    "fit_through_query": cutoff,
                    "cycle_rank_increase": bool(
                        counts[r["product_index"], 1] > counts[r["parent_index"], 1]
                    ),
                    "raw_predictions": {name: float(values[name][offset]) for name in MODELS},
                    "predictions": {
                        name: float(np.clip(values[name][offset], 0, 1)) for name in MODELS
                    },
                }
            )
    return {
        "case": stream["case"],
        "predictions": predictions,
        "snapshots": snapshots,
        "metrics": metrics(predictions),
        "topology_metrics": {
            label: metrics([r for r in predictions if r["cycle_rank_increase"] == flag])
            for label, flag in (("cycle_increase", True), ("other", False))
        },
        "excluded": stream["exclusions"]
        + [
            {"id": r["id"], "reason": "warmup"}
            for r in rows
            if r["query"] <= RECIPE["warmup_queries"]
        ],
    }


def metrics(rows: list[dict]) -> dict:
    result = {}
    for model in MODELS:
        pairs, concordant, sign_n, sign_correct = 0, 0.0, 0, 0.0
        for i, row in enumerate(rows):
            true_delta = row["score"] - row["parent_score"]
            predicted_delta = row["predictions"][model] - row["parent_score"]
            if abs(true_delta) > 1e-12:
                sign_n += 1
                sign_correct += (
                    0.5
                    if abs(predicted_delta) <= 1e-12
                    else float(true_delta * predicted_delta > 0)
                )
            for other in rows[i + 1 :]:
                difference = row["score"] - other["score"]
                if (
                    row["fit_through_query"] != other["fit_through_query"]
                    or abs(difference) <= 1e-12
                ):
                    continue
                pairs += 1
                predicted = row["predictions"][model] - other["predictions"][model]
                concordant += 0.5 if abs(predicted) <= 1e-12 else float(difference * predicted > 0)
        result[model] = {
            "n": len(rows),
            "mae": float(np.mean([abs(r["predictions"][model] - r["score"]) for r in rows]))
            if rows
            else None,
            "ranking_pairs": pairs,
            "concordance": concordant / pairs if pairs else None,
            "sign_n": sign_n,
            "sign_accuracy": sign_correct / sign_n if sign_n else None,
        }
    return result
