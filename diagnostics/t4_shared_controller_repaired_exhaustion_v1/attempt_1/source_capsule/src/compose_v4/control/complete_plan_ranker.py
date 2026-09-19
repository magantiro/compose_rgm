"""Frozen endpoint-gain ranker for complete executable plan proposals."""

from __future__ import annotations

import math

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_chronological import state_features

RECIPE = {
    "schema_version": "complete_plan_ranker_recipe_v2",
    "target": "completed_option_score-parent_score",
    "kernel": "pmo_chronological_product_state_kernel_v1",
    "ridge": 1.0,
    "clip_delta": [-1.0, 1.0],
    "uncertainty": "none; max training kernel similarity is applicability only",
    "dtype": "float64",
}


def _validate_example(row: dict) -> tuple[str, float]:
    product = row["product_smiles"]
    parent = float(row["parent_score"])
    score = float(row["score"])
    if (
        not isinstance(product, str)
        or not product
        or not math.isfinite(parent)
        or not math.isfinite(score)
        or not 0 <= parent <= 1
        or not 0 <= score <= 1
    ):
        raise ValueError("invalid scored complete-option example")
    return product, score - parent


def fit(examples: list[dict], *, data_sha256: str) -> dict:
    if not examples or not isinstance(data_sha256, str) or len(data_sha256) != 64:
        raise ValueError("plan ranker requires scored examples and a data identity")
    products, deltas = zip(*(_validate_example(row) for row in examples), strict=True)
    kernel, _ = state_features(list(products))
    deltas = np.asarray(deltas, dtype=np.float64)
    mean = float(deltas.mean())
    coefficients = np.linalg.solve(kernel + np.eye(len(examples)) * RECIPE["ridge"], deltas - mean)
    model = {
        "schema_version": "complete_plan_ranker_v2",
        "recipe": RECIPE,
        "data_sha256": data_sha256,
        "training_products": list(products),
        "mean_delta": mean,
        "coefficients": coefficients.tolist(),
        "training_examples": len(examples),
    }
    model["model_sha256"] = identity(model)
    return model


def validate(model: dict) -> None:
    body = {key: value for key, value in model.items() if key != "model_sha256"}
    coefficients = np.asarray(model.get("coefficients", []), dtype=np.float64)
    if (
        model.get("schema_version") != "complete_plan_ranker_v2"
        or model.get("recipe") != RECIPE
        or identity(body) != model.get("model_sha256")
        or len(model.get("training_products", [])) != model.get("training_examples")
        or coefficients.shape != (model.get("training_examples"),)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("invalid complete-plan ranker")


def predict(model: dict, candidates: list[dict]) -> list[dict]:
    validate(model)
    if not candidates:
        raise ValueError("plan ranker requires at least one candidate")
    products = [row["smiles"] for row in candidates]
    if len(products) != len(set(products)):
        raise ValueError("plan ranker candidates must be canonically distinct")
    training = model["training_products"]
    if set(training) & set(products):
        raise ValueError("prospective plan candidates overlap ranker training products")
    full_kernel, _ = state_features(training + products)
    cross = full_kernel[len(training) :, : len(training)]
    delta = model["mean_delta"] + cross @ np.asarray(model["coefficients"])
    delta = np.clip(delta, *RECIPE["clip_delta"])
    support = cross.max(axis=1)
    result = []
    for row, predicted_delta, maximum_kernel in zip(candidates, delta, support, strict=True):
        parent_score = float(row["parent_score"])
        if not math.isfinite(parent_score) or not 0 <= parent_score <= 1:
            raise ValueError("candidate has invalid paid parent score")
        result.append(
            {
                "candidate_id": row["candidate_id"],
                "worker_id": row["worker_id"],
                "smiles": row["smiles"],
                "source": row["source"],
                "parent_score": parent_score,
                "phase": int(row["phase"]),
                "slot": int(row["slot"]),
                "role": row["role"],
                "release": float(row["release"]),
                "primitive_steps": int(row["primitive_steps"]),
                "predicted_delta": float(predicted_delta),
                "predicted_score": float(np.clip(parent_score + predicted_delta, 0.0, 1.0)),
                "max_training_kernel": float(maximum_kernel),
            }
        )
    return result
