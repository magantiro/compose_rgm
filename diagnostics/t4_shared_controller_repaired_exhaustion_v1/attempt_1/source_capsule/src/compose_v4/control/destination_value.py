"""What good chemistry looks like for one target, learned without any route.

Three levels of evidence are easy to confuse, so they are named here and kept apart:

1. TEACHER REPLAY -- the winning trajectory is in training and the model reproduces it.
   A debugging aid. It is not autonomous evidence and must never be reported as such.
2. DESTINATION-SUPERVISED NAVIGATION -- the model is shown scored molecules for the
   target, including good ones, with genealogy and ordering REMOVED. It knows what good
   territory looks like; it was never told how to reach it. Whether search can navigate
   there is then a property of the controller, not of the supervision.
3. WINNER-BLIND AUTONOMY -- the good region is withheld too. This is the clean claim.

This module implements the value used at level 2 and level 3. `fit` sees a bag of
(molecule, score) pairs in shuffled order: no parents, no children, no call indices, no
trajectory. `strip_genealogy` enforces that structurally rather than by convention,
because a single leaked parent field would turn a navigation result into a replay result
and the two are indistinguishable from the output.

The value is deliberately NOT structural similarity to a known winner. Similarity to
scored endpoints was already measured on this branch and is not a dependable utility
label; a molecule can sit close to a winner and dock badly. What is fitted here is the
score itself, as a function of whole-molecule descriptors.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "destination_value_v1"

# Fields that would let a value model reconstruct search order or parentage. Anything
# here is removed before fitting, so a navigation claim cannot quietly become a replay.
GENEALOGY_FIELDS = (
    "genealogical_parent",
    "parent_score",
    "parent_probability",
    "query",
    "ancestral_primitives",
    "entry_id",
    "receipt_id",
    "run",
    "replicate",
    "checkpoint",
    "input_smiles",
    "input_state_sha256",
    "input_endpoint_similarity",
    "rule_counts",
    "primitive_count",
    "net_created",
    "net_deleted",
    "changed_slots",
    "changed_slot_count",
    "ring_count_change",
    "block_labels",
)


def strip_genealogy(rows, rng=None) -> list[dict]:
    """A shuffled bag of (molecule, score). Everything path-shaped is dropped."""
    stripped = [{k: v for k, v in row.items() if k not in GENEALOGY_FIELDS} for row in rows]
    order = (rng or np.random.default_rng(0)).permutation(len(stripped))
    return [stripped[i] for i in order]


def assert_route_blind(rows) -> None:
    """Fail loudly if any path-shaped field survived into the training bag."""
    leaked = sorted({field for row in rows for field in GENEALOGY_FIELDS if field in row})
    if leaked:
        raise ValueError(f"destination value must be route-blind; leaked fields: {leaked}")


def fit(features, scores, *, penalty: float = 1e-2) -> dict:
    """Ridge regression onto the docking score. Convex, closed form, no seed.

    Scores are negated so that a HIGHER value means a better molecule, which keeps every
    downstream comparison in one direction.
    """
    features = np.asarray(features, dtype=float)
    target = -np.asarray(scores, dtype=float)
    design = np.column_stack([features, np.ones(len(features))])
    ridge = penalty * np.eye(design.shape[1])
    ridge[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.T @ design / len(target) + ridge, design.T @ target / len(target)
    )
    residual = design @ weights - target
    return {
        "schema_version": SCHEMA_VERSION,
        "weights": weights,
        "penalty": penalty,
        "molecules": len(target),
        "train_rmse": float(np.sqrt(residual @ residual / len(target))),
    }


def value(model, features) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return np.column_stack([features, np.ones(len(features))]) @ model["weights"]


def withhold_region(rows, *, threshold: float) -> list[dict]:
    """Level 3: remove the good region itself, so the model has never seen the answer."""
    return [row for row in rows if row["score"] > threshold]
