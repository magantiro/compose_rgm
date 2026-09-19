"""Task-trained preference policy over sampled complete executable edits.

This policy ranks parent/product pairs. It is neither a calibrated score model
nor a primitive-step future-value head. Chemistry and oracle I/O live elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from compose_v4.control.continuation import _tilted
from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_chronological import state_features

RECIPE = {
    "version": "complete_edit_preference_v1",
    "kernel": "mean_parent_product_normalized_graph_feature_change",
    "loss": "same_parent_score_gap_weighted_pair_logistic",
    "parent_weighting": "equal_weight_informative_parents",
    "incumbent_comparison": True,
    "regularization": 0.01,
    "eigen_tolerance": 1e-10,
    "max_iterations": 200,
    "kappa": 1.0,
    "exploration": 0.1,
    "uncertainty": "not_estimated",
}


def edit_kernel(left: list[dict], right: list[dict]) -> np.ndarray:
    molecules = sorted({r[k] for r in left + right for k in ("parent_smiles", "smiles")})
    state, _ = state_features(molecules)
    indices = {s: i for i, s in enumerate(molecules)}
    lp, lq = ([indices[r[k]] for r in left] for k in ("parent_smiles", "smiles"))
    rp, rq = ([indices[r[k]] for r in right] for k in ("parent_smiles", "smiles"))
    parent, product = state[np.ix_(lp, rp)], state[np.ix_(lq, rq)]
    change = parent + product - state[np.ix_(lp, rq)] - state[np.ix_(lq, rp)]
    ln = np.sqrt(np.maximum(state[lp, lp] + state[lq, lq] - 2 * state[lp, lq], 0))
    rn = np.sqrt(np.maximum(state[rp, rp] + state[rq, rq] - 2 * state[rp, rq], 0))
    norms = ln[:, None] * rn[None, :]
    difference = np.divide(change, norms, out=np.zeros_like(change), where=norms > 1e-12)
    return (parent + product + difference) / 3


def training_rows(observations: list[dict]) -> list[dict]:
    rows, labels = {}, {}
    for row in observations:
        for molecule, score in (
            (row["parent_smiles"], row["parent_score"]),
            (row["smiles"], row["score"]),
        ):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid observed score: {molecule}")
            if molecule in labels and abs(labels[molecule] - score) > 1e-12:
                raise ValueError(f"conflicting deterministic oracle labels: {molecule}")
            labels[molecule] = score
        rows[row["parent_smiles"], row["smiles"]] = {
            k: row[k] for k in ("parent_smiles", "smiles", "score")
        }
        # The incumbent is a comparison anchor, not a newly introduced self-edit.
        rows[row["parent_smiles"], row["parent_smiles"]] = {
            "parent_smiles": row["parent_smiles"],
            "smiles": row["parent_smiles"],
            "score": row["parent_score"],
        }
    return [rows[key] for key in sorted(rows)]


class BranchPolicy:
    def __init__(self, payload: dict):
        if (
            payload["recipe"] != RECIPE
            or identity({k: v for k, v in payload.items() if k != "model_sha256"})
            != payload["model_sha256"]
        ):
            raise ValueError("branch policy identity/recipe mismatch")
        self.payload = payload

    @classmethod
    def fit(cls, observations: list[dict], *, source_sha256: str):
        rows = training_rows(observations)
        kernel = edit_kernel(rows, rows)
        eigenvalues, vectors = np.linalg.eigh(kernel)
        if eigenvalues.min() < -1e-8:
            raise ValueError("edit kernel is not positive semidefinite")
        positive = eigenvalues > RECIPE["eigen_tolerance"]
        values, vectors = eigenvalues[positive], vectors[:, positive]
        features = vectors * np.sqrt(values)
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["parent_smiles"]].append(i)
        pairs, weights = [], []
        for members in groups.values():
            contrasts = [
                (i, j, rows[i]["score"] - rows[j]["score"])
                for i, j in combinations(members, 2)
                if abs(rows[i]["score"] - rows[j]["score"]) > 1e-12
            ]
            total = sum(abs(gap) for _, _, gap in contrasts)
            for i, j, gap in contrasts:
                pairs.append((i, j) if gap > 0 else (j, i))
                weights.append(abs(gap) / total)
        if not pairs:
            raise ValueError("no informative same-parent comparisons for policy fitting")
        weights = np.asarray(weights)
        weights /= weights.sum()
        design = np.asarray([features[i] - features[j] for i, j in pairs])
        penalty = RECIPE["regularization"]

        def loss(theta):
            margin = design @ theta
            return (
                float(weights @ np.logaddexp(0, -margin) + penalty * (theta @ theta) / 2),
                -(design.T @ (weights * expit(-margin))) + penalty * theta,
            )

        optimized = minimize(
            loss,
            np.zeros(len(values)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": RECIPE["max_iterations"], "ftol": 1e-10, "gtol": 1e-7},
        )
        if not optimized.success or not np.isfinite(optimized.x).all():
            raise RuntimeError(f"branch policy fit failed: {optimized.message}")
        coefficients = vectors @ (optimized.x / np.sqrt(values))
        payload = {
            "schema_version": "branch_policy_v1",
            "recipe": RECIPE,
            "source_sha256": source_sha256,
            "observations_sha256": identity(observations),
            "training_rows": rows,
            "coefficients": coefficients.tolist(),
            "fit": {
                "pairs": len(pairs),
                "parents": len(groups),
                "iterations": optimized.nit,
                "objective": optimized.fun,
                "converged": True,
            },
        }
        payload["model_sha256"] = identity(payload)
        return cls(payload)

    def utilities(self, rows: list[dict]) -> np.ndarray:
        return edit_kernel(rows, self.payload["training_rows"]) @ np.asarray(
            self.payload["coefficients"]
        )

    def distribution(self, rows: list[dict], reference, *, guided: bool):
        base = np.asarray(reference, dtype=float)
        if (
            base.shape != (len(rows),)
            or not len(base)
            or np.any(base <= 0)
            or not np.isclose(base.sum(), 1)
        ):
            raise ValueError("branch policy requires a full-support normalized sampled pool")
        utilities = self.utilities(rows)
        q, eta, kl = base.copy(), 0.0, 0.0
        if guided:
            q, eta, kl = _tilted(
                base,
                np.exp(utilities - utilities.max()),
                kappa=RECIPE["kappa"],
                exploration=RECIPE["exploration"],
            )
        return q, {
            "utilities": utilities.tolist(),
            "reference": base.tolist(),
            "probabilities": q.tolist(),
            "eta": eta,
            "kl": kl,
            "model_sha256": self.payload["model_sha256"],
        }
