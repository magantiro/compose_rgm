"""Task-outcome ranking of complete edit plans, not a future-value model.

Every scored plan still needs compilation and executor validation. The learned
policy acts on proposal marks; it is not claimed to preserve the learned molecular
law. A positive base mixture preserves support among the supplied plans.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.linalg import solve

from compose_v4.chem.molecular_graph import BOND_CLASS_TO_ORDER, ELEMENTS, is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_replay import cut_features, cut_from_payload
from compose_v4.rewrite.trace_shard import decode_state

RECIPE = {
    "schema_version": "complete_edit_chooser_v1",
    "ridge": 1.0,
    "bag_hash_width": 8192,
    "kernel": "mean_three_count_dot_tanimotos_joint_product_and_fixed_numeric_rbf",
    "numeric_count_scale": 8.0,
    "source_weighting": "equal_total_weight_per_canonical_parent",
    "reference_exploration": 0.2,
    "temperature": 0.05,
    "target": "observed_complete_endpoint_score_minus_observed_parent_score",
    "completion_target": "observed_compiler_status_only_not_oracle_reward",
    "uncertainty": "not_estimated",
}


def _counts(graph, slots):
    slots = sorted(slots)
    atoms = Counter(int(graph.atom_types[i]) for i in slots)
    bonds = Counter(
        float(graph.bonds[i, j])
        for pos, i in enumerate(slots)
        for j in slots[pos + 1 :]
        if graph.bonds[i, j]
    )
    # Type IDs are executor attributes, not a learned or pruned vocabulary.
    return (
        [float(atoms[i]) for i in range(len(ELEMENTS))]
        + [float(bonds[b]) for b in BOND_CLASS_TO_ORDER if b != 0]
        + [float(len(slots)), float(sum(bonds.values()) - len(slots) + bool(slots))]
    )


@dataclass
class EditFeatures:
    """Bounded-lifetime cache shared by an input batch, not hidden global state."""

    graphs: dict = field(default_factory=dict)
    cuts: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def _graph(self, payload):
        key = identity(payload)
        if key not in self.graphs:
            self.graphs[key] = decode_state(payload)
        return key, self.graphs[key]

    def _cut(self, payload, cut_payload):
        key, graph = self._graph(payload)
        cut = cut_from_payload(cut_payload)
        cache_key = key, cut
        if cache_key not in self.cuts:
            self.cuts[cache_key] = cut_features(graph, cut)
        if cache_key not in self.counts:
            self.counts[cache_key] = _counts(graph, cut.component)
        return key, graph, cut, self.cuts[cache_key], self.counts[cache_key]

    def __call__(self, row):
        key, graph, left, source, removed = self._cut(row["source"], row["source_cut"])
        _, _, _, donor, added = self._cut(row["donor"], row["donor_cut"])
        if key not in self.counts:
            self.counts[key] = _counts(graph, np.flatnonzero(is_element(graph.atom_types)))
        numeric = np.asarray(self.counts[key] + removed + added, dtype=np.float64)
        numeric /= RECIPE["numeric_count_scale"]
        numeric = np.r_[numeric, len(left.component) / graph.n_real_atoms]
        return {
            "bags": [source["context"], source["removed"], donor["removed"]],
            "numeric": numeric.tolist(),
        }


def _bag_matrix(features, index):
    rows, columns, values = [], [], []
    for i, feature in enumerate(features):
        for key, value in feature["bags"][index].items():
            rows.append(i)
            columns.append(int(identity([key])[:16], 16) % RECIPE["bag_hash_width"])
            values.append(value)
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(features), RECIPE["bag_hash_width"]), dtype=np.float64
    )


def edit_kernel(left, right):
    if not left or not right:
        return np.empty((len(left), len(right)))
    components = []
    for index in range(3):
        a, b = _bag_matrix(left, index), _bag_matrix(right, index)
        dot = (a @ b.T).toarray()
        denom = (
            np.asarray(a.multiply(a).sum(axis=1)) + np.asarray(b.multiply(b).sum(axis=1)).T - dot
        )
        components.append(np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0))
    a = np.asarray([r["numeric"] for r in left])
    b = np.asarray([r["numeric"] for r in right])
    distance = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
    components.extend([np.prod(components, axis=0), np.exp(-distance / 2)])
    return np.mean(components, axis=0)


def source_weights(parents):
    counts = Counter(parents)
    return np.array([len(parents) / (len(counts) * counts[p]) for p in parents])


def fit_regression(kernel, targets, parents):
    y = np.asarray(targets, dtype=np.float64)
    if kernel.shape != (len(y), len(y)) or not len(y) or not np.isfinite(y).all():
        raise ValueError("edit chooser needs finite targets and a nonempty square kernel")
    weights = source_weights(parents)
    mean = float(np.average(y, weights=weights))
    alpha = solve(kernel + np.diag(RECIPE["ridge"] / weights), y - mean, assume_a="pos")
    return {"mean": mean, "alpha": alpha.tolist()}


def fit_chooser(rows, features, *, input_identity):
    if len(rows) != len(features) or not rows:
        raise ValueError("chooser rows and features must be nonempty and aligned")
    scored = [i for i, r in enumerate(rows) if r["score"] is not None]
    if not scored:
        raise ValueError("no paid completed edits to fit")
    for r in rows:
        if r["parent_score"] is None:
            raise ValueError("paid source score required for complete-edit gain target")
        if r["status"] != "compiled" and r["score"] is not None:
            raise ValueError("failed compilation cannot have a fabricated oracle target")
        for value in (r["parent_score"], r["score"]):
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("chooser score outside declared normalized task domain")
    matrix = edit_kernel(features, features)
    reward = fit_regression(
        matrix[np.ix_(scored, scored)],
        [rows[i]["score"] - rows[i]["parent_score"] for i in scored],
        [rows[i]["parent_smiles"] for i in scored],
    )
    completion = fit_regression(
        matrix,
        [r["status"] == "compiled" for r in rows],
        [r["parent_smiles"] for r in rows],
    )
    payload = {
        "recipe": RECIPE,
        "input_identity": input_identity,
        "features": features,
        "reward_indices": scored,
        "reward": reward,
        "completion": completion,
        "training_rows": [r["row_id"] for r in rows],
        "training_counts": {
            "attempts": len(rows),
            "scored": len(scored),
            "improved": sum(rows[i]["score"] > rows[i]["parent_score"] + 1e-12 for i in scored),
            "parents": len({r["parent_smiles"] for r in rows}),
        },
    }
    return {**payload, "model_sha256": identity(payload)}


@dataclass(frozen=True)
class EditChooser:
    payload: dict

    def __post_init__(self):
        if (
            self.payload["recipe"] != RECIPE
            or identity({k: v for k, v in self.payload.items() if k != "model_sha256"})
            != self.payload["model_sha256"]
        ):
            raise ValueError("edit chooser recipe or model identity mismatch")

    def predict(self, features, *, batch_size=256):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive integer batch_size required")
        gains, completions = [], []
        for start in range(0, len(features), batch_size):
            matrix = edit_kernel(features[start : start + batch_size], self.payload["features"])
            gains.extend(
                self.payload["reward"]["mean"]
                + matrix[:, self.payload["reward_indices"]] @ self.payload["reward"]["alpha"]
            )
            completions.extend(
                self.payload["completion"]["mean"] + matrix @ self.payload["completion"]["alpha"]
            )
        return np.asarray(gains), np.clip(completions, 0, 1)

    @staticmethod
    def distribution(gains, base):
        gains, base = np.asarray(gains, dtype=float), np.asarray(base, dtype=float)
        if gains.shape != base.shape or gains.ndim != 1 or not gains.size:
            raise ValueError("aligned nonempty one-dimensional scores and base required")
        if not np.isfinite(gains).all() or not np.isfinite(base).all() or (base <= 0).any():
            raise ValueError("finite scores and strictly positive base mass required")
        base = base / base.sum()
        mass = base * np.exp(np.maximum((gains - gains.max()) / RECIPE["temperature"], -700))
        epsilon = RECIPE["reference_exploration"]
        return epsilon * base + (1 - epsilon) * mass / mass.sum()
