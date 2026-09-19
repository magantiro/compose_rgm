"""Context-conditioned replay of observed beneficial complete donor edits.

This is a nonparametric proposal policy, not a future-value model. Its finite
experience memory does not define COMPOSE's molecular or primitive support.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.donor_program import PendantCut, pendant_cuts, transplant_plan
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

RECIPE = {
    "schema_version": "contextual_edit_replay_v1",
    "reference_exploration": 0.2,
    "wl_radius": 2,
    "context_smoothing": 0.05,
    "context_power": 2,
    "endpoint_beta": 10.0,
    "positive_tolerance": 1e-12,
    "weighting": "positive_gain_normalized_per_source_times_exp_endpoint_score",
    "support": "original_uniform_donor_channel_plus_observed_donor_cut_replay",
}


def cut_from_payload(value: dict) -> PendantCut:
    return PendantCut(value["anchor"], value["root"], tuple(value["component"]))


def cut_features(graph, cut: PendantCut) -> dict:
    """Permutation-invariant local graph features, never reconstructed replay states."""
    if cut not in pendant_cuts(graph):
        raise ValueError("edit-replay feature cut is not an oriented single bridge")
    real = set(map(int, np.flatnonzero(is_element(graph.atom_types))))
    removed = set(cut.component)

    def fingerprints(atoms, port, *, rooted):
        labels = {
            i: identity(
                [
                    int(graph.atom_types[i]),
                    int(graph.formal_charges[i]),
                    int(graph.implicit_h_counts[i]),
                    i == port,
                ]
            )
            for i in atoms
        }
        result = Counter()
        for radius in range(RECIPE["wl_radius"] + 1):
            result.update(f"{radius}:{labels[i]}" for i in ([port] if rooted else sorted(atoms)))
            labels = {
                i: identity(
                    [
                        labels[i],
                        sorted(
                            (int(graph.bonds[i, j]), labels[j]) for j in atoms if graph.bonds[i, j]
                        ),
                    ]
                )
                for i in atoms
            }
        return dict(sorted(result.items()))

    return {
        "context": fingerprints(real - removed, cut.anchor, rooted=True),
        "removed": fingerprints(removed, cut.root, rooted=False),
    }


def count_tanimoto(left: dict, right: dict) -> float:
    keys = left.keys() | right.keys()
    denominator = sum(max(left.get(k, 0), right.get(k, 0)) for k in keys)
    return (
        sum(min(left.get(k, 0), right.get(k, 0)) for k in keys) / denominator
        if denominator
        else 1.0
    )


def fit_replay(rows: list[dict]) -> dict:
    """Fit from unique, scored source/product edges, retaining exclusion provenance."""
    unique, excluded, labels = {}, [], {}
    for row in rows:
        for smi, score in (
            (row["parent_smiles"], row["parent_score"]),
            (row["smiles"], row["score"]),
        ):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid replay score")
            if smi in labels and abs(labels[smi] - score) > 1e-12:
                raise ValueError("conflicting replay oracle labels")
            labels[smi] = score
        key = row["parent_smiles"], row["smiles"]
        if key in unique:
            unique[key]["origins"].append(row["origin"])
        else:
            unique[key] = {**row, "origins": [row["origin"]]}
    entries = []
    parent_gain = defaultdict(float)
    for key, row in sorted(unique.items()):
        gain = row["score"] - row["parent_score"]
        if gain <= RECIPE["positive_tolerance"]:
            excluded.append(
                {
                    "edge": list(key),
                    "origins": row["origins"],
                    "reason": "nonpositive_gain",
                    "gain": gain,
                }
            )
            continue
        source, donor = decode_state(row["source"]), decode_state(row["donor"])
        a, b = cut_from_payload(row["source_cut"]), cut_from_payload(row["donor_cut"])
        plan = transplant_plan(source, donor, a, b)
        if (
            canonical_state_key(source) != row["parent_smiles"]
            or plan["status"] != "planned"
            or canonical_state_key(plan["target"]) != row["smiles"]
        ):
            raise ValueError("scored replay edge does not match its exact operand graphs")
        entries.append({**row, "gain": gain, "features": cut_features(source, a)})
        parent_gain[row["parent_smiles"]] += gain
    if not entries:
        raise ValueError("no observed positive donor edits for replay")
    weights = np.array(
        [
            r["gain"]
            / parent_gain[r["parent_smiles"]]
            * np.exp(RECIPE["endpoint_beta"] * r["score"])
            for r in entries
        ]
    )
    weights /= weights.sum()
    payload = {
        "schema_version": "edit_replay_bank_v1",
        "recipe": RECIPE,
        "entries": entries,
        "weights": weights.tolist(),
        "excluded": excluded,
        "raw_rows": len(rows),
        "unique_edges": len(unique),
        "positive_parents": len(parent_gain),
    }
    payload["bank_sha256"] = identity(payload)
    return payload


class EditReplay:
    def __init__(self, payload: dict):
        if (
            payload["recipe"] != RECIPE
            or identity({k: v for k, v in payload.items() if k != "bank_sha256"})
            != payload["bank_sha256"]
        ):
            raise ValueError("edit-replay recipe or bank identity mismatch")
        self.payload = payload

    def distribution(self, graph):
        cuts = pendant_cuts(graph)
        if not cuts:
            return cuts, np.empty((0, len(self.payload["entries"])))
        features = [cut_features(graph, cut) for cut in cuts]
        smoothing = RECIPE["context_smoothing"]
        kernel = np.array(
            [
                [
                    np.prod(
                        [
                            smoothing
                            + (1 - smoothing) * count_tanimoto(feature[k], entry["features"][k])
                            for k in ("context", "removed")
                        ]
                    )
                    ** RECIPE["context_power"]
                    for entry in self.payload["entries"]
                ]
                for feature in features
            ]
        )
        mass = kernel * np.array(self.payload["weights"])[None, :]
        return cuts, mass / mass.sum()

    def draw(self, graph, donor_graphs, rng, *, mode, cached=None):
        if mode not in ("uniform", "replay"):
            raise ValueError(f"unknown donor proposal mode {mode}")
        cuts, probabilities = self.distribution(graph) if cached is None else cached
        if not cuts:
            return None
        if mode == "uniform" or rng.random() < RECIPE["reference_exploration"]:
            i = int(rng.integers(len(donor_graphs)))
            right = pendant_cuts(donor_graphs[i])
            if not right:
                return None
            return {
                "component": "uniform",
                "donor_index": i,
                "source_cut": cuts[int(rng.integers(len(cuts)))],
                "donor_cut": right[int(rng.integers(len(right)))],
            }
        at = int(rng.choice(probabilities.size, p=probabilities.ravel()))
        cut_index, entry_index = np.unravel_index(at, probabilities.shape)
        entry = self.payload["entries"][entry_index]
        return {
            "component": "replay",
            "donor_index": entry["donor_index"],
            "source_cut": cuts[cut_index],
            "donor_cut": cut_from_payload(entry["donor_cut"]),
            "experience_index": int(entry_index),
            "conditional_pair_probability": float(probabilities[cut_index, entry_index]),
        }
