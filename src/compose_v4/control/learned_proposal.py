"""Return-weighted proposal energy, not a property predictor or Doob value.

The molecular reference and executor are inputs and remain unchanged. A bounded
energy admits lazy exact rejection sampling without enumerating every product.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from compose_v4.control.docking_value import identity
from compose_v4.rewrite.kernel import canonical_state_key

RECIPE = {
    "version": "return_weighted_proposal_nce_v1",
    "dimension": 4096,
    "noise_samples": 4,
    "return_beta": 10.0,
    "regularization": 0.001,
    "max_iterations": 200,
    "energy_bound": math.sqrt(2),
    "reference_mixture": 0.1,
    "seed": 20260924,
    "dtype": "float64",
    "features": "signed_hash_graph_environment_changes_option_region_phase_v1",
}


@lru_cache(maxsize=16384)
def environments(smiles):
    # Parsing is for invariant features only, NEVER exact-state reconstruction.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid proposal feature molecule: {smiles}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseCountFingerprint(mol)
    features = {f"morgan:{k}": float(v) for k, v in fp.GetNonzeroElements().items()}
    features.update(
        {
            "atoms": float(mol.GetNumHeavyAtoms()),
            "bonds": float(mol.GetNumBonds()),
            "aromatic_atoms": float(sum(a.GetIsAromatic() for a in mol.GetAtoms())),
            "cycle_rank": float(mol.GetNumBonds() - mol.GetNumAtoms() + 1),
        }
    )
    return features


@lru_cache(maxsize=262144)
def hash_token(token):
    h = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(h[:4], "little") % RECIPE["dimension"], 1.0 if h[4] & 1 else -1.0


def features(node, option, graph):
    """Canonical graph features; no raw slot IDs, target structures or labels."""
    source = environments(canonical_state_key(node.graph))
    result = defaultdict(float)

    def add(token, value):
        index, sign = hash_token(token)
        result[index] += sign * value

    region = node.region.released_fraction
    contexts = (f"option:{option}", f"scale:{int(region * 4)}")
    if node.stage == "what":
        option_tokens = sorted({option, *option.split(":")})
        for token in option_tokens:
            add(f"what:{token}", 1.0 / len(option_tokens))
            for name, count in source.items():
                add(f"what:{token}:{name}", count / 10 / len(option_tokens))
            add(f"what:{token}:scale", region)
    elif node.stage == "how":
        product = environments(canonical_state_key(graph))
        for name in sorted(source.keys() | product.keys()):
            value = product.get(name, 0) - source.get(name, 0)
            add(f"product:{name}", product.get(name, 0) / 10)
            if value:
                add(f"change:{name}", value / 4)
                for context in contexts:
                    add(f"change:{context}:{name}", value / 4)
                phase = node.active.step / node.active.horizon
                add(f"phase_change:{name}", phase * value / 4)
    else:
        raise ValueError("proposal learning does not change WHERE")
    return {int(k): float(v) for k, v in sorted(result.items()) if v}


def matrix(rows):
    indices, values, indptr = [], [], [0]
    for row in rows:
        for key, value in sorted(row.items(), key=lambda kv: int(kv[0])):
            if not 0 <= int(key) < RECIPE["dimension"] or not np.isfinite(value):
                raise ValueError("malformed proposal feature")
            indices.append(int(key))
            values.append(value)
        indptr.append(len(values))
    return csr_matrix((values, indices, indptr), shape=(len(rows), RECIPE["dimension"]))


def fit(examples, *, data_sha256):
    if not examples or any(len(e["features"]) != 1 + RECIPE["noise_samples"] for e in examples):
        raise ValueError("proposal training requires one observed and four noise alternatives")
    weights = np.asarray([e["weight"] for e in examples], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("proposal training weights must be finite and positive")
    weights /= weights.sum()
    x = matrix([row for e in examples for row in e["features"]])
    width = 1 + RECIPE["noise_samples"]
    penalty = RECIPE["regularization"]

    def loss(theta):
        raw = (x @ theta).reshape(-1, width)
        normalizers = logsumexp(raw, axis=1)
        p = np.exp(raw - normalizers[:, None])
        p[:, 0] -= 1
        return (
            float(weights @ (normalizers - raw[:, 0]) + penalty * (theta @ theta) / 2),
            np.asarray(x.T @ (p * weights[:, None]).ravel()) + penalty * theta,
        )

    optimized = minimize(
        loss,
        np.zeros(RECIPE["dimension"]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": RECIPE["max_iterations"], "ftol": 1e-10, "gtol": 1e-7},
    )
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise RuntimeError(f"proposal fitting failed: {optimized.message}")
    payload = {
        "schema_version": "learned_proposal_v1",
        "recipe": RECIPE,
        "data_sha256": data_sha256,
        "coefficients": optimized.x.tolist(),
        "fit": {
            "decisions": len(examples),
            "objective": float(optimized.fun),
            "initial_objective": math.log(width),
            "iterations": int(optimized.nit),
            "converged": True,
        },
    }
    payload["model_sha256"] = identity(payload)
    return ProposalPolicy(payload)


class ProposalPolicy:
    def __init__(self, payload):
        if (
            payload["recipe"] != RECIPE
            or identity({k: v for k, v in payload.items() if k != "model_sha256"})
            != payload["model_sha256"]
        ):
            raise ValueError("proposal model recipe/hash mismatch")
        self.payload = payload
        self.coefficients = np.asarray(payload["coefficients"], dtype=float)
        if (
            self.coefficients.shape != (RECIPE["dimension"],)
            or not np.isfinite(self.coefficients).all()
        ):
            raise ValueError("invalid proposal coefficients")

    def energy(self, node, option, graph):
        row = features(node, option, graph)
        value = sum(self.coefficients[k] * v for k, v in row.items())
        return float(np.clip(value, -RECIPE["energy_bound"], RECIPE["energy_bound"]))

    def distribution(self, node, row):
        if node.stage != "what":
            raise ValueError("only the existing WHAT row may be reweighted")
        energy = np.asarray([self.energy(node, s.active.option, s.graph) for s in row.successors])
        base = row.reference
        tilted = base * np.exp(energy - energy.max())
        tilted /= tilted.sum()
        epsilon = RECIPE["reference_mixture"]
        q = epsilon * base + (1 - epsilon) * tilted
        kl = float(np.sum(q * np.log(q / base)))
        if not np.isclose(q.sum(), 1) or kl > 1 + 1e-10 or np.any(q < epsilon * base):
            raise RuntimeError("proposal KL/exploration invariant violated")
        return q, {
            "reference": base.tolist(),
            "probabilities": q.tolist(),
            "energy": energy.tolist(),
            "kl": kl,
            "model_sha256": self.payload["model_sha256"],
        }

    def sample(self, hierarchy, node, rng):
        if node.stage != "how":
            raise ValueError("lazy proposal sampling requires HOW state")
        tilted = rng.random() >= RECIPE["reference_mixture"]
        attempts = 0
        while True:
            product = hierarchy.sample_reference(node, rng)
            attempts += 1
            if product is None:
                return None, {"attempts": attempts, "status": "empty_reference"}
            energy = self.energy(node, node.active.option, product.graph)
            if not tilted or rng.random() < math.exp(energy - RECIPE["energy_bound"]):
                return product, {
                    "attempts": attempts,
                    "tilted_component": tilted,
                    "accepted_energy": energy,
                    "kl_upper_bound": 1.0,
                    "model_sha256": self.payload["model_sha256"],
                }
