"""Optional local endpoint channel; the broad reference/program channels remain."""

from __future__ import annotations

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.donor_memory import proposal_identity
from compose_v4.experiments.pmo_chronological import state_features


def active(seed: int, phase: int, slot: int) -> bool:
    rng = np.random.default_rng(np.random.SeedSequence([seed, phase, slot, 776]))
    # First coin is identical to the existing donor/reference mixture. Never
    # replace donor draws; replace half of reference draws, leaving both broad
    # proposal channels positive. No task label changes these coins.
    return bool(rng.random() >= 0.5 and rng.random() < 0.5)


def policy_identity(memory_id: str, model_id: str) -> str:
    return identity(
        {
            "broad_policy": proposal_identity(memory_id),
            "local_model": model_id,
            "recipe": "half_reference_draws_uniform_among_top16_v1",
        }
    )


def predict(model_lock: dict, candidates: list[str]) -> np.ndarray:
    model = model_lock["model"]
    if (
        identity({k: v for k, v in model.items() if k != "snapshot_sha256"})
        != model["snapshot_sha256"]
    ):
        raise ValueError("local endpoint model hash mismatch")
    train = model_lock["training_smiles"]
    if [model_lock["all_smiles"][i] for i in model["train_indices"]] != train:
        raise ValueError("local endpoint coefficients and training identities differ")
    if not candidates or len(set(candidates)) != len(candidates) or set(train) & set(candidates):
        raise ValueError("local endpoint prediction requires novel distinct candidates")
    kernel, _ = state_features(train + candidates)
    return model["mean"] + kernel[len(train) :, : len(train)] @ np.asarray(model["coefficients"])


def select(smiles: list[str], predictions: np.ndarray, rng: np.random.Generator) -> str:
    if (
        not smiles
        or len(smiles) != len(predictions)
        or len(set(smiles)) != len(smiles)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("invalid local endpoint choice pool")
    ranked = sorted(zip(smiles, predictions, strict=True), key=lambda p: (-p[1], p[0]))
    return ranked[int(rng.integers(min(16, len(ranked))))][0]
