"""Canonical molecular transitions for QED editing from the shared reference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.production_successor_kernel import (
    FactorizedCanonicalSuccessorKernel,
    _one_state_batch,
)
from compose_v4.model.reference_checkpoint import LoadedReference, load_frozen_reference


@dataclass(frozen=True)
class SharedReferenceConfig:
    checkpoint: Path
    checkpoint_sha256: str
    catalog_fingerprint: str
    time: float
    persistent_slots: int = 48
    catalog_path: Path | None = None
    catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time) or not 0.0 <= self.time <= 1.0:
            raise ValueError("reference time must be finite and in [0, 1]")
        if type(self.persistent_slots) is not int or self.persistent_slots < 1:
            raise ValueError("persistent_slots must be a positive integer")
        if (self.catalog_path is None) != (self.catalog_sha256 is None):
            raise ValueError("catalog_path and catalog_sha256 must be supplied together")


class QEDSharedReference:
    """The embedded canonical jump law of one hash-verified frozen model."""

    def __init__(self, reference: LoadedReference, config: SharedReferenceConfig):
        if reference.checkpoint_sha256 != config.checkpoint_sha256:
            raise ValueError("loaded reference differs from the QED configuration")
        if reference.catalog_fingerprint != config.catalog_fingerprint:
            raise ValueError("loaded reference catalog differs from the QED configuration")
        self.reference = reference
        self.config = config
        self.kernel = FactorizedCanonicalSuccessorKernel(reference.model, time=config.time)

    @classmethod
    def load(cls, config: SharedReferenceConfig) -> QEDSharedReference:
        reference = load_frozen_reference(
            config.checkpoint,
            expected_sha256=config.checkpoint_sha256,
            expected_catalog_fingerprint=config.catalog_fingerprint,
            catalog_path=config.catalog_path,
            expected_catalog_sha256=config.catalog_sha256,
        )
        return cls(reference, config)

    def identity(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.reference.checkpoint_sha256,
            "catalog_fingerprint": self.reference.catalog_fingerprint,
            "catalog_sha256": self.reference.catalog_sha256,
            "time": self.config.time,
            "persistent_slots": self.config.persistent_slots,
            "max_active_atoms": self.reference.max_active_atoms,
            "active_atom_boundary": "condition_on_supported_successors",
            "transition_law": "canonical_successor_embedded_jump",
        }

    def _check_state(self, state: MolecularGraph) -> None:
        if state.n_atoms != self.config.persistent_slots:
            raise ValueError(
                f"QED state has {state.n_atoms} slots, expected {self.config.persistent_slots}"
            )
        if not 1 <= state.n_real_atoms <= self.reference.max_active_atoms:
            raise ValueError("QED state is outside the reference's active-atom support")

    def encode(self, state: MolecularGraph) -> np.ndarray:
        """Return the frozen encoder's state embedding for a matched value head."""
        self._check_state(state)
        batch = _one_state_batch(
            self.reference.model,
            state,
            self.config.time,
            prepared_batch=None,
        )
        with torch.no_grad():
            _nodes, global_state, _pairs = self.reference.model._encode_batch(batch)
        embedding = global_state[0].detach().cpu().numpy().astype(np.float64)
        if embedding.shape != (256,) or not np.isfinite(embedding).all():
            raise ValueError("shared reference returned an invalid 256-wide state embedding")
        return embedding

    def successors(self, state: MolecularGraph):
        self._check_state(state)
        candidates = self.kernel.successors(state).successors
        supported = tuple(
            successor
            for successor in candidates
            if successor.state.n_real_atoms <= self.reference.max_active_atoms
        )
        if not supported:
            return ()
        mass = sum(successor.probability for successor in supported)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("supported successor mass must be positive and finite")
        return tuple(
            replace(successor, probability=successor.probability / mass) for successor in supported
        )

    def sample(self, state: MolecularGraph, rng: np.random.Generator) -> MolecularGraph | None:
        """Draw one canonical successor, or return None for an empty fiber."""
        successors = self.successors(state)
        if not successors:
            return None
        probabilities = np.asarray(
            [successor.probability for successor in successors], dtype=np.float64
        )
        if not np.isfinite(probabilities).all() or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=2e-5
        ):
            raise ValueError("canonical successor probabilities do not sum to one")
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(successors), p=probabilities))
        return successors[index].state

    def require_matching_value_head(self, metadata: Mapping[str, Any]) -> None:
        """A value head must name the same transition model it was fitted under."""
        observed = metadata.get("r_theta_sha256")
        if observed != self.reference.checkpoint_sha256:
            raise ValueError(
                "QED value head names a different reference checkpoint: "
                f"{observed!r}, expected {self.reference.checkpoint_sha256}"
            )
