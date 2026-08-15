"""A lightweight five-objective surrogate, fit ONLY on molecules already paid for.

It exists so a navigator can tell whether an edit moved toward a region without
evaluating anything. Every label it learns from was charged to the strict budget
when it was acquired; nothing here can obtain a new one. The oracle's
`navigation_lockout` enforces that at the point of evaluation, so this class does
not have to be trusted -- it has to be correct.

⚠️ THE REPRESENTATION IS AN OPTIMISTIC CONTROL, NOT THE ONE WE PLAN TO DEPLOY.
Morgan bits are free, and they are also exactly what the JNK3 and GSK3B oracles
are random forests over -- so this surrogate is fitting the same function class
on the same features as the oracle it predicts. Measured ranking quality here is
a CEILING. It is adequate for deciding whether the region-targeting MECHANISM is
worth pursuing, and it is not evidence about a surrogate over frozen R_theta
representations. That has to be measured on the representation we will actually
steer with.

Qualified separately before use: `scripts/task3_surrogate_signal.py`. At a few
hundred paid labels it ranks SA well (rho 0.60-0.66) and JNK3/GSK3B poorly
(0.22-0.32), because a random ZINC draw contains almost no actives. That is the
reason the mechanism test seeds its archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Neighbours averaged per prediction. Small, because the regime that matters is
#: a few hundred labels, and there is nothing to tune it on that would not be
#: the very data whose scarcity is the question.
K = 5


@dataclass
class TanimotoKNN:
    """Tanimoto-weighted k-NN over Morgan bits. No fitting step, so updating is
    just appending -- which is what a policy needs as labels arrive."""

    k: int = K
    _fingerprints: list[np.ndarray] = field(default_factory=list, repr=False)
    _values: list[tuple[float, ...]] = field(default_factory=list, repr=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _targets: np.ndarray | None = field(default=None, init=False, repr=False)
    _norms: np.ndarray | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self._values)

    def update(self, smiles: list[str], values: list[tuple[float, ...]]) -> int:
        """Add paid-for labels. Returns how many were actually usable."""

        from compose_v4.benchmark.oracles.forest import morgan_bits

        added = 0
        for item, value in zip(smiles, values):
            row = morgan_bits(item)
            if row is None:
                continue
            self._fingerprints.append(row)
            self._values.append(tuple(float(v) for v in value))
            added += 1
        if added:
            self._matrix = np.vstack(self._fingerprints)
            self._targets = np.asarray(self._values, dtype=float)
            self._norms = self._matrix.sum(axis=1)
        return added

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Predicted five-objective vectors, one row per input molecule.

        Unknown or unparseable molecules get the worst vector rather than a
        neutral one: a surrogate that cannot see a molecule must not make it
        look attractive.
        """

        from compose_v4.benchmark.oracles.forest import morgan_bits

        out = np.zeros((len(smiles), 5), dtype=float)
        if self._matrix is None or not len(self._matrix):
            return out
        rows, keep = [], []
        for position, item in enumerate(smiles):
            row = morgan_bits(item)
            if row is not None:
                rows.append(row)
                keep.append(position)
        if not rows:
            return out
        query = np.vstack(rows)
        intersection = query @ self._matrix.T
        union = (query.sum(axis=1)[:, None] + self._norms[None, :] - intersection)
        similarity = intersection / np.maximum(union, 1e-9)
        k = min(self.k, similarity.shape[1])
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        weights = np.take_along_axis(similarity, top, axis=1)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        out[keep] = np.einsum("ij,ijk->ik", weights, self._targets[top])
        return out
