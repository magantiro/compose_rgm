"""A cheap student of the frozen five-objective oracle.

NOT a replacement oracle. The benchmark's objective functions are immutable
artifacts; this only approximates them from labels we have already purchased,
so COMPOSE can inspect many speculative molecules without spending budget. A
surrogate value may steer the search and can never enter the reported Pareto
front -- only measured molecules do.

Tanimoto k-NN over Morgan fingerprints, deliberately. It has no training step,
so "update the surrogate" is appending a labelled point, which makes the L1 and
L4 arms share an update rule exactly rather than approximately. A fitted model
would introduce optimisation noise that differs between arms purely because
they see different label counts, confounding the very comparison being run.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TanimotoKNN"]


class TanimotoKNN:
    def __init__(self, n_objectives: int, k: int = 5, radius: int = 2,
                 n_bits: int = 2048) -> None:
        self.k = int(k)
        self.n_obj = int(n_objectives)
        self.radius, self.n_bits = int(radius), int(n_bits)
        self._keys: list[str] = []
        self._fps: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._seen: set[str] = set()
        self._gen = None

    def _fp(self, smiles: str):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        if self._gen is None:
            self._gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.n_bits)
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        arr = np.frombuffer(
            bytes(self._gen.GetFingerprintAsNumPy(m).astype(np.uint8)),
            dtype=np.uint8).astype(np.float32)
        return arr

    def fit_add(self, smiles: str, y) -> None:
        """Append one measured label. Idempotent per canonical string."""
        if smiles in self._seen:
            return
        f = self._fp(smiles)
        if f is None:
            return
        self._seen.add(smiles)
        self._keys.append(smiles)
        self._fps.append(f)
        self._y.append(np.asarray(y, dtype=float))
        self._M = None

    @property
    def n_labels(self) -> int:
        return len(self._keys)

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """Tanimoto-weighted mean of the k nearest measured molecules."""
        out = np.zeros((len(smiles_list), self.n_obj), dtype=float)
        if not self._fps:
            return out
        M = np.vstack(self._fps)                    # (n, bits)
        Y = np.vstack(self._y)                      # (n, obj)
        m_norm = M.sum(axis=1)
        for i, s in enumerate(smiles_list):
            f = self._fp(s)
            if f is None:
                continue
            inter = M @ f
            union = m_norm + f.sum() - inter
            tan = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            k = min(self.k, len(tan))
            idx = np.argpartition(-tan, k - 1)[:k]
            w = tan[idx]
            if w.sum() <= 0:                        # no overlap at all
                out[i] = Y.mean(axis=0)
            else:
                out[i] = (w[:, None] * Y[idx]).sum(axis=0) / w.sum()
        return out
