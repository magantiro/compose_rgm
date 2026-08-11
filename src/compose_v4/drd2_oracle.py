"""The classic DRD2 activity oracle, evaluated in plain numpy.

This is the RBF-SVM behind the similarity-constrained DRD2 benchmark (Olivecrona
et al.'s REINVENT classifier, redistributed with wengong-jin/iclr19-graph2graph
and used to produce the VJTNN and GrIDDD success rates).  The upstream artifact
is a Python-3.6 sklearn pickle; `scripts/drd2_oracle_extract.py` opens it once
and freezes the parameters, and this module evaluates them directly.  Nothing
here imports sklearn or unpickles anything, so the oracle cannot drift when the
environment does.

The arithmetic is libsvm's, reproduced exactly:

    K(x, s)  = exp(-gamma ||x - s||^2)
    d(x)     = sum_i a_i K(x, s_i) + b            (libsvm's decision value)
    P(active | x) = 1 / (1 + exp(A d(x) + B))     (Platt scaling)

The Platt orientation is not inferred from first principles -- libsvm's sign
convention for the binary case is easy to get backwards and a flipped oracle
would still look plausible.  It is DETERMINED against the original estimator on
a fixed molecule panel by `scripts/drd2_oracle_parity.py` and then pinned in the
manifest, so a mistake shows up as a parity failure rather than as a silently
inverted objective.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

#: Morgan radius for the ORACLE fingerprint (counts, features). Not the same
#: fingerprint as the similarity constraint -- see `tanimoto_to`.
ORACLE_RADIUS = 3
ORACLE_SIZE = 2048
#: Morgan radius / bits for the SIMILARITY constraint (plain ECFP4 bits).
SIMILARITY_RADIUS = 2
SIMILARITY_BITS = 2048


def oracle_fingerprint(smiles: str, size: int = ORACLE_SIZE) -> np.ndarray | None:
    """Count-based FCFP6 folded by modulo, exactly as the benchmark defines it.

    Radius 3, ``useCounts=True``, ``useFeatures=True``, folded with ``% size`` on
    the raw Morgan identifier.  Any deviation -- bit vectors, radius 2, plain
    ECFP -- yields a different oracle whose scores are not comparable to any
    published DRD2 number.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprint(
        mol, ORACLE_RADIUS, useCounts=True, useFeatures=True)
    folded = np.zeros(size, dtype=np.float64)
    for identifier, value in fingerprint.GetNonzeroElements().items():
        folded[identifier % size] += float(value)
    return folded


def tanimoto_to(reference_smiles: str, smiles: str) -> float:
    """ECFP4-bit Tanimoto, the similarity used by the constrained benchmark.

    Deliberately a different fingerprint from the oracle's: the benchmark scores
    activity with count-based FCFP6 and constrains similarity with ECFP4 bits.
    """

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    a = Chem.MolFromSmiles(reference_smiles)
    b = Chem.MolFromSmiles(smiles)
    if a is None or b is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(
        a, SIMILARITY_RADIUS, nBits=SIMILARITY_BITS, useChirality=False)
    fb = AllChem.GetMorganFingerprintAsBitVect(
        b, SIMILARITY_RADIUS, nBits=SIMILARITY_BITS, useChirality=False)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _sigmoid_predict(decision: np.ndarray, prob_a: float,
                     prob_b: float) -> np.ndarray:
    """libsvm's ``sigmoid_predict``, including its branch for numerical stability."""

    z = prob_a * np.asarray(decision, dtype=np.float64) + prob_b
    return np.where(z >= 0.0,
                    np.exp(-z) / (1.0 + np.exp(-z)),
                    1.0 / (1.0 + np.exp(z)))


def _binary_probability_coupling(sigmoid: np.ndarray,
                                 max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """libsvm's ``multiclass_probability`` specialised to k=2, reproduced exactly.

    The exact fixed point of this iteration is the Platt sigmoid itself, so it
    looks redundant -- but libsvm stops as soon as ``max_error < 0.005/k``,
    which leaves a residual of up to ~2e-3 relative to the true fixed point.
    That early-stopped iterate, not the sigmoid, is what ``predict_proba``
    returns and therefore what every published DRD2 number was measured
    against.  At a 0.5 success threshold the difference decides borderline
    molecules, so the iteration is reproduced rather than short-circuited.
    """

    sigmoid = np.atleast_1d(np.asarray(sigmoid, dtype=np.float64))
    r01 = sigmoid
    r10 = 1.0 - sigmoid
    q00, q11 = r10 * r10, r01 * r01
    q01 = -r10 * r01
    q10 = q01

    p0 = np.full(sigmoid.shape, 0.5)
    p1 = np.full(sigmoid.shape, 0.5)
    eps = 0.005 / 2.0

    # A degenerate row (a saturated sigmoid) has a zero diagonal and no update
    # to make; it is frozen at its already-correct value instead of dividing by
    # zero.
    degenerate = (q00 <= 0.0) | (q11 <= 0.0)

    for _ in range(max_iter):
        # Recomputed from scratch each sweep, as libsvm does "for numerical
        # accuracy".
        qp0 = q00 * p0 + q01 * p1
        qp1 = q10 * p0 + q11 * p1
        pQp = p0 * qp0 + p1 * qp1
        max_error = np.maximum(np.abs(qp0 - pQp), np.abs(qp1 - pQp))
        live = (max_error >= eps) & ~degenerate
        if not live.any():
            break
        keep = ~live

        # --- t = 0 -------------------------------------------------------
        # p[t] += diff, then the loop over j divides EVERY p[j] -- including
        # p[t] -- by (1 + diff). The increment is therefore itself rescaled.
        diff = np.where(live, (-qp0 + pQp) / np.where(q00 > 0.0, q00, 1.0), 0.0)
        scale = 1.0 + diff
        new_pQp = (pQp + diff * (diff * q00 + 2.0 * qp0)) / (scale * scale)
        new_qp0 = (qp0 + diff * q00) / scale
        new_qp1 = (qp1 + diff * q01) / scale
        new_p0 = (p0 + diff) / scale
        new_p1 = p1 / scale
        pQp = np.where(keep, pQp, new_pQp)
        qp0 = np.where(keep, qp0, new_qp0)
        qp1 = np.where(keep, qp1, new_qp1)
        p0 = np.where(keep, p0, new_p0)
        p1 = np.where(keep, p1, new_p1)

        # --- t = 1 -------------------------------------------------------
        diff = np.where(live, (-qp1 + pQp) / np.where(q11 > 0.0, q11, 1.0), 0.0)
        scale = 1.0 + diff
        new_pQp = (pQp + diff * (diff * q11 + 2.0 * qp1)) / (scale * scale)
        new_qp1 = (qp1 + diff * q11) / scale
        new_qp0 = (qp0 + diff * q10) / scale
        new_p1 = (p1 + diff) / scale
        new_p0 = p0 / scale
        pQp = np.where(keep, pQp, new_pQp)
        qp1 = np.where(keep, qp1, new_qp1)
        qp0 = np.where(keep, qp0, new_qp0)
        p1 = np.where(keep, p1, new_p1)
        p0 = np.where(keep, p0, new_p0)

    return p0, p1


class DRD2Oracle:
    """Frozen RBF-SVM scorer. Construct once per process; scoring is batched."""

    def __init__(self, parameters_path: Path, *, platt_orientation: str = "direct"):
        data = np.load(Path(parameters_path))
        self.support_vectors = np.ascontiguousarray(data["support_vectors"], np.float64)
        self.dual_coef = np.ascontiguousarray(data["dual_coef"], np.float64).ravel()
        self.intercept = float(np.asarray(data["intercept"]).ravel()[0])
        self.prob_a = float(np.asarray(data["prob_a"]).ravel()[0])
        self.prob_b = float(np.asarray(data["prob_b"]).ravel()[0])
        self.gamma = float(np.asarray(data["gamma"]).ravel()[0])
        if platt_orientation not in ("direct", "flipped"):
            raise ValueError(f"unknown platt orientation {platt_orientation!r}")
        self.platt_orientation = platt_orientation
        # ||s||^2 is reused for every query, so it is computed once.
        self._sv_sqnorm = np.einsum("ij,ij->i", self.support_vectors,
                                    self.support_vectors)

    @classmethod
    def from_manifest(cls, manifest_path: Path, parameters_path: Path | None = None):
        """Build using the orientation the parity check pinned in the manifest."""

        manifest = json.loads(Path(manifest_path).read_text())
        orientation = manifest.get("platt_orientation")
        if orientation is None:
            raise ValueError(
                "manifest carries no platt_orientation; run drd2_oracle_parity.py "
                "before using the oracle -- the sign is verified, never assumed")
        if parameters_path is None:
            parameters_path = Path(manifest_path).with_name("drd2_svm_parameters.npz")
        return cls(parameters_path, platt_orientation=orientation)

    def decision_values(self, fingerprints: np.ndarray) -> np.ndarray:
        """libsvm decision values for a (n, 2048) count matrix."""

        x = np.ascontiguousarray(np.atleast_2d(fingerprints), np.float64)
        # ||x - s||^2 expanded so the heavy term is one GEMM.
        cross = x @ self.support_vectors.T
        sq = (np.einsum("ij,ij->i", x, x)[:, None]
              - 2.0 * cross + self._sqnorm_row())
        np.maximum(sq, 0.0, out=sq)
        kernel = np.exp(-self.gamma * sq)
        return kernel @ self.dual_coef + self.intercept

    def _sqnorm_row(self) -> np.ndarray:
        return self._sv_sqnorm[None, :]

    def probabilities(self, fingerprints: np.ndarray) -> np.ndarray:
        """P(active) for a (n, 2048) count matrix, as the benchmark computes it."""

        decision = self.decision_values(fingerprints)
        sigmoid = _sigmoid_predict(decision, self.prob_a, self.prob_b)
        first, second = _binary_probability_coupling(sigmoid)
        return second if self.platt_orientation == "flipped" else first

    def margins(self, fingerprints: np.ndarray) -> np.ndarray:
        """logit P(active) -- the SVM margin, increasing with activity.

        With the flipped orientation P(active) = sigmoid(A d + B) exactly, so
        ``A d + B`` IS the log-odds and is strictly monotone in the probability:
        ranking by margin and ranking by probability are the SAME ranking, so a
        greedy controller behaves identically under either.

        The margin exists for numerical RESOLUTION, not for a different ordering.
        Across the benchmark's own source molecules the probability spans
        0.00003 to 0.048 while the margin spans about 1.2, so a short rollout
        into the low-activity regime can register progress that the probability
        cannot.

        Monotonicity does NOT survive taking expectations: E[max margin] and
        E[max probability] are different objectives and may order candidates
        differently.  The margin is therefore a mechanism-detection signal only,
        never a restatement of the benchmark outcome, which stays P(active)
        against the 0.5 threshold.
        """

        return self.prob_a * self.decision_values(fingerprints) + self.prob_b

    def _map_many(self, smiles: Sequence[str], reduce, fill: float,
                  batch: int) -> np.ndarray:
        smiles = list(smiles)
        out = np.full(len(smiles), fill, dtype=np.float64)
        index: list[int] = []
        rows: list[np.ndarray] = []

        def flush() -> None:
            if not rows:
                return
            out[index] = reduce(np.vstack(rows))
            index.clear()
            rows.clear()

        for position, item in enumerate(smiles):
            row = oracle_fingerprint(item)
            if row is None:
                continue
            index.append(position)
            rows.append(row)
            if len(rows) >= batch:
                flush()
        flush()
        return out

    def margin_many(self, smiles: Sequence[str], *, batch: int = 512) -> np.ndarray:
        """Margins per SMILES; unparseable molecules take -inf, not 0.0.

        Zero is a perfectly good margin -- it is exactly P = 0.5, the success
        threshold -- so an unparseable molecule must not be given one. That
        would make a non-molecule look like a solved target.
        """

        return self._map_many(smiles, self.margins, -np.inf, batch)

    def score_many(self, smiles: Sequence[str],
                   *, batch: int = 512) -> np.ndarray:
        """P(active) per SMILES; unparseable molecules score 0.0.

        Scoring is batched because the kernel is a dense (n x 2159) GEMM -- one
        molecule at a time wastes most of the achievable throughput, which
        matters when a single decision state has ~600 successors to rank.
        """

        smiles = list(smiles)
        out = np.zeros(len(smiles), dtype=np.float64)
        pending_index: list[int] = []
        pending_rows: list[np.ndarray] = []

        def flush() -> None:
            if not pending_rows:
                return
            matrix = np.vstack(pending_rows)
            out[pending_index] = self.probabilities(matrix)
            pending_index.clear()
            pending_rows.clear()

        for position, item in enumerate(smiles):
            row = oracle_fingerprint(item)
            if row is None:
                continue
            pending_index.append(position)
            pending_rows.append(row)
            if len(pending_rows) >= batch:
                flush()
        flush()
        return out

    def score(self, smiles: str) -> float:
        return float(self.score_many([smiles])[0])


@functools.lru_cache(maxsize=1)
def load_default_oracle(manifest_path: str) -> DRD2Oracle:
    return DRD2Oracle.from_manifest(Path(manifest_path))
