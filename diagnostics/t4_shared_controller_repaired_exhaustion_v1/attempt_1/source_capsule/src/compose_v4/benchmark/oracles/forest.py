"""The JNK3 / GSK3B random forests, evaluated in plain numpy.

These two objectives are scikit-learn 0.21.3 pickles that a modern scikit-learn
cannot even open -- the tree node dtype changed in 0.22.  `scripts/
molleo_task3_oracle_extract.py` opens them exactly once in a pinned environment
and writes the flat node tables this module walks.  Nothing here imports sklearn
or unpickles anything, so the oracle cannot drift when the environment does.

WHICH MODEL THIS IS, AND WHY THAT IS NOT A DETAIL
-------------------------------------------------
Two mutually incompatible models circulate for these same two tasks:

    TDC (what MOLLEO calls)   Morgan bits, radius 2, **2048**
    HN-GFN / MARS             Morgan bits, radius 2, **1024**

They are different forests, not different encodings of one forest.  This module
is the 2048-feature TDC-format model, and `n_features` is asserted at load, so
handing it the other fingerprint raises rather than quietly scoring molecules
against the wrong oracle.

THE ARITHMETIC IS sklearn's, REPRODUCED EXACTLY
-----------------------------------------------
`RandomForestClassifier.predict_proba` normalises each leaf's class counts and
then averages over trees.  The frozen table stores the already-normalised
P(class 1) per leaf, so scoring is: walk each tree to its leaf, average.  Every
internal split is `x[feature] <= 0.5` -- verified at extraction time, because
the features are bits -- which is why no threshold array is carried.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: TDC's kinase fingerprint. Not the DRD2 one, which is counts/features at
#: radius 3 -- these two oracles genuinely disagree about what a molecule is.
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048


def morgan_bits(smiles: str, n_bits: int = FINGERPRINT_BITS) -> np.ndarray | None:
    """``AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`` as float64.

    Exactly TDC's call for both gsk3b and jnk3.  Returns None for an
    unparseable SMILES rather than a zero row: an all-zero fingerprint is a
    perfectly legal molecule-shaped input and would be scored as one.
    """

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=n_bits)
    row = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bits, row)
    return row


class FrozenForest:
    """A frozen RandomForestClassifier scored by a batched tree walk.

    All 100 trees are walked simultaneously: the node table is one flat array,
    so a step of the walk is a single gather over an ``(n_molecules, n_trees)``
    index matrix.  Cost is therefore ``max_depth`` gathers for the whole batch
    rather than ``max_depth x n_trees`` per molecule, which is what makes a
    10,000-evaluation budget cheap to spend.
    """

    def __init__(self, parameters_path: Path):
        data = np.load(Path(parameters_path))
        self.children_left = np.ascontiguousarray(data["children_left"], np.int32)
        self.children_right = np.ascontiguousarray(data["children_right"], np.int32)
        self.feature = np.ascontiguousarray(data["feature"], np.int32)
        self.leaf_p1 = np.ascontiguousarray(data["leaf_p1"], np.float64)
        self.tree_root = np.ascontiguousarray(data["tree_root"], np.int32)
        self.n_features = int(data["n_features"])
        self.split_threshold = float(data["split_threshold"])
        # `feature` is int16 in the file; widening once avoids a per-step cast.
        self._safe_feature = np.maximum(self.feature, 0)
        # No tree can be deeper than it has nodes, so this bounds the walk. It
        # exists so a corrupt node table raises instead of spinning forever.
        sizes = np.diff(np.append(self.tree_root, len(self.children_left)))
        self._walk_bound = int(sizes.max())

    @property
    def n_trees(self) -> int:
        return len(self.tree_root)

    def probabilities(self, fingerprints: np.ndarray) -> np.ndarray:
        """P(active) for an ``(n, 2048)`` bit matrix -- sklearn's predict_proba[:, 1]."""

        x = np.atleast_2d(np.asarray(fingerprints, dtype=np.float64))
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"this forest expects {self.n_features} features, got {x.shape[1]}. "
                f"The 1024-bit variant of this task is a DIFFERENT model, not a "
                f"different encoding of this one.")
        n = x.shape[0]
        node = np.broadcast_to(self.tree_root, (n, self.n_trees)).astype(np.int64)
        node = np.ascontiguousarray(node)
        rows = np.arange(n, dtype=np.int64)[:, None]
        # A leaf points at itself and carries feature -1, so the walk is done
        # when no node is still internal.
        for _ in range(self._walk_bound):
            feature = self.feature[node]
            internal = feature >= 0
            if not internal.any():
                break
            bit_is_set = x[rows, self._safe_feature[node]] > self.split_threshold
            step = np.where(bit_is_set, self.children_right[node],
                            self.children_left[node])
            node = np.where(internal, step, node)
        else:  # pragma: no cover - a cycle in the node table is impossible
            raise RuntimeError("tree walk did not terminate; node table is corrupt")
        return self.leaf_p1[node].sum(axis=1) / self.n_trees

    def score_many(self, smiles: list[str], *, batch: int = 512) -> np.ndarray:
        """P(active) per SMILES; unparseable molecules score 0.0.

        Zero is the correct answer for "not a molecule" here because the
        objective is an activity probability and the benchmark treats an
        unparseable string as worthless -- but note that for the MINIMISED
        objectives zero is the BEST value, so `task3` never routes an
        unparseable molecule through this path.
        """

        smiles = list(smiles)
        out = np.zeros(len(smiles), dtype=np.float64)
        index: list[int] = []
        rows: list[np.ndarray] = []

        def flush() -> None:
            if not rows:
                return
            out[index] = self.probabilities(np.vstack(rows))
            index.clear()
            rows.clear()

        for position, item in enumerate(smiles):
            row = morgan_bits(item, self.n_features)
            if row is None:
                continue
            index.append(position)
            rows.append(row)
            if len(rows) >= batch:
                flush()
        flush()
        return out

    def score(self, smiles: str) -> float:
        return float(self.score_many([smiles])[0])
