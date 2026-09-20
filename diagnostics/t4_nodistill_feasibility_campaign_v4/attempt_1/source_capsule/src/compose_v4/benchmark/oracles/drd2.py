"""DRD2 for Task 3: the same frozen SVM, made independent of batch size.

THE DEFECT THIS AVOIDS
----------------------
`docs/ORACLE_BATCH_INVARIANCE_DEFECT.md` records that `DRD2Oracle` scores the
same molecule differently depending on how many molecules were asked about at
once.  The cause is one line -- `kernel @ dual_coef` dispatches to a dot product
for a single row and to GEMV for a batch, and the two accumulate in different
orders.  The magnitude is ~2e-14: chemically meaningless, and exactly the
magnitude that flips a tie.

That defect is deliberately NOT fixed in `drd2_oracle.py`, because frozen QED
results were measured with it and fixing it would re-baseline them.  Task 3 has
nothing frozen yet, so it gets the correct behaviour from the start: here the
reduction is a row-wise sum over a fixed number of support vectors, which is
independent of how many rows are being scored.

WHY THIS MATTERS FOR A BENCHMARK SPECIFICALLY
---------------------------------------------
The oracle meter caches by molecule and charges once.  If the value depended on
the batch a molecule happened to arrive in, then the cached value and the value
a re-run would compute could differ, and a "resumed" run would not reproduce the
run it resumed.  Durability is worth nothing if replay is not exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compose_v4.drd2_oracle import DRD2Oracle


class BatchInvariantDRD2(DRD2Oracle):
    """`DRD2Oracle` with a reduction that does not depend on the batch shape.

    Only `decision_values` is overridden; `probabilities`, `margins` and the
    manifest constructor all route through it, so the Platt orientation stays
    exactly the one the DRD2 parity check pinned.
    """

    def decision_values(self, fingerprints: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.atleast_2d(fingerprints), np.float64)
        cross = x @ self.support_vectors.T
        sq = (np.einsum("ij,ij->i", x, x)[:, None]
              - 2.0 * cross + self._sqnorm_row())
        np.maximum(sq, 0.0, out=sq)
        kernel = np.exp(-self.gamma * sq)
        # `kernel @ dual_coef` is the batch-dependent line. Summing along the
        # support-vector axis instead reduces each row over a FIXED length with
        # numpy's pairwise algorithm, so a row's result is the same whether it
        # was scored alone or alongside a thousand others.
        return (kernel * self.dual_coef).sum(axis=1) + self.intercept
