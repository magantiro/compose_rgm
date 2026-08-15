#!/usr/bin/env python
"""Can a surrogate rank JNK3 and GSK3B from a few hundred PAID labels?

This runs BEFORE any steering comparison, and it exists to prevent one specific
misdiagnosis. If adaptive region targeting fails to beat a fixed scalarization,
there are two possible causes -- the targeting does not help, or the surrogate
cannot see the objectives well enough for any targeting to act on. Those call
for opposite responses. So the surrogate is qualified on its own first.

THE PROTOCOL IS THE ONE A RUN ACTUALLY FACES
--------------------------------------------
Labels are not sampled at random from a finished run; they arrive in the order
the run bought them. So training uses the FIRST N molecules of a real ledger and
testing uses the NEXT block, which reproduces both the ordering and -- crucially
-- the base rate an early run sees. JNK3 activity is scarce: in a random
ZINC-250k draw roughly 98% of molecules score near zero, so at N = 200 the
training set may contain no actives at all. That, not model capacity, is the
thing most likely to break early steering.

⚠️ THE REPRESENTATION HERE IS AN OPTIMISTIC CONTROL, NOT THE REAL TEST.
Morgan bits are used because they are free. But the JNK3 and GSK3B oracles ARE
random forests over Morgan bits, so a fingerprint surrogate is fitting the same
function class on the same features as the oracle it is predicting. It should be
read as a CEILING -- "this is the best a lightweight surrogate could do here" --
and never as evidence that a surrogate over frozen R_theta representations will
work as well. That comparison has to be run on the representation we will
actually steer with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

NAMES = ("qed", "jnk3", "sa", "gsk3b", "drd2")
TRAIN_SIZES = (120, 250, 500, 1000, 2000)
#: Tanimoto k-NN: no fitting, no hyperparameters to tune on the very data whose
#: scarcity is the question, and it degrades gracefully with few labels.
K = 5


def read_ledger(path: Path) -> tuple[list[str], np.ndarray]:
    smiles, values = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                smiles.append(record["smiles"])
                values.append(record["v"])
    return smiles, np.asarray(values, dtype=float)


def fingerprints(smiles: list[str]) -> np.ndarray:
    from compose_v4.benchmark.oracles.forest import morgan_bits

    rows = [morgan_bits(s) for s in smiles]
    return np.vstack([r if r is not None else np.zeros(2048) for r in rows])


def knn_predict(train_fp: np.ndarray, train_y: np.ndarray,
                test_fp: np.ndarray, k: int = K) -> np.ndarray:
    """Tanimoto-weighted k-NN, in blocks so the similarity matrix stays small."""

    train_norm = train_fp.sum(axis=1)
    test_norm = test_fp.sum(axis=1)
    out = np.zeros((len(test_fp), train_y.shape[1]))
    for start in range(0, len(test_fp), 512):
        block = test_fp[start:start + 512]
        intersection = block @ train_fp.T
        union = (test_norm[start:start + 512, None] + train_norm[None, :]
                 - intersection)
        similarity = np.divide(intersection, np.maximum(union, 1e-9))
        top = np.argpartition(-similarity, min(k, similarity.shape[1] - 1),
                              axis=1)[:, :k]
        weights = np.take_along_axis(similarity, top, axis=1)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        out[start:start + 512] = np.einsum("ij,ijk->ik", weights, train_y[top])
    return out


def enrichment(predicted: np.ndarray, truth: np.ndarray, top: int = 50) -> float:
    """Fraction of the true top-`top` recovered by the surrogate's top-`top`.

    Ranking metrics like Spearman are dominated by the inactive mass; what a
    policy actually needs is to find the few molecules worth attacking.
    """

    if len(truth) <= top:
        return float("nan")
    true_best = set(np.argsort(-truth)[:top].tolist())
    predicted_best = np.argsort(-predicted)[:top]
    return float(sum(1 for i in predicted_best if i in true_best) / top)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, action="append", required=True,
                        help="a run's evaluations.jsonl; repeatable")
    parser.add_argument("--test-block", type=int, default=2000)
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_surrogate_signal.json"))
    args = parser.parse_args()

    report: dict = {
        "representation": "Morgan bits r=2 2048 (OPTIMISTIC: same feature family "
                          "as the jnk3/gsk3b oracles themselves)",
        "model": f"Tanimoto-weighted {K}-NN",
        "protocol": "train on the first N of a real ledger, test on the next block",
        "runs": {},
    }
    for ledger in args.ledger:
        smiles, values = read_ledger(ledger)
        name = ledger.parent.name
        print(f"\n=== {name} ({len(smiles):,} paid labels) ===")
        fp = fingerprints(smiles)
        rows = {}
        print(f"{'N train':>8}{'actives':>9}{'rho jnk3':>10}{'rho gsk3b':>11}"
              f"{'top50 jnk3':>12}")
        for n in TRAIN_SIZES:
            if n + args.test_block > len(smiles):
                continue
            train_fp, train_y = fp[:n], values[:n]
            test_fp, test_y = fp[n:n + args.test_block], values[n:n + args.test_block]
            predicted = knn_predict(train_fp, train_y, test_fp)
            # "actives" in the training window: the scarcity that breaks early
            # steering shows up here before it shows up in the correlations.
            actives = int((train_y[:, 1] > 0.1).sum())
            entry = {"train_actives_jnk3_over_0.1": actives}
            for axis, objective in enumerate(NAMES):
                rho = spearmanr(predicted[:, axis], test_y[:, axis]).statistic
                entry[f"spearman_{objective}"] = (float(rho) if np.isfinite(rho)
                                                  else None)
            entry["top50_recall_jnk3"] = enrichment(predicted[:, 1], test_y[:, 1])
            entry["top50_recall_gsk3b"] = enrichment(predicted[:, 3], test_y[:, 3])
            rows[str(n)] = entry
            print(f"{n:>8}{actives:>9}{entry['spearman_jnk3'] or float('nan'):>10.3f}"
                  f"{entry['spearman_gsk3b'] or float('nan'):>11.3f}"
                  f"{entry['top50_recall_jnk3']:>12.2f}")
        report["runs"][name] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
