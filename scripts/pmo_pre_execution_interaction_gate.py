"""Does parent CHEMISTRY tell us which macro is worth attempting from that parent?

`Q_post` is strong, but it reads the realized molecule and so cannot guide what to PROPOSE.  The
upstream question is whether pre-execution information -- parent chemistry plus macro intent --
carries signal, and specifically whether the two INTERACT.

THE DECISIVE COMPARISON IS INTERACTION vs ADDITIVE, not either against random.  An additive model
already knows "this is a promising parent" and "this is a generally promising macro"; only an
interaction model can express "this ring operation suits THIS chemical environment and not that
one".  Beating random would prove nothing beyond the trivial fact that good parents have good
children, which is why the additive model is the bar.

    Q_pre(G, o)  >  Q_parent(G) + Q_macro(o)   on held-out lineages?

TARGET.  `Q_pre` does not choose an oracle call; it chooses where to spend cheap generation.  So
the target is the value of ATTEMPTING an intent, `E[max over m realizations]` -- if I spend m cheap
attempts on this parent and this family, how good is the best thing I surface?  That is estimable
from the beam without new oracle calls because it generated many realizations per parent.

Family PRESENCE indicators are used, never the exact family tuple: 356 distinct tuples over 4,128
edges is the same sparse keying that stopped `pmo_credit` accumulating evidence.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

EDGES = "diagnostics/pmo_blind_traversal_v1/control_dataset_edges_v1.jsonl"
OUT = "diagnostics/pmo_sibling_ranking_v1/pre_execution_interaction_gate_v1.json"
ATTEMPTS = 4


def _ridge(x, y, penalty=1.0):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = scale < 1e-12
    mean = np.where(constant, 0.0, mean)
    scale = np.where(constant, 1.0, scale)
    z = (x - mean) / scale
    ridge = penalty * np.eye(z.shape[1])
    ridge[-1, -1] = 0.0
    weights = np.linalg.solve(z.T @ z + ridge * len(y), z.T @ y)
    return lambda q: ((q - mean) / scale) @ weights


def main():
    from compose_v4.control.pmo_contextual_macro import (
        MACRO_FAMILIES,
        _descriptors,
        _folded,
        _mol,
    )

    with open(EDGES) as handle:
        rows = [json.loads(line) for line in handle]
    starts = sorted({r["start"] for r in rows})
    if len(starts) < 2:
        print("need at least two start lineages")
        return 1

    # One row per (parent, family-intent) CELL, valued by the best of `ATTEMPTS` realizations.
    cells = collections.defaultdict(list)
    for r in rows:
        key = (r["start"], r["generation"], r["parent"], tuple(sorted(set(r["families"]))))
        cells[key].append(r)
    usable = {k: v for k, v in cells.items() if len(v) >= 2}
    print(f"cells with >=2 realizations: {len(usable)} of {len(cells)}")

    parent_cache: dict[str, np.ndarray] = {}

    def parent_block(smiles):
        if smiles not in parent_cache:
            mol = _mol(smiles)
            parent_cache[smiles] = np.concatenate([_descriptors(mol), _folded(mol)])
        return parent_cache[smiles]

    records = []
    for (start, _gen, parent, families), group in usable.items():
        scores = np.array([g["score"] for g in group])
        target = float(np.sort(scores)[-min(ATTEMPTS, len(scores)) :].max())
        present = np.asarray([float(f in families) for f in MACRO_FAMILIES])
        intent = np.concatenate(
            [present, [float(group[0].get("requested_modules", 0)) / 4.0]]
        )
        records.append(
            {
                "start": start,
                "parent_score": float(group[0]["parent_score"]),
                "parent": parent_block(parent),
                "intent": intent,
                "target": target,
            }
        )

    holdout = starts[-1]
    train = [r for r in records if r["start"] != holdout]
    test = [r for r in records if r["start"] == holdout]
    print(f"train cells {len(train)} | test cells {len(test)} on '{holdout}'")
    if len(test) < 20 or len(train) < 50:
        print("not enough cells for a held-out comparison yet")
        return 1

    def build(kind, rs):
        parent = np.vstack([np.concatenate([[r["parent_score"]], r["parent"]]) for r in rs])
        intent = np.vstack([r["intent"] for r in rs])
        one = np.ones((len(rs), 1))
        if kind == "parent_only":
            return np.hstack([parent, one])
        if kind == "macro_only":
            return np.hstack([intent, one])
        if kind == "additive":
            return np.hstack([parent, intent, one])
        # INTERACTION: every family indicator crossed with a compact parent descriptor, so the
        # macro's effect is allowed to depend on the chemistry it is applied to.
        compact = parent[:, :9]
        cross = np.einsum("ij,ik->ijk", compact, intent).reshape(len(rs), -1)
        return np.hstack([parent, intent, cross, one])

    y_train = np.array([r["target"] for r in train])
    y_test = np.array([r["target"] for r in test])
    report = {
        "schema_version": "pmo_pre_execution_interaction_gate_v1",
        "evidence_role": "held_out_development_gate",
        "new_charged_oracle_calls": 0,
        "target": f"best of up to {ATTEMPTS} realizations per (parent, family-intent) cell",
        "holdout_start": holdout,
        "train_cells": len(train),
        "test_cells": len(test),
        "arms": {},
    }
    baseline = float(np.mean((y_test - y_train.mean()) ** 2))
    print(f"\n{'model':16s} {'held-out R2':>12s} {'spearman':>10s} {'top-10 cell mean':>18s}")
    for kind in ("parent_only", "macro_only", "additive", "interaction"):
        predict = _ridge(build(kind, train), y_train)
        guess = predict(build(kind, test))
        r2 = 1.0 - float(np.mean((y_test - guess) ** 2)) / baseline
        order = np.argsort(np.argsort(guess))
        truth = np.argsort(np.argsort(y_test))
        rho = float(np.corrcoef(order, truth)[0, 1]) if guess.std() > 0 else float("nan")
        top = float(np.mean(y_test[np.argsort(-guess)[:10]]))
        report["arms"][kind] = {"held_out_r2": r2, "spearman": rho, "top10_cell_mean": top}
        print(f"{kind:16s} {r2:>12.4f} {rho:>10.3f} {top:>18.4f}")
    print(f"\n{'overall test-cell mean':16s} {y_test.mean():>12.4f}")
    a, i = report["arms"]["additive"], report["arms"]["interaction"]
    print(
        f"\nINTERACTION minus ADDITIVE: R2 {i['held_out_r2'] - a['held_out_r2']:+.4f}  "
        f"rho {i['spearman'] - a['spearman']:+.3f}  top10 {i['top10_cell_mean'] - a['top10_cell_mean']:+.4f}"
    )
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
