#!/usr/bin/env python3
"""Region-conditioned element-insertion prior, derived from the corpus.

The model's `connected_atom_order_log_prior` is a single GLOBAL 3x4 base-rate over
(bond-order in {1,2,3}) x (element in C/N/O/F) for a connected atom insertion. The
heteroatom-abundance evidence shows this base-rate should depend on the region
(tail=carbon, linker=O/N, head=N). This script estimates P(order, element | region)
from the corpus: for each molecule, a BFS spanning tree from atom 0 assigns each
non-root atom the order of its parent bond (a defensible proxy for the order it was
inserted with) and its element; atoms are grouped by lipid_region_labels. Output is
the per-region prior table (the artifact a gated region-conditioned prior loads) and
a divergence showing how different the regions are from the global prior.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/build_region_conditioned_prior.py --n 15000
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.lipids.region_labels import REGION_NAMES, lipid_region_labels

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles"
ELEMENTS = ["C", "N", "O", "F"]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENTS)}
ORDERS = [1, 2, 3]


def _spanning_insertions(mol: Chem.Mol):
    """Yield (atom_idx, parent_bond_order) for every non-root atom via BFS from 0."""
    n = mol.GetNumAtoms()
    if n == 0:
        return
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        i = queue.popleft()
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j not in seen:
                seen.add(j)
                order = int(round(bond.GetBondTypeAsDouble()))
                yield j, max(1, min(3, order))
                queue.append(j)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return round(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=REPO_ROOT
                        / "artifacts/datasets/compose_lipid_pretraining_v1/region_conditioned_prior_v1.json")
    args = parser.parse_args()

    lines = [ln.strip().split()[0] for ln in args.corpus.open() if ln.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(lines), size=min(args.n, len(lines)), replace=False)

    # counts[region] = 3x4 (order x element); root_counts[region] = 4 (element)
    counts = {name: np.zeros((3, 4)) for name in REGION_NAMES.values()}
    root_counts = {name: np.zeros(4) for name in REGION_NAMES.values()}
    n_ok = 0
    for i in idx:
        mol = Chem.MolFromSmiles(lines[i])
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        labels = lipid_region_labels(mol)
        n_ok += 1
        # root atom (grow_root prior)
        r0 = mol.GetAtomWithIdx(0).GetSymbol()
        if r0 in ELEM_IDX:
            root_counts[REGION_NAMES[labels[0]]][ELEM_IDX[r0]] += 1
        for j, order in _spanning_insertions(mol):
            sym = mol.GetAtomWithIdx(j).GetSymbol()
            if sym in ELEM_IDX:
                counts[REGION_NAMES[labels[j]]][order - 1, ELEM_IDX[sym]] += 1

    smoothing = 0.5
    def _norm(m):
        m = m + smoothing
        return (m / m.sum()).round(5)

    global_conn = sum(counts.values())
    per_region_conn = {n: _norm(counts[n]) for n in REGION_NAMES.values()}
    per_region_root = {n: _norm(root_counts[n]) for n in REGION_NAMES.values()}
    global_conn_norm = _norm(global_conn)

    out = {
        "format": "compose_lipid_region_conditioned_prior_v1",
        "purpose": "region-conditioned base-rate for the model's connected_atom_order prior (grow_option) and grow_root prior; the model learns a residual above the per-region base-rate (corpus_residual_v1).",
        "axes": {"order": ORDERS, "element": ELEMENTS, "region": list(REGION_NAMES.values())},
        "sample_molecules": n_ok,
        "connected_atom_order_prior": {
            "global": global_conn_norm.tolist(),
            "per_region": {n: per_region_conn[n].tolist() for n in REGION_NAMES.values()},
        },
        "root_atom_prior": {n: per_region_root[n].tolist() for n in REGION_NAMES.values()},
        "js_divergence_to_global": {n: _js(per_region_conn[n].ravel(), global_conn_norm.ravel())
                                    for n in REGION_NAMES.values()},
        "js_tail_vs_linker": _js(per_region_conn["tail"].ravel(), per_region_conn["linker"].ravel()),
        "interpretation": "Large per-region JS to the global prior => a single global element-insertion prior is a poor base-rate; the region-conditioned table is the correct initialization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")

    print(f"sampled {n_ok} lipids")
    print("P(element | region), summed over bond order:")
    for n in ("head", "linker", "tail", "other"):
        pe = per_region_conn[n].sum(axis=0)
        print(f"  {n:>7}: " + "  ".join(f"{e}={pe[k]:.2f}" for e, k in ELEM_IDX.items())
              + f"   JS_to_global={out['js_divergence_to_global'][n]}")
    print(f"JS(tail vs linker) = {out['js_tail_vs_linker']}   (0=identical, 1=disjoint)")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
