#!/usr/bin/env python3
"""Structural-freedom audit: does the corpus span every DOF at realistic ratios?

For every ionizable-lipid degree of freedom -- number of tails, tail length,
tail branchedness (# branches), unsaturation, linker type, linker length, head
size, head class, overall size/charge/rings -- report the corpus distribution
vs the LNPDB R0 anchor and the Jensen-Shannon divergence. Covering an axis is
not enough; its RATIOS must match reality (low JS). High-JS axes are flagged.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/audit_structural_freedom.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.lipids.corpus_bias import jensen_shannon, lipid_topology_features, probability_vector
from compose_v4.oracles.head_domain import basic_nitrogens, head_region_atoms

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

_LINKER_SMARTS = {
    "ester": "[CX3](=O)[OX2][#6]", "amide": "[NX3][CX3](=O)[#6]",
    "carbamate": "[NX3][CX3](=O)[OX2]", "urea": "[NX3][CX3](=O)[NX3]",
    "thioether": "[#6][SX2][#6]", "disulfide": "[SX2][SX2]",
    "acetal": "[CX4]([OX2])([OX2])", "phosphate": "[PX4](=O)([OX2])[OX2]",
}
_LINKER_PATTS = {k: Chem.MolFromSmarts(v) for k, v in _LINKER_SMARTS.items()}


def _bin(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def dof_keys(smiles: str) -> dict[str, str] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    f = lipid_topology_features(mol)
    # linker type: dominant present motif (priority order)
    linker = "none"
    for name in ("phosphate", "disulfide", "acetal", "carbamate", "urea", "thioether", "amide", "ester"):
        if mol.HasSubstructMatch(_LINKER_PATTS[name]):
            linker = name
            break
    # head size: heavy atoms in the ionizable-head region
    head_atoms = head_region_atoms(mol)
    head_size = len(head_atoms)
    n_basic = len(basic_nitrogens(mol))
    return {
        "n_tails": f["long_tail_bin"],
        "tail_length": _bin(f["max_aliphatic_tail_depth"], [7, 12, 17, 22], ["<=7", "8-12", "13-17", "18-22", ">22"]),
        "n_branches": _bin(f["carbon_branch_point_count"], [0, 1, 2], ["0", "1", "2", "3plus"]),
        "unsaturation": _bin(f["cc_unsaturation_count"], [0, 1, 2], ["0", "1", "2", "3plus"]),
        "linker_type": linker,
        "head_size": _bin(head_size, [4, 8, 14], ["<=4", "5-8", "9-14", ">14"]),
        "n_basic_N": _bin(n_basic, [1, 2, 3], ["1", "2", "3", "4plus"]),
        "molecule_size": f["size_bin"],
        "charge": f["charge_bin"],
        "rings": f["ring_bin"],
    }


def _distributions(smis: list[str]) -> list[dict]:
    out = []
    for s in smis:
        k = dof_keys(s)
        if k is not None:
            out.append(k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/structural_freedom_audit.json")
    parser.add_argument("--r0-sample", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    corpus_smis = [r["canonical_smiles"] for r in csv.DictReader(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1/pilot_products.csv").open())]
    r0_all = [r["canonical_isomeric_smiles"] for r in csv.DictReader(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv").open())]
    r0_smis = [r0_all[i] for i in rng.choice(len(r0_all), size=min(args.r0_sample, len(r0_all)), replace=False)]

    corpus = _distributions(corpus_smis)
    r0 = _distributions(r0_smis)
    axes = list(corpus[0].keys())

    report = {}
    for axis in axes:
        cf = [d[axis] for d in corpus]
        rf = [d[axis] for d in r0]
        cats = sorted(set(cf) | set(rf))
        cp = probability_vector(cf, cats)
        rp = probability_vector(rf, cats)
        report[axis] = {
            "jensen_shannon_to_r0": round(jensen_shannon(cp, rp), 4),
            "corpus": {c: round(float(p), 3) for c, p in zip(cats, cp)},
            "r0": {c: round(float(p), 3) for c, p in zip(cats, rp)},
            "n_categories_covered": int((cp > 0).sum()),
            "ratio_match": "good" if jensen_shannon(cp, rp) < 0.1 else "off",
        }

    out = {
        "format": "compose_lipid_structural_freedom_audit_v1",
        "corpus_n": len(corpus), "r0_n": len(r0),
        "degrees_of_freedom": report,
        "well_matched_axes": [a for a in axes if report[a]["ratio_match"] == "good"],
        "off_ratio_axes": [a for a in axes if report[a]["ratio_match"] == "off"],
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"structural-freedom audit: corpus {len(corpus)} vs R0 {len(r0)}")
    print(f"{'DOF':16s}{'JS-to-R0':>10}  match")
    for axis in axes:
        r = report[axis]
        print(f"  {axis:16s}{r['jensen_shannon_to_r0']:>8.3f}  {r['ratio_match']}")
    print(f"\nwell-matched: {out['well_matched_axes']}")
    print(f"off-ratio (need tuning): {out['off_ratio_axes']}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
