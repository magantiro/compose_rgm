#!/usr/bin/env python3
"""Materialize the full reaction-grounded R1 corpus + training manifest for the generator.

Until now only the 26.5k R0+AGILE union was materialized as training data; the
~464k route-certified R1 products were enumerable but not written. This makes the
corpus TRAINING-READY: it enumerates the full unique R1 set (the RGM reachable
support), attaches per-product provenance + cheap DOF bins + a REALISM SAMPLING
WEIGHT (so sampling R1 by weight matches R0's real marginals while the whole
reachable set stays available), and writes a training manifest (R0+R1 layering,
kernel profile the model must support, split policy, provenance hashes).

Realism weights are fit by iterative proportional fitting over the axes we can
compute cheaply at 464k scale WITHOUT the head-region BFS -- linker type,
n_tails, tail_length, molecule size. head_size ratio-matching is validated on the
diagnostic pilot (structural_freedom_audit.json, JS 0.021) rather than re-fit here.

Run (background; ~20-25 min at 464k):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/materialize_training_corpus.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.lipids.corpus_bias import jensen_shannon, lipid_topology_features, probability_vector
from scripts.enumerate_corpus_pilot import (
    _LINKER_PATT,
    _LINKER_PRIORITY,
    _bin,
    _r0_sample,
    enumerate_all,
)

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1"
# axes fit at full scale (no head-region BFS); head_size validated on the pilot audit.
_WEIGHT_AXES = ("linker_type", "n_tails", "tail_length", "size")


def _cheap_dof(mol) -> dict[str, str]:
    f = lipid_topology_features(mol)
    linker = "none"
    for name in _LINKER_PRIORITY:
        if mol.HasSubstructMatch(_LINKER_PATT[name]):
            linker = name
            break
    return {
        "linker_type": linker,
        "n_tails": f["long_tail_bin"],
        "tail_length": _bin(f["max_aliphatic_tail_depth"], [7, 12, 17, 22],
                            ["<=7", "8-12", "13-17", "18-22", ">22"]),
        "size": f["size_bin"],
    }


def _ipf_weights(keys: list[dict], r0_keys: list[dict], iters: int = 50) -> np.ndarray:
    targets: dict[str, dict[str, float]] = {}
    for axis in _WEIGHT_AXES:
        c = Counter(k[axis] for k in r0_keys)
        tot = sum(c.values()) or 1
        targets[axis] = {b: v / tot for b, v in c.items()}
    bins_by_axis = {axis: defaultdict(list) for axis in _WEIGHT_AXES}
    for i, k in enumerate(keys):
        for axis in _WEIGHT_AXES:
            bins_by_axis[axis][k[axis]].append(i)
    n = len(keys)
    w = np.ones(n)
    for _ in range(iters):
        for axis in _WEIGHT_AXES:
            W = w.sum()
            for b, idxs in bins_by_axis[axis].items():
                cur = w[idxs].sum()
                tgt = targets[axis].get(b, 1e-6) * W
                if cur > 0:
                    w[idxs] *= tgt / cur
        w *= n / w.sum()
    return w  # mean ~1.0 over the corpus


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATASET)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("enumerating full R1 reachable support ...", flush=True)
    products, per_family_raw = enumerate_all()
    print(f"  unique R1 products: {len(products):,}", flush=True)

    print("computing cheap DOF keys + realism weights ...", flush=True)
    mols = [Chem.MolFromSmiles(p.canonical_smiles) for p in products]
    keys = [_cheap_dof(m) for m in mols]
    r0_smiles = _r0_sample(rng, n=3000)
    r0_keys = [_cheap_dof(m) for m in (Chem.MolFromSmiles(s) for s in r0_smiles) if m is not None]
    weights = _ipf_weights(keys, r0_keys)

    # write the R1 corpus with provenance + cheap bins + realism weight
    r1_path = args.out_dir / "r1_reaction_grounded_corpus_v1.csv"
    with r1_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["canonical_smiles", "reaction_family", "reactant_ids", "reactant_roles",
                    "size_bin", "charge_bin", "ring_bin", "n_tails_bin", "tail_length_bin",
                    "linker_type", "realism_weight"])
        for p, k, wt in zip(products, keys, weights):
            w.writerow([p.canonical_smiles, p.reaction_id, "|".join(p.reactant_ids),
                        "|".join(p.reactant_roles), p.product_bins["size_bin"],
                        p.product_bins["charge_bin"], p.product_bins["ring_bin"],
                        k["n_tails"], k["tail_length"], k["linker_type"], round(float(wt), 5)])
    r1_sha = _sha256(r1_path)

    # weighted-realism check on the fit axes (head_size validated separately on the pilot)
    js_weighted = {}
    p_weight = weights / weights.sum()
    draw = rng.choice(len(products), size=min(6000, len(products)), replace=False, p=p_weight)
    for axis in _WEIGHT_AXES:
        cf = [keys[i][axis] for i in draw]
        rf = [k[axis] for k in r0_keys]
        cats = sorted(set(cf) | set(rf))
        js_weighted[axis] = round(jensen_shannon(probability_vector(cf, cats),
                                                 probability_vector(rf, cats)), 4)

    # kernel profile (elements / charge / size / stereo) on a sample -- what the model must support
    sample = [mols[i] for i in rng.choice(len(mols), size=min(8000, len(mols)), replace=False)]
    elements: set[str] = set()
    heavy, n_stereo, n_charged = [], 0, 0
    for m in sample:
        for a in m.GetAtoms():
            elements.add(a.GetSymbol())
        heavy.append(m.GetNumHeavyAtoms())
        if Chem.GetFormalCharge(m) != 0:
            n_charged += 1
        if Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False):
            n_stereo += 1
    h = np.array(heavy)

    fam_counts = Counter(p.reaction_id for p in products)
    r0_count = sum(1 for _ in csv.DictReader((DATASET / "r0_observed_real_structures.csv").open()))
    r0_sha = _sha256(DATASET / "r0_observed_real_structures.csv")

    manifest = {
        "format": "compose_lipid_training_corpus_v1",
        "purpose": "training-ready structural pretraining corpus for the COMPOSE-Lipid generator (R0 anchor + R1 reachable support)",
        "layers": {
            "r0_observed_real": {
                "path": "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv",
                "row_count": r0_count, "sha256": r0_sha, "role": "high-weight real anchor",
                "within_layer": "uniform over unique canonical observed structures",
                "never_inherits_labels": True,
            },
            "r1_reaction_grounded": {
                "path": "artifacts/datasets/compose_lipid_pretraining_v1/r1_reaction_grounded_corpus_v1.csv",
                "row_count": len(products), "sha256": r1_sha,
                "role": "reaction-grounded reachable support (RGM rewrite kernel target set)",
                "within_layer": "sample by realism_weight to match R0 marginals; full set available for coverage",
                "provenance_columns": ["reaction_family", "reactant_ids", "reactant_roles"],
                "family_counts": dict(sorted(fam_counts.items())),
            },
        },
        "realism_weighting": {
            "method": "iterative proportional fitting to R0 marginals; weight column = realism_weight (mean ~1.0)",
            "axes_fit_at_scale": list(_WEIGHT_AXES),
            "weighted_js_to_r0": js_weighted,
            "head_size_note": "head_size ratio-match validated on the diagnostic pilot (structural_freedom_audit.json, JS 0.021), not re-fit here to avoid head-region BFS at 464k scale",
        },
        "kernel_readiness_profile": {
            "elements_present": sorted(elements),
            "heavy_atoms": {"min": int(h.min()), "median": int(np.median(h)), "max": int(h.max()),
                            "frac_le_64": round(float((h <= 64).mean()), 3),
                            "frac_le_96": round(float((h <= 96).mean()), 3)},
            "fraction_charged": round(n_charged / len(sample), 4),
            "fraction_with_stereocenter": round(n_stereo / len(sample), 4),
            "note": "the C/N/O/F small-molecule kernel is NOT lipid-ready; extend elements (S,P) + heavy-atom range + declare a stereo policy before training",
        },
        "splits": {
            "frozen_diagnostic": "artifacts/datasets/compose_lipid_pretraining_v1/splits_v1/ (R0 + 2,159 R1; leak-free)",
            "policy": "leakage-resistant group-key hashing: a structure is val/test if its group (reaction_family | heteroatom_core | head_region | study) hashes to that fold. Apply the same hash to R1 at load time; reaction_family split is directly available from the reaction_family column.",
        },
        "provenance": {
            "building_block_pool": "configs/lipid_reactions/building_block_pool_v1.json",
            "reaction_registries": ["configs/lipid_reactions/qualified_reactions_v1.json",
                                    "configs/lipid_reactions/qualified_reaction_families_v1.json"],
            "raw_applications_by_family": dict(sorted(per_family_raw.items())),
            "seed": args.seed, "rdkit": Chem.rdBase.rdkitVersion,
        },
        "gates": [
            "structural pretraining only; R1 never carries biological/delivery labels",
            "corpus stays general/linker-agnostic; the Michael/propiolate linker is a downstream fine-tune",
            "lipid generator training authorized only after Paper 1 P1-G7",
        ],
    }
    (args.out_dir / "training_corpus_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"R1 corpus written: {r1_path.relative_to(REPO_ROOT)}  ({len(products):,} rows, sha {r1_sha[:12]})")
    print(f"weighted JS-to-R0 (fit axes): {js_weighted}")
    print(f"elements: {sorted(elements)}  heavy median {int(np.median(h))} max {int(h.max())}")
    print(f"families: {dict(sorted(fam_counts.items()))}")
    print(f"manifest: {(args.out_dir/'training_corpus_manifest_v1.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
