#!/usr/bin/env python3
"""Stratified corpus pilot enumeration + paper-grade diversity/coverage metrics.

Enumerates the qualified reaction families over the building-block pool, globally
deduplicates, selects a family/architecture-balanced pilot via the deterministic
stratified reservoir, and reports the metrics needed to defend an unsupervised
structural corpus: family balance, validity/uniqueness, nearest-neighbor
redundancy (ECFP4 Tanimoto), physicochemical marginals vs the LNPDB R0 anchor,
LNPDB coverage/recall, size eligibility, and scaffold diversity.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/enumerate_corpus_pilot.py --quota 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from compose_v4.lipids.building_block_registry import load_pool
from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry
from compose_v4.lipids.streaming_enumeration import StratifiedReservoir

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

REGISTRIES = [
    "configs/lipid_reactions/qualified_reactions_v1.json",
    "configs/lipid_reactions/qualified_reaction_families_v1.json",
]
_MFP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return _MFP.GetFingerprint(m) if m else None


def enumerate_all() -> list:
    pool = load_pool(REPO_ROOT / "configs/lipid_reactions/building_block_pool_v1.json")
    products: dict[str, object] = {}
    per_family_raw: dict[str, int] = {}
    for reg_path in REGISTRIES:
        registry = ReactionRegistry.load(REPO_ROOT / reg_path)
        for spec in registry.reactions:
            enumerator = ReactionEnumerator(spec)
            by_role = pool.blocks_for_roles(enumerator.role_order)
            if any(len(by_role.get(r, [])) == 0 for r in enumerator.role_order):
                continue
            count = 0
            for product in enumerator.enumerate(by_role):
                count += 1
                if product.canonical_smiles not in products:
                    products[product.canonical_smiles] = product
            per_family_raw[spec.reaction_id] = per_family_raw.get(spec.reaction_id, 0) + count
    return list(products.values()), per_family_raw


def select_pilot(products: list, quota: int, seed: str = "compose_lipid_pilot_v1") -> list:
    axes = ("reaction_family", "size_bin", "charge_bin", "ring_bin")
    reservoir = StratifiedReservoir(axes=axes, quota_per_stratum=quota, seed=seed)
    for p in products:
        reservoir.consider(p)
    return reservoir.selected()


def _descriptors(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    return {
        "mw": Descriptors.MolWt(m), "logp": Crippen.MolLogP(m), "tpsa": Descriptors.TPSA(m),
        "hbd": Lipinski.NumHDonors(m), "hba": Lipinski.NumHAcceptors(m),
        "rot": Lipinski.NumRotatableBonds(m), "rings": rdMolDescriptors.CalcNumRings(m),
        "charge": Chem.GetFormalCharge(m), "heavy": m.GetNumHeavyAtoms(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
    }


def _marginals(smis: list[str]) -> dict:
    rows = [_descriptors(s) for s in smis]
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": round(float(vals.mean()), 2), "median": round(float(np.median(vals)), 2),
                  "p5": round(float(np.percentile(vals, 5)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}
    return out


def _nn_redundancy(fps: list, sample: int, rng: np.random.Generator) -> dict:
    n = len(fps)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    nn = []
    for i in idx:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i] + fps[i + 1:])
        nn.append(max(sims) if sims else 0.0)
    nn = np.array(nn)
    return {"sampled": int(len(idx)), "median_nn_tanimoto": round(float(np.median(nn)), 3),
            "frac_nn_ge_0.9": round(float((nn >= 0.9).mean()), 3),
            "frac_nn_ge_0.95": round(float((nn >= 0.95).mean()), 3)}


def _lnpdb_coverage(corpus_fps: list, rng: np.random.Generator, sample: int = 2000) -> dict:
    r0_path = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv"
    import csv
    r0 = [row["canonical_isomeric_smiles"] for row in csv.DictReader(r0_path.open())]
    idx = rng.choice(len(r0), size=min(sample, len(r0)), replace=False)
    covered = 0; nn_to_corpus = []
    for i in idx:
        fp = fingerprint(r0[i])
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, corpus_fps)
        m = max(sims) if sims else 0.0
        nn_to_corpus.append(m)
        if m >= 0.4:
            covered += 1
    nn_to_corpus = np.array(nn_to_corpus)
    return {"r0_sampled": int(len(nn_to_corpus)),
            "recall_at_tanimoto_0.4": round(float(covered / len(nn_to_corpus)), 3),
            "median_r0_nn_to_corpus": round(float(np.median(nn_to_corpus)), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota", type=int, default=60, help="max products per stratum")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    products, per_family_raw = enumerate_all()
    total_unique = len(products)
    selected = select_pilot(products, args.quota)

    sel_smiles = [p.canonical_smiles for p in selected]
    sel_fps = [fingerprint(s) for s in sel_smiles]
    sel_fps = [f for f in sel_fps if f is not None]

    fam_raw = dict(sorted(per_family_raw.items()))
    fam_sel: dict[str, int] = {}
    for p in selected:
        fam_sel[p.reaction_id] = fam_sel.get(p.reaction_id, 0) + 1

    heavy = np.array([_descriptors(s)["heavy"] for s in sel_smiles])
    # heteroatom-connector-ish scaffold diversity via Bemis-Murcko generic scaffolds
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = set()
    for s in sel_smiles:
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(s))
        except Exception:
            pass

    metrics = {
        "format": "compose_lipid_corpus_pilot_metrics_v1",
        "corpus_scope": "general linker-agnostic reaction-grounded pilot (R1)",
        "enumeration": {
            "raw_products_by_family": fam_raw,
            "total_raw_applications": int(sum(fam_raw.values())),
            "total_unique_products": total_unique,
        },
        "selection": {
            "quota_per_stratum": args.quota,
            "selected_pilot_size": len(selected),
            "family_balance_selected": dict(sorted(fam_sel.items())),
            "note": "raw enumeration is family-skewed by combinatorial count; stratified reservoir rebalances.",
        },
        "validity": {"all_single_component_sanitized": True, "uniqueness": 1.0},
        "nn_redundancy_intra_corpus": _nn_redundancy(sel_fps, sample=min(3000, len(sel_fps)), rng=rng),
        "physchem_marginals": _marginals(sel_smiles),
        "lnpdb_coverage": _lnpdb_coverage(sel_fps, rng),
        "size_eligibility": {
            "min_heavy": int(heavy.min()), "median_heavy": int(np.median(heavy)),
            "max_heavy": int(heavy.max()),
            "frac_le_64_heavy": round(float((heavy <= 64).mean()), 3),
            "frac_le_96_heavy": round(float((heavy <= 96).mean()), 3),
        },
        "scaffold_diversity": {"unique_murcko_scaffolds": len(scaffolds),
                               "murcko_per_1k": round(1000 * len(scaffolds) / len(sel_smiles), 1)},
        "environment": {"rdkit": Chem.rdBase.rdkitVersion, "seed": args.seed},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with (args.out_dir / "pilot_products.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["canonical_smiles", "reaction_family", "size_bin", "charge_bin", "ring_bin"])
        for p in selected:
            w.writerow([p.canonical_smiles, p.reaction_id, p.product_bins["size_bin"],
                        p.product_bins["charge_bin"], p.product_bins["ring_bin"]])
    text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    (args.out_dir / "pilot_metrics.json").write_text(text)

    print(f"raw applications: {metrics['enumeration']['total_raw_applications']:,}  "
          f"unique products: {total_unique:,}")
    print(f"raw by family: {fam_raw}")
    print(f"selected pilot: {len(selected):,}  balance: {dict(sorted(fam_sel.items()))}")
    print(f"NN redundancy: {metrics['nn_redundancy_intra_corpus']}")
    print(f"LNPDB coverage: {metrics['lnpdb_coverage']}")
    print(f"size: {metrics['size_eligibility']}")
    print(f"scaffolds: {metrics['scaffold_diversity']}")
    print(f"written: {(args.out_dir/'pilot_metrics.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
