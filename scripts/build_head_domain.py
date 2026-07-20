#!/usr/bin/env python3
"""Freeze the known-head applicability domain from R0 real lipids + corpus amines.

Operationalizes the milestone-4 decision (head axis doesn't transfer -> rank
within known heads, abstain on novel ones). The domain is the set of ionizable-
head regions over the ~15.4k real observed lipids plus the corpus amine heads;
its size (hundreds of scaffolds) is why 'known head' is a large, useful domain.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/build_head_domain.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import RDLogger

from compose_v4.oracles.head_domain import domain_from_head_smiles, unique_head_smiles

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/oracles/head_domain_v1")
    parser.add_argument("--similarity-threshold", type=float, default=0.4)
    args = parser.parse_args()

    r0 = [row["canonical_isomeric_smiles"] for row in csv.DictReader(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv").open())]
    pool = json.loads((REPO_ROOT / "configs/lipid_reactions/building_block_pool_v1.json").read_text())
    amines = [b["canonical_smiles"] for b in pool["blocks"] if b["handle"] == "amine"]
    pka_vals = [float(row["head_pka"]) for row in csv.DictReader(
        (REPO_ROOT / "artifacts/oracles/head_pka_v1/a549_head_pka.csv").open())
        if row["head_pka"] not in ("", "None")]
    pka_min, pka_max = min(pka_vals), max(pka_vals)

    heads = unique_head_smiles(r0 + amines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "known_head_regions.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["head_region_smiles"])
        for h in heads:
            w.writerow([h])
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # validation: known classes admit, novel/OOD abstain
    domain = domain_from_head_smiles(heads, pka_min, pka_max, args.similarity_threshold)
    checks = {
        "DMAPA-head lipid (known)": ("CCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)N(C)CCN(C)C", 9.5, True),
        "piperazine-head (known)": ("CCCCCCCCCCCCC(=O)N1CCN(CCCCCCCCCCC)CC1", 8.5, True),
        "MC3-like DMA head (known)": ("CCCCCCCCC=CCC=CCCCCCCCC(=O)OCC(CN(C)C)OC(=O)CCCCCCCC=CCC=CCCCCC", 9.5, True),
        "guanidine-head (novel/OOD)": ("CCCCCCCCCCCCCCCCNC(=N)N", 12.5, False),
    }
    validation = {}
    for name, (smi, pk, expect) in checks.items():
        r = domain.admission(smi, head_pka=pk)
        validation[name] = {"admitted": r["admitted"], "max_head_similarity": r["max_head_similarity"],
                            "expected_admit": expect, "correct": r["admitted"] == expect}

    manifest = {
        "format": "compose_head_domain_v1",
        "purpose": "known-head applicability domain: rank within, abstain (->active learning) outside",
        "known_head_regions": len(heads),
        "sources": {"r0_real_lipids": len(r0), "corpus_amine_heads": len(amines)},
        "head_pka_range": [round(pka_min, 2), round(pka_max, 2)],
        "similarity_threshold": args.similarity_threshold,
        "known_head_regions_csv": {"path": str(csv_path.relative_to(REPO_ROOT)), "sha256": csv_sha},
        "validation": validation,
        "all_validation_correct": all(v["correct"] for v in validation.values()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"known-head domain: {len(heads)} head-region scaffolds "
          f"(from {len(r0)} R0 lipids + {len(amines)} corpus amines)")
    print(f"head-pKa range: [{pka_min:.1f}, {pka_max:.1f}]")
    for name, v in validation.items():
        print(f"  {name:30s} admit={v['admitted']}  sim={v['max_head_similarity']}  {'OK' if v['correct'] else 'WRONG'}")
    print(f"all validation correct: {manifest['all_validation_correct']}")


if __name__ == "__main__":
    main()
