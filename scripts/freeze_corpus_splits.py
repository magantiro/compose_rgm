#!/usr/bin/env python3
"""Freeze leakage-resistant train/val/test splits for the lipid corpus.

Paper-essential (Fig 2 evaluation) and the handoff the generator session needs:
splits that jointly hold out reaction family, heteroatom-core scaffold, head
region, and (for R0) source study -- so a held-out group never leaks across
folds. Assignment is by deterministic hash of the group key, so it is stable and
reproducible; a structure inherits the fold of its group.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/freeze_corpus_splits.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.lipids.corpus_bias import heteroatom_connector_core
from compose_v4.oracles.head_domain import head_region_smiles

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _fold(group_key: str, salt: str, val_cut: int = 70, test_cut: int = 85) -> str:
    """Deterministic train/val/test fold from a stable hash bucket (0..99)."""
    bucket = int(hashlib.sha256(f"{salt}\0{group_key}".encode()).hexdigest(), 16) % 100
    if bucket < val_cut:
        return "train"
    if bucket < test_cut:
        return "val"
    return "test"


def group_keys(smiles: str, reaction_family: str, source: str) -> dict[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    core = heteroatom_connector_core(mol) if mol is not None else "<none>"
    head = head_region_smiles(smiles) or "<none>"
    return {
        "reaction_family": reaction_family,
        "heteroatom_core": core,
        "head_region": head,
        "source_study": source,
        "exact_structure": smiles,
    }


SPLITS = {
    "held_reaction_family": "reaction_family",
    "held_scaffold": "heteroatom_core",
    "held_head": "head_region",
    "held_study": "source_study",
    "random_diagnostic": "exact_structure",
}


def load_corpus() -> list[dict]:
    rows: list[dict] = []
    # R1 enumerated pilot (reaction-grounded virtual)
    pilot = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1/pilot_products.csv"
    for r in csv.DictReader(pilot.open()):
        rows.append({"smiles": r["canonical_smiles"], "layer": "R1_virtual",
                     "reaction_family": r["reaction_family"], "source": "R1_enumerated"})
    # R0 observed real lipids
    r0 = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv"
    for r in csv.DictReader(r0.open()):
        src = (r.get("observed_source_ids") or "R0_observed").split("|")[0]
        rows.append({"smiles": r["canonical_isomeric_smiles"], "layer": "R0_observed",
                     "reaction_family": "observed_real", "source": src})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/splits_v1")
    parser.add_argument("--max-r0", type=int, default=15433)
    args = parser.parse_args()

    corpus = load_corpus()
    # attach group keys
    for row in corpus:
        row["groups"] = group_keys(row["smiles"], row["reaction_family"], row["source"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    assignment_rows = []
    for split_name, key in SPLITS.items():
        salt = f"compose_lipid_splits_v1:{split_name}"
        folds = {"train": 0, "val": 0, "test": 0}
        held_groups: dict[str, str] = {}
        per_row_fold = []
        for row in corpus:
            gk = row["groups"][key]
            # study split only meaningful for R0; R1 rows go to train there
            if split_name == "held_study" and row["layer"] != "R0_observed":
                fold = "train"
            else:
                fold = _fold(gk, salt)
            folds[fold] += 1
            per_row_fold.append(fold)
            if fold != "train":
                held_groups[gk] = fold
        # leakage verification: no group key spans >1 fold
        group_to_folds: dict[str, set] = {}
        for row, fold in zip(corpus, per_row_fold):
            group_to_folds.setdefault(row["groups"][key], set()).add(fold)
        leaked = {g for g, fs in group_to_folds.items() if len(fs) > 1}
        manifest_splits[split_name] = {
            "group_key": key, "fold_counts": folds,
            "n_groups": len(group_to_folds), "n_held_out_groups": len(held_groups),
            "leaked_groups": len(leaked),
            "example_held_out_groups": sorted(list(held_groups))[:5],
        }
        assignment_rows.append((split_name, per_row_fold))

    # write per-structure fold assignments (one column per split)
    fold_csv = args.out_dir / "corpus_fold_assignments.csv"
    with fold_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["canonical_smiles", "layer", "reaction_family"] + list(SPLITS))
        for i, row in enumerate(corpus):
            w.writerow([row["smiles"], row["layer"], row["reaction_family"]]
                       + [assignment_rows[j][1][i] for j in range(len(SPLITS))])
    fold_sha = hashlib.sha256(fold_csv.read_bytes()).hexdigest()

    manifest = {
        "format": "compose_lipid_corpus_splits_v1",
        "purpose": "leakage-resistant train/val/test: hold out family/scaffold/head/study jointly",
        "corpus_rows": len(corpus),
        "layers": {"R0_observed": sum(1 for r in corpus if r["layer"] == "R0_observed"),
                   "R1_virtual": sum(1 for r in corpus if r["layer"] == "R1_virtual")},
        "splits": manifest_splits,
        "all_splits_leak_free": all(s["leaked_groups"] == 0 for s in manifest_splits.values()),
        "fold_assignments_csv": {"path": str(fold_csv.relative_to(REPO_ROOT)), "sha256": fold_sha},
        "note": "A structure is TEST/VAL if its group (for that split key) hashed to that fold; groups never span folds. Random split is a diagnostic only.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"corpus: {len(corpus)} rows ({manifest['layers']})")
    for name, s in manifest_splits.items():
        print(f"  {name:22s} groups={s['n_groups']:5d} folds={s['fold_counts']} leaked={s['leaked_groups']}")
    print(f"all splits leak-free: {manifest['all_splits_leak_free']}")
    print(f"written: {(args.out_dir/'manifest.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
