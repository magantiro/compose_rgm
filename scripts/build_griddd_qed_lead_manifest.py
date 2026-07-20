#!/usr/bin/env python3
"""Freeze exact Jin QED leads and a separately labeled GrIDDD reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, QED, rdFingerprintGenerator

from compose_v4.experiments.griddd_conditional import (
    GridDDBenchmarkFairnessContract,
)


JIN_COMMIT = "e02c14ae857be0e122413845162aeac0251b09c6"
GRIDDD_COMMIT = "cc3dc31ac341216a0bbcc62f63ae6831581ff40b"
ZINC_SOURCE_COMMIT = "37b9f96470d4471c0593cffefa448e0a8a184ef6"
JIN_TEST_EXPECTED_SHA256 = (
    "704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090"
)
ZINC_SOURCE_EXPECTED_SHA256 = (
    "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
)
ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
ALLOWED_DICTIONARY_CHARGES = {"N+1", "O-1"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ordered_smiles_sha256(smiles: list[str]) -> str:
    return _sha256_bytes(("\n".join(smiles) + "\n").encode("utf-8"))


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty lead table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _molecule_row(index: int, original_smiles: str) -> dict[str, object]:
    molecule = Chem.MolFromSmiles(original_smiles)
    if molecule is None:
        raise ValueError(f"invalid Jin lead at line {index + 1}")
    canonical_isomeric = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    canonical_nonisomeric = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    return {
        "lead_index": index,
        "source_line_number": index + 1,
        "original_smiles": original_smiles,
        "canonical_isomeric_smiles": canonical_isomeric,
        "canonical_nonisomeric_smiles": canonical_nonisomeric,
        "qed": format(float(QED.qed(molecule)), ".17g"),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "elements": ";".join(elements),
    }


def _load_jin_exact(path: Path) -> list[dict[str, object]]:
    if _file_sha256(path) != JIN_TEST_EXPECTED_SHA256:
        raise ValueError("Jin QED test file does not match the frozen public artifact")
    smiles = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(smiles) != 800:
        raise ValueError("Jin QED test artifact must contain exactly 800 leads")
    return [_molecule_row(index, value) for index, value in enumerate(smiles)]


def _dictionary_policy_canonical_smiles(raw_smiles: str) -> str | None:
    """Approximate the released GrIDDD ZINC dictionary-policy preprocessing.

    This reconstruction is deliberately not called exact.  The repository does
    not ship its generated ``qed_smiles.smiles`` and does not bind the executed
    CLI overrides, processed cache, or RDKit build.
    """

    molecule = Chem.MolFromSmiles(raw_smiles)
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, Chem.rdchem.KekulizeException):
        return None
    canonical = Chem.MolToSmiles(molecule)
    rebuilt = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        if charge:
            token = f"{symbol}{'+' if charge > 0 else ''}{charge}"
            if token not in ALLOWED_DICTIONARY_CHARGES:
                return None
        elif symbol not in ALLOWED_ELEMENTS:
            return None
        copied = Chem.Atom(symbol)
        copied.SetFormalCharge(charge)
        old_to_new[atom.GetIdx()] = rebuilt.AddAtom(copied)
    bonds: list[tuple[int, int, Chem.BondType]] = []
    for bond in molecule.GetBonds():
        left = old_to_new.get(bond.GetBeginAtomIdx())
        right = old_to_new.get(bond.GetEndAtomIdx())
        if left is None or right is None:
            continue
        bonds.append((min(left, right), max(left, right), bond.GetBondType()))
    for left, right, bond_type in sorted(bonds, key=lambda row: (row[0], row[1])):
        rebuilt.AddBond(left, right, bond_type)
    try:
        reconstructed = rebuilt.GetMol()
        Chem.RemoveStereochemistry(reconstructed)
        Chem.SanitizeMol(reconstructed)
        reconstructed_smiles = Chem.MolToSmiles(reconstructed)
    except (ValueError, Chem.rdchem.KekulizeException):
        return None
    if reconstructed_smiles != canonical:
        return None
    if len(Chem.GetMolFrags(reconstructed)) != 1:
        return None
    return canonical


def _load_zinc_rows(path: Path) -> list[str]:
    if _file_sha256(path) != ZINC_SOURCE_EXPECTED_SHA256:
        raise ValueError("ZINC source CSV does not match the frozen public artifact")
    with path.open(encoding="utf-8", newline="") as handle:
        return [str(row["smiles"]) for row in csv.DictReader(handle)]


def _reconstruct_griddd(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    raw_smiles = _load_zinc_rows(path)
    permutation = np.random.RandomState(42).permutation(len(raw_smiles))
    train_count = int(0.8 * len(raw_smiles))
    test_count = int(0.1 * len(raw_smiles))
    validation_count = len(raw_smiles) - train_count - test_count
    train_indices = permutation[:train_count]
    test_indices = permutation[train_count + validation_count :]
    processed_train = {
        value
        for index in train_indices
        if (value := _dictionary_policy_canonical_smiles(raw_smiles[int(index)]))
        is not None
    }
    selected: list[dict[str, object]] = []
    rejected = 0
    train_duplicate = 0
    processed_test = 0
    for source_index in test_indices:
        canonical = _dictionary_policy_canonical_smiles(
            raw_smiles[int(source_index)]
        )
        if canonical is None:
            rejected += 1
            continue
        if canonical in processed_train:
            train_duplicate += 1
            continue
        processed_test += 1
        molecule = Chem.MolFromSmiles(canonical)
        assert molecule is not None
        qed = float(QED.qed(molecule))
        if 0.70 <= qed <= 0.80 and len(selected) < 800:
            selected.append(
                {
                    "lead_index": len(selected),
                    "zinc_source_row_zero_based": int(source_index),
                    "canonical_smiles": canonical,
                    "qed": format(qed, ".17g"),
                    "heavy_atoms": molecule.GetNumHeavyAtoms(),
                    "formal_charge": Chem.GetFormalCharge(molecule),
                    "elements": ";".join(
                        sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
                    ),
                }
            )
    if len(selected) != 800:
        raise RuntimeError("GrIDDD reconstruction did not yield 800 eligible leads")
    return selected, {
        "source_rows": len(raw_smiles),
        "train_rows": train_count,
        "validation_rows": validation_count,
        "test_rows": test_count,
        "processed_test_rows": processed_test,
        "rejected_test_rows": rejected,
        "canonical_train_duplicates_removed": train_duplicate,
    }


def _fingerprint_api_equivalence(smiles: list[str]) -> bool:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        legacy = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            2,
            nBits=2048,
            useChirality=False,
        )
        modern = generator.GetFingerprint(molecule)
        if DataStructs.TanimotoSimilarity(legacy, modern) != 1.0:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jin-test", type=Path, required=True)
    parser.add_argument("--zinc-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    exact_rows = _load_jin_exact(args.jin_test)
    reconstruction_rows, reconstruction_counts = _reconstruct_griddd(args.zinc_csv)
    exact_path = args.output_dir / "jin_iclr19_qed_test_exact_v1.csv"
    reconstruction_path = (
        args.output_dir
        / "griddd_qed_test_reconstruction_dictionary_local_runtime_v1.csv"
    )
    manifest_path = args.output_dir / "griddd_qed_lead_manifest_v1.json"
    _atomic_csv(exact_rows, exact_path)
    _atomic_csv(reconstruction_rows, reconstruction_path)

    exact_smiles = [str(row["canonical_isomeric_smiles"]) for row in exact_rows]
    reconstruction_smiles = [
        str(row["canonical_smiles"]) for row in reconstruction_rows
    ]
    exact_qed = [float(row["qed"]) for row in exact_rows]
    reconstruction_qed = [float(row["qed"]) for row in reconstruction_rows]
    fairness = GridDDBenchmarkFairnessContract().to_dict()
    payload: dict[str, object] = {
        "format": "compose_v4_griddd_qed_lead_manifest_v1",
        "decision": {
            "jin_iclr19_exact_public_list_available": True,
            "griddd_release_exact_list_available": False,
            "griddd_primary_protocol_launch_authorized": False,
            "no_substitute_labeled_exact": True,
        },
        "jin_iclr19_exact": {
            "status": "exact_public_artifact",
            "repository": "https://github.com/wengong-jin/iclr19-graph2graph",
            "commit": JIN_COMMIT,
            "source_path": "data/qed/test.txt",
            "source_url": (
                "https://raw.githubusercontent.com/wengong-jin/"
                f"iclr19-graph2graph/{JIN_COMMIT}/data/qed/test.txt"
            ),
            "source_sha256": JIN_TEST_EXPECTED_SHA256,
            "lead_count": len(exact_rows),
            "canonicalization": "RDKit canonical isomeric SMILES",
            "ordered_canonical_smiles_sha256": _ordered_smiles_sha256(
                exact_smiles
            ),
            "table": str(exact_path),
            "table_sha256": _file_sha256(exact_path),
            "qed_minimum": min(exact_qed),
            "qed_maximum": max(exact_qed),
            "all_qed_in_closed_interval_0p70_0p80": all(
                0.70 <= value <= 0.80 for value in exact_qed
            ),
        },
        "griddd_release": {
            "status": "exact_list_blocked",
            "repository": "https://github.com/mninniri/GrIDDD",
            "commit": GRIDDD_COMMIT,
            "selection_code_path": "griddd/datasets/zinc250k_dataset.py",
            "paper": "https://arxiv.org/abs/2506.15725",
            "selection_summary": (
                "shuffle source CSV with pandas random_state=42; use final 10% "
                "as test after an 80/10/10 split; preprocess; select first 800 "
                "processed test rows with inclusive QED 0.70-0.80"
            ),
            "blockers": [
                "generated raw/qed_smiles.smiles is not committed",
                "processed test_qed checkpoint is not committed",
                "executed Hydra CLI overrides are not frozen",
                "README requests charges_policy=dictionary while resolved experiment YAML defaults to partial",
                "the released output is not hash-bound to the requested RDKit 2023.03.2 environment",
            ],
        },
        "griddd_code_reconstruction": {
            "status": "reproducible_reconstruction_not_exact",
            "must_not_be_labeled": "official GrIDDD 800",
            "source_csv_url": (
                "https://raw.githubusercontent.com/aspuru-guzik-group/"
                f"chemical_vae/{ZINC_SOURCE_COMMIT}/models/zinc_properties/"
                "250k_rndm_zinc_drugs_clean_3.csv"
            ),
            "source_csv_sha256": ZINC_SOURCE_EXPECTED_SHA256,
            "settings": {
                "random_state": 42,
                "split": [0.8, 0.1, 0.1],
                "charges_policy": "dictionary",
                "qed_interval": [0.70, 0.80],
                "selection": "first 800 eligible processed test rows",
            },
            "runtime": {
                "rdkit": rdBase.rdkitVersion,
                "numpy": np.__version__,
            },
            "counts": reconstruction_counts,
            "lead_count": len(reconstruction_rows),
            "ordered_canonical_smiles_sha256": _ordered_smiles_sha256(
                reconstruction_smiles
            ),
            "table": str(reconstruction_path),
            "table_sha256": _file_sha256(reconstruction_path),
            "qed_minimum": min(reconstruction_qed),
            "qed_maximum": max(reconstruction_qed),
            "all_qed_in_closed_interval_0p70_0p80": all(
                0.70 <= value <= 0.80 for value in reconstruction_qed
            ),
            "comparison_to_jin_exact": {
                "ordered_equal": reconstruction_smiles == exact_smiles,
                "canonical_set_intersection": len(
                    set(reconstruction_smiles) & set(exact_smiles)
                ),
            },
        },
        "metric_contract": {
            "qed": "rdkit.Chem.QED.qed",
            "similarity": "Tanimoto",
            "fingerprint": {
                "implementation_in_griddd": (
                    "AllChem.GetMorganFingerprintAsBitVect(radius=2, "
                    "nBits=2048, useChirality=False)"
                ),
                "compose_implementation": (
                    "rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)"
                ),
                "equivalent_on_all_frozen_rows": (
                    _fingerprint_api_equivalence(
                        exact_smiles + reconstruction_smiles
                    )
                ),
            },
        },
        "fairness_contract": fairness,
        "builder": {
            "path": str(Path(__file__)),
            "sha256": _file_sha256(Path(__file__)),
        },
        "no_model_launch": True,
    }
    _atomic_json(payload, manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "jin_exact": len(exact_rows),
                "griddd_reconstruction": len(reconstruction_rows),
                "griddd_exact_available": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
