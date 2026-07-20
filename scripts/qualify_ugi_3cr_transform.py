#!/usr/bin/env python3
"""Qualify the AGILE Ugi-3CR (acid-free alpha-amino-amide) reaction transform.

Correctness gate: reconstruct every released AGILE measured product exactly from
its A/B/C components using one frozen atom-mapped reaction SMARTS, and reject a
panel of chemoselectivity negatives.  Writes a deterministic audit artifact whose
SHA-256 is bound into the qualified reaction registry
(``configs/lipid_reactions/qualified_reactions_v1.json``).

Reaction (Ugi three-component, acid-free alpha-amino amide):
    amine + aldehyde + isocyanide  ->  R1-NH-CH(R2)-C(=O)-NH-R4

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/qualify_ugi_3cr_transform.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Frozen atom-mapped transform. Map indices: 1 amine N, 2 aldehyde C (central),
# 3 isocyanide C (-> amide carbonyl), 4 isocyanide N (-> amide N-H).
UGI_3CR_SMARTS = (
    "[NX3;H2,H1:1].[CX3H1:2]=[OX1].[C;-1,+0;X1:3]#[N;+1,+0;X2:4]"
    ">>[N:1][CH1:2][C+0:3](=O)[NH1+0:4]"
)

# Role reactive-handle SMARTS (chemoselectivity gates used by the enumerator).
AMINE_HANDLE = "[NX3;H2,H1]"
ALDEHYDE_HANDLE = "[CX3H1]=[OX1]"
ISOCYANIDE_HANDLE = "[C;-1,+0;X1]#[N;+1,+0;X2]"

# Chemoselectivity negatives: each swaps one component for an incompatible group.
_NEG_AMINE = "CN(C)CCN"
_NEG_ALD = "O=C(CCCCCCCC)OCCCCCC=O"
_NEG_ISO = "CCCCCCCCCCCC[N+]#[C-]"
NEGATIVE_CASES = [
    {"reason": "tertiary-amine-only substrate has no N-H handle",
     "reactants": ["CCN(CC)CC", _NEG_ALD, _NEG_ISO]},
    {"reason": "ketone is not an aldehyde (no C-H on the carbonyl)",
     "reactants": [_NEG_AMINE, "CCCCC(=O)CCCC", _NEG_ISO]},
    {"reason": "nitrile carbon is internal (X2), not a terminal isocyanide carbon",
     "reactants": [_NEG_AMINE, _NEG_ALD, "CCCCCCCCCCCCC#N"]},
    {"reason": "alkane supplies no isocyanide handle",
     "reactants": [_NEG_AMINE, _NEG_ALD, "CCCCCCCCCCCC"]},
    {"reason": "carboxylic acid is not an aldehyde",
     "reactants": [_NEG_AMINE, "CCCCCCCC(=O)O", _NEG_ISO]},
]


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def make_reaction():
    reaction = AllChem.ReactionFromSmarts(UGI_3CR_SMARTS)
    if reaction is None:
        raise SystemExit("Ugi-3CR SMARTS failed to parse")
    return reaction


def run_products(reaction, a: str, b: str, c: str) -> set[str]:
    mols = [Chem.MolFromSmiles(x) for x in (a, b, c)]
    if any(m is None for m in mols):
        return set()
    out: set[str] = set()
    for product_set in reaction.RunReactants(tuple(mols)):
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            if len(Chem.GetMolFrags(product)) != 1:
                continue
            smiles = canonical(Chem.MolToSmiles(product))
            if smiles:
                out.add(smiles)
    return out


def _agile_path() -> Path:
    manifest = json.loads(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json").read_text()
    )
    for source in manifest["sources"]:
        if source["source_id"] == "agile_measured1200":
            return Path(source["local_artifact_path"])
    raise SystemExit("agile_measured1200 not found in source manifest")


def qualify(agile_path: Path) -> dict:
    reaction = make_reaction()
    rows = list(csv.DictReader(agile_path.open()))
    total = len(rows)
    exact = 0
    multi_product = 0
    first_positive_examples: list[dict] = []
    misses: list[dict] = []
    for row in rows:
        a, b, c = row["A_smiles"], row["B_smiles"], row["C_smiles"]
        target = canonical(row["combined_mol_SMILES"])
        products = run_products(reaction, a, b, c)
        if target in products:
            exact += 1
            if len(products) > 1:
                multi_product += 1
            if len(first_positive_examples) < 6:
                first_positive_examples.append(
                    {"reactants": [a, b, c], "expected": target,
                     "enumerated_product_count": len(products)}
                )
        elif len(misses) < 10:
            misses.append({"reactants": [a, b, c], "expected": target,
                           "enumerated": sorted(products)[:3]})

    negatives = []
    for case in NEGATIVE_CASES:
        a, b, c = case["reactants"]
        products = run_products(reaction, a, b, c)
        negatives.append({**case, "enumerated_product_count": len(products),
                          "rejected": len(products) == 0})

    audit = {
        "format": "compose_lipid_ugi_3cr_qualification_v1",
        "reaction_id": "ugi_3cr_agile",
        "reaction_version": 2,
        "reaction_name": "Ugi three-component (acid-free alpha-amino amide)",
        "atom_mapped_reaction_smarts": UGI_3CR_SMARTS,
        "role_handles": {
            "amine_head": AMINE_HANDLE,
            "oxoester_aldehyde_body_tail": ALDEHYDE_HANDLE,
            "isocyanide_tail": ISOCYANIDE_HANDLE,
        },
        "validation_source": {
            "source_id": "agile_measured1200",
            "path": str(agile_path),
            "sha256": hashlib.sha256(agile_path.read_bytes()).hexdigest(),
            "product_field": "combined_mol_SMILES",
            "component_fields": ["A_smiles", "B_smiles", "C_smiles"],
        },
        "positive_reconstruction": {
            "total_products": total,
            "exact_reconstructions": exact,
            "exact_fraction": round(exact / total, 6) if total else 0.0,
            "rows_with_multiple_enumerated_products": multi_product,
            "examples": first_positive_examples,
            "misses": misses,
        },
        "negative_rejection": {
            "cases": negatives,
            "all_rejected": all(case["rejected"] for case in negatives),
        },
        "environment": {"rdkit": Chem.rdBase.rdkitVersion},
    }
    return audit


def write_json_deterministic(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_qualification.json",
    )
    args = parser.parse_args()

    audit = qualify(_agile_path())
    audit_sha = write_json_deterministic(args.audit_output, audit)

    pos = audit["positive_reconstruction"]
    neg = audit["negative_rejection"]
    print(f"exact reconstruction: {pos['exact_reconstructions']}/{pos['total_products']} "
          f"= {100 * pos['exact_fraction']:.2f}%")
    print(f"rows with >1 enumerated product: {pos['rows_with_multiple_enumerated_products']}")
    print(f"negatives all rejected: {neg['all_rejected']} ({len(neg['cases'])} cases)")
    print(f"audit: {args.audit_output.relative_to(REPO_ROOT)}")
    print(f"audit sha256: {audit_sha}")

    qualified = pos["exact_fraction"] == 1.0 and neg["all_rejected"]
    print(f"QUALIFICATION {'PASS' if qualified else 'FAIL'}")
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
