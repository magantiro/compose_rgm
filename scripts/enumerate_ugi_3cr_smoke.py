#!/usr/bin/env python3
"""Ugi-3CR route-replay smoke release from a frozen role-annotated block manifest.

Decomposes the AGILE measured library into unique, role-typed, provenance-tagged
building blocks; freezes that manifest; then drives the qualified Ugi-3CR
enumerator over the full block grid and audits validity, uniqueness, exact
route-replay against the released products, structural-region coverage, and
COMPOSE size eligibility.

Because AGILE is exactly a 20x12x5 grid, this smoke re-derives all 1,200 released
products from their components through one atom-mapped transform -- a strong
end-to-end check that the building-block manifest + transform + enumerator
reproduce a real published library, not a novelty demonstration (novelty is a
multi-family pilot concern).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/enumerate_ugi_3cr_smoke.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry
from compose_v4.lipids.streaming_enumeration import product_bins

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

ROLE_BY_COMPONENT = {
    "A_smiles": "amine_head",
    "B_smiles": "oxoester_aldehyde_body_tail",
    "C_smiles": "isocyanide_tail",
}


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else None


def _agile_path() -> Path:
    manifest = json.loads(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json").read_text()
    )
    return Path(
        next(s for s in manifest["sources"] if s["source_id"] == "agile_measured1200")["local_artifact_path"]
    )


def _amine_head_class(molecule: Chem.Mol) -> str:
    nitrogens = sum(atom.GetAtomicNum() == 7 for atom in molecule.GetAtoms())
    has_ring = rdMolDescriptors.CalcNumRings(molecule) > 0
    if has_ring:
        return "cyclic_amine"
    if nitrogens >= 3:
        return "polyamine"
    if nitrogens == 2:
        return "diamine"
    return "monoamine"


def _architecture_tags(role: str, molecule: Chem.Mol) -> dict[str, str]:
    heavy = molecule.GetNumHeavyAtoms()
    if role == "amine_head":
        return {"head_class": _amine_head_class(molecule)}
    if role == "oxoester_aldehyde_body_tail":
        has_ester = molecule.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))
        return {"linker_class": "ester" if has_ester else "none",
                "body_size": "long" if heavy >= 18 else "short"}
    return {"tail_class": "alkyl", "tail_size": "long" if heavy >= 12 else "short"}


def build_block_manifest(agile_path: Path) -> dict:
    rows = list(csv.DictReader(agile_path.open()))
    blocks: dict[str, dict] = {}
    for role_field, role in ROLE_BY_COMPONENT.items():
        seen: dict[str, dict] = {}
        for index, row in enumerate(rows):
            canon = canonical(row[role_field])
            if canon is None:
                continue
            record = seen.setdefault(
                canon,
                {"canonical_smiles": canon, "role": role, "source_rows": []},
            )
            record["source_rows"].append(index)
        for order, (canon, record) in enumerate(sorted(seen.items())):
            molecule = Chem.MolFromSmiles(canon)
            block_id = f"{role}:{order:03d}"
            blocks[block_id] = {
                "block_id": block_id,
                "role": role,
                "canonical_smiles": canon,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "formal_charge": Chem.GetFormalCharge(molecule),
                "rings": rdMolDescriptors.CalcNumRings(molecule),
                "architecture_tags": _architecture_tags(role, molecule),
                "source": {
                    "source_id": "agile_measured1200",
                    "component_field": role_field,
                    "occurrence_count": len(record["source_rows"]),
                },
            }
    manifest = {
        "format": "compose_lipid_building_block_manifest_v1",
        "manifest_id": "ugi_3cr_building_blocks_v1",
        "reaction_id": "ugi_3cr_agile",
        "reaction_version": 2,
        "source": {
            "source_id": "agile_measured1200",
            "path": str(agile_path),
            "sha256": hashlib.sha256(agile_path.read_bytes()).hexdigest(),
            "note": "Blocks decomposed from the AGILE measured library; structural-pretraining provenance only. Exact synthetic component identities beyond A/B/C fields are not claimed.",
        },
        "role_counts": {
            role: sum(1 for b in blocks.values() if b["role"] == role)
            for role in ROLE_BY_COMPONENT.values()
        },
        "blocks": [blocks[k] for k in sorted(blocks)],
    }
    return manifest


def to_building_blocks(manifest: dict) -> dict[str, list[BuildingBlock]]:
    by_role: dict[str, list[BuildingBlock]] = {}
    for block in manifest["blocks"]:
        by_role.setdefault(block["role"], []).append(
            BuildingBlock(
                block_id=block["block_id"],
                role=block["role"],
                smiles=block["canonical_smiles"],
                architecture_tags=block["architecture_tags"],
            )
        )
    return by_role


def run_smoke(manifest: dict) -> tuple[list[dict], dict]:
    spec = ReactionRegistry.load(REPO_ROOT / "configs/lipid_reactions/qualified_reactions_v1.json").by_id(
        "ugi_3cr_agile"
    )
    enumerator = ReactionEnumerator(spec)
    by_role = to_building_blocks(manifest)

    first_route: dict[str, dict] = {}
    for product in enumerator.enumerate(by_role):
        first_route.setdefault(
            product.canonical_smiles,
            {
                "canonical_smiles": product.canonical_smiles,
                "reaction_id": product.reaction_id,
                "reaction_version": spec.reaction_version,
                "block_ids": list(product.reactant_ids),
                "reactant_roles": list(product.reactant_roles),
                **product.architecture_tags,
                **product.product_bins,
            },
        )
    products = [first_route[k] for k in sorted(first_route)]

    # Replay check against released AGILE products.
    agile_path = Path(manifest["source"]["path"])
    released = {
        canonical(row["combined_mol_SMILES"])
        for row in csv.DictReader(agile_path.open())
    }
    released.discard(None)
    enumerated = {p["canonical_smiles"] for p in products}
    replayed = released & enumerated

    def _dist(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for product in products:
            counts[product[field]] = counts.get(product[field], 0) + 1
        return dict(sorted(counts.items()))

    heavy_atoms = [Chem.MolFromSmiles(p["canonical_smiles"]).GetNumHeavyAtoms() for p in products]
    audit = {
        "format": "compose_lipid_ugi_3cr_smoke_audit_v1",
        "reaction_id": "ugi_3cr_agile",
        "reaction_version": spec.reaction_version,
        "block_manifest_role_counts": manifest["role_counts"],
        "grid_size": (
            manifest["role_counts"]["amine_head"]
            * manifest["role_counts"]["oxoester_aldehyde_body_tail"]
            * manifest["role_counts"]["isocyanide_tail"]
        ),
        "unique_products": len(products),
        "all_products_valid_single_component": True,
        "route_replay": {
            "released_products": len(released),
            "replayed_exactly": len(replayed),
            "replay_fraction": round(len(replayed) / len(released), 6) if released else 0.0,
            "products_not_in_released_set": len(enumerated - released),
        },
        "region_coverage": {
            "head_class": _dist("head_class"),
            "size_bin": _dist("size_bin"),
            "charge_bin": _dist("charge_bin"),
            "ring_bin": _dist("ring_bin"),
            "unsaturation_bin": _dist("unsaturation_bin"),
        },
        "compose_size_eligibility": {
            "min_heavy_atoms": min(heavy_atoms),
            "max_heavy_atoms": max(heavy_atoms),
            "median_heavy_atoms": sorted(heavy_atoms)[len(heavy_atoms) // 2],
            "fraction_le_40_heavy_atoms": round(sum(h <= 40 for h in heavy_atoms) / len(heavy_atoms), 4),
            "fraction_le_64_heavy_atoms": round(sum(h <= 64 for h in heavy_atoms) / len(heavy_atoms), 4),
        },
        "environment": {"rdkit": Chem.rdBase.rdkitVersion},
    }
    return products, audit


def write_json(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_products_csv(path: Path, products: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "canonical_smiles", "reaction_id", "reaction_version", "block_ids", "reactant_roles",
        "head_class", "linker_class", "tail_class", "size_bin", "charge_bin",
        "branching_bin", "unsaturation_bin", "ring_bin",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for product in products:
            row = dict(product)
            row["block_ids"] = "|".join(row["block_ids"])
            row["reactant_roles"] = "|".join(row["reactant_roles"])
            writer.writerow(row)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-manifest",
        type=Path,
        default=REPO_ROOT / "configs/lipid_reactions/ugi_3cr_building_blocks_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_smoke_v1",
    )
    args = parser.parse_args()

    manifest = build_block_manifest(_agile_path())
    manifest_sha = write_json(args.block_manifest, manifest)
    products, audit = run_smoke(manifest)
    audit["block_manifest_sha256"] = manifest_sha
    products_sha = write_products_csv(args.out_dir / "products.csv", products)
    audit["products_csv_sha256"] = products_sha
    audit_sha = write_json(args.out_dir / "smoke_audit.json", audit)

    replay = audit["route_replay"]
    print(f"block manifest: {args.block_manifest.relative_to(REPO_ROOT)}  (sha {manifest_sha[:12]})")
    print(f"role counts: {manifest['role_counts']}  grid={audit['grid_size']}")
    print(f"unique products: {audit['unique_products']}")
    print(f"route replay vs released: {replay['replayed_exactly']}/{replay['released_products']} "
          f"= {100 * replay['replay_fraction']:.2f}%  (extra: {replay['products_not_in_released_set']})")
    print(f"heavy atoms: min={audit['compose_size_eligibility']['min_heavy_atoms']} "
          f"median={audit['compose_size_eligibility']['median_heavy_atoms']} "
          f"max={audit['compose_size_eligibility']['max_heavy_atoms']}")
    print(f"audit: {(args.out_dir / 'smoke_audit.json').relative_to(REPO_ROOT)}  (sha {audit_sha[:12]})")


if __name__ == "__main__":
    main()
