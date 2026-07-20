#!/usr/bin/env python3
"""Reportable unconditional-sufficiency panel for a generated molecule set.

Combines the repository's ``molecular_quality_report`` and ``ring_taxonomy_report``
(so the standard metrics match the frozen pancake baseline exactly) with the
chemistry-failure motifs isolated in
``docs/audits/2026-07-20_unconditional_chemistry_failure_audit.md``
(triple bonds, heteroatom-heavy rings, ring O-O / N-N, adjacent aromatic
``[nH]``, and small rings) that the base eval does not report.

The reference is the matched neutral C/N/O/F <=40-heavy-atom deduplicated
projection of the held-out set, built with the same ``_canonical_cnof_smiles``
filter the rollout evaluator uses, so generated and reference motif rates are
directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.data.cnof import _canonical_cnof_smiles
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report

RDLogger.DisableLog("rdApp.*")


def _load_smiles_file(path: Path, limit: int | None = None) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        lines = lines[:limit]
    return lines


def build_cnof_reference(path: Path, *, max_atoms: int, limit: int | None) -> list[str]:
    """Canonical neutral-CNOF<=max_atoms deduplicated reference (order-preserving)."""

    raw = _load_smiles_file(path, limit=limit)
    seen: set[str] = set()
    reference: list[str] = []
    for smiles in raw:
        canonical = _canonical_cnof_smiles((smiles, max_atoms))
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        reference.append(canonical)
    return reference


def _is_aromatic_nh(atom: Chem.Atom) -> bool:
    return (
        atom.GetIsAromatic()
        and atom.GetSymbol() == "N"
        and atom.GetTotalNumHs() >= 1
    )


def motif_panel(smiles: list[str]) -> dict[str, float]:
    """Chemistry-failure motif prevalence, matched to the audit definitions."""

    n = 0
    any_triple = 0
    triple_bonds = 0
    total_bonds = 0
    triple_in_ring = 0
    heteroatoms = 0
    nitrogen = 0
    oxygen = 0
    ge3_hetero_ring_mol = 0
    ring_oo_mol = 0
    ring_nn_mol = 0
    adjacent_nh_mol = 0
    multi_nh_ring_mol = 0
    small_ring_mol = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n += 1
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        # per-atom composition
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                heteroatoms += 1
            if atom.GetSymbol() == "N":
                nitrogen += 1
            elif atom.GetSymbol() == "O":
                oxygen += 1
        # bonds
        has_triple = False
        for bond in mol.GetBonds():
            total_bonds += 1
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                triple_bonds += 1
                has_triple = True
                if bond.IsInRing():
                    triple_in_ring += 1
        if has_triple:
            any_triple += 1
        # ring-level motifs
        has_ge3_hetero = False
        has_ring_oo = False
        has_ring_nn = False
        has_small = False
        has_multi_nh_ring = False
        for ring in atom_rings:
            if len(ring) <= 4:
                has_small = True
            hetero = sum(
                1 for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() != "C"
            )
            if hetero >= 3:
                has_ge3_hetero = True
            nh_count = sum(
                1 for idx in ring if _is_aromatic_nh(mol.GetAtomWithIdx(idx))
            )
            if nh_count >= 2:
                has_multi_nh_ring = True
        # bond-in-ring O-O / N-N and adjacent aromatic nH
        has_adjacent_nh = False
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                continue
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            sa, sb = a.GetSymbol(), b.GetSymbol()
            if sa == "O" and sb == "O":
                has_ring_oo = True
            if sa == "N" and sb == "N":
                has_ring_nn = True
            if _is_aromatic_nh(a) and _is_aromatic_nh(b):
                has_adjacent_nh = True
        ge3_hetero_ring_mol += int(has_ge3_hetero)
        ring_oo_mol += int(has_ring_oo)
        ring_nn_mol += int(has_ring_nn)
        adjacent_nh_mol += int(has_adjacent_nh)
        multi_nh_ring_mol += int(has_multi_nh_ring)
        small_ring_mol += int(has_small)
    if n == 0:
        return {"molecules": 0}
    return {
        "molecules": n,
        "triple_molecule_fraction": any_triple / n,
        "triple_bonds_per_molecule": triple_bonds / n,
        "triple_bond_share_of_all_bonds": (triple_bonds / total_bonds) if total_bonds else 0.0,
        "triple_bonds_in_rings": triple_in_ring,
        "heteroatoms_per_molecule": heteroatoms / n,
        "nitrogen_per_molecule": nitrogen / n,
        "oxygen_per_molecule": oxygen / n,
        "ge3_hetero_ring_molecule_fraction": ge3_hetero_ring_mol / n,
        "ring_oo_molecule_fraction": ring_oo_mol / n,
        "ring_nn_molecule_fraction": ring_nn_mol / n,
        "adjacent_aromatic_nh_molecule_fraction": adjacent_nh_mol / n,
        "multi_aromatic_nh_ring_molecule_fraction": multi_nh_ring_mol / n,
        "small_ring_molecule_fraction": small_ring_mol / n,
    }


def _generated_from_input(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        for key in ("generated_smiles", "generated_nonnull_smiles"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                return [str(s) for s in value if s]
        raise ValueError(f"{path} lacks a generated_smiles/generated_nonnull_smiles list")
    return _load_smiles_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path, help="rollout metrics JSON or a .smiles file")
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--reference-limit", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--label", type=str, default="generated")
    args = parser.parse_args()

    generated = _generated_from_input(args.generated)
    reference = build_cnof_reference(
        args.reference_file, max_atoms=args.max_atoms, limit=args.reference_limit
    )
    train = build_cnof_reference(
        args.train_file, max_atoms=args.max_atoms, limit=args.train_limit
    )

    quality = molecular_quality_report(
        tuple(generated),
        reference_smiles=tuple(reference),
        train_smiles=tuple(train),
        include_fcd=False,
    )
    taxonomy_generated = ring_taxonomy_report(tuple(generated))
    taxonomy_reference = ring_taxonomy_report(tuple(reference))
    motifs_generated = motif_panel(generated)
    motifs_reference = motif_panel(reference)

    panel = {
        "format": "compose_v4_unconditional_sufficiency_panel_v1",
        "label": args.label,
        "counts": {
            "generated": len(generated),
            "reference_cnof_matched": len(reference),
            "train_cnof_matched": len(train),
        },
        "quality": quality,
        "ring_taxonomy": {
            "generated": taxonomy_generated,
            "reference": taxonomy_reference,
        },
        "motifs": {
            "generated": motifs_generated,
            "reference": motifs_reference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")

    q = quality
    dd = q["descriptor_distributions"]
    print(f"=== {args.label}: {len(generated)} generated vs {len(reference)} matched reference ===")
    print(f"valid {q['valid_fraction']:.4f}  unique {q['unique_fraction']:.4f}  novel {q['novel_to_train_fraction']:.4f}  intdiv {q['similarity']['internal_diversity']:.4f}")
    mg, mr = motifs_generated, motifs_reference
    print(f"triples(mol): gen {mg['triple_molecule_fraction']:.4f}  ref {mr['triple_molecule_fraction']:.4f}   (per-mol gen {mg['triple_bonds_per_molecule']:.3f})")
    print(f">=3-hetero ring: gen {mg['ge3_hetero_ring_molecule_fraction']:.4f}  ref {mr['ge3_hetero_ring_molecule_fraction']:.4f}")
    print(f"ring O-O: gen {mg['ring_oo_molecule_fraction']:.4f}  ref {mr['ring_oo_molecule_fraction']:.4f}")
    print(f"adjacent arom nH: gen {mg['adjacent_aromatic_nh_molecule_fraction']:.4f}  ref {mr['adjacent_aromatic_nh_molecule_fraction']:.4f}")
    print(f"small ring(mol): gen {mg['small_ring_molecule_fraction']:.4f}  ref {mr['small_ring_molecule_fraction']:.4f}")
    rs = q["ring_systems"]
    print(f"fused: gen {rs['generated_fused_fraction']:.4f}  ref {rs['reference_fused_fraction']:.4f}   spiro gen {rs['generated_spiro_fraction']:.4f} ref {rs['reference_spiro_fraction']:.4f}   bridged gen {rs['generated_bridged_fraction']:.4f} ref {rs['reference_bridged_fraction']:.4f}")
    print(f"QED: gen {dd['qed']['generated_mean']:.4f}  ref {dd['qed']['reference_mean']:.4f}    SA: gen {dd['sa_score']['generated_mean']:.4f}  ref {dd['sa_score']['reference_mean']:.4f}")
    print(f"aromatic-atom frac: gen {dd['aromatic_atom_fraction']['generated_mean']:.4f}  ref {dd['aromatic_atom_fraction']['reference_mean']:.4f}")
    print(f"panel written: {args.output}")


if __name__ == "__main__":
    main()
