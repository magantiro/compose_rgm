#!/usr/bin/env python3
"""Compare a bounded unconditional rollout with the retained pancake audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from compose_v4.data.cnof import _canonical_cnof_smiles


def _metric(metrics: dict[str, object], *path: str) -> object:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _load_metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"metrics payload is not an object: {path}")
    return payload


def _generated_smiles(metrics: dict[str, object]) -> tuple[str, ...]:
    values = metrics.get("generated_smiles")
    if not isinstance(values, list) or not values:
        raise ValueError("metrics payload has no generated SMILES")
    return tuple(str(value) for value in values)


def _matched_reference(path: Path, *, max_atoms: int = 40) -> tuple[str, ...]:
    accepted: dict[str, None] = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        canonical = _canonical_cnof_smiles((fields[0], max_atoms))
        if canonical is not None:
            accepted.setdefault(canonical, None)
    if not accepted:
        raise ValueError("matched reference is empty")
    return tuple(accepted)


def _ring_bonds(molecule: Chem.Mol, ring: Sequence[int]) -> tuple[Chem.Bond, ...]:
    members = frozenset(int(atom) for atom in ring)
    return tuple(
        bond
        for bond in molecule.GetBonds()
        if bond.IsInRing()
        and bond.GetBeginAtomIdx() in members
        and bond.GetEndAtomIdx() in members
    )


def chemistry_summary(smiles: Sequence[str]) -> dict[str, object]:
    molecules = tuple(Chem.MolFromSmiles(text) for text in smiles)
    if any(molecule is None for molecule in molecules):
        raise ValueError("chemistry comparison received invalid SMILES")

    element_counts: Counter[str] = Counter()
    bond_counts: Counter[str] = Counter()
    triple_types: Counter[str] = Counter()
    heavy_atoms: list[int] = []
    bonds_per_molecule: list[int] = []
    heteroatoms: list[int] = []
    heteroatom_fractions: list[float] = []
    triple_molecules = 0
    ring_oo_molecules = 0
    ring_nn_molecules = 0
    adjacent_aromatic_nh_molecules = 0
    multi_aromatic_nh_ring_molecules = 0
    triple_bonds = 0
    ring_triple_bonds = 0
    ring_count = 0
    aromatic_rings = 0
    heterocyclic_rings = 0
    nonaromatic_unsaturated_rings = 0
    rings_with_three_heteroatoms = 0

    for molecule in molecules:
        assert molecule is not None
        atoms = tuple(molecule.GetAtoms())
        bonds = tuple(molecule.GetBonds())
        heavy = int(molecule.GetNumHeavyAtoms())
        hetero = sum(atom.GetAtomicNum() not in {1, 6} for atom in atoms)
        heavy_atoms.append(heavy)
        bonds_per_molecule.append(len(bonds))
        heteroatoms.append(hetero)
        heteroatom_fractions.append(hetero / max(heavy, 1))
        element_counts.update(atom.GetSymbol() for atom in atoms)
        bond_counts.update(str(bond.GetBondType()) for bond in bonds)

        triples = tuple(
            bond for bond in bonds if bond.GetBondType() == Chem.BondType.TRIPLE
        )
        triple_bonds += len(triples)
        triple_molecules += int(bool(triples))
        ring_triple_bonds += sum(bond.IsInRing() for bond in triples)
        for bond in triples:
            symbols = sorted(
                (
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol(),
                )
            )
            triple_types["#".join(symbols)] += 1

        ring_oo_molecules += int(
            any(
                bond.IsInRing()
                and bond.GetBeginAtom().GetSymbol() == "O"
                and bond.GetEndAtom().GetSymbol() == "O"
                for bond in bonds
            )
        )
        ring_nn_molecules += int(
            any(
                bond.IsInRing()
                and bond.GetBeginAtom().GetSymbol() == "N"
                and bond.GetEndAtom().GetSymbol() == "N"
                for bond in bonds
            )
        )
        adjacent_aromatic_nh_molecules += int(
            any(
                bond.GetIsAromatic()
                and all(
                    atom.GetSymbol() == "N"
                    and atom.GetIsAromatic()
                    and atom.GetTotalNumHs() > 0
                    for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
                )
                for bond in bonds
            )
        )

        rings = tuple(tuple(ring) for ring in Chem.GetSymmSSSR(molecule))
        ring_count += len(rings)
        has_multi_aromatic_nh_ring = False
        for ring in rings:
            ring_atoms = tuple(molecule.GetAtomWithIdx(int(index)) for index in ring)
            ring_edges = _ring_bonds(molecule, ring)
            aromatic = all(atom.GetIsAromatic() for atom in ring_atoms)
            aromatic_rings += int(aromatic)
            heterocyclic_rings += int(any(atom.GetAtomicNum() != 6 for atom in ring_atoms))
            rings_with_three_heteroatoms += int(
                sum(atom.GetAtomicNum() != 6 for atom in ring_atoms) >= 3
            )
            nonaromatic_unsaturated_rings += int(
                not aromatic
                and any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in ring_edges)
            )
            aromatic_nh = sum(
                atom.GetSymbol() == "N"
                and atom.GetIsAromatic()
                and atom.GetTotalNumHs() > 0
                for atom in ring_atoms
            )
            has_multi_aromatic_nh_ring |= aromatic_nh >= 2
        multi_aromatic_nh_ring_molecules += int(has_multi_aromatic_nh_ring)

    samples = len(molecules)
    total_bonds = sum(bond_counts.values())
    return {
        "samples": samples,
        "mean_heavy_atoms": float(np.mean(heavy_atoms)),
        "mean_bonds": float(np.mean(bonds_per_molecule)),
        "mean_heteroatoms": float(np.mean(heteroatoms)),
        "mean_heteroatom_fraction": float(np.mean(heteroatom_fractions)),
        "high_heteroatom_fraction_ge_0p4": float(
            np.mean(np.asarray(heteroatom_fractions) >= 0.4)
        ),
        "mean_element_counts": {
            symbol: element_counts[symbol] / samples
            for symbol in ("C", "N", "O", "F")
        },
        "bond_type_counts": dict(sorted(bond_counts.items())),
        "bond_type_fractions": {
            name: count / max(total_bonds, 1)
            for name, count in sorted(bond_counts.items())
        },
        "triple_bond_types": dict(sorted(triple_types.items())),
        "triple_bonds": triple_bonds,
        "triple_bonds_per_molecule": triple_bonds / samples,
        "triple_bond_fraction_of_bonds": triple_bonds / max(total_bonds, 1),
        "triple_containing_molecule_fraction": triple_molecules / samples,
        "ring_triple_bonds": ring_triple_bonds,
        "ring_oo_molecule_fraction": ring_oo_molecules / samples,
        "ring_nn_molecule_fraction": ring_nn_molecules / samples,
        "adjacent_aromatic_nh_molecule_fraction": (
            adjacent_aromatic_nh_molecules / samples
        ),
        "multi_aromatic_nh_ring_molecule_fraction": (
            multi_aromatic_nh_ring_molecules / samples
        ),
        "ring_count": ring_count,
        "aromatic_ring_fraction": aromatic_rings / max(ring_count, 1),
        "heterocyclic_ring_fraction": heterocyclic_rings / max(ring_count, 1),
        "nonaromatic_unsaturated_ring_fraction": (
            nonaromatic_unsaturated_rings / max(ring_count, 1)
        ),
        "rings_with_three_heteroatoms_fraction": (
            rings_with_three_heteroatoms / max(ring_count, 1)
        ),
    }


def _endpoint_report(metrics: dict[str, object], chemistry: dict[str, object]) -> dict[str, object]:
    prevalence = _metric(metrics, "generated_ring_taxonomy", "molecule_prevalence")
    return {
        "samples": _metric(metrics, "rollout", "samples"),
        "safety": {
            key: _metric(metrics, "rollout", key)
            for key in (
                "non_null_fraction",
                "valid_fraction",
                "connected_or_null_fraction",
                "unique_fraction",
                "event_budget_exhaustion_fraction",
            )
        },
        "distribution": {
            key: _metric(metrics, "rollout", key)
            for key in (
                "mean_atoms",
                "reference_mean_atoms",
                "atom_count_total_variation",
                "mean_bonds",
                "reference_mean_bonds",
                "bond_order_distribution_total_variation",
                "element_distribution_total_variation",
                "mean_cycle_rank",
                "reference_mean_cycle_rank",
                "cycle_rank_total_variation",
                "mean_events",
            )
        },
        "chemistry": chemistry,
        "ring_means": _metric(metrics, "generated_ring_taxonomy", "means"),
        "ring_size_counts": _metric(
            metrics, "generated_ring_taxonomy", "ring_size_counts"
        ),
        "ring_molecule_prevalence": prevalence,
        "event_rule_counts": metrics.get("rollout_event_counts"),
        "event_rule_fractions": _metric(metrics, "rollout", "event_rule_fractions"),
        "trajectory_diagnostics": _metric(
            metrics, "rollout", "trajectory_diagnostics"
        ),
        "quality": {
            name: _metric(
                metrics,
                "molecular_quality",
                "descriptor_distributions",
                name,
                "generated_mean",
            )
            for name in ("qed", "sa_score")
        },
    }


def _render_grid(path: Path, smiles: Sequence[str]) -> None:
    molecules = [Chem.MolFromSmiles(text) for text in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("cannot render invalid SMILES")
    legends = []
    for index, molecule in enumerate(molecules):
        assert molecule is not None
        triples = sum(
            bond.GetBondType() == Chem.BondType.TRIPLE for bond in molecule.GetBonds()
        )
        rings = len(Chem.GetSymmSSSR(molecule))
        hetero = sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())
        legends.append(
            f"#{index} | HA={molecule.GetNumHeavyAtoms()} | rings={rings} | "
            f"triple={triples} | hetero={hetero}"
        )
    image = Draw.MolsToGridImage(
        molecules,
        legends=legends,
        molsPerRow=5,
        subImgSize=(320, 240),
        useSVG=False,
    )
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_metrics", type=Path)
    parser.add_argument("pancake_metrics", type=Path)
    parser.add_argument("reference_smiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args()

    candidate_metrics = _load_metrics(args.candidate_metrics)
    pancake_metrics = _load_metrics(args.pancake_metrics)
    candidate_smiles = _generated_smiles(candidate_metrics)
    pancake_smiles = _generated_smiles(pancake_metrics)
    reference_smiles = _matched_reference(args.reference_smiles)

    candidate = _endpoint_report(
        candidate_metrics,
        chemistry_summary(candidate_smiles),
    )
    pancake = _endpoint_report(
        pancake_metrics,
        chemistry_summary(pancake_smiles),
    )
    reference = chemistry_summary(reference_smiles)
    report = {
        "format": "compose_v4_unconditional_chemistry_posthoc_comparison_v1",
        "decision_use": "posthoc_nonpromotional_preview_gate_missed",
        "candidate": candidate,
        "pancake": pancake,
        "matched_reference": reference,
        "limitations": {
            "candidate_samples": len(candidate_smiles),
            "pancake_samples": len(pancake_smiles),
            "minimum_preregistered_promotion_samples": 200,
            "promotion_authorized": False,
            "candidate_incomplete": bool(candidate_metrics.get("incomplete", False)),
            "candidate_declared_shards": candidate_metrics.get("declared_shards"),
            "candidate_completed_shards": candidate_metrics.get("completed_shards"),
            "candidate_missing_shard_indices": candidate_metrics.get(
                "missing_shard_indices"
            ),
            "note": (
                "The candidate missed the preregistered 0.65 family-accuracy "
                "preview gate and this bounded diagnostic is descriptive only. "
                "If candidate_incomplete is true, shard completion may bias the "
                "interim subset toward faster trajectories."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.grid.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    _render_grid(args.grid, candidate_smiles)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
