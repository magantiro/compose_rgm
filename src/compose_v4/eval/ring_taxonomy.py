"""Overlapping ring-taxonomy statistics for molecular distribution audits."""

from __future__ import annotations

from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def ring_taxonomy_report(smiles: tuple[str, ...]) -> dict[str, object]:
    """Summarize topological and electronic ring frequencies.

    Topology labels intentionally overlap: a molecule may be fused and bridged,
    for example. Ring counts use RDKit's symmetrized SSSR representation, while
    cycle rank is reported separately as a basis-independent graph invariant.
    """

    if not smiles:
        raise ValueError("smiles must be non-empty")
    molecule_features: Counter[str] = Counter()
    ring_features: Counter[str] = Counter()
    ring_sizes: Counter[int] = Counter()
    cycle_ranks = []
    ring_system_counts = []

    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"invalid SMILES in ring audit: {text!r}")
        atom_rings = tuple(frozenset(ring) for ring in molecule.GetRingInfo().AtomRings())
        bond_rings = tuple(tuple(ring) for ring in molecule.GetRingInfo().BondRings())
        if len(atom_rings) != len(bond_rings):
            raise RuntimeError("RDKit atom/bond ring bases disagree")

        n_atoms = molecule.GetNumAtoms()
        n_bonds = molecule.GetNumBonds()
        cycle_rank = n_bonds - n_atoms + len(Chem.GetMolFrags(molecule))
        cycle_ranks.append(int(cycle_rank))
        ring_system_counts.append(_ring_system_count(atom_rings))

        if not atom_rings:
            molecule_features["acyclic"] += 1
            continue
        molecule_features["cyclic"] += 1
        molecule_features["monocyclic"] += int(cycle_rank == 1)
        molecule_features["polycyclic"] += int(cycle_rank >= 2)

        fused = any(
            len(left & right) >= 2
            for index, left in enumerate(atom_rings)
            for right in atom_rings[index + 1 :]
        )
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0
        bridgeheads = int(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule))
        molecule_features["fused"] += int(fused)
        molecule_features["spiro"] += int(spiro)
        molecule_features["bridged"] += int(bridgeheads > 0)
        molecule_features["polybridged_or_cage_candidate"] += int(bridgeheads >= 3)
        molecule_features["small_ring_3_or_4"] += int(
            any(len(ring) in {3, 4} for ring in atom_rings)
        )
        molecule_features["macrocycle_ge_12"] += int(
            any(len(ring) >= 12 for ring in atom_rings)
        )

        molecule_has_aromatic = False
        molecule_has_heterocycle = False
        molecule_has_saturated = False
        molecule_has_unsaturated_nonaromatic = False
        molecule_has_charged_ring = False
        for atom_ring, bond_ring in zip(atom_rings, bond_rings):
            ring_sizes[len(atom_ring)] += 1
            atoms = [molecule.GetAtomWithIdx(int(index)) for index in atom_ring]
            bonds = [molecule.GetBondWithIdx(int(index)) for index in bond_ring]
            aromatic = bool(bonds) and all(bond.GetIsAromatic() for bond in bonds)
            saturated = bool(bonds) and all(
                bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds
            )
            heterocycle = any(atom.GetAtomicNum() != 6 for atom in atoms)
            charged = any(atom.GetFormalCharge() != 0 for atom in atoms)

            ring_features["aromatic"] += int(aromatic)
            ring_features["saturated"] += int(saturated)
            ring_features["unsaturated_nonaromatic"] += int(
                not aromatic and not saturated
            )
            ring_features["heterocycle"] += int(heterocycle)
            ring_features["carbocycle"] += int(not heterocycle)
            ring_features["charged"] += int(charged)
            molecule_has_aromatic |= aromatic
            molecule_has_heterocycle |= heterocycle
            molecule_has_saturated |= saturated
            molecule_has_unsaturated_nonaromatic |= not aromatic and not saturated
            molecule_has_charged_ring |= charged

        molecule_features["aromatic"] += int(molecule_has_aromatic)
        molecule_features["heterocycle"] += int(molecule_has_heterocycle)
        molecule_features["saturated_ring"] += int(molecule_has_saturated)
        molecule_features["unsaturated_nonaromatic_ring"] += int(
            molecule_has_unsaturated_nonaromatic
        )
        molecule_features["charged_ring"] += int(molecule_has_charged_ring)

    n_molecules = len(smiles)
    n_rings = sum(ring_sizes.values())
    return {
        "molecule_count": n_molecules,
        "molecule_prevalence": {
            name: {
                "count": int(molecule_features[name]),
                "fraction": float(molecule_features[name] / n_molecules),
            }
            for name in (
                "acyclic",
                "cyclic",
                "monocyclic",
                "polycyclic",
                "aromatic",
                "saturated_ring",
                "unsaturated_nonaromatic_ring",
                "heterocycle",
                "charged_ring",
                "fused",
                "spiro",
                "bridged",
                "polybridged_or_cage_candidate",
                "small_ring_3_or_4",
                "macrocycle_ge_12",
            )
        },
        "ring_count": int(n_rings),
        "ring_class_counts": {
            name: {
                "count": int(ring_features[name]),
                "fraction_of_rings": float(
                    ring_features[name] / n_rings if n_rings else 0.0
                ),
            }
            for name in (
                "aromatic",
                "saturated",
                "unsaturated_nonaromatic",
                "heterocycle",
                "carbocycle",
                "charged",
            )
        },
        "ring_size_counts": {
            str(size): int(count) for size, count in sorted(ring_sizes.items())
        },
        "means": {
            "sssr_rings_per_molecule": float(n_rings / n_molecules),
            "cycle_rank": float(np.mean(cycle_ranks)),
            "ring_systems": float(np.mean(ring_system_counts)),
        },
    }


def _ring_system_count(atom_rings: tuple[frozenset[int], ...]) -> int:
    if not atom_rings:
        return 0
    unseen = set(range(len(atom_rings)))
    systems = 0
    while unseen:
        systems += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            neighbors = {
                other
                for other in unseen
                if atom_rings[current] & atom_rings[other]
            }
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return systems
