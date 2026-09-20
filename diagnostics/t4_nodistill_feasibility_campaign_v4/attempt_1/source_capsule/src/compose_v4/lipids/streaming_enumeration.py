"""Memory-bounded deterministic selection for route-certified lipid products."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def stable_priority(namespace: str, identity: str) -> int:
    """Return a reproducible 128-bit priority; lower values rank first."""

    payload = f"{namespace}\0{identity}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def canonicalize_product(molecule: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    """Sanitize and canonicalize one connected product without changing charge/tautomer."""

    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    if len(Chem.GetMolFrags(molecule)) != 1:
        return None
    canonical = Chem.MolToSmiles(
        Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    parsed = Chem.MolFromSmiles(canonical)
    if parsed is None:
        return None
    return canonical, parsed


def product_bins(molecule: Chem.Mol) -> dict[str, str]:
    """Compute cheap product-level axes used for stratified streaming selection."""

    heavy_atoms = molecule.GetNumHeavyAtoms()
    if heavy_atoms <= 40:
        size_bin = "le40"
    elif heavy_atoms <= 64:
        size_bin = "41_64"
    elif heavy_atoms <= 96:
        size_bin = "65_96"
    elif heavy_atoms <= 128:
        size_bin = "97_128"
    else:
        size_bin = "gt128"
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    charge_bin = "negative" if charge < 0 else "positive" if charge > 0 else "neutral"
    carbon_branch_points = sum(
        atom.GetAtomicNum() == 6 and atom.GetDegree() >= 3 and not atom.GetIsAromatic()
        for atom in molecule.GetAtoms()
    )
    branching_bin = "0" if carbon_branch_points == 0 else "1" if carbon_branch_points == 1 else "2plus"
    unsaturated_bonds = sum(
        bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        and not bond.GetIsAromatic()
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    )
    return {
        "size_bin": size_bin,
        "charge_bin": charge_bin,
        "branching_bin": branching_bin,
        "unsaturation_bin": "present" if unsaturated_bonds else "none",
        "ring_bin": "ring" if rdMolDescriptors.CalcNumRings(molecule) else "acyclic",
    }


@dataclass(frozen=True)
class CandidateProduct:
    """One route-replayable unique product candidate."""

    canonical_smiles: str
    reaction_id: str
    reactant_ids: tuple[str, ...]
    reactant_roles: tuple[str, ...]
    architecture_tags: Mapping[str, str]
    product_bins: Mapping[str, str]

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_smiles.encode()).hexdigest()

    def stratum(self, axes: tuple[str, ...]) -> tuple[str, ...]:
        values: dict[str, str] = {
            "reaction_family": self.reaction_id,
            **self.architecture_tags,
            **self.product_bins,
        }
        missing = [axis for axis in axes if axis not in values]
        if missing:
            raise KeyError(f"candidate lacks stratification axes: {missing}")
        return tuple(values[axis] for axis in axes)


class StratifiedReservoir:
    """Keep the lowest deterministic hashes per stratum using bounded memory.

    Exact global molecular deduplication must happen before ``consider`` (for a
    million-scale run, use a disk-backed unique index). This class solves the
    separate problem of deterministic quota selection without materializing a
    Cartesian library in memory.
    """

    def __init__(
        self,
        *,
        axes: Iterable[str],
        quota_per_stratum: int,
        seed: str,
    ) -> None:
        self.axes = tuple(axes)
        if not self.axes:
            raise ValueError("at least one stratification axis is required")
        if quota_per_stratum < 1:
            raise ValueError("quota_per_stratum must be positive")
        self.quota_per_stratum = quota_per_stratum
        self.seed = seed
        self._heaps: dict[tuple[str, ...], list[tuple[int, str, CandidateProduct]]] = {}

    def consider(self, candidate: CandidateProduct) -> None:
        stratum = candidate.stratum(self.axes)
        priority = stable_priority(self.seed, candidate.identity)
        heap = self._heaps.setdefault(stratum, [])
        entry = (-priority, candidate.identity, candidate)
        if len(heap) < self.quota_per_stratum:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def selected(self) -> list[CandidateProduct]:
        selected: list[tuple[tuple[str, ...], int, str, CandidateProduct]] = []
        for stratum, heap in self._heaps.items():
            for negative_priority, identity, candidate in heap:
                selected.append((stratum, -negative_priority, identity, candidate))
        selected.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[-1] for item in selected]

    @property
    def stratum_counts(self) -> dict[tuple[str, ...], int]:
        return {stratum: len(heap) for stratum, heap in sorted(self._heaps.items())}
