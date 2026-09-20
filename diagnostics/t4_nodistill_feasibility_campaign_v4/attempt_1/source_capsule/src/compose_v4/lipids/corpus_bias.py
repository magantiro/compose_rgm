"""Chemistry-aware diversity diagnostics for layered lipid corpora."""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


_MOTIFS = {
    "ester": Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"),
    "amide": Chem.MolFromSmarts("[NX3][CX3](=O)"),
    "carbonate": Chem.MolFromSmarts("[OX2][CX3](=O)[OX2]"),
    "disulfide": Chem.MolFromSmarts("[SX2][SX2]"),
    "acetal": Chem.MolFromSmarts("[CX4]([OX2])([OX2])"),
}


def murcko_scaffold_key(molecule: Chem.Mol) -> str:
    """Return a Murcko key while making the acyclic collapse explicit."""

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "<ACYCLIC>"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def heteroatom_connector_core(molecule: Chem.Mol) -> str:
    """Return the minimal atom-induced connector spanning all heteroatoms.

    Murcko scaffolds collapse acyclic lipids to an empty graph. This proxy keeps
    all heteroatoms and every shortest-path atom needed to connect them, while
    removing terminal carbon-only tails. It is a structural grouping key, not
    a claim about synthetic building blocks.
    """

    heteroatoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 6]
    if not heteroatoms:
        return "<NO_HETEROATOM>"
    keep = set(heteroatoms)
    for left_index, left in enumerate(heteroatoms):
        for right in heteroatoms[left_index + 1 :]:
            keep.update(Chem.GetShortestPath(molecule, left, right))
    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old_index in sorted(keep):
        old_to_new[old_index] = editable.AddAtom(Chem.Atom(molecule.GetAtomWithIdx(old_index)))
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in keep and end in keep:
            editable.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
            new_bond = editable.GetBondBetweenAtoms(old_to_new[begin], old_to_new[end])
            if new_bond is not None:
                new_bond.SetIsAromatic(bond.GetIsAromatic())
    core = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(core)
            return Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
    except Exception:
        # The atom-induced graph can expose aromatic valence at a cut boundary.
        # A stable atom/bond signature preserves grouping without inventing caps.
        atoms = ".".join(
            f"{molecule.GetAtomWithIdx(index).GetSymbol()}:{index}" for index in sorted(keep)
        )
        bonds = ".".join(
            sorted(
                f"{min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}-"
                f"{max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}:{bond.GetBondType()}"
                for bond in molecule.GetBonds()
                if bond.GetBeginAtomIdx() in keep and bond.GetEndAtomIdx() in keep
            )
        )
        return f"<UNSANITIZED_CONNECTOR>|{atoms}|{bonds}"


def _aliphatic_tail_proxy(molecule: Chem.Mol) -> tuple[int, int]:
    """Return count of long aliphatic termini and maximum heteroatom depth.

    A long tail terminus is a non-aromatic carbon with at most one aliphatic
    carbon neighbor and graph distance at least six from the nearest heteroatom
    through non-aromatic carbon atoms. This is a transparent topology proxy,
    not an exact head/linker/tail decomposition.
    """

    allowed = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
    }
    heteroatoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 6]
    distances: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()
    for heteroatom in heteroatoms:
        for neighbor in molecule.GetAtomWithIdx(heteroatom).GetNeighbors():
            index = neighbor.GetIdx()
            if index in allowed and index not in distances:
                distances[index] = 1
                queue.append((index, 1))
    while queue:
        index, distance = queue.popleft()
        for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors():
            next_index = neighbor.GetIdx()
            if next_index in allowed and next_index not in distances:
                distances[next_index] = distance + 1
                queue.append((next_index, distance + 1))
    termini: list[int] = []
    for index in allowed:
        carbon_neighbors = sum(
            neighbor.GetIdx() in allowed
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
        )
        if carbon_neighbors <= 1 and distances.get(index, 0) >= 6:
            termini.append(index)
    return len(termini), max(distances.values(), default=0)


def lipid_topology_features(molecule: Chem.Mol) -> dict[str, object]:
    """Compute interpretable molecule-level profile and lipid topology proxies."""

    formal_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    chiral_centers = Chem.FindMolChiralCenters(
        molecule,
        includeUnassigned=True,
        includeCIP=True,
        useLegacyImplementation=False,
    )
    ring_info = molecule.GetRingInfo()
    aromatic_ring_count = sum(
        all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        for ring in ring_info.AtomRings()
    )
    branch_points = sum(
        atom.GetAtomicNum() == 6
        and not atom.GetIsAromatic()
        and sum(
            neighbor.GetAtomicNum() == 6 and not neighbor.GetIsAromatic()
            for neighbor in atom.GetNeighbors()
        )
        >= 3
        for atom in molecule.GetAtoms()
    )
    cc_unsaturation = sum(
        bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        and not bond.GetIsAromatic()
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    )
    long_tail_count, max_tail_depth = _aliphatic_tail_proxy(molecule)
    motif_counts = {
        name: len(molecule.GetSubstructMatches(pattern))
        for name, pattern in _MOTIFS.items()
        if pattern is not None
    }
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    if branch_points == 0:
        branching_bin = "0"
    elif branch_points == 1:
        branching_bin = "1"
    else:
        branching_bin = "2plus"
    if long_tail_count == 0:
        long_tail_bin = "0"
    elif long_tail_count == 1:
        long_tail_bin = "1"
    elif long_tail_count == 2:
        long_tail_bin = "2"
    else:
        long_tail_bin = "3plus"
    degradable_names = ("ester", "carbonate", "disulfide", "acetal")
    degradable_signature = "+".join(
        name for name in degradable_names if motif_counts[name] > 0
    ) or "none_detected"
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
    return {
        "heavy_atoms": heavy_atoms,
        "size_bin": size_bin,
        "formal_charge": formal_charge,
        "charge_bin": "negative" if formal_charge < 0 else "positive" if formal_charge > 0 else "neutral",
        "elements": "|".join(elements),
        "stereogenic_atom_count": len(chiral_centers),
        "stereo_bin": "present" if chiral_centers else "none",
        "ring_count": ring_info.NumRings(),
        "ring_bin": "ring" if ring_info.NumRings() else "acyclic",
        "aromatic_ring_count": aromatic_ring_count,
        "aromatic_bin": "aromatic" if aromatic_ring_count else "nonaromatic",
        "carbon_branch_point_count": branch_points,
        "branching_bin": branching_bin,
        "cc_unsaturation_count": cc_unsaturation,
        "unsaturation_bin": "present" if cc_unsaturation else "none",
        "long_aliphatic_terminus_count": long_tail_count,
        "long_tail_bin": long_tail_bin,
        "max_aliphatic_tail_depth": max_tail_depth,
        "tail_architecture_proxy": (
            f"tails={long_tail_bin}|branch={branching_bin}|"
            f"unsaturation={'present' if cc_unsaturation else 'none'}"
        ),
        **{f"motif_{name}_count": count for name, count in motif_counts.items()},
        "cleavable_motif_proxy": degradable_signature,
        "murcko_scaffold": murcko_scaffold_key(molecule),
        "heteroatom_connector_core": heteroatom_connector_core(molecule),
    }


def probability_vector(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    counts = Counter(values)
    total = len(values)
    if total == 0:
        raise ValueError("cannot form a distribution from no values")
    return np.asarray([counts[category] / total for category in categories], dtype=float)


def weighted_probability_vector(
    values: Sequence[str], categories: Sequence[str], weights: np.ndarray
) -> np.ndarray:
    if len(values) != len(weights):
        raise ValueError("value/weight lengths differ")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    result = np.zeros(len(categories), dtype=float)
    category_to_index = {category: index for index, category in enumerate(categories)}
    for value, weight in zip(values, weights, strict=True):
        result[category_to_index[value]] += float(weight)
    return result / total


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in bits on aligned probability vectors."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("probability shapes differ")
    if not np.isclose(left.sum(), 1.0) or not np.isclose(right.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    midpoint = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray) -> float:
        mask = source > 0
        return float(np.sum(source[mask] * np.log2(source[mask] / midpoint[mask])))

    return 0.5 * (kl_divergence(left) + kl_divergence(right))


def hill_effective_numbers(values: Iterable[str]) -> dict[str, float]:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if counts.size == 0:
        return {"richness_q0": 0.0, "shannon_q1": 0.0, "simpson_q2": 0.0}
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "richness_q0": float(counts.size),
        "shannon_q1": math.exp(entropy),
        "simpson_q2": 1.0 / float(np.sum(probabilities**2)),
    }


def expected_weighted_unique_coverage(
    within_layer_probabilities: np.ndarray,
    novelty_weights: np.ndarray,
    layer_probability: float,
    draws: int,
) -> float:
    """Expected novelty-weighted unique auxiliary coverage after ``draws``."""

    probabilities = np.asarray(within_layer_probabilities, dtype=float)
    novelty = np.asarray(novelty_weights, dtype=float)
    if probabilities.shape != novelty.shape:
        raise ValueError("probability/novelty shapes differ")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("within-layer probabilities must sum to one")
    if not 0 <= layer_probability <= 1:
        raise ValueError("layer_probability must be in [0, 1]")
    inclusion = -np.expm1(draws * np.log1p(-layer_probability * probabilities))
    return float(np.sum(novelty * inclusion))


def choose_equal_priority_minimax(
    layer_probabilities: Sequence[float],
    normalized_realism_penalty: Sequence[float],
    normalized_coverage_utility: Sequence[float],
) -> dict[str, float]:
    """Choose the point minimizing worst normalized regret without a weight coefficient."""

    if not (
        len(layer_probabilities)
        == len(normalized_realism_penalty)
        == len(normalized_coverage_utility)
    ):
        raise ValueError("tradeoff curves have different lengths")
    candidates = []
    for probability, realism, coverage in zip(
        layer_probabilities,
        normalized_realism_penalty,
        normalized_coverage_utility,
        strict=True,
    ):
        regret = max(float(realism), 1.0 - float(coverage))
        candidates.append((regret, float(probability), float(realism), float(coverage)))
    regret, probability, realism, coverage = min(candidates)
    return {
        "layer_probability": probability,
        "normalized_realism_penalty": realism,
        "normalized_coverage_utility": coverage,
        "max_regret": regret,
    }
