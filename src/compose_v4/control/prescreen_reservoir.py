"""Select a structurally diverse reservoir from an oracle-ranked prescreen prefix.

WHY A DEEPER PREFIX, AND WHY DIVERSITY RATHER THAN A DEEPER TOP-N.  Measured on the frozen
ZINC250k prescreen table for the three structural-gap rows, best ECFP4-to-reference is FLAT
from K=40 to K=5000 (celecoxib 0.3478 -> 0.3556; troglitazone and thiothixene unchanged).
Since ECFP4-to-reference is essentially the rediscovery oracle for those tasks, that
flatness means a deeper prefix buys NO additional oracle-rank quality.  What it buys is
scaffold coverage: thiothixene's defining thioxanthene tricycle is absent from the top 40
and its best donor sits at RANK 41, one position outside the cutoff.  So the rule here
keeps the official oracle as the only ranking signal and spends the widened prefix on
TARGET-BLIND structural spread.

INFORMATION BOUNDARY.  The only task-specific quantity consumed is the official oracle
score that the declared prescreen already paid for.  Diversity is measured between
CANDIDATES -- never against a reference, a target fingerprint, a target scaffold or a
motif.  This module imports no reference and cannot: `select_reservoir` is given scores
and SMILES and nothing else.

THE HEAD IS KEPT ON PURPOSE.  `head_fraction` of the bank is the plain top of the oracle
ranking, so the selection can never trade away the strongest molecules for spread; the
remainder is a greedy max-min over structural distance, which is what reaches rank 41.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

#: Default widened prefix. The motif curves rise steeply to ~1000 and flatten well before
#: 5000, and a deeper prefix costs nothing at selection time because the scores are already
#: paid for -- but it does admit weaker molecules, which is why the head is preserved.
DEFAULT_PREFIX = 2000


@dataclass(frozen=True)
class ReservoirEntry:
    smiles: str
    oracle_score: float
    oracle_rank: int
    scaffold: str
    ring_signature: tuple[int, ...]
    selected_by: str


def _features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rings = tuple(sorted(len(r) for r in mol.GetRingInfo().AtomRings()))
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except (ValueError, RuntimeError):
        scaffold = ""
    return {
        "mol": mol,
        "morgan": AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
        "atom_pair": rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol),
        "torsion": rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol),
        "scaffold": scaffold,
        "rings": rings,
    }


def _distance(a, b) -> float:
    """Mean complement-Tanimoto over three complementary representations.

    Morgan alone collapses scaffold families that differ by ring topology; atom-pair and
    torsion carry the through-bond geometry that distinguishes a fused tricycle from two
    rings on a linker, which is exactly the distinction thiothixene turns on.
    """
    return float(np.mean([
        1.0 - DataStructs.TanimotoSimilarity(a["morgan"], b["morgan"]),
        1.0 - DataStructs.TanimotoSimilarity(a["atom_pair"], b["atom_pair"]),
        1.0 - DataStructs.TanimotoSimilarity(a["torsion"], b["torsion"]),
    ]))


def select_scaffold_coverage(
    ranked: list[tuple[float, str]],
    count: int,
    *,
    prefix: int = DEFAULT_PREFIX,
) -> list[ReservoirEntry]:
    """Walk the oracle ranking and keep the FIRST molecule of each new Murcko scaffold.

    MEASURED, and this is why it is the default: on thiothixene at count=40 it reaches
    RANK 41 -- the thioxanthene-core donor the shipped top-40 misses by one position --
    while looking only as deep as rank 44. On troglitazone at count=16 it holds BOTH
    thiazolidinedione donors (ranks 1 and 3). Max-min distance does neither: it spends
    its picks on outliers and reached the core only at a 200-molecule donor pool.

    So the repair is not "widen to K=5000". It is "stop spending the bank on repeated
    scaffolds", which is a far smaller and better-founded change: the ranking is still
    purely the official oracle, and the only thing added is a deduplication key that
    knows nothing about any target.
    """
    picked: list[ReservoirEntry] = []
    seen_scaffold: set[str] = set()
    seen_smiles: set[str] = set()
    for rank, (score, smiles) in enumerate(ranked[:prefix], start=1):
        if len(picked) >= count:
            break
        if smiles in seen_smiles:
            continue
        feats = _features(smiles)
        if feats is None:
            continue
        seen_smiles.add(smiles)
        if feats["scaffold"] in seen_scaffold:
            continue
        seen_scaffold.add(feats["scaffold"])
        picked.append(ReservoirEntry(
            smiles=smiles, oracle_score=float(score), oracle_rank=rank,
            scaffold=feats["scaffold"], ring_signature=feats["rings"],
            selected_by="scaffold_coverage"))
    if not picked:
        raise ValueError("no parsable candidates in the prescreen prefix")
    return picked


def select_reservoir(
    ranked: list[tuple[float, str]],
    count: int,
    *,
    prefix: int = DEFAULT_PREFIX,
    head_fraction: float = 0.25,
) -> list[ReservoirEntry]:
    """Max-min structural spread. KEPT, but NOT the default -- see the gate result.

    It failed the zero-oracle gate at bank size 16 (troglitazone ring-signature coverage
    fell 9 -> 7, and it did not reach the thioxanthene core at all), because max-min
    selects outliers rather than sweeping scaffold families. `select_scaffold_coverage`
    is the rule that passes. Retained so the comparison stays reproducible.

    Selection is deterministic: no RNG, so a bank is reproducible from its inputs.
    """
    if count <= 0:
        raise ValueError("reservoir size must be positive")
    if not 0.0 <= head_fraction <= 1.0:
        raise ValueError("head_fraction is a proportion")
    pool: list[tuple[int, float, str, dict]] = []
    seen: set[str] = set()
    for rank, (score, smiles) in enumerate(ranked[:prefix], start=1):
        if smiles in seen:
            continue
        feats = _features(smiles)
        if feats is None:
            continue
        seen.add(smiles)
        pool.append((rank, float(score), smiles, feats))
    if not pool:
        raise ValueError("no parsable candidates in the prescreen prefix")

    n_head = min(len(pool), max(1, round(count * head_fraction)))
    chosen = list(pool[:n_head])
    chosen_by = {row[2]: "oracle_head" for row in chosen}

    # Greedy max-min: repeatedly take the candidate furthest from everything already held.
    remaining = pool[n_head:]
    best_distance = {
        row[2]: min(_distance(row[3], held[3]) for held in chosen) for row in remaining
    }
    while len(chosen) < count and remaining:
        pick = max(remaining, key=lambda row: (best_distance[row[2]], -row[0]))
        remaining.remove(pick)
        chosen.append(pick)
        chosen_by[pick[2]] = "structural_spread"
        for row in remaining:
            d = _distance(row[3], pick[3])
            best_distance[row[2]] = min(best_distance[row[2]], d)
    return [
        ReservoirEntry(
            smiles=smiles,
            oracle_score=score,
            oracle_rank=rank,
            scaffold=feats["scaffold"],
            ring_signature=feats["rings"],
            selected_by=chosen_by[smiles],
        )
        for rank, score, smiles, feats in chosen
    ]
