"""Deterministic SCAFFOLD-KEYED train/validation/test partitioning of corruption source molecules.

Why not reuse the corpus split: ``load_organic_corpus_split`` partitions by INDEX under a deterministic
shuffle, so two molecules sharing a Bemis-Murcko scaffold can land in different partitions. For an EDIT
model that is a leak with teeth -- corrupting a source produces many trajectories around one scaffold, so a
scaffold present in both train and validation lets the model memorize a scaffold's edit neighbourhood and
score well on "held-out" data it has effectively seen.

The rule implemented here: **the scaffold is the partition unit.** A molecule's partition is a
deterministic function of its scaffold hash, so every molecule sharing a scaffold -- and therefore every
trajectory generated from any of them -- lands in the same partition. Assignment happens BEFORE trajectory
generation, never after.

Guarantees (asserted by ``verify_partition_disjointness``):
  * zero SOURCE overlap across partitions;
  * zero SCAFFOLD overlap across partitions;
  * reproducible from (scaffold, salt) alone -- no RNG state, no ordering dependence.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

PARTITIONS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.90, 0.05, 0.05)
_HASH_BUCKETS = 10_000


def _acyclic_key(mol) -> str:
    """Partition key for a molecule with no ring system.

    Bemis-Murcko returns the EMPTY scaffold for acyclics, so keying on it directly would drop every acyclic
    molecule into one bucket -- catastrophic for balance. Heavy-atom count alone is leak-safe but far too
    coarse: measured on held-out broad-organic data, 60 acyclic molecules collapsed into 25 keys and landed
    59 train / 1 validation / 0 test, leaving the held-out sets with no acyclic coverage at all.

    We therefore key on a topology-preserving, label-reduced skeleton: every heavy atom is relabelled to a
    single type and bond orders are discarded, then a Weisfeiler-Lehman graph hash identifies the shape.
    Molecules sharing a carbon skeleton stay confined to one partition (the leak guarantee is per-key and is
    preserved), while genuinely different skeletons can separate.
    """
    import networkx as nx

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), label="C")          # carbonized: shape only, not composition
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    try:
        shape = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", iterations=3)[:16]
    except Exception:  # noqa: BLE001 -- fall back to the coarse key rather than failing assignment
        shape = "nowl"
    return f"<acyclic:{mol.GetNumHeavyAtoms()}:{shape}>"


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None when RDKit cannot parse the molecule.

    Acyclic molecules have no ring system and are keyed by a label-reduced skeleton hash instead -- see
    ``_acyclic_key`` for why the empty scaffold and the heavy-atom count are both inadequate.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # noqa: BLE001 -- unparseable molecules are simply unassignable
        return None
    if not scaffold:
        return _acyclic_key(mol)
    return scaffold


def _bucket(scaffold: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}\x00{scaffold}".encode()).hexdigest()
    return int(digest[:8], 16) % _HASH_BUCKETS


def partition_for_scaffold(
    scaffold: str, *, salt: str = "ringcore-v1", ratios: tuple[float, float, float] = DEFAULT_RATIOS
) -> str:
    """Deterministic partition for a scaffold. Pure function of (scaffold, salt) -- no ordering, no RNG."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"partition ratios must sum to 1, got {ratios}")
    bucket = _bucket(scaffold, salt)
    train_cut = ratios[0] * _HASH_BUCKETS
    val_cut = (ratios[0] + ratios[1]) * _HASH_BUCKETS
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "validation"
    return "test"


def assign_partitions(
    smiles_list,
    *,
    salt: str = "ringcore-v1",
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> tuple[dict[str, str], dict[str, str], dict]:
    """Assign every source molecule to a partition via its scaffold.

    Returns ``(partition_by_smiles, scaffold_by_smiles, stats)``. Molecules whose scaffold cannot be
    computed are DROPPED (recorded in stats) rather than silently defaulted into train.
    """
    partition_by_smiles: dict[str, str] = {}
    scaffold_by_smiles: dict[str, str] = {}
    unassignable: list[str] = []
    for smiles in smiles_list:
        scaffold = murcko_scaffold(smiles)
        if scaffold is None:
            unassignable.append(smiles)
            continue
        scaffold_by_smiles[smiles] = scaffold
        partition_by_smiles[smiles] = partition_for_scaffold(scaffold, salt=salt, ratios=ratios)

    per_partition = Counter(partition_by_smiles.values())
    scaffolds_per_partition: dict[str, set[str]] = defaultdict(set)
    for smiles, part in partition_by_smiles.items():
        scaffolds_per_partition[part].add(scaffold_by_smiles[smiles])
    total = max(len(partition_by_smiles), 1)
    stats = {
        "salt": salt,
        "ratios": list(ratios),
        "assigned_sources": len(partition_by_smiles),
        "unassignable_sources": len(unassignable),
        "sources_per_partition": dict(per_partition),
        "source_fraction_per_partition": {
            k: round(v / total, 4) for k, v in per_partition.items()
        },
        "scaffolds_per_partition": {k: len(v) for k, v in scaffolds_per_partition.items()},
        "unique_scaffolds": len(set(scaffold_by_smiles.values())),
    }
    return partition_by_smiles, scaffold_by_smiles, stats


def verify_partition_disjointness(
    partition_by_smiles: dict[str, str], scaffold_by_smiles: dict[str, str]
) -> dict:
    """Assert zero SOURCE and zero SCAFFOLD overlap across partitions. Raises on any leak."""
    sources: dict[str, set[str]] = defaultdict(set)
    scaffolds: dict[str, set[str]] = defaultdict(set)
    for smiles, part in partition_by_smiles.items():
        sources[part].add(smiles)
        scaffolds[part].add(scaffold_by_smiles[smiles])

    leaks: list[str] = []
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            shared_sources = sources[a] & sources[b]
            shared_scaffolds = scaffolds[a] & scaffolds[b]
            if shared_sources:
                leaks.append(f"{a}/{b} share {len(shared_sources)} source molecules")
            if shared_scaffolds:
                leaks.append(f"{a}/{b} share {len(shared_scaffolds)} scaffolds")
    if leaks:
        raise ValueError("partition leakage detected: " + "; ".join(leaks))
    return {
        "source_overlap": 0,
        "scaffold_overlap": 0,
        "sources_per_partition": {k: len(v) for k, v in sources.items()},
        "scaffolds_per_partition": {k: len(v) for k, v in scaffolds.items()},
        "verified": True,
    }
