"""Split-first observed one/two-boundary regions, without a ring/element whitelist.

The corpus provides candidate structural content, not endpoint reward. Runtime
programs still require shared exact compilation, fragment admission and a finite
score from the frozen COMPOSE model. Extraction and runtime support are distinct.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

from compose_v4.benchmark.training_attachment_fragments import atom_context, physical_sha256
from compose_v4.data.scaffold_partition import murcko_scaffold, partition_for_scaffold


def observed_regions(mol):
    """Whole components behind one/two BRICS boundary cuts, preserving ring systems."""
    source = Chem.Mol(mol)
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    cuts = []
    for (left, right), _ in BRICS.FindBRICSBonds(source):
        bond = source.GetBondBetweenAtoms(left, right)
        if bond.GetBondType() == Chem.BondType.SINGLE:
            cuts.append((bond.GetIdx(), left, right))
    cuts.sort()
    for arity in (1, 2):
        for chosen in combinations(cuts, arity):
            fragmented = Chem.FragmentOnBonds(
                source,
                [c[0] for c in chosen],
                addDummies=True,
                dummyLabels=[(i + 1, i + 1) for i in range(arity)],
            )
            for piece in Chem.GetMolFrags(fragmented, asMols=True):
                dummies = [a for a in piece.GetAtoms() if a.GetAtomicNum() == 0]
                if len(dummies) != arity or {a.GetIsotope() for a in dummies} != set(
                    range(1, arity + 1)
                ):
                    continue
                contexts = []
                for dummy in sorted(dummies, key=lambda a: a.GetIsotope()):
                    root = dummy.GetNeighbors()[0].GetAtomMapNum() - 1
                    _, left, right = chosen[dummy.GetIsotope() - 1]
                    external = right if root == left else left
                    contexts.append(atom_context(source.GetAtomWithIdx(external)))
                for atom in piece.GetAtoms():
                    atom.SetAtomMapNum(0)
                yield tuple(contexts), Chem.MolToSmiles(piece, canonical=True)


def build_region_catalog(
    source: Path,
    *,
    expected_sha256: str,
    excluded_canonical: frozenset[str],
    training_molecules: int = 10_000,
):
    if physical_sha256(source) != expected_sha256:
        raise ValueError(f"source hash mismatch: {source}")
    if training_molecules < 1:
        raise ValueError("training_molecules must be positive")
    counts = Counter()
    regions = defaultdict(list)
    accepted_rows, seen = [], set()
    for row_index, text in enumerate(source.open(encoding="utf-8"), 1):
        counts["scanned_rows"] += 1
        mol = Chem.MolFromSmiles(text.strip())
        if mol is None:
            counts["invalid_source"] += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in excluded_canonical:
            counts["benchmark_reference_excluded"] += 1
            continue
        if canonical in seen:
            counts["duplicate_source"] += 1
            continue
        seen.add(canonical)
        scaffold = murcko_scaffold(canonical)
        if scaffold is None:
            counts["unassignable_scaffold"] += 1
            continue
        partition = partition_for_scaffold(scaffold)
        if partition != "train":
            counts[f"excluded_{partition}"] += 1
            continue
        if not 1 <= mol.GetNumHeavyAtoms() <= 40 or len(Chem.GetMolFrags(mol)) != 1:
            counts["outside_source_graph_size_or_connectivity"] += 1
            continue
        accepted_rows.append(row_index)
        # Partition and benchmark-identity admission precede every extraction.
        for contexts, rooted in observed_regions(mol):
            regions[(contexts, rooted)].append(row_index)
        if len(accepted_rows) == training_molecules:
            break
    if len(accepted_rows) != training_molecules:
        raise ValueError("source exhausted before predeclared training-subset size")
    entries = []
    for (contexts, rooted), rows in sorted(regions.items()):
        mol = Chem.MolFromSmiles(rooted)
        entries.append(
            {
                "contexts": list(contexts),
                "rooted_smiles": rooted,
                "heavy_atoms": sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms()),
                "ring_count": mol.GetRingInfo().NumRings(),
                "aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
                "source_stereo_annotations": sum(
                    a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()
                ),
                "elements": sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()}),
                "occurrences": len(rows),
                "source_rows": rows,
            }
        )
    return {
        "schema": "split_first_training_region_catalog_v1",
        "source": str(source.resolve()),
        "source_sha256": expected_sha256,
        "split": {
            "algorithm": "murcko+carbonized-wl3",
            "version": 2,
            "salt": "ringcore-v1",
            "ratios": [0.9, 0.05, 0.05],
            "partition": "train",
        },
        "subset_rule": "first requested eligible train molecules in frozen source order",
        "training_molecules": training_molecules,
        "accepted_source_rows": accepted_rows,
        "exclusions": dict(sorted(counts.items())),
        "entries": entries,
        "stereochemistry": "source annotations retained; exact graph executor has its existing stereo support only",
        "candidate_source": "observed regions, not a QED/SA-ranked library",
    }
