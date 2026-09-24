"""Split-first observed smaller-side regions for localized scaffold decoration.

Only the topology of a training cut determines which side is a candidate
pendant. No benchmark prompt, QED, SA, target or drug label enters extraction.
The output is structural proposal content, not a new executable operator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from rdkit import Chem

from compose_v4.benchmark.training_attachment_fragments import atom_context, physical_sha256


def observed_pendant_sides(molecule: Chem.Mol) -> tuple[dict, ...]:
    """Return every smaller component behind an acyclic single-bond cut.

    Equal-size sides both remain eligible; they each receive one observation
    before the source molecule's total mass is normalized below.
    """
    source = Chem.Mol(molecule)
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    total = source.GetNumHeavyAtoms()
    observed = []
    for bond in source.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            source.GetAtomWithIdx(left).GetAtomicNum() < 2
            or source.GetAtomWithIdx(right).GetAtomicNum() < 2
        ):
            continue
        fragmented = Chem.FragmentOnBonds(
            source, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)]
        )
        pieces = Chem.GetMolFrags(fragmented, asMols=True)
        if len(pieces) != 2:
            raise ValueError("an acyclic source bond failed to separate two components")
        for piece in pieces:
            size = piece.GetNumHeavyAtoms()
            if size > total - size:
                continue
            dummies = [atom for atom in piece.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(dummies) != 1 or dummies[0].GetIsotope() != 1 or dummies[0].GetDegree() != 1:
                raise ValueError("cut side lost its unique one-boundary dummy")
            root = dummies[0].GetNeighbors()[0].GetAtomMapNum() - 1
            if root not in (left, right):
                raise ValueError("cut side root is not a source bond endpoint")
            external = right if root == left else left
            context = atom_context(source.GetAtomWithIdx(external))
            stereo = sum(
                atom.GetAtomicNum() != 0 and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                for atom in piece.GetAtoms()
            )
            for atom in piece.GetAtoms():
                atom.SetAtomMapNum(0)
            rooted = Chem.MolToSmiles(piece, canonical=True)
            observed.append(
                {
                    "context": context,
                    "rooted_smiles": rooted,
                    "heavy_atoms": size,
                    "ring_count": piece.GetRingInfo().NumRings(),
                    "source_stereo_annotations": stereo,
                }
            )
    return tuple(observed)


def build_pendant_catalog(region_catalog: dict, source: Path) -> dict:
    """Reuse exactly the frozen training rows; weight each molecule once."""
    if region_catalog.get("schema") != "split_first_training_region_catalog_v1":
        raise ValueError("source catalog is not the split-first training region catalog")
    if physical_sha256(source) != region_catalog["source_sha256"]:
        raise ValueError(f"training molecular source hash changed: {source}")
    accepted = region_catalog["accepted_source_rows"]
    if (
        not accepted
        or accepted != sorted(set(accepted))
        or len(accepted) != region_catalog["training_molecules"]
    ):
        raise ValueError("frozen training row identities are malformed")
    selected = set(accepted)
    counts = Counter()
    weights: dict[tuple[str, str], Fraction] = defaultdict(Fraction)
    source_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict] = {}
    verified = []
    with source.open(encoding="utf-8") as handle:
        for row_index, text in enumerate(handle, 1):
            if row_index > accepted[-1]:
                break
            if row_index not in selected:
                continue
            molecule = Chem.MolFromSmiles(text.strip())
            if molecule is None or not 1 <= molecule.GetNumHeavyAtoms() <= 40:
                raise ValueError(f"frozen training row {row_index} changed support")
            verified.append(row_index)
            observed = observed_pendant_sides(molecule)
            counts["training_molecules"] += 1
            if not observed:
                counts["molecules_without_bridge_pendant"] += 1
                continue
            counts["molecules_with_pendant"] += 1
            counts["observed_cut_sides"] += len(observed)
            unit = Fraction(1, len(observed))
            for item in observed:
                key = item["context"], item["rooted_smiles"]
                if key in metadata and metadata[key] != item:
                    raise ValueError(f"rooted pendant metadata differs across training rows: {key}")
                metadata[key] = item
                weights[key] += unit
                source_rows[key].append(row_index)
                if item["heavy_atoms"] == 1:
                    counts["one_atom_observations"] += 1
                if item["ring_count"]:
                    counts["ring_containing_observations"] += 1
    if verified != accepted:
        raise ValueError("not every frozen training row was recovered in source order")
    entries = []
    for key in sorted(metadata):
        item = metadata[key]
        entries.append(
            {
                **item,
                "source_balanced_weight": float(weights[key]),
                "occurrences": len(source_rows[key]),
                "source_rows": source_rows[key],
            }
        )
    if (
        abs(sum(e["source_balanced_weight"] for e in entries) - counts["molecules_with_pendant"])
        > 1e-8
    ):
        raise ValueError("source-balanced pendant mass differs from contributing molecule count")
    return {
        "schema": "split_first_training_pendant_catalog_v1",
        "source": str(source.resolve()),
        "source_sha256": region_catalog["source_sha256"],
        "split": region_catalog["split"],
        "training_molecules": len(accepted),
        "accepted_source_rows": accepted,
        "source_balance": "each molecule has unit mass over its observed smaller-side bridge cuts",
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "census": dict(sorted(counts.items())),
        "entries": entries,
    }
