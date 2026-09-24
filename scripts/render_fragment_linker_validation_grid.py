"""Render deterministic first-distinct linker outcomes for qualitative review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    artifact = json.loads(args.artifact.read_text())
    rows = artifact["results"]["linker_design"]["per_drug"]
    molecules = []
    legends = []
    for drug, entries in sorted(rows.items()):
        if len(entries) != 1:
            raise ValueError(f"expected one seed for {drug}")
        row = entries[0]
        if len(row["committed_endpoint_smiles"]) != len(row["realized_linker_lengths"]):
            raise ValueError(f"molecule/length mismatch for {drug}")
        seen = set()
        chosen = 0
        for smiles, length in zip(
            row["committed_endpoint_smiles"], row["realized_linker_lengths"], strict=True
        ):
            if smiles in seen:
                continue
            seen.add(smiles)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"unparseable committed molecule for {drug}: {smiles}")
            molecules.append(mol)
            legends.append(f"{drug}  path={length}  seed={row['seeded_linker_length']}")
            chosen += 1
            if chosen == 2:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid = Draw.MolsToGridImage(
        molecules,
        legends=legends,
        molsPerRow=2,
        subImgSize=(420, 300),
        useSVG=False,
    )
    grid.save(args.output)
    print(f"rendered {len(molecules)} molecules to {args.output}")


if __name__ == "__main__":
    main()
