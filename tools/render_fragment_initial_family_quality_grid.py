"""Render deterministic molecule examples for a quality audit, not selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "diagnostics/fragment_initial_family_dev_v1"
SOURCE = FOLDER / "quality_seed10_n20_all10.json"
DRUGS = ("ERLOTINIB", "FUTIBATINIB")
ARMS = ("baseline", "conditioned")


def main() -> None:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    records = json.loads(SOURCE.read_text())["molecules"]
    selected = []
    for drug in DRUGS:
        for arm in ARMS:
            pool = sorted(
                (r for r in records if r["drug"] == drug and r["arm"] == arm),
                key=lambda r: r["attempt_index"],
            )
            for quality_pass in (True, False):
                samples = [r for r in pool if r["quality_pass"] is quality_pass][:3]
                if len(samples) != 3:
                    raise RuntimeError(f"not enough examples for {drug}/{arm}/{quality_pass}")
                selected.extend(samples)
    molecules = [Chem.MolFromSmiles(r["smiles"]) for r in selected]
    if any(mol is None for mol in molecules):
        raise RuntimeError("a selected molecule does not parse")
    legends = [
        f"{r['drug'][:4]} {r['arm'][:4]} #{r['attempt_index']} Q{r['qed']:.2f} SA{r['sa']:.2f}"
        for r in selected
    ]
    image = Draw.MolsToGridImage(molecules, molsPerRow=6, subImgSize=(220, 190), legends=legends)
    with tempfile.NamedTemporaryFile(dir=FOLDER, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    image.save(temporary)
    os.replace(temporary, FOLDER / "quality_examples_seed10.png")
    manifest = {
        "schema": "fragment_initial_family_quality_grid_v1",
        "selection": "first three distinct emitted quality pass and first three quality fail per drug/arm in attempt order",
        "input_sha256": source_hash,
        "rows": [
            {
                "drug": r["drug"],
                "arm": r["arm"],
                "attempt_index": r["attempt_index"],
                "smiles": r["smiles"],
                "qed": r["qed"],
                "sa": r["sa"],
                "quality_pass": r["quality_pass"],
            }
            for r in selected
        ],
    }
    output = FOLDER / "quality_examples_seed10.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=FOLDER, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
