"""Visualize first saved completions, without quality-based selection or scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-shard", type=Path, required=True)
    parser.add_argument("--baseline-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    program = json.loads(args.program_shard.read_text())
    baseline = json.loads(args.baseline_shard.read_text())
    if (program["drug"], program["task"]) != (baseline["drug"], baseline["task"]):
        raise ValueError("preview shards do not share the exact prompt")
    samples = [("supplied core", program["start_smiles"])]
    for label, records in (
        ("frozen local", baseline["attempt_records"]),
        ("complete program", program["attempts"]),
    ):
        for row in records[:2]:
            smiles = row["committed_smiles"]
            if smiles is not None:
                samples.append((f"{label}, attempt {row['attempt_index']}", smiles))
    mols = [Chem.MolFromSmiles(s) for _, s in samples]
    if any(m is None for m in mols):
        raise ValueError("preview contains an invalid emitted SMILES")
    grid = Draw.MolsToGridImage(
        mols, legends=[s for s, _ in samples], molsPerRow=3, subImgSize=(400, 300)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(args.output))
    _atomic_json(
        args.output.with_suffix(".json"),
        {
            "schema": "fragment_program_preview_v1",
            "selection": "core plus first two attempts per arm; no quality selection",
            "inputs": {
                str(p.resolve()): physical_sha256(p)
                for p in (args.program_shard, args.baseline_shard, Path(__file__))
            },
            "rdkit": rdBase.rdkitVersion,
            "molecules": samples,
            "image_sha256": physical_sha256(args.output),
            "scope": "qualitative support preview, not benchmark quality evidence",
        },
    )


if __name__ == "__main__":
    main()
