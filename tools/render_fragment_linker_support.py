"""Render every selected support endpoint and record structural distributions.

Fixed prompt/attempt order, no property score, quality filtering or selection.
Blue/green highlight the supplied cores; orange highlights connector atoms.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary_path = args.support_dir / "summary.json"
    manifest_path = args.support_dir / "manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if summary["manifest_sha256"] != physical_sha256(manifest_path):
        raise ValueError("support manifest hash changed")
    paths = [
        args.support_dir / "attempts" / f"{drug}_{index:03d}.json"
        for drug in manifest["configuration"]["prompts"]
        for index in range(2)
    ]
    if len(paths) != 20 or set(summary["attempt_hashes"]) != {
        str(path.relative_to(args.support_dir)) for path in paths
    }:
        raise ValueError("render requires the complete fixed support census")
    records, molecules, colors, legends = [], [], [], []
    for path in paths:
        if summary["attempt_hashes"][str(path.relative_to(args.support_dir))] != physical_sha256(
            path
        ):
            raise ValueError(f"support attempt hash changed: {path}")
        row = json.loads(path.read_text())
        panel = row["panel"]
        if not panel["output_count"]:
            records.append({"drug": row["drug"], "attempt": row["attempt_index"], "output": False})
            continue
        selected = panel["offered"][panel["selection"]["selected_draw"]]
        if selected["endpoint"] != panel["selected_smiles"]:
            raise ValueError("selected program does not identify the rendered endpoint")
        provenance = selected["provenance"]
        molecule = Chem.MolFromSmiles(selected["endpoint"])
        maps = provenance["fidelity"]["core_maps"]
        atom_colors = {index: (1.0, 0.70, 0.30) for index in range(molecule.GetNumAtoms())}
        for index in maps[0]:
            atom_colors[index] = (0.55, 0.75, 1.0)
        for index in maps[1]:
            atom_colors[index] = (0.50, 0.85, 0.60)
        proposal = provenance["training_connector"]
        record = {
            "drug": row["drug"],
            "attempt": row["attempt_index"],
            "output": True,
            "endpoint": selected["endpoint"],
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "ring_count": molecule.GetRingInfo().NumRings(),
            "connector_heavy_atoms": molecule.GetNumHeavyAtoms() - proposal["core_heavy_atoms"],
            "connector_rings": molecule.GetRingInfo().NumRings() - proposal["core_ring_count"],
            "connector_path_atoms": provenance["fidelity"]["linker_internal_atoms"],
            "primitives": len(selected["trace"]["actions"]),
            "core_maps": maps,
        }
        records.append(record)
        molecules.append(molecule)
        colors.append(atom_colors)
        legends.append(
            f"{row['drug']}  attempt {row['attempt_index']}\n"
            f"atoms {record['heavy_atoms']} | rings {record['ring_count']} | "
            f"edits {record['primitives']} | connector {record['connector_heavy_atoms']} atoms"
        )
    args.output_dir.mkdir(parents=True)
    image_paths = []
    for start in range(0, len(molecules), 10):
        stop = start + 10
        image = Draw.MolsToGridImage(
            molecules[start:stop],
            molsPerRow=2,
            subImgSize=(700, 400),
            legends=legends[start:stop],
            highlightAtomLists=[list(c) for c in colors[start:stop]],
            highlightAtomColors=colors[start:stop],
            useSVG=False,
            returnPNG=True,
        )
        path = args.output_dir / f"selected_{start // 10 + 1}.png"
        path.write_bytes(image)
        image_paths.append(path)
    fields = (
        "heavy_atoms",
        "ring_count",
        "connector_heavy_atoms",
        "connector_rings",
        "connector_path_atoms",
        "primitives",
    )
    distributions = {}
    for field in fields:
        values = [record[field] for record in records if record["output"]]
        distributions[field] = {
            "histogram": dict(sorted(Counter(values).items())),
            "minimum": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "maximum": max(values) if values else None,
        }
    result = {
        "schema": "fragment_linker_support_visual_structural_audit_v1",
        "inputs": {
            str(path.resolve()): physical_sha256(path)
            for path in [summary_path, manifest_path, *paths, Path(__file__).resolve()]
        },
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "randomness": "none; fixed prompt and attempt order, all selected outputs",
        "draw_style": "core 1 blue; core 2 green; new connector orange",
        "quality_evaluations": 0,
        "oracle_calls": 0,
        "attempts": len(records),
        "rendered_outputs": len(molecules),
        "distributions": distributions,
        "records": records,
        "images": {path.name: physical_sha256(path) for path in image_paths},
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps({k: result[k] for k in ("rendered_outputs", "distributions")}))


if __name__ == "__main__":
    main()
