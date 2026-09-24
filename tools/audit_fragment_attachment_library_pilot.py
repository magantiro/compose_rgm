"""Decompose the frozen attachment-library pilot without selecting new samples."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from compose_v4.benchmark.fragment_constrained import sascorer
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def _atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _row(shard: dict) -> dict:
    attempts = shard["attempt_records"]
    records = []
    for attempt in attempts:
        smiles = attempt["committed_smiles"]
        if smiles is None:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid committed molecule: {smiles}")
        records.append(
            {
                "smiles": smiles,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "qed": QED.qed(molecule),
                "sa": sascorer.calculateScore(molecule),
            }
        )
    unique = {row["smiles"]: row for row in records}
    return {
        "attempts": len(attempts),
        "committed": len(records),
        "no_output": len(attempts) - len(records),
        "unique": len(unique),
        "mean_heavy_atoms": float(np.mean([row["heavy_atoms"] for row in records]))
        if records
        else None,
        "mean_qed": float(np.mean([row["qed"] for row in records])) if records else None,
        "mean_sa": float(np.mean([row["sa"] for row in records])) if records else None,
        "qed_pass_attempts": sum(row["qed"] >= 0.6 for row in records),
        "sa_pass_attempts": sum(row["sa"] <= 4.0 for row in records),
        "joint_pass_attempts": sum(row["qed"] >= 0.6 and row["sa"] <= 4.0 for row in records),
        "unique_joint_pass": sum(row["qed"] >= 0.6 and row["sa"] <= 4.0 for row in unique.values()),
        "official_quality": shard["metrics"]["quality"],
        "official_diversity": shard["metrics"]["diversity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.pilot_dir.resolve()
    manifest_path = directory / "manifest.json"
    summary_path = directory / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "fragment_attachment_library_pilot_v1":
        raise ValueError(f"wrong pilot manifest: {manifest_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"complete pilot summary missing: {summary_path}")
    paths = tuple(sorted((directory / "shards").glob("*.json")))
    if len(paths) != 40:
        raise ValueError(f"expected 40 matched pilot shards, got {len(paths)}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    per_prompt = {}
    for path in paths:
        shard = json.loads(path.read_text())
        row = _row(shard)
        key = (shard["task"], shard["arm"])
        groups[key].append(row)
        per_prompt[f"{shard['task']}|{shard['drug']}|{shard['arm']}"] = row
    task_summary = {}
    for (task, arm), rows in sorted(groups.items()):
        if len(rows) != 10:
            raise ValueError(f"not ten drugs for {task}/{arm}")
        task_summary[f"{task}|{arm}"] = {
            "attempts": sum(row["attempts"] for row in rows),
            "committed": sum(row["committed"] for row in rows),
            "no_output": sum(row["no_output"] for row in rows),
            "qed_pass_attempts": sum(row["qed_pass_attempts"] for row in rows),
            "sa_pass_attempts": sum(row["sa_pass_attempts"] for row in rows),
            "joint_pass_attempts": sum(row["joint_pass_attempts"] for row in rows),
            "unique_joint_pass": sum(row["unique_joint_pass"] for row in rows),
            "mean_heavy_atoms": float(np.mean([row["mean_heavy_atoms"] for row in rows])),
            "mean_qed": float(np.mean([row["mean_qed"] for row in rows])),
            "mean_sa": float(np.mean([row["mean_sa"] for row in rows])),
            "mean_official_quality": float(np.mean([row["official_quality"] for row in rows])),
            "mean_official_diversity": float(np.mean([row["official_diversity"] for row in rows])),
        }
    output = {
        "schema": "fragment_attachment_library_pilot_audit_v1",
        "evidence_role": "post hoc decomposition of one frozen development pilot",
        "pilot_manifest_sha256": physical_sha256(manifest_path),
        "pilot_summary_sha256": physical_sha256(summary_path),
        "shard_sha256": {path.name: physical_sha256(path) for path in paths},
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "analysis_code_sha256": physical_sha256(Path(__file__).resolve()),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "per_prompt": per_prompt,
        "summary": task_summary,
    }
    path = directory / "analysis.json"
    if path.exists():
        raise FileExistsError(f"preserving existing analysis: {path}")
    _atomic_json(path, output)
    print(json.dumps(task_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
