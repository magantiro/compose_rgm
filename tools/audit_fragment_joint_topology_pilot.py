"""Decompose sealed joint-completion rows by task topology and quality cause.

This is a read-only development audit. It never selects proposals, and partial
rows are explicitly labelled partial rather than projected to a ten-prompt run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED

from compose_v4.benchmark.fragment_constrained import sascorer
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

TASKS = ("motif_extension", "scaffold_decoration")


def _summary(samples: list[dict]) -> dict:
    outputs = [row for row in samples if row["smiles"] is not None]
    return {
        "attempts": len(samples),
        "outputs": len(outputs),
        "qed_pass_outputs": sum(row["qed_pass"] for row in outputs),
        "sa_pass_outputs": sum(row["sa_pass"] for row in outputs),
        "joint_pass_outputs": sum(row["joint_pass"] for row in outputs),
        "mean_heavy_atoms": sum(row["heavy_atoms"] for row in outputs) / len(outputs)
        if outputs
        else None,
        "mean_planned_heavy_atoms": sum(row["planned_heavy_atoms"] for row in outputs)
        / len(outputs)
        if outputs
        else None,
        "mean_qed": sum(row["qed"] for row in outputs) / len(outputs) if outputs else None,
        "mean_sa": sum(row["sa"] for row in outputs) / len(outputs) if outputs else None,
    }


def audit(pilot_dir: Path) -> dict:
    manifest_path = pilot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "fragment_joint_prior_gate_v1" or manifest["mode"] != "pilot":
        raise ValueError("not the frozen joint-completion metric pilot")
    for raw, digest in manifest["inputs"].items():
        if physical_sha256(Path(raw)) != digest:
            raise ValueError(f"frozen joint-completion pilot input changed: {raw}")
    row_paths = sorted((pilot_dir / "rows").glob("*.json"))
    if not row_paths:
        raise ValueError("no sealed prompt rows")
    samples, prompts, hashes = [], [], {}
    for path in row_paths:
        row = json.loads(path.read_text())
        if row["task"] not in TASKS or path.stem != f"{row['task']}_{row['drug']}":
            raise ValueError(f"unexpected prompt row identity: {path}")
        if len(row["attempts"]) != 20:
            raise ValueError(f"prompt row not sealed at twenty attempts: {path}")
        hashes[str(path.resolve())] = physical_sha256(path)
        seen = set()
        unique_quality = 0
        for index, attempt in enumerate(row["attempts"]):
            if attempt["attempt_index"] != index or not attempt["complete"]:
                raise ValueError(f"incomplete/out-of-order attempt in {path}")
            text = attempt["committed_smiles"]
            result = {
                "task": row["task"],
                "drug": row["drug"],
                "attempt_index": index,
                "smiles": text,
            }
            if text is not None:
                mol = Chem.MolFromSmiles(text)
                if mol is None or Chem.MolToSmiles(mol) != text:
                    raise ValueError(f"noncanonical or invalid committed endpoint in {path}")
                qed, sa = QED.qed(mol), sascorer.calculateScore(mol)
                plan = attempt["joint_plan"]
                result.update(
                    qed=qed,
                    sa=sa,
                    qed_pass=qed >= 0.6,
                    sa_pass=sa <= 4.0,
                    joint_pass=qed >= 0.6 and sa <= 4.0,
                    heavy_atoms=mol.GetNumHeavyAtoms(),
                    rings=mol.GetRingInfo().NumRings(),
                    planned_heavy_atoms=plan["planned_heavy_atoms"],
                    planned_rings=plan["planned_rings"],
                    lane=attempt["selected_capabilities"]["t4_lane"] or "joint_training_regions",
                    prompt_fidelity=attempt["prompt_fidelity"],
                )
                if text not in seen and result["joint_pass"]:
                    unique_quality += 1
                seen.add(text)
            samples.append(result)
        if abs(row["official"]["quality"] - 100 * unique_quality / 20) > 1e-9:
            raise ValueError(f"local QED/SA decomposition differs from official quality: {path}")
        prompts.append(
            {
                "task": row["task"],
                "drug": row["drug"],
                "official": row["official"],
                "decomposition": _summary(samples[-20:]),
            }
        )
    by_task = {task: _summary([r for r in samples if r["task"] == task]) for task in TASKS}
    by_lane = defaultdict(list)
    for row in samples:
        if row["smiles"] is not None:
            by_lane[(row["task"], row["lane"])].append(row)
    result = {
        "schema": "fragment_joint_topology_pilot_audit_v1",
        "role": "development diagnostic on sealed rows; partial until ten prompts per task",
        "manifest_sha256": physical_sha256(manifest_path),
        "input_sha256": dict(
            sorted(
                {
                    str(manifest_path.resolve()): physical_sha256(manifest_path),
                    str(Path(__file__).resolve()): physical_sha256(Path(__file__)),
                    str(Path(QED.__file__).resolve()): physical_sha256(Path(QED.__file__)),
                    str(Path(sascorer.__file__).resolve()): physical_sha256(
                        Path(sascorer.__file__)
                    ),
                    **hashes,
                }.items()
            )
        ),
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prompts": prompts,
        "by_task": by_task,
        "by_task_and_lane": {
            f"{task}:{lane}": _summary(items) for (task, lane), items in sorted(by_lane.items())
        },
        "samples": samples,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = audit(args.pilot_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".joint-audit-") as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"by_task": result["by_task"], "rows": len(result["prompts"])}))


if __name__ == "__main__":
    main()
