"""Replay a named fragment run's finite panels with uniform endpoint selection.

This is a conditional panel-selection ablation. It preserves all eight recorded
offers, their native-support admission, and the official evaluator. It does not
replace the primitive transition law or regenerate the offers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import RDLogger, rdBase

from compose_v4.benchmark.fragment_common_panel_ablation import (
    select_common_panel,
    supported_panel,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if self_hash(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"contract self-hash mismatch: {path}")
    payload = envelope["payload"]
    if payload["schema"] not in {"fragment_motif_official_v1", "fragment_linker_official_v1"}:
        raise ValueError(f"wrong fragment contract: {path}")
    return payload, envelope["payload_sha256"]


def _quality_flags(samples: list[str]) -> dict[str, bool]:
    # The imported functions are the same verified upstream functions used by
    # official_prompt_metrics; apply the criterion to every attempt, including
    # repeated molecules, for the separate attempt-denominator quality yield.
    from in_virtuo_gen.utils.mol import compute_properties, is_drug_like_and_synthesizable

    unique = list(dict.fromkeys(sample for sample in samples if sample))
    properties = compute_properties(unique)
    if len(properties) != len(unique):
        raise RuntimeError("upstream quality computation returned the wrong row count")
    return {
        sample: bool(prop is not None and is_drug_like_and_synthesizable(prop))
        for sample, prop in zip(unique, properties, strict=True)
    }


def run(contract_path: Path, run_root: Path, output: Path) -> dict:
    RDLogger.DisableLog("rdApp.warning")
    payload, contract_hash = load_contract(contract_path)
    task = (
        FragmentTask.MOTIF_EXTENSION
        if payload["schema"] == "fragment_motif_official_v1"
        else FragmentTask.LINKER_DESIGN
    )
    source_dir = run_root / payload["output_dir"]
    manifest = source_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"named run manifest missing: {manifest}")
    recorded_manifest = json.loads(manifest.read_text())
    if recorded_manifest["contract_payload_sha256"] != contract_hash:
        raise ValueError("run manifest is not bound to the named contract")
    summary = source_dir / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"full named run is not complete: {summary}")
    manifest_digest = sha256(manifest)
    prompts = {
        prompt.drug_name: prompt
        for prompt in load_genmol_prompts(
            run_root / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        )
        if prompt.task == task
    }
    if set(prompts) != set(payload["drugs"]) or len(prompts) != 10:
        raise ValueError("prompt panel differs from the named run contract")

    results: list[dict] = []
    all_attempts: list[dict] = []
    for seed in payload["seeds"]:
        for drug in payload["drugs"]:
            lock_path = source_dir / "locks" / f"seed{seed}" / f"{drug}.json"
            lock = json.loads(lock_path.read_text())
            if lock["manifest_sha256"] != manifest_digest:
                raise ValueError(f"prompt lock has the wrong manifest: {lock_path}")
            if len(lock["attempt_hashes"]) != 100 or len(lock["samples"]) != 100:
                raise ValueError(f"incomplete prompt lock: {lock_path}")
            learned: list[str] = []
            uniform: list[str] = []
            for index in range(100):
                relative = f"attempts/seed{seed}/{drug}_{index:03d}.json"
                attempt_path = source_dir / relative
                if sha256(attempt_path) != lock["attempt_hashes"][relative]:
                    raise ValueError(f"attempt hash mismatch: {attempt_path}")
                attempt = json.loads(attempt_path.read_text())
                if (attempt["seed"], attempt["drug"], attempt["attempt_index"]) != (
                    seed, drug, index
                ):
                    raise ValueError(f"attempt identity mismatch: {attempt_path}")
                panel = attempt["panel"]
                admitted = supported_panel(panel["offered"])
                selected = panel["selected_smiles"] or ""
                if selected and selected not in {smi for smi, _ in admitted}:
                    raise ValueError(f"learned selection outside recorded support: {attempt_path}")
                if selected != lock["samples"][index]:
                    raise ValueError(f"learned selection differs from sealed lock: {attempt_path}")
                picked = select_common_panel(
                    panel["offered"],
                    law="uniform",
                    seed_material=f"{contract_hash}|{seed}|{drug}|{index}|uniform-v1",
                ) or ""
                learned.append(selected)
                uniform.append(picked)
                all_attempts.append(
                    {
                        "seed": seed, "drug": drug, "attempt_index": index,
                        "source_sha256": lock["attempt_hashes"][relative],
                        "admitted_count": len(admitted),
                        "learned": selected, "uniform": picked,
                    }
                )
            row = {"seed": seed, "drug": drug, "attempts": 100,
                   "lock_sha256": sha256(lock_path), "arms": {}}
            for law, samples in (("learned", learned), ("uniform", uniform)):
                metrics = official_prompt_metrics(samples, expected_samples=100)
                flags = _quality_flags(samples)
                fidelity = [
                    bool(sample and check_fragment_constraint(prompts[drug], sample).satisfied)
                    for sample in samples
                ]
                row["arms"][law] = {
                    "metrics": metrics,
                    "outputs": sum(bool(sample) for sample in samples),
                    "prompt_faithful_outputs": sum(fidelity),
                    "prompt_faithful_quality_yield": sum(
                        faithful and flags.get(sample, False)
                        for faithful, sample in zip(fidelity, samples, strict=True)
                    ) / 100.0,
                }
            results.append(row)

    baseline = json.loads(summary.read_text())
    for name in ("validity", "uniqueness", "quality", "diversity"):
        replayed = float(np.mean([row["arms"]["learned"]["metrics"][name] for row in results]))
        if abs(replayed - baseline["official_mean"][name]) > 1e-8:
            raise ValueError(f"learned metric replay mismatch for {name}: {replayed}")
    per_prompt = []
    for drug in payload["drugs"]:
        cohort = [row for row in results if row["drug"] == drug]
        per_prompt.append({
            "drug": drug,
            "arms": {
                law: {
                    name: float(np.mean([row["arms"][law]["metrics"][name] for row in cohort]))
                    for name in ("validity", "uniqueness", "quality", "diversity")
                }
                for law in ("learned", "uniform")
            },
        })
    report = {
        "schema": "fragment_common_panel_selection_ablation_v1",
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "source_run_revision": payload["source_revision"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "conditional finite-panel selection on common native-supported offers; not a primitive transition-reference ablation",
        "task": task.value,
        "source_contract_payload_sha256": contract_hash,
        "source_manifest_sha256": manifest_digest,
        "source_summary_sha256": sha256(summary),
        "source_checkpoint_sha256": payload["checkpoint_sha256"],
        "evaluator_sha256": payload["official_evaluator_sha256"],
        "seed_derivation": "SHA256(contract payload|seed|drug|attempt|uniform-v1), first 64 bits",
        "execution_environment": {
            "device": "cpu",
            "precision": "float64 metric aggregation; original checkpoint scores reused",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "executable": sys.executable,
        },
        "attempts": len(all_attempts),
        "rows": results,
        "per_prompt": per_prompt,
        "selections": all_attempts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.run_root, args.output)
    print(f"{report['task']}: {report['attempts']} paired panel selections -> {args.output}")


if __name__ == "__main__":
    main()
