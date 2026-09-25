"""Frozen three-seed decoration evaluation of the qualified joint-mass constructor.

Each seed writes to its own directory. The prior development comparison is
used only to select the frozen constructor; its samples are never reused here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_source_coupled_decoration_pilot_v1 import (
    ROOT,
    _run_cell,
    preflight,
)
from run_fragment_source_coupled_decoration_pilot_v1 import (
    load_contract as load_development_contract,
)
from run_fragment_training_linker_metric_pilot import immutable_json

from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

CONTRACT = ROOT / "configs/fragment_decoration_official_v2.json"
METRICS = ("quality", "uniqueness", "diversity", "validity")


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract() -> tuple[dict, str]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid decoration benchmark contract: {CONTRACT}")
    contract = envelope["payload"]
    if identity(contract) != envelope["payload_sha256"]:
        raise ValueError("decoration benchmark contract hash mismatch")
    if (
        contract["schema"] != "fragment_decoration_official_v2"
        or contract["task"] != "scaffold_decoration"
        or contract["arm"] != "frozen"
        or contract["seeds"] != [8, 9, 10]
        or contract["attempts_per_prompt_seed"] != 100
        or contract["offers_per_attempt"] != 8
        or contract["output_dir"] != "diagnostics/fragment_decoration_official_v2"
        or contract["minimum_free_bytes"] != 5368709120
    ):
        raise ValueError("decoration benchmark scientific envelope changed")
    return contract, envelope["payload_sha256"]


def preflight_official(contract: dict) -> tuple[dict, tuple, dict, dict, str, dict]:
    development, development_hash = load_development_contract()
    if development_hash != contract["development_contract_payload_sha256"]:
        raise ValueError("decoration development contract changed")
    development_summary = ROOT / development["output_dir"] / "summary.json"
    if physical_sha256(development_summary) != contract["development_summary_sha256"]:
        raise ValueError("decoration development result changed")
    if json.loads(development_summary.read_text())["promotion_gate_pass"] is not False:
        raise ValueError("expected frozen development constructor to remain selected")
    prompts, catalog, prior, _ = preflight(development)
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("decoration benchmark prompt identities changed")
    for raw, digest in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != digest:
            raise ValueError(f"decoration benchmark material changed: {raw}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    return development, prompts, catalog, prior, revision, versions


def seed_summary(rows: list[dict], seed: int, drugs: list[str]) -> dict:
    if (
        len(rows) != len(drugs)
        or [row["drug"] for row in rows] != drugs
        or any(row["seed"] != seed or row["attempts"] != 100 for row in rows)
        or any(
            row["outputs"] != row["valid_connected_outputs"]
            or row["outputs"] != row["constraint_fidelity_outputs"]
            for row in rows
        )
    ):
        raise ValueError(f"decoration seed {seed} has an incomplete prompt population")
    return {
        "seed": seed,
        "outputs": sum(row["outputs"] for row in rows),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in rows),
        "constraint_fidelity_outputs": sum(row["constraint_fidelity_outputs"] for row in rows),
        **{metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in METRICS},
    }


def summarize(output: Path, contract: dict, manifest_sha: str) -> dict:
    per_seed = []
    row_hashes = {}
    for seed in contract["seeds"]:
        rows = []
        for drug in contract["drugs"]:
            path = output / f"seed{seed}" / "rows" / "frozen" / f"{drug}.json"
            row = json.loads(path.read_text())
            if row["manifest_sha256"] != manifest_sha:
                raise ValueError(f"decoration row changed manifest: {path}")
            lock_path = output / f"seed{seed}" / "locks" / "frozen" / f"{drug}.json"
            if row["lock_sha256"] != physical_sha256(lock_path):
                raise ValueError(f"decoration row changed sample lock: {path}")
            row_hashes[str(path.relative_to(output))] = physical_sha256(path)
            rows.append(row)
        per_seed.append(seed_summary(rows, seed, contract["drugs"]))
    return {
        "schema": "fragment_decoration_official_result_v2",
        "role": "independent fresh-seed published-scale decoration evaluation",
        "attempts": len(contract["seeds"]) * len(contract["drugs"]) * 100,
        "offered_candidate_draws": len(contract["seeds"]) * len(contract["drugs"]) * 800,
        "outputs": sum(row["outputs"] for row in per_seed),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in per_seed),
        "constraint_fidelity_outputs": sum(row["constraint_fidelity_outputs"] for row in per_seed),
        "per_seed": per_seed,
        "official_mean": {
            metric: float(np.mean([row[metric] for row in per_seed])) for metric in METRICS
        },
        "official_sd": {
            metric: float(np.std([row[metric] for row in per_seed], ddof=1)) for metric in METRICS
        },
        "manifest_sha256": manifest_sha,
        "row_sha256": row_hashes,
        "oracle_calls": 0,
    }


def prepared_seed_directory(output: Path, seed: int) -> Path:
    """Create the seed namespace before the reused cell runner checks free space."""
    path = output / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run-seed", "summarize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, payload_sha = load_contract()
    development, prompts, catalog, prior, revision, versions = preflight_official(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("decoration output differs from frozen contract")
    manifest = {
        "schema": "fragment_decoration_official_manifest_v2",
        "contract": contract,
        "contract_payload_sha256": payload_sha,
        "development_contract": development,
        "code_revision": revision,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32 model; float64 proposal probabilities",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if args.seed is not None or output.exists():
            raise ValueError("prepare requires no seed and a fresh output directory")
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "payload_sha256": payload_sha}))
        return
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("decoration benchmark manifest missing or changed")
    manifest_sha = physical_sha256(manifest_path)
    if args.phase == "summarize":
        if args.seed is not None:
            raise ValueError("summarize takes no seed")
        summary = summarize(output, contract, manifest_sha)
        immutable_json(output / "summary.json", summary)
        print(json.dumps({"completed": True, "official_mean": summary["official_mean"]}))
        return
    if args.seed not in contract["seeds"]:
        raise ValueError("run-seed needs a declared seed")
    seed_output = prepared_seed_directory(output, args.seed)
    if (seed_output / "complete.json").exists():
        raise FileExistsError(f"decoration seed already complete: {args.seed}")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(development["checkpoint_path"]))
    sampler = JointMassPendantSampler(catalog, prior)
    seed_contract = dict(development, seed=args.seed, attempts_per_prompt_arm=100)
    for prompt in prompts:
        lock_path = _run_cell(
            output=seed_output,
            manifest_sha=manifest_sha,
            prompt=prompt,
            arm="frozen",
            sampler=sampler,
            model=model,
            contract=seed_contract,
        )
        lock = json.loads(lock_path.read_text())
        samples = lock["samples"]
        if len(samples) != 100:
            raise ValueError("decoration prompt has incomplete attempt denominator")
        attempts = [
            json.loads(
                (
                    seed_output / "attempts" / "frozen" / f"{prompt.drug_name}_{i:03d}.json"
                ).read_text()
            )
            for i in range(100)
        ]
        row = {
            "seed": args.seed,
            "drug": prompt.drug_name,
            "attempts": 100,
            "outputs": sum(row["panel"]["output_count"] for row in attempts),
            "valid_connected_outputs": sum(row["selected_valid_connected"] for row in attempts),
            "constraint_fidelity_outputs": sum(
                row["selected_constraint_fidelity"] for row in attempts
            ),
            "metrics": official_prompt_metrics(samples, expected_samples=100),
            "lock_sha256": physical_sha256(lock_path),
            "manifest_sha256": manifest_sha,
        }
        immutable_json(seed_output / "rows" / "frozen" / f"{prompt.drug_name}.json", row)
        print(
            json.dumps({"seed": args.seed, "drug": prompt.drug_name, "metrics": row["metrics"]}),
            flush=True,
        )
    rows = [
        json.loads((seed_output / "rows" / "frozen" / f"{drug}.json").read_text())
        for drug in contract["drugs"]
    ]
    completion = seed_summary(rows, args.seed, contract["drugs"])
    completion["manifest_sha256"] = manifest_sha
    immutable_json(seed_output / "complete.json", completion)
    print(json.dumps({"seed_completed": args.seed, "metrics": completion}), flush=True)


if __name__ == "__main__":
    main()
