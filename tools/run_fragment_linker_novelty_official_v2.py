"""Frozen three-seed linker evaluation of the qualified novelty-4 selector.

The proposal law, model, structural checks, eight offers, and official metric
remain those of the qualified development pilot. Only fresh seeds are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_linker_broaden_pilot_v1 import cell_support_summary, preflight
from run_fragment_linker_novelty_pilot_v1 import load_contract as load_development_contract
from run_fragment_linker_official_v1 import attempt_samples
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog, sample_linker_panel
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_linker_novelty_official_v2.json"
METRICS = ("quality", "uniqueness", "diversity", "validity")


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract() -> tuple[dict, str]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid linker benchmark contract: {CONTRACT}")
    contract = envelope["payload"]
    if identity(contract) != envelope["payload_sha256"]:
        raise ValueError("linker benchmark contract hash mismatch")
    if (
        contract["schema"] != "fragment_linker_novelty_official_v2"
        or contract["task"] != "linker_design"
        or contract["selector"] != "novelty4"
        or contract["seeds"] != [6, 7, 8]
        or contract["attempts_per_prompt_seed"] != 100
        or contract["candidate_draws_per_attempt"] != 8
        or contract["output_dir"] != "diagnostics/fragment_linker_novelty_official_v2"
        or contract["minimum_free_bytes"] != 5368709120
    ):
        raise ValueError("linker benchmark scientific envelope changed")
    return contract, envelope["payload_sha256"]


def preflight_official(contract: dict) -> tuple[dict, tuple, dict, str]:
    development, development_hash = load_development_contract()
    if development_hash != contract["development_contract_payload_sha256"]:
        raise ValueError("linker novelty development contract changed")
    summary_path = ROOT / development["output_dir"] / "summary.json"
    if physical_sha256(summary_path) != contract["development_summary_sha256"]:
        raise ValueError("linker novelty development result changed")
    if json.loads(summary_path.read_text())["promotion_gate_passed"] is not True:
        raise ValueError("linker novelty selector did not pass its frozen development gate")
    versions, prompts = preflight(development)
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("linker benchmark prompt identities changed")
    for raw, digest in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != digest:
            raise ValueError(f"linker benchmark material changed: {raw}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return development, prompts, versions, revision


def seed_summary(rows: list[dict], seed: int, drugs: list[str]) -> dict:
    if (
        len(rows) != len(drugs)
        or [row["drug"] for row in rows] != drugs
        or any(row["seed"] != seed or row["attempts"] != 100 for row in rows)
        or any(
            row["outputs"] != row["valid_connected_outputs"]
            or row["outputs"] != row["exact_core_path_fidelity_outputs"]
            for row in rows
        )
    ):
        raise ValueError(f"linker seed {seed} has an incomplete or unfaithful prompt population")
    return {
        "seed": seed,
        "outputs": sum(row["outputs"] for row in rows),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in rows),
        "exact_core_path_fidelity_outputs": sum(
            row["exact_core_path_fidelity_outputs"] for row in rows
        ),
        **{metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in METRICS},
    }


def summarize(output: Path, contract: dict, manifest_sha: str) -> dict:
    per_seed = []
    row_hashes = {}
    for seed in contract["seeds"]:
        rows = []
        for drug in contract["drugs"]:
            path = output / f"seed{seed}" / "rows" / f"{drug}.json"
            row = json.loads(path.read_text())
            lock_path = output / f"seed{seed}" / "locks" / f"{drug}.json"
            if row["manifest_sha256"] != manifest_sha or row["lock_sha256"] != physical_sha256(
                lock_path
            ):
                raise ValueError(f"linker row changed manifest or sample lock: {path}")
            row_hashes[str(path.relative_to(output))] = physical_sha256(path)
            rows.append(row)
        per_seed.append(seed_summary(rows, seed, contract["drugs"]))
    return {
        "schema": "fragment_linker_novelty_official_result_v2",
        "role": "independent fresh-seed published-scale linker evaluation",
        "attempts": len(contract["seeds"]) * len(contract["drugs"]) * 100,
        "offered_candidate_draws": len(contract["seeds"]) * len(contract["drugs"]) * 800,
        "outputs": sum(row["outputs"] for row in per_seed),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in per_seed),
        "exact_core_path_fidelity_outputs": sum(
            row["exact_core_path_fidelity_outputs"] for row in per_seed
        ),
        "per_seed": per_seed,
        "official_mean": {
            metric: float(np.mean([row[metric] for row in per_seed])) for metric in METRICS
        },
        "official_sd": {
            metric: float(np.std([row[metric] for row in per_seed], ddof=1)) for metric in METRICS
        },
        "morphing": "identical-input alias; not an independent replicate",
        "manifest_sha256": manifest_sha,
        "row_sha256": row_hashes,
        "oracle_calls": 0,
    }


def run_seed(
    output: Path,
    seed: int,
    prompts: tuple,
    catalog,
    prior,
    model,
    manifest_sha: str,
    minimum_free_bytes: int,
) -> None:
    seed_output = output / f"seed{seed}"
    if (seed_output / "complete.json").exists():
        raise FileExistsError(f"linker seed already complete: {seed}")
    rows = []
    for prompt in prompts:
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, seed))
        emitted: set[str] = set()
        attempts = []
        attempt_hashes = {}
        for index in range(100):
            path = seed_output / "attempts" / f"{prompt.drug_name}_{index:03d}.json"
            start = seed_output / "starts" / f"{prompt.drug_name}_{index:03d}.json"
            before = {
                "seed": seed,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "rng_state_before": rng.bit_generator.state,
                "manifest_sha256": manifest_sha,
            }
            if path.exists():
                if not start.exists() or json.loads(start.read_text()) != before:
                    raise ValueError(f"completed linker attempt lost start identity: {path}")
                receipt = json.loads(path.read_text())
                if receipt.get("seed") != seed:
                    raise ValueError(f"saved linker attempt changed seed: {path}")
                restore_completed_attempt(
                    receipt, drug=prompt.drug_name, attempt_index=index, rng=rng
                )
            else:
                if start.exists():
                    raise RuntimeError(f"started linker attempt cannot be redrawn: {start}")
                if shutil.disk_usage(output).free < minimum_free_bytes:
                    raise RuntimeError("linker benchmark stopped at frozen disk-free floor")
                _atomic_json(start, before)
                began = time.monotonic()
                panel = sample_linker_panel(
                    prompt,
                    catalog,
                    prior,
                    model,
                    rng,
                    cell_allocation="frozen",
                    prior_emitted=frozenset(emitted),
                )
                selected = panel.selected
                fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
                valid = bool(
                    selected
                    and is_valid_state(selected.endpoint)
                    and is_connected_or_null(selected.endpoint)
                )
                faithful = bool(
                    selected
                    and fidelity["satisfied"]
                    and selected.provenance["exact_mapped_core_identity_checked"]
                    and selected.provenance["source_core_locked_all_states"]
                )
                if selected and not (valid and faithful):
                    raise RuntimeError(
                        f"invalid/nonfaithful linker output: {seed}/{prompt.drug_name}/{index}"
                    )
                receipt = {
                    "seed": seed,
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": panel.receipt,
                    "wall_seconds": time.monotonic() - began,
                    "selected_valid_connected": valid,
                    "selected_exact_core_path_fidelity": faithful,
                    "selected_fidelity": fidelity,
                }
                _atomic_json(path, receipt)
            attempts.append(receipt)
            selected_smiles = receipt["panel"]["selected_smiles"]
            if selected_smiles is not None:
                emitted.add(selected_smiles)
            attempt_hashes[str(path.relative_to(seed_output))] = physical_sha256(path)
            if (index + 1) % 10 == 0:
                _atomic_json(
                    seed_output / "progress.json",
                    {
                        "schema": "fragment_linker_novelty_official_progress_v2",
                        "seed": seed,
                        "drug": prompt.drug_name,
                        "completed_attempts_in_prompt": index + 1,
                        "completed_prompt_rows": len(rows),
                    },
                )
        samples = attempt_samples(attempts, drug=prompt.drug_name)
        lock_path = seed_output / "locks" / f"{prompt.drug_name}.json"
        immutable_json(
            lock_path,
            {
                "manifest_sha256": manifest_sha,
                "attempt_hashes": attempt_hashes,
                "samples": samples,
            },
        )
        row = {
            "seed": seed,
            "drug": prompt.drug_name,
            "attempts": 100,
            "outputs": sum(receipt["panel"]["output_count"] for receipt in attempts),
            "valid_connected_outputs": sum(
                receipt["selected_valid_connected"] for receipt in attempts
            ),
            "exact_core_path_fidelity_outputs": sum(
                receipt["selected_exact_core_path_fidelity"] for receipt in attempts
            ),
            "metrics": official_prompt_metrics(samples, expected_samples=100),
            "lock_sha256": physical_sha256(lock_path),
            "manifest_sha256": manifest_sha,
        }
        immutable_json(seed_output / "rows" / f"{prompt.drug_name}.json", row)
        rows.append(row)
        print(
            json.dumps({"seed": seed, "drug": prompt.drug_name, "metrics": row["metrics"]}),
            flush=True,
        )
    completion = seed_summary(rows, seed, [prompt.drug_name for prompt in prompts])
    completion["manifest_sha256"] = manifest_sha
    immutable_json(seed_output / "complete.json", completion)
    print(json.dumps({"seed_completed": seed, "metrics": completion}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run-seed", "summarize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, payload_sha = load_contract()
    development, prompts, versions, revision = preflight_official(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("linker output differs from frozen contract")
    manifest = {
        "schema": "fragment_linker_novelty_official_manifest_v2",
        "contract": contract,
        "contract_payload_sha256": payload_sha,
        "development_contract": development,
        "code_revision": revision,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32 model; float64 proposal probabilities",
        "score_blind_cell_support": cell_support_summary(prompts, ("frozen", "novelty4")),
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if args.seed is not None or output.exists():
            raise ValueError("prepare requires no seed and a fresh output directory")
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "payload_sha256": payload_sha}))
        return
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("linker benchmark manifest missing or changed")
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
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(development["checkpoint_path"]))
    catalog = load_linker_catalog(
        ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
    )
    prior = JointCompletionPrior.from_dict(
        json.loads((ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json").read_text())
    )
    run_seed(
        output,
        args.seed,
        prompts,
        catalog,
        prior,
        model,
        manifest_sha,
        contract["minimum_free_bytes"],
    )


if __name__ == "__main__":
    main()
