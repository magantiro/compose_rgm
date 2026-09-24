"""Logged independent-seed motif or decoration development pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_motif_focused_programs import sample_motif_panel
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler
from compose_v4.benchmark.pendant_completion_policy import PendantCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "motif_extension": ROOT / "configs/fragment_motif_focused_dev_v1.json",
    "scaffold_decoration": ROOT / "configs/fragment_pendant_decoration_dev_v1.json",
}
CATALOGS = {
    "motif_extension": ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json",
    "scaffold_decoration": ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json",
}
PRIOR = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(task: str) -> tuple[dict, str]:
    envelope = json.loads(CONTRACTS[task].read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("motif development contract is not self-hashed")
    contract = envelope["payload"]
    if _identity(contract) != envelope["payload_sha256"]:
        raise ValueError("motif development contract self-hash mismatch")
    if (
        contract["schema"] != "fragment_focused_dev_v1"
        or contract["task"] != task
        or contract["attempts_per_prompt"] != 20
        or contract["candidate_draws_per_attempt"] != 8
        or contract["seed"] != 1
        or contract["quality_selection"] is not False
        or contract["workers"] != 1
    ):
        raise ValueError("motif development contract changes the predeclared envelope")
    return contract, envelope["payload_sha256"]


def attempt_samples(attempts: list[dict], *, drug: str) -> list[str]:
    if len(attempts) != 20 or [(row["drug"], row["attempt_index"]) for row in attempts] != [
        (drug, index) for index in range(20)
    ]:
        raise ValueError("motif quality requires twenty ordered attempts")
    if any(
        row["panel"]["offered_count"] != 8
        or [candidate["draw"] for candidate in row["panel"]["offered"]] != list(range(8))
        for row in attempts
    ):
        raise ValueError("motif quality requires eight recorded offers per attempt")
    return [row["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for row in attempts]


def preflight(contract: dict) -> tuple[dict, tuple, dict | None]:
    task = contract["task"]
    materials = [
        (CATALOGS[task], contract["catalog_sha256"]),
        (PROMPTS, contract["prompt_sha256"]),
        (Path(contract["checkpoint_path"]), contract["checkpoint_sha256"]),
    ]
    if task == "motif_extension":
        materials.append((PRIOR, contract["prior_sha256"]))
    for path, expected in materials:
        if physical_sha256(path) != expected:
            raise ValueError(f"motif pilot material changed: {path}")
    baseline = ROOT / "diagnostics/fragment_joint_completion_matched_pilot_v1/summary.json"
    if physical_sha256(baseline) != contract["baseline_summary_sha256"]:
        raise ValueError("frozen joint-completion baseline changed")
    if task == "scaffold_decoration":
        support = ROOT / "diagnostics/fragment_pendant_support_v1/summary.json"
        if physical_sha256(support) != contract["support_summary_sha256"]:
            raise ValueError("pendant support artifact changed")
        if json.loads(support.read_text())["support_pass"] is not True:
            raise ValueError("pendant support gate did not pass")
    from fetch_official_fragment_evaluator import verify_only

    verified = verify_only()
    if contract["official_evaluator_sha256"] not in verified.values():
        raise ValueError("pinned official motif evaluator unavailable")
    prompts = tuple(prompt for prompt in load_genmol_prompts(PROMPTS) if prompt.task.value == task)
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("frozen motif prompt identity/order changed")
    prior_data = json.loads(PRIOR.read_text()) if task == "motif_extension" else None
    if prior_data is not None and prior_data["catalog_sha256"] != contract["catalog_sha256"]:
        raise ValueError("motif prior/catalog training identities disagree")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != contract["versions"]:
        raise ValueError("motif development model/chemistry environment changed")
    for raw, expected in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != expected:
            raise ValueError(f"motif implementation changed: {raw}")
    return versions, prompts, prior_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--task", required=True, choices=tuple(CONTRACTS))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash = load_contract(args.task)
    versions, prompts, prior_data = preflight(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("motif pilot output differs from frozen contract")
    manifest = {
        "schema": "fragment_motif_focused_dev_manifest_v1",
        "contract_payload_sha256": contract_hash,
        "contract": contract,
        "versions": versions,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": "cpu",
        "threads": 1,
        "precision": "float32",
        "role": "single-fresh-seed development; not published-scale benchmark",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(output)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(manifest_path)})
        )
        return
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("motif pilot manifest missing or changed")
    if (output / "summary.json").exists():
        raise FileExistsError("motif development result already complete")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    if args.task == "motif_extension":
        entries = json.loads(CATALOGS[args.task].read_text())["entries"]
        sampler = JointCompletionSampler(entries, JointCompletionPrior.from_dict(prior_data))
        sample_panel = sample_motif_panel
    else:
        sampler = PendantCompletionSampler(json.loads(CATALOGS[args.task].read_text()))
        sample_panel = sample_pendant_panel
    model, _ = load_factorized_rollout_checkpoint(Path(contract["checkpoint_path"]))
    rows, hashes = [], {}
    work_seconds = 0.0
    for prompt in prompts:
        context = build_prompt_context(prompt)
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 1))
        attempts, attempt_hashes = [], {}
        for index in range(20):
            if shutil.disk_usage(output).free < 5_000_000_000:
                raise RuntimeError("pilot stopped before attempt: less than 5 GB disk free")
            path = output / "attempts" / f"{prompt.drug_name}_{index:03d}.json"
            start = output / "starts" / f"{prompt.drug_name}_{index:03d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            else:
                if start.exists():
                    raise RuntimeError(
                        f"interrupted started motif attempt; no automatic redraw: {start}"
                    )
                _atomic_json(
                    start,
                    {
                        "drug": prompt.drug_name,
                        "attempt_index": index,
                        "rng_state_before": rng.bit_generator.state,
                        "manifest_sha256": physical_sha256(manifest_path),
                    },
                )
                began = time.monotonic()
                panel = sample_panel(context, sampler, model, rng)
                selected = panel.selected
                fidelity = False
                if selected:
                    constraint = ProgramConstraint.from_context(context)
                    lock = constraint.lock(context.start_state)
                    fidelity = constraint.complete(selected.endpoint) and all(
                        lock.permits(decode_state(state)) for state in selected.trace["states"]
                    )
                    verify_prompt_endpoint(context, selected)
                row = {
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": panel.receipt,
                    "wall_seconds": time.monotonic() - began,
                    "selected_valid_connected": bool(
                        selected
                        and is_valid_state(selected.endpoint)
                        and is_connected_or_null(selected.endpoint)
                    ),
                    "selected_constraint_fidelity": fidelity,
                }
                if selected and not (row["selected_valid_connected"] and fidelity):
                    raise RuntimeError(
                        "motif selected output violates chemistry or fragment constraint"
                    )
                _atomic_json(path, row)
            attempts.append(row)
            work_seconds += row["wall_seconds"]
            attempt_hashes[str(path.relative_to(output))] = physical_sha256(path)
            if index % 5 == 4:
                completed = len(rows) * 20 + index + 1
                progress = {
                    "schema": "fragment_focused_dev_progress_v1",
                    "task": args.task,
                    "completed_attempts": completed,
                    "total_attempts": 200,
                    "outputs": sum(item["outputs"] for item in rows)
                    + sum(item["panel"]["output_count"] for item in attempts),
                    "candidate_work_seconds": work_seconds,
                    "estimated_remaining_candidate_seconds": work_seconds
                    / completed
                    * (200 - completed),
                    "sealed_prompts": len(rows),
                    "disk_free_bytes": shutil.disk_usage(output).free,
                    "official_metrics_so_far": {
                        key: float(np.mean([item["metrics"][key] for item in rows]))
                        for key in ("validity", "uniqueness", "quality", "diversity")
                    }
                    if rows
                    else None,
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                }
                _atomic_json(output / "progress.json", progress)
                print(json.dumps({"progress": progress}), flush=True)
        samples = attempt_samples(attempts, drug=prompt.drug_name)
        lock_path = output / "locks" / f"{prompt.drug_name}.json"
        immutable_json(
            lock_path,
            {
                "manifest_sha256": physical_sha256(manifest_path),
                "attempt_hashes": attempt_hashes,
                "samples": samples,
            },
        )
        row_path = output / "rows" / f"{prompt.drug_name}.json"
        if row_path.exists():
            metric_row = json.loads(row_path.read_text())
            if metric_row["lock_sha256"] != physical_sha256(lock_path):
                raise ValueError("saved motif row lost its sample lock")
        else:
            metric_row = {
                "drug": prompt.drug_name,
                "attempts": 20,
                "outputs": sum(item["panel"]["output_count"] for item in attempts),
                "valid_connected_outputs": sum(
                    item["selected_valid_connected"] for item in attempts
                ),
                "constraint_fidelity_outputs": sum(
                    item["selected_constraint_fidelity"] for item in attempts
                ),
                "metrics": official_prompt_metrics(samples, expected_samples=20),
                "lock_sha256": physical_sha256(lock_path),
            }
            _atomic_json(row_path, metric_row)
        rows.append(metric_row)
        hashes.update(attempt_hashes)
        hashes[str(lock_path.relative_to(output))] = physical_sha256(lock_path)
        hashes[str(row_path.relative_to(output))] = physical_sha256(row_path)
        _atomic_json(
            output / "progress.json",
            {
                "schema": "fragment_focused_dev_progress_v1",
                "task": args.task,
                "completed_attempts": len(rows) * 20,
                "total_attempts": 200,
                "outputs": sum(item["outputs"] for item in rows),
                "candidate_work_seconds": work_seconds,
                "estimated_remaining_candidate_seconds": work_seconds
                / (len(rows) * 20)
                * (200 - len(rows) * 20),
                "sealed_prompts": len(rows),
                "official_metrics_so_far": {
                    key: float(np.mean([item["metrics"][key] for item in rows]))
                    for key in ("validity", "uniqueness", "quality", "diversity")
                },
                "disk_free_bytes": shutil.disk_usage(output).free,
                "drug": prompt.drug_name,
                "attempt_index": 19,
            },
        )
        print(
            json.dumps({"sealed_prompt": prompt.drug_name, "metrics": metric_row["metrics"]}),
            flush=True,
        )
    if len(rows) != 10 or any(row["attempts"] != 20 for row in rows):
        raise RuntimeError("motif development result missing a sealed prompt")
    for raw, expected in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != expected:
            raise ValueError(f"development implementation changed during run: {raw}")
    metrics = {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in ("validity", "uniqueness", "quality", "diversity")
    }
    summary = {
        "schema": "fragment_focused_dev_result_v1",
        "task": args.task,
        "role": "fresh-seed development, not a formal comparator win",
        "attempts": 200,
        "offered_candidate_draws": 1600,
        "outputs": sum(row["outputs"] for row in rows),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in rows),
        "constraint_fidelity_outputs": sum(row["constraint_fidelity_outputs"] for row in rows),
        "candidate_status": dict(
            sorted(
                Counter(
                    candidate["status"]
                    for path in (output / "attempts").glob("*.json")
                    for candidate in json.loads(path.read_text())["panel"]["offered"]
                ).items()
            )
        ),
        "metrics": metrics,
        "checks": {
            "quality_at_least_predeclared": metrics["quality"] >= contract["minimum_quality"],
            "uniqueness_at_least_predeclared": metrics["uniqueness"]
            >= contract["minimum_uniqueness"],
            "diversity_at_least_predeclared": metrics["diversity"] >= contract["minimum_diversity"],
            "output_at_least_95_percent": sum(row["outputs"] for row in rows) >= 190,
            "all_committed_valid_connected": all(
                row["valid_connected_outputs"] == row["outputs"] for row in rows
            ),
            "all_committed_constraint_faithful": all(
                row["constraint_fidelity_outputs"] == row["outputs"] for row in rows
            ),
        },
        "manifest_sha256": physical_sha256(manifest_path),
        "artifact_sha256": dict(sorted(hashes.items())),
        "oracle_calls": 0,
    }
    summary["development_pass"] = all(summary["checks"].values())
    _atomic_json(output / "summary.json", summary)
    print(
        json.dumps({"completed": True, "metrics": metrics, "checks": summary["checks"]}), flush=True
    )


if __name__ == "__main__":
    main()
