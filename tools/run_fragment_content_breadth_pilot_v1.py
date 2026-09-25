"""Matched, fresh-seed decoration pilot for training-content breadth.

Both arms use the same joint mass law, exact executor, frozen reference model,
eight-offer selector, prompts and evaluator. Only within-cell content weights
change. Samples are locked before any benchmark-quality calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint, verify_prompt_endpoint
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_content_breadth_pilot_v1.json"
ARMS = ("frozen", "uniform_within_cell")
METRICS = ("quality", "uniqueness", "diversity", "validity")


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract() -> tuple[dict, str]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid self-hashed decoration contract: {CONTRACT}")
    contract = envelope["payload"]
    if identity(contract) != envelope["payload_sha256"]:
        raise ValueError("decoration pilot contract hash mismatch")
    if (
        contract["schema"] != "fragment_content_breadth_pilot_v1"
        or contract["task"] != "scaffold_decoration"
        or tuple(contract["arms"]) != ARMS
        or len(contract["drugs"]) != 10
        or len(set(contract["drugs"])) != 10
        or contract["seed"] != 6
        or contract["attempts_per_prompt_arm"] != 20
        or contract["candidate_draws_per_attempt"] != 8
        or contract["workers"] != 1
        or contract["quality_selection"] is not False
        or contract["oracle_calls"] != 0
        or contract["promotion_gate"]
        != {
            "minimum_diversity": 0.56,
            "minimum_diversity_gain": 0.01,
            "maximum_quality_drop_points": 2.0,
            "minimum_uniqueness": 90.0,
            "minimum_outputs_per_arm": 190,
        }
    ):
        raise ValueError("decoration pilot scientific envelope changed")
    return contract, envelope["payload_sha256"]


def preflight(contract: dict) -> tuple[tuple, dict, dict]:
    if Path(sys.prefix).resolve() != Path(contract["python_executable"]).parent.parent.resolve():
        raise ValueError(f"wrong decoration pilot Python: {sys.executable}")
    for raw, digest in contract["material_sha256"].items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if physical_sha256(path) != digest:
            raise ValueError(f"decoration pilot material changed: {path}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("decoration pilot source revision is not an ancestor of HEAD")
    if any(
        subprocess.run(command, cwd=ROOT, check=False).returncode
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"])
    ):
        raise ValueError("tracked decoration pilot source worktree is dirty")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != contract["versions"]:
        raise ValueError("decoration pilot software versions changed")
    from fetch_official_fragment_evaluator import verify_only

    if contract["official_evaluator_sha256"] not in verify_only().values():
        raise ValueError("pinned official evaluator unavailable")
    prompt_path = Path(contract["prompt_path"])
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(prompt_path)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("decoration pilot prompt order changed")
    catalog = json.loads(Path(contract["catalog_path"]).read_text())
    prior = json.loads(Path(contract["mass_prior_path"]).read_text())
    JointMassPendantSampler(catalog, prior)
    JointMassPendantSampler(catalog, prior, content_allocation="uniform_within_cell")
    return prompts, catalog, prior


def _run_cell(
    *, output: Path, manifest_sha: str, prompt, arm: str, sampler, model, contract: dict
) -> dict:
    context = build_prompt_context(prompt)
    constraint = ProgramConstraint.from_context(context)
    protected = constraint.lock(context.start_state)
    rng = np.random.default_rng(
        prompt_rng_seed(prompt.drug_name, prompt.task.value, contract["seed"])
    )
    attempts: list[dict] = []
    attempt_hashes: dict[str, str] = {}
    count = contract["attempts_per_prompt_arm"]
    for index in range(count):
        relative = Path("attempts") / arm / f"{prompt.drug_name}_{index:03d}.json"
        path = output / relative
        start = output / "starts" / arm / f"{prompt.drug_name}_{index:03d}.json"
        if path.exists():
            if not start.exists():
                raise ValueError(f"completed attempt lacks a start receipt: {path}")
            expected_start = {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "rng_state_before": rng.bit_generator.state,
                "manifest_sha256": manifest_sha,
            }
            if json.loads(start.read_text()) != expected_start:
                raise ValueError(f"decoration attempt start receipt changed: {start}")
            receipt = json.loads(path.read_text())
            if receipt["arm"] != arm:
                raise ValueError(f"decoration attempt arm changed: {path}")
            restore_completed_attempt(receipt, drug=prompt.drug_name, attempt_index=index, rng=rng)
        else:
            if start.exists():
                raise RuntimeError(f"started attempt cannot be redrawn: {start}")
            if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                raise RuntimeError("decoration pilot stopped at frozen disk-free floor")
            _atomic_json(
                start,
                {
                    "arm": arm,
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "rng_state_before": rng.bit_generator.state,
                    "manifest_sha256": manifest_sha,
                },
            )
            began = time.monotonic()
            panel = sample_pendant_panel(context, sampler, model, rng)
            selected = panel.selected
            valid = bool(
                selected
                and is_valid_state(selected.endpoint)
                and is_connected_or_null(selected.endpoint)
            )
            faithful = bool(
                selected
                and constraint.complete(selected.endpoint)
                and all(
                    protected.permits(decode_state(state)) for state in selected.trace["states"]
                )
            )
            if selected:
                verify_prompt_endpoint(context, selected)
            if selected and not (valid and faithful):
                raise RuntimeError(f"invalid/nonfaithful output: {arm}/{prompt.drug_name}/{index}")
            receipt = {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "panel": panel.receipt,
                "wall_seconds": time.monotonic() - began,
                "selected_valid_connected": valid,
                "selected_constraint_fidelity": faithful,
            }
            _atomic_json(path, receipt)
        attempts.append(receipt)
        attempt_hashes[str(relative)] = physical_sha256(path)
        if (index + 1) % 5 == 0:
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "fragment_content_breadth_pilot_progress_v1",
                    "arm": arm,
                    "drug": prompt.drug_name,
                    "completed_attempts_in_cell": index + 1,
                    "total_attempts_in_cell": count,
                    "outputs_in_cell": sum(row["panel"]["output_count"] for row in attempts),
                    "disk_free_bytes": shutil.disk_usage(output).free,
                },
            )
    samples = [row["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for row in attempts]
    lock_path = output / "locks" / arm / f"{prompt.drug_name}.json"
    immutable_json(
        lock_path,
        {"manifest_sha256": manifest_sha, "attempt_hashes": attempt_hashes, "samples": samples},
    )
    row_path = output / "rows" / arm / f"{prompt.drug_name}.json"
    if row_path.exists():
        row = json.loads(row_path.read_text())
        if row["lock_sha256"] != physical_sha256(lock_path):
            raise ValueError(f"saved decoration row lost its sample lock: {row_path}")
    else:
        row = {
            "arm": arm,
            "drug": prompt.drug_name,
            "attempts": count,
            "outputs": sum(item["panel"]["output_count"] for item in attempts),
            "valid_connected_outputs": sum(item["selected_valid_connected"] for item in attempts),
            "constraint_fidelity_outputs": sum(
                item["selected_constraint_fidelity"] for item in attempts
            ),
            "metrics": official_prompt_metrics(samples, expected_samples=count),
            "lock_sha256": physical_sha256(lock_path),
        }
        _atomic_json(row_path, row)
    print(json.dumps({"sealed": f"{arm}/{prompt.drug_name}"}), flush=True)
    return row


def summarize(rows: dict[str, list[dict]], contract: dict) -> dict:
    count = contract["attempts_per_prompt_arm"]
    for arm in ARMS:
        if len(rows[arm]) != 10 or any(row["attempts"] != count for row in rows[arm]):
            raise ValueError(f"decoration pilot has incomplete {arm} rows")
    means = {
        arm: {
            metric: float(np.mean([row["metrics"][metric] for row in rows[arm]]))
            for metric in METRICS
        }
        for arm in ARMS
    }
    outputs = {arm: sum(row["outputs"] for row in rows[arm]) for arm in ARMS}
    gate = contract["promotion_gate"]
    changed = means["uniform_within_cell"]
    frozen = means["frozen"]
    checks = {
        "minimum_diversity": changed["diversity"] >= gate["minimum_diversity"],
        "minimum_diversity_gain": changed["diversity"] - frozen["diversity"]
        >= gate["minimum_diversity_gain"],
        "maximum_quality_drop": changed["quality"]
        >= frozen["quality"] - gate["maximum_quality_drop_points"],
        "minimum_uniqueness": changed["uniqueness"] >= gate["minimum_uniqueness"],
        "minimum_outputs": all(
            value >= gate["minimum_outputs_per_arm"] for value in outputs.values()
        ),
        "chemical_validity_of_outputs": all(
            row["outputs"] == row["valid_connected_outputs"] for arm in ARMS for row in rows[arm]
        ),
        "exact_prompt_fidelity": all(
            row["outputs"] == row["valid_connected_outputs"] == row["constraint_fidelity_outputs"]
            for arm in ARMS
            for row in rows[arm]
        ),
    }
    return {
        "schema": "fragment_content_breadth_pilot_result_v1",
        "role": "matched fresh-seed development; not a final benchmark",
        "seed": contract["seed"],
        "attempts_per_arm": 10 * count,
        "offered_candidate_draws_per_arm": 10 * count * contract["candidate_draws_per_attempt"],
        "outputs": outputs,
        "metrics": means,
        "checks": checks,
        "promotion_gate_pass": all(checks.values()),
        "quality_guidance_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash = load_contract()
    prompts, catalog, prior = preflight(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("decoration pilot output differs from frozen contract")
    manifest = {
        "schema": "fragment_content_breadth_pilot_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": "cpu",
        "threads": 1,
        "precision": "float32 model; float64 proposal probabilities",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(output)
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "payload_sha256": contract_hash}))
        return
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("decoration pilot manifest missing or changed")
    authorization = output / "authorization.json"
    if not authorization.exists():
        raise PermissionError("exact payload-bound decoration launch authorization is missing")
    approved = json.loads(authorization.read_text())
    if approved != {"approved": True, "payload_sha256": contract_hash}:
        raise PermissionError("decoration launch authorization does not bind this payload")
    if (output / "summary.json").exists():
        raise FileExistsError("decoration pilot already complete")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(contract["checkpoint_path"]))
    samplers = {
        "frozen": JointMassPendantSampler(catalog, prior),
        "uniform_within_cell": JointMassPendantSampler(
            catalog, prior, content_allocation="uniform_within_cell"
        ),
    }
    manifest_sha = physical_sha256(manifest_path)
    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for prompt in prompts:
        for arm in ARMS:
            rows[arm].append(
                _run_cell(
                    output=output,
                    manifest_sha=manifest_sha,
                    prompt=prompt,
                    arm=arm,
                    sampler=samplers[arm],
                    model=model,
                    contract=contract,
                )
            )
    summary = summarize(rows, contract)
    summary.update(
        {
            "contract_payload_sha256": contract_hash,
            "manifest_sha256": manifest_sha,
            "row_sha256": {
                str(path.relative_to(output)): physical_sha256(path)
                for path in sorted((output / "rows").glob("*/*.json"))
            },
        }
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"completed": True, "checks": summary["checks"]}), flush=True)


if __name__ == "__main__":
    main()
