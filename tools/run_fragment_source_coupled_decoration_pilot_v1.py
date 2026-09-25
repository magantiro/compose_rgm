"""Matched, quality-blind proposal test for source-coupled decoration.

All attempted molecules are locked before the official QED/SA evaluator runs.
The runner is bound to a self-hashed contract and a separate launch receipt.
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
from compose_v4.benchmark.source_coupled_pendant_policy import SourceCoupledPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_source_coupled_decoration_pilot_v1.json"
ARMS = ("frozen", "source_coupled")
METRICS = ("quality", "uniqueness", "diversity", "validity")


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract() -> tuple[dict, str]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid self-hashed source-coupled contract: {CONTRACT}")
    contract = envelope["payload"]
    if _identity(contract) != envelope["payload_sha256"]:
        raise ValueError("source-coupled contract payload hash mismatch")
    if (
        contract["schema"] != "fragment_source_coupled_decoration_pilot_v1"
        or contract["task"] != "scaffold_decoration"
        or tuple(contract["arms"]) != ARMS
        or len(contract["drugs"]) != 10
        or len(set(contract["drugs"])) != 10
        or contract["seed"] != 7
        or contract["attempts_per_prompt_arm"] != 20
        or contract["offers_per_attempt"] != 8
        or contract["workers"] != 1
        or contract["source_coupling_probability"] != 0.5
        or contract["quality_guided_selection"] is not False
        or contract["promotion_gate"]
        != {
            "minimum_quality_gain_points": 3.0,
            "minimum_diversity": 0.56,
            "minimum_uniqueness_percent": 90.0,
            "minimum_outputs_per_arm": 190,
            "all_committed_valid_and_faithful": True,
        }
    ):
        raise ValueError("source-coupled scientific comparison envelope changed")
    return contract, envelope["payload_sha256"]


def preflight(contract: dict) -> tuple[tuple, dict, dict, str]:
    if Path(sys.executable).resolve() != Path(contract["python_executable"]).resolve():
        raise ValueError("wrong source-coupled pilot Python environment")
    for raw, digest in contract["material_sha256"].items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if physical_sha256(path) != digest:
            raise ValueError(f"source-coupled pilot material changed: {path}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("source-coupled pilot source revision is not an ancestor of HEAD")
    if any(
        subprocess.run(command, cwd=ROOT, check=False).returncode
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"])
    ):
        raise ValueError("source-coupled pilot tracked source worktree is dirty")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != contract["versions"]:
        raise ValueError("source-coupled pilot software environment changed")
    from fetch_official_fragment_evaluator import verify_only

    if contract["official_evaluator_sha256"] not in verify_only().values():
        raise ValueError("pinned official fragment evaluator unavailable")
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(Path(contract["prompt_path"]))
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("source-coupled pilot prompt population changed")
    catalog = json.loads(Path(contract["catalog_path"]).read_text())
    prior = json.loads(Path(contract["mass_prior_path"]).read_text())
    JointMassPendantSampler(catalog, prior)
    SourceCoupledPendantSampler(
        catalog, prior, coupled_probability=contract["source_coupling_probability"]
    )
    return prompts, catalog, prior, revision


def _run_cell(*, output, manifest_sha, prompt, arm, sampler, model, contract) -> Path:
    context = build_prompt_context(prompt)
    constraint = ProgramConstraint.from_context(context)
    protected = constraint.lock(context.start_state)
    rng = np.random.default_rng(
        prompt_rng_seed(prompt.drug_name, prompt.task.value, contract["seed"])
    )
    count = contract["attempts_per_prompt_arm"]
    attempts = []
    attempt_hashes = {}
    for index in range(count):
        relative = Path("attempts") / arm / f"{prompt.drug_name}_{index:03d}.json"
        path = output / relative
        start = output / "starts" / arm / f"{prompt.drug_name}_{index:03d}.json"
        before = {
            "arm": arm,
            "drug": prompt.drug_name,
            "attempt_index": index,
            "rng_state_before": rng.bit_generator.state,
            "manifest_sha256": manifest_sha,
        }
        if path.exists():
            if not start.exists() or json.loads(start.read_text()) != before:
                raise ValueError(
                    f"completed source-coupled attempt lost its start identity: {path}"
                )
            attempt = json.loads(path.read_text())
            if attempt.get("arm") != arm:
                raise ValueError(f"completed source-coupled attempt changed arms: {path}")
            restore_completed_attempt(attempt, drug=prompt.drug_name, attempt_index=index, rng=rng)
        else:
            if start.exists():
                raise RuntimeError(f"started attempt cannot be redrawn: {start}")
            if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                raise RuntimeError("source-coupled pilot stopped at frozen disk-free floor")
            _atomic_json(start, before)
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
            attempt = {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "panel": panel.receipt,
                "wall_seconds": time.monotonic() - began,
                "selected_valid_connected": valid,
                "selected_constraint_fidelity": faithful,
            }
            _atomic_json(path, attempt)
        attempts.append(attempt)
        attempt_hashes[str(relative)] = physical_sha256(path)
        if (index + 1) % 5 == 0:
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "fragment_source_coupled_decoration_progress_v1",
                    "arm": arm,
                    "drug": prompt.drug_name,
                    "completed_attempts_in_cell": index + 1,
                    "total_attempts_in_cell": count,
                    "outputs_in_cell": sum(row["panel"]["output_count"] for row in attempts),
                },
            )
    samples = [row["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for row in attempts]
    lock_path = output / "locks" / arm / f"{prompt.drug_name}.json"
    immutable_json(
        lock_path,
        {"manifest_sha256": manifest_sha, "attempt_hashes": attempt_hashes, "samples": samples},
    )
    print(
        json.dumps(
            {"locked": f"{arm}/{prompt.drug_name}", "outputs": sum(bool(s) for s in samples)}
        ),
        flush=True,
    )
    return lock_path


def summarize(rows: dict[str, list[dict]], contract: dict) -> dict:
    count = contract["attempts_per_prompt_arm"]
    for arm in ARMS:
        if len(rows[arm]) != 10 or any(row["attempts"] != count for row in rows[arm]):
            raise ValueError(f"incomplete source-coupled {arm} result population")
    means = {
        arm: {
            metric: float(np.mean([row["metrics"][metric] for row in rows[arm]]))
            for metric in METRICS
        }
        for arm in ARMS
    }
    outputs = {arm: sum(row["outputs"] for row in rows[arm]) for arm in ARMS}
    gate = contract["promotion_gate"]
    changed = means["source_coupled"]
    frozen = means["frozen"]
    checks = {
        "minimum_quality_gain": changed["quality"] - frozen["quality"]
        >= gate["minimum_quality_gain_points"],
        "minimum_diversity": changed["diversity"] >= gate["minimum_diversity"],
        "minimum_uniqueness": changed["uniqueness"] >= gate["minimum_uniqueness_percent"],
        "minimum_outputs": all(
            value >= gate["minimum_outputs_per_arm"] for value in outputs.values()
        ),
        "all_committed_valid_and_faithful": all(
            row["outputs"] == row["valid_connected_outputs"] == row["constraint_fidelity_outputs"]
            for arm in ARMS
            for row in rows[arm]
        ),
    }
    return {
        "schema": "fragment_source_coupled_decoration_result_v1",
        "role": "matched fresh-seed development comparison; not a final benchmark",
        "attempts_per_arm": 10 * count,
        "offers_per_arm": 10 * count * contract["offers_per_attempt"],
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
    contract, payload_sha = load_contract()
    prompts, catalog, prior, revision = preflight(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("source-coupled pilot output differs from frozen contract")
    manifest = {
        "schema": "fragment_source_coupled_decoration_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": payload_sha,
        "code_revision": revision,
        "device": "cpu",
        "threads": 1,
        "precision": "float32 model; float64 proposal probabilities",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(output)
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "payload_sha256": payload_sha}))
        return
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("source-coupled pilot manifest missing or changed")
    authorization_path = output / "authorization.json"
    if not authorization_path.exists() or json.loads(authorization_path.read_text()) != {
        "approved": True,
        "payload_sha256": payload_sha,
    }:
        raise PermissionError("exact payload-bound source-coupled launch authorization is missing")
    if (output / "summary.json").exists():
        raise FileExistsError("source-coupled pilot already complete")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(contract["checkpoint_path"]))
    samplers = {
        "frozen": JointMassPendantSampler(catalog, prior),
        "source_coupled": SourceCoupledPendantSampler(
            catalog, prior, coupled_probability=contract["source_coupling_probability"]
        ),
    }
    manifest_sha = physical_sha256(manifest_path)
    locks = []
    for prompt in prompts:
        for arm in ARMS:
            locks.append(
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
    if len(locks) != 20:
        raise ValueError("source-coupled pilot did not lock all prompt-arm cells")
    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for prompt in prompts:
        for arm in ARMS:
            lock_path = output / "locks" / arm / f"{prompt.drug_name}.json"
            lock = json.loads(lock_path.read_text())
            if lock["manifest_sha256"] != manifest_sha:
                raise ValueError(f"source-coupled lock changed manifest: {lock_path}")
            samples = lock["samples"]
            if len(samples) != contract["attempts_per_prompt_arm"]:
                raise ValueError(f"source-coupled lock has incomplete denominator: {lock_path}")
            attempts = []
            for index in range(contract["attempts_per_prompt_arm"]):
                path = output / "attempts" / arm / f"{prompt.drug_name}_{index:03d}.json"
                if lock["attempt_hashes"].get(str(path.relative_to(output))) != physical_sha256(
                    path
                ):
                    raise ValueError(f"source-coupled attempt changed after lock: {path}")
                attempts.append(json.loads(path.read_text()))
            row = {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempts": contract["attempts_per_prompt_arm"],
                "outputs": sum(bool(sample) for sample in samples),
                "valid_connected_outputs": sum(
                    item["selected_valid_connected"] for item in attempts
                ),
                "constraint_fidelity_outputs": sum(
                    item["selected_constraint_fidelity"] for item in attempts
                ),
                "metrics": official_prompt_metrics(
                    samples, expected_samples=contract["attempts_per_prompt_arm"]
                ),
                "lock_sha256": physical_sha256(lock_path),
            }
            immutable_json(output / "rows" / arm / f"{prompt.drug_name}.json", row)
            rows[arm].append(row)
    summary = summarize(rows, contract)
    summary.update(
        {
            "contract_payload_sha256": payload_sha,
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
