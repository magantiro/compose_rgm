"""Run one matched, fresh-seed decoration content-prior comparison.

Both arms use the same checkpoint, exact compiler, joint size law, eight-offer
model selection, prompts, evaluator, and attempt count. Only the train-derived
within-category pendant-content law changes. Preparation never scores samples;
the run phase requires an exact payload-bound authorization receipt.
"""

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

from compose_v4.benchmark.conditional_content_pendant_policy import (
    ConditionalContentPendantSampler,
)
from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_conditional_content_decoration_pilot_v1.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
MASS = ROOT / "diagnostics/fragment_training_decoration_mass_prior_v1/prior.json"
CONTENT = ROOT / "diagnostics/fragment_training_decoration_content_context_v1/prior.json"
SCHEMA = "fragment_conditional_content_decoration_pilot_v1"


def identity(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def preflight() -> tuple[dict, str, tuple, dict, dict, dict]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("decoration pilot contract lacks a self-hashed envelope")
    contract = envelope["payload"]
    if identity(contract) != envelope["payload_sha256"]:
        raise ValueError("decoration pilot contract payload changed")
    if (
        contract["schema"] != SCHEMA
        or contract["task"] != "scaffold_decoration"
        or contract["arms"] != ["joint_mass", "conditional_content"]
        or contract["attempts_per_prompt_per_arm"] != 10
        or contract["candidate_draws_per_attempt"] != 8
        or contract["seed"] != 5
        or contract["workers"] != 1
        or contract["quality_selection"] is not False
        or contract["promotion_gate"]
        != {
            "minimum_quality_gain_points": 5.0,
            "minimum_quality_percent": 36.37,
            "minimum_diversity": 0.56,
            "minimum_uniqueness_percent": 88.58,
            "minimum_outputs_per_arm": 95,
            "all_committed_valid_and_faithful": True,
        }
    ):
        raise ValueError("decoration pilot contract changed the comparison")
    materials = (
        (PROMPTS, contract["prompt_sha256"]),
        (CATALOG, contract["catalog_sha256"]),
        (MASS, contract["mass_prior_sha256"]),
        (CONTENT, contract["content_prior_sha256"]),
        (Path(contract["checkpoint_path"]), contract["checkpoint_sha256"]),
        *tuple((ROOT / name, digest) for name, digest in contract["material_sha256"].items()),
    )
    for path, digest in materials:
        if physical_sha256(path) != digest:
            raise ValueError(f"decoration pilot material changed: {path}")
    from fetch_official_fragment_evaluator import verify_only

    if contract["official_evaluator_sha256"] not in verify_only().values():
        raise ValueError("pinned official fragment evaluator unavailable")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != contract["versions"]:
        raise ValueError("decoration pilot software environment changed")
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if [prompt.drug_name for prompt in prompts] != contract["drugs"]:
        raise ValueError("decoration pilot prompt set or order changed")
    catalog, mass, content = (json.loads(path.read_text()) for path in (CATALOG, MASS, CONTENT))
    JointMassPendantSampler(catalog, mass)
    ConditionalContentPendantSampler(
        catalog, mass, content, catalog_sha256=contract["catalog_sha256"]
    )
    return contract, envelope["payload_sha256"], prompts, catalog, mass, content


def _attempt_samples(attempts: list[dict], drug: str, count: int) -> list[str]:
    if [(row["drug"], row["attempt_index"]) for row in attempts] != [
        (drug, index) for index in range(count)
    ]:
        raise ValueError("decoration samples lack ordered attempt denominator")
    if any(
        row["panel"]["offered_count"] != 8
        or [offer["draw"] for offer in row["panel"]["offered"]] != list(range(8))
        for row in attempts
    ):
        raise ValueError("decoration attempt lacks eight recorded offers")
    return [row["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for row in attempts]


def _run_arm_prompt(
    *, output: Path, manifest_hash: str, prompt, arm: str, sampler, model, count: int
) -> dict:
    context = build_prompt_context(prompt)
    constraint = ProgramConstraint.from_context(context)
    lock = constraint.lock(context.start_state)
    rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 5))
    attempts, attempt_hashes = [], {}
    for index in range(count):
        if shutil.disk_usage(output).free < 5_000_000_000:
            raise RuntimeError("decoration pilot stopped before attempt: disk below 5 GB")
        path = output / "attempts" / arm / f"{prompt.drug_name}_{index:03d}.json"
        start = output / "starts" / arm / f"{prompt.drug_name}_{index:03d}.json"
        if path.exists():
            if not start.exists():
                raise ValueError(f"completed attempt lacks its start receipt: {path}")
            start_receipt = json.loads(start.read_text())
            if start_receipt != {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "rng_state_before": rng.bit_generator.state,
                "manifest_sha256": manifest_hash,
            }:
                raise ValueError(f"decoration attempt start receipt changed: {start}")
            row = json.loads(path.read_text())
            if row["arm"] != arm:
                raise ValueError(f"decoration attempt arm changed: {path}")
            restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
        else:
            if start.exists():
                raise RuntimeError(f"started attempt cannot be redrawn: {start}")
            _atomic_json(
                start,
                {
                    "arm": arm,
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "rng_state_before": rng.bit_generator.state,
                    "manifest_sha256": manifest_hash,
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
                and all(lock.permits(decode_state(state)) for state in selected.trace["states"])
            )
            if selected:
                verify_prompt_endpoint(context, selected)
            if selected and not (valid and faithful):
                raise RuntimeError(f"invalid/nonfaithful output: {arm}/{prompt.drug_name}/{index}")
            row = {
                "arm": arm,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "panel": panel.receipt,
                "wall_seconds": time.monotonic() - began,
                "selected_valid_connected": valid,
                "selected_constraint_fidelity": faithful,
            }
            _atomic_json(path, row)
        attempts.append(row)
        attempt_hashes[str(path.relative_to(output))] = physical_sha256(path)
        if (index + 1) % 2 == 0:
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "fragment_conditional_content_decoration_progress_v1",
                    "arm": arm,
                    "drug": prompt.drug_name,
                    "completed_attempts_in_cell": index + 1,
                    "total_attempts_in_cell": count,
                    "outputs_in_cell": sum(item["panel"]["output_count"] for item in attempts),
                    "disk_free_bytes": shutil.disk_usage(output).free,
                },
            )
    samples = _attempt_samples(attempts, prompt.drug_name, count)
    lock_path = output / "locks" / arm / f"{prompt.drug_name}.json"
    immutable_json(
        lock_path,
        {"manifest_sha256": manifest_hash, "attempt_hashes": attempt_hashes, "samples": samples},
    )
    row_path = output / "rows" / arm / f"{prompt.drug_name}.json"
    if row_path.exists():
        row = json.loads(row_path.read_text())
        if row["lock_sha256"] != physical_sha256(lock_path):
            raise ValueError("saved decoration row lost its sample lock")
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
            "candidate_status": dict(
                sorted(
                    Counter(
                        offer["status"] for item in attempts for offer in item["panel"]["offered"]
                    ).items()
                )
            ),
            "metrics": official_prompt_metrics(samples, expected_samples=count),
            "lock_sha256": physical_sha256(lock_path),
        }
        _atomic_json(row_path, row)
    print(
        json.dumps({"sealed": f"{arm}/{prompt.drug_name}", "metrics": row["metrics"]}), flush=True
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash, prompts, catalog, mass, content = preflight()
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("decoration output differs from frozen contract")
    manifest = {
        "schema": "fragment_conditional_content_decoration_manifest_v1",
        "contract_payload_sha256": contract_hash,
        "contract": contract,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": "cpu",
        "threads": 1,
        "precision": "float32",
        "role": "one matched ten-prompt development comparison",
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
        raise PermissionError("exact payload-bound scored-run authorization is missing")
    approved = json.loads(authorization.read_text())
    if approved.get("payload_sha256") != contract_hash or approved.get("approved") is not True:
        raise PermissionError("decoration scored-run authorization does not bind this payload")
    if (output / "summary.json").exists():
        raise FileExistsError("decoration pilot already complete")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(contract["checkpoint_path"]))
    samplers = {
        "joint_mass": JointMassPendantSampler(catalog, mass),
        "conditional_content": ConditionalContentPendantSampler(
            catalog, mass, content, catalog_sha256=contract["catalog_sha256"]
        ),
    }
    manifest_hash = physical_sha256(manifest_path)
    rows = {arm: [] for arm in contract["arms"]}
    for prompt in prompts:
        for arm in contract["arms"]:
            rows[arm].append(
                _run_arm_prompt(
                    output=output,
                    manifest_hash=manifest_hash,
                    prompt=prompt,
                    arm=arm,
                    sampler=samplers[arm],
                    model=model,
                    count=contract["attempts_per_prompt_per_arm"],
                )
            )
    for name, digest in contract["material_sha256"].items():
        if physical_sha256(ROOT / name) != digest:
            raise ValueError(f"decoration implementation changed during run: {name}")
    means = {
        arm: {
            key: float(np.mean([row["metrics"][key] for row in rows[arm]]))
            for key in ("validity", "uniqueness", "quality", "diversity")
        }
        for arm in contract["arms"]
    }
    outputs = {arm: sum(row["outputs"] for row in rows[arm]) for arm in contract["arms"]}
    gate = contract["promotion_gate"]
    checks = {
        "quality_gain_at_least_5_points": means["conditional_content"]["quality"]
        - means["joint_mass"]["quality"]
        >= gate["minimum_quality_gain_points"],
        "quality_above_ivg": means["conditional_content"]["quality"]
        > gate["minimum_quality_percent"],
        "diversity_at_least_ivg": means["conditional_content"]["diversity"]
        >= gate["minimum_diversity"],
        "uniqueness_above_ivg": means["conditional_content"]["uniqueness"]
        > gate["minimum_uniqueness_percent"],
        "both_output_at_least_95_percent": all(
            value >= gate["minimum_outputs_per_arm"] for value in outputs.values()
        ),
        "all_committed_valid_and_faithful": all(
            row["outputs"] == row["valid_connected_outputs"] == row["constraint_fidelity_outputs"]
            for arm in contract["arms"]
            for row in rows[arm]
        ),
    }
    summary = {
        "schema": "fragment_conditional_content_decoration_result_v1",
        "manifest_sha256": manifest_hash,
        "role": "matched development comparison; not a formal comparator result",
        "attempts_per_arm": 100,
        "candidate_draws_per_arm": 800,
        "outputs": outputs,
        "metrics": means,
        "checks": checks,
        "promotion_gate_pass": all(checks.values()),
        "property_guidance_calls": 0,
        "row_sha256": {
            str(path.relative_to(output)): physical_sha256(path)
            for path in sorted((output / "rows").glob("*/*.json"))
        },
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"completed": True, "metrics": means, "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
