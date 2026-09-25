"""Fresh-seed, published-scale evaluation of the frozen focused motif sampler.

Each complete eight-offer attempt is durable. Existing attempts replay their
recorded RNG transition before any new proposal. A started attempt without a
completed receipt fails closed, and prompt metrics are computed only after its
100 ordered samples have been locked.
"""

from __future__ import annotations

import argparse
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
from run_fragment_linker_official_v1 import _identity
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_motif_focused_programs import sample_motif_panel
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)
from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_motif_official_v1.json"
DEVELOPMENT = ROOT / "diagnostics/fragment_motif_focused_dev_v1"
CATALOG = ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
PRIOR = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
METRICS = ("validity", "uniqueness", "quality", "diversity")


def load_contract(path: Path = CONTRACT) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed motif contract: {path}")
    payload = envelope["payload"]
    if _identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"motif contract self-hash mismatch: {path}")
    if (
        payload["schema"] != "fragment_motif_official_v1"
        or payload["task"] != "motif_extension"
        or payload["prompt_count"] != 10
        or payload["attempts_per_prompt_seed"] != 100
        or payload["candidate_draws_per_attempt"] != 8
        or payload["seeds"] != [2, 3, 4]
        or payload["development_seed_excluded"] != 1
        or payload["workers"] != 1
        or payload["quality_selection"] is not False
    ):
        raise ValueError("motif contract changes frozen evaluation envelope")
    return payload, envelope["payload_sha256"]


def preflight(contract: dict) -> tuple[dict, tuple, JointCompletionSampler]:
    if Path(sys.prefix).resolve() != Path(contract["python_executable"]).parent.parent.resolve():
        raise ValueError(f"wrong motif benchmark Python: {sys.executable}")
    if physical_sha256(Path(contract["checkpoint_path"])) != contract["checkpoint_sha256"]:
        raise ValueError("frozen motif checkpoint changed")
    if (
        physical_sha256(DEVELOPMENT / "manifest.json") != contract["development_manifest_sha256"]
        or physical_sha256(DEVELOPMENT / "summary.json") != contract["development_summary_sha256"]
    ):
        raise ValueError("qualified motif development result changed")
    dev_manifest = json.loads((DEVELOPMENT / "manifest.json").read_text())
    dev_summary = json.loads((DEVELOPMENT / "summary.json").read_text())
    if (
        dev_summary["development_pass"] is not True
        or dev_summary["manifest_sha256"] != physical_sha256(DEVELOPMENT / "manifest.json")
        or dev_manifest["contract"]["seed"] != 1
    ):
        raise ValueError("qualified motif development chain is invalid")
    for raw, digest in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != digest:
            raise ValueError(f"motif material changed: {raw}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("frozen motif source revision is not an ancestor of HEAD")
    if (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
    ):
        raise ValueError("tracked motif source worktree is dirty")
    from fetch_official_fragment_evaluator import verify_only

    if contract["official_evaluator_sha256"] not in verify_only().values():
        raise ValueError("pinned official evaluator unavailable")
    prompts = tuple(
        p for p in load_genmol_prompts(PROMPTS) if p.task is FragmentTask.MOTIF_EXTENSION
    )
    if [p.drug_name for p in prompts] != contract["drugs"]:
        raise ValueError("motif prompt identity/order changed")
    prior_data = json.loads(PRIOR.read_text())
    if prior_data["catalog_sha256"] != contract["material_sha256"][str(CATALOG.relative_to(ROOT))]:
        raise ValueError("motif prior/catalog training identities disagree")
    entries = json.loads(CATALOG.read_text())["entries"]
    sampler = JointCompletionSampler(entries, JointCompletionPrior.from_dict(prior_data))
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != dev_manifest["versions"]:
        raise ValueError("motif model/chemistry environment changed")
    return versions, prompts, sampler


def attempt_samples(attempts: list[dict], *, drug: str) -> list[str]:
    if len(attempts) != 100 or [(a["drug"], a["attempt_index"]) for a in attempts] != [
        (drug, i) for i in range(100)
    ]:
        raise ValueError("motif metric row requires 100 ordered attempts")
    if any(
        a["panel"]["offered_count"] != 8
        or [offer["draw"] for offer in a["panel"]["offered"]] != list(range(8))
        for a in attempts
    ):
        raise ValueError("motif metric row requires eight recorded offers per attempt")
    return [a["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for a in attempts]


def _attempt_path(output: Path, seed: int, drug: str, index: int) -> Path:
    return output / "attempts" / f"seed{seed}" / f"{drug}_{index:03d}.json"


def benchmark_summary(rows: list[dict], contract: dict) -> dict:
    expected = {(seed, drug) for seed in contract["seeds"] for drug in contract["drugs"]}
    observed = {(row["seed"], row["drug"]) for row in rows}
    if len(expected) != 30 or len(rows) != 30 or observed != expected:
        raise ValueError("motif result needs exactly ten prompt rows at each fresh seed")
    if any(row["attempts"] != 100 for row in rows):
        raise ValueError("motif result has an incomplete 100-attempt row")
    if any(
        row["valid_connected_outputs"] != row["outputs"]
        or row["constraint_fidelity_outputs"] != row["outputs"]
        for row in rows
    ):
        raise ValueError("motif result contains an invalid or nonfaithful committed output")
    per_seed = [
        {
            "seed": seed,
            **{
                metric: float(
                    np.mean([row["metrics"][metric] for row in rows if row["seed"] == seed])
                )
                for metric in METRICS
            },
        }
        for seed in contract["seeds"]
    ]
    return {
        "schema": "fragment_motif_official_result_v1",
        "role": "independent fresh-seed published-scale motif evaluation",
        "attempts": 3000,
        "offered_candidate_draws": 24000,
        "outputs": sum(row["outputs"] for row in rows),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in rows),
        "constraint_fidelity_outputs": sum(row["constraint_fidelity_outputs"] for row in rows),
        "per_seed": per_seed,
        "official_mean": {
            metric: float(np.mean([row[metric] for row in per_seed])) for metric in METRICS
        },
        "official_std": {
            metric: float(np.std([row[metric] for row in per_seed])) for metric in METRICS
        },
        "oracle_calls": 0,
    }


def _run(contract: dict, manifest_path: Path, output: Path, prompts: tuple, sampler) -> None:
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model = None
    rows, hashes = [], {}
    work_seconds = 0.0
    manifest_hash = physical_sha256(manifest_path)
    for seed in contract["seeds"]:
        for prompt in prompts:
            context = build_prompt_context(prompt)
            rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, seed))
            attempts, attempt_hashes = [], {}
            for index in range(100):
                path = _attempt_path(output, seed, prompt.drug_name, index)
                start = output / "starts" / f"seed{seed}" / f"{prompt.drug_name}_{index:03d}.json"
                if path.exists():
                    receipt = json.loads(path.read_text())
                    restore_completed_attempt(
                        receipt, drug=prompt.drug_name, attempt_index=index, rng=rng
                    )
                else:
                    if start.exists():
                        raise RuntimeError(
                            f"interrupted started motif attempt; audited recovery required: {start}"
                        )
                    if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                        raise RuntimeError("motif run stopped at frozen disk-free safety floor")
                    if model is None:
                        model, _ = load_factorized_rollout_checkpoint(
                            Path(contract["checkpoint_path"])
                        )
                        if any(
                            p.dtype != torch.float32 or p.device.type != "cpu"
                            for p in model.parameters()
                        ):
                            raise ValueError("motif checkpoint must load as CPU float32")
                    _atomic_json(
                        start,
                        {
                            "seed": seed,
                            "drug": prompt.drug_name,
                            "attempt_index": index,
                            "rng_state_before": rng.bit_generator.state,
                            "manifest_sha256": manifest_hash,
                        },
                    )
                    began = time.monotonic()
                    panel = sample_motif_panel(context, sampler, model, rng)
                    selected = panel.selected
                    fidelity = False
                    if selected:
                        constraint = ProgramConstraint.from_context(context)
                        lock = constraint.lock(context.start_state)
                        fidelity = constraint.complete(selected.endpoint) and all(
                            lock.permits(decode_state(state)) for state in selected.trace["states"]
                        )
                        verify_prompt_endpoint(context, selected)
                    receipt = {
                        "seed": seed,
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
                    if selected and not (
                        receipt["selected_valid_connected"]
                        and receipt["selected_constraint_fidelity"]
                    ):
                        raise RuntimeError(
                            f"selected motif violates chemistry/fragment invariant: {seed}/{prompt.drug_name}/{index}"
                        )
                    _atomic_json(path, receipt)
                if receipt["seed"] != seed or not start.is_file():
                    raise ValueError("saved motif attempt/start identity changed")
                started = json.loads(start.read_text())
                if (
                    started["seed"] != seed
                    or started["drug"] != prompt.drug_name
                    or started["attempt_index"] != index
                    or started["manifest_sha256"] != manifest_hash
                    or started["rng_state_before"] != receipt["panel"]["rng_state_before"]
                ):
                    raise ValueError("saved motif attempt/start RNG or manifest identity changed")
                attempts.append(receipt)
                work_seconds += receipt["wall_seconds"]
                attempt_hashes[str(path.relative_to(output))] = physical_sha256(path)
                if index % 10 == 9:
                    completed = len(rows) * 100 + index + 1
                    _atomic_json(
                        output / "progress.json",
                        {
                            "schema": "fragment_motif_official_progress_v1",
                            "manifest_sha256": manifest_hash,
                            "completed_attempts": completed,
                            "total_attempts": 3000,
                            "seed": seed,
                            "drug": prompt.drug_name,
                            "prompt_completed_attempts": index + 1,
                            "outputs": sum(row["outputs"] for row in rows)
                            + sum(a["panel"]["output_count"] for a in attempts),
                            "candidate_work_seconds": work_seconds,
                            "estimated_remaining_candidate_seconds": work_seconds
                            * (3000 - completed)
                            / completed,
                            "disk_free_bytes": shutil.disk_usage(output).free,
                            "sealed_prompt_rows": len(rows),
                        },
                    )
            samples = attempt_samples(attempts, drug=prompt.drug_name)
            lock_path = output / "locks" / f"seed{seed}" / f"{prompt.drug_name}.json"
            immutable_json(
                lock_path,
                {
                    "manifest_sha256": manifest_hash,
                    "attempt_hashes": attempt_hashes,
                    "samples": samples,
                },
            )
            row_path = output / "rows" / f"seed{seed}" / f"{prompt.drug_name}.json"
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if row["lock_sha256"] != physical_sha256(lock_path):
                    raise ValueError("saved motif row lost exact sample lock")
            else:
                row = {
                    "seed": seed,
                    "drug": prompt.drug_name,
                    "task": "motif_extension",
                    "attempts": 100,
                    "outputs": sum(a["panel"]["output_count"] for a in attempts),
                    "valid_connected_outputs": sum(a["selected_valid_connected"] for a in attempts),
                    "constraint_fidelity_outputs": sum(
                        a["selected_constraint_fidelity"] for a in attempts
                    ),
                    "metrics": official_prompt_metrics(samples, expected_samples=100),
                    "lock_sha256": physical_sha256(lock_path),
                }
                _atomic_json(row_path, row)
            rows.append(row)
            hashes[str(lock_path.relative_to(output))] = physical_sha256(lock_path)
            hashes[str(row_path.relative_to(output))] = physical_sha256(row_path)
            hashes.update(attempt_hashes)
            print(
                json.dumps(
                    {"completed_prompt": prompt.drug_name, "seed": seed, "metrics": row["metrics"]}
                ),
                flush=True,
            )
    summary = benchmark_summary(rows, contract)
    summary.update(
        {
            "contract_payload_sha256": _identity(contract),
            "manifest_sha256": manifest_hash,
            "artifact_hashes": dict(sorted(hashes.items())),
            "execution_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"completed": True, "official_mean": summary["official_mean"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash = load_contract()
    versions, prompts, sampler = preflight(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("official motif output differs from frozen contract")
    manifest = {
        "schema": "fragment_motif_official_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "source_revision": contract["source_revision"],
        "development_seed": 1,
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(f"official motif output namespace already exists: {output}")
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(manifest_path)})
        )
        return
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("official motif launch manifest is missing or changed")
    if (output / "summary.json").exists():
        raise FileExistsError("official motif result already complete; no rerun")
    _run(contract, manifest_path, output, prompts, sampler)


if __name__ == "__main__":
    main()
