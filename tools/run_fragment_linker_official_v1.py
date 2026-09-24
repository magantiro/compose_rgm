"""Frozen, independent three-seed linker/morphing benchmark evaluation.

Each eight-draw generation attempt is durable. A started but incomplete attempt
fails closed; completed attempts replay their saved RNG chain on resume. Quality
is computed only after all 100 attempts for one prompt/seed are locked.
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
from audit_fragment_training_linker import CATALOG_SHA, CHECKPOINT_SHA, ROOT
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from fetch_official_fragment_evaluator import verify_only
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog, sample_linker_panel
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_prompt_metrics,
)
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state

CONTRACT = ROOT / "configs/fragment_linker_official_v1.json"
DEVELOPMENT = ROOT / "diagnostics/fragment_training_linker_metric_pilot_v2"
CATALOG = ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
PRIOR = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
METRICS = ("validity", "uniqueness", "quality", "diversity")


def _identity(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(path: Path = CONTRACT) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed linker contract: {path}")
    payload = envelope["payload"]
    if _identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"linker contract hash mismatch: {path}")
    if (
        payload["schema"] != "fragment_linker_official_v1"
        or payload["task"] != "linker_design"
        or payload["prompt_count"] != 10
        or payload["attempts_per_prompt_seed"] != 100
        or payload["candidate_draws_per_attempt"] != 8
        or payload["seeds"] != [1, 2, 3]
        or payload["development_seed_excluded"] != 0
        or payload["workers"] != 1
        or payload["threads"] != 1
        or payload["quality_selection"] is not False
    ):
        raise ValueError("linker contract changes frozen benchmark envelope")
    return payload, envelope["payload_sha256"]


def benchmark_summary(rows: list[dict], contract: dict) -> dict:
    if len(contract["drugs"]) != 10 or len(set(contract["drugs"])) != 10:
        raise ValueError("linker result contract requires ten distinct prompts")
    expected = {(seed, drug) for seed in contract["seeds"] for drug in contract["drugs"]}
    observed = {(row["seed"], row["drug"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("linker result needs exactly ten prompt rows at each fresh seed")
    if any(row["attempts"] != 100 for row in rows):
        raise ValueError("linker result has an incomplete 100-attempt prompt row")
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
        "schema": "fragment_linker_official_result_v1",
        "role": "independent fresh-seed published-scale linker evaluation",
        "attempts": 3000,
        "offered_candidate_draws": 24000,
        "outputs": sum(row["outputs"] for row in rows),
        "connected_valid_outputs": sum(row["connected_valid_outputs"] for row in rows),
        "exact_core_path_fidelity_outputs": sum(
            row["exact_core_path_fidelity_outputs"] for row in rows
        ),
        "per_seed": per_seed,
        "official_mean": {
            metric: float(np.mean([row[metric] for row in per_seed])) for metric in METRICS
        },
        "morphing": "identical-input alias; not an independent replicate",
        "oracle_calls": 0,
    }


def preflight(contract: dict) -> tuple[dict, tuple]:
    if Path(sys.prefix).resolve() != Path(contract["python_executable"]).parent.parent.resolve():
        raise ValueError(f"wrong linker benchmark Python: {sys.executable}")
    checkpoint = Path(contract["checkpoint_path"])
    if physical_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("frozen linker checkpoint changed")
    development_manifest = DEVELOPMENT / "manifest.json"
    development_summary = DEVELOPMENT / "summary.json"
    if (
        physical_sha256(development_manifest) != contract["development_manifest_sha256"]
        or physical_sha256(development_summary) != contract["development_summary_sha256"]
    ):
        raise ValueError("qualified linker development result changed")
    old = json.loads(development_manifest.read_text())
    result = json.loads(development_summary.read_text())
    if result["manifest_sha256"] != physical_sha256(development_manifest):
        raise ValueError("development summary is detached from its manifest")
    for raw, digest in old["inputs"].items():
        if physical_sha256(Path(raw)) != digest:
            raise ValueError(f"qualified linker dependency changed: {raw}")
    for raw, digest in contract["material_sha256"].items():
        if physical_sha256(ROOT / raw) != digest:
            raise ValueError(f"official linker material changed: {raw}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("frozen linker source revision is not an ancestor of HEAD")
    if (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
    ):
        raise ValueError("tracked linker source worktree is dirty")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != old["versions"]:
        raise ValueError("linker model/chemistry environment changed")
    verified = verify_only()
    if contract["official_evaluator_sha256"] not in verified.values():
        raise ValueError("pinned official IVG evaluator unavailable")
    prompts = tuple(p for p in load_genmol_prompts(PROMPTS) if p.task is FragmentTask.LINKER_DESIGN)
    morphing = {
        p.drug_name: p
        for p in load_genmol_prompts(PROMPTS)
        if p.task is FragmentTask.SCAFFOLD_MORPHING
    }
    if [p.drug_name for p in prompts] != contract["drugs"] or any(
        p.fragments != morphing[p.drug_name].fragments for p in prompts
    ):
        raise ValueError("linker/morphing prompt identity changed")
    return versions, prompts


def _attempt_path(output: Path, seed: int, drug: str, index: int) -> Path:
    return output / "attempts" / f"seed{seed}" / f"{drug}_{index:03d}.json"


def attempt_samples(attempts: list[dict], *, drug: str) -> list[str]:
    if len(attempts) != 100 or [(row["drug"], row["attempt_index"]) for row in attempts] != [
        (drug, index) for index in range(100)
    ]:
        raise ValueError("official linker row requires 100 ordered attempts")
    if any(
        row["panel"]["offered_count"] != 8
        or [offered["draw"] for offered in row["panel"]["offered"]] != list(range(8))
        for row in attempts
    ):
        raise ValueError("official linker row requires eight recorded draws per attempt")
    return [row["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for row in attempts]


def _run(contract: dict, manifest_path: Path, output: Path, prompts: tuple) -> None:
    catalog = load_linker_catalog(CATALOG, expected_sha256=CATALOG_SHA)
    prior = JointCompletionPrior.from_dict(json.loads(PRIOR.read_text()))
    model = None
    row_hashes: dict[str, str] = {}
    rows: list[dict] = []
    work_seconds = 0.0
    manifest_hash = physical_sha256(manifest_path)
    for seed in contract["seeds"]:
        for prompt in prompts:
            rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, seed))
            attempts = []
            attempt_hashes = {}
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
                            f"interrupted started linker attempt; explicit recovery required: {start}"
                        )
                    if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                        raise RuntimeError(
                            "linker run stopped at its frozen disk-free safety floor"
                        )
                    if model is None:
                        model, _ = load_factorized_rollout_checkpoint(
                            Path(contract["checkpoint_path"])
                        )
                        if any(
                            p.dtype != torch.float32 or p.device.type != "cpu"
                            for p in model.parameters()
                        ):
                            raise ValueError("linker checkpoint must load as CPU float32")
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
                    panel = sample_linker_panel(prompt, catalog, prior, model, rng)
                    selected = panel.selected
                    fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
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
                        "selected_exact_core_path_fidelity": bool(
                            selected
                            and fidelity["satisfied"]
                            and selected.provenance["exact_mapped_core_identity_checked"]
                            and selected.provenance["source_core_locked_all_states"]
                        ),
                        "selected_fidelity": fidelity,
                    }
                    if selected and not (
                        receipt["selected_valid_connected"]
                        and receipt["selected_exact_core_path_fidelity"]
                    ):
                        raise RuntimeError(
                            f"selected linker violates hard chemistry/core/path invariant: {seed}/{prompt.drug_name}/{index}"
                        )
                    _atomic_json(path, receipt)
                if (
                    receipt["seed"] != seed
                    or start.exists()
                    and json.loads(start.read_text())["manifest_sha256"] != manifest_hash
                ):
                    raise ValueError("saved linker attempt/manifest identity changed")
                attempts.append(receipt)
                work_seconds += receipt["wall_seconds"]
                attempt_hashes[str(path.relative_to(output))] = physical_sha256(path)
                if index % 10 == 9:
                    completed = seed_progress(contract, rows, index + 1)
                    progress = {
                        "schema": "fragment_linker_official_progress_v1",
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
                    }
                    _atomic_json(
                        output / "progress.json",
                        progress,
                    )
                    print(json.dumps({"progress": progress}), flush=True)
            samples = attempt_samples(attempts, drug=prompt.drug_name)
            lock = output / "locks" / f"seed{seed}" / f"{prompt.drug_name}.json"
            immutable_json(
                lock,
                {
                    "manifest_sha256": manifest_hash,
                    "attempt_hashes": attempt_hashes,
                    "samples": samples,
                },
            )
            row_path = output / "rows" / f"seed{seed}" / f"{prompt.drug_name}.json"
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if row["lock_sha256"] != physical_sha256(lock):
                    raise ValueError("saved linker metric row lost its exact sample lock")
            else:
                row = {
                    "seed": seed,
                    "drug": prompt.drug_name,
                    "task": "linker_design",
                    "attempts": 100,
                    "outputs": sum(a["panel"]["output_count"] for a in attempts),
                    "connected_valid_outputs": sum(a["selected_valid_connected"] for a in attempts),
                    "exact_core_path_fidelity_outputs": sum(
                        a["selected_exact_core_path_fidelity"] for a in attempts
                    ),
                    "metrics": official_prompt_metrics(samples, expected_samples=100),
                    "lock_sha256": physical_sha256(lock),
                }
                _atomic_json(row_path, row)
            rows.append(row)
            row_hashes[str(row_path.relative_to(output))] = physical_sha256(row_path)
            row_hashes[str(lock.relative_to(output))] = physical_sha256(lock)
            row_hashes.update(attempt_hashes)
            print(
                json.dumps(
                    {
                        "completed_prompt": prompt.drug_name,
                        "seed": seed,
                        "outputs": row["outputs"],
                        "metrics": row["metrics"],
                    }
                ),
                flush=True,
            )
    summary = benchmark_summary(rows, contract)
    summary.update(
        {
            "contract_payload_sha256": _identity(contract),
            "manifest_sha256": manifest_hash,
            "artifact_hashes": dict(sorted(row_hashes.items())),
            "execution_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"completed": True, "official_mean": summary["official_mean"]}), flush=True)


def seed_progress(contract: dict, rows: list[dict], current: int) -> int:
    return len(rows) * contract["attempts_per_prompt_seed"] + current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash = load_contract()
    versions, prompts = preflight(contract)
    manifest = {
        "schema": "fragment_linker_official_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "source_revision": contract["source_revision"],
        "development_seed": 0,
        "morphing_pairs_verified_identical": len(prompts),
    }
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("official linker output directory differs from frozen contract")
    path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(f"official linker output namespace already exists: {output}")
        _atomic_json(path, manifest)
        print(json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(path)}))
        return
    if not path.is_file() or json.loads(path.read_text()) != manifest:
        raise ValueError("official linker launch manifest is missing or changed")
    if (output / "summary.json").exists():
        raise FileExistsError("official linker result already complete; no rerun")
    _run(contract, path, output, prompts)


if __name__ == "__main__":
    main()
