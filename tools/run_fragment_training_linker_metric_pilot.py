"""One immutable 200-attempt linker development measurement, reusing its prefix.

No generator/scorer changes: continue the exact support RNG chains and compute
official metrics only after locking all twenty attempts for each prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from audit_fragment_training_linker import CATALOG_SHA, CHECKPOINT_SHA, ROOT
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from fetch_official_fragment_evaluator import default_cache_dir, verify_only
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_support import restore_completed_attempt, support_summary

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

SUPPORT = ROOT / "diagnostics/fragment_training_linker_panel_support_v1"
CONTRACT = ROOT / "configs/fragment_training_linker_metric_pilot_v1.json"


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable pilot artifact changed: {path}")
    else:
        _atomic_json(path, value)


def verify_support_prefix():
    manifest_path, summary_path = SUPPORT / "manifest.json", SUPPORT / "summary.json"
    manifest, summary = (json.loads(p.read_text()) for p in (manifest_path, summary_path))
    if summary["manifest_sha256"] != physical_sha256(manifest_path) or not summary["support_pass"]:
        raise ValueError("linker support is not sealed and passing")
    for raw, expected in manifest["inputs"].items():
        if physical_sha256(Path(raw)) != expected:
            raise ValueError(f"support-bound sampler/model/prior/source changed: {raw}")
    rows = {}
    for drug in manifest["configuration"]["prompts"]:
        for index in range(2):
            relative = f"attempts/{drug}_{index:03d}.json"
            path = SUPPORT / relative
            if summary["attempt_hashes"].get(relative) != physical_sha256(path):
                raise ValueError(f"support prefix hash changed: {relative}")
            rows[(drug, index)] = json.loads(path.read_text())
    expected = support_summary(list(rows.values()))
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("support summary disagrees with its exact attempt receipts")
    return manifest, summary, rows


def prompt_samples(attempts, *, drug):
    if len(attempts) != 20 or [(a["drug"], a["attempt_index"]) for a in attempts] != [
        (drug, index) for index in range(20)
    ]:
        raise ValueError("official metrics require all twenty ordered prompt attempts")
    if any(a["panel"]["offered_count"] != 8 for a in attempts):
        raise ValueError("official metrics require eight recorded candidate draws per attempt")
    return [a["panel"]["selected_smiles"] or FAILED_SAMPLE_PLACEHOLDER for a in attempts]


def imported_prefix_attempt(source_row, source_path):
    return {
        **source_row,
        "origin": {
            "role": "reused_support_prefix",
            "source": str(source_path.resolve()),
            "sha256": physical_sha256(source_path),
        },
    }


def pilot_summary(rows, contract):
    if (
        len(rows) != 10
        or len({row["drug"] for row in rows}) != 10
        or any(row["attempts"] != 20 for row in rows)
    ):
        raise ValueError("pilot summary requires all ten twenty-attempt prompt rows")
    metrics = {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in ("validity", "uniqueness", "quality", "diversity")
    }
    outputs = sum(row["outputs"] for row in rows)
    faithful = sum(row["exact_core_path_fidelity_outputs"] for row in rows)
    return {
        "schema": "fragment_training_linker_metric_pilot_result_v1",
        "role": "single-seed development, reused support prefix; not a formal benchmark win",
        "metrics": metrics,
        "attempts": 200,
        "outputs": outputs,
        "output_coverage": outputs / 200,
        "exact_core_path_fidelity_outputs": faithful,
        "exact_core_path_fidelity_precision": faithful / outputs if outputs else None,
        "support_attempts_reused": 20,
        "new_attempts": 180,
        "candidate_draws_total_including_reused": 1600,
        "new_candidate_draws": 1440,
        "checks": {
            "output_at_least_90_percent": outputs / 200 >= contract["required_output_coverage"],
            "all_selected_exact_core_path_fidelity": outputs > 0 and faithful == outputs,
        },
        "reported_comparator_differences": {
            name: {key: metrics[key] - comparator[key] for key in metrics}
            for name, comparator in contract["reported_comparators"].items()
        },
        "morphing": "identical-input alias; same outputs, not an independent replicate",
        "oracle_calls": 0,
        "quality_evaluated_after_prompt_locks": True,
    }


def prepare_manifest(checkpoint):
    support, summary, prefix = verify_support_prefix()
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != support["versions"]:
        raise ValueError("metric pilot software differs from the passing support environment")
    if physical_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("metric pilot checkpoint differs from passing support")
    contract = json.loads(CONTRACT.read_text())
    if (
        contract["attempts_per_prompt"] != 20
        or contract["candidate_draws_per_attempt"] != 8
        or contract["support_attempts_reused_per_prompt"] != 2
        or contract["seed"] != 0
        or contract["new_attempts"] != 180
        or contract["new_candidate_draws"] != 1440
        or contract["max_workers"] != 1
        or contract["quality_selection"]
    ):
        raise ValueError("metric pilot is outside the authorized frozen envelope")
    all_prompts = load_genmol_prompts(
        ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    )
    prompts = tuple(p for p in all_prompts if p.task is FragmentTask.LINKER_DESIGN)
    morphing = {p.drug_name: p for p in all_prompts if p.task is FragmentTask.SCAFFOLD_MORPHING}
    if [p.drug_name for p in prompts] != support["configuration"]["prompts"] or any(
        p.fragments != morphing[p.drug_name].fragments for p in prompts
    ):
        raise ValueError("linker/morphing source pairs or support prompt order changed")
    official = verify_only()
    paths = {
        Path(__file__).resolve(),
        CONTRACT,
        SUPPORT / "summary.json",
        SUPPORT / "manifest.json",
        ROOT / "docs/FRAGMENT_TRAINING_LINKER_METRIC_PILOT_2026-09-24.md",
        ROOT / "docs/FRAGMENT_CONSTRAINED_BENCHMARK.md",
        ROOT / "src/compose_v4/benchmark/fragment_official_metrics.py",
        ROOT / "tools/fetch_official_fragment_evaluator.py",
        ROOT / "diagnostics/fragment_linker_comparator_references_v1/receipt.json",
        ROOT / "diagnostics/fragment_linker_comparator_references_v1/genmol_README.md",
    }
    paths.update(default_cache_dir() / "pkg" / relative for relative in official)
    paths.update(SUPPORT / relative for relative in summary["attempt_hashes"])
    ref = json.loads(
        (ROOT / "diagnostics/fragment_linker_comparator_references_v1/receipt.json").read_text()
    )
    readme = ROOT / contract["reported_comparators"]["genmol_v2_author_readme"]["source"]
    if physical_sha256(readme) != ref["sha256"] or any(
        contract["reported_comparators"]["genmol_v2_author_readme"][key] != value
        for key, value in ref["metrics"].items()
    ):
        raise ValueError("GenMol V2 reported comparator drift")
    inputs = {
        **support["inputs"],
        **{str(path.resolve()): physical_sha256(path) for path in sorted(paths)},
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest = {
        "schema": "fragment_training_linker_metric_pilot_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "inputs": inputs,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "support_prefix": "exact attempt 0/1 receipts and RNG states, reused without recalculation",
        "morphing_pairs_verified_identical": len(prompts),
        "predicted_new_candidate_wall_seconds": summary["candidate_and_selection_wall_seconds"] * 9,
        "prediction_role": "operational estimate from the twenty-attempt support; not a stopping rule",
    }
    return manifest, prompts, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest, prompts, prefix = prepare_manifest(args.checkpoint)
    manifest_path = args.output_dir / "manifest.json"
    if args.phase == "prepare":
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": "prepared_only",
                    "manifest_sha256": physical_sha256(manifest_path),
                    "contract_sha256": manifest["contract_payload_sha256"],
                }
            )
        )
        return
    if not manifest_path.exists():
        raise ValueError("prepare the immutable pilot manifest before launch")
    frozen = json.loads(manifest_path.read_text())
    if {k: v for k, v in frozen.items() if k != "code_revision"} != {
        k: v for k, v in manifest.items() if k != "code_revision"
    }:
        raise ValueError("frozen pilot identities changed before launch/resume")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("pilot is already complete")
    catalog = load_linker_catalog(
        ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json",
        expected_sha256=CATALOG_SHA,
    )
    prior = JointCompletionPrior.from_dict(
        json.loads((ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json").read_text())
    )
    model, rows, all_hashes = None, [], {}
    for prompt in prompts:
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 0))
        attempts, attempt_hashes = [], {}
        for index in range(20):
            name = f"{prompt.drug_name}_{index:03d}"
            path = args.output_dir / "attempts" / f"{name}.json"
            start_path = args.output_dir / "starts" / f"{name}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if index < 2 and row != imported_prefix_attempt(
                    prefix[(prompt.drug_name, index)], SUPPORT / "attempts" / f"{name}.json"
                ):
                    raise ValueError("imported support prefix differs from its frozen source row")
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            elif index < 2:
                row = imported_prefix_attempt(
                    prefix[(prompt.drug_name, index)], SUPPORT / "attempts" / f"{name}.json"
                )
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
                _atomic_json(path, row)
            else:
                if start_path.exists():
                    raise ValueError(f"interrupted started panel {name}; no automatic regeneration")
                if model is None:
                    model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
                    if any(
                        p.dtype != torch.float32 or p.device.type != "cpu"
                        for p in model.parameters()
                    ):
                        raise ValueError("frozen pilot requires CPU float32 model parameters")
                _atomic_json(
                    start_path,
                    {
                        "drug": prompt.drug_name,
                        "attempt_index": index,
                        "rng_state_before": rng.bit_generator.state,
                        "manifest_sha256": physical_sha256(manifest_path),
                    },
                )
                started = time.monotonic()
                result = sample_linker_panel(prompt, catalog, prior, model, rng)
                selected = result.selected
                fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
                row = {
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": result.receipt,
                    "origin": {"role": "new_pilot_attempt"},
                    "wall_seconds": time.monotonic() - started,
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
                _atomic_json(path, row)
            attempts.append(row)
            attempt_hashes[str(path.relative_to(args.output_dir))] = physical_sha256(path)
            progress = {
                "drug": prompt.drug_name,
                "completed_prompt_attempts": len(attempts),
                "completed_metric_rows": len(rows),
                "total_prompts": 10,
                "total_attempts": 200,
            }
            _atomic_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        samples = prompt_samples(attempts, drug=prompt.drug_name)
        lock_path = args.output_dir / "locks" / f"{prompt.drug_name}.json"
        immutable_json(
            lock_path,
            {
                "attempt_hashes": attempt_hashes,
                "samples": samples,
                "manifest_sha256": physical_sha256(manifest_path),
            },
        )
        row_path = args.output_dir / "rows" / f"{prompt.drug_name}.json"
        if row_path.exists():
            metric_row = json.loads(row_path.read_text())
            if metric_row["lock_sha256"] != physical_sha256(lock_path):
                raise ValueError("saved metric row does not match its immutable prompt lock")
        else:
            metrics = official_prompt_metrics(samples, expected_samples=20)
            metric_row = {
                "drug": prompt.drug_name,
                "task": prompt.task.value,
                "attempts": 20,
                "outputs": sum(a["panel"]["output_count"] for a in attempts),
                "exact_core_path_fidelity_outputs": sum(
                    a["selected_exact_core_path_fidelity"] for a in attempts
                ),
                "metrics": metrics,
                "lock_sha256": physical_sha256(lock_path),
                "support_attempts_reused": 2,
            }
            _atomic_json(row_path, metric_row)
        rows.append(metric_row)
        all_hashes.update(attempt_hashes)
        all_hashes[str(row_path.relative_to(args.output_dir))] = physical_sha256(row_path)
        all_hashes[str(lock_path.relative_to(args.output_dir))] = physical_sha256(lock_path)
        print(json.dumps({"locked_prompt_metrics": metric_row}), flush=True)
    if any(physical_sha256(Path(path)) != digest for path, digest in frozen["inputs"].items()):
        raise ValueError("pilot dependencies changed during execution")
    summary = {
        **pilot_summary(rows, frozen["contract"]),
        "manifest_sha256": physical_sha256(manifest_path),
        "artifact_hashes": dict(sorted(all_hashes.items())),
        "execution_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
