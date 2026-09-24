"""Resume the failed linker pilot under the sealed perceived-ring repair.

The completed v1 prefix is imported byte-for-byte. The one interrupted v1
start is explicitly replayed once from its saved pre-attempt RNG state. A new
interrupted v2 start fails closed; this command never retries it automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from audit_fragment_training_linker import CATALOG_SHA, CHECKPOINT_SHA, ROOT
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_metric_pilot import pilot_summary, prompt_samples
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog, sample_linker_panel
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state

V1 = ROOT / "diagnostics/fragment_training_linker_metric_pilot_v1"
CONTRACT = ROOT / "configs/fragment_training_linker_metric_pilot_v2.json"
REPAIR_DOC = ROOT / "docs/FRAGMENT_LINKER_RING_COUNT_REPAIR_2026-09-24.md"
SAMPLER = ROOT / "src/compose_v4/benchmark/fragment_linker_sampler.py"
START_NAME = "LIOTHYRONINE_007"


def _copy_verified(source: Path, destination: Path, digest: str) -> None:
    if physical_sha256(source) != digest:
        raise ValueError(f"source v1 artifact changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if physical_sha256(destination) != digest:
            raise ValueError(f"imported v1 artifact changed: {destination}")
        return
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".import-", delete=False
    ) as temp:
        temp_name = Path(temp.name)
    try:
        shutil.copyfile(source, temp_name)
        if physical_sha256(temp_name) != digest:
            raise ValueError(f"v1 artifact copy changed: {source}")
        os.replace(temp_name, destination)
    finally:
        temp_name.unlink(missing_ok=True)


def prepare_manifest() -> tuple[dict, tuple, dict]:
    repair = json.loads(CONTRACT.read_text())
    original_path = V1 / "manifest.json"
    if physical_sha256(original_path) != repair["source_v1_manifest_sha256"]:
        raise ValueError("source v1 manifest changed")
    original = json.loads(original_path.read_text())
    old_sampler_hash = original["inputs"].get(str(SAMPLER.resolve()))
    old_sampler = subprocess.check_output(
        [
            "git",
            "show",
            f"{original['code_revision']}:src/compose_v4/benchmark/fragment_linker_sampler.py",
        ],
        cwd=ROOT,
    )
    if hashlib.sha256(old_sampler).hexdigest() != old_sampler_hash:
        raise ValueError("source v1 sampler cannot be recovered at its recorded revision")
    for raw, expected in original["inputs"].items():
        if Path(raw) == SAMPLER:
            continue
        if physical_sha256(Path(raw)) != expected:
            raise ValueError(f"source v1 input changed: {raw}")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != original["versions"]:
        raise ValueError("v2 chemistry/model environment differs from v1")
    if (
        physical_sha256(ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json")
        != CATALOG_SHA
    ):
        raise ValueError("training connector catalog changed")
    if (
        physical_sha256(ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json")
        != original["inputs"][
            str((ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json").resolve())
        ]
    ):
        raise ValueError("joint prior changed")
    prompts = tuple(
        p
        for p in load_genmol_prompts(
            ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        )
        if p.task == FragmentTask.LINKER_DESIGN
    )
    if len(prompts) != 10 or prompts[6].drug_name != "LIOTHYRONINE":
        raise ValueError("source prompt order changed")
    if (
        repair["source_v1_completed_attempts"] != 127
        or repair["source_v1_completed_prompt_rows"] != 6
        or repair["remaining_attempts_including_explicit_replay"] != 73
        or repair["interrupted_started_attempt"] != START_NAME
        or repair["candidate_draws_per_attempt"] != 8
        or repair["max_workers"] != 1
        or repair["seed"] != 0
    ):
        raise ValueError("v2 repair contract changed its bounded execution envelope")
    expected_prefix = {
        f"{prompt.drug_name}_{index:03d}"
        for position, prompt in enumerate(prompts)
        for index in range(20 if position < 6 else 7 if position == 6 else 0)
    }
    observed_prefix = {path.stem for path in (V1 / "attempts").glob("*.json")}
    if observed_prefix != expected_prefix:
        raise ValueError("source v1 complete-attempt set is not the recorded 127-prefix")
    starts = {path.stem for path in (V1 / "starts").glob("*.json")}
    if starts - observed_prefix != {START_NAME}:
        raise ValueError("source v1 has a different set of interrupted starts")
    start_path = V1 / "starts" / f"{START_NAME}.json"
    start = json.loads(start_path.read_text())
    if start["manifest_sha256"] != physical_sha256(original_path):
        raise ValueError("interrupted v1 start has a different manifest")
    if any(start[k] != v for k, v in (("drug", "LIOTHYRONINE"), ("attempt_index", 7))):
        raise ValueError("interrupted v1 start identity changed")
    expected_rows = {prompt.drug_name for prompt in prompts[:6]}
    if {path.stem for path in (V1 / "rows").glob("*.json")} != expected_rows or {
        path.stem for path in (V1 / "locks").glob("*.json")
    } != expected_rows:
        raise ValueError("source v1 completed row/lock set changed")
    imported = {
        f"attempts/{name}.json": physical_sha256(V1 / "attempts" / f"{name}.json")
        for name in sorted(expected_prefix)
    }
    for name in sorted(expected_rows):
        for folder in ("rows", "locks"):
            relative = f"{folder}/{name}.json"
            imported[relative] = physical_sha256(V1 / relative)
        lock = json.loads((V1 / "locks" / f"{name}.json").read_text())
        row = json.loads((V1 / "rows" / f"{name}.json").read_text())
        if row["lock_sha256"] != imported[f"locks/{name}.json"]:
            raise ValueError(f"v1 row does not match its prompt lock: {name}")
        if any(
            imported.get(relative) != digest for relative, digest in lock["attempt_hashes"].items()
        ):
            raise ValueError(f"v1 prompt lock has a changed attempt: {name}")
    inputs = {
        **{raw: digest for raw, digest in original["inputs"].items() if Path(raw) != SAMPLER},
        **{
            str(path.resolve()): physical_sha256(path)
            for path in (
                SAMPLER,
                CONTRACT,
                REPAIR_DOC,
                Path(__file__).resolve(),
                original_path,
                start_path,
            )
        },
        **{str((V1 / relative).resolve()): digest for relative, digest in imported.items()},
    }
    combined_contract = {**original["contract"], "repair_v2": repair}
    payload = json.dumps(
        combined_contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    manifest = {
        "schema": "fragment_training_linker_metric_pilot_manifest_v2",
        "contract": combined_contract,
        "contract_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "source_v1_manifest_sha256": repair["source_v1_manifest_sha256"],
        "source_v1_sampler_sha256": old_sampler_hash,
        "v1_imported_artifact_sha256": dict(sorted(imported.items())),
        "v1_interrupted_start_sha256": physical_sha256(start_path),
        "v1_failure": {
            "kind": "nonadditive_rdkit_perceived_ring_count",
            "offered_draw": 4,
            "connector": "[1*]N1C2CNCC1C2[2*]",
            "planned_cell": [21, 4],
            "actual_cell": [21, 3],
        },
        "inputs": dict(sorted(inputs.items())),
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "role": "explicit bounded recovery; not an independent benchmark replicate",
    }
    return manifest, prompts, original


def run(output: Path, manifest: dict, prompts: tuple, source_manifest: dict) -> None:
    if any(physical_sha256(Path(path)) != digest for path, digest in manifest["inputs"].items()):
        raise ValueError("v2 material input drift before execution")
    catalog = load_linker_catalog(
        ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json",
        expected_sha256=CATALOG_SHA,
    )
    prior = JointCompletionPrior.from_dict(
        json.loads((ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json").read_text())
    )
    checkpoint = next(
        Path(raw) for raw, digest in source_manifest["inputs"].items() if digest == CHECKPOINT_SHA
    )
    if physical_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("frozen checkpoint changed")
    model, rows, hashes = None, [], {}
    for position, prompt in enumerate(prompts):
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 0))
        attempts = []
        old_count = 20 if position < 6 else 7 if position == 6 else 0
        for index in range(20):
            name = f"{prompt.drug_name}_{index:03d}"
            relative = f"attempts/{name}.json"
            path = output / relative
            if index < old_count:
                expected = manifest["v1_imported_artifact_sha256"][relative]
                _copy_verified(V1 / relative, path, expected)
                row = json.loads(path.read_text())
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            elif path.exists():
                row = json.loads(path.read_text())
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            else:
                start_path = output / "starts" / f"{name}.json"
                if start_path.exists():
                    raise ValueError(
                        f"interrupted v2 started panel {name}; no automatic regeneration"
                    )
                origin = {"role": "new_v2_attempt"}
                if name == START_NAME:
                    old_start = V1 / "starts" / f"{name}.json"
                    if (
                        json.loads(old_start.read_text())["rng_state_before"]
                        != rng.bit_generator.state
                    ):
                        raise ValueError("explicit replay RNG differs from the failed v1 start")
                    origin = {
                        "role": "explicit_v1_failed_start_replay",
                        "source": str(old_start.resolve()),
                        "sha256": manifest["v1_interrupted_start_sha256"],
                    }
                if model is None:
                    model, _ = load_factorized_rollout_checkpoint(checkpoint)
                    if any(
                        p.dtype != torch.float32 or p.device.type != "cpu"
                        for p in model.parameters()
                    ):
                        raise ValueError("v2 requires the frozen CPU float32 model")
                _atomic_json(
                    start_path,
                    {
                        "drug": prompt.drug_name,
                        "attempt_index": index,
                        "rng_state_before": rng.bit_generator.state,
                        "manifest_sha256": physical_sha256(output / "manifest.json"),
                        "origin": origin,
                    },
                )
                started = time.monotonic()
                panel = sample_linker_panel(prompt, catalog, prior, model, rng)
                if name == START_NAME:
                    mismatch = panel.receipt["offered"][4]
                    if (
                        mismatch["status"] != "compiler_or_constraint_abstention"
                        or mismatch.get("proposal_receipt", {}).get("planned_final_cell") != [21, 4]
                        or mismatch.get("proposal_receipt", {}).get("actual_final_cell") != [21, 3]
                    ):
                        raise ValueError(
                            "explicit v1 failure witness did not replay as a consumed refusal"
                        )
                selected = panel.selected
                fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
                row = {
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": panel.receipt,
                    "origin": origin,
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
            hashes[relative] = physical_sha256(path)
            _atomic_json(
                output / "progress.json",
                {
                    "drug": prompt.drug_name,
                    "completed_prompt_attempts": len(attempts),
                    "completed_metric_rows": len(rows),
                    "total_prompts": 10,
                    "total_attempts": 200,
                    "v1_imported_complete_attempts": 127,
                },
            )
            print(
                json.dumps({"drug": prompt.drug_name, "attempt": index, "complete": True}),
                flush=True,
            )
        lock_relative = f"locks/{prompt.drug_name}.json"
        row_relative = f"rows/{prompt.drug_name}.json"
        lock_path, row_path = output / lock_relative, output / row_relative
        if position < 6:
            for relative in (lock_relative, row_relative):
                _copy_verified(
                    V1 / relative,
                    output / relative,
                    manifest["v1_imported_artifact_sha256"][relative],
                )
            metric_row = json.loads(row_path.read_text())
            if json.loads(lock_path.read_text())["samples"] != prompt_samples(
                attempts, drug=prompt.drug_name
            ):
                raise ValueError(f"imported prompt lock changed samples: {prompt.drug_name}")
        else:
            samples = prompt_samples(attempts, drug=prompt.drug_name)
            lock = {
                "attempt_hashes": {
                    f"attempts/{prompt.drug_name}_{i:03d}.json": hashes[
                        f"attempts/{prompt.drug_name}_{i:03d}.json"
                    ]
                    for i in range(20)
                },
                "samples": samples,
                "manifest_sha256": physical_sha256(output / "manifest.json"),
            }
            if lock_path.exists():
                if json.loads(lock_path.read_text()) != lock:
                    raise ValueError(f"v2 prompt lock changed: {prompt.drug_name}")
            else:
                _atomic_json(lock_path, lock)
            if row_path.exists():
                metric_row = json.loads(row_path.read_text())
                if metric_row["lock_sha256"] != physical_sha256(lock_path):
                    raise ValueError(f"v2 row differs from its locked prompt: {prompt.drug_name}")
            else:
                metric_row = {
                    "drug": prompt.drug_name,
                    "task": prompt.task.value,
                    "attempts": 20,
                    "outputs": sum(a["panel"]["output_count"] for a in attempts),
                    "exact_core_path_fidelity_outputs": sum(
                        a["selected_exact_core_path_fidelity"] for a in attempts
                    ),
                    "metrics": official_prompt_metrics(samples, expected_samples=20),
                    "lock_sha256": physical_sha256(lock_path),
                    "support_attempts_reused": 2,
                }
                _atomic_json(row_path, metric_row)
        rows.append(metric_row)
        hashes[lock_relative], hashes[row_relative] = (
            physical_sha256(lock_path),
            physical_sha256(row_path),
        )
        print(json.dumps({"locked_prompt_metrics": metric_row}), flush=True)
    if any(physical_sha256(Path(path)) != digest for path, digest in manifest["inputs"].items()):
        raise ValueError("v2 material input drift during execution")
    result = {
        **pilot_summary(rows, manifest["contract"]),
        "schema": "fragment_training_linker_metric_pilot_result_v2",
        "role": "bounded v1 failure recovery; 127 imported attempts; not independent replication",
        "source_v1_manifest_sha256": manifest["source_v1_manifest_sha256"],
        "v1_imported_attempts": 127,
        "v2_new_attempts_including_explicit_replay": 73,
        "new_attempts": 73,
        "new_candidate_draws": 584,
        "manifest_sha256": physical_sha256(output / "manifest.json"),
        "artifact_hashes": dict(sorted(hashes.items())),
    }
    _atomic_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest, prompts, original = prepare_manifest()
    manifest_path = args.output_dir / "manifest.json"
    if args.phase == "prepare":
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps({"prepared": str(manifest_path), "sha256": physical_sha256(manifest_path)})
        )
        return
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("v2 manifest missing or changed; prepare from committed source first")
    run(args.output_dir, manifest, prompts, original)


if __name__ == "__main__":
    main()
