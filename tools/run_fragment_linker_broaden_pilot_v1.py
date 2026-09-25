"""Matched 100-attempt linker pilot for one frozen cell-allocation change.

The two arms share prompts, seed, checkpoint, executor, eight-offer budget and
learned selector. Samples are locked before the official evaluator is called.
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
from audit_fragment_training_linker import CATALOG_SHA, CHECKPOINT_SHA
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from fetch_official_fragment_evaluator import verify_only
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_linker_official_v1 import attempt_samples
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import (
    compatible_linker_cells,
    linker_cell_probabilities,
    load_linker_catalog,
    sample_linker_panel,
)
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_linker_broaden_pilot_v1.json"
CATALOG = ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
PRIOR = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
ARMS = ("frozen", "sqrt_train_mass")
METRICS = ("validity", "uniqueness", "quality", "diversity")


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(path: Path = CONTRACT) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed linker pilot contract: {path}")
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"linker pilot contract hash mismatch: {path}")
    if (
        payload["schema"] != "fragment_linker_broaden_pilot_v1"
        or payload["arms"] != list(ARMS)
        or payload["task"] != "linker_design"
        or len(payload["drugs"]) != 10
        or len(set(payload["drugs"])) != 10
        or payload["seed"] != 4
        or payload["attempts_per_prompt_arm"] != 100
        or payload["candidate_draws_per_attempt"] != 8
        or payload["workers"] != 1
        or payload["threads"] != 1
        or payload["quality_selection"] is not False
        or payload["oracle_calls"] != 0
    ):
        raise ValueError("linker pilot contract changes matched scientific envelope")
    return payload, envelope["payload_sha256"]


def summarize(rows: list[dict], contract: dict) -> dict:
    arms = tuple(contract["arms"])
    if arms not in (ARMS, ("frozen", "novelty4")):
        raise ValueError("unrecognized matched linker pilot arms")
    expected = {(arm, drug) for arm in arms for drug in contract["drugs"]}
    observed = {(row["arm"], row["drug"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("linker pilot needs both complete ten-prompt arms")
    if any(row["attempts"] != 100 for row in rows):
        raise ValueError("linker pilot contains an incomplete prompt row")
    by_arm = {
        arm: {
            **{
                metric: float(
                    np.mean([row["metrics"][metric] for row in rows if row["arm"] == arm])
                )
                for metric in METRICS
            },
            "outputs": sum(row["outputs"] for row in rows if row["arm"] == arm),
            "exact_core_path_fidelity_outputs": sum(
                row["exact_core_path_fidelity_outputs"] for row in rows if row["arm"] == arm
            ),
        }
        for arm in arms
    }
    old, new = by_arm["frozen"], by_arm[arms[1]]
    gate = contract["promotion_gate"]
    passed = (
        new["quality"] >= old["quality"] + gate["min_quality_gain_points"]
        and new["uniqueness"] >= old["uniqueness"] + gate["min_uniqueness_gain_points"]
        and new["diversity"] >= old["diversity"] - gate["max_diversity_drop"]
        and all(
            by_arm[arm]["outputs"] >= gate["min_outputs_per_arm"]
            and by_arm[arm]["exact_core_path_fidelity_outputs"] == by_arm[arm]["outputs"]
            and by_arm[arm]["validity"] == 100.0
            for arm in arms
        )
    )
    return {
        "schema": (
            "fragment_linker_broaden_pilot_result_v1"
            if arms == ARMS
            else "fragment_linker_novelty_pilot_result_v1"
        ),
        "role": "single-seed matched development; not an independent final benchmark",
        "seed": contract["seed"],
        "attempts_per_arm": 1000,
        "offered_candidate_draws_per_arm": 8000,
        "by_arm": by_arm,
        "promotion_gate_passed": passed,
        "morphing": "identical-input alias; not a second experiment",
    }


def cell_support_summary(prompts: tuple, arms: tuple[str, str] = ARMS) -> list[dict]:
    """Record the score-blind structural support before sampling begins."""
    catalog = load_linker_catalog(CATALOG, expected_sha256=CATALOG_SHA)
    prior = JointCompletionPrior.from_dict(json.loads(PRIOR.read_text()))
    rows = []
    for prompt in prompts:
        grouped, support = compatible_linker_cells(prompt, catalog)
        if not grouped:
            raise ValueError(f"no compatible linker cell for {prompt.drug_name}")
        cells = tuple(grouped)
        masses = {
            arm: linker_cell_probabilities(
                prior, cells, allocation="frozen" if arm == "novelty4" else arm
            )
            for arm in arms
        }
        rows.append(
            {
                "drug": prompt.drug_name,
                "reachable_cells": len(cells),
                "compatible_training_connectors": support["atom_capacity_compatible_entries"],
                "effective_cells": {
                    arm: float(1.0 / np.sum(probabilities**2))
                    for arm, probabilities in masses.items()
                },
            }
        )
    return rows


def preflight(contract: dict) -> tuple[dict, tuple]:
    if Path(sys.prefix).resolve() != Path(contract["python_executable"]).parent.parent.resolve():
        raise ValueError(f"wrong linker pilot Python: {sys.executable}")
    if physical_sha256(Path(contract["checkpoint_path"])) != CHECKPOINT_SHA:
        raise ValueError("frozen linker checkpoint changed")
    for raw, digest in contract["material_sha256"].items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if physical_sha256(path) != digest:
            raise ValueError(f"linker pilot material changed: {path}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("linker pilot source revision is not an ancestor of HEAD")
    if any(
        subprocess.run(command, cwd=ROOT, check=False).returncode
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"])
    ):
        raise ValueError("tracked linker pilot source worktree is dirty")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if versions != contract["versions"]:
        raise ValueError("linker pilot runtime versions changed")
    if contract["official_evaluator_sha256"] not in verify_only().values():
        raise ValueError("pinned official evaluator unavailable")
    prompts = tuple(p for p in load_genmol_prompts(PROMPTS) if p.task is FragmentTask.LINKER_DESIGN)
    if [p.drug_name for p in prompts] != contract["drugs"]:
        raise ValueError("linker pilot prompt identity changed")
    return versions, prompts


def run(contract: dict, manifest_path: Path, output: Path, prompts: tuple) -> None:
    arms = tuple(contract["arms"])
    if arms not in (ARMS, ("frozen", "novelty4")):
        raise ValueError("unrecognized matched linker pilot arms")
    catalog = load_linker_catalog(CATALOG, expected_sha256=CATALOG_SHA)
    prior = JointCompletionPrior.from_dict(json.loads(PRIOR.read_text()))
    model = None
    rows: list[dict] = []
    artifact_hashes: dict[str, str] = {}
    manifest_sha = physical_sha256(manifest_path)
    work_seconds = 0.0
    for prompt in prompts:
        for arm in arms:
            rng = np.random.default_rng(
                prompt_rng_seed(prompt.drug_name, prompt.task.value, contract["seed"])
            )
            emitted: set[str] = set()
            attempts: list[dict] = []
            attempt_hashes: dict[str, str] = {}
            for index in range(100):
                relative = Path("attempts") / arm / f"{prompt.drug_name}_{index:03d}.json"
                path = output / relative
                start = output / "starts" / arm / f"{prompt.drug_name}_{index:03d}.json"
                if path.exists():
                    receipt = json.loads(path.read_text())
                    if receipt["arm"] != arm:
                        raise ValueError(f"saved attempt arm changed: {path}")
                    restore_completed_attempt(
                        receipt, drug=prompt.drug_name, attempt_index=index, rng=rng
                    )
                else:
                    if start.exists():
                        raise RuntimeError(f"interrupted started linker pilot attempt: {start}")
                    if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                        raise RuntimeError("linker pilot stopped at frozen disk-free floor")
                    if model is None:
                        model, _ = load_factorized_rollout_checkpoint(
                            Path(contract["checkpoint_path"])
                        )
                        if any(
                            p.dtype != torch.float32 or p.device.type != "cpu"
                            for p in model.parameters()
                        ):
                            raise ValueError("linker pilot checkpoint must load as CPU float32")
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
                    panel = sample_linker_panel(
                        prompt,
                        catalog,
                        prior,
                        model,
                        rng,
                        cell_allocation="frozen" if arm == "novelty4" else arm,
                        prior_emitted=frozenset(emitted) if arm == "novelty4" else None,
                    )
                    selected = panel.selected
                    fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
                    receipt = {
                        "arm": arm,
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
                            f"linker pilot committed invalid or unfaithful output: {path}"
                        )
                    _atomic_json(path, receipt)
                if (
                    start.exists()
                    and json.loads(start.read_text())["manifest_sha256"] != manifest_sha
                ):
                    raise ValueError(f"saved attempt manifest changed: {start}")
                attempts.append(receipt)
                selected_smiles = receipt["panel"]["selected_smiles"]
                if selected_smiles is not None:
                    emitted.add(selected_smiles)
                work_seconds += receipt["wall_seconds"]
                attempt_hashes[str(relative)] = physical_sha256(path)
                if index % 10 == 9:
                    completed = len(rows) * 100 + index + 1
                    _atomic_json(
                        output / "progress.json",
                        {
                            "schema": (
                                "fragment_linker_broaden_pilot_progress_v1"
                                if arms == ARMS
                                else "fragment_linker_novelty_pilot_progress_v1"
                            ),
                            "manifest_sha256": manifest_sha,
                            "completed_attempts": completed,
                            "total_attempts": 2000,
                            "arm": arm,
                            "drug": prompt.drug_name,
                            "outputs_in_cell": sum(a["panel"]["output_count"] for a in attempts),
                            "sealed_prompt_rows": len(rows),
                            "candidate_work_seconds": work_seconds,
                            "estimated_remaining_candidate_seconds": work_seconds
                            * (2000 - completed)
                            / completed,
                            "disk_free_bytes": shutil.disk_usage(output).free,
                        },
                    )
                    print(
                        json.dumps(
                            {"progress_attempts": completed, "arm": arm, "drug": prompt.drug_name}
                        ),
                        flush=True,
                    )
            samples = attempt_samples(attempts, drug=prompt.drug_name)
            lock = output / "locks" / arm / f"{prompt.drug_name}.json"
            immutable_json(
                lock,
                {
                    "manifest_sha256": manifest_sha,
                    "attempt_hashes": attempt_hashes,
                    "samples": samples,
                },
            )
            row_path = output / "rows" / arm / f"{prompt.drug_name}.json"
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if row["lock_sha256"] != physical_sha256(lock):
                    raise ValueError(f"saved linker pilot row lost its sample lock: {row_path}")
            else:
                row = {
                    "arm": arm,
                    "drug": prompt.drug_name,
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
            artifact_hashes[str(row_path.relative_to(output))] = physical_sha256(row_path)
            artifact_hashes[str(lock.relative_to(output))] = physical_sha256(lock)
            artifact_hashes.update(attempt_hashes)
            print(
                json.dumps(
                    {"completed_prompt": prompt.drug_name, "arm": arm, "metrics": row["metrics"]}
                ),
                flush=True,
            )
    summary = summarize(rows, contract)
    summary.update(
        {
            "contract_payload_sha256": identity(contract),
            "manifest_sha256": manifest_sha,
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "execution_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"completed": True, "by_arm": summary["by_arm"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, contract_hash = load_contract()
    versions, prompts = preflight(contract)
    manifest = {
        "schema": "fragment_linker_broaden_pilot_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "source_revision": contract["source_revision"],
        "score_blind_cell_support": cell_support_summary(prompts),
    }
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("linker pilot output differs from frozen contract")
    path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(f"linker pilot output namespace already exists: {output}")
        _atomic_json(path, manifest)
        print(json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(path)}))
        return
    if not path.is_file() or json.loads(path.read_text()) != manifest:
        raise ValueError("linker pilot launch manifest missing or changed")
    if (output / "summary.json").exists():
        raise FileExistsError("linker pilot already complete")
    run(contract, path, output, prompts)


if __name__ == "__main__":
    main()
