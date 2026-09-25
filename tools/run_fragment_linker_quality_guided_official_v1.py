"""Fresh three-seed evaluation of explicitly QED/SA-guided linker panel selection."""

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
from audit_fragment_training_linker import CATALOG_SHA
from rdkit import Chem, RDLogger, rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_linker_official_v1 import attempt_samples
from run_fragment_training_linker_metric_pilot import immutable_json
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog, sample_linker_panel
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "fragment-linker-novelty-v1"
CONTRACT = ROOT / "configs/fragment_linker_quality_guided_official_v1.json"
METRICS = ("quality", "uniqueness", "diversity", "validity")


def identity(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def material_path(name: str) -> Path:
    if name.startswith("local/"):
        return ROOT / name.removeprefix("local/")
    if name.startswith("source/"):
        return SOURCE / name.removeprefix("source/")
    if name.startswith("checkpoint/"):
        return Path("/Users/rmaganti/compose_fragment_ckpt") / name.removeprefix("checkpoint/")
    raise ValueError(f"invalid linker material locator: {name}")


def load_contract() -> tuple[dict, str]:
    envelope = json.loads(CONTRACT.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid quality-guided linker contract: {CONTRACT}")
    contract = envelope["payload"]
    if identity(contract) != envelope["payload_sha256"]:
        raise ValueError("quality-guided linker contract hash mismatch")
    if (
        contract["schema"] != "fragment_linker_quality_guided_official_v1"
        or contract["task"] != "linker_design"
        or contract["selector"] != "unseen_quality_then_unseen_then_full_native_softmax"
        or contract["seeds"] != [9, 10, 11]
        or contract["attempts_per_prompt_seed"] != 100
        or contract["candidate_draws_per_attempt"] != 8
        or contract["output_dir"] != "diagnostics/fragment_linker_quality_guided_official_v1"
        or contract["minimum_free_bytes"] != 5368709120
        or contract["property_thresholds"] != {"qed_ge": 0.6, "sa_le": 4.0}
        or Path(contract["checkpoint_path"]) != material_path("checkpoint/ringcore_a7546e2_best.pt")
    ):
        raise ValueError("quality-guided linker scientific envelope changed")
    return contract, envelope["payload_sha256"]


def preflight(contract: dict) -> tuple[tuple, dict]:
    if SOURCE.name != "fragment-linker-novelty-v1":
        raise ValueError("frozen linker artifact source changed")
    for name, expected in contract["material_sha256"].items():
        path = material_path(name)
        if physical_sha256(path) != expected:
            raise ValueError(f"quality-guided linker material changed: {path}")
    prompt_path = SOURCE / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    prompts = tuple(
        p for p in load_genmol_prompts(prompt_path) if p.task is FragmentTask.LINKER_DESIGN
    )
    if [p.drug_name for p in prompts] != contract["drugs"]:
        raise ValueError("quality-guided linker prompt population or order changed")
    replay_path = ROOT / "diagnostics/fragment_linker_quality_guided_replay_v1/result.json"
    if physical_sha256(replay_path) != contract["development_replay_sha256"]:
        raise ValueError("quality-guided linker development replay changed")
    replay = json.loads(replay_path.read_text())
    if replay["development_screen_passed"] is not True:
        raise ValueError("quality-guided linker selector failed its frozen development screen")
    version = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    if version != contract["versions"]:
        raise ValueError(f"quality-guided linker software mismatch: {version}")
    # The official evaluator remains in the immutable sibling cache. The
    # local benchmark adapter reads an identical, hash-verified cache link.
    from fetch_official_fragment_evaluator import verify_only

    verify_only(cache_dir=SOURCE / ".official_eval_cache")
    verify_only(cache_dir=ROOT / ".official_eval_cache")
    return prompts, version


def quality_scorer_factory(cache: dict[str, tuple[float, float]]):
    pkg = ROOT / ".official_eval_cache/pkg"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from in_virtuo_gen.utils.mol import compute_single_property

    def score(smiles: str) -> tuple[float, float]:
        if smiles not in cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or Chem.MolToSmiles(molecule, canonical=True) != smiles:
                raise ValueError(f"property guidance received invalid endpoint: {smiles}")
            values = compute_single_property(smiles)
            if values is None or len(values) != 2:
                raise ValueError(f"pinned QED/SA scorer failed for endpoint: {smiles}")
            sa, qed = float(values[0]), float(values[1])
            if not np.isfinite(sa) or not np.isfinite(qed):
                raise ValueError(f"pinned QED/SA scorer returned nonfinite value: {smiles}")
            cache[smiles] = (sa, qed)
        return cache[smiles]

    return score


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
        raise ValueError(f"quality-guided linker seed {seed} is incomplete or unfaithful")
    return {
        "seed": seed,
        "outputs": sum(row["outputs"] for row in rows),
        "valid_connected_outputs": sum(row["valid_connected_outputs"] for row in rows),
        "exact_core_path_fidelity_outputs": sum(
            row["exact_core_path_fidelity_outputs"] for row in rows
        ),
        "property_evaluations": sum(row["property_evaluations"] for row in rows),
        "property_candidate_inspections": sum(
            row["property_candidate_inspections"] for row in rows
        ),
        **{metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in METRICS},
    }


def summarize(output: Path, contract: dict, manifest_sha: str) -> dict:
    per_seed, row_hashes = [], {}
    for seed in contract["seeds"]:
        rows = []
        for drug in contract["drugs"]:
            path = output / f"seed{seed}/rows/{drug}.json"
            row = json.loads(path.read_text())
            lock = output / f"seed{seed}/locks/{drug}.json"
            if row["manifest_sha256"] != manifest_sha or row["lock_sha256"] != physical_sha256(
                lock
            ):
                raise ValueError(f"quality-guided linker row changed lock or manifest: {path}")
            rows.append(row)
            row_hashes[str(path.relative_to(output))] = physical_sha256(path)
        per_seed.append(seed_summary(rows, seed, contract["drugs"]))
    return {
        "schema": "fragment_linker_quality_guided_official_result_v1",
        "role": "fresh-seed, property-guided linker evaluation; distinct from unguided fragment results",
        "attempts": 3000,
        "offered_candidate_draws": 24000,
        "outputs": sum(row["outputs"] for row in per_seed),
        "property_evaluations": sum(row["property_evaluations"] for row in per_seed),
        "property_candidate_inspections": sum(
            row["property_candidate_inspections"] for row in per_seed
        ),
        "per_seed": per_seed,
        "official_mean": {
            metric: float(np.mean([row[metric] for row in per_seed])) for metric in METRICS
        },
        "official_sd": {
            metric: float(np.std([row[metric] for row in per_seed], ddof=1)) for metric in METRICS
        },
        "morphing": "same generated linker outputs only if benchmark convention is retained",
        "manifest_sha256": manifest_sha,
        "row_sha256": row_hashes,
        "external_oracle_calls": 0,
    }


def run_seed(output: Path, seed: int, prompts: tuple, contract: dict, manifest_sha: str) -> None:
    if (output / f"seed{seed}/complete.json").exists():
        raise FileExistsError(f"quality-guided linker seed already complete: {seed}")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(Path(contract["checkpoint_path"]))
    catalog = load_linker_catalog(
        SOURCE / "diagnostics/fragment_training_region_catalog_v1/catalog.json",
        expected_sha256=CATALOG_SHA,
    )
    prior = JointCompletionPrior.from_dict(
        json.loads(
            (SOURCE / "diagnostics/fragment_joint_completion_prior_v1/prior.json").read_text()
        )
    )
    property_cache: dict[str, tuple[float, float]] = {}
    scorer = quality_scorer_factory(property_cache)
    rows = []
    for prompt in prompts:
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, seed))
        emitted: set[str] = set()
        attempts, attempt_hashes = [], {}
        for index in range(100):
            stem = f"{prompt.drug_name}_{index:03d}"
            path = output / f"seed{seed}/attempts/{stem}.json"
            start = output / f"seed{seed}/starts/{stem}.json"
            before = {
                "seed": seed,
                "drug": prompt.drug_name,
                "attempt_index": index,
                "rng_state_before": rng.bit_generator.state,
                "manifest_sha256": manifest_sha,
            }
            previous_property_count = len(property_cache)
            if path.exists():
                if not start.exists() or json.loads(start.read_text()) != before:
                    raise ValueError(f"completed guided linker attempt lost start identity: {path}")
                receipt = json.loads(path.read_text())
                restore_completed_attempt(
                    receipt, drug=prompt.drug_name, attempt_index=index, rng=rng
                )
                if receipt["panel"]["qed_sa_guidance"] is not True:
                    raise ValueError(f"resumed attempt lacks declared property guidance: {path}")
                for offer in receipt["panel"]["offered"]:
                    if offer["status"] == "model_supported":
                        scorer(offer["endpoint"])
                if len(property_cache) - previous_property_count != receipt["property_evaluations"]:
                    raise ValueError(f"resumed property-evaluation accounting changed: {path}")
                selection = receipt["panel"]["selection"]
                if selection is not None:
                    sa, qed = scorer(receipt["panel"]["selected_smiles"])
                    if (
                        abs(sa - selection["selected_sa"]) > 1e-12
                        or abs(qed - selection["selected_qed"]) > 1e-12
                    ):
                        raise ValueError(f"resumed selected property values changed: {path}")
            else:
                if start.exists():
                    raise RuntimeError(f"started guided linker attempt cannot be redrawn: {start}")
                if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
                    raise RuntimeError("guided linker benchmark stopped at frozen disk-free floor")
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
                    archive_selection="quality_priority",
                    quality_scorer=scorer,
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
                        f"invalid/nonfaithful guided linker output: {seed}/{prompt.drug_name}/{index}"
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
                    "property_evaluations": len(property_cache) - previous_property_count,
                    "property_candidate_inspections": panel.receipt["model_supported_count"],
                }
                _atomic_json(path, receipt)
            attempts.append(receipt)
            selected_smiles = receipt["panel"]["selected_smiles"]
            if selected_smiles is not None:
                emitted.add(selected_smiles)
            attempt_hashes[str(path.relative_to(output / f"seed{seed}"))] = physical_sha256(path)
            if (index + 1) % 10 == 0:
                _atomic_json(
                    output / f"seed{seed}/progress.json",
                    {
                        "schema": "fragment_linker_quality_guided_progress_v1",
                        "seed": seed,
                        "drug": prompt.drug_name,
                        "completed_attempts_in_prompt": index + 1,
                        "completed_prompt_rows": len(rows),
                    },
                )
        samples = attempt_samples(attempts, drug=prompt.drug_name)
        lock = output / f"seed{seed}/locks/{prompt.drug_name}.json"
        immutable_json(
            lock,
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
            "outputs": sum(record["panel"]["output_count"] for record in attempts),
            "valid_connected_outputs": sum(
                record["selected_valid_connected"] for record in attempts
            ),
            "exact_core_path_fidelity_outputs": sum(
                record["selected_exact_core_path_fidelity"] for record in attempts
            ),
            "property_evaluations": sum(record["property_evaluations"] for record in attempts),
            "property_candidate_inspections": sum(
                record["property_candidate_inspections"] for record in attempts
            ),
            "metrics": official_prompt_metrics(samples, expected_samples=100),
            "lock_sha256": physical_sha256(lock),
            "manifest_sha256": manifest_sha,
        }
        immutable_json(output / f"seed{seed}/rows/{prompt.drug_name}.json", row)
        rows.append(row)
        print(
            json.dumps({"seed": seed, "drug": prompt.drug_name, "metrics": row["metrics"]}),
            flush=True,
        )
    completion = seed_summary(rows, seed, [prompt.drug_name for prompt in prompts])
    completion["manifest_sha256"] = manifest_sha
    immutable_json(output / f"seed{seed}/complete.json", completion)
    print(json.dumps({"seed_completed": seed, "metrics": completion}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run-seed", "summarize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.warning")
    contract, payload_sha = load_contract()
    prompts, versions = preflight(contract)
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("guided linker output differs from frozen contract")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema": "fragment_linker_quality_guided_official_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": payload_sha,
        "code_revision": revision,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32 model; float64 proposal and property probabilities",
        "hardware": platform.machine(),
        "quality_selection": "QED>=0.6 and SA<=4 before return; eight-offer fixed panel",
        "comparison_status": "property-guided; not an unguided fragment-generation baseline",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if args.seed is not None or output.exists():
            raise ValueError("prepare requires no seed and a fresh output directory")
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "payload_sha256": payload_sha}))
        return
    if not manifest_path.is_file():
        raise ValueError("prepare the guided linker manifest before running")
    prepared = json.loads(manifest_path.read_text())
    if {k: v for k, v in prepared.items() if k != "code_revision"} != {
        k: v for k, v in manifest.items() if k != "code_revision"
    }:
        raise ValueError("guided linker manifest or source materials changed")
    manifest_sha = physical_sha256(manifest_path)
    if args.phase == "summarize":
        if args.seed is not None:
            raise ValueError("summarize takes no seed")
        result = summarize(output, contract, manifest_sha)
        immutable_json(output / "summary.json", result)
        print(json.dumps({"completed": True, "official_mean": result["official_mean"]}))
        return
    if args.seed not in contract["seeds"]:
        raise ValueError("run-seed needs a declared seed")
    run_seed(output, args.seed, prompts, contract, manifest_sha)


if __name__ == "__main__":
    main()
