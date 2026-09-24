"""Matched 10-prompt motif/decorate development gate, with every attempt retained.

This is not a full paper benchmark. It compares the unchanged learned sampler
against one opt-in training-derived constructive lane on the same prompts and
deterministic seeds. The official evaluator sees all chemically committed
endpoints plus explicit no-output placeholders, never task-filtered survivors.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from run_fragment_constrained_suite import prompt_reference_smiles, prompt_rng_seed, run_task

from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    build_prompt_context,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_distance,
    official_prompt_metrics,
)
from compose_v4.benchmark.training_attachment_fragments import (
    AttachmentCatalog,
    atom_context,
    physical_sha256,
    prepare_catalog,
    sample_catalog_completion,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
TASKS = (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION)
METRICS = ("validity", "uniqueness", "quality", "diversity")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _load_catalog(directory: Path):
    manifest_path = directory / "manifest.json"
    catalog_path = directory / "catalog.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "training_attachment_catalog_build_v1":
        raise ValueError(f"not a training attachment catalog: {manifest_path}")
    if physical_sha256(catalog_path) != manifest["catalog_sha256"]:
        raise ValueError(f"catalog SHA-256 mismatch: {catalog_path}")
    catalog = AttachmentCatalog.from_dict(json.loads(catalog_path.read_text()))
    if catalog.source_sha256 != manifest["source_sha256"]:
        raise ValueError("catalog/manifest source identity mismatch")
    if manifest.get("prompt_manifest_sha256") != physical_sha256(PROMPTS):
        raise ValueError("catalog was not split against these benchmark references")
    if tuple(manifest.get("excluded_reference_rows", ())) != catalog.excluded_reference_rows:
        raise ValueError("catalog/manifest reference exclusion ledger mismatch")
    return prepare_catalog(catalog), manifest


def _catalog_row(prompt, runtime, system, samples: int, seed: int) -> dict:
    context = build_prompt_context(prompt)
    rng_seed = prompt_rng_seed(prompt.drug_name, prompt.task.value, seed)
    rng = np.random.default_rng(rng_seed)
    chemical_samples: list[str] = []
    fidelity = 0
    fallback_count = 0
    attempt_records = []
    for attempt_index in range(samples):
        emitted, receipt = sample_catalog_completion(context, runtime, system, rng)
        committed = receipt.committed_smiles
        chemical_samples.append(committed or FAILED_SAMPLE_PLACEHOLDER)
        fidelity += int(emitted is not None)
        fallback_count += receipt.fallback_count
        attempt_records.append(
            {
                "attempt_index": attempt_index,
                "rng_seed": rng_seed,
                "committed_smiles": committed,
                "emitted_smiles": emitted,
                "offered_fragments": receipt.attempted_fragments,
                "accepted_actions": receipt.accepted_actions,
                "fallback_count": receipt.fallback_count,
                "failure_reason": receipt.failure_reason,
            }
        )
    metrics = official_prompt_metrics(chemical_samples, expected_samples=samples)
    distance = official_distance(chemical_samples, prompt_reference_smiles(prompt))
    metrics["distance"] = None if np.isnan(distance) else distance
    if any(
        record["committed_smiles"] is not None
        and Chem.MolFromSmiles(record["committed_smiles"]) is None
        for record in attempt_records
    ):
        raise AssertionError("catalog lane committed a chemically invalid endpoint")
    independent_fidelity = sum(
        check_fragment_constraint(prompt, row["committed_smiles"]).satisfied
        for row in attempt_records
        if row["committed_smiles"] is not None
    )
    if fidelity != independent_fidelity:
        raise AssertionError("catalog task-fidelity census disagrees with final constraint")
    return {
        "schema": "fragment_attachment_library_pilot_shard_v1",
        "arm": "catalog",
        "task": prompt.task.value,
        "drug": prompt.drug_name,
        "seed": seed,
        "rng_seed": rng_seed,
        "attempts": samples,
        "metrics": metrics,
        "task_fidelity_count": fidelity,
        "fallback_count": fallback_count,
        "attempt_records": attempt_records,
    }


def _baseline_row(prompt, model, system, samples: int, seed: int) -> dict:
    config = SamplerConfig()
    control = AttachmentControlConfig(
        enabled=True,
        condition_initial_locked_family=True,
        hard_lock_effective_chemistry=True,
    )
    result = run_task(
        model,
        system,
        (prompt,),
        prompt.task,
        seeds=1,
        samples=samples,
        config=config,
        control=control,
        verbose=False,
        seed_list=[seed],
        drugs=[prompt.drug_name],
    )
    [row] = result["per_drug"][prompt.drug_name]
    if row["attempts"] != samples or len(row["attempt_records"]) != samples:
        raise AssertionError("baseline did not persist every attempted generation")
    return {
        "schema": "fragment_attachment_library_pilot_shard_v1",
        "arm": "baseline",
        "task": prompt.task.value,
        "drug": prompt.drug_name,
        "seed": seed,
        "rng_seed": row["rng_seed"],
        "attempts": samples,
        "metrics": row["official"],
        "task_fidelity_count": row["emitted_nonempty"],
        "fallback_count": 0,
        "attempt_records": row["attempt_records"],
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.samples != 20 or args.seed != 0:
        raise ValueError("first predeclared pilot is exactly 20 attempts/prompt at seed 0")
    if physical_sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError(f"checkpoint hash mismatch: {args.checkpoint}")
    runtime, catalog_manifest = _load_catalog(args.catalog_dir)
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, checkpoint_meta = load_factorized_rollout_checkpoint(args.checkpoint)
    prompts = tuple(prompt for prompt in load_genmol_prompts(PROMPTS) if prompt.task in TASKS)
    if len(prompts) != 20:
        raise AssertionError("pilot requires ten prompts per task")
    library_coverage = {}
    for prompt in prompts:
        context = build_prompt_context(prompt)
        core = Chem.MolFromSmiles(context.start_smiles)
        if core is None or context.attachment is None:
            raise ValueError(f"unusable benchmark attachment context: {prompt.drug_name}")
        sites = []
        for site, count in context.attachment.requirements:
            key = atom_context(core.GetAtomWithIdx(site))
            compatible = tuple(
                entry
                for entry in runtime.by_context.get(key, ())
                if entry.heavy_atoms <= 40 - context.start_state.n_real_atoms
            )
            sites.append(
                {
                    "site": site,
                    "required_count": count,
                    "context": key,
                    "compatible_distinct_fragments": len(compatible),
                    "compatible_training_occurrences": sum(
                        entry.occurrences for entry in compatible
                    ),
                }
            )
        library_coverage[f"{prompt.task.value}|{prompt.drug_name}"] = sites
    output_dir = args.output_dir.resolve()
    manifest = {
        "schema": "fragment_attachment_library_pilot_v1",
        "scientific_role": "matched development pilot, not a full benchmark",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prompt_manifest": str(PROMPTS),
        "prompt_sha256": physical_sha256(PROMPTS),
        "catalog_dir": str(args.catalog_dir.resolve()),
        "catalog_sha256": catalog_manifest["catalog_sha256"],
        "training_source_sha256": catalog_manifest["source_sha256"],
        "library_coverage": library_coverage,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_steps": checkpoint_meta.get("completed_steps"),
        "pilot_code_sha256": physical_sha256(Path(__file__).resolve()),
        "catalog_code_sha256": physical_sha256(
            ROOT / "src/compose_v4/benchmark/training_attachment_fragments.py"
        ),
        "evaluator_sha256": physical_sha256(
            ROOT / "src/compose_v4/benchmark/fragment_official_metrics.py"
        ),
        "samples_per_prompt": args.samples,
        "seed": args.seed,
        "seed_derivation": "BLAKE2b drug|task|seed, 8-byte digest modulo 2^32",
        "baseline_sampler": asdict(SamplerConfig()),
        "baseline_attachment_control": asdict(
            AttachmentControlConfig(
                enabled=True,
                condition_initial_locked_family=True,
                hard_lock_effective_chemistry=True,
            )
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "metric_denominator": "20 attempted generations per prompt, no task-fidelity censoring",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("pilot manifest differs from existing partial run")
    else:
        _atomic_json(manifest_path, manifest)
    system = de_novo_rewrite_system()
    rows = []
    for prompt in prompts:
        for arm in ("baseline", "catalog"):
            shard = output_dir / "shards" / f"{prompt.task.value}__{prompt.drug_name}__{arm}.json"
            if shard.exists():
                row = json.loads(shard.read_text())
            else:
                row = (
                    _baseline_row(prompt, model, system, args.samples, args.seed)
                    if arm == "baseline"
                    else _catalog_row(prompt, runtime, system, args.samples, args.seed)
                )
                _atomic_json(shard, row)
            if row["attempts"] != args.samples or len(row["attempt_records"]) != args.samples:
                raise RuntimeError(f"incomplete pilot shard: {shard}")
            rows.append(row)
            print(
                f"{prompt.task.value} {prompt.drug_name} {arm}: "
                f"Q={row['metrics']['quality']:.2f} "
                f"U={row['metrics']['uniqueness']:.2f} "
                f"D={row['metrics']['diversity']:.3f} "
                f"V={row['metrics']['validity']:.2f}",
                flush=True,
            )
    summary = {}
    for task in TASKS:
        task_summary = {}
        for arm in ("baseline", "catalog"):
            selected = [row for row in rows if row["task"] == task.value and row["arm"] == arm]
            task_summary[arm] = {
                key: float(np.mean([row["metrics"][key] for row in selected])) for key in METRICS
            }
            task_summary[arm]["attempts"] = sum(row["attempts"] for row in selected)
            task_summary[arm]["fidelity_count"] = sum(
                row["task_fidelity_count"] for row in selected
            )
            task_summary[arm]["fallback_count"] = sum(row["fallback_count"] for row in selected)
        summary[task.value] = task_summary
    _atomic_json(output_dir / "summary.json", {"manifest": manifest, "summary": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
