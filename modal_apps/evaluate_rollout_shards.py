"""Distributed, restart-safe ancestral rollout evaluation on Modal.

This app exists because marked-rewrite sampling is sequential within one
trajectory but embarrassingly parallel across trajectories.  Each CPU
container writes one independently seeded shard to the artifact volume; a
single reducer then merges exactly the requested number of attempts and runs
the same fixed-reference molecular metrics used by the standard evaluator.

Example
-------
modal run --detach modal_apps/evaluate_rollout_shards.py \
    --run-label compose-v4-stage3-best6250-eval2000 \
    --source-run-label compose-v4-stage3-full-ring-hierarchical-v1 \
    --checkpoint-name checkpoint.recovery.pt \
    --samples 2000 --shards 80
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
if REMOTE_ROOT.is_dir():
    sys.path.insert(0, str(REMOTE_ROOT))
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
        "fcd-torch==1.0.7",
    )
    .env(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(REMOTE_ROOT), str(REMOTE_ROOT / "src"))
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
)

app = modal.App("compose-v4-sharded-rollout-evaluation")
guacamol_volume = modal.Volume.from_name("guacamol", create_if_missing=False)
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)


def _basename(value: str, label: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{label} must be a nonempty basename")
    return value


def _partition_sample_counts(samples: int, shards: int) -> tuple[int, ...]:
    if samples <= 0 or shards <= 0:
        raise ValueError("samples and shards must be positive")
    if shards > samples:
        raise ValueError("shards cannot exceed samples")
    quotient, remainder = divmod(int(samples), int(shards))
    return tuple(
        quotient + int(index < remainder)
        for index in range(shards)
    )


def _derive_shard_seed(base_seed: int, shard_index: int) -> int:
    if shard_index < 0:
        raise ValueError("shard index must be nonnegative")
    sequence = np.random.SeedSequence((int(base_seed), int(shard_index)))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _shard_filename(shard_index: int, shard_count: int) -> str:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index is outside the declared shard count")
    return f"shard-{shard_index:05d}-of-{shard_count:05d}.pt"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _expected_shard_metadata(
    *,
    source_run_label: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    shard_index: int,
    shard_count: int,
    samples: int,
    base_seed: int,
) -> dict[str, object]:
    return {
        "format": "compose_v4_rollout_shard_v1",
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "samples": int(samples),
        "base_seed": int(base_seed),
        "shard_seed": _derive_shard_seed(base_seed, shard_index),
    }


def _validate_shard_payload(
    payload: object,
    expected: dict[str, object],
) -> tuple[object, ...]:
    if not isinstance(payload, dict):
        raise ValueError("rollout shard payload must be a dictionary")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"rollout shard metadata mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    rollouts = payload.get("rollouts")
    if not isinstance(rollouts, (tuple, list)):
        raise ValueError("rollout shard lacks a rollout sequence")
    if len(rollouts) != int(expected["samples"]):
        raise ValueError("rollout shard has the wrong number of attempts")
    return tuple(rollouts)


def _merge_shard_payloads(
    payloads: Iterable[object],
    expected_by_index: tuple[dict[str, object], ...],
) -> tuple[object, ...]:
    materialized = list(payloads)
    if len(materialized) != len(expected_by_index):
        raise ValueError("one payload is required for every rollout shard")
    indexed: dict[int, object] = {}
    for payload in materialized:
        if not isinstance(payload, dict) or "shard_index" not in payload:
            raise ValueError("rollout shard does not declare its index")
        index = int(payload["shard_index"])
        if index in indexed:
            raise ValueError(f"duplicate rollout shard index: {index}")
        indexed[index] = payload
    merged: list[object] = []
    for index, expected in enumerate(expected_by_index):
        if index not in indexed:
            raise ValueError(f"missing rollout shard index: {index}")
        merged.extend(_validate_shard_payload(indexed[index], expected))
    return tuple(merged)


@app.function(
    image=image,
    cpu=4.0,
    memory=24576,
    timeout=6 * 3600,
    volumes={"/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def sample_rollout_shard(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str,
    shard_index: int,
    shard_count: int,
    samples: int,
    base_seed: int,
    workers: int,
) -> dict[str, object]:
    """Sample or reuse one deterministic shard without loading any teacher paths."""

    import torch

    from compose_v4.experiments.parallel_tracelet_sampling import (
        sample_tracelet_ancestral_many,
    )
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    run_label = _basename(run_label, "run label")
    source_run_label = _basename(source_run_label, "source run label")
    checkpoint_name = _basename(checkpoint_name, "checkpoint name")
    if samples <= 0 or workers <= 0:
        raise ValueError("shard samples and workers must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index is outside the declared shard count")

    torch.set_num_threads(1)
    artifact_volume.reload()
    checkpoint_path = Path("/artifacts") / source_run_label / checkpoint_name
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing source checkpoint: {checkpoint_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    expected = _expected_shard_metadata(
        source_run_label=source_run_label,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha256,
        shard_index=shard_index,
        shard_count=shard_count,
        samples=samples,
        base_seed=base_seed,
    )
    shard_path = (
        Path("/artifacts")
        / run_label
        / "shards"
        / _shard_filename(shard_index, shard_count)
    )
    if shard_path.is_file() and shard_path.stat().st_size > 0:
        cached = torch.load(shard_path, map_location="cpu", weights_only=False)
        _validate_shard_payload(cached, expected)
        return {
            **expected,
            "path": str(shard_path),
            "reused": True,
        }

    model, checkpoint = load_factorized_rollout_checkpoint(checkpoint_path)
    source_prior = checkpoint["tree_source_prior"]
    rollouts = sample_tracelet_ancestral_many(
        model,
        seed=int(expected["shard_seed"]),
        samples=samples,
        workers=min(workers, samples),
        n_slots=40,
        operational_horizon=16.0,
        time_step=0.1,
        max_events=128,
        torch_threads_per_worker=1,
        source_prior=source_prior,
    )
    payload = {
        **expected,
        "selected_validation": checkpoint.get("selected_validation"),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "rollout_state_source": checkpoint.get("rollout_state_source"),
        "n_slots": 40,
        "operational_horizon": 16.0,
        "time_step": 0.1,
        "max_events": 128,
        "source_prior": "carbon_tree",
        "tree_source_prior": source_prior,
        "rollouts": rollouts,
    }
    _atomic_torch_save(payload, shard_path)
    artifact_volume.commit()
    return {
        **expected,
        "path": str(shard_path),
        "reused": False,
        "selected_validation": checkpoint.get("selected_validation"),
    }


def _read_smiles(path: Path) -> tuple[str, ...]:
    return tuple(
        fields[0]
        for line in path.read_text().splitlines()
        if (fields := line.strip().split())
    )


def _matched_cnof_reference(path: Path, *, max_atoms: int) -> tuple[str, ...]:
    from compose_v4.data.cnof import _canonical_cnof_smiles

    accepted: dict[str, None] = {}
    for text in _read_smiles(path):
        canonical = _canonical_cnof_smiles((text, max_atoms))
        if canonical is not None:
            accepted.setdefault(canonical, None)
    return tuple(accepted)


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=4 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def merge_and_evaluate(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str,
    samples: int,
    shards: int,
    base_seed: int,
) -> dict[str, object]:
    """Merge every attempt in shard order and run fixed-reference metrics once."""

    import torch

    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.data.cnof import load_cnof_corpus_split
    from compose_v4.eval.molecular_quality import molecular_quality_report
    from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
    from compose_v4.experiments.cnof_conditional import corpus_rollout_metrics
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    run_label = _basename(run_label, "run label")
    source_run_label = _basename(source_run_label, "source run label")
    checkpoint_name = _basename(checkpoint_name, "checkpoint name")
    sample_counts = _partition_sample_counts(samples, shards)
    artifact_volume.reload()
    guacamol_volume.reload()
    checkpoint_path = Path("/artifacts") / source_run_label / checkpoint_name
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    expected = tuple(
        _expected_shard_metadata(
            source_run_label=source_run_label,
            checkpoint_name=checkpoint_name,
            checkpoint_sha256=checkpoint_sha256,
            shard_index=index,
            shard_count=shards,
            samples=count,
            base_seed=base_seed,
        )
        for index, count in enumerate(sample_counts)
    )
    shard_paths = tuple(
        Path("/artifacts") / run_label / "shards" / _shard_filename(index, shards)
        for index in range(shards)
    )
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} rollout shards: {missing[:5]}")
    payloads = tuple(
        torch.load(path, map_location="cpu", weights_only=False)
        for path in shard_paths
    )
    rollouts = _merge_shard_payloads(payloads, expected)
    if len(rollouts) != samples:
        raise RuntimeError(f"merged {len(rollouts)} rollouts; expected {samples}")

    _, checkpoint = load_factorized_rollout_checkpoint(checkpoint_path)
    source_prior = checkpoint["tree_source_prior"]
    run_dir = Path("/artifacts") / run_label
    merged_path = run_dir / "rollouts.pt"
    _atomic_torch_save(
        {
            "format": "compose_v4_merged_rollouts_v1",
            "rollouts": rollouts,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "selected_validation": checkpoint.get("selected_validation"),
            "base_seed": base_seed,
            "shard_seeds": tuple(item["shard_seed"] for item in expected),
            "shard_sample_counts": sample_counts,
            "n_slots": 40,
            "operational_horizon": 16.0,
            "time_step": 0.1,
            "max_events": 128,
            "source_prior": "carbon_tree",
            "tree_source_prior": source_prior,
        },
        merged_path,
    )
    artifact_volume.commit()

    train_file = Path("/guacamol/guacamol_subset_500000_seed0.smiles")
    reference_file = Path("/guacamol/guacamol_heldout_val_5000_seed0.smiles")
    for required in (train_file, reference_file):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing evaluation data: {required}")
    split = load_cnof_corpus_split(
        train_file,
        train_size=50000,
        validation_size=2000,
        test_size=2000,
        max_atoms=40,
        seed=20260717,
        scan_all=True,
        workers=16,
    )
    generated_smiles = tuple(
        text
        for rollout in rollouts
        if (text := molecular_graph_to_smiles(rollout.final_state)) is not None
    )
    reference_smiles = _read_smiles(reference_file)[:5000]
    matched_reference = _matched_cnof_reference(reference_file, max_atoms=40)
    event_counts = Counter(
        rule
        for rollout in rollouts
        for rule in rollout.event_rules
    )
    report: dict[str, object] = {
        "evaluation_kind": "distributed_checkpoint_rollout_only",
        "teacher_path_cache_loaded": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "rollout_state_source": checkpoint.get("rollout_state_source"),
        "selected_validation": checkpoint.get("selected_validation"),
        "requested_samples": samples,
        "merged_attempts": len(rollouts),
        "generated_nonnull_smiles": len(generated_smiles),
        "generated_smiles": generated_smiles,
        "rollout_event_counts": dict(sorted(event_counts.items())),
        "rollout": corpus_rollout_metrics(
            rollouts,
            train_smiles=split.train,
            reference_smiles=split.test,
        ),
        "generated_ring_taxonomy": (
            ring_taxonomy_report(generated_smiles) if generated_smiles else None
        ),
        "molecular_quality": molecular_quality_report(
            generated_smiles,
            reference_smiles=reference_smiles,
            train_smiles=split.train,
            include_fcd=True,
            fcd_reference_limit=5000,
            fcd_generated_limit=2000,
            fcd_device="cpu",
            seed=20260717,
        ),
        "molecular_quality_cnof_matched": molecular_quality_report(
            generated_smiles,
            reference_smiles=matched_reference,
            train_smiles=split.train,
            include_fcd=True,
            fcd_reference_limit=None,
            fcd_generated_limit=2000,
            fcd_device="cpu",
            seed=20260717,
        ),
        "reference": {
            "full_count": len(reference_smiles),
            "matched_neutral_cnof_max40_count": len(matched_reference),
            "full_sha256": _file_sha256(reference_file),
        },
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "eligible": split.eligible_molecules,
            "scanned_lines": split.scanned_lines,
        },
        "sampler": {
            "target_available": False,
            "beam_search": False,
            "ancestral_ctmc": True,
            "base_seed": base_seed,
            "seed_scheme": "SeedSequence((base_seed, shard_index)) then per-trajectory spawn",
            "shards": shards,
            "shard_seeds": tuple(item["shard_seed"] for item in expected),
            "shard_sample_counts": sample_counts,
            "operational_horizon": 16.0,
            "time_step": 0.1,
            "max_events": 128,
        },
    }
    metrics_path = run_dir / "metrics.json"
    _atomic_json_save(report, metrics_path)
    manifest = {
        "run_label": run_label,
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "samples": samples,
        "shards": shards,
        "base_seed": base_seed,
        "shard_sample_counts": sample_counts,
        "shard_seeds": tuple(item["shard_seed"] for item in expected),
        "teacher_path_cache_loaded": False,
        "all_attempts_retained": True,
    }
    _atomic_json_save(manifest, run_dir / "manifest.json")
    artifact_volume.commit()
    summary = {
        "run_label": run_label,
        "samples": samples,
        "generated_nonnull_smiles": len(generated_smiles),
        "valid_fraction": report["rollout"]["valid_fraction"],
        "full_reference_fcd": report["molecular_quality"]["frechet_chemnet_distance"],
        "cnof_matched_fcd": report["molecular_quality_cnof_matched"]["frechet_chemnet_distance"],
        "metrics": str(metrics_path),
        "rollouts": str(merged_path),
    }
    print(json.dumps({"phase": "distributed_rollout_complete", **summary}, sort_keys=True))
    return summary


@app.local_entrypoint()
def main(
    run_label: str = "compose-v4-sharded-rollout-eval",
    source_run_label: str = "compose-v4-stage3-full-ring-hierarchical-v1",
    checkpoint_name: str = "checkpoint.recovery.pt",
    samples: int = 2000,
    shards: int = 80,
    base_seed: int = 20260724,
    workers_per_shard: int = 4,
    smoke: bool = False,
) -> None:
    run_label = _basename(run_label, "run label")
    source_run_label = _basename(source_run_label, "source run label")
    checkpoint_name = _basename(checkpoint_name, "checkpoint name")
    if workers_per_shard <= 0:
        raise ValueError("workers per shard must be positive")
    if smoke:
        samples = 4
        shards = 2
        workers_per_shard = 2
    counts = _partition_sample_counts(samples, shards)
    arguments = [
        (
            run_label,
            source_run_label,
            checkpoint_name,
            index,
            shards,
            count,
            base_seed,
            workers_per_shard,
        )
        for index, count in enumerate(counts)
    ]
    print(
        json.dumps(
            {
                "phase": "distributed_rollout_start",
                "run_label": run_label,
                "samples": samples,
                "shards": shards,
                "workers_per_shard": workers_per_shard,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    shard_results = list(
        sample_rollout_shard.starmap(
            arguments,
            order_outputs=False,
        )
    )
    print(
        json.dumps(
            {
                "phase": "distributed_rollout_shards_complete",
                "completed": len(shard_results),
                "reused": sum(bool(item.get("reused")) for item in shard_results),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = merge_and_evaluate.remote(
        run_label,
        source_run_label,
        checkpoint_name,
        samples,
        shards,
        base_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
