"""Train/evaluate from-scratch, empirical, or prior-tilted tracelet generators."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Executor
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.experiments.cnof_conditional import PathRecord, corpus_rollout_metrics
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    build_tree_transport_path_records,
    sample_tracelet_conditional_batch,
    tracelet_fiber_executor,
    tracelet_path_executor,
    tracelet_conditional_metrics,
    train_tracelet_conditional_model,
)
from compose_v4.experiments.factorized_mark_conditional import (
    factorized_mark_metrics,
    sample_factorized_mark_batch,
    train_factorized_mark_model,
)
from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)
from compose_v4.experiments.tracelet_corpus_marginal import (
    fit_tracelet_corpus_marginal_rate_model,
)
from compose_v4.experiments.tracelet_prior_tilted import (
    PriorTiltedTraceletRateModel,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    SparseBinaryRows,
)
from compose_v4.model.device import resolve_torch_device
from compose_v4.rewrite.typed_ring_catalog import (
    RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
    TypedRingCatalog,
    build_typed_ring_catalog,
    build_typed_ring_catalog_from_paths,
)
from compose_v4.rewrite.ring_system_fiber import structured_ring_trace_supported
from compose_v4.rewrite.kernel import canonical_state_key


TRANSPORT_SUPPORT_PROJECTION_VERSION = 3
EVALUATION_BATCH_CACHE_FORMAT_VERSION = 1


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Replace a checkpoint only after its complete serialization succeeds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_shared_torch_save(payload: object, path: Path) -> None:
    """Atomically publish a content-addressed cache under concurrent runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recovery_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.recovery{checkpoint.suffix}")


def _best_so_far_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.best_so_far{checkpoint.suffix}")


def _validation_baseline_for_run(
    observed_validation: dict[str, float],
    resume_state: dict[str, object] | None,
) -> dict[str, float]:
    """Preserve the original step-zero baseline across exact resumptions."""

    if resume_state is None:
        return dict(observed_validation)
    baseline = resume_state.get("initial_validation")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("recovery checkpoint lacks the original initial_validation")
    try:
        return {str(key): float(value) for key, value in baseline.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("recovery initial_validation must contain numeric metrics") from error


def _path_cache_fingerprint(signature: dict[str, object]) -> str:
    """Return a stable identifier without copying the full split into every shard."""

    serialized = json.dumps(
        signature,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _evaluation_batch_cache_signature(
    path_cache_signature: dict[str, object],
    *,
    training_backend: str,
    seed: int,
    validation_examples: int,
    test_examples: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    bond_representation: str,
    ring_electronic_mode: str,
) -> dict[str, object]:
    """Identify every scientific choice that fixes the evaluation tensors."""

    return {
        "format_version": EVALUATION_BATCH_CACHE_FORMAT_VERSION,
        "path_cache_fingerprint": _path_cache_fingerprint(path_cache_signature),
        "training_backend": str(training_backend),
        "seed": int(seed),
        "validation_examples": int(validation_examples),
        "test_examples": int(test_examples),
        "late_time_fraction": float(late_time_fraction),
        "operational_horizon": float(operational_horizon),
        "progress_stratification_fraction": float(progress_stratification_fraction),
        "bond_representation": str(bond_representation),
        "ring_electronic_mode": str(ring_electronic_mode),
    }


def _evaluation_batch_cache_path(
    cache_dir: Path,
    signature: dict[str, object],
) -> Path:
    fingerprint = _path_cache_fingerprint(signature)
    return cache_dir / f"evaluation-batches-v{EVALUATION_BATCH_CACHE_FORMAT_VERSION}-{fingerprint}.pt"


def _load_evaluation_batch_cache(
    path: Path,
    *,
    signature: dict[str, object],
    validation_examples: int,
    test_examples: int,
) -> tuple[object, object]:
    """Load a complete cache and reject stale or partially written payloads."""

    payload = torch.load(path, weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise ValueError(f"evaluation batch cache/config mismatch: {path}")
    validation_batch = payload.get("validation_batch")
    test_batch = payload.get("test_batch")
    observed_sizes = (
        getattr(validation_batch, "batch_size", None),
        getattr(test_batch, "batch_size", None),
    )
    expected_sizes = (int(validation_examples), int(test_examples))
    if observed_sizes != expected_sizes:
        raise ValueError(
            "evaluation batch cache has invalid partition sizes: "
            f"{observed_sizes} != {expected_sizes}"
        )
    # Cache v1 originally stored exact ring support densely.  New code keeps
    # the same scientific signature and upgrades those rows to CSR in memory,
    # so an already published production cache remains reusable without a
    # chemistry rebuild.
    for batch in (validation_batch, test_batch):
        if not isinstance(batch, FactorizedMarkBatch):
            continue
        if not hasattr(batch, "ring_grow_support_sparse"):
            object.__setattr__(batch, "ring_grow_support_sparse", None)
        dense_support = getattr(batch, "ring_grow_support_mask", None)
        if dense_support is not None:
            object.__setattr__(
                batch,
                "ring_grow_support_sparse",
                SparseBinaryRows.from_dense(dense_support),
            )
            object.__setattr__(batch, "ring_grow_support_mask", None)
    return validation_batch, test_batch


def _legacy_evaluation_signature_matches(
    candidate: dict[str, object],
    expected: dict[str, object],
) -> bool:
    """Accept an eager cache only when every scientific coupling field agrees.

    Format 3 eager caches predate compact checkpoint and shard-layout fields.
    They are safe for checkpoint evaluation because no path is compiled or
    mutated, but they must never be silently reused for training.
    """

    required_fields = {
        "seed",
        "max_atoms",
        "ring_proposals",
        "max_cycle_templates",
        "max_attach_templates",
        "max_ear_templates",
        "source_prior",
        "tree_couplings_per_target",
        "tree_transport",
    }
    if not required_fields <= candidate.keys():
        return False
    if any(candidate[key] != expected[key] for key in required_fields):
        return False
    # Split payloads changed representation across eager-cache versions and
    # can omit targets rejected by the typed-ring support filter.  They are
    # therefore audited below from the canonical endpoint keys and exact
    # coupling multiplicities rather than compared as serialized containers.
    optional_scientific_fields = {"tree_size_prior"}
    return all(
        candidate[key] == expected[key] for key in optional_scientific_fields if key in candidate
    )


def _path_cache_signature_mismatch(
    candidate: object,
    expected: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Summarize scalar metadata differences without logging corpus payloads."""

    if not isinstance(candidate, dict):
        return {
            "signature": {
                "candidate": type(candidate).__name__,
                "expected": "dict",
            }
        }
    ignored = {"train_smiles", "validation_smiles", "test_smiles"}
    differences: dict[str, dict[str, object]] = {}
    for key in sorted((candidate.keys() | expected.keys()) - ignored):
        candidate_value = candidate.get(key, "<missing>")
        expected_value = expected.get(key, "<missing>")
        if candidate_value != expected_value:
            differences[key] = {
                "candidate": candidate_value,
                "expected": expected_value,
            }
    return differences


def _validate_legacy_evaluation_partition(
    records: tuple[PathRecord, ...],
    split_smiles: tuple[str, ...],
    *,
    n_slots: int,
    couplings_per_target: int,
    partition: str,
) -> None:
    """Verify endpoint identity and coupling multiplicity in an eager cache."""

    allowed = {
        canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph(text), n_slots))
        for text in split_smiles
    }
    counts = Counter(record.target_key for record in records)
    unexpected = set(counts) - allowed
    wrong_multiplicity = {
        key: count for key, count in counts.items() if count != couplings_per_target
    }
    minimum_supported = int(0.95 * len(allowed))
    if unexpected or wrong_multiplicity or len(counts) < minimum_supported:
        raise ValueError(
            "legacy evaluation cache endpoint audit failed for "
            f"{partition}: unexpected={len(unexpected)}, "
            f"wrong_multiplicity={len(wrong_multiplicity)}, "
            f"supported={len(counts)}/{len(allowed)}"
        )


def _sharded_path_cache_root(path: Path) -> Path:
    return path.with_name(f"{path.name}.shards")


def _sharded_path_manifest(path: Path) -> Path:
    return _sharded_path_cache_root(path) / "manifest.pt"


def _load_sharded_path_manifest(
    path: Path,
    *,
    signature: dict[str, object],
) -> dict[str, object] | None:
    manifest_path = _sharded_path_manifest(path)
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return None
    payload = torch.load(manifest_path, weights_only=False)
    if payload.get("signature") != signature:
        raise ValueError(f"compiled path shard/config mismatch: {manifest_path}")
    return payload


def _save_sharded_path_manifest(
    path: Path,
    *,
    signature: dict[str, object],
    ring_catalog: TypedRingCatalog | None,
    supported_smiles: dict[str, tuple[str, ...]],
    complete: bool,
    supported_record_keys: dict[str, tuple[tuple[object, object], ...]] | None = None,
) -> None:
    _atomic_torch_save(
        {
            "signature": signature,
            "signature_fingerprint": _path_cache_fingerprint(signature),
            "ring_catalog": ring_catalog,
            "supported_smiles": supported_smiles,
            "supported_record_keys": supported_record_keys,
            "complete": bool(complete),
            # A complete manifest is written only after the executable
            # transport paths have been projected through the final ring
            # catalog.  Training may therefore reuse this projection instead
            # of repeating graph-isomorphism support checks on every launch.
            "transport_support_projection_version": (
                TRANSPORT_SUPPORT_PROJECTION_VERSION if complete else None
            ),
        },
        _sharded_path_manifest(path),
    )


def _filter_records_by_finalized_manifest(
    records: tuple[PathRecord, ...],
    supported_smiles: tuple[str, ...],
    *,
    n_slots: int,
    couplings_per_target: int,
    partition: str,
    supported_record_keys: tuple[tuple[object, object], ...] | None = None,
) -> tuple[PathRecord, ...]:
    """Recover the audited support projection from a finalized manifest.

    Version-2 manifests store exact source/target coupling identities because
    two independent source trees for one target can differ in catalog support.
    The target-only fallback exists for incomplete/legacy cache diagnostics;
    production training requires the versioned coupling keys.
    """

    if supported_record_keys is None:
        supported_keys = tuple(
            canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph(text), n_slots))
            for text in supported_smiles
        )
        expected_counts = Counter(supported_keys)
        for key in expected_counts:
            expected_counts[key] *= couplings_per_target

        def record_key(record: PathRecord) -> object:
            return record.target_key

    else:
        expected_counts = Counter(supported_record_keys)
        record_key = _path_record_support_key
    retained = tuple(record for record in records if record_key(record) in expected_counts)
    actual_counts = Counter(record_key(record) for record in retained)
    missing = set(expected_counts) - set(actual_counts)
    wrong_multiplicity = {
        key: (actual_counts.get(key, 0), expected)
        for key, expected in expected_counts.items()
        if actual_counts.get(key, 0) != expected
    }
    if missing or wrong_multiplicity:
        raise ValueError(
            "finalized manifest endpoint audit failed for "
            f"{partition}: missing={len(missing)}, "
            f"wrong_multiplicity={len(wrong_multiplicity)}"
        )
    return retained


def _path_record_support_key(record: PathRecord) -> tuple[object, object]:
    """Identify one deterministic source-to-target coupling in a path shard."""

    return (
        record.target_key,
        canonical_state_key(record.path.trace.source),
    )


def _tree_transport_shard_seed(base_seed: int, shard_index: int) -> int:
    state = np.random.SeedSequence([int(base_seed), int(shard_index)]).generate_state(
        1,
        dtype=np.uint32,
    )
    return int(state[0])


def _compile_tracelet_proposal_partition_shards(
    *,
    partition: str,
    smiles: tuple[str, ...],
    args: argparse.Namespace,
    signature: dict[str, object],
    executor: Executor | None,
) -> tuple[PathRecord, ...]:
    """Load or atomically compile ring-catalog proposal traces."""

    cache_root = _sharded_path_cache_root(args.path_cache) / "proposal"
    fingerprint = _path_cache_fingerprint(signature)
    records: list[PathRecord] = []
    total_shards = (len(smiles) + args.path_shard_size - 1) // args.path_shard_size
    for shard_index, start in enumerate(range(0, len(smiles), args.path_shard_size)):
        stop = min(start + args.path_shard_size, len(smiles))
        shard_smiles = smiles[start:stop]
        shard_path = cache_root / f"{partition}-{shard_index:05d}.pt"
        expected = {
            "signature_fingerprint": fingerprint,
            "partition": partition,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "target_count": len(shard_smiles),
        }
        if shard_path.is_file() and shard_path.stat().st_size > 0:
            candidate = torch.load(shard_path, weights_only=False)
            actual = {key: candidate.get(key) for key in expected}
            if actual != expected:
                raise ValueError(f"compiled proposal shard metadata mismatch: {shard_path}")
            shard_records = tuple(candidate.get("records", ()))
            if len(shard_records) != len(shard_smiles):
                raise ValueError(f"compiled proposal shard is incomplete: {shard_path}")
            phase = "compiled_proposal_shard_loaded"
            elapsed = 0.0
        else:
            started = perf_counter()
            shard_records = build_tracelet_path_records(
                shard_smiles,
                n_slots=args.max_atoms,
                typed_ring_payloads=True,
                workers=0,
                checkpoint_interval=args.path_checkpoint_interval,
                executor=executor,
            )
            elapsed = perf_counter() - started
            _atomic_torch_save({**expected, "records": shard_records}, shard_path)
            phase = "compiled_proposal_shard_saved"
        records.extend(shard_records)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "partition": partition,
                    "shard": shard_index + 1,
                    "shards": total_shards,
                    "targets_complete": stop,
                    "targets_total": len(smiles),
                    "records": len(shard_records),
                    "seconds": elapsed,
                    "bytes": shard_path.stat().st_size,
                    "path": str(shard_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(records)


def _compile_tree_transport_partition_shards(
    *,
    partition: str,
    smiles: tuple[str, ...],
    base_seed: int,
    args: argparse.Namespace,
    source_prior: DegreeBoundedCarbonTreePrior,
    ring_catalog: TypedRingCatalog | None,
    signature: dict[str, object],
    executor: Executor | None,
) -> tuple[PathRecord, ...]:
    """Load or atomically compile one deterministic transport partition."""

    cache_root = _sharded_path_cache_root(args.path_cache) / "transport"
    fingerprint = _path_cache_fingerprint(signature)
    records: list[PathRecord] = []
    total_shards = (len(smiles) + args.path_shard_size - 1) // args.path_shard_size
    for shard_index, start in enumerate(range(0, len(smiles), args.path_shard_size)):
        stop = min(start + args.path_shard_size, len(smiles))
        shard_smiles = smiles[start:stop]
        shard_path = cache_root / f"{partition}-{shard_index:05d}.pt"
        expected = {
            "signature_fingerprint": fingerprint,
            "partition": partition,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "target_count": len(shard_smiles),
        }
        if shard_path.is_file() and shard_path.stat().st_size > 0:
            candidate = torch.load(shard_path, weights_only=False)
            actual = {key: candidate.get(key) for key in expected}
            if actual != expected:
                raise ValueError(f"compiled path shard metadata mismatch: {shard_path}")
            candidate_records = tuple(candidate.get("records", ()))
            expected_records = len(shard_smiles) * args.tree_couplings_per_target
            if len(candidate_records) != expected_records:
                raise ValueError(f"compiled path shard is incomplete: {shard_path}")
            shard_records = candidate_records
            phase = "compiled_path_shard_loaded"
            elapsed = 0.0
        elif args.require_path_cache:
            raise FileNotFoundError(f"required compiled path shard is missing: {shard_path}")
        else:
            started = perf_counter()
            shard_records = build_tree_transport_path_records(
                shard_smiles,
                n_slots=args.max_atoms,
                source_prior=source_prior,
                seed=_tree_transport_shard_seed(base_seed, shard_index),
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=ring_catalog,
                workers=0,
                checkpoint_interval=args.path_checkpoint_interval,
                executor=executor,
            )
            elapsed = perf_counter() - started
            _atomic_torch_save({**expected, "records": shard_records}, shard_path)
            phase = "compiled_path_shard_saved"
        records.extend(shard_records)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "partition": partition,
                    "shard": shard_index + 1,
                    "shards": total_shards,
                    "targets_complete": stop,
                    "targets_total": len(smiles),
                    "records": len(shard_records),
                    "seconds": elapsed,
                    "bytes": shard_path.stat().st_size,
                    "path": str(shard_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument(
        "--model",
        choices=("from_scratch", "prior_only", "prior_tilted"),
        default="from_scratch",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument(
        "--fast-split",
        action="store_true",
        help="stop after the requested eligible molecules; diagnostic screens only",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument(
        "--training-backend",
        choices=("exact_fiber", "factorized_marks"),
        default="exact_fiber",
        help="exact successor oracle or scalable dense marked-rewrite training",
    )
    parser.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="persistent deterministic workers for factorized batch preparation",
    )
    parser.add_argument(
        "--data-prefetch-factor",
        type=int,
        default=2,
        help="whole batches queued per persistent factorized-data worker",
    )
    parser.add_argument(
        "--use-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use bfloat16 autocast for the dense CUDA training path",
    )
    parser.add_argument(
        "--skip-rollouts",
        action="store_true",
        help="stop after train/validation/test; intended for throughput gates",
    )
    parser.add_argument(
        "--rate-factorization",
        choices=("hierarchical", "quotient_energy", "superposed"),
        default="hierarchical",
    )
    parser.add_argument(
        "--ring-proposals",
        choices=("generic_carbon", "typed_catalog"),
        default="generic_carbon",
    )
    parser.add_argument(
        "--ring-electronic-mode",
        choices=("factorized_local", "catalog_exact"),
        default="factorized_local",
        help=(
            "semantic structured-label decoder (generalizes across electronic "
            "assignments) or the diagnostic observed-alias decoder"
        ),
    )
    parser.add_argument("--max-cycle-templates", type=int, default=128)
    parser.add_argument("--max-attach-templates", type=int, default=128)
    parser.add_argument("--max-ear-templates", type=int, default=128)
    parser.add_argument("--max-ring-system-templates", type=int, default=4096)
    parser.add_argument(
        "--source-prior",
        choices=("null", "carbon_tree"),
        default="null",
    )
    parser.add_argument(
        "--tree-size-prior",
        choices=("empirical", "uniform"),
        default="empirical",
        help="source N is sampled independently of the paired target",
    )
    parser.add_argument(
        "--tree-couplings-per-target",
        type=int,
        default=1,
        help="independent source-tree draws compiled for each data endpoint",
    )
    parser.add_argument(
        "--tree-transport",
        choices=("primitive", "size_matched_graft", "flexible_size_graft"),
        default="primitive",
        help=(
            "primitive delete/regrow baseline, equal-size atom-retaining Graft, "
            "or independently sized grow/shrink plus Graft transport"
        ),
    )
    parser.add_argument(
        "--bond-representation",
        choices=("kekule", "aromatic"),
        default="kekule",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linear AdamW warmup updates; factorized marked training only",
    )
    parser.add_argument(
        "--minimum-learning-rate-fraction",
        type=float,
        default=1.0,
        help="terminal cosine-decay learning-rate fraction",
    )
    parser.add_argument(
        "--evaluation-every",
        type=int,
        default=0,
        help="validation updates between evaluations; 0 uses ten evaluations",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="non-improving evaluations before stopping; 0 disables stopping",
    )
    parser.add_argument(
        "--early-stopping-min-relative-delta",
        type=float,
        default=0.0,
        help=(
            "relative validation-loss decrease required to reset patience; "
            "the absolute best checkpoint is still retained"
        ),
    )
    parser.add_argument("--action-kl-weight", type=float, default=0.0)
    parser.add_argument("--hazard-tilt-weight", type=float, default=0.0)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=64,
        help="streamed model-evaluation microbatch size",
    )
    parser.add_argument("--late-time-fraction", type=float, default=0.5)
    parser.add_argument("--operational-horizon", type=float, default=7.0)
    parser.add_argument(
        "--progress-stratification-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--teacher-ordering",
        choices=("sequential", "causal_frontier"),
        default="sequential",
        help="use one compiler order or all currently enabled causal rewrites",
    )
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--rollout-samples", type=int, default=1000)
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument(
        "--fiber-workers",
        type=int,
        default=1,
        help="persistent CPU processes used to construct exact legal action fibers",
    )
    parser.add_argument(
        "--path-workers",
        type=int,
        default=0,
        help="processes used for deterministic path compilation",
    )
    parser.add_argument(
        "--path-checkpoint-interval",
        type=int,
        default=8,
        help=(
            "retain one packed graph checkpoint per N path edits; sampled "
            "states replay at most N-1 validated rewrites"
        ),
    )
    parser.add_argument(
        "--path-shard-size",
        type=int,
        default=2048,
        help="target molecules per atomic, resumable transport-path shard",
    )
    parser.add_argument(
        "--compile-paths-only",
        action="store_true",
        help="compile and validate the path cache, then exit before model work",
    )
    parser.add_argument(
        "--require-path-cache",
        action="store_true",
        help="fail instead of compiling paths; used to keep GPUs out of preprocessing",
    )
    parser.add_argument(
        "--evaluation-cache-dir",
        type=Path,
        help=(
            "shared directory for content-addressed fixed validation/test batches"
        ),
    )
    parser.add_argument(
        "--compile-evaluation-cache",
        action="store_true",
        help="build the fixed evaluation batches during a CPU compile-only stage",
    )
    parser.add_argument(
        "--require-evaluation-cache",
        action="store_true",
        help="fail rather than build fixed evaluation batches in this process",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        help="CPU workers for fixed evaluation batches; defaults to data-workers",
    )
    parser.add_argument(
        "--allow-legacy-evaluation-cache",
        action="store_true",
        help=(
            "load a scientifically matching eager cache from an older storage "
            "format; requires --load-checkpoint and --require-path-cache"
        ),
    )
    parser.add_argument(
        "--corpus-workers",
        type=int,
        default=0,
        help="processes used for a full-corpus chemistry eligibility scan",
    )
    parser.add_argument("--quality-metrics", action="store_true")
    parser.add_argument("--include-fcd", action="store_true")
    parser.add_argument(
        "--quality-reference-file",
        type=Path,
        help="optional held-out/reference SMILES file for distribution metrics",
    )
    parser.add_argument("--quality-reference-limit", type=int, default=5000)
    parser.add_argument(
        "--fcd-generated-limit",
        type=int,
        default=0,
        help="0 uses every valid generated molecule",
    )
    parser.add_argument(
        "--fcd-device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/tracelet_gate.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--provenance-sha256",
        type=str,
        default="",
        help="immutable wrapper identity for exact recovery compatibility",
    )
    parser.add_argument(
        "--path-cache",
        type=Path,
        help="persist compiled source-to-target paths for restart-safe preprocessing",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="evaluate a trained neural checkpoint without retraining",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume interrupted training from a periodic recovery checkpoint",
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=200,
        help="training steps between exact recovery snapshots; 0 disables them",
    )
    parser.add_argument(
        "--rollout-cache",
        type=Path,
        help="persist completed rollouts before optional quality metrics",
    )
    parser.add_argument(
        "--load-rollouts",
        type=Path,
        help="reuse a persisted rollout cache instead of sampling",
    )
    args = parser.parse_args()

    if args.load_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("--load-checkpoint and --resume-checkpoint are mutually exclusive")
    if args.recovery_every < 0:
        raise ValueError("--recovery-every must be non-negative")
    if not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("--warmup-steps must lie in [0, steps]")
    if not 0.0 < args.minimum_learning_rate_fraction <= 1.0:
        raise ValueError("--minimum-learning-rate-fraction must lie in (0, 1]")
    if args.evaluation_every < 0 or args.early_stopping_patience < 0:
        raise ValueError("--evaluation-every and --early-stopping-patience must be non-negative")
    if not 0.0 <= args.early_stopping_min_relative_delta < 1.0:
        raise ValueError("--early-stopping-min-relative-delta must lie in [0, 1)")
    if min(args.data_workers, args.path_workers, args.corpus_workers) < 0:
        raise ValueError("data, path, and corpus worker counts must be non-negative")
    if args.evaluation_workers is not None and args.evaluation_workers < 0:
        raise ValueError("evaluation-workers must be non-negative")
    if args.evaluation_batch_size <= 0:
        raise ValueError("evaluation-batch-size must be positive")
    if args.data_prefetch_factor <= 0:
        raise ValueError("data-prefetch-factor must be positive")
    if args.path_checkpoint_interval <= 0:
        raise ValueError("path-checkpoint-interval must be positive")
    if args.path_shard_size <= 0:
        raise ValueError("path-shard-size must be positive")
    if args.compile_paths_only and args.path_cache is None:
        raise ValueError("compile-paths-only requires --path-cache")
    if args.compile_evaluation_cache and not args.compile_paths_only:
        raise ValueError("compile-evaluation-cache requires --compile-paths-only")
    if (args.compile_evaluation_cache or args.require_evaluation_cache) and (
        args.evaluation_cache_dir is None
    ):
        raise ValueError(
            "evaluation-cache-dir is required when compiling or requiring its cache"
        )
    if args.compile_evaluation_cache and args.require_evaluation_cache:
        raise ValueError(
            "compile-evaluation-cache and require-evaluation-cache are mutually exclusive"
        )
    if args.allow_legacy_evaluation_cache and (
        args.load_checkpoint is None or not args.require_path_cache
    ):
        raise ValueError(
            "allow-legacy-evaluation-cache requires load-checkpoint and require-path-cache"
        )
    if args.training_backend == "factorized_marks" and (
        args.model != "from_scratch"
        or args.ring_proposals != "typed_catalog"
        or args.teacher_ordering != "sequential"
    ):
        raise ValueError(
            "factorized marked training requires from_scratch, typed_catalog, "
            "and the sequential teacher"
        )

    if args.ring_proposals == "typed_catalog" and args.model != "from_scratch":
        raise ValueError("the typed-catalog diagnostic currently requires --model from_scratch")
    if args.source_prior == "carbon_tree" and (
        args.model != "from_scratch" or args.ring_proposals != "typed_catalog"
    ):
        raise ValueError(
            "carbon-tree transport currently requires --model from_scratch "
            "and --ring-proposals typed_catalog"
        )
    if args.source_prior == "carbon_tree" and args.teacher_ordering != "sequential":
        raise ValueError("carbon-tree transport first gates the sequential teacher")
    if args.tree_transport != "primitive" and args.source_prior != "carbon_tree":
        raise ValueError("non-primitive tree transport requires --source-prior carbon_tree")
    if args.tree_couplings_per_target <= 0:
        raise ValueError("--tree-couplings-per-target must be positive")
    if args.quality_reference_limit <= 0:
        raise ValueError("--quality-reference-limit must be positive")
    if args.teacher_ordering == "causal_frontier" and args.ring_proposals != "typed_catalog":
        raise ValueError(
            "causal-frontier teachers require topology-committed typed ring "
            "traces; use --ring-proposals typed_catalog"
        )
    if args.teacher_ordering == "causal_frontier" and args.progress_stratification_fraction != 0.0:
        raise ValueError(
            "causal-frontier teachers cannot be combined with family progress stratification"
        )

    if args.model == "prior_only" and args.steps != 800:
        # The argument is ignored, but accepting an explicit common command
        # line keeps the three-way ablation launch mechanically identical.
        pass
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_torch_device(args.device)
    print(
        json.dumps(
            {
                "phase": "device",
                "requested": args.device,
                "selected": str(device),
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=not args.fast_split,
        workers=args.corpus_workers,
    )
    print(json.dumps({"phase": "split_loaded"}), flush=True)
    tree_source_prior = None
    if args.source_prior == "carbon_tree":
        if args.tree_size_prior == "empirical":
            size_counts = Counter(
                smiles_to_molecular_graph(text).n_real_atoms for text in split.train
            )
            tree_source_prior = DegreeBoundedCarbonTreePrior.from_size_counts(dict(size_counts))
        else:
            tree_source_prior = DegreeBoundedCarbonTreePrior(
                sizes=tuple(range(1, args.max_atoms + 1))
            )
        print(
            json.dumps(
                {
                    "phase": "tree_source_prior",
                    "size_mode": args.tree_size_prior,
                    "sizes": tree_source_prior.sizes,
                    "probabilities": tree_source_prior.probabilities,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    path_cache_signature = {
        # Revision 7 invalidates paths compiled before Graft actions were
        # quotiented by molecular state.  Reusing a v6 manifest would train
        # against gauge-only reroutes even though the current compiler no
        # longer emits them.
        "format_version": 7,
        "seed": args.seed,
        "max_atoms": args.max_atoms,
        "ring_proposals": args.ring_proposals,
        "max_cycle_templates": args.max_cycle_templates,
        "max_attach_templates": args.max_attach_templates,
        "max_ear_templates": args.max_ear_templates,
        "max_ring_system_templates": args.max_ring_system_templates,
        "ring_system_electronic_alias_version": RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
        "source_prior": args.source_prior,
        "tree_size_prior": args.tree_size_prior,
        "tree_couplings_per_target": args.tree_couplings_per_target,
        "tree_transport": args.tree_transport,
        "path_checkpoint_interval": args.path_checkpoint_interval,
        "path_shard_size": args.path_shard_size,
        "train_smiles": split.train,
        "validation_smiles": split.validation,
        "test_smiles": split.test,
    }
    loaded_path_cache = None
    loaded_sharded_manifest = None
    supported_record_keys = None
    if (
        args.path_cache is not None
        and args.path_cache.is_file()
        and args.path_cache.stat().st_size > 0
    ):
        candidate = torch.load(
            args.path_cache,
            weights_only=False,
            mmap=bool(args.allow_legacy_evaluation_cache),
        )
        candidate_signature = candidate.get("signature")
        exact_signature = candidate_signature == path_cache_signature
        legacy_evaluation_signature = (
            args.allow_legacy_evaluation_cache
            and isinstance(candidate_signature, dict)
            and _legacy_evaluation_signature_matches(
                candidate_signature,
                path_cache_signature,
            )
        )
        if not exact_signature and not legacy_evaluation_signature:
            print(
                json.dumps(
                    {
                        "phase": "compiled_path_cache_signature_mismatch",
                        "path": str(args.path_cache),
                        "differences": _path_cache_signature_mismatch(
                            candidate_signature,
                            path_cache_signature,
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise ValueError(f"compiled path cache/config mismatch: {args.path_cache}")
        loaded_path_cache = candidate
        ring_catalog = candidate.get("ring_catalog")
        train_records = tuple(candidate["train_records"])
        validation_records = tuple(candidate["validation_records"])
        test_records = tuple(candidate["test_records"])
        if legacy_evaluation_signature and not exact_signature:
            for partition, records, smiles in (
                ("train", train_records, split.train),
                ("validation", validation_records, split.validation),
                ("test", test_records, split.test),
            ):
                _validate_legacy_evaluation_partition(
                    records,
                    smiles,
                    n_slots=args.max_atoms,
                    couplings_per_target=args.tree_couplings_per_target,
                    partition=partition,
                )
        print(
            json.dumps(
                {
                    "phase": "compiled_path_cache_loaded",
                    "path": str(args.path_cache),
                    "train": len(train_records),
                    "validation": len(validation_records),
                    "test": len(test_records),
                    "legacy_storage_format": bool(
                        legacy_evaluation_signature and not exact_signature
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if loaded_path_cache is None and tree_source_prior is not None and args.path_cache is not None:
        loaded_sharded_manifest = _load_sharded_path_manifest(
            args.path_cache,
            signature=path_cache_signature,
        )
        if loaded_sharded_manifest is not None:
            ring_catalog = loaded_sharded_manifest["ring_catalog"]
            supported = loaded_sharded_manifest["supported_smiles"]
            supported_train = tuple(supported["train"])
            supported_validation = tuple(supported["validation"])
            supported_test = tuple(supported["test"])
            supported_record_keys = loaded_sharded_manifest.get("supported_record_keys")
            print(
                json.dumps(
                    {
                        "phase": "compiled_path_manifest_loaded",
                        "path": str(_sharded_path_manifest(args.path_cache)),
                        "complete": bool(loaded_sharded_manifest.get("complete")),
                        "train_targets": len(supported_train),
                        "validation_targets": len(supported_validation),
                        "test_targets": len(supported_test),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    cache_is_ready = loaded_path_cache is not None or (
        loaded_sharded_manifest is not None and bool(loaded_sharded_manifest.get("complete"))
    )
    if args.require_path_cache and not cache_is_ready:
        location = args.path_cache if args.path_cache is not None else "<unset>"
        raise FileNotFoundError(f"complete compiled path cache required: {location}")

    tree_paths_ready = False
    if loaded_path_cache is None and tree_source_prior is not None:
        if args.path_cache is not None:
            with tracelet_path_executor(
                args.path_workers,
                n_slots=args.max_atoms,
                typed_ring_payloads=True,
                ring_catalog=None,
                checkpoint_interval=args.path_checkpoint_interval,
            ) as path_executor:
                raw_train_records = _compile_tree_transport_partition_shards(
                    partition="train",
                    smiles=split.train,
                    base_seed=args.seed + 10,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=None,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
                raw_validation_records = _compile_tree_transport_partition_shards(
                    partition="validation",
                    smiles=split.validation,
                    base_seed=args.seed + 11,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=None,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
                raw_test_records = _compile_tree_transport_partition_shards(
                    partition="test",
                    smiles=split.test,
                    base_seed=args.seed + 12,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=None,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
        else:
            raw_train_records = build_tree_transport_path_records(
                split.train,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 10,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=None,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
            raw_validation_records = build_tree_transport_path_records(
                split.validation,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 11,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=None,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
            raw_test_records = build_tree_transport_path_records(
                split.test,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 12,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=None,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )

        finalized_manifest = (
            loaded_sharded_manifest is not None
            and bool(loaded_sharded_manifest.get("complete"))
            and loaded_sharded_manifest.get("transport_support_projection_version")
            == TRANSPORT_SUPPORT_PROJECTION_VERSION
            and isinstance(supported_record_keys, dict)
        )
        if finalized_manifest:
            if ring_catalog is None:
                raise RuntimeError("complete transport manifest is missing its ring-system catalog")
            train_records = _filter_records_by_finalized_manifest(
                raw_train_records,
                supported_train,
                n_slots=args.max_atoms,
                couplings_per_target=args.tree_couplings_per_target,
                partition="train",
                supported_record_keys=tuple(supported_record_keys["train"]),
            )
            validation_records = _filter_records_by_finalized_manifest(
                raw_validation_records,
                supported_validation,
                n_slots=args.max_atoms,
                couplings_per_target=args.tree_couplings_per_target,
                partition="validation",
                supported_record_keys=tuple(supported_record_keys["validation"]),
            )
            test_records = _filter_records_by_finalized_manifest(
                raw_test_records,
                supported_test,
                n_slots=args.max_atoms,
                couplings_per_target=args.tree_couplings_per_target,
                partition="test",
                supported_record_keys=tuple(supported_record_keys["test"]),
            )
            print(
                json.dumps(
                    {
                        "phase": "finalized_transport_support_loaded",
                        **ring_catalog.statistics(),
                        "train_supported": len(train_records),
                        "train_total": len(raw_train_records),
                        "validation_supported": len(validation_records),
                        "validation_total": len(raw_validation_records),
                        "test_supported": len(test_records),
                        "test_total": len(raw_test_records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            # Fresh or unversioned caches must rebuild from executable paths.
            # This prevents a stale catalog from omitting a Kekule/lowering
            # alias used by another saved tree coupling even when its semantic
            # topology is supported.
            ring_catalog = build_typed_ring_catalog_from_paths(
                (record.path for record in raw_train_records),
                max_cycle_templates=args.max_cycle_templates,
                max_attach_templates=args.max_attach_templates,
                max_ear_templates=args.max_ear_templates,
                max_ring_system_templates=args.max_ring_system_templates,
            )
            if ring_catalog is None:
                raise RuntimeError("tree transport requires a full ring-system catalog")
            train_records = tuple(
                record
                for record in raw_train_records
                if structured_ring_trace_supported(record.path.trace, ring_catalog)
            )
            validation_records = tuple(
                record
                for record in raw_validation_records
                if structured_ring_trace_supported(record.path.trace, ring_catalog)
            )
            test_records = tuple(
                record
                for record in raw_test_records
                if structured_ring_trace_supported(record.path.trace, ring_catalog)
            )
            print(
                json.dumps(
                    {
                        "phase": "full_ring_transport_catalog",
                        **ring_catalog.statistics(),
                        "train_supported": len(train_records),
                        "train_total": len(raw_train_records),
                        "validation_supported": len(validation_records),
                        "validation_total": len(raw_validation_records),
                        "test_supported": len(test_records),
                        "test_total": len(raw_test_records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.path_cache is not None:
                supported_keys = {
                    "train": {record.target_key for record in train_records},
                    "validation": {record.target_key for record in validation_records},
                    "test": {record.target_key for record in test_records},
                }

                def retained_smiles(partition: str, items: tuple[str, ...]) -> tuple[str, ...]:
                    return tuple(
                        text
                        for text in items
                        if canonical_state_key(
                            pad_molecular_graph(smiles_to_molecular_graph(text), args.max_atoms)
                        )
                        in supported_keys[partition]
                    )

                _save_sharded_path_manifest(
                    args.path_cache,
                    signature=path_cache_signature,
                    ring_catalog=ring_catalog,
                    supported_smiles={
                        "train": retained_smiles("train", split.train),
                        "validation": retained_smiles("validation", split.validation),
                        "test": retained_smiles("test", split.test),
                    },
                    complete=True,
                    supported_record_keys={
                        "train": tuple(
                            _path_record_support_key(record) for record in train_records
                        ),
                        "validation": tuple(
                            _path_record_support_key(record) for record in validation_records
                        ),
                        "test": tuple(_path_record_support_key(record) for record in test_records),
                    },
                )
        tree_paths_ready = True

    if loaded_path_cache is None and loaded_sharded_manifest is None and tree_source_prior is None:
        ring_catalog = None
        if args.ring_proposals == "typed_catalog":
            if tree_source_prior is not None and args.path_cache is not None:
                with tracelet_path_executor(
                    args.path_workers,
                    n_slots=args.max_atoms,
                    typed_ring_payloads=True,
                    ring_catalog=None,
                    checkpoint_interval=args.path_checkpoint_interval,
                ) as proposal_executor:
                    proposal_records = _compile_tracelet_proposal_partition_shards(
                        partition="train",
                        smiles=split.train,
                        args=args,
                        signature=path_cache_signature,
                        executor=proposal_executor,
                    )
            else:
                proposal_records = build_tracelet_path_records(
                    split.train,
                    n_slots=args.max_atoms,
                    typed_ring_payloads=True,
                    workers=args.path_workers,
                    checkpoint_interval=args.path_checkpoint_interval,
                )
            ring_catalog = build_typed_ring_catalog(
                (record.path.trace for record in proposal_records),
                max_cycle_templates=args.max_cycle_templates,
                max_attach_templates=args.max_attach_templates,
                max_ear_templates=args.max_ear_templates,
                max_ring_system_templates=args.max_ring_system_templates,
            )
            print(
                json.dumps(
                    {"phase": "typed_ring_catalog", **ring_catalog.statistics()},
                    sort_keys=True,
                ),
                flush=True,
            )
            train_records = tuple(
                record
                for record in proposal_records
                if ring_catalog.supports_trace(record.path.trace)
            )
            print(
                json.dumps(
                    {
                        "phase": "typed_ring_training_support",
                        "supported": len(train_records),
                        "total": len(proposal_records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            train_records = build_tracelet_path_records(
                split.train,
                n_slots=args.max_atoms,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
        print(json.dumps({"phase": "train_paths_compiled"}), flush=True)
        validation_records = build_tracelet_path_records(
            split.validation,
            n_slots=args.max_atoms,
            typed_ring_payloads=ring_catalog is not None,
            workers=args.path_workers,
            checkpoint_interval=args.path_checkpoint_interval,
        )
        test_records = build_tracelet_path_records(
            split.test,
            n_slots=args.max_atoms,
            typed_ring_payloads=ring_catalog is not None,
            workers=args.path_workers,
            checkpoint_interval=args.path_checkpoint_interval,
        )
        if ring_catalog is not None:
            raw_validation_count = len(validation_records)
            raw_test_count = len(test_records)
            validation_records = tuple(
                record
                for record in validation_records
                if ring_catalog.supports_trace(record.path.trace)
            )
            test_records = tuple(
                record for record in test_records if ring_catalog.supports_trace(record.path.trace)
            )
            print(
                json.dumps(
                    {
                        "phase": "typed_ring_evaluation_support",
                        "validation_supported": len(validation_records),
                        "validation_total": raw_validation_count,
                        "test_supported": len(test_records),
                        "test_total": raw_test_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if loaded_path_cache is None and tree_source_prior is not None and not tree_paths_ready:
        if loaded_sharded_manifest is None:
            train_keys = {record.target_key for record in train_records}
            validation_keys = {record.target_key for record in validation_records}
            test_keys = {record.target_key for record in test_records}

            def supported_smiles(
                items: tuple[str, ...],
                keys: set[str],
            ) -> tuple[str, ...]:
                return tuple(
                    text
                    for text in items
                    if canonical_state_key(
                        pad_molecular_graph(
                            smiles_to_molecular_graph(text),
                            args.max_atoms,
                        )
                    )
                    in keys
                )

            supported_train = supported_smiles(split.train, train_keys)
            supported_validation = supported_smiles(split.validation, validation_keys)
            supported_test = supported_smiles(split.test, test_keys)
            if args.path_cache is not None:
                _save_sharded_path_manifest(
                    args.path_cache,
                    signature=path_cache_signature,
                    ring_catalog=ring_catalog,
                    supported_smiles={
                        "train": supported_train,
                        "validation": supported_validation,
                        "test": supported_test,
                    },
                    complete=False,
                )
                print(
                    json.dumps(
                        {
                            "phase": "compiled_path_manifest_saved",
                            "path": str(_sharded_path_manifest(args.path_cache)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        # The proposal paths are no longer needed once support has been
        # projected back to SMILES. Release them before accumulating transport
        # paths rather than holding both multi-million-state corpora at once.
        train_records = ()
        validation_records = ()
        test_records = ()
        if "proposal_records" in locals():
            proposal_records = ()

        if args.path_cache is not None:
            with tracelet_path_executor(
                args.path_workers,
                n_slots=args.max_atoms,
                typed_ring_payloads=True,
                ring_catalog=ring_catalog,
                checkpoint_interval=args.path_checkpoint_interval,
            ) as path_executor:
                train_records = _compile_tree_transport_partition_shards(
                    partition="train",
                    smiles=supported_train,
                    base_seed=args.seed + 10,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=ring_catalog,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
                validation_records = _compile_tree_transport_partition_shards(
                    partition="validation",
                    smiles=supported_validation,
                    base_seed=args.seed + 11,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=ring_catalog,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
                test_records = _compile_tree_transport_partition_shards(
                    partition="test",
                    smiles=supported_test,
                    base_seed=args.seed + 12,
                    args=args,
                    source_prior=tree_source_prior,
                    ring_catalog=ring_catalog,
                    signature=path_cache_signature,
                    executor=path_executor,
                )
            _save_sharded_path_manifest(
                args.path_cache,
                signature=path_cache_signature,
                ring_catalog=ring_catalog,
                supported_smiles={
                    "train": supported_train,
                    "validation": supported_validation,
                    "test": supported_test,
                },
                complete=True,
                supported_record_keys={
                    "train": tuple(_path_record_support_key(record) for record in train_records),
                    "validation": tuple(
                        _path_record_support_key(record) for record in validation_records
                    ),
                    "test": tuple(_path_record_support_key(record) for record in test_records),
                },
            )
        else:
            train_records = build_tree_transport_path_records(
                supported_train,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 10,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=ring_catalog,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
            validation_records = build_tree_transport_path_records(
                supported_validation,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 11,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=ring_catalog,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
            test_records = build_tree_transport_path_records(
                supported_test,
                n_slots=args.max_atoms,
                source_prior=tree_source_prior,
                seed=args.seed + 12,
                couplings_per_target=args.tree_couplings_per_target,
                transport_mode=args.tree_transport,
                typed_ring_payloads=True,
                ring_catalog=ring_catalog,
                workers=args.path_workers,
                checkpoint_interval=args.path_checkpoint_interval,
            )
        print(
            json.dumps(
                {
                    "phase": "tree_transport_paths_compiled",
                    "train": len(train_records),
                    "validation": len(validation_records),
                    "test": len(test_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if loaded_path_cache is None and tree_source_prior is None and args.path_cache is not None:
        _atomic_torch_save(
            {
                "signature": path_cache_signature,
                "ring_catalog": ring_catalog,
                "train_records": train_records,
                "validation_records": validation_records,
                "test_records": test_records,
            },
            args.path_cache,
        )
        print(
            json.dumps(
                {"phase": "compiled_path_cache_saved", "path": str(args.path_cache)},
                sort_keys=True,
            ),
            flush=True,
        )
    empty_partitions = [
        name
        for name, records in (
            ("train", train_records),
            ("validation", validation_records),
            ("test", test_records),
        )
        if not records
    ]
    if empty_partitions:
        raise ValueError(
            "typed ring support left empty path partitions: "
            f"{empty_partitions}; enlarge the training/catalog split or increase "
            "the typed-template limits"
        )
    print(json.dumps({"phase": "evaluation_paths_compiled"}), flush=True)
    if args.compile_paths_only:
        print(
            json.dumps(
                {
                    "phase": "compiled_paths_ready",
                    "train": len(train_records),
                    "validation": len(validation_records),
                    "test": len(test_records),
                    "path_cache": (None if args.path_cache is None else str(args.path_cache)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.compile_evaluation_cache:
            return

    evaluation_signature = _evaluation_batch_cache_signature(
        path_cache_signature,
        training_backend=args.training_backend,
        seed=args.seed,
        validation_examples=args.validation_examples,
        test_examples=args.test_examples,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
        progress_stratification_fraction=args.progress_stratification_fraction,
        bond_representation=args.bond_representation,
        ring_electronic_mode=args.ring_electronic_mode,
    )
    evaluation_cache_path = (
        None
        if args.evaluation_cache_dir is None
        else _evaluation_batch_cache_path(
            args.evaluation_cache_dir,
            evaluation_signature,
        )
    )
    evaluation_workers = (
        args.data_workers
        if args.evaluation_workers is None
        else args.evaluation_workers
    )
    validation_examples = None
    test_examples = None
    if evaluation_cache_path is not None and evaluation_cache_path.is_file():
        cache_load_started = perf_counter()
        validation_examples, test_examples = _load_evaluation_batch_cache(
            evaluation_cache_path,
            signature=evaluation_signature,
            validation_examples=args.validation_examples,
            test_examples=args.test_examples,
        )
        print(
            json.dumps(
                {
                    "phase": "evaluation_batch_cache_loaded",
                    "path": str(evaluation_cache_path),
                    "seconds": perf_counter() - cache_load_started,
                    "validation_examples": args.validation_examples,
                    "test_examples": args.test_examples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif args.require_evaluation_cache:
        raise FileNotFoundError(
            "required evaluation batch cache is missing: "
            f"{evaluation_cache_path}"
        )

    evaluation_build_started = perf_counter()
    if validation_examples is None and args.training_backend == "factorized_marks":
        validation_examples = sample_factorized_mark_batch(
            validation_records,
            batch_size=args.validation_examples,
            seed=args.seed + 1,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=args.progress_stratification_fraction,
            use_aromatic_bond_view=args.bond_representation == "aromatic",
            workers=evaluation_workers,
            ring_catalog=ring_catalog,
            ring_electronic_mode=args.ring_electronic_mode,
        )
        print(
            json.dumps(
                {
                    "phase": "validation_factorized_batch_built",
                    "seconds": perf_counter() - evaluation_build_started,
                    "workers": evaluation_workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        test_build_started = perf_counter()
        test_examples = sample_factorized_mark_batch(
            test_records,
            batch_size=args.test_examples,
            seed=args.seed + 2,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=args.progress_stratification_fraction,
            use_aromatic_bond_view=args.bond_representation == "aromatic",
            workers=evaluation_workers,
            ring_catalog=ring_catalog,
            ring_electronic_mode=args.ring_electronic_mode,
        )
        print(
            json.dumps(
                {
                    "phase": "test_factorized_batch_built",
                    "seconds": perf_counter() - test_build_started,
                    "workers": evaluation_workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif validation_examples is None:
        fiber_cache = {}
        with tracelet_fiber_executor(
            args.fiber_workers,
            ring_catalog=ring_catalog,
        ) as evaluation_fiber_executor:
            validation_examples = sample_tracelet_conditional_batch(
                validation_records,
                batch_size=args.validation_examples,
                rng=np.random.default_rng(args.seed + 1),
                fiber_cache=fiber_cache,
                late_time_fraction=args.late_time_fraction,
                operational_horizon=args.operational_horizon,
                ring_catalog=ring_catalog,
                causal_teachers=args.teacher_ordering == "causal_frontier",
                fiber_executor=evaluation_fiber_executor,
            )
            print(
                json.dumps(
                    {
                        "phase": "validation_examples_built",
                        "cache_size": len(fiber_cache),
                    }
                ),
                flush=True,
            )
            test_examples = sample_tracelet_conditional_batch(
                test_records,
                batch_size=args.test_examples,
                rng=np.random.default_rng(args.seed + 2),
                fiber_cache=fiber_cache,
                late_time_fraction=args.late_time_fraction,
                operational_horizon=args.operational_horizon,
                ring_catalog=ring_catalog,
                causal_teachers=args.teacher_ordering == "causal_frontier",
                fiber_executor=evaluation_fiber_executor,
            )
            print(
                json.dumps(
                    {"phase": "test_examples_built", "cache_size": len(fiber_cache)}
                ),
                flush=True,
            )

    if validation_examples is None or test_examples is None:
        raise RuntimeError("evaluation batch construction produced an empty partition")
    if evaluation_cache_path is not None and not evaluation_cache_path.is_file():
        _atomic_shared_torch_save(
            {
                "signature": evaluation_signature,
                "validation_batch": validation_examples,
                "test_batch": test_examples,
            },
            evaluation_cache_path,
        )
        print(
            json.dumps(
                {
                    "phase": "evaluation_batch_cache_saved",
                    "path": str(evaluation_cache_path),
                    "bytes": evaluation_cache_path.stat().st_size,
                    "validation_examples": args.validation_examples,
                    "test_examples": args.test_examples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.compile_paths_only:
        print(
            json.dumps(
                {
                    "phase": "compiled_evaluation_batches_ready",
                    "path": (
                        None
                        if evaluation_cache_path is None
                        else str(evaluation_cache_path)
                    ),
                    "validation_examples": args.validation_examples,
                    "test_examples": args.test_examples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    # The empirical marginal model scans every compiled path.  Neural
    # from-scratch training does not consume it, so fitting it here would keep
    # an allocated GPU idle for a purely dead CPU preprocessing pass.
    prior = (
        None
        if args.model == "from_scratch"
        else fit_tracelet_corpus_marginal_rate_model(train_records)
    )

    if args.model == "from_scratch" and args.training_backend == "factorized_marks":
        if ring_catalog is None:
            raise RuntimeError("factorized marked model requires a typed ring catalog")
        model = FactorizedTraceletRateModel(
            ring_catalog,
            hidden_dim=args.hidden_dim,
            message_passing_steps=args.message_passing_steps,
            ring_electronic_mode=args.ring_electronic_mode,
        ).to(device)
    elif args.model == "from_scratch":
        model = TraceletRateModel(
            hidden_dim=args.hidden_dim,
            message_passing_steps=args.message_passing_steps,
            rate_factorization=args.rate_factorization,
            use_aromatic_bond_view=args.bond_representation == "aromatic",
            ring_catalog=ring_catalog,
        ).to(device)
    elif args.model == "prior_tilted":
        if prior is None:
            raise RuntimeError("prior-tilted training requires an empirical prior")
        model = PriorTiltedTraceletRateModel(
            prior,
            hidden_dim=args.hidden_dim,
            message_passing_steps=args.message_passing_steps,
            use_aromatic_bond_view=args.bond_representation == "aromatic",
        ).to(device)
    else:
        if prior is None:
            raise RuntimeError("corpus-marginal evaluation requires a fitted prior")
        model = prior

    checkpoint_metadata = {
        "model": args.model,
        "training_backend": args.training_backend,
        "use_bf16": args.use_bf16,
        "data_workers": args.data_workers,
        "data_prefetch_factor": args.data_prefetch_factor,
        "path_workers": args.path_workers,
        "path_checkpoint_interval": args.path_checkpoint_interval,
        "corpus_workers": args.corpus_workers,
        "hidden_dim": args.hidden_dim,
        "message_passing_steps": args.message_passing_steps,
        "rate_factorization": args.rate_factorization,
        "bond_representation": args.bond_representation,
        "ring_proposals": args.ring_proposals,
        "max_cycle_templates": args.max_cycle_templates,
        "max_attach_templates": args.max_attach_templates,
        "max_ear_templates": args.max_ear_templates,
        "max_ring_system_templates": args.max_ring_system_templates,
        "ring_electronic_mode": (
            args.ring_electronic_mode
            if args.training_backend == "factorized_marks"
            else "not_applicable"
        ),
        "ring_catalog": ring_catalog,
        "progress_stratification_fraction": args.progress_stratification_fraction,
        "teacher_ordering": args.teacher_ordering,
        "source_prior": args.source_prior,
        "tree_size_prior": args.tree_size_prior,
        "tree_couplings_per_target": args.tree_couplings_per_target,
        "tree_transport": args.tree_transport,
        "tree_source_prior": tree_source_prior,
        "seed": args.seed,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer_kind": (
            "adamw_decoupled_v1" if args.training_backend == "factorized_marks" else "adam"
        ),
        "warmup_steps": args.warmup_steps,
        "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
        "evaluation_every": args.evaluation_every,
        "evaluation_batch_size": args.evaluation_batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_relative_delta": (
            args.early_stopping_min_relative_delta
        ),
        "late_time_fraction": args.late_time_fraction,
        "operational_horizon": args.operational_horizon,
        "provenance_sha256": args.provenance_sha256,
    }
    expected = {
        key: checkpoint_metadata[key]
        for key in (
            "model",
            "training_backend",
            "hidden_dim",
            "message_passing_steps",
            "rate_factorization",
            "bond_representation",
            "ring_proposals",
            "ring_electronic_mode",
            "teacher_ordering",
            "source_prior",
            "tree_size_prior",
            "tree_couplings_per_target",
            "tree_transport",
        )
    }
    checkpoint_defaults = {
        "training_backend": "exact_fiber",
        "teacher_ordering": "sequential",
        "source_prior": "null",
        "tree_size_prior": "empirical",
        "tree_couplings_per_target": 1,
        "tree_transport": "primitive",
        "ring_electronic_mode": "factorized_local",
    }

    loaded_checkpoint = None
    resume_state = None
    selected_checkpoint_path = args.load_checkpoint or args.resume_checkpoint
    if selected_checkpoint_path is not None:
        if not isinstance(model, torch.nn.Module):
            raise ValueError("checkpoint loading requires a neural model")
        checkpoint_payload = torch.load(
            selected_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        mismatches = {
            key: (
                checkpoint_payload.get(key, checkpoint_defaults.get(key)),
                value,
            )
            for key, value in expected.items()
            if checkpoint_payload.get(key, checkpoint_defaults.get(key)) != value
        }
        if args.resume_checkpoint is not None:
            resume_expected = {
                key: checkpoint_metadata[key]
                for key in (
                    "training_steps",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "optimizer_kind",
                    "warmup_steps",
                    "minimum_learning_rate_fraction",
                    "evaluation_every",
                    "evaluation_batch_size",
                    "early_stopping_patience",
                    "early_stopping_min_relative_delta",
                    "late_time_fraction",
                    "operational_horizon",
                    "progress_stratification_fraction",
                    "use_bf16",
                    "provenance_sha256",
                )
            }
            mismatches.update(
                {
                    key: (checkpoint_payload.get(key), value)
                    for key, value in resume_expected.items()
                    if checkpoint_payload.get(key) != value
                }
            )
        if mismatches:
            raise ValueError(f"checkpoint/config mismatch: {mismatches}")
        if args.load_checkpoint is not None:
            loaded_checkpoint = checkpoint_payload
            model.load_state_dict(loaded_checkpoint["state_dict"])
            phase = "checkpoint_loaded"
        else:
            resume_state = checkpoint_payload
            model.load_state_dict(resume_state["current_state_dict"])
            phase = "recovery_checkpoint_loaded"
        print(
            json.dumps(
                {"phase": phase, "path": str(selected_checkpoint_path)},
                sort_keys=True,
            ),
            flush=True,
        )

    if isinstance(model, FactorizedTraceletRateModel):
        observed_validation = factorized_mark_metrics(
            model,
            validation_examples,
            use_bf16=args.use_bf16,
            microbatch_size=args.evaluation_batch_size,
        )
    else:
        observed_validation = tracelet_conditional_metrics(model, validation_examples)
    validation_phase = (
        "resume_validation" if resume_state is not None else "initial_validation"
    )
    print(json.dumps({"phase": validation_phase, **observed_validation}), flush=True)
    nonfinite_validation = {
        key: value
        for key, value in observed_validation.items()
        if isinstance(value, (int, float)) and not bool(np.isfinite(value))
    }
    if nonfinite_validation:
        raise RuntimeError(
            "initial validation contains non-finite metrics; refusing to train: "
            f"{nonfinite_validation}"
        )
    initial_validation = _validation_baseline_for_run(
        observed_validation,
        resume_state,
    )
    if resume_state is not None:
        print(
            json.dumps(
                {"phase": "initial_validation_restored", **initial_validation},
                sort_keys=True,
            ),
            flush=True,
        )
    nonfinite_baseline = {
        key: value
        for key, value in initial_validation.items()
        if not bool(np.isfinite(value))
    }
    if nonfinite_baseline:
        raise RuntimeError(
            "restored initial validation contains non-finite metrics: "
            f"{nonfinite_baseline}"
        )
    history = []
    recovery_path = (
        _recovery_path(args.checkpoint) if args.checkpoint is not None else args.resume_checkpoint
    )
    best_so_far_path = (
        _best_so_far_path(args.checkpoint) if args.checkpoint is not None else None
    )

    def save_recovery(training_state: dict[str, object]) -> None:
        if recovery_path is None:
            return
        payload = {
            **checkpoint_metadata,
            **training_state,
            "checkpoint_kind": "exact_training_recovery",
            "initial_validation": initial_validation,
        }
        _atomic_torch_save(payload, recovery_path)
        if best_so_far_path is not None:
            _atomic_torch_save(
                {
                    **checkpoint_metadata,
                    "checkpoint_kind": "interim_best_evaluation_model",
                    "state_dict": training_state["best_state_dict"],
                    "selected_validation": training_state["best_metrics"],
                    "initial_validation": initial_validation,
                    "completed_steps": training_state["completed_steps"],
                },
                best_so_far_path,
            )
            print(
                json.dumps(
                    {
                        "phase": "interim_best_checkpoint_saved",
                        "path": str(best_so_far_path),
                        "completed_steps": training_state["completed_steps"],
                        "selected_step": training_state["best_metrics"].get(
                            "selected_step"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "phase": "recovery_checkpoint_saved",
                    "path": str(recovery_path),
                    "completed_steps": training_state["completed_steps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if isinstance(model, FactorizedTraceletRateModel) and loaded_checkpoint is None:
        history, selected_validation = train_factorized_mark_model(
            model,
            train_records,
            validation_examples,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + 3,
            workers=args.data_workers,
            data_prefetch_factor=args.data_prefetch_factor,
            ring_electronic_mode=args.ring_electronic_mode,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=(args.progress_stratification_fraction),
            use_aromatic_bond_view=args.bond_representation == "aromatic",
            use_bf16=args.use_bf16,
            ring_catalog=ring_catalog,
            evaluation_interval=(args.evaluation_every if args.evaluation_every > 0 else None),
            warmup_steps=args.warmup_steps,
            minimum_learning_rate_fraction=(args.minimum_learning_rate_fraction),
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_relative_delta=(
                args.early_stopping_min_relative_delta
            ),
            progress_callback=lambda metrics: print(
                json.dumps({"phase": "training", **metrics}, sort_keys=True),
                flush=True,
            ),
            checkpoint_interval=args.recovery_every,
            checkpoint_callback=(save_recovery if recovery_path is not None else None),
            resume_state=resume_state,
            profile_timing=args.fast_split,
            evaluation_batch_size=args.evaluation_batch_size,
            initial_validation_metrics=observed_validation,
        )
    elif isinstance(model, torch.nn.Module) and loaded_checkpoint is None:
        training_fiber_cache = {}
        with tracelet_fiber_executor(
            args.fiber_workers,
            ring_catalog=ring_catalog,
        ) as training_fiber_executor:
            history, selected_validation = train_tracelet_conditional_model(
                model,
                train_records,
                validation_examples,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                action_kl_weight=args.action_kl_weight,
                hazard_tilt_weight=args.hazard_tilt_weight,
                seed=args.seed + 3,
                fiber_cache=training_fiber_cache,
                late_time_fraction=args.late_time_fraction,
                operational_horizon=args.operational_horizon,
                progress_stratification_fraction=(args.progress_stratification_fraction),
                ring_catalog=ring_catalog,
                causal_teachers=args.teacher_ordering == "causal_frontier",
                causal_cache={},
                progress_callback=lambda metrics: print(
                    json.dumps({"phase": "training", **metrics}, sort_keys=True),
                    flush=True,
                ),
                checkpoint_interval=args.recovery_every,
                checkpoint_callback=(save_recovery if recovery_path is not None else None),
                resume_state=resume_state,
                fiber_executor=training_fiber_executor,
            )
    elif loaded_checkpoint is not None:
        selected_validation = dict(loaded_checkpoint.get("selected_validation", initial_validation))
    else:
        selected_validation = dict(initial_validation)
        selected_validation["selected_step"] = 0.0
    if isinstance(model, FactorizedTraceletRateModel):
        final_test = factorized_mark_metrics(
            model,
            test_examples,
            use_bf16=args.use_bf16,
            microbatch_size=args.evaluation_batch_size,
        )
    else:
        final_test = tracelet_conditional_metrics(model, test_examples)
    if args.checkpoint is not None and isinstance(model, torch.nn.Module):
        _atomic_torch_save(
            {
                **checkpoint_metadata,
                "checkpoint_kind": "selected_evaluation_model",
                "state_dict": model.state_dict(),
                "selected_validation": selected_validation,
            },
            args.checkpoint,
        )
        print(
            json.dumps(
                {"phase": "checkpoint_saved", "path": str(args.checkpoint)},
                sort_keys=True,
            ),
            flush=True,
        )

    if args.skip_rollouts:
        report = {
            "seed": args.seed,
            "model": args.model,
            "training_backend": args.training_backend,
            "training": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "message_passing_steps": args.message_passing_steps,
                "data_workers": args.data_workers,
                "data_prefetch_factor": args.data_prefetch_factor,
                "path_workers": args.path_workers,
                "path_checkpoint_interval": args.path_checkpoint_interval,
                "corpus_workers": args.corpus_workers,
                "use_bf16": args.use_bf16,
                "optimizer_kind": (
                    "adamw_decoupled_v1" if args.training_backend == "factorized_marks" else "adam"
                ),
                "warmup_steps": args.warmup_steps,
                "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
                "evaluation_every": args.evaluation_every,
                "evaluation_batch_size": args.evaluation_batch_size,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_relative_delta": (
                    args.early_stopping_min_relative_delta
                ),
                "history": history,
                "device": str(device),
            },
            "initial_validation": initial_validation,
            "selected_validation": selected_validation,
            "final_test": final_test,
            "generated_nonnull_smiles": 0,
            "rollout": {"skipped": True},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    progress_interval = max(args.rollout_samples // 10, 1)

    def report_sampling_progress(completed: int, total: int) -> None:
        if completed % progress_interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "phase": "sampling",
                        "completed": completed,
                        "total": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.load_rollouts is not None:
        rollout_payload = torch.load(args.load_rollouts, weights_only=False)
        rollouts = tuple(rollout_payload["rollouts"])
        if len(rollouts) != args.rollout_samples:
            raise ValueError(
                "rollout cache size does not match --rollout-samples: "
                f"{len(rollouts)} != {args.rollout_samples}"
            )
        print(
            json.dumps(
                {"phase": "rollouts_loaded", "path": str(args.load_rollouts)},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        if (
            isinstance(model, torch.nn.Module)
            and args.rollout_workers > 1
            and next(model.parameters()).device.type != "cpu"
        ):
            model = model.to("cpu")
            print(
                json.dumps(
                    {
                        "phase": "sampling_device_handoff",
                        "from": str(device),
                        "to": "cpu",
                        "workers": args.rollout_workers,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        rollouts = sample_tracelet_ancestral_many(
            model,
            seed=args.seed + 4,
            samples=args.rollout_samples,
            workers=args.rollout_workers,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            max_events=args.max_events,
            torch_threads_per_worker=args.torch_threads,
            progress_callback=report_sampling_progress,
            source_prior=tree_source_prior,
        )
        if args.rollout_cache is not None:
            args.rollout_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "rollouts": rollouts,
                    "seed": args.seed + 4,
                    "n_slots": args.max_atoms,
                    "operational_horizon": args.operational_horizon,
                    "time_step": args.time_step,
                    "max_events": args.max_events,
                    "source_prior": args.source_prior,
                    "tree_size_prior": args.tree_size_prior,
                    "tree_source_prior": tree_source_prior,
                },
                args.rollout_cache,
            )
            print(
                json.dumps(
                    {"phase": "rollouts_saved", "path": str(args.rollout_cache)},
                    sort_keys=True,
                ),
                flush=True,
            )
    generated_smiles = tuple(
        text
        for item in rollouts
        if (text := molecular_graph_to_smiles(item.final_state)) is not None
    )
    event_counts = Counter(rule for item in rollouts for rule in item.event_rules)
    report: dict[str, object] = {
        "seed": args.seed,
        "model": args.model,
        "source_prior": {
            "kind": args.source_prior,
            "tree_size_prior": (
                args.tree_size_prior if args.source_prior == "carbon_tree" else None
            ),
            "sizes": tree_source_prior.sizes if tree_source_prior is not None else None,
            "probabilities": (
                tree_source_prior.probabilities if tree_source_prior is not None else None
            ),
            "couplings_per_target": (
                args.tree_couplings_per_target if args.source_prior == "carbon_tree" else None
            ),
            "transport": (args.tree_transport if args.source_prior == "carbon_tree" else None),
        },
        "frequency_prior_role": (
            "standalone_baseline"
            if args.model == "prior_only"
            else "initial_base_measure"
            if args.model == "prior_tilted"
            else "none"
        ),
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "eligible": split.eligible_molecules,
            "scanned_lines": split.scanned_lines,
            "selection": "first_eligible_screen" if args.fast_split else "full_corpus_random",
            "max_atoms": args.max_atoms,
        },
        "training": {
            "steps": (
                args.steps
                if isinstance(model, torch.nn.Module) and loaded_checkpoint is None
                else None
                if loaded_checkpoint is not None
                else 0
            ),
            "evaluation_only": loaded_checkpoint is not None,
            "loaded_checkpoint": (
                str(args.load_checkpoint) if args.load_checkpoint is not None else None
            ),
            "resumed_from": (
                str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
            ),
            "recovery_checkpoint": (str(recovery_path) if recovery_path is not None else None),
            "recovery_every": args.recovery_every,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim if isinstance(model, torch.nn.Module) else None,
            "message_passing_steps": (
                args.message_passing_steps if isinstance(model, torch.nn.Module) else None
            ),
            "rate_factorization": (
                args.rate_factorization if isinstance(model, torch.nn.Module) else None
            ),
            "training_backend": args.training_backend,
            "data_workers": args.data_workers,
            "data_prefetch_factor": args.data_prefetch_factor,
            "path_workers": args.path_workers,
            "path_checkpoint_interval": args.path_checkpoint_interval,
            "corpus_workers": args.corpus_workers,
            "use_bf16": args.use_bf16,
            "optimizer_kind": (
                "adamw_decoupled_v1" if args.training_backend == "factorized_marks" else "adam"
            ),
            "warmup_steps": args.warmup_steps,
            "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
            "evaluation_every": args.evaluation_every,
            "evaluation_batch_size": args.evaluation_batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_relative_delta": (
                args.early_stopping_min_relative_delta
            ),
            "bond_representation": (
                args.bond_representation if isinstance(model, torch.nn.Module) else None
            ),
            "action_kl_weight": args.action_kl_weight,
            "hazard_tilt_weight": args.hazard_tilt_weight,
            "progress_stratification_fraction": (args.progress_stratification_fraction),
            "teacher_ordering": args.teacher_ordering,
            "history": history,
            "device": str(device),
        },
        "initial_validation": initial_validation,
        "selected_validation": selected_validation,
        "final_test": final_test,
        "rollout": corpus_rollout_metrics(
            rollouts,
            train_smiles=split.train,
            reference_smiles=split.test,
        ),
        "rollout_event_counts": dict(sorted(event_counts.items())),
        "generated_nonnull_smiles": len(generated_smiles),
        "generated_smiles": generated_smiles,
        "sampler": {
            "target_available": False,
            "beam_search": False,
            "ancestral_ctmc": True,
            "operational_horizon": args.operational_horizon,
            "time_step": args.time_step,
            "max_events": args.max_events,
            "workers": args.rollout_workers,
            "seed_scheme": "seed_sequence_per_trajectory_v1",
        },
    }
    if generated_smiles:
        report["generated_ring_taxonomy"] = ring_taxonomy_report(generated_smiles)
    if args.quality_metrics and generated_smiles:
        quality_reference = split.test
        if args.quality_reference_file is not None:
            quality_reference = _read_smiles_file(
                args.quality_reference_file,
                limit=args.quality_reference_limit,
            )
        report["molecular_quality"] = molecular_quality_report(
            generated_smiles,
            reference_smiles=quality_reference,
            train_smiles=split.train,
            include_fcd=args.include_fcd,
            fcd_reference_limit=args.quality_reference_limit,
            fcd_generated_limit=(args.fcd_generated_limit or None),
            fcd_device=args.fcd_device,
            seed=args.seed + 5,
        )
        report["molecular_quality_reference"] = {
            "file": (
                str(args.quality_reference_file)
                if args.quality_reference_file is not None
                else None
            ),
            "source": "external" if args.quality_reference_file is not None else "split.test",
            "count": len(quality_reference),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _read_smiles_file(path: Path, *, limit: int) -> tuple[str, ...]:
    smiles = []
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            smiles.append(fields[0])
            if len(smiles) >= limit:
                break
    if not smiles:
        raise ValueError(f"no SMILES found in quality reference file: {path}")
    return tuple(smiles)


if __name__ == "__main__":
    main()
