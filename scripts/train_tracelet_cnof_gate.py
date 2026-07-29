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
    CNOF_VOCABULARY,
    IDX_TO_ELEMENT,
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, load_organic_corpus_split
from compose_v4.experiments.hierarchical_sampler import build_layered_sampler
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.experiments.cnof_conditional import PathRecord, corpus_rollout_metrics
from compose_v4.experiments import zero_mixture_instrumentation as _zmi
from compose_v4.experiments.analogue_prior import build_analogue_prior_records
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records
from compose_v4.rewrite.source_corruption import TEACHER_REPRESENTABILITY_FILTER_VERSION
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
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
    attach_property_conditions,
    factorized_mark_metrics,
    sample_factorized_mark_batch,
    train_factorized_mark_model,
)
from compose_v4.experiments.factorized_mark_priors import (
    fit_factorized_mark_empirical_priors,
)
from compose_v4.experiments.molecular_property_conditioning import (
    fit_property_condition_normalizer,
    standardized_record_conditions,
)
from compose_v4.experiments.training_support_cache import (
    RING_SUPPORT_SEMANTICS_VERSION,
    ShardedTrainingSupportCache,
)
from compose_v4.experiments.training_support_compiler import (
    attach_ring_teacher_semantic_certificates,
    compile_training_support_shards,
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
    MARK_RULE_NAMES,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    OperatorCapabilities,
    SparseBinaryRows,
)
from compose_v4.model.device import resolve_torch_device
from compose_v4.rewrite.typed_ring_catalog import (
    RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
    TypedRingCatalog,
    build_typed_ring_catalog,
    build_typed_ring_catalog_from_paths,
    ring_catalog_fingerprint,
)
from compose_v4.rewrite.ring_system_fiber import structured_ring_trace_supported
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

# RING_CORE_V1 production ring-catalog seeds (deterministic; reproduces the production fingerprint
# 639ff6078c32d43c). Under the zero-mixture --scaled-manifest path the catalog is reconstructed from THESE,
# independent of any de-novo corpus path or B's checkpoint (owner mandate §2).
_RING_CORE_CATALOG_SEEDS = ("c1ccccc1", "c1ccncc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1")


TRANSPORT_SUPPORT_PROJECTION_VERSION = 3
# v2: the eval/validation/test batch now enumerates the editing families (ring_system_restate/delete,
# cyclic graft) to match training's dynamic candidates -- v1 caches (built without these) are invalid.
EVALUATION_BATCH_CACHE_FORMAT_VERSION = 2
TRAINING_SUPPORT_STREAM_FORMAT_VERSION = 1

# Stable hash of the operator-registry family list -- part of the eval-cache key so a cache built under a
# different operator registry (family set) is never reused (capability/operator invalidation, not just
# FORMAT_VERSION).
MARK_RULE_NAMES_HASH = hashlib.sha256("|".join(MARK_RULE_NAMES).encode()).hexdigest()[:16]


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


def _step_snapshot_path(checkpoint: Path, completed_steps: int) -> Path:
    """Step-stamped snapshot (``checkpoint.step1500.pt``). The rolling recovery file OVERWRITES itself, so
    a post-hoc checkpoint-selection rule (choose the step minimizing a held-out metric) has nothing to
    select from once a run ends. Opt-in via --snapshot-checkpoints; off by default (byte-identical)."""
    return checkpoint.with_name(f"{checkpoint.stem}.step{int(completed_steps)}{checkpoint.suffix}")


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


def _compatible_checkpoint_initialization(
    target_state: dict[str, torch.Tensor],
    checkpoint_payload: dict[str, object],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...], tuple[str, ...]]:
    """Transfer exactly shape-compatible tensors and retain new-head initialization.

    This is intentionally narrower than ``strict=False`` checkpoint loading.  A
    source tensor is accepted only when both its fully qualified parameter name
    and shape match the current model.  Changed template tables and newly added
    heads therefore keep the current model's deterministic initialization.
    """

    source_state = checkpoint_payload.get("state_dict")
    if source_state is None:
        source_state = checkpoint_payload.get("best_state_dict")
    if not isinstance(source_state, dict) or not source_state:
        raise ValueError(
            "compatible initialization checkpoint lacks state_dict or best_state_dict"
        )
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in source_state.items()
    ):
        raise ValueError("compatible initialization state must map names to tensors")

    initialized = {name: value.detach().clone() for name, value in target_state.items()}
    transferred: list[str] = []
    retained: list[str] = []
    for name, target in target_state.items():
        source = source_state.get(name)
        if isinstance(source, torch.Tensor) and source.shape == target.shape:
            initialized[name] = (
                source.detach().to(device=target.device, dtype=target.dtype).clone()
            )
            transferred.append(name)
        else:
            retained.append(name)
    if not transferred:
        raise ValueError("compatible initialization found no shape-compatible tensors")
    return initialized, tuple(transferred), tuple(retained)


# Categorical output heads whose row axis is the atom (element, valence) vocabulary. When the vocabulary is
# widened (CNOF 4 -> ORGANIC 15) these heads change shape, so the shape-exact compatible transfer would leave
# them entirely fresh. Semantic partial transfer instead copies the rows of every SHARED class (matched by
# (element, valence) LABEL, not position) and keeps fresh init only for genuinely new classes.
#   "row"        : tensor rows are the vocab classes in order        (Linear weight/bias: n_classes rows).
#   "role_major" : tensor is reshape(k, n_classes, d) -> row r*n_classes + c is (role r, class c)
#                  (grow_option Embedding: 3*n_classes rows; see factorized_tracelet_rate_model.py:3550).
_ATOM_VOCAB_HEAD_LAYOUT: dict[str, tuple[str, int]] = {
    "restate_head.weight": ("row", 1),
    "restate_head.bias": ("row", 1),
    "grow_root_head.weight": ("row", 1),
    "grow_root_head.bias": ("row", 1),
    "grow_option.weight": ("role_major", 3),
}


def _semantic_row_map(
    source_classes: tuple, dest_classes: tuple
) -> list[int | None]:
    """Map dest row -> source row by semantic class LABEL (never raw position). ``None`` for a genuinely new
    destination class. Fails loudly on duplicate labels or a source class absent from the destination (which
    would silently drop learned rows)."""

    if len(set(source_classes)) != len(source_classes):
        raise ValueError("source vocabulary has duplicate class labels")
    if len(set(dest_classes)) != len(dest_classes):
        raise ValueError("destination vocabulary has duplicate class labels")
    dest_index = {label: j for j, label in enumerate(dest_classes)}
    missing = [label for label in source_classes if label not in dest_index]
    if missing:
        raise ValueError(
            f"source vocabulary classes absent from destination (would drop learned rows): {missing}"
        )
    source_index = {label: i for i, label in enumerate(source_classes)}
    return [source_index.get(label) for label in dest_classes]


def _semantic_partial_checkpoint_initialization(
    target_state: dict[str, torch.Tensor],
    checkpoint_payload: dict[str, object],
    *,
    source_classes: tuple,
    dest_classes: tuple,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...], tuple[str, ...], list[dict], str]:
    """Shape-exact transfer PLUS semantic row-copy for the widened (element, valence) heads.

    Every shape-identical tensor transfers as in ``_compatible_checkpoint_initialization`` (body, embeddings,
    hazard, non-widened heads). For each widened atom-vocabulary head, the rows of every shared class copy
    from the source by label; genuinely new classes keep the model's fresh initialization. Returns
    ``(initialized_state, copied_exact, fresh_or_partial, row_table, transfer_map_hash)``. The row table is a
    per-row audit (head, class, source row, dest row, status, source/dest checksum)."""

    initialized, transferred, retained = _compatible_checkpoint_initialization(
        target_state, checkpoint_payload
    )
    source_state = checkpoint_payload.get("state_dict")
    if source_state is None:
        source_state = checkpoint_payload.get("best_state_dict")
    row_map = _semantic_row_map(source_classes, dest_classes)  # dest_j -> source_i | None
    n_dest, n_src = len(dest_classes), len(source_classes)
    row_table: list[dict] = []

    def _cksum(tensor: torch.Tensor) -> str:
        # .cpu() BEFORE .numpy(): the warm-start runs on the training device (cuda on A100), and a CUDA tensor
        # cannot convert to numpy directly. The CPU dry-launch could not catch this (device=cpu makes .numpy()
        # work); it fired only on the A100 bounded run's semantic-transfer row-table checksum.
        return hashlib.sha256(
            tensor.detach().cpu().to(torch.float64).numpy().tobytes()
        ).hexdigest()[:12]

    for name, (layout, k) in _ATOM_VOCAB_HEAD_LAYOUT.items():
        if name not in retained or name not in source_state:
            continue  # not widened here (same vocab) or absent -> nothing semantic to do
        src = source_state[name]
        dst = initialized[name].detach().clone()
        record_rows = name.endswith(".weight")  # audit one representative tensor per head (skip bias echo)
        for r in range(k):
            for j, i in enumerate(row_map):
                dest_row = r * n_dest + j
                if i is None:
                    if record_rows:
                        row_table.append({
                            "head": name, "class": list(dest_classes[j]), "role": r,
                            "source_row": None, "dest_row": dest_row,
                            "status": "FRESH_NEW_CLASS", "dest_checksum": _cksum(dst[dest_row]),
                        })
                    continue
                source_row = r * n_src + i
                dst[dest_row] = src[source_row].to(device=dst.device, dtype=dst.dtype)
                if record_rows:
                    row_table.append({
                        "head": name, "class": list(dest_classes[j]), "role": r,
                        "source_row": source_row, "dest_row": dest_row,
                        "status": "COPIED_WITH_INDEX_MAP",
                        "source_checksum": _cksum(src[source_row]), "dest_checksum": _cksum(dst[dest_row]),
                    })
        initialized[name] = dst

    copied_exact = tuple(sorted(transferred))
    fresh_or_partial = tuple(sorted(retained))  # widened heads are now PARTIALLY copied (see row_table)
    map_payload = {
        "source_classes": [list(c) for c in source_classes],
        "dest_classes": [list(c) for c in dest_classes],
        "row_map": row_map,
        "heads": sorted(_ATOM_VOCAB_HEAD_LAYOUT),
    }
    transfer_map_hash = hashlib.sha256(
        json.dumps(map_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return initialized, copied_exact, fresh_or_partial, row_table, transfer_map_hash


_SEMANTIC_TRANSFER_IMPL_VERSION = "semantic_partial_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vocabulary_hash(vocabulary) -> str:
    return hashlib.sha256(
        json.dumps([list(cls) for cls in vocabulary.classes], sort_keys=True).encode()
    ).hexdigest()[:16]


def _path_cache_fingerprint(signature: dict[str, object]) -> str:
    """Return a stable identifier without copying the full split into every shard."""

    serialized = json.dumps(
        signature,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _ring_support_semantics_mode(ring_electronic_mode: str) -> str:
    """Collapse neural scoring variants that share identical exact support."""

    if ring_electronic_mode in {
        "factorized_local",
        "factorized_contextual",
    }:
        return "factorized_local"
    if ring_electronic_mode == "catalog_exact":
        return "catalog_exact"
    raise ValueError("unknown ring electronic decoding mode")


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
    property_conditioning: dict[str, object] | None = None,
    operator_capability_fingerprint: str | None = None,
    operator_registry_hash: str | None = None,
) -> dict[str, object]:
    """Identify every scientific choice that fixes the evaluation tensors."""

    return {
        "format_version": EVALUATION_BATCH_CACHE_FORMAT_VERSION,
        "path_cache_fingerprint": _path_cache_fingerprint(path_cache_signature),
        # The editing-family enumeration capabilities + operator-registry semantics that determine which
        # dynamic candidates the eval batch enumerates. A cache built under one capability set must never be
        # reused under another (a de-novo-capability cache has no editing candidates and would fail scoring).
        "operator_capability_fingerprint": operator_capability_fingerprint,
        "operator_registry_hash": operator_registry_hash,
        "training_backend": str(training_backend),
        "seed": int(seed),
        "validation_examples": int(validation_examples),
        "test_examples": int(test_examples),
        "late_time_fraction": float(late_time_fraction),
        "operational_horizon": float(operational_horizon),
        "progress_stratification_fraction": float(progress_stratification_fraction),
        "bond_representation": str(bond_representation),
        "ring_electronic_mode": _ring_support_semantics_mode(
            ring_electronic_mode
        ),
        "property_conditioning": property_conditioning,
    }


_CNOF_SYMBOLS = frozenset({"C", "N", "O", "F"})


def _record_cold_elements(record):
    """Non-CNOF element symbols present in a record's trace target (drives the sampler cold-element floors)."""
    target = record.path.trace.target
    real = is_element(target.atom_types)
    return {IDX_TO_ELEMENT[int(t)] for t in target.atom_types[real]} - _CNOF_SYMBOLS


def _build_scaled_manifest_sampler(train_records, *, denovo_keep, n_corruption, manifest_path, seed):
    """Build the hierarchical record sampler from a scaled manifest + the layer boundaries of train_records
    (denovo[:denovo_keep] + corruption[n_corruption] + mmp[rest]). Returns None if the manifest has no
    positive-weight layer. Tags align 1:1 with train_records (the dataset guard enforces it)."""
    _zmi.bump("edit_manifest_loads")
    manifest = json.loads(Path(manifest_path).read_text())
    locked = manifest.get("locked_mixture") or {}
    weights = {k: float(v) for k, v in (locked.get("layer_weights") or {}).items() if float(v) > 0}
    if not weights:
        return None
    edges = tuple(locked.get("curriculum_bin_edges") or (1, 2, 4))
    floor = float(locked.get("cold_element_floor", 0.0))
    records_by_layer: dict[str, tuple] = {}
    if denovo_keep:
        records_by_layer["denovo"] = tuple(train_records[:denovo_keep])
    records_by_layer["corruption"] = tuple(train_records[denovo_keep:denovo_keep + n_corruption])
    records_by_layer["mmp"] = tuple(train_records[denovo_keep + n_corruption:])
    return build_layered_sampler(
        records_by_layer, layer_weights=weights, path_length_bins=edges,
        cold_element_floor=floor, cold_elements_of=_record_cold_elements, seed=seed)


def _load_precompiled_corpus(args, *, partition: str, seed: int):
    """Load one partition of the validated precompiled corpus and its explicit three-layer sampler.

    This is the production data path. It NEVER calls build_corrupted_prior_records /
    build_cycle_op_records / build_analogue_prior_records: those regenerate a few hundred sources per
    launch (the DATA_STARVED_BASELINE), carry no partition discipline and no contract hashes. Any
    contract drift, missing shard or empty positively-weighted layer aborts here -- before the GPU.
    """
    from compose_v4.data.production_edit_corpus import (
        load_production_edit_corpus,
        measure_realized_layer_frequencies,
    )
    from ring_core_identity import (
        LAYER_WEIGHTS_HASH,
        PRODUCTION_LAYER_WEIGHTS,
        recompute_layer_weights_hash,
    )

    if recompute_layer_weights_hash() != LAYER_WEIGHTS_HASH:
        raise SystemExit(
            "production layer weights drifted from their locked hash "
            f"({recompute_layer_weights_hash()} != {LAYER_WEIGHTS_HASH}); refusing to train"
        )
    root = Path(args.precompiled_corpus)
    contract = json.loads((root / "BUILD_COMPLETE.json").read_text()).get("contract", {})
    corpus = load_production_edit_corpus(
        root,
        mmp_pool_path=Path(args.precompiled_mmp_pool),
        packed_root=(Path(args.packed_corpus) if args.packed_corpus else None),
        packed_mmp_root=(Path(args.packed_mmp_corpus) if args.packed_mmp_corpus else None),
        require_packed_mmp=bool(args.packed_corpus),
        partition=partition,
        layer_weights=PRODUCTION_LAYER_WEIGHTS,
        expected_contract=contract,
        path_length_bins=(5, 9, 13),
        checkpoint_interval=args.path_checkpoint_interval,
        seed=seed,
    )
    storage = corpus.provenance["layer_storage"]
    if args.packed_corpus:
        unpacked = sorted(k for k, v in storage.items() if v != "packed")
        if unpacked:
            raise SystemExit(
                f"BENCHMARK_CONTRACT violation: layers not served from packed storage: {unpacked} "
                f"(storage={storage}); refusing to run on a fallback path"
            )
    realized = measure_realized_layer_frequencies(corpus, draws=20000, seed=seed + 1)
    print(
        json.dumps(
            {
                "phase": "precompiled_corpus",
                "partition": partition,
                "layer_storage": corpus.provenance["layer_storage"],
                "records_by_layer": corpus.provenance["records_by_layer"],
                "transitions_by_layer": corpus.provenance["transitions_by_layer"],
                "configured_layer_weights": dict(PRODUCTION_LAYER_WEIGHTS),
                "realized_layer_frequency": {k: round(v, 5) for k, v in realized.items()},
                "layer_weights_hash": LAYER_WEIGHTS_HASH,
                "mmp_partition_filter": corpus.provenance["mmp_partition_filter"],
                "reducer_checksum": corpus.provenance["reducer_checksum"],
                "authorization": corpus.provenance["authorization"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return corpus


def _build_ring_core_seed_ring_catalog(max_atoms: int) -> TypedRingCatalog:
    """Reconstruct the RING_CORE_V1 production ring catalog DETERMINISTICALLY from the fixed seeds -- NOT from
    de-novo corpus paths and NOT from B's checkpoint (owner mandate §2). Same 5-seed builder the warm-start
    dry-run + freeze manifest used (fingerprint 639ff6078c32d43c). Reconstructing the catalog from 5 fixed
    seed molecules is NOT a de-novo training-dataset instantiation."""

    def _seed_trace(smi: str):
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), max_atoms)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=max_atoms
        )
        return compile_carbon_tree_to_target(
            source, target, use_bond_reroute=True, align_source=True
        )

    _zmi.bump("production_catalog_constructions")
    return build_typed_ring_catalog(tuple(_seed_trace(s) for s in _RING_CORE_CATALOG_SEEDS))


def _training_support_cache_signature(
    path_cache_signature: dict[str, object],
    *,
    seed: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    ring_electronic_mode: str,
    corrupted_prior_mix: bool = False,
    organic_vocabulary: bool = False,
    analogue_trace_pool: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
) -> dict[str, object]:
    """Identify the infinite deterministic row stream independent of its horizon."""

    return {
        "format_version": TRAINING_SUPPORT_STREAM_FORMAT_VERSION,
        "ring_support_semantics_version": RING_SUPPORT_SEMANTICS_VERSION,
        "path_cache_fingerprint": _path_cache_fingerprint(path_cache_signature),
        "seed": int(seed),
        "late_time_fraction": float(late_time_fraction),
        "operational_horizon": float(operational_horizon),
        "progress_stratification_fraction": float(progress_stratification_fraction),
        "ring_electronic_mode": _ring_support_semantics_mode(
            ring_electronic_mode
        ),
        # The mixed (de-novo + corrupted-prior) records are a different deterministic row stream than
        # B's carbon-only cache; this discriminator forces a recompile instead of a silent misalignment.
        "corrupted_prior_mix": bool(corrupted_prior_mix),
        # Organic (15-class) teacher marks are a different deterministic row stream than the CNOF ones.
        "organic_vocabulary": bool(organic_vocabulary),
        # Real analogue-pair (A->B / B->A MMP) traces are yet another deterministic row stream.
        "analogue_trace_pool": bool(analogue_trace_pool),
        # Compositional ring-op (cycle_open/cycle_close) supervision is another deterministic row stream.
        "cycle_op_mix": bool(cycle_op_mix),
        # Disabling the legacy grow macro drops grow support rows -> a different support stream.
        "disable_ring_grow_macro": bool(disable_ring_grow_macro),
    }


def _evaluation_batch_cache_path(
    cache_dir: Path,
    signature: dict[str, object],
) -> Path:
    fingerprint = _path_cache_fingerprint(signature)
    return (
        cache_dir / f"evaluation-batches-v{EVALUATION_BATCH_CACHE_FORMAT_VERSION}-{fingerprint}.pt"
    )


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
        if not hasattr(batch, "ring_teacher_semantic_certificates"):
            object.__setattr__(batch, "ring_teacher_semantic_certificates", None)
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
        "--empirical-mark-prior-mode",
        choices=("none", "corpus_residual_v1"),
        default="none",
        help=(
            "add fixed corpus mark-category log probabilities underneath the "
            "learned atom/bond/ring-electronic residual logits"
        ),
    )
    parser.add_argument(
        "--empirical-mark-prior-smoothing",
        type=float,
        default=1.0,
        help="positive additive count smoothing for corpus_residual_v1",
    )
    parser.add_argument(
        "--ring-family-mass-mode",
        choices=("boolean", "catalog_topology_local_support"),
        default="boolean",
        help=(
            "Boolean ring-family gating or catalog topology-group prior mass "
            "conditioned on the fast local structural support DP"
        ),
    )
    parser.add_argument(
        "--ring-template-factorization",
        choices=("flat", "topology_cycle_hierarchical"),
        default="flat",
        help=(
            "Flat complete-template softmax or an explicit learned hierarchy "
            "over topology class and cycle-size group before template detail"
        ),
    )
    parser.add_argument(
        "--property-condition",
        action="append",
        choices=("qed", "logp", "molecular_weight"),
        default=[],
        help=(
            "standardized endpoint property supplied to the marked-rate model; "
            "repeat for multi-property conditioning"
        ),
    )
    parser.add_argument(
        "--condition-dropout-probability",
        type=float,
        default=0.15,
        help="classifier-free all-target dropout used during conditioned training",
    )
    parser.add_argument(
        "--ring-proposals",
        choices=("generic_carbon", "typed_catalog"),
        default="generic_carbon",
    )
    parser.add_argument(
        "--ring-electronic-mode",
        choices=(
            "factorized_local",
            "factorized_contextual",
            "catalog_exact",
        ),
        default="factorized_local",
        help=(
            "independent semantic labels, contextual semantic labels with "
            "learned composition/adjacency potentials, or the diagnostic "
            "observed-alias decoder"
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
        "--trainable-parameter-scope",
        choices=(
            "all",
            "chemistry_marks_only",
            "ring_topology_only",
            "chemistry_and_topology",
        ),
        default="all",
        help=(
            "all parameters, only atom/bond and ring-electronic mark heads, "
            "only the shared ring topology-and-cycle selector, or the union of "
            "the chemistry-mark and ring-topology heads"
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linear AdamW warmup updates; factorized marked training only",
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help=(
            "learning-rate schedule horizon; 0 uses --steps. Set larger than "
            "--steps for a bounded pilot of a longer intended optimization run"
        ),
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
        help=("shared directory for content-addressed fixed validation/test batches"),
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
        "--training-support-cache-dir",
        type=Path,
        help="shared root for content-addressed sparse training-support shards",
    )
    parser.add_argument(
        "--training-support-shard-size",
        type=int,
        default=16000,
        help="deterministic training rows per atomically published support shard",
    )
    parser.add_argument(
        "--training-support-workers",
        type=int,
        default=0,
        help="CPU processes used only by the sparse support compiler",
    )
    parser.add_argument(
        "--training-support-microbatch-size",
        type=int,
        default=8,
        help="row tasks assigned dynamically to each support worker",
    )
    parser.add_argument(
        "--training-support-prefetch-factor",
        type=int,
        default=2,
        help="support microbatches queued per compiler worker",
    )
    parser.add_argument(
        "--compile-training-support-start-step",
        type=int,
        default=0,
        help=(
            "first training step in the deterministic support range; must begin "
            "at a support-shard boundary"
        ),
    )
    parser.add_argument(
        "--compile-training-support-steps",
        type=int,
        default=0,
        help=(
            "number of training steps to compile from the requested start and "
            "then exit; 0 disables compilation"
        ),
    )
    parser.add_argument(
        "--require-training-support-cache",
        action="store_true",
        help="forbid online support chemistry in the training process",
    )
    parser.add_argument(
        "--training-support-wait-seconds",
        type=float,
        default=0.0,
        help="maximum time a trainer worker waits for a concurrently compiled shard",
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
        "--initialize-checkpoint",
        type=Path,
        help=(
            "initialize neural weights from a compatible selected checkpoint, "
            "then train with a fresh optimizer and the current schedule"
        ),
    )
    parser.add_argument(
        "--initialize-compatible-checkpoint",
        type=Path,
        help=(
            "initialize only tensors whose full parameter name and shape match; "
            "new or resized heads retain their current initialization"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume interrupted training from a periodic recovery checkpoint",
    )
    parser.add_argument(
        "--allow-resume-provenance-mismatch",
        action="store_true",
        help=(
            "permit an explicitly audited implementation-only acceleration to "
            "resume an otherwise configuration-identical recovery checkpoint"
        ),
    )
    parser.add_argument(
        "--allow-resume-step-extension",
        action="store_true",
        help=(
            "permit only training_steps to increase when resuming an otherwise "
            "configuration-identical recovery checkpoint; a shorter explicit "
            "schedule remains at its minimum learning-rate floor"
        ),
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=200,
        help="training steps between exact recovery snapshots; 0 disables them",
    )
    parser.add_argument(
        "--snapshot-checkpoints",
        action="store_true",
        help=(
            "also persist a step-stamped checkpoint (checkpoint.step<N>.pt) at every recovery point, so a "
            "post-hoc checkpoint-selection rule can score several steps; the rolling recovery file "
            "overwrites itself and would otherwise leave only the final step"
        ),
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
    parser.add_argument(
        "--corrupted-prior-mix",
        action="store_true",
        help="append corrupted-molecule source-prior (B-edit) records to train_records and enable the "
        "ring_system_restate + cyclic bond_reroute editing marks (fine-tune B -> B-edit)",
    )
    parser.add_argument(
        "--cycle-op-mix",
        action="store_true",
        help="enable the compositional cycle_close/cycle_open ring families + append real-molecule "
        "ring-bond (cycle_open/cycle_close) supervision to the edit records (support-complete ring "
        "growth/closure). Requires --corrupted-prior-mix.",
    )
    parser.add_argument(
        "--disable-ring-grow-macro",
        action="store_true",
        help="RING_CORE_V1: disable the legacy whole-ring ring_system_grow macro so ring ADDITION is "
        "purely compositional (cycle_close). The grow head params are retained (warm-start-safe); the "
        "family is masked dead (never sampled/taught). Typically paired with --cycle-op-mix.",
    )
    parser.add_argument(
        "--dry-launch",
        action="store_true",
        help="CPU dry-launch: run the REAL trainer through model/warm-start/dataloader/first forward + one "
        "editing-validation forward, then EXIT before backward/optimizer (optimizer_steps=0). Serializes the "
        "zero-mixture instrumentation counters -- proves the --scaled-manifest path reaches the first editing "
        "batch with a finite loss + zero de-novo work. No parameter is updated.",
    )
    parser.add_argument("--dry-launch-output", default="", help="path to write the dry-launch JSON artifact")
    parser.add_argument(
        "--organic-vocabulary",
        action="store_true",
        help="predict/edit the whole drug-like organic subset (C,N,O,F,S,P,Cl,Br,I,B) via 15 "
        "(element, valence) atom classes instead of CNOF-only; the micro editing path + the corruption "
        "go organic. Ring atom GENERATION stays CNOF (heteroaromatic rings are read + edited, not built "
        "de-novo). Warm-start B with --initialize-compatible-checkpoint (body/embeddings transfer, the "
        "wider atom-type heads keep fresh init).",
    )
    parser.add_argument(
        "--corrupted-prior-count",
        type=int,
        default=6000,
        help="number of leads to corrupt for the B-edit mix; bounds the fine-tune data-gen cost and, "
        "with a matched de-novo subset, gives a ~50/50 edit ratio under uniform sampling",
    )
    parser.add_argument(
        "--analogue-trace-pool",
        type=str,
        default=None,
        help="path to a verified analogue-pair trace pool (scripts/build_analogue_trace_pool.py output, "
        "JSONL): appends real-molecule A->B / B->A MMP edit traces as GM PathRecords (Layer 1 of the "
        "universal-edit-prior mixture); omit for corruption-only",
    )
    parser.add_argument(
        "--analogue-trace-count",
        type=int,
        default=0,
        help="cap on analogue traces appended (0 = the whole pool); tune against --corrupted-prior-count "
        "to set the corruption:MMP mixture proportions",
    )
    parser.add_argument(
        "--scaled-manifest",
        type=str,
        default=None,
        help="path to a scaled B-edit data manifest (scripts/build_scaled_edit_manifest.py output): drives "
        "the hierarchical training sampler (LOCKED layer weights + curriculum bins + cold-element floors) "
        "instead of uniform sampling. Requires --corrupted-prior-mix; drops the de-novo retention subset "
        "(the mixture is manifest-controlled, de-novo retained via the warm-start).",
    )
    parser.add_argument(
        "--benchmark-steps", type=int, default=0,
        help="OPERATIONAL throughput benchmark: run the REAL training loop for N steps and emit a "
        "steady-state timing report, then exit. The checkpoint is meaningless and must be discarded; the "
        "loss must NOT be used for scientific interpretation.",
    )
    parser.add_argument(
        "--benchmark-warmup", type=int, default=30,
        help="steps excluded from the steady-state statistics (kernel autotuning, first shard open, "
        "dataloader cache population all land here)",
    )
    parser.add_argument("--benchmark-output", default="", help="path for the benchmark JSON artifact")
    parser.add_argument(
        "--precompiled-corpus",
        type=str,
        default=None,
        help="root of a VALIDATED precompiled edit corpus (modal_apps/precompile_edit_data_app.py output, "
        "containing BUILD_COMPLETE.json + <layer>/<partition>/*.jsonl.gz). Replaces the in-memory "
        "corruption/cycle/analogue builders entirely: records come from the validated artifact, the "
        "explicit three-layer sampler drives the mixture, and every contract hash is verified before any "
        "GPU work. Mutually exclusive with --corrupted-prior-count/--analogue-trace-count sizing.",
    )
    parser.add_argument(
        "--packed-corpus",
        type=str,
        default=None,
        help="root of the PACKED store for the audit layers (pack_trace_states_app output). Serves "
        "pre-materialized states so no trace is replayed at load (measured 21.36 -> 0.217 ms/trace). "
        "Must cover every audit shard exactly.",
    )
    parser.add_argument(
        "--packed-mmp-corpus",
        type=str,
        default=None,
        help="root of the PACKED MMP store (pack_mmp_pool_app output). Required with --packed-corpus: "
        "without it the MMP layer falls back to replaying the raw pool (~83 min per partition).",
    )
    parser.add_argument(
        "--precompiled-mmp-pool",
        type=str,
        default=None,
        help="analogue-trace pool consumed as the mmp_analogue layer of --precompiled-corpus. Rows are "
        "routed by the SAME ringcore-v1 scaffold key as the shards, so held-out partitions stay disjoint "
        "(the raw pool predates that partitioner; ~8%% of it falls in validation/test).",
    )
    args = parser.parse_args()
    _zmi.reset()  # zero-mixture instrumentation accumulates over this whole run (catalog/manifest/pool/train)

    if args.precompiled_corpus and not args.precompiled_mmp_pool:
        raise SystemExit(
            "--precompiled-corpus requires --precompiled-mmp-pool: the mmp_analogue layer carries a "
            "positive weight, and a silently-empty layer would renormalize the mixture"
        )

    checkpoint_modes = sum(
        item is not None
        for item in (
            args.load_checkpoint,
            args.initialize_checkpoint,
            args.initialize_compatible_checkpoint,
            args.resume_checkpoint,
        )
    )
    if checkpoint_modes > 1:
        raise ValueError(
            "--load-checkpoint, --initialize-checkpoint, and "
            "--initialize-compatible-checkpoint, and --resume-checkpoint are "
            "mutually exclusive"
        )
    if args.allow_resume_provenance_mismatch and args.resume_checkpoint is None:
        raise ValueError("--allow-resume-provenance-mismatch requires --resume-checkpoint")
    if args.allow_resume_step_extension and args.resume_checkpoint is None:
        raise ValueError("--allow-resume-step-extension requires --resume-checkpoint")
    if args.recovery_every < 0:
        raise ValueError("--recovery-every must be non-negative")
    if not 0.0 <= args.condition_dropout_probability <= 1.0:
        raise ValueError("--condition-dropout-probability must lie in [0, 1]")
    if (
        not np.isfinite(args.empirical_mark_prior_smoothing)
        or args.empirical_mark_prior_smoothing <= 0.0
    ):
        raise ValueError("--empirical-mark-prior-smoothing must be positive")
    if len(set(args.property_condition)) != len(args.property_condition):
        raise ValueError("--property-condition values must be unique")
    if args.property_condition and (
        args.model != "from_scratch" or args.training_backend != "factorized_marks"
    ):
        raise ValueError("property conditioning currently requires factorized from-scratch training")
    if not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("--warmup-steps must lie in [0, steps]")
    if args.schedule_steps < 0:
        raise ValueError("--schedule-steps must be non-negative")
    resolved_schedule_steps = args.steps if args.schedule_steps == 0 else args.schedule_steps
    if not 0.0 < args.minimum_learning_rate_fraction <= 1.0:
        raise ValueError("--minimum-learning-rate-fraction must lie in (0, 1]")
    if args.evaluation_every < 0 or args.early_stopping_patience < 0:
        raise ValueError("--evaluation-every and --early-stopping-patience must be non-negative")
    if not 0.0 <= args.early_stopping_min_relative_delta < 1.0:
        raise ValueError("--early-stopping-min-relative-delta must lie in [0, 1)")
    if (
        min(
            args.data_workers,
            args.path_workers,
            args.corpus_workers,
            args.training_support_workers,
        )
        < 0
    ):
        raise ValueError("data, path, corpus, and support worker counts must be non-negative")
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
    if (
        args.training_support_shard_size <= 0
        or args.training_support_microbatch_size <= 0
        or args.training_support_prefetch_factor <= 0
    ):
        raise ValueError("training-support compiler dimensions must be positive")
    if args.compile_training_support_start_step < 0:
        raise ValueError("compile-training-support-start-step must be non-negative")
    if args.compile_training_support_start_step and not args.compile_training_support_steps:
        raise ValueError("compile-training-support-start-step requires a non-empty compile range")
    if not 0 <= args.compile_training_support_steps <= args.steps:
        raise ValueError("compile-training-support-steps must lie in [0, steps]")
    if args.compile_training_support_start_step + args.compile_training_support_steps > args.steps:
        raise ValueError("compiled training-support range exceeds the training stream")
    if args.training_support_wait_seconds < 0.0:
        raise ValueError("training-support-wait-seconds must be non-negative")
    support_mode = bool(args.compile_training_support_steps or args.require_training_support_cache)
    if support_mode and args.training_support_cache_dir is None:
        raise ValueError("training-support-cache-dir is required for support caching")
    if support_mode and args.training_backend != "factorized_marks":
        raise ValueError("training support caching requires factorized marked training")
    if args.compile_training_support_steps and not args.require_path_cache:
        raise ValueError("training support compilation requires --require-path-cache")
    if args.compile_training_support_steps and args.require_training_support_cache:
        raise ValueError(
            "compile-training-support-steps and require-training-support-cache "
            "are mutually exclusive"
        )
    if args.scaled_manifest and not args.corrupted_prior_mix:
        raise ValueError("--scaled-manifest requires --corrupted-prior-mix")
    if args.cycle_op_mix and not args.corrupted_prior_mix:
        raise ValueError("--cycle-op-mix requires --corrupted-prior-mix")
    if args.disable_ring_grow_macro and not args.cycle_op_mix:
        # Disabling the legacy grow macro removes whole-ring addition; compositional cycle_close must be
        # present to keep ring-generation support (RING_CORE_V1). Guards against a support-losing config.
        raise ValueError("--disable-ring-grow-macro requires --cycle-op-mix (compositional ring support)")
    if args.scaled_manifest and (
        args.require_training_support_cache or args.compile_training_support_steps
    ):
        # The hierarchical sampler changes the row->record mapping, but the separate support COMPILER
        # (compile_training_support_shards) draws uniformly -> a pre-compiled cache would misalign. Until the
        # compiler shares the sampler, --scaled-manifest must compute support on the fly.
        raise ValueError(
            "--scaled-manifest is incompatible with a pre-compiled training-support cache (the hierarchical "
            "sampler changes the row->record mapping); run without --require-training-support-cache so "
            "support is computed on the fly, or tune --corrupted-prior-count/--analogue-trace-count for the "
            "mixture under uniform sampling")
    support_start_row = args.compile_training_support_start_step * args.batch_size
    support_stop_row = (
        args.compile_training_support_start_step + args.compile_training_support_steps
    ) * args.batch_size
    if args.compile_training_support_steps and (
        support_start_row % args.training_support_shard_size
        or support_stop_row % args.training_support_shard_size
    ):
        raise ValueError("compiled support range must align to shard boundaries")
    if args.compile_paths_only and args.path_cache is None:
        raise ValueError("compile-paths-only requires --path-cache")
    if args.compile_evaluation_cache and not args.compile_paths_only:
        raise ValueError("compile-evaluation-cache requires --compile-paths-only")
    if (args.compile_evaluation_cache or args.require_evaluation_cache) and (
        args.evaluation_cache_dir is None
    ):
        raise ValueError("evaluation-cache-dir is required when compiling or requiring its cache")
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
    if (
        args.empirical_mark_prior_mode != "none"
        or args.ring_family_mass_mode != "boolean"
        or args.ring_template_factorization != "flat"
    ) and args.training_backend != "factorized_marks":
        raise ValueError(
            "empirical mark priors and ring factorizations require factorized_marks"
        )
    if (
        args.trainable_parameter_scope != "all"
        and args.training_backend != "factorized_marks"
    ):
        raise ValueError(
            "restricted trainable parameter scopes require factorized_marks"
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
    # Broad-organic B-edit uses the SHARED scope loader (the same mining/validation/eval use) so the
    # production edit path never runs the CNOF-neutral filter. The CNOF loader stays only for the de-novo
    # CNOF base / ablation (organic_vocabulary off).
    if args.organic_vocabulary:
        corpus_scope = BROAD_ORGANIC_V1
        if args.max_atoms != corpus_scope.max_atoms:
            # EQUALITY: the padded state capacity (n_slots == --max-atoms) must EQUAL the scope bound so the
            # all-intermediate |V(x_k)| <= max_atoms invariant holds. A larger capacity would let an edit
            # grow a scope-max molecule past the bound; a smaller one would drop scope-eligible molecules.
            # This is the warm-start state-space contract: B, B-edit, scope, and evaluation must all use the
            # SAME max_atoms.
            raise ValueError(
                f"--max-atoms {args.max_atoms} != scope max_atoms {corpus_scope.max_atoms}; the padded "
                "state capacity must equal the scope bound (state-space contract)")
        split = load_organic_corpus_split(
            args.smiles_file,
            scope=corpus_scope,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            seed=args.seed,
            scan_all=not args.fast_split,
            workers=args.corpus_workers,
        )
        corpus_scope_descriptor = split.scope
        print(json.dumps({"phase": "corpus_scope", **corpus_scope_descriptor,
                          "census": {k: split.census[k] for k in ("retained", "total", "rejected")}},
                         sort_keys=True), flush=True)
    else:
        corpus_scope_descriptor = {"scope_name": "cnof_neutral", "scope_hash": None}
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
    # Zero-mixture (RING_CORE_V1 / --scaled-manifest): denovo_keep=0, so the carbon-tree source prior is NEVER
    # constructed -- the de-novo path branch (gated on `tree_source_prior is not None`) then stays dead and no
    # carbon-tree dataset is instantiated. The decision happens here, before any de-novo work.
    if args.source_prior == "carbon_tree" and not args.scaled_manifest:
        _zmi.bump("carbon_tree_prior_constructions")
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
    if args.scaled_manifest:
        # ZERO-MIXTURE (RING_CORE_V1): denovo_keep=0, so NO de-novo path is compiled, NO carbon-tree dataset
        # is instantiated, and NO de-novo/path cache is resolved. train_records=() (the corrupted-prior +
        # cycle-op edit records are appended below); the production ring catalog is reconstructed from the
        # fixed seeds, independent of any corpus path or B's checkpoint (owner mandate §1, §2).
        train_records = ()
        validation_records = ()
        test_records = ()
        ring_catalog = _build_ring_core_seed_ring_catalog(args.max_atoms)
        tree_paths_ready = True
        _zmi.bump("production_catalog_fingerprint_checks")
        print(
            json.dumps(
                {"phase": "zero_mixture_denovo_skipped",
                 "ring_catalog_fingerprint": ring_catalog_fingerprint(ring_catalog)},
                sort_keys=True,
            ),
            flush=True,
        )
    elif loaded_path_cache is None and tree_source_prior is not None:
        _zmi.bump("denovo_path_compile_calls")
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

    if (
        loaded_path_cache is None
        and loaded_sharded_manifest is None
        and tree_source_prior is None
        and not args.scaled_manifest  # zero-mixture already built the seed catalog above; do NOT compile de-novo proposals
    ):
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
                _zmi.bump("denovo_path_compile_calls")
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
    if empty_partitions and not args.scaled_manifest:
        # Under zero-mixture (--scaled-manifest) the de-novo train/validation/test partitions are
        # INTENTIONALLY empty (denovo_keep=0); the corrupted-prior + cycle-op edit records are appended
        # below and become the training data. So an empty de-novo partition is expected here, not an error.
        raise ValueError(
            "typed ring support left empty path partitions: "
            f"{empty_partitions}; enlarge the training/catalog split or increase "
            "the typed-template limits"
        )
    # Layer boundaries of the final train_records tuple, for the scaled-manifest hierarchical sampler.
    scaled_sampler_denovo_keep = 0
    scaled_sampler_n_corruption = 0
    precompiled_corpus = None
    precompiled_validation_corpus = None
    if args.precompiled_corpus:
        # PRODUCTION PATH. Records come from the validated artifact; the in-memory builders below are
        # unreachable in this branch by construction, which is the point -- regenerating a few hundred
        # sources at launch is what produced the data-starved baseline.
        precompiled_corpus = _load_precompiled_corpus(args, partition="train", seed=args.seed + 21)
        precompiled_validation_corpus = _load_precompiled_corpus(
            args, partition="validation", seed=args.seed + 22
        )
        train_records = precompiled_corpus.records
        validation_records = precompiled_validation_corpus.records
        edit_record_index_sampler_precompiled = precompiled_corpus.sampler
    elif args.corrupted_prior_mix:
        # Append corrupted-molecule source-prior (B-edit) records IN MEMORY -- never persisted to the
        # path cache, so B's carbon-tree cache stays reusable under --train-only. Sized ~50/50 against
        # the de-novo records; the support-cache signature carries a corrupted_prior_mix discriminator
        # so the mixed tuple recompiles its support instead of silently reusing the carbon-only cache.
        edit_smiles = tuple(split.train)[: max(1, args.corrupted_prior_count)]
        edit_records, edit_attempted = build_corrupted_prior_records(
            edit_smiles,
            n_slots=args.max_atoms,
            depth_max=5,
            seed=args.seed + 7,
            catalog=ring_catalog,  # enables the clean ring-opening corruption family (de-cyclize)
            vocabulary=ORGANIC_VOCABULARY if args.organic_vocabulary else CNOF_VOCABULARY,
        )
        if args.cycle_op_mix:
            # Compositional ring-op supervision (P3): real-molecule cycle_open/cycle_close teacher traces,
            # folded into the in-memory edit records so support-complete ring growth/closure gets positive
            # supervision (every ring bond, both directions, round-trip-verified). Enabled at the model via
            # enable_cycle_ops below.
            cycle_records, cycle_attempted = build_cycle_op_records(
                edit_smiles, n_slots=args.max_atoms, seed=args.seed + 8,
            )
            edit_records = tuple(edit_records) + tuple(cycle_records)
            print(
                json.dumps(
                    {"phase": "cycle_op_mix", "cycle_records": len(cycle_records),
                     "cycle_attempted": cycle_attempted},
                    sort_keys=True,
                ),
                flush=True,
            )
        # Keep an equal-sized de-novo subset so uniform sampling sees a real ~50/50 edit ratio; the model
        # already knows de-novo via the warm-start, so a subset suffices for retention. (Sizing this to
        # the full corpus would run the CPU corruption for tens of hours on the training GPU.)
        # Under --scaled-manifest the hierarchical sampler controls the mixture, so drop the de-novo
        # retention subset (de-novo is retained via the warm-start); else keep it for the uniform ~50/50.
        denovo_keep = 0 if args.scaled_manifest else min(len(train_records), len(edit_records))
        scaled_sampler_denovo_keep = denovo_keep
        scaled_sampler_n_corruption = len(edit_records)
        train_records = tuple(train_records)[:denovo_keep] + tuple(edit_records)
        print(
            json.dumps(
                {
                    "phase": "corrupted_prior_mix",
                    "edit_records": len(edit_records),
                    "edit_attempted": edit_attempted,
                    "train_total": len(train_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.analogue_trace_pool and not args.precompiled_corpus:
        # Append verified real analogue-pair edit traces (A->B / B->A MMP transitions between real drug
        # molecules) as GM PathRecords -- Layer 1 of the universal-edit-prior mixture (see
        # docs/PAPER_MASTER_PLAN.md §0b). Appended IN MEMORY like the corruption records; the support-cache
        # signature carries an analogue_trace_pool discriminator so the mixed tuple recompiles its support.
        _zmi.bump("edit_pool_open_calls")
        analogue_records = build_analogue_prior_records(
            args.analogue_trace_pool,
            count=(args.analogue_trace_count or None),
        )
        train_records = tuple(train_records) + tuple(analogue_records)
        print(
            json.dumps(
                {
                    "phase": "analogue_prior_mix",
                    "analogue_records": len(analogue_records),
                    "train_total": len(train_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.scaled_manifest and not validation_records and not args.precompiled_corpus:
        # Zero-mixture edit VALIDATION (owner mandate §4): the de-novo validation partition is empty, so the
        # held-out editing diagnostic is built from split.VALIDATION molecules (disjoint from split.train,
        # which sourced the training edit records) -- covering the corrupted-prior + cycle-op editing families
        # + broad-organic/charged contexts. NO de-novo/legacy evaluation cache is required.
        _held_out = tuple(split.validation)[: max(1, min(args.corrupted_prior_count, len(split.validation)))]
        _val_edit, _ = build_corrupted_prior_records(
            _held_out, n_slots=args.max_atoms, depth_max=5, seed=args.seed + 17,
            catalog=ring_catalog,
            vocabulary=ORGANIC_VOCABULARY if args.organic_vocabulary else CNOF_VOCABULARY,
        )
        if args.cycle_op_mix:
            _val_cycle, _ = build_cycle_op_records(_held_out, n_slots=args.max_atoms, seed=args.seed + 18)
            _val_edit = tuple(_val_edit) + tuple(_val_cycle)
        validation_records = tuple(_val_edit)
        print(
            json.dumps(
                {"phase": "zero_mixture_edit_validation", "validation_records": len(validation_records)},
                sort_keys=True,
            ),
            flush=True,
        )
    training_support_cache = None
    if args.training_support_cache_dir is not None:
        training_support_signature = _training_support_cache_signature(
            path_cache_signature,
            seed=args.seed + 3,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=args.progress_stratification_fraction,
            ring_electronic_mode=args.ring_electronic_mode,
            corrupted_prior_mix=args.corrupted_prior_mix,
            organic_vocabulary=args.organic_vocabulary,
            analogue_trace_pool=bool(args.analogue_trace_pool),
            cycle_op_mix=bool(args.cycle_op_mix),
            disable_ring_grow_macro=bool(args.disable_ring_grow_macro),
        )
        training_support_cache = ShardedTrainingSupportCache(
            args.training_support_cache_dir,
            training_support_signature,
            total_rows=args.steps * args.batch_size,
            shard_size=args.training_support_shard_size,
            wait_timeout_seconds=args.training_support_wait_seconds,
        )
    if args.compile_training_support_steps:
        if training_support_cache is None or ring_catalog is None:
            raise RuntimeError("training support compiler lacks its cache or ring catalog")
        start_index = args.compile_training_support_start_step * args.batch_size
        stop_index = (
            args.compile_training_support_start_step + args.compile_training_support_steps
        ) * args.batch_size
        last_reported = 0

        def report_support_progress(metrics: dict[str, float]) -> None:
            nonlocal last_reported
            compiled = int(metrics["compiled_rows"])
            if compiled - last_reported < 4096 and int(metrics["absolute_stop"]) < stop_index:
                return
            last_reported = compiled
            print(
                json.dumps(
                    {"phase": "training_support_compilation", **metrics},
                    sort_keys=True,
                ),
                flush=True,
            )

        def report_support_shard(metrics: dict[str, float]) -> None:
            print(
                json.dumps(
                    {"phase": "training_support_shard_saved", **metrics},
                    sort_keys=True,
                ),
                flush=True,
            )

        started = perf_counter()
        paths = compile_training_support_shards(
            train_records,
            cache=training_support_cache,
            start_index=start_index,
            stop_index=stop_index,
            seed=args.seed + 3,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=args.progress_stratification_fraction,
            ring_catalog=ring_catalog,
            ring_electronic_mode=args.ring_electronic_mode,
            workers=args.training_support_workers,
            microbatch_size=args.training_support_microbatch_size,
            prefetch_factor=args.training_support_prefetch_factor,
            progress_callback=report_support_progress,
            shard_callback=report_support_shard,
        )
        elapsed = perf_counter() - started
        # Touch both ends through the validating mmap reader before declaring
        # a prefix ready for an expensive GPU consumer.
        training_support_cache.require(start_index)
        training_support_cache.require(stop_index - 1)
        print(
            json.dumps(
                {
                    "phase": "training_support_cache_ready",
                    "root": str(training_support_cache.root),
                    "start_step": args.compile_training_support_start_step,
                    "steps": args.compile_training_support_steps,
                    "start_row": start_index,
                    "stop_row": stop_index,
                    "rows": stop_index - start_index,
                    "shards": len(paths),
                    "seconds": elapsed,
                    "rows_per_second": (stop_index - start_index) / max(elapsed, 1e-12),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
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

    property_condition_names = tuple(str(name) for name in args.property_condition)
    property_normalizer = (
        fit_property_condition_normalizer(train_records, property_condition_names)
        if property_condition_names
        else None
    )
    train_property_conditions = (
        standardized_record_conditions(train_records, property_normalizer)
        if property_normalizer is not None
        else None
    )
    validation_property_conditions = (
        standardized_record_conditions(validation_records, property_normalizer)
        if property_normalizer is not None
        else None
    )
    test_property_conditions = (
        standardized_record_conditions(test_records, property_normalizer)
        if property_normalizer is not None
        else None
    )
    property_conditioning_signature = (
        None
        if property_normalizer is None
        else {
            **property_normalizer.to_dict(),
            "condition_dropout_probability": float(
                args.condition_dropout_probability
            ),
        }
    )

    _eval_capabilities = OperatorCapabilities(
        compute_ring_grow_support=not args.disable_ring_grow_macro,
        compute_ring_restates=args.corrupted_prior_mix,
        compute_cyclic_graft=args.corrupted_prior_mix,
        compute_ring_opening=args.corrupted_prior_mix,
    )
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
        property_conditioning=property_conditioning_signature,
        operator_capability_fingerprint=_eval_capabilities.fingerprint(),
        operator_registry_hash=MARK_RULE_NAMES_HASH,
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
        args.data_workers if args.evaluation_workers is None else args.evaluation_workers
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
        if (
            args.training_backend == "factorized_marks"
            and args.ring_electronic_mode
            in {"factorized_local", "factorized_contextual"}
            and isinstance(validation_examples, FactorizedMarkBatch)
            and isinstance(test_examples, FactorizedMarkBatch)
        ):
            needs_teacher_certificates = any(
                getattr(batch, "ring_teacher_semantic_certificates", None) is None
                or any(
                    name == "ring_system_grow" and certificate is None
                    for name, certificate in zip(
                        batch.teacher_rule_names,
                        getattr(batch, "ring_teacher_semantic_certificates", None)
                        or (None,) * batch.batch_size,
                    )
                )
                for batch in (validation_examples, test_examples)
            )
            if needs_teacher_certificates:
                certificate_started = perf_counter()
                validation_examples = attach_ring_teacher_semantic_certificates(
                    validation_examples,
                    ring_catalog=ring_catalog,
                    ring_electronic_mode=args.ring_electronic_mode,
                    workers=evaluation_workers,
                )
                test_examples = attach_ring_teacher_semantic_certificates(
                    test_examples,
                    ring_catalog=ring_catalog,
                    ring_electronic_mode=args.ring_electronic_mode,
                    workers=evaluation_workers,
                )
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
                            "phase": "evaluation_teacher_certificates_compiled",
                            "path": str(evaluation_cache_path),
                            "seconds": perf_counter() - certificate_started,
                            "workers": evaluation_workers,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    elif (
        args.require_evaluation_cache
        and evaluation_cache_path is not None
        and property_conditioning_signature is not None
    ):
        base_evaluation_signature = _evaluation_batch_cache_signature(
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
            property_conditioning=None,
        )
        base_cache_path = _evaluation_batch_cache_path(
            args.evaluation_cache_dir,
            base_evaluation_signature,
        )
        if not base_cache_path.is_file():
            raise FileNotFoundError(
                "required conditioned and reusable base evaluation caches are "
                f"missing: {evaluation_cache_path}; {base_cache_path}"
            )
        cache_load_started = perf_counter()
        validation_examples, test_examples = _load_evaluation_batch_cache(
            base_cache_path,
            signature=base_evaluation_signature,
            validation_examples=args.validation_examples,
            test_examples=args.test_examples,
        )
        if not isinstance(validation_examples, FactorizedMarkBatch) or not isinstance(
            test_examples, FactorizedMarkBatch
        ):
            raise TypeError("property conditioning requires factorized evaluation caches")
        validation_examples = attach_property_conditions(
            validation_examples,
            validation_records,
            seed=args.seed + 1,
            target_property_conditions=validation_property_conditions,
        )
        test_examples = attach_property_conditions(
            test_examples,
            test_records,
            seed=args.seed + 2,
            target_property_conditions=test_property_conditions,
        )
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
                    "phase": "conditioned_evaluation_cache_upgraded",
                    "base_path": str(base_cache_path),
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
            f"required evaluation batch cache is missing: {evaluation_cache_path}"
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
            target_property_conditions=validation_property_conditions,
            condition_dropout_probability=0.0,
            # The active model's editing-family capabilities -- one shared representability contract, so an
            # editing teacher never lands outside the eval batch's dynamic candidates.
            capabilities=_eval_capabilities,
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
        # Zero-mixture: the de-novo test partition is empty; reuse the held-out edit validation records for the
        # test batch (the test metric is diagnostic only, never a training/selection input). Keeps the batch
        # non-empty without a de-novo cache.
        _test_source = test_records if test_records else validation_records
        test_examples = sample_factorized_mark_batch(
            _test_source,
            batch_size=args.test_examples,
            seed=args.seed + 2,
            late_time_fraction=args.late_time_fraction,
            operational_horizon=args.operational_horizon,
            progress_stratification_fraction=args.progress_stratification_fraction,
            use_aromatic_bond_view=args.bond_representation == "aromatic",
            workers=evaluation_workers,
            ring_catalog=ring_catalog,
            ring_electronic_mode=args.ring_electronic_mode,
            target_property_conditions=test_property_conditions,
            condition_dropout_probability=0.0,
            capabilities=_eval_capabilities,
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
                json.dumps({"phase": "test_examples_built", "cache_size": len(fiber_cache)}),
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
                    "path": (None if evaluation_cache_path is None else str(evaluation_cache_path)),
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
    empirical_mark_priors = None
    if args.empirical_mark_prior_mode == "corpus_residual_v1":
        prior_fit_started = perf_counter()
        empirical_mark_priors = fit_factorized_mark_empirical_priors(
            train_records,
            smoothing=args.empirical_mark_prior_smoothing,
        )
        print(
            json.dumps(
                {
                    "phase": "factorized_empirical_mark_priors_fitted",
                    "seconds": perf_counter() - prior_fit_started,
                    "smoothing": args.empirical_mark_prior_smoothing,
                    "observations": {
                        "root_atom": empirical_mark_priors.root_atom_observations,
                        "connected_atom": (
                            empirical_mark_priors.connected_atom_observations
                        ),
                        "atom_restate": (
                            empirical_mark_priors.atom_restate_observations
                        ),
                        "bond_reorder": (
                            empirical_mark_priors.bond_reorder_observations
                        ),
                        "ring_electronic": (
                            empirical_mark_priors.ring_electronic_observations
                        ),
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.model == "from_scratch" and args.training_backend == "factorized_marks":
        if ring_catalog is None:
            raise RuntimeError("factorized marked model requires a typed ring catalog")
        model = FactorizedTraceletRateModel(
            ring_catalog,
            hidden_dim=args.hidden_dim,
            message_passing_steps=args.message_passing_steps,
            ring_electronic_mode=args.ring_electronic_mode,
            rate_factorization=args.rate_factorization,
            property_condition_dim=len(property_condition_names),
            empirical_mark_prior_mode=args.empirical_mark_prior_mode,
            empirical_mark_priors=empirical_mark_priors,
            ring_family_mass_mode=args.ring_family_mass_mode,
            ring_template_factorization=args.ring_template_factorization,
            enable_ring_restates=args.corrupted_prior_mix,
            enable_cyclic_graft=args.corrupted_prior_mix,
            enable_heteroatom_scan=args.corrupted_prior_mix,
            enable_ring_opening=args.corrupted_prior_mix,
            enable_cycle_ops=args.cycle_op_mix,
            enable_ring_grow_macro=not args.disable_ring_grow_macro,
            atom_vocabulary=ORGANIC_VOCABULARY if args.organic_vocabulary else None,
        ).to(device)
        if args.cycle_op_mix and args.disable_ring_grow_macro:
            # RING_CORE_V1 launch identity (owner finalization §4): print every frozen hash + capability flag
            # at startup so a launch that drifts from the frozen contract is visible in the run log.
            import ring_core_identity as _rci

            # Owner-locked production scheduler guard (decision 2026-07-28): the schedule-check run and the
            # eventual full RingCore run MUST share the locked LR schedule. Assert the run's ACTUAL scheduler
            # args reproduce SCHEDULER_CONFIG_HASH; abort BEFORE GPU on any drift (outside the try below so it
            # is never swallowed). training_steps is intentionally excluded -- only the schedule must match.
            _run_scheduler_hash = _rci.scheduler_config_hash_from_args(
                warmup_steps=args.warmup_steps,
                schedule_steps=resolved_schedule_steps,
                minimum_learning_rate_fraction=args.minimum_learning_rate_fraction,
                peak_learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            if _run_scheduler_hash != _rci.SCHEDULER_CONFIG_HASH:
                raise SystemExit(
                    f"RING_CORE_V1 scheduler drift: run scheduler hash {_run_scheduler_hash} != "
                    f"owner-locked {_rci.SCHEDULER_CONFIG_HASH}. A RingCore launch MUST use the locked "
                    f"production scheduler (AdamW peak_lr=3e-4 wd=1e-5 warmup=500 schedule_steps=3000 "
                    f"min_lr_fraction=0.05 cosine+linear-warmup); got warmup={args.warmup_steps} "
                    f"schedule_steps={resolved_schedule_steps} min_lr={args.minimum_learning_rate_fraction} "
                    f"peak_lr={args.learning_rate} weight_decay={args.weight_decay}."
                )
            try:
                print(json.dumps({
                    "phase": "ring_core_launch_identity",
                    "capability_hash": _rci.CAPABILITY_HASH,
                    "operator_registry_hash": _rci.recompute_operator_registry_hash(),
                    "cycle_op_semantic_hash": _rci.recompute_cycle_op_semantic_hash(),
                    "calibration_policy_hash": _rci.CALIBRATION_POLICY_HASH,
                    "scheduler_config_hash": _run_scheduler_hash,
                    "scope_hash": _rci.SCOPE_HASH,
                    "base_b_sha256": _rci.BASE_B_SHA256,
                    "eval_operator_capability_fingerprint": _eval_capabilities.fingerprint(),
                    "teacher_representability_filter_version": TEACHER_REPRESENTABILITY_FILTER_VERSION,
                    "evaluation_batch_cache_format_version": EVALUATION_BATCH_CACHE_FORMAT_VERSION,
                    "denovo_keep": 0 if args.scaled_manifest else None,
                    "enable_cycle_ops": True,
                    "enable_ring_macros": False,
                    "enable_ring_grow_macro": not args.disable_ring_grow_macro,
                    "training_steps": args.steps,
                    "schedule_steps": resolved_schedule_steps,
                    "warmup_steps": args.warmup_steps,
                    "max_atoms": args.max_atoms,
                    "scaled_manifest": args.scaled_manifest or None,
                }, sort_keys=True), flush=True)
            except Exception:  # noqa: BLE001 -- identity print must never break a launch
                pass
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
        "corrupted_prior_mix": bool(args.corrupted_prior_mix),
        "enable_cycle_ops": bool(args.cycle_op_mix),
        # RING_CORE_V1: legacy whole-ring grow macro disabled (ring addition is purely compositional).
        # enable_ring_macros is the forward-looking P4 hybrid flag; always False until RING_HYBRID_V2.
        "enable_ring_grow_macro": not bool(args.disable_ring_grow_macro),
        "enable_ring_macros": False,
        # Teacher-in-exact-candidates filter (declared production data contract): every selected teacher
        # belongs to the model's exact dynamic candidate set for its state; unrepresentable corruption traces
        # are excluded at the data source. Versioned so a filtered corpus is never conflated across versions.
        "teacher_representability_filter_version": (
            TEACHER_REPRESENTABILITY_FILTER_VERSION if args.corrupted_prior_mix else None
        ),
        "eval_operator_capability_fingerprint": (
            _eval_capabilities.fingerprint() if args.training_backend == "factorized_marks" else None
        ),
        # Persist the organic-vocab flag so inference can reconstruct the wider heads + editing families
        # (the enable_* flags are derived from corrupted_prior_mix at load, mirroring construction).
        "organic_vocabulary": bool(args.organic_vocabulary),
        # Pin the corpus SCOPE so any data/checkpoint load under a different scope can fail loudly (mining,
        # training, validation, sampling, eval must all agree on the same broad-organic scope).
        "corpus_scope": corpus_scope_descriptor.get("scope_name"),
        "corpus_scope_hash": corpus_scope_descriptor.get("scope_hash"),
        # Pin the versioned ring-system definition used for corruption (clean ring-opening), enumeration,
        # the executor, and sampling, so a checkpoint records the exact catalog it was trained against.
        "ring_catalog_fingerprint": (
            ring_catalog_fingerprint(ring_catalog) if ring_catalog is not None else None
        ),
        "use_bf16": args.use_bf16,
        "data_workers": args.data_workers,
        "data_prefetch_factor": args.data_prefetch_factor,
        "path_workers": args.path_workers,
        "path_checkpoint_interval": args.path_checkpoint_interval,
        "corpus_workers": args.corpus_workers,
        "hidden_dim": args.hidden_dim,
        "message_passing_steps": args.message_passing_steps,
        "rate_factorization": args.rate_factorization,
        "empirical_mark_prior_mode": args.empirical_mark_prior_mode,
        "empirical_mark_prior_smoothing": args.empirical_mark_prior_smoothing,
        "empirical_mark_priors": (
            None
            if empirical_mark_priors is None
            else empirical_mark_priors.to_dict()
        ),
        "ring_family_mass_mode": args.ring_family_mass_mode,
        "ring_template_factorization": args.ring_template_factorization,
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
        "property_condition_dim": len(property_condition_names),
        "property_conditioning": property_conditioning_signature,
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
        "trainable_parameter_scope": args.trainable_parameter_scope,
        "optimizer_kind": (
            "adamw_decoupled_v1" if args.training_backend == "factorized_marks" else "adam"
        ),
        "warmup_steps": args.warmup_steps,
        "schedule_steps": resolved_schedule_steps,
        "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
        "evaluation_every": args.evaluation_every,
        "evaluation_batch_size": args.evaluation_batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_relative_delta": (args.early_stopping_min_relative_delta),
        "late_time_fraction": args.late_time_fraction,
        "operational_horizon": args.operational_horizon,
        "provenance_sha256": args.provenance_sha256,
    }
    checkpoint_metadata["max_atoms"] = args.max_atoms
    if args.cycle_op_mix and args.disable_ring_grow_macro:
        # RING_CORE_V1 self-identification: persist the frozen capability + live code hashes so the rollout
        # harness can verify checkpoint identity. Defensive import (never break training); if absent, the
        # harness recomputes the code hashes + verifies the enable flags / scope / max_atoms instead.
        try:
            import ring_core_identity

            checkpoint_metadata.update(ring_core_identity.ring_core_checkpoint_metadata())
        except Exception:  # noqa: BLE001
            pass
    expected = {
        key: checkpoint_metadata[key]
        for key in (
            "model",
            "training_backend",
            "hidden_dim",
            "message_passing_steps",
            "rate_factorization",
            "empirical_mark_prior_mode",
            "empirical_mark_prior_smoothing",
            "empirical_mark_priors",
            "ring_family_mass_mode",
            "ring_template_factorization",
            "bond_representation",
            "ring_proposals",
            "ring_electronic_mode",
            "teacher_ordering",
            "source_prior",
            "tree_size_prior",
            "tree_couplings_per_target",
            "tree_transport",
            "property_condition_dim",
            "property_conditioning",
            "trainable_parameter_scope",
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
        "empirical_mark_prior_mode": "none",
        "empirical_mark_prior_smoothing": 1.0,
        "empirical_mark_priors": None,
        "ring_family_mass_mode": "boolean",
        "ring_template_factorization": "flat",
        "property_condition_dim": 0,
        "property_conditioning": None,
        "trainable_parameter_scope": "all",
    }

    loaded_checkpoint = None
    resume_state = None
    initialized_checkpoint_path = (
        args.initialize_checkpoint or args.initialize_compatible_checkpoint
    )
    selected_checkpoint_path = (
        args.load_checkpoint
        or initialized_checkpoint_path
        or args.resume_checkpoint
    )
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
        if args.initialize_compatible_checkpoint is not None:
            # A direct conditional model is deliberately warm-started from
            # the qualified unconditional checkpoint.  The zero-residual
            # condition adapter and the fixed, parameter-free empirical base
            # measures are the only scientific configuration differences
            # permitted by this shape-exact transfer path.
            mismatches.pop("property_condition_dim", None)
            mismatches.pop("property_conditioning", None)
            mismatches.pop("empirical_mark_prior_mode", None)
            mismatches.pop("empirical_mark_prior_smoothing", None)
            mismatches.pop("empirical_mark_priors", None)
            mismatches.pop("ring_family_mass_mode", None)
            mismatches.pop("ring_template_factorization", None)
            # rate_factorization (hierarchical <-> superposed) changes only the
            # forward rate-composition in _masked_family_logits, never a tensor
            # shape, so the shape-exact transfer stays valid and the newly added
            # topology-group head still fresh-inits.  Permitted exactly like
            # ring_template_factorization above -- this is the section-9.3
            # ring-hazard fix (superposed + topology_cycle_hierarchical).
            mismatches.pop("rate_factorization", None)
            mismatches.pop("trainable_parameter_scope", None)
            source_ring_mode = checkpoint_payload.get(
                "ring_electronic_mode",
                checkpoint_defaults["ring_electronic_mode"],
            )
            if {
                str(source_ring_mode),
                str(args.ring_electronic_mode),
            }.issubset({"factorized_local", "factorized_contextual"}):
                mismatches.pop("ring_electronic_mode", None)
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
                    "schedule_steps",
                    "minimum_learning_rate_fraction",
                    "evaluation_every",
                    "evaluation_batch_size",
                    "early_stopping_patience",
                    "early_stopping_min_relative_delta",
                    "late_time_fraction",
                    "operational_horizon",
                    "progress_stratification_fraction",
                    "use_bf16",
                    "trainable_parameter_scope",
                )
            }
            if not args.allow_resume_provenance_mismatch:
                resume_expected["provenance_sha256"] = checkpoint_metadata["provenance_sha256"]
            resume_mismatches = {
                key: (
                    checkpoint_payload.get(
                        key,
                        (
                            checkpoint_payload.get("training_steps")
                            if key == "schedule_steps"
                            else None
                        ),
                    ),
                    value,
                )
                for key, value in resume_expected.items()
                if checkpoint_payload.get(
                    key,
                    (
                        checkpoint_payload.get("training_steps")
                        if key == "schedule_steps"
                        else None
                    ),
                )
                != value
            }
            if args.allow_resume_step_extension:
                previous_steps = int(checkpoint_payload["training_steps"])
                if args.steps <= previous_steps:
                    raise ValueError(
                        "resume step extension requires --steps to exceed the "
                        "checkpoint training horizon"
                    )
                resume_mismatches.pop("training_steps", None)
            mismatches.update(resume_mismatches)
        if mismatches:
            raise ValueError(f"checkpoint/config mismatch: {mismatches}")
        if args.load_checkpoint is not None:
            loaded_checkpoint = checkpoint_payload
            model.load_state_dict(loaded_checkpoint["state_dict"])
            phase = "checkpoint_loaded"
        elif args.initialize_checkpoint is not None:
            model.load_state_dict(checkpoint_payload["state_dict"])
            phase = "checkpoint_initialized_fresh_optimizer"
        elif args.initialize_compatible_checkpoint is not None:
            source_vocab = (
                ORGANIC_VOCABULARY
                if checkpoint_payload.get("organic_vocabulary")
                else CNOF_VOCABULARY
            )
            dest_vocab = getattr(model, "atom_vocabulary", None)
            widening = (
                dest_vocab is not None
                and tuple(dest_vocab.classes) != tuple(source_vocab.classes)
            )
            if widening:
                # B->B-edit vocabulary expansion: SEMANTIC partial transfer -- shared (element,valence) rows
                # copy from the source by LABEL, genuinely-new S/P/Cl/Br/I/B rows keep the fresh init. The
                # shape-exact path alone would discard B's learned CNOF output rows on the widened heads.
                (
                    initialized_state,
                    transferred,
                    retained,
                    transfer_row_table,
                    transfer_map_hash,
                ) = _semantic_partial_checkpoint_initialization(
                    model.state_dict(),
                    checkpoint_payload,
                    source_classes=source_vocab.classes,
                    dest_classes=dest_vocab.classes,
                )
                phase = "checkpoint_semantically_initialized_fresh_optimizer"
            else:
                initialized_state, transferred, retained = (
                    _compatible_checkpoint_initialization(
                        model.state_dict(),
                        checkpoint_payload,
                    )
                )
                transfer_row_table, transfer_map_hash = [], None
                phase = "checkpoint_compatibly_initialized_fresh_optimizer"
            model.load_state_dict(initialized_state, strict=True)
            # Warm-start transfer provenance recorded in the saved checkpoint (fail-closed on reload).
            checkpoint_metadata["warmstart_source_sha256"] = _file_sha256(
                selected_checkpoint_path
            )
            checkpoint_metadata["warmstart_transfer_impl_version"] = (
                _SEMANTIC_TRANSFER_IMPL_VERSION
            )
            checkpoint_metadata["warmstart_semantic_transfer_map_hash"] = transfer_map_hash
            checkpoint_metadata["warmstart_source_vocab_hash"] = _vocabulary_hash(source_vocab)
            checkpoint_metadata["warmstart_dest_vocab_hash"] = (
                _vocabulary_hash(dest_vocab) if dest_vocab is not None else None
            )
            checkpoint_metadata["warmstart_rows_copied"] = sum(
                1 for row in transfer_row_table if row["status"] == "COPIED_WITH_INDEX_MAP"
            )
            checkpoint_metadata["warmstart_rows_fresh"] = sum(
                1 for row in transfer_row_table if row["status"] == "FRESH_NEW_CLASS"
            )
            checkpoint_metadata["warmstart_row_table_hash"] = (
                hashlib.sha256(
                    json.dumps(transfer_row_table, sort_keys=True).encode()
                ).hexdigest()[:16]
                if transfer_row_table
                else None
            )
        else:
            resume_state = checkpoint_payload
            model.load_state_dict(resume_state["current_state_dict"])
            phase = "recovery_checkpoint_loaded"
        print(
            json.dumps(
                {
                    "phase": phase,
                    "path": str(selected_checkpoint_path),
                    **(
                        {
                            "transferred_tensors": len(transferred),
                            "retained_initialized_tensors": len(retained),
                            "retained_initialized_names": retained,
                        }
                        if args.initialize_compatible_checkpoint is not None
                        else {}
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.resume_checkpoint is not None and args.allow_resume_provenance_mismatch:
            print(
                json.dumps(
                    {
                        "phase": "implementation_only_resume_accepted",
                        "checkpoint_provenance_sha256": checkpoint_payload.get("provenance_sha256"),
                        "current_provenance_sha256": args.provenance_sha256,
                    },
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
    validation_phase = "resume_validation" if resume_state is not None else "initial_validation"
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
        key: value for key, value in initial_validation.items() if not bool(np.isfinite(value))
    }
    if nonfinite_baseline:
        raise RuntimeError(
            f"restored initial validation contains non-finite metrics: {nonfinite_baseline}"
        )
    history = []
    recovery_path = (
        _recovery_path(args.checkpoint) if args.checkpoint is not None else args.resume_checkpoint
    )
    best_so_far_path = _best_so_far_path(args.checkpoint) if args.checkpoint is not None else None

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
        if args.snapshot_checkpoints and args.checkpoint is not None:
            # Step-stamped copy of the SAME exact-recovery payload, so a post-hoc selection rule can score
            # each step's current_state_dict. The rolling recovery file overwrites itself every interval.
            snapshot_path = _step_snapshot_path(
                args.checkpoint, int(training_state["completed_steps"])  # type: ignore[arg-type]
            )
            _atomic_torch_save(payload, snapshot_path)
            print(
                json.dumps(
                    {
                        "phase": "step_snapshot_saved",
                        "path": str(snapshot_path),
                        "completed_steps": training_state["completed_steps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
                        "selected_step": training_state["best_metrics"].get("selected_step"),
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
        edit_record_index_sampler = None
        if args.precompiled_corpus:
            # The precompiled corpus already built its explicit three-layer sampler over exactly
            # this record tuple, so reuse it rather than re-deriving boundaries positionally.
            edit_record_index_sampler = edit_record_index_sampler_precompiled
        elif args.scaled_manifest:
            edit_record_index_sampler = _build_scaled_manifest_sampler(
                train_records,
                denovo_keep=scaled_sampler_denovo_keep,
                n_corruption=scaled_sampler_n_corruption,
                manifest_path=args.scaled_manifest,
                seed=args.seed + 11,
            )
            print(json.dumps({
                "phase": "hierarchical_sampler",
                "records": len(train_records),
                "realized_count_fractions": (None if edit_record_index_sampler is None
                                             else edit_record_index_sampler.realized_layer_fractions()),
            }, sort_keys=True), flush=True)
        _trainer_result = train_factorized_mark_model(
            model,
            train_records,
            validation_examples,
            dry_launch=args.dry_launch,
            dry_launch_output=(args.dry_launch_output or None),
            record_index_sampler=edit_record_index_sampler,
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
            schedule_steps=resolved_schedule_steps,
            minimum_learning_rate_fraction=(args.minimum_learning_rate_fraction),
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_relative_delta=(args.early_stopping_min_relative_delta),
            progress_callback=lambda metrics: print(
                json.dumps({"phase": "training", **metrics}, sort_keys=True),
                flush=True,
            ),
            checkpoint_interval=args.recovery_every,
            checkpoint_callback=(save_recovery if recovery_path is not None else None),
            resume_state=resume_state,
            # A benchmark REQUIRES synchronized phase boundaries; without them the forward/backward/
            # optimizer splits are just async-launch times and the report would be fiction.
            profile_timing=args.fast_split or bool(args.benchmark_steps),
            benchmark_steps=args.benchmark_steps,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_output=(args.benchmark_output or None),
            evaluation_batch_size=args.evaluation_batch_size,
            initial_validation_metrics=observed_validation,
            training_support_cache=(
                training_support_cache if args.require_training_support_cache else None
            ),
            require_cached_support=args.require_training_support_cache,
            target_property_conditions=train_property_conditions,
            condition_dropout_probability=args.condition_dropout_probability,
            trainable_parameter_scope=args.trainable_parameter_scope,
        )
        if args.dry_launch:
            _dry_ok, _dry_report = _zmi.zero_mixture_ok()
            _finite = bool(_trainer_result.get("train_loss_finite")) and bool(
                _trainer_result.get("validation_loss_finite")
            )
            _verdict = "GO_CPU_DRY_LAUNCH" if (_dry_ok and _finite) else "NO_GO_ZERO_MIXTURE_GATE"
            print(
                json.dumps(
                    {
                        "phase": "dry_launch_complete",
                        "verdict": _verdict,
                        "zero_mixture_ok": _dry_ok,
                        "train_gm_loss": _trainer_result.get("train_gm_loss"),
                        "validation_gm_loss": _trainer_result.get("validation_gm_loss"),
                        "counters": _dry_report["counters"],
                        "failing_zero": _dry_report["failing_zero_counters"],
                        "failing_positive": _dry_report["failing_positive_counters"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        history, selected_validation = _trainer_result
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
                "trainable_parameter_scope": args.trainable_parameter_scope,
                "warmup_steps": args.warmup_steps,
                "schedule_steps": resolved_schedule_steps,
                "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
                "evaluation_every": args.evaluation_every,
                "evaluation_batch_size": args.evaluation_batch_size,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_relative_delta": (args.early_stopping_min_relative_delta),
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
            "initialized_from": (
                str(initialized_checkpoint_path)
                if initialized_checkpoint_path is not None
                else None
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
            "trainable_parameter_scope": args.trainable_parameter_scope,
            "warmup_steps": args.warmup_steps,
            "schedule_steps": resolved_schedule_steps,
            "minimum_learning_rate_fraction": (args.minimum_learning_rate_fraction),
            "evaluation_every": args.evaluation_every,
            "evaluation_batch_size": args.evaluation_batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_relative_delta": (args.early_stopping_min_relative_delta),
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
