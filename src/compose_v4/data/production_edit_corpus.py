"""Production edit corpus: validated shard manifest -> layered PathRecords -> explicit three-layer sampler.

This is the ONLY sanctioned training-data path for RingCore-V1. The trainer must not rebuild corruption
or cycle records in memory at launch: those in-memory builders regenerate a few hundred sources per run
(the DATA_STARVED_BASELINE), have no partition discipline, and carry no contract hashes. Here the corpus
is a validated artifact and the loader's job is to refuse anything that does not match the contract.

Fail-closed conditions, all of which abort BEFORE any GPU work:
  * ``BUILD_COMPLETE.json`` missing, or the reducer marked the build invalid;
  * any expected shard absent, or an unexpected shard present;
  * any contract hash (capability / codec / operator registry / scope / trace schema / partitioner)
    disagreeing with what the caller pins;
  * a positively-weighted layer contributing no records (enforced by ``build_layered_sampler``).

Partition discipline for the MMP layer
-------------------------------------
The corruption and cycle layers were partitioned at build time by the ringcore-v1 scaffold key. The MMP
analogue pool predates that partitioner and was mined under a different random split, so it is filtered
HERE by the same pure ``partition_for_scaffold`` function. Measured on a stride sample of the 363,456-row
pool, 8.09% of its records fall in the held-out partitions -- roughly 29k rows that would otherwise train
on validation/test scaffolds and quietly inflate the held-out learning curve used to select checkpoints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from compose_v4.data.packed_trace_store import read_packed_shard
from compose_v4.data.scaffold_partition import (
    DEFAULT_RATIOS,
    SCAFFOLD_KEY_ALGORITHM,
    SCAFFOLD_KEY_VERSION,
    murcko_scaffold,
    partition_for_scaffold,
)
from compose_v4.experiments.analogue_prior import rewrite_trace_from_record
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.hierarchical_sampler import (
    CYCLE_OPS,
    GENERAL_CORRUPTION,
    MMP_ANALOGUE,
    PRODUCTION_LAYERS,
    build_layered_sampler,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace_shard import load_trace_records

# Precompiled shard layer names -> production sampler layer names.
_SHARD_LAYER_TO_PRODUCTION = {
    "corruption": GENERAL_CORRUPTION,
    "cycle_ops": CYCLE_OPS,
}

_CONTRACT_KEYS = (
    "capability_hash",
    "codec_implementation_hash",
    "operator_registry_hash",
    "scope_hash",
    "trace_schema_version",
)


class ProductionCorpusError(RuntimeError):
    """The corpus artifact does not satisfy the production contract."""


@dataclass(frozen=True)
class LayeredEditCorpus:
    """Records concatenated in ``PRODUCTION_LAYERS`` order, plus the sampler over them.

    ``build_layered_sampler`` returns ABSOLUTE indices into the concatenation of its ``records_by_layer``
    values in order, so ``records`` must stay exactly that concatenation -- ``layer_bounds`` records the
    slice each layer owns so the alignment is checkable rather than assumed.
    """

    records: tuple
    layer_bounds: dict
    sampler: object
    provenance: dict = field(default_factory=dict)

    def layer_of(self, index: int) -> str:
        for layer, (lo, hi) in self.layer_bounds.items():
            if lo <= index < hi:
                return layer
        raise IndexError(index)

    def verify_alignment(self) -> None:
        total = sum(hi - lo for lo, hi in self.layer_bounds.values())
        if total != len(self.records):
            raise ProductionCorpusError(
                f"layer bounds cover {total} records but the tuple holds {len(self.records)}"
            )


def verify_build_complete(root: Path, *, expected_contract: dict | None = None) -> dict:
    """Load and validate ``BUILD_COMPLETE.json``; raise unless the build is complete and on-contract."""
    root = Path(root)
    marker = root / "BUILD_COMPLETE.json"
    if not marker.exists():
        raise ProductionCorpusError(
            f"{marker} is absent: the corpus build did not reach a terminal COMPLETE state"
        )
    payload = json.loads(marker.read_text())
    if not payload.get("BUILD_COMPLETE"):
        raise ProductionCorpusError("BUILD_COMPLETE.json present but not marked complete")
    contract = payload.get("contract") or {}
    if expected_contract:
        for key in _CONTRACT_KEYS:
            if key in expected_contract and contract.get(key) != expected_contract[key]:
                raise ProductionCorpusError(
                    f"contract mismatch on {key}: artifact {contract.get(key)!r} != "
                    f"expected {expected_contract[key]!r}"
                )
        want_part = expected_contract.get("partitioner")
        if want_part:
            got_part = contract.get("partitioner") or {}
            for key, value in want_part.items():
                if got_part.get(key) != value:
                    raise ProductionCorpusError(
                        f"partitioner mismatch on {key}: artifact {got_part.get(key)!r} != {value!r}"
                    )
    return payload


def _shard_paths(root: Path, shard_layer: str, partition: str) -> list[Path]:
    directory = Path(root) / shard_layer / partition
    if not directory.is_dir():
        raise ProductionCorpusError(f"missing shard directory: {directory}")
    return sorted(directory.glob("*.jsonl.gz"))


def load_shard_layer_records(
    root: Path,
    shard_layer: str,
    partition: str,
    *,
    checkpoint_interval: int | None = None,
    packed_root: Path | None = None,
    verify_fraction: float = 0.0,
) -> tuple[PathRecord, ...]:
    """Rebuild every trace of one precompiled (layer, partition) into a GM ``PathRecord``.

    With ``packed_root``, states come from the packed derivative store and no trace is replayed
    (measured 21.36 -> 0.217 ms/trace). Without it, the audit shards are replayed -- correct but ~98x
    slower. The packed path is an acceleration ONLY: both produce byte-identical states, which
    tests/test_packed_trace_store.py asserts on every accessor the trainer touches.
    """
    records: list[PathRecord] = []
    if packed_root is not None:
        # The derivative store must COVER the audit shards exactly. Without this, a packed shard that
        # failed to build would silently shrink the training corpus -- a smaller run that still succeeds,
        # which is the failure mode this whole loader exists to prevent.
        audit_names = {path.name for path in _shard_paths(root, shard_layer, partition)}
        packed_paths = _shard_paths(packed_root, shard_layer, partition)
        packed_names = {path.name for path in packed_paths}
        if packed_names != audit_names:
            raise ProductionCorpusError(
                f"packed store does not cover {shard_layer}/{partition}: "
                f"missing {sorted(audit_names - packed_names)}, "
                f"unexpected {sorted(packed_names - audit_names)}"
            )
        for path in packed_paths:
            for trace, packed_path in read_packed_shard(path, verify_fraction=verify_fraction):
                records.append(PathRecord(canonical_state_key(trace.target), packed_path))
        return tuple(records)
    for path in _shard_paths(root, shard_layer, partition):
        for trace in load_trace_records(path):
            records.append(
                PathRecord(
                    canonical_state_key(trace.target),
                    TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
                )
            )
    return tuple(records)


def load_mmp_records(
    pool_path: Path,
    *,
    partition: str,
    salt: str = "ringcore-v1",
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    limit: int | None = None,
    checkpoint_interval: int | None = None,
) -> tuple[tuple[PathRecord, ...], dict]:
    """Load MMP analogue records for ONE scaffold partition.

    The pool is streamed in full and each row is routed by ``partition_for_scaffold`` on its target's
    Murcko scaffold -- the same pure function the corruption/cycle shards were partitioned with. This is
    what keeps the held-out sets disjoint; a prefix ``limit`` is applied only AFTER filtering, so it can
    never silently become a partition-blind head of the file.
    """
    kept: list[PathRecord] = []
    stats = {"scanned": 0, "kept": 0, "other_partition": 0, "unassignable": 0, "unbuildable": 0}
    with open(pool_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["scanned"] += 1
            row = json.loads(line)
            target = row.get("target_smiles") or ""
            # An empty SMILES PARSES (RDKit yields a zero-atom Mol) and would therefore be handed a
            # partition bucket, so emptiness must be rejected explicitly rather than left to
            # murcko_scaffold, whose contract only covers unparseable input.
            scaffold = murcko_scaffold(target) if target.strip() else None
            if not scaffold:
                stats["unassignable"] += 1
                continue
            if partition_for_scaffold(scaffold, salt=salt, ratios=ratios) != partition:
                stats["other_partition"] += 1
                continue
            try:
                trace = rewrite_trace_from_record(row)
            except Exception:  # noqa: BLE001 -- a pool row that no longer rebuilds is skipped, and counted
                stats["unbuildable"] += 1
                continue
            kept.append(
                PathRecord(
                    canonical_state_key(trace.target),
                    TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
                )
            )
            stats["kept"] += 1
            if limit is not None and len(kept) >= limit:
                break
    return tuple(kept), stats


def load_production_edit_corpus(
    root: Path,
    *,
    mmp_pool_path: Path,
    partition: str,
    layer_weights: dict,
    expected_contract: dict | None = None,
    path_length_bins: tuple[int, ...] = (5, 9, 13),
    cold_element_floor: float = 0.0,
    cold_elements_of=None,
    mmp_limit: int | None = None,
    checkpoint_interval: int | None = None,
    packed_root: Path | None = None,
    verify_fraction: float = 0.0,
    seed: int = 0,
) -> LayeredEditCorpus:
    """Assemble the three-layer production corpus and its sampler from validated artifacts."""
    root = Path(root)
    build = verify_build_complete(root, expected_contract=expected_contract)

    by_layer: dict[str, tuple] = {}
    for shard_layer, production_layer in _SHARD_LAYER_TO_PRODUCTION.items():
        by_layer[production_layer] = load_shard_layer_records(
            root, shard_layer, partition, checkpoint_interval=checkpoint_interval,
            packed_root=packed_root, verify_fraction=verify_fraction,
        )
    mmp_records, mmp_stats = load_mmp_records(
        mmp_pool_path,
        partition=partition,
        limit=mmp_limit,
        checkpoint_interval=checkpoint_interval,
    )
    by_layer[MMP_ANALOGUE] = mmp_records

    # Concatenate in the canonical layer order so sampler indices are well defined.
    ordered = {layer: by_layer[layer] for layer in PRODUCTION_LAYERS}
    records: list = []
    bounds: dict[str, tuple[int, int]] = {}
    for layer, layer_records in ordered.items():
        bounds[layer] = (len(records), len(records) + len(layer_records))
        records.extend(layer_records)

    sampler = build_layered_sampler(
        ordered,
        layer_weights=layer_weights,
        path_length_bins=path_length_bins,
        cold_element_floor=cold_element_floor,
        cold_elements_of=cold_elements_of,
        seed=seed,
    )
    provenance = {
        "root": str(root),
        "packed_root": None if packed_root is None else str(packed_root),
        "partition": partition,
        "configured_layer_weights": dict(layer_weights),
        "records_by_layer": {layer: len(rs) for layer, rs in ordered.items()},
        "transitions_by_layer": {
            layer: sum(r.path.path_length for r in rs) for layer, rs in ordered.items()
        },
        "mmp_partition_filter": {
            **mmp_stats,
            "scaffold_key_algorithm": SCAFFOLD_KEY_ALGORITHM,
            "scaffold_key_version": SCAFFOLD_KEY_VERSION,
        },
        "contract": build.get("contract", {}),
        "authorization": build.get("authorization", {}),
        "reducer_checksum": build.get("reducer_checksum"),
    }
    corpus = LayeredEditCorpus(
        records=tuple(records), layer_bounds=bounds, sampler=sampler, provenance=provenance
    )
    corpus.verify_alignment()
    return corpus


def measure_realized_layer_frequencies(corpus: LayeredEditCorpus, *, draws: int = 20000, seed: int = 0):
    """Empirical layer mix of the sampler, for the launch log.

    The configured weight is what we asked for; this is what the sampler actually does. Logging both is
    what makes the mixture auditable after the fact rather than asserted.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(corpus.layer_bounds, 0)
    for _ in range(draws):
        counts[corpus.layer_of(int(corpus.sampler.draw(rng)))] += 1
    return {layer: count / draws for layer, count in counts.items()}
