"""Compile one source chunk selection-free, in a single chemistry pass.

WHY THIS EXISTS
---------------
The P50 path compiles the entries a *selection* names, and then a separate stage
collates them.  Both call ``prepare_factorized_mark_batch`` on the same states,
so the most expensive chemistry in the pipeline -- the semantic cycle-close and
atom-restate admission masks, measured at 1.48 s and 1.06 s per state -- is
computed **twice**.  Measured on real drug-like leads, fusing the two behind one
shared ``chemistry_feature_cache`` takes the prep path from 46.91 s to 25.77 s
for twelve entries (1.82x), with the compiled entries byte-identical; collation
alone falls from 21.53 s to 0.85 s because every state hits the cache the
compile just filled.

The second reason is that a leaf carrying ``selection_sha256`` is bound to one
sampling law.  Changing the law -- to importance-corrected corpus sampling, or
to run an epsilon ablation -- invalidates every compiled leaf and pays the whole
corpus cost again.  Compiling a whole chunk instead makes the chemistry a
per-corpus cost and leaves sampling free forever after.

WHAT IT DOES NOT DO
-------------------
It selects nothing, weights nothing and trains nothing.  It compiles every
Active8-accepted transition of one chunk in one partition role and returns both
products of that single chemistry pass: the compact teacher fibers, and the
model-ready collated batch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import hashlib
import io

import torch

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
)
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.data.editing_v2_process_v2_gate_zero import (
    ProcessV2GateZeroError,
    iter_process_v2_role_shards,
)
from compose_v4.experiments.editing_v2_process_v2_p50_runtime import (
    ProcessV2P50RuntimeError,
    _candidate_from_transition,
    _COMPILE_SUPPORT_TIME,
    compile_prepared_entries,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    ProcessV2T1Source,
    resolve_process_v2_t1_entries,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.trace_shard import decode_state

TRAIN_ROLE = "train"


class ProcessV2ChunkCompileError(RuntimeError):
    """One chunk could not be compiled selection-free."""


def chunk_transition_rows(
    source: ProcessV2T1Source,
    *,
    task_identity_sha256: str,
    partition_role: str = TRAIN_ROLE,
) -> tuple[dict[str, Any], ...]:
    """Every Active8-accepted transition of one chunk, in published order.

    Reuses the gate's own role-shard reader so a sealed role cannot be opened by
    this path either: the role is a parameter of the reader, not a filter
    applied afterwards.
    """

    try:
        shards = iter_process_v2_role_shards(
            source.active8_run_root,
            contracts=source.contracts,
            index=source.index,
            partition_role=partition_role,
            task_identity_sha256=task_identity_sha256,
        )
        rows = [row for _shard, published in shards for row in published]
    except ProcessV2GateZeroError as error:
        raise ProcessV2ChunkCompileError(str(error)) from error
    if not rows:
        raise ProcessV2ChunkCompileError(
            f"chunk {task_identity_sha256} has no accepted {partition_role} transitions"
        )
    return tuple(rows)


def compile_process_v2_chunk_shard(
    source: ProcessV2T1Source,
    model: FactorizedTraceletRateModel,
    *,
    task_identity_sha256: str,
    partition_role: str = TRAIN_ROLE,
    limit: int | None = None,
    chemistry_feature_cache_limit: int = 16_384,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile a whole chunk and collate it from ONE chemistry pass.

    ``limit`` truncates the chunk for a bounded pilot; it changes how much is
    compiled, never how any individual entry is compiled.
    """

    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ProcessV2ChunkCompileError("chunk compile limit must be a positive int")

    rows = chunk_transition_rows(
        source,
        task_identity_sha256=task_identity_sha256,
        partition_role=partition_role,
    )
    if limit is not None:
        rows = rows[:limit]
    entries = [_candidate_from_transition(row) for row in rows]

    resolver_entries = [
        {**entry, "panel_entry_sha256": entry["p50_entry_sha256"]} for entry in entries
    ]
    try:
        resolved = resolve_process_v2_t1_entries(source, resolver_entries)
    except (ProcessV2T1PanelError, ProcessV2P50RuntimeError) as error:
        raise ProcessV2ChunkCompileError(str(error)) from error

    addressed_by_entry: dict[str, Any] = {}
    for item in resolved:
        for entry in item.panel_entries:
            addressed_by_entry[str(entry["panel_entry_sha256"])] = item.addressed_trace

    sources: list[Any] = []
    targets: list[Any] = []
    for entry in resolver_entries:
        addressed = addressed_by_entry.get(str(entry["panel_entry_sha256"]))
        if addressed is None:
            raise ProcessV2ChunkCompileError(
                "a chunk transition did not resolve to an addressed trace"
            )
        progress = int(entry["progress_index"])
        source_state = addressed.path.state_at(progress)
        target_state = addressed.path.state_at(progress + 1)
        # The published evidence is the authority on which exact states this
        # transition names; a replay that lands elsewhere is a corpus fault, not
        # a compile detail, so it fails here rather than being compiled anyway.
        if (
            persistent_slot_state_sha256(source_state) != entry["source_state_sha256"]
            or persistent_slot_state_sha256(target_state) != entry["target_state_sha256"]
        ):
            raise ProcessV2ChunkCompileError(
                "a chunk transition replays to another exact state"
            )
        sources.append(source_state)
        targets.append(target_state)

    # The collator owns the cache and is a frozen dataclass, so it is built
    # FIRST and its cache handed to the compile.  That ordering is the whole
    # optimisation: the compile fills the cache, collation then hits it for
    # every state instead of recomputing the masks.
    collator = FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        chemistry_feature_cache_limit=chemistry_feature_cache_limit,
    )
    shared_cache = collator._chemistry_feature_cache

    compiled = compile_prepared_entries(
        model,
        sources,
        targets,
        resolver_entries,
        task_identity_sha256=task_identity_sha256,
        progress_callback=progress_callback,
        chemistry_feature_cache=shared_cache,
    )
    batch = collator(
        [
            FactorizedMarkExample(
                state=decode_state(entry["exact_state"]),
                time=_COMPILE_SUPPORT_TIME,
                teacher_action=None,
                teacher_rule_name=None,
                teacher_rate=1.0,
                importance_weight=1.0,
            )
            for entry in compiled
        ]
    )
    if batch.batch_size != len(compiled):
        raise ProcessV2ChunkCompileError("chunk collation changed entry cardinality")

    families: dict[str, int] = {}
    cells: dict[str, int] = {}
    for entry in compiled:
        families[str(entry["model_family"])] = (
            families.get(str(entry["model_family"]), 0) + 1
        )
        cells[str(entry["capability_cell_id"])] = (
            cells.get(str(entry["capability_cell_id"]), 0) + 1
        )
    return {
        "task_identity_sha256": task_identity_sha256,
        "partition_role": partition_role,
        "entry_count": len(compiled),
        "entries": compiled,
        "batch": batch,
        "family_counts": dict(sorted(families.items())),
        "capability_cell_counts": dict(sorted(cells.items())),
        "chemistry_states_cached": len(shared_cache),
    }


CHUNKS_DIRNAME = "chunks"
ENTRIES_FILENAME = "ENTRIES.json"
BATCH_FILENAME = "BATCH.pt"
RECEIPT_FILENAME = "RECEIPT.json"
SHARD_SCHEMA = "compose.editing_v2.process_v2_chunk_shard"
SHARD_SCHEMA_VERSION = 1
SHARD_STATUS = "PROCESS_V2_CHUNK_SHARD_EVIDENCE_ONLY_NO_AUTHORITY"


def publish_process_v2_chunk_shard(
    result: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Publish one compiled chunk immutably: entries, tensors, and a receipt.

    BOTH products of the single chemistry pass are stored, deliberately:

    * ``ENTRIES.json`` -- the compact teacher fibers, ~2.5 kB per entry. This is
      what a resampler reads, so changing the sampling law, reweighting, or
      running an epsilon arm costs nothing beyond reading it.
    * ``BATCH.pt`` -- the collated model-ready tensors, ~92 kB per entry. This
      is what training reads. It is stored rather than rebuilt because
      collation is NOT chemistry-free despite its name: it runs the admission
      masks, so collating per batch would re-pay the entire corpus chemistry on
      every epoch.

    The receipt binds both by content hash and is self-hashed, so a consumer
    can authenticate the tensors it is about to train on without parsing them.
    """

    task_identity_sha256 = str(result["task_identity_sha256"])
    entries = list(result["entries"])
    directory = Path(output_root) / CHUNKS_DIRNAME / task_identity_sha256

    entries_payload = {
        "schema": SHARD_SCHEMA,
        "schema_version": SHARD_SCHEMA_VERSION,
        "task_identity_sha256": task_identity_sha256,
        "partition_role": str(result["partition_role"]),
        "entry_count": len(entries),
        "entries": entries,
    }
    entries_bytes = canonical_bytes(entries_payload) + b"\n"
    write_bytes_if_absent(directory / ENTRIES_FILENAME, entries_bytes)

    buffer = io.BytesIO()
    torch.save(result["batch"], buffer)
    batch_bytes = buffer.getvalue()
    write_bytes_if_absent(directory / BATCH_FILENAME, batch_bytes)

    body = {
        "schema": f"{SHARD_SCHEMA}_receipt",
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": SHARD_STATUS,
        **authority_false_block(),
        "task_identity_sha256": task_identity_sha256,
        "partition_role": str(result["partition_role"]),
        "entry_count": len(entries),
        "entries_file_sha256": hashlib.sha256(entries_bytes).hexdigest(),
        "entries_payload_sha256": canonical_sha256(entries_payload),
        "batch_file_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "batch_bytes": len(batch_bytes),
        "chemistry_states_compiled": int(result["chemistry_states_cached"]),
        "family_counts": dict(result["family_counts"]),
        "capability_cell_counts": dict(result["capability_cell_counts"]),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    write_bytes_if_absent(
        directory / RECEIPT_FILENAME, canonical_bytes(receipt) + b"\n"
    )
    return receipt


__all__ = [
    "publish_process_v2_chunk_shard",
    "SHARD_STATUS",
    "SHARD_SCHEMA_VERSION",
    "SHARD_SCHEMA",
    "RECEIPT_FILENAME",
    "ENTRIES_FILENAME",
    "CHUNKS_DIRNAME",
    "BATCH_FILENAME",
    "TRAIN_ROLE",
    "ProcessV2ChunkCompileError",
    "chunk_transition_rows",
    "compile_process_v2_chunk_shard",
]
