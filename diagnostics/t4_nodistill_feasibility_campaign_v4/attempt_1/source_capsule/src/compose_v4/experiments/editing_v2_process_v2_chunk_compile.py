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
It weights nothing and trains nothing.  It compiles Active8-accepted transitions
of one chunk in one partition role and returns both products of that single
chemistry pass: the compact teacher fibers, and the model-ready collated batch.

By default it compiles EVERY such transition.  ``source_subset`` narrows which
source molecules are materialized, which is a different thing from binding a
sampling law: it says "materialize chemistry for these sources" and still says
nothing about how the resulting rows are weighted, so the law remains free to
change without invalidating a leaf.  The subset's digest is returned and
published, because the same ``offset`` names different rows under different
subsets and the two must never be confused.

It exists because the compile unit is a contiguous byte range while a selection
is a set of molecules, and those disagree badly when the selection is scattered:
the V2 recipe's 4,921 ring_system_restate rows sit ~195-per-task across all 27
synthetic train tasks, so compiling whole tasks materializes 140,743 records to
obtain 4,921 -- 28.6x the wanted chemistry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import hashlib
import io
import time

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
    offset: int = 0,
    limit: int | None = None,
    source_subset: Iterable[str] | None = None,
    chemistry_feature_cache_limit: int = 16_384,
    compile_batch_size: int = 25,
    deadline_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile a whole chunk and collate it from ONE chemistry pass.

    ``offset``/``limit`` select a contiguous SLICE of the chunk.  They exist so
    the unit of parallelism can be smaller than a chunk: chunk sizes here differ
    by 4.5x (~2,827 entries in a synthetic-walk chunk against ~12,858 in a
    multistep one), so one-container-per-chunk floors the wall clock at the
    largest chunk no matter how many containers run.  Slicing divides the work
    evenly instead.  Neither changes how any individual entry is compiled.

    ``compile_batch_size`` exists for OBSERVABILITY, not for throughput.  The
    expensive chemistry lives inside one batched
    ``compile_teacher_successor_fibers_support_only`` call, so compiling a whole
    slice in one shot emits no progress at all until it is finished -- a worker
    could run an entire 45-minute timeout and report nothing before dying, which
    is exactly what happened.  Compiling in sub-batches over the SAME shared
    chemistry cache makes rate measurable while it is happening; the compiled
    entries are identical either way.

    ``deadline_seconds`` bounds the compile against a wall.  When the measured
    rate projects past it, the compile STOPS and returns what it finished rather
    than being killed with nothing: slices are offset-addressed, so a short
    slice is a valid published slice and ``next_offset`` names the remainder to
    re-queue.  Degrading to less work beats losing all of it.

    The deadline is enforced BETWEEN sub-batches, so it is precise to one
    sub-batch and the first one always runs -- there is no rate to project from
    until something has been measured.  ``compile_batch_size`` must therefore be
    small enough that a single sub-batch fits inside the reserve; at 25 entries
    and the measured 2,046 ms/entry that is ~51s against a five-minute reserve.
    A sub-batch that overruns anyway is reported in the progress event as
    ``overran_deadline``, so the cause is visible rather than inferred from a
    kill.
    """

    if type(offset) is not int or offset < 0:
        raise ProcessV2ChunkCompileError("chunk compile offset must be a nonnegative int")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ProcessV2ChunkCompileError("chunk compile limit must be a positive int")
    if type(compile_batch_size) is not int or compile_batch_size <= 0:
        raise ProcessV2ChunkCompileError("compile batch size must be a positive int")
    if deadline_seconds is not None and (
        not isinstance(deadline_seconds, (int, float)) or deadline_seconds <= 0
    ):
        raise ProcessV2ChunkCompileError("deadline seconds must be a positive number")

    rows = chunk_transition_rows(
        source,
        task_identity_sha256=task_identity_sha256,
        partition_role=partition_role,
    )
    # A SOURCE SUBSET, when given, restricts the stream to named source molecules
    # BEFORE slicing, so offset/limit address the selected stream and no worker
    # is spent on a range that contains nothing wanted.
    #
    # This exists because the compile unit is a contiguous byte range, not a set
    # of molecules, and the two disagree badly when a selection is scattered.
    # MEASURED: the 4,921 ring_system_restate rows the V2 recipe wants are spread
    # across all 27 synthetic train tasks at ~195 each, inside 140,743 total
    # records -- compiling whole tasks pays for 28.6x the wanted chemistry
    # (~$47 against ~$1.66).
    #
    # It restricts WHICH CHEMISTRY IS MATERIALIZED and says nothing about how
    # rows are weighted afterwards, so the sampling law stays out of
    # compilation. The subset digest is returned and published so a slice
    # compiled from a subset is never mistaken for one compiled whole: the same
    # offset means different rows under different subsets.
    source_subset_sha256 = None
    if source_subset is not None:
        wanted = frozenset(str(key) for key in source_subset)
        if not wanted:
            raise ProcessV2ChunkCompileError("source subset must name at least one source")
        source_subset_sha256 = canonical_sha256(sorted(wanted))
        rows = [
            row
            for row in rows
            if str(row["candidate_evidence"]["source_canonical_key"]) in wanted
        ]
        if not rows:
            raise ProcessV2ChunkCompileError(
                f"source subset {source_subset_sha256[:16]} selects no transitions "
                f"in task {task_identity_sha256[:12]}"
            )
    selected_row_count = len(rows)
    rows = rows[offset:] if limit is None else rows[offset : offset + limit]
    if not rows:
        raise ProcessV2ChunkCompileError(
            f"chunk slice offset={offset} limit={limit} selects no transitions"
        )
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

    compiled: list[dict[str, Any]] = []
    started = time.monotonic()
    stopped_on_deadline = False
    for batch_start in range(0, len(resolver_entries), compile_batch_size):
        batch_end = min(batch_start + compile_batch_size, len(resolver_entries))
        compiled.extend(
            compile_prepared_entries(
                model,
                sources[batch_start:batch_end],
                targets[batch_start:batch_end],
                resolver_entries[batch_start:batch_end],
                task_identity_sha256=task_identity_sha256,
                chemistry_feature_cache=shared_cache,
            )
        )
        elapsed = time.monotonic() - started
        rate = elapsed / len(compiled)
        remaining = len(resolver_entries) - len(compiled)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "process_v2_chunk_compile_progress",
                    "task_identity_sha256": task_identity_sha256,
                    "completed": len(compiled),
                    "total": len(resolver_entries),
                    "elapsed_seconds": elapsed,
                    "ms_per_entry": rate * 1000.0,
                    "projected_remaining_seconds": rate * remaining,
                    "deadline_seconds": deadline_seconds,
                    "overran_deadline": bool(
                        deadline_seconds is not None and elapsed > float(deadline_seconds)
                    ),
                }
            )
        # Project BEFORE committing to another sub-batch, so the decision is
        # made on measured rate rather than discovered at the wall.
        if (
            deadline_seconds is not None
            and remaining
            and elapsed + rate * min(compile_batch_size, remaining) > float(deadline_seconds)
        ):
            stopped_on_deadline = True
            break
    # The one-shot path sorts its whole output; concatenated sub-batches must be
    # sorted the same way or the published order would depend on batching.
    compiled.sort(key=lambda row: row["p50_entry_sha256"])
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
        "source_subset_sha256": source_subset_sha256,
        "source_subset_selected_rows": selected_row_count,
        "task_identity_sha256": task_identity_sha256,
        "partition_role": partition_role,
        "entry_offset": int(offset),
        "entry_count": len(compiled),
        "entries": compiled,
        "batch": batch,
        "family_counts": dict(sorted(families.items())),
        "capability_cell_counts": dict(sorted(cells.items())),
        "chemistry_states_cached": len(shared_cache),
        "requested_entry_count": len(resolver_entries),
        "stopped_on_deadline": stopped_on_deadline,
        # Named so a short slice is re-queueable rather than silently lost.
        "next_offset": int(offset) + len(compiled),
        "unfinished_entry_count": len(resolver_entries) - len(compiled),
        "compile_seconds": time.monotonic() - started,
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
    offset = int(result["entry_offset"])
    # Slice-addressed, because the compile unit is a slice: two workers may hold
    # different ranges of the same chunk and must not collide on one path.
    #
    # A SOURCE SUBSET changes what an offset means -- it indexes the FILTERED
    # stream -- so the subset digest is part of the address, not just the
    # payload. Without it two logically different materializations address the
    # same location: republishing a corrected subset compile over an earlier
    # whole-task run raised ImmutableArtifactError on
    # ``000000140-000000015/ENTRIES.json``, same path, different rows. The
    # immutable writer caught that one; the next could be a silent overwrite
    # under a writer that permitted it, or a corpus that quietly mixes both.
    #
    # Whole-task slices keep their historical two-field name, so every artifact
    # already on disk stays addressable and no published corpus is orphaned.
    subset_digest = result.get("source_subset_sha256")
    slice_id = f"{offset:09d}-{len(entries):09d}"
    if subset_digest:
        slice_id = f"{slice_id}-{str(subset_digest)[:16]}"
    directory = Path(output_root) / CHUNKS_DIRNAME / task_identity_sha256 / slice_id

    entries_payload = {
        "schema": SHARD_SCHEMA,
        "schema_version": SHARD_SCHEMA_VERSION,
        "task_identity_sha256": task_identity_sha256,
        "partition_role": str(result["partition_role"]),
        "entry_offset": offset,
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
        "entry_offset": offset,
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
