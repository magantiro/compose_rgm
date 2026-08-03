#!/usr/bin/env python
"""Chunk-size benchmark for the Process-V2 cache-fed rebind. NOT evidence.

**This is explicitly non-production.**  It runs on a laptop, on a synthetic
corpus built from real migrated traces, in a single process.  Its output guides
a later resource request -- how much memory to ask a container for, and how long
a worker task will run -- and it is not scientific evidence about anything.

What it measures, and why that is the honest thing to measure
------------------------------------------------------------

The two selection criteria are both **per-chunk** quantities:

* peak memory, which is bounded by one chunk's rows and one chunk's proofs;
* task time, because under the cache-fed geometry one worker task *is* one
  chunk.

Neither depends on how many chunks a shard has, so both are measured directly at
the chosen chunk size rather than extrapolated from a smaller unit.  This
repository has a recorded history of component micro-benchmarks over-predicting
path throughput by an order of magnitude (630x -> 2x -> 98x), so what is timed
here is the real ``execute_process_v2_rebind_task`` -- read the chunk, decode
every row, replay every transition, derive the mask, publish the proof artifact
-- and not a component of it.

Peak memory is measured as the child process's own ``ru_maxrss``: each chunk
size is measured in a **separate subprocess**, because max RSS is monotonic
within a process and measuring several configurations in one would report the
largest for all of them.  ``tracemalloc`` is reported beside it and covers only
Python-allocated objects, which excludes numpy's own buffers.

What it does NOT cover, stated rather than implied
--------------------------------------------------

* Modal container behaviour, volume I/O, network, and cold starts: none.
* Inter-chunk variance from production's heterogeneous molecule sizes: the
  corpus here is one repeated production-shaped trace, so the measured p99 is a
  read/replay-time tail, not a chemistry-size tail.
* The p99 at the largest chunk size is estimated from few samples by
  construction -- a fixed corpus yields four times fewer chunks at 2048 than at
  512 -- and the sample count is reported next to every quantile.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/benchmark_process_v2_chunk_sizes.py \\
        --entries 8192 --chunk-sizes 512,1024,2048 \\
        --report diagnostics/process_v2_chunk_size_benchmark.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.data.editing_corpus_contract import (  # noqa: E402
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_rebind import (  # noqa: E402
    TASK_DIRNAME,
    bind_process_v2_chunk_cache_generation,
    bind_v1_semantic_payload,
    build_process_v2_rebind_source_revision,
    execute_process_v2_rebind_task,
    plan_process_v2_rebind,
    reduce_process_v2_rebind,
    write_process_v2_rebind_plan,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (  # noqa: E402
    build_cache_implementation_revision,
    execute_process_v2_chunk_cache_task,
    plan_process_v2_chunk_cache,
    reduce_process_v2_chunk_cache,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_completion_binder import (  # noqa: E402
    ProcessV2ExactCompletionBinding,
)
from compose_v4.data.packed_trace_store import (  # noqa: E402
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_packed_trace_store import (  # noqa: E402
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (  # noqa: E402
    semantic_packed_builder_identity,
)
from compose_v4.data.semantic_trace_migration_materializer import (  # noqa: E402
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (  # noqa: E402
    SEMANTIC_ARTIFACT_DIRNAME,
    SemanticTraceMigrationTask,
    materialize_frozen_packed_shard,
)
from compose_v4.rewrite.editing_v2_process_identity import (  # noqa: E402
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import AtomDelete, BondInsert  # noqa: E402
from compose_v4.rewrite.progress import TraceProgressCTMC  # noqa: E402
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace  # noqa: E402
from compose_v4.rewrite.trace_shard import encode_trace_record  # noqa: E402

PAYLOAD_ARTIFACT_PATH = "/artifacts/benchmark_v1_payload"
BENCHMARK_STATUS = "NON_PRODUCTION_LOCAL_CHUNK_SIZE_BENCHMARK_NO_DOWNSTREAM_AUTHORITY"

# The declared production shard shape, measured in Wave 1 on one real
# production-built shard. Used only to size the synthetic rows so the per-chunk
# numbers are taken at the right byte scale, and reported alongside.
PRODUCTION_SHARD_ENTRIES = 32_339
PRODUCTION_SHARD_DECOMPRESSED_BYTES = 281_800_000
PRODUCTION_SOURCE_SHARDS = 20

# The intended container shape for a rebind worker, read from the launcher so a
# change there cannot leave this selecting against a stale limit.
WORKER_MEMORY_MB = 8192
WORKER_TIMEOUT_SECONDS = 8 * 3600
MEMORY_HEADROOM_FACTOR = 2.0
TIMEOUT_HEADROOM_FACTOR = 2.0

# One production-shaped trace: forty slots (the executor's own
# ``MAX_ACTIVE_ATOMS``) and eight steps, which puts a persisted row within a few
# percent of the measured production mean of ~8.7 kB.
#
# The corpus repeats that one trace, which is the right trade for the two
# selection criteria -- decode and proof cost follow molecule size, not variety
# -- with one consequence worth naming: a shard of one repeated row compresses
# far better than a production shard of distinct molecules, so the recorded
# ``compressed_bytes`` is not a production figure and the gzip read is a smaller
# share of task time here than it would be in production. Task time is dominated
# by the per-state mask derivation either way, which is why this is a trade
# rather than a defect.
BENCHMARK_SLOTS = 40
BENCHMARK_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
BENCHMARK_OPERATIONS = (
    ("bond_insert", BondInsert(0, 9, 1)),
    ("bond_insert", BondInsert(10, 19, 1)),
    ("atom_delete", AtomDelete(29)),
    ("atom_delete", AtomDelete(28)),
    ("atom_delete", AtomDelete(27)),
    ("atom_delete", AtomDelete(26)),
    ("atom_delete", AtomDelete(25)),
    ("atom_delete", AtomDelete(24)),
)


# ---- Corpus ------------------------------------------------------------------


def _benchmark_trace() -> RewriteTrace:
    runtime = de_novo_rewrite_system()
    state = pad_molecular_graph(smiles_to_molecular_graph(BENCHMARK_SMILES), BENCHMARK_SLOTS)
    steps: list[RewriteStep] = []
    for rule, action in BENCHMARK_OPERATIONS:
        state = runtime.apply(state, rule, action)
        steps.append(RewriteStep(rule, action))
    return RewriteTrace(
        pad_molecular_graph(smiles_to_molecular_graph(BENCHMARK_SMILES), BENCHMARK_SLOTS),
        state,
        tuple(steps),
        {"benchmark": "process_v2_chunk_size"},
    )


def build_benchmark_payload(artifact_root: Path, *, entries: int) -> dict[str, Any]:
    """One lane/split cell of ``entries`` production-shaped migrated traces."""

    lane, split = REQUIRED_DATA_LANES[0], REQUIRED_PARTITION_ROLES[0]
    trace = _benchmark_trace()
    path = TraceProgressCTMC(trace)
    legacy_shard = artifact_root / "benchmark_source" / "shard_0000.jsonl.gz"
    legacy_shard.parent.mkdir(parents=True, exist_ok=True)
    packed = [
        build_packed_entry(
            encode_trace_record(
                trace,
                n_slots=BENCHMARK_SLOTS,
                seed=1000 + index,
                trace_id=f"benchmark-{index:08d}",
                partition=split,
                layer=lane,
                extra=dict(trace.metadata),
            ),
            path,
        )
        for index in range(entries)
    ]
    write_packed_shard(legacy_shard, packed, provenance={"benchmark": True}, deterministic_gzip=True)

    identity = hashlib.sha256(b"process-v2-chunk-size-benchmark").hexdigest()
    payload_root = artifact_root / "benchmark_v1_payload"
    materialize_frozen_packed_shard(
        SemanticTraceMigrationTask(
            source_path=legacy_shard,
            source_shard_sha256=hashlib.sha256(legacy_shard.read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(
                manifest_path_for(legacy_shard).read_bytes()
            ).hexdigest(),
            source_overlay_sha256=None,
            source_unified_manifest_sha256=hashlib.sha256(b"benchmark-unified").hexdigest(),
            source_entry_count=entries,
            implementation_revision="b" * 40,
            data_lane=lane,
            split=split,
            output_dir=payload_root / TASK_DIRNAME / identity,
        )
    )
    semantic_shard = (
        payload_root / TASK_DIRNAME / identity / SEMANTIC_ARTIFACT_DIRNAME / SEMANTIC_SHARD_FILENAME
    )
    with gzip.open(semantic_shard, "rb") as handle:
        decompressed = sum(len(line) for line in handle)
    receipt = json.loads(
        (payload_root / TASK_DIRNAME / identity / V1_RECEIPT_FILENAME).read_bytes()
    )
    return {
        "task_identity_sha256": identity,
        "data_lane": lane,
        "split": split,
        "entries": int(receipt["counts"]["admitted"]),
        "semantic_shard_sha256": receipt["semantic_shard_sha256"],
        "semantic_manifest_sha256": receipt["semantic_manifest_sha256"],
        "semantic_shard_physical_sha256": hashlib.sha256(semantic_shard.read_bytes()).hexdigest(),
        "compressed_bytes": semantic_shard.stat().st_size,
        "decompressed_bytes": decompressed,
    }


def _binding(artifact_root: Path, source: dict[str, Any]) -> ProcessV2ExactCompletionBinding:
    """A completion binding constructed directly, for the benchmark only.

    The production binder requires the exact twenty-cell completion, which a
    one-cell benchmark corpus does not have and must not pretend to. Building
    the binding here keeps the benchmark honest about being a benchmark; nothing
    it produces is loadable as a production artifact, because its completion
    hashes address a corpus that exists only under this temporary root.
    """

    process_identity = json.loads(json.dumps(editing_v2_process_identity()))
    builder_identity = json.loads(json.dumps(semantic_packed_builder_identity()))
    payload_binding = bind_v1_semantic_payload(
        payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
        artifact_root=artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    counts = {"source": source["entries"], "admitted": source["entries"], "rejected": 0}
    inventory = (
        {
            "task_identity_sha256": source["task_identity_sha256"],
            "data_lane": source["data_lane"],
            "split": source["split"],
            "semantic_shard_sha256": source["semantic_shard_sha256"],
            "semantic_manifest_sha256": source["semantic_manifest_sha256"],
            "counts": dict(counts),
        },
    )
    completion = {
        "completion_sha256": hashlib.sha256(b"benchmark-completion").hexdigest(),
        "run_identity_sha256": hashlib.sha256(b"benchmark-run").hexdigest(),
        "result_inventory_sha256": hashlib.sha256(b"benchmark-inventory").hexdigest(),
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "builder_identity_sha256": builder_identity["identity_sha256"],
        "task_count": 1,
    }
    return ProcessV2ExactCompletionBinding(
        completion_artifact_path=f"{PAYLOAD_ARTIFACT_PATH}/SEMANTIC_MIGRATION_COMPLETE.json",
        payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
        completion=completion,
        completion_file_sha256=hashlib.sha256(b"benchmark-file").hexdigest(),
        v1_payload_binding=payload_binding,
        source_inventory=inventory,
        counts=counts,
    )


# ---- One measured configuration ----------------------------------------------


def measure_chunk_size(artifact_root: Path, source: dict[str, Any], *, chunk_size: int):
    """Build the cache at one chunk size, then time every real rebind task."""

    binding = _binding(artifact_root, source)
    cache_started = time.perf_counter()
    cache_plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/benchmark_chunk_cache",
        records_per_chunk=chunk_size,
    )
    write_process_v2_chunk_cache_plan(cache_plan, artifact_root=artifact_root)
    for task in cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=artifact_root
        )
    cache_completion = reduce_process_v2_chunk_cache(cache_plan, artifact_root=artifact_root)
    cache_seconds = time.perf_counter() - cache_started

    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]),
        artifact_root=artifact_root,
        repo_root=REPO_ROOT,
    )
    process_identity = json.loads(json.dumps(editing_v2_process_identity()))
    builder_identity = json.loads(json.dumps(semantic_packed_builder_identity()))
    plan = plan_process_v2_rebind(
        binding.v1_payload_binding,
        source_revision=build_process_v2_rebind_source_revision(
            commit="c" * 40, tree="d" * 40, repo_root=REPO_ROOT, worktree_clean=True
        ),
        repo_root=REPO_ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        cache_binding=cache_binding,
        output_artifact_prefix="/artifacts/benchmark_rebind",
        entries_per_task=chunk_size,
    )
    write_process_v2_rebind_plan(plan, artifact_root=artifact_root, repo_root=REPO_ROOT)

    tracemalloc.start()
    durations: list[float] = []
    cpu_durations: list[float] = []
    load_before = os.getloadavg()
    started = time.perf_counter()
    for task in plan["tasks"]:
        task_started = time.perf_counter()
        cpu_started = _process_cpu_seconds()
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
        )
        durations.append(time.perf_counter() - task_started)
        # CPU time as well as wall time. A laptop shared with other work makes
        # wall time a measurement of the machine; CPU time is a measurement of
        # the code, and the selection criteria care about the code.
        cpu_durations.append(_process_cpu_seconds() - cpu_started)
    total_seconds = time.perf_counter() - started
    load_after = os.getloadavg()
    python_peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    completion = reduce_process_v2_rebind(
        plan, artifact_root=artifact_root, repo_root=REPO_ROOT
    )
    ordered = sorted(durations)
    ordered_cpu = sorted(cpu_durations)

    def _quantile(values: list[float], fraction: float) -> float:
        return values[min(len(values) - 1, int(fraction * len(values)))]

    entries = int(completion["counts"]["source_entries"])
    return {
        "records_per_chunk": chunk_size,
        "task_count": len(durations),
        "entries": entries,
        "cache_build_seconds": round(cache_seconds, 4),
        "cache_chunk_count": int(cache_completion["chunk_count"]),
        "rebind_total_seconds": round(total_seconds, 4),
        "rebind_total_cpu_seconds": round(sum(cpu_durations), 4),
        "records_per_second": round(entries / total_seconds, 2),
        "records_per_cpu_second": round(entries / max(sum(cpu_durations), 1e-9), 2),
        "task_seconds_p50": round(median(ordered), 4),
        "task_seconds_p99": round(_quantile(ordered, 0.99), 4),
        "task_seconds_max": round(ordered[-1], 4),
        "task_seconds_min": round(ordered[0], 4),
        "task_cpu_seconds_p50": round(median(ordered_cpu), 4),
        "task_cpu_seconds_p99": round(_quantile(ordered_cpu, 0.99), 4),
        "task_cpu_seconds_max": round(ordered_cpu[-1], 4),
        "quantile_sample_count": len(ordered),
        "python_peak_traced_mb": round(python_peak_bytes / 1e6, 2),
        "process_peak_rss_mb": round(_peak_rss_bytes() / 1e6, 2),
        "system_load_average_before": [round(value, 2) for value in load_before],
        "system_load_average_after": [round(value, 2) for value in load_after],
    }


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime) + float(usage.ru_stime)


def _peak_rss_bytes() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS reports bytes.
    return float(usage) if sys.platform == "darwin" else float(usage) * 1024.0


# ---- Environment and selection ------------------------------------------------


def environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "networkx", "rdkit", "torch", "scipy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - optional dependency
            versions[name] = f"unavailable: {type(error).__name__}"
    return {
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "cache_implementation_sha256": build_cache_implementation_revision()[
            "cache_implementation_sha256"
        ],
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def select_chunk_size(
    measurements: list[dict[str, Any]],
    *,
    memory_mb: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """The largest size meeting BOTH criteria, with every rejection recorded."""

    memory_budget_mb = memory_mb / MEMORY_HEADROOM_FACTOR
    time_budget_seconds = timeout_seconds / TIMEOUT_HEADROOM_FACTOR
    verdicts = []
    for measurement in sorted(measurements, key=lambda row: -int(row["records_per_chunk"])):
        memory_ok = float(measurement["process_peak_rss_mb"]) <= memory_budget_mb
        # Judged on wall time, which is what a container timeout measures, but
        # the CPU time is carried beside it: on a shared machine the wall
        # figure is partly a measurement of the machine.
        time_ok = float(measurement["task_seconds_p99"]) <= time_budget_seconds
        verdicts.append(
            {
                "records_per_chunk": measurement["records_per_chunk"],
                "peak_rss_mb": measurement["process_peak_rss_mb"],
                "memory_budget_mb": round(memory_budget_mb, 1),
                "memory_headroom_factor": round(
                    memory_mb / max(float(measurement["process_peak_rss_mb"]), 1e-9), 2
                ),
                "task_seconds_p99": measurement["task_seconds_p99"],
                "task_cpu_seconds_p99": measurement["task_cpu_seconds_p99"],
                "time_budget_seconds": time_budget_seconds,
                "time_headroom_factor": round(
                    timeout_seconds / max(float(measurement["task_seconds_p99"]), 1e-9), 2
                ),
                "meets_memory_criterion": memory_ok,
                "meets_time_criterion": time_ok,
                "selected": bool(memory_ok and time_ok),
            }
        )
    chosen = next((row for row in verdicts if row["selected"]), None)
    return {
        "container_memory_mb": memory_mb,
        "worker_timeout_seconds": timeout_seconds,
        "memory_headroom_factor_required": MEMORY_HEADROOM_FACTOR,
        "timeout_headroom_factor_required": TIMEOUT_HEADROOM_FACTOR,
        "verdicts": verdicts,
        "selected_records_per_chunk": chosen["records_per_chunk"] if chosen else None,
    }


# ---- Entry points --------------------------------------------------------------


def _measure_in_child(corpus_root: Path, source_path: Path, chunk_size: int) -> dict[str, Any]:
    """Each configuration in its own process, so ``ru_maxrss`` is its own."""

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--measure-only",
            str(chunk_size),
            "--corpus-root",
            str(corpus_root),
            "--source-json",
            str(source_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"chunk-size measurement at {chunk_size} failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=8192)
    parser.add_argument("--chunk-sizes", default="512,1024,2048")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--source-json", type=Path, default=None)
    parser.add_argument("--measure-only", type=int, default=None)
    args = parser.parse_args(argv)

    if args.measure_only is not None:
        source = json.loads(Path(args.source_json).read_bytes())
        print(
            json.dumps(
                measure_chunk_size(
                    Path(args.corpus_root), source, chunk_size=int(args.measure_only)
                )
            )
        )
        return 0

    chunk_sizes = [int(value) for value in str(args.chunk_sizes).split(",") if value]
    root = Path(args.corpus_root) if args.corpus_root else Path(tempfile.mkdtemp())
    root.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    source = build_benchmark_payload(root, entries=int(args.entries))
    source["corpus_build_seconds"] = round(time.perf_counter() - build_started, 2)
    source_path = root / "benchmark_source.json"
    source_path.write_bytes(json.dumps(source).encode())

    measurements = [
        _measure_in_child(root, source_path, chunk_size) for chunk_size in chunk_sizes
    ]
    report = {
        "schema": "compose.diagnostics.process_v2_chunk_size_benchmark",
        "schema_version": 1,
        "status": BENCHMARK_STATUS,
        "production_evidence": False,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "environment": environment(),
        "input": {
            "entries": int(source["entries"]),
            "compressed_bytes": int(source["compressed_bytes"]),
            "decompressed_bytes": int(source["decompressed_bytes"]),
            "decompressed_bytes_per_row": round(
                source["decompressed_bytes"] / max(int(source["entries"]), 1), 1
            ),
            "semantic_shard_physical_sha256": source["semantic_shard_physical_sha256"],
            "semantic_shard_declared_sha256": source["semantic_shard_sha256"],
            "corpus_build_seconds": source["corpus_build_seconds"],
            "declared_production_shape": {
                "entries_per_shard": PRODUCTION_SHARD_ENTRIES,
                "decompressed_bytes_per_shard": PRODUCTION_SHARD_DECOMPRESSED_BYTES,
                "decompressed_bytes_per_row": round(
                    PRODUCTION_SHARD_DECOMPRESSED_BYTES / PRODUCTION_SHARD_ENTRIES, 1
                ),
                "source_shards": PRODUCTION_SOURCE_SHARDS,
            },
        },
        "measurements": measurements,
        "selection": select_chunk_size(
            measurements, memory_mb=WORKER_MEMORY_MB, timeout_seconds=WORKER_TIMEOUT_SECONDS
        ),
        "covered": [
            "one real execute_process_v2_rebind_task per chunk: read, decode, replay, "
            "mask, publish",
            "the cache build's one fused pass per source shard",
            "peak resident memory of the measuring process, one configuration per process",
        ],
        "not_covered": [
            "Modal containers, volume I/O, network and cold starts",
            "inter-chunk variance from heterogeneous production molecules",
            "the p99 at the largest chunk size is estimated from few samples; see "
            "quantile_sample_count",
            "an idle machine: this ran on a shared laptop, so wall time is partly a "
            "measurement of the machine. system_load_average_before/after record what "
            "else was running, and task_cpu_seconds_* is the contention-robust figure",
            "a production-realistic compression ratio: the corpus repeats one trace, so "
            "compressed_bytes is far smaller than a shard of distinct molecules would be",
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
