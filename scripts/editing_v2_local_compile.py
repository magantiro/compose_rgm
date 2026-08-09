"""Compile V2 chunks locally, resumably, one published slice at a time.

WHY LOCAL
---------
Measured, not assumed: Modal runs this at ~7.5 s/entry on one vCPU while a local
core does ~3.9 s/entry, so Modal is roughly half as efficient per core and the
two high-value real lanes would cost about $9.19 there against a $2 ceiling.
Modal's only advantage is wall-clock parallelism (~1.3 h against ~7.6 h), which
is not worth paying a 2x per-core penalty for when the GPU budget is better
spent on training.

WHY RESUMABLE, PER SLICE
------------------------
Publication is immutable and write-if-absent, and slices are addressed by
``<task>/<offset>-<count>``, so a slice that finished is a slice that is safe.
A crash, a laptop sleep or a deliberate stop costs at most the slices currently
in flight -- never the hours already spent. Earlier in this project a run was
killed mid-flight and destroyed ~96 in-flight slices because the unit of
durability was the whole job; here it is one slice.

WORKER COUNT
------------
Six is the measured optimum on a 12-core machine: 589% CPU and 0 swapouts at 6,
against 536% and 1,409 swapouts at 8, and outright collapse at 10 (throughput
fell from 5,507 entries/h to 250). Each worker holds its own chemistry cache, so
the binding constraint is memory, not cores. More workers is slower.

The expensive setup -- opening the authenticated source and building the scratch
runtime, ~107 s -- is paid ONCE per worker in the pool initializer, not once per
slice, which is the difference between 139 setups and 6.
"""

from __future__ import annotations

import argparse
import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

#: Refuse to start below this much free RAM. Each worker carries torch, rdkit and
#: its own chemistry cache; starting into a swapping machine does not run slowly,
#: it thrashes and produces nothing. Measured: a run with 8.1 GB free and 7.6 GB
#: already swapped published ZERO slices in two hours at load 20.
MIN_FREE_GB_PER_WORKER = 1.2
MIN_FREE_GB_ABSOLUTE = 4.0

ROOT = Path("/private/tmp/compose-t1-collated-cache")
for _p in (ROOT / "src", ROOT / "tests", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_W: dict = {}


def _init(active8_root: str, gate_zero: str, artifact_root: str,
          materialized_state: str) -> None:
    import torch

    torch.set_num_threads(1)
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )

    source = open_process_v2_t1_source(
        Path(active8_root),
        gate_zero_decision_path=Path(gate_zero),
        artifact_root=Path(artifact_root),
        repo_root=ROOT,
    )
    state = load_materialized_scorer_state(Path(materialized_state))
    runtime, _binding, _receipt = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    _W["source"] = source
    _W["model"] = runtime.model


def _published(output_root: Path, task: str, offset: int, count: int) -> bool:
    """A slice is done when the receipt for THIS EXACT offset-count exists.

    Matching the offset prefix alone is wrong whenever two runs use different
    ``--slice-size``: a 50-entry slice published at offset 0 would satisfy a
    planned 250-entry slice at offset 0, silently dropping 200 entries with no
    error anywhere. Resuming a partly-compiled task at a different slice size
    is a normal thing to do, so the check has to be exact.

    Note this makes size changes SAFE, not free: ranges compiled under the old
    size are simply recompiled under the new one, and overlapping publications
    would double-count. Finish a task at one size, or clear it before changing.
    """

    receipt = (Path(output_root) / "chunks" / task
               / f"{offset:09d}-{count:09d}" / "RECEIPT.json")
    return receipt.exists()


def _compile_slice(job: tuple) -> dict:
    task, offset, limit, role, output_root, batch_size = job
    from compose_v4.experiments.editing_v2_process_v2_chunk_compile import (
        compile_process_v2_chunk_shard,
        publish_process_v2_chunk_shard,
    )

    if _published(Path(output_root), task, offset, limit):
        return {"task": task, "offset": offset, "skipped": True, "entries": 0}
    began = time.monotonic()
    try:
        result = compile_process_v2_chunk_shard(
            _W["source"], _W["model"],
            task_identity_sha256=task, partition_role=role,
            offset=offset, limit=limit, compile_batch_size=batch_size,
        )
        publish_process_v2_chunk_shard(result, output_root=Path(output_root))
        return {"task": task, "offset": offset, "skipped": False,
                "entries": len(result["entries"]),
                "seconds": time.monotonic() - began}
    except Exception as error:  # noqa: BLE001 - one bad slice must not kill the run
        return {"task": task, "offset": offset, "failed": f"{type(error).__name__}: {error}"[:200],
                "entries": 0, "seconds": time.monotonic() - began}


def _free_gb() -> float:
    """Free + inactive pages, which is what a new process can actually claim."""

    import subprocess

    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return float("inf")
    page = 4096
    free = inactive = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif line.startswith("Pages free:"):
            free = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            inactive = int(line.split(":")[1].strip().rstrip("."))
    return (free + inactive) * page / 1024 ** 3


def _own_process_group() -> None:
    """Put workers in our own group so killing the parent kills the children.

    `multiprocessing` spawn workers do NOT carry the script name in their command
    line -- they run `from multiprocessing.spawn import spawn_main` -- so
    `pkill -f <script>` matches only the parent and silently orphans every
    worker. Twelve such orphans once spun for two hours at load 20 while the
    parent looked dead. Killing the group is the only reliable teardown.
    """

    try:
        os.setpgrp()
    except OSError:
        return
    pgid = os.getpgrp()

    def _teardown(*_args):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_teardown)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, lambda *_a: (_teardown(), os._exit(143)))
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--gate-zero", required=True)
    ap.add_argument("--manifest", default="diagnostics/editing_v2_v2_selection_manifest.json")
    ap.add_argument("--lanes", default="operator_aware_real_endpoint,linker_positional_topology_analogue")
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--materialized-state", required=True,
        help="Directory from materialize_process_v2_predecessor_state_app. Required "
             "locally: this host cannot re-derive the frozen scorer state from its "
             "seed (PyTorch CPU init is only reproducible on the platform that "
             "produced the constant), so the authenticated bytes are loaded instead.")
    ap.add_argument("--partition-role", default="train")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slice-size", type=int, default=250)
    ap.add_argument("--compile-batch-size", type=int, default=25)
    ap.add_argument("--max-slices", type=int, default=0,
                    help="0 = all; a small value smoke-tests that slices actually publish")
    ap.add_argument("--plan-only", action="store_true",
                    help="Print the slice plan and exit. Knowing the entry count and "
                         "projected wall clock BEFORE spawning is the difference between "
                         "choosing a worker count and discovering it hours in.")
    args = ap.parse_args()

    import gzip

    # FAIL CLOSED before spawning anything. The scratch runtime would raise on a
    # drifted catalog anyway, but inside six workers, 107 s in, with the real
    # cause buried under multiprocessing tracebacks. A divergent chemistry
    # kernel must stop the run here, and must never be neutralised: a local venv
    # on rdkit 2026.03.4 reconstructs 82fd910c instead of 639ff607, and a
    # 50-state parity gate showed 2 states whose canonical successor keys then
    # differ. Compiling under that would key half the corpus differently.
    import compose_v4.experiments.editing_gate_zero_runtime as _gz
    _expected = _gz.PRODUCTION_RINGCORE_CATALOG_FINGERPRINT
    try:
        _gz.build_production_ringcore_catalog(max_atoms=40)
    except _gz.EditingGateZeroRuntimeError as error:
        raise SystemExit(
            f"REFUSING TO COMPILE: catalog fingerprint drift.\n  {error}\n"
            "  Reproduce the pinned production environment (python 3.11, torch 2.4.0, "
            "numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5).\n"
            "  Do NOT call neutralize_catalog_drift() -- it overwrites the expected value."
        ) from error
    print(f"catalog fingerprint verified: {_expected}", flush=True)

    # Authenticate the scorer bytes here for the same reason: inside the pool this
    # surfaces as six identical tracebacks 107 s in, and a Pool RESPAWNS workers
    # that die in the initializer, so a failure looks like a busy run that never
    # publishes rather than like an error. That cost two hours once.
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        load_materialized_scorer_state,
    )
    _state = load_materialized_scorer_state(Path(args.materialized_state))
    print("materialized scorer authenticated: "
          f"{_state['receipt']['base_state_sha256'][:16]} "
          f"(built on {_state['receipt']['software']['system']}/"
          f"{_state['receipt']['software']['machine']})", flush=True)
    del _state

    free = _free_gb()
    needed = max(MIN_FREE_GB_ABSOLUTE, MIN_FREE_GB_PER_WORKER * args.workers)
    print(f"free memory {free:.1f} GB, need {needed:.1f} GB for {args.workers} workers", flush=True)
    if free < needed and not args.plan_only:
        raise SystemExit(
            f"REFUSING TO START: {free:.1f} GB free, {needed:.1f} GB needed for "
            f"{args.workers} workers.\n  Free memory or lower --workers. Starting into a "
            "swapping machine thrashes and publishes nothing."
        )
    _own_process_group()

    manifest = json.loads(Path(args.manifest).read_text())
    lanes = [lane for lane in args.lanes.split(",") if lane]
    a8 = Path(args.active8_root)

    jobs = []
    for lane in lanes:
        for task in sorted(manifest["selection"].get(lane, {})):
            stream = a8 / "tasks" / task / "transitions.jsonl.gz"
            if not stream.exists():
                continue
            total = sum(1 for _ in gzip.open(stream, "rt"))
            for offset in range(0, total, args.slice_size):
                limit = min(args.slice_size, total - offset)
                jobs.append((task, offset, limit, args.partition_role,
                             args.output_root, args.compile_batch_size))

    if args.max_slices > 0:
        # Bounded smoke test: prove a slice PUBLISHES before committing hours.
        jobs = [j for j in jobs
                if not _published(Path(args.output_root), j[0], j[1], j[2])][: args.max_slices]

    done_already = sum(1 for j in jobs
                       if _published(Path(args.output_root), j[0], j[1], j[2]))
    planned = sum(j[2] for j in jobs)
    print(f"lanes {lanes}", flush=True)
    print(f"slices {len(jobs)} ({done_already} already published), "
          f"planned entries {planned:,}, workers {args.workers}", flush=True)

    if args.plan_only:
        # 614 entries/h/worker measured on this host at slice-size 50, steady state
        # (excludes the ~107 s per-worker setup, which is paid once).
        remaining = planned - sum(j[2] for j in jobs
                                  if _published(Path(args.output_root), j[0], j[1], j[2]))
        for count in (2, 4, 6):
            print(f"  {count} workers -> {remaining / (614 * count):.1f}h")
        return 0

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    entries = failures = skipped = 0
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_init,
                      initargs=(args.active8_root, args.gate_zero, args.artifact_root,
                                args.materialized_state)) as pool:
        for n, row in enumerate(pool.imap_unordered(_compile_slice, jobs), 1):
            if row.get("skipped"):
                skipped += 1
            elif row.get("failed"):
                failures += 1
                print(f"  FAILED {row['task'][:12]}@{row['offset']}: {row['failed']}", flush=True)
            else:
                entries += row["entries"]
            if n % 10 == 0 or n == len(jobs):
                el = time.monotonic() - began
                rate = entries / max(el, 1e-9)
                left = max(planned - entries, 0)
                print(f"  [{n}/{len(jobs)}] entries {entries:,}  skipped {skipped}  "
                      f"failed {failures}  {rate * 3600:,.0f}/h  "
                      f"eta {left / max(rate, 1e-9) / 3600:.1f}h", flush=True)

    print(f"\ncompiled {entries:,} entries, skipped {skipped}, failed {failures}, "
          f"{(time.monotonic() - began) / 3600:.2f}h")
    print(f"output: {args.output_root}")
    return 1 if failures and entries == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
