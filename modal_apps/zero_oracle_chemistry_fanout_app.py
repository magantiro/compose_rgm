"""Modal: a reusable shard-map fan-out for ZERO-ORACLE CPU chemistry.

WHY
---
Several COMPOSE diagnostics are embarrassingly parallel, cost no oracle call and no
docking call, and are pure CPU -- the T4-v2 feasibility grid, the support-stage audits,
the realization sweeps.  Run serially on a laptop they take tens of minutes to hours, and
they contend with every other agent on the same machine, which has already corrupted one
measurement.  Modal bills core-seconds, so moving them costs the same compute and buys
back the wall clock and an uncontended timer.

SHAPE
-----
    plan     (local)        the workload emits a WORK LIST of independent units
    run_unit (N containers) one unit per call; the row is written to the volume the
                            moment it is computed, and streamed back to the launcher,
                            which also writes it locally -- a crash at the end of the run
                            loses nothing that has already finished
    reduce   (1 container)  read every unit row off the volume, hand them to the
                            workload's reducer, write the merged artifact + a sidecar
                            manifest

The unit is the shard.  Modal's ``.map`` distributes the calls across at most
``ZOC_MAX_CONTAINERS`` containers and reuses warm ones, so the container count is a dial
on wall clock, not on cost.

ISOLATION
---------
Own app name, own volume, own output namespace.  No other volume is mounted, read or
written, and nothing under ``src/compose_v4``, ``configs/`` or any pre-existing
``modal_apps`` module is imported or modified.  The ten pinned T4 campaigns and the PMO
campaign are untouched by this file.

ZERO ORACLE
-----------
Three independent guards: (1) the image carries NO docking binary and no TDC install --
it is the pinned chemistry image and nothing else; (2) ``block_network`` is on by default,
so a container cannot reach a receptor mirror or a TDC asset host even if some future
import tried; (3) ``assert_zero_oracle`` runs inside every unit and refuses to proceed if
an oracle module or docking binary has appeared.  Every unit row carries that evidence.

RUN
---
    export ZOC_MAX_CONTAINERS=40
    modal run --profile nitya modal_apps/zero_oracle_chemistry_fanout_app.py \
        --workload t4_v2_feasibility --budget 6000 --seed 1 \
        --out diagnostics/t4_v2_feasibility_modal.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
APP_NAME = "compose-zero-oracle-chemistry-fanout"
VOLUME_NAME = "compose-zero-oracle-fanout"
ARTIFACT_MOUNT = Path("/artifacts")
SCHEMA_VERSION = "zero_oracle_chemistry_fanout_v1"

#: Tuned per launch from the environment, because ``max_containers`` is decided when the
#: app module is imported and ``modal run`` imports it locally first.
MAX_CONTAINERS = int(os.environ.get("ZOC_MAX_CONTAINERS", "32"))
#: Modal's autoscaler ramps on backlog, and a map of SHORT units finishes before the ramp
#: does -- an 18-unit map at max_containers=18 was measured running on exactly ONE
#: container.  Pre-warming is therefore required to measure a container count rather than
#: an autoscaler transient; it is idle-billed, so it is set per launch, not by default.
MIN_CONTAINERS = int(os.environ.get("ZOC_MIN_CONTAINERS", "0"))
BUFFER_CONTAINERS = int(os.environ.get("ZOC_BUFFER_CONTAINERS", "0"))
#: Default ON.  Set ZOC_BLOCK_NETWORK=0 only if a workload legitimately needs the network,
#: which for a zero-oracle chemistry workload it never should.
BLOCK_NETWORK = os.environ.get("ZOC_BLOCK_NETWORK", "1") != "0"
UNIT_TIMEOUT_SECONDS = int(os.environ.get("ZOC_UNIT_TIMEOUT", str(60 * 60)))
#: Re-launching a crashed run skips units whose row is already on the volume.  Unit ids
#: are content-addressed over the work item, so a hit is the SAME unit, never a stale
#: one.  Deliberately OFF by default: a resumed unit returns in milliseconds, which would
#: silently turn any repeat launch -- including a throughput measurement -- into a cache
#: read reported as compute.
RESUME = os.environ.get("ZOC_RESUME", "0") == "1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modal_apps import zero_oracle_chemistry_workloads as workloads

_IGNORE = ("**/__pycache__/**", "**/*.pyc")


def _declared_inputs() -> tuple[str, ...]:
    """Every registered workload's extra input files, unioned and deduplicated.

    Explicit by design: ``diagnostics/`` is 875 MB and must never be copied wholesale.
    """

    seen: list[str] = []
    for definition in workloads.WORKLOADS.values():
        for relative in definition["inputs"]:
            if relative not in seen:
                seen.append(relative)
    return tuple(seen)


def _runtime_image() -> modal.Image:
    """The pinned production chemistry environment, and nothing else.

    Versions are the ones COMPOSE pins for every claim-bearing evaluation (python 3.11,
    torch 2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5).  Deliberately
    absent relative to the T4 scored image: qvina02, the receptor set, openbabel and TDC.
    Their absence is what makes the zero-oracle claim structural rather than a promise.
    """

    result = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.0",
            "numpy==1.26.4",
            "scipy==1.13.1",
            "networkx==3.3",
            "rdkit==2024.3.5",
        )
        .env(
            {
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(REMOTE_ROOT),
                        str(REMOTE_ROOT / "src"),
                        str(REMOTE_ROOT / "scripts"),
                    )
                ),
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                # Nested numeric threads over-subscribe a 1-core container and slow the
                # whole fan-out; one thread per container is the measured setting.
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True, ignore=_IGNORE)
        .add_local_dir(
            ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True, ignore=_IGNORE
        )
        # Copied to the SAME repository-relative position it occupies locally, so
        # ``ROOT = parents[1]`` and ``from modal_apps import ...`` resolve identically on
        # both sides.  That symmetry is what lets one function be the equivalence
        # reference for the other.
        .add_local_file(
            ROOT / "modal_apps/zero_oracle_chemistry_workloads.py",
            str(REMOTE_ROOT / "modal_apps/zero_oracle_chemistry_workloads.py"),
            copy=True,
        )
    )
    for relative in _declared_inputs():
        result = result.add_local_file(
            ROOT / relative, str(REMOTE_ROOT / relative), copy=True
        )
    return result


image = _runtime_image()
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def manifest_path_for(path: Path) -> Path:
    """``merged.json`` -> ``merged.manifest.json``.

    ``with_suffix``, never concatenation: ``str(path) + '.manifest.json'`` yields
    ``merged.json.manifest.json``, which every reader then fails to find.
    """

    return path.with_suffix(".manifest.json")


def source_commit(root: Path = ROOT) -> str:
    """The launch commit, or ``"unknown"``.

    The debian_slim image has no ``git`` binary, and a ``subprocess.run(['git', ...])``
    raising ``FileNotFoundError`` at the END of a container has destroyed a finished run
    in this repository before.  This is only ever called on the LAUNCHER, and it still
    catches, because the cost of guessing wrong is an entire run.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "unknown"


def run_namespace(workload_name: str, parameters: dict, commit: str) -> str:
    """Content-addressed output namespace; two different plans never collide."""

    identity = workloads.canonical_sha256(
        {
            "schema": SCHEMA_VERSION,
            "workload": workload_name,
            "parameters": parameters,
            "source_commit": commit,
        }
    )
    return f"{workload_name}/{identity[:20]}"


# ---- Remote functions ---------------------------------------------------------------


@app.function(
    image=image,
    volumes={str(ARTIFACT_MOUNT): volume},
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=UNIT_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    min_containers=MIN_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=60,
    block_network=BLOCK_NETWORK,
    retries=modal.Retries(max_retries=3),
)
def run_unit(item: dict, run_id: str = "", commit: str = "") -> dict:
    """Execute exactly one work unit and persist it BEFORE returning."""

    started = time.time()
    print(
        f"[unit:start] {item['unit_id']} kind={item['kind']} "
        f"cell={item.get('cell')} arm={item.get('arm') or item.get('component') or '-'}",
        flush=True,
    )
    destination = ARTIFACT_MOUNT / run_id / "units" / f"{item['unit_id']}.json"
    if RESUME and destination.exists():
        try:
            existing = json.loads(destination.read_text())
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and existing.get("unit_id") == item["unit_id"]:
            existing["resumed"] = True
            print(f"[unit:resume] {item['unit_id']} already on the volume", flush=True)
            return existing

    row = workloads.execute_item(item)
    row["run_id"] = run_id
    row["source_commit"] = commit
    row["item"] = item

    row["resumed"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(row, indent=1, sort_keys=True, default=str) + "\n"
    )
    volume.commit()
    print(
        f"[unit:done] {item['unit_id']} {row['wall_seconds']}s "
        f"(container {round(time.time() - started, 1)}s) -> {destination}",
        flush=True,
    )
    return row


@app.function(
    image=image,
    volumes={str(ARTIFACT_MOUNT): volume},
    cpu=(2.0, 2.0),
    memory=8192,
    timeout=60 * 60,
    block_network=BLOCK_NETWORK,
    retries=modal.Retries(max_retries=2),
)
def reduce_run(run_id: str, workload_name: str, plan: list[dict], meta: dict) -> dict:
    """Merge every persisted unit row into the workload's artifact."""

    print(f"[reduce] reading {run_id}", flush=True)
    volume.reload()
    unit_dir = ARTIFACT_MOUNT / run_id / "units"
    rows = [
        json.loads(path.read_text()) for path in sorted(unit_dir.glob("*.json"))
    ]
    print(f"[reduce] {len(rows)} unit rows for {len(plan)} planned units", flush=True)

    definition = workloads.workload(workload_name)
    merged = definition["reduce"](plan, rows)

    unit_seconds = sum(float(row.get("wall_seconds") or 0.0) for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workload": workload_name,
        "run_id": run_id,
        "planned_units": len(plan),
        "completed_units": len(rows),
        "unit_seconds_total": round(unit_seconds, 2),
        "unit_seconds_mean": round(unit_seconds / max(len(rows), 1), 2),
        "unit_seconds_max": round(
            max((float(row.get("wall_seconds") or 0.0) for row in rows), default=0.0), 2
        ),
        "distinct_containers": len({row.get("hostname") for row in rows}),
        "resumed_units": sum(1 for row in rows if row.get("resumed")),
        "oracle_calls": 0,
        "docking_calls": 0,
        "zero_oracle_violations": [
            row["unit_id"]
            for row in rows
            if row.get("zero_oracle_evidence", {}).get("forbidden_modules_imported")
            or row.get("zero_oracle_evidence", {}).get("forbidden_binaries_present")
        ],
        "merged_identity_sha256": workloads.canonical_sha256(merged),
        "meta": meta,
    }

    merged_path = ARTIFACT_MOUNT / run_id / "merged.json"
    merged_path.write_text(json.dumps(merged, indent=1, sort_keys=True, default=str) + "\n")
    manifest_path_for(merged_path).write_text(
        json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n"
    )
    volume.commit()
    print(
        f"[reduce] wrote {merged_path} and {manifest_path_for(merged_path)}", flush=True
    )
    return {"merged": merged, "manifest": manifest}


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=10 * 60,
    block_network=BLOCK_NETWORK,
)
def plan_units(workload_name: str, parameters: dict) -> list[dict]:
    """Build the work list INSIDE the pinned image.

    Planning is chemistry-free -- it reads a JSON audit and validates names -- but the
    planner lives in the same module as the executor and imports RDKit transitively.
    Running it remotely keeps the launcher thin (a laptop needs only the Modal client)
    and removes any chance that the plan is built under one chemistry version while the
    units run under another.
    """

    definition = workloads.workload(workload_name)
    items = definition["plan"](**parameters)
    print(f"[plan] {len(items)} units for {workload_name}", flush=True)
    return items


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=10 * 60,
    block_network=BLOCK_NETWORK,
)
def environment_report() -> dict:
    """What the container actually is.  Proves the pins and the absent oracle paths."""

    import networkx
    import numpy
    import rdkit
    import scipy

    report = {
        "python": sys.version.split()[0],
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
        "block_network": BLOCK_NETWORK,
    }
    try:
        import torch

        report["torch"] = torch.__version__
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        report["torch"] = f"absent: {error}"
    report["zero_oracle_evidence"] = workloads.assert_zero_oracle()
    return report


# ---- Launcher -----------------------------------------------------------------------


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@app.local_entrypoint()
def main(
    workload: str = "t4_v2_feasibility",
    cells: str = "",
    arms: str = "",
    budget: int = 6000,
    seed: int = 1,
    delta: float = 0.6,
    ablate: bool = True,
    attribution: bool = True,
    granularity: str = "arm",
    out: str = "",
    stream_dir: str = "",
    plan_only: bool = False,
    environment_only: bool = False,
) -> None:
    """Plan locally, fan out, stream rows back, reduce, write the artifact."""

    if environment_only:
        print(json.dumps(environment_report.remote(), indent=1, sort_keys=True))
        return

    commit = source_commit()
    workloads.workload(workload)  # fail fast on an unknown name, before any remote call
    parameters = {
        "cells": list(_parse_list(cells)),
        "arms": list(_parse_list(arms)),
        "budget": budget,
        "seed": seed,
        "delta": delta,
        "ablate": ablate,
        "attribution": attribution,
        "granularity": granularity,
    }
    print(f"[plan] workload={workload} commit={commit}", flush=True)
    items = plan_units.remote(workload, parameters)
    run_id = run_namespace(workload, parameters, commit)
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(
        f"[plan] {len(items)} units {kinds} -> run_id={run_id} "
        f"max_containers={MAX_CONTAINERS} min_containers={MIN_CONTAINERS} "
        f"block_network={BLOCK_NETWORK}",
        flush=True,
    )
    if plan_only:
        print(json.dumps(items, indent=1, sort_keys=True))
        return

    stream = Path(stream_dir) if stream_dir else None
    if stream is not None:
        stream.mkdir(parents=True, exist_ok=True)

    started = time.time()
    completed = 0
    unit_seconds = 0.0
    # ``order_outputs=False`` yields each row the instant it lands, which is what makes
    # the local mirror an as-they-complete write rather than a buffered one.
    for row in run_unit.map(
        items, kwargs={"run_id": run_id, "commit": commit}, order_outputs=False
    ):
        completed += 1
        unit_seconds += float(row.get("wall_seconds") or 0.0)
        if stream is not None:
            (stream / f"{row['unit_id']}.json").write_text(
                json.dumps(row, indent=1, sort_keys=True, default=str) + "\n"
            )
        print(
            f"[map] {completed}/{len(items)} {row['kind']} {row.get('cell')} "
            f"{row['wall_seconds']}s elapsed={round(time.time() - started, 1)}s",
            flush=True,
        )
    wall = time.time() - started
    print(
        f"[map] done: {completed} units, {round(unit_seconds, 1)} container-seconds, "
        f"{round(wall, 1)}s wall",
        flush=True,
    )

    meta = {
        "source_commit": commit,
        "parameters": parameters,
        "max_containers": MAX_CONTAINERS,
        "resume_enabled": RESUME,
        "min_containers": MIN_CONTAINERS,
        "buffer_containers": BUFFER_CONTAINERS,
        "block_network": BLOCK_NETWORK,
        "map_wall_seconds": round(wall, 2),
        "map_container_seconds": round(unit_seconds, 2),
    }
    result = reduce_run.remote(run_id, workload, items, meta)
    print(json.dumps(result["manifest"], indent=1, sort_keys=True), flush=True)

    if out:
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result["merged"], indent=1, sort_keys=True, default=str) + "\n"
        )
        manifest_path_for(destination).write_text(
            json.dumps(result["manifest"], indent=1, sort_keys=True, default=str) + "\n"
        )
        print(f"[out] {destination} + {manifest_path_for(destination)}", flush=True)
