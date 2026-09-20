"""Generate or evaluate the zero-oracle retained-interface v3 support gate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compose_v4.experiments.t4_nodistill_retained_interface_gate_v3 import (
    ARTIFACT_ROOT_RELATIVE_PATH,
    allocated_particle_names,
    evaluate_locks,
    load_contract,
    lock_cell,
    lock_particle,
    merge_cell_particles,
)


def _lock_one(arguments: tuple[str, str, str]) -> dict:
    root, cell_key, output = arguments
    return lock_cell(
        repository_root=Path(root),
        cell_key=cell_key,
        output_path=Path(output),
    )


def _lock_particle_one(arguments: tuple[str, str, str, str]) -> dict:
    root, cell_key, particle_name, output = arguments
    return lock_particle(
        repository_root=Path(root),
        cell_key=cell_key,
        particle_name=particle_name,
        output_path=Path(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-cell")
    lock.add_argument("--cell-key", required=True)
    lock.add_argument("--output", type=Path)
    panel = commands.add_parser("lock-panel")
    panel.add_argument("--artifact-root", type=Path)
    panel.add_argument("--workers", type=int, default=1)
    particles = commands.add_parser("lock-particles")
    particles.add_argument("--artifact-root", type=Path)
    particles.add_argument("--workers", type=int, default=1)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--artifact-root", type=Path)
    evaluate.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_root = (
        root / ARTIFACT_ROOT_RELATIVE_PATH
        if getattr(arguments, "artifact_root", None) is None
        else arguments.artifact_root.resolve()
    )
    if arguments.command == "lock-cell":
        output = (
            artifact_root / "locks" / f"{arguments.cell_key}.json.gz"
            if arguments.output is None
            else arguments.output.resolve()
        )
        result: object = lock_cell(
            repository_root=root,
            cell_key=arguments.cell_key,
            output_path=output,
        )
    elif arguments.command == "lock-panel":
        contract, _identity = load_contract(root)
        workers = int(arguments.workers)
        if not 1 <= workers <= len(contract["cells"]):
            raise ValueError("workers must be within one and the five-cell panel size")
        jobs = [
            (
                str(root),
                str(row["cell_key"]),
                str(artifact_root / "locks" / f"{row['cell_key']}.json.gz"),
            )
            for row in contract["cells"]
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            result = sorted(
                executor.map(_lock_one, jobs), key=lambda row: row["cell_key"]
            )
    elif arguments.command == "lock-particles":
        contract, _identity = load_contract(root)
        workers = int(arguments.workers)
        if not 1 <= workers <= 12:
            raise ValueError("particle workers must be within one and twelve")
        missing_cells = [
            str(row["cell_key"])
            for row in contract["cells"]
            if not (artifact_root / "locks" / f"{row['cell_key']}.json.gz").exists()
        ]
        jobs_by_cell = []
        for cell_key in missing_cells:
            cell_jobs = []
            for particle_name in allocated_particle_names(
                repository_root=root, cell_key=cell_key
            ):
                output = (
                    artifact_root / "particles" / cell_key / f"{particle_name}.json.gz"
                )
                if not output.exists():
                    cell_jobs.append((str(root), cell_key, particle_name, str(output)))
            jobs_by_cell.append(cell_jobs)
        jobs = [
            cell_jobs[position]
            for position in range(max(map(len, jobs_by_cell), default=0))
            for cell_jobs in jobs_by_cell
            if position < len(cell_jobs)
        ]
        completed = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_lock_particle_one, job): job for job in jobs}
            total = len(futures)
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                completed.append(row)
                print(
                    json.dumps(
                        {
                            "heartbeat": f"{index}/{total}",
                            "cell_key": row["cell_key"],
                            "particle_name": row["particle_name"],
                            "exact_candidates": row["exact_candidates"],
                            "raw_compile_successes": row["raw_compile_successes"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        merged = []
        for cell_key in missing_cells:
            merged.append(
                merge_cell_particles(
                    repository_root=root,
                    cell_key=cell_key,
                    artifact_root=artifact_root,
                    output_path=artifact_root / "locks" / f"{cell_key}.json.gz",
                )
            )
        result = {
            "particle_locks_created": len(completed),
            "particle_locks_reused": sum(
                len(allocated_particle_names(repository_root=root, cell_key=cell_key))
                for cell_key in missing_cells
            )
            - len(completed),
            "merged_cells": sorted(merged, key=lambda row: row["cell_key"]),
        }
    else:
        output = (
            artifact_root / "result.json"
            if arguments.output is None
            else arguments.output.resolve()
        )
        result = evaluate_locks(
            repository_root=root,
            artifact_root=artifact_root,
            output_path=output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
