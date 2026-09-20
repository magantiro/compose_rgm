"""Run durable route-free v4 support shards and the frozen reduction."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compose_v4.experiments.t4_nodistill_feasibility_headroom_gate_v4 import (
    ARTIFACT_ROOT_RELATIVE_PATH,
    allocated_plan_names,
    evaluate_locks,
    load_contract,
    lock_joint_stop,
    lock_plan,
    merge_cell_shards,
)


def _lock_one(job: tuple[str, str, str, str, str]) -> dict:
    root, kind, cell_key, unit, output = job
    if kind == "plan":
        return lock_plan(
            repository_root=Path(root),
            cell_key=cell_key,
            plan_name=unit,
            output_path=Path(output),
        )
    if kind == "joint_stop":
        return lock_joint_stop(
            repository_root=Path(root),
            cell_key=cell_key,
            output_path=Path(output),
        )
    raise ValueError(f"unknown v4 shard kind: {kind}")


def _cell_complete(root: Path, artifact_root: Path, cell_key: str) -> bool:
    return (
        all(
            (artifact_root / "plans" / cell_key / f"{name}.json.gz").exists()
            for name in allocated_plan_names(repository_root=root, cell_key=cell_key)
        )
        and (artifact_root / "joint_stop" / f"{cell_key}.json.gz").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-shards")
    lock.add_argument("--artifact-root", type=Path)
    lock.add_argument("--workers", type=int, default=8)
    lock.add_argument("--cell-key", action="append")
    merge = commands.add_parser("merge-cell")
    merge.add_argument("--artifact-root", type=Path)
    merge.add_argument("--cell-key", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--artifact-root", type=Path)
    evaluate.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_root = (
        root / ARTIFACT_ROOT_RELATIVE_PATH
        if arguments.artifact_root is None
        else arguments.artifact_root.resolve()
    )
    if arguments.command == "lock-shards":
        contract, _contract_identity = load_contract(root)
        workers = int(arguments.workers)
        if not 1 <= workers <= 12:
            raise ValueError("v4 workers must be within one and twelve")
        declared = [str(row["cell_key"]) for row in contract["cells"]]
        requested = declared if not arguments.cell_key else list(arguments.cell_key)
        if not requested or any(cell not in declared for cell in requested):
            raise ValueError("requested cell is outside the frozen v4 panel")
        jobs = []
        reused = 0
        # Keep motivating-cell priority while each cell is independently plan-sharded.
        for cell_key in requested:
            for plan_name in allocated_plan_names(
                repository_root=root, cell_key=cell_key
            ):
                output = artifact_root / "plans" / cell_key / f"{plan_name}.json.gz"
                if output.exists():
                    reused += 1
                else:
                    jobs.append(
                        (
                            str(root),
                            "plan",
                            cell_key,
                            plan_name,
                            str(output),
                        )
                    )
            joint = artifact_root / "joint_stop" / f"{cell_key}.json.gz"
            if joint.exists():
                reused += 1
            else:
                jobs.append(
                    (str(root), "joint_stop", cell_key, "joint_stop", str(joint))
                )
        completed = []
        merged = []
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_lock_one, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                completed.append(row)
                print(
                    json.dumps(
                        {
                            "heartbeat": f"{index}/{len(futures)}",
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                            **row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                cell_key = str(row["cell_key"])
                cell_lock = artifact_root / "locks" / f"{cell_key}.json.gz"
                if (
                    cell_key in requested
                    and not cell_lock.exists()
                    and _cell_complete(root, artifact_root, cell_key)
                ):
                    reduced = merge_cell_shards(
                        repository_root=root,
                        cell_key=cell_key,
                        artifact_root=artifact_root,
                        output_path=cell_lock,
                    )
                    merged.append(reduced)
                    print(
                        json.dumps(
                            {
                                "cell_reduced": reduced,
                                "elapsed_seconds": round(time.monotonic() - started, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        for cell_key in requested:
            cell_lock = artifact_root / "locks" / f"{cell_key}.json.gz"
            if not cell_lock.exists() and _cell_complete(root, artifact_root, cell_key):
                merged.append(
                    merge_cell_shards(
                        repository_root=root,
                        cell_key=cell_key,
                        artifact_root=artifact_root,
                        output_path=cell_lock,
                    )
                )
        result: object = {
            "created": len(completed),
            "reused": reused,
            "merged": merged,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
    elif arguments.command == "merge-cell":
        output = artifact_root / "locks" / f"{arguments.cell_key}.json.gz"
        result = merge_cell_shards(
            repository_root=root,
            cell_key=arguments.cell_key,
            artifact_root=artifact_root,
            output_path=output,
        )
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
