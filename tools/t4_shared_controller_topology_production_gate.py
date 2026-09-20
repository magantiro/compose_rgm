#!/usr/bin/env python3
"""Run the sealed zero-oracle topology production-path gate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compose_v4.experiments.t4_shared_controller_topology_production_gate import (
    ARTIFACT_ROOT_RELATIVE_PATH,
    load_contract,
    reduce_cell_shards,
    run_cell_shard,
    run_gate,
)


def _run_cell_job(
    repository_root: str, cell_key: str, output_path: str
) -> dict[str, object]:
    return run_cell_shard(
        repository_root=Path(repository_root),
        cell_key=cell_key,
        output_path=Path(output_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    output = args.output or (
        repository_root / ARTIFACT_ROOT_RELATIVE_PATH / "result.json"
    )
    shard_root = args.shard_root or output.parent / "cell_shards"
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1:
        result = run_gate(
            repository_root=repository_root,
            output_path=output,
            shard_root=shard_root,
        )
    else:
        contract, _identity = load_contract(repository_root)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_cell_job,
                    str(repository_root),
                    cell["cell_key"],
                    str(shard_root / f"{cell['cell_key']}.json"),
                ): cell["cell_key"]
                for cell in contract["cells"]
            }
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        result = reduce_cell_shards(
            repository_root=repository_root,
            shard_root=shard_root,
            output_path=output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "PASS_PRODUCTION_PATH_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
