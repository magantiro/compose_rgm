"""Run the deterministic, zero-oracle NoDistill joint STOP support gate."""

from __future__ import annotations

import argparse
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.nodistill_joint_stop import ARMS, JointStopConfig
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_nodistill_joint_stop_gate import (
    evaluate_published_locks,
    generate_case_lock,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "diagnostics/t4_nodistill_joint_stop_gate/attempt_1"


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _generate_twice(root_text: str, case: dict, arm: str, generation: dict):
    root = Path(root_text)
    config = JointStopConfig(**generation)
    first = generate_case_lock(root, case, arm, config)
    second = generate_case_lock(root, case, arm, config)
    if first != second:
        return first, False
    return first, True


def run(output: Path, *, workers: int) -> dict:
    contract = load_contract(ROOT)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("NoDistill gate uses one to four deterministic CPU shards")
    started = perf_counter()
    jobs = [(case, arm) for case in contract["cases"] for arm in ARMS]
    generated = {}
    determinism = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {
            pool.submit(
                _generate_twice,
                str(ROOT),
                case,
                arm,
                contract["generation"],
            ): (case, arm)
            for case, arm in jobs
        }
        for future in as_completed(futures):
            case, arm = futures[future]
            lock, stable = future.result()
            key = (case["case_id"], arm)
            generated[key] = lock
            determinism[key] = stable

    # Publication is the phase boundary.  The evaluator, including teacher
    # comparison, is not imported with any candidate into the worker runtime.
    lock_paths = {}
    lock_manifest = []
    for case, arm in jobs:
        key = (case["case_id"], arm)
        path = output / "locks" / f"{case['case_id']}__{arm}.json"
        publish_json(path, generated[key])
        lock_paths[key] = path
        lock_manifest.append(
            {
                "case_id": case["case_id"],
                "arm": arm,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "lock_id": generated[key]["lock_id"],
                "deterministic_rerun": determinism[key],
            }
        )
    manifest_path = output / "lock_manifest.json"
    publish_json(
        manifest_path,
        {
            "schema_version": "t4_nodistill_joint_stop_lock_manifest_v1",
            "contract_sha256": sha256_file(
                ROOT / "configs/t4_nodistill_joint_stop_gate_v1.json"
            ),
            "locks": lock_manifest,
            "all_locks_published_before_evaluation": True,
            "new_oracle_calls": 0,
        },
    )

    result = evaluate_published_locks(ROOT, contract, lock_paths, determinism)
    result["provenance"] = {
        "code_revision": _revision(ROOT),
        "contract_path": "configs/t4_nodistill_joint_stop_gate_v1.json",
        "contract_sha256": sha256_file(
            ROOT / "configs/t4_nodistill_joint_stop_gate_v1.json"
        ),
        "lock_manifest_path": str(manifest_path.relative_to(ROOT)),
        "lock_manifest_sha256": sha256_file(manifest_path),
        "input_sha256": contract["inputs_sha256"],
        "seed": contract["generation"]["seed"],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "hardware": "local CPU; process-level deterministic shards",
        "precision": "exact integer graph state and float64 RDKit properties",
        "workers": workers,
        "wall_seconds": perf_counter() - started,
    }
    publish_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    result = run(output, workers=args.workers)
    print(result["gate"]["decision"])
    print(output / "result.json")
    return 0 if result["gate"]["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
