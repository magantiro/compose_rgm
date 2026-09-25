"""Run the predeclared uniform-native-mark superstructure comparison arm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_fragment_superstructure_official_v2 import (
    ROOT,
    identity,
    preflight,
    read_shard,
    reduce_shards,
    sha256,
    shard_path,
)

DEFAULT_CONTRACT = ROOT / "configs/fragment_superstructure_reference_ablation_v1.json"


def load_contract(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    digest = envelope["payload_sha256"]
    if identity(payload) != digest:
        raise ValueError(f"ablation contract self-hash mismatch: {path}")
    if (
        payload["schema"] != "compose_fragment_superstructure_reference_ablation_v1"
        or payload["mark_law"] != "uniform_native"
    ):
        raise ValueError(f"unexpected ablation contract: {path}")
    return payload, digest


def checked_shard(path: Path, drug: str, seed: int, contract: dict) -> dict:
    raw = json.loads(path.read_text())
    if raw.get("mark_law") != "uniform_native":
        raise ValueError(f"not a uniform-native-mark shard: {path}")
    return read_shard(path, drug, seed, contract)


def run_shard(contract: dict, output_dir: Path, drug: str, seed: int) -> Path:
    path = shard_path(output_dir, drug, seed)
    if path.exists():
        checked_shard(path, drug, seed, contract)
        return path
    if shutil.disk_usage(output_dir).free < contract["minimum_free_bytes"]:
        raise RuntimeError("uniform superstructure run reached its frozen disk-free floor")
    if Path(sys.executable).resolve() != Path(contract["python_executable"]).resolve():
        raise ValueError("wrong frozen superstructure Python interpreter")
    command = [
        sys.executable,
        "tools/run_fragment_constrained_suite.py",
        "--checkpoint", contract["checkpoint"]["path"],
        "--output", str(path),
        "--task", "superstructure_generation",
        "--drug", drug,
        "--seed-list", str(seed),
        "--seeds", "1",
        "--samples", "100",
        "--max-events", str(contract["sampler_config"]["max_events"]),
        "--operational-horizon", str(contract["sampler_config"]["operational_horizon"]),
        "--mark-attempts-per-event", str(contract["sampler_config"]["mark_attempts_per_event"]),
        "--attachment-control",
        "--condition-initial-locked-family",
        "--hard-lock-effective-chemistry",
        "--mark-law", "uniform_native",
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH="src:tools:scripts", PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
        KMP_DUPLICATE_LIB_OK="TRUE"
    )
    completed = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"uniform shard failed {drug}/seed{seed}: {completed.returncode}; "
            f"stdout={completed.stdout[-1000:]}; stderr={completed.stderr[-1000:]}"
        )
    if not path.is_file():
        raise RuntimeError(f"uniform shard produced no output: {path}")
    checked_shard(path, drug, seed, contract)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    contract, contract_hash = load_contract(args.contract)
    if args.workers < 1 or args.workers > contract["max_workers"]:
        raise ValueError("worker count outside frozen contract")
    learned_path = Path(contract["learned_arm_result_path"])
    if not learned_path.is_file() or sha256(learned_path) != contract["learned_arm_result_sha256"]:
        raise RuntimeError("completed learned superstructure arm is missing or changed")
    learned = json.loads(learned_path.read_text())
    if (
        learned.get("contract_payload_sha256") != contract["learned_arm_contract_payload_sha256"]
        or learned.get("attempts") != 3000
        or learned.get("committed") != 3000
        or learned.get("prompt_compliant") != 3000
    ):
        raise RuntimeError("learned arm is not the completed matched 3,000-attempt result")
    drugs = preflight(contract)
    output_dir = ROOT / contract["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(drug, seed) for seed in contract["seeds"] for drug in drugs]
    print(f"uniform-native contract {contract_hash}, {len(tasks)} shards", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_shard, contract, output_dir, drug, seed): (drug, seed)
            for drug, seed in tasks
        }
        for future in as_completed(futures):
            drug, seed = futures[future]
            path = future.result()
            print(f"complete {drug} seed{seed}: {path}", flush=True)
    for drug, seed in tasks:
        checked_shard(shard_path(output_dir, drug, seed), drug, seed, contract)
    result = reduce_shards(contract, contract_hash, drugs)
    print(json.dumps(result["official_mean"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
