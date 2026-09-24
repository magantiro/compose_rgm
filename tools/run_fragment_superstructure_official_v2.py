"""Run and reduce the frozen repaired-superstructure official fragment row.

Each (drug, seed) is one resumable 100-attempt shard. This wrapper verifies a
self-hashed contract, the clean source revision and all material file hashes
before starting. Existing shards are validated, never overwritten. The full
three-seed result is published only after all thirty shards are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/fragment_superstructure_official_v2.json"
METRICS = ("validity", "uniqueness", "quality", "diversity", "distance")


def identity(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_contract(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed contract: {path}")
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"contract self-hash mismatch: {path}")
    if payload["schema"] != "compose_fragment_superstructure_official_v2":
        raise ValueError(f"unexpected contract schema: {path}")
    return payload, envelope["payload_sha256"]


def preflight(contract: dict) -> list[str]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(f"frozen source revision is not an ancestor of {revision}")
    output_prefix = contract["output_dir"].rstrip("/") + "/"
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True
    )
    unexpected = [line for line in status.splitlines() if not line[3:].startswith(output_prefix)]
    if unexpected:
        raise RuntimeError(f"dirty source tree outside output namespace: {unexpected[:10]}")
    for relative, expected in contract["material_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"material hash mismatch or missing file: {relative}")
    checkpoint = Path(contract["checkpoint"]["path"])
    if not checkpoint.is_file() or sha256(checkpoint) != contract["checkpoint"]["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch or missing file: {checkpoint}")
    from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only

    verified = verify_only()
    expected_evaluator = contract["official_evaluator_sha256"]
    if expected_evaluator != OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]:
        raise RuntimeError("contract evaluator identity is not the pinned official blob")
    if expected_evaluator not in verified.values():
        raise RuntimeError("pinned official evaluator not available")
    manifest = ROOT / contract["prompt_manifest"]
    drugs = sorted(
        prompt.drug_name
        for prompt in load_genmol_prompts(manifest)
        if prompt.task == FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    if len(drugs) != 10 or len(set(drugs)) != 10:
        raise RuntimeError("not exactly ten distinct official superstructure prompts")
    return drugs


def shard_path(output_dir: Path, drug: str, seed: int) -> Path:
    return output_dir / "shards" / f"superstructure_generation__{drug}__seed{seed}.json"


def read_shard(path: Path, drug: str, seed: int, contract: dict) -> dict:
    payload = json.loads(path.read_text())
    if payload["schema"] != "compose_fragment_official_suite_v2":
        raise RuntimeError(f"unexpected shard schema: {path}")
    if payload["protocol"]["samples_per_prompt"] != 100 or payload["protocol"]["seed_list"] != [
        seed
    ]:
        raise RuntimeError(f"shard protocol mismatch: {path}")
    if payload["sampler"]["config"] != contract["sampler_config"]:
        raise RuntimeError(f"shard sampler mismatch: {path}")
    control = payload["attachment_control"]["config"]
    if any(
        control.get(key) is not True
        for key in ("enabled", "condition_initial_locked_family", "hard_lock_effective_chemistry")
    ):
        raise RuntimeError(f"shard control mismatch: {path}")
    per_drug = payload["results"]["superstructure_generation"]["per_drug"]
    if set(per_drug) != {drug} or len(per_drug[drug]) != 1:
        raise RuntimeError(f"shard prompt mismatch: {path}")
    [row] = per_drug[drug]
    if row["seed"] != seed or row["attempts"] != 100 or len(row["attempt_records"]) != 100:
        raise RuntimeError(f"shard attempt census mismatch: {path}")
    if len(row["chemical_samples"]) != 100 or len(row["emitted_samples"]) != 100:
        raise RuntimeError(f"shard chemical/task population length mismatch: {path}")
    if sum(bool(sample) for sample in row["chemical_samples"]) != row["committed_endpoints"]:
        raise RuntimeError(f"chemical-output census mismatch: {path}")
    if row["committed_endpoints"] != row["committed_chemically_valid"]:
        raise RuntimeError(f"chemically invalid committed endpoint: {path}")
    if row["committed_fragment_preserving"] != row["committed_endpoints"]:
        raise RuntimeError(f"fragment loss on a committed endpoint: {path}")
    if row["emitted_nonempty"] != sum(bool(x) for x in row["emitted_samples"]):
        raise RuntimeError(f"task-fidelity census mismatch: {path}")
    if row["official"]["validity"] != float(row["committed_endpoints"]):
        raise RuntimeError(f"official chemical validity mismatch: {path}")
    return row


def run_shard(contract: dict, output_dir: Path, drug: str, seed: int) -> tuple[str, int, Path]:
    path = shard_path(output_dir, drug, seed)
    if path.exists():
        read_shard(path, drug, seed, contract)
        return drug, seed, path
    python = sys.executable
    if Path(python).resolve() != Path(contract["python_executable"]).resolve():
        raise RuntimeError(f"wrong Python interpreter: {python}")
    command = [
        python,
        "tools/run_fragment_constrained_suite.py",
        "--checkpoint",
        contract["checkpoint"]["path"],
        "--output",
        str(path),
        "--task",
        "superstructure_generation",
        "--drug",
        drug,
        "--seed-list",
        str(seed),
        "--seeds",
        "1",
        "--samples",
        "100",
        "--max-events",
        str(contract["sampler_config"]["max_events"]),
        "--operational-horizon",
        str(contract["sampler_config"]["operational_horizon"]),
        "--mark-attempts-per-event",
        str(contract["sampler_config"]["mark_attempts_per_event"]),
        "--attachment-control",
        "--condition-initial-locked-family",
        "--hard-lock-effective-chemistry",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src:tools:scripts",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        }
    )
    completed = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"shard failed {drug}/seed{seed}: exit {completed.returncode}; "
            f"stdout={completed.stdout[-2000:]}; stderr={completed.stderr[-2000:]}"
        )
    if not path.is_file():
        raise RuntimeError(f"successful shard wrote no artifact: {path}")
    read_shard(path, drug, seed, contract)
    return drug, seed, path


def reduce_shards(contract: dict, contract_hash: str, drugs: list[str]) -> dict:
    output_dir = ROOT / contract["output_dir"]
    rows = []
    hashes = {}
    for seed in contract["seeds"]:
        for drug in drugs:
            path = shard_path(output_dir, drug, seed)
            if not path.is_file():
                raise RuntimeError(f"missing official shard: {path}")
            row = read_shard(path, drug, seed, contract)
            rescored = official_prompt_metrics(row["chemical_samples"], expected_samples=100)
            for name in ("validity", "uniqueness", "quality", "diversity"):
                if abs(rescored[name] - row["official"][name]) > 1e-9:
                    raise RuntimeError(f"official chemical metric replay mismatch: {path}/{name}")
            hashes[str(path.relative_to(ROOT))] = sha256(path)
            rows.append(
                {
                    "drug": drug,
                    "seed": seed,
                    "attempts": row["attempts"],
                    "committed": row["committed_endpoints"],
                    "fragment_preserving": row["committed_fragment_preserving"],
                    "prompt_compliant": row["emitted_nonempty"],
                    "official": row["official"],
                    "official_task_filtered": row["official_task_filtered"],
                }
            )
    per_seed = []
    for seed in contract["seeds"]:
        cohort = [row for row in rows if row["seed"] == seed]
        per_seed.append(
            {
                "seed": seed,
                **{
                    name: sum(row["official"][name] for row in cohort) / len(cohort)
                    for name in METRICS
                },
            }
        )
    result = {
        "schema": "fragment_superstructure_official_v2_result_v1",
        "contract_payload_sha256": contract_hash,
        "source_revision": contract["source_revision"],
        "execution_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": dict(sorted(hashes.items())),
        "attempts": sum(row["attempts"] for row in rows),
        "committed": sum(row["committed"] for row in rows),
        "fragment_preserving": sum(row["fragment_preserving"] for row in rows),
        "prompt_compliant": sum(row["prompt_compliant"] for row in rows),
        "per_seed": per_seed,
        "official_mean": {
            name: sum(row[name] for row in per_seed) / len(per_seed) for name in METRICS
        },
        "official_std": {
            name: (
                sum(
                    (row[name] - sum(s[name] for s in per_seed) / len(per_seed)) ** 2
                    for row in per_seed
                )
                / len(per_seed)
            )
            ** 0.5
            for name in METRICS
        },
        "rows": rows,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    contract, contract_hash = load_contract(args.contract)
    if args.workers < 1 or args.workers > contract["max_workers"]:
        raise ValueError("worker count outside frozen contract")
    drugs = preflight(contract)
    output_dir = ROOT / contract["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(drug, seed) for seed in contract["seeds"] for drug in drugs]
    print(
        f"contract {contract_hash} revision {contract['source_revision']} shards {len(tasks)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_shard, contract, output_dir, drug, seed): (drug, seed)
            for drug, seed in tasks
        }
        for future in as_completed(futures):
            drug, seed, path = future.result()
            print(f"complete {drug} seed{seed}: {path}", flush=True)
    result = reduce_shards(contract, contract_hash, drugs)
    print(json.dumps(result["official_mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
