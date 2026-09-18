"""Run one sealed route-prior versus uniform production-support probe arm."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from rdkit import rdBase

from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_route_guided_support import (
    fit_held_target_prior,
    load_decisions,
    run_arm,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _load_contract(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    actual = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise ValueError(f"contract hash mismatch for {path}: {actual}")
    return payload, actual


def _write_ledger(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--arm", choices=("uniform_same_pool", "route_prior"), required=True)
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract, contract_hash = _load_contract(args.contract)
    if args.cell not in contract["cells"]:
        parser.error(f"cell must be one of {sorted(contract['cells'])}")
    attempts = contract["attempts_by_cell"][args.cell]
    if args.attempts is not None:
        if not contract["allow_engineering_smoke_override"]:
            parser.error("the contract forbids an attempt override")
        attempts = args.attempts

    inputs = contract["inputs"]
    for relative, expected in inputs.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {relative}: {actual}")

    registry = unseal(ROOT / contract["source_registry"])
    unit = registry["cells"][args.cell]
    if unit["target"] != contract["cells"][args.cell]["target"]:
        raise ValueError(f"target mismatch for {args.cell}")

    rows = load_decisions(ROOT / contract["decision_corpus"])
    model, vocabulary, split_audit = fit_held_target_prior(rows, unit["target"])
    if not split_audit["held_target_absent_from_training"]:
        raise RuntimeError("held target entered route-prior training")

    last_reported = {"attempt": -1}

    def progress(attempt, total, feasible):
        stride = max(1, min(25, total // 10))
        if attempt == 0 or attempt + 1 == total or attempt - last_reported["attempt"] >= stride:
            last_reported["attempt"] = attempt
            print(
                json.dumps(
                    {
                        "cell": args.cell,
                        "arm": args.arm,
                        "attempt": attempt + 1,
                        "attempts": total,
                        "feasible_unique_so_far": feasible,
                    }
                ),
                flush=True,
            )

    result = run_arm(
        decode_state(unit["source_state"]),
        unit["original_seed"],
        unit["target"],
        model,
        vocabulary,
        arm=args.arm,
        attempts=attempts,
        seed=contract["seed_by_cell"][args.cell],
        delta=contract["delta"],
        depths=tuple(contract["depths"]),
        candidates_per_step=contract["candidates_per_step"],
        route_exploration=contract["route_exploration"],
        temperature=contract["temperature"],
        progress=progress,
    )
    ledger_path = args.output.with_suffix(".jsonl.gz")
    ledger_hash = _write_ledger(ledger_path, result.pop("ledger"))
    payload = {
        **result,
        "cell": args.cell,
        "target": unit["target"],
        "delta": contract["delta"],
        "seed": contract["seed_by_cell"][args.cell],
        "split_audit": split_audit,
        "contract": str(args.contract),
        "contract_payload_sha256": contract_hash,
        "decision_corpus_sha256": inputs[contract["decision_corpus"]],
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_hash,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "evidence_status": (
            "zero-oracle generated support; not docking utility and not autonomous T4 performance"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seal(args.output, payload)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("raw_factor_rates", "feasible_factor_rates")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
