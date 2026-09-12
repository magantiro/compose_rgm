"""Read-only progress and compact result collection for the approved feedback run."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--details", action="store_true", help="also collect saved proposal and value ledgers"
    )
    args = parser.parse_args()
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    profiles = {
        "t4_macro_feedback_spawn_v1": ("t4_macro_feedback", 10, 8),
        "t4_macro_lookahead_spawn_v1": ("t4_macro_lookahead", 4, 16),
        "t4_winner_route_docking_spawn_v1": ("t4_winner_route_docking", 0, 0),
    }
    expected_limit = 6 if receipt["schema_version"] == "t4_winner_route_docking_spawn_v1" else 40
    if receipt["schema_version"] not in profiles or receipt["oracle_call_limit"] != expected_limit:
        raise ValueError("expected a recognized bounded diagnostic receipt")
    kind, round_count, worker_count = profiles[receipt["schema_version"]]
    prefix = receipt["volume_path"].lstrip("/")
    run_id = receipt["task"]["run_id"]
    if prefix != kind + "/" + run_id:
        raise ValueError("feedback receipt namespace mismatch")
    volume = modal.Volume.from_name(receipt["volume"])

    def read(name):
        try:
            return b"".join(volume.read_file(prefix + "/" + name))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None

    names = ["result.json", "failure.json", "progress.json", "heartbeat.json"]
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = dict(zip(names, executor.map(read, names), strict=True))
        reports = {k: json.loads(v) for k, v in values.items() if v is not None}
        for key, report in reports.items():
            if key == "result.json":
                if kind == "t4_winner_route_docking":
                    print(
                        json.dumps(
                            {
                                "status": report["status"],
                                "new_calls": report["new_oracle_attempts"],
                                "elapsed_seconds": report["elapsed_seconds"],
                                "docked": [
                                    {
                                        k: r[k]
                                        for k in (
                                            "role",
                                            "ds",
                                            "qed",
                                            "sa",
                                            "sim",
                                            "smiles",
                                            "docking_seconds",
                                        )
                                    }
                                    for r in report["docked"]
                                ],
                            },
                            indent=2,
                        )
                    )
                    continue
                print(
                    json.dumps(
                        {
                            "status": report["status"],
                            "new_calls": report["new_oracle_attempts"],
                            "best_score": report["best"]["ds"],
                            "smiles": report["best"]["smiles"],
                            "rounds": [
                                {
                                    k: r[k]
                                    for k in (
                                        "round",
                                        "options",
                                        "completed",
                                        "max_option_depth",
                                        "best_score",
                                        "new_oracle_attempts",
                                        "proposal_seconds",
                                    )
                                }
                                for r in report["rounds"]
                            ],
                        },
                        indent=2,
                    )
                )
            else:
                print(key, json.dumps(report, sort_keys=True))
        if "result.json" not in reports and "failure.json" not in reports:
            number = max((r.get("round", 1) for r in reports.values()), default=1) - 1
            names = [
                f"rounds/{number:02}/workers/{i:02}/{run_id}/heartbeat.json"
                for i in range(worker_count)
            ]
            for i, value in enumerate(executor.map(read, names)):
                if value is not None:
                    worker = json.loads(value)
                    print(
                        "worker",
                        i,
                        json.dumps(
                            {
                                k: worker[k]
                                for k in (
                                    "phase",
                                    "primitive_depth",
                                    "branch",
                                    "elapsed_seconds",
                                    "updated_at_utc",
                                )
                                if k in worker
                            }
                        ),
                    )
        if args.collect:
            destination = ROOT / "diagnostics" / kind / run_id

            def retain(name, data):
                target = destination / name
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError(f"immutable collected artifact changed: {target}")
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(target)

            retain("spawn.json", receipt_bytes)
            names = [
                "launch.json",
                "runtime_gate.json",
                "result.json",
                "failure.json",
                "initial.json",
            ]
            names += [f"rounds/{i:02}/after.json" for i in range(round_count)]
            if kind == "t4_winner_route_docking":
                names += ["candidate_lock.json", "docking_started.json"]
                names += [
                    f"rows/{i:02}/{name}.json" for i in range(6) for name in ("started", "result")
                ]
                if args.details:
                    names += [
                        f"poses/{i:02}/{name}"
                        for i in range(6)
                        for name in ("manifest.json", "l.mol", "l.pdbqt", "o.pdbqt")
                    ]
            if args.details:
                names += [
                    f"rounds/{i:02}/{name}.json"
                    for i in range(round_count)
                    for name in ("generated", "prior/value", "posterior/value")
                ]
            hashes = {}
            for name, data in zip(names, executor.map(read, names), strict=True):
                if data is not None:
                    retain(name, data)
                    hashes[name] = hashlib.sha256(data).hexdigest()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            retain(
                f"collections/{stamp}.json",
                json.dumps(
                    {
                        "prefix": prefix,
                        "volume": receipt["volume"],
                        "sha256": hashes,
                        "collected_at_utc": stamp,
                    },
                    sort_keys=True,
                    indent=2,
                ).encode(),
            )


if __name__ == "__main__":
    main()
