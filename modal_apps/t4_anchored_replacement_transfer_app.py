"""Matched zero-oracle anchored-replacement transfer on JAK2 seeds 0 and 2."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from time import perf_counter

import modal

from modal_apps.genmol_t4_opt_app import REMOTE_ROOT, ROOT
from modal_apps.genmol_t4_opt_app import image as base_image

APP_NAME = "compose-t4-anchored-replacement-transfer"
CONTRACT = "configs/t4_anchored_replacement_transfer_v1.json"

image = base_image.add_local_dir(
    ROOT / "src",
    str(REMOTE_ROOT / "src"),
    copy=True,
    ignore=("**/__pycache__/**", "**/*.pyc"),
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=300,
    max_containers=64,
    retries=0,
    scaledown_window=30,
)
def worker(task):
    import numpy as np

    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

    started = perf_counter()
    try:
        rows = expand(
            task["root_smiles"],
            0.0,
            Fiber(task["root_smiles"], task["delta"], support=task["support"]),
            np.random.default_rng(task["seed"]),
            draws=task["draws_per_attempt"],
            horizon=task["horizon"],
            proposal_lane=task["lane"],
        )
        error = None
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as exc:
        rows = []
        error = f"{type(exc).__name__}: {exc}"
    return {
        "cell": task["cell"],
        "lane": task["lane"],
        "seed": task["seed"],
        "candidates": rows,
        "elapsed_seconds": perf_counter() - started,
        "error": error,
        "new_oracle_calls": 0,
    }


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


def _contract() -> tuple[dict, str]:
    from compose_v4.experiments.continuation_profile import canonical_bytes

    envelope = json.loads((ROOT / CONTRACT).read_text())
    payload = envelope["payload"]
    actual = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise ValueError(f"contract hash mismatch: {actual}")
    return payload, actual


@app.local_entrypoint()
def main(
    output: str = "diagnostics/t4_anchored_replacement_transfer_v1/result.json",
) -> None:
    from rdkit import rdBase

    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_anchored_replacement_transfer import (
        SCHEMA_VERSION,
        compare_transfer_cells,
    )
    from compose_v4.experiments.t4_fiber_lane_support import summarize_attempts
    from compose_v4.experiments.t4_matched_pilot import _stamp, seal

    contract, contract_hash = _contract()
    for relative, expected in contract["inputs"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {relative}: {actual}")

    attempt_count = int(contract["attempts_per_lane_per_cell"])
    tasks = []
    for cell_index, (cell, cell_spec) in enumerate(sorted(contract["cells"].items())):
        for lane_index, lane in enumerate(contract["lanes"]):
            for attempt in range(attempt_count):
                tasks.append(
                    {
                        "cell": cell,
                        "lane": lane,
                        "seed": (
                            int(contract["seed"])
                            + cell_index * 10_000_000
                            + lane_index * 1_000_000
                            + attempt
                        ),
                        "root_smiles": cell_spec["root_smiles"],
                        "delta": contract["delta"],
                        "support": contract["support"],
                        "horizon": contract["horizon"],
                        "draws_per_attempt": int(contract["draws_per_attempt"]),
                    }
                )
    results = list(worker.map(tasks, order_outputs=False))
    summaries = {
        cell: {
            lane: summarize_attempts(
                (
                    row
                    for row in results
                    if row["cell"] == cell and row["lane"] == lane
                ),
                expected_attempts=attempt_count,
            )
            for lane in contract["lanes"]
        }
        for cell in contract["cells"]
    }
    comparison = compare_transfer_cells(
        summaries, minimum_panel_yield=int(contract["minimum_panel_yield"])
    )

    destination = ROOT / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    ledger = destination.with_suffix(".jsonl.gz")
    with gzip.open(ledger, "wt") as handle:
        for row in sorted(results, key=lambda item: (item["cell"], item["lane"], item["seed"])):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "contract_payload_sha256": contract_hash,
        "code_revision": _revision(),
        "configuration": contract,
        "summaries": summaries,
        "comparison": comparison,
        "ledger": str(ledger.relative_to(ROOT)),
        "ledger_sha256": sha256_file(ledger),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "new_oracle_calls": 0,
        "evidence_status": "zero-oracle cross-source proposal-support evidence",
    }
    seal(destination, payload)
    print(json.dumps(comparison, indent=2))
