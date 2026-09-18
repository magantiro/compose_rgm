"""Parallel zero-oracle route-guided support probe on independent CPU workers."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import modal

from modal_apps.genmol_t4_opt_app import REMOTE_ROOT, ROOT
from modal_apps.genmol_t4_opt_app import image as base_image

APP_NAME = "compose-t4-route-guided-support"
CONTRACT = "configs/t4_route_guided_support_probe_v3.json"

image = base_image
image = image.add_local_dir(
    ROOT / "src",
    str(REMOTE_ROOT / "src"),
    copy=True,
    ignore=("**/__pycache__/**", "**/*.pyc"),
)
for relative in (
    "modal_apps/t4_route_guided_support_app.py",
    "diagnostics/t4_proposal_prior/construction_corpus_v1/decisions.jsonl.gz",
    "diagnostics/t4_proposal_prior/construction_corpus_v1/census.json",
):
    image = image.add_local_file(ROOT / relative, str(REMOTE_ROOT / relative), copy=True)

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

    from compose_v4.control.constructive_policy import PolicyShape
    from compose_v4.experiments.t4_route_guided_support import run_arm
    from compose_v4.rewrite.trace_shard import decode_state

    model = {
        "shape": PolicyShape(**task["model"]["shape"]),
        "theta": np.asarray(task["model"]["theta"], dtype=float),
    }
    result = run_arm(
        decode_state(task["source_state"]),
        task["original_seed"],
        task["target"],
        model,
        task["vocabulary"],
        arm=task["arm"],
        attempts=1,
        seed=task["worker_seed"],
        delta=task["delta"],
        depths=tuple(task["depths"]),
        candidates_per_step=task["candidates_per_step"],
        route_exploration=task["route_exploration"],
        temperature=task["temperature"],
    )
    result["worker_seed"] = task["worker_seed"]
    return result


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


def _contract(path: Path) -> tuple[dict, str]:
    from compose_v4.experiments.continuation_profile import canonical_bytes

    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    actual = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise ValueError(f"contract hash mismatch: {actual}")
    return payload, actual


def _ledger(path: Path, rows: list[dict]) -> str:
    from compose_v4.experiments.continuation_profile import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(path)


@app.local_entrypoint()
def main(
    cell: str = "jak2_1",
    arm: str = "route_prior",
    output: str = "diagnostics/t4_route_guided_support_probe_v1/jak2_1_route_prior.json",
) -> None:
    import numpy as np
    from rdkit import rdBase

    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
    from compose_v4.experiments.t4_route_guided_support import (
        aggregate_attempt_results,
        fit_held_target_prior,
        load_decisions,
    )

    contract, contract_hash = _contract(ROOT / CONTRACT)
    if cell not in contract["cells"]:
        raise ValueError(f"cell must be one of {sorted(contract['cells'])}")
    if arm not in contract["arms"]:
        raise ValueError(f"arm must be one of {contract['arms']}")
    for relative, expected in contract["inputs"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {relative}: {actual}")

    registry = unseal(ROOT / contract["source_registry"])
    unit = registry["cells"][cell]
    rows = load_decisions(ROOT / contract["decision_corpus"])
    model, vocabulary, split_audit = fit_held_target_prior(rows, unit["target"])
    if not split_audit["held_target_absent_from_training"]:
        raise RuntimeError("held target entered route-prior training")

    attempt_budget = contract["attempts_by_cell"][cell]
    base_seed = contract["seed_by_cell"][cell]
    serial_model = {
        "shape": {
            "site_features": model["shape"].site_features,
            "mode_features": model["shape"].mode_features,
        },
        "theta": np.asarray(model["theta"]).tolist(),
    }
    tasks = [
        {
            "cell": cell,
            "arm": arm,
            "worker_seed": base_seed + attempt,
            "source_state": unit["source_state"],
            "original_seed": unit["original_seed"],
            "target": unit["target"],
            "model": serial_model,
            "vocabulary": vocabulary,
            "delta": contract["delta"],
            "depths": contract["depths"],
            "candidates_per_step": contract["candidates_per_step"],
            "route_exploration": contract["route_exploration"],
            "temperature": contract["temperature"],
        }
        for attempt in range(attempt_budget)
    ]
    results = list(worker.map(tasks, order_outputs=False))
    aggregate = aggregate_attempt_results(results, arm=arm, attempt_budget=attempt_budget)
    ledger_path = ROOT / Path(output).with_suffix(".jsonl.gz")
    ledger_hash = _ledger(ledger_path, aggregate.pop("ledger"))
    payload = {
        **aggregate,
        "cell": cell,
        "target": unit["target"],
        "delta": contract["delta"],
        "seed_range": [base_seed, base_seed + attempt_budget - 1],
        "split_audit": split_audit,
        "contract": CONTRACT,
        "contract_payload_sha256": contract_hash,
        "ledger": str(ledger_path.relative_to(ROOT)),
        "ledger_sha256": ledger_hash,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "evidence_status": (
            "zero-oracle generated support; not docking utility and not autonomous T4 performance"
        ),
    }
    destination = ROOT / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    seal(destination, payload)
    print(
        json.dumps({key: value for key, value in payload.items() if "rates" not in key}, indent=2)
    )
