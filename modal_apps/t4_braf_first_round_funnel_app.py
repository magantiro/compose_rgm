"""Parallel zero-oracle reconstruction of failed BRAF first-round proposal funnels."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
OUTPUT = Path("/braf_funnel")
CONTRACT = "configs/t4_braf_first_round_funnel_v1.json"
SOURCE_CONTRACT = "configs/t4_integrated_route_fiber_braf_v1.json"
ROUTE_SMOKE = "diagnostics/t4_integrated_route_fiber_braf_v1/route_expert_smoke.json"
VOLUME_NAME = "compose-t4-braf-first-round-funnel-v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True)
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(ROOT / SOURCE_CONTRACT, str(REMOTE / SOURCE_CONTRACT), copy=True)
    .add_local_file(ROOT / ROUTE_SMOKE, str(REMOTE / ROUTE_SMOKE), copy=True)
    .add_local_file(
        ROOT / "modal_apps/t4_braf_first_round_funnel_app.py",
        str(REMOTE / "modal_apps/t4_braf_first_round_funnel_app.py"),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("compose-t4-braf-first-round-funnel-v1")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 4096,
    "retries": 0,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 30,
}


def _load_envelope(path: Path) -> dict:
    from compose_v4.control.docking_value import identity

    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"{path}: expected a sealed payload envelope")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"{path}: payload identity mismatch")
    return envelope


def _load_contract() -> dict:
    from compose_v4.experiments.continuation_profile import sha256_file

    envelope = _load_envelope(REMOTE / CONTRACT)
    for relative, expected in envelope["payload"]["inputs_sha256"].items():
        actual = sha256_file(REMOTE / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
    return envelope


def _publish_json(path: Path, payload: dict) -> str:
    from compose_v4.control.docking_value import identity

    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    volume.commit()
    return envelope["payload_sha256"]


def _publish_ledger(path: Path, rows: list[dict]) -> str:
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
    ):
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            compressed.write(line.encode())
    temporary.replace(path)
    volume.commit()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(**common, max_containers=6, timeout=3600)
def audit_worker(task: dict) -> dict:
    import os
    import platform

    from rdkit import rdBase

    from compose_v4.experiments.t4_proposal_funnel_audit import (
        audit_expansion_lane,
        audit_route_candidates,
    )

    contract_envelope = _load_contract()
    contract = contract_envelope["payload"]
    if task["contract_payload_sha256"] != contract_envelope["payload_sha256"]:
        raise ValueError("audit task contract identity mismatch")
    cell_spec = contract["cells"][task["cell"]]
    lane = task["lane"]
    if lane == "route_complete_region":
        route = _load_envelope(REMOTE / ROUTE_SMOKE)["payload"]
        observed = next(row for row in route["cells"] if row["cell"] == task["cell"])
        lane_spec = contract["lanes"][lane]
        summary, assessments = audit_route_candidates(
            cell=task["cell"],
            seed_smiles=cell_spec["smiles"],
            delta=contract["delta"],
            support=contract["support"],
            proposed_pool=lane_spec["proposed_pool"],
            realization_limit=lane_spec["realization_limit"],
            telemetry=observed["telemetry"],
            candidates=observed["candidates"],
        )
    else:
        lane_spec = contract["lanes"][lane]
        summary, assessments = audit_expansion_lane(
            cell=task["cell"],
            seed_smiles=cell_spec["smiles"],
            delta=contract["delta"],
            support=contract["support"],
            lane=lane,
            proposal_seed=cell_spec["proposal_seeds"][lane],
            draws=lane_spec["draws"],
            horizon=lane_spec["horizon"],
        )
    folder = OUTPUT / task["run_id"] / task["cell"]
    ledger_path = folder / f"{lane}.jsonl.gz"
    ledger_sha = _publish_ledger(ledger_path, assessments)
    payload = {
        **summary,
        "contract_payload_sha256": contract_envelope["payload_sha256"],
        "ledger": str(ledger_path.relative_to(OUTPUT)),
        "ledger_sha256": ledger_sha,
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "cpu_count": os.cpu_count(),
        },
        "evidence_status": "zero-oracle deterministic first-round funnel reconstruction",
    }
    _publish_json(folder / f"{lane}.summary.json", payload)
    print(
        f"[{task['cell']}:{lane}] attempted={payload['programs_attempted']} "
        f"synthesized={payload['programs_synthesized']} "
        f"executed={payload['cumulative_funnel']['distinct_executed_endpoints']} "
        f"eligible={payload['cumulative_funnel']['all_compose_valid']}",
        flush=True,
    )
    return payload


@app.function(**common, max_containers=1, timeout=120)
def finalize(task: dict, summaries: list[dict]) -> dict:
    contract_envelope = _load_contract()
    expected = {
        (cell, lane)
        for cell in contract_envelope["payload"]["cells"]
        for lane in contract_envelope["payload"]["lanes"]
    }
    observed = {(row["cell"], row["lane"]) for row in summaries}
    if observed != expected:
        raise ValueError(f"incomplete audit summaries: {sorted(expected - observed)}")
    payload = {
        "schema_version": "t4_braf_first_round_funnel_result_v1",
        "run_id": task["run_id"],
        "contract_payload_sha256": contract_envelope["payload_sha256"],
        "summaries": sorted(summaries, key=lambda row: (row["cell"], row["lane"])),
        "new_oracle_calls": 0,
        "evidence_status": "zero-oracle deterministic first-round funnel reconstruction",
        "claim_boundary": contract_envelope["payload"]["claim_boundary"],
    }
    _publish_json(OUTPUT / task["run_id"] / "result.json", payload)
    return payload


@app.function(**common, max_containers=1, timeout=2 * 3600)
def drive(task: dict) -> dict:
    tasks = [
        {**task, "cell": cell, "lane": lane}
        for cell in task["cells"]
        for lane in task["lanes"]
    ]
    summaries = list(audit_worker.map(tasks, order_outputs=True))
    return finalize.remote(task, summaries)


@app.function(**common, max_containers=1, timeout=120)
def remote_status(task: dict) -> dict:
    volume.reload()
    folder = OUTPUT / task["run_id"]
    result = folder / "result.json"
    if result.exists():
        return {"status": "complete", **_load_envelope(result)["payload"]}
    summaries = []
    for cell in task["cells"]:
        for lane in task["lanes"]:
            path = folder / cell / f"{lane}.summary.json"
            if path.exists():
                payload = _load_envelope(path)["payload"]
                summaries.append(
                    {
                        "cell": cell,
                        "lane": lane,
                        "status": "complete",
                        "executed": payload["cumulative_funnel"][
                            "distinct_executed_endpoints"
                        ],
                        "eligible": payload["cumulative_funnel"]["all_compose_valid"],
                    }
                )
            else:
                summaries.append({"cell": cell, "lane": lane, "status": "pending"})
    return {"status": "running", "run_id": task["run_id"], "summaries": summaries}


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


def _local_task(run_id: str | None = None) -> dict:
    from compose_v4.control.docking_value import identity

    envelope = json.loads((ROOT / CONTRACT).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("local audit contract identity mismatch")
    resolved = run_id or identity(
        {
            "contract_payload_sha256": envelope["payload_sha256"],
            "code_revision": _revision(),
        }
    )
    return {
        "run_id": resolved,
        "contract_payload_sha256": envelope["payload_sha256"],
        "cells": sorted(envelope["payload"]["cells"]),
        "lanes": sorted(envelope["payload"]["lanes"]),
        "code_revision": _revision(),
    }


@app.local_entrypoint()
def main(mode: str = "launch", run_id: str | None = None) -> None:
    task = _local_task(run_id)
    if mode == "status":
        print(json.dumps(remote_status.remote(task), indent=2))
        return
    if mode != "launch":
        raise ValueError("mode must be launch or status")
    call = drive.spawn(task)
    print(
        json.dumps(
            {
                "run_id": task["run_id"],
                "function_call_id": call.object_id,
                "volume": VOLUME_NAME,
                "contract_payload_sha256": task["contract_payload_sha256"],
                "code_revision": task["code_revision"],
            },
            indent=2,
        )
    )
