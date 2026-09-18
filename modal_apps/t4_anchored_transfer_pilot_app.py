"""Dock the immutable JAK2 seed-0 anchored-transfer lock exactly once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from modal_apps.genmol_t4_opt_app import BOXES, image

app = modal.App("t4-anchored-transfer-pilot-v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=1, timeout=900, max_containers=12, retries=0)
def dock_one(request: dict) -> dict:
    """Prepare and dock one locked endpoint without retry or replacement."""

    import os
    import subprocess

    from compose_v4.experiments.continuation_profile import sha256_file

    target = request["target"]
    physical = {
        "qvina02": sha256_file(Path("/opt/dock/qvina02")),
        "receptor": sha256_file(Path(f"/opt/dock/receptors/{target}.pdbqt")),
    }
    if physical != request["evaluator_sha256"]:
        raise ValueError(f"docking evaluator identity mismatch: {physical}")
    item = request["candidate"]
    directory = f"/tmp/anchored_transfer_{item['index']}_{item['endpoint_sha256'][:12]}"
    os.makedirs(directory, exist_ok=True)
    molecule = f"{directory}/ligand.mol"
    ligand = f"{directory}/ligand.pdbqt"
    output = f"{directory}/out.pdbqt"
    try:
        subprocess.run(
            ["obabel", f"-:{item['smiles']}", "--gen3D", "-O", molecule],
            capture_output=True,
            timeout=120,
            check=True,
        )
        subprocess.run(
            ["obabel", molecule, "-O", ligand],
            capture_output=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as error:
        return {**item, "score": None, "failure": f"prepare:{type(error).__name__}"}
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    command = [
        "/opt/dock/qvina02",
        "--receptor",
        f"/opt/dock/receptors/{target}.pdbqt",
        "--ligand",
        ligand,
        "--out",
        output,
        "--center_x",
        str(cx),
        "--center_y",
        str(cy),
        "--center_z",
        str(cz),
        "--size_x",
        str(sx),
        "--size_y",
        str(sy),
        "--size_z",
        str(sz),
        "--seed",
        str(request["docking_seed"]),
        "--cpu",
        "1",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=600, check=True, text=True
        )
    except (subprocess.SubprocessError, OSError) as error:
        return {**item, "score": None, "failure": f"dock:{type(error).__name__}"}
    score = None
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "1":
            try:
                score = float(fields[1])
            except ValueError:
                score = None
            break
    return {
        **item,
        "score": score,
        "failure": None if score is not None else "no_score",
    }


@app.local_entrypoint()
def main(
    contract: str = "configs/t4_anchored_transfer_pilot_v1.json",
    lock: str = (
        "diagnostics/t4_anchored_replacement_transfer_pilot_v1/candidate_lock.json"
    ),
    output: str = "diagnostics/t4_anchored_replacement_transfer_pilot_v1/result.json",
) -> None:
    """Verify the frozen lock, execute it, and atomically seal the result."""

    import platform
    import subprocess
    import time

    from rdkit import rdBase

    from compose_v4.experiments.t4_anchored_replacement_pilot import summarize_docking
    from compose_v4.experiments.t4_anchored_transfer_lock import verify_transfer_lock
    from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

    contract_path, lock_path, output_path = Path(contract), Path(lock), Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite scored result: {output_path}")
    envelope = json.loads(contract_path.read_text())
    contract_payload = unseal(contract_path)
    publication = unseal(lock_path)
    candidate_lock = publication["assessment"]["lock"]
    verify_transfer_lock(candidate_lock)
    if _sha256(lock_path) != contract_payload["inputs"]["candidate_lock_sha256"]:
        raise ValueError("candidate-lock physical hash mismatch")
    if candidate_lock["lock_id"] != contract_payload["candidate_lock_id"]:
        raise ValueError("candidate-lock identity mismatch")
    if candidate_lock["charged_calls"] != contract_payload["charged_call_limit"]:
        raise ValueError("candidate count disagrees with the scored contract")

    candidates = [
        {
            "index": index,
            "smiles": row["endpoint"],
            "endpoint_sha256": row["endpoint_sha256"],
            "properties": row["properties"],
        }
        for index, row in enumerate(candidate_lock["selection"])
    ]
    requests = [
        {
            "target": candidate_lock["target"],
            "docking_seed": candidate_lock["docking_seed"],
            "evaluator_sha256": contract_payload["evaluator_sha256"],
            "candidate": candidate,
        }
        for candidate in candidates
    ]
    started = time.time()
    results = []
    for row in dock_one.map(requests, order_outputs=False):
        results.append(row)
        shown = "FAIL" if row["score"] is None else f"{row['score']:.2f}"
        print(
            f"[anchored-transfer] {len(results):02d}/{len(candidates):02d} "
            f"{shown:>6} {row['endpoint_sha256'][:12]}",
            flush=True,
        )
    results.sort(key=lambda row: int(row["index"]))
    summary = summarize_docking(
        results, expected_digests=[row["endpoint_sha256"] for row in candidates]
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "schema_version": "t4_anchored_transfer_pilot_publication_v1",
        "contract_file_sha256": _sha256(contract_path),
        "contract_payload_sha256": envelope["payload_sha256"],
        "candidate_lock_id": candidate_lock["lock_id"],
        "candidate_lock_file_sha256": _sha256(lock_path),
        "code_revision": revision,
        "results": results,
        "summary": summary,
        "evaluator_sha256": contract_payload["evaluator_sha256"],
        "elapsed_seconds": time.time() - started,
        "new_oracle_calls": summary["charged_calls"],
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "claim_boundary": (
            "bounded prospective utility evidence on one second-source JAK2 "
            "delta-0.6 development cell; not FiberControl or benchmark evidence"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seal(temporary, payload)
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2), flush=True)
