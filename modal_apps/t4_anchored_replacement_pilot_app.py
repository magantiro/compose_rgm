"""Dock the immutable anchored-replacement JAK2 candidate lock once.

The local entry point verifies the self-hashed contract and candidate lock before the
remote function sees a molecule.  The remote function verifies the physical docking
binary and receptor, logs every result, performs no retry, and returns every attempted
row.  The local process publishes one sealed result atomically.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

from modal_apps.genmol_t4_opt_app import BOXES, image

app = modal.App("t4-anchored-replacement-pilot-v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=8, timeout=60 * 60, max_containers=3)
def dock_locked(payload: str) -> str:
    """Run every locked query once and stream per-query progress."""

    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from compose_v4.experiments.continuation_profile import sha256_file

    request = json.loads(payload)
    target = request["target"]
    expected = request["evaluator_sha256"]
    physical = {
        "qvina02": sha256_file(Path("/opt/dock/qvina02")),
        "receptor": sha256_file(Path(f"/opt/dock/receptors/{target}.pdbqt")),
    }
    if physical != expected:
        raise ValueError(f"docking evaluator identity mismatch: {physical} != {expected}")
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    seed = int(request["docking_seed"])

    def dock(item):
        index, smiles = int(item["index"]), str(item["smiles"])
        directory = f"/tmp/anchored_{index}_{item['endpoint_sha256'][:12]}"
        os.makedirs(directory, exist_ok=True)
        mol = f"{directory}/ligand.mol"
        ligand = f"{directory}/ligand.pdbqt"
        out = f"{directory}/out.pdbqt"
        try:
            subprocess.run(
                ["obabel", f"-:{smiles}", "--gen3D", "-O", mol],
                capture_output=True,
                timeout=120,
                check=True,
            )
            subprocess.run(
                ["obabel", mol, "-O", ligand],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            return {**item, "score": None, "failure": f"prepare:{type(error).__name__}"}
        command = [
            "/opt/dock/qvina02",
            "--receptor",
            f"/opt/dock/receptors/{target}.pdbqt",
            "--ligand",
            ligand,
            "--out",
            out,
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
            str(seed),
            "--cpu",
            "1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=600,
                check=True,
                text=True,
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
        return {**item, "score": score, "failure": None if score is not None else "no_score"}

    started = time.time()
    print(f"[anchored] docking {len(request['candidates'])} locked candidates", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(dock, request["candidates"]):
            completed.append(row)
            shown = "FAIL" if row["score"] is None else f"{row['score']:.2f}"
            print(
                f"[anchored] {len(completed):02d}/{len(request['candidates']):02d} "
                f"{shown:>6} {row['endpoint_sha256'][:12]}",
                flush=True,
            )
    return json.dumps(
        {
            "results": completed,
            "evaluator_sha256": physical,
            "elapsed_seconds": time.time() - started,
        }
    )


@app.local_entrypoint()
def main(
    contract: str = "configs/t4_anchored_replacement_pilot_v1.json",
    lock: str = "diagnostics/t4_anchored_replacement_pilot_v1/candidate_lock.json",
    output: str = "diagnostics/t4_anchored_replacement_pilot_v1/result.json",
) -> None:
    """Verify, execute and atomically publish the exact locked panel."""

    import platform
    import subprocess

    from rdkit import rdBase

    from compose_v4.experiments.t4_anchored_replacement_lock import verify_lock
    from compose_v4.experiments.t4_anchored_replacement_pilot import summarize_docking
    from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

    contract_path, lock_path, output_path = Path(contract), Path(lock), Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite scored result: {output_path}")
    contract_envelope = json.loads(contract_path.read_text())
    contract_payload = unseal(contract_path)
    lock_publication = unseal(lock_path)
    candidate_lock = lock_publication["assessment"]["lock"]
    verify_lock(candidate_lock)
    if _sha256(lock_path) != contract_payload["inputs"]["candidate_lock_sha256"]:
        raise ValueError("candidate lock file hash disagrees with scored contract")
    if candidate_lock["lock_id"] != contract_payload["candidate_lock_id"]:
        raise ValueError("candidate lock identity disagrees with scored contract")
    if candidate_lock["charged_calls"] != contract_payload["charged_call_limit"]:
        raise ValueError("candidate count disagrees with scored contract")

    candidates = [
        {
            "index": index,
            "smiles": row["endpoint"],
            "endpoint_sha256": row["endpoint_sha256"],
            "properties": row["properties"],
        }
        for index, row in enumerate(candidate_lock["selection"])
    ]
    request = {
        "target": candidate_lock["target"],
        "docking_seed": candidate_lock["docking_seed"],
        "evaluator_sha256": contract_payload["evaluator_sha256"],
        "candidates": candidates,
    }
    answer = json.loads(dock_locked.remote(json.dumps(request, sort_keys=True)))
    summary = summarize_docking(
        answer["results"],
        expected_digests=[row["endpoint_sha256"] for row in candidates],
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "schema_version": "t4_anchored_replacement_pilot_publication_v1",
        "contract_file_sha256": _sha256(contract_path),
        "contract_payload_sha256": contract_envelope["payload_sha256"],
        "candidate_lock_id": candidate_lock["lock_id"],
        "candidate_lock_file_sha256": _sha256(lock_path),
        "code_revision": revision,
        "results": answer["results"],
        "summary": summary,
        "evaluator_sha256": answer["evaluator_sha256"],
        "elapsed_seconds": answer["elapsed_seconds"],
        "new_oracle_calls": summary["charged_calls"],
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "claim_boundary": (
            "bounded answer-known JAK2 delta-0.6 development evidence; not held-out, "
            "not replicated, and not a benchmark-wide IVG comparison"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seal(temporary, payload)
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2), flush=True)


@app.local_entrypoint()
def confirm(
    contract: str = "configs/t4_anchored_replacement_confirmation_v1.json",
    output: str = "diagnostics/t4_anchored_replacement_confirmation_v1/result.json",
) -> None:
    """Run the frozen top-three by fresh-seed confirmation grid."""

    import platform
    import subprocess

    from rdkit import rdBase

    from compose_v4.experiments.t4_anchored_replacement_pilot import summarize_confirmation
    from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

    contract_path, output_path = Path(contract), Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite confirmation result: {output_path}")
    envelope = json.loads(contract_path.read_text())
    payload = unseal(contract_path)
    first_pass_path = Path(payload["inputs"]["first_pass_result"])
    if _sha256(first_pass_path) != payload["inputs"]["first_pass_result_sha256"]:
        raise ValueError("first-pass result hash disagrees with confirmation contract")

    handles = []
    for seed in payload["docking_seeds"]:
        request = {
            "target": payload["target"],
            "docking_seed": seed,
            "evaluator_sha256": payload["evaluator_sha256"],
            "candidates": [
                {
                    "index": index,
                    "smiles": row["smiles"],
                    "endpoint_sha256": row["endpoint_sha256"],
                    "first_pass_score": row["first_pass_score"],
                }
                for index, row in enumerate(payload["candidates"])
            ],
        }
        handles.append((seed, dock_locked.spawn(json.dumps(request, sort_keys=True))))

    rows, evaluator = [], None
    for seed, handle in handles:
        answer = json.loads(handle.get())
        if evaluator is None:
            evaluator = answer["evaluator_sha256"]
        elif evaluator != answer["evaluator_sha256"]:
            raise ValueError("confirmation workers used different evaluator identities")
        rows.extend({**row, "docking_seed": seed} for row in answer["results"])
    summary = summarize_confirmation(
        rows,
        expected_digests=[row["endpoint_sha256"] for row in payload["candidates"]],
        expected_seeds=[int(seed) for seed in payload["docking_seeds"]],
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    result = {
        "schema_version": "t4_anchored_replacement_confirmation_publication_v1",
        "contract_file_sha256": _sha256(contract_path),
        "contract_payload_sha256": envelope["payload_sha256"],
        "code_revision": revision,
        "results": rows,
        "summary": summary,
        "evaluator_sha256": evaluator,
        "new_oracle_calls": summary["charged_calls"],
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "claim_boundary": (
            "post-selection confirmation of three answer-known JAK2 development "
            "endpoints; not a fresh candidate evaluation or benchmark-wide result"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seal(temporary, result)
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2), flush=True)
