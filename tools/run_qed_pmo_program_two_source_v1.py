"""Matched two-source QED diagnostic for the PMO population program controller.

This is not the 800-source GrIDDD evaluation. Both arms use the same source,
legal COMPOSE executor, proposal budget, local QED scorer and feedback ledger.
The PMO arm adds the task-blind joint-dependency jump channel and population
allocation to the older Dynamic-v2.1 program controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.control.dynamic_program_synthesis_v21 import (
    DynamicV21ProgramOptimizer,
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, initialization_lock
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.pmo_population_v1 import configuration
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/qed_pmo_program_two_source_v1.json"
SOURCES = ROOT / "configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv"
CHECKPOINT = ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ARMS = ("dynamic_v21", "pmo_population")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(path: Path = CONTRACT) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed QED pilot contract: {path}")
    payload = envelope["payload"]
    if digest(payload) != envelope["payload_sha256"]:
        raise ValueError(f"QED pilot contract hash mismatch: {path}")
    if (
        payload["schema"] != "qed_pmo_program_two_source_v1"
        or payload["source_indices"] != [0, 1]
        or payload["arms"] != list(ARMS)
        or payload["charged_qed_calls_per_source_arm"] != 25
        or payload["rounds"] != 3
        or payload["queries_per_round"] != 8
        or payload["attempts_per_batch"] != 64
        or payload["proposal_wall_seconds"] != 20.0
        or payload["similarity_floor"] != 0.4
        or payload["qed_success_floor"] != 0.9
        or payload["oracle"] != "local_rdkit_qed_only"
        or payload["workers"] != 1
    ):
        raise ValueError("QED pilot scientific envelope changed")
    return payload, envelope["payload_sha256"]


def source_rows(contract: dict) -> list[dict]:
    with SOURCES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for index in contract["source_indices"]:
        row = rows[index]
        if int(row["lead_index"]) != index:
            raise ValueError(f"Jin source index mismatch: {index}")
        smiles = row["canonical_isomeric_smiles"]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or not 0.7 <= QED.qed(molecule) <= 0.8:
            raise ValueError(f"Jin source outside declared supported cohort: {index}")
        state = production_state_from_smiles(smiles, 48)
        selected.append(
            {
                "index": index,
                "smiles": smiles,
                "source_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
                "qed": float(QED.qed(molecule)),
                "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
                "encoded_state": encode_state(state),
            }
        )
    return selected


def preflight(contract: dict) -> list[dict]:
    if Path(sys.prefix).resolve() != Path(contract["python_executable"]).parent.parent.resolve():
        raise ValueError(f"wrong QED pilot Python: {sys.executable}")
    versions = {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion}
    if versions != contract["versions"]:
        raise ValueError("QED pilot scientific runtime changed")
    for relative, expected in contract["material_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"QED pilot material changed: {relative}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["source_revision"], revision],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("QED pilot source revision is not an ancestor of HEAD")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise ValueError("tracked QED pilot source is dirty")
    return source_rows(contract)


def source_seed(smiles: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"qed-pmo-program-two-source-v1|{index}|{smiles}".encode()).digest()[:4],
        "big",
    )


def properties(smiles: str, source_fingerprint) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"scored QED endpoint is not an RDKit molecule: {smiles}")
    similarity = float(
        DataStructs.TanimotoSimilarity(source_fingerprint, FINGERPRINT.GetFingerprint(molecule))
    )
    return {"qed": float(QED.qed(molecule)), "similarity": similarity}


def run_one(contract: dict, source: dict, arm: str, output: Path, checkpoint: dict) -> dict:
    index = source["index"]
    folder = output / arm / f"source_{index:03d}"
    result_path = folder / "result.json"
    if result_path.exists():
        raise FileExistsError(f"QED pilot source already completed: {result_path}")
    seed = source_seed(source["smiles"], index)
    source_fingerprint = FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(source["smiles"]))

    def evaluate(smiles: str) -> float:
        measured = properties(smiles, source_fingerprint)
        return measured["qed"] if measured["similarity"] >= contract["similarity_floor"] else 0.0

    task = ProgramTask(
        f"qed_jin_{index:03d}_{arm}",
        digest(
            {
                "schema": contract["schema"],
                "source_sha256": source["source_sha256"],
                "similarity_floor": contract["similarity_floor"],
                "score": "QED(y) if source-relative Tanimoto >= floor else zero",
            }
        ),
        "pmo",
    )
    initialization = initialization_lock(
        [{"state": source["encoded_state"], "source_id": f"jin{index:03d}"}],
        count=1,
        seed=seed,
        source_sha256=source["source_sha256"],
    )
    config = replace(
        configuration(seed),
        attempts_per_batch=contract["attempts_per_batch"],
        candidates_per_batch=contract["queries_per_round"],
        wall_seconds=contract["proposal_wall_seconds"],
        require_broad_runtime=False,
    )
    ledger = ProgramQueryLedger(
        folder / "qed_ledger",
        task,
        evaluate,
        budget=contract["charged_qed_calls_per_source_arm"],
    )

    def progress(row: dict) -> None:
        event = {"arm": arm, "source_index": index, **row}
        publish_json(folder / "progress.json", event)
        print(json.dumps(event, sort_keys=True), flush=True)

    campaign = run_program_campaign(
        output=folder / "campaign",
        task=task,
        config=config,
        initialization=initialization,
        library=(),
        ledger=ledger,
        rounds=contract["rounds"],
        queries_per_round=contract["queries_per_round"],
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        progress=progress,
        optimizer_type=(
            DynamicV21ProgramOptimizer if arm == "dynamic_v21" else PmoPopulationController
        ),
        optimizer_kwargs={} if arm == "dynamic_v21" else {"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    scored = []
    for row in ledger.rows:
        if row["status"] != "complete":
            raise RuntimeError("QED pilot ledger contains an unresolved charge")
        measured = properties(row["endpoint"], source_fingerprint)
        scored.append(
            {
                "smiles": row["endpoint"],
                "role": row["role"],
                **measured,
                "score": row["score"],
                "similarity_eligible": measured["similarity"] >= contract["similarity_floor"],
                "success": (
                    measured["similarity"] >= contract["similarity_floor"]
                    and measured["qed"] >= contract["qed_success_floor"]
                ),
            }
        )
    channel_counts: Counter[str] = Counter()
    for pending in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        for candidate in batch["candidates"]:
            channel_counts[
                str(candidate.get("provenance", {}).get("planner_channel", "unknown"))
            ] += 1
    record = {
        "schema": "qed_pmo_program_two_source_result_v1",
        "role": "two-source matched development diagnostic; not a GrIDDD benchmark result",
        "source_index": index,
        "source_sha256": source["source_sha256"],
        "arm": arm,
        "seed": seed,
        "charged_qed_calls": len(scored),
        "call_ceiling": contract["charged_qed_calls_per_source_arm"],
        "successful_endpoints": sum(row["success"] for row in scored),
        "any_success": any(row["success"] for row in scored),
        "best_eligible_qed": max(
            (row["qed"] for row in scored if row["similarity_eligible"]), default=None
        ),
        "selected_channel_counts": dict(sorted(channel_counts.items())),
        "scored": scored,
        "campaign_rounds": campaign["history"],
    }
    publish_json(result_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    contract, contract_hash = load_contract()
    sources = preflight(contract)
    output = ROOT / contract["output_dir"]
    manifest = {
        "schema": "qed_pmo_program_two_source_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "source_identities": [
            {key: source[key] for key in ("index", "smiles", "source_sha256", "qed", "heavy_atoms")}
            for source in sources
        ],
        "checkpoint_role": "task-blind PMO joint-dependency plan latents; no QED fitting",
    }
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(f"QED pilot output already exists: {output}")
        publish_json(manifest_path, manifest)
        print(json.dumps({"prepared_only": True, "manifest_sha256": sha256_file(manifest_path)}))
        return
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("QED pilot manifest missing or changed")
    checkpoint = json.loads(CHECKPOINT.read_text())["payload"]["checkpoints"]["shared_all_routes"]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("QED pilot checkpoint is not the task-blind shared fit")
    for source in sources:
        for arm in ARMS:
            run_one(contract, source, arm, output, checkpoint)


if __name__ == "__main__":
    main()
