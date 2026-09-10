"""Answer-known six-molecule diagnostic. Winner first; never optimizer training."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_macro_beam import exact_archive_graph
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_partial_docking import dock_saved_row

KIND = "t4_winner_route_docking"
CONTRACT_PATH = f"configs/{KIND}.json"
STAGES = (
    "remodel_linker",
    "pendant_benzene",
    "fused_six_ring",
    "ring_carbonyl",
    "core_carbonyl_insertion",
)


def lock_panel(contract, repo_root, artifact_root, task):
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("winner diagnostic requires pinned RDKit")
    witness_path = repo_root / contract["witness"]["path"]
    best_path = artifact_root / contract["incumbent"]["path"]
    verify_file(witness_path, contract["witness"]["sha256"])
    verify_file(best_path, contract["incumbent"]["sha256"])
    witness = json.loads(witness_path.read_text())
    previous = json.loads(best_path.read_text())
    attempts = [a for a in witness["attempts"] if a["status"] == "exact_winner"]
    if len(attempts) != 1 or previous["status"] != "complete":
        raise ValueError("requires completed exact witness and incumbent episode")
    stages = attempts[0]["stages"]
    if tuple(s["name"] for s in stages) != STAGES:
        raise ValueError("saved route stage census changed")
    rows = [{"role": s["name"], "smiles": s["endpoint"], "state": s["states"][-1]} for s in stages]
    best = previous["best"]
    if rows[-1]["smiles"] != witness["target"] or best["ds"] != contract["winner_continue_below"]:
        raise ValueError("winner or incumbent identity drift")
    rows = [
        rows[-1],
        *rows[:-1],
        {"role": "incumbent_redock", "smiles": best["smiles"], "state": best["state"]},
    ]
    if (
        len(rows) != 6
        or contract["compute"]["oracle_call_limit"] != 6
        or len({r["smiles"] for r in rows}) != 6
    ):
        raise ValueError("panel limit or canonical identity drift")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(task["smiles"]))
    for row in rows:
        exact_archive_graph(row)
        row.update(
            calculate_properties(
                Chem.MolFromSmiles(row["smiles"]),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            )
        )
        row["oracle_eligible"] = acceptable_endpoint(row)
        if not row["oracle_eligible"]:
            raise ValueError(f"locked diagnostic endpoint is ineligible: {row['role']}")
    return {
        "schema_version": "t4_winner_route_candidate_lock_v1",
        "task": task,
        "required_rdkit": contract["required_rdkit"],
        "take": rows,
        "winner_continue_below": contract["winner_continue_below"],
        "input_sha256": {
            str(witness_path): sha256_file(witness_path),
            str(best_path): sha256_file(best_path),
        },
        "training_allowed": False,
        "optimizer_continuation": False,
        "interpretation": "answer-known diagnostic, not autonomous discovery or prospective benchmark",
    }


def run_panel(lock, output, execute, *, commit, progress):
    path = output / "candidate_lock.json"
    if path.exists():
        if unseal(path) != lock:
            raise ValueError("saved panel differs from current lock")
    else:
        seal(path, lock)
        commit()
    digest = sha256_file(path)
    barrier = output / "docking_started.json"
    if barrier.exists():
        if json.loads(barrier.read_text())["candidate_lock_sha256"] != digest:
            raise ValueError("batch barrier differs from panel")
    else:
        publish_json(
            barrier,
            {"candidate_lock_sha256": digest, "maximum_attempts": 6, "started_at_utc": _stamp()},
        )
        commit()
    rows = []
    status = "complete"
    for index, candidate in enumerate(lock["take"]):
        row = execute(index, digest)
        if (
            row["index"] != index
            or row["smiles"] != candidate["smiles"]
            or row["candidate_lock_sha256"] != digest
        ):
            raise ValueError("docking identity differs from locked candidate")
        rows.append({**candidate, **row})
        progress.update(
            phase="winner_checked" if index == 0 else "route_docking",
            completed=len(rows),
            last_role=candidate["role"],
            last_score=row["ds"],
        )
        publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
        commit()
        if index == 0 and (row["ds"] is None or row["ds"] >= lock["winner_continue_below"]):
            status = (
                "winner_oracle_failed" if row["ds"] is None else "winner_advantage_not_observed"
            )
            break
    return {
        "schema_version": "t4_winner_route_docking_result_v1",
        "status": status,
        "docked": rows,
        "new_oracle_attempts": len(rows),
        "oracle_failures": sum(r["ds"] is None for r in rows),
        "candidate_lock_sha256": digest,
        "training_allowed": False,
        "optimizer_continuation": False,
        "new_generator_calls": 0,
        "new_executor_calls": 0,
        "interpretation": lock["interpretation"],
    }


def run_remote(task, repo_root, artifact_root, volume, validate_revision, dock):
    from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        contract = json.loads((repo_root / CONTRACT_PATH).read_text())
        lock = lock_panel(contract, repo_root, artifact_root, actual_task)

        def execute(index, digest):
            def preserving_dock(smiles, tag):
                score = dock(smiles, tag)
                folder = output / "poses" / f"{index:02}"
                folder.mkdir(parents=True, exist_ok=True)
                files = {}
                for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
                    source = Path("/tmp") / tag / name
                    if source.exists():
                        shutil.copyfile(source, folder / name)
                        files[name] = sha256_file(folder / name)
                publish_json(folder / "manifest.json", {"smiles": smiles, "sha256": files})
                return score

            return dock_saved_row(
                {**task, "index": index, "candidate_lock_sha256": digest},
                repo_root,
                artifact_root,
                volume,
                validate_revision,
                preserving_dock,
                run_kind=KIND,
                batch_limit=6,
            )

        result = run_panel(lock, output, execute, commit=commit, progress=progress)
        result["software"] = {
            "rdkit": rdBase.rdkitVersion,
            "python": platform.python_version(),
            "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
        }
        result["randomness"] = (
            "unchanged production docking: Open Babel and QuickVina seeds not explicitly set"
        )
        return result

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        None,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )
