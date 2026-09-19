"""Approved saved-prefix docking diagnostic, never an optimizer continuation."""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import calculate_properties, feasible_endpoint
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_warm_continuation import endpoint, payload_hash

CONTRACT_PATH = "configs/t4_partial_docking.json"
KIND = "t4_partial_docking"


def lock_saved_batch(
    source: Path, contract: dict, task: dict, *, qed_min: float, sa_max: float
) -> dict:
    """Re-evaluate saved endpoints only; retain every exclusion and exact product."""
    for relative, digest in contract["source"]["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("source artifact path escapes approved prefix")
        verify_file(path, digest)
    failure = json.loads((source / "failure.json").read_text())
    if failure["error_type"] != "ContinuationBudgetExceeded" or failure["parent_complete"] != 7:
        raise ValueError("source is not the approved interrupted seven-parent run")
    warm = unseal(source / "warm_start.json")
    if warm["oracle_attempts"] != 20 or warm["round"] != 1:
        raise ValueError("partial diagnostic requires all 20 source guided calls")
    paths = sorted(k for k in contract["source"]["files"] if k.startswith("round_2/parents/"))
    units = [unseal(source / path) for path in paths]
    if [u["parent_index"] for u in units] != list(range(7)):
        raise ValueError("partial diagnostic requires exactly the approved seven parent units")
    source_task = units[0]["task"]
    if any(u["task"] != source_task for u in units) or (
        source_task["code_revision"] != contract["source"]["code_revision"]
        or source_task["warm_start_sha256"] != payload_hash(warm)
        or source_task["delta"] != task["delta"]
        or source_task["smiles"] != task["smiles"]
    ):
        raise ValueError("saved proposals disagree with approved task/archive identity")
    allowed = contract["authorized_smiles"]
    if (
        len(allowed) != 13
        or len(set(allowed)) != 13
        or contract["compute"]["oracle_call_limit"] != 13
    ):
        raise ValueError("only the approved set of 13 saved molecules is authorized")
    prior = {Chem.MolToSmiles(Chem.MolFromSmiles(c["smiles"])) for c in warm["archive"]}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(task["smiles"]))
    pool, seen = [], set()
    for unit in units:
        for candidate in unit["candidates"]:
            exact = endpoint({"work": [unit["work"]]}, candidate)
            smiles = candidate["smiles"]
            properties = calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=task["delta"],
                qed_min=qed_min,
                sa_max=sa_max,
            )
            if properties is None:
                raise ValueError("unparseable exact saved endpoint")
            reasons = []
            if smiles in prior:
                reasons.append("already_evaluated_or_seed")
            if smiles in seen:
                reasons.append("canonical_duplicate")
            if smiles not in allowed:
                reasons.append("not_in_approved_batch")
            if not feasible_endpoint(properties):
                reasons.append("existing_t4_constraint_violation")
            if (
                candidate["option"] in ("build_ring_system", "build_fused_ring")
                and not candidate["program_complete"]
            ):
                reasons.append("incomplete_compound_option")
            seen.add(smiles)
            pool.append(
                {
                    **candidate,
                    **properties,
                    "state": exact,
                    "oracle_eligible": not reasons,
                    "endpoint_exclusion_reasons": reasons,
                }
            )
    if not set(allowed) <= seen:
        raise ValueError("approved molecule missing from saved exact proposals")
    take = sorted(
        (c for c in pool if c["oracle_eligible"]), key=lambda c: (c["bundle_id"], c["smiles"])
    )
    if len(take) > 13 or len({c["bundle_id"] for c in take}) != len(take):
        raise ValueError("approved partial batch exceeds call/bundle bounds")
    return {
        "schema_version": "t4_partial_candidate_lock_v1",
        "task": task,
        "source": contract["source"],
        "source_task": source_task,
        "source_warm_start_sha256": payload_hash(warm),
        "constraints": {"qed_min": qed_min, "sa_max": sa_max, "delta": task["delta"]},
        "required_rdkit": contract["required_rdkit"],
        "pool": pool,
        "take": take,
        "bundles": [b for u in units for b in u["bundles"]],
        "work": [u["work"] for u in units],
        "oracle_calls": 0,
        "new_executor_calls": 0,
        "new_generator_calls": 0,
        "locked_at_utc": _stamp(),
        "software": {
            "rdkit": rdBase.rdkitVersion,
            "python": platform.python_version(),
            "sa_scorer_sha256": sha256_file(Path(sascorer.__file__)),
            "sa_fragment_scores_sha256": sha256_file(
                Path(sascorer.__file__).with_name("fpscores.pkl.gz")
            ),
        },
        "interpretation_scope": "partial seven-parent diagnostic, not completed optimizer round",
    }


def run_batch(
    task,
    contract,
    source,
    output,
    dock_parallel,
    *,
    qed_min,
    sa_max,
    commit=lambda: None,
    progress=None,
):
    progress = {} if progress is None else progress
    path = output / "candidate_lock.json"
    if path.exists():
        lock = unseal(path)
        if lock["task"] != task or lock["source"] != contract["source"]:
            raise ValueError("saved candidate lock differs from approved task")
    else:
        lock = lock_saved_batch(source, contract, task, qed_min=qed_min, sa_max=sa_max)
        seal(path, lock)
        commit()
    digest = sha256_file(path)
    result_path, attempt_path = output / "docking.json", output / "docking_started.json"
    if result_path.exists():
        result = unseal(result_path)
        if result["candidate_lock_sha256"] != digest:
            raise ValueError("saved docking differs from candidate lock")
        return result
    if attempt_path.exists():
        raise RuntimeError("partial batch already attempted; no implicit oracle retry")
    publish_json(
        attempt_path,
        {
            "candidate_lock_sha256": digest,
            "attempts": len(lock["take"]),
            "started_at_utc": _stamp(),
        },
    )
    commit()
    progress.update(phase="docking", locked_candidates=len(lock["take"]), completed=0)
    start = time.perf_counter()
    rows, errors = {}, []
    for row in dock_parallel(lock, digest, output):
        if isinstance(row, BaseException):
            errors.append(str(row))
            continue
        index = row["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(lock["take"])
            or index in rows
            or row["candidate_lock_sha256"] != digest
            or row["smiles"] != lock["take"][index]["smiles"]
        ):
            raise ValueError(
                "worker result has duplicate, missing, or mismatched candidate identity"
            )
        if row["ds"] is not None and not math.isfinite(row["ds"]):
            raise ValueError("nonfinite docking result")
        rows[index] = row
        scores = [r["ds"] for r in rows.values() if r["ds"] is not None]
        progress.update(completed=len(rows), best_batch_score=min(scores, default=None))
        publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
        commit()
    if errors or len(rows) != len(lock["take"]):
        raise RuntimeError(f"incomplete docking batch; preserve all attempts, no retries: {errors}")
    docked = [{**candidate, **rows[index]} for index, candidate in enumerate(lock["take"])]
    warm = unseal(source / "warm_start.json")
    old = [c for c in warm["archive"] if c["ds"] is not None and c["v"] <= 0]
    valid = [c for c in docked if c["ds"] is not None]
    best = min([*old, *valid], key=lambda c: c["ds"], default=None)
    result = {
        "schema_version": "t4_partial_docking_result_v1",
        "status": "complete",
        "optimizer_round_completed": False,
        "candidate_lock_sha256": digest,
        "docked": docked,
        "new_oracle_attempts": len(docked),
        "oracle_failures": sum(c["ds"] is None for c in docked),
        "cumulative_guided_calls": 20 + len(docked),
        "remaining_new_call_allowance": 40 - len(docked),
        "best_source": min(old, key=lambda c: c["ds"], default=None),
        "best_batch": min(valid, key=lambda c: c["ds"], default=None),
        "best_so_far": best,
        "new_executor_calls": 0,
        "new_generator_calls": 0,
        "docking_seconds": time.perf_counter() - start,
        "completed_at_utc": _stamp(),
        "interpretation_scope": lock["interpretation_scope"],
    }
    seal(result_path, result)
    commit()
    return result


def dock_saved_row(
    task,
    repo_root,
    artifact_root,
    volume,
    validate_revision,
    dock,
    *,
    run_kind=KIND,
    batch_limit=13,
):
    validate_revision(task["image_revision"])
    verify_file(repo_root / "modal_apps/genmol_t4_opt_app.py", task["app_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", task["run_id"]):
        raise ValueError("malformed partial docking run ID")
    volume.reload()
    root = artifact_root / run_kind / task["run_id"]
    path = root / "candidate_lock.json"
    verify_file(path, task["candidate_lock_sha256"])
    lock = unseal(path)
    if rdBase.rdkitVersion != lock["required_rdkit"]:
        raise ValueError("worker RDKit differs from locked pinned runtime")
    for name, path in {
        "qvina02": Path("/opt/dock/qvina02"),
        "receptor": Path("/opt/dock/receptors/parp1.pdbqt"),
    }.items():
        verify_file(path, lock["task"]["expected_input_sha256"][name])
    index = task["index"]
    if type(index) is not int or not 0 <= index < len(lock["take"]) <= batch_limit:
        raise ValueError("worker index outside locked batch")
    barrier = json.loads((root / "docking_started.json").read_text())
    if barrier["candidate_lock_sha256"] != task["candidate_lock_sha256"]:
        raise ValueError("worker lacks matching committed batch barrier")
    candidate = lock["take"][index]
    if not feasible_endpoint(candidate) or not candidate["oracle_eligible"]:
        raise ValueError("ineligible endpoint sent to docking worker")
    folder = root / "rows" / f"{index:02d}"
    result_path, started_path = folder / "result.json", folder / "started.json"
    if result_path.exists():
        result = unseal(result_path)
        if result["candidate_lock_sha256"] != task["candidate_lock_sha256"]:
            raise ValueError("row receipt belongs to another candidate lock")
        return result
    if started_path.exists():
        raise RuntimeError("saved row already attempted; no implicit re-docking")
    start = time.perf_counter()
    receipt = {
        "index": index,
        "smiles": candidate["smiles"],
        "candidate_lock_sha256": task["candidate_lock_sha256"],
        "started_at_utc": _stamp(),
    }
    publish_json(started_path, receipt)
    volume.commit()
    score = dock(candidate["smiles"], f"{task['run_id']}_{index}")
    if score is not None and not math.isfinite(score):
        raise ValueError("oracle returned nonfinite score")
    result = {
        **receipt,
        "ds": score,
        "completed_at_utc": _stamp(),
        "docking_seconds": time.perf_counter() - start,
    }
    seal(result_path, result)
    volume.commit()
    return result


def run_remote(
    task, repo_root, artifact_root, volume, validate_revision, parallel, *, qed_min, sa_max
):
    from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote

    def runner(actual_task, _prepare, _dock, output, **kwargs):
        contract = json.loads((repo_root / CONTRACT_PATH).read_text())
        if rdBase.rdkitVersion != contract["required_rdkit"]:
            raise ValueError("partial docking requires the frozen remote RDKit version")

        def dock_parallel(lock, digest, _output):
            return parallel(
                [
                    {**task, "candidate_lock_sha256": digest, "index": i}
                    for i in range(len(lock["take"]))
                ]
            )

        result = run_batch(
            actual_task,
            contract,
            artifact_root / contract["source"]["volume_path"],
            output,
            dock_parallel,
            qed_min=qed_min,
            sa_max=sa_max,
            **kwargs,
        )
        result["openbabel_version"] = subprocess.check_output(["obabel", "-V"], text=True).strip()
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
