"""Dock a complete frozen repair census without surrogate or winner selection."""

from __future__ import annotations

import json
import subprocess
from time import perf_counter

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_partial_docking import dock_saved_row
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "t4_proposal_docking"
CONTRACT_PATH = f"configs/{KIND}.json"


def lock_batch(contract, artifact_root, task):
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("saved proposals require pinned RDKit")

    def read(asset):
        path = artifact_root / asset["path"]
        verify_file(path, asset["sha256"])
        return unseal(path)

    archive = read(contract["archive"])
    if archive["oracle_attempts"] != 51:
        raise ValueError("expected the complete 51-call source archive")
    known = {r["smiles"] for r in archive["archive"]}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(task["smiles"]))
    selected, census = {}, []
    for source in contract["sources"]:
        generated, scored = read(source["generation"]), read(source["scored"])
        if len(generated["products"]) != len(scored):
            raise ValueError("generation/scoring row count changed")
        eligible_count = 0
        for product, row in zip(generated["products"], scored, strict=True):
            if any(row[k] != v for k, v in product.items()):
                raise ValueError("scored row differs from saved exact product")
            if row["in_prior_archive"] != (row["smiles"] in known):
                raise ValueError("archive novelty identity changed")
            if not row["oracle_eligible"] or row["in_prior_archive"]:
                continue
            smiles = canonical_state_key(decode_state(row["state"]))
            if smiles != row["smiles"]:
                raise ValueError("exact molecular state differs from SMILES metadata")
            props = calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            )
            if not acceptable_endpoint({"smiles": smiles, **props}) or any(
                abs(props[k] - row[k]) > 1e-10 for k in props
            ):
                raise ValueError("frozen endpoint eligibility/properties changed")
            eligible_count += 1
            provenance = {
                "generation": source["generation"],
                "source_state": generated["source"],
                "source_smiles": generated["source_smiles"],
                "witnesses": row["witnesses"],
            }
            if smiles not in selected:
                selected[smiles] = {**row, "origins": []}
            selected[smiles]["origins"].append(provenance)
        census.append({"source": source, "products": len(scored), "new_eligible": eligible_count})
    if (
        len(selected) != contract["compute"]["oracle_call_limit"]
        or set(selected) != set(contract["authorized_smiles"])
        or len(selected) != 16
    ):
        raise ValueError("complete saved batch differs from authorized 16 identities")
    old = [r for r in archive["archive"] if r["ds"] is not None and acceptable_endpoint(r)]
    return {
        "schema_version": "t4_proposal_docking_lock_v1",
        "task": task,
        "required_rdkit": contract["required_rdkit"],
        "archive": contract["archive"],
        "take": [selected[k] for k in sorted(selected)],
        "census": census,
        "best_prior": min(old, key=lambda r: (r["ds"], r["smiles"])),
        "oracle_calls": 0,
        "new_generator_calls": 0,
        "locked_at_utc": _stamp(),
        "selection": "all new eligible products; no predicted score or winner selection",
    }


def run_locked_batch(lock, output, execute, *, commit, progress):
    path = output / "candidate_lock.json"
    if path.exists():
        saved = unseal(path)
        if {k: v for k, v in saved.items() if k != "locked_at_utc"} != {
            k: v for k, v in lock.items() if k != "locked_at_utc"
        }:
            raise ValueError("saved lock differs from requested batch")
        lock = saved
    else:
        seal(path, lock)
        commit()
    digest = sha256_file(path)
    barrier = output / "docking_started.json"
    if barrier.exists():
        if json.loads(barrier.read_text())["candidate_lock_sha256"] != digest:
            raise ValueError("batch barrier differs from lock")
    else:
        publish_json(
            barrier,
            {
                "candidate_lock_sha256": digest,
                "attempts": len(lock["take"]),
                "started_at_utc": _stamp(),
            },
        )
        commit()
    started, docked = perf_counter(), []
    for index, candidate in enumerate(lock["take"]):
        progress.update(phase="docking", completed=len(docked), total=len(lock["take"]))
        row = execute(index, digest)
        if (
            row["index"] != index
            or row["smiles"] != candidate["smiles"]
            or row["candidate_lock_sha256"] != digest
        ):
            raise ValueError("docking row identity differs from locked candidate")
        docked.append({**candidate, **row})
        values = [r["ds"] for r in docked if r["ds"] is not None]
        progress.update(completed=len(docked), best_batch_score=min(values, default=None))
        publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
        commit()
        print(
            f"docked {len(docked)}/{len(lock['take'])}: score={row['ds']} best={min(values, default=None)}",
            flush=True,
        )
    valid = [r for r in docked if r["ds"] is not None]
    return {
        "schema_version": "t4_proposal_docking_result_v1",
        "status": "complete",
        "candidate_lock_sha256": digest,
        "docked": docked,
        "new_oracle_attempts": len(docked),
        "oracle_failures": len(docked) - len(valid),
        "cumulative_source_and_diagnostic_calls": 51 + len(docked),
        "best_prior": lock["best_prior"],
        "best_batch": min(valid, key=lambda r: r["ds"], default=None),
        "best_so_far": min([lock["best_prior"], *valid], key=lambda r: r["ds"]),
        "docking_phase_seconds_this_invocation": perf_counter() - started,
        "optimizer_round_completed": False,
        "new_generator_calls": 0,
        "interpretation": "saved repair census, not adaptive search or IVG winner recovery",
    }


def run_remote(task, repo_root, artifact_root, volume, validate_revision, dock):
    from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        contract = json.loads((repo_root / CONTRACT_PATH).read_text())
        lock = lock_batch(contract, artifact_root, actual_task)

        def execute(index, digest):
            return dock_saved_row(
                {**task, "index": index, "candidate_lock_sha256": digest},
                repo_root,
                artifact_root,
                volume,
                validate_revision,
                dock,
                run_kind=KIND,
                batch_limit=16,
            )

        result = run_locked_batch(lock, output, execute, commit=commit, progress=progress)
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
