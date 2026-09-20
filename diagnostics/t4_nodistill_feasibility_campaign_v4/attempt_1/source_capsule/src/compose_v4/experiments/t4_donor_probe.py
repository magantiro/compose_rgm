"""One matched, score-blind T4 proposal batch with fixed docking controls."""

from __future__ import annotations

import json
from functools import partial
from time import perf_counter

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.pmo_branch_policy import _frozen_save, worker_identity
from compose_v4.experiments.pmo_donor_comparison import CHANNEL, CHANNEL_ID
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_macro_beam import exact_archive_graph
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal

KIND = "t4_donor_probe"
APP_NAME = "compose-t4-donor-probe"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/T4_DONOR_PROBE.md"
SOURCE_SHA = "025a5a5586d440b3bfd07b65877998d10e259bae86d2b28f410155b3f1b82728"


def prepare(root, archive_path, result_path):
    verify_file(result_path, SOURCE_SHA)
    result = json.loads(result_path.read_text())
    saved = unseal(archive_path)
    rows = saved["state"]["archive"]
    if result["status"] != "complete" or result["winner_used"] or len(rows) != 135:
        raise ValueError("T4 probe requires the original completed 134-attempt broad archive")
    if saved["summary"] != result["rounds"][-1]:
        raise ValueError("T4 archive does not match the completed source round")
    unique = {}
    for row in sorted(rows, key=lambda r: (r["ds"] is None, r["ds"] or 0, r["smiles"])):
        exact_archive_graph(row)
        unique.setdefault(row["smiles"], row)
    elite = [r for r in unique.values() if r["ds"] is not None]
    if elite[0]["smiles"] != result["best"]["smiles"] or elite[0]["ds"] != -11.0:
        raise ValueError("source T4 incumbent changed")
    chosen = [rows[0], *elite[:7]]
    parents = []
    for i, row in enumerate(chosen):
        node = MolecularSearchState.start(exact_archive_graph(row), budget=64, root_id=f"t4/{i}")
        parents.append(
            {
                "id": f"t4/{i}",
                "node": encode_search_state(node),
                "smiles": row["smiles"],
                "score": row["ds"],
                "chain": [],
                "primitives": 0,
            }
        )
    data = {
        "schema_version": "t4_donor_prepared_v1",
        "parents": parents,
        "donors": [r["state"] for r in elite[:100]],
        "prior_smiles": sorted(unique),
        "seed_smiles": rows[0]["smiles"],
        "anchors": [
            {"role": role, "smiles": row["smiles"], "state": row["state"]}
            for role, row in (("seed", rows[0]), ("incumbent", elite[0]))
        ],
        "historical_oracle_attempts": 134,
        "inputs": {
            str(archive_path): sha256_file(archive_path),
            str(result_path): sha256_file(result_path),
        },
    }
    publish_json(root / PREPARED, data)
    pmo = json.loads((root / "configs/pmo_donor_comparison.json").read_text())
    c = {
        "schema_version": "t4_donor_probe_contract_v1",
        "seed": 20260928,
        "target": "parp1",
        "delta": 0.4,
        "qed_min": 0.6,
        "sa_max": 4.0,
        "particles": 16,
        "boundaries": 1,
        "draws_per_bundle": 1,
        "primitive_budget": 64,
        "arms": ["baseline", "hybrid"],
        "proposal_ids": {"baseline": None, "hybrid": CHANNEL_ID},
        "channel": CHANNEL,
        "include_region_replacement": True,
        "initial_indices": list(range(8)) * 2,
        "docks_per_arm": 8,
        "anchor_repeats": 2,
        "new_oracle_limit": 20,
        "winner_donors": False,
        "training_authorized": False,
        "expected_input_sha256": pmo["expected_input_sha256"],
        "docking_input_sha256": {
            k: result["configuration"]["expected_input_sha256"][k] for k in ("receptor", "qvina02")
        },
        "runtime_contract_sha256": pmo["runtime_contract_sha256"],
        "law_caches": pmo["law_caches"],
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        "compute": {
            "max_workers": 14,
            "driver_containers": 1,
            "cpu": 1,
            "worker_timeout": 540,
            "driver_timeout": 1200,
            "retries": 0,
            "reserved_usd_cap": 5,
            "expected_minutes": [3, 10],
        },
        "interpretation": "one warm exposed T4 proposal batch, not a full optimizer or replicated benchmark; no surrogate or winner-based endpoint selection",
    }
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if c["contract_sha256"] != identity({k: v for k, v in c.items() if k != "contract_sha256"}):
        raise ValueError("T4 donor contract self-hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["docks_per_arm"],
        c["anchor_repeats"],
    ) != (20260928, 16, 1, 20, 8, 2):
        raise ValueError("T4 donor scope changed")
    if (c["target"], c["delta"], c["qed_min"], c["sa_max"]) != ("parp1", 0.4, 0.6, 4.0) or c[
        "channel"
    ] != CHANNEL:
        raise ValueError("T4 chemistry or benchmark recipe changed")
    if c["training_authorized"] or c["winner_donors"]:
        raise ValueError("no model training or winner injection in T4 transfer")
    if (
        c["arms"] != ["baseline", "hybrid"]
        or c["proposal_ids"] != {"baseline": None, "hybrid": CHANNEL_ID}
        or c["initial_indices"] != list(range(8)) * 2
        or c["primitive_budget"] != 64
        or c["draws_per_bundle"] != 1
        or not c["include_region_replacement"]
        or c["compute"]["max_workers"] + c["compute"]["driver_containers"] != 15
    ):
        raise ValueError("T4 proposal support or compute allocation changed")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)


def candidate_lock(c, data, results):
    """Deterministic bundle-first allocation using eligibility, never docking predictions."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(data["seed_smiles"]))

    def properties(row):
        exact_archive_graph(
            {"smiles": row["smiles"], "state": row.get("state", row.get("node", {}).get("graph"))}
        )
        p = calculate_properties(
            Chem.MolFromSmiles(row["smiles"]),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=c["delta"],
            qed_min=c["qed_min"],
            sa_max=c["sa_max"],
        )
        return {**row, **p, "oracle_eligible": acceptable_endpoint({**row, **p})}

    pool, chosen = [], {a: [] for a in c["arms"]}
    for arm in c["arms"]:
        seen = set(data["prior_smiles"])
        workers = sorted(
            (r for r in results if r.get("proposal_policy_sha256") == c["proposal_ids"][arm]),
            key=lambda r: r["slot"],
        )
        for worker in workers:
            for row in worker["candidates"]:
                row = {**properties(row), "arm": arm, "slot": worker["slot"]}
                row["already_observed_or_duplicate"] = row["smiles"] in seen
                pool.append(row)
                if (
                    row["oracle_eligible"]
                    and row["smiles"] not in seen
                    and len(chosen[arm]) < c["docks_per_arm"]
                ):
                    chosen[arm].append(row)
                seen.add(row["smiles"])
    grouped = {}
    for arm in c["arms"]:
        for row in chosen[arm]:
            if row["smiles"] not in grouped:
                grouped[row["smiles"]] = {**row, "arms": [], "origins": []}
            grouped[row["smiles"]]["arms"].append(arm)
            grouped[row["smiles"]]["origins"].append(
                {k: row[k] for k in ("arm", "slot", "id", "bundle")}
            )
    take = [grouped[s] for s in sorted(grouped)]
    for anchor in data["anchors"]:
        for repeat in range(c["anchor_repeats"]):
            row = properties(anchor)
            if not row["oracle_eligible"]:
                raise ValueError("fixed docking control is not eligible")
            take.append({**row, "repeat": repeat, "arms": []})
    if len(take) > c["new_oracle_limit"]:
        raise ValueError("T4 transfer exceeds docking cap")
    return {
        "schema_version": "t4_donor_candidate_lock_v1",
        "required_rdkit": "2024.03.5",
        "task": {"expected_input_sha256": c["docking_input_sha256"]},
        "take": take,
        "pool": pool,
        "selected_counts": {a: len(v) for a, v in chosen.items()},
    }


def driver_remote(task, root, artifact_root, volume, validate, parallel):
    with session(task, root, artifact_root, volume, validate) as (c, store, progress):
        final_path = store.output / "result.json"
        if final_path.exists():
            previous = json.loads(final_path.read_text())
            if previous["run_id"] != task["run_id"] or previous["configuration"] != c:
                raise ValueError("saved T4 result belongs to another run")
            return previous
        start = perf_counter()
        data = json.loads((root / PREPARED).read_text())
        results = store.read("proposals")
        if results is None:
            tasks = []
            for arm in c["arms"]:
                for slot, index in enumerate(c["initial_indices"]):
                    parent, policy = data["parents"][index], c["proposal_ids"][arm]
                    tasks.append(
                        {
                            **task,
                            "phase": 1,
                            "slot": slot,
                            "parent": parent,
                            "proposal_id": policy,
                            "worker_id": worker_identity(1, slot, parent, policy),
                        }
                    )
            _frozen_save(store, "parents", tasks)
            expected = {t["worker_id"]: t for t in tasks}
            results = []
            progress.update(phase="proposals", workers_total=len(tasks))
            for row in parallel(tasks):
                if (
                    not row["replay_verified"]
                    or row["worker_id"] not in expected
                    or row.get("proposal_policy_sha256")
                    != expected[row["worker_id"]].get("proposal_id")
                ):
                    raise ValueError("unverified T4 proposal")
                results.append(row)
                progress.update(workers_complete=len(results))
                print(f"[T4 transfer] proposals={len(results)}/{len(tasks)}", flush=True)
            if len({r["worker_id"] for r in results}) != len(tasks):
                raise ValueError("T4 proposal census differs")
            store.save("proposals", results, durable=True)
        lock = candidate_lock(c, data, results)
        _frozen_save(store, "candidate_lock", lock)
        digest = sha256_file(store.output / "candidate_lock.json")
        barrier = store.output / "docking_started.json"
        if not barrier.exists():
            publish_json(
                barrier,
                {
                    "candidate_lock_sha256": digest,
                    "attempts": len(lock["take"]),
                    "started_at_utc": _stamp(),
                },
            )
            store.flush(force=True)
        docking, indices = [], set()
        progress.update(phase="docking", locked_candidates=len(lock["take"]))
        tasks = [
            {
                **task,
                "stage": "dock",
                "index": i,
                "candidate_lock_sha256": digest,
                "app_sha256": sha256_file(root / "modal_apps/genmol_t4_opt_app.py"),
            }
            for i in range(len(lock["take"]))
        ]
        for row in parallel(tasks):
            index = row["index"]
            if (
                type(index) is not int
                or not 0 <= index < len(tasks)
                or index in indices
                or row["candidate_lock_sha256"] != digest
                or row["smiles"] != lock["take"][index]["smiles"]
            ):
                raise ValueError("docking row does not match T4 lock")
            indices.add(index)
            docking.append({**lock["take"][index], **row})
            progress.update(
                docked=len(docking),
                best_batch=min((r["ds"] for r in docking if r["ds"] is not None), default=None),
            )
            print(f"[T4 transfer] docked={len(docking)}/{len(tasks)} score={row['ds']}", flush=True)
        if len({r["index"] for r in docking}) != len(tasks):
            raise ValueError("incomplete docking census")
        result = {
            "schema_version": "t4_donor_probe_result_v1",
            "status": "complete_development",
            "configuration": c,
            "run_id": task["run_id"],
            "image_revision": task["image_revision"],
            "workers": results,
            "docked": sorted(docking, key=lambda r: r["index"]),
            "selected_counts": lock["selected_counts"],
            "new_oracle_calls": len(docking),
            "historical_oracle_attempts": 134,
            "seconds": perf_counter() - start,
            "software": software(),
            "rdkit": rdBase.rdkitVersion,
            "finished_at": _stamp(),
            "interpretation": c["interpretation"],
        }
        publish_json(store.output / "result.json", result)
        store.flush(force=True)
        return result
