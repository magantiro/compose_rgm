"""Matched frozen versus evolving donor memory, broad reference proposals retained."""

from __future__ import annotations

import json
import tarfile
from functools import partial

from compose_v4.control.docking_value import identity
from compose_v4.control.donor_memory import RECIPE, build_memory, checked_parent
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_evolving_memory"
APP_NAME = "compose-pmo-evolving-memory"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_EVOLVING_MEMORY.md"


def prepare(root, prior_path, *, replicate=0, reuse_result=None):
    if type(replicate) is not int or replicate not in (0, 1):
        raise ValueError("memory comparison permits only one earned fresh-seed replication")
    store = Store(prior_path.parent, lambda: None)
    result, data = store.read("result"), store.read("prepared")
    if (
        result["status"] != "complete_development"
        or result["schema_version"] != "edit_replay_probe_result_v1"
    ):
        raise ValueError("evolving memory requires the completed contextual replay comparison")
    for key in ("configuration", "training", "candidate_lock"):
        verify_file(prior_path.with_name(f"{key}.json"), result[f"{key}_sha256"])
    observed, nodes = data["known"].copy(), {r["smiles"]: r for r in data["parents"]}
    for row in result["oracle_rows"]:
        if row["status"] != "complete" or row["smiles"] in observed:
            raise ValueError("historical oracle query is incomplete or repeated")
        observed[row["smiles"]] = row["score"]
    for arm in result["arms"].values():
        for row in arm["candidates"]:
            checked_parent(row)
            if observed[row["smiles"]] != row["score"]:
                raise ValueError("historical candidate has inconsistent score")
            nodes.setdefault(row["smiles"], row)
    parents = sorted(nodes.values(), key=lambda r: (-r["score"], r["smiles"]))[:16]
    if len(parents) != 16 or parents[0]["score"] != 0.649519052838329:
        raise ValueError("evolving memory must start from the actual current champion")
    memory = {}
    for i, state in enumerate(data["donors"]):
        key = canonical_state_key(decode_state(state))
        memory.setdefault(
            key,
            {
                "smiles": key,
                "score": observed[key],
                "state": state,
                "origin": f"original_donor/{i}",
            },
        )
    for row in parents:
        checked_parent(row)
        memory.setdefault(
            row["smiles"],
            {
                "smiles": row["smiles"],
                "score": row["score"],
                "state": row["node"]["graph"],
                "origin": row["id"],
            },
        )
    initial = [memory[s] for s in sorted(memory)]
    # Both laws must start identical, including exact representatives and weights.
    if build_memory(initial, {}, mode="fixed") != build_memory(
        initial, {r["smiles"]: r for r in parents}, mode="evolving"
    ):
        raise ValueError("matched initial donor memories differ")
    prepared = {
        "schema_version": "evolving_memory_prepared_v1",
        "parents": parents,
        "observed": observed,
        "initial_donor_memory": initial,
        "historical_prescreen_calls": result["historical_prescreen_calls"],
        "historical_development_physical_calls": result["historical_development_physical_calls"]
        + result["new_oracle_calls"],
        "inputs": {
            str(path): sha256_file(path)
            for path in (prior_path, prior_path.with_name("prepared.json"))
        },
        "initialization": "same top 16 latest exact canonical states; original donor bank plus these parents; no public winner",
    }
    previous_run = None
    if replicate:
        if reuse_result is None:
            raise ValueError("memory replication requires the positive completed first comparison")
        previous_run = json.loads(reuse_result.read_text())
        if (
            previous_run["status"] != "complete_development"
            or previous_run["configuration"]["schema_version"] != "evolving_memory_contract_v1"
            or previous_run["configuration"].get("replicate", 0) != 0
            or previous_run["arms"]["evolving"]["best"]
            <= max(previous_run["initial_metrics"]["best"], previous_run["arms"]["fixed"]["best"])
        ):
            raise ValueError("first memory comparison did not earn replication")
        with tarfile.open(reuse_result.with_name("source_snapshot.tar.gz"), "r:gz") as snapshot:
            old_bytes = snapshot.extractfile(
                previous_run["configuration"]["prepared"]["path"]
            ).read()
        import hashlib

        if (
            hashlib.sha256(old_bytes).hexdigest()
            != previous_run["configuration"]["prepared"]["sha256"]
        ):
            raise ValueError("first memory comparison prepared snapshot changed")
        old_data = json.loads(old_bytes)
        if any(prepared[k] != old_data[k] for k in ("parents", "initial_donor_memory", "observed")):
            raise ValueError(
                "replication changed initial states, donor memory or historical labels"
            )
        for row in previous_run["oracle_rows"]:
            if row["status"] != "complete" or row["smiles"] in prepared["observed"]:
                raise ValueError("replication cache has repeated or incomplete labels")
            prepared["observed"][row["smiles"]] = row["score"]
        prepared["historical_development_physical_calls"] += previous_run["new_oracle_calls"]
        prepared["replication_cache"] = {
            "path": str(reuse_result),
            "sha256": sha256_file(reuse_result),
            "role": "paid query lookup only; never an initial parent or donor",
        }
    elif reuse_result is not None:
        raise ValueError("reuse-result is reserved for the earned replication")
    publish_json(root / PREPARED, prepared)
    previous = json.loads((root / "configs/pmo_archive_branching.json").read_text())
    c = {
        k: previous[k]
        for k in ("task", "runtime_contract_sha256", "expected_input_sha256", "law_caches")
    }
    c.update(
        schema_version="evolving_memory_contract_v1",
        artifact_kind=KIND,
        authorization="active user controller goal authorizes bounded PMO implementation and experiments, up to 30 containers",
        seed=20261002 + replicate,
        replicate=replicate,
        particles=16,
        boundaries=6,
        draws_per_bundle=1,
        primitive_budget=64,
        beta=10.0,
        initial_indices=list(range(16)),
        arms=["fixed", "evolving"],
        donor_memory_modes={"fixed": "fixed", "evolving": "evolving"},
        memory_recipe=RECIPE,
        parent_selection_modes={"fixed": "archive", "evolving": "archive"},
        archive_exploration=0.2,
        new_oracle_limit=192,
        include_region_replacement=True,
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        winner_donors=False,
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 29,
            "driver_containers": 1,
            "worker_tasks_max": 192,
            "worker_timeout": 180,
            "driver_timeout": 900,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [4, 12],
            "reserved_cpu_hour_bound": 9.85,
            "reserved_usd_cap": 10,
        },
        interpretation="warm prescreened development; same broad reference and ranked donor mixture, only donor memory updates differ; no exact original-reference Doob or official PMO AUC claim",
    )
    if previous_run is not None:
        if previous_run["configuration"]["expected_input_sha256"] != c["expected_input_sha256"]:
            raise ValueError("replication reference-model identities changed")
        metadata = root / f"diagnostics/{KIND}/cache_metadata"
        origin = {
            "source_result_sha256": sha256_file(reuse_result),
            "run_id": previous_run["run_id"],
        }
        publish_json(
            metadata / "launch.json", {**origin, "image_revision": previous_run["image_revision"]}
        )
        publish_json(
            metadata / "runtime_gate.json", {**origin, "input_sha256": c["expected_input_sha256"]}
        )
        c["law_caches"].append(
            {
                "path": f"{KIND}/{previous_run['run_id']}",
                **origin,
                "launch_sha256": sha256_file(metadata / "launch.json"),
                "cache_paths": sorted(
                    f"workers/{w['worker_id']}"
                    for w in previous_run["workers"]
                    if w["law_work"]["fresh_laws"] > 0
                ),
            }
        )
        c["replication_decision"] = (
            "Same initial exact parents, donors, proposal distributions, parent rule, rounds and query ceiling; only seed changes. New labels are request-only cache entries. A repeated lead supports replication, not external SOTA; null/reversal leaves the first gain seed-dependent."
        )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return {
        "contract_sha256": c["contract_sha256"],
        "initial_donors": len(initial),
        "initial_best": parents[0]["score"],
        "historical_calls": prepared["historical_development_physical_calls"],
    }


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("evolving memory contract hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["draws_per_bundle"],
        c["new_oracle_limit"],
        c["primitive_budget"],
        c["archive_exploration"],
        c["beta"],
    ) != (20261002 + c.get("replicate", 0), 16, 6, 1, 192, 64, 0.2, 10.0):
        raise ValueError("evolving memory recipe changed")
    if type(c.get("replicate", 0)) is not int or c.get("replicate", 0) not in (0, 1):
        raise ValueError("undeclared memory replication")
    if (
        c["arms"] != ["fixed", "evolving"]
        or c["memory_recipe"] != RECIPE
        or c["donor_memory_modes"] != {"fixed": "fixed", "evolving": "evolving"}
        or c["parent_selection_modes"] != {"fixed": "archive", "evolving": "archive"}
        or c["initial_indices"] != list(range(16))
        or not c["include_region_replacement"]
    ):
        raise ValueError("matched memory comparison/support changed")
    if (
        c["artifact_kind"] != KIND
        or c["task"] != "perindopril_mpo"
        or any(
            c[k]
            for k in (
                "reference_training_authorized",
                "value_training_authorized",
                "docking_authorized",
                "winner_donors",
            )
        )
    ):
        raise ValueError("evolving memory scope changed")
    if (
        c["compute"]["max_workers"],
        c["compute"]["driver_containers"],
        c["compute"]["reserved_usd_cap"],
    ) != (29, 1, 10):
        raise ValueError("evolving memory compute authorization changed")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
