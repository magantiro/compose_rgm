"""A tested local endpoint channel in the retained broad population controller."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.local_endpoint_selector import predict, select
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.pmo_branch_policy import propose_remote
from compose_v4.experiments.pmo_donor_comparison import (
    ReferenceComponent,
)
from compose_v4.experiments.pmo_donor_comparison import (
    worker_remote as broad_worker,
)
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_repair_neighbors import enumerate_products
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "pmo_local_guidance"
APP_NAME = "compose-pmo-local-guidance"
APP, CONTRACT = f"modal_apps/{KIND}_app.py", f"configs/{KIND}.json"
PREPARED, PROTOCOL = f"diagnostics/{KIND}/prepared.json", "docs/PMO_LOCAL_GUIDANCE.md"


def neighbor_candidate(parent, row, candidate_id):
    origin = decode_search_state(parent["node"])
    mark = row["witnesses"][0]
    codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
    family, action = codec.decode_action(mark)
    lineage = origin.lineage.observe(family, action)
    graph = decode_state(row["state"])
    node = MolecularSearchState(graph, lineage, 63, origin.root_id)
    result = {
        "id": candidate_id,
        "node": encode_search_state(node),
        "smiles": row["smiles"],
        "chain": parent["chain"] + [candidate_id],
        "primitives": parent["primitives"] + 1,
        "primitive_count": 1,
        "parent_smiles": parent["smiles"],
        "parent_score": parent["score"],
        "bundle": {
            "option": "local_endpoint_selector",
            "bundle_id": identity([candidate_id, mark]),
            "r_release": 1.0,
            "scope": "all primitive sites; one realized edit, not a whole-molecule rewrite",
        },
        "structural_change": structural_displacement(origin.graph, graph, origin.lineage, lineage),
        "topology": topology(graph),
    }
    if "score" in row:
        result["score"] = row["score"]
    return result


def prepare(root: Path, prior_path: Path):
    cross = unseal(prior_path)
    directory = prior_path.parent
    cross_config = unseal(directory / "configuration.json")
    verify_file(directory / "configuration.json", cross["configuration_sha256"])
    report_path = root / "diagnostics/pmo_cross_parent_selection/report.json"
    report = json.loads(report_path.read_text())
    verify_file(prior_path, report["input_sha256"][str(prior_path)])
    if report["decision"] != "integrate_local_channel_then_matched_broad_comparison":
        raise ValueError("local channel requires the passing audited cross-parent test")
    paths = [Path(p) for p in cross_config["inputs"]]
    for p in paths:
        verify_file(p, cross_config["inputs"][str(p)])
    model_path = next(p for p in paths if p.name == "query_lock.json")
    prepared_path = next(p for p in paths if p.name == "prepared.json")
    origin_path = next(
        p for p in paths if p.name == "result.json" and "workers" in json.loads(p.read_text())
    )
    model, history, origin = (
        unseal(model_path),
        unseal(prepared_path),
        json.loads(origin_path.read_text()),
    )
    previous = unseal(prepared_path.with_name("result.json"))
    selection = unseal(model_path.with_name("result.json"))
    selection_config = unseal(model_path.with_name("configuration.json"))
    local_path = next(
        Path(p) for p in selection_config["input_sha256"] if p.endswith("/result.json")
    )
    local = unseal(local_path)
    observed = history["known"].copy()
    for result in (previous, cross):
        for q in result["oracle_rows"]:
            if q["smiles"] in observed or q["status"] != "complete":
                raise ValueError("local-guidance historical query is repeated or unresolved")
            observed[q["smiles"]] = q["score"]
    nodes = {}

    def retain(p):
        if p is None:
            return
        if (
            canonical_state_key(decode_search_state(p["node"]).graph) != p["smiles"]
            or observed[p["smiles"]] != p["score"]
        ):
            raise ValueError("starting archive lacks exact graph or paid score")
        nodes.setdefault(p["smiles"], p)

    for p in origin["initial_parents"] + history["roots"]:
        retain(p)
    for rd in origin["rounds"]:
        for arm in rd["arms"].values():
            for p in arm["proposals"]:
                retain(p)
    for group in previous["first"] + list(previous["second"].values()):
        for p in group:
            retain(p)
    for result in (local, selection):
        for row in result["rows"]:
            retain(
                neighbor_candidate(
                    origin["initial_parents"][0], row, "paid_local/" + identity(row["state"])
                )
            )
    query = unseal(directory / "query_lock.json")
    for i, batch in enumerate(query["batches"]):
        gen = unseal(directory / f"generation/{i}.json")
        for row in gen["products"]:
            if row["smiles"] in query["new"]:
                retain(
                    neighbor_candidate(
                        batch["parent"],
                        {**row, "score": observed[row["smiles"]]},
                        "paid_transfer/" + identity(row["state"]),
                    )
                )
    parents = sorted(nodes.values(), key=lambda p: (-p["score"], p["smiles"]))[:16]
    if len(parents) != 16 or parents[0]["score"] != cross["summary"]["best"]:
        raise ValueError("local-guidance initial champion or population changed")
    inputs = set(paths) | {
        prior_path,
        directory / "configuration.json",
        directory / "query_lock.json",
        report_path,
        local_path,
        model_path.with_name("result.json"),
        model_path.with_name("configuration.json"),
        prepared_path.with_name("result.json"),
    }
    inputs.update(directory / f"generation/{i}.json" for i in range(2))
    data = {
        "schema_version": "pmo_local_guidance_prepared_v1",
        "parents": parents,
        "observed": observed,
        "initial_donor_memory": history["donor_memory"]["rows"],
        "endpoint_model": {k: model[k] for k in ("model", "training_smiles", "all_smiles")},
        "input_sha256": {str(p): sha256_file(p) for p in sorted(inputs)},
        "historical_prescreen_calls": cross_config["historical_prescreen_calls"],
        "historical_development_physical_calls": cross_config[
            "historical_development_physical_calls"
        ]
        + cross["summary"]["new_calls"],
        "initialization": "same best sixteen own exact observed states, fixed original 116 donor memory; no public endpoint",
    }
    publish_json(root / PREPARED, data)
    c = dict(origin["configuration"])
    for k in ("contract_sha256", "replication_decision", "replicate"):
        c.pop(k, None)
    c.update(
        schema_version="pmo_local_guidance_contract_v1",
        artifact_kind=KIND,
        seed=20261008,
        boundaries=4,
        arms=["baseline", "guided"],
        new_oracle_limit=128,
        donor_memory_modes={"baseline": "fixed", "guided": "fixed"},
        parent_selection_modes={"baseline": "archive", "guided": "archive"},
        local_selector_arms=["guided"],
        endpoint_model_sha256=model["model"]["snapshot_sha256"],
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        interpretation="warm development comparison of fixed-memory broad controller versus broad controller plus frozen endpoint channel; not future-value/Doob or matched PMO AUC",
    )
    c["compute"] = {**c["compute"], "worker_tasks_max": 128, "reserved_cpu_hour_bound": 6.65}
    metadata = root / f"diagnostics/{KIND}/cache_metadata"
    cache_origin = {"source_result_sha256": sha256_file(origin_path), "run_id": origin["run_id"]}
    publish_json(
        metadata / "launch.json", {**cache_origin, "image_revision": origin["image_revision"]}
    )
    publish_json(
        metadata / "runtime_gate.json", {**cache_origin, "input_sha256": c["expected_input_sha256"]}
    )
    c["law_caches"].append(
        {
            "path": f"pmo_evolving_memory/{origin['run_id']}",
            "metadata_path": f"{KIND}/cache_metadata/{origin['run_id']}",
            **cache_origin,
            "launch_sha256": sha256_file(metadata / "launch.json"),
            "cache_paths": sorted(
                f"workers/{w['worker_id']}"
                for w in origin["workers"]
                if w["law_work"].get("fresh_laws", 0) > 0
            ),
        }
    )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return {
        "contract_sha256": c["contract_sha256"],
        "initial_best": parents[0]["score"],
        "donors": len(data["initial_donor_memory"]),
        "history_calls": data["historical_development_physical_calls"],
    }


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("local guidance contract hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["draws_per_bundle"],
        c["new_oracle_limit"],
        c["primitive_budget"],
    ) != (20261008, 16, 4, 1, 128, 64):
        raise ValueError("local guidance recipe exceeds the declared scope")
    if (
        c["arms"] != ["baseline", "guided"]
        or c["local_selector_arms"] != ["guided"]
        or c["donor_memory_modes"] != {"baseline": "fixed", "guided": "fixed"}
        or c["parent_selection_modes"] != {"baseline": "archive", "guided": "archive"}
    ):
        raise ValueError("local guidance changed its baseline or memory/selection rules")
    if (
        c["archive_exploration"],
        c["include_region_replacement"],
        c["task"],
        c["compute"]["max_workers"],
        c["compute"]["driver_containers"],
        c["compute"]["reserved_usd_cap"],
    ) != (0.2, True, "perindopril_mpo", 29, 1, 10):
        raise ValueError("local guidance changed chemistry, task or cost scope")
    if any(
        c[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
            "winner_donors",
        )
    ):
        raise ValueError("undeclared training, docking or public-winner injection")
    for k in ("prepared", "protocol"):
        verify_file(root / c[k]["path"], c[k]["sha256"])
    data = json.loads((root / c["prepared"]["path"]).read_text())
    if (
        data["endpoint_model"]["model"]["snapshot_sha256"] != c["endpoint_model_sha256"]
        or len(data["initial_donor_memory"]) != 116
    ):
        raise ValueError("frozen model or donor memory mismatch")
    return c


def local_worker(task, root, c, store, progress, runtime, law, started, initialization):
    data = json.loads((root / c["prepared"]["path"]).read_text())
    origin = decode_search_state(task["parent"]["node"])
    meter = ExecutorMeter(None)
    with meter.instrument():
        progress.update(phase="local_full_support_census")
        generation = store.read("local_generation")
        if generation is None:
            generation = enumerate_products(origin.graph, law, runtime["system"])
            store.save("local_generation", generation)
        if generation["source"] != encode_state(origin.graph):
            raise ValueError("local generation cache changed its exact parent")
        products = {p["smiles"]: p for p in generation["products"]}
        exclusions = set(task["local_exclusions"])
        if not set(data["observed"]) <= exclusions:
            raise ValueError("local selector lost historical query exclusions")
        pool = sorted(set(products) - exclusions)
        candidate, selection_seconds = None, 0.0
        if pool:
            progress.update(phase="local_endpoint_prediction", candidates=len(pool))
            before = perf_counter()
            prediction = predict(data["endpoint_model"], pool)
            rng = np.random.default_rng(
                np.random.SeedSequence([c["seed"], task["phase"], task["slot"], 777])
            )
            chosen = select(pool, prediction, rng)
            row = products[chosen]
            candidate = neighbor_candidate(
                task["parent"], row, f"workers/{task['worker_id']}/draws/00"
            )
            mark = row["witnesses"][0]
            codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
            actual = runtime["system"].apply(origin.graph, *codec.decode_action(mark))
            if encode_state(actual) != row["state"] or canonical_state_key(actual) != chosen:
                raise ValueError("selected local product failed exact primitive replay")
            store.save(
                "local_selection",
                {
                    "pool": pool,
                    "predictions": prediction.tolist(),
                    "chosen": chosen,
                    "model_sha256": c["endpoint_model_sha256"],
                    "exclusions": sorted(exclusions),
                },
            )
            store.save(
                "draws/00",
                {
                    "source": generation["source"],
                    "actions": [mark],
                    "states": [row["state"]],
                    "smiles": chosen,
                },
            )
            selection_seconds = perf_counter() - before
        store.save("executor", meter.attempts)
    elapsed = perf_counter() - started
    result = {
        "worker_id": task["worker_id"],
        "phase": task["phase"],
        "slot": task["slot"],
        "candidates": [] if candidate is None else [candidate],
        "attempts": [
            {
                "draw": 0,
                "status": "complete" if candidate else "no_novel_local_successor",
                "bundle": None if candidate is None else candidate["bundle"],
                "proposal_seconds": elapsed,
                "proposal_attempts": generation["counts"].get("positive_mass_marks", 0),
                "primitive_steps": int(candidate is not None),
                "what_allocation": None,
                "component": "local_endpoint",
                "available_products": len(products),
                "unqueried_products": len(pool),
                "selection_seconds": selection_seconds,
            }
        ],
        "replay_verified": True,
        "seconds": elapsed,
        "initialization_seconds": initialization,
        "law_work": law.counts,
        "executor_calls": meter.calls,
        "oracle_calls": 0,
        "proposal_policy_sha256": task["proposal_id"],
        "code_revision": task["image_revision"]["commit"],
        "io_timings": dict(store.timings),
        "reference_probability_certified": False,
    }
    store.save("complete", result)
    return result


def worker_remote(task, root, artifact_root, volume, validate, runtime_factory):
    if task.get("local_selector"):
        return propose_remote(
            task,
            root,
            artifact_root,
            volume,
            validate,
            runtime_factory,
            run_session=session,
            proposal_factory=lambda r, c, t: ReferenceComponent(t["proposal_id"]),
        )
    return broad_worker(
        task,
        root,
        artifact_root,
        volume,
        validate,
        runtime_factory,
        contract_loader=load_contract,
        run_session=session,
    )


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
