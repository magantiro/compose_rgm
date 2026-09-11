"""Matched broad-reference versus donor/reference mixture on identical starts."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.donor_program import compile_transplant, pendant_cuts
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import propose_remote
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_donor_comparison"
APP_NAME = "compose-pmo-donor-comparison"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_DONOR_COMPARISON.md"
CHANNEL = {
    "version": "donor_reference_mixture_v1",
    "donor_probability": 0.5,
    "donor_distribution": "uniform_frozen_prescreen_bank",
    "cut_distribution": "uniform_oriented_single_bridge",
    "compiler_max_steps": 64,
    "compiler_max_expansions": 128,
}
CHANNEL_ID = identity(CHANNEL)


def prepare(root, prior_path, *, replicate=0, reuse_result=None):
    if type(replicate) is not int or replicate not in (0, 1):
        raise ValueError("donor comparison permits the initial run and one fresh-seed replication")
    prior = json.loads(prior_path.read_text())
    if (
        prior["schema_version"] != "donor_probe_result_v1"
        or prior["status"] != "complete_development"
    ):
        raise ValueError("donor comparison requires the complete probe receipt")
    batch_path = prior_path.with_name("batch.json")
    verify_file(batch_path, prior["batch_sha256"])
    batch = json.loads(batch_path.read_text())
    if len(batch["parents"]) != 100 or len(prior["parity"]) != 100:
        raise ValueError("initial donor bank census changed")
    parents, observed = [], {}
    for i, row in enumerate(batch["parents"]):
        if (
            prior["parity"][i]["parent"] != i
            or abs(prior["parity"][i]["score"] - row["historical_score"]) > 1e-12
        ):
            raise ValueError("parent lacks historical oracle parity")
        node = MolecularSearchState.start(
            decode_state(row["state"]), budget=64, root_id=f"prescreen/{i}"
        )
        parents.append(
            {
                "id": f"prescreen/{i}",
                "node": encode_search_state(node),
                "smiles": row["smiles"],
                "score": prior["parity"][i]["score"],
                "chain": [],
                "primitives": 0,
            }
        )
        observed[row["smiles"]] = prior["parity"][i]["score"]
    for row in prior["new_rows"]:
        if row["smiles"] in observed:
            raise ValueError("prior new endpoint is not new")
        observed[row["smiles"]] = row["score"]
    prepared = {
        "schema_version": "donor_comparison_prepared_v1",
        "parents": parents,
        "observed": observed,
        "donors": [row["state"] for row in batch["parents"]],
        "initialization": "first 16 original prescreen parents, no new probe winners injected",
        "historical_prescreen_calls": prior["historical_prescreen_reported_calls"],
        "prior_physical_validation_and_new_calls": prior["physical_calls_this_probe"],
        "historical_provenance_limit": prior["historical_provenance_limit"],
        "inputs": {
            str(prior_path): sha256_file(prior_path),
            str(batch_path): sha256_file(batch_path),
        },
    }
    publish_json(root / PREPARED, prepared)
    old = json.loads((root / "configs/pmo_learned_proposal.json").read_text())
    c = {k: old[k] for k in ("task", "expected_input_sha256", "runtime_contract_sha256")}
    c.update(
        schema_version="donor_comparison_contract_v1",
        seed=20260926 + replicate,
        replicate=replicate,
        particles=16,
        boundaries=6,
        arms=["baseline", "hybrid"],
        selection_modes={"baseline": "immediate", "hybrid": "immediate"},
        beta=10.0,
        draws_per_bundle=1,
        primitive_budget=64,
        include_region_replacement=True,
        initial_indices=list(range(16)),
        new_oracle_limit=192,
        proposal_ids={"baseline": None, "hybrid": CHANNEL_ID},
        channel=CHANNEL,
        law_caches=[],
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        prescreen=True,
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
            "expected_minutes": [4, 10],
            "reserved_cpu_hour_bound": 9.85,
            "reserved_usd_cap": 10,
        },
        authorization="2026-09-11 user authorizes controller and macro changes, prescreen reuse and efficient experiments",
        interpretation="prescreened exposed development; identical actual-score SMC but different proposal path laws; no exact original-reference Doob or kappa claim; not PMO AUC",
    )
    c["contract_sha256"] = identity(c)
    if reuse_result is not None:
        previous = json.loads(reuse_result.read_text())
        if (
            previous["status"] != "complete_development"
            or previous["configuration"]["schema_version"] != "donor_comparison_contract_v1"
            or previous["configuration"]["expected_input_sha256"] != c["expected_input_sha256"]
        ):
            raise ValueError("law reuse requires a complete compatible donor comparison")
        # Deterministic wrapper conversion for the existing exact-law cache API.
        # The raw molecular laws remain in their original worker directories.
        metadata = root / "diagnostics/pmo_donor_comparison/cache_metadata"
        origin = {"source_result_sha256": sha256_file(reuse_result), "run_id": previous["run_id"]}
        publish_json(
            metadata / "launch.json", {**origin, "image_revision": previous["image_revision"]}
        )
        publish_json(
            metadata / "runtime_gate.json", {**origin, "input_sha256": c["expected_input_sha256"]}
        )
        c["law_caches"] = [
            {
                "path": f"{KIND}/{previous['run_id']}",
                "launch_sha256": sha256_file(metadata / "launch.json"),
                "cache_paths": sorted(
                    f"workers/{w['worker_id']}"
                    for w in previous["workers"]
                    if w["law_work"]["fresh_laws"] > 0
                ),
                **origin,
            }
        ]
        c["contract_sha256"] = identity({k: v for k, v in c.items() if k != "contract_sha256"})
    if replicate:
        c["compute"]["max_workers"] = 14  # Leave 15 containers for the parallel T4 lane.
        c["compute"]["expected_minutes"] = [4, 12]
        c["replication_decision"] = (
            "Same starts, donors, proposal mixture, beta, particles and boundaries; only RNG seed changes. "
            "A second best-score gain over initialization and baseline supports replication, not external SOTA. "
            "A null or reversal leaves the first gain seed-dependent; do not increase this unchanged recipe. "
            "Timeout or failed execution remains inconclusive. Prior result enters only exact-law cache reuse."
        )
        c["contract_sha256"] = identity({k: v for k, v in c.items() if k != "contract_sha256"})
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("donor comparison contract hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["primitive_budget"],
        c["beta"],
        c["draws_per_bundle"],
    ) != (20260926 + c.get("replicate", 0), 16, 6, 192, 64, 10.0, 1):
        raise ValueError("donor comparison exceeds locked recipe")
    if type(c.get("replicate", 0)) is not int or c.get("replicate", 0) not in (0, 1):
        raise ValueError("undeclared donor comparison replication")
    if c["channel"] != CHANNEL or c["proposal_ids"] != {"baseline": None, "hybrid": CHANNEL_ID}:
        raise ValueError("donor proposal law changed")
    if c["arms"] != ["baseline", "hybrid"] or c["selection_modes"] != {
        "baseline": "immediate",
        "hybrid": "immediate",
    }:
        raise ValueError("both arms must share actual-score SMC selection")
    if c["initial_indices"] != list(range(16)) or not c["include_region_replacement"]:
        raise ValueError("initialization or broad reference support changed")
    if any(
        c[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
            "winner_donors",
        )
    ):
        raise ValueError("undeclared training/docking/winner injection")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


class ReferenceComponent:
    """Identity policy carrying the mixture's provenance, not a learned model."""

    def __init__(self, proposal_id=CHANNEL_ID):
        self.payload = {"model_sha256": proposal_id}

    def distribution(self, node, row):
        return row.reference, {
            "reference": row.reference.tolist(),
            "probabilities": row.reference.tolist(),
            "kl": 0.0,
        }

    def sample(self, hierarchy, node, rng):
        return hierarchy.sample_reference(node, rng), {
            "attempts": 1,
            "mixture_component": "reference",
        }


def donor_candidate(parent, result, candidate_id, donor_index, source_cut, donor_cut):
    if result["status"] != "compiled":
        return None
    origin = decode_search_state(parent["node"])
    lineage = origin.lineage
    for mark in result["actions"]:
        family, action = decode_action(mark)
        lineage = lineage.observe(family, action)
    graph = decode_state(result["states"][-1])
    node = MolecularSearchState(graph, lineage, 64 - result["primitive_steps"], origin.root_id)
    return {
        "id": candidate_id,
        "node": encode_search_state(node),
        "smiles": result["smiles"],
        "chain": parent["chain"] + [candidate_id],
        "primitives": parent["primitives"] + result["primitive_steps"],
        "primitive_count": result["primitive_steps"],
        "parent_smiles": parent["smiles"],
        "parent_score": parent["score"],
        "bundle": {
            "option": "donor_transplant",
            "bundle_id": identity([candidate_id, donor_index]),
            "r_release": result["released_fraction"],
            "donor_index": donor_index,
            "source_cut": asdict(source_cut),
            "donor_cut": asdict(donor_cut),
        },
        "structural_change": structural_displacement(origin.graph, graph, origin.lineage, lineage),
        "topology": topology(graph),
    }


def worker_remote(
    task,
    root,
    artifact_root,
    volume,
    validate,
    runtime_factory,
    *,
    contract_loader=None,
    run_session=None,
):
    c = (contract_loader or load_contract)(root)
    scoped_session = run_session or session
    rng = np.random.default_rng(
        np.random.SeedSequence([c["seed"], task["phase"], task["slot"], 776])
    )
    has_memory = bool(c.get("donor_memory_modes"))
    use_donor = (has_memory or task.get("proposal_id") == CHANNEL_ID) and rng.random() < CHANNEL[
        "donor_probability"
    ]
    if not use_donor:
        return propose_remote(
            task,
            root,
            artifact_root,
            volume,
            validate,
            runtime_factory,
            run_session=scoped_session,
            proposal_factory=lambda root, c, t: (
                ReferenceComponent(t["proposal_id"]) if t.get("proposal_id") else None
            ),
        )
    with scoped_session(task, root, artifact_root, volume, validate) as (c, store, progress):
        previous = store.read("complete")
        if previous is not None:
            return previous
        start = perf_counter()
        data = json.loads((root / c["prepared"]["path"]).read_text())
        parent = task["parent"]
        origin = decode_search_state(parent["node"])
        memory = None
        if has_memory:
            from compose_v4.control.donor_memory import validate_memory

            memory_path = (
                artifact_root
                / c["artifact_kind"]
                / task["run_id"]
                / f"memory/{task['donor_memory_id']}.json"
            )
            verify_file(memory_path, task["donor_memory_sha256"])
            memory = Store(memory_path.parent, lambda: None).read(memory_path.stem)
            validate_memory(memory, task["donor_memory_id"])
            i = int(rng.choice(len(memory["rows"]), p=memory["probabilities"]))
            donor = decode_state(memory["rows"][i]["state"])
        else:
            i = int(rng.integers(len(data["donors"])))
            donor = decode_state(data["donors"][i])
        left, right = pendant_cuts(origin.graph), pendant_cuts(donor)
        candidate = None
        progress.update(phase="donor_compile", donor=i)
        if left and right:
            a, b = left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]
            result = compile_transplant(origin.graph, donor, a, b)
            store.save("draws/00", result)
            candidate = donor_candidate(
                parent, result, f"workers/{task['worker_id']}/draws/00", i, a, b
            )
            if candidate is not None and memory is not None:
                candidate["bundle"].update(
                    donor_memory_id=memory["memory_id"],
                    donor_smiles=memory["rows"][i]["smiles"],
                    donor_probability=memory["probabilities"][i],
                )
        else:
            result = {"status": "no_pendant_cut"}
            store.save("draws/00", result)
        elapsed = perf_counter() - start
        receipt = {
            "worker_id": task["worker_id"],
            "phase": task["phase"],
            "slot": task["slot"],
            "candidates": [] if candidate is None else [candidate],
            "attempts": [
                {
                    "draw": 0,
                    "status": "complete" if candidate else result["status"],
                    "bundle": None if candidate is None else candidate["bundle"],
                    "proposal_seconds": elapsed,
                    "proposal_attempts": result.get("primitive_steps", 0),
                    "primitive_steps": result.get("primitive_steps", 0),
                    "what_allocation": None,
                    "component": "donor",
                }
            ],
            "replay_verified": True,
            "seconds": elapsed,
            "initialization_seconds": 0.0,
            "law_work": {"fresh_laws": 0, "law_seconds": 0.0},
            "executor_calls": result.get("attempts", 0) + result.get("primitive_steps", 0),
            "oracle_calls": 0,
            "proposal_policy_sha256": task.get("proposal_id", CHANNEL_ID),
            "code_revision": task["image_revision"]["commit"],
            "io_timings": dict(store.timings),
            "reference_probability_certified": False,
        }
        store.save("complete", receipt)
        return receipt


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
