"""Online learned complete-plan proposals in the broad population controller."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from functools import partial
from time import perf_counter

import numpy as np
import torch

from compose_v4.control.docking_value import identity
from compose_v4.control.donor_memory import validate_memory
from compose_v4.control.donor_program import compile_transplant, pendant_cuts, transplant_plan
from compose_v4.control.edit_replay import cut_from_payload
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.control.plan_policy import RECIPE, PlanPolicy, check_snapshot, fit_endpoint
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_donor_comparison import (
    donor_candidate,
)
from compose_v4.experiments.pmo_donor_comparison import (
    worker_remote as broad_worker,
)
from compose_v4.experiments.pmo_edit_chooser import RUNS, read_input
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_plan_policy"
APP_NAME = "compose-pmo-plan-policy"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_PLAN_POLICY.md"


def _paid_inputs(root):
    inputs = {}
    base_path = root / "diagnostics/pmo_local_guidance/prepared.json"
    base = read_input(base_path, inputs)
    observed = dict(base["observed"])
    nodes = {p["smiles"]: p for p in base["parents"]}
    for path in RUNS:
        run = read_input(path, inputs)
        if run["status"] != "complete_development":
            raise ValueError("plan policy requires complete prior runs")
        for row in run["oracle_rows"]:
            if row["status"] != "complete" or (
                row["smiles"] in observed and observed[row["smiles"]] != row["score"]
            ):
                raise ValueError("inconsistent paid historical label")
            observed[row["smiles"]] = row["score"]
        for round_ in run["rounds"]:
            for arm in round_["arms"].values():
                for node in arm["proposals"]:
                    if node is not None:
                        nodes.setdefault(node["smiles"], node)
    parents = sorted(nodes.values(), key=lambda p: (-p["score"], p["smiles"]))[:16]
    for parent in parents:
        if (
            canonical_state_key(decode_search_state(parent["node"]).graph) != parent["smiles"]
            or observed[parent["smiles"]] != parent["score"]
        ):
            raise ValueError("initial parent identity or score changed")
    if len(observed) != 2217 or parents[0]["score"] != 0.6835298930947339:
        raise ValueError("plan comparison lost paid history or its champion")
    return base, observed, parents, inputs


def prepare(root):
    started = perf_counter()
    base, observed, parents, inputs = _paid_inputs(root)
    path = root / PREPARED
    if path.exists():
        old = json.loads(path.read_text())
        if old["input_sha256"] != inputs or old["encoder"]["recipe"] != RECIPE:
            raise ValueError("existing plan preparation has incompatible inputs")
        load_contract(root)
        return {"reused": True, "initial_best": parents[0]["score"]}
    encoder, policy, training = fit_endpoint(observed)
    prepared = {
        "schema_version": "plan_policy_prepared_v1",
        "parents": parents,
        "observed": observed,
        "initial_donor_memory": base["initial_donor_memory"],
        "historical_prescreen_calls": base["historical_prescreen_calls"],
        "historical_development_physical_calls": 2217,
        "input_sha256": inputs,
        "encoder": encoder,
        "initial_policy": policy,
        "initialization": "own paid history and exact states; no public winners",
    }
    publish_json(path, prepared)
    training.update(
        seconds=perf_counter() - started,
        new_oracle_calls=0,
        input_sha256=inputs,
        encoder_id=encoder["encoder_id"],
        software={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        code_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        implementation={
            p: sha256_file(root / p)
            for p in (
                "src/compose_v4/control/plan_policy.py",
                "src/compose_v4/experiments/pmo_plan_policy.py",
            )
        },
    )
    publish_json(root / f"diagnostics/{KIND}/training.json", training)
    _write_contract(root, encoder["encoder_id"], path)
    return {
        "initial_best": parents[0]["score"],
        "training_seconds": training["seconds"],
        "calibration_rmse": training["calibration_rmse"],
        "constant_rmse": training["constant_rmse"],
        "new_oracle_limit": 128,
    }


def _write_contract(root, encoder_id, prepared_path):
    old = json.loads((root / "configs/pmo_local_guidance_replication.json").read_text())
    contract = {
        k: old[k]
        for k in ("task", "runtime_contract_sha256", "expected_input_sha256", "law_caches")
    }
    contract.update(
        schema_version="plan_policy_contract_v1",
        artifact_kind=KIND,
        authorization="2026-09-11 user approved trying the learned search policy and continuing",
        seed=RECIPE["seed"],
        particles=16,
        boundaries=4,
        draws_per_bundle=1,
        new_oracle_limit=128,
        primitive_budget=64,
        initial_indices=list(range(16)),
        arms=["frozen", "learning"],
        donor_memory_modes={"frozen": "fixed", "learning": "fixed"},
        parent_selection_modes={"frozen": "archive", "learning": "archive"},
        archive_exploration=0.2,
        include_region_replacement=True,
        plan_policy_arms=["frozen", "learning"],
        plan_policy_recipe=RECIPE,
        encoder_id=encoder_id,
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        winner_donors=False,
        prepared={"path": PREPARED, "sha256": sha256_file(prepared_path)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 29,
            "driver_containers": 1,
            "worker_tasks_max": 128,
            "worker_timeout": 180,
            "driver_timeout": 900,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [5, 15],
            "reserved_cpu_hour_bound": 6.65,
            "reserved_usd_cap": 10,
        },
        interpretation=(
            "warm proposal-policy component test with broad reference retained; "
            "not original-R_theta Doob, future value or official PMO AUC"
        ),
    )
    contract["contract_sha256"] = identity(contract)
    publish_json(root / CONTRACT, contract)


def load_contract(root):
    contract = json.loads((root / CONTRACT).read_text())
    if (
        identity({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
    ):
        raise ValueError("plan-policy contract hash mismatch")
    expected = (RECIPE["seed"], 16, 4, 128, 64, ["frozen", "learning"])
    actual = tuple(
        contract[k]
        for k in ("seed", "particles", "boundaries", "new_oracle_limit", "primitive_budget", "arms")
    )
    if actual != expected:
        raise ValueError("plan-policy comparison exceeds authorized scope")
    if (
        contract["plan_policy_recipe"] != RECIPE
        or contract["task"] != "perindopril_mpo"
        or not contract["include_region_replacement"]
        or contract["donor_memory_modes"] != {"frozen": "fixed", "learning": "fixed"}
        or contract["parent_selection_modes"] != {"frozen": "archive", "learning": "archive"}
    ):
        raise ValueError("plan-policy recipe, task, support, or baseline changed")
    if any(
        contract[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
            "winner_donors",
        )
    ):
        raise ValueError("unauthorized scientific contract change")
    for key in ("prepared", "protocol"):
        verify_file(root / contract[key]["path"], contract[key]["sha256"])
    return contract


def plan_identity(memory_id, encoder_id, policy_id):
    return identity(
        {
            "memory": memory_id,
            "encoder": encoder_id,
            "policy": policy_id,
            "recipe": RECIPE,
        }
    )


def validate_worker(task, contract):
    if task.get("plan_policy_id") is None or task.get("donor_memory_id") is None:
        raise ValueError("plan-policy worker lacks its immutable policy or memory")
    expected = plan_identity(
        task["donor_memory_id"], contract["encoder_id"], task["plan_policy_id"]
    )
    if task.get("proposal_id") != expected:
        raise ValueError("plan-policy worker identity mismatch")


def round_context(data, state, store, arm, memory_context):
    snapshot = state[arm].get("plan_policy", data["initial_policy"])
    check_snapshot(snapshot)
    path = f"policies/{snapshot['policy_id']}"
    from compose_v4.experiments.pmo_branch_policy import _frozen_save

    _frozen_save(store, path, snapshot)
    memory_id = memory_context["donor_memory_id"]
    return {
        "plan_policy_id": snapshot["policy_id"],
        "plan_policy_sha256": sha256_file(store.output / f"{path}.json"),
        "proposal_id": plan_identity(
            memory_id, data["encoder"]["encoder_id"], snapshot["policy_id"]
        ),
    }


def update_round(data, previous, proposed, results, assignments, arm):
    before = previous.get("plan_policy", data["initial_policy"])
    if arm == "frozen":
        return before, {"mode": "fixed", "changed": False, "scored_decisions": 0}
    decisions = []
    for child, key in zip(proposed, assignments, strict=True):
        if child is None or key is None:
            continue
        row = results[key].get("plan_decision")
        if row is not None:
            decisions.append({**row, "gain": child["score"] - child["parent_score"]})
    started = perf_counter()
    snapshot, audit = PlanPolicy(data["encoder"], before).update(decisions)
    return snapshot, {
        **audit,
        "seconds": perf_counter() - started,
        "before": before["policy_id"],
        "after": snapshot["policy_id"],
    }


def _plan_pool(source, memory, rng):
    source_cuts = pendant_cuts(source)
    donors, rejected, plans = {}, Counter(), {}
    for _ in range(RECIPE["plans_per_draw"]):
        index = int(rng.choice(len(memory["rows"]), p=memory["probabilities"]))
        if index not in donors:
            donor = decode_state(memory["rows"][index]["state"])
            donors[index] = donor, pendant_cuts(donor)
        donor, donor_cuts = donors[index]
        if not source_cuts or not donor_cuts:
            rejected["no_cut"] += 1
            continue
        left = source_cuts[int(rng.integers(len(source_cuts)))]
        right = donor_cuts[int(rng.integers(len(donor_cuts)))]
        plan = transplant_plan(source, donor, left, right)
        if plan["status"] != "planned":
            rejected[plan["status"]] += 1
            continue
        try:
            smiles = canonical_state_key(plan["target"])
        except (ValueError, RuntimeError) as error:
            rejected[f"invalid_plan/{type(error).__name__}"] += 1
            continue
        if smiles == canonical_state_key(source):
            rejected["self_plan"] += 1
        elif smiles in plans:
            rejected["canonical_duplicate"] += 1
        else:
            plans[smiles] = {
                "donor_index": index,
                "source_cut": asdict(left),
                "donor_cut": asdict(right),
                "release": plan["released_fraction"],
            }
    return plans, donors, rejected


def worker_remote(task, root, artifact_root, volume, validate, runtime_factory):
    contract = load_contract(root)
    rng = np.random.default_rng(
        np.random.SeedSequence([contract["seed"], task["phase"], task["slot"], 776])
    )
    # Use the same first coin as the existing donor/reference mixture.
    if rng.random() >= 0.5:
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
    with session(task, root, artifact_root, volume, validate) as (_, store, progress):
        done = store.read("complete")
        if done is not None:
            return done
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        started = perf_counter()
        data = json.loads((root / PREPARED).read_text())
        run_dir = artifact_root / KIND / task["run_id"]
        memory_path = run_dir / f"memory/{task['donor_memory_id']}.json"
        verify_file(memory_path, task["donor_memory_sha256"])
        memory = Store(memory_path.parent, lambda: None).read(memory_path.stem)
        validate_memory(memory, task["donor_memory_id"])
        policy_path = run_dir / f"policies/{task['plan_policy_id']}.json"
        verify_file(policy_path, task["plan_policy_sha256"])
        snapshot = Store(policy_path.parent, lambda: None).read(policy_path.stem)
        policy = PlanPolicy(data["encoder"], snapshot)
        parent = task["parent"]
        source = decode_search_state(parent["node"]).graph
        progress.update(phase="complete_plan_proposals")
        plans, donors, rejected = _plan_pool(source, memory, rng)
        candidate, decision = None, None
        result = {"status": "no_supported_plan"}
        if plans:
            products = sorted(plans)
            release = [plans[smiles]["release"] for smiles in products]
            probabilities = policy.distribution(parent["smiles"], products, release)
            selected = int(rng.choice(len(products), p=probabilities))
            chosen = plans[products[selected]]
            decision = {
                "policy_id": snapshot["policy_id"],
                "source": parent["smiles"],
                "products": products,
                "release": release,
                "probabilities": probabilities.tolist(),
                "selected": selected,
            }
            decision["decision_id"] = identity([task["worker_id"], decision])
            store.save(
                "plan_lock",
                {"decision": decision, "plans": plans, "rejected": dict(rejected)},
            )
            progress.update(phase="selected_plan_compile", plans=len(products))
            left = cut_from_payload(chosen["source_cut"])
            right = cut_from_payload(chosen["donor_cut"])
            result = compile_transplant(source, donors[chosen["donor_index"]][0], left, right)
            store.save("draws/00", result)
            candidate = donor_candidate(
                parent,
                result,
                f"workers/{task['worker_id']}/draws/00",
                chosen["donor_index"],
                left,
                right,
            )
            if candidate is not None:
                if candidate["smiles"] != products[selected]:
                    raise ValueError("compiled endpoint differs from selected plan")
                candidate["bundle"].update(
                    plan_policy_id=snapshot["policy_id"],
                    plan_decision_id=decision["decision_id"],
                )
        elapsed = perf_counter() - started
        receipt = {
            "worker_id": task["worker_id"],
            "phase": task["phase"],
            "slot": task["slot"],
            "candidates": [] if candidate is None else [candidate],
            "plan_decision": decision,
            "attempts": [
                {
                    "draw": 0,
                    "status": "complete" if candidate else result["status"],
                    "bundle": None if candidate is None else candidate["bundle"],
                    "proposal_seconds": elapsed,
                    "proposal_attempts": RECIPE["plans_per_draw"],
                    "primitive_steps": result.get("primitive_steps", 0),
                    "what_allocation": None,
                    "component": "learned_plan",
                    "unique_plans": len(plans),
                    "rejected": dict(rejected),
                }
            ],
            "replay_verified": True,
            "seconds": elapsed,
            "initialization_seconds": 0.0,
            "law_work": {"fresh_laws": 0, "law_seconds": 0.0},
            "executor_calls": result.get("attempts", 0) + result.get("primitive_steps", 0),
            "oracle_calls": 0,
            "proposal_policy_sha256": task["proposal_id"],
            "code_revision": task["image_revision"]["commit"],
            "io_timings": dict(store.timings),
            "reference_probability_certified": False,
        }
        store.save("complete", receipt)
        return receipt


session = partial(
    particle_session,
    contract_loader=load_contract,
    app_path=APP,
    kind=KIND,
    proposal_validator=validate_worker,
)
driver_remote = partial(
    particle_driver,
    run_session=session,
    round_context_factory=round_context,
    round_update_factory=update_round,
)
