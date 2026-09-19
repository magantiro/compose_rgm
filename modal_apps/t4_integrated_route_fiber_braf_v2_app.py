"""Preemption-safe BRAF repair with retained-core proposals and FiberControl.

Every round first publishes an immutable candidate and query lock to a durable Modal
volume.  Docking then runs once per locked molecule with no retry or replacement.  The
three cells and all proposal workers execute concurrently on single-CPU containers.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
OUTPUT = Path("/integrated_braf_v3")
CONTRACT = "configs/t4_integrated_route_fiber_braf_v3.json"
CHECKPOINT = "diagnostics/t4_integrated_route_fiber_braf_v1/route_expert_checkpoint.json"
PREFLIGHT = "diagnostics/t4_integrated_route_fiber_braf_v2/retained_core_preflight.json"
VOLUME_NAME = "compose-t4-integrated-route-fiber-braf-v3"
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .apt_install("openbabel", "curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl --fail -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        f"curl --fail -sSL -o /opt/dock/receptors/braf.pdbqt {MOOD}/receptors/braf.pdbqt",
    )
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(ROOT / CHECKPOINT, str(REMOTE / CHECKPOINT), copy=True)
    .add_local_file(ROOT / PREFLIGHT, str(REMOTE / PREFLIGHT), copy=True)
    .add_local_file(
        ROOT / "docs/GENMOL_T4_SEEDS.json",
        str(REMOTE / "docs/GENMOL_T4_SEEDS.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps/t4_integrated_route_fiber_braf_v2_app.py",
        str(REMOTE / "modal_apps/t4_integrated_route_fiber_braf_v2_app.py"),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("compose-t4-integrated-route-fiber-braf-v3")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 4096,
    "retries": 0,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 20,
}


def _load_contract() -> dict:
    from compose_v4.experiments.t4_matched_pilot import unseal

    return unseal(REMOTE / CONTRACT)


def _validate_task(task: dict, *, role: str) -> dict:
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file

    contract = _load_contract()
    if task.get("contract_payload_sha256") != identity(contract):
        raise ValueError("integrated controller contract identity mismatch")
    if role not in {"proposal", "dock", "cell", "driver", "status"}:
        raise ValueError(f"unknown integrated runtime role {role!r}")
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(REMOTE / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
    return contract


def _publish(path: Path, payload: dict) -> str:
    from compose_v4.control.docking_value import identity

    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    import json

    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    volume.commit()
    return envelope["payload_sha256"]


def _read(path: Path) -> dict:
    import json

    from compose_v4.control.docking_value import identity

    envelope = json.loads(path.read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"durable payload hash mismatch: {path}")
    return envelope["payload"]


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@app.function(**common, max_containers=36, timeout=1800)
def proposal_worker(task: dict) -> dict:
    """One expert on one measured parent; no task-oracle access."""

    import json
    import time

    import numpy as np

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.docking_value import identity
    from compose_v4.control.route_distilled_goal_expert import (
        RouteDistilledGoalExpert,
        propose_route_expert_candidates,
    )
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

    contract = _validate_task(task, role="proposal")
    expert = task["expert"]
    started = time.time()
    fiber = Fiber(task["original_seed"], contract["delta"], support=contract["support"])
    telemetry = {}
    if expert in {"shallow", "anchored_replacement"}:
        records = expand(
            task["parent"],
            task["parent_score"],
            fiber,
            np.random.default_rng(task["proposal_seed"]),
            draws=contract["proposal"][expert]["draws"],
            multi_region=True,
            horizon=contract["proposal"][expert]["horizon"],
            proposal_lane=expert,
        )
        telemetry = {"raw_draws": contract["proposal"][expert]["draws"]}
    elif expert == "route_complete_region":
        envelope = json.loads((REMOTE / CHECKPOINT).read_text())
        if identity(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError("route expert checkpoint payload hash mismatch")
        route = contract["proposal"][expert]
        model = RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])
        source = pad_molecular_graph(smiles_to_molecular_graph(task["parent"]), 48)
        proposed, telemetry = propose_route_expert_candidates(
            source,
            model,
            pool_size=route["pool_size"],
            realization_limit=route["realization_limit"],
            beam_width=route["beam_width"],
            expansion_width=route["expansion_width"],
            max_bindings_per_template=route["max_bindings_per_template"],
            maximum_expansions=route["maximum_expansions"],
        )
        records = []
        for row in proposed:
            properties = fiber.check(row["smiles"])
            if properties is None or properties["smiles"] == task["parent"]:
                continue
            records.append(
                {
                    **row,
                    **properties,
                    "parent": task["parent"],
                    "parent_score": task["parent_score"],
                    "delta": contract["delta"],
                }
            )
    elif expert == "retained_core_prune":
        from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
        from compose_v4.control.retained_core_pruning import (
            enumerate_retained_core_prunes,
        )

        retained = contract["proposal"][expert]
        source = pad_molecular_graph(smiles_to_molecular_graph(task["parent"]), 48)
        proposed = enumerate_retained_core_prunes(
            source,
            maximum_fragment_atoms=retained["maximum_fragment_atoms"],
            maximum_stages=retained["maximum_stages"],
            maximum_primitives=retained["maximum_primitives"],
            maximum_prefixes=retained["maximum_prefixes"],
        )
        records = []
        for row in proposed:
            properties = fiber.check(molecular_graph_to_smiles(row.product))
            if properties is None or properties["smiles"] == task["parent"]:
                continue
            deleted = sum(stage["deleted_atoms"] for stage in row.stages)
            records.append(
                {
                    **properties,
                    "parent": task["parent"],
                    "parent_score": task["parent_score"],
                    "families": ["retained_deletion"],
                    "program_families": ["retained_core_prune"],
                    "proposal_lane": expert,
                    "regions": len(row.stages),
                    "created": 0,
                    "deleted": deleted,
                    "delta": contract["delta"],
                    "primitive_edits": len(row.actions),
                    "retained_core_stages": list(row.stages),
                }
            )
        telemetry = {
            "exact_programs": len(proposed),
            "eligible_unique": len(records),
            "maximum_fragment_atoms": retained["maximum_fragment_atoms"],
            "maximum_stages": retained["maximum_stages"],
        }
    else:
        raise ValueError(f"unknown proposal expert {expert!r}")
    return {
        "expert": expert,
        "parent": task["parent"],
        "parent_score": task["parent_score"],
        "proposal_seed": task["proposal_seed"],
        "records": records,
        "telemetry": telemetry,
        "eligible_unique": len(records),
        "elapsed_seconds": time.time() - started,
    }


@app.function(**common, max_containers=24, timeout=720)
def dock_worker(task: dict) -> dict:
    """One locked docking request, charged once, with no retry."""

    import time

    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_docking_adapter import dock_t4

    contract = _validate_task(task, role="dock")
    receipt_path = OUTPUT / task["receipt_path"]
    volume.reload()
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") == "complete":
            return receipt["answer"]
        raise RuntimeError(
            f"unresolved docking reservation for {task['query_id']}; retry forbidden"
        )
    physical = {
        "qvina02": sha256_file(Path("/opt/dock/qvina02")),
        "receptor": sha256_file(Path("/opt/dock/receptors/braf.pdbqt")),
    }
    if physical != contract["evaluator_sha256"]:
        raise ValueError(f"docking evaluator identity mismatch: {physical}")
    _publish(
        receipt_path,
        {
            "schema_version": "t4_integrated_query_receipt_v1",
            "status": "reserved",
            "query_id": task["query_id"],
            "smiles": task["smiles"],
            "evaluator_sha256": physical,
        },
    )
    started = time.time()
    score = dock_t4(
        task["smiles"],
        task["query_id"],
        contract["docking_seed"],
        box={
            "coordinates": contract["docking_box"],
            "receptor": "/opt/dock/receptors/braf.pdbqt",
        },
    )
    answer = {
        "query_id": task["query_id"],
        "smiles": task["smiles"],
        "score": score,
        "failure": None if score is not None else "oracle_no_score",
        "elapsed_seconds": time.time() - started,
        "evaluator_sha256": physical,
    }
    _publish(
        receipt_path,
        {
            "schema_version": "t4_integrated_query_receipt_v1",
            "status": "complete",
            "query_id": task["query_id"],
            "smiles": task["smiles"],
            "answer": answer,
        },
    )
    return answer


@app.function(**common, max_containers=3, timeout=3600)
def run_phase(task: dict) -> dict:
    """Execute at most one root or scored round and durably checkpoint it."""

    import time

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
        EXPERTS,
        attach_features,
        durable_phase_action,
        expert_census,
        merge_expert_pools,
        select_batch,
    )

    contract = _validate_task(task, role="cell")
    cell = next(row for row in contract["cells"] if row["cell"] == task["cell"])
    folder = OUTPUT / task["run_id"] / task["cell"]
    final_path = folder / "result.json"
    checkpoint_path = folder / "checkpoint.json"
    volume.reload()
    if final_path.exists():
        return _read(final_path)

    checkpoint = _read(checkpoint_path) if checkpoint_path.exists() else None
    round_index = 0 if checkpoint is None else int(checkpoint["rounds_completed"]) + 1
    lock_path = folder / f"round_{round_index:03d}_lock.json"
    phase_result_path = folder / f"round_{round_index:03d}_result.json"
    if phase_result_path.exists():
        phase_result = _read(phase_result_path)
        if not checkpoint_path.exists() or _read(checkpoint_path) != phase_result["checkpoint"]:
            _publish(checkpoint_path, phase_result["checkpoint"])
        return {
            "status": "running",
            "cell": cell["cell"],
            "phase": round_index,
            "charged_calls": phase_result["checkpoint"]["charged_calls"],
            "best": min(phase_result["checkpoint"]["archive"].values()),
        }

    lock = _read(lock_path) if lock_path.exists() else None
    statuses = []
    if lock is not None:
        for query in lock["queries"]:
            receipt = folder / "receipts" / f"{query['query_id']}.json"
            statuses.append(_read(receipt).get("status", "invalid") if receipt.exists() else "missing")
    action = durable_phase_action(lock_exists=lock is not None, receipt_statuses=statuses)
    if action == "fail_closed":
        charged_before = int(lock.get("charged_before", 0)) if lock else 0
        charged = charged_before + len(lock.get("queries", [])) if lock else charged_before
        result = {
            "schema_version": "t4_integrated_route_fiber_result_v2",
            "status": "operational_fail_closed",
            "cell": cell["cell"],
            "charged_calls": charged,
            "unresolved_phase": round_index,
            "receipt_statuses": statuses,
            "failure": "immutable query lock lacks a complete receipt; no query was resubmitted",
            "prior_operational_waste": contract["prior_operational_waste"],
        }
        _publish(final_path, result)
        return result

    if round_index == 0:
        if action == "start":
            query_id = f"{task['run_id']}_{cell['cell']}_root"
            lock = {
                "schema_version": "t4_integrated_round_lock_v2",
                "contract_payload_sha256": task["contract_payload_sha256"],
                "cell": cell["cell"],
                "round": 0,
                "kind": "root",
                "charged_before": 0,
                "queries": [{"query_id": query_id, "smiles": cell["smiles"]}],
            }
            lock_hash = _publish(lock_path, lock)
            query = lock["queries"][0]
            answer = dock_worker.remote(
                {
                    **task,
                    **query,
                    "receipt_path": str(
                        Path(task["run_id"])
                        / cell["cell"]
                        / "receipts"
                        / f"{query_id}.json"
                    ),
                }
            )
        else:
            lock_hash = __import__("compose_v4.control.docking_value", fromlist=["identity"]).identity(lock)
            query = lock["queries"][0]
            answer = _read(folder / "receipts" / f"{query['query_id']}.json")["answer"]
        if answer["score"] is None:
            result = {
                "schema_version": "t4_integrated_route_fiber_result_v2",
                "status": "root_oracle_failure",
                "cell": cell["cell"],
                "charged_calls": 1,
                "root_lock_sha256": lock_hash,
                "root_result": answer,
            }
            _publish(final_path, result)
            return result
        rng = np.random.default_rng(cell["controller_seed"])
        checkpoint = {
            "schema_version": "t4_integrated_route_fiber_checkpoint_v2",
            "status": "running",
            "cell": cell["cell"],
            "contract_payload_sha256": task["contract_payload_sha256"],
            "charged_calls": 1,
            "budget_remaining": contract["charged_calls_per_cell"] - 1,
            "archive": {cell["smiles"]: float(answer["score"])},
            "features": [],
            "improvements": [],
            "history": [],
            "rounds": [],
            "rounds_completed": 0,
            "root_result": answer,
            "rng_state": _jsonable(rng.bit_generator.state),
        }
        _publish(
            phase_result_path,
            {
                "schema_version": "t4_integrated_phase_result_v1",
                "round": 0,
                "checkpoint": checkpoint,
            },
        )
        _publish(checkpoint_path, checkpoint)
        print(f"[{cell['cell']}] durable root calls=1 score={answer['score']:.2f}", flush=True)
        return {"status": "running", "cell": cell["cell"], "phase": 0, "charged_calls": 1, "best": answer["score"]}

    rng = np.random.default_rng()
    rng.bit_generator.state = checkpoint["rng_state"]
    state = SearchState(
        archive={key: float(value) for key, value in checkpoint["archive"].items()},
        budget=int(checkpoint["budget_remaining"]),
        rounds=int(checkpoint["rounds_completed"]),
        history=list(checkpoint["history"]),
    )
    value = ProgramValue(penalty=contract["value_penalty"])
    features = list(checkpoint["features"])
    improvements = list(checkpoint["improvements"])
    value.fit(features, improvements)
    rounds = list(checkpoint["rounds"])
    fiber = Fiber(cell["smiles"], contract["delta"], support=contract["support"])
    started = time.time()

    if action == "start":
        parents = state.parents(limit=contract["parents"], rng=rng, explore=contract["parent_explore"])
        requests = []
        for parent_index, parent in enumerate(parents):
            for expert_index, expert in enumerate(EXPERTS):
                requests.append(
                    {
                        **task,
                        "expert": expert,
                        "parent": parent,
                        "parent_score": state.archive[parent],
                        "original_seed": cell["smiles"],
                        "proposal_seed": int(cell["controller_seed"] + 1_000_003 * round_index + 10_007 * parent_index + 101 * expert_index),
                    }
                )
        worker_results = list(proposal_worker.map(requests, order_outputs=True, return_exceptions=True))
        pools = {expert: [] for expert in EXPERTS}
        worker_telemetry = []
        for request, answer in zip(requests, worker_results, strict=True):
            if isinstance(answer, Exception):
                worker_telemetry.append({"expert": request["expert"], "parent": request["parent"], "status": "failed", "error": repr(answer)})
                continue
            pools[answer["expert"]].extend(answer["records"])
            worker_telemetry.append({**answer, "records": None, "status": "complete"})
        merged = merge_expert_pools(pools)
        candidates = attach_features([row for row in merged if row["smiles"] not in state.archive], state, fiber)
        room = min(contract["batch"], state.budget)
        selected = select_batch(
            candidates,
            value,
            state,
            rng,
            round_index=round_index,
            batch=room,
            exploration=min(contract["exploration"], room),
            expert_floor_rounds=contract["expert_floor_rounds"],
        )
        if not selected:
            result = {
                "schema_version": "t4_integrated_route_fiber_result_v2",
                "status": "candidate_exhaustion",
                "cell": cell["cell"],
                "charged_calls": checkpoint["charged_calls"],
                "root_result": checkpoint["root_result"],
                "rounds": rounds,
                "archive": dict(sorted(state.archive.items())),
                "final_best": state.incumbent,
            }
            _publish(final_path, result)
            return result
        queries = [
            {
                "query_id": f"{task['run_id']}_{cell['cell']}_r{round_index:03d}_q{index:02d}",
                "smiles": row["smiles"],
                "selection_kind": row["selection_kind"],
                "proposal_experts": row["proposal_experts"],
                "parent": row["parent"],
                "parent_score": row["parent_score"],
            }
            for index, row in enumerate(selected)
        ]
        lock = {
            "schema_version": "t4_integrated_round_lock_v2",
            "contract_payload_sha256": task["contract_payload_sha256"],
            "cell": cell["cell"],
            "round": round_index,
            "charged_before": checkpoint["charged_calls"],
            "parents": parents,
            "pool_census": expert_census(candidates),
            "selected_census": expert_census(selected),
            "worker_telemetry": _jsonable(worker_telemetry),
            "candidate_pool": _jsonable(candidates),
            "selected_rows": _jsonable(selected),
            "queries": queries,
            "rng_state_after_selection": _jsonable(rng.bit_generator.state),
        }
        lock_hash = _publish(lock_path, lock)
        dock_tasks = [
            {
                **task,
                "query_id": query["query_id"],
                "smiles": query["smiles"],
                "receipt_path": str(Path(task["run_id"]) / cell["cell"] / "receipts" / f"{query['query_id']}.json"),
            }
            for query in queries
        ]
        answers = list(dock_worker.map(dock_tasks, order_outputs=True, return_exceptions=True))
    else:
        lock_hash = __import__("compose_v4.control.docking_value", fromlist=["identity"]).identity(lock)
        selected = lock["selected_rows"]
        candidates = lock["candidate_pool"]
        worker_telemetry = lock["worker_telemetry"]
        queries = lock["queries"]
        rng.bit_generator.state = lock["rng_state_after_selection"]
        answers = [_read(folder / "receipts" / f"{query['query_id']}.json")["answer"] for query in queries]

    if any(isinstance(answer, Exception) for answer in answers):
        statuses = []
        for query in queries:
            receipt = folder / "receipts" / f"{query['query_id']}.json"
            statuses.append(_read(receipt).get("status", "invalid") if receipt.exists() else "missing")
        result = {
            "schema_version": "t4_integrated_route_fiber_result_v2",
            "status": "operational_fail_closed",
            "cell": cell["cell"],
            "charged_calls": int(lock["charged_before"]) + len(queries),
            "unresolved_phase": round_index,
            "receipt_statuses": statuses,
            "failure": "one or more locked docking calls failed or remained unresolved; no retry",
        }
        _publish(final_path, result)
        return result

    previous = state.incumbent
    observed = []
    for row, answer in zip(selected, answers, strict=True):
        scored = {**row, **answer}
        observed.append(scored)
        if answer["score"] is None:
            continue
        features.append(np.asarray(row["features"], dtype=float).tolist())
        improvements.append(float(row["parent_score"] - answer["score"]))
        state.archive[row["smiles"]] = float(answer["score"])
    charged = int(checkpoint["charged_calls"]) + len(observed)
    state.budget = contract["charged_calls_per_cell"] - charged
    state.rounds = round_index
    improved = state.incumbent < previous
    state.history.append({"round": round_index, "improved": improved})
    value.fit(features, improvements)
    round_result = {
        "round": round_index,
        "charged_calls": charged,
        "charged_this_round": len(observed),
        "candidate_lock_payload_sha256": lock_hash,
        "pool_census": lock["pool_census"],
        "selected_census": lock["selected_census"],
        "selection_kind_census": {kind: sum(row["selection_kind"] == kind for row in selected) for kind in ("expert_floor", "model", "exploration")},
        "docked": _jsonable(observed),
        "round_best": min((row["score"] for row in observed if row.get("score") is not None), default=None),
        "best_so_far": state.incumbent,
        "improved": improved,
        "value_training_rows": len(improvements),
        "timing_seconds": {"phase_total": time.time() - started, "proposal_max": max((row.get("elapsed_seconds", 0.0) for row in worker_telemetry), default=0.0)},
        "rng_state_after": _jsonable(rng.bit_generator.state),
    }
    rounds.append(round_result)
    checkpoint = {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v2",
        "status": "running",
        "cell": cell["cell"],
        "contract_payload_sha256": task["contract_payload_sha256"],
        "charged_calls": charged,
        "budget_remaining": state.budget,
        "archive": dict(sorted(state.archive.items())),
        "features": features,
        "improvements": improvements,
        "history": state.history,
        "rounds": rounds,
        "rounds_completed": round_index,
        "root_result": checkpoint["root_result"],
        "rng_state": _jsonable(rng.bit_generator.state),
    }
    _publish(phase_result_path, {"schema_version": "t4_integrated_phase_result_v1", "round": round_index, "checkpoint": checkpoint, "round_result": round_result})
    _publish(checkpoint_path, checkpoint)
    print(f"[{cell['cell']}] durable round={round_index} calls={charged} best={state.incumbent:.2f} round_best={round_result['round_best']}", flush=True)

    if state.budget <= 0:
        result = {
            "schema_version": "t4_integrated_route_fiber_result_v2",
            "status": "complete_budget",
            "contract_payload_sha256": task["contract_payload_sha256"],
            "cell": cell["cell"],
            "root_result": checkpoint["root_result"],
            "charged_calls": charged,
            "rounds": rounds,
            "archive": dict(sorted(state.archive.items())),
            "final_best": state.incumbent,
            "best_smiles": min(state.archive, key=state.archive.get),
            "training_rows": len(improvements),
            "new_oracle_calls": charged,
            "prior_operational_waste": contract["prior_operational_waste"],
            "claim_boundary": contract["claim_boundary"],
        }
        _publish(final_path, result)
        return result
    return {"status": "running", "cell": cell["cell"], "phase": round_index, "charged_calls": charged, "best": state.incumbent}


@app.function(**common, max_containers=1, timeout=4000)
def drive(task: dict) -> dict:
    import json

    contract = _validate_task(task, role="driver")
    launch_path = OUTPUT / task["run_id"] / "launch.json"
    _publish(launch_path, task)
    tasks = [{**task, "cell": row["cell"]} for row in contract["cells"]]
    answers = list(run_phase.map(tasks, order_outputs=True, return_exceptions=True))
    records = []
    for row, answer in zip(contract["cells"], answers, strict=True):
        if isinstance(answer, Exception):
            records.append({"cell": row["cell"], "status": "failed", "error": repr(answer)})
        else:
            records.append(
                {
                    "cell": row["cell"],
                    "status": answer["status"],
                    "charged_calls": answer["charged_calls"],
                    "final_best": answer.get("final_best", answer.get("best")),
                    "phase": answer.get("phase"),
                }
            )
    summary = {
        "schema_version": "t4_integrated_route_fiber_phase_summary_v1",
        "run_id": task["run_id"],
        "contract_payload_sha256": task["contract_payload_sha256"],
        "records": records,
        "finished": all(row["status"] != "running" for row in records),
        "new_oracle_calls": sum(row.get("charged_calls", 0) for row in records),
        "claim_boundary": contract["claim_boundary"],
    }
    _publish(OUTPUT / task["run_id"] / "progress.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["finished"]:
        continuation = drive.spawn(task)
        summary["continuation_function_call_id"] = continuation.object_id
        _publish(OUTPUT / task["run_id"] / "progress.json", summary)
    else:
        _publish(OUTPUT / task["run_id"] / "summary.json", summary)
    return summary


@app.function(**common, max_containers=1, timeout=120)
def remote_status(task: dict) -> dict:
    import json

    _validate_task(task, role="status")
    folder = OUTPUT / task["run_id"]
    volume.reload()
    cells = []
    for cell in ("braf_0", "braf_1"):
        result_path = folder / cell / "result.json"
        checkpoint_path = folder / cell / "checkpoint.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())["payload"]
            cells.append(
                {
                    "cell": cell,
                    "status": payload["status"],
                    "calls": payload["charged_calls"],
                    "best": payload.get("final_best"),
                    "rounds": len(payload.get("rounds", [])),
                }
            )
        elif checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text())["payload"]
            cells.append(
                {
                    "cell": cell,
                    "status": payload["status"],
                    "calls": payload["charged_calls"],
                    "best": min(payload["archive"].values()),
                    "rounds": len(payload["rounds"]),
                }
            )
        else:
            cells.append({"cell": cell, "status": "not_started", "calls": 0})
    return {"run_id": task["run_id"], "cells": cells}


def _local_task() -> dict:
    import subprocess

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_matched_pilot import unseal

    contract = unseal(ROOT / CONTRACT)
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"local runtime input mismatch for {relative}: {actual}")
    runtime_paths = sorted({*contract["runtime_inputs_sha256"], CONTRACT})
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *runtime_paths], cwd=ROOT, check=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *runtime_paths],
        cwd=ROOT,
        text=True,
    )
    if untracked.strip():
        raise ValueError(f"untracked integrated runtime inputs: {untracked}")
    body = {
        "schema_version": "t4_integrated_route_fiber_launch_v2",
        "contract_payload_sha256": identity(contract),
        "contract_file_sha256": sha256_file(ROOT / CONTRACT),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "automatic_retries": 0,
        "charged_call_ceiling": contract["total_charged_call_ceiling"],
    }
    return {**body, "run_id": identity(body)}


@app.local_entrypoint()
def main(mode: str = "launch", run_id: str = "") -> None:
    import json

    from compose_v4.experiments.t4_matched_pilot import seal, unseal

    if mode == "launch":
        task = _local_task()
        call = drive.spawn(task)
        receipt = {
            "schema_version": "t4_integrated_route_fiber_launch_receipt_v1",
            "task": task,
            "function_call_id": call.object_id,
            "volume": VOLUME_NAME,
            "output_prefix": task["run_id"],
        }
        destination = (
            ROOT / "diagnostics/t4_integrated_route_fiber_braf_v3/launches" / (task["run_id"] + ".json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        seal(destination, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return
    if mode not in {"advance", "status"} or not run_id:
        raise ValueError("use mode=launch, or mode=advance/status with run_id")
    receipt_path = (
        ROOT / "diagnostics/t4_integrated_route_fiber_braf_v3/launches" / f"{run_id}.json"
    )
    receipt = unseal(receipt_path)
    task = receipt["task"]
    if mode == "advance":
        call = drive.spawn(task)
        print(json.dumps({"run_id": run_id, "function_call_id": call.object_id}, sort_keys=True))
        return
    print(json.dumps(remote_status.remote(task), indent=2))
