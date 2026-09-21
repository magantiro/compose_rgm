"""Protonation-aware rescue arm for 5HT1B seed 2 at delta 0.6, 248 calls.

This is the PROTONATION-AWARE arm.  Its proposal mechanism is the explicit
protonation-aware retained-subgraph expert, NOT the bridge-separated region-draw
law used by the concurrent region-repair rescue arms.  The two mechanisms are
deliberately never mixed in one arm: if this cell becomes searchable, the arm
identity is what attributes it.

Every round first publishes an immutable candidate and query lock to a durable Modal
volume.  Docking then runs once per locked molecule with no retry or replacement.  The
proposal workers execute concurrently on single-CPU containers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REMOTE = Path("/compose")
OUTPUT = Path("/protonation_rescue_5ht1b2_d06")
LEGACY_OUTPUT = OUTPUT
CONTRACT = "configs/t4_5ht1b2_protonation_rescue_d06_v1.json"
CHECKPOINT = "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
VOLUME_NAME = "compose-t4-5ht1b2-protonation-rescue-d06"
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"


CONTRACT_PATH = ROOT / CONTRACT
ARM = "d06"
PREPARED_STATUS = "PREPARED_AWAITING_OWNER_AUTHORIZATION"
REQUIRED_EXPERT = "protonation_aware_retained_subgraph"
AUTHORIZATION_METADATA_KEYS = (
    "authorization",
    "authorization_form",
    "authorization_sentence",
    "authorized_at_utc",
    "authorized_payload_sha256",
    "owner_response_verbatim",
)


def _assert_wired_and_externally_authorized() -> None:
    """Refuse to construct the arm unless the mechanism is requested and consent is EXTERNAL.

    Two separate refusals, because they fail for different reasons.

    The first is the mechanism.  This arm exists only to test the protonation-aware
    proposal expert; if that expert is not actually requested in the contract's proposal
    block and cold-start floor, the run would reproduce the candidate exhaustion it exists
    to fix while still charging its whole budget -- the same failure the region-law arms
    guard against with their ``region_law`` field assertion.

    The second is the SHAPE of authorization.  An earlier revision of this file demanded
    an authorized status INSIDE the contract payload.  That is backwards: writing consent
    into the payload moves the payload's hash off the very value that was consented to.
    So the payload must stay PREPARED and must carry no consent metadata at all.  Owner
    authorization lives in the external receipt, names the payload by hash, and reaches
    this wrapper through ``--authorization-payload-sha256``, which is checked against the
    payload identity in ``validate_rescue_preflight_v2`` before anything spawns.

    The check lives at import time, which is before ``modal run`` can spawn anything.  It
    deliberately confers NO launch authority -- it only refuses obvious misuse.
    """

    payload = json.loads(CONTRACT_PATH.read_text())["payload"]
    status = payload.get("status")
    if status != PREPARED_STATUS:
        raise RuntimeError(
            f"{CONTRACT} status is {status!r}, not {PREPARED_STATUS!r}; authorization "
            "must point at this payload by hash from the external receipt and must never "
            "be written into the payload"
        )
    present = sorted(key for key in AUTHORIZATION_METADATA_KEYS if key in payload)
    if present:
        raise RuntimeError(
            f"{CONTRACT} carries authorization metadata {present} inside the scientific "
            "payload; recording consent must not change the bytes consented to"
        )
    proposal = payload.get("proposal") or {}
    if REQUIRED_EXPERT not in proposal:
        raise RuntimeError(
            f"proposal.{REQUIRED_EXPERT} is absent; without it this arm would run the "
            "unchanged v1 controller and spend the whole rescue budget reproducing the "
            "candidate exhaustion it exists to fix"
        )
    floors = (
        ((payload.get("allocation_policy") or {}).get("expected_all_pools_nonempty") or {})
        .get("expert_floor_counts")
        or {}
    )
    if int(floors.get(REQUIRED_EXPERT, 0)) < 1:
        raise RuntimeError(
            f"allocation_policy expert floor for {REQUIRED_EXPERT} is not at least 1; the "
            "protonation expert would never be guaranteed a cold-start slot"
        )


_assert_wired_and_externally_authorized()

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
        f"curl --fail -sSL -o /opt/dock/receptors/5ht1b.pdbqt {MOOD}/receptors/5ht1b.pdbqt",
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"]
    )
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(ROOT / CHECKPOINT, str(REMOTE / CHECKPOINT), copy=True)
    .add_local_file(
        ROOT / "docs/GENMOL_T4_SEEDS.json",
        str(REMOTE / "docs/GENMOL_T4_SEEDS.json"),
        copy=True,
    )
    .add_local_file(
        ROOT
        / "modal_apps/t4_5ht1b2_protonation_rescue_d06_app.py",
        str(
            REMOTE
            / "modal_apps/t4_5ht1b2_protonation_rescue_d06_app.py"
        ),
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
legacy_volume = volume
app = modal.App("compose-t4-5ht1b2-protonation-rescue-d06")
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

    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
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


def _proposal_worker_impl(task: dict) -> dict:
    """One expert on one measured parent; no task-oracle access."""

    import json
    import time

    import numpy as np

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.docking_value import identity
    from compose_v4.control.protonation_aware_proposal import (
        ProtonationAwareProposalConfig,
        propose_protonation_aware_candidates,
    )
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
            scale_balanced=route["scale_balanced"],
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
    elif expert == "protonation_aware_retained_subgraph":
        envelope = json.loads((REMOTE / CHECKPOINT).read_text())
        if identity(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError("route expert checkpoint payload hash mismatch")
        model = RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])
        source = pad_molecular_graph(smiles_to_molecular_graph(task["parent"]), 48)
        proposal = contract["proposal"][expert]
        proposed, telemetry = propose_protonation_aware_candidates(
            source,
            model,
            config=ProtonationAwareProposalConfig(
                seed=task["proposal_seed"],
                shallow_draws=proposal["shallow_draws"],
                retained_maximum_fragment_atoms=proposal[
                    "retained_maximum_fragment_atoms"
                ],
                retained_maximum_stages=proposal["retained_maximum_stages"],
                retained_maximum_prefixes=proposal["retained_maximum_prefixes"],
                route_pool_size=proposal["route_pool_size"],
                route_realization_limit=proposal["route_realization_limit"],
                route_beam_width=proposal["route_beam_width"],
                route_expansion_width=proposal["route_expansion_width"],
                route_max_bindings_per_template=proposal[
                    "route_max_bindings_per_template"
                ],
                route_maximum_expansions=proposal["route_maximum_expansions"],
                route_candidate_timeout_seconds=proposal[
                    "route_candidate_timeout_seconds"
                ],
            ),
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
                    "proposal_lane": expert,
                    "families": sorted(set(row.get("program_families") or [])),
                    "parent": task["parent"],
                    "parent_score": task["parent_score"],
                    "delta": contract["delta"],
                }
            )
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


@app.function(**common, max_containers=36, timeout=1800)
def proposal_worker(task: dict) -> dict:
    return _proposal_worker_impl(task)


@app.function(**common, max_containers=36, timeout=1800)
def durable_proposal_worker(task: dict) -> dict:
    """Persist one deterministic expert result independently of its collector."""

    import time

    receipt_path = OUTPUT / task["proposal_receipt_path"]
    volume.reload()
    if receipt_path.exists():
        existing = _read(receipt_path)
        if existing.get("status") in {"complete", "failed"}:
            return existing
    started = time.time()
    _publish(
        receipt_path,
        {
            "schema_version": "t4_integrated_proposal_receipt_v1",
            "status": "running",
            "expert": task["expert"],
            "parent": task["parent"],
            "proposal_seed": task["proposal_seed"],
            "started_at": started,
        },
    )
    try:
        answer = _proposal_worker_impl(task)
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        receipt = {
            "schema_version": "t4_integrated_proposal_receipt_v1",
            "status": "failed",
            "expert": task["expert"],
            "parent": task["parent"],
            "proposal_seed": task["proposal_seed"],
            "error": repr(error),
            "elapsed_seconds": time.time() - started,
        }
    else:
        receipt = {
            "schema_version": "t4_integrated_proposal_receipt_v1",
            "status": "complete",
            "expert": task["expert"],
            "parent": task["parent"],
            "proposal_seed": task["proposal_seed"],
            "answer": answer,
            "elapsed_seconds": time.time() - started,
        }
    _publish(receipt_path, receipt)
    return receipt


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
        "receptor": sha256_file(Path("/opt/dock/receptors/5ht1b.pdbqt")),
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
            "receptor": "/opt/dock/receptors/5ht1b.pdbqt",
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
    from compose_v4.control.t4_cold_start_allocation import cold_start_allocation_v1
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
        proposal_collection_action,
        query_collection_action,
    )
    from compose_v4.experiments.t4_protonation_integrated_route_fiber import (
        EXPERTS,
        attach_features,
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

    if not checkpoint_path.exists():
        legacy = contract.get("legacy_resume", {}).get(cell["cell"])
        if legacy is not None:
            from compose_v4.experiments.continuation_profile import sha256_file

            legacy_volume.reload()
            legacy_lock_path = LEGACY_OUTPUT / legacy["round_lock_path"]
            if sha256_file(legacy_lock_path) != legacy["round_lock_sha256"]:
                raise ValueError("legacy round lock physical hash mismatch")
            legacy_lock = _read(legacy_lock_path)
            if (
                __import__(
                    "compose_v4.control.docking_value", fromlist=["identity"]
                ).identity(legacy_lock)
                != legacy["round_lock_payload_sha256"]
            ):
                raise ValueError("legacy round lock payload hash mismatch")
            if legacy["mode"] == "fail_closed":
                result = {
                    "schema_version": "t4_integrated_route_fiber_result_v2",
                    "status": "operational_fail_closed",
                    "cell": cell["cell"],
                    "charged_calls": len(legacy_lock["queries"]),
                    "unresolved_phase": int(legacy_lock["round"]),
                    "failure": (
                        "legacy immutable query lock has no durable result; no query "
                        "was resubmitted"
                    ),
                    "legacy_resume": legacy,
                }
                _publish(final_path, result)
                return result
            if legacy["mode"] != "checkpoint":
                raise ValueError(f"unknown legacy resume mode {legacy['mode']!r}")
            legacy_checkpoint_path = LEGACY_OUTPUT / legacy["checkpoint_path"]
            if sha256_file(legacy_checkpoint_path) != legacy["checkpoint_sha256"]:
                raise ValueError("legacy checkpoint physical hash mismatch")
            legacy_checkpoint = _read(legacy_checkpoint_path)
            if (
                __import__(
                    "compose_v4.control.docking_value", fromlist=["identity"]
                ).identity(legacy_checkpoint)
                != legacy["checkpoint_payload_sha256"]
            ):
                raise ValueError("legacy checkpoint payload hash mismatch")
            unresolved_lock = None
            if legacy.get("unresolved_round_lock_path"):
                unresolved_path = LEGACY_OUTPUT / legacy["unresolved_round_lock_path"]
                if (
                    sha256_file(unresolved_path)
                    != legacy["unresolved_round_lock_sha256"]
                ):
                    raise ValueError(
                        "legacy unresolved round lock physical hash mismatch"
                    )
                unresolved_lock = _read(unresolved_path)
                if (
                    __import__(
                        "compose_v4.control.docking_value", fromlist=["identity"]
                    ).identity(unresolved_lock)
                    != legacy["unresolved_round_lock_payload_sha256"]
                ):
                    raise ValueError(
                        "legacy unresolved round lock payload hash mismatch"
                    )
            from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
                migrate_v1_checkpoint,
            )

            checkpoint = migrate_v1_checkpoint(
                legacy_checkpoint,
                cell=cell["cell"],
                original_seed=cell["smiles"],
                new_contract_payload_sha256=task["contract_payload_sha256"],
                charged_call_ceiling=contract["charged_calls_per_cell"],
                legacy_run_id=legacy["run_id"],
                legacy_checkpoint_sha256=legacy["checkpoint_sha256"],
                legacy_checkpoint_payload_sha256=legacy["checkpoint_payload_sha256"],
                legacy_round_lock_sha256=legacy["round_lock_sha256"],
                legacy_round_lock_payload_sha256=legacy["round_lock_payload_sha256"],
                unresolved_round_lock=unresolved_lock,
                legacy_unresolved_round_lock_sha256=legacy.get(
                    "unresolved_round_lock_sha256"
                ),
                legacy_unresolved_round_lock_payload_sha256=legacy.get(
                    "unresolved_round_lock_payload_sha256"
                ),
                parents=contract["parents"],
                parent_explore=contract["parent_explore"],
                value_penalty=contract["value_penalty"],
                batch=contract["batch"],
                exploration=contract["exploration"],
                expert_floor_rounds=contract["expert_floor_rounds"],
                route_scale_floor_rounds=contract["route_scale_floor_rounds"],
            )
            _publish(checkpoint_path, checkpoint)
            _publish(
                folder / "legacy_migration.json",
                {
                    "schema_version": "t4_integrated_legacy_migration_v1",
                    "cell": cell["cell"],
                    "legacy_resume": legacy,
                    "charged_calls": checkpoint["charged_calls"],
                    "best": min(checkpoint["archive"].values()),
                },
            )
            return {
                "status": "running",
                "cell": cell["cell"],
                "phase": checkpoint["rounds_completed"],
                "charged_calls": checkpoint["charged_calls"],
                "best": min(checkpoint["archive"].values()),
                "recovered_from": "sealed_v1_checkpoint",
            }

    checkpoint = _read(checkpoint_path) if checkpoint_path.exists() else None
    round_index = 0 if checkpoint is None else int(checkpoint["rounds_completed"]) + 1
    lock_path = folder / f"round_{round_index:03d}_lock.json"
    phase_result_path = folder / f"round_{round_index:03d}_result.json"
    if phase_result_path.exists():
        phase_result = _read(phase_result_path)
        if (
            not checkpoint_path.exists()
            or _read(checkpoint_path) != phase_result["checkpoint"]
        ):
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
            statuses.append(
                _read(receipt).get("status", "invalid")
                if receipt.exists()
                else "missing"
            )
    action = query_collection_action(
        lock_exists=lock is not None,
        receipt_statuses=statuses,
        now=time.time(),
        deadline=None if lock is None else lock.get("query_deadline"),
    )
    if action == "wait":
        return {
            "status": "queries_running",
            "cell": cell["cell"],
            "phase": round_index,
            "charged_calls": int(lock["charged_before"]),
            "receipt_statuses": statuses,
            "query_deadline": lock["query_deadline"],
        }

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
                "query_deadline": time.time() + contract["query_wait_seconds"],
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
        elif action == "recover":
            lock_hash = __import__(
                "compose_v4.control.docking_value", fromlist=["identity"]
            ).identity(lock)
            query = lock["queries"][0]
            answer = _read(folder / "receipts" / f"{query['query_id']}.json")["answer"]
        else:
            result = {
                "schema_version": "t4_integrated_route_fiber_result_v2",
                "status": "root_oracle_failure",
                "cell": cell["cell"],
                "charged_calls": 1,
                "receipt_statuses": statuses,
                "failure": "locked root query remained unresolved; no retry",
            }
            _publish(final_path, result)
            return result
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
        print(
            f"[{cell['cell']}] durable root calls=1 score={answer['score']:.2f}",
            flush=True,
        )
        return {
            "status": "running",
            "cell": cell["cell"],
            "phase": 0,
            "charged_calls": 1,
            "best": answer["score"],
        }

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
        proposal_manifest_path = folder / f"round_{round_index:03d}_proposals.json"
        if not proposal_manifest_path.exists():
            parents = state.parents(
                limit=contract["parents"], rng=rng, explore=contract["parent_explore"]
            )
            requests = []
            for parent_index, parent in enumerate(parents):
                for expert_index, expert in enumerate(EXPERTS):
                    receipt_relative = str(
                        Path(task["run_id"])
                        / cell["cell"]
                        / f"round_{round_index:03d}_proposal_receipts"
                        / f"p{parent_index:02d}_{expert}.json"
                    )
                    requests.append(
                        {
                            **task,
                            "expert": expert,
                            "parent": parent,
                            "parent_score": state.archive[parent],
                            "original_seed": cell["smiles"],
                            "proposal_seed": int(
                                cell["controller_seed"]
                                + 1_000_003 * round_index
                                + 10_007 * parent_index
                                + 101 * expert_index
                            ),
                            "proposal_receipt_path": receipt_relative,
                        }
                    )
            manifest = {
                "schema_version": "t4_integrated_proposal_manifest_v1",
                "cell": cell["cell"],
                "round": round_index,
                "created_at": time.time(),
                "deadline": time.time() + contract["proposal_wait_seconds"],
                "parents": parents,
                "requests": requests,
                "rng_state_after_parent_selection": _jsonable(rng.bit_generator.state),
            }
            _publish(proposal_manifest_path, manifest)
            call_ids = []
            for request in requests:
                call_ids.append(durable_proposal_worker.spawn(request).object_id)
            _publish(
                folder / f"round_{round_index:03d}_proposal_dispatch.json",
                {
                    "schema_version": "t4_integrated_proposal_dispatch_v1",
                    "round": round_index,
                    "call_ids": call_ids,
                },
            )
            return {
                "status": "proposals_running",
                "cell": cell["cell"],
                "phase": round_index,
                "charged_calls": checkpoint["charged_calls"],
                "proposal_jobs": len(requests),
            }
        manifest = _read(proposal_manifest_path)
        requests = manifest["requests"]
        receipt_rows = []
        statuses = []
        for request in requests:
            receipt_path = OUTPUT / request["proposal_receipt_path"]
            if receipt_path.exists():
                receipt = _read(receipt_path)
                statuses.append(receipt.get("status", "invalid"))
                receipt_rows.append(receipt)
            else:
                statuses.append("missing")
                receipt_rows.append(None)
        collection = proposal_collection_action(
            receipt_statuses=statuses,
            now=time.time(),
            deadline=manifest["deadline"],
        )
        if collection == "wait":
            return {
                "status": "proposals_running",
                "cell": cell["cell"],
                "phase": round_index,
                "charged_calls": checkpoint["charged_calls"],
                "proposal_statuses": {
                    name: statuses.count(name)
                    for name in ("complete", "failed", "running", "missing")
                },
                "proposal_deadline": manifest["deadline"],
            }
        parents = manifest["parents"]
        rng.bit_generator.state = manifest["rng_state_after_parent_selection"]
        pools = {expert: [] for expert in EXPERTS}
        worker_telemetry = []
        for request, receipt in zip(requests, receipt_rows, strict=True):
            if receipt is None or receipt.get("status") != "complete":
                worker_telemetry.append(
                    {
                        "expert": request["expert"],
                        "parent": request["parent"],
                        "status": "abstained",
                        "receipt_status": (
                            "missing" if receipt is None else receipt.get("status")
                        ),
                        "error": None if receipt is None else receipt.get("error"),
                    }
                )
                continue
            answer = receipt["answer"]
            pools[answer["expert"]].extend(answer["records"])
            worker_telemetry.append({**answer, "records": None, "status": "complete"})
        merged = merge_expert_pools(pools)
        candidates = attach_features(
            [row for row in merged if row["smiles"] not in state.archive], state, fiber
        )
        room = min(contract["batch"], state.budget)
        available_experts = sorted(
            {expert for row in candidates for expert in row.get("proposal_experts", ())}
        )
        allocation_plan = cold_start_allocation_v1(
            round_index=round_index,
            batch_size=room,
            available_experts=available_experts,
        )
        selected = select_batch(
            candidates,
            value,
            state,
            rng,
            round_index=round_index,
            batch=room,
            exploration=(
                allocation_plan.exploration_slots
                if allocation_plan.active
                else min(contract["exploration"], room)
            ),
            expert_floor_rounds=1 if allocation_plan.active else 0,
            expert_floor_counts=allocation_plan.expert_floor_counts,
            route_scale_floor_rounds=1 if allocation_plan.active else 0,
            route_scale_floor_counts=allocation_plan.route_scale_floor_counts,
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
            "allocation_plan": _jsonable(allocation_plan.__dict__),
            "worker_telemetry": _jsonable(worker_telemetry),
            "candidate_pool": _jsonable(candidates),
            "selected_rows": _jsonable(selected),
            "queries": queries,
            "query_deadline": time.time() + contract["query_wait_seconds"],
            "rng_state_after_selection": _jsonable(rng.bit_generator.state),
        }
        lock_hash = _publish(lock_path, lock)
        dock_tasks = [
            {
                **task,
                "query_id": query["query_id"],
                "smiles": query["smiles"],
                "receipt_path": str(
                    Path(task["run_id"])
                    / cell["cell"]
                    / "receipts"
                    / f"{query['query_id']}.json"
                ),
            }
            for query in queries
        ]
        answers = list(
            dock_worker.map(dock_tasks, order_outputs=True, return_exceptions=True)
        )
    else:
        lock_hash = __import__(
            "compose_v4.control.docking_value", fromlist=["identity"]
        ).identity(lock)
        selected = lock["selected_rows"]
        candidates = lock["candidate_pool"]
        worker_telemetry = lock["worker_telemetry"]
        queries = lock["queries"]
        rng.bit_generator.state = lock["rng_state_after_selection"]
        answers = []
        for query in queries:
            receipt_path = folder / "receipts" / f"{query['query_id']}.json"
            receipt = (
                _read(receipt_path) if receipt_path.exists() else {"status": "missing"}
            )
            if receipt.get("status") == "complete":
                answers.append(receipt["answer"])
            else:
                answers.append(
                    {
                        "query_id": query["query_id"],
                        "smiles": query["smiles"],
                        "score": None,
                        "failure": f"unresolved_locked_query_{receipt.get('status', 'invalid')}",
                        "elapsed_seconds": None,
                        "evaluator_sha256": contract["evaluator_sha256"],
                    }
                )

    if any(isinstance(answer, Exception) for answer in answers):
        volume.reload()
        statuses = []
        for query in queries:
            receipt = folder / "receipts" / f"{query['query_id']}.json"
            statuses.append(
                _read(receipt).get("status", "invalid")
                if receipt.exists()
                else "missing"
            )
        collection = query_collection_action(
            lock_exists=True,
            receipt_statuses=statuses,
            now=time.time(),
            deadline=lock["query_deadline"],
        )
        if collection == "wait":
            return {
                "status": "queries_running",
                "cell": cell["cell"],
                "phase": round_index,
                "charged_calls": int(lock["charged_before"]),
                "receipt_statuses": statuses,
                "query_deadline": lock["query_deadline"],
            }
        answers = []
        for query in queries:
            receipt_path = folder / "receipts" / f"{query['query_id']}.json"
            receipt = (
                _read(receipt_path) if receipt_path.exists() else {"status": "missing"}
            )
            if receipt.get("status") == "complete":
                answers.append(receipt["answer"])
            else:
                answers.append(
                    {
                        "query_id": query["query_id"],
                        "smiles": query["smiles"],
                        "score": None,
                        "failure": f"unresolved_locked_query_{receipt.get('status', 'invalid')}",
                        "elapsed_seconds": None,
                        "evaluator_sha256": contract["evaluator_sha256"],
                    }
                )

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
        "selection_kind_census": {
            kind: sum(row["selection_kind"] == kind for row in selected)
            for kind in ("route_scale_floor", "expert_floor", "model", "exploration")
        },
        "docked": _jsonable(observed),
        "round_best": min(
            (row["score"] for row in observed if row.get("score") is not None),
            default=None,
        ),
        "best_so_far": state.incumbent,
        "improved": improved,
        "value_training_rows": len(improvements),
        "timing_seconds": {
            "phase_total": time.time() - started,
            "proposal_max": max(
                (row.get("elapsed_seconds", 0.0) for row in worker_telemetry),
                default=0.0,
            ),
        },
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
        **(
            {"migration_provenance": checkpoint["migration_provenance"]}
            if "migration_provenance" in checkpoint
            else {}
        ),
    }
    _publish(
        phase_result_path,
        {
            "schema_version": "t4_integrated_phase_result_v1",
            "round": round_index,
            "checkpoint": checkpoint,
            "round_result": round_result,
        },
    )
    _publish(checkpoint_path, checkpoint)
    print(
        f"[{cell['cell']}] durable round={round_index} calls={charged} best={state.incumbent:.2f} round_best={round_result['round_best']}",
        flush=True,
    )

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
            "new_oracle_calls": charged
            - int(
                checkpoint.get("migration_provenance", {}).get(
                    "legacy_charged_calls", 0
                )
            ),
            "prior_operational_waste": contract.get("prior_operational_waste", 0),
            "claim_boundary": contract["claim_boundary"],
        }
        _publish(final_path, result)
        return result
    return {
        "status": "running",
        "cell": cell["cell"],
        "phase": round_index,
        "charged_calls": charged,
        "best": state.incumbent,
    }


@app.function(**common, max_containers=1, timeout=4000)
def drive(task: dict) -> dict:
    import json
    import time

    from compose_v4.experiments.t4_campaign_driver import campaign_driver_action

    contract = _validate_task(task, role="driver")
    folder = OUTPUT / task["run_id"]
    launch_path = folder / "launch.json"
    driver_path = folder / "driver_state.json"
    volume.reload()
    if not launch_path.exists():
        _publish(launch_path, task)

    generation = task.get("continuation_generation")
    if generation is None:
        if driver_path.exists():
            state = _read(driver_path)
            if task.get("confirmed_prior_call_terminal") is not True:
                return {
                    "status": "driver_wait",
                    "run_id": task["run_id"],
                    "driver_state": state,
                    "reason": "a reserved or running driver may still be live",
                }
            if state.get("state") not in {"reserved", "running", "terminal"}:
                raise ValueError("manual resume found invalid durable driver state")
            generation = int(state.get("generation", 0)) + 1
        else:
            generation = 0
        _publish(
            driver_path,
            {
                "schema_version": "t4_fail_closed_driver_state_v1",
                "state": "running",
                "generation": generation,
                "function_call_id": None,
                "manual_terminal_confirmation": bool(
                    task.get("confirmed_prior_call_terminal")
                ),
            },
        )
    else:
        deadline = time.time() + 60.0
        while True:
            volume.reload()
            if driver_path.exists():
                state = _read(driver_path)
                if state.get("state") == "running" and int(
                    state.get("generation", -1)
                ) == int(generation):
                    break
                if int(state.get("generation", -1)) != int(generation):
                    raise RuntimeError(
                        "continuation generation no longer owns the driver"
                    )
            if time.time() >= deadline:
                raise RuntimeError(
                    "continuation remained reserved; fail closed without running a phase"
                )
            time.sleep(1.0)
        time.sleep(contract["phase_poll_seconds"])

    tasks = [{**task, "cell": row["cell"]} for row in contract["cells"]]
    answers = list(run_phase.map(tasks, order_outputs=True, return_exceptions=True))
    records = []
    for row, answer in zip(contract["cells"], answers, strict=True):
        if isinstance(answer, Exception):
            records.append(
                {"cell": row["cell"], "status": "failed", "error": repr(answer)}
            )
        else:
            records.append(
                {
                    "cell": row["cell"],
                    "status": answer["status"],
                    "charged_calls": answer["charged_calls"],
                    "final_best": answer.get("final_best", answer.get("best")),
                    "phase": answer.get("phase"),
                    "new_oracle_calls": answer.get("new_oracle_calls", 0),
                }
            )
    statuses = [row["status"] for row in records]
    _publish(
        driver_path,
        {
            "schema_version": "t4_fail_closed_driver_state_v1",
            "state": "terminal",
            "generation": int(generation),
            "function_call_id": _read(driver_path).get("function_call_id"),
        },
    )
    action = campaign_driver_action(statuses, continuation_state="terminal")
    summary = {
        "schema_version": "t4_integrated_route_fiber_phase_summary_v1",
        "run_id": task["run_id"],
        "contract_payload_sha256": task["contract_payload_sha256"],
        "records": records,
        "finished": action == "finish",
        "driver_action": action,
        "driver_generation": int(generation),
        "charged_calls_total": sum(row.get("charged_calls", 0) for row in records),
        "new_oracle_calls": sum(row.get("new_oracle_calls", 0) for row in records),
        "claim_boundary": contract["claim_boundary"],
    }
    _publish(folder / "progress.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if action == "reserve_and_spawn":
        next_generation = int(generation) + 1
        _publish(
            driver_path,
            {
                "schema_version": "t4_fail_closed_driver_state_v1",
                "state": "reserved",
                "generation": next_generation,
                "function_call_id": None,
            },
        )
        continuation = drive.spawn(
            {
                **task,
                "continuation": True,
                "continuation_generation": next_generation,
                "confirmed_prior_call_terminal": False,
            }
        )
        _publish(
            driver_path,
            {
                "schema_version": "t4_fail_closed_driver_state_v1",
                "state": "running",
                "generation": next_generation,
                "function_call_id": continuation.object_id,
            },
        )
        summary["continuation_function_call_id"] = continuation.object_id
        _publish(folder / "progress.json", summary)
    else:
        _publish(folder / "summary.json", summary)
    return summary


@app.function(**common, max_containers=1, timeout=120)
def remote_status(task: dict) -> dict:
    import json

    _validate_task(task, role="status")
    folder = OUTPUT / task["run_id"]
    volume.reload()
    cells = []
    for cell in ("5ht1b_2",):
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


def _local_task(authorization_payload_sha256: str) -> dict:
    import subprocess

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.t4_matched_pilot import unseal
    from compose_v4.experiments.t4_protonation_rescue_contract_v2 import (
        validate_rescue_preflight_v2,
    )

    contract = unseal(ROOT / CONTRACT)
    preflight = validate_rescue_preflight_v2(
        ROOT,
        ROOT / CONTRACT,
        arm=ARM,
        authorization_payload_sha256=authorization_payload_sha256,
        require_clean_runtime=True,
    )
    # The ceiling this run may charge comes from the CONTRACT, never from a constant in
    # this file.  Cross-check the two readings so a wrapper pointed at the wrong contract
    # cannot inherit the other arm's budget.
    if preflight["charged_call_ceiling"] != contract["total_charged_call_ceiling"]:
        raise ValueError("preflight and contract disagree on the charged-call ceiling")
    body = {
        "schema_version": "t4_5ht1b2_protonation_rescue_launch_v1",
        "contract_payload_sha256": identity(contract),
        "contract_file_sha256": preflight["contract_file_sha256"],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "automatic_retries": 0,
        "charged_call_ceiling": contract["total_charged_call_ceiling"],
    }
    return {**body, "run_id": identity(body)}


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    run_id: str = "",
    authorization_payload_sha256: str = "",
    confirm_prior_call_terminal: bool = False,
) -> None:
    import json

    from compose_v4.experiments.t4_matched_pilot import seal, unseal

    if mode in {"preflight", "launch_dry_run"}:
        from compose_v4.experiments.t4_protonation_rescue_contract_v2 import (
            validate_rescue_preflight_v2,
        )

        report = validate_rescue_preflight_v2(
            ROOT,
            ROOT / CONTRACT,
            arm=ARM,
            authorization_payload_sha256=authorization_payload_sha256 or None,
            require_clean_runtime=True,
        )
        if mode == "preflight":
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        # launch_dry_run builds exactly what mode=launch would spawn and then STOPS: no
        # reservation is written, no function is spawned, no oracle call is charged.  It
        # exists so the launch path can be exercised before it is trusted with a budget.
        if not authorization_payload_sha256:
            raise ValueError(
                "a separately authorized contract payload SHA-256 is required"
            )
        task = _local_task(authorization_payload_sha256)
        destination = (
            ROOT
            / "diagnostics/t4_5ht1b2_protonation_rescue_d06/launches"
            / (task["run_id"] + ".json")
        )
        print(
            json.dumps(
                {
                    "schema_version": "t4_5ht1b2_protonation_rescue_launch_dry_run_v1",
                    "status": "DRY_RUN_NOTHING_SPAWNED",
                    "arm": ARM,
                    "oracle_calls_charged": 0,
                    "preflight": report,
                    "task": task,
                    "volume": VOLUME_NAME,
                    "would_write_receipt": str(destination.relative_to(ROOT)),
                    "receipt_already_exists": destination.exists(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if mode == "launch":
        if not authorization_payload_sha256:
            raise ValueError(
                "a separately authorized contract payload SHA-256 is required"
            )
        task = _local_task(authorization_payload_sha256)
        destination = (
            ROOT
            / "diagnostics/t4_5ht1b2_protonation_rescue_d06/launches"
            / (task["run_id"] + ".json")
        )
        if destination.exists():
            raise RuntimeError(
                "launch reservation already exists; inspect it instead of respawning"
            )
        reservation = {
            "schema_version": "t4_5ht1b2_protonation_rescue_launch_receipt_v1",
            "status": "reserved",
            "task": task,
            "function_call_id": None,
            "volume": VOLUME_NAME,
            "output_prefix": task["run_id"],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        seal(destination, reservation)
        call = drive.spawn(task)
        receipt = {
            **reservation,
            "status": "running",
            "function_call_id": call.object_id,
        }
        seal(destination, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return
    if mode not in {"advance", "status"} or not run_id:
        raise ValueError(
            "use mode=preflight/launch_dry_run/launch, or advance/status with run_id"
        )
    receipt_path = (
        ROOT
        / "diagnostics/t4_5ht1b2_protonation_rescue_d06/launches"
        / f"{run_id}.json"
    )
    receipt = unseal(receipt_path)
    task = receipt["task"]
    if mode == "advance":
        if not confirm_prior_call_terminal:
            raise ValueError(
                "manual advance requires authoritative confirmation that the prior "
                "driver call is terminal"
            )
        call = drive.spawn(
            {
                **task,
                "confirmed_prior_call_terminal": True,
            }
        )
        print(
            json.dumps(
                {"run_id": run_id, "function_call_id": call.object_id}, sort_keys=True
            )
        )
        return
    print(json.dumps(remote_status.remote(task), indent=2))
