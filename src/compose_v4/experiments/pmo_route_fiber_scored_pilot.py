"""Zero-oracle preparation and durable runtime for the frozen PMO factorial.

Preparation creates score-blind candidate locks only. The runtime accepts an
evaluator callable so focused tests can use synthetic labels, but a production
caller must also provide a separately sealed launch authorization. This module
never constructs an oracle itself.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.control.fiber_control import ProgramValue, SearchState, acquisition
from compose_v4.control.program_campaign import ProgramQueryLedger
from compose_v4.control.program_task import ProgramTask, pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.pmo_ivg_oracle_parity import (
    verify_environment,
    verify_pytdc_sources,
)
from compose_v4.experiments.pmo_legal_action_policy import enumerate_rule_successors
from compose_v4.experiments.pmo_route_fiber_production_yield import (
    CHECKPOINT_SCHEMA,
    LEGAL_RUNTIME,
    MAXIMUM_COMPONENTS,
    MAXIMUM_PRIMITIVES,
    RUNTIME_FORBIDDEN_KEYS,
    START,
    STOP,
    _fold_checkpoints,
    _recursive_keys,
    _sample_legal_successor,
    _weighted_order,
)
from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
    ARMS,
    CONTRACT,
    TASKS,
    verify_contract,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT_OUTPUT = "diagnostics/pmo_route_fiber_scored_pilot_v1"
PREFLIGHT = f"{ROOT_OUTPUT}/preflight.json"
LOCK_MANIFEST = f"{ROOT_OUTPUT}/candidate_locks.json"
RUNTIME_DIRECTORY = f"{ROOT_OUTPUT}/runtime"
RUN_DIRECTORY = f"{ROOT_OUTPUT}/runs"
INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
ROUTE_CHECKPOINT = (
    "diagnostics/pmo_route_fiber_pilot/route_transition_checkpoint_v3.json"
)
PRODUCTION_RESULT = "diagnostics/pmo_route_fiber_pilot/production_yield_v3.json"
DYNAMIC_CONTRACT = "configs/pmo_dynamic_v21_development_v1.json"

POOL_SCHEMA = "pmo_route_fiber_candidate_pool_v1"
MANIFEST_SCHEMA = "pmo_route_fiber_candidate_lock_manifest_v1"
PREFLIGHT_SCHEMA = "pmo_route_fiber_scored_preflight_v1"
UNIT_SCHEMA = "pmo_route_fiber_scored_unit_v1"
CHECKPOINT_SCHEMA_UNIT = "pmo_route_fiber_scored_unit_checkpoint_v1"

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
GENERIC_RULES = (
    "atom_delete",
    "atom_insert",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
)


def seal(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    publish_json(path, envelope)
    return envelope


def unseal(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid sealed artifact: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"sealed artifact payload changed: {path}")
    return envelope["payload"]


def _seed(contract_id: str, *parts: object) -> int:
    raw = json.dumps([contract_id, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def _runtime_bundle_path(root: Path, fold: int) -> Path:
    return root / RUNTIME_DIRECTORY / f"fold_{fold}.json"


def _pool_path(root: Path, task: str, source: str, round_index: int) -> Path:
    return (
        root
        / ROOT_OUTPUT
        / "locks"
        / task
        / source
        / f"round_{round_index:02d}"
        / "pool.json"
    )


def _query_path(root: Path, task: str, arm: str, round_index: int) -> Path:
    return (
        root
        / ROOT_OUTPUT
        / "locks"
        / task
        / arm
        / f"round_{round_index:02d}"
        / "query.json"
    )


def _read_initialization(root: Path) -> dict[str, Any]:
    initialized = json.loads((root / INITIALIZATION).read_text())
    body = {key: value for key, value in initialized.items() if key != "lock_sha256"}
    if identity(body) != initialized.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")
    if initialized.get("count") != 16 or len(initialized.get("candidates", ())) != 16:
        raise ValueError("PMO scored pilot requires exactly 16 initialization sources")
    if any("score" in row or "task" in row for row in initialized["candidates"]):
        raise ValueError("PMO initialization contains task or score information")
    return initialized


def prepare_runtime_bundles(root: Path) -> dict[int, dict[str, Any]]:
    """Extract only each selected sanitized numeric fold checkpoint."""
    route = unseal(root / ROUTE_CHECKPOINT)
    with gzip.open(root / LEGAL_RUNTIME, "rt") as handle:
        legal_envelope = json.load(handle)
    if identity(legal_envelope["payload"]) != legal_envelope["payload_sha256"]:
        raise ValueError("PMO legal-action checkpoint payload changed")
    legal = _fold_checkpoints(legal_envelope["payload"])
    transitions = {int(row["fold_index"]): row for row in route.get("folds", ())}
    if sorted(transitions) != [0, 1, 2] or sorted(legal) != [0, 1, 2]:
        raise ValueError("PMO runtime fold coverage changed")

    bundles = {}
    for fold in sorted({row["fold"] for row in TASKS.values()}):
        body = {
            "schema_version": "pmo_route_fiber_sanitized_fold_bundle_v1",
            "fold_index": fold,
            "route_transition": transitions[fold],
            "legal_action_ranker": legal[fold],
            "new_oracle_calls": 0,
        }
        if RUNTIME_FORBIDDEN_KEYS & _recursive_keys(body):
            raise ValueError(f"fold {fold} retained forbidden teacher fields")
        bundles[fold] = seal(_runtime_bundle_path(root, fold), body)
    return bundles


def _action_census(actions: list[dict[str, Any]]) -> dict[str, Any]:
    rules = [str(action["executor_rule"]) for action in actions]
    histogram = {rule: rules.count(rule) for rule in GENERIC_RULES}
    return {
        "generic_rule_histogram": histogram,
        "created_atom_count": histogram["atom_insert"],
        "deleted_atom_count": histogram["atom_delete"],
    }


def _fingerprint(endpoint: str) -> list[int]:
    molecule = Chem.MolFromSmiles(endpoint)
    if molecule is None:
        raise ValueError(f"generated endpoint is invalid: {endpoint}")
    return list(map(int, GENERATOR.GetFingerprint(molecule).GetOnBits()))


def _candidate(
    *,
    source_row: dict[str, Any],
    source_index: int,
    proposal_source: str,
    lane: str,
    endpoint: str,
    endpoint_state: dict[str, Any],
    actions: list[dict[str, Any]],
    component_count: int,
    created_dependencies: list[dict[str, Any]],
    cycle_dependencies: list[dict[str, Any]],
    program_identity: str,
    attempt_identity: str,
) -> dict[str, Any]:
    census = _action_census(actions)
    structural = {
        "route_lane_indicator": int(lane == "route"),
        "primitive_count": len(actions),
        "dependency_region_count": int(component_count),
        "created_atom_count": census["created_atom_count"],
        "deleted_atom_count": census["deleted_atom_count"],
        "created_dependency_count": len(created_dependencies),
        "cycle_dependency_count": len(cycle_dependencies),
        "generic_rule_histogram": census["generic_rule_histogram"],
    }
    parent_id = identity(source_row["state"])
    body = {
        "parent_id": parent_id,
        "parent_index": source_index,
        "proposal_source": proposal_source,
        "proposal_lane": lane,
        "program_identity": program_identity,
        "attempt_identity": attempt_identity,
        "canonical_endpoint": endpoint,
        "endpoint_state": endpoint_state,
        "exact_replay": True,
        "validity": True,
        "primitive_count": len(actions),
        "dependency_region_count": int(component_count),
        "created_handle_dependencies": created_dependencies,
        "cycle_dependencies": cycle_dependencies,
        "generic_rule_histogram": census["generic_rule_histogram"],
        "selection_features": structural,
        "structural_fingerprint": _fingerprint(endpoint),
    }
    return {**body, "candidate_id": identity(body)}


def _v0_attempt(source_row: dict[str, Any], source_index: int, rng, draw: int) -> dict:
    source = decode_state(source_row["state"])
    try:
        if (
            source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
            and draw < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
        ):
            _, program, _, trace, metadata = synthesize_named_module_sequence(
                source,
                rng,
                ("substituent_delete",),
                max_primitives=32,
                max_blocks=8,
            )
        else:
            _, program, _, trace, metadata = synthesize_dynamic_program(
                source,
                rng,
                max_modules=3,
                max_primitives=32,
                max_blocks=8,
            )
        region = dependency_region_program(
            trace["states"],
            trace["actions"],
            config=DependencyRegionConfig(
                runtime_maximum_primitives=32,
                runtime_maximum_components=8,
            ),
        )
        if not region["exact_replay"]:
            raise RuntimeError("v0 candidate failed dependency-region exact replay")
        return {
            "status": "complete",
            "candidate": _candidate(
                source_row=source_row,
                source_index=source_index,
                proposal_source="old_v0",
                lane="v0",
                endpoint=trace["endpoint"],
                endpoint_state=trace["states"][-1],
                actions=trace["actions"],
                component_count=region["component_count"],
                created_dependencies=region["created_dependency_edges"],
                cycle_dependencies=region["cycle_dependency_edges"],
                program_identity=program.program_id,
                attempt_identity=identity(
                    {
                        "source": identity(source_row["state"]),
                        "program": program.program_id,
                        "draw": draw,
                    }
                ),
            ),
            "generated_program": program.payload(),
            "execution_actions": trace["actions"],
            "execution_states": trace["states"],
            "module_families": [row["family"] for row in metadata["modules"]],
        }
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as error:
        return {"status": "rejected", "reason": f"{type(error).__name__}: {error}"}


def _route_attempt(
    source_row: dict[str, Any],
    source_index: int,
    bundle: dict[str, Any],
    rng,
) -> dict:
    transition = bundle["route_transition"]
    legal = bundle["legal_action_ranker"]
    if transition.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("route-transition runtime schema changed")
    source = decode_state(source_row["state"])
    graph = source
    states = [encode_state(source)]
    actions: list[dict[str, Any]] = []
    previous = START
    telemetry = {
        "rule_draws": 0,
        "empty_rule_fibers": 0,
        "legal_successors_enumerated": 0,
        "chosen_probabilities": [],
        "chosen_rule_sequence": [],
        "stopped": False,
    }
    try:
        for _ in range(MAXIMUM_PRIMITIVES):
            row = transition["transitions"][previous]
            ordered = _weighted_order(
                tuple(map(str, row["classes"])),
                np.asarray(row["probabilities"], dtype=float),
                rng,
            )
            advanced = False
            for token in ordered:
                telemetry["rule_draws"] += 1
                if token == STOP:
                    if actions:
                        telemetry["stopped"] = True
                        advanced = True
                        break
                    continue
                candidates = tuple(enumerate_rule_successors(graph, token))
                telemetry["legal_successors_enumerated"] += len(candidates)
                if not candidates:
                    telemetry["empty_rule_fibers"] += 1
                    continue
                selected, probability, _ = _sample_legal_successor(
                    graph, candidates, legal, rng
                )
                graph = selected.successor
                states.append(encode_state(graph))
                actions.append(selected.action_record)
                telemetry["chosen_probabilities"].append(probability)
                telemetry["chosen_rule_sequence"].append(token)
                previous = token
                advanced = True
                break
            if telemetry["stopped"] or not advanced:
                break
        if not actions:
            return {"status": "rejected", "reason": "no_executable_program"}
        region = dependency_region_program(
            states,
            actions,
            config=DependencyRegionConfig(
                runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
                runtime_maximum_components=MAXIMUM_COMPONENTS,
            ),
        )
        endpoint = canonical_state_key(graph)
        if not region["exact_replay"] or endpoint == canonical_state_key(source):
            return {"status": "rejected", "reason": "nonexact_or_self_endpoint"}
        program_id = identity(
            {"source": identity(source_row["state"]), "actions": actions}
        )
        return {
            "status": "complete",
            "candidate": _candidate(
                source_row=source_row,
                source_index=source_index,
                proposal_source="additive_route",
                lane="route",
                endpoint=endpoint,
                endpoint_state=states[-1],
                actions=actions,
                component_count=region["component_count"],
                created_dependencies=region["created_dependency_edges"],
                cycle_dependencies=region["cycle_dependency_edges"],
                program_identity=program_id,
                attempt_identity=program_id,
            ),
            "execution_actions": actions,
            "execution_states": states,
            "telemetry": telemetry,
        }
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as error:
        return {"status": "rejected", "reason": f"{type(error).__name__}: {error}"}


def _publish_pool(
    path: Path,
    *,
    contract_id: str,
    task: str,
    fold: int,
    round_index: int,
    proposal_source: str,
    attempts: list[dict[str, Any]],
    seen: set[str],
    implementation_sha256: str,
) -> dict[str, Any]:
    candidates = []
    duplicate_ids = []
    for attempt in attempts:
        if attempt["status"] != "complete":
            continue
        candidate = attempt["candidate"]
        endpoint = candidate["canonical_endpoint"]
        if endpoint in seen or any(
            row["canonical_endpoint"] == endpoint for row in candidates
        ):
            duplicate_ids.append(candidate["candidate_id"])
            continue
        candidates.append(candidate)
    if len(candidates) < 8:
        raise RuntimeError(
            f"candidate pool shortfall before scoring: {task}/{proposal_source}/"
            f"{round_index}: {len(candidates)} unique candidates"
        )
    seen.update(row["canonical_endpoint"] for row in candidates)
    payload = {
        "schema_version": POOL_SCHEMA,
        "contract_payload_sha256": contract_id,
        "task": task,
        "fold_index": fold,
        "round": round_index,
        "proposal_source": proposal_source,
        "attempt_budget": 32,
        "attempts": attempts,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "duplicate_candidate_ids": duplicate_ids,
        "score_fields_present": False,
        "new_oracle_calls": 0,
        "implementation_sha256": implementation_sha256,
    }
    forbidden = {"reward", "control_score", "winner_identity", "comparator_outcome"}
    if forbidden & _recursive_keys(payload):
        raise ValueError("score-blind candidate lock retained a forbidden field")
    return seal(path, payload)


def prepare_candidate_locks(root: Path, *, progress=None) -> dict[str, Any]:
    """Generate every immutable score-blind pool before scoring."""
    contract = verify_contract(root, root / CONTRACT)
    contract_id = contract["payload_sha256"]
    initialized = _read_initialization(root)
    bundles = prepare_runtime_bundles(root)
    implementation = sha256_file(Path(__file__))
    report = progress or (lambda row: None)
    manifest_rows = []

    for task, task_spec in TASKS.items():
        fold = int(task_spec["fold"])
        bundle = bundles[fold]["payload"]
        seen = {
            "old_v0": {
                Chem.MolToSmiles(Chem.MolFromSmiles(row["endpoint"]))
                for row in initialized["candidates"]
            },
            "additive_route": {
                Chem.MolToSmiles(Chem.MolFromSmiles(row["endpoint"]))
                for row in initialized["candidates"]
            },
        }
        for round_index in range(1, 5):
            old_attempts = []
            additive_attempts = []
            for source_index, source_row in enumerate(initialized["candidates"]):
                first = _v0_attempt(
                    source_row,
                    source_index,
                    np.random.default_rng(
                        _seed(contract_id, "v0", round_index, source_index, 0)
                    ),
                    0,
                )
                second = _v0_attempt(
                    source_row,
                    source_index,
                    np.random.default_rng(
                        _seed(contract_id, "v0", round_index, source_index, 1)
                    ),
                    1,
                )
                route = _route_attempt(
                    source_row,
                    source_index,
                    bundle,
                    np.random.default_rng(
                        _seed(
                            contract_id,
                            "route",
                            fold,
                            round_index,
                            source_index,
                        )
                    ),
                )
                old_attempts.extend((first, second))
                additive_attempts.extend((first, route))
                report(
                    {
                        "phase": "candidate_generation",
                        "task": task,
                        "round": round_index,
                        "source": source_index + 1,
                        "sources": 16,
                        "route_status": route["status"],
                        "new_oracle_calls": 0,
                    }
                )
            for proposal_source, attempts in (
                ("old_v0", old_attempts),
                ("additive_route", additive_attempts),
            ):
                path = _pool_path(root, task, proposal_source, round_index)
                if path.exists():
                    envelope = {
                        "payload": unseal(path),
                        "payload_sha256": json.loads(path.read_text())[
                            "payload_sha256"
                        ],
                    }
                else:
                    envelope = _publish_pool(
                        path,
                        contract_id=contract_id,
                        task=task,
                        fold=fold,
                        round_index=round_index,
                        proposal_source=proposal_source,
                        attempts=attempts,
                        seen=seen[proposal_source],
                        implementation_sha256=implementation,
                    )
                payload = envelope["payload"]
                if (
                    payload["contract_payload_sha256"] != contract_id
                    or payload["task"] != task
                    or payload["fold_index"] != fold
                    or payload["round"] != round_index
                    or payload["proposal_source"] != proposal_source
                    or payload["candidate_count"] < 8
                    or payload["score_fields_present"] is not False
                ):
                    raise ValueError(
                        f"candidate pool does not match frozen contract: {path}"
                    )
                seen[proposal_source].update(
                    row["canonical_endpoint"] for row in payload["candidates"]
                )
                manifest_rows.append(
                    {
                        "task": task,
                        "fold_index": fold,
                        "round": round_index,
                        "proposal_source": proposal_source,
                        "path": str(path.relative_to(root)),
                        "file_sha256": sha256_file(path),
                        "payload_sha256": envelope["payload_sha256"],
                        "candidate_count": payload["candidate_count"],
                    }
                )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "contract_payload_sha256": contract_id,
        "pools": manifest_rows,
        "pool_count": len(manifest_rows),
        "all_locked_before_scoring": len(manifest_rows) == 16,
        "new_oracle_calls": 0,
    }
    return seal(root / LOCK_MANIFEST, manifest)


def pmo_program_features(
    candidate: dict[str, Any],
    state: SearchState,
    *,
    parent_reward: float,
) -> np.ndarray:
    """Frozen task-free PMO transformation features for online FiberControl."""
    raw = candidate["selection_features"]
    incumbent_reward = (
        -state.incumbent if math.isfinite(state.incumbent) else parent_reward
    )
    rules = raw["generic_rule_histogram"]
    return np.asarray(
        [
            -parent_reward,
            incumbent_reward - parent_reward,
            raw["route_lane_indicator"],
            raw["primitive_count"] / 8,
            raw["dependency_region_count"] / 8,
            raw["created_atom_count"] / 6,
            raw["deleted_atom_count"] / 6,
            raw["created_dependency_count"] / 4,
            raw["cycle_dependency_count"] / 4,
            *(rules[name] / 8 for name in GENERIC_RULES),
            1.0,
        ],
        dtype=float,
    )


def _selection_seed(
    contract_id: str,
    fold: int,
    proposal_source: str,
    round_index: int,
    selection: str,
) -> int:
    policy = "round1_shared" if round_index == 1 else selection
    return _seed(contract_id, "selection", fold, proposal_source, round_index, policy)


def _validate_launch_receipt(
    root: Path,
    contract_id: str,
    *,
    task: str,
    arm: str,
    oracle_protocol: str,
    launch_receipt: Path | None,
) -> None:
    if oracle_protocol.startswith("synthetic:"):
        if launch_receipt is not None:
            raise ValueError(
                "synthetic fixture must not accept a scored launch receipt"
            )
        return
    if launch_receipt is None:
        raise PermissionError(
            "non-synthetic PMO evaluation requires a separately sealed scored launch receipt"
        )
    launch = unseal(launch_receipt)
    if (
        launch.get("schema_version") != "pmo_route_fiber_scored_launch_v1"
        or launch.get("contract_payload_sha256") != contract_id
        or launch.get("scored_launch_authorized") is not True
        or launch.get("charged_call_ceiling") != 384
        or launch.get("automatic_retries") != 0
        or task not in launch.get("tasks", ())
        or arm not in launch.get("arms", ())
        or launch.get("oracle_protocols", {}).get(task) != oracle_protocol
    ):
        raise PermissionError(
            "scored launch receipt does not match the frozen PMO pilot"
        )
    preflight_path = root / PREFLIGHT
    manifest_path = root / LOCK_MANIFEST
    if launch.get("preflight_file_sha256") != sha256_file(preflight_path) or launch.get(
        "candidate_manifest_file_sha256"
    ) != sha256_file(manifest_path):
        raise PermissionError(
            "scored launch receipt is not bound to preflight and locks"
        )


def _curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = []
    curve = []
    for index, row in enumerate(rows, start=1):
        if row["status"] != "complete":
            raise ValueError("PMO curve cannot include an incomplete receipt")
        scores.append(float(row["score"]))
        top = sorted(scores, reverse=True)[:10]
        curve.append(
            {
                "charged_calls": index,
                "best_reward": max(scores),
                "top10_mean": sum(top) / len(top),
            }
        )
    return curve


def _checkpoint_payload(
    *,
    contract_id: str,
    task: str,
    arm: str,
    round_index: int,
    ledger: ProgramQueryLedger,
    model: ProgramValue,
    state: SearchState,
    query_lock_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_UNIT,
        "contract_payload_sha256": contract_id,
        "task": task,
        "arm": arm,
        "completed_round": round_index,
        "charged_calls": len(ledger.rows),
        "query_lock_ids": query_lock_ids,
        "search_state": {
            "archive": state.archive,
            "budget": state.budget,
            "rounds": state.rounds,
            "history": state.history,
        },
        "program_value": {
            "penalty": model.penalty,
            "n": model.n,
            "weights": None if model.weights is None else model.weights.tolist(),
            "mean": (
                None
                if model.weights is None
                else np.asarray(model._mean, dtype=float).tolist()
            ),
            "scale": (
                None
                if model.weights is None
                else np.asarray(model._scale, dtype=float).tolist()
            ),
        },
        "score_curve": _curve(ledger.rows),
    }


def _fit_value(
    observations: list[dict[str, Any]],
) -> ProgramValue:
    model = ProgramValue(penalty=1.0)
    if observations:
        model.fit(
            [row["features"] for row in observations],
            [row["improvement"] for row in observations],
        )
    return model


def run_locked_unit(
    root: Path,
    output: Path,
    *,
    task: str,
    arm: str,
    oracle_protocol: str,
    evaluate: Callable[[str], float],
    progress=None,
    stop_after_round: int | None = None,
    query_lock_root: Path | None = None,
    launch_receipt: Path | None = None,
    flush=None,
) -> dict[str, Any]:
    """Run one locked unit with a caller-supplied evaluator.

    Production code must additionally bind this call to a separately authorized
    launch receipt. Focused tests use explicit synthetic protocol identities.
    """
    contract = verify_contract(root, root / CONTRACT)
    contract_id = contract["payload_sha256"]
    if task not in TASKS or arm not in ARMS:
        raise ValueError("task or arm is outside the frozen PMO factorial")
    if not oracle_protocol:
        raise ValueError("identified evaluator protocol required")
    _validate_launch_receipt(
        root,
        contract_id,
        task=task,
        arm=arm,
        oracle_protocol=oracle_protocol,
        launch_receipt=launch_receipt,
    )
    lock_manifest = unseal(root / LOCK_MANIFEST)
    if (
        lock_manifest["contract_payload_sha256"] != contract_id
        or lock_manifest["pool_count"] != 16
        or lock_manifest["all_locked_before_scoring"] is not True
    ):
        raise ValueError("candidate-lock manifest is incomplete")
    initialized = _read_initialization(root)
    spec = ARMS[arm]
    proposal_source = spec["proposal_source"]
    selection = spec["selection"]
    task_object = ProgramTask(task, oracle_protocol, "pmo")
    ledger = ProgramQueryLedger(
        output / "oracle", task_object, evaluate, budget=48, flush=flush
    )
    report = progress or (lambda row: None)
    lock_root = root if query_lock_root is None else query_lock_root
    initialization_lock = {
        "schema_version": "pmo_route_fiber_initialization_query_lock_v1",
        "contract_payload_sha256": contract_id,
        "task": task,
        "arm": arm,
        "candidate_ids": [
            identity({"index": index, "state": row["state"]})
            for index, row in enumerate(initialized["candidates"])
        ],
        "endpoints": [row["endpoint"] for row in initialized["candidates"]],
    }
    initial_query_lock = seal(
        output / "initialization_query_lock.json", initialization_lock
    )
    for row in initialized["candidates"]:
        ledger.query(
            row["endpoint"],
            lock_id=initial_query_lock["payload_sha256"],
            role="initialization",
        )
    parent_rewards = {
        index: float(
            ledger.cache[Chem.MolToSmiles(Chem.MolFromSmiles(row["endpoint"]))]["score"]
        )
        for index, row in enumerate(initialized["candidates"])
    }
    observations: list[dict[str, Any]] = []
    query_lock_ids: list[str] = []
    state = SearchState(
        archive={
            row["endpoint"]: -parent_rewards[index]
            for index, row in enumerate(initialized["candidates"])
        },
        budget=ledger.remaining,
        rounds=0,
        history=[],
    )
    began = perf_counter()

    for round_index in range(1, 5):
        pool_path = _pool_path(root, task, proposal_source, round_index)
        pool = unseal(pool_path)
        candidates = pool["candidates"]
        prepared = []
        for candidate in candidates:
            parent_reward = parent_rewards[int(candidate["parent_index"])]
            prepared.append(
                {
                    **candidate,
                    "parent_score": -parent_reward,
                    "features": pmo_program_features(
                        candidate, state, parent_reward=parent_reward
                    ),
                    "fingerprint": set(candidate["structural_fingerprint"]),
                }
            )
        model = _fit_value(observations)
        rng = np.random.default_rng(
            _selection_seed(
                contract_id,
                int(TASKS[task]["fold"]),
                proposal_source,
                round_index,
                selection,
            )
        )
        if selection == "blind" or round_index == 1:
            selected_indices = sorted(
                map(int, rng.choice(len(prepared), 8, replace=False))
            )
            predicted = [None] * len(prepared)
        else:
            selected_indices = acquisition(
                prepared,
                model,
                state,
                rng,
                beta=1.0,
                batch=8,
                diversity=0.5,
                exploration=2,
            )
            predictions = model.predict(
                np.asarray([row["features"] for row in prepared])
            )
            predicted = list(map(float, predictions))
        query_payload = {
            "schema_version": "pmo_route_fiber_query_lock_v1",
            "contract_payload_sha256": contract_id,
            "task": task,
            "arm": arm,
            "proposal_source": proposal_source,
            "selection": selection,
            "round": round_index,
            "pool_path": str(pool_path.relative_to(root)),
            "pool_payload_sha256": json.loads(pool_path.read_text())["payload_sha256"],
            "selected_indices": selected_indices,
            "selected_candidate_ids": [
                prepared[index]["candidate_id"] for index in selected_indices
            ],
            "predicted_improvements": [predicted[index] for index in selected_indices],
            "score_fields_present": False,
        }
        query_path = _query_path(lock_root, task, arm, round_index)
        if query_path.exists():
            existing = unseal(query_path)
            if existing != query_payload:
                raise ValueError(f"query lock changed during resume: {query_path}")
            query_envelope = json.loads(query_path.read_text())
        else:
            query_envelope = seal(query_path, query_payload)
        query_lock_ids.append(query_envelope["payload_sha256"])
        for index in selected_indices:
            candidate = prepared[index]
            receipt = ledger.query(
                candidate["canonical_endpoint"],
                lock_id=query_envelope["payload_sha256"],
                role="candidate",
            )
            reward = float(receipt["score"])
            parent_reward = parent_rewards[int(candidate["parent_index"])]
            observations.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "features": candidate["features"].tolist(),
                    "improvement": reward - parent_reward,
                    "reward": reward,
                    "parent_reward": parent_reward,
                    "proposal_lane": candidate["proposal_lane"],
                }
            )
            state.archive[candidate["canonical_endpoint"]] = -reward
        state.rounds = round_index
        state.budget = ledger.remaining
        state.history.append(
            {
                "round": round_index,
                "query_lock_id": query_envelope["payload_sha256"],
                "selected_candidate_ids": query_payload["selected_candidate_ids"],
            }
        )
        model = _fit_value(observations)
        checkpoint = _checkpoint_payload(
            contract_id=contract_id,
            task=task,
            arm=arm,
            round_index=round_index,
            ledger=ledger,
            model=model,
            state=state,
            query_lock_ids=query_lock_ids,
        )
        seal(output / "checkpoint.json", checkpoint)
        curve = _curve(ledger.rows)
        elapsed = perf_counter() - began
        completed_work = 16 + 8 * round_index
        estimated = elapsed / max(1, completed_work - 16) * (48 - completed_work)
        status = {
            "schema_version": "pmo_route_fiber_progress_v1",
            "task": task,
            "arm": arm,
            "phase": "round_complete",
            "round": round_index,
            "charged_calls": len(ledger.rows),
            "remaining_calls": ledger.remaining,
            "candidate_pool_size": len(candidates),
            "candidate_shortfall": max(0, 8 - len(candidates)),
            "current_best_reward": curve[-1]["best_reward"],
            "current_top10_mean": curve[-1]["top10_mean"],
            "proposal_seconds": 0.0,
            "oracle_seconds": sum(
                float(row.get("seconds", 0.0)) for row in ledger.rows
            ),
            "wall_seconds": elapsed,
            "estimated_remaining_seconds": estimated,
        }
        publish_json(output / "progress.json", status)
        report(status)
        if stop_after_round == round_index:
            return {
                "schema_version": UNIT_SCHEMA,
                "status": "paused_fixture",
                "task": task,
                "arm": arm,
                "charged_calls": len(ledger.rows),
                "completed_round": round_index,
                "selected_candidate_ids": [row["candidate_id"] for row in observations],
                "scores": [float(row["score"]) for row in ledger.rows],
            }

    curve = _curve(ledger.rows)
    rewards = [float(row["score"]) for row in ledger.rows]
    result = {
        "schema_version": UNIT_SCHEMA,
        "status": "complete",
        "task": task,
        "arm": arm,
        "proposal_source": proposal_source,
        "selection": selection,
        "charged_calls": len(ledger.rows),
        "best_reward": curve[-1]["best_reward"],
        "top10_mean": curve[-1]["top10_mean"],
        "auc_top10_48": pmo_top_ten_auc(rewards, budget=48, frequency=1, finish=True),
        "score_curve": curve,
        "observations": observations,
        "selected_candidate_ids": [row["candidate_id"] for row in observations],
        "new_oracle_calls": len(ledger.rows),
    }
    seal(output / "result.json", result)
    if flush is not None:
        flush()
    return result


def build_scored_launch_receipt(root: Path, *, code_revision: str) -> dict[str, Any]:
    """Build the separately authorized receipt without changing the frozen contract."""
    contract = verify_contract(root, root / CONTRACT)
    preflight = unseal(root / PREFLIGHT)
    manifest = unseal(root / LOCK_MANIFEST)
    if (
        preflight.get("decision")
        != "ZERO_ORACLE_PREPARATION_PASSED_NOT_AUTHORIZED_TO_SCORE"
        or preflight.get("new_oracle_calls") != 0
        or manifest.get("pool_count") != 16
        or manifest.get("all_locked_before_scoring") is not True
    ):
        raise ValueError("PMO route/FiberControl launch prerequisites changed")
    if not isinstance(code_revision, str) or len(code_revision) != 40:
        raise ValueError("launch requires an exact Git revision")
    return {
        "schema_version": "pmo_route_fiber_scored_launch_v1",
        "contract_payload_sha256": contract["payload_sha256"],
        "preflight_file_sha256": sha256_file(root / PREFLIGHT),
        "candidate_manifest_file_sha256": sha256_file(root / LOCK_MANIFEST),
        "scored_launch_authorized": True,
        "authorization_recorded_at_utc": "2026-09-19",
        "authorization_scope": (
            "user directed the exact frozen PMO route-prior x FiberControl pilot "
            "to run under its 384-call, zero-retry contract"
        ),
        "code_revision": code_revision,
        "tasks": sorted(TASKS),
        "arms": sorted(ARMS),
        "oracle_protocols": {
            task: f"native-pmo:{task}" for task in sorted(TASKS)
        },
        "charged_call_ceiling": 384,
        "automatic_retries": 0,
        "worker_count": 8,
        "cpu_per_worker": 1,
    }


def aggregate_scored_units(
    launch: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reduce the eight isolated units only after every unit has completed."""
    expected = {(task, arm) for task in TASKS for arm in ARMS}
    observed = {(row.get("task"), row.get("arm")) for row in results}
    if observed != expected or len(results) != len(expected):
        raise ValueError(f"PMO scored unit census mismatch: {observed} != {expected}")
    if any(
        row.get("schema_version") != UNIT_SCHEMA
        or row.get("status") != "complete"
        or row.get("charged_calls") != 48
        or row.get("new_oracle_calls") != 48
        for row in results
    ):
        raise ValueError("PMO scored unit is incomplete or violates its call budget")
    total = sum(int(row["charged_calls"]) for row in results)
    if total != 384 or total > int(launch["charged_call_ceiling"]):
        raise ValueError("PMO scored aggregate violates the frozen call ceiling")

    by_task = {}
    for task in sorted(TASKS):
        arms = {row["arm"]: row for row in results if row["task"] == task}
        combined = arms["additive_route_fiber_control"]
        better_single = max(
            arms["additive_route_blind"]["auc_top10_48"],
            arms["old_v0_fiber_control"]["auc_top10_48"],
        )
        route_improvements = sum(
            row["proposal_lane"] == "route" and row["improvement"] > 0
            for row in combined["observations"]
        )
        by_task[task] = {
            "arms": arms,
            "combined_minus_v0_blind_auc": (
                combined["auc_top10_48"] - arms["old_v0_blind"]["auc_top10_48"]
            ),
            "combined_minus_best_single_auc": (
                combined["auc_top10_48"] - better_single
            ),
            "combined_route_parent_improvements": route_improvements,
        }

    both_base = all(
        row["arms"]["additive_route_fiber_control"]["auc_top10_48"]
        > row["arms"]["old_v0_blind"]["auc_top10_48"]
        and row["arms"]["additive_route_fiber_control"]["best_reward"]
        > row["arms"]["old_v0_blind"]["best_reward"]
        for row in by_task.values()
    )
    single_deltas = [
        row["combined_minus_best_single_auc"] for row in by_task.values()
    ]
    promotion = (
        both_base
        and max(single_deltas) >= 0.02
        and min(single_deltas) >= -0.01
        and all(
            row["combined_route_parent_improvements"] > 0
            for row in by_task.values()
        )
    )
    route_failed_both = all(
        row["arms"]["additive_route_fiber_control"]["auc_top10_48"]
        <= row["arms"]["old_v0_fiber_control"]["auc_top10_48"]
        and row["arms"]["additive_route_blind"]["auc_top10_48"]
        <= row["arms"]["old_v0_blind"]["auc_top10_48"]
        for row in by_task.values()
    )
    decision = (
        "PROMOTE"
        if promotion
        else ("KILL_REVISION" if route_failed_both else "NO_PROMOTION")
    )
    return {
        "schema_version": "pmo_route_fiber_scored_result_v1",
        "status": "complete",
        "decision": decision,
        "launch_payload_sha256": identity(launch),
        "charged_calls": total,
        "automatic_retries": 0,
        "tasks": by_task,
    }


def build_preflight(root: Path) -> dict[str, Any]:
    """Verify pinned runtime identities and every zero-score lock."""
    contract = verify_contract(root, root / CONTRACT)
    dynamic = json.loads((root / DYNAMIC_CONTRACT).read_text())["payload"]
    qualified = {
        "environment": verify_environment(dynamic),
        "pytdc_source_sha256": verify_pytdc_sources(dynamic),
    }
    if not platform.python_version().startswith("3.11."):
        raise ValueError(
            f"PMO pinned preflight requires Python 3.11, got {platform.python_version()}"
        )
    asset = root / contract["payload"]["oracle"]["gsk3b_asset"]["path"]
    expected_asset = contract["payload"]["oracle"]["gsk3b_asset"]
    if asset.stat().st_size != expected_asset["bytes"]:
        raise ValueError("GSK3B asset byte count changed")
    verify_file(asset, expected_asset["sha256"])
    qualified["gsk3b_asset_sha256"] = expected_asset["sha256"]

    production = unseal(root / PRODUCTION_RESULT)
    if (
        production["decision"] != "ZERO_ORACLE_PRODUCTION_SUPPORT_PASSED_YIELD_TIED"
        or production["new_oracle_calls"] != 0
    ):
        raise ValueError("sealed PMO zero-oracle production result changed")
    locks = unseal(root / LOCK_MANIFEST)
    if (
        locks["pool_count"] != 16
        or locks["all_locked_before_scoring"] is not True
        or locks["new_oracle_calls"] != 0
    ):
        raise ValueError("candidate-lock manifest is incomplete")
    pool_audit = []
    for row in locks["pools"]:
        path = root / row["path"]
        verify_file(path, row["file_sha256"])
        pool = unseal(path)
        if (
            pool["candidate_count"] < 8
            or pool["score_fields_present"] is not False
            or pool["new_oracle_calls"] != 0
            or not all(
                candidate["exact_replay"] and candidate["validity"]
                for candidate in pool["candidates"]
            )
        ):
            raise ValueError(f"candidate lock failed: {path}")
        pool_audit.append(
            {
                "path": row["path"],
                "candidate_count": pool["candidate_count"],
                "exact_execution_precision": 1.0,
                "unique_endpoints": len(
                    {
                        candidate["canonical_endpoint"]
                        for candidate in pool["candidates"]
                    }
                ),
            }
        )
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "decision": "ZERO_ORACLE_PREPARATION_PASSED_NOT_AUTHORIZED_TO_SCORE",
        "contract_payload_sha256": contract["payload_sha256"],
        "qualified_environment": qualified,
        "production_result_payload_sha256": identity(production),
        "candidate_lock_manifest_payload_sha256": identity(locks),
        "pool_audit": pool_audit,
        "pinned_environment_decision_equivalence": {
            "criterion": (
                "all two-task v0 and additive locks provide at least eight unique "
                "valid exact endpoints; endpoint byte identity with the v3 development "
                "environment is not claimed"
            ),
            "passed": True,
        },
        "new_oracle_calls": 0,
        "scored_launch_authorized": False,
        "next_safe_action": (
            "run synthetic ledger/resume verification, commit preflight and locks, "
            "amend AGENTS and obtain the exact 384-call authorization"
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": importlib.metadata.version("numpy"),
        },
    }
    return seal(root / PREFLIGHT, payload)


def status(root: Path) -> dict[str, Any]:
    """Read-only, RNG-free preparation and future-run status."""
    result = {
        "contract": str(root / CONTRACT),
        "contract_present": (root / CONTRACT).exists(),
        "preflight_present": (root / PREFLIGHT).exists(),
        "candidate_manifest_present": (root / LOCK_MANIFEST).exists(),
        "oracle_calls_authorized": 0,
        "scored_launch_authorized": False,
        "units": [],
    }
    if (root / LOCK_MANIFEST).exists():
        manifest = unseal(root / LOCK_MANIFEST)
        result["candidate_pools"] = manifest["pool_count"]
        result["candidate_lock_complete"] = manifest["all_locked_before_scoring"]
    for task in TASKS:
        for arm in ARMS:
            folder = root / RUN_DIRECTORY / task / arm
            progress_path = folder / "progress.json"
            row = {"task": task, "arm": arm, "started": folder.exists()}
            if progress_path.exists():
                row["progress"] = json.loads(progress_path.read_text())
            if (folder / "result.json").exists():
                row["result"] = unseal(folder / "result.json")
            result["units"].append(row)
    return result


__all__ = [
    "LOCK_MANIFEST",
    "PREFLIGHT",
    "aggregate_scored_units",
    "build_preflight",
    "build_scored_launch_receipt",
    "pmo_program_features",
    "prepare_candidate_locks",
    "prepare_runtime_bundles",
    "run_locked_unit",
    "seal",
    "status",
    "unseal",
]
