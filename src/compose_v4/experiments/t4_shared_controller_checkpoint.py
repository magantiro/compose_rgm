"""Pure, fail-closed checkpoint transitions for one shared T4 controller cell.

This module performs no proposal generation, docking, network, Modal, or file
operation.  It advances a cell only across already settled query-lock boundaries.
The durable orchestration layer remains responsible for publishing the returned
checkpoint and immutable round plan.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    integrated_features,
    merge_expert_pools,
    select_batch,
)
from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    make_query_lock,
    validate_query_lock,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)

CHECKPOINT_SCHEMA = "t4_shared_controller_checkpoint_v1"
ROUND_PLAN_SCHEMA = "t4_shared_controller_round_plan_v1"

RUNNING = "running"
COMPLETE_BUDGET = "complete_budget"
CANDIDATE_EXHAUSTION = "candidate_exhaustion"
_STATUSES = {RUNNING, COMPLETE_BUDGET, CANDIDATE_EXHAUSTION}

_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "terminal_reason",
    "cell_key",
    "source_smiles",
    "delta",
    "budget_ceiling",
    "charged_count",
    "budget_remaining",
    "archive",
    "rounds_completed",
    "history",
    "rounds",
    "value_features",
    "value_improvements",
    "controller_config",
    "controller_config_identity",
    "controller_seed",
    "rng_state",
    "root_query",
    "last_settled_query_lock_identity",
}


@dataclass(frozen=True)
class RestoredController:
    """Validated runtime objects reconstructed only from serialized state."""

    checkpoint: dict[str, Any]
    state: SearchState
    value: ProgramValue
    rng: np.random.Generator
    config: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not deterministic JSON") from error
    return json.loads(encoded)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _validate_controller_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("controller config must be a mapping")
    normalized = _canonical_copy(dict(config), label="controller config")
    required = {
        "parents",
        "parent_explore",
        "batch",
        "exploration",
        "expert_floor_rounds",
        "route_scale_floor_rounds",
        "value_penalty",
    }
    missing = sorted(required.difference(normalized))
    if missing:
        raise ValueError(f"controller config is missing fields: {missing}")
    _integer(normalized["parents"], label="controller parents", minimum=1)
    batch = _integer(normalized["batch"], label="controller batch", minimum=1)
    if batch != 8:
        raise ValueError("shared controller query locks require batch=8")
    exploration = _integer(normalized["exploration"], label="controller exploration")
    if exploration > batch:
        raise ValueError("controller exploration may not exceed batch")
    for key in ("expert_floor_rounds", "route_scale_floor_rounds"):
        _integer(normalized[key], label=f"controller {key}")
    parent_explore = _finite_number(
        normalized["parent_explore"], label="controller parent_explore"
    )
    if not 0.0 <= parent_explore <= 1.0:
        raise ValueError("controller parent_explore must be within zero and one")
    penalty = _finite_number(
        normalized["value_penalty"], label="controller value_penalty"
    )
    if penalty < 0.0:
        raise ValueError("controller value_penalty must be nonnegative")
    counts = normalized.get("route_scale_floor_counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise TypeError("route_scale_floor_counts must be a mapping")
        unknown = sorted(set(counts).difference({"small", "medium", "large"}))
        if unknown:
            raise ValueError(f"unknown route-scale floor bands: {unknown}")
        for band, count in counts.items():
            _integer(count, label=f"route scale floor {band}")
    return normalized


def _validate_expected_binding(
    checkpoint: Mapping[str, Any],
    *,
    cell_key: str,
    source_smiles: str,
    delta: float,
    budget_ceiling: int,
    controller_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = _validate_controller_config(controller_config)
    if checkpoint.get("cell_key") != cell_key:
        raise ValueError("checkpoint cell mismatch")
    if checkpoint.get("source_smiles") != source_smiles:
        raise ValueError("checkpoint source mismatch")
    expected_delta = _finite_number(delta, label="expected delta")
    if checkpoint.get("delta") != expected_delta:
        raise ValueError("checkpoint delta mismatch")
    if checkpoint.get("budget_ceiling") != budget_ceiling:
        raise ValueError("checkpoint budget ceiling mismatch")
    if checkpoint.get("controller_config") != config:
        raise ValueError("checkpoint controller config drift")
    if checkpoint.get("controller_config_identity") != payload_identity(config):
        raise ValueError("checkpoint controller config identity drift")
    return config


def initialize_after_settled_root(
    *,
    cell_key: str,
    source_smiles: str,
    delta: float,
    budget_ceiling: int,
    controller_config: Mapping[str, Any],
    controller_seed: int,
    root_query_lock: Mapping[str, Any],
    settled_root: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the first durable checkpoint after one successful settled root."""

    if not isinstance(cell_key, str) or not cell_key:
        raise ValueError("cell key must be nonempty")
    if not isinstance(source_smiles, str) or not source_smiles:
        raise ValueError("source SMILES must be nonempty")
    delta_value = _finite_number(delta, label="delta")
    ceiling = _integer(budget_ceiling, label="budget ceiling", minimum=1)
    if ceiling != 49:
        raise ValueError("nine-cell shared controller requires a 49-call ceiling")
    seed = _integer(controller_seed, label="controller seed")
    config = _validate_controller_config(controller_config)

    lock = _canonical_copy(dict(root_query_lock), label="root query lock")
    validate_query_lock(lock, charged_call_ceiling=ceiling)
    if (
        lock.get("cell_key") != cell_key
        or lock.get("round") != 0
        or lock.get("charged_before") != 0
        or len(lock["queries"]) != 1
        or lock["queries"][0].get("smiles") != source_smiles
    ):
        raise ValueError("settled root lock does not match its cell and source")
    settlement = _canonical_copy(dict(settled_root), label="root settlement")
    if (
        settlement.get("action") != "recover"
        or settlement.get("charged_after") != 1
        or len(settlement.get("observations", ())) != 1
    ):
        raise RuntimeError("root query lock is not completely settled")
    observation = settlement["observations"][0]
    query = lock["queries"][0]
    if (
        observation.get("query_id") != query["query_id"]
        or observation.get("smiles") != source_smiles
    ):
        raise ValueError("settled root observation does not match its lock")
    score = _finite_number(observation.get("score"), label="settled root score")
    if observation.get("failure") not in (None, ""):
        raise ValueError("settled root may not contain an evaluator failure")

    rng = np.random.default_rng(seed)
    lock_identity = payload_identity(lock)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": RUNNING if ceiling > 1 else COMPLETE_BUDGET,
        "terminal_reason": None if ceiling > 1 else "budget_exhausted",
        "cell_key": cell_key,
        "source_smiles": source_smiles,
        "delta": delta_value,
        "budget_ceiling": ceiling,
        "charged_count": 1,
        "budget_remaining": ceiling - 1,
        "archive": {source_smiles: score},
        "rounds_completed": 0,
        "history": [],
        "rounds": [],
        "value_features": [],
        "value_improvements": [],
        "controller_config": config,
        "controller_config_identity": payload_identity(config),
        "controller_seed": seed,
        "rng_state": _jsonable(rng.bit_generator.state),
        "root_query": {
            "query_id": query["query_id"],
            "query_lock_payload_sha256": lock_identity,
            "score": score,
        },
        "last_settled_query_lock_identity": lock_identity,
    }
    return restore_checkpoint(
        checkpoint,
        cell_key=cell_key,
        source_smiles=source_smiles,
        delta=delta_value,
        budget_ceiling=ceiling,
        controller_config=config,
    ).checkpoint


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    cell_key: str,
    source_smiles: str,
    delta: float,
    budget_ceiling: int,
    controller_config: Mapping[str, Any],
) -> RestoredController:
    """Validate and reconstruct all scientific state, failing on any drift."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    normalized = _canonical_copy(dict(checkpoint), label="checkpoint")
    if set(normalized) != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS.difference(normalized))
        extra = sorted(set(normalized).difference(_CHECKPOINT_KEYS))
        raise ValueError(f"checkpoint field drift; missing={missing}, extra={extra}")
    if normalized["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema drift")
    config = _validate_expected_binding(
        normalized,
        cell_key=cell_key,
        source_smiles=source_smiles,
        delta=delta,
        budget_ceiling=budget_ceiling,
        controller_config=controller_config,
    )
    status = normalized["status"]
    if status not in _STATUSES:
        raise ValueError("checkpoint terminal status is invalid")
    seed = _integer(normalized["controller_seed"], label="controller seed")
    charged = _integer(normalized["charged_count"], label="charged count", minimum=1)
    remaining = _integer(normalized["budget_remaining"], label="budget remaining")
    if charged > budget_ceiling or remaining != budget_ceiling - charged:
        raise ValueError("checkpoint budget accounting drift")

    archive = normalized["archive"]
    if not isinstance(archive, dict) or source_smiles not in archive:
        raise ValueError("checkpoint archive omitted its exact source")
    normalized_archive = {
        str(smiles): _finite_number(score, label=f"archive score for {smiles}")
        for smiles, score in archive.items()
    }
    if list(archive) != sorted(archive):
        raise ValueError("checkpoint archive must be canonically sorted")

    rounds_completed = _integer(
        normalized["rounds_completed"], label="rounds completed"
    )
    rounds = normalized["rounds"]
    history = normalized["history"]
    if not isinstance(rounds, list) or not isinstance(history, list):
        raise TypeError("checkpoint rounds and history must be lists")
    if len(rounds) != len(history) or len(rounds) != rounds_completed:
        raise ValueError("checkpoint round and history lengths drift")
    if [row.get("round") for row in rounds] != list(range(1, rounds_completed + 1)):
        raise ValueError("checkpoint rounds are skipped or rewound")
    if [row.get("round") for row in history] != list(range(1, rounds_completed + 1)):
        raise ValueError("checkpoint history is skipped or rewound")
    for round_row, history_row in zip(rounds, history, strict=True):
        if history_row != {
            "round": round_row["round"],
            "improved": round_row.get("improved"),
        }:
            raise ValueError("checkpoint history disagrees with its round ledger")

    root_query = normalized["root_query"]
    if not isinstance(root_query, dict) or set(root_query) != {
        "query_id",
        "query_lock_payload_sha256",
        "score",
    }:
        raise ValueError("checkpoint root query provenance drift")
    root_score = _finite_number(root_query["score"], label="root query score")
    if normalized_archive[source_smiles] != root_score:
        raise ValueError("checkpoint root score disagrees with archive")
    lock_identities = [root_query["query_lock_payload_sha256"]]
    charged_from_rounds = 1
    successful_observations = 0
    prior_charged = 1
    reconstructed_archive = {source_smiles: root_score}
    for row in rounds:
        required = {
            "round",
            "parents",
            "query_lock_payload_sha256",
            "query_ids",
            "charged_before",
            "charged_this_round",
            "charged_after",
            "observations",
            "improved",
            "best_so_far",
            "value_training_rows",
        }
        if set(row) != required:
            raise ValueError("checkpoint round ledger field drift")
        if row["charged_before"] != prior_charged:
            raise ValueError("checkpoint round charged count is not contiguous")
        count = _integer(
            row["charged_this_round"], label="round charged count", minimum=1
        )
        if row["charged_after"] != prior_charged + count:
            raise ValueError("checkpoint round budget accounting drift")
        query_ids = row["query_ids"]
        observations = row["observations"]
        if (
            not isinstance(query_ids, list)
            or not isinstance(observations, list)
            or len(query_ids) != count
            or len(observations) != count
            or len(set(query_ids)) != count
            or [item.get("query_id") for item in observations] != query_ids
        ):
            raise ValueError("checkpoint round query identities drift")
        for observation in observations:
            if observation.get("score") is not None:
                smiles = observation.get("smiles")
                if not isinstance(smiles, str) or not smiles:
                    raise ValueError(
                        "checkpoint round observation omitted its molecule"
                    )
                if smiles in reconstructed_archive:
                    raise ValueError("checkpoint round ledger redocked a molecule")
                reconstructed_archive[smiles] = _finite_number(
                    observation["score"], label="round observation score"
                )
                successful_observations += 1
        charged_from_rounds += count
        prior_charged = row["charged_after"]
        lock_identities.append(row["query_lock_payload_sha256"])
        if row["value_training_rows"] != successful_observations:
            raise ValueError("checkpoint value-training count drift")
        if row["best_so_far"] != min(reconstructed_archive.values()):
            raise ValueError("round incumbent disagrees with its observation ledger")
    if charged_from_rounds != charged:
        raise ValueError("checkpoint charged count disagrees with round ledger")
    if len(lock_identities) != len(set(lock_identities)):
        raise ValueError("checkpoint reused a settled query lock")
    if normalized["last_settled_query_lock_identity"] != lock_identities[-1]:
        raise ValueError("checkpoint last settled query-lock identity drift")
    if reconstructed_archive != normalized_archive:
        raise ValueError("checkpoint archive disagrees with observation ledger")

    features = normalized["value_features"]
    improvements = normalized["value_improvements"]
    if not isinstance(features, list) or not isinstance(improvements, list):
        raise TypeError("checkpoint value observations must be lists")
    if len(features) != len(improvements) or len(features) != successful_observations:
        raise ValueError("checkpoint value observations are misaligned")
    feature_width = None
    for index, row in enumerate(features):
        if not isinstance(row, list) or not row:
            raise ValueError("checkpoint value feature row is empty")
        converted = [
            _finite_number(value, label=f"value feature {index}") for value in row
        ]
        feature_width = len(converted) if feature_width is None else feature_width
        if len(converted) != feature_width:
            raise ValueError("checkpoint value feature width drift")
    for value in improvements:
        _finite_number(value, label="value improvement")

    if remaining == 0:
        if (
            status != COMPLETE_BUDGET
            or normalized["terminal_reason"] != "budget_exhausted"
        ):
            raise ValueError("exhausted checkpoint is not terminal")
    elif status == COMPLETE_BUDGET:
        raise ValueError("budget-complete checkpoint retains uncharged budget")
    elif status == RUNNING and normalized["terminal_reason"] is not None:
        raise ValueError("running checkpoint has a terminal reason")
    elif status == CANDIDATE_EXHAUSTION and normalized["terminal_reason"] != (
        "no_unseen_candidates"
    ):
        raise ValueError("candidate-exhaustion checkpoint reason drift")

    rng = np.random.default_rng(seed)
    try:
        rng.bit_generator.state = normalized["rng_state"]
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint NumPy RNG state is invalid") from error
    state = SearchState(
        archive=normalized_archive,
        budget=remaining,
        rounds=rounds_completed,
        history=copy.deepcopy(history),
    )
    value = ProgramValue(penalty=float(config["value_penalty"]))
    if features:
        value.fit(features, improvements)
    return RestoredController(normalized, state, value, rng, config)


def _normalize_proposal_pools(
    proposal_pools: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    state: SearchState,
    parents: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(proposal_pools, Mapping):
        raise TypeError("proposal pools must be a mapping")
    unknown = sorted(set(proposal_pools).difference(EXPERTS))
    if unknown:
        raise ValueError(f"unknown proposal pools: {unknown}")
    parent_set = set(parents)
    normalized = {expert: [] for expert in EXPERTS}
    for expert in EXPERTS:
        rows = proposal_pools.get(expert, ())
        for source in rows:
            if not isinstance(source, Mapping):
                raise TypeError("proposal row must be a mapping")
            row = _canonical_copy(dict(source), label="proposal row")
            parent = row.get("parent")
            if parent not in state.archive:
                raise ValueError("proposal references a foreign parent")
            if row.get("parent_score") != state.archive[parent]:
                raise ValueError("proposal parent score disagrees with archive")
            if parent not in parent_set:
                continue
            fingerprint = row.get("fingerprint")
            if not isinstance(fingerprint, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in fingerprint
            ):
                raise ValueError("proposal fingerprint must be an integer list")
            row["fingerprint"] = set(fingerprint)
            normalized[expert].append(row)
    return normalized


def _serialize_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for source in rows:
        row = dict(source)
        row["features"] = np.asarray(row["features"], dtype=float).tolist()
        row["fingerprint"] = sorted(row["fingerprint"])
        serialized.append(_canonical_copy(row, label="selected proposal"))
    return serialized


def select_parent_and_batch(
    checkpoint: Mapping[str, Any],
    proposal_pools: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    cell_key: str,
    source_smiles: str,
    delta: float,
    budget_ceiling: int,
    controller_config: Mapping[str, Any],
    query_deadline: float,
) -> dict[str, Any]:
    """Select the next parents and batch, returning an immutable round plan.

    Supplied proposal rows may cover any archived parent.  Rows outside the
    deterministic selected parent set are ignored; rows for non-archive parents
    or with mismatched parent scores fail closed.
    """

    restored = restore_checkpoint(
        checkpoint,
        cell_key=cell_key,
        source_smiles=source_smiles,
        delta=delta,
        budget_ceiling=budget_ceiling,
        controller_config=controller_config,
    )
    if restored.checkpoint["status"] != RUNNING:
        raise RuntimeError("terminal checkpoint may not select another round")
    deadline = _finite_number(query_deadline, label="query deadline")
    config = restored.config
    state = restored.state
    rng = restored.rng
    round_index = state.rounds + 1
    parents = state.parents(
        limit=int(config["parents"]),
        rng=rng,
        explore=float(config["parent_explore"]),
    )
    pools = _normalize_proposal_pools(proposal_pools, state=state, parents=parents)
    merged = merge_expert_pools(pools)
    candidates = []
    for source in merged:
        if source["smiles"] in state.archive:
            continue
        row = dict(source)
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = set(row["fingerprint"])
        candidates.append(row)
    room = min(int(config["batch"]), state.budget)
    selected = select_batch(
        candidates,
        restored.value,
        state,
        rng,
        round_index=round_index,
        batch=room,
        exploration=min(int(config["exploration"]), room),
        expert_floor_rounds=int(config["expert_floor_rounds"]),
        route_scale_floor_rounds=int(config["route_scale_floor_rounds"]),
        route_scale_floor_counts=config.get("route_scale_floor_counts"),
    )
    if not selected:
        terminal = copy.deepcopy(restored.checkpoint)
        terminal.update(
            {
                "status": CANDIDATE_EXHAUSTION,
                "terminal_reason": "no_unseen_candidates",
                "rng_state": _jsonable(rng.bit_generator.state),
            }
        )
        validated = restore_checkpoint(
            terminal,
            cell_key=cell_key,
            source_smiles=source_smiles,
            delta=delta,
            budget_ceiling=budget_ceiling,
            controller_config=config,
        ).checkpoint
        return {
            "schema_version": ROUND_PLAN_SCHEMA,
            "status": CANDIDATE_EXHAUSTION,
            "checkpoint": validated,
        }

    selected_rows = _serialize_selected(selected)
    lock = make_query_lock(
        cell_key=cell_key,
        round_index=round_index,
        charged_before=restored.checkpoint["charged_count"],
        selected_rows=selected_rows,
        query_deadline=deadline,
        charged_call_ceiling=budget_ceiling,
        batch_size=int(config["batch"]),
    )
    return {
        "schema_version": ROUND_PLAN_SCHEMA,
        "status": "pending_settlement",
        "base_checkpoint_payload_sha256": payload_identity(restored.checkpoint),
        "cell_key": cell_key,
        "source_smiles": source_smiles,
        "delta": float(delta),
        "controller_config_identity": restored.checkpoint["controller_config_identity"],
        "round": round_index,
        "parents": parents,
        "selected_rows": selected_rows,
        "query_lock": lock,
        "query_lock_payload_sha256": payload_identity(lock),
        "rng_state_after_selection": _jsonable(rng.bit_generator.state),
    }


def _validate_round_plan(
    plan: Mapping[str, Any], restored: RestoredController
) -> dict[str, Any]:
    normalized = _canonical_copy(dict(plan), label="round plan")
    required = {
        "schema_version",
        "status",
        "base_checkpoint_payload_sha256",
        "cell_key",
        "source_smiles",
        "delta",
        "controller_config_identity",
        "round",
        "parents",
        "selected_rows",
        "query_lock",
        "query_lock_payload_sha256",
        "rng_state_after_selection",
    }
    if set(normalized) != required or normalized.get("schema_version") != (
        ROUND_PLAN_SCHEMA
    ):
        raise ValueError("round plan schema or fields drift")
    if normalized.get("status") != "pending_settlement":
        raise RuntimeError("round plan is not awaiting a settled query lock")
    checkpoint = restored.checkpoint
    if (
        normalized["query_lock_payload_sha256"]
        == checkpoint["last_settled_query_lock_identity"]
    ):
        raise RuntimeError("query lock was already applied")
    if normalized["base_checkpoint_payload_sha256"] != payload_identity(checkpoint):
        raise ValueError("round plan was not derived from this checkpoint")
    if (
        normalized["cell_key"] != checkpoint["cell_key"]
        or normalized["source_smiles"] != checkpoint["source_smiles"]
        or normalized["delta"] != checkpoint["delta"]
        or normalized["controller_config_identity"]
        != checkpoint["controller_config_identity"]
    ):
        raise ValueError("round plan cell, source, delta, or config drift")
    if normalized["round"] != checkpoint["rounds_completed"] + 1:
        raise ValueError("round plan skipped or rewound a round")
    lock = normalized["query_lock"]
    validate_query_lock(lock, charged_call_ceiling=checkpoint["budget_ceiling"])
    if payload_identity(lock) != normalized["query_lock_payload_sha256"]:
        raise ValueError("round plan query-lock identity drift")
    if (
        lock["cell_key"] != checkpoint["cell_key"]
        or lock["round"] != normalized["round"]
        or lock["charged_before"] != checkpoint["charged_count"]
    ):
        raise ValueError("round plan query lock disagrees with checkpoint")
    selected = normalized["selected_rows"]
    if not isinstance(selected, list) or len(selected) != len(lock["queries"]):
        raise ValueError("round plan selected rows do not match query lock")
    fields = (
        "smiles",
        "selection_kind",
        "proposal_experts",
        "parent",
        "parent_score",
    )
    if any(
        any(row.get(field) != query.get(field) for field in fields)
        for row, query in zip(selected, lock["queries"], strict=True)
    ):
        raise ValueError("round plan selected rows drift from query lock")
    rng = np.random.default_rng(checkpoint["controller_seed"])
    try:
        rng.bit_generator.state = normalized["rng_state_after_selection"]
    except (TypeError, ValueError) as error:
        raise ValueError("round plan NumPy RNG state is invalid") from error
    return normalized


def apply_settled_observations(
    checkpoint: Mapping[str, Any],
    round_plan: Mapping[str, Any],
    settled_batch: Mapping[str, Any],
    *,
    cell_key: str,
    source_smiles: str,
    delta: float,
    budget_ceiling: int,
    controller_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one terminal query-lock settlement exactly once."""

    restored = restore_checkpoint(
        checkpoint,
        cell_key=cell_key,
        source_smiles=source_smiles,
        delta=delta,
        budget_ceiling=budget_ceiling,
        controller_config=controller_config,
    )
    if restored.checkpoint["status"] != RUNNING:
        raise RuntimeError("terminal checkpoint may not accept observations")
    plan = _validate_round_plan(round_plan, restored)
    settlement = _canonical_copy(dict(settled_batch), label="settled observation batch")
    if settlement.get("action") not in {"recover", "recover_with_unresolved"}:
        raise RuntimeError("query lock is unresolved and may not advance checkpoint")
    observations = settlement.get("observations")
    statuses = settlement.get("receipt_statuses")
    lock = plan["query_lock"]
    queries = lock["queries"]
    if not isinstance(observations, list) or not isinstance(statuses, list):
        raise TypeError("settled batch omitted observations or receipt statuses")
    if len(observations) != len(queries) or len(statuses) != len(queries):
        raise ValueError("settled batch row count differs from query lock")
    query_ids = [query["query_id"] for query in queries]
    observed_ids = [row.get("query_id") for row in observations]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("settled batch contains duplicate query IDs")
    if observed_ids != query_ids:
        raise ValueError("settled batch contains foreign or reordered query IDs")
    expected_charged = restored.checkpoint["charged_count"] + len(queries)
    if settlement.get("charged_after") != expected_charged:
        raise ValueError("settled batch charged count drift")

    selected = plan["selected_rows"]
    state = restored.state
    previous = state.incumbent
    features = copy.deepcopy(restored.checkpoint["value_features"])
    improvements = copy.deepcopy(restored.checkpoint["value_improvements"])
    round_observations = []
    for row, query, observation in zip(selected, queries, observations, strict=True):
        for key in (
            "query_id",
            "smiles",
            "selection_kind",
            "proposal_experts",
            "parent",
            "parent_score",
        ):
            expected = query.get(key)
            actual = observation.get(key)
            if actual != expected:
                raise ValueError(f"settled observation {key} drift")
        if row["smiles"] in state.archive:
            raise ValueError("settled batch attempts to redock an archived molecule")
        score = observation.get("score")
        failure = observation.get("failure")
        if score is None:
            if not isinstance(failure, str) or not failure:
                raise ValueError("unscored settled observation requires a failure")
            normalized_score = None
        else:
            normalized_score = _finite_number(score, label="settled observation score")
            if failure not in (None, ""):
                raise ValueError("scored settled observation may not report failure")
            feature_row = [
                _finite_number(value, label="selected value feature")
                for value in row["features"]
            ]
            features.append(feature_row)
            improvements.append(float(row["parent_score"] - normalized_score))
            state.archive[row["smiles"]] = normalized_score
        round_observations.append(
            {
                "query_id": query["query_id"],
                "smiles": query["smiles"],
                "score": normalized_score,
                "failure": failure,
                "selection_kind": query.get("selection_kind"),
                "proposal_experts": query.get("proposal_experts", []),
                "parent": query.get("parent"),
                "parent_score": query.get("parent_score"),
            }
        )

    round_index = plan["round"]
    state.budget = budget_ceiling - expected_charged
    state.rounds = round_index
    improved = state.incumbent < previous
    state.history.append({"round": round_index, "improved": improved})
    round_result = {
        "round": round_index,
        "parents": plan["parents"],
        "query_lock_payload_sha256": plan["query_lock_payload_sha256"],
        "query_ids": query_ids,
        "charged_before": restored.checkpoint["charged_count"],
        "charged_this_round": len(queries),
        "charged_after": expected_charged,
        "observations": round_observations,
        "improved": improved,
        "best_so_far": state.incumbent,
        "value_training_rows": len(improvements),
    }
    status = COMPLETE_BUDGET if state.budget == 0 else RUNNING
    result = {
        **copy.deepcopy(restored.checkpoint),
        "status": status,
        "terminal_reason": "budget_exhausted" if status == COMPLETE_BUDGET else None,
        "charged_count": expected_charged,
        "budget_remaining": state.budget,
        "archive": dict(sorted(state.archive.items())),
        "rounds_completed": round_index,
        "history": state.history,
        "rounds": [*restored.checkpoint["rounds"], round_result],
        "value_features": features,
        "value_improvements": improvements,
        "rng_state": plan["rng_state_after_selection"],
        "last_settled_query_lock_identity": plan["query_lock_payload_sha256"],
    }
    return restore_checkpoint(
        result,
        cell_key=cell_key,
        source_smiles=source_smiles,
        delta=delta,
        budget_ceiling=budget_ceiling,
        controller_config=controller_config,
    ).checkpoint


__all__ = [
    "CANDIDATE_EXHAUSTION",
    "CHECKPOINT_SCHEMA",
    "COMPLETE_BUDGET",
    "ROUND_PLAN_SCHEMA",
    "RUNNING",
    "RestoredController",
    "apply_settled_observations",
    "initialize_after_settled_root",
    "restore_checkpoint",
    "select_parent_and_batch",
]
