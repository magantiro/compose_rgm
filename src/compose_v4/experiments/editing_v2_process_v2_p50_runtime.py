"""Minimal Process-V2 P50 selection, preparation, and training runtime.

The bounded P50 pilot answers one question: after fifty scratch updates on the
balanced Active8 training stream, does every load-bearing family receive a
finite learning signal without regressing on a small validation-only sentinel?

This module deliberately avoids the historical P50 infrastructure.  It scans
authenticated Active8 metadata once, selects exactly 3,200 train rows, opens
only the unique selected train states and a 64-state validation sentinel, and
compiles those successor fibers once on CPU.  The GPU path performs no
chemistry or corpus scan.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_gate_zero import (
    ProcessV2GateZeroError,
    iter_process_v2_role_shards,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    DEVELOPMENT_CELL_ROLES,
    P50_RECIPE_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50ScopedPrerequisites,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    ProcessV2T1Source,
    resolve_process_v2_t1_entries,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    ProcessV2T1RuntimeError,
    build_process_v2_score_revised_scratch_runtime,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    ACTION_ROUTE_PREFIXES,
    TOTAL_HAZARD_PREFIX,
    _assert_hazard_identity,
    _attach_successor_family_coordinates,
    _clone_state_dict,
    _hazard_state,
    _index_factorized_batch,
    _rng_state,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_from_payload,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    SuccessorTrainingError,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
    compile_teacher_successor_fibers_support_only,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SELECTION_SCHEMA = "compose.editing_v2.process_v2_p50_selection"
SELECTION_SCHEMA_VERSION = 1
SELECTION_STATUS = "PROCESS_V2_P50_SELECTION_NO_DOWNSTREAM_AUTHORITY"
PREPARED_SCHEMA = "compose.editing_v2.process_v2_p50_prepared_inputs"
PREPARED_SCHEMA_VERSION = 2
PREPARED_STATUS = "PROCESS_V2_P50_PREPARED_INPUTS_NO_DOWNSTREAM_AUTHORITY"
SELECTION_FILENAME = "PROCESS_V2_P50_SELECTION.json"
PREPARED_FILENAME = "PROCESS_V2_P50_PREPARED_INPUTS.json"
VALIDATION_EXAMPLES_PER_SUPPORTED_CELL = 4
VALIDATION_ROLE = "validation"
TRAIN_ROLE = "train"
_COMPILE_SUPPORT_TIME = 0.5


class ProcessV2P50RuntimeError(RuntimeError):
    """The bounded Process-V2 P50 stream, input, or training run is invalid."""


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2P50RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _alias_payload(alias: TeacherSuccessorAlias) -> dict[str, Any]:
    return {
        "family_name": alias.family_name,
        "table_name": alias.table_name,
        "coordinate": list(alias.coordinate),
    }


def _alias_from_payload(value: object) -> TeacherSuccessorAlias:
    if not isinstance(value, Mapping) or set(value) != {
        "family_name",
        "table_name",
        "coordinate",
    }:
        raise ProcessV2P50RuntimeError("a compact P50 successor alias is malformed")
    coordinate = value["coordinate"]
    if (
        not isinstance(value["family_name"], str)
        or not value["family_name"]
        or not isinstance(value["table_name"], str)
        or not value["table_name"]
        or not isinstance(coordinate, list)
        or not coordinate
        or any(type(item) is not int or item < 0 for item in coordinate)
    ):
        raise ProcessV2P50RuntimeError("a compact P50 successor coordinate is malformed")
    return TeacherSuccessorAlias(
        family_name=str(value["family_name"]),
        table_name=str(value["table_name"]),
        coordinate=tuple(coordinate),
    )


def _teacher_fiber_payload(fiber: TeacherSuccessorFiber) -> dict[str, Any]:
    return {
        "source_key": fiber.source_key,
        "target_key": fiber.target_key,
        "target_state_sha256": fiber.target_state_sha256,
        "aliases": [_alias_payload(alias) for alias in fiber.aliases],
        "state_support": {
            "source_key": fiber.state_support.source_key,
            "source_state_sha256": fiber.state_support.source_state_sha256,
            "virtual_aliases": [
                _alias_payload(alias) for alias in fiber.state_support.virtual_aliases
            ],
        },
    }


def _teacher_fiber_from_payload(value: object) -> TeacherSuccessorFiber:
    if not isinstance(value, Mapping) or set(value) != {
        "source_key",
        "target_key",
        "target_state_sha256",
        "aliases",
        "state_support",
    }:
        raise ProcessV2P50RuntimeError("a compact P50 teacher fiber is malformed")
    support = value["state_support"]
    aliases = value["aliases"]
    if (
        not isinstance(support, Mapping)
        or set(support) != {"source_key", "source_state_sha256", "virtual_aliases"}
        or not isinstance(aliases, list)
        or not isinstance(support["virtual_aliases"], list)
    ):
        raise ProcessV2P50RuntimeError("a compact P50 state support is malformed")
    try:
        state_support = StateProductiveSupport(
            source_key=str(support["source_key"]),
            source_state_sha256=_require_sha(
                support["source_state_sha256"], field="P50 source state"
            ),
            virtual_aliases=tuple(
                _alias_from_payload(alias) for alias in support["virtual_aliases"]
            ),
        )
        return TeacherSuccessorFiber(
            source_key=str(value["source_key"]),
            target_key=str(value["target_key"]),
            target_state_sha256=_require_sha(
                value["target_state_sha256"], field="P50 target state"
            ),
            aliases=tuple(_alias_from_payload(alias) for alias in aliases),
            state_support=state_support,
        )
    except ValueError as error:
        raise ProcessV2P50RuntimeError("a compact P50 teacher fiber is invalid") from error


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "task_identity_sha256": candidate["task_identity_sha256"],
            "entry_index": candidate["entry_index"],
            "trace_id": candidate["trace_id"],
            "progress_index": candidate["progress_index"],
            "assignment_sha256": candidate["assignment_sha256"],
        }
    )


def _candidate_from_transition(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = row.get("candidate_evidence")
    if not isinstance(evidence, Mapping):
        raise ProcessV2P50RuntimeError("an accepted P50 transition lacks candidate evidence")
    if (
        row.get("terminal") is not False
        or evidence.get("supported") is not True
        or evidence.get("teacher_coordinate_legal") is not True
        or evidence.get("teacher_executes_to_exact_successor") is not True
        or evidence.get("productive_canonical_successor") is not True
        or evidence.get("exclusion_reason") is not None
    ):
        raise ProcessV2P50RuntimeError(
            "an Active8-accepted P50 transition violates its frozen evidence contract"
        )
    body = {
        "v1_task_identity_sha256": str(row["v1_task_identity_sha256"]),
        "task_identity_sha256": str(row["task_identity_sha256"]),
        "entry_index": int(row["entry_index"]),
        "trace_id": str(row["trace_id"]),
        "step_index": int(row["step_index"]),
        "progress_index": int(row["progress_index"]),
        "partition_role": str(row["partition_role"]),
        "data_lane": str(row["data_lane"]),
        "executor_rule": str(row["executor_rule"]),
        "model_family": str(row["model_family"]),
        "capability_cell_id": str(row["capability_cell_id"]),
        "source_state_sha256": _require_sha(
            evidence["source_state_sha256"], field="source_state_sha256"
        ),
        "target_state_sha256": _require_sha(
            evidence["target_state_sha256"], field="target_state_sha256"
        ),
        "source_canonical_key": str(evidence["source_canonical_key"]),
        "canonical_successor_key": str(evidence["canonical_successor_key"]),
        "action_sha256": _require_sha(evidence["action_sha256"], field="action_sha256"),
        "assignment_sha256": _require_sha(row["assignment_sha256"], field="assignment_sha256"),
    }
    return {**body, "p50_entry_sha256": _candidate_identity(body)}


def _required_cells(repo_root: Path) -> tuple[str, ...]:
    roles = load_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=repo_root)
    return tuple(str(cell) for cell in roles["required_cell_ids"])


def _rank(
    candidate: Mapping[str, Any],
    *,
    policy_sha256: str,
    purpose: str,
    cycle_index: int = 0,
) -> str:
    return canonical_sha256(
        {
            "algorithm": "process_v2_p50_candidate_hash_permutation_v1",
            "policy_sha256": policy_sha256,
            "purpose": purpose,
            "cycle_index": cycle_index,
            "capability_cell_id": candidate["capability_cell_id"],
            "p50_entry_sha256": candidate["p50_entry_sha256"],
        }
    )


def _time_hex(*, stream_index: int, candidate: Mapping[str, Any], seed: int) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "algorithm": "sha256_counter_open_unit_interval_53bit_v1",
                "seed": seed,
                "stream_index": stream_index,
                "p50_entry_sha256": candidate["p50_entry_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    numerator = (int.from_bytes(digest[:8], "big") >> 11) + 1
    value = numerator / ((1 << 53) + 1)
    if not 0.0 < value < 1.0:
        raise AssertionError("P50 time derivation escaped its open interval")
    return value.hex()


def _iter_role_candidates(
    source: ProcessV2T1Source, *, partition_role: str
) -> Iterable[dict[str, Any]]:
    try:
        shards = iter_process_v2_role_shards(
            source.active8_run_root,
            contracts=source.contracts,
            index=source.index,
            partition_role=partition_role,
        )
        for _shard, rows in shards:
            for row in rows:
                yield _candidate_from_transition(row)
    except ProcessV2GateZeroError as error:
        raise ProcessV2P50RuntimeError(str(error)) from error


def _training_schedule(
    source: ProcessV2T1Source,
    *,
    required_cells: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    total = int(policy["optimization"]["scheduled_nonterminal_examples"])
    cells = tuple(required_cells)
    maximum_per_cell = math.ceil(total / len(cells))
    kept: dict[str, list[tuple[int, str, dict[str, Any]]]] = {cell: [] for cell in cells}
    counts: Counter[str] = Counter()
    for candidate in _iter_role_candidates(source, partition_role=TRAIN_ROLE):
        cell = str(candidate["capability_cell_id"])
        if cell not in kept:
            continue
        counts[cell] += 1
        rank = _rank(
            candidate,
            policy_sha256=str(policy["contract_sha256"]),
            purpose="train",
        )
        item = (-int(rank, 16), str(candidate["p50_entry_sha256"]), candidate)
        ranked = kept[cell]
        if len(ranked) < maximum_per_cell:
            heapq.heappush(ranked, item)
        elif int(rank, 16) < -ranked[0][0]:
            heapq.heapreplace(ranked, item)
    missing = sorted(cell for cell in cells if counts[cell] == 0)
    if missing:
        raise ProcessV2P50RuntimeError(f"P50 train stream omits required semantic cells: {missing}")
    pools = {
        cell: sorted(
            (item[2] for item in kept[cell]),
            key=lambda candidate: (
                _rank(
                    candidate,
                    policy_sha256=str(policy["contract_sha256"]),
                    purpose="train",
                ),
                candidate["p50_entry_sha256"],
            ),
        )
        for cell in cells
    }
    rows: list[dict[str, Any]] = []
    seed = int(policy["optimization"]["seed"])
    for stream_index in range(total):
        cell = cells[stream_index % len(cells)]
        occurrence = stream_index // len(cells)
        population = int(counts[cell])
        cycle_index, offset = divmod(occurrence, population)
        if cycle_index == 0:
            ordered = pools[cell]
        else:
            if len(pools[cell]) != population:
                raise ProcessV2P50RuntimeError(
                    "bounded P50 selector discarded a candidate needed by a later cycle"
                )
            ordered = sorted(
                pools[cell],
                key=lambda candidate: (
                    _rank(
                        candidate,
                        policy_sha256=str(policy["contract_sha256"]),
                        purpose="train",
                        cycle_index=cycle_index,
                    ),
                    candidate["p50_entry_sha256"],
                ),
            )
        candidate = ordered[offset]
        body = {
            "stream_index": stream_index,
            "optimizer_step": stream_index // int(policy["optimization"]["batch_size"]),
            "batch_offset": stream_index % int(policy["optimization"]["batch_size"]),
            "p50_entry_sha256": candidate["p50_entry_sha256"],
            "model_family": candidate["model_family"],
            "capability_cell_id": cell,
            "time_hex": _time_hex(stream_index=stream_index, candidate=candidate, seed=seed),
            "objective_coefficient": 1,
        }
        rows.append({**body, "stream_row_sha256": canonical_sha256(body)})
    scheduled_entry_ids = {str(row["p50_entry_sha256"]) for row in rows}
    selected = {
        candidate["p50_entry_sha256"]: candidate
        for cell_pool in pools.values()
        for candidate in cell_pool
        if candidate["p50_entry_sha256"] in scheduled_entry_ids
    }
    return rows, [selected[key] for key in sorted(selected)], dict(sorted(counts.items()))


def _validation_panel(
    source: ProcessV2T1Source,
    *,
    required_cells: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    by_cell_source: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {
        cell: {} for cell in required_cells
    }
    raw_counts: Counter[str] = Counter()
    for candidate in _iter_role_candidates(source, partition_role=VALIDATION_ROLE):
        cell = str(candidate["capability_cell_id"])
        if cell not in by_cell_source:
            continue
        raw_counts[cell] += 1
        rank = _rank(
            candidate,
            policy_sha256=str(policy["contract_sha256"]),
            purpose="validation",
        )
        source_sha = str(candidate["source_state_sha256"])
        previous = by_cell_source[cell].get(source_sha)
        if previous is None or (rank, candidate["p50_entry_sha256"]) < (
            previous[0],
            previous[1]["p50_entry_sha256"],
        ):
            by_cell_source[cell][source_sha] = (rank, candidate)
    unsupported = [cell for cell in required_cells if not by_cell_source[cell]]
    supported = tuple(cell for cell in required_cells if cell not in set(unsupported))
    if not supported:
        raise ProcessV2P50RuntimeError("P50 validation role has no required-cell support")
    if any(
        len(by_cell_source[cell]) < VALIDATION_EXAMPLES_PER_SUPPORTED_CELL for cell in supported
    ):
        sparse = {
            cell: len(by_cell_source[cell])
            for cell in supported
            if len(by_cell_source[cell]) < VALIDATION_EXAMPLES_PER_SUPPORTED_CELL
        }
        raise ProcessV2P50RuntimeError(
            f"P50 validation support is below its prospective four-state floor: {sparse}"
        )

    slot_cells = [cell for cell in supported for _ in range(VALIDATION_EXAMPLES_PER_SUPPORTED_CELL)]
    options = {
        cell: [
            item[1]
            for item in sorted(
                by_cell_source[cell].values(),
                key=lambda pair: (pair[0], pair[1]["p50_entry_sha256"]),
            )
        ]
        for cell in supported
    }
    assigned_source_to_slot: dict[str, int] = {}
    assigned_candidate_by_slot: dict[int, dict[str, Any]] = {}

    def assign(slot: int, visited: set[str]) -> bool:
        cell = slot_cells[slot]
        for candidate in options[cell]:
            source_sha = str(candidate["source_state_sha256"])
            if source_sha in visited:
                continue
            visited.add(source_sha)
            previous_slot = assigned_source_to_slot.get(source_sha)
            if previous_slot is None or assign(previous_slot, visited):
                assigned_source_to_slot[source_sha] = slot
                assigned_candidate_by_slot[slot] = candidate
                return True
        return False

    for slot in range(len(slot_cells)):
        if not assign(slot, set()):
            raise ProcessV2P50RuntimeError(
                "P50 validation cells have no globally source-unique four-state matching"
            )
    selected = [assigned_candidate_by_slot[index] for index in range(len(slot_cells))]
    if len({row["source_state_sha256"] for row in selected}) != len(selected):
        raise AssertionError("P50 validation matching repeated an exact source")
    return selected, unsupported, dict(sorted(raw_counts.items()))


def build_process_v2_p50_selection(
    source: ProcessV2T1Source,
    *,
    prerequisites: ProcessV2P50ScopedPrerequisites,
) -> dict[str, Any]:
    """Scan authenticated metadata once and freeze train/validation addresses."""

    policy = load_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=source.repo_root)
    cells = _required_cells(source.repo_root)
    training_rows, train_entries, train_counts = _training_schedule(
        source, required_cells=cells, policy=policy
    )
    validation_entries, unsupported, validation_counts = _validation_panel(
        source, required_cells=cells, policy=policy
    )
    training_source_ids = {entry["source_state_sha256"] for entry in train_entries}
    validation_source_ids = {entry["source_state_sha256"] for entry in validation_entries}
    if training_source_ids & validation_source_ids:
        raise ProcessV2P50RuntimeError(
            "P50 train and validation selections share an exact source state"
        )
    unique_entries = {
        entry["p50_entry_sha256"]: entry for entry in (*train_entries, *validation_entries)
    }
    family_opportunities = Counter(row["model_family"] for row in training_rows)
    cell_opportunities = Counter(row["capability_cell_id"] for row in training_rows)
    family_step_opportunities = {
        family: len(
            {int(row["optimizer_step"]) for row in training_rows if row["model_family"] == family}
        )
        for family in sorted(family_opportunities)
    }
    cell_step_opportunities = {
        cell: len(
            {
                int(row["optimizer_step"])
                for row in training_rows
                if row["capability_cell_id"] == cell
            }
        )
        for cell in sorted(cell_opportunities)
    }
    if (
        len(training_rows) != prerequisites.optimizer_steps * prerequisites.batch_size
        or set(cell_opportunities) != set(cells)
        or set(family_opportunities) != set(prerequisites.active_families)
        or any(value <= 0 for value in cell_opportunities.values())
    ):
        raise ProcessV2P50RuntimeError("P50 schedule is not the complete balanced Active8 stream")
    body = {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": SELECTION_STATUS,
        **authority_false_block(),
        "prerequisites": prerequisites.as_payload(),
        "prerequisites_binding_sha256": prerequisites.binding_sha256,
        "p50_recipe_policy_sha256": policy["contract_sha256"],
        "active8_run_identity_sha256": source.plan["run_identity_sha256"],
        "required_cells": list(cells),
        "training_candidate_counts_by_cell": train_counts,
        "validation_candidate_counts_by_cell": validation_counts,
        "validation_examples_per_supported_cell": VALIDATION_EXAMPLES_PER_SUPPORTED_CELL,
        "validation_unsupported_required_cells": unsupported,
        "training_family_opportunities": dict(sorted(family_opportunities.items())),
        "training_cell_opportunities": dict(sorted(cell_opportunities.items())),
        "training_family_step_opportunities": family_step_opportunities,
        "training_cell_step_opportunities": cell_step_opportunities,
        "training_stream": training_rows,
        "training_stream_sha256": canonical_sha256(training_rows),
        "validation_entry_sha256s": sorted(
            entry["p50_entry_sha256"] for entry in validation_entries
        ),
        "validation_inventory_sha256": canonical_sha256(
            sorted(entry["p50_entry_sha256"] for entry in validation_entries)
        ),
        "unique_entry_count": len(unique_entries),
        "entries": [unique_entries[key] for key in sorted(unique_entries)],
    }
    return {**body, "selection_sha256": canonical_sha256(body)}


def validate_process_v2_p50_selection(value: object) -> dict[str, Any]:
    """Validate the bounded metadata selection without reopening chemistry."""

    if not isinstance(value, Mapping):
        raise ProcessV2P50RuntimeError("P50 selection must be an object")
    selection = dict(value)
    try:
        verify_self_hash(selection, field="selection_sha256", label="the P50 selection")
        require_authority_false(selection, label="the P50 selection")
    except ValueError as error:
        raise ProcessV2P50RuntimeError(str(error)) from error
    prerequisites = selection.get("prerequisites")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("status") != SELECTION_STATUS
        or selection.get("training_stream_sha256")
        != canonical_sha256(selection.get("training_stream"))
        or selection.get("validation_inventory_sha256")
        != canonical_sha256(selection.get("validation_entry_sha256s"))
        or selection.get("unique_entry_count") != len(selection.get("entries", ()))
        or not isinstance(prerequisites, Mapping)
        or selection.get("prerequisites_binding_sha256") != canonical_sha256(prerequisites)
        or selection.get("p50_recipe_policy_sha256")
        != prerequisites.get("p50_recipe_policy_sha256")
        or prerequisites.get("optimizer_steps") != 50
        or prerequisites.get("batch_size") != 64
        or not isinstance(selection.get("required_cells"), list)
        or len(selection["required_cells"]) != 17
        or len(selection["required_cells"]) != len(set(selection["required_cells"]))
    ):
        raise ProcessV2P50RuntimeError("P50 selection identity or census disagrees")
    entries = selection["entries"]
    if not isinstance(entries, list):
        raise ProcessV2P50RuntimeError("P50 selection entries must be a list")
    entry_ids = [str(entry["p50_entry_sha256"]) for entry in entries]
    if entry_ids != sorted(entry_ids) or len(entry_ids) != len(set(entry_ids)):
        raise ProcessV2P50RuntimeError("P50 selection entries repeat or are unordered")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ProcessV2P50RuntimeError("a P50 selection entry is not an object")
        if (
            entry.get("p50_entry_sha256") != _candidate_identity(entry)
            or entry.get("partition_role") not in {TRAIN_ROLE, VALIDATION_ROLE}
            or entry.get("p50_entry_sha256") in by_id
        ):
            raise ProcessV2P50RuntimeError("a P50 selection entry identity disagrees")
        by_id[str(entry["p50_entry_sha256"])] = entry
    stream = selection["training_stream"]
    if not isinstance(stream, list) or len(stream) != 3200:
        raise ProcessV2P50RuntimeError("P50 training stream is not exactly 3,200 rows")
    for index, row in enumerate(stream):
        try:
            time_value = float.fromhex(str(row.get("time_hex")))
        except (TypeError, ValueError) as error:
            raise ProcessV2P50RuntimeError("a P50 training time is invalid") from error
        body = {key: item for key, item in row.items() if key != "stream_row_sha256"}
        entry = by_id.get(str(row.get("p50_entry_sha256")))
        if (
            row.get("stream_index") != index
            or row.get("optimizer_step") != index // 64
            or row.get("batch_offset") != index % 64
            or row.get("stream_row_sha256") != canonical_sha256(body)
            or entry is None
            or entry.get("partition_role") != TRAIN_ROLE
            or row.get("model_family") != entry.get("model_family")
            or row.get("capability_cell_id") != entry.get("capability_cell_id")
            or row.get("objective_coefficient") != 1
            or not 0.0 < time_value < 1.0
        ):
            raise ProcessV2P50RuntimeError("a P50 training stream row disagrees")
    validation_ids = selection["validation_entry_sha256s"]
    if (
        not isinstance(validation_ids, list)
        or validation_ids != sorted(validation_ids)
        or len(validation_ids) != len(set(validation_ids))
        or any(
            identifier not in by_id or by_id[identifier]["partition_role"] != VALIDATION_ROLE
            for identifier in validation_ids
        )
    ):
        raise ProcessV2P50RuntimeError("P50 validation inventory disagrees")
    return selection


def compile_process_v2_p50_entries(
    selection: Mapping[str, Any],
    *,
    source: ProcessV2T1Source,
    task_identity_sha256: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile one source-chunk leaf for the entries the selection touches."""

    entries = [
        dict(entry)
        for entry in selection["entries"]
        if entry["task_identity_sha256"] == task_identity_sha256
    ]
    if not entries:
        raise ProcessV2P50RuntimeError("P50 compile task has no selected entries")
    resolver_entries = [
        {**entry, "panel_entry_sha256": entry["p50_entry_sha256"]} for entry in entries
    ]
    try:
        resolved = resolve_process_v2_t1_entries(source, resolver_entries)
        scratch, _binding, _score_receipt = build_process_v2_score_revised_scratch_runtime(source)
    except (ProcessV2T1PanelError, ProcessV2T1RuntimeError) as error:
        raise ProcessV2P50RuntimeError(str(error)) from error
    addressed_by_entry: dict[str, Any] = {}
    for item in resolved:
        for entry in item.panel_entries:
            addressed_by_entry[str(entry["panel_entry_sha256"])] = item.addressed_trace
    sources: list[Any] = []
    targets: list[Any] = []
    for entry in resolver_entries:
        addressed = addressed_by_entry[str(entry["panel_entry_sha256"])]
        progress = int(entry["progress_index"])
        source_state = addressed.path.state_at(progress)
        target_state = addressed.path.state_at(progress + 1)
        step = addressed.trace.steps[progress]
        observed_action_sha256 = rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
            schema_version=action_codec_v4.SCHEMA_VERSION,
        )
        if (
            observed_action_sha256 != entry["action_sha256"]
            or persistent_slot_state_sha256(source_state) != entry["source_state_sha256"]
            or persistent_slot_state_sha256(target_state) != entry["target_state_sha256"]
        ):
            raise ProcessV2P50RuntimeError(
                "a selected P50 action or exact state changed after selection"
            )
        sources.append(source_state)
        targets.append(target_state)
    try:
        compact_rows = compile_teacher_successor_fibers_support_only(
            scratch.model,
            tuple(sources),
            tuple(targets),
            teacher_action_sha256s=tuple(str(entry["action_sha256"]) for entry in resolver_entries),
            teacher_families=tuple(str(entry["model_family"]) for entry in resolver_entries),
            times=tuple(_COMPILE_SUPPORT_TIME for _entry in resolver_entries),
        )
    except (SuccessorTrainingError, ValueError) as error:
        raise ProcessV2P50RuntimeError(
            "could not compile compact exact P50 teacher-successor fibers"
        ) from error

    compiled: list[dict[str, Any]] = []
    for index, (entry, source_state, compact) in enumerate(
        zip(resolver_entries, sources, compact_rows, strict=True), start=1
    ):
        state_payload = encode_state(source_state)
        if encode_state(decode_state(state_payload)) != state_payload:
            raise ProcessV2P50RuntimeError("P50 exact-state encoding is not byte-stable")
        fiber = compact.teacher_fiber
        prepared_body = {
            "p50_entry_sha256": entry["p50_entry_sha256"],
            "model_family": entry["model_family"],
            "capability_cell_id": entry["capability_cell_id"],
            "partition_role": entry["partition_role"],
            "support_time_hex": float(_COMPILE_SUPPORT_TIME).hex(),
            "source_state_sha256": entry["source_state_sha256"],
            "target_state_sha256": entry["target_state_sha256"],
            "successor_canonical_key": entry["canonical_successor_key"],
            "teacher_action_sha256": entry["action_sha256"],
            "objective_coefficient": 1,
            "raw_mark_count": compact.raw_mark_count,
            "decoded_mark_count": compact.decoded_mark_count,
            "marks_by_family": dict(compact.marks_by_family),
            "production_successor_alias_multiplicity": len(fiber.aliases),
            "virtual_alias_count": len(fiber.state_support.virtual_aliases),
            "exact_state": state_payload,
            "teacher_successor_fiber": _teacher_fiber_payload(fiber),
            "exact_teacher_alias": _alias_payload(compact.exact_teacher_alias),
            "model_scores_or_probabilities_stored": False,
            "hazard_included": False,
        }
        prepared = {
            **prepared_body,
            "p50_compiled_entry_sha256": canonical_sha256(prepared_body),
        }
        compiled.append(prepared)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "process_v2_p50_compile_entry",
                    "task_identity_sha256": task_identity_sha256,
                    "completed": index,
                    "total": len(entries),
                }
            )
    compiled.sort(key=lambda row: row["p50_entry_sha256"])
    body = {
        "task_identity_sha256": task_identity_sha256,
        "selection_sha256": selection["selection_sha256"],
        "entry_count": len(compiled),
        "entries": compiled,
    }
    return {**body, "leaf_sha256": canonical_sha256(body)}


def convert_process_v2_p50_v1_leaf(
    selection: Mapping[str, Any], legacy_leaf: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert one exhaustive v1 leaf without re-enumerating chemistry."""

    leaf = dict(legacy_leaf)
    leaf_body = {key: item for key, item in leaf.items() if key != "leaf_sha256"}
    task_id = str(leaf.get("task_identity_sha256"))
    selected = {
        str(entry["p50_entry_sha256"]): entry
        for entry in selection["entries"]
        if entry["task_identity_sha256"] == task_id
    }
    if (
        set(leaf)
        != {
            "task_identity_sha256",
            "selection_sha256",
            "entry_count",
            "entries",
            "leaf_sha256",
        }
        or leaf.get("leaf_sha256") != canonical_sha256(leaf_body)
        or leaf.get("selection_sha256") != selection.get("selection_sha256")
        or leaf.get("entry_count") != len(leaf.get("entries", ()))
        or not selected
    ):
        raise ProcessV2P50RuntimeError("a legacy P50 leaf binding disagrees")

    converted: list[dict[str, Any]] = []
    for raw in leaf["entries"]:
        entry = dict(raw)
        old_body = {key: item for key, item in entry.items() if key != "p50_compiled_entry_sha256"}
        identifier = str(entry.get("p50_entry_sha256"))
        selection_entry = selected.get(identifier)
        try:
            state = decode_state(entry["exact_state"])
            partition = compiled_successor_map_from_payload(entry["successor_partition"])
            fiber = teacher_successor_fiber_from_exact_digest(
                partition, str(entry["target_state_sha256"])
            )
        except (KeyError, TypeError, ValueError, SuccessorTrainingError) as error:
            raise ProcessV2P50RuntimeError("a legacy P50 entry is invalid") from error
        all_marks = (
            tuple(mark for group in partition.successor_groups for mark in group.marks)
            + partition.virtual_marks
        )
        family_counts: dict[str, int] = {}
        for mark in all_marks:
            family = mark.alias.family_name
            family_counts[family] = family_counts.get(family, 0) + 1
        exact_aliases = tuple(
            mark.alias
            for group in partition.successor_groups
            for mark in group.marks
            if mark.successor_state_sha256 == entry.get("target_state_sha256")
            and mark.action_sha256 == entry.get("teacher_action_sha256")
            and mark.alias.family_name == entry.get("model_family")
        )
        raw_mark_count = len(all_marks)
        if (
            selection_entry is None
            or entry.get("p50_compiled_entry_sha256") != canonical_sha256(old_body)
            or encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry.get("source_state_sha256")
            or partition.source_state_sha256 != entry.get("source_state_sha256")
            or entry.get("raw_mark_count") != raw_mark_count
            or entry.get("canonical_successor_count") != len(partition.successor_groups)
            or entry.get("productive_alias_count")
            != sum(len(group.marks) for group in partition.successor_groups)
            or entry.get("production_successor_alias_multiplicity") != len(fiber.aliases)
            or entry.get("virtual_alias_count") != len(partition.virtual_marks)
            or len(exact_aliases) != 1
            or entry.get("hazard_included") is not False
            or entry.get("model_scores_or_probabilities_stored") is not False
            or any(
                entry.get(field) != selection_entry.get(selection_field)
                for field, selection_field in (
                    ("model_family", "model_family"),
                    ("capability_cell_id", "capability_cell_id"),
                    ("partition_role", "partition_role"),
                    ("source_state_sha256", "source_state_sha256"),
                    ("target_state_sha256", "target_state_sha256"),
                    ("successor_canonical_key", "canonical_successor_key"),
                    ("teacher_action_sha256", "action_sha256"),
                )
            )
        ):
            raise ProcessV2P50RuntimeError("a legacy P50 entry identity disagrees")
        new_body = {
            "p50_entry_sha256": identifier,
            "model_family": entry["model_family"],
            "capability_cell_id": entry["capability_cell_id"],
            "partition_role": entry["partition_role"],
            "support_time_hex": entry["support_time_hex"],
            "source_state_sha256": entry["source_state_sha256"],
            "target_state_sha256": entry["target_state_sha256"],
            "successor_canonical_key": entry["successor_canonical_key"],
            "teacher_action_sha256": entry["teacher_action_sha256"],
            "objective_coefficient": entry["objective_coefficient"],
            "raw_mark_count": raw_mark_count,
            "decoded_mark_count": raw_mark_count,
            "marks_by_family": dict(sorted(family_counts.items())),
            "production_successor_alias_multiplicity": len(fiber.aliases),
            "virtual_alias_count": len(fiber.state_support.virtual_aliases),
            "exact_state": entry["exact_state"],
            "teacher_successor_fiber": _teacher_fiber_payload(fiber),
            "exact_teacher_alias": _alias_payload(exact_aliases[0]),
            "model_scores_or_probabilities_stored": False,
            "hazard_included": False,
        }
        converted.append(
            {
                **new_body,
                "p50_compiled_entry_sha256": canonical_sha256(new_body),
            }
        )
    if {entry["p50_entry_sha256"] for entry in converted} != set(selected):
        raise ProcessV2P50RuntimeError("a legacy P50 leaf omits selected task entries")
    converted.sort(key=lambda entry: entry["p50_entry_sha256"])
    body = {
        "task_identity_sha256": task_id,
        "selection_sha256": selection["selection_sha256"],
        "entry_count": len(converted),
        "entries": converted,
    }
    return {**body, "leaf_sha256": canonical_sha256(body)}


def build_process_v2_p50_prepared_inputs(
    selection: Mapping[str, Any], *, leaves: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reduce complete CPU leaves into the chemistry-free GPU input."""

    if not leaves:
        raise ProcessV2P50RuntimeError("P50 preparation has no compiled leaves")
    task_ids: set[str] = set()
    for leaf in leaves:
        body = {key: item for key, item in leaf.items() if key != "leaf_sha256"}
        task_id = str(leaf.get("task_identity_sha256"))
        if (
            set(leaf)
            != {
                "task_identity_sha256",
                "selection_sha256",
                "entry_count",
                "entries",
                "leaf_sha256",
            }
            or leaf.get("leaf_sha256") != canonical_sha256(body)
            or leaf.get("selection_sha256") != selection.get("selection_sha256")
            or leaf.get("entry_count") != len(leaf.get("entries", ()))
            or task_id in task_ids
        ):
            raise ProcessV2P50RuntimeError("a P50 prepared leaf identity disagrees")
        task_ids.add(task_id)
    expected = {str(entry["p50_entry_sha256"]) for entry in selection["entries"]}
    entries = [dict(entry) for leaf in leaves for entry in leaf["entries"]]
    observed = [str(entry["p50_entry_sha256"]) for entry in entries]
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise ProcessV2P50RuntimeError("P50 prepared leaves omit or repeat a selected entry")
    entries.sort(key=lambda row: row["p50_entry_sha256"])
    body = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **authority_false_block(),
        "selection_sha256": selection["selection_sha256"],
        "prerequisites_binding_sha256": selection["prerequisites_binding_sha256"],
        "initial_model_state_sha256": selection["prerequisites"]["t1_initial_model_state_sha256"],
        "training_stream": selection["training_stream"],
        "training_stream_sha256": selection["training_stream_sha256"],
        "required_cells": selection["required_cells"],
        "training_family_step_opportunities": selection["training_family_step_opportunities"],
        "training_cell_step_opportunities": selection["training_cell_step_opportunities"],
        "validation_entry_sha256s": selection["validation_entry_sha256s"],
        "validation_inventory_sha256": selection["validation_inventory_sha256"],
        "validation_unsupported_required_cells": selection["validation_unsupported_required_cells"],
        "entry_count": len(entries),
        "entries": entries,
    }
    return {**body, "prepared_sha256": canonical_sha256(body)}


def validate_process_v2_p50_prepared_inputs(value: object) -> dict[str, Any]:
    """Validate every compiled fiber before the chemistry-free GPU path."""

    if not isinstance(value, Mapping):
        raise ProcessV2P50RuntimeError("P50 prepared inputs must be an object")
    prepared = dict(value)
    try:
        verify_self_hash(prepared, field="prepared_sha256", label="the P50 prepared inputs")
        require_authority_false(prepared, label="the P50 prepared inputs")
    except ValueError as error:
        raise ProcessV2P50RuntimeError(str(error)) from error
    if (
        prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("schema_version") != PREPARED_SCHEMA_VERSION
        or prepared.get("status") != PREPARED_STATUS
        or prepared.get("training_stream_sha256")
        != canonical_sha256(prepared.get("training_stream"))
        or prepared.get("validation_inventory_sha256")
        != canonical_sha256(prepared.get("validation_entry_sha256s"))
        or prepared.get("entry_count") != len(prepared.get("entries", ()))
    ):
        raise ProcessV2P50RuntimeError("P50 prepared input identity or census disagrees")
    observed: set[str] = set()
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in prepared["entries"]:
        entry = dict(raw)
        body = {key: item for key, item in entry.items() if key != "p50_compiled_entry_sha256"}
        identifier = _require_sha(entry.get("p50_entry_sha256"), field="P50 entry")
        try:
            state = decode_state(entry["exact_state"])
            teacher = _teacher_fiber_from_payload(entry["teacher_successor_fiber"])
            exact_teacher_alias = _alias_from_payload(entry["exact_teacher_alias"])
        except (KeyError, TypeError, ValueError, SuccessorTrainingError) as error:
            raise ProcessV2P50RuntimeError("a compiled P50 entry is invalid") from error
        marks_by_family = entry.get("marks_by_family")
        if (
            identifier in observed
            or entry.get("p50_compiled_entry_sha256") != canonical_sha256(body)
            or encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
            or teacher.state_support.source_state_sha256 != entry["source_state_sha256"]
            or teacher.target_state_sha256 != entry["target_state_sha256"]
            or teacher.target_key != entry["successor_canonical_key"]
            or exact_teacher_alias not in teacher.aliases
            or entry.get("production_successor_alias_multiplicity") != len(teacher.aliases)
            or entry.get("virtual_alias_count") != len(teacher.state_support.virtual_aliases)
            or type(entry.get("raw_mark_count")) is not int
            or type(entry.get("decoded_mark_count")) is not int
            or not 0 < entry["decoded_mark_count"] <= entry["raw_mark_count"]
            or not isinstance(marks_by_family, Mapping)
            or any(
                not isinstance(family, str) or type(count) is not int or count < 0
                for family, count in marks_by_family.items()
            )
            or sum(marks_by_family.values()) != entry["raw_mark_count"]
            or entry.get("partition_role") not in {TRAIN_ROLE, VALIDATION_ROLE}
            or entry.get("hazard_included") is not False
            or entry.get("model_scores_or_probabilities_stored") is not False
        ):
            raise ProcessV2P50RuntimeError("a compiled P50 entry identity disagrees")
        observed.add(identifier)
        by_id[identifier] = entry
    train_ids = {str(row["p50_entry_sha256"]) for row in prepared["training_stream"]}
    validation_ids = set(prepared["validation_entry_sha256s"])
    if (
        observed != train_ids | validation_ids
        or any(by_id[identifier]["partition_role"] != TRAIN_ROLE for identifier in train_ids)
        or any(
            by_id[identifier]["partition_role"] != VALIDATION_ROLE for identifier in validation_ids
        )
    ):
        raise ProcessV2P50RuntimeError("P50 prepared train/validation roles disagree")
    return prepared


@dataclass(frozen=True)
class LoadedProcessV2P50Inputs:
    prepared: Mapping[str, Any]
    states_by_id: Mapping[str, Any]
    fibers_by_id: Mapping[str, TeacherSuccessorFiber]
    metadata_by_id: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class MaterializedProcessV2P50Batch:
    """One CPU-collated row per unique prepared P50 entry."""

    batch: Any
    entry_ids: tuple[str, ...]
    index_by_entry_id: Mapping[str, int]


def load_process_v2_p50_inputs(value: Mapping[str, Any]) -> LoadedProcessV2P50Inputs:
    prepared_value = validate_process_v2_p50_prepared_inputs(value)
    entries = tuple(prepared_value["entries"])
    states: dict[str, Any] = {}
    fibers: dict[str, TeacherSuccessorFiber] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        identifier = str(entry["p50_entry_sha256"])
        state = decode_state(entry["exact_state"])
        fiber = _teacher_fiber_from_payload(entry["teacher_successor_fiber"])
        states[identifier] = state
        fibers[identifier] = fiber
        metadata[identifier] = MappingProxyType(dict(entry))
    return LoadedProcessV2P50Inputs(
        prepared=MappingProxyType(prepared_value),
        states_by_id=MappingProxyType(states),
        fibers_by_id=MappingProxyType(fibers),
        metadata_by_id=MappingProxyType(metadata),
    )


def _collator(model: FactorizedTraceletRateModel) -> FactorizedMarkCollator:
    return FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
    )


def _batch(
    runtime: LoadedProcessV2P50Inputs,
    model: FactorizedTraceletRateModel,
    identifiers: Sequence[str],
    times: Sequence[float],
    *,
    materialized: MaterializedProcessV2P50Batch | None = None,
) -> tuple[Any, tuple[TeacherSuccessorFiber, ...], tuple[Mapping[str, Any], ...]]:
    entries = tuple(runtime.metadata_by_id[identifier] for identifier in identifiers)
    if materialized is None:
        examples = [
            FactorizedMarkExample(
                state=runtime.states_by_id[identifier],
                time=float(time_value),
                teacher_action=None,
                teacher_rule_name=None,
                teacher_rate=1.0,
                importance_weight=1.0,
            )
            for identifier, time_value in zip(identifiers, times, strict=True)
        ]
        batch = _attach_successor_family_coordinates(_collator(model)(examples), entries)
    else:
        try:
            indices = tuple(materialized.index_by_entry_id[identifier] for identifier in identifiers)
        except KeyError as error:
            raise ProcessV2P50RuntimeError(
                "the P50 stream references an entry outside the collated cache"
            ) from error
        batch = _index_factorized_batch(materialized.batch, indices)
        batch = replace(
            batch,
            times=torch.tensor(
                tuple(float(value) for value in times),
                dtype=batch.times.dtype,
                device=batch.times.device,
            ),
        )
    fibers = tuple(runtime.fibers_by_id[identifier] for identifier in identifiers)
    return batch, fibers, entries


def collate_process_v2_p50_entries(
    entries: Sequence[Mapping[str, Any]], model: FactorizedTraceletRateModel
) -> tuple[Any, tuple[str, ...]]:
    """Collate unique prepared entries once, without scoring or chemistry execution."""

    resolved = tuple(dict(entry) for entry in entries)
    if not resolved:
        raise ProcessV2P50RuntimeError("a P50 collated leaf has no entries")
    identifiers: list[str] = []
    examples: list[FactorizedMarkExample] = []
    for entry in resolved:
        body = {key: item for key, item in entry.items() if key != "p50_compiled_entry_sha256"}
        identifier = _require_sha(entry.get("p50_entry_sha256"), field="P50 entry")
        try:
            state = decode_state(entry["exact_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProcessV2P50RuntimeError("a P50 collated entry state is invalid") from error
        if (
            entry.get("p50_compiled_entry_sha256") != canonical_sha256(body)
            or encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry.get("source_state_sha256")
            or identifier in identifiers
        ):
            raise ProcessV2P50RuntimeError("a P50 collated entry identity disagrees")
        identifiers.append(identifier)
        examples.append(
            FactorizedMarkExample(
                state=state,
                time=_COMPILE_SUPPORT_TIME,
                teacher_action=None,
                teacher_rule_name=None,
                teacher_rate=1.0,
                importance_weight=1.0,
            )
        )
    batch = _attach_successor_family_coordinates(_collator(model)(examples), resolved)
    if batch.batch_size != len(identifiers):
        raise ProcessV2P50RuntimeError("a P50 collated batch changed entry cardinality")
    return batch, tuple(identifiers)


def materialize_process_v2_p50_batch(
    runtime: LoadedProcessV2P50Inputs,
    *,
    batch: Any,
    entry_ids: Sequence[str],
) -> MaterializedProcessV2P50Batch:
    """Validate a consolidated collated batch against the immutable prepared input."""

    identifiers = tuple(str(identifier) for identifier in entry_ids)
    expected = tuple(str(entry["p50_entry_sha256"]) for entry in runtime.prepared["entries"])
    if identifiers != expected or batch.batch_size != len(expected):
        raise ProcessV2P50RuntimeError("the P50 collated batch inventory disagrees")
    observed_state_sha256s = tuple(
        persistent_slot_state_sha256(state) for state in batch.states
    )
    expected_state_sha256s = tuple(
        str(runtime.metadata_by_id[identifier]["source_state_sha256"])
        for identifier in identifiers
    )
    expected_families = tuple(
        str(runtime.metadata_by_id[identifier]["model_family"]) for identifier in identifiers
    )
    if (
        observed_state_sha256s != expected_state_sha256s
        or any(action is not None for action in batch.teacher_actions)
        or tuple(batch.teacher_rule_names) != expected_families
    ):
        raise ProcessV2P50RuntimeError("the P50 collated batch contents disagree")
    return MaterializedProcessV2P50Batch(
        batch=batch,
        entry_ids=identifiers,
        index_by_entry_id=MappingProxyType(
            {identifier: index for index, identifier in enumerate(identifiers)}
        ),
    )


def _evaluation_rows(
    runtime: LoadedProcessV2P50Inputs,
    model: FactorizedTraceletRateModel,
    identifiers: Sequence[str],
    *,
    batch_size: int,
    materialized: MaterializedProcessV2P50Batch | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(identifiers), batch_size):
            selected = tuple(identifiers[start : start + batch_size])
            times = tuple(_COMPILE_SUPPORT_TIME for _ in selected)
            batch, fibers, entries = _batch(
                runtime, model, selected, times, materialized=materialized
            )
            prediction = forward_teacher_successor_batch(model, batch.to(model.device), fibers)
            nlls = -prediction.selected_productive_successor_log_probability.detach().cpu()
            for entry, nll in zip(entries, nlls, strict=True):
                rows.append(
                    {
                        "p50_entry_sha256": entry["p50_entry_sha256"],
                        "model_family": entry["model_family"],
                        "capability_cell_id": entry["capability_cell_id"],
                        "canonical_successor_nll": float(nll),
                    }
                )
    model.train(was_training)
    return rows


def _aggregate_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(selected: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(row["canonical_successor_nll"]) for row in selected) / len(selected)

    return {
        "overall_mean_successor_nll": mean(rows),
        "by_family": {
            family: mean([row for row in rows if row["model_family"] == family])
            for family in sorted({str(row["model_family"]) for row in rows})
        },
        "by_cell": {
            cell: mean([row for row in rows if row["capability_cell_id"] == cell])
            for cell in sorted({str(row["capability_cell_id"]) for row in rows})
        },
    }


def run_process_v2_p50(
    model: FactorizedTraceletRateModel,
    runtime: LoadedProcessV2P50Inputs,
    *,
    policy: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    materialized: MaterializedProcessV2P50Batch | None = None,
) -> dict[str, Any]:
    """Run exactly fifty scratch, hazard-free canonical-successor updates."""

    optimization = policy["optimization"]
    if (
        state_dict_semantic_sha256(model.state_dict())
        != runtime.prepared["initial_model_state_sha256"]
        or int(optimization["optimizer_steps"]) != 50
        or int(optimization["batch_size"]) != 64
    ):
        raise ProcessV2P50RuntimeError("P50 scratch state or bounded recipe disagrees")
    torch.use_deterministic_algorithms(True)
    seed = int(optimization["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hazard_initial = _hazard_state(model)
    hazard_initial_state_sha256 = state_dict_semantic_sha256(hazard_initial)
    named_parameters = dict(model.named_parameters())
    trainable: list[torch.Tensor] = []
    for name, parameter in named_parameters.items():
        parameter.requires_grad_(not name.startswith(TOTAL_HAZARD_PREFIX))
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    validation_ids = tuple(runtime.prepared["validation_entry_sha256s"])
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "process_v2_p50_baseline_start",
                "validation_example_count": len(validation_ids),
                "collated_cache_used": materialized is not None,
            }
        )
    baseline_rows = _evaluation_rows(
        runtime,
        model,
        validation_ids,
        batch_size=int(optimization["batch_size"]),
        materialized=materialized,
    )
    baseline = _aggregate_validation(baseline_rows)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "process_v2_p50_baseline_complete",
                "validation_example_count": len(validation_ids),
            }
        )
    families = tuple(policy["active_families"])
    required_cells = tuple(runtime.prepared["required_cells"])
    family_examples = Counter(row["model_family"] for row in runtime.prepared["training_stream"])
    cell_examples = Counter(
        row["capability_cell_id"] for row in runtime.prepared["training_stream"]
    )
    family_exposure = {
        family: {
            "observed_example_count": 0,
            "finite_nonzero_global_gradient_exposure_steps": 0,
            "all_exposure_step_global_gradients_finite": True,
            "cumulative_exposure_step_global_gradient_l2": 0.0,
            "finite_nonzero_action_route_gradient_exposure_steps": 0,
            "all_exposure_step_action_route_gradients_finite": True,
            "cumulative_exposure_step_action_route_gradient_l2": 0.0,
            "required_revision_parameter": (
                "graft_relation_head.weight"
                if family == "bond_reroute"
                else (
                    "ring_restate_context_head.weight" if family == "ring_system_restate" else None
                )
            ),
            "finite_nonzero_revision_parameter_gradient_exposure_steps": 0,
            "all_exposure_step_revision_parameter_gradients_finite": True,
            "cumulative_exposure_step_revision_parameter_gradient_l2": 0.0,
        }
        for family in families
    }
    cell_exposure = {
        cell: {
            "observed_example_count": 0,
            "finite_nonzero_global_gradient_exposure_steps": 0,
            "all_exposure_step_global_gradients_finite": True,
            "cumulative_exposure_step_global_gradient_l2": 0.0,
        }
        for cell in required_cells
    }
    losses: list[float] = []
    trajectory: list[dict[str, Any]] = []
    stream = tuple(runtime.prepared["training_stream"])
    model.train()
    for step in range(int(optimization["optimizer_steps"])):
        rows = stream[
            step * int(optimization["batch_size"]) : (step + 1) * int(optimization["batch_size"])
        ]
        identifiers = tuple(str(row["p50_entry_sha256"]) for row in rows)
        times = tuple(float.fromhex(str(row["time_hex"])) for row in rows)
        batch, fibers, entries = _batch(
            runtime, model, identifiers, times, materialized=materialized
        )
        device_batch = batch.to(model.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(model, device_batch, fibers)
        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise ProcessV2P50RuntimeError(f"nonfinite P50 loss at step {step + 1}")
        loss.backward()
        squared_gradient_l2 = 0.0
        all_gradients_finite = True
        for name, parameter in named_parameters.items():
            if name.startswith(TOTAL_HAZARD_PREFIX):
                if parameter.grad is not None:
                    raise ProcessV2P50RuntimeError(
                        f"frozen hazard parameter received a P50 gradient: {name}"
                    )
            elif parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                all_gradients_finite = False
            if parameter.grad is not None and not name.startswith(TOTAL_HAZARD_PREFIX):
                squared_gradient_l2 += float(parameter.grad.detach().norm()) ** 2
        global_gradient_l2 = math.sqrt(squared_gradient_l2)
        if not all_gradients_finite or not math.isfinite(global_gradient_l2):
            raise ProcessV2P50RuntimeError(f"nonfinite P50 gradient at step {step + 1}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise ProcessV2P50RuntimeError(f"nonfinite P50 gradient norm at step {step + 1}")
        optimizer.step()
        _assert_hazard_identity(model, hazard_initial)
        losses.append(float(loss.detach().cpu()))
        trajectory.append(
            {
                "optimizer_step": step + 1,
                "mean_batch_canonical_successor_nll": losses[-1],
                "global_gradient_finite": all_gradients_finite,
                "global_gradient_l2": global_gradient_l2,
            }
        )
        row_family_counts = Counter(str(entry["model_family"]) for entry in entries)
        row_cell_counts = Counter(str(entry["capability_cell_id"]) for entry in entries)
        finite_nonzero = all_gradients_finite and global_gradient_l2 > 0.0
        for family, count in row_family_counts.items():
            exposure = family_exposure[family]
            route_gradients = [
                parameter.grad
                for name, parameter in named_parameters.items()
                if name.startswith(ACTION_ROUTE_PREFIXES[family])
                and parameter.requires_grad
                and parameter.grad is not None
            ]
            route_finite = bool(route_gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in route_gradients
            )
            route_l2 = math.sqrt(
                sum(float(gradient.detach().norm()) ** 2 for gradient in route_gradients)
            )
            if not route_finite or not math.isfinite(route_l2):
                raise ProcessV2P50RuntimeError(
                    f"nonfinite or absent {family} action-route gradient at step {step + 1}"
                )
            exposure["observed_example_count"] += count
            exposure["finite_nonzero_global_gradient_exposure_steps"] += int(finite_nonzero)
            exposure["all_exposure_step_global_gradients_finite"] &= all_gradients_finite
            exposure["cumulative_exposure_step_global_gradient_l2"] += global_gradient_l2
            exposure["finite_nonzero_action_route_gradient_exposure_steps"] += int(route_l2 > 0.0)
            exposure["all_exposure_step_action_route_gradients_finite"] &= route_finite
            exposure["cumulative_exposure_step_action_route_gradient_l2"] += route_l2
            revision_name = exposure["required_revision_parameter"]
            if revision_name is not None:
                revision_gradient = named_parameters[revision_name].grad
                revision_finite = revision_gradient is not None and bool(
                    torch.isfinite(revision_gradient).all()
                )
                revision_l2 = (
                    float(revision_gradient.detach().norm())
                    if revision_gradient is not None
                    else 0.0
                )
                if not revision_finite or not math.isfinite(revision_l2):
                    raise ProcessV2P50RuntimeError(
                        f"nonfinite or absent {revision_name} gradient at step {step + 1}"
                    )
                exposure["finite_nonzero_revision_parameter_gradient_exposure_steps"] += int(
                    revision_l2 > 0.0
                )
                exposure["all_exposure_step_revision_parameter_gradients_finite"] &= revision_finite
                exposure["cumulative_exposure_step_revision_parameter_gradient_l2"] += revision_l2
        for cell, count in row_cell_counts.items():
            exposure = cell_exposure[cell]
            exposure["observed_example_count"] += count
            exposure["finite_nonzero_global_gradient_exposure_steps"] += int(finite_nonzero)
            exposure["all_exposure_step_global_gradients_finite"] &= all_gradients_finite
            exposure["cumulative_exposure_step_global_gradient_l2"] += global_gradient_l2
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "process_v2_p50_optimizer_step",
                    "completed": step + 1,
                    "total": int(optimization["optimizer_steps"]),
                    "loss": losses[-1],
                }
            )
    final_rows = _evaluation_rows(
        runtime,
        model,
        validation_ids,
        batch_size=int(optimization["batch_size"]),
        materialized=materialized,
    )
    final = _aggregate_validation(final_rows)
    _assert_hazard_identity(model, hazard_initial)
    hazard_final_state_sha256 = state_dict_semantic_sha256(_hazard_state(model))
    baseline_by_id = {row["p50_entry_sha256"]: row for row in baseline_rows}
    final_by_id = {row["p50_entry_sha256"]: row for row in final_rows}
    observable_cells = tuple(
        cell
        for cell in required_cells
        if cell not in set(runtime.prepared["validation_unsupported_required_cells"])
    )
    validation_entry_metrics = sorted(
        (
            {
                "validation_entry_sha256": identifier,
                "family": baseline_by_id[identifier]["model_family"],
                "semantic_cell_id": baseline_by_id[identifier]["capability_cell_id"],
                "baseline_canonical_successor_nll": baseline_by_id[identifier][
                    "canonical_successor_nll"
                ],
                "final_canonical_successor_nll": final_by_id[identifier]["canonical_successor_nll"],
            }
            for identifier in validation_ids
        ),
        key=lambda row: (
            families.index(row["family"]),
            observable_cells.index(row["semantic_cell_id"]),
            row["validation_entry_sha256"],
        ),
    )
    family_training_evidence = [
        {
            "family": family,
            "planned_example_count": int(family_examples[family]),
            "planned_optimizer_step_opportunities": int(
                runtime.prepared["training_family_step_opportunities"][family]
            ),
            **family_exposure[family],
        }
        for family in families
    ]
    semantic_cell_training_evidence = [
        {
            "semantic_cell_id": cell,
            "planned_example_count": int(cell_examples[cell]),
            "planned_optimizer_step_opportunities": int(
                runtime.prepared["training_cell_step_opportunities"][cell]
            ),
            **cell_exposure[cell],
        }
        for cell in required_cells
    ]
    return {
        "optimizer_steps_completed": len(losses),
        "losses": losses,
        "trajectory": trajectory,
        "baseline_validation": baseline,
        "final_validation": final,
        "validation_entry_metrics": validation_entry_metrics,
        "validation_unsupported_required_cells": list(
            runtime.prepared["validation_unsupported_required_cells"]
        ),
        "family_training_evidence": family_training_evidence,
        "semantic_cell_training_evidence": semantic_cell_training_evidence,
        "final_model_state": _clone_state_dict(model),
        "final_model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "hazard_unchanged": True,
        "hazard_initial_state_sha256": hazard_initial_state_sha256,
        "hazard_final_state_sha256": hazard_final_state_sha256,
    }


__all__ = [
    "PREPARED_FILENAME",
    "SELECTION_FILENAME",
    "LoadedProcessV2P50Inputs",
    "ProcessV2P50RuntimeError",
    "VALIDATION_EXAMPLES_PER_SUPPORTED_CELL",
    "build_process_v2_p50_prepared_inputs",
    "build_process_v2_p50_selection",
    "collate_process_v2_p50_entries",
    "compile_process_v2_p50_entries",
    "convert_process_v2_p50_v1_leaf",
    "load_process_v2_p50_inputs",
    "materialize_process_v2_p50_batch",
    "run_process_v2_p50",
    "validate_process_v2_p50_prepared_inputs",
    "validate_process_v2_p50_selection",
]
