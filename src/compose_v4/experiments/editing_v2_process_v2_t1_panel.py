"""Process-V2-native T1 panel selection and exact-state resolution.

This is the narrow bridge from the current Active8/Gate-0 artifact seam to the
existing successor-level T1 compiler.  It deliberately does not build a second
decision index or a new training framework:

* Gate 0 must already have published an authenticated ``PASS``;
* only Gate-0-eligible (currently train) decision shards are opened;
* repeated rows are collapsed at canonical-successor level and exact source
  states with more than one observed canonical target are excluded;
* the frozen Process-V2 panel cardinalities select a deterministic, cell-aware
  bounded panel; and
* selected trace addresses reopen the existing Process-V2 cache/rebind path,
  preserving exact persistent-slot states.

The artifact grants no training or P50 authority.  Successor-partition
compilation and capacity optimization remain downstream stages.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_process_v2_rebind import (
    chunk_target_for_task,
    mounted_process_v2_artifact_path,
)
from compose_v4.data.editing_v2_process_v2_active8_map import (
    read_rebind_chunk_decisions,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    PLAN_FILENAME,
    load_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    ProcessV2Active8ReduceError,
    validate_process_v2_active8_completion,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    read_process_v2_chunk_target,
)
from compose_v4.data.editing_v2_process_v2_gate_zero import (
    GateZeroContracts,
    GateZeroSourceIndex,
    ProcessV2GateZeroError,
    iter_gate_zero_eligible_shards,
    reduce_gate_zero,
    reduction_order,
    resolve_gate_zero_eligible_stream,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import ImmutableArtifactError, write_bytes_if_absent
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    T1_PANEL_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.rewrite.kernel import canonical_state_key

PANEL_SCHEMA = "compose.editing_v2.process_v2_t1_panel"
PANEL_SCHEMA_VERSION = 1
PANEL_STATUS = "PROCESS_V2_T1_PANEL_PREPARED_NO_DOWNSTREAM_AUTHORITY"
PANEL_FILENAME = "PROCESS_V2_T1_PANEL.json"
SELECTION_RULE = "unique_source_single_canonical_target_cell_round_robin_v1"

_DECISION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "decision",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "contracts_binding_sha256",
        "gate_zero_structural_contract_sha256",
        "process_identity_sha256",
        "source_index_sha256",
        "enforced_structural_clauses",
        "reduction",
        "accounting",
        "role_census",
        "sealed_role_metadata",
        "model_family_counts",
        "executor_rule_counts",
        "capability_cell_counts",
        "required_cell_counts",
        "conditional_cell_counts",
        "separate_lane_cell_counts",
        "violation_counts",
        "total_violations",
        "missing_active8_families",
        "missing_required_cells",
        "checks",
        "negative_receipts",
        "decision_sha256",
    }
)

_ENTRY_FIELDS = frozenset(
    {
        "v1_task_identity_sha256",
        "entry_index",
        "trace_id",
        "step_index",
        "task_identity_sha256",
        "data_lane",
        "partition_role",
        "progress_index",
        "executor_rule",
        "model_family",
        "family_context",
        "capability_cell_id",
        "source_state_sha256",
        "target_state_sha256",
        "source_canonical_key",
        "canonical_successor_key",
        "action_sha256",
        "assignment_sha256",
        "source_occurrence_count",
        "selection_rank_sha256",
        "panel_entry_sha256",
    }
)


class ProcessV2T1PanelError(RuntimeError):
    """The current Process-V2 evidence cannot produce an honest T1 panel."""


@dataclass(frozen=True)
class ProcessV2T1Source:
    """Fully authenticated upstream inputs; no molecular chunk is opened."""

    active8_run_root: Path
    artifact_root: Path
    repo_root: Path
    contracts: GateZeroContracts
    index: GateZeroSourceIndex
    decision: Mapping[str, Any]
    decision_file_sha256: str
    completion: Mapping[str, Any]
    plan: Mapping[str, Any]
    policy: Mapping[str, Any]
    policy_file_sha256: str


@dataclass(frozen=True)
class ProcessV2T1ResolvedTrace:
    """One exact trace plus the selected panel entries it contains."""

    addressed_trace: Any
    panel_entries: tuple[Mapping[str, Any], ...]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2T1PanelError(f"{field} must be a full lowercase SHA-256")
    return value


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1PanelError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ProcessV2T1PanelError(f"{label} must be an object: {path}")
    if raw != canonical_bytes(value) + b"\n":
        raise ProcessV2T1PanelError(f"{label} is not canonical newline-framed JSON: {path}")
    return value, raw


def _validate_gate_zero_pass(
    path: Path,
    *,
    contracts: GateZeroContracts,
    index: GateZeroSourceIndex,
) -> tuple[dict[str, Any], str]:
    decision, raw = _load_json(path, label="the Process-V2 Gate-0 decision")
    if set(decision) != _DECISION_FIELDS:
        raise ProcessV2T1PanelError("the Process-V2 Gate-0 decision field set disagrees")
    try:
        verify_self_hash(decision, field="decision_sha256", label="the Gate-0 decision")
        require_authority_false(decision, label="the Gate-0 decision")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    if (
        decision["schema"] != GATE_ZERO_DECISION_SCHEMA
        or decision["schema_version"] != GATE_ZERO_DECISION_SCHEMA_VERSION
        or decision["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or decision["decision"] != "PASS"
    ):
        raise ProcessV2T1PanelError("T1 requires a current Process-V2 Gate-0 PASS")
    if (
        decision["process_identity_sha256"] != contracts.process_identity_sha256
        or decision["contracts_binding_sha256"] != contracts.binding_sha256
        or decision["gate_zero_structural_contract_sha256"] != contracts.contract_sha256
        or decision["source_index_sha256"] != index.index_sha256
        or decision["active8_completion_sha256"] != index.active8_completion_sha256
        or decision["active8_sentinel_sha256"] != index.active8_sentinel_sha256
    ):
        raise ProcessV2T1PanelError("the Gate-0 PASS names another Active8/process input")
    expected_digest = canonical_sha256(
        [
            [str(shard["task_identity_sha256"]), str(shard["decision_shard_sha256"])]
            for shard in reduction_order(index.eligible())
        ]
    )
    if (
        decision["reduction"].get("digest_sha256") != expected_digest
        or decision["accounting"].get("eligible_shards") != len(index.eligible())
        or decision["accounting"].get("decision_shards_read") != len(index.eligible())
        or decision["total_violations"] != 0
        or decision["missing_active8_families"]
        or decision["missing_required_cells"]
        or not decision["checks"]
        or any(value is not True for value in decision["checks"].values())
    ):
        raise ProcessV2T1PanelError("the Gate-0 PASS does not reconcile with its source index")
    with tempfile.TemporaryDirectory() as temporary:
        derived = reduce_gate_zero(
            Path(index.active8_run_root),
            gate_zero_root=Path(temporary),
            contracts=contracts,
            index=index,
        )
    if canonical_bytes(derived) != canonical_bytes(decision):
        raise ProcessV2T1PanelError(
            "the Gate-0 decision differs from a fresh reduction of its bound shards"
        )
    return decision, hashlib.sha256(raw).hexdigest()


def open_process_v2_t1_source(
    active8_run_root: Path,
    *,
    gate_zero_decision_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> ProcessV2T1Source:
    """Authenticate Active8, Gate 0, the frozen policy, and the exact plan."""

    active8_root = Path(active8_run_root)
    artifact_root = Path(artifact_root)
    repo_root = Path(repo_root)
    try:
        contracts, index = resolve_gate_zero_eligible_stream(
            active8_root, repo_root=repo_root
        )
    except ProcessV2GateZeroError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    decision, decision_file_sha256 = _validate_gate_zero_pass(
        Path(gate_zero_decision_path), contracts=contracts, index=index
    )
    completion_value, _ = _load_json(
        active8_root / COMPLETION_FILENAME, label="the Active8 completion"
    )
    try:
        completion = validate_process_v2_active8_completion(completion_value)
    except ProcessV2Active8ReduceError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    plan = load_process_v2_active8_plan(active8_root / PLAN_FILENAME, repo_root=repo_root)
    mounted_root = mounted_process_v2_artifact_path(
        str(plan["run_artifact_root"]),
        artifact_root=artifact_root,
        field="active8 plan run_artifact_root",
    )
    if mounted_root.resolve() != active8_root.resolve():
        raise ProcessV2T1PanelError("the Active8 plan resolves to another run root")
    if (
        completion["completion_sha256"] != decision["active8_completion_sha256"]
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["run_identity_sha256"] != plan["run_identity_sha256"]
        or completion["binding_sha256"] != plan["binding_sha256"]
    ):
        raise ProcessV2T1PanelError("the Active8 completion, plan, and Gate-0 PASS disagree")
    policy = load_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=repo_root)
    policy_path = repo_root / T1_PANEL_POLICY
    if (
        policy["process_identity"]["process_identity_sha256"]
        != contracts.process_identity_sha256
        or tuple(policy["active_families"]) != contracts.active_families
        or policy["panel_kind"]
        != "unique_state_single_target_canonical_successor_capacity"
        or policy["repeated_state_panel_included"] is not False
        or policy["hazard_included"] is not False
    ):
        raise ProcessV2T1PanelError("the frozen Process-V2 T1 panel policy disagrees")
    return ProcessV2T1Source(
        active8_run_root=active8_root,
        artifact_root=artifact_root,
        repo_root=repo_root,
        contracts=contracts,
        index=index,
        decision=decision,
        decision_file_sha256=decision_file_sha256,
        completion=completion,
        plan=plan,
        policy=policy,
        policy_file_sha256=_file_sha256(policy_path),
    )


def iter_process_v2_t1_candidates(
    source: ProcessV2T1Source,
) -> Iterable[dict[str, Any]]:
    """Stream the exact train-only accepted-transition source once."""

    try:
        shards = iter_gate_zero_eligible_shards(
            source.active8_run_root,
            contracts=source.contracts,
            index=source.index,
        )
        for _shard, rows in shards:
            for row in rows:
                evidence = row["candidate_evidence"]
                yield {
                    "v1_task_identity_sha256": str(row["v1_task_identity_sha256"]),
                    "entry_index": int(row["entry_index"]),
                    "trace_id": str(row["trace_id"]),
                    "step_index": int(row["step_index"]),
                    "task_identity_sha256": str(row["task_identity_sha256"]),
                    "data_lane": str(row["data_lane"]),
                    "partition_role": str(row["partition_role"]),
                    "progress_index": int(row["progress_index"]),
                    "executor_rule": str(row["executor_rule"]),
                    "model_family": str(row["model_family"]),
                    "family_context": str(row["family_context"]),
                    "capability_cell_id": str(row["capability_cell_id"]),
                    "source_state_sha256": _require_sha(
                        evidence["source_state_sha256"], field="source_state_sha256"
                    ),
                    "target_state_sha256": _require_sha(
                        evidence["target_state_sha256"], field="target_state_sha256"
                    ),
                    "source_canonical_key": str(evidence["source_canonical_key"]),
                    "canonical_successor_key": str(evidence["canonical_successor_key"]),
                    "action_sha256": _require_sha(
                        evidence["action_sha256"], field="action_sha256"
                    ),
                    "assignment_sha256": _require_sha(
                        row["assignment_sha256"], field="assignment_sha256"
                    ),
                }
    except ProcessV2GateZeroError as error:
        raise ProcessV2T1PanelError(str(error)) from error


def _selection_rank(candidate: Mapping[str, Any], *, seed: str) -> str:
    return canonical_sha256(
        {
            "selection_rule": SELECTION_RULE,
            "selection_seed_sha256": seed,
            "source_state_sha256": candidate["source_state_sha256"],
            "canonical_successor_key": candidate["canonical_successor_key"],
            "model_family": candidate["model_family"],
            "capability_cell_id": candidate["capability_cell_id"],
        }
    )


def _selected_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    source: ProcessV2T1Source,
    scratch_dir: Path | None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Collapse and select through a bounded temporary SQLite index."""

    seed = canonical_sha256(
        {
            "decision_sha256": source.decision["decision_sha256"],
            "panel_policy_sha256": source.policy["contract_sha256"],
            "process_identity_sha256": source.contracts.process_identity_sha256,
        }
    )
    with tempfile.TemporaryDirectory(dir=scratch_dir) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "candidates.sqlite3"))
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE occurrences (
                    assignment_sha256 TEXT PRIMARY KEY,
                    source_state_sha256 TEXT NOT NULL,
                    canonical_successor_key TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    capability_cell_id TEXT NOT NULL,
                    selection_rank_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            raw_count = 0
            stream = hashlib.sha256()
            for item in candidates:
                candidate = dict(item)
                if candidate["partition_role"] not in source.contracts.decision_eligible_roles:
                    raise ProcessV2T1PanelError("a sealed role reached T1 panel selection")
                family = str(candidate["model_family"])
                cell = str(candidate["capability_cell_id"])
                if family not in source.contracts.active_families or cell not in (
                    source.contracts.registered_cell_ids
                ):
                    raise ProcessV2T1PanelError("an unregistered family/cell reached T1")
                rank = _selection_rank(candidate, seed=seed)
                payload = json.dumps(
                    candidate,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                try:
                    connection.execute(
                        "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            candidate["assignment_sha256"],
                            candidate["source_state_sha256"],
                            candidate["canonical_successor_key"],
                            family,
                            cell,
                            rank,
                            payload,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ProcessV2T1PanelError(
                        "the T1 source repeats one accepted assignment"
                    ) from error
                raw_count += 1
                stream.update(str(candidate["assignment_sha256"]).encode("ascii") + b"\n")
            connection.commit()
            if raw_count != int(source.decision["accounting"]["transitions"]):
                raise ProcessV2T1PanelError(
                    "the T1 stream count differs from the authenticated Gate-0 PASS"
                )
            census = connection.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN target_count = 1 AND context_count = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN target_count > 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN context_count > 1 THEN 1 ELSE 0 END)
                FROM (
                    SELECT source_state_sha256,
                           COUNT(DISTINCT canonical_successor_key) AS target_count,
                           COUNT(DISTINCT model_family || char(31) || capability_cell_id)
                               AS context_count
                    FROM occurrences GROUP BY source_state_sha256
                )
                """
            ).fetchone()
            rows = connection.execute(
                """
                WITH eligible AS (
                    SELECT source_state_sha256, COUNT(*) AS occurrence_count
                    FROM occurrences
                    GROUP BY source_state_sha256
                    HAVING COUNT(DISTINCT canonical_successor_key) = 1
                       AND COUNT(DISTINCT model_family || char(31) || capability_cell_id) = 1
                ), ranked AS (
                    SELECT o.*, e.occurrence_count,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.source_state_sha256
                               ORDER BY o.assignment_sha256
                           ) AS representative_rank
                    FROM occurrences o JOIN eligible e USING (source_state_sha256)
                )
                SELECT payload_json, selection_rank_sha256, occurrence_count
                FROM ranked WHERE representative_rank = 1
                ORDER BY model_family, capability_cell_id, selection_rank_sha256,
                         source_state_sha256
                """
            ).fetchall()
        finally:
            connection.close()

    by_family_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_inventory: list[str] = []
    for payload, rank, occurrence_count in rows:
        candidate = json.loads(payload)
        candidate["selection_rank_sha256"] = str(rank)
        candidate["source_occurrence_count"] = int(occurrence_count)
        by_family_cell[(candidate["model_family"], candidate["capability_cell_id"])].append(
            candidate
        )
        candidate_inventory.append(str(candidate["source_state_sha256"]))

    minimum = {str(k): int(v) for k, v in source.policy["minimum_entries_by_family"].items()}
    maximum = {str(k): int(v) for k, v in source.policy["maximum_entries_by_family"].items()}
    required_by_family: dict[str, list[str]] = defaultdict(list)
    for cell in source.contracts.required_cell_ids:
        _namespace, family, _context = cell.split(":", 2)
        required_by_family[family].append(cell)

    selected: list[dict[str, Any]] = []
    for family in source.contracts.active_families:
        cells = sorted(required_by_family[family])
        missing = [cell for cell in cells if not by_family_cell.get((family, cell))]
        if missing:
            raise ProcessV2T1PanelError(
                f"T1 panel has no unique-state candidate for required cells {missing}"
            )
        offsets = dict.fromkeys(cells, 0)
        family_selected: list[dict[str, Any]] = []
        while len(family_selected) < maximum[family]:
            progressed = False
            for cell in cells:
                bucket = by_family_cell[(family, cell)]
                offset = offsets[cell]
                if offset < len(bucket) and len(family_selected) < maximum[family]:
                    family_selected.append(bucket[offset])
                    offsets[cell] = offset + 1
                    progressed = True
            if not progressed:
                break
        if len(family_selected) < minimum[family]:
            raise ProcessV2T1PanelError(
                f"T1 panel has {len(family_selected)} unique {family} states; "
                f"the frozen minimum is {minimum[family]}"
            )
        selected.extend(family_selected)

    entries: list[dict[str, Any]] = []
    for candidate in sorted(
        selected,
        key=lambda item: (
            item["model_family"],
            item["capability_cell_id"],
            item["selection_rank_sha256"],
        ),
    ):
        body = dict(candidate)
        entry = {**body, "panel_entry_sha256": canonical_sha256(body)}
        if set(entry) != _ENTRY_FIELDS:
            raise ProcessV2T1PanelError("the Process-V2 T1 panel entry fields disagree")
        entries.append(entry)
    measurement = {
        "accepted_transition_count": raw_count,
        "accepted_transition_stream_sha256": stream.hexdigest(),
        "distinct_source_state_count": int(census[0] or 0),
        "eligible_single_target_single_context_source_count": int(census[1] or 0),
        "multi_target_source_count": int(census[2] or 0),
        "multi_context_source_count": int(census[3] or 0),
        "eligible_source_inventory_sha256": canonical_sha256(sorted(candidate_inventory)),
    }
    return tuple(entries), measurement


def build_process_v2_t1_panel(
    source: ProcessV2T1Source,
    *,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the frozen-cardinality native panel; publish nothing on failure."""

    entries, measurement = _selected_candidates(
        iter_process_v2_t1_candidates(source), source=source, scratch_dir=scratch_dir
    )
    family_counts = Counter(str(entry["model_family"]) for entry in entries)
    cell_counts = Counter(str(entry["capability_cell_id"]) for entry in entries)
    implementation_sha256 = _file_sha256(Path(__file__))
    body = {
        "schema": PANEL_SCHEMA,
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": PANEL_STATUS,
        **authority_false_block(),
        "selection_rule": SELECTION_RULE,
        "process_identity_sha256": source.contracts.process_identity_sha256,
        "active8_completion_sha256": source.index.active8_completion_sha256,
        "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
        "active8_plan_sha256": source.plan["plan_sha256"],
        "active8_run_identity_sha256": source.plan["run_identity_sha256"],
        "active8_binding_sha256": source.plan["binding_sha256"],
        "gate_zero_decision_file_sha256": source.decision_file_sha256,
        "gate_zero_decision_sha256": source.decision["decision_sha256"],
        "gate_zero_source_index_sha256": source.index.index_sha256,
        "gate_zero_contracts_binding_sha256": source.contracts.binding_sha256,
        "panel_policy_file_sha256": source.policy_file_sha256,
        "panel_policy_sha256": source.policy["contract_sha256"],
        "panel_implementation_sha256": implementation_sha256,
        "support_time_hex": source.policy["support_time_hex"],
        "panel_kind": source.policy["panel_kind"],
        "family_counts": dict(sorted(family_counts.items())),
        "capability_cell_counts": dict(sorted(cell_counts.items())),
        "source_measurement": measurement,
        "entries": list(entries),
    }
    return {**body, "panel_sha256": canonical_sha256(body)}


def validate_process_v2_t1_panel(
    value: object, *, source: ProcessV2T1Source
) -> dict[str, Any]:
    """Validate a panel against the exact current upstream evidence."""

    if not isinstance(value, Mapping):
        raise ProcessV2T1PanelError("the Process-V2 T1 panel must be an object")
    panel = dict(value)
    expected_top = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "selection_rule",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "active8_binding_sha256",
        "gate_zero_decision_file_sha256",
        "gate_zero_decision_sha256",
        "gate_zero_source_index_sha256",
        "gate_zero_contracts_binding_sha256",
        "panel_policy_file_sha256",
        "panel_policy_sha256",
        "panel_implementation_sha256",
        "support_time_hex",
        "panel_kind",
        "family_counts",
        "capability_cell_counts",
        "source_measurement",
        "entries",
        "panel_sha256",
    }
    if set(panel) != expected_top:
        raise ProcessV2T1PanelError("the Process-V2 T1 panel field set disagrees")
    try:
        verify_self_hash(panel, field="panel_sha256", label="the Process-V2 T1 panel")
        require_authority_false(panel, label="the Process-V2 T1 panel")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    expected = {
        "schema": PANEL_SCHEMA,
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": PANEL_STATUS,
        "selection_rule": SELECTION_RULE,
        "process_identity_sha256": source.contracts.process_identity_sha256,
        "active8_completion_sha256": source.index.active8_completion_sha256,
        "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
        "active8_plan_sha256": source.plan["plan_sha256"],
        "active8_run_identity_sha256": source.plan["run_identity_sha256"],
        "active8_binding_sha256": source.plan["binding_sha256"],
        "gate_zero_decision_file_sha256": source.decision_file_sha256,
        "gate_zero_decision_sha256": source.decision["decision_sha256"],
        "gate_zero_source_index_sha256": source.index.index_sha256,
        "gate_zero_contracts_binding_sha256": source.contracts.binding_sha256,
        "panel_policy_file_sha256": source.policy_file_sha256,
        "panel_policy_sha256": source.policy["contract_sha256"],
        "panel_implementation_sha256": _file_sha256(Path(__file__)),
        "support_time_hex": source.policy["support_time_hex"],
        "panel_kind": source.policy["panel_kind"],
    }
    if any(panel.get(key) != item for key, item in expected.items()):
        raise ProcessV2T1PanelError("the Process-V2 T1 panel binds another input")
    entries = panel["entries"]
    if not isinstance(entries, list) or not entries:
        raise ProcessV2T1PanelError("the Process-V2 T1 panel has no entries")
    source_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_FIELDS:
            raise ProcessV2T1PanelError("a Process-V2 T1 panel entry field set disagrees")
        verify_self_hash(entry, field="panel_entry_sha256", label="a T1 panel entry")
        source_id = str(entry["source_state_sha256"])
        if source_id in source_ids:
            raise ProcessV2T1PanelError("the unique-state T1 panel repeats an exact source")
        source_ids.add(source_id)
        family_counts[str(entry["model_family"])] += 1
        cell_counts[str(entry["capability_cell_id"])] += 1
    if dict(sorted(family_counts.items())) != panel["family_counts"] or dict(
        sorted(cell_counts.items())
    ) != panel["capability_cell_counts"]:
        raise ProcessV2T1PanelError("the T1 panel counts disagree with its entries")
    for family in source.contracts.active_families:
        if not (
            int(source.policy["minimum_entries_by_family"][family])
            <= family_counts[family]
            <= int(source.policy["maximum_entries_by_family"][family])
        ):
            raise ProcessV2T1PanelError("the T1 panel violates frozen family cardinality")
    if not set(source.contracts.required_cell_ids).issubset(cell_counts):
        raise ProcessV2T1PanelError("the T1 panel omits a required capability cell")
    rebuilt = build_process_v2_t1_panel(source)
    if canonical_bytes(rebuilt) != canonical_bytes(panel):
        raise ProcessV2T1PanelError(
            "the T1 panel differs from deterministic selection over its bound source"
        )
    return panel


def write_process_v2_t1_panel(
    panel: Mapping[str, Any], *, output_root: Path, source: ProcessV2T1Source
) -> Path:
    """Publish one immutable validated native panel."""

    validated = validate_process_v2_t1_panel(panel, source=source)
    path = Path(output_root) / PANEL_FILENAME
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1PanelError(str(error)) from error
    return path


def load_process_v2_t1_panel(
    path: Path, *, source: ProcessV2T1Source
) -> dict[str, Any]:
    """Reopen canonical panel bytes and rederive selection from the bound source."""

    if Path(path).name != PANEL_FILENAME:
        raise ProcessV2T1PanelError(f"the T1 panel must be named {PANEL_FILENAME}")
    panel, _raw = _load_json(Path(path), label="the Process-V2 T1 panel")
    return validate_process_v2_t1_panel(panel, source=source)


def resolve_process_v2_t1_entries(
    source: ProcessV2T1Source,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[ProcessV2T1ResolvedTrace, ...]:
    """Reopen selected exact traces once per source chunk and verify each state."""

    if not entries:
        raise ProcessV2T1PanelError("no T1 entries were supplied for exact resolution")
    task_index = {
        str(task["task_identity_sha256"]): dict(task) for task in source.plan["tasks"]
    }
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        identity = str(entry["task_identity_sha256"])
        if identity not in task_index:
            raise ProcessV2T1PanelError("a T1 entry names an unplanned Active8 task")
        by_task[identity].append(entry)

    resolved: list[ProcessV2T1ResolvedTrace] = []
    observed_entry_ids: set[str] = set()
    for task_identity in sorted(by_task):
        task = task_index[task_identity]
        selected = by_task[task_identity]
        wanted_by_index: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for entry in selected:
            index = int(entry["entry_index"])
            if not int(task["entry_start"]) <= index < int(task["entry_stop"]):
                raise ProcessV2T1PanelError("a T1 entry lies outside its source chunk")
            wanted_by_index[index].append(entry)
        rebind_output = mounted_process_v2_artifact_path(
            str(task["rebind_task_artifact_path"]),
            artifact_root=source.artifact_root,
            field="task.rebind_task_artifact_path",
        )
        decisions, pinned_identity = read_rebind_chunk_decisions(
            rebind_output,
            entry_start=int(task["entry_start"]),
            entry_stop=int(task["entry_stop"]),
            task_identity_sha256=str(task["rebind_task_identity_sha256"]),
            chunk_file_sha256=str(task["chunk_file_sha256"]),
            pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        )
        for index in wanted_by_index:
            if not decisions[index]["admitted"]:
                raise ProcessV2T1PanelError("a selected Active8 transition was not rebind-admitted")
        target = chunk_target_for_task(
            task,
            pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        )
        source_output = mounted_process_v2_artifact_path(
            target.source_artifact_path,
            artifact_root=source.artifact_root,
            field="task.cache_source_artifact_path",
        )
        for row in read_process_v2_chunk_target(
            source_output,
            target=target,
            expected_process_identity=pinned_identity,
            sentinel_replay_entries=0,
            recover_row_errors=False,
            repo_root=source.repo_root,
        ):
            selected_here = wanted_by_index.get(int(row.entry_index))
            if not selected_here:
                continue
            if row.addressed is None:
                raise ProcessV2T1PanelError("a selected cache row has no exact trace")
            addressed = row.addressed
            if any(str(entry["trace_id"]) != addressed.address.trace_id for entry in selected_here):
                raise ProcessV2T1PanelError("a selected T1 address resolves another trace")
            for entry in selected_here:
                progress = int(entry["progress_index"])
                if progress != int(entry["step_index"]) or not (
                    0 <= progress < len(addressed.trace.steps)
                ):
                    raise ProcessV2T1PanelError("a selected T1 progress address is invalid")
                before = addressed.path.state_at(progress)
                after = addressed.path.state_at(progress + 1)
                if (
                    persistent_slot_state_sha256(before) != entry["source_state_sha256"]
                    or persistent_slot_state_sha256(after) != entry["target_state_sha256"]
                    or canonical_state_key(before) != entry["source_canonical_key"]
                    or canonical_state_key(after) != entry["canonical_successor_key"]
                ):
                    raise ProcessV2T1PanelError(
                        "a selected T1 transition disagrees with its exact cached states"
                    )
                entry_id = str(entry.get("panel_entry_sha256") or entry["assignment_sha256"])
                if entry_id in observed_entry_ids:
                    raise ProcessV2T1PanelError("a T1 panel entry resolved twice")
                observed_entry_ids.add(entry_id)
            resolved.append(
                ProcessV2T1ResolvedTrace(
                    addressed_trace=addressed,
                    panel_entries=tuple(
                        sorted(selected_here, key=lambda item: int(item["progress_index"]))
                    ),
                )
            )
    expected_entry_ids = {
        str(entry.get("panel_entry_sha256") or entry["assignment_sha256"])
        for entry in entries
    }
    if observed_entry_ids != expected_entry_ids:
        raise ProcessV2T1PanelError("the cache/rebind path did not resolve every T1 entry")
    return tuple(resolved)


__all__ = [
    "PANEL_FILENAME",
    "PANEL_SCHEMA",
    "PANEL_SCHEMA_VERSION",
    "ProcessV2T1PanelError",
    "ProcessV2T1ResolvedTrace",
    "ProcessV2T1Source",
    "build_process_v2_t1_panel",
    "iter_process_v2_t1_candidates",
    "load_process_v2_t1_panel",
    "open_process_v2_t1_source",
    "resolve_process_v2_t1_entries",
    "validate_process_v2_t1_panel",
    "write_process_v2_t1_panel",
]
