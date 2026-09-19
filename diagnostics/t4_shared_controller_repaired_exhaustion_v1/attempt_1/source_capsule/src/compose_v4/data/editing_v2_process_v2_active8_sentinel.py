"""The Active8 release sentinel: the only re-enumeration anywhere in the pipeline.

WHY IT EXISTS
-------------
There is a residual that no amount of internal consistency can reach. If
family-local teacher evidence is produced incorrectly and then propagated
consistently through every aggregate, no downstream consistency check can
detect it because nothing downstream re-enters the molecular state.

The sentinel converts that residual from undetectable to SAMPLED. It re-enters
the chunk, reconstructs complete quotient geometry through the production
evaluator, and compares the shared teacher-admission projection and cell
assignment against what was published. It cannot prove the whole corpus; it
makes the undetectable failure detectable at a bounded, prospectively frozen
rate.

WHEN IT RUNS
------------
After every task has published and BEFORE the authoritative completion.  Any
mismatch blocks completion: preparation and mismatch receipts remain auditable,
but no authoritative completion is published.

SELECTION
---------
Ranking is by SHA-256 over the frozen ``SENTINEL_SALT`` and the pair, so it was
fixed before any result existed and cannot be chosen after seeing one.  At or
below ``SENTINEL_EXHAUSTIVE_THRESHOLD`` unique accepted
``(source_state_sha256, action_sha256)`` pairs the sentinel checks all of them;
above it, the deduplicated union of ``SENTINEL_PER_CELL_EXAMPLES`` per REQUIRED
capability cell and ``SENTINEL_GLOBAL_EXAMPLES`` globally.  The per-cell half
exists because a global sample of a skewed corpus can miss a rare required cell
entirely.

THE DICTIONARY ORACLE
---------------------
``_dictionary_successor_oracle`` regroups the production marked law with an
ordinary Python dictionary and compares the result against the production
canonical quotient.  It reuses the production executor and the production
canonical key -- those DEFINE the process -- and reimplements only the grouping,
which is the step being checked.  It is a BOUNDED TEST INSTRUMENT: it runs on
at most ``SENTINEL_ORACLE_EXAMPLES`` examples, it never decides admission, and
no number it produces other than ``oracle_mismatches`` is ever published.
``tests/test_editing_v2_process_v2_active8_pipeline.py`` drives it against the
repository's independent test-only successor oracle, which production code is
forbidden to import, so the instrument itself is checked independently.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import mounted_process_v2_artifact_path
from compose_v4.data.editing_v2_process_v2_active8_map import (
    ProcessV2Active8MapError,
    _chunk_target,
    _classification_block,
    _validated_rebind_chunk_receipt,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    read_process_v2_chunk_target,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    PIPELINE_STATUS_NO_AUTHORITY,
    SENTINEL_BLOCKS_COMPLETION,
    SENTINEL_EXHAUSTIVE_THRESHOLD,
    SENTINEL_GLOBAL_EXAMPLES,
    SENTINEL_OCCURRENCE_FIELDS,
    SENTINEL_ORACLE_EXAMPLES,
    SENTINEL_PAIR_FIELDS,
    SENTINEL_PARTITION_FIELDS,
    SENTINEL_PARTITION_MATCHED,
    SENTINEL_PARTITION_MISMATCH,
    SENTINEL_PARTITION_RESULT_FIELDS,
    SENTINEL_PARTITION_RESULT_FILENAME,
    SENTINEL_PARTITION_RESULT_SCHEMA,
    SENTINEL_PARTITION_RESULT_SCHEMA_VERSION,
    SENTINEL_PARTITIONS_DIRNAME,
    SENTINEL_PER_CELL_EXAMPLES,
    SENTINEL_PLAN_FIELDS,
    SENTINEL_PLAN_SCHEMA,
    SENTINEL_PLAN_SCHEMA_VERSION,
    SENTINEL_RESULT_FIELDS,
    SENTINEL_SALT,
    SENTINEL_SCHEMA,
    SENTINEL_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    self_hashed,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProductionProcessV2SemanticExactCandidateChecker,
)
from compose_v4.experiments.production_successor_kernel import canonical_successor_result
from compose_v4.rewrite.kernel import canonical_state_key

SENTINEL_PASSED = "SENTINEL_PASSED"
SENTINEL_FAILED = "SENTINEL_FAILED"
SELECTION_EXHAUSTIVE = "exhaustive_all_unique_accepted_pairs"
SELECTION_RANKED = "deduplicated_union_of_per_cell_and_global_ranked_examples"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProcessV2Active8SentinelError(RuntimeError):
    """The sentinel cannot be run over the published Active8 evidence."""


def sentinel_rank(source_state_sha256: str, action_sha256: str) -> str:
    """The prospectively frozen rank of one accepted pair."""

    digest = hashlib.sha256()
    digest.update(SENTINEL_SALT.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(source_state_sha256).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(action_sha256).encode("utf-8"))
    return digest.hexdigest()


def _pair(transition: Mapping[str, Any]) -> tuple[str, str]:
    evidence = transition["candidate_evidence"]
    return str(evidence["source_state_sha256"]), str(evidence["action_sha256"])


def _pair_rank_key(pair: tuple[str, str]) -> tuple[str, str, str]:
    """Total ordering for a pair, including deterministic digest-collision ties."""

    return sentinel_rank(*pair), pair[0], pair[1]


def create_sentinel_pair_index(connection: sqlite3.Connection) -> None:
    """Create the bounded reducer's exact unique-pair and cell-membership index."""

    connection.executescript(
        """
        CREATE TABLE sentinel_pairs (
            source_state_sha256 TEXT NOT NULL,
            action_sha256 TEXT NOT NULL,
            rank_sha256 TEXT NOT NULL,
            PRIMARY KEY (source_state_sha256, action_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE sentinel_pair_cells (
            capability_cell_id TEXT NOT NULL,
            source_state_sha256 TEXT NOT NULL,
            action_sha256 TEXT NOT NULL,
            PRIMARY KEY (
                capability_cell_id,
                source_state_sha256,
                action_sha256
            )
        ) WITHOUT ROWID;
        """
    )


def index_sentinel_transition(
    connection: sqlite3.Connection, transition: Mapping[str, Any]
) -> None:
    """Add one accepted transition without retaining its occurrence payload."""

    source_sha256, action_sha256 = _pair(transition)
    connection.execute(
        """
        INSERT OR IGNORE INTO sentinel_pairs (
            source_state_sha256,
            action_sha256,
            rank_sha256
        ) VALUES (?, ?, ?)
        """,
        (
            source_sha256,
            action_sha256,
            sentinel_rank(source_sha256, action_sha256),
        ),
    )
    cell = transition["capability_cell_id"]
    if cell is not None:
        connection.execute(
            """
            INSERT OR IGNORE INTO sentinel_pair_cells (
                capability_cell_id,
                source_state_sha256,
                action_sha256
            ) VALUES (?, ?, ?)
            """,
            (str(cell), source_sha256, action_sha256),
        )


def _indexed_pairs(
    connection: sqlite3.Connection,
    *,
    capability_cell_id: str | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    if capability_cell_id is None:
        statement = """
            SELECT source_state_sha256, action_sha256
            FROM sentinel_pairs
            ORDER BY rank_sha256, source_state_sha256, action_sha256
        """
        parameters: tuple[Any, ...] = ()
    else:
        statement = """
            SELECT pairs.source_state_sha256, pairs.action_sha256
            FROM sentinel_pair_cells AS cells
            JOIN sentinel_pairs AS pairs USING (
                source_state_sha256,
                action_sha256
            )
            WHERE cells.capability_cell_id = ?
            ORDER BY
                pairs.rank_sha256,
                pairs.source_state_sha256,
                pairs.action_sha256
        """
        parameters = (str(capability_cell_id),)
    if limit is not None:
        statement += " LIMIT ?"
        parameters = (*parameters, limit)
    return [
        (str(source), str(action)) for source, action in connection.execute(statement, parameters)
    ]


def select_sentinel_pairs_from_index(
    connection: sqlite3.Connection, *, required_cell_ids: Sequence[str]
) -> dict[str, Any]:
    """Select from the disk index with byte-equivalent eager semantics."""

    row = connection.execute("SELECT COUNT(*) FROM sentinel_pairs").fetchone()
    if row is None:
        raise ProcessV2Active8SentinelError(
            "the sentinel pair index did not return its unique-pair census"
        )
    unique_count = int(row[0])
    if unique_count <= SENTINEL_EXHAUSTIVE_THRESHOLD:
        selected = _indexed_pairs(connection)
        return {
            "selection_mode": SELECTION_EXHAUSTIVE,
            "unique_accepted_pairs": unique_count,
            "per_cell_examples": 0,
            "global_examples": 0,
            "selected": selected,
        }
    per_cell: set[tuple[str, str]] = set()
    for cell in sorted({str(cell) for cell in required_cell_ids}):
        per_cell.update(
            _indexed_pairs(
                connection,
                capability_cell_id=cell,
                limit=SENTINEL_PER_CELL_EXAMPLES,
            )
        )
    globally = _indexed_pairs(connection, limit=SENTINEL_GLOBAL_EXAMPLES)
    selected = sorted(per_cell | set(globally), key=_pair_rank_key)
    return {
        "selection_mode": SELECTION_RANKED,
        "unique_accepted_pairs": unique_count,
        "per_cell_examples": len(per_cell),
        "global_examples": len(globally),
        "selected": selected,
    }


def select_sentinel_pairs(
    transitions: Sequence[Mapping[str, Any]], *, required_cell_ids: Sequence[str]
) -> dict[str, Any]:
    """Choose the pairs to re-enumerate, deterministically and salt-ranked."""

    occurrences: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    cells: dict[str, set[tuple[str, str]]] = {cell: set() for cell in required_cell_ids}
    for transition in transitions:
        pair = _pair(transition)
        occurrences.setdefault(pair, []).append(transition)
        cell = transition["capability_cell_id"]
        if cell in cells:
            cells[str(cell)].add(pair)
    # One re-enumeration per pair, compared against EVERY transition carrying it.
    # Deduplicating to a representative and checking only that one would leave a
    # moved count in any other occurrence of the same pair undetected, which is
    # precisely the residual this instrument exists to sample.
    for published in occurrences.values():
        published.sort(
            key=lambda transition: (
                str(transition["v1_task_identity_sha256"]),
                int(transition["entry_index"]),
                int(transition["step_index"]),
            )
        )
    representatives = {pair: rows[0] for pair, rows in occurrences.items()}
    unique = sorted(occurrences)
    if len(unique) <= SENTINEL_EXHAUSTIVE_THRESHOLD:
        selected = sorted(unique, key=_pair_rank_key)
        return {
            "selection_mode": SELECTION_EXHAUSTIVE,
            "unique_accepted_pairs": len(unique),
            "per_cell_examples": 0,
            "global_examples": 0,
            "selected": selected,
            "representatives": representatives,
            "occurrences": occurrences,
        }
    per_cell: set[tuple[str, str]] = set()
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=_pair_rank_key)
        per_cell.update(ranked[:SENTINEL_PER_CELL_EXAMPLES])
    globally = sorted(unique, key=_pair_rank_key)[:SENTINEL_GLOBAL_EXAMPLES]
    selected = sorted(per_cell | set(globally), key=_pair_rank_key)
    return {
        "selection_mode": SELECTION_RANKED,
        "unique_accepted_pairs": len(unique),
        "per_cell_examples": len(per_cell),
        "global_examples": len(globally),
        "selected": selected,
        "representatives": representatives,
        "occurrences": occurrences,
    }


def _dictionary_successor_oracle(result: Any, state: Any, *, system: Any) -> dict[str, int]:
    """Regroup the production marked law with an ordinary dictionary.

    BOUNDED TEST INSTRUMENT.  It reuses the production executor and canonical
    key and reimplements only the aggregation, so it can disagree with the
    production quotient exactly where the production grouping is wrong.
    """

    source_key = canonical_state_key(state)
    aliases: dict[str, int] = {}
    for mark in result.marked_law.marks:
        successor = system.apply(state, mark.executor_rule_name, mark.action)
        key = canonical_state_key(successor)
        if key == source_key:
            continue
        aliases[key] = aliases.get(key, 0) + 1
    return aliases


def _occurrence_projection(transition: Mapping[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(transition[field]) for field in SENTINEL_OCCURRENCE_FIELDS}


def _pair_key(pair: Mapping[str, Any]) -> list[str]:
    return [str(pair["source_state_sha256"]), str(pair["action_sha256"])]


def _partition_identity_body(
    partition: Mapping[str, Any], *, context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "binding_sha256": str(context["binding_sha256"]),
        "plan_sha256": str(context["plan_sha256"]),
        "run_identity_sha256": str(context["run_identity_sha256"]),
        "task_inventory_sha256": str(context["task_inventory_sha256"]),
        "result_inventory_sha256": str(context["result_inventory_sha256"]),
        **{key: value for key, value in partition.items() if key != "partition_identity_sha256"},
    }


def prepare_release_sentinel_plan(
    plan: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    *,
    result_inventory_sha256: str,
    max_pairs_per_partition: int,
) -> dict[str, Any]:
    """Freeze the exact selection into source-chunk-aligned worker partitions."""

    selection = select_sentinel_pairs(
        transitions, required_cell_ids=list(plan["binding"]["required_cell_ids"])
    )
    selected_occurrences = (
        transition
        for pair in selection["selected"]
        for transition in selection["occurrences"][pair]
    )
    return prepare_release_sentinel_plan_from_selection(
        plan,
        selection,
        selected_occurrences,
        result_inventory_sha256=result_inventory_sha256,
        max_pairs_per_partition=max_pairs_per_partition,
    )


def prepare_release_sentinel_plan_from_selection(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    selected_transitions: Iterable[Mapping[str, Any]],
    *,
    result_inventory_sha256: str,
    max_pairs_per_partition: int,
) -> dict[str, Any]:
    """Build the unchanged plan from a bounded selection and its occurrences."""

    if type(max_pairs_per_partition) is not int or max_pairs_per_partition <= 0:
        raise ProcessV2Active8SentinelError("max_pairs_per_partition must be a positive integer")
    if _SHA256.fullmatch(str(result_inventory_sha256)) is None:
        raise ProcessV2Active8SentinelError("the result inventory identity is not a SHA-256")
    selected: list[tuple[str, str]] = list(selection["selected"])
    if (
        any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(value, str) for value in pair)
            for pair in selected
        )
        or len(set(selected)) != len(selected)
        or selected != sorted(selected, key=_pair_rank_key)
    ):
        raise ProcessV2Active8SentinelError(
            "the bounded sentinel selection pair inventory is not canonical"
        )
    selection_counts = {
        "selection_mode": selection["selection_mode"],
        "unique_accepted_pairs": selection["unique_accepted_pairs"],
        "selected_pairs": len(selected),
        "per_cell_examples": selection["per_cell_examples"],
        "global_examples": selection["global_examples"],
        "oracle_examples": min(len(selected), SENTINEL_ORACLE_EXAMPLES),
    }
    _require_selection_counts(selection_counts)
    occurrences: dict[tuple[str, str], list[Mapping[str, Any]]] = {pair: [] for pair in selected}
    for transition in selected_transitions:
        pair = _pair(transition)
        if pair not in occurrences:
            raise ProcessV2Active8SentinelError(
                "the sentinel occurrence stream contains an unselected pair"
            )
        occurrences[pair].append(transition)
    for pair, published in occurrences.items():
        if not published:
            raise ProcessV2Active8SentinelError(
                f"the selected sentinel pair has no occurrence: {pair}"
            )
        published.sort(
            key=lambda transition: (
                str(transition["v1_task_identity_sha256"]),
                int(transition["entry_index"]),
                int(transition["step_index"]),
            )
        )
    representatives = {pair: rows[0] for pair, rows in occurrences.items()}
    oracle_pairs = set(selected[:SENTINEL_ORACLE_EXAMPLES])
    tasks = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for source_sha256, action_sha256 in selected:
        pair = (source_sha256, action_sha256)
        anchor = representatives[pair]
        task_identity = str(anchor["task_identity_sha256"])
        if task_identity not in tasks:
            raise ProcessV2Active8SentinelError(
                f"a selected transition names an unplanned task: {task_identity}"
            )
        pair_body = {
            "source_state_sha256": source_sha256,
            "action_sha256": action_sha256,
            "rank_sha256": sentinel_rank(source_sha256, action_sha256),
            "source_task_identity_sha256": task_identity,
            "source_entry_index": int(anchor["entry_index"]),
            "source_step_index": int(anchor["step_index"]),
            "source_model_family": str(anchor["model_family"]),
            "oracle_required": pair in oracle_pairs,
            "occurrences": [_occurrence_projection(item) for item in occurrences[pair]],
        }
        pair_record = self_hashed(pair_body, field="pair_sha256")
        if tuple(pair_record) != SENTINEL_PAIR_FIELDS:
            raise ProcessV2Active8SentinelError("the sentinel pair field set disagrees")
        by_task.setdefault(task_identity, []).append(pair_record)

    context = {
        "binding_sha256": str(plan["binding_sha256"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "run_identity_sha256": str(plan["run_identity_sha256"]),
        "task_inventory_sha256": str(plan["task_inventory_sha256"]),
        "result_inventory_sha256": str(result_inventory_sha256),
    }
    partitions: list[dict[str, Any]] = []
    for task_identity in sorted(by_task):
        task_pairs = by_task[task_identity]
        for start in range(0, len(task_pairs), max_pairs_per_partition):
            items = task_pairs[start : start + max_pairs_per_partition]
            selected_keys = [_pair_key(item) for item in items]
            partition_body = {
                "partition_index": len(partitions),
                "source_task_identity_sha256": task_identity,
                "selected_pairs": len(items),
                "selected_pairs_sha256": canonical_sha256(selected_keys),
                "oracle_examples": sum(bool(item["oracle_required"]) for item in items),
                "pairs": items,
            }
            partitions.append(
                {
                    **partition_body,
                    "partition_identity_sha256": canonical_sha256(
                        _partition_identity_body(partition_body, context=context)
                    ),
                }
            )
    selected_keys = [[source, action] for source, action in selected]
    body = {
        "schema": SENTINEL_PLAN_SCHEMA,
        "schema_version": SENTINEL_PLAN_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "salt": SENTINEL_SALT,
        **context,
        "unique_accepted_pairs": int(selection_counts["unique_accepted_pairs"]),
        "selection_mode": str(selection_counts["selection_mode"]),
        "selected_pairs": len(selected),
        "selected_pairs_sha256": canonical_sha256(selected_keys),
        "per_cell_examples": int(selection_counts["per_cell_examples"]),
        "global_examples": int(selection_counts["global_examples"]),
        "oracle_examples": min(len(selected), SENTINEL_ORACLE_EXAMPLES),
        "max_pairs_per_partition": max_pairs_per_partition,
        "partitions": partitions,
        "partition_inventory_sha256": canonical_sha256(
            [item["partition_identity_sha256"] for item in partitions]
        ),
    }
    sentinel_plan = self_hashed(body, field="sentinel_plan_sha256")
    validate_release_sentinel_plan(sentinel_plan, active8_plan=plan)
    return sentinel_plan


def validate_release_sentinel_plan(
    value: Mapping[str, Any], *, active8_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate selection, partition geometry, and every nested content hash."""

    if not isinstance(value, Mapping) or set(value) != set(SENTINEL_PLAN_FIELDS):
        raise ProcessV2Active8SentinelError("the sentinel plan field set disagrees")
    sentinel_plan = dict(value)
    verify_self_hash(sentinel_plan, field="sentinel_plan_sha256", label="the sentinel plan")
    if (
        sentinel_plan["schema"] != SENTINEL_PLAN_SCHEMA
        or sentinel_plan["schema_version"] != SENTINEL_PLAN_SCHEMA_VERSION
        or sentinel_plan["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or sentinel_plan["salt"] != SENTINEL_SALT
    ):
        raise ProcessV2Active8SentinelError("the sentinel plan protocol disagrees")
    for field, false_value in authority_false_block().items():
        if sentinel_plan[field] is not false_value:
            raise ProcessV2Active8SentinelError(
                f"the sentinel plan grants forbidden authority through {field}"
            )
    context = {
        "binding_sha256": str(active8_plan["binding_sha256"]),
        "plan_sha256": str(active8_plan["plan_sha256"]),
        "run_identity_sha256": str(active8_plan["run_identity_sha256"]),
        "task_inventory_sha256": str(active8_plan["task_inventory_sha256"]),
        "result_inventory_sha256": str(sentinel_plan["result_inventory_sha256"]),
    }
    for field, expected in context.items():
        observed = sentinel_plan[field]
        if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
            raise ProcessV2Active8SentinelError(f"the sentinel plan {field} is not a SHA-256")
        if field != "result_inventory_sha256" and observed != expected:
            raise ProcessV2Active8SentinelError(
                f"the sentinel plan {field} addresses another Active8 run"
            )
    max_pairs = sentinel_plan["max_pairs_per_partition"]
    if type(max_pairs) is not int or max_pairs <= 0:
        raise ProcessV2Active8SentinelError(
            "the sentinel partition size must be a positive integer"
        )
    planned_tasks = {str(task["task_identity_sha256"]) for task in active8_plan["tasks"]}
    partitions = sentinel_plan["partitions"]
    if not isinstance(partitions, list):
        raise ProcessV2Active8SentinelError("the sentinel partitions must be a list")
    identities: list[str] = []
    all_pairs: list[dict[str, Any]] = []
    occurrence_locations: set[tuple[str, int, int, str, str]] = set()
    for expected_index, partition in enumerate(partitions):
        if not isinstance(partition, Mapping) or set(partition) != set(SENTINEL_PARTITION_FIELDS):
            raise ProcessV2Active8SentinelError("a sentinel partition field set disagrees")
        if (
            type(partition["partition_index"]) is not int
            or partition["partition_index"] != expected_index
        ):
            raise ProcessV2Active8SentinelError("the sentinel partition indices are not canonical")
        task_identity = partition["source_task_identity_sha256"]
        if task_identity not in planned_tasks:
            raise ProcessV2Active8SentinelError(
                "a sentinel partition names an unplanned source task"
            )
        pairs = partition["pairs"]
        if (
            not isinstance(pairs, list)
            or not pairs
            or len(pairs) > max_pairs
            or type(partition["selected_pairs"]) is not int
            or partition["selected_pairs"] != len(pairs)
        ):
            raise ProcessV2Active8SentinelError("a sentinel partition pair inventory is malformed")
        pair_keys: list[list[str]] = []
        oracle_examples = 0
        for pair in pairs:
            if not isinstance(pair, Mapping) or set(pair) != set(SENTINEL_PAIR_FIELDS):
                raise ProcessV2Active8SentinelError("a sentinel selected-pair field set disagrees")
            verify_self_hash(pair, field="pair_sha256", label="a sentinel selected pair")
            source_sha256, action_sha256 = _pair_key(pair)
            if (
                _SHA256.fullmatch(source_sha256) is None
                or _SHA256.fullmatch(action_sha256) is None
                or pair["rank_sha256"] != sentinel_rank(source_sha256, action_sha256)
                or pair["source_task_identity_sha256"] != task_identity
                or type(pair["source_entry_index"]) is not int
                or pair["source_entry_index"] < 0
                or type(pair["source_step_index"]) is not int
                or pair["source_step_index"] < 0
                or not isinstance(pair["source_model_family"], str)
            ):
                raise ProcessV2Active8SentinelError("a sentinel selected-pair identity disagrees")
            occurrences = pair["occurrences"]
            if not isinstance(occurrences, list) or not occurrences:
                raise ProcessV2Active8SentinelError(
                    "a sentinel selected pair has no published occurrence"
                )
            representative_found = False
            for occurrence in occurrences:
                if not isinstance(occurrence, Mapping) or set(occurrence) != set(
                    SENTINEL_OCCURRENCE_FIELDS
                ):
                    raise ProcessV2Active8SentinelError("a sentinel occurrence field set disagrees")
                evidence = occurrence["candidate_evidence"]
                if (
                    not isinstance(evidence, Mapping)
                    or evidence.get("source_state_sha256") != source_sha256
                    or evidence.get("action_sha256") != action_sha256
                    or type(occurrence["entry_index"]) is not int
                    or occurrence["entry_index"] < 0
                    or type(occurrence["step_index"]) is not int
                    or occurrence["step_index"] < 0
                ):
                    raise ProcessV2Active8SentinelError(
                        "a sentinel occurrence belongs to another selected pair"
                    )
                location = (
                    str(occurrence["v1_task_identity_sha256"]),
                    occurrence["entry_index"],
                    occurrence["step_index"],
                    source_sha256,
                    action_sha256,
                )
                if location in occurrence_locations:
                    raise ProcessV2Active8SentinelError(
                        "the sentinel plan repeats a published occurrence"
                    )
                occurrence_locations.add(location)
                if (
                    occurrence["task_identity_sha256"] == task_identity
                    and occurrence["entry_index"] == pair["source_entry_index"]
                    and occurrence["step_index"] == pair["source_step_index"]
                ):
                    representative_found = True
            if not representative_found:
                raise ProcessV2Active8SentinelError(
                    "a sentinel pair does not contain its source occurrence"
                )
            if type(pair["oracle_required"]) is not bool:
                raise ProcessV2Active8SentinelError("a sentinel pair oracle flag is not boolean")
            oracle_examples += bool(pair["oracle_required"])
            pair_keys.append([source_sha256, action_sha256])
            all_pairs.append(dict(pair))
        if (
            not isinstance(partition["selected_pairs_sha256"], str)
            or _SHA256.fullmatch(partition["selected_pairs_sha256"]) is None
            or canonical_sha256(pair_keys) != partition["selected_pairs_sha256"]
        ):
            raise ProcessV2Active8SentinelError(
                "a sentinel partition selected-pair digest disagrees"
            )
        if (
            type(partition["oracle_examples"]) is not int
            or partition["oracle_examples"] != oracle_examples
        ):
            raise ProcessV2Active8SentinelError("a sentinel partition oracle count disagrees")
        expected_identity = canonical_sha256(_partition_identity_body(partition, context=context))
        if (
            not isinstance(partition["partition_identity_sha256"], str)
            or _SHA256.fullmatch(partition["partition_identity_sha256"]) is None
            or partition["partition_identity_sha256"] != expected_identity
        ):
            raise ProcessV2Active8SentinelError("a sentinel partition identity disagrees")
        identities.append(str(partition["partition_identity_sha256"]))
    if (
        not isinstance(sentinel_plan["partition_inventory_sha256"], str)
        or _SHA256.fullmatch(sentinel_plan["partition_inventory_sha256"]) is None
        or canonical_sha256(identities) != sentinel_plan["partition_inventory_sha256"]
    ):
        raise ProcessV2Active8SentinelError("the sentinel partition inventory digest disagrees")
    by_rank = sorted(
        all_pairs,
        key=lambda pair: (
            str(pair["rank_sha256"]),
            str(pair["source_state_sha256"]),
            str(pair["action_sha256"]),
        ),
    )
    selected_keys = [_pair_key(pair) for pair in by_rank]
    if (
        len({tuple(pair) for pair in selected_keys}) != len(selected_keys)
        or not isinstance(sentinel_plan["selected_pairs_sha256"], str)
        or _SHA256.fullmatch(sentinel_plan["selected_pairs_sha256"]) is None
        or sentinel_plan["selected_pairs"] != len(selected_keys)
        or sentinel_plan["selected_pairs_sha256"] != canonical_sha256(selected_keys)
    ):
        raise ProcessV2Active8SentinelError("the sentinel plan selected-pair inventory disagrees")
    expected_oracle = min(len(selected_keys), SENTINEL_ORACLE_EXAMPLES)
    if sentinel_plan["oracle_examples"] != expected_oracle or [
        bool(pair["oracle_required"]) for pair in by_rank
    ] != [index < expected_oracle for index in range(len(by_rank))]:
        raise ProcessV2Active8SentinelError("the sentinel plan oracle selection disagrees")
    _require_selection_counts(sentinel_plan)
    max_per_cell_examples = min(
        int(sentinel_plan["unique_accepted_pairs"]),
        SENTINEL_PER_CELL_EXAMPLES * len(active8_plan["binding"]["required_cell_ids"]),
    )
    if int(sentinel_plan["per_cell_examples"]) > max_per_cell_examples:
        raise ProcessV2Active8SentinelError(
            "the sentinel per-cell selection exceeds its frozen bound"
        )
    return sentinel_plan


def _require_selection_counts(value: Mapping[str, Any]) -> None:
    count_fields = (
        "unique_accepted_pairs",
        "selected_pairs",
        "per_cell_examples",
        "global_examples",
        "oracle_examples",
    )
    if any(type(value[field]) is not int or value[field] < 0 for field in count_fields):
        raise ProcessV2Active8SentinelError(
            "the sentinel selection counts must be nonnegative integers"
        )
    if value["selected_pairs"] > value["unique_accepted_pairs"]:
        raise ProcessV2Active8SentinelError("the sentinel selects more pairs than exist")
    if (value["selected_pairs"] == 0) != (value["unique_accepted_pairs"] == 0):
        raise ProcessV2Active8SentinelError(
            "the sentinel empty selection does not match its unique-pair census"
        )
    if value["selection_mode"] == SELECTION_EXHAUSTIVE:
        if (
            value["unique_accepted_pairs"] > SENTINEL_EXHAUSTIVE_THRESHOLD
            or value["selected_pairs"] != value["unique_accepted_pairs"]
            or value["per_cell_examples"] != 0
            or value["global_examples"] != 0
        ):
            raise ProcessV2Active8SentinelError(
                "the exhaustive sentinel selection counts do not reconcile"
            )
    elif value["selection_mode"] == SELECTION_RANKED:
        if value["unique_accepted_pairs"] <= SENTINEL_EXHAUSTIVE_THRESHOLD:
            raise ProcessV2Active8SentinelError(
                "the ranked sentinel selection is below the exhaustive threshold"
            )
        expected_global = min(value["unique_accepted_pairs"], SENTINEL_GLOBAL_EXAMPLES)
        if (
            value["global_examples"] != expected_global
            or value["selected_pairs"] < value["global_examples"]
            or value["selected_pairs"] < value["per_cell_examples"]
            or value["selected_pairs"] > value["global_examples"] + value["per_cell_examples"]
        ):
            raise ProcessV2Active8SentinelError(
                "the ranked sentinel selection counts do not reconcile"
            )
    else:
        raise ProcessV2Active8SentinelError("the sentinel selection mode is unknown")


def sentinel_partition_identities(sentinel_plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact worker inventory in deterministic plan order."""

    return tuple(
        str(partition["partition_identity_sha256"]) for partition in sentinel_plan["partitions"]
    )


def run_release_sentinel_partition(
    plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    partition_identity_sha256: str,
    *,
    checker: ProductionProcessV2SemanticExactCandidateChecker,
    namespace: str,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Re-enumerate one independently addressed source-chunk partition."""

    validated = validate_release_sentinel_plan(sentinel_plan, active8_plan=plan)
    matching = [
        partition
        for partition in validated["partitions"]
        if partition["partition_identity_sha256"] == partition_identity_sha256
    ]
    if len(matching) != 1:
        raise ProcessV2Active8SentinelError(
            "the requested sentinel partition is absent or ambiguous"
        )
    partition = matching[0]
    tasks = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    task_identity = str(partition["source_task_identity_sha256"])
    task = tasks.get(task_identity)
    if task is None:
        raise ProcessV2Active8SentinelError("the sentinel partition names an unplanned task")
    rebind_output = mounted_process_v2_artifact_path(
        str(task["rebind_task_artifact_path"]),
        artifact_root=Path(artifact_root),
        field="task.rebind_task_artifact_path",
    )
    try:
        rebind_receipt = _validated_rebind_chunk_receipt(
            rebind_output,
            entry_start=int(task["entry_start"]),
            entry_stop=int(task["entry_stop"]),
            task_identity_sha256=str(task["rebind_task_identity_sha256"]),
            chunk_file_sha256=str(task["chunk_file_sha256"]),
            pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        )
    except ProcessV2Active8MapError as error:
        raise ProcessV2Active8SentinelError(
            "the sentinel source is not authorized by the bound rebind receipt"
        ) from error
    expected_process_identity = dict(rebind_receipt["pinned_process_identity"])
    pairs = list(partition["pairs"])
    wanted: dict[int, list[Mapping[str, Any]]] = {}
    for pair in pairs:
        wanted.setdefault(int(pair["source_entry_index"]), []).append(pair)
    target = _chunk_target(task)
    source_output = mounted_process_v2_artifact_path(
        target.source_artifact_path,
        artifact_root=Path(artifact_root),
        field="task.cache_source_artifact_path",
    )
    evidence_mismatches: list[str] = []
    cell_mismatches: list[str] = []
    oracle_mismatches: list[str] = []
    evaluated: list[list[str]] = []
    oracle_checked = 0
    for read in read_process_v2_chunk_target(
        source_output,
        target=target,
        expected_process_identity=expected_process_identity,
        sentinel_replay_entries=0,
        recover_row_errors=False,
        repo_root=Path(repo_root),
    ):
        selected = wanted.get(int(read.entry_index))
        if not selected or read.addressed is None:
            continue
        for pair in selected:
            step_index = int(pair["source_step_index"])
            addressed = read.addressed
            audit = checker.evaluate(addressed, step_index)
            source = addressed.path.state_at(step_index)
            successor = addressed.path.state_at(step_index + 1)
            exact = audit.evidence
            teacher_coordinate_legal = exact.matching_mark_count == 1
            teacher_executes_to_exact_successor = (
                audit.exact_successor_mark_count == 1
            )
            productive_canonical_successor = (
                exact.source_canonical_key != exact.canonical_successor_key
            )
            observed = {
                "supported": bool(exact.supported),
                "exclusion_reason": exact.exclusion_reason,
                "action_sha256": exact.action_sha256,
                "source_state_sha256": exact.source_state_sha256,
                "target_state_sha256": exact.target_state_sha256,
                "source_canonical_key": exact.source_canonical_key,
                "canonical_successor_key": exact.canonical_successor_key,
                "teacher_coordinate_legal": teacher_coordinate_legal,
                "teacher_executes_to_exact_successor": (
                    teacher_executes_to_exact_successor
                ),
                "productive_canonical_successor": productive_canonical_successor,
            }
            block = _classification_block(
                source,
                successor,
                addressed.trace.steps[step_index],
                family=str(pair["source_model_family"]),
                supported=bool(audit.evidence.supported),
                step_index=step_index,
                path_length=int(addressed.address.path_length),
                namespace=namespace,
            )
            for published in pair["occurrences"]:
                where = (
                    f"{published['v1_task_identity_sha256']}"
                    f"[{published['entry_index']}]:{published['step_index']}"
                )
                if observed != dict(published["candidate_evidence"]):
                    evidence_mismatches.append(where)
                if (
                    block["capability_cell_id"] != published["capability_cell_id"]
                    or block["family_context"] != published["family_context"]
                    or block["audit_axes"] != published["audit_axes"]
                ):
                    cell_mismatches.append(where)
            if pair["oracle_required"]:
                oracle_checked += 1
                if _oracle_disagrees(checker, source):
                    oracle_mismatches.append(f"{task_identity}[{read.entry_index}]:{step_index}")
            evaluated.append(_pair_key(pair))
    selected_keys = [_pair_key(pair) for pair in pairs]
    if sorted(evaluated) != sorted(selected_keys) or len(evaluated) != len(selected_keys):
        raise ProcessV2Active8SentinelError(
            "the sentinel partition did not evaluate its exact selected-pair inventory"
        )
    mismatched = bool(evidence_mismatches or cell_mismatches or oracle_mismatches)
    body = {
        "schema": SENTINEL_PARTITION_RESULT_SCHEMA,
        "schema_version": SENTINEL_PARTITION_RESULT_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "partition_outcome": (
            SENTINEL_PARTITION_MISMATCH if mismatched else SENTINEL_PARTITION_MATCHED
        ),
        "sentinel_plan_sha256": str(validated["sentinel_plan_sha256"]),
        "partition_identity_sha256": str(partition["partition_identity_sha256"]),
        "partition_index": int(partition["partition_index"]),
        "binding_sha256": str(validated["binding_sha256"]),
        "plan_sha256": str(validated["plan_sha256"]),
        "run_identity_sha256": str(validated["run_identity_sha256"]),
        "task_inventory_sha256": str(validated["task_inventory_sha256"]),
        "result_inventory_sha256": str(validated["result_inventory_sha256"]),
        "selected_pairs": int(partition["selected_pairs"]),
        "selected_pairs_sha256": str(partition["selected_pairs_sha256"]),
        "evaluated_pairs": len(evaluated),
        "evaluated_pairs_sha256": canonical_sha256(selected_keys),
        "oracle_examples": oracle_checked,
        "evidence_mismatches": sorted(evidence_mismatches),
        "cell_mismatches": sorted(cell_mismatches),
        "oracle_mismatches": sorted(oracle_mismatches),
    }
    result = self_hashed(body, field="partition_result_sha256")
    validate_release_sentinel_partition_result(result, sentinel_plan=validated)
    return result


def validate_release_sentinel_partition_result(
    value: Mapping[str, Any], *, sentinel_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate one result against a sealed sentinel-plan envelope.

    Authoritative callers additionally validate the complete plan against the
    Active8 run before reaching this helper.  Requiring the envelope here keeps
    the standalone API from accepting an unsealed mutation while avoiding a
    second, weaker reimplementation of full plan validation.
    """

    if not isinstance(sentinel_plan, Mapping) or set(sentinel_plan) != set(SENTINEL_PLAN_FIELDS):
        raise ProcessV2Active8SentinelError("the sentinel plan field set disagrees")
    try:
        verify_self_hash(
            sentinel_plan,
            field="sentinel_plan_sha256",
            label="the sentinel plan",
        )
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8SentinelError(str(error)) from error
    if (
        sentinel_plan["schema"] != SENTINEL_PLAN_SCHEMA
        or sentinel_plan["schema_version"] != SENTINEL_PLAN_SCHEMA_VERSION
        or sentinel_plan["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or sentinel_plan["salt"] != SENTINEL_SALT
    ):
        raise ProcessV2Active8SentinelError("the sentinel plan protocol disagrees")
    for field, false_value in authority_false_block().items():
        if sentinel_plan[field] is not false_value:
            raise ProcessV2Active8SentinelError(
                f"the sentinel plan grants forbidden authority through {field}"
            )

    if not isinstance(value, Mapping) or set(value) != set(SENTINEL_PARTITION_RESULT_FIELDS):
        raise ProcessV2Active8SentinelError("the sentinel partition result field set disagrees")
    result = dict(value)
    verify_self_hash(
        result,
        field="partition_result_sha256",
        label="the sentinel partition result",
    )
    if (
        result["schema"] != SENTINEL_PARTITION_RESULT_SCHEMA
        or result["schema_version"] != SENTINEL_PARTITION_RESULT_SCHEMA_VERSION
        or result["status"] != PIPELINE_STATUS_NO_AUTHORITY
    ):
        raise ProcessV2Active8SentinelError("the sentinel partition result protocol disagrees")
    for field, false_value in authority_false_block().items():
        if result[field] is not false_value:
            raise ProcessV2Active8SentinelError(
                f"the sentinel partition result grants authority through {field}"
            )
    if result["sentinel_plan_sha256"] != sentinel_plan["sentinel_plan_sha256"]:
        raise ProcessV2Active8SentinelError(
            "the sentinel partition result belongs to another sentinel plan"
        )
    partitions = {
        str(partition["partition_identity_sha256"]): partition
        for partition in sentinel_plan["partitions"]
    }
    partition = partitions.get(str(result["partition_identity_sha256"]))
    if partition is None:
        raise ProcessV2Active8SentinelError(
            "the sentinel partition result is not in the planned inventory"
        )
    for field in (
        "partition_index",
        "selected_pairs",
        "selected_pairs_sha256",
    ):
        if result[field] != partition[field]:
            raise ProcessV2Active8SentinelError(
                f"the sentinel partition result {field} disagrees with its plan"
            )
    for field in (
        "binding_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "task_inventory_sha256",
        "result_inventory_sha256",
    ):
        if result[field] != sentinel_plan[field]:
            raise ProcessV2Active8SentinelError(
                f"the sentinel partition result {field} addresses another run"
            )
    expected_keys = [_pair_key(pair) for pair in partition["pairs"]]
    if (
        type(result["partition_index"]) is not int
        or type(result["selected_pairs"]) is not int
        or type(result["evaluated_pairs"]) is not int
        or type(result["oracle_examples"]) is not int
        or result["evaluated_pairs"] != len(expected_keys)
        or result["evaluated_pairs_sha256"] != canonical_sha256(expected_keys)
        or result["oracle_examples"] != partition["oracle_examples"]
    ):
        raise ProcessV2Active8SentinelError(
            "the sentinel partition result coverage does not reconcile"
        )
    mismatch_fields = (
        "evidence_mismatches",
        "cell_mismatches",
        "oracle_mismatches",
    )
    for field in mismatch_fields:
        values = result[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) for item in values)
            or values != sorted(values)
        ):
            raise ProcessV2Active8SentinelError(
                f"the sentinel partition result {field} is malformed"
            )
    mismatched = any(result[field] for field in mismatch_fields)
    expected_outcome = SENTINEL_PARTITION_MISMATCH if mismatched else SENTINEL_PARTITION_MATCHED
    if result["partition_outcome"] != expected_outcome:
        raise ProcessV2Active8SentinelError(
            "the sentinel partition result outcome contradicts its mismatches"
        )
    return result


def sentinel_partition_result_path(
    active8_plan: Mapping[str, Any],
    partition_identity_sha256: str,
    *,
    artifact_root: Path,
) -> Path:
    """Resolve one content-addressed partition result inside the run root."""

    if _SHA256.fullmatch(str(partition_identity_sha256)) is None:
        raise ProcessV2Active8SentinelError("the sentinel partition identity is not a SHA-256")
    run_root = mounted_process_v2_artifact_path(
        str(active8_plan["run_artifact_root"]),
        artifact_root=Path(artifact_root),
        field="plan.run_artifact_root",
    )
    return (
        run_root
        / SENTINEL_PARTITIONS_DIRNAME
        / str(partition_identity_sha256)
        / SENTINEL_PARTITION_RESULT_FILENAME
    )


def write_release_sentinel_partition_result(
    result: Mapping[str, Any],
    *,
    active8_plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    artifact_root: Path,
) -> Path:
    """Publish a result immutably; an exact retry reuses the same bytes."""

    validated_plan = validate_release_sentinel_plan(sentinel_plan, active8_plan=active8_plan)
    validated = validate_release_sentinel_partition_result(result, sentinel_plan=validated_plan)
    path = sentinel_partition_result_path(
        active8_plan,
        str(validated["partition_identity_sha256"]),
        artifact_root=Path(artifact_root),
    )
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2Active8SentinelError(
            f"immutable sentinel partition publication failed: {path}"
        ) from error
    return path


def load_release_sentinel_partition_result(
    partition_identity_sha256: str,
    *,
    active8_plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Reopen canonical bytes and independently validate one partition result."""

    validated_plan = validate_release_sentinel_plan(sentinel_plan, active8_plan=active8_plan)
    path = sentinel_partition_result_path(
        active8_plan,
        partition_identity_sha256,
        artifact_root=Path(artifact_root),
    )
    try:
        raw = path.read_bytes()
        result = json.loads(raw)
    except OSError as error:
        raise ProcessV2Active8SentinelError(
            f"the sentinel partition result is absent: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ProcessV2Active8SentinelError(
            f"the sentinel partition result is not valid JSON: {path}"
        ) from error
    if canonical_bytes(result) + b"\n" != raw:
        raise ProcessV2Active8SentinelError("the sentinel partition result is not canonical JSON")
    return validate_release_sentinel_partition_result(result, sentinel_plan=validated_plan)


def completed_release_sentinel_partition_ids(
    *,
    active8_plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    artifact_root: Path,
) -> set[str]:
    """Return only exact planned partition results safe to reuse on restart."""

    validated_plan = validate_release_sentinel_plan(sentinel_plan, active8_plan=active8_plan)
    complete: set[str] = set()
    for identity in sentinel_partition_identities(validated_plan):
        try:
            load_release_sentinel_partition_result(
                identity,
                active8_plan=active8_plan,
                sentinel_plan=validated_plan,
                artifact_root=Path(artifact_root),
            )
        except (ProcessV2Active8SentinelError, OSError, ValueError):
            continue
        complete.add(identity)
    return complete


def reduce_release_sentinel_partitions(
    plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    partition_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce the exact partition inventory; input order cannot change bytes."""

    validated = validate_release_sentinel_plan(sentinel_plan, active8_plan=plan)
    by_identity: dict[str, dict[str, Any]] = {}
    for value in partition_results:
        result = validate_release_sentinel_partition_result(value, sentinel_plan=validated)
        identity = str(result["partition_identity_sha256"])
        if identity in by_identity:
            raise ProcessV2Active8SentinelError(
                "the sentinel reducer received one partition result twice"
            )
        by_identity[identity] = result
    expected = sentinel_partition_identities(validated)
    if set(by_identity) != set(expected) or len(by_identity) != len(expected):
        raise ProcessV2Active8SentinelError(
            "the sentinel result inventory is incomplete or contains an extra partition"
        )
    ordered = [by_identity[identity] for identity in expected]
    evidence_mismatches = sorted(
        item for result in ordered for item in result["evidence_mismatches"]
    )
    cell_mismatches = sorted(item for result in ordered for item in result["cell_mismatches"])
    oracle_mismatches = sorted(item for result in ordered for item in result["oracle_mismatches"])
    body = {
        "schema": SENTINEL_SCHEMA,
        "schema_version": SENTINEL_SCHEMA_VERSION,
        "status": (
            SENTINEL_PASSED
            if not (evidence_mismatches or cell_mismatches or oracle_mismatches)
            else SENTINEL_FAILED
        ),
        "salt": SENTINEL_SALT,
        "binding_sha256": str(validated["binding_sha256"]),
        "plan_sha256": str(validated["plan_sha256"]),
        "run_identity_sha256": str(validated["run_identity_sha256"]),
        "task_inventory_sha256": str(validated["task_inventory_sha256"]),
        "result_inventory_sha256": str(validated["result_inventory_sha256"]),
        "unique_accepted_pairs": int(validated["unique_accepted_pairs"]),
        "selection_mode": str(validated["selection_mode"]),
        "selected_pairs": int(validated["selected_pairs"]),
        "selected_pairs_sha256": str(validated["selected_pairs_sha256"]),
        "per_cell_examples": int(validated["per_cell_examples"]),
        "global_examples": int(validated["global_examples"]),
        "oracle_examples": sum(int(result["oracle_examples"]) for result in ordered),
        "evidence_mismatches": evidence_mismatches,
        "cell_mismatches": cell_mismatches,
        "oracle_mismatches": oracle_mismatches,
    }
    sentinel = self_hashed(body, field="sentinel_sha256")
    if tuple(sentinel) != SENTINEL_RESULT_FIELDS:
        raise ProcessV2Active8SentinelError("the sentinel result field set disagrees")
    return sentinel


def run_release_sentinel(
    plan: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    *,
    result_inventory_sha256: str,
    checker: ProductionProcessV2SemanticExactCandidateChecker,
    namespace: str,
    artifact_root: Path,
    repo_root: Path,
    max_pairs_per_partition: int = 256,
) -> dict[str, Any]:
    """Local reference wrapper over the same prepare/map/reduce interfaces."""

    sentinel_plan = prepare_release_sentinel_plan(
        plan,
        transitions,
        result_inventory_sha256=result_inventory_sha256,
        max_pairs_per_partition=max_pairs_per_partition,
    )
    results = [
        run_release_sentinel_partition(
            plan,
            sentinel_plan,
            identity,
            checker=checker,
            namespace=namespace,
            artifact_root=Path(artifact_root),
            repo_root=Path(repo_root),
        )
        for identity in sentinel_partition_identities(sentinel_plan)
    ]
    return reduce_release_sentinel_partitions(plan, sentinel_plan, results)


def _oracle_disagrees(
    checker: ProductionProcessV2SemanticExactCandidateChecker, state: Any
) -> bool:
    """Compare the production canonical quotient against the dictionary oracle."""

    result = canonical_successor_result(checker.model, state, checker.time, system=checker.system)
    produced = {
        str(successor.key): int(successor.alias_count) for successor in result.batch.successors
    }
    return produced != _dictionary_successor_oracle(result, state, system=checker.system)


def require_sentinel_passed(
    sentinel: Mapping[str, Any],
    *,
    binding_sha256: str | None = None,
    plan_sha256: str | None = None,
    run_identity_sha256: str | None = None,
    task_inventory_sha256: str | None = None,
    result_inventory_sha256: str | None = None,
) -> None:
    """Any mismatch blocks completion; there is no partial release."""

    if not SENTINEL_BLOCKS_COMPLETION:  # pragma: no cover - frozen True in the seam
        raise ProcessV2Active8SentinelError("the sentinel must block completion")
    if set(sentinel) != set(SENTINEL_RESULT_FIELDS):
        raise ProcessV2Active8SentinelError("the sentinel result field set disagrees")
    if (
        canonical_sha256(
            {key: value for key, value in sentinel.items() if key != "sentinel_sha256"}
        )
        != sentinel["sentinel_sha256"]
    ):
        raise ProcessV2Active8SentinelError("the sentinel self-hash disagrees")
    if (
        sentinel["schema"] != SENTINEL_SCHEMA
        or sentinel["schema_version"] != SENTINEL_SCHEMA_VERSION
        or sentinel["salt"] != SENTINEL_SALT
    ):
        raise ProcessV2Active8SentinelError("the sentinel protocol identity disagrees")
    expected = {
        "binding_sha256": binding_sha256,
        "plan_sha256": plan_sha256,
        "run_identity_sha256": run_identity_sha256,
        "task_inventory_sha256": task_inventory_sha256,
        "result_inventory_sha256": result_inventory_sha256,
    }
    for field, value in expected.items():
        observed = sentinel[field]
        if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
            raise ProcessV2Active8SentinelError(f"the sentinel {field} is not a SHA-256")
        if value is not None and observed != value:
            raise ProcessV2Active8SentinelError(
                f"the sentinel {field} addresses another Active8 run"
            )
    counts = {
        field: sentinel[field]
        for field in (
            "unique_accepted_pairs",
            "selected_pairs",
            "per_cell_examples",
            "global_examples",
            "oracle_examples",
        )
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ProcessV2Active8SentinelError("the sentinel counts must be nonnegative integers")
    if _SHA256.fullmatch(str(sentinel["selected_pairs_sha256"])) is None:
        raise ProcessV2Active8SentinelError("the sentinel selected-pair digest is not a SHA-256")
    _require_selection_counts(sentinel)
    if counts["oracle_examples"] != min(counts["selected_pairs"], SENTINEL_ORACLE_EXAMPLES):
        raise ProcessV2Active8SentinelError("the sentinel oracle count does not reconcile")
    mismatch_fields = (
        "evidence_mismatches",
        "cell_mismatches",
        "oracle_mismatches",
    )
    if any(
        not isinstance(sentinel[field], list)
        or any(not isinstance(item, str) for item in sentinel[field])
        or sentinel[field] != sorted(sentinel[field])
        for field in mismatch_fields
    ):
        raise ProcessV2Active8SentinelError("the sentinel mismatch receipts are malformed")
    has_mismatch = any(bool(sentinel[field]) for field in mismatch_fields)
    if sentinel["status"] == SENTINEL_PASSED and has_mismatch:
        raise ProcessV2Active8SentinelError("the sentinel claims PASS while publishing mismatches")
    if sentinel["status"] != SENTINEL_PASSED:
        raise ProcessV2Active8SentinelError(
            "the release sentinel found a mismatch, so the Active8 run publishes no "
            f"completion: evidence={sentinel['evidence_mismatches']} "
            f"cells={sentinel['cell_mismatches']} oracle={sentinel['oracle_mismatches']}"
        )


__all__ = [
    "SELECTION_EXHAUSTIVE",
    "SELECTION_RANKED",
    "SENTINEL_FAILED",
    "SENTINEL_PASSED",
    "ProcessV2Active8SentinelError",
    "completed_release_sentinel_partition_ids",
    "create_sentinel_pair_index",
    "index_sentinel_transition",
    "load_release_sentinel_partition_result",
    "prepare_release_sentinel_plan",
    "prepare_release_sentinel_plan_from_selection",
    "reduce_release_sentinel_partitions",
    "require_sentinel_passed",
    "run_release_sentinel",
    "run_release_sentinel_partition",
    "select_sentinel_pairs",
    "select_sentinel_pairs_from_index",
    "sentinel_partition_identities",
    "sentinel_partition_result_path",
    "sentinel_rank",
    "validate_release_sentinel_partition_result",
    "validate_release_sentinel_plan",
    "write_release_sentinel_partition_result",
]
