"""Gate 0 for the Process-V2 vertical pipeline: the structural readiness decision.

WHAT THIS STAGE IS
------------------
Active8 performs the sole expensive chunk pass.  It assigns
``capability_cell_id`` at WRITE time, while the exact source state, the action,
the teacher-family coordinate and the executor result are still in memory. It
writes the raw structural axes and teacher-admission evidence beside the cell,
each row authenticated by its own ``assignment_sha256``.

Gate 0 is therefore a SMALL DETERMINISTIC POST-ACTIVE8 REDUCER over those rows.
It is not a second distributed subsystem: there is no plan, no map task, no
per-task partial and no fan-out here.  It imports no capability-cell classifier,
reconstructs no model, decodes no molecular state, re-enumerates no successor
fiber and performs no per-trace point lookup.

Most sharply: **Gate 0 opens zero molecular cache chunks.**  The chunk cache
(``chunk-NNNNNN.jsonl.gz``) belongs to Active8's single pass.  Gate 0 reads only
Active8's published decision shards, which already carry everything it
aggregates.  That claim is measured, not asserted, by an instrument the tests
first prove can see a chunk open when one genuinely happens.

THE DATA STRUCTURE
------------------
    <active8_run_root>/tasks/<task_identity_sha256>/RECEIPT.json
        Active8 task result metadata: schema, partition role, data lane and the
        census.  Read for EVERY role, so the census and the sealed-role hashes
        stay complete.

    <active8_run_root>/tasks/<task_identity_sha256>/transitions.jsonl.gz
        The decision shard: one JSON object per accepted transition, each
        holding exactly ``ACCEPTED_TRANSITION_FIELDS``.  Read only for a
        decision-eligible, nonempty shard.  A sealed role's shard is never
        opened -- "never opened", not "never counted".

    <gate_zero_root>/DECISION.json
        The one artifact this stage publishes.

Every name above -- the directory, both filenames, and the receipt's own field
set -- is IMPORTED from the pipeline seam, never redeclared here.  The seam
names the fields AND the layout precisely because two modules independently
declaring the same filename is the same divergence as two modules independently
spelling the same field.  ``<active8_run_root>`` is the Active8 map stage's
``<output_artifact_prefix>/<run_identity_sha256>``.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* Role filtering is a SHARD-level predicate on metadata, evaluated before any
  transitions path is constructed, so held-out sealing is free rather than a
  per-row filter.
* ``terminal`` is CONSUMED from the seam, never recomputed.  It describes the
  teacher's SOURCE progress position and is False for every accepted action,
  including a final action whose successor is terminal.
* ``supported`` is READ and enforced, because the structural contract declares
  ``require_every_teacher_supported``.
* Whole-fiber geometry is not recomputed or inferred here. It belongs to the
  bounded release sentinel and cached T1 panels.
* The fold order is stated (``reduction_order``: ascending
  ``task_identity_sha256``) and the combiner is key-sorted counter addition,
  which is commutative and associative.  Nothing is published on an incomplete
  reduction.
* No stage here authorizes anything.  Every authority field is False in a PASS
  and in a FAIL alike, and a FAIL is a valid completed result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from compose_v4.data.editing_corpus_contract import REQUIRED_PARTITION_ROLES
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    ProcessV2Active8ReduceError,
    validate_process_v2_active8_completion,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACCEPTED_TRANSITION_FIELDS,
    ACTION_KEY_FIELDS,
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_DECISION_SHARD_FILENAME,
    ACTIVE8_RECEIPT_FIELDS,
    ACTIVE8_RECEIPT_FILENAME,
    ACTIVE8_TASK_SCHEMA,
    ACTIVE8_TASK_SCHEMA_VERSION,
    ACTIVE8_TASKS_DIRNAME,
    CANDIDATE_EVIDENCE_FIELDS,
    GATE_ZERO_DECISION_FILENAME,
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    canonical_bytes,
    canonical_sha256,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    load_process_v2_chain_artifact,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_process_v2_identity

# ---- Errors ------------------------------------------------------------------


class ProcessV2GateZeroError(RuntimeError):
    """The Gate-0 inputs are malformed, so no honest census exists to publish."""


class ProcessV2GateZeroIncomplete(ProcessV2GateZeroError):
    """A decision-eligible shard is absent; the reduction publishes nothing."""


# ---- Layout ------------------------------------------------------------------

#: Emitted, all False, in a PASS and in a FAIL alike.
_AUTHORITY: Mapping[str, bool] = {
    "bounded_p50_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
    "gate_zero_authorized": False,
    "long_training_authorized": False,
    "t1_authorized": False,
    "training_authorized": False,
}

#: Typed negative-receipt categories.  A well-formed row that violates the
#: structural contract is RECORDED and fails the gate; it does not raise.  A
#: malformed artifact raises, because there is then no honest census to publish.
_VIOLATION_CATEGORIES: tuple[str, ...] = (
    "unsupported_teacher",
    "terminal_assignment",
    "illegal_teacher_coordinate",
    "inexact_teacher_successor",
    "nonproductive_teacher_successor",
    "self_transition_teacher",
    "evidence_inconsistency",
    "unregistered_capability_cell",
    "unknown_model_family",
    "unknown_family_context",
    "cell_disagrees_with_axes",
    "missing_audit_axes",
    "duplicate_action_assignment",
)


# ---- Deterministic serialization ---------------------------------------------


def _canonical_line(value: object) -> bytes:
    """One newline-framed record.  Framing only; the hash is over the body."""

    return canonical_bytes(value) + b"\n"


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2GateZeroError(f"{field} must be a full lowercase SHA-256")
    return value


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProcessV2GateZeroError(f"{field} must be a nonempty string")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProcessV2GateZeroError(f"{field} must be a nonnegative integer")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ProcessV2GateZeroError(f"{field} must be boolean")
    return value


def _publish(path: Path, content: bytes) -> bool:
    """Publish immutably; a differing prior body is a collision, not an update."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise ProcessV2GateZeroError(f"immutable Gate-0 collision at {target}")
        return True
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return False


# ---- The bound contracts ------------------------------------------------------


@dataclass(frozen=True)
class GateZeroContracts:
    """The three frozen contracts Gate 0 reads, validated through the chain."""

    contract_sha256: str
    process_identity_sha256: str
    active_families: tuple[str, ...]
    family_contexts: Mapping[str, tuple[str, ...]]
    data_lanes: tuple[str, ...]
    decision_eligible_roles: tuple[str, ...]
    sealed_roles: tuple[str, ...]
    required_cell_ids: tuple[str, ...]
    conditional_cell_ids: tuple[str, ...]
    separate_lane_cell_ids: tuple[str, ...]
    registered_cell_ids: frozenset[str]
    receipt_limit: int
    structural_checks: Mapping[str, Any]

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(
            {
                "contract_sha256": self.contract_sha256,
                "process_identity_sha256": self.process_identity_sha256,
                "required_cell_ids": list(self.required_cell_ids),
                "conditional_cell_ids": list(self.conditional_cell_ids),
                "separate_lane_cell_ids": list(self.separate_lane_cell_ids),
                "decision_eligible_roles": list(self.decision_eligible_roles),
                "sealed_roles": list(self.sealed_roles),
                "active_families": list(self.active_families),
            }
        )


#: Structural clauses Gate 0 both REQUIRES the contract to declare and enforces.
#: The previous attempt refused a contract omitting ``require_every_teacher_
#: supported`` and then never read it; every name here has an enforcement site
#: in :func:`reduce_gate_zero`.
_REQUIRED_STRUCTURAL_CLAUSES: tuple[str, ...] = (
    "require_every_active8_family",
    "require_every_teacher_supported",
    "require_one_assignment_per_accepted_action",
    "require_productive_nonself_successor",
    "require_teacher_coordinate_legal",
    "require_teacher_executes_to_exact_successor",
    "require_zero_terminal_assignments",
)


def build_gate_zero_contracts(
    structural: Mapping[str, Any],
    cells: Mapping[str, Any],
    roles: Mapping[str, Any],
    *,
    live_process_identity_sha256: str,
) -> GateZeroContracts:
    """Cross-validate the three contract payloads.  Pure: it reads no file.

    Separated from :func:`load_gate_zero_contracts` so every refusal below has a
    reachable witness.  Through the chain loader these payloads are already
    hash-validated, which makes a forged one unreachable in production and the
    guards untestable if they were inlined there.
    """

    checks = structural.get("structural_checks")
    if not isinstance(checks, Mapping):
        raise ProcessV2GateZeroError("gate-zero contract declares no structural_checks")
    for clause in _REQUIRED_STRUCTURAL_CLAUSES:
        if checks.get(clause) is not True:
            raise ProcessV2GateZeroError(
                f"gate-zero contract must declare {clause} true; Gate 0 enforces it"
            )

    eligible = tuple(checks["decision_eligible_partition_roles"])
    sealed = tuple(checks["sealed_nondecision_partition_roles"])
    if set(eligible) & set(sealed):
        raise ProcessV2GateZeroError("a partition role is both decision-eligible and sealed")
    if set(eligible) | set(sealed) != set(REQUIRED_PARTITION_ROLES):
        raise ProcessV2GateZeroError(
            "decision-eligible and sealed roles must partition REQUIRED_PARTITION_ROLES"
        )

    families = tuple(cells["active_families"])
    if tuple(structural["active_families"]) != families or tuple(roles["active_families"]) != (
        families
    ):
        raise ProcessV2GateZeroError("the three contracts disagree on the active families")
    contexts = {
        str(family): tuple(str(item) for item in values)
        for family, values in cells["family_contexts"].items()
    }
    if set(contexts) != set(families):
        raise ProcessV2GateZeroError("family_contexts does not cover the active families")

    required = tuple(str(item) for item in roles["required_cell_ids"])
    conditional = tuple(str(item) for item in roles["conditional_cell_ids"])
    separate = tuple(str(item) for item in roles["separate_lane_cell_ids"])
    registered = [*required, *conditional, *separate]
    if len(set(registered)) != len(registered):
        raise ProcessV2GateZeroError("a capability cell carries two development roles")
    policy = roles["partition_policy"]
    if policy.get("registered_cells_must_be_classified_exactly_once") is not True:
        raise ProcessV2GateZeroError(
            "cell-role policy does not require exactly-once classification"
        )
    declared = (
        int(policy["required_cell_count"]),
        int(policy["conditional_cell_count"]),
        int(policy["separate_lane_cell_count"]),
    )
    if declared != (len(required), len(conditional), len(separate)):
        raise ProcessV2GateZeroError("cell-role counts disagree with the enumerated ids")

    namespace = str(cells["cell_identity_policy"]["namespace"])
    expected_cells = {
        f"{namespace}:{family}:{context}" for family in families for context in contexts[family]
    }
    if set(registered) != expected_cells:
        raise ProcessV2GateZeroError(
            "the registered cells are not exactly the family x context registry"
        )

    lanes = tuple(str(item) for item in cells["bindings"]["data_lanes"])
    pinned = _require_sha(
        structural["process_identity"]["process_identity_sha256"],
        field="process_identity.process_identity_sha256",
    )
    live = _require_sha(live_process_identity_sha256, field="live_process_identity_sha256")
    if pinned != live:
        raise ProcessV2GateZeroError(
            f"gate-zero contract pins process identity {pinned}, live is {live}"
        )
    return GateZeroContracts(
        contract_sha256=_require_sha(structural["contract_sha256"], field="contract_sha256"),
        process_identity_sha256=live,
        active_families=families,
        family_contexts=contexts,
        data_lanes=lanes,
        decision_eligible_roles=eligible,
        sealed_roles=sealed,
        required_cell_ids=required,
        conditional_cell_ids=conditional,
        separate_lane_cell_ids=separate,
        registered_cell_ids=frozenset(registered),
        receipt_limit=int(checks["classification_failure_receipt_limit"]),
        structural_checks=dict(checks),
    )


def load_gate_zero_contracts(*, repo_root: Path) -> GateZeroContracts:
    """Read the three chain artifacts, then cross-validate them."""

    repo_root = Path(repo_root)
    return build_gate_zero_contracts(
        load_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=repo_root),
        load_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=repo_root),
        load_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=repo_root),
        live_process_identity_sha256=str(
            editing_process_v2_identity()["process_identity_sha256"]
        ),
    )


# ---- Resolved decision metadata, for every role ------------------------------


@dataclass(frozen=True)
class GateZeroSourceIndex:
    """Every Active8 shard's resolved metadata; no shard content is opened.

    ``shards`` covers EVERY role, so the census and the sealed-role hashes stay
    complete.  ``eligible_task_identities`` is the subset whose rows are read:
    a decision-eligible role AND a nonempty accepted census.
    """

    active8_run_root: str
    active8_completion_sha256: str
    active8_sentinel_sha256: str
    contracts_binding_sha256: str
    shards: tuple[Mapping[str, Any], ...]
    eligible_task_identities: tuple[str, ...]
    role_census: Mapping[str, Mapping[str, int]]
    sealed_role_metadata: Mapping[str, Mapping[str, Any]]
    index_sha256: str

    def eligible(self) -> tuple[Mapping[str, Any], ...]:
        chosen = set(self.eligible_task_identities)
        return tuple(shard for shard in self.shards if shard["task_identity_sha256"] in chosen)


def _read_active8_task_metadata(
    path: Path, *, task_identity_sha256: str, contracts: GateZeroContracts
) -> tuple[dict[str, Any], str]:
    """Read ONE Active8 task result document.  Opens no decision shard."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProcessV2GateZeroError(f"active8 task result is unreadable: {path}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2GateZeroError(f"active8 task result is not JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ProcessV2GateZeroError(f"active8 task result is not an object: {path}")
    if set(payload) != set(ACTIVE8_RECEIPT_FIELDS):
        missing = sorted(set(ACTIVE8_RECEIPT_FIELDS) - set(payload))
        extra = sorted(set(payload) - set(ACTIVE8_RECEIPT_FIELDS))
        raise ProcessV2GateZeroError(
            f"{path}: receipt must carry exactly ACTIVE8_RECEIPT_FIELDS "
            f"(missing {missing}, unexpected {extra})"
        )
    try:
        verify_self_hash(payload, field="receipt_sha256", label=f"{path}: the Active8 receipt")
    except ProcessV2SchemaError as error:
        raise ProcessV2GateZeroError(str(error)) from error
    if payload.get("schema") != ACTIVE8_TASK_SCHEMA:
        raise ProcessV2GateZeroError(
            f"{path}: schema is {payload.get('schema')!r}, expected {ACTIVE8_TASK_SCHEMA!r}"
        )
    if payload.get("schema_version") != ACTIVE8_TASK_SCHEMA_VERSION:
        raise ProcessV2GateZeroError(f"{path}: unsupported active8 task schema version")
    identity = _require_sha(payload.get("task_identity_sha256"), field="task_identity_sha256")
    if identity != task_identity_sha256:
        raise ProcessV2GateZeroError(f"{path}: task identity disagrees with its directory")
    role = _require_str(payload.get("partition_role"), field="partition_role")
    if role not in REQUIRED_PARTITION_ROLES:
        raise ProcessV2GateZeroError(f"{path}: partition_role {role!r} is not a corpus role")
    lane = _require_str(payload.get("data_lane"), field="data_lane")
    if lane not in contracts.data_lanes:
        raise ProcessV2GateZeroError(f"{path}: data_lane {lane!r} is not a bound lane")
    # The census fields are TOP-LEVEL receipt fields, spliced into
    # ACTIVE8_RECEIPT_FIELDS; they are not nested under a `census` key.
    census = {
        field: _require_nonnegative_int(payload[field], field=field)
        for field in ACTIVE8_CENSUS_FIELDS
    }
    if census["source_entries"] != (
        census["upstream_rejected_entries"]
        + census["active8_accepted_entries"]
        + census["active8_excluded_entries"]
    ):
        raise ProcessV2GateZeroError(f"{path}: census does not close on source_entries")
    metadata = {
        "task_identity_sha256": identity,
        "partition_role": role,
        "data_lane": lane,
        "census": census,
        "source_chunk_identity_sha256": _require_sha(
            payload["source_chunk_identity_sha256"], field="source_chunk_identity_sha256"
        ),
        # Carried so the reduction can authenticate the shard it opens and
        # cross-check its length, both WITHOUT opening anything here.
        "transition_count": _require_nonnegative_int(
            payload["transition_count"], field="transition_count"
        ),
        "decision_shard_sha256": _require_sha(
            payload["decision_shard_sha256"], field="decision_shard_sha256"
        ),
        "receipt_sha256": _require_sha(
            payload["receipt_sha256"], field="receipt_sha256"
        ),
    }
    return metadata, hashlib.sha256(raw).hexdigest()


def read_active8_decision_index(
    active8_run_root: Path, *, contracts: GateZeroContracts
) -> GateZeroSourceIndex:
    """Resolve every role's decision metadata without opening a single shard.

    Reading every role is what keeps the census and the sealed-role hashes
    complete.  Eligibility is a SHARD-level predicate on that metadata, so a
    sealed role is excluded before any shard path is constructed.
    """

    root = Path(active8_run_root)
    completion_path = root / COMPLETION_FILENAME
    try:
        completion_payload = json.loads(completion_path.read_bytes())
        completion = validate_process_v2_active8_completion(completion_payload)
    except (OSError, json.JSONDecodeError, ProcessV2Active8ReduceError) as error:
        raise ProcessV2GateZeroIncomplete(
            f"active8 run has no valid sentinel-gated completion: {completion_path}"
        ) from error
    completion_inventory = {
        str(task_identity): str(receipt_sha256)
        for task_identity, receipt_sha256 in completion["result_inventory"]
    }
    parent = root / ACTIVE8_TASKS_DIRNAME
    if not parent.is_dir():
        raise ProcessV2GateZeroError(f"active8 run root has no task directory: {parent}")
    identities = sorted(entry.name for entry in parent.iterdir() if entry.is_dir())
    if not identities:
        raise ProcessV2GateZeroError(f"active8 run root publishes no tasks: {parent}")
    if set(identities) != set(completion_inventory):
        raise ProcessV2GateZeroError(
            "active8 task directories differ from the committed completion inventory"
        )

    shards: list[dict[str, Any]] = []
    eligible: list[str] = []
    role_totals: dict[str, Counter[str]] = {role: Counter() for role in REQUIRED_PARTITION_ROLES}
    sealed_witness: dict[str, list[list[str]]] = {role: [] for role in contracts.sealed_roles}
    for identity in identities:
        _require_sha(identity, field="active8 task directory name")
        metadata, metadata_sha = _read_active8_task_metadata(
            parent / identity / ACTIVE8_RECEIPT_FILENAME,
            task_identity_sha256=identity,
            contracts=contracts,
        )
        role = metadata["partition_role"]
        if metadata["receipt_sha256"] != completion_inventory[identity]:
            raise ProcessV2GateZeroError(
                "an Active8 receipt differs from the committed completion inventory"
            )
        decision_eligible_role = role in contracts.decision_eligible_roles
        nonempty = metadata["census"]["active8_accepted_entries"] > 0
        shards.append(
            {
                **metadata,
                "metadata_file_sha256": metadata_sha,
                "decision_eligible_role": decision_eligible_role,
                "nonempty": nonempty,
                "rows_read": decision_eligible_role and nonempty,
            }
        )
        if decision_eligible_role and nonempty:
            eligible.append(identity)
        counter = role_totals[role]
        counter["shards"] += 1
        counter.update(metadata["census"])
        if role in sealed_witness:
            sealed_witness[role].append([identity, metadata_sha])

    role_census = {
        role: {key: int(value) for key, value in sorted(counter.items())}
        for role, counter in sorted(role_totals.items())
    }
    receipt_census = {
        field: sum(role_census[role].get(field, 0) for role in role_census)
        for field in ACTIVE8_CENSUS_FIELDS
    }
    if receipt_census != dict(completion["census"]):
        raise ProcessV2GateZeroError(
            "the Active8 receipt census differs from the committed completion"
        )
    if sum(int(shard["transition_count"]) for shard in shards) != int(
        completion["accepted_transitions"]
    ):
        raise ProcessV2GateZeroError(
            "the Active8 receipt transition count differs from the committed completion"
        )
    # Resolution facts only.  How many of a sealed role's shards were OPENED is
    # not knowable here -- nothing has been opened yet -- so it is measured at
    # the reduction's one open site instead of asserted as a literal.
    sealed_role_metadata = {
        role: {
            "shards": len(witness),
            "metadata_resolved": len(witness),
            "metadata_digest_sha256": canonical_sha256(sorted(witness)),
        }
        for role, witness in sorted(sealed_witness.items())
    }
    body = {
        "active8_run_root": str(root),
        "active8_completion_sha256": str(completion["completion_sha256"]),
        "active8_sentinel_sha256": str(completion["sentinel"]["sentinel_sha256"]),
        "contracts_binding_sha256": contracts.binding_sha256,
        "shards": shards,
        "eligible_task_identities": sorted(eligible),
        "role_census": role_census,
        "sealed_role_metadata": sealed_role_metadata,
    }
    return GateZeroSourceIndex(
        active8_run_root=str(root),
        active8_completion_sha256=str(completion["completion_sha256"]),
        active8_sentinel_sha256=str(completion["sentinel"]["sentinel_sha256"]),
        contracts_binding_sha256=contracts.binding_sha256,
        shards=tuple(shards),
        eligible_task_identities=tuple(sorted(eligible)),
        role_census=role_census,
        sealed_role_metadata=sealed_role_metadata,
        index_sha256=canonical_sha256(body),
    )


# ---- Reading one decision shard ----------------------------------------------


def _read_decision_shard(path: Path) -> tuple[tuple[dict[str, Any], ...], str]:
    """Read one decision shard: rows plus the content hash of its bytes."""

    try:
        compressed = path.read_bytes()
    except OSError as error:
        raise ProcessV2GateZeroIncomplete(f"decision shard is absent: {path}") from error
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise ProcessV2GateZeroError(f"decision shard is not gzip: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProcessV2GateZeroError(f"{path}:{number} is not JSON") from error
        if not isinstance(row, Mapping):
            raise ProcessV2GateZeroError(f"{path}:{number} is not an object")
        rows.append(dict(row))
    return tuple(rows), hashlib.sha256(compressed).hexdigest()


def _validated_row(row: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Check the seam shape and the row self-hash.  A shape break RAISES."""

    if set(row) != set(ACCEPTED_TRANSITION_FIELDS):
        missing = sorted(set(ACCEPTED_TRANSITION_FIELDS) - set(row))
        extra = sorted(set(row) - set(ACCEPTED_TRANSITION_FIELDS))
        raise ProcessV2GateZeroError(
            f"{where}: accepted transition must carry exactly ACCEPTED_TRANSITION_FIELDS "
            f"(missing {missing}, unexpected {extra})"
        )
    evidence = row["candidate_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != set(CANDIDATE_EVIDENCE_FIELDS):
        raise ProcessV2GateZeroError(
            f"{where}: candidate_evidence must carry exactly {sorted(CANDIDATE_EVIDENCE_FIELDS)}"
        )
    try:
        verify_self_hash(row, field="assignment_sha256", label=f"{where}: accepted transition")
    except ProcessV2SchemaError as error:
        raise ProcessV2GateZeroError(str(error)) from error
    return dict(row)


# ---- The reduction -----------------------------------------------------------


def reduction_order(shards: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """The ONE stated fold order: ascending ``task_identity_sha256``.

    Order-independence is explicit rather than inherited from however the
    filesystem enumerated the shards: the caller may present them in any order,
    this function fixes the sequence, and the fold itself is key-sorted counter
    addition, which is commutative and associative.
    """

    identities = [str(shard["task_identity_sha256"]) for shard in shards]
    if len(set(identities)) != len(identities):
        raise ProcessV2GateZeroError("two decision shards claim the same task identity")
    return tuple(sorted(shards, key=lambda shard: str(shard["task_identity_sha256"])))


@dataclass
class _Aggregate:
    """The one mutable fold state.  Every counter combines by key-sorted sum."""

    families: Counter[str]
    executor_rules: Counter[str]
    cells: Counter[str]
    violations: Counter[str]
    receipts: list[dict[str, Any]]
    action_keys: set[tuple[Any, ...]]
    traces: set[tuple[Any, ...]]
    transitions: int
    shard_digests: list[list[str]]


def _fold_shard(
    aggregate: _Aggregate,
    rows: Sequence[Mapping[str, Any]],
    *,
    shard: Mapping[str, Any],
    where: str,
    contracts: GateZeroContracts,
) -> None:
    """Fold one shard's rows into the aggregate.  No row is looked up by key."""

    def record(category: str, row: Mapping[str, Any], detail: str) -> None:
        aggregate.violations[category] += 1
        if len(aggregate.receipts) < contracts.receipt_limit:
            aggregate.receipts.append(
                {
                    "category": category,
                    "detail": detail,
                    "assignment_sha256": row["assignment_sha256"],
                    "capability_cell_id": row["capability_cell_id"],
                    "task_identity_sha256": shard["task_identity_sha256"],
                }
            )

    for number, raw_row in enumerate(rows, start=1):
        row = _validated_row(raw_row, where=f"{where}:{number}")
        if row["task_identity_sha256"] != shard["task_identity_sha256"]:
            raise ProcessV2GateZeroError(f"{where}:{number}: row belongs to another Active8 task")
        if row["partition_role"] != shard["partition_role"] or row["data_lane"] != (
            shard["data_lane"]
        ):
            raise ProcessV2GateZeroError(f"{where}:{number}: row role/lane differs from its shard")
        _require_sha(row["v1_task_identity_sha256"], field="v1_task_identity_sha256")
        _require_nonnegative_int(row["entry_index"], field="entry_index")

        key = tuple(row[field] for field in ACTION_KEY_FIELDS)
        if key in aggregate.action_keys:
            record("duplicate_action_assignment", row, "action key repeats in the corpus")
        aggregate.action_keys.add(key)
        aggregate.traces.add(key[:-1])
        aggregate.transitions += 1

        # `terminal` is CONSUMED from the seam, never recomputed from the
        # successor: it describes the teacher's SOURCE progress position.
        if row["terminal"] is not False:
            record("terminal_assignment", row, "source progress position is terminal")

        evidence = row["candidate_evidence"]
        supported = _require_bool(evidence["supported"], field="candidate_evidence.supported")
        exclusion_reason = evidence["exclusion_reason"]
        if exclusion_reason is not None and (
            not isinstance(exclusion_reason, str) or not exclusion_reason
        ):
            raise ProcessV2GateZeroError(
                "candidate_evidence.exclusion_reason must be null or nonempty text"
            )
        if not supported:
            record(
                "unsupported_teacher",
                row,
                f"supported={supported!r} exclusion={exclusion_reason!r}",
            )
        flags = {
            field: _require_bool(evidence[field], field=f"candidate_evidence.{field}")
            for field in (
                "teacher_coordinate_legal",
                "teacher_executes_to_exact_successor",
                "productive_canonical_successor",
            )
        }
        if not flags["teacher_coordinate_legal"]:
            record("illegal_teacher_coordinate", row, "teacher coordinate is absent")
        if not flags["teacher_executes_to_exact_successor"]:
            record(
                "inexact_teacher_successor",
                row,
                "teacher replay differs from stored successor",
            )
        if not flags["productive_canonical_successor"]:
            record(
                "nonproductive_teacher_successor",
                row,
                "teacher successor is not productive",
            )
        source_key = _require_str(
            evidence["source_canonical_key"], field="source_canonical_key"
        )
        successor_key = _require_str(
            evidence["canonical_successor_key"], field="canonical_successor_key"
        )
        if source_key == successor_key:
            record("self_transition_teacher", row, source_key)
        status_consistent = (
            supported and exclusion_reason is None and all(flags.values())
        ) or (
            not supported and isinstance(exclusion_reason, str) and bool(exclusion_reason)
        )
        evidence_consistent = (
            status_consistent
            and flags["productive_canonical_successor"] == (source_key != successor_key)
            and (
                not flags["teacher_executes_to_exact_successor"]
                or flags["teacher_coordinate_legal"]
            )
        )
        if not evidence_consistent:
            record("evidence_inconsistency", row, json.dumps(flags, sort_keys=True))

        family = _require_str(row["model_family"], field="model_family")
        context = _require_str(row["family_context"], field="family_context")
        cell = _require_str(row["capability_cell_id"], field="capability_cell_id")
        if family not in contracts.active_families:
            record("unknown_model_family", row, family)
        elif context not in contracts.family_contexts[family]:
            record("unknown_family_context", row, f"{family}:{context}")
        if cell not in contracts.registered_cell_ids:
            record("unregistered_capability_cell", row, cell)
        elif not cell.endswith(f":{family}:{context}"):
            record("cell_disagrees_with_axes", row, f"{cell} vs {family}:{context}")
        axes = row["audit_axes"]
        if not axes or not isinstance(axes, (Mapping, list, tuple)):
            record("missing_audit_axes", row, "the raw structural axes are absent")

        aggregate.families[family] += 1
        aggregate.executor_rules[_require_str(row["executor_rule"], field="executor_rule")] += 1
        aggregate.cells[cell] += 1


def reduce_gate_zero(
    active8_run_root: Path,
    *,
    gate_zero_root: Path,
    contracts: GateZeroContracts,
    index: GateZeroSourceIndex | None = None,
) -> dict[str, Any]:
    """Read every eligible decision shard once and publish the Gate-0 decision.

    Refuses -- publishing nothing -- unless every eligible shard is present.
    The decision is honest in both directions: a FAIL is a completed result, and
    no threshold is relaxed and no cell dropped to reach a PASS.  Every
    authority field is False either way.
    """

    resolved = read_active8_decision_index(active8_run_root, contracts=contracts)
    if index is not None and index != resolved:
        raise ProcessV2GateZeroError(
            "the supplied Gate-0 source index differs from the authenticated Active8 run"
        )
    index = resolved
    if index.contracts_binding_sha256 != contracts.binding_sha256:
        raise ProcessV2GateZeroError("the resolved index was built against other contracts")
    if str(Path(active8_run_root)) != index.active8_run_root:
        raise ProcessV2GateZeroError("the resolved index addresses another Active8 run")
    eligible = reduction_order(index.eligible())
    if not eligible:
        raise ProcessV2GateZeroIncomplete("no decision-eligible nonempty Active8 shard exists")
    for shard in eligible:
        if shard["partition_role"] not in contracts.decision_eligible_roles:
            raise ProcessV2GateZeroError("a sealed partition role reached the reduction")

    aggregate = _Aggregate(
        families=Counter(),
        executor_rules=Counter(),
        cells=Counter(),
        violations=Counter(),
        receipts=[],
        action_keys=set(),
        traces=set(),
        transitions=0,
        shard_digests=[],
    )
    parent = Path(active8_run_root) / ACTIVE8_TASKS_DIRNAME
    # This is the ONE place the stage opens content, so `opened_by_role` is
    # measured at the open site rather than declared afterwards.
    opened_by_role: Counter[str] = Counter()
    for shard in eligible:
        identity = str(shard["task_identity_sha256"])
        path = parent / identity / ACTIVE8_DECISION_SHARD_FILENAME
        opened_by_role[str(shard["partition_role"])] += 1
        rows, digest = _read_decision_shard(path)
        if digest != shard["decision_shard_sha256"]:
            raise ProcessV2GateZeroError(
                f"{path}: shard bytes do not match the decision_shard_sha256 its receipt declares"
            )
        if len(rows) != shard["transition_count"]:
            raise ProcessV2GateZeroError(
                f"{path}: holds {len(rows)} transitions, its receipt declares "
                f"{shard['transition_count']}"
            )
        if not rows:
            raise ProcessV2GateZeroError(f"{path}: a nonempty shard holds no transitions")
        traces_before = len(aggregate.traces)
        _fold_shard(aggregate, rows, shard=shard, where=str(path), contracts=contracts)
        accepted = shard["census"]["active8_accepted_entries"]
        if len(aggregate.traces) - traces_before != accepted:
            raise ProcessV2GateZeroError(
                f"{path}: shard covers {len(aggregate.traces) - traces_before} accepted traces, "
                f"its census says {accepted}"
            )
        aggregate.shard_digests.append([identity, digest])

    sealed_role_metadata = {
        role: {**block, "decision_shards_opened": int(opened_by_role[role])}
        for role, block in index.sealed_role_metadata.items()
    }

    violations = {
        category: int(aggregate.violations[category]) for category in _VIOLATION_CATEGORIES
    }
    families = dict(sorted(aggregate.families.items()))
    cells = dict(sorted(aggregate.cells.items()))
    missing_families = [
        family for family in contracts.active_families if families.get(family, 0) < 1
    ]
    missing_required_cells = [
        cell for cell in contracts.required_cell_ids if cells.get(cell, 0) < 1
    ]

    checks = {
        "every_active8_family_present": not missing_families,
        "every_required_cell_present": not missing_required_cells,
        "terminal_assignment_count_is_zero": violations["terminal_assignment"] == 0,
        "every_teacher_supported": violations["unsupported_teacher"] == 0,
        "every_teacher_coordinate_legal": violations["illegal_teacher_coordinate"] == 0,
        "every_teacher_replays_exact_successor": violations["inexact_teacher_successor"]
        == 0,
        "productive_nonself_successor": (
            violations["nonproductive_teacher_successor"] == 0
            and violations["self_transition_teacher"] == 0
        ),
        "teacher_evidence_consistent": violations["evidence_inconsistency"] == 0,
        "every_cell_registered": (
            violations["unregistered_capability_cell"] == 0
            and violations["unknown_model_family"] == 0
            and violations["unknown_family_context"] == 0
            and violations["cell_disagrees_with_axes"] == 0
        ),
        "audit_axes_present": violations["missing_audit_axes"] == 0,
        "one_assignment_per_accepted_action": violations["duplicate_action_assignment"] == 0,
        "sealed_roles_unopened": all(
            block["decision_shards_opened"] == 0 for block in sealed_role_metadata.values()
        ),
    }
    passed = all(checks.values())

    body: dict[str, Any] = {
        "schema": GATE_ZERO_DECISION_SCHEMA,
        "schema_version": GATE_ZERO_DECISION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **_AUTHORITY,
        "decision": "PASS" if passed else "FAIL",
        "active8_completion_sha256": index.active8_completion_sha256,
        "active8_sentinel_sha256": index.active8_sentinel_sha256,
        "contracts_binding_sha256": contracts.binding_sha256,
        "gate_zero_structural_contract_sha256": contracts.contract_sha256,
        "process_identity_sha256": contracts.process_identity_sha256,
        "source_index_sha256": index.index_sha256,
        # The clauses this run enforced, read back from the contract that
        # declared them, so the decision says what it was measured against.
        "enforced_structural_clauses": {
            clause: contracts.structural_checks[clause]
            for clause in _REQUIRED_STRUCTURAL_CLAUSES
        },
        "reduction": {
            "order": "ascending_task_identity_sha256",
            "combiner": "key_sorted_counter_addition",
            "shards": len(eligible),
            "digest_sha256": canonical_sha256(aggregate.shard_digests),
        },
        "accounting": {
            "resolved_shards": len(index.shards),
            "eligible_shards": len(eligible),
            "decision_shards_read": len(aggregate.shard_digests),
            # Measured at the one open site.  That Gate 0 opens no MOLECULAR
            # cache chunk is not restated here as a self-derived zero: this
            # module constructs no chunk path, and the property is measured
            # process-wide by an instrumented test with a positive control.
            "decision_shards_opened_by_role": dict(sorted(opened_by_role.items())),
            "transitions": aggregate.transitions,
            "distinct_action_keys": len(aggregate.action_keys),
            "accepted_traces": len(aggregate.traces),
        },
        "role_census": index.role_census,
        "sealed_role_metadata": sealed_role_metadata,
        "model_family_counts": families,
        "executor_rule_counts": dict(sorted(aggregate.executor_rules.items())),
        "capability_cell_counts": cells,
        "required_cell_counts": {
            cell: cells.get(cell, 0) for cell in sorted(contracts.required_cell_ids)
        },
        "conditional_cell_counts": {
            cell: cells.get(cell, 0) for cell in sorted(contracts.conditional_cell_ids)
        },
        "separate_lane_cell_counts": {
            cell: cells.get(cell, 0) for cell in sorted(contracts.separate_lane_cell_ids)
        },
        "violation_counts": dict(sorted(violations.items())),
        "total_violations": sum(violations.values()),
        "missing_active8_families": missing_families,
        "missing_required_cells": missing_required_cells,
        "checks": dict(sorted(checks.items())),
        "negative_receipts": aggregate.receipts,
    }
    decision = {**body, "decision_sha256": canonical_sha256(body)}
    _publish(
        Path(gate_zero_root) / GATE_ZERO_DECISION_FILENAME,
        _canonical_line(decision),
    )
    return decision


def run_gate_zero(
    active8_run_root: Path, *, gate_zero_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Bind the contracts, resolve every role's metadata, reduce, publish."""

    contracts = load_gate_zero_contracts(repo_root=repo_root)
    return reduce_gate_zero(
        active8_run_root, gate_zero_root=gate_zero_root, contracts=contracts
    )


def resolve_gate_zero_eligible_stream(
    active8_run_root: Path,
    *,
    repo_root: Path,
) -> tuple[GateZeroContracts, GateZeroSourceIndex]:
    """Authenticate the train-only metadata stream without opening a shard."""

    contracts = load_gate_zero_contracts(repo_root=repo_root)
    index = read_active8_decision_index(active8_run_root, contracts=contracts)
    if not reduction_order(index.eligible()):
        raise ProcessV2GateZeroIncomplete("no decision-eligible nonempty Active8 shard exists")
    return contracts, index


def iter_gate_zero_eligible_shards(
    active8_run_root: Path,
    *,
    contracts: GateZeroContracts,
    index: GateZeroSourceIndex,
) -> Iterator[tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]]:
    """Yield authenticated train-only shards while holding one shard in memory."""

    eligible = reduction_order(index.eligible())
    if not eligible:
        raise ProcessV2GateZeroIncomplete("no decision-eligible nonempty Active8 shard exists")
    parent = Path(active8_run_root) / ACTIVE8_TASKS_DIRNAME
    for shard in eligible:
        if shard["partition_role"] not in contracts.decision_eligible_roles:
            raise ProcessV2GateZeroError("a sealed partition role reached the T1 stream")
        identity = str(shard["task_identity_sha256"])
        path = parent / identity / ACTIVE8_DECISION_SHARD_FILENAME
        rows, digest = _read_decision_shard(path)
        if digest != shard["decision_shard_sha256"]:
            raise ProcessV2GateZeroError(
                f"{path}: shard bytes do not match the decision_shard_sha256 its receipt declares"
            )
        if len(rows) != shard["transition_count"]:
            raise ProcessV2GateZeroError(
                f"{path}: holds {len(rows)} transitions, its receipt declares "
                f"{shard['transition_count']}"
            )
        validated = tuple(
            _validated_row(row, where=f"{path}:{number}")
            for number, row in enumerate(rows, start=1)
        )
        yield shard, validated


def iter_process_v2_role_shards(
    active8_run_root: Path,
    *,
    contracts: GateZeroContracts,
    index: GateZeroSourceIndex,
    partition_role: str,
) -> Iterator[tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]]:
    """Yield authenticated Active8 shards for one explicitly requested role.

    Gate 0 intentionally calls only :func:`iter_gate_zero_eligible_shards`, so
    held-out decision shards remain unopened during structural gating.  A
    later, separately authorized evaluation stage may use this role-scoped
    reader to open validation evidence without ever constructing a path for
    controller-validation or final-test data.
    """

    if partition_role not in REQUIRED_PARTITION_ROLES:
        raise ProcessV2GateZeroError(
            f"unknown Process-V2 partition role: {partition_role!r}"
        )
    selected = reduction_order(
        tuple(
            shard
            for shard in index.shards
            if shard["partition_role"] == partition_role and shard["nonempty"]
        )
    )
    parent = Path(active8_run_root) / ACTIVE8_TASKS_DIRNAME
    for shard in selected:
        identity = str(shard["task_identity_sha256"])
        path = parent / identity / ACTIVE8_DECISION_SHARD_FILENAME
        rows, digest = _read_decision_shard(path)
        if digest != shard["decision_shard_sha256"]:
            raise ProcessV2GateZeroError(
                f"{path}: bytes do not match the receipt decision_shard_sha256"
            )
        if len(rows) != shard["transition_count"]:
            raise ProcessV2GateZeroError(
                f"{path}: holds {len(rows)} transitions, its receipt declares "
                f"{shard['transition_count']}"
            )
        validated = tuple(
            _validated_row(row, where=f"{path}:{number}")
            for number, row in enumerate(rows, start=1)
        )
        if any(row["partition_role"] != partition_role for row in validated):
            raise ProcessV2GateZeroError(
                f"{path}: contains a transition outside role {partition_role!r}"
            )
        yield shard, validated
