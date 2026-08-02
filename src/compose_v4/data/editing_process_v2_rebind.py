"""Proof-bound rebind of the completed V1 semantic payload under Process V2.

Process V2 expands the learned ``atom_delete`` fiber, so every V1 downstream
artifact is invalidated by decision.  The completed V1 migration payload is
still immutable *chemical* data, and this module is the only sanctioned way to
carry it forward: not by relabelling bytes, but by re-proving every chemical
claim it makes and emitting new, separately identified V2 evidence.

What a task proves, per V1 record
---------------------------------

1. the record, its receipt, its schema, its physical hashes and its exact
   pinned process identity are internally consistent;
2. every persisted semantic action round-trips through
   :mod:`compose_v4.rewrite.action_codec_v4` to a byte-equal record;
3. every teacher transition replays through the *unchanged* executor
   :func:`editing_v2_semantic_rewrite_system`;
4. every replayed successor equals the persisted exact persistent-slot state on
   all four arrays, and its canonical key equals the persisted key;
5. the *complete effective* Process-V2 atom-delete mask agrees, at every
   progress state, with the bounded independent legality oracle implemented
   below;
6. source address, lane, split, lineage and evidence-profile identities are
   carried through byte-for-byte; nothing is re-partitioned and no split is
   re-derived.

Two outcomes that must never be conflated
-----------------------------------------

An **integrity mismatch** says the payload is not what it claims to be: a row
is unreadable, a schema, self-hash, pinned identity or provenance field
disagrees, an action does not round-trip, the unchanged executor rejects a
persisted teacher, a replayed successor array or canonical key differs, or the
production mask disagrees with the independent oracle.  Nothing about the
payload can then be trusted, so the whole task aborts and **publishes
nothing**; every finding is returned, reason-coded and addressed, on
:class:`ProcessV2RebindMismatch`.

An **unsupported teacher** says the payload is exactly what it claims to be and
the *prospectively frozen* Process-V2 fiber does not contain one of its teacher
actions.  That is an expected support exclusion, not a defect, and the measured
correction round found it is reachable on real chemistry: applying the
authoritative charge policy uniformly excludes inherited leaf deletions that
the V1 dense rule admitted.  Such a trace is **rejected whole**, its exact
identity and reason code are recorded, and the scan continues.  No trace is
ever admitted in part: a rejected trace contributes zero rows.

Why a pinned identity is required
---------------------------------

``editing_v2_process_identity()`` hashes its implementation sources, so the
live V1 identity value necessarily moves when Process V2 lands.  A historical
payload therefore cannot be read through the live-identity path by
construction.  The caller must name the superseded identity explicitly; the
pinned object is validated for internal self-consistency and then every
chemical fact is re-proven from the bytes.  Naming an identity is not trusting
it.

What the published censuses mean
--------------------------------

``rejected_traces_by_code``, ``unsupported_teacher_steps_by_code``,
``rejected_traces`` and every count are **measured** over the exact rows the
task read, and the reducer sums them.  A published artifact carries no
integrity-mismatch histogram, because publishing on an integrity mismatch is
forbidden: such a field could only ever be empty, and an always-empty field
published beside real measurements reads as evidence when it is a constant.
The integrity histogram is a measurement of the refusal, so it lives on the
raised :class:`ProcessV2RebindMismatch` and in
:meth:`ProcessV2RebindMismatch.as_report`, which is never written into the
artifact namespace.

Neither a complete task nor a complete reduction grants Gate 0, T1, P50,
checkpoint-selection or training authority.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    BOND_NULL,
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
    is_scar,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedStoreError,
    load_semantic_packed_manifest,
    read_semantic_packed_artifact_range_rows,
    validate_semantic_packed_entry_ranges,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    DECISION_FILENAME,
    MATERIALIZATION_SCHEMA,
    MATERIALIZATION_SCHEMA_VERSION,
    MATERIALIZATION_STATUS,
    SEMANTIC_ARTIFACT_DIRNAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite import editing_v2_process_identity as process_identity_module
from compose_v4.rewrite.editing_v2_process_identity import PROCESS_SEMANTICS
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite import process_v2_atom_delete as process_v2_atom_delete_module
from compose_v4.rewrite.operators import (
    AtomDelete,
    apply_atom_delete,
    is_valid_atom_delete,
)
from compose_v4.rewrite.trace_shard_v3 import (
    MAX_ACTIVE_ATOMS,
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
)

# ---- Frozen artifact identity ------------------------------------------------

# Schema version 2 records one intentional, incompatible change: a task now
# separates integrity mismatches from support exclusions, so it publishes a
# measured rejection census and an exact rejected-trace inventory instead of
# two structurally empty mismatch fields.  Version 1 artifacts are not readable
# as version 2 and are not relabelled.
PLAN_SCHEMA = "compose.data.editing_process_v2_rebind_plan"
PLAN_SCHEMA_VERSION = 2
PLAN_STATUS = "FROZEN_COMPLETE_NO_TRAINING_AUTHORITY"
SOURCE_REVISION_SCHEMA = "compose.data.editing_process_v2_rebind_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 2
V1_PAYLOAD_BINDING_SCHEMA = "compose.data.editing_process_v2_rebind_v1_payload_binding"
V1_PAYLOAD_BINDING_SCHEMA_VERSION = 2
PROOF_SCHEMA = "compose.data.editing_process_v2_rebind_proof"
PROOF_SCHEMA_VERSION = 2
REJECTION_SCHEMA = "compose.data.editing_process_v2_rebind_rejection"
REJECTION_SCHEMA_VERSION = 2
MANIFEST_SCHEMA = "compose.data.editing_process_v2_rebind_manifest"
MANIFEST_SCHEMA_VERSION = 2
RECEIPT_SCHEMA = "compose.data.editing_process_v2_rebind_receipt"
RECEIPT_SCHEMA_VERSION = 2
TASK_STATUS = "PROVEN_V1_COMPATIBLE_NO_TRAINING_AUTHORITY"
COMPLETION_SCHEMA = "compose.data.editing_process_v2_rebind_completion"
COMPLETION_SCHEMA_VERSION = 2
COMPLETION_STATUS = "COMPLETE_V1_COMPATIBILITY_PROOF_NO_TRAINING_AUTHORITY"
REFUSAL_SCHEMA = "compose.data.editing_process_v2_rebind_refusal"
REFUSAL_SCHEMA_VERSION = 2
REFUSAL_STATUS = "INTEGRITY_MISMATCH_NOTHING_PUBLISHED"

PLAN_FILENAME = "PROCESS_V2_REBIND_PLAN.json"
COMPLETION_FILENAME = "PROCESS_V2_REBIND_COMPLETE.json"
MANIFEST_FILENAME = "MANIFEST.json"
PROOF_FILENAME = "proofs.jsonl.gz"
RECEIPT_FILENAME = "RECEIPT.json"
TASK_DIRNAME = "tasks"

DEFAULT_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_rebind"
DEFAULT_ENTRIES_PER_TASK = 64

# Structural candidate sources are diagnostic labels only.  One admission
# authority decides every ``atom_delete`` candidate; the two labels exist so a
# census can say *which* structural population an admission or an exclusion came
# from, never so that one population escapes a gate.
INDEPENDENT_CONNECTED_NONLEAF_MINIMUM_DEGREE = 2
INHERITED_CANDIDATE_SOURCE = "inherited_root_singleton_leaf"
CONNECTED_NONLEAF_CANDIDATE_SOURCE = "connected_nonleaf"
CANDIDATE_SOURCES = (INHERITED_CANDIDATE_SOURCE, CONNECTED_NONLEAF_CANDIDATE_SOURCE)

# The complete effective Process-V2 atom-delete mask, resolved by name at call
# time so an absent authority names itself instead of failing at import.
PROCESS_V2_ATOM_DELETE_MASK_SYMBOL = "process_v2_atom_delete_mask"

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_SOURCE_FILES = (
    "src/compose_v4/chem/aromaticity.py",
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/data/charge_policy.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_process_v2_rebind.py",
    "src/compose_v4/data/semantic_packed_trace_store.py",
    "src/compose_v4/data/semantic_trace_migration_materializer.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/process_v2_atom_delete.py",
    "src/compose_v4/rewrite/trace_shard_v3.py",
)
_SOURCE_REVISION_FIELDS = {
    "schema",
    "schema_version",
    "commit",
    "tree",
    "worktree_clean",
    "implementation_files",
    "implementation_files_sha256",
    "process_v2_identity_sha256",
    "source_revision_sha256",
}
_V1_TASK_BINDING_FIELDS = {
    "v1_task_identity_sha256",
    "v1_task_artifact_path",
    "data_lane",
    "split",
    "v1_receipt_file_sha256",
    "v1_receipt_sha256",
    "v1_decision_ledger_sha256",
    "v1_semantic_shard_sha256",
    "v1_semantic_manifest_sha256",
    "v1_source_binding",
    "v1_entries",
}
_V1_PAYLOAD_BINDING_FIELDS = {
    "schema",
    "schema_version",
    "payload_root_artifact_path",
    "payload_root_files",
    "pinned_process_identity_sha256",
    "pinned_builder_identity_sha256",
    "v1_tasks",
    "v1_tasks_sha256",
    "v1_entry_count",
    "binding_sha256",
}
_TASK_FIELDS = _V1_TASK_BINDING_FIELDS | {
    "entry_start",
    "entry_stop",
    "task_identity_sha256",
    "output_artifact_path",
}
_PLAN_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "implementation_revision",
    "source_revision",
    "pinned_process_identity",
    "pinned_builder_identity",
    "process_v2_identity",
    "v1_payload_binding",
    "v1_payload_binding_sha256",
    "run_identity_sha256",
    "run_artifact_root",
    "output_artifact_prefix",
    "entries_per_task",
    "expected_task_count",
    "expected_entry_count",
    "tasks",
    "task_inventory_sha256",
    "plan_sha256",
}
_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "v1_task_identity_sha256",
    "v1_task_artifact_path",
    "data_lane",
    "split",
    "entry_start",
    "entry_stop",
    "v1_entries",
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
    "family_histogram",
    "teacher_census",
    "process_v2_atom_delete_census",
    "rejected_traces_by_code",
    "unsupported_teacher_steps_by_code",
    "rejected_traces",
    "v1_record_inventory_sha256",
    "proof_stream_sha256",
    "pinned_process_identity_sha256",
    "pinned_builder_identity_sha256",
    "process_v2_identity_sha256",
    "manifest_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "implementation_revision",
    "run_identity_sha256",
    "task_identity_sha256",
    "v1_task_identity_sha256",
    "v1_task_artifact_path",
    "data_lane",
    "split",
    "entry_start",
    "entry_stop",
    "v1_entries",
    "v1_source_binding",
    "v1_semantic_shard_sha256",
    "v1_semantic_manifest_sha256",
    "pinned_process_identity",
    "pinned_builder_identity",
    "process_v2_identity",
    "proof_ledger_sha256",
    "proof_stream_sha256",
    "manifest_physical_sha256",
    "manifest_sha256",
    "counts",
    "rejected_traces_by_code",
    "unsupported_teacher_steps_by_code",
    "receipt_sha256",
}
_PROOF_FIELDS = {
    "schema",
    "schema_version",
    "entry_index",
    "trace_id",
    "data_lane",
    "split",
    "source_address",
    "lineage",
    "v1_record_sha256",
    "path_length",
    "canonical_state_keys",
    "family_histogram",
    "replayed_transitions",
    "atom_delete_teachers_by_candidate_source",
    "process_v2_atom_delete_candidates",
    "proof_sha256",
}
_REJECTION_FIELDS = {
    "schema",
    "schema_version",
    "entry_index",
    "trace_id",
    "data_lane",
    "split",
    "v1_record_sha256",
    "path_length",
    "step_index",
    "exclusion_code",
    "detail",
    "unsupported_teacher_steps",
    "rejection_sha256",
}
_COUNT_FIELDS = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
)
_TEACHER_CENSUS_FIELDS = (
    "atom_delete_teachers",
    *(f"atom_delete_teachers_{source}" for source in CANDIDATE_SOURCES),
    "teacher_transitions",
)
_MASK_CENSUS_FIELDS = (
    "states_evaluated",
    "states_with_candidates",
    "candidate_slots",
)


class ProcessV2RebindIntegrityCode(str, Enum):
    """Every way one V1 record can prove not to be what it claims to be.

    Any of these aborts the whole task and publishes nothing.  None of them is
    a statement about the *support* of the frozen Process-V2 fiber.
    """

    V1_RECORD_UNREADABLE = "v1_record_unreadable"
    V1_RECORD_SCHEMA_DISAGREES = "v1_record_schema_disagrees"
    V1_RECORD_SELF_HASH_DISAGREES = "v1_record_self_hash_disagrees"
    V1_PROCESS_IDENTITY_DISAGREES = "v1_process_identity_disagrees"
    V1_PROVENANCE_DISAGREES = "v1_provenance_disagrees"
    ACTION_CODEC_ROUNDTRIP_DISAGREES = "action_codec_roundtrip_disagrees"
    EXECUTOR_REPLAY_FAILED = "executor_replay_failed"
    SUCCESSOR_ARRAY_DISAGREES = "successor_array_disagrees"
    CANONICAL_KEY_DISAGREES = "canonical_key_disagrees"
    PROCESS_V2_MASK_DISAGREES = "process_v2_mask_disagrees"


class ProcessV2RebindExclusionCode(str, Enum):
    """Every way an intact V1 record can fall outside the frozen V2 support.

    Any of these rejects exactly one trace, records its identity, and lets the
    scan continue.  None of them says the payload is corrupt.
    """

    ATOM_DELETE_OUTSIDE_PROCESS_V2_MASK = "atom_delete_outside_process_v2_mask"
    TEACHER_RULE_OUTSIDE_FROZEN_SUPPORT = "teacher_rule_outside_frozen_support"
    TEACHER_FAMILY_OUTSIDE_FROZEN_SUPPORT = "teacher_family_outside_frozen_support"


@dataclass(frozen=True)
class ProcessV2RebindFinding:
    """One reason-coded integrity refusal, addressed to its record and step."""

    entry_index: int
    step_index: int | None
    code: ProcessV2RebindIntegrityCode
    detail: str
    trace_id: str | None = None
    v1_record_sha256: str | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "entry_index": self.entry_index,
            "step_index": self.step_index,
            "code": self.code.value,
            "detail": self.detail,
            "trace_id": self.trace_id,
            "v1_record_sha256": self.v1_record_sha256,
        }


@dataclass(frozen=True)
class ProcessV2RebindRejection:
    """One whole trace excluded by the prospectively frozen Process-V2 fiber.

    ``step_index`` addresses the first unsupported teacher in trace order and
    ``unsupported_teacher_steps`` counts every unsupported step in the trace,
    so a census can report both the trace and the step denominator.
    """

    entry_index: int
    trace_id: str
    data_lane: str
    split: str
    v1_record_sha256: str
    path_length: int
    step_index: int
    code: ProcessV2RebindExclusionCode
    detail: str
    unsupported_teacher_steps: int

    def as_payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": REJECTION_SCHEMA,
            "schema_version": REJECTION_SCHEMA_VERSION,
            "entry_index": self.entry_index,
            "trace_id": self.trace_id,
            "data_lane": self.data_lane,
            "split": self.split,
            "v1_record_sha256": self.v1_record_sha256,
            "path_length": self.path_length,
            "step_index": self.step_index,
            "exclusion_code": self.code.value,
            "detail": self.detail,
            "unsupported_teacher_steps": self.unsupported_teacher_steps,
        }
        return {**body, "rejection_sha256": _canonical_sha256(body)}


class ProcessV2RebindError(RuntimeError):
    """The rebind plan, task, payload binding, or reduction is invalid."""


class ProcessV2RebindMismatch(ProcessV2RebindError):
    """A V1 record is not what it claims, so the task publishes nothing.

    ``mismatches_by_code`` is a measurement of this refusal, not a constant: it
    counts the findings actually raised.  It is deliberately not written into
    the artifact namespace, because publishing anything at all on an integrity
    mismatch is forbidden.
    """

    def __init__(self, findings: Sequence[ProcessV2RebindFinding]) -> None:
        self.findings = tuple(findings)
        counter: Counter[str] = Counter(finding.code.value for finding in self.findings)
        self.mismatches_by_code: dict[str, int] = dict(sorted(counter.items()))
        codes = sorted(self.mismatches_by_code)
        first = self.findings[0] if self.findings else None
        location = (
            f"entry {first.entry_index} step {first.step_index}" if first else "no entry"
        )
        super().__init__(
            f"Process-V2 rebind refuses to publish {len(self.findings)} finding(s) "
            f"{codes} (first at {location}: {first.detail if first else ''})"
        )

    def as_report(self) -> dict[str, Any]:
        """Return the complete, deterministic, unpublished refusal report."""

        body: dict[str, Any] = {
            "schema": REFUSAL_SCHEMA,
            "schema_version": REFUSAL_SCHEMA_VERSION,
            "status": REFUSAL_STATUS,
            "training_authorized": False,
            "published": False,
            "mismatch_count": len(self.findings),
            "mismatches_by_code": dict(self.mismatches_by_code),
            "findings": [finding.as_payload() for finding in self.findings],
        }
        return {**body, "refusal_sha256": _canonical_sha256(body)}


class ProcessV2RebindIncomplete(ProcessV2RebindError):
    """At least one planned rebind range has not published a complete proof."""


# ---- Deterministic serialization ---------------------------------------------


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, object], *, field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or _HEX_RE.fullmatch(value) is None:
        raise ProcessV2RebindError(f"{field} must be a lowercase SHA-256")
    return value


def _require_git_object(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64} or _HEX_RE.fullmatch(value) is None:
        raise ProcessV2RebindError(f"{field} must be a lowercase Git-style object identity")
    return value


def _require_artifact_path(value: object, *, field: str) -> str:
    raw = value if isinstance(value, str) else ""
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "artifacts"
        or ".." in path.parts
        or str(path) != raw
        or raw.endswith("/")
    ):
        raise ProcessV2RebindError(f"{field} must be a normalized path below /artifacts")
    return raw


def _mounted_artifact_path(artifact_path: str, *, artifact_root: Path, field: str) -> Path:
    normalized = _require_artifact_path(artifact_path, field=field)
    root = Path(artifact_root).resolve()
    resolved = (root / PurePosixPath(normalized).relative_to("/artifacts")).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProcessV2RebindError(f"{field} resolves outside the artifact root") from error
    return resolved


def mounted_process_v2_artifact_path(
    artifact_path: str,
    *,
    artifact_root: Path,
    field: str,
) -> Path:
    """Resolve one ``/artifacts`` path under a mounted root, refusing escapes.

    Published so a consumer of these artifacts resolves them through the same
    normalization and containment check the writer used, instead of rebuilding
    a second, possibly weaker one.
    """

    return _mounted_artifact_path(artifact_path, artifact_root=artifact_root, field=field)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2RebindError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ProcessV2RebindError(f"{label} must be an object: {path}")
    return value


def _publish_json_atomically(target: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    """Write once, or prove the published bytes are already exactly these."""

    content = _canonical_json_bytes(payload, newline=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ProcessV2RebindError(f"immutable {label} collision at {target}")
        return target
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


# ---- Process identity resolution ---------------------------------------------


def _process_identity_symbol(name: str):
    """Return one frozen-identity entry point or name the exact missing symbol."""

    symbol = getattr(process_identity_module, name, None)
    if symbol is None:
        raise ProcessV2RebindError(
            "the Process-V2 rebind requires "
            f"compose_v4.rewrite.editing_v2_process_identity.{name}, which is absent"
        )
    return symbol


def editing_process_v2_identity() -> dict[str, Any]:
    """Return the live Process-V2 semantic identity every artifact is bound to."""

    identity = _process_identity_symbol("editing_process_v2_identity")()
    if not isinstance(identity, Mapping):
        raise ProcessV2RebindError("the Process-V2 identity must be an object")
    return dict(identity)


def validate_pinned_process_identity(value: object) -> dict[str, Any]:
    """Accept only an internally self-consistent historical process identity."""

    if not isinstance(value, Mapping):
        raise ProcessV2RebindError("a pinned process identity must be an object")
    validate = _process_identity_symbol("validate_frozen_process_identity")
    try:
        validated = validate(dict(value))
    except ValueError as error:
        raise ProcessV2RebindError(
            "the pinned V1 process identity is not internally self-consistent"
        ) from error
    if not isinstance(validated, Mapping):
        raise ProcessV2RebindError("frozen process identity validation returned no object")
    return dict(validated)


def validate_pinned_builder_identity(value: object) -> dict[str, Any]:
    """Accept only a self-hash-consistent historical packed-builder identity."""

    if not isinstance(value, Mapping):
        raise ProcessV2RebindError("a pinned builder identity must be an object")
    builder = dict(value)
    if builder.get("identity_sha256") != _self_hash(builder, field="identity_sha256"):
        raise ProcessV2RebindError(
            "the pinned V1 packed-builder identity self-hash does not match its contents"
        )
    return builder


# ---- Independent bounded Process-V2 atom-delete oracle ------------------------


def _independent_real_atom_graph(state: MolecularGraph) -> nx.Graph:
    """Build the real-atom graph from the persistent-slot bond matrix directly."""

    real = np.flatnonzero(is_element(state.atom_types))
    adjacency = state.bonds[np.ix_(real, real)] != BOND_NULL
    graph = nx.from_numpy_array(adjacency.astype(np.int8))
    return nx.relabel_nodes(
        graph,
        {index: int(slot) for index, slot in enumerate(real)},
        copy=True,
    )


def _independent_within_declared_support(state: MolecularGraph) -> bool:
    """Broad-organic representability of every real slot, at most 40 active."""

    real = np.flatnonzero(is_element(state.atom_types))
    if int(real.size) > MAX_ACTIVE_ATOMS:
        return False
    contribution = np.asarray(BOND_CLASS_TO_H_CHANGE, dtype=np.int64)
    bond_order_sums = contribution[state.bonds[real].astype(np.int64)].sum(axis=1)
    for offset, slot in enumerate(int(value) for value in real):
        if (
            ORGANIC_VOCABULARY.class_index(
                int(state.atom_types[slot]),
                int(bond_order_sums[offset]),
                int(state.implicit_h_counts[slot]),
                int(state.formal_charges[slot]),
            )
            is None
        ):
            return False
    return True


def _independent_scar_incident(state: MolecularGraph, slot: int) -> bool:
    """True when any bonded neighbour of ``slot`` occupies a SCAR site."""

    return bool(np.any((state.bonds[slot] != BOND_NULL) & is_scar(state.atom_types)))


def independent_process_v2_atom_delete_slots(state: MolecularGraph) -> tuple[int, ...]:
    """Bounded independent oracle for the *complete* Process-V2 delete fiber.

    This derives every ``atom_delete`` candidate, inherited and newly
    introduced alike, from one uniform admission rule.  Six gates apply to both
    structural candidate sources: real element under the authoritative
    predicate, the unchanged executor validator, successor connectivity, the
    authoritative charge policy, declared broad-organic at-most-40-active-atom
    support, and successor canonicalizability.  Three further gates apply only
    to a connected-nonleaf candidate, meaning a real slot of real-atom degree at
    least two: non-aromatic under resonance-invariant perception, not a graph
    articulation point, and not incident to a SCAR.  A degree-at-most-one slot
    is never exempt from the six common gates; the structural source is a
    diagnostic label, not an exemption.

    The gates are conjunctive, so the admitted set does not depend on the order
    in which they are evaluated.  This oracle deliberately evaluates them in its
    own order and builds its own real-atom graph, and it never imports or calls
    :mod:`compose_v4.rewrite.process_v2_atom_delete`.

    Claim boundary: this is a second *derivation*, not a second *design*.  It
    shares the executor, the charge policy, aromatic perception, the
    canonicalizer and the declared vocabulary, because those are the
    authorities and duplicating them would be the very approximation the
    handoff forbids.  Agreement therefore catches a coding error, a plumbing
    error, a mask-composition error or drift between the two implementations.
    It cannot catch a shared misreading of the declared semantics, because both
    encode the same reading.  It is a bounded proof and test oracle and is not
    a production successor-kernel implementation.
    """

    if not is_valid_state(state) or not is_connected_or_null(state):
        return ()
    real = np.flatnonzero(is_element(state.atom_types))
    if int(real.size) == 0:
        return ()
    graph = _independent_real_atom_graph(state)
    cut_vertices = frozenset(int(vertex) for vertex in nx.articulation_points(graph))
    aromatic_slot = np.any(resonance_invariant_bond_classes(state) == BOND_AROMATIC, axis=1)
    admitted: list[int] = []
    for slot in (int(value) for value in real):
        connected_nonleaf = (
            int(graph.degree[slot]) >= INDEPENDENT_CONNECTED_NONLEAF_MINIMUM_DEGREE
        )
        if connected_nonleaf and (
            bool(aromatic_slot[slot])
            or slot in cut_vertices
            or _independent_scar_incident(state, slot)
        ):
            continue
        action = AtomDelete(slot)
        if not is_valid_atom_delete(state, action):
            continue
        successor = apply_atom_delete(state, action)
        if not is_connected_or_null(successor):
            continue
        if not charge_policy_preserved(state, successor):
            continue
        if not _independent_within_declared_support(successor):
            continue
        try:
            canonical_state_key(successor)
        except InvalidRewrite:
            continue
        admitted.append(slot)
    return tuple(admitted)


def process_v2_atom_delete_mask_authority():
    """Return the complete effective mask authority or name the missing symbol.

    The authority is resolved by name at call time rather than imported at
    module load, so a worktree whose Process-V2 module has not yet published
    the complete effective mask fails with an exact, actionable message instead
    of an import error, and never silently validates a narrower mask.
    """

    symbol = getattr(
        process_v2_atom_delete_module,
        PROCESS_V2_ATOM_DELETE_MASK_SYMBOL,
        None,
    )
    if symbol is None:
        raise ProcessV2RebindError(
            "the Process-V2 rebind requires compose_v4.rewrite.process_v2_atom_delete."
            f"{PROCESS_V2_ATOM_DELETE_MASK_SYMBOL}, the complete effective atom-delete "
            "mask authority, which is absent"
        )
    return symbol


def authority_process_v2_atom_delete_slots(state: MolecularGraph) -> tuple[int, ...]:
    """Return the production effective Process-V2 delete slots for one state."""

    mask = np.asarray(process_v2_atom_delete_mask_authority()(state))
    if mask.dtype != np.bool_ or mask.shape != (state.n_atoms,):
        raise ProcessV2RebindError(
            "the Process-V2 atom-delete mask authority must return a boolean mask of "
            f"shape ({state.n_atoms},), got dtype {mask.dtype} shape {mask.shape}"
        )
    return tuple(int(slot) for slot in np.flatnonzero(mask))


def independent_process_v2_atom_delete_candidate_source(
    state: MolecularGraph,
    slot: int,
) -> str:
    """Label one slot's structural candidate source, for diagnostics only."""

    if not 0 <= slot < state.n_atoms or not bool(is_element(state.atom_types[slot])):
        raise ProcessV2RebindError(
            f"slot {slot} is not a real atom, so it has no candidate source"
        )
    real = np.flatnonzero(is_element(state.atom_types))
    degree = int(np.count_nonzero(state.bonds[slot, real] != BOND_NULL))
    return (
        CONNECTED_NONLEAF_CANDIDATE_SOURCE
        if degree >= INDEPENDENT_CONNECTED_NONLEAF_MINIMUM_DEGREE
        else INHERITED_CANDIDATE_SOURCE
    )


# ---- Source revision ----------------------------------------------------------


def _implementation_files(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root)
    files: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise ProcessV2RebindError(f"rebind implementation source is absent: {source}")
        files[relative] = _file_sha256(source)
    return files


def build_process_v2_rebind_source_revision(
    *,
    commit: str,
    tree: str,
    repo_root: Path,
    worktree_clean: bool,
) -> dict[str, Any]:
    """Bind the complete serialized proof implementation to one Git revision."""

    commit = _require_git_object(commit, field="source revision commit")
    tree = _require_git_object(tree, field="source revision tree")
    if worktree_clean is not True:
        raise ProcessV2RebindError("the Process-V2 rebind requires a clean committed worktree")
    implementation_files = _implementation_files(Path(repo_root))
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "implementation_files": implementation_files,
        "implementation_files_sha256": _canonical_sha256(implementation_files),
        "process_v2_identity_sha256": editing_process_v2_identity()["process_identity_sha256"],
    }
    return {**body, "source_revision_sha256": _canonical_sha256(body)}


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProcessV2RebindError(
            f"cannot establish the rebind Git identity: git {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def repository_process_v2_rebind_source_revision(*, repo_root: Path) -> dict[str, Any]:
    """Return a revision only when every serialized file belongs to clean HEAD."""

    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProcessV2RebindError("the Process-V2 rebind refuses a dirty serialized-code tree")
    for relative in _SOURCE_FILES:
        if _git(root, "rev-parse", f"HEAD:{relative}") != _git(root, "hash-object", "--", relative):
            raise ProcessV2RebindError(f"rebind source is off the bound revision: {relative}")
    return build_process_v2_rebind_source_revision(
        commit=commit,
        tree=tree,
        repo_root=root,
        worktree_clean=True,
    )


def validate_process_v2_rebind_source_revision(
    value: object,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate a clean-tree binding against the live serialized files."""

    if not isinstance(value, Mapping):
        raise ProcessV2RebindError("the rebind source revision must be an object")
    payload = dict(value)
    if set(payload) != _SOURCE_REVISION_FIELDS:
        raise ProcessV2RebindError("rebind source revision fields disagree")
    _require_git_object(payload["commit"], field="source revision commit")
    _require_git_object(payload["tree"], field="source revision tree")
    live_files = _implementation_files(Path(repo_root))
    if (
        payload["schema"] != SOURCE_REVISION_SCHEMA
        or payload["schema_version"] != SOURCE_REVISION_SCHEMA_VERSION
        or payload["worktree_clean"] is not True
        or payload["implementation_files"] != live_files
        or payload["implementation_files_sha256"] != _canonical_sha256(live_files)
        or payload["process_v2_identity_sha256"]
        != editing_process_v2_identity()["process_identity_sha256"]
        or payload["source_revision_sha256"] != _self_hash(payload, field="source_revision_sha256")
    ):
        raise ProcessV2RebindError("the rebind source revision is stale or malformed")
    return payload


# ---- V1 payload binding -------------------------------------------------------


def _bind_v1_task(
    task_dir: Path,
    *,
    task_artifact_path: str,
    pinned_process_identity: Mapping[str, Any],
    pinned_builder_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable V1 task payload under the pinned identities."""

    inventory = {path.name for path in task_dir.iterdir()}
    if inventory != {DECISION_FILENAME, V1_RECEIPT_FILENAME, SEMANTIC_ARTIFACT_DIRNAME}:
        raise ProcessV2RebindError(
            f"V1 task payload inventory is incomplete or unexpected: {task_dir}"
        )
    receipt_path = task_dir / V1_RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path, label="V1 migration receipt")
    if receipt.get("receipt_sha256") != _self_hash(receipt, field="receipt_sha256"):
        raise ProcessV2RebindError(f"V1 migration receipt self-hash disagrees: {receipt_path}")
    if (
        receipt.get("schema") != MATERIALIZATION_SCHEMA
        or receipt.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or receipt.get("status") != MATERIALIZATION_STATUS
        or receipt.get("training_authorized") is not False
    ):
        raise ProcessV2RebindError(f"V1 migration receipt contract disagrees: {receipt_path}")
    if receipt.get("process_identity") != dict(pinned_process_identity):
        raise ProcessV2RebindError(
            f"V1 migration receipt does not carry the pinned process identity: {receipt_path}"
        )
    if receipt.get("builder_identity") != dict(pinned_builder_identity):
        raise ProcessV2RebindError(
            f"V1 migration receipt does not carry the pinned builder identity: {receipt_path}"
        )
    decision_sha256 = _file_sha256(task_dir / DECISION_FILENAME)
    if receipt.get("decision_ledger_sha256") != decision_sha256:
        raise ProcessV2RebindError(
            f"V1 decision-ledger SHA-256 disagrees with its receipt: {task_dir}"
        )
    for field in ("data_lane", "split"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ProcessV2RebindError(f"V1 migration receipt {field} must be nonempty")
    shard_sha256 = _require_sha256(
        receipt.get("semantic_shard_sha256"),
        field="V1 receipt semantic_shard_sha256",
    )
    manifest_sha256 = _require_sha256(
        receipt.get("semantic_manifest_sha256"),
        field="V1 receipt semantic_manifest_sha256",
    )
    try:
        manifest = load_semantic_packed_manifest(
            task_dir / SEMANTIC_ARTIFACT_DIRNAME,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_source_binding=receipt.get("source_binding"),
            expected_process_identity=pinned_process_identity,
            expected_builder_identity=pinned_builder_identity,
        )
    except SemanticPackedStoreError as error:
        raise ProcessV2RebindError(
            f"V1 semantic packed artifact is invalid under the pinned identities: {task_dir}"
        ) from error
    if manifest["data_lane"] != receipt["data_lane"] or manifest["split"] != receipt["split"]:
        raise ProcessV2RebindError(f"V1 semantic lane or split disagrees with its receipt: {task_dir}")
    return {
        "v1_task_identity_sha256": task_dir.name,
        "v1_task_artifact_path": task_artifact_path,
        "data_lane": str(receipt["data_lane"]),
        "split": str(receipt["split"]),
        "v1_receipt_file_sha256": _file_sha256(receipt_path),
        "v1_receipt_sha256": str(receipt["receipt_sha256"]),
        "v1_decision_ledger_sha256": decision_sha256,
        "v1_semantic_shard_sha256": shard_sha256,
        "v1_semantic_manifest_sha256": manifest_sha256,
        "v1_source_binding": dict(receipt["source_binding"]),
        "v1_entries": int(manifest["entries"]),
    }


def bind_v1_semantic_payload(
    *,
    payload_root_artifact_path: str,
    artifact_root: Path,
    pinned_process_identity: Mapping[str, object],
    pinned_builder_identity: Mapping[str, object],
) -> dict[str, Any]:
    """Bind one completed V1 migration payload as immutable chemical data.

    Every regular file directly under the payload root, including the V1
    ``SEMANTIC_MIGRATION_PLAN.json`` and ``SEMANTIC_MIGRATION_COMPLETE.json``
    when present, is bound by physical SHA-256 only.  Those two artifacts
    cannot be revalidated semantically because they assert the *live* V1
    identity, which Process V2 supersedes; binding their bytes records exactly
    what was read without pretending the superseded assertion still holds.
    """

    process_identity = validate_pinned_process_identity(pinned_process_identity)
    builder_identity = validate_pinned_builder_identity(pinned_builder_identity)
    root = _mounted_artifact_path(
        payload_root_artifact_path,
        artifact_root=artifact_root,
        field="payload_root_artifact_path",
    )
    if not root.is_dir():
        raise ProcessV2RebindError(f"the V1 migration payload root is absent: {root}")
    task_root = root / TASK_DIRNAME
    if not task_root.is_dir():
        raise ProcessV2RebindError(f"the V1 migration payload has no task namespace: {task_root}")
    payload_root_files: dict[str, str] = {}
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name == TASK_DIRNAME:
            continue
        if not entry.is_file():
            raise ProcessV2RebindError(
                f"unexpected object in the V1 migration payload root: {entry}"
            )
        payload_root_files[entry.name] = _file_sha256(entry)
    v1_tasks: list[dict[str, Any]] = []
    for entry in sorted(task_root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir() or _HEX_RE.fullmatch(entry.name) is None or len(entry.name) != 64:
            raise ProcessV2RebindError(
                f"unexpected object in the V1 migration task namespace: {entry}"
            )
        v1_tasks.append(
            _bind_v1_task(
                entry,
                task_artifact_path=f"{payload_root_artifact_path}/{TASK_DIRNAME}/{entry.name}",
                pinned_process_identity=process_identity,
                pinned_builder_identity=builder_identity,
            )
        )
    if not v1_tasks:
        raise ProcessV2RebindError(f"the V1 migration payload has no task results: {task_root}")
    body: dict[str, Any] = {
        "schema": V1_PAYLOAD_BINDING_SCHEMA,
        "schema_version": V1_PAYLOAD_BINDING_SCHEMA_VERSION,
        "payload_root_artifact_path": _require_artifact_path(
            payload_root_artifact_path,
            field="payload_root_artifact_path",
        ),
        "payload_root_files": payload_root_files,
        "pinned_process_identity_sha256": process_identity["process_identity_sha256"],
        "pinned_builder_identity_sha256": builder_identity["identity_sha256"],
        "v1_tasks": v1_tasks,
        "v1_tasks_sha256": _canonical_sha256(v1_tasks),
        "v1_entry_count": sum(int(task["v1_entries"]) for task in v1_tasks),
    }
    return {**body, "binding_sha256": _canonical_sha256(body)}


def validate_v1_semantic_payload_binding(
    value: object,
    *,
    pinned_process_identity: Mapping[str, object],
    pinned_builder_identity: Mapping[str, object],
) -> dict[str, Any]:
    """Validate the recorded payload binding without re-reading the payload."""

    if not isinstance(value, Mapping):
        raise ProcessV2RebindError("the V1 payload binding must be an object")
    binding = dict(value)
    if set(binding) != _V1_PAYLOAD_BINDING_FIELDS:
        raise ProcessV2RebindError("V1 payload binding fields disagree")
    process_identity = validate_pinned_process_identity(pinned_process_identity)
    builder_identity = validate_pinned_builder_identity(pinned_builder_identity)
    tasks = binding["v1_tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ProcessV2RebindError("the V1 payload binding must name at least one task")
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or set(task) != _V1_TASK_BINDING_FIELDS:
            raise ProcessV2RebindError(f"V1 payload task binding {index} fields disagree")
        identity = _require_sha256(
            task["v1_task_identity_sha256"],
            field=f"v1_tasks[{index}].v1_task_identity_sha256",
        )
        if identity in seen:
            raise ProcessV2RebindError("the V1 payload binding repeats a task identity")
        seen.add(identity)
        _require_artifact_path(
            task["v1_task_artifact_path"],
            field=f"v1_tasks[{index}].v1_task_artifact_path",
        )
        for field in (
            "v1_receipt_file_sha256",
            "v1_receipt_sha256",
            "v1_decision_ledger_sha256",
            "v1_semantic_shard_sha256",
            "v1_semantic_manifest_sha256",
        ):
            _require_sha256(task[field], field=f"v1_tasks[{index}].{field}")
        if type(task["v1_entries"]) is not int or task["v1_entries"] < 0:
            raise ProcessV2RebindError(f"v1_tasks[{index}].v1_entries must be nonnegative")
        for field in ("data_lane", "split"):
            if not isinstance(task[field], str) or not task[field]:
                raise ProcessV2RebindError(f"v1_tasks[{index}].{field} must be nonempty")
    if (
        binding["schema"] != V1_PAYLOAD_BINDING_SCHEMA
        or binding["schema_version"] != V1_PAYLOAD_BINDING_SCHEMA_VERSION
        or binding["pinned_process_identity_sha256"] != process_identity["process_identity_sha256"]
        or binding["pinned_builder_identity_sha256"] != builder_identity["identity_sha256"]
        or binding["v1_tasks_sha256"] != _canonical_sha256(tasks)
        or binding["v1_entry_count"] != sum(int(task["v1_entries"]) for task in tasks)
        or binding["binding_sha256"] != _self_hash(binding, field="binding_sha256")
    ):
        raise ProcessV2RebindError("the V1 payload binding identity or census disagrees")
    _require_artifact_path(
        binding["payload_root_artifact_path"],
        field="payload_root_artifact_path",
    )
    files = binding["payload_root_files"]
    if not isinstance(files, dict) or any(
        not isinstance(name, str) or Path(name).name != name for name in files
    ):
        raise ProcessV2RebindError("V1 payload root files must be basename-keyed")
    for name, digest in files.items():
        _require_sha256(digest, field=f"payload_root_files[{name}]")
    return binding


# ---- Plan ---------------------------------------------------------------------


def _entry_ranges(entries: int, *, entries_per_task: int) -> tuple[tuple[int, int], ...]:
    if entries == 0:
        ranges: tuple[tuple[int, int], ...] = ((0, 0),)
    else:
        ranges = tuple(
            (start, min(start + entries_per_task, entries))
            for start in range(0, entries, entries_per_task)
        )
    return validate_semantic_packed_entry_ranges(ranges, entries=entries)


def validate_process_v2_rebind_task_ranges(
    entry_ranges: Iterable[tuple[int, int]],
    *,
    entries: int,
) -> tuple[tuple[int, int], ...]:
    """Require an unordered range set to partition one exact shard census.

    Ranges are sorted before delegation, so a duplicate, an overlap, a gap and
    an incomplete cover are all rejected regardless of the order in which
    independent task results were discovered.
    """

    ordered = tuple(sorted((int(start), int(stop)) for start, stop in entry_ranges))
    try:
        return validate_semantic_packed_entry_ranges(ordered, entries=entries)
    except ValueError as error:
        raise ProcessV2RebindError(str(error)) from error


def plan_process_v2_rebind(
    v1_payload_binding: Mapping[str, Any],
    *,
    source_revision: Mapping[str, Any],
    repo_root: Path,
    pinned_process_identity: Mapping[str, object],
    pinned_builder_identity: Mapping[str, object],
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
    entries_per_task: int = DEFAULT_ENTRIES_PER_TASK,
) -> dict[str, Any]:
    """Build a deterministic, content-addressed entry-range task manifest."""

    if type(entries_per_task) is not int or entries_per_task <= 0:
        raise ProcessV2RebindError("entries_per_task must be a positive integer")
    revision = validate_process_v2_rebind_source_revision(source_revision, repo_root=repo_root)
    binding = validate_v1_semantic_payload_binding(
        v1_payload_binding,
        pinned_process_identity=pinned_process_identity,
        pinned_builder_identity=pinned_builder_identity,
    )
    process_identity = validate_pinned_process_identity(pinned_process_identity)
    builder_identity = validate_pinned_builder_identity(pinned_builder_identity)
    process_v2_identity = editing_process_v2_identity()
    prefix = _require_artifact_path(output_artifact_prefix, field="output_artifact_prefix")
    run_identity_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_revision_sha256": revision["source_revision_sha256"],
        "v1_payload_binding_sha256": binding["binding_sha256"],
        "process_v2_identity_sha256": process_v2_identity["process_identity_sha256"],
        "output_artifact_prefix": prefix,
        "entries_per_task": entries_per_task,
    }
    run_identity_sha256 = _canonical_sha256(run_identity_body)
    run_artifact_root = f"{prefix}/{run_identity_sha256}"
    tasks: list[dict[str, Any]] = []
    for v1_task in binding["v1_tasks"]:
        for entry_start, entry_stop in _entry_ranges(
            int(v1_task["v1_entries"]),
            entries_per_task=entries_per_task,
        ):
            task_body = {
                **dict(v1_task),
                "entry_start": entry_start,
                "entry_stop": entry_stop,
            }
            task_identity_sha256 = _canonical_sha256(
                {"run_identity_sha256": run_identity_sha256, "task": task_body}
            )
            tasks.append(
                {
                    **task_body,
                    "task_identity_sha256": task_identity_sha256,
                    "output_artifact_path": (
                        f"{run_artifact_root}/{TASK_DIRNAME}/{task_identity_sha256}"
                    ),
                }
            )
    body: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        "training_authorized": False,
        "implementation_revision": revision["commit"],
        "source_revision": revision,
        "pinned_process_identity": process_identity,
        "pinned_builder_identity": builder_identity,
        "process_v2_identity": process_v2_identity,
        "v1_payload_binding": binding,
        "v1_payload_binding_sha256": binding["binding_sha256"],
        "run_identity_sha256": run_identity_sha256,
        "run_artifact_root": run_artifact_root,
        "output_artifact_prefix": prefix,
        "entries_per_task": entries_per_task,
        "expected_task_count": len(tasks),
        "expected_entry_count": int(binding["v1_entry_count"]),
        "tasks": tasks,
        "task_inventory_sha256": _canonical_sha256(tasks),
    }
    plan = {**body, "plan_sha256": _canonical_sha256(body)}
    validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    return plan


def validate_process_v2_rebind_plan(value: object, *, repo_root: Path) -> dict[str, Any]:
    """Validate plan identity, live V2 binding, task order and range coverage."""

    if not isinstance(value, Mapping):
        raise ProcessV2RebindError("the rebind plan must be an object")
    plan = dict(value)
    if set(plan) != _PLAN_FIELDS:
        raise ProcessV2RebindError("rebind plan fields disagree")
    revision = validate_process_v2_rebind_source_revision(
        plan["source_revision"],
        repo_root=repo_root,
    )
    process_identity = validate_pinned_process_identity(plan["pinned_process_identity"])
    builder_identity = validate_pinned_builder_identity(plan["pinned_builder_identity"])
    binding = validate_v1_semantic_payload_binding(
        plan["v1_payload_binding"],
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    process_v2_identity = editing_process_v2_identity()
    prefix = _require_artifact_path(
        plan.get("output_artifact_prefix"),
        field="output_artifact_prefix",
    )
    entries_per_task = plan.get("entries_per_task")
    if type(entries_per_task) is not int or entries_per_task <= 0:
        raise ProcessV2RebindError("entries_per_task must be a positive integer")
    run_identity_sha256 = _canonical_sha256(
        {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "source_revision_sha256": revision["source_revision_sha256"],
            "v1_payload_binding_sha256": binding["binding_sha256"],
            "process_v2_identity_sha256": process_v2_identity["process_identity_sha256"],
            "output_artifact_prefix": prefix,
            "entries_per_task": entries_per_task,
        }
    )
    run_artifact_root = f"{prefix}/{run_identity_sha256}"
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ProcessV2RebindError("the rebind plan must contain at least one task")
    v1_tasks = {task["v1_task_identity_sha256"]: task for task in binding["v1_tasks"]}
    observed: dict[str, list[tuple[int, int]]] = {identity: [] for identity in v1_tasks}
    task_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or set(task) != _TASK_FIELDS:
            raise ProcessV2RebindError(f"rebind task {index} fields disagree")
        task = dict(task)
        identity = task["v1_task_identity_sha256"]
        if identity not in v1_tasks:
            raise ProcessV2RebindError(f"rebind task {index} names an unbound V1 task payload")
        if {key: task[key] for key in _V1_TASK_BINDING_FIELDS} != v1_tasks[identity]:
            raise ProcessV2RebindError(f"rebind task {index} V1 payload binding disagrees")
        entry_start = task["entry_start"]
        entry_stop = task["entry_stop"]
        if type(entry_start) is not int or type(entry_stop) is not int:
            raise ProcessV2RebindError(f"rebind task {index} entry bounds must be integers")
        observed[identity].append((entry_start, entry_stop))
        task_body = {key: task[key] for key in _V1_TASK_BINDING_FIELDS | {"entry_start", "entry_stop"}}
        task_id = _canonical_sha256(
            {"run_identity_sha256": run_identity_sha256, "task": task_body}
        )
        if task_id in task_ids:
            raise ProcessV2RebindError("the rebind plan repeats a task identity and entry range")
        if (
            task["task_identity_sha256"] != task_id
            or task["output_artifact_path"] != f"{run_artifact_root}/{TASK_DIRNAME}/{task_id}"
        ):
            raise ProcessV2RebindError(f"rebind task {index} identity or output path disagrees")
        task_ids.add(task_id)
    for identity, ranges in observed.items():
        validate_process_v2_rebind_task_ranges(
            ranges,
            entries=int(v1_tasks[identity]["v1_entries"]),
        )
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
        or plan["training_authorized"] is not False
        or plan["implementation_revision"] != revision["commit"]
        or plan["process_v2_identity"] != process_v2_identity
        or plan["v1_payload_binding_sha256"] != binding["binding_sha256"]
        or plan["run_identity_sha256"] != run_identity_sha256
        or plan["run_artifact_root"] != run_artifact_root
        or plan["expected_task_count"] != len(tasks)
        or plan["expected_entry_count"] != int(binding["v1_entry_count"])
        or plan["task_inventory_sha256"] != _canonical_sha256(tasks)
        or plan["plan_sha256"] != _self_hash(plan, field="plan_sha256")
    ):
        raise ProcessV2RebindError("rebind plan identity, order, provenance or authority disagrees")
    return plan


def write_process_v2_rebind_plan(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> Path:
    """Atomically publish or exactly reuse the content-addressed plan."""

    validated = validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    run_root = _mounted_artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    return _publish_json_atomically(
        run_root / PLAN_FILENAME,
        validated,
        label="Process-V2 rebind plan",
    )


def load_process_v2_rebind_plan(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Load one published plan and verify its exact semantic identity."""

    return validate_process_v2_rebind_plan(
        _load_json_object(Path(path), label="Process-V2 rebind plan"),
        repo_root=repo_root,
    )


# ---- Proof --------------------------------------------------------------------


def _teacher_support_exclusion(
    action_record: Mapping[str, Any],
) -> tuple[ProcessV2RebindExclusionCode, str] | None:
    """Reject a teacher whose rule or family is outside the frozen support.

    The frozen support is read from its registries rather than retyped:
    :data:`compose_v4.rewrite.action_codec_v4.ACTIVE8_EXECUTOR_RULES` for the
    executor surface and
    :data:`compose_v4.data.editing_corpus_contract.ACTIVE8_FAMILIES` for the
    development families.  This runs before the codec decode so an unsupported
    rule is reported as the support exclusion it is, not as a codec defect.
    """

    executor_rule = action_record.get("executor_rule")
    model_family = action_record.get("model_family")
    if executor_rule not in action_codec_v4.ACTIVE8_EXECUTOR_RULES:
        return (
            ProcessV2RebindExclusionCode.TEACHER_RULE_OUTSIDE_FROZEN_SUPPORT,
            f"executor rule {executor_rule!r} is outside the frozen Active8 surface",
        )
    if model_family not in ACTIVE8_FAMILIES:
        return (
            ProcessV2RebindExclusionCode.TEACHER_FAMILY_OUTSIDE_FROZEN_SUPPORT,
            f"model family {model_family!r} is outside the frozen Active8 families",
        )
    return None


def _prove_record(
    *,
    entry_index: int,
    record: Mapping[str, Any] | None,
    addressed: Any,
    read_error: str | None,
    task: Mapping[str, Any],
    pinned_process_identity: Mapping[str, Any],
    runtime: Any,
    counts: Counter[str],
    families: Counter[str],
    unsupported_steps: Counter[str],
) -> tuple[
    dict[str, Any] | None,
    ProcessV2RebindRejection | None,
    list[ProcessV2RebindFinding],
]:
    """Re-prove one V1 record, then decide admission. Never admits partially.

    Returns ``(proof, rejection, findings)``.  At most one of ``proof`` and
    ``rejection`` is ever populated.  ``findings`` is nonempty only for an
    integrity mismatch, which the caller turns into a task-wide refusal.

    Integrity wins over support, and it is decided over the *whole* record: a
    step whose teacher falls outside the frozen fiber is still codec-checked,
    replayed and compared, so a corrupt payload can never hide behind an
    unsupported teacher.  The only step whose integrity cannot be established
    is one whose executor rule the frozen codec cannot decode at all.
    """

    findings: list[ProcessV2RebindFinding] = []
    exclusions: list[tuple[int, ProcessV2RebindExclusionCode, str]] = []
    trace_id: str | None = None
    record_sha256: str | None = None

    def reject(
        code: ProcessV2RebindIntegrityCode,
        detail: str,
        step: int | None = None,
    ) -> None:
        findings.append(
            ProcessV2RebindFinding(
                entry_index=entry_index,
                step_index=step,
                code=code,
                detail=detail,
                trace_id=trace_id,
                v1_record_sha256=record_sha256,
            )
        )

    def exclude(step: int, code: ProcessV2RebindExclusionCode, detail: str) -> None:
        exclusions.append((step, code, detail))
        unsupported_steps[code.value] += 1

    if read_error is not None or record is None or addressed is None:
        reject(
            ProcessV2RebindIntegrityCode.V1_RECORD_UNREADABLE,
            read_error or "the persisted V1 row could not be read",
        )
        return None, None, findings

    if isinstance(record.get("record_sha256"), str):
        record_sha256 = str(record["record_sha256"])
    if isinstance(record.get("trace_id"), str):
        trace_id = str(record["trace_id"])

    if record.get("record_sha256") != _self_hash(record, field="record_sha256"):
        reject(
            ProcessV2RebindIntegrityCode.V1_RECORD_SELF_HASH_DISAGREES,
            "persisted record self-hash does not match its contents",
        )
        return None, None, findings
    if (
        record.get("schema") != TRACE_SCHEMA
        or record.get("schema_version") != TRACE_SCHEMA_VERSION
        or record.get("action_codec_schema_version") != action_codec_v4.SCHEMA_VERSION
        or record.get("action_codec_implementation_hash")
        != action_codec_v4.codec_implementation_hash()
    ):
        reject(
            ProcessV2RebindIntegrityCode.V1_RECORD_SCHEMA_DISAGREES,
            "persisted trace schema or action-codec identity disagrees",
        )
        return None, None, findings
    if (
        record.get("process_semantics") != PROCESS_SEMANTICS
        or record.get("process_identity_sha256")
        != pinned_process_identity["process_identity_sha256"]
        or record.get("process_contract_sha256") != pinned_process_identity["contract_sha256"]
    ):
        reject(
            ProcessV2RebindIntegrityCode.V1_PROCESS_IDENTITY_DISAGREES,
            "persisted record does not carry the pinned V1 process identity",
        )
        return None, None, findings

    address = addressed.address
    states = tuple(addressed.path.states)
    keys = record.get("canonical_state_keys")
    steps = record.get("steps")
    source_address = record.get("source_address")
    lineage = record.get("lineage")
    if (
        record.get("trace_id") != address.trace_id
        or record.get("data_lane") != task["data_lane"]
        or record.get("split") != task["split"]
        or record.get("data_lane") != address.layer
        or record.get("split") != address.partition
        or record.get("path_length") != address.path_length
        or not isinstance(source_address, dict)
        or not source_address
        or not isinstance(lineage, dict)
        or not isinstance(record.get("family_histogram"), dict)
        or not isinstance(keys, list)
        or not isinstance(steps, list)
        or len(keys) != len(states)
        or len(steps) != len(states) - 1
        or keys[0] != address.source_key
        or keys[-1] != address.target_key
    ):
        reject(
            ProcessV2RebindIntegrityCode.V1_PROVENANCE_DISAGREES,
            "persisted provenance disagrees with the validated shard address",
        )
        return None, None, findings

    try:
        if canonical_state_key(states[0]) != keys[0]:
            reject(
                ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES,
                "recomputed source canonical key differs from the persisted key",
                0,
            )
    except InvalidRewrite as error:
        reject(
            ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES,
            f"source state is not canonicalizable: {error}",
            0,
        )

    # The complete effective Process-V2 atom-delete mask, at every progress
    # state, against the independent oracle.  This runs before the teacher loop
    # so a support decision consults the exact array the model would score, and
    # only after that array has been proven to agree with a second derivation.
    candidates: list[list[int]] = []
    for state_index, state in enumerate(states):
        authority = authority_process_v2_atom_delete_slots(state)
        oracle = independent_process_v2_atom_delete_slots(state)
        if authority != oracle:
            reject(
                ProcessV2RebindIntegrityCode.PROCESS_V2_MASK_DISAGREES,
                f"production effective mask {list(authority)} disagrees with the "
                f"independent oracle {list(oracle)}",
                state_index,
            )
        candidates.append(list(authority))
        counts["mask_states_evaluated"] += 1
        counts["mask_candidate_slots"] += len(authority)
        if authority:
            counts["mask_states_with_candidates"] += 1

    replayed = 0
    atom_delete_teachers: Counter[str] = Counter()
    for step_index, entry in enumerate(steps):
        if not isinstance(entry, dict) or set(entry) != {
            "action",
            "source_key",
            "successor_key",
        }:
            reject(
                ProcessV2RebindIntegrityCode.V1_RECORD_SCHEMA_DISAGREES,
                "persisted step fields disagree",
                step_index,
            )
            continue
        action_record = entry["action"]
        if not isinstance(action_record, dict) or not all(
            isinstance(action_record.get(field), str)
            for field in ("executor_rule", "model_family")
        ):
            # A missing or non-string rule or family is a schema defect, not a
            # statement about support, so it must not be miscoded as one.
            reject(
                ProcessV2RebindIntegrityCode.V1_RECORD_SCHEMA_DISAGREES,
                "persisted step action does not name an executor rule and model family",
                step_index,
            )
            continue
        unsupported = _teacher_support_exclusion(action_record)
        if unsupported is not None:
            exclude(step_index, unsupported[0], unsupported[1])
            continue
        try:
            rule, action = action_codec_v4.decode_action(action_record)
            reencoded = action_codec_v4.encode_action(rule, action)
        except ValueError as error:
            reject(
                ProcessV2RebindIntegrityCode.ACTION_CODEC_ROUNDTRIP_DISAGREES,
                f"persisted action does not decode under ActionCodecV4: {error}",
                step_index,
            )
            continue
        if reencoded != action_record:
            reject(
                ProcessV2RebindIntegrityCode.ACTION_CODEC_ROUNDTRIP_DISAGREES,
                "re-encoded action differs from the persisted action record",
                step_index,
            )
            continue
        if (
            entry["source_key"] != keys[step_index]
            or entry["successor_key"] != keys[step_index + 1]
        ):
            reject(
                ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES,
                "persisted step keys disagree with the persisted state keys",
                step_index,
            )
            continue
        if rule == "atom_delete":
            slot = int(action.v)
            state = states[step_index]
            if not 0 <= slot < state.n_atoms or not bool(is_element(state.atom_types[slot])):
                reject(
                    ProcessV2RebindIntegrityCode.V1_PROVENANCE_DISAGREES,
                    f"persisted atom_delete teacher targets slot {slot}, which is not "
                    "a real atom of its persisted source state",
                    step_index,
                )
                continue
            source_label = independent_process_v2_atom_delete_candidate_source(state, slot)
            if slot not in candidates[step_index]:
                # Record the exclusion but keep verifying: an integrity defect
                # must never be maskable behind an unsupported teacher, so this
                # step is still replayed and compared below.
                exclude(
                    step_index,
                    ProcessV2RebindExclusionCode.ATOM_DELETE_OUTSIDE_PROCESS_V2_MASK,
                    f"atom_delete teacher at slot {slot} ({source_label}) is outside the "
                    "effective Process-V2 mask "
                    f"{candidates[step_index]}",
                )
            else:
                atom_delete_teachers[source_label] += 1
        try:
            successor = runtime.apply(states[step_index], rule, action)
        except ValueError as error:
            reject(
                ProcessV2RebindIntegrityCode.EXECUTOR_REPLAY_FAILED,
                f"unchanged executor rejected the persisted teacher: {error}",
                step_index,
            )
            continue
        if not all(
            np.array_equal(getattr(successor, field), getattr(states[step_index + 1], field))
            for field in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
        ):
            reject(
                ProcessV2RebindIntegrityCode.SUCCESSOR_ARRAY_DISAGREES,
                "replayed successor differs from the persisted persistent-slot state",
                step_index,
            )
            continue
        try:
            successor_key = canonical_state_key(successor)
        except InvalidRewrite as error:
            reject(
                ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES,
                f"replayed successor is not canonicalizable: {error}",
                step_index,
            )
            continue
        if successor_key != keys[step_index + 1]:
            reject(
                ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES,
                "replayed successor canonical key differs from the persisted key",
                step_index,
            )
            continue
        replayed += 1

    if findings:
        return None, None, findings
    if exclusions:
        # Whole-trace exclusion: no proof row, no partial credit, and every
        # per-transition counter this record would have contributed is dropped.
        step_index, code, detail = exclusions[0]
        counts["rejected_entries"] += 1
        return (
            None,
            ProcessV2RebindRejection(
                entry_index=entry_index,
                trace_id=str(record["trace_id"]),
                data_lane=str(record["data_lane"]),
                split=str(record["split"]),
                v1_record_sha256=str(record["record_sha256"]),
                path_length=int(record["path_length"]),
                step_index=step_index,
                code=code,
                detail=detail,
                unsupported_teacher_steps=len(exclusions),
            ),
            findings,
        )

    counts["admitted_entries"] += 1
    counts["admitted_states"] += len(states)
    counts["admitted_transitions"] += replayed
    for source_label in CANDIDATE_SOURCES:
        counts[f"atom_delete_teachers_{source_label}"] += atom_delete_teachers[source_label]
    for admitted_step in steps:
        families[str(admitted_step["action"]["model_family"])] += 1
    proof_body: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "schema_version": PROOF_SCHEMA_VERSION,
        "entry_index": entry_index,
        "trace_id": str(record["trace_id"]),
        "data_lane": str(record["data_lane"]),
        "split": str(record["split"]),
        "source_address": dict(source_address),
        "lineage": dict(lineage),
        "v1_record_sha256": str(record["record_sha256"]),
        "path_length": int(record["path_length"]),
        "canonical_state_keys": [str(key) for key in keys],
        "family_histogram": dict(record["family_histogram"]),
        "replayed_transitions": replayed,
        "atom_delete_teachers_by_candidate_source": {
            source_label: int(atom_delete_teachers[source_label])
            for source_label in CANDIDATE_SOURCES
        },
        "process_v2_atom_delete_candidates": candidates,
    }
    return {**proof_body, "proof_sha256": _canonical_sha256(proof_body)}, None, findings


def _prove_entry_range(
    task: Mapping[str, Any],
    *,
    v1_task_dir: Path,
    pinned_process_identity: Mapping[str, Any],
    pinned_builder_identity: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[ProcessV2RebindRejection],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    """Prove one half-open entry range, accounting for every record it read.

    Raises :class:`ProcessV2RebindMismatch` on any integrity finding, so the
    caller publishes nothing.  A support exclusion is not a mismatch: it
    returns as a rejection and the range still completes.
    """

    semantic_dir = v1_task_dir / SEMANTIC_ARTIFACT_DIRNAME
    entry_start = int(task["entry_start"])
    entry_stop = int(task["entry_stop"])
    proofs: list[dict[str, Any]] = []
    rejections: list[ProcessV2RebindRejection] = []
    findings: list[ProcessV2RebindFinding] = []
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    unsupported_steps: Counter[str] = Counter()
    observed_indices: list[int] = []
    try:
        rows = read_semantic_packed_artifact_range_rows(
            semantic_dir,
            expected_shard_sha256=task["v1_semantic_shard_sha256"],
            expected_manifest_sha256=task["v1_semantic_manifest_sha256"],
            expected_source_binding=task["v1_source_binding"],
            expected_process_identity=pinned_process_identity,
            expected_builder_identity=pinned_builder_identity,
            entry_start=entry_start,
            entry_stop=entry_stop,
            # The rebind replays every transition itself, which strictly
            # dominates the reader's first-N sentinel replay.
            sentinel_replay_entries=0,
            # An unreadable row is this module's V1_RECORD_UNREADABLE finding,
            # addressed to its exact physical entry, not an untyped crash.
            recover_row_errors=True,
        )
        runtime = editing_v2_semantic_rewrite_system()
        for row in rows:
            observed_indices.append(row.entry_index)
            counts["source_entries"] += 1
            proof, rejection, row_findings = _prove_record(
                entry_index=row.entry_index,
                record=row.record,
                addressed=row.addressed,
                read_error=row.error,
                task=task,
                pinned_process_identity=pinned_process_identity,
                runtime=runtime,
                counts=counts,
                families=families,
                unsupported_steps=unsupported_steps,
            )
            findings.extend(row_findings)
            if proof is not None:
                proofs.append(proof)
            if rejection is not None:
                rejections.append(rejection)
    except SemanticPackedStoreError as error:
        raise ProcessV2RebindError(
            f"the V1 semantic entry range [{entry_start}, {entry_stop}) is unreadable "
            f"under the pinned identities: {error}"
        ) from error
    if findings:
        raise ProcessV2RebindMismatch(findings)
    if observed_indices != list(range(entry_start, entry_stop)):
        raise ProcessV2RebindError(
            f"the V1 semantic entry range [{entry_start}, {entry_stop}) did not read "
            "its exact physical entry indices"
        )
    if len(proofs) + len(rejections) != entry_stop - entry_start:
        raise ProcessV2RebindError(
            f"the V1 semantic entry range [{entry_start}, {entry_stop}) accounts for "
            f"{len(proofs)} admitted and {len(rejections)} rejected records"
        )
    return proofs, rejections, counts, families, unsupported_steps


def _write_proof_ledger(path: Path, proofs: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Write the deterministic gzip proof ledger; return physical and stream hashes."""

    stream = hashlib.sha256()
    raw_handle = path.open("wb")
    try:
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
        try:
            for proof in proofs:
                line = _canonical_json_bytes(proof, newline=True)
                stream.update(line)
                compressed.write(line)
        finally:
            compressed.close()
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    finally:
        raw_handle.close()
    return _file_sha256(path), stream.hexdigest()


# ---- Task execution -----------------------------------------------------------


def _task_by_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    _require_sha256(task_identity_sha256, field="task_identity_sha256")
    matches = [
        task for task in plan["tasks"] if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise ProcessV2RebindError("task identity is absent or duplicated in the frozen plan")
    return dict(matches[0])


def _validate_reason_histogram(
    value: object,
    *,
    allowed: set[str],
    field: str,
) -> dict[str, int]:
    """Require a measured reason histogram over immutable reason codes only."""

    if not isinstance(value, dict):
        raise ProcessV2RebindError(f"{field} must be an object")
    for key, count in value.items():
        if key not in allowed:
            raise ProcessV2RebindError(f"{field} names an unknown reason code {key!r}")
        if type(count) is not int or count <= 0:
            raise ProcessV2RebindError(f"{field}[{key}] must be a positive integer")
    if list(value) != sorted(value):
        raise ProcessV2RebindError(f"{field} must be sorted by reason code")
    return {str(key): int(count) for key, count in value.items()}


def _validate_rejected_trace_inventory(value: object) -> list[dict[str, Any]]:
    """Require an exact, ascending, reason-coded rejected-trace inventory."""

    if not isinstance(value, list):
        raise ProcessV2RebindError("manifest.rejected_traces must be a list")
    allowed = {code.value for code in ProcessV2RebindExclusionCode}
    rows: list[dict[str, Any]] = []
    previous = -1
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _REJECTION_FIELDS:
            raise ProcessV2RebindError(f"manifest.rejected_traces[{index}] fields disagree")
        if (
            row["schema"] != REJECTION_SCHEMA
            or row["schema_version"] != REJECTION_SCHEMA_VERSION
            or row["rejection_sha256"] != _self_hash(row, field="rejection_sha256")
            or row["exclusion_code"] not in allowed
            or type(row["entry_index"]) is not int
            or type(row["step_index"]) is not int
            or type(row["unsupported_teacher_steps"]) is not int
            or row["unsupported_teacher_steps"] <= 0
            or not isinstance(row["trace_id"], str)
            or not row["trace_id"]
        ):
            raise ProcessV2RebindError(f"manifest.rejected_traces[{index}] disagrees")
        _require_sha256(
            row["v1_record_sha256"],
            field=f"manifest.rejected_traces[{index}].v1_record_sha256",
        )
        if int(row["entry_index"]) <= previous:
            raise ProcessV2RebindError(
                "manifest.rejected_traces must be ascending and unique by entry index"
            )
        previous = int(row["entry_index"])
        rows.append(dict(row))
    return rows


def validate_process_v2_rebind_task_result(
    output_dir: Path,
    *,
    expected_receipt: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Reconcile one complete published proof artifact from its own bytes."""

    output = Path(output_dir)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {
        MANIFEST_FILENAME,
        PROOF_FILENAME,
        RECEIPT_FILENAME,
    }:
        raise ProcessV2RebindError(
            f"Process-V2 rebind output inventory is incomplete or unexpected: {output}"
        )
    receipt = _load_json_object(output / RECEIPT_FILENAME, label="Process-V2 rebind receipt")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ProcessV2RebindError("Process-V2 rebind receipt fields disagree")
    if receipt["receipt_sha256"] != _self_hash(receipt, field="receipt_sha256"):
        raise ProcessV2RebindError("Process-V2 rebind receipt self-hash disagrees")
    if expected_receipt is not None and receipt != dict(expected_receipt):
        raise ProcessV2RebindError("Process-V2 rebind receipt differs from the expected result")
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["status"] != TASK_STATUS
        or receipt["training_authorized"] is not False
    ):
        raise ProcessV2RebindError("Process-V2 rebind receipt contract disagrees")
    if receipt["process_v2_identity"] != editing_process_v2_identity():
        raise ProcessV2RebindError("Process-V2 rebind receipt process identity is stale")
    validate_pinned_process_identity(receipt["pinned_process_identity"])
    validate_pinned_builder_identity(receipt["pinned_builder_identity"])

    manifest_path = output / MANIFEST_FILENAME
    manifest = _load_json_object(manifest_path, label="Process-V2 rebind manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ProcessV2RebindError("Process-V2 rebind manifest fields disagree")
    if manifest["manifest_sha256"] != _self_hash(manifest, field="manifest_sha256"):
        raise ProcessV2RebindError("Process-V2 rebind manifest self-hash disagrees")
    if (
        manifest["manifest_sha256"] != receipt["manifest_sha256"]
        or _file_sha256(manifest_path) != receipt["manifest_physical_sha256"]
        or manifest["status"] != TASK_STATUS
        or manifest["training_authorized"] is not False
        or manifest["process_v2_identity_sha256"]
        != receipt["process_v2_identity"]["process_identity_sha256"]
        or set(manifest["teacher_census"]) != set(_TEACHER_CENSUS_FIELDS)
        or set(manifest["process_v2_atom_delete_census"]) != set(_MASK_CENSUS_FIELDS)
    ):
        raise ProcessV2RebindError("Process-V2 rebind manifest does not bind its receipt")
    _validate_reason_histogram(
        manifest["rejected_traces_by_code"],
        allowed={code.value for code in ProcessV2RebindExclusionCode},
        field="manifest.rejected_traces_by_code",
    )
    _validate_reason_histogram(
        manifest["unsupported_teacher_steps_by_code"],
        allowed={code.value for code in ProcessV2RebindExclusionCode},
        field="manifest.unsupported_teacher_steps_by_code",
    )
    rejected = _validate_rejected_trace_inventory(manifest["rejected_traces"])

    proof_path = output / PROOF_FILENAME
    if _file_sha256(proof_path) != receipt["proof_ledger_sha256"]:
        raise ProcessV2RebindError("Process-V2 rebind proof-ledger SHA-256 disagrees")
    stream = hashlib.sha256()
    admitted_indices: list[int] = []
    inventory: list[list[Any]] = []
    states = 0
    transitions = 0
    with gzip.open(proof_path, "rb") as handle:
        for row_index, raw_line in enumerate(handle):
            if not raw_line.endswith(b"\n") or not raw_line.strip():
                raise ProcessV2RebindError(f"rebind proof row {row_index} is not canonical JSONL")
            stream.update(raw_line)
            try:
                proof = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ProcessV2RebindError(f"rebind proof row {row_index} is malformed") from error
            if (
                not isinstance(proof, dict)
                or set(proof) != _PROOF_FIELDS
                or _canonical_json_bytes(proof, newline=True) != raw_line
                or proof["proof_sha256"] != _self_hash(proof, field="proof_sha256")
                or proof["schema"] != PROOF_SCHEMA
                or proof["schema_version"] != PROOF_SCHEMA_VERSION
                or set(proof["atom_delete_teachers_by_candidate_source"]) != set(CANDIDATE_SOURCES)
            ):
                raise ProcessV2RebindError(f"rebind proof row {row_index} disagrees")
            admitted_indices.append(int(proof["entry_index"]))
            inventory.append([int(proof["entry_index"]), str(proof["v1_record_sha256"])])
            states += len(proof["canonical_state_keys"])
            transitions += int(proof["replayed_transitions"])
    if stream.hexdigest() != receipt["proof_stream_sha256"]:
        raise ProcessV2RebindError("Process-V2 rebind proof stream SHA-256 disagrees")
    inventory.extend(
        [int(row["entry_index"]), str(row["v1_record_sha256"])] for row in rejected
    )
    inventory.sort()
    covered = [entry_index for entry_index, _ in inventory]
    exact_range = list(range(int(receipt["entry_start"]), int(receipt["entry_stop"])))
    if admitted_indices != sorted(admitted_indices) or covered != exact_range:
        raise ProcessV2RebindError(
            "Process-V2 rebind admitted and rejected records do not cover the exact range"
        )
    if (
        manifest["source_entries"] != len(exact_range)
        or manifest["admitted_entries"] != len(admitted_indices)
        or manifest["rejected_entries"] != len(rejected)
        or manifest["admitted_states"] != states
        or manifest["admitted_transitions"] != transitions
        or manifest["v1_record_inventory_sha256"] != _canonical_sha256(inventory)
        or manifest["proof_stream_sha256"] != receipt["proof_stream_sha256"]
        or manifest["rejected_traces_by_code"] != receipt["rejected_traces_by_code"]
        or manifest["unsupported_teacher_steps_by_code"]
        != receipt["unsupported_teacher_steps_by_code"]
        or sum(manifest["rejected_traces_by_code"].values()) != len(rejected)
        or receipt["counts"]
        != {
            "source_entries": manifest["source_entries"],
            "admitted_entries": manifest["admitted_entries"],
            "rejected_entries": manifest["rejected_entries"],
            "admitted_states": manifest["admitted_states"],
            "admitted_transitions": manifest["admitted_transitions"],
        }
    ):
        raise ProcessV2RebindError("Process-V2 rebind proof census disagrees with its manifest")
    return receipt


def _publish_task(
    output: Path,
    *,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    (staging / MANIFEST_FILENAME).write_bytes(_canonical_json_bytes(manifest, newline=True))
    (staging / RECEIPT_FILENAME).write_bytes(_canonical_json_bytes(receipt, newline=True))
    validate_process_v2_rebind_task_result(staging, expected_receipt=receipt)
    if output.exists():
        return validate_process_v2_rebind_task_result(output, expected_receipt=receipt)
    try:
        os.rename(staging, output)
    except OSError:
        if not output.exists():
            raise
        return validate_process_v2_rebind_task_result(output, expected_receipt=receipt)
    return dict(receipt)


def execute_process_v2_rebind_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Prove or exactly reuse one restart-safe entry-range rebind task."""

    validated = validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    task = _task_by_identity(validated, task_identity_sha256)
    pinned_process_identity = validated["pinned_process_identity"]
    pinned_builder_identity = validated["pinned_builder_identity"]
    v1_task_dir = _mounted_artifact_path(
        task["v1_task_artifact_path"],
        artifact_root=artifact_root,
        field="task.v1_task_artifact_path",
    )
    output = _mounted_artifact_path(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    if output.exists():
        existing = validate_process_v2_rebind_task_result(output)
        if any(
            existing.get(field) != task[field]
            for field in (
                "v1_task_identity_sha256",
                "v1_task_artifact_path",
                "data_lane",
                "split",
                "entry_start",
                "entry_stop",
                "v1_entries",
                "v1_source_binding",
                "v1_semantic_shard_sha256",
                "v1_semantic_manifest_sha256",
            )
        ) or existing.get("task_identity_sha256") != task_identity_sha256:
            raise ProcessV2RebindError(f"immutable Process-V2 rebind collision at {output}")
        return {
            "task_identity_sha256": task_identity_sha256,
            "output_artifact_path": task["output_artifact_path"],
            "receipt_sha256": existing["receipt_sha256"],
            "counts": existing["counts"],
            "reused": True,
        }

    proofs, rejections, counts, families, unsupported_steps = _prove_entry_range(
        task,
        v1_task_dir=v1_task_dir,
        pinned_process_identity=pinned_process_identity,
        pinned_builder_identity=pinned_builder_identity,
    )
    rejected_rows = [rejection.as_payload() for rejection in rejections]
    rejected_traces_by_code = dict(
        sorted(Counter(rejection.code.value for rejection in rejections).items())
    )
    unsupported_teacher_steps_by_code = dict(sorted(unsupported_steps.items()))
    record_inventory = sorted(
        [
            *([int(proof["entry_index"]), str(proof["v1_record_sha256"])] for proof in proofs),
            *([int(row["entry_index"]), str(row["v1_record_sha256"])] for row in rejected_rows),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.", suffix=".staging")
    )
    try:
        proof_ledger_sha256, proof_stream_sha256 = _write_proof_ledger(
            staging / PROOF_FILENAME,
            proofs,
        )
        manifest_body: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": TASK_STATUS,
            "training_authorized": False,
            "v1_task_identity_sha256": task["v1_task_identity_sha256"],
            "v1_task_artifact_path": task["v1_task_artifact_path"],
            "data_lane": task["data_lane"],
            "split": task["split"],
            "entry_start": task["entry_start"],
            "entry_stop": task["entry_stop"],
            "v1_entries": task["v1_entries"],
            "source_entries": counts["source_entries"],
            "admitted_entries": counts["admitted_entries"],
            "rejected_entries": counts["rejected_entries"],
            "admitted_states": counts["admitted_states"],
            "admitted_transitions": counts["admitted_transitions"],
            "family_histogram": dict(sorted(families.items())),
            "teacher_census": {
                "atom_delete_teachers": sum(
                    counts[f"atom_delete_teachers_{source}"] for source in CANDIDATE_SOURCES
                ),
                **{
                    f"atom_delete_teachers_{source}": counts[f"atom_delete_teachers_{source}"]
                    for source in CANDIDATE_SOURCES
                },
                "teacher_transitions": counts["admitted_transitions"],
            },
            "process_v2_atom_delete_census": {
                "states_evaluated": counts["mask_states_evaluated"],
                "states_with_candidates": counts["mask_states_with_candidates"],
                "candidate_slots": counts["mask_candidate_slots"],
            },
            # Measured over the exact rows this range read.  An integrity
            # mismatch never reaches this point, so no mismatch histogram is
            # published; it is a measurement of a refusal and is carried on
            # ProcessV2RebindMismatch instead.
            "rejected_traces_by_code": rejected_traces_by_code,
            "unsupported_teacher_steps_by_code": unsupported_teacher_steps_by_code,
            "rejected_traces": rejected_rows,
            "v1_record_inventory_sha256": _canonical_sha256(record_inventory),
            "proof_stream_sha256": proof_stream_sha256,
            "pinned_process_identity_sha256": pinned_process_identity["process_identity_sha256"],
            "pinned_builder_identity_sha256": pinned_builder_identity["identity_sha256"],
            "process_v2_identity_sha256": validated["process_v2_identity"][
                "process_identity_sha256"
            ],
        }
        manifest = {**manifest_body, "manifest_sha256": _canonical_sha256(manifest_body)}
        manifest_bytes = _canonical_json_bytes(manifest, newline=True)
        receipt_body: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": TASK_STATUS,
            "training_authorized": False,
            "implementation_revision": validated["implementation_revision"],
            "run_identity_sha256": validated["run_identity_sha256"],
            "task_identity_sha256": task_identity_sha256,
            "v1_task_identity_sha256": task["v1_task_identity_sha256"],
            "v1_task_artifact_path": task["v1_task_artifact_path"],
            "data_lane": task["data_lane"],
            "split": task["split"],
            "entry_start": task["entry_start"],
            "entry_stop": task["entry_stop"],
            "v1_entries": task["v1_entries"],
            "v1_source_binding": task["v1_source_binding"],
            "v1_semantic_shard_sha256": task["v1_semantic_shard_sha256"],
            "v1_semantic_manifest_sha256": task["v1_semantic_manifest_sha256"],
            "pinned_process_identity": dict(pinned_process_identity),
            "pinned_builder_identity": dict(pinned_builder_identity),
            "process_v2_identity": dict(validated["process_v2_identity"]),
            "proof_ledger_sha256": proof_ledger_sha256,
            "proof_stream_sha256": proof_stream_sha256,
            "manifest_physical_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"],
            "counts": {field: counts[field] for field in _COUNT_FIELDS},
            "rejected_traces_by_code": rejected_traces_by_code,
            "unsupported_teacher_steps_by_code": unsupported_teacher_steps_by_code,
        }
        receipt = {**receipt_body, "receipt_sha256": _canonical_sha256(receipt_body)}
        published = _publish_task(
            output,
            receipt=receipt,
            manifest=manifest,
            staging=staging,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "task_identity_sha256": task_identity_sha256,
        "output_artifact_path": task["output_artifact_path"],
        "receipt_sha256": published["receipt_sha256"],
        "counts": published["counts"],
        "reused": False,
    }


def _validate_task_result_against_plan(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    output = _mounted_artifact_path(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    receipt = validate_process_v2_rebind_task_result(output)
    if (
        receipt["run_identity_sha256"] != plan["run_identity_sha256"]
        or receipt["task_identity_sha256"] != task["task_identity_sha256"]
        or receipt["implementation_revision"] != plan["implementation_revision"]
        or receipt["pinned_process_identity"] != plan["pinned_process_identity"]
        or receipt["pinned_builder_identity"] != plan["pinned_builder_identity"]
        or receipt["process_v2_identity"] != plan["process_v2_identity"]
        or any(
            receipt[field] != task[field]
            for field in (
                "v1_task_identity_sha256",
                "v1_task_artifact_path",
                "data_lane",
                "split",
                "entry_start",
                "entry_stop",
                "v1_entries",
                "v1_source_binding",
                "v1_semantic_shard_sha256",
                "v1_semantic_manifest_sha256",
            )
        )
    ):
        raise ProcessV2RebindError(
            f"Process-V2 rebind result mismatches task {task['task_identity_sha256']}"
        )
    return receipt


def completed_process_v2_rebind_task_ids(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> set[str]:
    """Return exact reusable task identities and reject unexpected task objects."""

    validated = validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    task_root = _mounted_artifact_path(
        f"{validated['run_artifact_root']}/{TASK_DIRNAME}",
        artifact_root=artifact_root,
        field="plan task root",
    )
    expected = {task["task_identity_sha256"] for task in validated["tasks"]}
    if not task_root.exists():
        return set()
    observed = {path.name for path in task_root.iterdir()}
    # A task publishes by directory rename.  A hard termination can leave only
    # its hidden, non-authoritative staging sibling, which is not a result and
    # must not prevent the exact task from being retried.  Every other
    # unexpected object remains a hard failure.
    private_staging = {
        name
        for name in observed
        if any(name.startswith(f".{task_id}.") and name.endswith(".staging") for task_id in expected)
    }
    unexpected = (observed - private_staging) - expected
    if unexpected:
        raise ProcessV2RebindError(
            f"Process-V2 rebind task namespace contains unexpected objects: {sorted(unexpected)}"
        )
    tasks_by_id = {task["task_identity_sha256"]: task for task in validated["tasks"]}
    complete: set[str] = set()
    for task_id in sorted(observed - private_staging):
        _validate_task_result_against_plan(
            validated,
            tasks_by_id[task_id],
            artifact_root=artifact_root,
        )
        complete.add(task_id)
    return complete


def _counter_sum(target: Counter[str], value: object, *, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ProcessV2RebindError(f"{field} must be an object")
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise ProcessV2RebindError(f"{field} must map strings to nonnegative integers")
        target[key] += count


def reduce_process_v2_rebind(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Publish completion after an exact reduction that accounts for every record.

    Every planned range must have published a complete result, and a range that
    hit an integrity mismatch never publishes, so a missing range is the only
    way an integrity failure can reach this point and it raises
    :class:`ProcessV2RebindIncomplete`.  Support exclusions are expected: the
    completion carries their summed, measured census and requires
    admitted plus rejected to equal the exact source census of the payload.
    """

    validated = validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    run_root = _mounted_artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    published_plan_path = run_root / PLAN_FILENAME
    expected_plan_bytes = _canonical_json_bytes(validated, newline=True)
    if not published_plan_path.is_file() or published_plan_path.read_bytes() != expected_plan_bytes:
        raise ProcessV2RebindError(
            "the Process-V2 rebind reduction requires the exact published plan bytes"
        )
    complete = completed_process_v2_rebind_task_ids(
        validated,
        artifact_root=artifact_root,
        repo_root=repo_root,
    )
    expected = {task["task_identity_sha256"] for task in validated["tasks"]}
    missing = expected - complete
    if missing:
        raise ProcessV2RebindIncomplete(
            f"the Process-V2 rebind is missing {len(missing)} planned range results"
        )
    v1_entries = {
        task["v1_task_identity_sha256"]: int(task["v1_entries"])
        for task in validated["v1_payload_binding"]["v1_tasks"]
    }
    observed_ranges: dict[str, list[tuple[int, int]]] = {identity: [] for identity in v1_entries}
    results: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    teacher_totals: Counter[str] = Counter()
    mask_totals: Counter[str] = Counter()
    rejected_trace_totals: Counter[str] = Counter()
    unsupported_step_totals: Counter[str] = Counter()
    rejected_trace_inventory: list[dict[str, Any]] = []
    for task in validated["tasks"]:
        receipt = _validate_task_result_against_plan(
            validated,
            task,
            artifact_root=artifact_root,
        )
        observed_ranges[receipt["v1_task_identity_sha256"]].append(
            (int(receipt["entry_start"]), int(receipt["entry_stop"]))
        )
        _counter_sum(totals, receipt["counts"], field="task receipt counts")
        _counter_sum(
            rejected_trace_totals,
            receipt["rejected_traces_by_code"],
            field="rejected_traces_by_code",
        )
        _counter_sum(
            unsupported_step_totals,
            receipt["unsupported_teacher_steps_by_code"],
            field="unsupported_teacher_steps_by_code",
        )
        manifest = _load_json_object(
            _mounted_artifact_path(
                task["output_artifact_path"],
                artifact_root=artifact_root,
                field="task.output_artifact_path",
            )
            / MANIFEST_FILENAME,
            label="Process-V2 rebind manifest",
        )
        _counter_sum(family_totals, manifest["family_histogram"], field="family_histogram")
        _counter_sum(teacher_totals, manifest["teacher_census"], field="teacher_census")
        _counter_sum(
            mask_totals,
            manifest["process_v2_atom_delete_census"],
            field="process_v2_atom_delete_census",
        )
        rejected_trace_inventory.extend(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "v1_task_identity_sha256": task["v1_task_identity_sha256"],
                "data_lane": str(row["data_lane"]),
                "split": str(row["split"]),
                "entry_index": int(row["entry_index"]),
                "trace_id": str(row["trace_id"]),
                "v1_record_sha256": str(row["v1_record_sha256"]),
                "step_index": int(row["step_index"]),
                "exclusion_code": str(row["exclusion_code"]),
                "unsupported_teacher_steps": int(row["unsupported_teacher_steps"]),
            }
            for row in manifest["rejected_traces"]
        )
        results.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "v1_task_identity_sha256": task["v1_task_identity_sha256"],
                "data_lane": task["data_lane"],
                "split": task["split"],
                "entry_start": task["entry_start"],
                "entry_stop": task["entry_stop"],
                "output_artifact_path": task["output_artifact_path"],
                "receipt_sha256": receipt["receipt_sha256"],
                "proof_ledger_sha256": receipt["proof_ledger_sha256"],
                "manifest_sha256": receipt["manifest_sha256"],
                "counts": receipt["counts"],
            }
        )
    for identity, ranges in observed_ranges.items():
        validate_process_v2_rebind_task_ranges(ranges, entries=v1_entries[identity])
    if totals["source_entries"] != int(validated["expected_entry_count"]):
        raise ProcessV2RebindError(
            "the Process-V2 rebind reduction does not account for every V1 record"
        )
    if totals["admitted_entries"] + totals["rejected_entries"] != totals["source_entries"]:
        raise ProcessV2RebindError(
            "the Process-V2 rebind reduction admits and rejects an inconsistent census"
        )
    if sum(rejected_trace_totals.values()) != totals["rejected_entries"] or len(
        rejected_trace_inventory
    ) != int(totals["rejected_entries"]):
        raise ProcessV2RebindError(
            "the Process-V2 rebind rejection census does not name every rejected trace"
        )
    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        "training_authorized": False,
        "gate_zero_run": False,
        "run_identity_sha256": validated["run_identity_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "plan_file_sha256": hashlib.sha256(expected_plan_bytes).hexdigest(),
        "source_revision_sha256": validated["source_revision"]["source_revision_sha256"],
        "pinned_process_identity_sha256": validated["pinned_process_identity"][
            "process_identity_sha256"
        ],
        "pinned_builder_identity_sha256": validated["pinned_builder_identity"]["identity_sha256"],
        "process_v2_identity_sha256": validated["process_v2_identity"]["process_identity_sha256"],
        "v1_payload_binding_sha256": validated["v1_payload_binding_sha256"],
        "task_inventory_sha256": validated["task_inventory_sha256"],
        "task_count": len(results),
        "result_inventory": results,
        "result_inventory_sha256": _canonical_sha256(results),
        "counts": {field: totals[field] for field in _COUNT_FIELDS},
        "family_histogram": dict(sorted(family_totals.items())),
        "teacher_census": {
            field: teacher_totals[field] for field in sorted(_TEACHER_CENSUS_FIELDS)
        },
        "process_v2_atom_delete_census": {
            field: mask_totals[field] for field in sorted(_MASK_CENSUS_FIELDS)
        },
        "rejected_traces_by_code": dict(sorted(rejected_trace_totals.items())),
        "unsupported_teacher_steps_by_code": dict(sorted(unsupported_step_totals.items())),
        "rejected_trace_inventory_sha256": _canonical_sha256(rejected_trace_inventory),
    }
    completion = {**body, "completion_sha256": _canonical_sha256(body)}
    run_root.mkdir(parents=True, exist_ok=True)
    _publish_json_atomically(
        run_root / COMPLETION_FILENAME,
        completion,
        label="Process-V2 rebind completion",
    )
    return completion


__all__ = [
    "CANDIDATE_SOURCES",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "COMPLETION_SCHEMA_VERSION",
    "COMPLETION_STATUS",
    "CONNECTED_NONLEAF_CANDIDATE_SOURCE",
    "DEFAULT_ENTRIES_PER_TASK",
    "DEFAULT_OUTPUT_ARTIFACT_PREFIX",
    "INDEPENDENT_CONNECTED_NONLEAF_MINIMUM_DEGREE",
    "INHERITED_CANDIDATE_SOURCE",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "PROCESS_V2_ATOM_DELETE_MASK_SYMBOL",
    "PROOF_FILENAME",
    "PROOF_SCHEMA",
    "PROOF_SCHEMA_VERSION",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA_VERSION",
    "REFUSAL_SCHEMA",
    "REFUSAL_SCHEMA_VERSION",
    "REFUSAL_STATUS",
    "REJECTION_SCHEMA",
    "REJECTION_SCHEMA_VERSION",
    "SOURCE_REVISION_SCHEMA",
    "SOURCE_REVISION_SCHEMA_VERSION",
    "TASK_DIRNAME",
    "TASK_STATUS",
    "V1_PAYLOAD_BINDING_SCHEMA",
    "V1_PAYLOAD_BINDING_SCHEMA_VERSION",
    "ProcessV2RebindError",
    "ProcessV2RebindExclusionCode",
    "ProcessV2RebindFinding",
    "ProcessV2RebindIncomplete",
    "ProcessV2RebindIntegrityCode",
    "ProcessV2RebindMismatch",
    "ProcessV2RebindRejection",
    "authority_process_v2_atom_delete_slots",
    "bind_v1_semantic_payload",
    "build_process_v2_rebind_source_revision",
    "completed_process_v2_rebind_task_ids",
    "editing_process_v2_identity",
    "execute_process_v2_rebind_task",
    "independent_process_v2_atom_delete_candidate_source",
    "independent_process_v2_atom_delete_slots",
    "load_process_v2_rebind_plan",
    "mounted_process_v2_artifact_path",
    "process_v2_atom_delete_mask_authority",
    "plan_process_v2_rebind",
    "reduce_process_v2_rebind",
    "repository_process_v2_rebind_source_revision",
    "validate_pinned_builder_identity",
    "validate_pinned_process_identity",
    "validate_process_v2_rebind_plan",
    "validate_process_v2_rebind_source_revision",
    "validate_process_v2_rebind_task_ranges",
    "validate_process_v2_rebind_task_result",
    "validate_v1_semantic_payload_binding",
    "write_process_v2_rebind_plan",
]
