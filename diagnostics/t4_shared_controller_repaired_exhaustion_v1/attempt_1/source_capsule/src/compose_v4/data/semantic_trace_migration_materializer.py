"""Whole-shard offline migration into the semantic Editing V2 corpus.

Each task scans one immutable historical packed shard exactly once.  Every
source trace receives one deterministic admission decision, and accepted rows
are published as a complete semantic packed artifact.  No read-time action
translation exists downstream of this boundary.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.packed_trace_store import (
    manifest_path_for,
    read_frozen_source_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import overlay_path_for
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    SemanticPackedStoreError,
    load_semantic_packed_manifest,
    semantic_packed_builder_identity,
    write_semantic_packed_artifact,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)
from compose_v4.rewrite.semantic_trace_migration import (
    migrate_and_encode_semantic_trace,
)

MATERIALIZATION_SCHEMA = "compose.data.semantic_trace_migration_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
MATERIALIZATION_STATUS = "COMPLETE_GATE0_NOT_RUN_NO_TRAINING_AUTHORITY"
DECISION_SCHEMA = "compose.data.semantic_trace_migration_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_FILENAME = "decisions.jsonl.gz"
RECEIPT_FILENAME = "RECEIPT.json"
SEMANTIC_ARTIFACT_DIRNAME = "semantic"
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "implementation_revision",
    "source_binding",
    "data_lane",
    "split",
    "process_identity",
    "builder_identity",
    "decision_ledger_sha256",
    "decision_stream_sha256",
    "semantic_shard_sha256",
    "semantic_manifest_sha256",
    "counts",
    "rejections_by_code",
    "accepted_family_histogram",
    "receipt_sha256",
}
_DECISION_FIELDS = {
    "schema",
    "schema_version",
    "source_address",
    "derived_trace_id",
    "decision",
    "semantic_record_sha256",
    "rejection",
    "decision_sha256",
}


class SemanticTraceMaterializationError(RuntimeError):
    """A whole-shard migration task is malformed or incomplete."""


@dataclass(frozen=True)
class SemanticTraceMigrationTask:
    source_path: Path
    source_shard_sha256: str
    source_manifest_sha256: str
    source_overlay_sha256: str | None
    source_unified_manifest_sha256: str
    source_entry_count: int
    implementation_revision: str
    data_lane: str
    split: str
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.data_lane or not self.split:
            raise ValueError("data_lane and split must be nonempty")
        if type(self.source_entry_count) is not int or self.source_entry_count < 0:
            raise ValueError("source_entry_count must be a nonnegative integer")
        if len(self.implementation_revision) not in {40, 64} or any(
            character not in "0123456789abcdef"
            for character in self.implementation_revision
        ):
            raise ValueError(
                "implementation_revision must be a lowercase Git-style hex revision"
            )
        for field in (
            "source_shard_sha256",
            "source_manifest_sha256",
            "source_unified_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.source_overlay_sha256 is not None:
            _require_sha256(
                self.source_overlay_sha256,
                field="source_overlay_sha256",
            )


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _source_binding(task: SemanticTraceMigrationTask) -> dict[str, object]:
    return {
        "source_shard_name": task.source_path.name,
        "source_shard_sha256": task.source_shard_sha256,
        "source_manifest_sha256": task.source_manifest_sha256,
        "source_overlay_sha256": task.source_overlay_sha256,
        "source_unified_manifest_sha256": task.source_unified_manifest_sha256,
        "source_entry_count": task.source_entry_count,
    }


def _validate_task_source_bytes(task: SemanticTraceMigrationTask) -> None:
    source = Path(task.source_path)
    manifest = manifest_path_for(source)
    if not source.is_file() or not manifest.is_file():
        raise SemanticTraceMaterializationError(
            "frozen source shard or manifest is absent"
        )
    if _file_sha256(source) != task.source_shard_sha256:
        raise SemanticTraceMaterializationError(
            "frozen source shard SHA-256 disagrees with the task"
        )
    if _file_sha256(manifest) != task.source_manifest_sha256:
        raise SemanticTraceMaterializationError(
            "frozen source manifest SHA-256 disagrees with the task"
        )
    overlay = overlay_path_for(source)
    if task.source_overlay_sha256 is None:
        if overlay.exists():
            raise SemanticTraceMaterializationError(
                "frozen source has an undeclared provenance overlay"
            )
    elif not overlay.is_file() or _file_sha256(overlay) != task.source_overlay_sha256:
        raise SemanticTraceMaterializationError(
            "frozen source provenance-overlay SHA-256 disagrees with the task"
        )


def _source_address(task: SemanticTraceMigrationTask, addressed) -> dict[str, object]:
    address = addressed.address
    body: dict[str, object] = {
        "source_shard_sha256": task.source_shard_sha256,
        "source_manifest_sha256": task.source_manifest_sha256,
        "source_overlay_sha256": task.source_overlay_sha256,
        "source_shard_name": task.source_path.name,
        "entry_index": address.entry_index,
        "trace_id": address.trace_id,
        "layer": address.layer,
        "partition": address.partition,
        "source_key": address.source_key,
        "target_key": address.target_key,
        "path_length": address.path_length,
    }
    return {**body, "source_address_sha256": _canonical_sha256(body)}


def _derived_trace_id(source_address: Mapping[str, object]) -> str:
    return f"semantic-{source_address['source_address_sha256']}"


def _decision_self_hash(value: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: item for key, item in value.items() if key != "decision_sha256"}
    )


def _iter_canonical_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        for index, raw_line in enumerate(handle):
            if not raw_line.endswith(b"\n") or not raw_line.strip():
                raise SemanticTraceMaterializationError(
                    f"accepted scratch row {index} is not canonical JSONL"
                )
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise SemanticTraceMaterializationError(
                    f"accepted scratch row {index} is malformed"
                ) from error
            if _canonical_json_bytes(value, newline=True) != raw_line:
                raise SemanticTraceMaterializationError(
                    f"accepted scratch row {index} is not canonically serialized"
                )
            yield value


def _write_decision_ledger(
    path: Path,
    decisions_path: Path,
) -> tuple[str, str]:
    """Compress a canonical scratch ledger and return physical and stream hashes."""

    stream_digest = hashlib.sha256()
    with (
        Path(path).open("wb") as raw_handle,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed,
        Path(decisions_path).open("rb") as source,
    ):
        while block := source.read(1 << 20):
            stream_digest.update(block)
            compressed.write(block)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _file_sha256(path), stream_digest.hexdigest()


def _receipt_self_hash(receipt: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticTraceMaterializationError(
            f"semantic migration receipt is unreadable: {path}"
        ) from error
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise SemanticTraceMaterializationError(
            "semantic migration receipt fields disagree"
        )
    if receipt.get("receipt_sha256") != _receipt_self_hash(receipt):
        raise SemanticTraceMaterializationError(
            "semantic migration receipt self-hash disagrees"
        )
    return receipt


def validate_semantic_trace_materialization(
    output_dir: Path,
    *,
    expected_receipt: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Reconcile one complete published migration artifact from its bytes."""

    output = Path(output_dir)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {
        DECISION_FILENAME,
        RECEIPT_FILENAME,
        SEMANTIC_ARTIFACT_DIRNAME,
    }:
        raise SemanticTraceMaterializationError(
            "semantic migration output inventory is incomplete or unexpected"
        )
    receipt = _load_receipt(output / RECEIPT_FILENAME)
    if expected_receipt is not None and receipt != dict(expected_receipt):
        raise SemanticTraceMaterializationError(
            "semantic migration receipt differs from the expected result"
        )
    if (
        receipt.get("schema") != MATERIALIZATION_SCHEMA
        or receipt.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or receipt.get("status") != MATERIALIZATION_STATUS
        or receipt.get("training_authorized") is not False
    ):
        raise SemanticTraceMaterializationError(
            "semantic migration receipt contract disagrees"
        )
    if receipt.get("process_identity") != editing_v2_process_identity():
        raise SemanticTraceMaterializationError(
            "semantic migration process identity is stale"
        )
    if receipt.get("builder_identity") != semantic_packed_builder_identity():
        raise SemanticTraceMaterializationError(
            "semantic migration builder identity is stale"
        )

    decision_path = output / DECISION_FILENAME
    if _file_sha256(decision_path) != receipt["decision_ledger_sha256"]:
        raise SemanticTraceMaterializationError(
            "semantic migration decision-ledger SHA-256 disagrees"
        )
    stream_digest = hashlib.sha256()
    observed: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    indices: list[int] = []
    with gzip.open(decision_path, "rb") as handle:
        for row_index, raw_line in enumerate(handle):
            if not raw_line.endswith(b"\n") or not raw_line.strip():
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} is not canonical JSONL"
                )
            stream_digest.update(raw_line)
            try:
                decision = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} is malformed"
                ) from error
            if (
                not isinstance(decision, dict)
                or set(decision) != _DECISION_FIELDS
                or _canonical_json_bytes(decision, newline=True) != raw_line
                or decision.get("decision_sha256") != _decision_self_hash(decision)
            ):
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} disagrees"
                )
            source_address = decision["source_address"]
            if not isinstance(source_address, dict):
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} lacks an address"
                )
            address_body = {
                key: value
                for key, value in source_address.items()
                if key != "source_address_sha256"
            }
            if source_address.get("source_address_sha256") != _canonical_sha256(
                address_body
            ):
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} address disagrees"
                )
            if decision["derived_trace_id"] != _derived_trace_id(source_address):
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} trace ID disagrees"
                )
            indices.append(source_address["entry_index"])
            observed["source"] += 1
            if decision["decision"] == "admitted":
                if decision["rejection"] is not None:
                    raise SemanticTraceMaterializationError(
                        f"semantic migration admitted row {row_index} has a rejection"
                    )
                _require_sha256(
                    decision["semantic_record_sha256"],
                    field="semantic_record_sha256",
                )
                observed["admitted"] += 1
            elif decision["decision"] == "rejected":
                rejection = decision["rejection"]
                if (
                    decision["semantic_record_sha256"] is not None
                    or not isinstance(rejection, dict)
                    or not isinstance(rejection.get("code"), str)
                ):
                    raise SemanticTraceMaterializationError(
                        f"semantic migration rejected row {row_index} is incomplete"
                    )
                observed["rejected"] += 1
                rejection_counts[rejection["code"]] += 1
            else:
                raise SemanticTraceMaterializationError(
                    f"semantic migration decision row {row_index} has an unknown outcome"
                )
    if stream_digest.hexdigest() != receipt["decision_stream_sha256"]:
        raise SemanticTraceMaterializationError(
            "semantic migration decision stream SHA-256 disagrees"
        )
    if indices != list(range(observed["source"])):
        raise SemanticTraceMaterializationError(
            "semantic migration decision indices are not contiguous"
        )
    observed_counts = {
        "source": observed["source"],
        "admitted": observed["admitted"],
        "rejected": observed["rejected"],
    }
    if observed_counts != receipt["counts"]:
        raise SemanticTraceMaterializationError(
            "semantic migration decision counts disagree"
        )
    if dict(sorted(rejection_counts.items())) != receipt["rejections_by_code"]:
        raise SemanticTraceMaterializationError(
            "semantic migration rejection census disagrees"
        )

    semantic_dir = output / SEMANTIC_ARTIFACT_DIRNAME
    manifest = load_semantic_packed_manifest(
        semantic_dir,
        expected_shard_sha256=receipt["semantic_shard_sha256"],
        expected_manifest_sha256=receipt["semantic_manifest_sha256"],
        expected_source_binding=receipt["source_binding"],
    )
    if manifest["decision_binding"] != {
        "decision_ledger_sha256": receipt["decision_ledger_sha256"],
        "source_count": observed["source"],
        "admitted_count": observed["admitted"],
        "rejected_count": observed["rejected"],
    }:
        raise SemanticTraceMaterializationError(
            "semantic packed derivative and decision ledger disagree"
        )
    if manifest["family_histogram"] != receipt["accepted_family_histogram"]:
        raise SemanticTraceMaterializationError(
            "semantic packed derivative family census disagrees"
        )
    return receipt


def materialize_frozen_packed_shard(
    task: SemanticTraceMigrationTask,
) -> dict[str, Any]:
    """Migrate one exact source shard in one pass and publish atomically."""

    source = Path(task.source_path)
    output = Path(task.output_dir)
    _validate_task_source_bytes(task)
    if output.exists():
        existing = validate_semantic_trace_materialization(output)
        expected_fields = {
            "source_binding": _source_binding(task),
            "data_lane": task.data_lane,
            "split": task.split,
            "implementation_revision": task.implementation_revision,
        }
        if any(
            existing.get(field) != value for field, value in expected_fields.items()
        ):
            raise SemanticTraceMaterializationError(
                f"immutable semantic migration collision at {output}"
            )
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".staging",
        )
    )
    accepted_scratch = staging / ".accepted.jsonl"
    decisions_scratch = staging / ".decisions.jsonl"
    counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    observed_indices: list[int] = []
    try:
        with (
            accepted_scratch.open("wb") as accepted_handle,
            decisions_scratch.open("wb") as decision_handle,
        ):
            for addressed in read_frozen_source_addressed_packed_shard(
                source,
                expected_shard_sha256=task.source_shard_sha256,
                expected_manifest_sha256=task.source_manifest_sha256,
                expected_overlay_sha256=task.source_overlay_sha256,
                verify_fraction=0.0,
            ):
                source_address = _source_address(task, addressed)
                observed_indices.append(addressed.address.entry_index)
                derived_trace_id = _derived_trace_id(source_address)
                record, rejection = migrate_and_encode_semantic_trace(
                    addressed.trace,
                    trace_id=derived_trace_id,
                    data_lane=task.data_lane,
                    split=task.split,
                    source_address=source_address,
                    lineage={
                        "builder_identity_sha256": semantic_packed_builder_identity()[
                            "identity_sha256"
                        ],
                        "source_layer": addressed.address.layer,
                        "source_partition": addressed.address.partition,
                    },
                    legacy_states=addressed.path.states,
                )
                decision_body: dict[str, object] = {
                    "schema": DECISION_SCHEMA,
                    "schema_version": DECISION_SCHEMA_VERSION,
                    "source_address": source_address,
                    "derived_trace_id": derived_trace_id,
                }
                if record is None:
                    if rejection is None:
                        raise SemanticTraceMaterializationError(
                            "migration returned neither a record nor a rejection"
                        )
                    rejection_payload = {
                        **asdict(rejection),
                        "code": rejection.code.value,
                    }
                    decision_body.update(
                        {
                            "decision": "rejected",
                            "semantic_record_sha256": None,
                            "rejection": rejection_payload,
                        }
                    )
                    counts["rejected"] += 1
                    rejection_counts[rejection.code.value] += 1
                else:
                    decision_body.update(
                        {
                            "decision": "admitted",
                            "semantic_record_sha256": record["record_sha256"],
                            "rejection": None,
                        }
                    )
                    accepted_handle.write(_canonical_json_bytes(record, newline=True))
                    counts["admitted"] += 1
                    family_counts.update(record["family_histogram"])
                decision = {
                    **decision_body,
                    "decision_sha256": _canonical_sha256(decision_body),
                }
                if decision["decision_sha256"] != _decision_self_hash(decision):
                    raise SemanticTraceMaterializationError(
                        "decision self-hash construction failed"
                    )
                decision_handle.write(_canonical_json_bytes(decision, newline=True))
                counts["source"] += 1
            accepted_handle.flush()
            os.fsync(accepted_handle.fileno())
            decision_handle.flush()
            os.fsync(decision_handle.fileno())
        if observed_indices != list(range(task.source_entry_count)):
            raise SemanticTraceMaterializationError(
                "frozen source did not yield one ordered decision per declared row"
            )
        if counts["source"] != counts["admitted"] + counts["rejected"]:
            raise SemanticTraceMaterializationError(
                "migration decisions do not account for every source trace"
            )
        decision_path = staging / DECISION_FILENAME
        decision_sha256, decision_stream_sha256 = _write_decision_ledger(
            decision_path,
            decisions_scratch,
        )
        decision_binding = {
            "decision_ledger_sha256": decision_sha256,
            "source_count": counts["source"],
            "admitted_count": counts["admitted"],
            "rejected_count": counts["rejected"],
        }
        semantic_dir = staging / SEMANTIC_ARTIFACT_DIRNAME
        write_semantic_packed_artifact(
            semantic_dir,
            _iter_canonical_jsonl(accepted_scratch),
            data_lane=task.data_lane,
            split=task.split,
            source_binding=_source_binding(task),
            decision_binding=decision_binding,
        )
        semantic_shard = semantic_dir / SHARD_FILENAME
        semantic_manifest = semantic_dir / MANIFEST_FILENAME
        receipt_body: dict[str, Any] = {
            "schema": MATERIALIZATION_SCHEMA,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": MATERIALIZATION_STATUS,
            "training_authorized": False,
            "implementation_revision": task.implementation_revision,
            "source_binding": _source_binding(task),
            "data_lane": task.data_lane,
            "split": task.split,
            "process_identity": editing_v2_process_identity(),
            "builder_identity": semantic_packed_builder_identity(),
            "decision_ledger_sha256": decision_sha256,
            "decision_stream_sha256": decision_stream_sha256,
            "semantic_shard_sha256": _file_sha256(semantic_shard),
            "semantic_manifest_sha256": _file_sha256(semantic_manifest),
            "counts": {
                "source": counts["source"],
                "admitted": counts["admitted"],
                "rejected": counts["rejected"],
            },
            "rejections_by_code": dict(sorted(rejection_counts.items())),
            "accepted_family_histogram": dict(sorted(family_counts.items())),
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": _canonical_sha256(receipt_body),
        }
        (staging / RECEIPT_FILENAME).write_bytes(
            _canonical_json_bytes(receipt, newline=True)
        )
        accepted_scratch.unlink()
        decisions_scratch.unlink()
        validate_semantic_trace_materialization(
            staging,
            expected_receipt=receipt,
        )
        if output.exists():
            return validate_semantic_trace_materialization(
                output,
                expected_receipt=receipt,
            )
        try:
            os.rename(staging, output)
        except OSError:
            if not output.exists():
                raise
            return validate_semantic_trace_materialization(
                output,
                expected_receipt=receipt,
            )
        return receipt
    except (SemanticPackedStoreError, ValueError) as error:
        raise SemanticTraceMaterializationError(str(error)) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "DECISION_FILENAME",
    "MATERIALIZATION_SCHEMA",
    "MATERIALIZATION_SCHEMA_VERSION",
    "MATERIALIZATION_STATUS",
    "RECEIPT_FILENAME",
    "SEMANTIC_ARTIFACT_DIRNAME",
    "SemanticTraceMaterializationError",
    "SemanticTraceMigrationTask",
    "materialize_frozen_packed_shard",
    "validate_semantic_trace_materialization",
]
