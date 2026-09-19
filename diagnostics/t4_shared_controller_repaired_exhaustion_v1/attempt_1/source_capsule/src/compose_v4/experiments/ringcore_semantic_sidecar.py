"""Freeze exact RingCore validation semantic-cell labels by packed address.

The sidecar is deliberately independent of checkpoint scores and successor
support.  It contains one row for every progress position of every supplied
immutable validation ``PathRecord``:

``(packed_shard_content_sha256, entry_index, progress_index) -> cell | null``.

Nonterminal cells are produced only by ``ringcore_semantic_axes``.  Terminal
rows are retained with a null cell.  The compressed artifact, its source
census, labeler contract, external corpus provenance, and semantic census are
all hash-bound.  This module cannot authorize training.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.experiments import ringcore_semantic_axes
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_data import (
    SuccessorSemanticCellKey,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.ringcore_semantic_axes import (
    BoundedSemanticCellCensus,
    SemanticAxisAssignment,
    context_from_path_record,
    label_validation_semantic_axes,
    semantic_axis_labeler_contract_sha256,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key

SEMANTIC_SIDECAR_SCHEMA = "compose.ringcore.validation_semantic_sidecar"
SEMANTIC_SIDECAR_SCHEMA_VERSION = 1
SEMANTIC_SIDECAR_MANIFEST_SCHEMA = "compose.ringcore.validation_semantic_sidecar_manifest"
SEMANTIC_SIDECAR_MANIFEST_VERSION = 1
SEMANTIC_SIDECAR_STATUS = "FROZEN_VALIDATION_SEMANTIC_SIDECAR_ONLY"
SEMANTIC_SIDECAR_PARTITION = "validation"
SEMANTIC_SIDECAR_COMPRESSION = "gzip_mtime_zero_level_9"
SEMANTIC_SIDECAR_ROW_ENCODING = "canonical_jsonl_utf8_v1"

_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "partition",
    "config",
    "config_sha256",
    "provenance",
    "provenance_sha256",
    "source_sha256",
    "semantic_census_sha256",
    "semantic_census_counts",
    "counts",
    "sidecar",
    "manifest_sha256",
}
_SIDECAR_FIELDS = {
    "schema",
    "schema_version",
    "row_encoding",
    "compression",
    "content_sha256",
    "file_sha256",
    "file_bytes",
}
_COUNT_FIELDS = {
    "traces",
    "rows",
    "terminal_rows",
    "nonterminal_rows",
    "nonempty_cells",
    "teacher_family_rows",
}
_CENSUS_COUNT_FIELDS = {
    "rows",
    "terminal_rows",
    "nonterminal_rows",
    "nonempty_cells",
}
_ROW_FIELDS = (
    "packed_shard_content_sha256",
    "entry_index",
    "progress_index",
    "semantic_cell_id",
)


class SemanticSidecarError(RuntimeError):
    """The semantic sidecar is incomplete, off-contract, or tampered with."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticSidecarError(
            "semantic-sidecar metadata is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def semantic_axis_labeler_source_sha256() -> str:
    """Hash the exact implementation source used to label the sidecar."""

    source = Path(ringcore_semantic_axes.__file__)
    if not source.is_file():
        raise SemanticSidecarError("semantic-axis labeler source is absent")
    return _bytes_sha256(source.read_bytes())


@dataclass(frozen=True)
class SemanticSidecarSourceShard:
    """Frozen expected census for one immutable packed validation shard."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    layer: str
    partition: str
    packed_manifest_sha256: str
    provenance_overlay_sha256: str
    packed_entry_count: int
    effective_trace_count: int
    progress_row_count: int
    excluded_entry_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "packed_shard_content_sha256",
            "packed_manifest_sha256",
            "provenance_overlay_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one nonempty basename")
        if not isinstance(self.layer, str) or not self.layer:
            raise ValueError("source-shard layer must be nonempty text")
        if self.partition != SEMANTIC_SIDECAR_PARTITION:
            raise ValueError("semantic sidecars accept validation shards only")
        for name in (
            "packed_entry_count",
            "effective_trace_count",
            "progress_row_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        excluded = self.excluded_entry_indices
        if (
            not isinstance(excluded, tuple)
            or excluded != tuple(sorted(set(excluded)))
            or any(
                type(index) is not int or not 0 <= index < self.packed_entry_count
                for index in excluded
            )
        ):
            raise ValueError("excluded_entry_indices must be sorted unique in-range integers")
        if self.effective_trace_count != (self.packed_entry_count - len(excluded)):
            raise ValueError("effective_trace_count disagrees with packed entries and exclusions")
        if self.progress_row_count < self.effective_trace_count:
            raise ValueError("progress_row_count cannot be smaller than the trace count")

    @property
    def key(self) -> tuple[str, str]:
        return (
            self.packed_shard_content_sha256,
            self.packed_shard_name,
        )


@dataclass(frozen=True)
class SemanticSidecarProvenance:
    """External immutable identities required before labeling begins."""

    source_census_artifact_sha256: str
    unified_packed_manifest_sha256: str
    representability_overlay_sha256: str
    labeler_source_sha256: str
    source_shards: tuple[SemanticSidecarSourceShard, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_census_artifact_sha256",
            "unified_packed_manifest_sha256",
            "representability_overlay_sha256",
            "labeler_source_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not isinstance(self.source_shards, tuple) or not self.source_shards:
            raise ValueError("source_shards must be a nonempty tuple")
        if any(not isinstance(shard, SemanticSidecarSourceShard) for shard in self.source_shards):
            raise TypeError("source_shards must contain SemanticSidecarSourceShard values")
        ordered = tuple(sorted(self.source_shards, key=lambda shard: shard.key))
        if len({shard.key for shard in ordered}) != len(ordered):
            raise ValueError("source_shards contains duplicate immutable shards")
        object.__setattr__(self, "source_shards", ordered)

    def payload(self) -> dict[str, Any]:
        return {
            "source_census_artifact_sha256": (self.source_census_artifact_sha256),
            "unified_packed_manifest_sha256": (self.unified_packed_manifest_sha256),
            "representability_overlay_sha256": (self.representability_overlay_sha256),
            "labeler_source_sha256": self.labeler_source_sha256,
            "source_shards": [asdict(shard) for shard in self.source_shards],
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.payload())


@dataclass(frozen=True)
class SemanticSidecarConfig:
    """Frozen resource bound and expected semantic-census identity."""

    maximum_nonempty_cells: int
    expected_semantic_census_sha256: str

    def __post_init__(self) -> None:
        if type(self.maximum_nonempty_cells) is not int or self.maximum_nonempty_cells <= 0:
            raise ValueError("maximum_nonempty_cells must be a positive resource bound")
        if not _is_sha256(self.expected_semantic_census_sha256):
            raise ValueError("expected_semantic_census_sha256 must be a lowercase SHA-256")

    def payload(self) -> dict[str, Any]:
        return {
            "partition": SEMANTIC_SIDECAR_PARTITION,
            "labeler_contract_sha256": (semantic_axis_labeler_contract_sha256()),
            "maximum_nonempty_cells": self.maximum_nonempty_cells,
            "expected_semantic_census_sha256": (self.expected_semantic_census_sha256),
            "row_encoding": SEMANTIC_SIDECAR_ROW_ENCODING,
            "compression": SEMANTIC_SIDECAR_COMPRESSION,
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.payload())


@dataclass(frozen=True)
class SemanticSidecarRow:
    """One exact progress-row assignment."""

    packed_shard_content_sha256: str
    entry_index: int
    progress_index: int
    semantic_cell_id: str | None

    def __post_init__(self) -> None:
        if not _is_sha256(self.packed_shard_content_sha256):
            raise ValueError("packed_shard_content_sha256 must be a lowercase SHA-256")
        if type(self.entry_index) is not int or self.entry_index < 0:
            raise ValueError("entry_index must be a nonnegative integer")
        if type(self.progress_index) is not int or self.progress_index < 0:
            raise ValueError("progress_index must be a nonnegative integer")
        if self.semantic_cell_id is not None and (
            not isinstance(self.semantic_cell_id, str) or not self.semantic_cell_id
        ):
            raise ValueError("semantic_cell_id must be null or nonempty text")

    @property
    def exact_key(self) -> SuccessorSemanticCellKey:
        return (
            self.packed_shard_content_sha256,
            self.entry_index,
            self.progress_index,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "packed_shard_content_sha256": (self.packed_shard_content_sha256),
            "entry_index": self.entry_index,
            "progress_index": self.progress_index,
            "semantic_cell_id": self.semantic_cell_id,
        }


def _row_sort_key(
    row: SemanticSidecarRow,
) -> SuccessorSemanticCellKey:
    return row.exact_key


def _sidecar_content_bytes(rows: tuple[SemanticSidecarRow, ...]) -> bytes:
    return b"".join(_canonical_json_bytes(row.payload()) + b"\n" for row in rows)


def _deterministic_gzip(content: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(content)
    return output.getvalue()


def _counts_payload(
    *,
    trace_count: int,
    rows: tuple[SemanticSidecarRow, ...],
    teacher_family_rows: Mapping[str, int],
) -> dict[str, Any]:
    terminal_rows = sum(row.semantic_cell_id is None for row in rows)
    cells = {row.semantic_cell_id for row in rows if row.semantic_cell_id is not None}
    return {
        "traces": trace_count,
        "rows": len(rows),
        "terminal_rows": terminal_rows,
        "nonterminal_rows": len(rows) - terminal_rows,
        "nonempty_cells": len(cells),
        "teacher_family_rows": dict(sorted(teacher_family_rows.items())),
    }


def _manifest_body(
    *,
    config: SemanticSidecarConfig,
    provenance: SemanticSidecarProvenance,
    source_sha256: str,
    semantic_census: Mapping[str, Any],
    trace_count: int,
    rows: tuple[SemanticSidecarRow, ...],
    teacher_family_rows: Mapping[str, int],
    compressed: bytes,
) -> dict[str, Any]:
    content = _sidecar_content_bytes(rows)
    return {
        "schema": SEMANTIC_SIDECAR_MANIFEST_SCHEMA,
        "schema_version": SEMANTIC_SIDECAR_MANIFEST_VERSION,
        "status": SEMANTIC_SIDECAR_STATUS,
        "training_authorized": False,
        "partition": SEMANTIC_SIDECAR_PARTITION,
        "config": config.payload(),
        "config_sha256": config.sha256,
        "provenance": provenance.payload(),
        "provenance_sha256": provenance.sha256,
        "source_sha256": source_sha256,
        "semantic_census_sha256": semantic_census["census_sha256"],
        "semantic_census_counts": semantic_census["counts"],
        "counts": _counts_payload(
            trace_count=trace_count,
            rows=rows,
            teacher_family_rows=teacher_family_rows,
        ),
        "sidecar": {
            "schema": SEMANTIC_SIDECAR_SCHEMA,
            "schema_version": SEMANTIC_SIDECAR_SCHEMA_VERSION,
            "row_encoding": SEMANTIC_SIDECAR_ROW_ENCODING,
            "compression": SEMANTIC_SIDECAR_COMPRESSION,
            "content_sha256": _bytes_sha256(content),
            "file_sha256": _bytes_sha256(compressed),
            "file_bytes": len(compressed),
        },
    }


@dataclass(frozen=True)
class SemanticSidecarArtifact:
    """In-memory result ready for an atomic, no-overwrite freeze."""

    config: SemanticSidecarConfig
    provenance: SemanticSidecarProvenance
    source_sha256: str
    semantic_census: Mapping[str, Any]
    trace_count: int
    rows: tuple[SemanticSidecarRow, ...]
    teacher_family_rows: Mapping[str, int]
    compressed: bytes
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        if type(self.trace_count) is not int or self.trace_count <= 0 or not self.rows:
            raise ValueError("semantic sidecar cannot be empty")
        if self.rows != tuple(sorted(self.rows, key=_row_sort_key)):
            raise ValueError("semantic sidecar rows are not in canonical order")
        keys = tuple(row.exact_key for row in self.rows)
        if len(keys) != len(set(keys)):
            raise ValueError("semantic sidecar contains duplicate exact rows")
        census = dict(self.semantic_census)
        if census.get("census_sha256") != self.config.expected_semantic_census_sha256:
            raise ValueError("semantic census disagrees with the frozen expectation")
        census_counts = census.get("counts")
        if not isinstance(census_counts, Mapping):
            raise ValueError("semantic census lacks its exact row counts")
        terminal_rows = sum(row.semantic_cell_id is None for row in self.rows)
        nonempty_cells = len(
            {row.semantic_cell_id for row in self.rows if row.semantic_cell_id is not None}
        )
        if (
            census_counts.get("rows") != len(self.rows)
            or census_counts.get("terminal_rows") != terminal_rows
            or census_counts.get("nonterminal_rows") != len(self.rows) - terminal_rows
            or census_counts.get("nonempty_cells") != nonempty_cells
        ):
            raise ValueError("semantic census counts disagree with sidecar rows")
        if semantic_axis_labeler_source_sha256() != (self.provenance.labeler_source_sha256):
            raise ValueError("semantic-axis labeler source changed")
        family_rows = dict(sorted(self.teacher_family_rows.items()))
        if any(
            not isinstance(family, str) or not family or type(count) is not int or count <= 0
            for family, count in family_rows.items()
        ):
            raise ValueError("teacher-family row census is malformed")
        object.__setattr__(
            self,
            "semantic_census",
            MappingProxyType(census),
        )
        object.__setattr__(
            self,
            "teacher_family_rows",
            MappingProxyType(family_rows),
        )
        content = _sidecar_content_bytes(self.rows)
        if _deterministic_gzip(content) != self.compressed:
            raise ValueError("compressed sidecar is not the canonical deterministic encoding")
        body = _manifest_body(
            config=self.config,
            provenance=self.provenance,
            source_sha256=self.source_sha256,
            semantic_census=self.semantic_census,
            trace_count=self.trace_count,
            rows=self.rows,
            teacher_family_rows=self.teacher_family_rows,
            compressed=self.compressed,
        )
        if self.manifest_sha256 != _stable_sha256(body):
            raise ValueError("manifest hash disagrees with exact artifact contents")

    @property
    def training_authorized(self) -> bool:
        return False

    def assert_training_authorized(self) -> None:
        raise SemanticSidecarError(
            "a semantic-cell sidecar labels validation rows and never authorizes training"
        )

    @property
    def semantic_cell_ids(
        self,
    ) -> Mapping[SuccessorSemanticCellKey, str | None]:
        return MappingProxyType({row.exact_key: row.semantic_cell_id for row in self.rows})

    def manifest(self) -> dict[str, Any]:
        body = _manifest_body(
            config=self.config,
            provenance=self.provenance,
            source_sha256=self.source_sha256,
            semantic_census=self.semantic_census,
            trace_count=self.trace_count,
            rows=self.rows,
            teacher_family_rows=self.teacher_family_rows,
            compressed=self.compressed,
        )
        return {**body, "manifest_sha256": self.manifest_sha256}


def _address_payload(record: PathRecord) -> dict[str, Any]:
    address = record.corpus_address
    if address is None:
        raise SemanticSidecarError("semantic sidecars require immutable packed addresses")
    return asdict(address)


def _label_trace_rows(
    record: PathRecord,
) -> tuple[
    tuple[tuple[SemanticSidecarRow, SemanticAxisAssignment], ...],
    dict[str, Any],
    Counter[str],
]:
    address = record.corpus_address
    if address is None:
        raise SemanticSidecarError("semantic sidecars require immutable packed addresses")
    if address.partition != SEMANTIC_SIDECAR_PARTITION:
        raise SemanticSidecarError("semantic sidecars accept validation PathRecords only")
    path = record.path
    if path.path_length != address.path_length or len(path.trace.steps) != address.path_length:
        raise SemanticSidecarError("packed address and trace path lengths disagree")
    if record.target_key != address.target_key:
        raise SemanticSidecarError("PathRecord target key disagrees with its immutable address")
    if persistent_slot_state_sha256(path.trace.source) != persistent_slot_state_sha256(
        path.state_at(0)
    ) or persistent_slot_state_sha256(path.trace.target) != persistent_slot_state_sha256(
        path.state_at(path.path_length)
    ):
        raise SemanticSidecarError("trace endpoints disagree with stored progress states")
    try:
        source_key = canonical_state_key(path.trace.source)
        target_key = canonical_state_key(path.trace.target)
    except Exception as error:
        raise SemanticSidecarError("trace endpoint is not a canonical production state") from error
    if source_key != address.source_key or target_key != address.target_key:
        raise SemanticSidecarError("trace endpoint identity disagrees with its immutable address")

    assignments_by_rule: dict[str, SemanticAxisAssignment] = {}
    family_rows: Counter[str] = Counter()
    teacher_actions: list[dict[str, str]] = []
    for progress_index, step in enumerate(path.trace.steps):
        rule_name = str(step.rule_name)
        family = canonical_family(rule_name)
        family_rows[family] += 1
        teacher_actions.append(
            {
                "rule_name": rule_name,
                "family_name": family,
                "action_sha256": rewrite_action_codec_sha256(
                    rule_name,
                    step.action,
                ),
            }
        )
        if rule_name not in assignments_by_rule:
            assignments_by_rule[rule_name] = label_validation_semantic_axes(
                context_from_path_record(
                    record,
                    progress_index=progress_index,
                )
            )
    terminal_assignment = label_validation_semantic_axes(
        context_from_path_record(
            record,
            progress_index=path.path_length,
        )
    )

    labeled: list[tuple[SemanticSidecarRow, SemanticAxisAssignment]] = []
    for progress_index in range(path.path_length + 1):
        if progress_index == path.path_length:
            assignment = terminal_assignment
        else:
            assignment = assignments_by_rule[path.trace.steps[progress_index].rule_name]
        labeled.append(
            (
                SemanticSidecarRow(
                    packed_shard_content_sha256=(address.packed_shard_content_sha256),
                    entry_index=address.entry_index,
                    progress_index=progress_index,
                    semantic_cell_id=assignment.semantic_cell_id,
                ),
                assignment,
            )
        )
    source_descriptor = {
        "address": _address_payload(record),
        "source_state_sha256": persistent_slot_state_sha256(path.trace.source),
        "target_state_sha256": persistent_slot_state_sha256(path.trace.target),
        "teacher_actions": teacher_actions,
    }
    return tuple(labeled), source_descriptor, family_rows


def build_semantic_cell_sidecar(
    records: Iterable[PathRecord],
    *,
    config: SemanticSidecarConfig,
    provenance: SemanticSidecarProvenance,
) -> SemanticSidecarArtifact:
    """Label and hash a complete immutable validation corpus."""

    if not isinstance(config, SemanticSidecarConfig):
        raise TypeError("config must be SemanticSidecarConfig")
    if not isinstance(provenance, SemanticSidecarProvenance):
        raise TypeError("provenance must be SemanticSidecarProvenance")
    if provenance.labeler_source_sha256 != semantic_axis_labeler_source_sha256():
        raise SemanticSidecarError("frozen provenance does not bind the current labeler source")

    shard_specs = {shard.key: shard for shard in provenance.source_shards}
    observed_indices: dict[tuple[str, str], set[int]] = {key: set() for key in shard_specs}
    observed_rows: Counter[tuple[str, str]] = Counter()
    rows: list[SemanticSidecarRow] = []
    source_descriptors: list[dict[str, Any]] = []
    teacher_family_rows: Counter[str] = Counter()
    census = BoundedSemanticCellCensus(maximum_nonempty_cells=config.maximum_nonempty_cells)
    trace_count = 0

    for record in records:
        if not isinstance(record, PathRecord):
            raise TypeError("records must contain PathRecord values only")
        address = record.corpus_address
        if address is None:
            raise SemanticSidecarError("semantic sidecars require immutable packed addresses")
        shard_key = (
            address.packed_shard_content_sha256,
            address.packed_shard_name,
        )
        try:
            shard = shard_specs[shard_key]
        except KeyError:
            raise SemanticSidecarError("PathRecord belongs to an undeclared packed shard") from None
        if address.layer != shard.layer or address.partition != shard.partition:
            raise SemanticSidecarError("PathRecord address disagrees with its declared shard lane")
        if address.entry_index in observed_indices[shard_key]:
            raise SemanticSidecarError("semantic sidecar source repeats a packed entry")
        observed_indices[shard_key].add(address.entry_index)
        labeled, descriptor, trace_family_rows = _label_trace_rows(record)
        for row, assignment in labeled:
            rows.append(row)
            census.add(assignment)
        observed_rows[shard_key] += len(labeled)
        source_descriptors.append(descriptor)
        teacher_family_rows.update(trace_family_rows)
        trace_count += 1

    if trace_count == 0:
        raise SemanticSidecarError("semantic sidecar source cannot be empty")
    for key, shard in shard_specs.items():
        expected_indices = set(range(shard.packed_entry_count)) - set(shard.excluded_entry_indices)
        if observed_indices[key] != expected_indices:
            missing = sorted(expected_indices - observed_indices[key])
            unexpected = sorted(observed_indices[key] - expected_indices)
            raise SemanticSidecarError(
                "source does not contain the complete declared shard census; "
                f"missing={missing[:20]!r}, unexpected={unexpected[:20]!r}"
            )
        if len(observed_indices[key]) != shard.effective_trace_count:
            raise SemanticSidecarError("source trace count disagrees with declared shard census")
        if observed_rows[key] != shard.progress_row_count:
            raise SemanticSidecarError(
                "source progress-row count disagrees with declared shard census"
            )

    ordered_rows = tuple(sorted(rows, key=_row_sort_key))
    source_sha256 = _stable_sha256(
        sorted(
            source_descriptors,
            key=lambda item: (
                item["address"]["packed_shard_content_sha256"],
                item["address"]["packed_shard_name"],
                item["address"]["entry_index"],
            ),
        )
    )
    semantic_census = census.payload()
    if semantic_census["census_sha256"] != config.expected_semantic_census_sha256:
        raise SemanticSidecarError("computed semantic census disagrees with the frozen expectation")
    content = _sidecar_content_bytes(ordered_rows)
    compressed = _deterministic_gzip(content)
    body = _manifest_body(
        config=config,
        provenance=provenance,
        source_sha256=source_sha256,
        semantic_census=semantic_census,
        trace_count=trace_count,
        rows=ordered_rows,
        teacher_family_rows=teacher_family_rows,
        compressed=compressed,
    )
    return SemanticSidecarArtifact(
        config=config,
        provenance=provenance,
        source_sha256=source_sha256,
        semantic_census=semantic_census,
        trace_count=trace_count,
        rows=ordered_rows,
        teacher_family_rows=dict(teacher_family_rows),
        compressed=compressed,
        manifest_sha256=_stable_sha256(body),
    )


def _atomic_write_if_absent(path: Path, content: bytes) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            return True
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    "semantic-sidecar artifact already exists with different "
                    f"content: {destination}"
                ) from None
            return False
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_semantic_cell_sidecar(
    sidecar_path: Path,
    manifest_path: Path,
    artifact: SemanticSidecarArtifact,
) -> None:
    """Atomically freeze deterministic sidecar and manifest without overwrite."""

    if not isinstance(artifact, SemanticSidecarArtifact):
        raise TypeError("artifact must be SemanticSidecarArtifact")
    sidecar = Path(sidecar_path)
    manifest = Path(manifest_path)
    if sidecar.resolve() == manifest.resolve():
        raise ValueError("sidecar and manifest paths must differ")
    manifest_bytes = _canonical_json_bytes(artifact.manifest()) + b"\n"
    sidecar_created = _atomic_write_if_absent(
        sidecar,
        artifact.compressed,
    )
    try:
        _atomic_write_if_absent(manifest, manifest_bytes)
    except Exception:
        if sidecar_created:
            sidecar.unlink()
        raise


@dataclass(frozen=True)
class LoadedSemanticCellSidecar:
    """Validated durable sidecar projection."""

    rows: tuple[SemanticSidecarRow, ...]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(dict(self.manifest)),
        )

    @property
    def training_authorized(self) -> bool:
        return False

    def assert_training_authorized(self) -> None:
        raise SemanticSidecarError(
            "a semantic-cell sidecar labels validation rows and never authorizes training"
        )

    @property
    def semantic_cell_ids(
        self,
    ) -> Mapping[SuccessorSemanticCellKey, str | None]:
        return MappingProxyType({row.exact_key: row.semantic_cell_id for row in self.rows})


def _require_exact_fields(
    payload: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SemanticSidecarError(f"{name} must be an object")
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        raise SemanticSidecarError(
            f"{name} fields disagree; missing={missing!r}, unexpected={unexpected!r}"
        )
    return payload


def read_semantic_cell_sidecar(
    sidecar_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
    expected_provenance_sha256: str,
    max_file_bytes: int = 1 << 30,
    max_content_bytes: int = 1 << 29,
    max_rows: int = 1_000_000,
) -> LoadedSemanticCellSidecar:
    """Load only after exact artifact, config, and provenance verification."""

    for name, value in (
        ("expected_manifest_sha256", expected_manifest_sha256),
        ("expected_config_sha256", expected_config_sha256),
        ("expected_provenance_sha256", expected_provenance_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if type(max_file_bytes) is not int or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    if type(max_content_bytes) is not int or max_content_bytes <= 0:
        raise ValueError("max_content_bytes must be positive")
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("max_rows must be positive")
    sidecar_file = Path(sidecar_path)
    manifest_file = Path(manifest_path)
    if not sidecar_file.is_file() or not manifest_file.is_file():
        raise SemanticSidecarError("semantic sidecar or manifest is absent")
    if sidecar_file.stat().st_size > max_file_bytes:
        raise SemanticSidecarError("compressed semantic sidecar exceeds its bound")
    try:
        manifest = json.loads(manifest_file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticSidecarError("semantic-sidecar manifest is invalid JSON") from error
    manifest = _require_exact_fields(
        manifest,
        _MANIFEST_FIELDS,
        name="manifest",
    )
    recorded_manifest_sha256 = manifest["manifest_sha256"]
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    observed_manifest_sha256 = _stable_sha256(body)
    if (
        recorded_manifest_sha256 != observed_manifest_sha256
        or recorded_manifest_sha256 != expected_manifest_sha256
    ):
        raise SemanticSidecarError("semantic-sidecar manifest hash mismatch")
    if (
        manifest["schema"] != SEMANTIC_SIDECAR_MANIFEST_SCHEMA
        or manifest["schema_version"] != SEMANTIC_SIDECAR_MANIFEST_VERSION
        or manifest["status"] != SEMANTIC_SIDECAR_STATUS
        or manifest["training_authorized"] is not False
        or manifest["partition"] != SEMANTIC_SIDECAR_PARTITION
    ):
        raise SemanticSidecarError(
            "semantic-sidecar manifest is not the frozen validation contract"
        )
    if manifest["config_sha256"] != expected_config_sha256:
        raise SemanticSidecarError("semantic-sidecar config hash mismatch")
    if manifest["provenance_sha256"] != expected_provenance_sha256:
        raise SemanticSidecarError("semantic-sidecar provenance hash mismatch")
    if _stable_sha256(manifest["config"]) != manifest["config_sha256"]:
        raise SemanticSidecarError("manifest config self-hash mismatch")
    if _stable_sha256(manifest["provenance"]) != manifest["provenance_sha256"]:
        raise SemanticSidecarError("manifest provenance self-hash mismatch")
    sidecar_meta = _require_exact_fields(
        manifest["sidecar"],
        _SIDECAR_FIELDS,
        name="sidecar metadata",
    )
    counts = _require_exact_fields(
        manifest["counts"],
        _COUNT_FIELDS,
        name="sidecar counts",
    )
    census_counts = _require_exact_fields(
        manifest["semantic_census_counts"],
        _CENSUS_COUNT_FIELDS,
        name="semantic census counts",
    )
    if manifest["semantic_census_sha256"] != manifest["config"].get(
        "expected_semantic_census_sha256"
    ):
        raise SemanticSidecarError("semantic census hash disagrees with the frozen config")
    for field in _CENSUS_COUNT_FIELDS:
        if counts[field] != census_counts[field]:
            raise SemanticSidecarError("semantic census counts disagree with sidecar counts")
    family_rows = counts["teacher_family_rows"]
    if (
        not isinstance(family_rows, dict)
        or any(
            not isinstance(family, str) or not family or type(count) is not int or count <= 0
            for family, count in family_rows.items()
        )
        or sum(family_rows.values()) != counts["nonterminal_rows"]
    ):
        raise SemanticSidecarError("teacher-family row census is malformed")
    if (
        sidecar_meta["schema"] != SEMANTIC_SIDECAR_SCHEMA
        or sidecar_meta["schema_version"] != SEMANTIC_SIDECAR_SCHEMA_VERSION
        or sidecar_meta["row_encoding"] != SEMANTIC_SIDECAR_ROW_ENCODING
        or sidecar_meta["compression"] != SEMANTIC_SIDECAR_COMPRESSION
    ):
        raise SemanticSidecarError("sidecar metadata is not the frozen encoding contract")
    compressed = sidecar_file.read_bytes()
    if (
        len(compressed) != sidecar_meta["file_bytes"]
        or _bytes_sha256(compressed) != sidecar_meta["file_sha256"]
    ):
        raise SemanticSidecarError("compressed semantic sidecar hash mismatch")
    rows: list[SemanticSidecarRow] = []
    content_digest = hashlib.sha256()
    content_bytes = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
            for line_index, line in enumerate(handle):
                content_bytes += len(line)
                if content_bytes > max_content_bytes:
                    raise SemanticSidecarError(
                        "semantic sidecar exceeds its uncompressed-byte bound"
                    )
                if line_index >= max_rows:
                    raise SemanticSidecarError("semantic sidecar exceeds its row bound")
                content_digest.update(line)
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SemanticSidecarError("semantic sidecar contains invalid JSONL") from error
                payload = _require_exact_fields(
                    payload,
                    set(_ROW_FIELDS),
                    name="semantic row",
                )
                try:
                    rows.append(
                        SemanticSidecarRow(**{field: payload[field] for field in _ROW_FIELDS})
                    )
                except (TypeError, ValueError) as error:
                    raise SemanticSidecarError(
                        "semantic sidecar contains a malformed row"
                    ) from error
    except SemanticSidecarError:
        raise
    except (OSError, EOFError) as error:
        raise SemanticSidecarError("semantic sidecar is not valid gzip") from error
    if content_digest.hexdigest() != sidecar_meta["content_sha256"]:
        raise SemanticSidecarError("semantic sidecar content hash mismatch")
    ordered = tuple(rows)
    if ordered != tuple(sorted(ordered, key=_row_sort_key)):
        raise SemanticSidecarError("semantic sidecar rows are not canonical")
    keys = tuple(row.exact_key for row in ordered)
    if len(keys) != len(set(keys)):
        raise SemanticSidecarError("semantic sidecar repeats an exact row")
    terminal_rows = sum(row.semantic_cell_id is None for row in ordered)
    nonempty_cells = len(
        {row.semantic_cell_id for row in ordered if row.semantic_cell_id is not None}
    )
    if (
        len(ordered) != counts["rows"]
        or terminal_rows != counts["terminal_rows"]
        or len(ordered) - terminal_rows != counts["nonterminal_rows"]
        or nonempty_cells != counts["nonempty_cells"]
        or terminal_rows != counts["traces"]
    ):
        raise SemanticSidecarError("decoded semantic sidecar disagrees with manifest counts")
    return LoadedSemanticCellSidecar(
        rows=ordered,
        manifest=manifest,
    )


__all__ = [
    "SEMANTIC_SIDECAR_COMPRESSION",
    "SEMANTIC_SIDECAR_MANIFEST_SCHEMA",
    "SEMANTIC_SIDECAR_MANIFEST_VERSION",
    "SEMANTIC_SIDECAR_PARTITION",
    "SEMANTIC_SIDECAR_ROW_ENCODING",
    "SEMANTIC_SIDECAR_SCHEMA",
    "SEMANTIC_SIDECAR_SCHEMA_VERSION",
    "SEMANTIC_SIDECAR_STATUS",
    "LoadedSemanticCellSidecar",
    "SemanticSidecarArtifact",
    "SemanticSidecarConfig",
    "SemanticSidecarError",
    "SemanticSidecarProvenance",
    "SemanticSidecarRow",
    "SemanticSidecarSourceShard",
    "build_semantic_cell_sidecar",
    "read_semantic_cell_sidecar",
    "semantic_axis_labeler_source_sha256",
    "write_semantic_cell_sidecar",
]
