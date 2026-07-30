"""Immutable whole-trace representability inventory for the active-8 editor.

This boundary is deliberately earlier than progress-row sampling.  A packed
trace is admitted only after *every* nonterminal teacher step:

* maps to one of the eight enabled model families;
* is representable by RewriteActionCodecV2;
* satisfies the frozen structural restrictions; and
* is present, with its exact persistent-slot output, in the production marked
  law for the corresponding source state.

Only then are all ``path_length + 1`` progress positions emitted.  Therefore a
trace with one bad middle step contributes neither neighboring nonterminal
rows nor its terminal row.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from compose_v4.chem.persistent_state_identity import (
    PERSISTENT_STATE_DIGEST_SCHEMA,
    PERSISTENT_STATE_DIGEST_VERSION,
    persistent_slot_state_sha256,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    manifest_path_for,
    read_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import overlay_path_for
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    compile_state_successor_map,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
)
from compose_v4.rewrite.action_codec import ActionCodecError, canonical_family
from compose_v4.rewrite.operators import AtomInsert

ACTIVE8_TRACE_INVENTORY_SCHEMA = "compose.data.active8_trace_inventory"
ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION = 1
ACTIVE8_TRACE_DECISION_SCHEMA = "compose.data.active8_trace_decision"
ACTIVE8_TRACE_DECISION_SCHEMA_VERSION = 1

ACTIVE8_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
)
_ACTIVE8_SET = frozenset(ACTIVE8_FAMILIES)
_EXPLICITLY_DISALLOWED_RULES = frozenset(
    {
        "ring_system_delete",
        "ring_system_grow",
    }
)
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/data/active8_trace_inventory.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/chem/persistent_state_identity.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
)


class Active8TraceInventoryError(RuntimeError):
    """The requested inventory cannot establish its fail-closed boundary."""


@dataclass(frozen=True)
class Active8TraceAdmission:
    """Verified O(1) whole-trace admission index for production corpus loading.

    The tuple for each physical packed shard is position-addressed by the
    immutable packed ``entry_index``.  Each element stores the exact trace ID
    and its accepted/excluded decision, so a reordered, missing, substituted or
    previously unseen trace fails before it can become a ``PathRecord``.
    """

    manifest_path: Path
    manifest_file_sha256: str
    inventory_sha256: str
    effective_source_corpus_cache_sha256: str
    decisions_by_digest: Mapping[str, tuple[tuple[str, bool], ...]]
    shard_digest_by_lane: Mapping[tuple[str, str, str], str]
    shard_metadata_by_digest: Mapping[str, Mapping[str, Any]]
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        for field in (
            "manifest_file_sha256",
            "inventory_sha256",
            "effective_source_corpus_cache_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")

    def expected_source_digest(
        self,
        *,
        packed_shard_name: str,
        layer: str,
        partition: str,
    ) -> str:
        lane = (str(layer), str(partition), str(packed_shard_name))
        try:
            return self.shard_digest_by_lane[lane]
        except KeyError:
            raise Active8TraceInventoryError(
                "packed source shard is absent from the active-8 inventory: "
                f"layer={layer!r}, partition={partition!r}, "
                f"shard={packed_shard_name!r}"
            ) from None

    def is_accepted(self, address: object) -> bool:
        """Return the frozen trace decision after verifying its exact address."""

        digest = str(address.packed_shard_content_sha256)
        entry_index = address.entry_index
        trace_id = address.trace_id
        if type(entry_index) is not int or entry_index < 0:
            raise Active8TraceInventoryError(
                "packed trace has an invalid active-8 entry index"
            )
        try:
            decisions = self.decisions_by_digest[digest]
        except KeyError:
            raise Active8TraceInventoryError(
                "packed trace shard digest is absent from the active-8 inventory"
            ) from None
        if entry_index >= len(decisions):
            raise Active8TraceInventoryError(
                "packed trace entry lies outside its active-8 shard census"
            )
        expected_trace_id, accepted = decisions[entry_index]
        if trace_id != expected_trace_id:
            raise Active8TraceInventoryError(
                "packed trace ID disagrees with its active-8 inventory address"
            )
        return accepted

    def assert_complete_source_shard(
        self,
        *,
        packed_shard_name: str,
        layer: str,
        partition: str,
        observed_digest: str | None,
        observed_entries: int,
    ) -> None:
        """Require that a loader consumed the complete declared physical shard."""

        expected_digest = self.expected_source_digest(
            packed_shard_name=packed_shard_name,
            layer=layer,
            partition=partition,
        )
        if observed_digest != expected_digest:
            raise Active8TraceInventoryError(
                "loaded packed shard bytes disagree with the active-8 inventory: "
                f"expected={expected_digest}, observed={observed_digest}"
            )
        expected_entries = len(self.decisions_by_digest[expected_digest])
        if observed_entries != expected_entries:
            raise Active8TraceInventoryError(
                "loaded packed shard census disagrees with the active-8 inventory: "
                f"expected={expected_entries}, observed={observed_entries}"
            )

    def assert_partition_shards(
        self,
        partition: str,
        observed: Iterable[tuple[str, str, str]],
    ) -> None:
        """Require exact packed-shard coverage for one corpus partition."""

        expected = {
            lane
            for lane in self.shard_digest_by_lane
            if lane[1] == partition
        }
        actual = set(observed)
        if actual != expected:
            raise Active8TraceInventoryError(
                "packed partition shard set disagrees with the active-8 inventory: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )


@dataclass(frozen=True)
class Active8SourceShard:
    """One unified-manifest-declared packed shard."""

    manifest_layer: str
    envelope_layer: str
    partition: str
    relative_path: str
    path: Path

    def __post_init__(self) -> None:
        if not self.manifest_layer or not self.envelope_layer or not self.partition:
            raise ValueError("source shard lane fields must be non-empty")
        relative = Path(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("source shard relative_path must be safe and relative")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class ExactCandidateEvidence:
    """Result of checking one teacher against the exact production marked law."""

    supported: bool
    action_sha256: str | None
    reason: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.supported:
            if not _is_sha256(self.action_sha256):
                raise ValueError("supported candidate evidence requires an action SHA-256")
            if self.reason is not None:
                raise ValueError("supported candidate evidence cannot carry a failure reason")
        else:
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("failed candidate evidence requires a reason")
            if self.action_sha256 is not None and not _is_sha256(self.action_sha256):
                raise ValueError("candidate evidence action digest is malformed")


class ExactCandidateChecker(Protocol):
    """Check one decoded teacher step against the exact active model support."""

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ExactCandidateEvidence: ...


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity(*, repo_root: Path | None = None) -> dict[str, object]:
    """Hash the complete code surface defining admission and row identity."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    sources: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCES:
        source = root / relative
        if not source.is_file():
            raise Active8TraceInventoryError(
                f"active-8 implementation source is absent: {source}"
            )
        sources[relative] = _sha256_file(source)
    return {
        "sources": sources,
        "implementation_sha256": _sha256_bytes(sources),
    }


class ProductionExactCandidateChecker:
    """Exact, checkpoint-independent support check using the live marked law.

    Model weights do not define support, but the model's immutable operator
    capabilities do.  A bounded LRU avoids recompiling frequently repeated
    exact source states without retaining the entire multi-million-row corpus.
    """

    def __init__(
        self,
        model: object,
        *,
        cache_size: int = 4096,
        time: float = 0.5,
        system: object | None = None,
    ) -> None:
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("exact-candidate cache_size must be positive")
        if not 0.0 < float(time) < 1.0:
            raise ValueError("exact-candidate enumeration time must lie in (0, 1)")
        self.model = model
        self.cache_size = cache_size
        self.time = float(time)
        self.system = system
        self._compiled: OrderedDict[str, object] = OrderedDict()

    def _compiled_state(self, state: object) -> object:
        state_sha256 = persistent_slot_state_sha256(state)
        cached = self._compiled.get(state_sha256)
        if cached is not None:
            self._compiled.move_to_end(state_sha256)
            return cached
        compiled = compile_state_successor_map(
            self.model,
            state,
            time=self.time,
            system=self.system,
        )
        self._compiled[state_sha256] = compiled
        self._compiled.move_to_end(state_sha256)
        while len(self._compiled) > self.cache_size:
            self._compiled.popitem(last=False)
        return compiled

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ExactCandidateEvidence:
        step = addressed.trace.steps[step_index]
        action_sha256: str | None = None
        try:
            action_sha256 = rewrite_action_codec_sha256(
                step.rule_name,
                step.action,
            )
            compiled = self._compiled_state(addressed.path.state_at(step_index))
            require_exact_successor_action_identity(
                compiled,
                target_state_sha256=persistent_slot_state_sha256(
                    addressed.path.state_at(step_index + 1)
                ),
                action_sha256=action_sha256,
            )
        except SuccessorTrainingError as error:
            return ExactCandidateEvidence(
                supported=False,
                action_sha256=action_sha256,
                reason="teacher_not_in_exact_candidates",
                detail=type(error).__name__,
            )
        except Exception as error:  # noqa: BLE001 -- any support defect excludes
            return ExactCandidateEvidence(
                supported=False,
                action_sha256=action_sha256,
                reason="unsupported_teacher_step",
                detail=type(error).__name__,
            )
        return ExactCandidateEvidence(
            supported=True,
            action_sha256=action_sha256,
        )


def _trace_key(addressed: AddressedPackedTrace) -> dict[str, object]:
    address = addressed.address
    return {
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "entry_index": address.entry_index,
        "trace_id": address.trace_id,
    }


def _base_decision(addressed: AddressedPackedTrace) -> dict[str, object]:
    address = addressed.address
    if address.path_length != addressed.path.path_length:
        raise Active8TraceInventoryError(
            "packed address and progress object disagree on path_length"
        )
    if len(addressed.trace.steps) != address.path_length:
        raise Active8TraceInventoryError(
            "packed trace and immutable address disagree on path_length"
        )
    return {
        "schema": ACTIVE8_TRACE_DECISION_SCHEMA,
        "schema_version": ACTIVE8_TRACE_DECISION_SCHEMA_VERSION,
        "trace_key": _trace_key(addressed),
        "packed_shard_name": address.packed_shard_name,
        "layer": address.layer,
        "partition": address.partition,
        "source_key": address.source_key,
        "target_key": address.target_key,
        "path_length": address.path_length,
    }


def _family_exclusions(addressed: AddressedPackedTrace) -> list[dict[str, object]]:
    exclusions: list[dict[str, object]] = []
    for step_index, step in enumerate(addressed.trace.steps):
        if step.rule_name in _EXPLICITLY_DISALLOWED_RULES:
            exclusions.append(
                {
                    "step_index": step_index,
                    "executor_rule": step.rule_name,
                    "family": step.rule_name,
                    "reason": "disallowed_family",
                }
            )
            continue
        try:
            family = canonical_family(step.rule_name)
        except ActionCodecError:
            exclusions.append(
                {
                    "step_index": step_index,
                    "executor_rule": step.rule_name,
                    "family": None,
                    "reason": "unknown_or_legacy_executor_rule",
                }
            )
            continue
        if family not in _ACTIVE8_SET:
            exclusions.append(
                {
                    "step_index": step_index,
                    "executor_rule": step.rule_name,
                    "family": family,
                    "reason": "disallowed_family",
                }
            )
            continue
        if (
            step.rule_name == "atom_insert"
            and isinstance(step.action, AtomInsert)
            and len(step.action.neighbors) > 1
        ):
            exclusions.append(
                {
                    "step_index": step_index,
                    "executor_rule": step.rule_name,
                    "family": family,
                    "reason": "unsupported_multi_neighbor_atom_insert",
                }
            )
            continue
        try:
            rewrite_action_codec_sha256(step.rule_name, step.action)
        except SuccessorTrainingError:
            exclusions.append(
                {
                    "step_index": step_index,
                    "executor_rule": step.rule_name,
                    "family": family,
                    "reason": "unsupported_action_encoding",
                }
            )
    return exclusions


def inventory_record_for_trace(
    addressed: AddressedPackedTrace,
    *,
    exact_candidate_checker: ExactCandidateChecker,
) -> dict[str, object]:
    """Return one accepted/excluded decision; excluded records contain no rows."""

    base = _base_decision(addressed)
    exclusions = _family_exclusions(addressed)
    evidence: dict[int, ExactCandidateEvidence] = {}
    if not exclusions:
        for step_index, step in enumerate(addressed.trace.steps):
            try:
                result = exact_candidate_checker(addressed, step_index)
            except Exception as error:  # noqa: BLE001 -- fail closed on checker defects
                result = ExactCandidateEvidence(
                    supported=False,
                    action_sha256=None,
                    reason="exact_candidate_checker_failure",
                    detail=type(error).__name__,
                )
            if not isinstance(result, ExactCandidateEvidence):
                raise Active8TraceInventoryError(
                    "exact candidate checker returned another result type"
                )
            if not result.supported:
                exclusions.append(
                    {
                        "step_index": step_index,
                        "executor_rule": step.rule_name,
                        "family": canonical_family(step.rule_name),
                        "reason": result.reason,
                        "detail": result.detail,
                    }
                )
            else:
                expected = rewrite_action_codec_sha256(
                    step.rule_name,
                    step.action,
                )
                if result.action_sha256 != expected:
                    exclusions.append(
                        {
                            "step_index": step_index,
                            "executor_rule": step.rule_name,
                            "family": canonical_family(step.rule_name),
                            "reason": "candidate_evidence_action_identity_mismatch",
                        }
                    )
                else:
                    evidence[step_index] = result
    if exclusions:
        return {
            **base,
            "decision": "excluded",
            # Crucial whole-trace invariant: an excluded trace never contains
            # progress rows, including otherwise-valid prefix/suffix states.
            "progress_rows": [],
            "exclusions": sorted(
                exclusions,
                key=lambda item: (
                    int(item["step_index"]),
                    str(item["reason"]),
                ),
            ),
        }

    progress_rows = []
    path_length = addressed.address.path_length
    for progress_index in range(path_length + 1):
        is_terminal = progress_index == path_length
        progress_rows.append(
            {
                "progress_index": progress_index,
                "is_terminal": is_terminal,
                "exact_state_sha256": persistent_slot_state_sha256(
                    addressed.path.state_at(progress_index)
                ),
                "teacher_family": (
                    None
                    if is_terminal
                    else canonical_family(
                        addressed.trace.steps[progress_index].rule_name
                    )
                ),
                "teacher_action_sha256": (
                    None
                    if is_terminal
                    else evidence[progress_index].action_sha256
                ),
            }
        )
    return {
        **base,
        "decision": "accepted",
        "progress_rows": progress_rows,
        "exclusions": [],
    }


def _decision_records_for_source(
    source: Active8SourceShard,
    *,
    source_bytes_sha256: str,
    exact_candidate_checker: ExactCandidateChecker,
    current: Counter,
    current_reasons: Counter[str],
    current_families: Counter[str],
    observed_digest: list[str | None],
) -> Iterable[dict[str, object]]:
    """Stream one source shard while accumulating its exact decision census."""

    for addressed in read_addressed_packed_shard(
        source.path,
        verify_fraction=0.0,
    ):
        address = addressed.address
        if (
            address.layer != source.envelope_layer
            or address.partition != source.partition
        ):
            raise Active8TraceInventoryError(
                "packed trace envelope disagrees with unified-manifest lane"
            )
        if observed_digest[0] is None:
            observed_digest[0] = address.packed_shard_content_sha256
            if observed_digest[0] != source_bytes_sha256:
                raise Active8TraceInventoryError(
                    "addressed physical shard digest differs from exact file bytes"
                )
        elif address.packed_shard_content_sha256 != observed_digest[0]:
            raise Active8TraceInventoryError(
                "one source shard yielded more than one physical digest"
            )
        record = inventory_record_for_trace(
            addressed,
            exact_candidate_checker=exact_candidate_checker,
        )
        current["traces"] += 1
        if record["decision"] == "accepted":
            current["accepted_traces"] += 1
            rows = record["progress_rows"]
            current["accepted_progress_rows"] += len(rows)
            current["accepted_nonterminal_rows"] += address.path_length
            current["accepted_terminal_rows"] += 1
            for row in rows:
                family = row["teacher_family"]
                if family is not None:
                    current_families[str(family)] += 1
        else:
            current["excluded_traces"] += 1
            for exclusion in record["exclusions"]:
                current_reasons[str(exclusion["reason"])] += 1
        yield record


def _atomic_deterministic_gzip_jsonl(
    path: Path,
    records: Iterable[dict[str, object]],
) -> tuple[int, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Active8TraceInventoryError(
            f"refusing to overwrite immutable inventory shard: {path}"
        )
    temporary_name: str | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary_name = raw.name
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                for record in records:
                    compressed.write(_canonical_json_bytes(record) + b"\n")
                    count += 1
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise Active8TraceInventoryError(
                f"immutable inventory shard collision: {path}"
            ) from error
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return count, _sha256_file(path)


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Active8TraceInventoryError(
            f"refusing to overwrite immutable inventory manifest: {path}"
        )
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise Active8TraceInventoryError(
                f"immutable inventory manifest collision: {path}"
            ) from error
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_active8_trace_inventory(
    shards: Iterable[Active8SourceShard],
    *,
    exact_candidate_checker: ExactCandidateChecker,
    source_manifest_path: Path,
    source_manifest: Mapping[str, object],
    support_contract_sha256: str,
    output_dir: Path,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Stream all source shards into immutable decision shards and a manifest."""

    if not _is_sha256(support_contract_sha256):
        raise ValueError("support_contract_sha256 must be a lowercase SHA-256")
    source_manifest_path = Path(source_manifest_path)
    if not source_manifest_path.is_file():
        raise Active8TraceInventoryError("source unified manifest is absent")
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    if _sha256_bytes(source_manifest) != _sha256_bytes(
        json.loads(source_manifest_path.read_text())
    ):
        raise Active8TraceInventoryError(
            "decoded source manifest differs from the exact manifest file"
        )
    ordered = tuple(
        sorted(
            shards,
            key=lambda shard: (
                shard.manifest_layer,
                shard.partition,
                shard.relative_path,
            ),
        )
    )
    if not ordered:
        raise Active8TraceInventoryError("at least one source shard is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ACTIVE8_TRACE_INVENTORY.json"
    if manifest_path.exists():
        raise Active8TraceInventoryError(
            f"immutable active-8 inventory already exists: {manifest_path}"
        )

    totals = Counter(
        source_shards=0,
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_progress_rows=0,
        accepted_nonterminal_rows=0,
        accepted_terminal_rows=0,
    )
    exclusions_by_reason: Counter[str] = Counter()
    family_rows: Counter[str] = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    shard_manifests: list[dict[str, object]] = []
    seen_source_digests: set[str] = set()

    for shard_index, source in enumerate(ordered):
        if not source.path.is_file():
            raise Active8TraceInventoryError(
                f"manifest-declared source shard is absent: {source.path}"
            )
        source_bytes_sha256 = _sha256_file(source.path)
        packed_manifest_path = manifest_path_for(source.path)
        if not packed_manifest_path.is_file():
            raise Active8TraceInventoryError(
                f"packed source manifest is absent: {packed_manifest_path}"
            )
        packed_manifest_sha256 = _sha256_file(packed_manifest_path)
        packed_overlay_path = overlay_path_for(source.path)
        packed_overlay_sha256 = (
            _sha256_file(packed_overlay_path)
            if packed_overlay_path.is_file()
            else None
        )
        current = Counter(
            traces=0,
            accepted_traces=0,
            excluded_traces=0,
            accepted_progress_rows=0,
            accepted_nonterminal_rows=0,
            accepted_terminal_rows=0,
        )
        current_reasons: Counter[str] = Counter()
        current_families: Counter[str] = Counter()
        observed_digest: list[str | None] = [None]

        derivative_name = f"active8-decisions-{shard_index:05d}.jsonl.gz"
        derivative_path = output_dir / derivative_name
        written, derivative_sha256 = _atomic_deterministic_gzip_jsonl(
            derivative_path,
            _decision_records_for_source(
                source,
                source_bytes_sha256=source_bytes_sha256,
                exact_candidate_checker=exact_candidate_checker,
                current=current,
                current_reasons=current_reasons,
                current_families=current_families,
                observed_digest=observed_digest,
            ),
        )
        if written != current["traces"]:
            raise Active8TraceInventoryError(
                "derived inventory shard lost trace decisions"
            )
        source_digest = observed_digest[0] or source_bytes_sha256
        if source_digest in seen_source_digests:
            raise Active8TraceInventoryError(
                "one physical packed shard is declared more than once"
            )
        seen_source_digests.add(source_digest)
        totals["source_shards"] += 1
        totals.update(current)
        exclusions_by_reason.update(current_reasons)
        family_rows.update(current_families)
        shard_report = {
            "manifest_layer": source.manifest_layer,
            "envelope_layer": source.envelope_layer,
            "partition": source.partition,
            "relative_path": source.relative_path,
            "packed_shard_name": source.path.name,
            "packed_shard_content_sha256": source_digest,
            "packed_manifest_sha256": packed_manifest_sha256,
            "packed_provenance_overlay_sha256": packed_overlay_sha256,
            "inventory_shard": derivative_name,
            "inventory_shard_sha256": derivative_sha256,
            "counts": dict(current),
            "exclusions_by_reason": dict(sorted(current_reasons.items())),
            "accepted_nonterminal_rows_by_family": {
                family: current_families[family] for family in ACTIVE8_FAMILIES
            },
        }
        shard_manifests.append(shard_report)
        if progress_callback is not None:
            progress_callback(shard_report)

    if totals["traces"] != totals["accepted_traces"] + totals["excluded_traces"]:
        raise Active8TraceInventoryError("trace decisions do not form an exact census")
    if totals["accepted_progress_rows"] != (
        totals["accepted_nonterminal_rows"] + totals["accepted_terminal_rows"]
    ):
        raise Active8TraceInventoryError(
            "accepted progress rows do not retain exactly one terminal per trace"
        )
    source_identity = {
        "unified_packed_manifest_sha256": source_manifest_sha256,
        "unified_packed_manifest_semantic_sha256": _sha256_bytes(source_manifest),
        "support_contract_sha256": support_contract_sha256,
        "physical_shard_binding_sha256": _sha256_bytes(
            [
                {
                    key: report[key]
                    for key in (
                        "manifest_layer",
                        "partition",
                        "relative_path",
                        "packed_shard_content_sha256",
                        "packed_manifest_sha256",
                        "packed_provenance_overlay_sha256",
                    )
                }
                for report in shard_manifests
            ]
        ),
    }
    source_identity["effective_source_corpus_cache_sha256"] = _sha256_bytes(
        source_identity
    )
    payload: dict[str, object] = {
        "schema": ACTIVE8_TRACE_INVENTORY_SCHEMA,
        "schema_version": ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION,
        "status": "IMMUTABLE_WHOLE_TRACE_ACTIVE8_BOUNDARY",
        "training_authorized": False,
        "selection_policy": {
            "unit": "complete_packed_trace",
            "active_families": list(ACTIVE8_FAMILIES),
            "all_nonterminal_teachers_must_match_exact_candidates": True,
            "excluded_trace_emits_progress_rows": False,
            "accepted_trace_retains_terminal_row": True,
            "multi_neighbor_atom_insert_supported": False,
        },
        "persistent_state_identity": {
            "schema": PERSISTENT_STATE_DIGEST_SCHEMA,
            "schema_version": PERSISTENT_STATE_DIGEST_VERSION,
        },
        "source_identity": source_identity,
        "implementation_identity": implementation_identity(),
        "counts": dict(totals),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "accepted_nonterminal_rows_by_family": {
            family: family_rows[family] for family in ACTIVE8_FAMILIES
        },
        "shards": shard_manifests,
    }
    payload["inventory_sha256"] = _sha256_bytes(payload)
    _atomic_json(manifest_path, payload)
    return payload


def load_active8_trace_inventory(path: Path) -> dict[str, object]:
    """Load and self-verify one immutable inventory manifest."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Active8TraceInventoryError(
            f"cannot read active-8 inventory manifest: {source}"
        ) from error
    if payload.get("schema") != ACTIVE8_TRACE_INVENTORY_SCHEMA:
        raise Active8TraceInventoryError("unexpected active-8 inventory schema")
    if payload.get("schema_version") != ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION:
        raise Active8TraceInventoryError("unexpected active-8 inventory schema version")
    expected = payload.pop("inventory_sha256", None)
    observed = _sha256_bytes(payload)
    payload["inventory_sha256"] = expected
    if not _is_sha256(expected) or expected != observed:
        raise Active8TraceInventoryError("active-8 inventory self-hash mismatch")
    return payload


def load_active8_trace_admission(
    path: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_inventory_sha256: str | None = None,
    expected_effective_source_corpus_cache_sha256: str | None = None,
) -> Active8TraceAdmission:
    """Load every decision into a verified immutable production admission index."""

    manifest_path = Path(path)
    manifest_file_sha256 = _sha256_file(manifest_path)
    if (
        expected_manifest_file_sha256 is not None
        and manifest_file_sha256 != expected_manifest_file_sha256
    ):
        raise Active8TraceInventoryError(
            "active-8 inventory file SHA-256 mismatch: "
            f"expected={expected_manifest_file_sha256}, "
            f"observed={manifest_file_sha256}"
        )
    manifest = load_active8_trace_inventory(manifest_path)
    inventory_sha256 = manifest["inventory_sha256"]
    if (
        expected_inventory_sha256 is not None
        and inventory_sha256 != expected_inventory_sha256
    ):
        raise Active8TraceInventoryError(
            "active-8 logical inventory SHA-256 mismatch"
        )
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise Active8TraceInventoryError(
            "active-8 inventory lacks its source identity"
        )
    effective_source_sha256 = source_identity.get(
        "effective_source_corpus_cache_sha256"
    )
    if not _is_sha256(effective_source_sha256):
        raise Active8TraceInventoryError(
            "active-8 inventory lacks its effective source-corpus identity"
        )
    if (
        expected_effective_source_corpus_cache_sha256 is not None
        and effective_source_sha256
        != expected_effective_source_corpus_cache_sha256
    ):
        raise Active8TraceInventoryError(
            "active-8 effective source-corpus SHA-256 mismatch"
        )

    decisions_by_digest: dict[str, tuple[tuple[str, bool], ...]] = {}
    shard_digest_by_lane: dict[tuple[str, str, str], str] = {}
    shard_metadata_by_digest: dict[str, Mapping[str, Any]] = {}
    total_traces = 0
    total_accepted = 0
    total_excluded = 0
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise Active8TraceInventoryError(
            "active-8 inventory has no source shards"
        )
    for raw_shard in shards:
        if not isinstance(raw_shard, Mapping):
            raise Active8TraceInventoryError(
                "active-8 shard manifest is not an object"
            )
        shard = dict(raw_shard)
        digest = shard.get("packed_shard_content_sha256")
        packed_shard_name = shard.get("packed_shard_name")
        envelope_layer = shard.get("envelope_layer")
        partition = shard.get("partition")
        decision_name = shard.get("inventory_shard")
        decision_sha256 = shard.get("inventory_shard_sha256")
        counts = shard.get("counts")
        if (
            not _is_sha256(digest)
            or not isinstance(packed_shard_name, str)
            or not packed_shard_name
            or not isinstance(envelope_layer, str)
            or not envelope_layer
            or not isinstance(partition, str)
            or not partition
            or not isinstance(decision_name, str)
            or Path(decision_name).name != decision_name
            or not _is_sha256(decision_sha256)
            or not isinstance(counts, Mapping)
        ):
            raise Active8TraceInventoryError(
                "active-8 source-shard identity is malformed"
            )
        if digest in decisions_by_digest:
            raise Active8TraceInventoryError(
                "active-8 inventory repeats a physical source shard"
            )
        lane = (envelope_layer, partition, packed_shard_name)
        if lane in shard_digest_by_lane:
            raise Active8TraceInventoryError(
                "active-8 inventory repeats a source-shard lane"
            )
        decision_path = manifest_path.parent / decision_name
        if (
            not decision_path.is_file()
            or _sha256_file(decision_path) != decision_sha256
        ):
            raise Active8TraceInventoryError(
                f"active-8 decision shard hash mismatch: {decision_path}"
            )
        decisions: list[tuple[str, bool]] = []
        with gzip.open(decision_path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Active8TraceInventoryError(
                        "active-8 decision shard contains invalid JSON"
                    ) from error
                key = record.get("trace_key")
                if (
                    record.get("schema") != ACTIVE8_TRACE_DECISION_SCHEMA
                    or record.get("schema_version")
                    != ACTIVE8_TRACE_DECISION_SCHEMA_VERSION
                    or not isinstance(key, Mapping)
                    or key.get("packed_shard_content_sha256") != digest
                    or key.get("entry_index") != len(decisions)
                    or not isinstance(key.get("trace_id"), str)
                    or not key["trace_id"]
                    or record.get("packed_shard_name") != packed_shard_name
                    or record.get("layer") != envelope_layer
                    or record.get("partition") != partition
                    or record.get("decision") not in {"accepted", "excluded"}
                ):
                    raise Active8TraceInventoryError(
                        "active-8 decision row disagrees with its source shard"
                    )
                accepted = record["decision"] == "accepted"
                progress_rows = record.get("progress_rows")
                if (
                    not isinstance(progress_rows, list)
                    or (
                        accepted
                        and len(progress_rows)
                        != int(record.get("path_length", -1)) + 1
                    )
                    or (not accepted and progress_rows)
                ):
                    raise Active8TraceInventoryError(
                        "active-8 decision row violates whole-trace admission"
                    )
                decisions.append((key["trace_id"], accepted))
                total_accepted += int(accepted)
                total_excluded += int(not accepted)
        expected_traces = counts.get("traces")
        if type(expected_traces) is not int or expected_traces != len(decisions):
            raise Active8TraceInventoryError(
                "active-8 decision shard disagrees with its trace census"
            )
        if counts.get("accepted_traces") != sum(
            int(accepted) for _, accepted in decisions
        ) or counts.get("excluded_traces") != sum(
            int(not accepted) for _, accepted in decisions
        ):
            raise Active8TraceInventoryError(
                "active-8 decision shard disagrees with its admission census"
            )
        total_traces += len(decisions)
        decisions_by_digest[digest] = tuple(decisions)
        shard_digest_by_lane[lane] = digest
        shard_metadata_by_digest[digest] = MappingProxyType(shard)

    manifest_counts = manifest.get("counts")
    if (
        not isinstance(manifest_counts, Mapping)
        or manifest_counts.get("source_shards") != len(decisions_by_digest)
        or manifest_counts.get("traces") != total_traces
        or manifest_counts.get("accepted_traces") != total_accepted
        or manifest_counts.get("excluded_traces") != total_excluded
        or total_traces != total_accepted + total_excluded
    ):
        raise Active8TraceInventoryError(
            "active-8 admission index disagrees with the manifest census"
        )
    return Active8TraceAdmission(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        inventory_sha256=inventory_sha256,
        effective_source_corpus_cache_sha256=effective_source_sha256,
        decisions_by_digest=MappingProxyType(decisions_by_digest),
        shard_digest_by_lane=MappingProxyType(shard_digest_by_lane),
        shard_metadata_by_digest=MappingProxyType(shard_metadata_by_digest),
        counts=MappingProxyType(
            {
                "source_shards": len(decisions_by_digest),
                "traces": total_traces,
                "accepted_traces": total_accepted,
                "excluded_traces": total_excluded,
            }
        ),
    )


def iter_inventory_decisions(
    manifest_path: Path,
) -> Iterable[dict[str, object]]:
    """Yield every decision after verifying manifest and derivative bytes."""

    manifest_path = Path(manifest_path)
    manifest = load_active8_trace_inventory(manifest_path)
    total = 0
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["inventory_shard"]
        if not path.is_file() or _sha256_file(path) != shard["inventory_shard_sha256"]:
            raise Active8TraceInventoryError(
                f"active-8 decision shard hash mismatch: {path}"
            )
        with gzip.open(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if (
                    record.get("schema") != ACTIVE8_TRACE_DECISION_SCHEMA
                    or record.get("schema_version")
                    != ACTIVE8_TRACE_DECISION_SCHEMA_VERSION
                ):
                    raise Active8TraceInventoryError(
                        "active-8 decision row has another schema"
                    )
                total += 1
                yield record
    if total != manifest["counts"]["traces"]:
        raise Active8TraceInventoryError(
            "active-8 decision shards disagree with manifest trace census"
        )


def accepted_trace_keys(manifest_path: Path) -> frozenset[tuple[str, int, str]]:
    """Materialize the immutable keys allowed to reach row sampling."""

    return frozenset(iter_accepted_trace_keys(manifest_path))


def iter_accepted_trace_keys(
    manifest_path: Path,
) -> Iterable[tuple[str, int, str]]:
    """Stream immutable admitted keys without materializing the full corpus."""

    last_entry_by_digest: dict[str, int] = {}
    for record in iter_inventory_decisions(manifest_path):
        raw = record["trace_key"]
        key = (
            str(raw["packed_shard_content_sha256"]),
            int(raw["entry_index"]),
            str(raw["trace_id"]),
        )
        previous = last_entry_by_digest.get(key[0], -1)
        if key[1] <= previous:
            raise Active8TraceInventoryError(
                "active-8 decision inventory repeats or reorders a physical trace key"
            )
        last_entry_by_digest[key[0]] = key[1]
        if record["decision"] == "accepted":
            rows = record["progress_rows"]
            if len(rows) != int(record["path_length"]) + 1:
                raise Active8TraceInventoryError(
                    "accepted trace does not retain every progress row"
                )
            if not rows[-1]["is_terminal"]:
                raise Active8TraceInventoryError(
                    "accepted trace does not retain its terminal row"
                )
            yield key
        elif record["decision"] == "excluded":
            if record["progress_rows"]:
                raise Active8TraceInventoryError(
                    "excluded trace leaked progress rows"
                )
        else:
            raise Active8TraceInventoryError("unknown active-8 trace decision")


__all__ = [
    "ACTIVE8_FAMILIES",
    "ACTIVE8_TRACE_INVENTORY_SCHEMA",
    "ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION",
    "Active8SourceShard",
    "Active8TraceAdmission",
    "Active8TraceInventoryError",
    "ExactCandidateEvidence",
    "ProductionExactCandidateChecker",
    "accepted_trace_keys",
    "build_active8_trace_inventory",
    "implementation_identity",
    "inventory_record_for_trace",
    "iter_accepted_trace_keys",
    "iter_inventory_decisions",
    "load_active8_trace_admission",
    "load_active8_trace_inventory",
]
