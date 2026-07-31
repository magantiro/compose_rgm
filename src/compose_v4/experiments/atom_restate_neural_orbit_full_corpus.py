"""Exact-lineage full-train driver for the atom-restatement orbit audit.

The row-level scientific calculation lives in
``atom_restate_neural_orbit_audit``.  This module adds the missing corpus
boundary: it accepts only the frozen Active8 whole-trace inventory, resolves
the exact train shards declared by its unified packed manifest, verifies every
packed shard/manifest/overlay byte identity, and includes only inventory-
admitted traces.

Rows are streamed to deterministic gzip and aggregated in bounded memory.  A
temporary directory is renamed into place only after both row evidence and the
summary validate, so a failed audit cannot publish a partial artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, BinaryIO

import numpy as np
from rdkit import rdBase

from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    Active8TraceInventoryError,
    load_active8_trace_admission,
    load_active8_trace_inventory,
)
from compose_v4.data.packed_charge_policy_audit import (
    DeclaredPackedShard,
    PackedChargePolicyAuditError,
    resolve_unified_manifest_shards,
)
from compose_v4.data.packed_trace_store import (
    PackedStoreError,
    manifest_path_for,
    read_frozen_source_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import overlay_path_for
from compose_v4.experiments.atom_restate_neural_orbit_audit import (
    AUDIT_SCHEMA,
    AUDIT_SCHEMA_VERSION,
    AUDIT_STATUS,
    CURRENT_POLICY_ID,
    AtomRestateNeuralOrbitAuditError,
    TeacherProgressAddress,
    audit_atom_restate_teacher,
    file_sha256,
    load_audit_contract,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.operators import AtomRestate


FULL_CORPUS_SCHEMA = "compose.experiments.atom_restate_neural_orbit_full_corpus"
FULL_CORPUS_SCHEMA_VERSION = 1
FULL_CORPUS_STATUS = "TRAIN_ONLY_DIAGNOSTIC_NOT_TRAINING_AUTHORITY"
DRIVER_CONTRACT_SCHEMA = "compose.experiments.atom_restate_neural_orbit_full_corpus_contract"
DRIVER_CONTRACT_SCHEMA_VERSION = 1
EXECUTABLE_DRIVER_STATUS = "FROZEN_TRAIN_ONLY_FULL_CORPUS_DIAGNOSTIC_NO_TRAINING_AUTHORITY"
BLOCKED_DRIVER_STATUS = "BLOCKED_MISSING_FROZEN_ACTIVE8_TRAIN_PARENT"
ROW_EVIDENCE_NAME = "atom_restate_neural_orbit_rows.jsonl.gz"
SUMMARY_NAME = "atom_restate_neural_orbit_summary.json"

RESUMABLE_PLAN_SCHEMA = "compose.experiments.atom_restate_neural_orbit_full_corpus_mapreduce_plan"
MAP_RECEIPT_SCHEMA = "compose.experiments.atom_restate_neural_orbit_full_corpus_map_receipt"
RESUMABLE_FINAL_SCHEMA = "compose.experiments.atom_restate_neural_orbit_full_corpus_mapreduce_final"
RESUMABLE_SCHEMA_VERSION = 1
RESUMABLE_STATUS = "TRAIN_ONLY_RESUMABLE_DIAGNOSTIC_NOT_TRAINING_AUTHORITY"

PINNED_DRIVER_CONTRACT_FILE_SHA256 = (
    "185035c4a2f97cce97276a3f4c83d2c5d9af782533edaa57b68c83e72a211c22"
)

_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/atom_restate_neural_orbit_full_corpus.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/experiments/atom_restate_neural_orbit_audit.py",
    "src/compose_v4/data/active8_trace_inventory.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/data/packed_charge_policy_audit.py",
    "src/compose_v4/data/provenance_overlay.py",
    "src/compose_v4/chem/aromaticity.py",
    "src/compose_v4/chem/graph_primitives.py",
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/persistent_state_identity.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "scripts/run_atom_restate_neural_orbit_full_corpus.py",
    "modal_apps/run_atom_restate_neural_orbit_full_corpus.py",
)


class AtomRestateFullCorpusAuditError(RuntimeError):
    """Full-corpus lineage, execution, or publication failed closed."""


@dataclass(frozen=True)
class FullCorpusAuditInputs:
    driver_contract_path: Path
    orbit_contract_path: Path
    active8_inventory_path: Path
    unified_manifest_path: Path
    audit_root: Path
    mmp_root: Path
    output_dir: Path
    repository_root: Path
    max_problem_address_examples: int = 256

    def __post_init__(self) -> None:
        for field in (
            "driver_contract_path",
            "orbit_contract_path",
            "active8_inventory_path",
            "unified_manifest_path",
            "audit_root",
            "mmp_root",
            "output_dir",
            "repository_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if (
            type(self.max_problem_address_examples) is not int
            or self.max_problem_address_examples < 0
        ):
            raise ValueError("max_problem_address_examples must be nonnegative")


@dataclass(frozen=True)
class BoundTrainShard:
    declared: DeclaredPackedShard
    packed_shard_content_sha256: str
    packed_manifest_sha256: str
    packed_provenance_overlay_sha256: str | None
    inventory_shard_sha256: str
    expected_traces: int
    expected_accepted_traces: int
    expected_excluded_traces: int
    expected_atom_restate_teachers: int

    def __post_init__(self) -> None:
        for field in (
            "packed_shard_content_sha256",
            "packed_manifest_sha256",
            "inventory_shard_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.packed_provenance_overlay_sha256 is not None:
            _require_sha256(
                self.packed_provenance_overlay_sha256,
                field="packed_provenance_overlay_sha256",
            )
        for field in (
            "expected_traces",
            "expected_accepted_traces",
            "expected_excluded_traces",
            "expected_atom_restate_teachers",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.declared.partition != "train":
            raise ValueError("a bound full-corpus shard must be train-only")
        if self.expected_traces != (self.expected_accepted_traces + self.expected_excluded_traces):
            raise ValueError("bound shard trace counts are inconsistent")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AtomRestateFullCorpusAuditError(f"{field} must be a lowercase SHA-256")
    return value


def _require_git_commit(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AtomRestateFullCorpusAuditError(
            f"{field} must be a lowercase 40-character Git commit"
        )
    return value


def _logical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def load_full_corpus_driver_contract(path: Path) -> dict[str, object]:
    """Load the self-hashed driver contract and its external physical pin."""

    path = Path(path)
    if not path.is_file():
        raise AtomRestateFullCorpusAuditError(f"full-corpus contract is absent: {path}")
    observed_file_sha256 = file_sha256(path)
    if (
        PINNED_DRIVER_CONTRACT_FILE_SHA256
        and observed_file_sha256 != PINNED_DRIVER_CONTRACT_FILE_SHA256
    ):
        raise AtomRestateFullCorpusAuditError(
            "full-corpus contract physical SHA-256 mismatch: "
            f"expected={PINNED_DRIVER_CONTRACT_FILE_SHA256}, "
            f"observed={observed_file_sha256}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomRestateFullCorpusAuditError("full-corpus contract is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AtomRestateFullCorpusAuditError("full-corpus contract must be a JSON object")
    status = payload.get("status")
    if (
        payload.get("schema") != DRIVER_CONTRACT_SCHEMA
        or payload.get("schema_version") != DRIVER_CONTRACT_SCHEMA_VERSION
        or status not in {EXECUTABLE_DRIVER_STATUS, BLOCKED_DRIVER_STATUS}
        or payload.get("partition_role") != "train_only"
        or payload.get("authorizes_training") is not False
        or payload.get("selects_support_policy") is not False
        or payload.get("allows_arbitrary_or_legacy_shards") is not False
    ):
        raise AtomRestateFullCorpusAuditError(
            "full-corpus contract schema or authorization boundary is invalid"
        )
    required_core_revision = payload.get("required_core_revision")
    _require_git_commit(required_core_revision, field="required_core_revision")
    parent = payload.get("frozen_active8_parent")
    expected_parent_fields = {
        "inventory_manifest_file_sha256",
        "inventory_sha256",
        "effective_source_corpus_cache_sha256",
        "support_contract_sha256",
        "unified_packed_manifest_sha256",
    }
    if status == EXECUTABLE_DRIVER_STATUS:
        if payload.get("authorizes_audit_execution") is not True:
            raise AtomRestateFullCorpusAuditError(
                "executable full-corpus contract lacks explicit audit authority"
            )
        if not isinstance(parent, dict) or set(parent) != expected_parent_fields:
            raise AtomRestateFullCorpusAuditError(
                "full-corpus contract lacks the exact five-part Active8 parent identity"
            )
        for field in sorted(expected_parent_fields):
            _require_sha256(parent[field], field=f"frozen_active8_parent.{field}")
    else:
        if payload.get("authorizes_audit_execution") is not False or parent is not None:
            raise AtomRestateFullCorpusAuditError(
                "blocked full-corpus contract cannot contain audit authority or a train parent"
            )
        observed = payload.get("observed_validation_only_parent")
        if (
            not isinstance(observed, dict)
            or observed.get("partitions") != ["validation"]
            or observed.get("train_shards") != 0
            or observed.get("source_shards") != 4
            or observed.get("accepted_traces") != 31815
        ):
            raise AtomRestateFullCorpusAuditError(
                "blocked contract does not preserve the exact validation-only finding"
            )
        for field in sorted(expected_parent_fields):
            _require_sha256(
                observed.get(field),
                field=f"observed_validation_only_parent.{field}",
            )
    orbit = payload.get("orbit_audit_contract")
    if not isinstance(orbit, dict) or set(orbit) != {
        "file_sha256",
        "contract_sha256",
    }:
        raise AtomRestateFullCorpusAuditError(
            "full-corpus contract lacks the exact orbit-audit contract identity"
        )
    _require_sha256(orbit["file_sha256"], field="orbit_audit_contract.file_sha256")
    _require_sha256(orbit["contract_sha256"], field="orbit_audit_contract.contract_sha256")
    if payload.get("output") != {
        "publication": "atomic_directory_rename",
        "row_evidence": ROW_EVIDENCE_NAME,
        "summary": SUMMARY_NAME,
        "row_encoding": "deterministic_gzip_jsonl_mtime_0",
    }:
        raise AtomRestateFullCorpusAuditError("full-corpus output publication contract is invalid")
    logical = payload.get("contract_sha256")
    _require_sha256(logical, field="contract_sha256")
    unhashed = dict(payload)
    unhashed.pop("contract_sha256")
    observed_logical = _logical_sha256(unhashed)
    if logical != observed_logical:
        raise AtomRestateFullCorpusAuditError(
            "full-corpus contract self-hash mismatch: "
            f"expected={logical}, observed={observed_logical}"
        )
    return payload


def _require_executable_driver_contract(
    driver_contract: Mapping[str, object],
) -> None:
    if (
        driver_contract.get("status") != EXECUTABLE_DRIVER_STATUS
        or driver_contract.get("authorizes_audit_execution") is not True
        or not isinstance(driver_contract.get("frozen_active8_parent"), Mapping)
    ):
        raise AtomRestateFullCorpusAuditError(
            "full-corpus atom-restatement audit is blocked: no frozen admitted "
            "Active8 train parent is available"
        )


def implementation_identity(*, repository_root: Path) -> dict[str, object]:
    """Hash the complete local code surface used by the driver."""

    root = Path(repository_root)
    sources: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise AtomRestateFullCorpusAuditError(
                f"full-corpus implementation source is absent: {path}"
            )
        sources[relative] = file_sha256(path)
    return {
        "sources": sources,
        "implementation_sha256": _logical_sha256(sources),
    }


def _git(
    repository_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=check,
        capture_output=True,
        text=True,
    )


def repository_source_revision(
    repository_root: Path,
    *,
    required_core_revision: str,
) -> dict[str, object]:
    """Require a clean committed source tree containing the frozen core."""

    root = Path(repository_root)
    try:
        commit = _git(root, "rev-parse", "HEAD").stdout.strip()
        status = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
        ancestor = _git(
            root,
            "merge-base",
            "--is-ancestor",
            required_core_revision,
            commit,
            check=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AtomRestateFullCorpusAuditError(
            "cannot establish the exact Git source revision"
        ) from exc
    _require_git_commit(commit, field="source_revision.commit")
    if status:
        raise AtomRestateFullCorpusAuditError(
            "full-corpus scientific audit requires a clean source worktree"
        )
    if ancestor.returncode != 0:
        raise AtomRestateFullCorpusAuditError(
            "source revision does not contain the frozen atom-restatement audit core"
        )
    return {
        "commit": commit,
        "worktree_clean": True,
        "required_core_revision": required_core_revision,
        "required_core_revision_is_ancestor": True,
    }


def _verify_orbit_contract(
    path: Path,
    driver_contract: Mapping[str, object],
) -> dict[str, object]:
    expected = driver_contract.get("orbit_audit_contract")
    if not isinstance(expected, Mapping):
        raise AtomRestateFullCorpusAuditError(
            "driver contract lacks its orbit-audit parent identity"
        )
    observed_file_sha256 = file_sha256(path)
    if observed_file_sha256 != expected["file_sha256"]:
        raise AtomRestateFullCorpusAuditError(
            "orbit-audit contract file differs from the full-corpus binding"
        )
    try:
        orbit = load_audit_contract(path)
    except AtomRestateNeuralOrbitAuditError as exc:
        raise AtomRestateFullCorpusAuditError(
            "orbit-audit contract failed its own verification"
        ) from exc
    if orbit.get("contract_sha256") != expected["contract_sha256"]:
        raise AtomRestateFullCorpusAuditError(
            "orbit-audit logical contract differs from the full-corpus binding"
        )
    return orbit


def _inventory_train_metadata(
    inventory_payload: Mapping[str, object],
) -> dict[tuple[str, str, str, str], Mapping[str, object]]:
    shards = inventory_payload.get("shards")
    if not isinstance(shards, list):
        raise AtomRestateFullCorpusAuditError("Active8 inventory has no shard list")
    result: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for raw in shards:
        if not isinstance(raw, Mapping):
            raise AtomRestateFullCorpusAuditError("Active8 inventory shard metadata is malformed")
        if raw.get("partition") != "train":
            continue
        key = (
            str(raw.get("manifest_layer")),
            str(raw.get("envelope_layer")),
            str(raw.get("partition")),
            str(raw.get("relative_path")),
        )
        if key in result:
            raise AtomRestateFullCorpusAuditError("Active8 inventory repeats a train source shard")
        result[key] = raw
    if not result:
        raise AtomRestateFullCorpusAuditError("Active8 inventory contains no train source shards")
    return result


def resolve_bound_train_shards(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
) -> tuple[Active8TraceAdmission, tuple[BoundTrainShard, ...], dict[str, object]]:
    """Resolve only the exact frozen Active8 train shard set."""

    parent = driver_contract.get("frozen_active8_parent")
    if not isinstance(parent, Mapping):
        raise AtomRestateFullCorpusAuditError(
            "driver contract lacks its frozen Active8 parent identity"
        )
    if file_sha256(inputs.active8_inventory_path) != parent["inventory_manifest_file_sha256"]:
        raise AtomRestateFullCorpusAuditError(
            "Active8 inventory file differs from the frozen driver parent"
        )
    if file_sha256(inputs.unified_manifest_path) != parent["unified_packed_manifest_sha256"]:
        raise AtomRestateFullCorpusAuditError(
            "unified packed manifest differs from the frozen driver parent"
        )
    try:
        admission = load_active8_trace_admission(
            inputs.active8_inventory_path,
            expected_manifest_file_sha256=parent["inventory_manifest_file_sha256"],
            expected_inventory_sha256=parent["inventory_sha256"],
            expected_effective_source_corpus_cache_sha256=parent[
                "effective_source_corpus_cache_sha256"
            ],
            expected_support_contract_sha256=parent["support_contract_sha256"],
        )
        inventory_payload = load_active8_trace_inventory(inputs.active8_inventory_path)
    except Active8TraceInventoryError as exc:
        raise AtomRestateFullCorpusAuditError(
            "Active8 inventory failed exact whole-trace verification"
        ) from exc
    if admission.unified_packed_manifest_sha256 != parent["unified_packed_manifest_sha256"]:
        raise AtomRestateFullCorpusAuditError(
            "Active8 admission names another unified packed manifest"
        )
    try:
        unified_payload, declared = resolve_unified_manifest_shards(
            inputs.unified_manifest_path,
            audit_root=inputs.audit_root,
            mmp_root=inputs.mmp_root,
        )
    except PackedChargePolicyAuditError as exc:
        raise AtomRestateFullCorpusAuditError(
            "cannot resolve the exact unified packed-corpus shards"
        ) from exc
    declared_train = {
        (
            item.manifest_layer,
            item.envelope_layer,
            item.partition,
            item.relative_path,
        ): item
        for item in declared
        if item.partition == "train"
    }
    inventory_train = _inventory_train_metadata(inventory_payload)
    if set(declared_train) != set(inventory_train):
        raise AtomRestateFullCorpusAuditError(
            "unified manifest and Active8 inventory declare different train shards: "
            f"missing={sorted(set(inventory_train) - set(declared_train))}, "
            f"unexpected={sorted(set(declared_train) - set(inventory_train))}"
        )

    bound = []
    for key in sorted(declared_train):
        source = declared_train[key]
        metadata = inventory_train[key]
        counts = metadata.get("counts")
        families = metadata.get("accepted_nonterminal_rows_by_family")
        if not isinstance(counts, Mapping) or not isinstance(families, Mapping):
            raise AtomRestateFullCorpusAuditError("Active8 train shard lacks exact counts")
        if metadata.get("packed_shard_name") != source.path.name:
            raise AtomRestateFullCorpusAuditError(
                "Active8 train shard basename disagrees with the unified manifest"
            )
        overlay_sha256 = metadata.get("packed_provenance_overlay_sha256")
        if overlay_sha256 is not None:
            _require_sha256(
                overlay_sha256,
                field="packed_provenance_overlay_sha256",
            )
        bound.append(
            BoundTrainShard(
                declared=source,
                packed_shard_content_sha256=_require_sha256(
                    metadata.get("packed_shard_content_sha256"),
                    field="packed_shard_content_sha256",
                ),
                packed_manifest_sha256=_require_sha256(
                    metadata.get("packed_manifest_sha256"),
                    field="packed_manifest_sha256",
                ),
                packed_provenance_overlay_sha256=overlay_sha256,
                inventory_shard_sha256=_require_sha256(
                    metadata.get("inventory_shard_sha256"),
                    field="inventory_shard_sha256",
                ),
                expected_traces=_nonnegative_count(counts, "traces"),
                expected_accepted_traces=_nonnegative_count(
                    counts,
                    "accepted_traces",
                ),
                expected_excluded_traces=_nonnegative_count(
                    counts,
                    "excluded_traces",
                ),
                expected_atom_restate_teachers=_nonnegative_count(
                    families,
                    "atom_restate",
                ),
            )
        )
    packed_digests = [item.packed_shard_content_sha256 for item in bound]
    if len(set(packed_digests)) != len(packed_digests):
        raise AtomRestateFullCorpusAuditError(
            "Active8 train inventory aliases one packed source digest through "
            "multiple manifest entries"
        )
    return admission, tuple(bound), unified_payload


def _nonnegative_count(mapping: Mapping[str, object], field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value < 0:
        raise AtomRestateFullCorpusAuditError(
            f"inventory count {field!r} must be a nonnegative integer"
        )
    return value


class _StreamingAggregate:
    """Bounded-memory aggregate over already-persisted row evidence."""

    def __init__(self, *, max_problem_address_examples: int) -> None:
        self.max_examples = max_problem_address_examples
        self.teacher_count = 0
        self.policy: dict[str, dict[str, Any]] = {}

    def add(self, row: Mapping[str, object]) -> None:
        if (
            row.get("schema") != AUDIT_SCHEMA
            or row.get("schema_version") != AUDIT_SCHEMA_VERSION
            or row.get("status") != AUDIT_STATUS
            or row.get("partition_role") != "train_only"
            or row.get("authorizes_training") is not False
            or row.get("selects_support_policy") is not False
        ):
            raise AtomRestateFullCorpusAuditError(
                "core audit emitted an invalid or authorizing row"
            )
        address = row.get("address")
        if not isinstance(address, Mapping) or address.get("partition") != "train":
            raise AtomRestateFullCorpusAuditError("core audit row lacks an exact train address")
        comparisons = row.get("policy_comparisons")
        if not isinstance(comparisons, list):
            raise AtomRestateFullCorpusAuditError("core audit row lacks policy comparisons")
        self.teacher_count += 1
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise AtomRestateFullCorpusAuditError("core policy comparison is malformed")
            policy_id = str(comparison.get("policy_id"))
            aggregate = self.policy.setdefault(
                policy_id,
                {
                    "policy_kind": comparison.get("policy_kind"),
                    "hypothetical": comparison.get("hypothetical"),
                    "teacher_count": 0,
                    "teacher_outside_policy_count": 0,
                    "ceiling_below_one_count": 0,
                    "minimum_ceiling": None,
                    "minimum_ceiling_fraction": None,
                    "minimum_ceiling_address": None,
                    "retained_mark_count": 0,
                    "class_size_histogram": Counter(),
                    "class_successor_cardinality_histogram": Counter(),
                    "semantic_counts": defaultdict(Counter),
                    "ceiling_fraction_histogram": Counter(),
                    "problem_address_examples": [],
                },
            )
            if aggregate["policy_kind"] != comparison.get("policy_kind") or aggregate[
                "hypothetical"
            ] != comparison.get("hypothetical"):
                raise AtomRestateFullCorpusAuditError(
                    f"policy {policy_id!r} changes identity across rows"
                )
            aggregate["teacher_count"] += 1
            aggregate["retained_mark_count"] += _nonnegative_count(
                comparison,
                "retained_mark_count",
            )
            for field in (
                "class_size_histogram",
                "class_successor_cardinality_histogram",
            ):
                raw_histogram = comparison.get(field)
                if not isinstance(raw_histogram, Mapping):
                    raise AtomRestateFullCorpusAuditError(f"policy comparison lacks {field}")
                aggregate[field].update(
                    {
                        str(key): _nonnegative_int(value, field=f"{field}.{key}")
                        for key, value in raw_histogram.items()
                    }
                )
            semantic = comparison.get("semantic_counts")
            if not isinstance(semantic, Mapping):
                raise AtomRestateFullCorpusAuditError("policy comparison lacks semantic counts")
            for axis, raw_counts in semantic.items():
                if not isinstance(raw_counts, Mapping):
                    raise AtomRestateFullCorpusAuditError(
                        f"semantic count axis {axis!r} is malformed"
                    )
                aggregate["semantic_counts"][str(axis)].update(
                    {
                        str(key): _nonnegative_int(
                            value,
                            field=f"semantic_counts.{axis}.{key}",
                        )
                        for key, value in raw_counts.items()
                    }
                )
            ceiling = comparison.get("family_conditional_teacher_probability_ceiling")
            if ceiling is None:
                aggregate["teacher_outside_policy_count"] += 1
                continue
            if not isinstance(ceiling, Mapping):
                raise AtomRestateFullCorpusAuditError("teacher probability ceiling is malformed")
            numerator = _positive_int(ceiling.get("numerator"), field="ceiling.numerator")
            denominator = _positive_int(
                ceiling.get("denominator"),
                field="ceiling.denominator",
            )
            if numerator > denominator:
                raise AtomRestateFullCorpusAuditError(
                    "teacher ceiling numerator exceeds its denominator"
                )
            value = numerator / denominator
            if abs(float(ceiling.get("value")) - value) > 1e-12:
                raise AtomRestateFullCorpusAuditError(
                    "teacher ceiling float disagrees with its exact fraction"
                )
            aggregate["ceiling_fraction_histogram"][f"{numerator}/{denominator}"] += 1
            previous_fraction = aggregate["minimum_ceiling_fraction"]
            if previous_fraction is None or (
                numerator * previous_fraction[1] < previous_fraction[0] * denominator
            ):
                aggregate["minimum_ceiling"] = value
                aggregate["minimum_ceiling_fraction"] = (
                    numerator,
                    denominator,
                )
                aggregate["minimum_ceiling_address"] = dict(address)
            if value < 1.0:
                aggregate["ceiling_below_one_count"] += 1
                examples = aggregate["problem_address_examples"]
                if len(examples) < self.max_examples:
                    examples.append(dict(address))

    def result(self, *, require_nonempty: bool = True) -> dict[str, object]:
        if self.teacher_count <= 0:
            if not require_nonempty:
                return {}
            raise AtomRestateFullCorpusAuditError(
                "frozen train corpus emitted zero atom-restatement teachers"
            )
        if CURRENT_POLICY_ID not in self.policy:
            raise AtomRestateFullCorpusAuditError(
                "streaming aggregate lacks the current broad policy"
            )
        result: dict[str, object] = {}
        for policy_id in sorted(self.policy):
            aggregate = self.policy[policy_id]
            if aggregate["teacher_count"] != self.teacher_count:
                raise AtomRestateFullCorpusAuditError(
                    f"policy {policy_id!r} does not cover every teacher row"
                )
            result[policy_id] = {
                "policy_kind": aggregate["policy_kind"],
                "hypothetical": aggregate["hypothetical"],
                "teacher_count": aggregate["teacher_count"],
                "teacher_outside_policy_count": aggregate["teacher_outside_policy_count"],
                "ceiling_below_one_count": aggregate["ceiling_below_one_count"],
                "minimum_ceiling": aggregate["minimum_ceiling"],
                "minimum_ceiling_fraction": (
                    {
                        "numerator": aggregate["minimum_ceiling_fraction"][0],
                        "denominator": aggregate["minimum_ceiling_fraction"][1],
                    }
                    if aggregate["minimum_ceiling_fraction"] is not None
                    else None
                ),
                "minimum_ceiling_address": aggregate["minimum_ceiling_address"],
                "retained_mark_count": aggregate["retained_mark_count"],
                "class_size_histogram": dict(sorted(aggregate["class_size_histogram"].items())),
                "class_successor_cardinality_histogram": dict(
                    sorted(aggregate["class_successor_cardinality_histogram"].items())
                ),
                "semantic_counts": {
                    axis: dict(sorted(counts.items()))
                    for axis, counts in sorted(aggregate["semantic_counts"].items())
                },
                "ceiling_fraction_histogram": dict(
                    sorted(aggregate["ceiling_fraction_histogram"].items())
                ),
                "problem_address_examples": aggregate["problem_address_examples"],
                "problem_address_examples_truncated": (
                    aggregate["ceiling_below_one_count"]
                    > len(aggregate["problem_address_examples"])
                ),
            }
        return result


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise AtomRestateFullCorpusAuditError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise AtomRestateFullCorpusAuditError(f"{field} must be positive")
    return result


class _AtomicPublisher:
    """Publish deterministic rows and summary together or publish nothing."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.temporary_dir: Path | None = None
        self.raw: BinaryIO | None = None
        self.compressed: gzip.GzipFile | None = None
        self.row_count = 0

    def __enter__(self) -> "_AtomicPublisher":
        if self.output_dir.exists():
            raise AtomRestateFullCorpusAuditError(
                f"refusing to overwrite full-corpus artifact: {self.output_dir}"
            )
        parent = self.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.",
                suffix=".tmp",
                dir=parent,
            )
        )
        self.raw = (self.temporary_dir / ROW_EVIDENCE_NAME).open("wb")
        self.compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self.raw,
            mtime=0,
        )
        return self

    def write_row(self, row: Mapping[str, object]) -> None:
        if self.compressed is None:
            raise AtomRestateFullCorpusAuditError("row publisher is not open")
        self.compressed.write(_canonical_json_bytes(row) + b"\n")
        self.row_count += 1

    def publish(self, summary: dict[str, object]) -> dict[str, object]:
        if self.temporary_dir is None or self.compressed is None or self.raw is None:
            raise AtomRestateFullCorpusAuditError("publisher is not open")
        self.compressed.close()
        self.compressed = None
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        self.raw = None
        rows_path = self.temporary_dir / ROW_EVIDENCE_NAME
        rows_sha256 = file_sha256(rows_path)
        summary["row_evidence"] = {
            "path": ROW_EVIDENCE_NAME,
            "file_sha256": rows_sha256,
            "row_count": self.row_count,
            "encoding": "deterministic_gzip_jsonl_mtime_0",
            "row_schema": AUDIT_SCHEMA,
            "row_schema_version": AUDIT_SCHEMA_VERSION,
        }
        unhashed = dict(summary)
        summary["summary_sha256"] = _logical_sha256(unhashed)
        summary_path = self.temporary_dir / SUMMARY_NAME
        with summary_path.open("wb") as handle:
            handle.write(
                json.dumps(
                    summary,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(self.temporary_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(self.temporary_dir, self.output_dir)
        self.temporary_dir = None
        parent_fd = os.open(self.output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return summary

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.compressed is not None:
            self.compressed.close()
            self.compressed = None
        if self.raw is not None:
            self.raw.close()
            self.raw = None
        if self.temporary_dir is not None:
            shutil.rmtree(self.temporary_dir)
            self.temporary_dir = None


def _bound_shard_identity(bound: BoundTrainShard) -> dict[str, object]:
    return {
        "manifest_layer": bound.declared.manifest_layer,
        "envelope_layer": bound.declared.envelope_layer,
        "partition": bound.declared.partition,
        "relative_path": bound.declared.relative_path,
        "packed_shard_name": bound.declared.path.name,
        "resolved_path": str(bound.declared.path.resolve()),
        "packed_shard_content_sha256": bound.packed_shard_content_sha256,
        "packed_manifest_path": str(manifest_path_for(bound.declared.path).resolve()),
        "packed_manifest_sha256": bound.packed_manifest_sha256,
        "packed_provenance_overlay_path": (
            str(overlay_path_for(bound.declared.path).resolve())
            if bound.packed_provenance_overlay_sha256 is not None
            else None
        ),
        "packed_provenance_overlay_sha256": (bound.packed_provenance_overlay_sha256),
        "inventory_shard_sha256": bound.inventory_shard_sha256,
        "expected_traces": bound.expected_traces,
        "expected_accepted_traces": bound.expected_accepted_traces,
        "expected_excluded_traces": bound.expected_excluded_traces,
        "expected_atom_restate_teachers": (bound.expected_atom_restate_teachers),
    }


def _software_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "platform": platform.platform(),
    }


def _scan_bound_shard(
    bound: BoundTrainShard,
    admission: Active8TraceAdmission,
    publisher: _AtomicPublisher,
    aggregate: _StreamingAggregate,
) -> dict[str, object]:
    counts = Counter(
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_nonterminal_rows=0,
        atom_restate_teachers=0,
    )
    observed_digest: str | None = None
    last_entry_index = -1
    try:
        for addressed in read_frozen_source_addressed_packed_shard(
            bound.declared.path,
            expected_shard_sha256=bound.packed_shard_content_sha256,
            expected_manifest_sha256=bound.packed_manifest_sha256,
            expected_overlay_sha256=bound.packed_provenance_overlay_sha256,
            verify_fraction=0.0,
        ):
            address = addressed.address
            if (
                address.layer != bound.declared.envelope_layer
                or address.partition != "train"
                or address.packed_shard_name != bound.declared.path.name
                or address.entry_index != last_entry_index + 1
            ):
                raise AtomRestateFullCorpusAuditError(
                    "decoded packed trace disagrees with its exact train shard envelope"
                )
            last_entry_index = address.entry_index
            observed_digest = address.packed_shard_content_sha256
            counts["traces"] += 1
            if not admission.is_accepted(address):
                counts["excluded_traces"] += 1
                continue
            counts["accepted_traces"] += 1
            counts["accepted_nonterminal_rows"] += address.path_length
            for progress_index, step in enumerate(addressed.trace.steps):
                family = canonical_family(step.rule_name)
                if family != "atom_restate":
                    continue
                if step.rule_name != "atom_restate" or not isinstance(
                    step.action,
                    AtomRestate,
                ):
                    raise AtomRestateFullCorpusAuditError(
                        "accepted atom-restatement teacher has another executor action"
                    )
                row = audit_atom_restate_teacher(
                    addressed.path.state_at(progress_index),
                    step.action,
                    addressed.path.state_at(progress_index + 1),
                    TeacherProgressAddress.from_packed_address(
                        address,
                        progress_index=progress_index,
                    ),
                )
                aggregate.add(row)
                publisher.write_row(row)
                counts["atom_restate_teachers"] += 1
    except PackedStoreError as exc:
        raise AtomRestateFullCorpusAuditError(
            f"frozen train shard failed physical or semantic verification: {bound.declared.path}"
        ) from exc
    if observed_digest is None:
        observed_digest = bound.packed_shard_content_sha256
    try:
        admission.assert_complete_source_shard(
            packed_shard_name=bound.declared.path.name,
            layer=bound.declared.envelope_layer,
            partition="train",
            observed_digest=observed_digest,
            observed_entries=counts["traces"],
        )
    except Active8TraceInventoryError as exc:
        raise AtomRestateFullCorpusAuditError(
            "decoded train shard is incomplete relative to Active8 admission"
        ) from exc
    expected = {
        "traces": bound.expected_traces,
        "accepted_traces": bound.expected_accepted_traces,
        "excluded_traces": bound.expected_excluded_traces,
        "atom_restate_teachers": bound.expected_atom_restate_teachers,
    }
    observed = {field: counts[field] for field in expected}
    if observed != expected:
        raise AtomRestateFullCorpusAuditError(
            "audited train shard counts disagree with Active8 inventory: "
            f"expected={expected}, observed={observed}"
        )
    return dict(counts)


def execute_bound_full_corpus_audit(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
    source_revision: Mapping[str, object],
) -> dict[str, object]:
    """Execute an already contract- and revision-bound audit.

    This split keeps fixture tests small.  The public entry point below is the
    only CLI path and always derives both bindings from the pinned contract and
    live clean Git repository.
    """

    _verify_orbit_contract(inputs.orbit_contract_path, driver_contract)
    admission, bound_shards, _ = resolve_bound_train_shards(
        inputs,
        driver_contract=driver_contract,
    )
    implementation = implementation_identity(repository_root=inputs.repository_root)
    aggregate = _StreamingAggregate(
        max_problem_address_examples=inputs.max_problem_address_examples
    )
    shard_reports = []
    totals = Counter(
        shards=0,
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_nonterminal_rows=0,
        atom_restate_teachers=0,
    )
    observed_lanes: set[tuple[str, str, str]] = set()
    with _AtomicPublisher(inputs.output_dir) as publisher:
        for bound in bound_shards:
            report = _scan_bound_shard(
                bound,
                admission,
                publisher,
                aggregate,
            )
            report["source"] = _bound_shard_identity(bound)
            shard_reports.append(report)
            totals.update({key: value for key, value in report.items() if key in totals})
            totals["shards"] += 1
            observed_lanes.add(
                (
                    bound.declared.envelope_layer,
                    "train",
                    bound.declared.path.name,
                )
            )
        try:
            admission.assert_partition_shards("train", observed_lanes)
        except Active8TraceInventoryError as exc:
            raise AtomRestateFullCorpusAuditError(
                "driver did not consume the exact complete Active8 train shard set"
            ) from exc
        if publisher.row_count != totals["atom_restate_teachers"]:
            raise AtomRestateFullCorpusAuditError(
                "row evidence count disagrees with audited teacher count"
            )
        policy_aggregates = aggregate.result()
        if aggregate.teacher_count != totals["atom_restate_teachers"]:
            raise AtomRestateFullCorpusAuditError("streaming aggregate lost atom-restatement rows")
        driver_contract_path = inputs.driver_contract_path.resolve()
        orbit_contract_path = inputs.orbit_contract_path.resolve()
        summary: dict[str, object] = {
            "schema": FULL_CORPUS_SCHEMA,
            "schema_version": FULL_CORPUS_SCHEMA_VERSION,
            "status": FULL_CORPUS_STATUS,
            "partition_role": "train_only",
            "authorizes_training": False,
            "selects_support_policy": False,
            "source_revision": dict(source_revision),
            "implementation": implementation,
            "software": _software_identity(),
            "determinism": {
                "random_seeds": [],
                "row_order": (
                    "manifest_layer_then_relative_path_then_entry_index_then_progress_index"
                ),
                "bounded_memory": True,
                "max_problem_address_examples": (inputs.max_problem_address_examples),
            },
            "inputs": {
                "driver_contract": {
                    "path": str(driver_contract_path),
                    "file_sha256": file_sha256(driver_contract_path),
                    "contract_sha256": driver_contract["contract_sha256"],
                },
                "orbit_audit_contract": {
                    "path": str(orbit_contract_path),
                    "file_sha256": file_sha256(orbit_contract_path),
                    "contract_sha256": driver_contract["orbit_audit_contract"]["contract_sha256"],
                },
                "active8_inventory": {
                    "path": str(inputs.active8_inventory_path.resolve()),
                    "manifest_file_sha256": admission.manifest_file_sha256,
                    "inventory_sha256": admission.inventory_sha256,
                    "effective_source_corpus_cache_sha256": (
                        admission.effective_source_corpus_cache_sha256
                    ),
                    "support_contract_sha256": (admission.support_contract_sha256),
                    "unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
                },
                "unified_packed_manifest": {
                    "path": str(inputs.unified_manifest_path.resolve()),
                    "file_sha256": file_sha256(inputs.unified_manifest_path),
                },
                "audit_root": str(inputs.audit_root.resolve()),
                "mmp_root": str(inputs.mmp_root.resolve()),
                "train_shards": [_bound_shard_identity(bound) for bound in bound_shards],
            },
            "counts": dict(totals),
            "policy_aggregates": policy_aggregates,
            "shards": shard_reports,
        }
        return publisher.publish(summary)


def run_full_corpus_audit(inputs: FullCorpusAuditInputs) -> dict[str, object]:
    """Public fail-closed entry point used by the command-line driver."""

    driver_contract = load_full_corpus_driver_contract(inputs.driver_contract_path)
    _require_executable_driver_contract(driver_contract)
    source_revision = repository_source_revision(
        inputs.repository_root,
        required_core_revision=str(driver_contract["required_core_revision"]),
    )
    return execute_bound_full_corpus_audit(
        inputs,
        driver_contract=driver_contract,
        source_revision=source_revision,
    )


def _validate_source_revision(
    source_revision: Mapping[str, object],
    *,
    required_core_revision: str,
) -> dict[str, object]:
    commit = _require_git_commit(
        source_revision.get("commit"),
        field="source_revision.commit",
    )
    if (
        source_revision.get("worktree_clean") is not True
        or source_revision.get("required_core_revision") != required_core_revision
        or source_revision.get("required_core_revision_is_ancestor") is not True
    ):
        raise AtomRestateFullCorpusAuditError(
            "source revision is not a clean descendant of the frozen audit core"
        )
    validated = {
        "commit": commit,
        "worktree_clean": True,
        "required_core_revision": required_core_revision,
        "required_core_revision_is_ancestor": True,
    }
    expected_implementation_sha256 = source_revision.get("implementation_sha256")
    if expected_implementation_sha256 is not None:
        validated["implementation_sha256"] = _require_sha256(
            expected_implementation_sha256,
            field="source_revision.implementation_sha256",
        )
    return validated


def _semantic_shard_identity(bound: BoundTrainShard) -> dict[str, object]:
    """Return a location-independent source identity for task addressing."""

    physical = _bound_shard_identity(bound)
    for field in (
        "resolved_path",
        "packed_manifest_path",
        "packed_provenance_overlay_path",
    ):
        physical.pop(field)
    return physical


def _plan_resumable_with_bound_shards(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
    source_revision: Mapping[str, object],
) -> tuple[
    dict[str, object],
    Active8TraceAdmission,
    tuple[BoundTrainShard, ...],
]:
    _require_executable_driver_contract(driver_contract)
    required_core_revision = _require_git_commit(
        driver_contract.get("required_core_revision"),
        field="required_core_revision",
    )
    validated_revision = _validate_source_revision(
        source_revision,
        required_core_revision=required_core_revision,
    )
    _verify_orbit_contract(inputs.orbit_contract_path, driver_contract)
    admission, bound_shards, _ = resolve_bound_train_shards(
        inputs,
        driver_contract=driver_contract,
    )
    implementation = implementation_identity(repository_root=inputs.repository_root)
    expected_implementation_sha256 = validated_revision.get("implementation_sha256")
    if (
        expected_implementation_sha256 is not None
        and implementation["implementation_sha256"] != expected_implementation_sha256
    ):
        raise AtomRestateFullCorpusAuditError(
            "runtime implementation bytes differ from the clean local source binding"
        )
    runtime_software = _software_identity()
    software = {field: runtime_software[field] for field in ("python", "numpy", "rdkit")}
    parent = driver_contract.get("frozen_active8_parent")
    orbit = driver_contract.get("orbit_audit_contract")
    if not isinstance(parent, Mapping) or not isinstance(orbit, Mapping):
        raise AtomRestateFullCorpusAuditError("driver contract parent identities are malformed")
    driver_contract_sha256 = _require_sha256(
        driver_contract.get("contract_sha256"),
        field="driver_contract.contract_sha256",
    )
    sources = [_semantic_shard_identity(bound) for bound in bound_shards]
    run_identity_payload = {
        "schema": RESUMABLE_PLAN_SCHEMA,
        "schema_version": RESUMABLE_SCHEMA_VERSION,
        "partition_role": "train_only",
        "driver_contract": {
            "file_sha256": file_sha256(inputs.driver_contract_path),
            "contract_sha256": driver_contract_sha256,
        },
        "orbit_audit_contract": {
            "file_sha256": file_sha256(inputs.orbit_contract_path),
            "contract_sha256": orbit["contract_sha256"],
        },
        "frozen_active8_parent": dict(parent),
        "source_revision": validated_revision,
        "implementation": implementation,
        "software": software,
        "sources": sources,
        "row_encoding": "deterministic_gzip_jsonl_mtime_0",
        "max_problem_address_examples": inputs.max_problem_address_examples,
    }
    run_identity_sha256 = _logical_sha256(run_identity_payload)
    tasks = []
    for source in sources:
        task_payload = {
            "schema": f"{RESUMABLE_PLAN_SCHEMA}.task",
            "schema_version": RESUMABLE_SCHEMA_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "source": source,
        }
        tasks.append(
            {
                **task_payload,
                "task_identity_sha256": _logical_sha256(task_payload),
            }
        )
    plan: dict[str, object] = {
        **run_identity_payload,
        "status": RESUMABLE_STATUS,
        "authorizes_training": False,
        "selects_support_policy": False,
        "run_identity_sha256": run_identity_sha256,
        "expected_map_tasks": len(tasks),
        "tasks": tasks,
    }
    plan["plan_sha256"] = _logical_sha256(plan)
    return plan, admission, bound_shards


def plan_resumable_full_corpus_audit(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
    source_revision: Mapping[str, object],
) -> dict[str, object]:
    """Plan exact immutable per-shard work without reading teacher rows."""

    plan, _, _ = _plan_resumable_with_bound_shards(
        inputs,
        driver_contract=driver_contract,
        source_revision=source_revision,
    )
    return plan


def _resumable_run_root(output_root: Path, run_identity_sha256: str) -> Path:
    _require_sha256(run_identity_sha256, field="run_identity_sha256")
    return Path(output_root) / "runs" / run_identity_sha256


def _load_self_hashed_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomRestateFullCorpusAuditError(
            f"resumable receipt is not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AtomRestateFullCorpusAuditError(f"resumable receipt must be a JSON object: {path}")
    claimed = _require_sha256(
        payload.get("summary_sha256"),
        field="summary_sha256",
    )
    unhashed = dict(payload)
    unhashed.pop("summary_sha256")
    observed = _logical_sha256(unhashed)
    if claimed != observed:
        raise AtomRestateFullCorpusAuditError(f"resumable receipt self-hash mismatch: {path}")
    return payload


def _verified_artifact_files(directory: Path) -> tuple[Path, Path]:
    if not directory.is_dir():
        raise AtomRestateFullCorpusAuditError(f"resumable artifact is not a directory: {directory}")
    observed = {path.name for path in directory.iterdir()}
    expected = {ROW_EVIDENCE_NAME, SUMMARY_NAME}
    if observed != expected:
        raise AtomRestateFullCorpusAuditError(
            "resumable artifact has missing or extra files: "
            f"directory={directory}, missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return directory / ROW_EVIDENCE_NAME, directory / SUMMARY_NAME


def _verify_map_receipt(
    task_dir: Path,
    *,
    task: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    rows_path, summary_path = _verified_artifact_files(task_dir)
    receipt = _load_self_hashed_summary(summary_path)
    if (
        receipt.get("schema") != MAP_RECEIPT_SCHEMA
        or receipt.get("schema_version") != RESUMABLE_SCHEMA_VERSION
        or receipt.get("status") != RESUMABLE_STATUS
        or receipt.get("partition_role") != "train_only"
        or receipt.get("authorizes_training") is not False
        or receipt.get("selects_support_policy") is not False
        or receipt.get("run_identity_sha256") != plan.get("run_identity_sha256")
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("task_identity_sha256") != task.get("task_identity_sha256")
        or receipt.get("source") != task.get("source")
    ):
        raise AtomRestateFullCorpusAuditError(
            f"map receipt disagrees with its exact task: {task_dir}"
        )
    row_evidence = receipt.get("row_evidence")
    counts = receipt.get("counts")
    if not isinstance(row_evidence, Mapping) or not isinstance(counts, Mapping):
        raise AtomRestateFullCorpusAuditError(
            f"map receipt lacks row evidence or counts: {task_dir}"
        )
    if (
        row_evidence.get("path") != ROW_EVIDENCE_NAME
        or file_sha256(rows_path) != row_evidence.get("file_sha256")
        or _nonnegative_count(row_evidence, "row_count")
        != _nonnegative_count(counts, "atom_restate_teachers")
    ):
        raise AtomRestateFullCorpusAuditError(
            f"map row evidence disagrees with its receipt: {task_dir}"
        )
    return receipt


def _expected_task_map(plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise AtomRestateFullCorpusAuditError("resumable plan has no map tasks")
    result: dict[str, Mapping[str, object]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise AtomRestateFullCorpusAuditError("resumable task is malformed")
        identity = _require_sha256(
            task.get("task_identity_sha256"),
            field="task_identity_sha256",
        )
        if identity in result:
            raise AtomRestateFullCorpusAuditError("resumable plan repeats a map-task identity")
        result[identity] = task
    return result


def verified_completed_map_tasks(
    plan: Mapping[str, object],
    *,
    output_root: Path,
) -> set[str]:
    """Return only byte-verified immutable map receipts safe to reuse."""

    tasks = _expected_task_map(plan)
    run_root = _resumable_run_root(
        output_root,
        str(plan["run_identity_sha256"]),
    )
    map_root = run_root / "map"
    if not map_root.exists():
        return set()
    if not map_root.is_dir():
        raise AtomRestateFullCorpusAuditError(f"resumable map root is not a directory: {map_root}")
    observed = {path.name for path in map_root.iterdir() if not path.name.startswith(".")}
    unexpected = observed - set(tasks)
    if unexpected:
        raise AtomRestateFullCorpusAuditError(
            f"resumable map root contains unexpected task objects: {sorted(unexpected)}"
        )
    completed = set()
    for identity in sorted(observed):
        _verify_map_receipt(
            map_root / identity,
            task=tasks[identity],
            plan=plan,
        )
        completed.add(identity)
    return completed


def map_resumable_full_corpus_shard(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
    source_revision: Mapping[str, object],
    output_root: Path,
    task_identity_sha256: str,
) -> dict[str, object]:
    """Build or verify one immutable, content-addressed train-shard receipt."""

    plan, admission, bound_shards = _plan_resumable_with_bound_shards(
        inputs,
        driver_contract=driver_contract,
        source_revision=source_revision,
    )
    tasks = _expected_task_map(plan)
    task = tasks.get(task_identity_sha256)
    if task is None:
        raise AtomRestateFullCorpusAuditError(
            "requested map task is not present in the exact current plan"
        )
    run_root = _resumable_run_root(output_root, str(plan["run_identity_sha256"]))
    task_dir = run_root / "map" / task_identity_sha256
    if task_dir.exists():
        receipt = _verify_map_receipt(task_dir, task=task, plan=plan)
        return {**receipt, "reused": True}
    source = task.get("source")
    matches = [bound for bound in bound_shards if _semantic_shard_identity(bound) == source]
    if len(matches) != 1:
        raise AtomRestateFullCorpusAuditError(
            "map task does not resolve to exactly one frozen train shard"
        )
    bound = matches[0]
    aggregate = _StreamingAggregate(
        max_problem_address_examples=inputs.max_problem_address_examples
    )
    with _AtomicPublisher(task_dir) as publisher:
        counts = _scan_bound_shard(bound, admission, publisher, aggregate)
        receipt: dict[str, object] = {
            "schema": MAP_RECEIPT_SCHEMA,
            "schema_version": RESUMABLE_SCHEMA_VERSION,
            "status": RESUMABLE_STATUS,
            "partition_role": "train_only",
            "authorizes_training": False,
            "selects_support_policy": False,
            "run_identity_sha256": plan["run_identity_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "task_identity_sha256": task_identity_sha256,
            "source": source,
            "source_revision": plan["source_revision"],
            "implementation": plan["implementation"],
            "software": plan["software"],
            "counts": counts,
            "policy_aggregates": aggregate.result(require_nonempty=False),
        }
        persisted = publisher.publish(receipt)
    return {**persisted, "reused": False}


def _verify_final_artifact(
    final_dir: Path,
    *,
    plan: Mapping[str, object],
) -> dict[str, object]:
    rows_path, summary_path = _verified_artifact_files(final_dir)
    summary = _load_self_hashed_summary(summary_path)
    if (
        summary.get("schema") != RESUMABLE_FINAL_SCHEMA
        or summary.get("schema_version") != RESUMABLE_SCHEMA_VERSION
        or summary.get("status") != RESUMABLE_STATUS
        or summary.get("partition_role") != "train_only"
        or summary.get("authorizes_training") is not False
        or summary.get("selects_support_policy") is not False
        or summary.get("run_identity_sha256") != plan.get("run_identity_sha256")
        or summary.get("plan_sha256") != plan.get("plan_sha256")
    ):
        raise AtomRestateFullCorpusAuditError(
            "final resumable artifact disagrees with the exact plan"
        )
    row_evidence = summary.get("row_evidence")
    counts = summary.get("counts")
    if not isinstance(row_evidence, Mapping) or not isinstance(counts, Mapping):
        raise AtomRestateFullCorpusAuditError(
            "final resumable artifact lacks row evidence or counts"
        )
    if file_sha256(rows_path) != row_evidence.get("file_sha256") or _nonnegative_count(
        row_evidence, "row_count"
    ) != _nonnegative_count(counts, "atom_restate_teachers"):
        raise AtomRestateFullCorpusAuditError(
            "final resumable row evidence disagrees with its summary"
        )
    return summary


def reduce_resumable_full_corpus_audit(
    inputs: FullCorpusAuditInputs,
    *,
    driver_contract: Mapping[str, object],
    source_revision: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    """Strictly reduce a complete exact set of immutable shard receipts."""

    plan, admission, bound_shards = _plan_resumable_with_bound_shards(
        inputs,
        driver_contract=driver_contract,
        source_revision=source_revision,
    )
    tasks = _expected_task_map(plan)
    completed = verified_completed_map_tasks(plan, output_root=output_root)
    if completed != set(tasks):
        raise AtomRestateFullCorpusAuditError(
            f"cannot reduce an incomplete map receipt set: missing={sorted(set(tasks) - completed)}"
        )
    run_root = _resumable_run_root(output_root, str(plan["run_identity_sha256"]))
    final_dir = run_root / "final"
    if final_dir.exists():
        return {
            **_verify_final_artifact(final_dir, plan=plan),
            "reused": True,
        }

    aggregate = _StreamingAggregate(
        max_problem_address_examples=inputs.max_problem_address_examples
    )
    totals = Counter(
        shards=0,
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_nonterminal_rows=0,
        atom_restate_teachers=0,
    )
    receipt_evidence = []
    observed_lanes: set[tuple[str, str, str]] = set()
    with _AtomicPublisher(final_dir) as publisher:
        for task_identity in sorted(tasks):
            task = tasks[task_identity]
            task_dir = run_root / "map" / task_identity
            receipt = _verify_map_receipt(task_dir, task=task, plan=plan)
            source = task["source"]
            if not isinstance(source, Mapping):
                raise AtomRestateFullCorpusAuditError("map task source identity is malformed")
            row_count = 0
            previous_position: tuple[int, int] | None = None
            try:
                with gzip.open(task_dir / ROW_EVIDENCE_NAME, "rt") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise AtomRestateFullCorpusAuditError(
                                "map row evidence must contain JSON objects"
                            )
                        address = row.get("address")
                        if not isinstance(address, Mapping):
                            raise AtomRestateFullCorpusAuditError(
                                "map row evidence lacks a train address"
                            )
                        if (
                            address.get("packed_shard_content_sha256")
                            != source.get("packed_shard_content_sha256")
                            or address.get("packed_shard_name") != source.get("packed_shard_name")
                            or address.get("layer") != source.get("envelope_layer")
                            or address.get("partition") != "train"
                        ):
                            raise AtomRestateFullCorpusAuditError(
                                "map row address escapes its immutable source task"
                            )
                        position = (
                            _nonnegative_count(address, "entry_index"),
                            _nonnegative_count(address, "progress_index"),
                        )
                        if previous_position is not None and position <= previous_position:
                            raise AtomRestateFullCorpusAuditError(
                                "map row addresses are duplicated or out of order"
                            )
                        previous_position = position
                        aggregate.add(row)
                        publisher.write_row(row)
                        row_count += 1
            except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
                raise AtomRestateFullCorpusAuditError(
                    f"map row evidence is unreadable: {task_dir}"
                ) from exc
            expected_rows = _nonnegative_count(
                receipt["row_evidence"],
                "row_count",
            )
            if row_count != expected_rows:
                raise AtomRestateFullCorpusAuditError(
                    "map row stream count disagrees with its immutable receipt"
                )
            counts = receipt.get("counts")
            if not isinstance(counts, Mapping):
                raise AtomRestateFullCorpusAuditError("map receipt counts are malformed")
            for field in totals:
                if field == "shards":
                    continue
                totals[field] += _nonnegative_count(counts, field)
            totals["shards"] += 1
            observed_lanes.add(
                (
                    str(source["envelope_layer"]),
                    "train",
                    str(source["packed_shard_name"]),
                )
            )
            receipt_evidence.append(
                {
                    "task_identity_sha256": task_identity,
                    "receipt_summary_sha256": receipt["summary_sha256"],
                    "row_evidence_file_sha256": receipt["row_evidence"]["file_sha256"],
                    "row_count": expected_rows,
                    "source": source,
                }
            )
        try:
            admission.assert_partition_shards("train", observed_lanes)
        except Active8TraceInventoryError as exc:
            raise AtomRestateFullCorpusAuditError(
                "resumable reducer did not cover the exact Active8 train shard set"
            ) from exc
        if publisher.row_count != totals["atom_restate_teachers"]:
            raise AtomRestateFullCorpusAuditError(
                "final row count disagrees with reduced teacher count"
            )
        policy_aggregates = aggregate.result()
        if aggregate.teacher_count != totals["atom_restate_teachers"]:
            raise AtomRestateFullCorpusAuditError(
                "final streaming aggregate lost atom-restatement rows"
            )
        summary: dict[str, object] = {
            "schema": RESUMABLE_FINAL_SCHEMA,
            "schema_version": RESUMABLE_SCHEMA_VERSION,
            "status": RESUMABLE_STATUS,
            "partition_role": "train_only",
            "authorizes_training": False,
            "selects_support_policy": False,
            "run_identity_sha256": plan["run_identity_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "source_revision": plan["source_revision"],
            "implementation": plan["implementation"],
            "software": plan["software"],
            "determinism": {
                "random_seeds": [],
                "task_order": "task_identity_sha256",
                "within_task_order": "entry_index_then_progress_index",
                "bounded_memory": True,
                "max_problem_address_examples": (inputs.max_problem_address_examples),
            },
            "inputs": {
                "driver_contract": plan["driver_contract"],
                "orbit_audit_contract": plan["orbit_audit_contract"],
                "frozen_active8_parent": plan["frozen_active8_parent"],
                "train_shards": [_bound_shard_identity(bound) for bound in bound_shards],
            },
            "counts": dict(totals),
            "policy_aggregates": policy_aggregates,
            "map_receipts": receipt_evidence,
        }
        persisted = publisher.publish(summary)
    return {**persisted, "reused": False}


__all__ = [
    "DRIVER_CONTRACT_SCHEMA",
    "DRIVER_CONTRACT_SCHEMA_VERSION",
    "FULL_CORPUS_SCHEMA",
    "FULL_CORPUS_SCHEMA_VERSION",
    "FULL_CORPUS_STATUS",
    "MAP_RECEIPT_SCHEMA",
    "RESUMABLE_FINAL_SCHEMA",
    "RESUMABLE_PLAN_SCHEMA",
    "RESUMABLE_SCHEMA_VERSION",
    "RESUMABLE_STATUS",
    "ROW_EVIDENCE_NAME",
    "SUMMARY_NAME",
    "AtomRestateFullCorpusAuditError",
    "BoundTrainShard",
    "FullCorpusAuditInputs",
    "execute_bound_full_corpus_audit",
    "implementation_identity",
    "load_full_corpus_driver_contract",
    "map_resumable_full_corpus_shard",
    "plan_resumable_full_corpus_audit",
    "reduce_resumable_full_corpus_audit",
    "repository_source_revision",
    "resolve_bound_train_shards",
    "run_full_corpus_audit",
    "verified_completed_map_tasks",
]
