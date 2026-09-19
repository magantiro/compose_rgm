"""Durable, publish-once orchestration for route combination particles.

The store in this module is proposal-only.  It deliberately does not share a
schema or namespace with charged-query locks and receipts.  A parent manifest
freezes the complete 28-job census, source and model identities, work budgets,
and code/configuration identities before any particle runs.  Job receipts are
immutable and are reduced all-or-nothing by the particle reducer.

No controller random state, oracle, docking function, target, cell, or delta is
used here.  A restart reuses a valid durable receipt.  Missing, corrupt,
deadline, or operationally failed jobs cause the whole additive particle lane
to abstain while preserving the independently valid legacy route pool.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.docking_value import identity
from compose_v4.control.route_complete_region_particles import (
    DEPTH_4_COMPLETE_COVERAGE,
    RECEIPT_SCHEMA_VERSION,
    SUPPORTED_COMPLETE_DEPTHS,
    CompleteCombinationJobSpec,
    complete_combination_job_specs,
    frozen_particle_budgets,
    reduce_route_complete_region_pool,
    run_complete_combination_job,
    validate_complete_combination_receipt,
)
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.experiments.t4_shared_controller_cell_runtime import ReceiptStore
from compose_v4.rewrite.trace_shard import encode_state

MANIFEST_SCHEMA_VERSION = "route_complete_region_particle_parent_manifest_v1"
MANIFEST_FILENAME = "manifest.json"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None


def _identity_map(name: str, identities: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(identities, Mapping) or not identities:
        raise ValueError(f"{name} identities must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_key, value in identities.items():
        key = str(raw_key)
        if _IDENTITY_NAME_PATTERN.fullmatch(key) is None or not _valid_hash(value):
            raise ValueError(f"invalid {name} identity: {key!r}")
        normalized[key] = value
    return dict(sorted(normalized.items()))


def make_complete_combination_parent_manifest(
    *,
    source_state_sha256: str,
    shared_checkpoint_sha256: str,
    expert_training_identity_sha256: str,
    code_identities: Mapping[str, str],
    config_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Create the deterministic self-hashed manifest for one source parent."""

    required_hashes = {
        "source_state_sha256": source_state_sha256,
        "shared_checkpoint_sha256": shared_checkpoint_sha256,
        "expert_training_identity_sha256": expert_training_identity_sha256,
    }
    for field, value in required_hashes.items():
        if not _valid_hash(value):
            raise ValueError(f"invalid manifest hash: {field}")
    specs = tuple(sorted(complete_combination_job_specs(), key=lambda row: row.job_id))
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **required_hashes,
        "code_identities": _identity_map("code", code_identities),
        "config_identities": _identity_map("config", config_identities),
        "particle_budgets": asdict(frozen_particle_budgets()),
        "job_count": len(specs),
        "jobs": [spec.payload() for spec in specs],
        "supported_complete_depths": list(SUPPORTED_COMPLETE_DEPTHS),
        "depth_4_complete_coverage": DEPTH_4_COMPLETE_COVERAGE,
        "controller_rng_consumed": False,
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def validate_complete_combination_parent_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a self-hashed manifest against the frozen local schedule."""

    if not isinstance(manifest, Mapping) or set(manifest) != {
        "payload",
        "payload_sha256",
    }:
        raise ValueError("particle parent manifest envelope is invalid")
    payload = manifest.get("payload")
    if (
        not isinstance(payload, dict)
        or not _valid_hash(manifest.get("payload_sha256"))
        or identity(payload) != manifest["payload_sha256"]
    ):
        raise ValueError("particle parent manifest self-hash mismatch")
    expected_fields = {
        "schema_version",
        "source_state_sha256",
        "shared_checkpoint_sha256",
        "expert_training_identity_sha256",
        "code_identities",
        "config_identities",
        "particle_budgets",
        "job_count",
        "jobs",
        "supported_complete_depths",
        "depth_4_complete_coverage",
        "controller_rng_consumed",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != (
        MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("particle parent manifest schema drift")
    for field in (
        "source_state_sha256",
        "shared_checkpoint_sha256",
        "expert_training_identity_sha256",
    ):
        if not _valid_hash(payload.get(field)):
            raise ValueError(f"particle parent manifest has invalid {field}")
    if payload.get("code_identities") != _identity_map(
        "code", payload.get("code_identities")
    ) or payload.get("config_identities") != _identity_map(
        "config", payload.get("config_identities")
    ):
        raise ValueError("particle parent manifest identities are not canonical")
    specs = tuple(sorted(complete_combination_job_specs(), key=lambda row: row.job_id))
    if (
        payload.get("particle_budgets") != asdict(frozen_particle_budgets())
        or payload.get("job_count") != len(specs)
        or payload.get("jobs") != [spec.payload() for spec in specs]
        or payload.get("supported_complete_depths") != list(SUPPORTED_COMPLETE_DEPTHS)
        or payload.get("depth_4_complete_coverage") is not False
        or payload.get("controller_rng_consumed") is not False
    ):
        raise ValueError("particle parent manifest changed its frozen schedule")
    return payload


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {"payload": payload, "payload_sha256": identity(payload)}


def _receipt_relative_path(spec: CompleteCombinationJobSpec) -> str:
    return (
        f"particles/b{spec.binding_particle}/d{spec.depth}/"
        f"s{spec.shard:02d}-of-{spec.shard_count:02d}/receipt.json"
    )


def _missing_at_deadline_receipt(
    spec: CompleteCombinationJobSpec, manifest_payload: Mapping[str, Any]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "job": spec.payload(),
        "status": "missing_at_deadline",
        "records": records,
        "telemetry": {
            "candidate_count": 0,
            "candidate_set_sha256": identity(records),
            "source_state_sha256": manifest_payload["source_state_sha256"],
            "expert_training_identity_sha256": manifest_payload[
                "expert_training_identity_sha256"
            ],
        },
    }
    return _envelope(payload)


class CompleteCombinationParticleReceiptStore:
    """Proposal-specific durable store for one source parent's 28 jobs."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        flush: Callable[[], None] | None = None,
    ):
        self.root = root.resolve()
        self._store = ReceiptStore(self.root, flush=flush)
        validated = validate_complete_combination_parent_manifest(manifest)
        self.manifest_payload = deepcopy(validated)
        self.manifest = _envelope(self.manifest_payload)
        self._specs = {spec.job_id: spec for spec in complete_combination_job_specs()}
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        try:
            existing_payload = self._store.read(MANIFEST_FILENAME)
        except FileNotFoundError:
            try:
                self._store.publish_once(MANIFEST_FILENAME, self.manifest_payload)
            except FileExistsError:
                existing_payload = self._store.read(MANIFEST_FILENAME)
            else:
                return
        existing = _envelope(existing_payload)
        validate_complete_combination_parent_manifest(existing)
        if existing != self.manifest:
            raise ValueError("durable particle parent manifest identity mismatch")

    def _checked_spec(
        self, spec: CompleteCombinationJobSpec
    ) -> CompleteCombinationJobSpec:
        if not isinstance(spec, CompleteCombinationJobSpec):
            raise TypeError("particle receipt job must be a job specification")
        expected = self._specs.get(spec.job_id)
        if expected != spec:
            raise ValueError("particle receipt job is outside the parent manifest")
        return expected

    def receipt_path(self, spec: CompleteCombinationJobSpec) -> str:
        """Return the unique immutable receipt path for a frozen job."""

        return _receipt_relative_path(self._checked_spec(spec))

    def load_receipt(self, spec: CompleteCombinationJobSpec) -> dict[str, Any]:
        """Load and strictly validate one already durable receipt."""

        spec = self._checked_spec(spec)
        payload = self._store.read(self.receipt_path(spec))
        receipt = _envelope(payload)
        validate_complete_combination_receipt(
            receipt,
            expected_spec=spec,
            source_state_sha256=self.manifest_payload["source_state_sha256"],
            expert_training_identity_sha256=self.manifest_payload[
                "expert_training_identity_sha256"
            ],
        )
        return receipt

    def publish_receipt(
        self, spec: CompleteCombinationJobSpec, receipt: Mapping[str, Any]
    ) -> str:
        """Validate and atomically publish one receipt, refusing overwrites."""

        spec = self._checked_spec(spec)
        payload = validate_complete_combination_receipt(
            receipt,
            expected_spec=spec,
            source_state_sha256=self.manifest_payload["source_state_sha256"],
            expert_training_identity_sha256=self.manifest_payload[
                "expert_training_identity_sha256"
            ],
        )
        return self._store.publish_once(self.receipt_path(spec), payload)

    def run_job_once(
        self,
        source: MolecularGraph,
        expert: RouteDistilledGoalExpert,
        spec: CompleteCombinationJobSpec,
        *,
        runner: Callable[
            [MolecularGraph, RouteDistilledGoalExpert, CompleteCombinationJobSpec],
            Mapping[str, Any],
        ] = run_complete_combination_job,
    ) -> tuple[dict[str, Any], bool]:
        """Reuse an existing valid receipt or compute and publish exactly once.

        The boolean is true only when a restart reused a durable receipt.
        """

        spec = self._checked_spec(spec)
        try:
            return self.load_receipt(spec), True
        except FileNotFoundError:
            pass
        if (
            identity(encode_state(source))
            != self.manifest_payload["source_state_sha256"]
        ):
            raise ValueError("runtime source differs from the frozen parent manifest")
        if (
            expert.training_identity
            != self.manifest_payload["expert_training_identity_sha256"]
        ):
            raise ValueError("runtime expert differs from the frozen parent manifest")
        receipt = dict(runner(source, expert, spec))
        try:
            self.publish_receipt(spec, receipt)
            return receipt, False
        except FileExistsError:
            return self.load_receipt(spec), True

    def mark_missing_at_deadline(
        self, spec: CompleteCombinationJobSpec
    ) -> tuple[dict[str, Any], bool]:
        """Publish an empty deadline tombstone, or reuse a prior valid receipt."""

        spec = self._checked_spec(spec)
        try:
            return self.load_receipt(spec), True
        except FileNotFoundError:
            pass
        receipt = _missing_at_deadline_receipt(spec, self.manifest_payload)
        try:
            self.publish_receipt(spec, receipt)
            return receipt, False
        except FileExistsError:
            return self.load_receipt(spec), True

    def settle(
        self, legacy: Iterable[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Deterministically reduce the exact job census, fail-closed."""

        receipts: list[Mapping[str, Any]] = []
        for job_id in sorted(self._specs):
            spec = self._specs[job_id]
            try:
                receipts.append(self.load_receipt(spec))
            except FileNotFoundError:
                continue
            except (AttributeError, OSError, TypeError, ValueError):
                receipts.append({})
        combined, telemetry = reduce_route_complete_region_pool(legacy, receipts)
        telemetry = {
            **telemetry,
            "particle_parent_manifest_sha256": self.manifest["payload_sha256"],
        }
        return combined, telemetry


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "CompleteCombinationParticleReceiptStore",
    "make_complete_combination_parent_manifest",
    "validate_complete_combination_parent_manifest",
]
