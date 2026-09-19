"""Typed loader for the frozen editing-V2 candidate provenance decisions.

This unit freezes only the identities needed to construct the candidate-stage
provenance registry.  It deliberately does not define semantic atom mappings
or molecular-distance metrics.  Each of those policies must be either bound by
one SHA-256 or named explicitly as unresolved, never both and never neither.

The candidate-stage key contract is a pass-through contract.  It copies the
persisted candidate-header identity strings without parsing molecules, invoking
RDKit, replaying the executor, or applying a current semantic canonicalizer.
This keeps historical compilation lineage distinct from future semantic trace
migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DECISIONS_SCHEMA = "compose.editing_v2_candidate_provenance_decisions"
DECISIONS_SCHEMA_VERSION = 1
DECISIONS_STATUS = "FROZEN_CANDIDATE_PROVENANCE_DECISIONS_NO_TRAINING_AUTHORITY"

IDENTITY_DEFINITIONS_SCHEMA = "compose.editing_v2_identity_definition_contract"
IDENTITY_DEFINITIONS_SCHEMA_VERSION = 2
RELATIONSHIP_NAMESPACE_SCHEMA = "compose.editing_v2_relationship_namespace"
RELATIONSHIP_NAMESPACE_SCHEMA_VERSION = 2

EVIDENCE_COMPONENTS = (
    "source_endpoint",
    "target_endpoint",
    "pair_relationship",
    "path",
    "intermediates",
    "action_sequence",
)

_HEX = frozenset("0123456789abcdef")


class EditingV2CandidateProvenanceDecisionsError(ValueError):
    """The frozen candidate-provenance decision unit is incomplete or invalid."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """One physical and semantic artifact identity."""

    path: str
    file_sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class CandidateMaterializationIdentity:
    """Exact completed candidate materialization and row-stream identity."""

    path: str
    manifest_file_sha256: str
    manifest_sha256: str
    rows_file_sha256: str
    rows_semantic_sha256: str
    address_stream_sha256: str


@dataclass(frozen=True)
class CandidateArtifactInputs:
    """Content-addressed candidate build and all direct frozen inputs."""

    volume_name: str
    run_root: str
    run_identity_sha256: str
    build_commit: str
    completion: ArtifactIdentity
    candidate_materialization: CandidateMaterializationIdentity
    source_manifest: ArtifactIdentity
    editing_corpus_contract: ArtifactIdentity
    routing_policy: ArtifactIdentity
    source_binding_registry: ArtifactIdentity
    upstream_overlay_completion: ArtifactIdentity


@dataclass(frozen=True)
class CandidateStagePassThroughIdentity:
    """No-recanonicalization identity contract for persisted header strings."""

    definition_id: str
    implementation_sha256: str
    source_of_truth: str
    identity_fields: tuple[str, ...]
    decoded_value_semantics: str
    parsing_performed: bool
    rdkit_invoked: bool
    executor_replay_performed: bool
    semantic_normalization_performed: bool
    identity_sha256: str


@dataclass(frozen=True)
class CandidateCompilerIdentity:
    """Candidate-stage registry identity, not historical trace compiler lineage."""

    implementation_sha256: str
    config_sha256: str
    operator_contract_sha256: str
    canonicalizer_sha256: str


@dataclass(frozen=True)
class CandidatePolicyBindings:
    """Resolved policy hashes or explicit deferred policy identifiers."""

    mapping_policy_sha256: str | None
    metric_policy_sha256: str | None
    unresolved_mapping_policies: tuple[str, ...]
    unresolved_metric_policies: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceIdentity:
    """Selectors used to derive provenance references from candidate headers."""

    definition_id: str
    implementation_sha256: str
    source_record_id_fields: tuple[str, ...]
    component_reference_fields: tuple[tuple[str, tuple[str, ...]], ...]
    evidence_identity_sha256: str

    def component_fields_dict(self) -> dict[str, list[str]]:
        return {name: list(fields) for name, fields in self.component_reference_fields}


@dataclass(frozen=True)
class IdentityDefinition:
    definition_id: str
    implementation_sha256: str
    identity_namespace: str


@dataclass(frozen=True)
class IdentityDefinitions:
    exact_molecule: IdentityDefinition
    partition_scaffold: IdentityDefinition
    source_group: IdentityDefinition


@dataclass(frozen=True)
class SourceGroupNamespace:
    namespace_id: str
    source_asset_id: str
    source_asset_sha256: str
    definition_id: str
    implementation_sha256: str
    cross_lane_sharing_authorized: bool


@dataclass(frozen=True)
class SourceAssetDecision:
    source_asset_id: str
    source_asset_path: str
    source_asset_sha256: str
    source_version: str
    access_basis: str
    source_group_namespace: SourceGroupNamespace


@dataclass(frozen=True)
class CandidateProvenanceDecisions:
    """Validated immutable candidate-stage provenance decision unit."""

    decision_id: str
    status: str
    training_authorized: bool
    candidate_artifact: CandidateArtifactInputs
    candidate_stage_pass_through: CandidateStagePassThroughIdentity
    compiler_identity: CandidateCompilerIdentity
    policy_bindings: CandidatePolicyBindings
    evidence_identity: EvidenceIdentity
    identity_definitions: IdentityDefinitions
    source_assets: tuple[SourceAssetDecision, ...]
    decision_sha256: str

    def registry_inputs(self) -> dict[str, object]:
        """Return exact kwargs for ``build_candidate_provenance_registry``.

        Candidate-artifact paths remain driver inputs.  The returned objects
        contain only the five explicit scientific-decision arguments expected
        by the registry builder.
        """

        identities = self.identity_definitions
        return {
            "compiler_identity": {
                "implementation_sha256": self.compiler_identity.implementation_sha256,
                "config_sha256": self.compiler_identity.config_sha256,
                "operator_contract_sha256": self.compiler_identity.operator_contract_sha256,
                "canonicalizer_sha256": self.compiler_identity.canonicalizer_sha256,
            },
            "policy_bindings": {
                "mapping_policy_sha256": self.policy_bindings.mapping_policy_sha256,
                "metric_policy_sha256": self.policy_bindings.metric_policy_sha256,
                "unresolved_mapping_policies": list(
                    self.policy_bindings.unresolved_mapping_policies
                ),
                "unresolved_metric_policies": list(self.policy_bindings.unresolved_metric_policies),
            },
            "evidence_identity": {
                "definition_id": self.evidence_identity.definition_id,
                "implementation_sha256": self.evidence_identity.implementation_sha256,
                "source_record_id_fields": list(self.evidence_identity.source_record_id_fields),
                "component_reference_fields": self.evidence_identity.component_fields_dict(),
                "evidence_identity_sha256": self.evidence_identity.evidence_identity_sha256,
            },
            "identity_definitions": {
                "schema": IDENTITY_DEFINITIONS_SCHEMA,
                "schema_version": IDENTITY_DEFINITIONS_SCHEMA_VERSION,
                "exact_molecule": _identity_definition_dict(identities.exact_molecule),
                "partition_scaffold": _identity_definition_dict(identities.partition_scaffold),
                "source_group": _identity_definition_dict(identities.source_group),
            },
            "source_assets": [_source_asset_dict(source) for source in self.source_assets],
        }


def _identity_definition_dict(value: IdentityDefinition) -> dict[str, object]:
    return {
        "definition_id": value.definition_id,
        "implementation_sha256": value.implementation_sha256,
        "identity_namespace": value.identity_namespace,
    }


def _source_asset_dict(value: SourceAssetDecision) -> dict[str, object]:
    namespace = value.source_group_namespace
    return {
        "source_asset_id": value.source_asset_id,
        "source_asset_path": value.source_asset_path,
        "source_asset_sha256": value.source_asset_sha256,
        "source_version": value.source_version,
        "access_basis": value.access_basis,
        "source_group_namespace": {
            "schema": RELATIONSHIP_NAMESPACE_SCHEMA,
            "schema_version": RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
            "relationship_type": "source_group",
            "namespace_id": namespace.namespace_id,
            "source_asset_id": namespace.source_asset_id,
            "source_asset_sha256": namespace.source_asset_sha256,
            "definition_id": namespace.definition_id,
            "implementation_sha256": namespace.implementation_sha256,
            "cross_lane_sharing_authorized": namespace.cross_lane_sharing_authorized,
        },
    }


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2CandidateProvenanceDecisionsError(
            "candidate provenance decisions are not canonical-JSON serializable"
        ) from error


def candidate_provenance_decisions_self_hash(payload: Mapping[str, Any]) -> str:
    """Hash the exact decision body, excluding only ``decision_sha256``."""

    body = dict(payload)
    body.pop("decision_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _require_object(value: object, *, fields: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2CandidateProvenanceDecisionsError(f"{field} must be an object")
    actual = set(value)
    if actual != fields:
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} fields disagree; missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must be nonempty normalized text"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _require_commit(value: object, *, field: str) -> str:
    commit = value if isinstance(value, str) else ""
    if len(commit) != 40 or any(character not in _HEX for character in commit):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must be a full lowercase 40-character commit SHA"
        )
    return commit


def _require_sorted_texts(
    value: object,
    *,
    field: str,
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EditingV2CandidateProvenanceDecisionsError(f"{field} must be a sequence of strings")
    values = tuple(
        _require_text(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if values != tuple(sorted(set(values))):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must be unique and deterministically sorted"
        )
    if nonempty and not values:
        raise EditingV2CandidateProvenanceDecisionsError(f"{field} must not be empty")
    return values


def _artifact_identity(value: object, *, field: str) -> ArtifactIdentity:
    payload = _require_object(
        value,
        fields={"path", "file_sha256", "semantic_sha256"},
        field=field,
    )
    return ArtifactIdentity(
        path=_require_text(payload["path"], field=f"{field}.path"),
        file_sha256=_require_sha256(payload["file_sha256"], field=f"{field}.file_sha256"),
        semantic_sha256=_require_sha256(
            payload["semantic_sha256"], field=f"{field}.semantic_sha256"
        ),
    )


def _candidate_artifact(value: object) -> CandidateArtifactInputs:
    field = "candidate_artifact"
    payload = _require_object(
        value,
        fields={
            "volume_name",
            "run_root",
            "run_identity_sha256",
            "build_commit",
            "completion",
            "candidate_materialization",
            "source_manifest",
            "editing_corpus_contract",
            "routing_policy",
            "source_binding_registry",
            "upstream_overlay_completion",
        },
        field=field,
    )
    materialization = _require_object(
        payload["candidate_materialization"],
        fields={
            "path",
            "manifest_file_sha256",
            "manifest_sha256",
            "rows_file_sha256",
            "rows_semantic_sha256",
            "address_stream_sha256",
        },
        field=f"{field}.candidate_materialization",
    )
    return CandidateArtifactInputs(
        volume_name=_require_text(payload["volume_name"], field=f"{field}.volume_name"),
        run_root=_require_text(payload["run_root"], field=f"{field}.run_root"),
        run_identity_sha256=_require_sha256(
            payload["run_identity_sha256"], field=f"{field}.run_identity_sha256"
        ),
        build_commit=_require_commit(payload["build_commit"], field=f"{field}.build_commit"),
        completion=_artifact_identity(payload["completion"], field=f"{field}.completion"),
        candidate_materialization=CandidateMaterializationIdentity(
            path=_require_text(
                materialization["path"], field=f"{field}.candidate_materialization.path"
            ),
            manifest_file_sha256=_require_sha256(
                materialization["manifest_file_sha256"],
                field=f"{field}.candidate_materialization.manifest_file_sha256",
            ),
            manifest_sha256=_require_sha256(
                materialization["manifest_sha256"],
                field=f"{field}.candidate_materialization.manifest_sha256",
            ),
            rows_file_sha256=_require_sha256(
                materialization["rows_file_sha256"],
                field=f"{field}.candidate_materialization.rows_file_sha256",
            ),
            rows_semantic_sha256=_require_sha256(
                materialization["rows_semantic_sha256"],
                field=f"{field}.candidate_materialization.rows_semantic_sha256",
            ),
            address_stream_sha256=_require_sha256(
                materialization["address_stream_sha256"],
                field=f"{field}.candidate_materialization.address_stream_sha256",
            ),
        ),
        source_manifest=_artifact_identity(
            payload["source_manifest"], field=f"{field}.source_manifest"
        ),
        editing_corpus_contract=_artifact_identity(
            payload["editing_corpus_contract"], field=f"{field}.editing_corpus_contract"
        ),
        routing_policy=_artifact_identity(
            payload["routing_policy"], field=f"{field}.routing_policy"
        ),
        source_binding_registry=_artifact_identity(
            payload["source_binding_registry"], field=f"{field}.source_binding_registry"
        ),
        upstream_overlay_completion=_artifact_identity(
            payload["upstream_overlay_completion"],
            field=f"{field}.upstream_overlay_completion",
        ),
    )


def _pass_through_identity(value: object) -> CandidateStagePassThroughIdentity:
    field = "candidate_stage_pass_through"
    payload = _require_object(
        value,
        fields={
            "definition_id",
            "implementation_sha256",
            "source_of_truth",
            "identity_fields",
            "decoded_value_semantics",
            "parsing_performed",
            "rdkit_invoked",
            "executor_replay_performed",
            "semantic_normalization_performed",
            "identity_sha256",
        },
        field=field,
    )
    body = dict(payload)
    supplied = _require_sha256(body.pop("identity_sha256"), field=f"{field}.identity_sha256")
    if supplied != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field}.identity_sha256 disagrees with its canonical body"
        )
    if (
        payload["source_of_truth"] != "persisted_candidate_header_identity_fields"
        or payload["decoded_value_semantics"] != "exact_decoded_string_equality"
        or payload["parsing_performed"] is not False
        or payload["rdkit_invoked"] is not False
        or payload["executor_replay_performed"] is not False
        or payload["semantic_normalization_performed"] is not False
    ):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must remain an exact no-recanonicalization pass-through contract"
        )
    return CandidateStagePassThroughIdentity(
        definition_id=_require_text(payload["definition_id"], field=f"{field}.definition_id"),
        implementation_sha256=_require_sha256(
            payload["implementation_sha256"], field=f"{field}.implementation_sha256"
        ),
        source_of_truth=payload["source_of_truth"],
        identity_fields=_require_sorted_texts(
            payload["identity_fields"], field=f"{field}.identity_fields", nonempty=True
        ),
        decoded_value_semantics=payload["decoded_value_semantics"],
        parsing_performed=False,
        rdkit_invoked=False,
        executor_replay_performed=False,
        semantic_normalization_performed=False,
        identity_sha256=supplied,
    )


def _policy_bindings(value: object) -> CandidatePolicyBindings:
    field = "policy_bindings"
    payload = _require_object(
        value,
        fields={
            "mapping_policy_sha256",
            "metric_policy_sha256",
            "unresolved_mapping_policies",
            "unresolved_metric_policies",
        },
        field=field,
    )
    normalized: dict[str, object] = {}
    for policy_type in ("mapping", "metric"):
        sha_field = f"{policy_type}_policy_sha256"
        unresolved_field = f"unresolved_{policy_type}_policies"
        digest = (
            None
            if payload[sha_field] is None
            else _require_sha256(payload[sha_field], field=f"{field}.{sha_field}")
        )
        unresolved = _require_sorted_texts(
            payload[unresolved_field],
            field=f"{field}.{unresolved_field}",
            nonempty=False,
        )
        if (digest is None) == (not unresolved):
            raise EditingV2CandidateProvenanceDecisionsError(
                f"{field}.{policy_type} must have exactly one of a resolved SHA-256 "
                "or explicit unresolved policy identifiers"
            )
        normalized[sha_field] = digest
        normalized[unresolved_field] = unresolved
    return CandidatePolicyBindings(
        mapping_policy_sha256=normalized["mapping_policy_sha256"],  # type: ignore[arg-type]
        metric_policy_sha256=normalized["metric_policy_sha256"],  # type: ignore[arg-type]
        unresolved_mapping_policies=normalized["unresolved_mapping_policies"],  # type: ignore[arg-type]
        unresolved_metric_policies=normalized["unresolved_metric_policies"],  # type: ignore[arg-type]
    )


def _evidence_identity(value: object) -> EvidenceIdentity:
    field = "evidence_identity"
    payload = _require_object(
        value,
        fields={
            "definition_id",
            "implementation_sha256",
            "source_record_id_fields",
            "component_reference_fields",
            "evidence_identity_sha256",
        },
        field=field,
    )
    components = _require_object(
        payload["component_reference_fields"],
        fields=set(EVIDENCE_COMPONENTS),
        field=f"{field}.component_reference_fields",
    )
    component_fields = tuple(
        (
            component,
            _require_sorted_texts(
                components[component],
                field=f"{field}.component_reference_fields.{component}",
                nonempty=True,
            ),
        )
        for component in EVIDENCE_COMPONENTS
    )
    body = {
        "definition_id": _require_text(payload["definition_id"], field=f"{field}.definition_id"),
        "implementation_sha256": _require_sha256(
            payload["implementation_sha256"], field=f"{field}.implementation_sha256"
        ),
        "source_record_id_fields": list(
            _require_sorted_texts(
                payload["source_record_id_fields"],
                field=f"{field}.source_record_id_fields",
                nonempty=True,
            )
        ),
        "component_reference_fields": {name: list(fields) for name, fields in component_fields},
    }
    supplied = _require_sha256(
        payload["evidence_identity_sha256"], field=f"{field}.evidence_identity_sha256"
    )
    if supplied != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field}.evidence_identity_sha256 disagrees with its canonical body"
        )
    return EvidenceIdentity(
        definition_id=body["definition_id"],  # type: ignore[arg-type]
        implementation_sha256=body["implementation_sha256"],  # type: ignore[arg-type]
        source_record_id_fields=tuple(body["source_record_id_fields"]),  # type: ignore[arg-type]
        component_reference_fields=component_fields,
        evidence_identity_sha256=supplied,
    )


def _identity_definition(value: object, *, field: str) -> IdentityDefinition:
    payload = _require_object(
        value,
        fields={"definition_id", "implementation_sha256", "identity_namespace"},
        field=field,
    )
    return IdentityDefinition(
        definition_id=_require_text(payload["definition_id"], field=f"{field}.definition_id"),
        implementation_sha256=_require_sha256(
            payload["implementation_sha256"], field=f"{field}.implementation_sha256"
        ),
        identity_namespace=_require_text(
            payload["identity_namespace"], field=f"{field}.identity_namespace"
        ),
    )


def _identity_definitions(value: object) -> IdentityDefinitions:
    field = "identity_definitions"
    payload = _require_object(
        value,
        fields={"schema", "schema_version", "exact_molecule", "partition_scaffold", "source_group"},
        field=field,
    )
    if (
        payload["schema"] != IDENTITY_DEFINITIONS_SCHEMA
        or payload["schema_version"] != IDENTITY_DEFINITIONS_SCHEMA_VERSION
    ):
        raise EditingV2CandidateProvenanceDecisionsError(f"{field} schema identity is unsupported")
    return IdentityDefinitions(
        exact_molecule=_identity_definition(
            payload["exact_molecule"], field=f"{field}.exact_molecule"
        ),
        partition_scaffold=_identity_definition(
            payload["partition_scaffold"], field=f"{field}.partition_scaffold"
        ),
        source_group=_identity_definition(payload["source_group"], field=f"{field}.source_group"),
    )


def _source_assets(value: object) -> tuple[SourceAssetDecision, ...]:
    field = "source_assets"
    if not isinstance(value, list) or not value:
        raise EditingV2CandidateProvenanceDecisionsError(f"{field} must be a nonempty list")
    sources: list[SourceAssetDecision] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        payload = _require_object(
            item,
            fields={
                "source_asset_id",
                "source_asset_path",
                "source_asset_sha256",
                "source_version",
                "access_basis",
                "source_group_namespace",
            },
            field=item_field,
        )
        source_id = _require_text(payload["source_asset_id"], field=f"{item_field}.source_asset_id")
        source_sha = _require_sha256(
            payload["source_asset_sha256"], field=f"{item_field}.source_asset_sha256"
        )
        namespace_field = f"{item_field}.source_group_namespace"
        namespace = _require_object(
            payload["source_group_namespace"],
            fields={
                "schema",
                "schema_version",
                "relationship_type",
                "namespace_id",
                "source_asset_id",
                "source_asset_sha256",
                "definition_id",
                "implementation_sha256",
                "cross_lane_sharing_authorized",
            },
            field=namespace_field,
        )
        if (
            namespace["schema"] != RELATIONSHIP_NAMESPACE_SCHEMA
            or namespace["schema_version"] != RELATIONSHIP_NAMESPACE_SCHEMA_VERSION
            or namespace["relationship_type"] != "source_group"
            or namespace["source_asset_id"] != source_id
            or namespace["source_asset_sha256"] != source_sha
            or namespace["cross_lane_sharing_authorized"] is not True
        ):
            raise EditingV2CandidateProvenanceDecisionsError(
                f"{namespace_field} must bind the exact source asset and authorize cross-lane sharing"
            )
        sources.append(
            SourceAssetDecision(
                source_asset_id=source_id,
                source_asset_path=_require_text(
                    payload["source_asset_path"], field=f"{item_field}.source_asset_path"
                ),
                source_asset_sha256=source_sha,
                source_version=_require_text(
                    payload["source_version"], field=f"{item_field}.source_version"
                ),
                access_basis=_require_text(
                    payload["access_basis"], field=f"{item_field}.access_basis"
                ),
                source_group_namespace=SourceGroupNamespace(
                    namespace_id=_require_text(
                        namespace["namespace_id"], field=f"{namespace_field}.namespace_id"
                    ),
                    source_asset_id=source_id,
                    source_asset_sha256=source_sha,
                    definition_id=_require_text(
                        namespace["definition_id"], field=f"{namespace_field}.definition_id"
                    ),
                    implementation_sha256=_require_sha256(
                        namespace["implementation_sha256"],
                        field=f"{namespace_field}.implementation_sha256",
                    ),
                    cross_lane_sharing_authorized=True,
                ),
            )
        )
    ids = [source.source_asset_id for source in sources]
    if ids != sorted(set(ids)):
        raise EditingV2CandidateProvenanceDecisionsError(
            f"{field} must be unique and deterministically sorted by source_asset_id"
        )
    return tuple(sources)


def validate_candidate_provenance_decisions(
    value: object,
) -> CandidateProvenanceDecisions:
    """Validate and normalize a frozen candidate-provenance decision object."""

    payload = _require_object(
        value,
        fields={
            "schema",
            "schema_version",
            "decision_id",
            "status",
            "training_authorized",
            "candidate_artifact",
            "candidate_stage_pass_through",
            "compiler_identity",
            "policy_bindings",
            "evidence_identity",
            "identity_definitions",
            "source_assets",
            "decision_sha256",
        },
        field="candidate provenance decisions",
    )
    if (
        payload["schema"] != DECISIONS_SCHEMA
        or payload["schema_version"] != DECISIONS_SCHEMA_VERSION
        or payload["status"] != DECISIONS_STATUS
        or payload["training_authorized"] is not False
    ):
        raise EditingV2CandidateProvenanceDecisionsError(
            "candidate provenance decisions schema, status, or authority is invalid"
        )
    supplied = _require_sha256(payload["decision_sha256"], field="decision_sha256")
    if supplied != candidate_provenance_decisions_self_hash(payload):
        raise EditingV2CandidateProvenanceDecisionsError(
            "candidate provenance decisions self-hash disagrees"
        )
    artifact = _candidate_artifact(payload["candidate_artifact"])
    pass_through = _pass_through_identity(payload["candidate_stage_pass_through"])
    compiler_payload = _require_object(
        payload["compiler_identity"],
        fields={
            "implementation_sha256",
            "config_sha256",
            "operator_contract_sha256",
            "canonicalizer_sha256",
        },
        field="compiler_identity",
    )
    compiler = CandidateCompilerIdentity(
        implementation_sha256=_require_sha256(
            compiler_payload["implementation_sha256"],
            field="compiler_identity.implementation_sha256",
        ),
        config_sha256=_require_sha256(
            compiler_payload["config_sha256"], field="compiler_identity.config_sha256"
        ),
        operator_contract_sha256=_require_sha256(
            compiler_payload["operator_contract_sha256"],
            field="compiler_identity.operator_contract_sha256",
        ),
        canonicalizer_sha256=_require_sha256(
            compiler_payload["canonicalizer_sha256"],
            field="compiler_identity.canonicalizer_sha256",
        ),
    )
    if (
        compiler.implementation_sha256 != pass_through.implementation_sha256
        or compiler.config_sha256 != artifact.routing_policy.semantic_sha256
        or compiler.operator_contract_sha256 != artifact.editing_corpus_contract.semantic_sha256
        or compiler.canonicalizer_sha256 != pass_through.identity_sha256
    ):
        raise EditingV2CandidateProvenanceDecisionsError(
            "compiler_identity must bind the candidate materializer, routing policy, "
            "editing-corpus contract, and pass-through identity exactly"
        )
    sources = _source_assets(payload["source_assets"])
    return CandidateProvenanceDecisions(
        decision_id=_require_text(payload["decision_id"], field="decision_id"),
        status=DECISIONS_STATUS,
        training_authorized=False,
        candidate_artifact=artifact,
        candidate_stage_pass_through=pass_through,
        compiler_identity=compiler,
        policy_bindings=_policy_bindings(payload["policy_bindings"]),
        evidence_identity=_evidence_identity(payload["evidence_identity"]),
        identity_definitions=_identity_definitions(payload["identity_definitions"]),
        source_assets=sources,
        decision_sha256=supplied,
    )


def load_candidate_provenance_decisions(
    path: str | Path,
) -> CandidateProvenanceDecisions:
    """Load a JSON decision unit and validate every frozen identity."""

    source = Path(path)
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2CandidateProvenanceDecisionsError(
            f"cannot load candidate provenance decisions: {source}"
        ) from error
    return validate_candidate_provenance_decisions(payload)


__all__ = [
    "CandidateProvenanceDecisions",
    "EditingV2CandidateProvenanceDecisionsError",
    "candidate_provenance_decisions_self_hash",
    "load_candidate_provenance_decisions",
    "validate_candidate_provenance_decisions",
]
