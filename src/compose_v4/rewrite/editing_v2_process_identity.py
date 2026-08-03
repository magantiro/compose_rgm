"""Content identity for the frozen Editing-V2 semantic processes.

Two semantic process *versions* share the ``editing_v2`` lane.  ``editing_v2``
names the LANE; the trailing ``_v1`` / ``_v2`` names the SEMANTIC PROCESS
VERSION.

* **V1** -- ``semantic_editing_v2_v1``, contract
  ``configs/editing_v2_semantic_process_v1.json``.  Its dense ``atom_delete``
  mask excludes every cyclic atom before executor validation, so it admits only
  root, singleton, and leaf slots, and it applies no charge gate to them.
* **V2** -- ``semantic_editing_v2_v2``, contract
  ``configs/editing_v2_semantic_process_v2.json``.  **One** admission authority
  in :mod:`compose_v4.rewrite.process_v2_atom_delete` decides *every*
  ``atom_delete`` candidate.  Root, singleton, and leaf deletion stays reachable
  as a capability, but only through the same exact gates every other candidate
  passes; connected-nonleaf candidates additionally carry the frozen aromatic,
  articulation, and SCAR-incidence exclusions.

The correction round matters for lineage.  An earlier candidate Process-V2
reading preserved the inherited root/singleton/leaf admission *set* bit-for-bit
and therefore exempted it from the authoritative charge policy, preserving a
legacy defect.  That candidate contract and the identity it computed to
(:data:`REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256`) were rejected before any
downstream run, so they are recorded as a rejected pre-run candidate, never as a
superseded production identity.

Both identities are computed by one private helper, so their field set and
derivation cannot drift apart.  They are distinguished by ``schema`` and by
``process_semantics``, never by the hash alone: a relabelled V1 object can never
be read as a V2 object.

The V1 identity *definition* -- schema string, schema version, semantics string,
contract path, implementation-source list, body key set -- is unchanged by the
V2 addition.  The V1 identity *value* does move whenever a listed implementation
source changes, which is the declared downstream-invalidation mechanism recorded
in the V2 contract, not a defect.

``scripts/verify_process_v2_hash_chain.py`` is the read-only tool that walks the
complete transitive re-pin chain and reports every disagreement between a pinned
pointer and the live value it addresses.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from compose_v4.rewrite.action_codec_v4 import (
    ACTIVE8_EXECUTOR_RULES,
    codec_implementation_hash,
)
from compose_v4.rewrite.action_codec_v4 import (
    SCHEMA_VERSION as ACTION_CODEC_SCHEMA_VERSION,
)

# ---- V1 semantic process (unchanged) ----
PROCESS_IDENTITY_SCHEMA = "compose.editing.semantic_process_identity"
PROCESS_IDENTITY_SCHEMA_VERSION = 1
PROCESS_SEMANTICS = "semantic_editing_v2_v1"
_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v1.json"
_IMPLEMENTATION_RELATIVE_PATHS = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/data/charge_policy.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/reference_successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/aromatic_kekule.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/ring_restate_semantics.py",
    "src/compose_v4/rewrite/semantic_atom_restate.py",
    "src/compose_v4/rewrite/semantic_cycle_close.py",
    "src/compose_v4/rewrite/semantic_cycle_open.py",
    "src/compose_v4/rewrite/semantic_trace.py",
    "src/compose_v4/rewrite/trace_shard_v3.py",
    "src/compose_v4/rewrite/tracelet_fiber.py",
)

# ---- V2 semantic process (connected-nonleaf atom deletion) ----
PROCESS_V2_IDENTITY_SCHEMA = "compose.editing.semantic_process_v2_identity"
PROCESS_V2_IDENTITY_SCHEMA_VERSION = 1
PROCESS_V2_SEMANTICS = "semantic_editing_v2_v2"
PROCESS_V2_CONTRACT_SCHEMA = "compose.editing.semantic_process_contract"
PROCESS_V2_CONTRACT_SCHEMA_VERSION = 2
PROCESS_V2_CONTRACT_STATUS = (
    "DESIGN_FROZEN_PROCESS_V2_SUPPORT_DECISION_NOT_TRAINING_OR_EXPERIMENT_AUTHORIZED"
)
# Measured on the branch base, before any Process-V2 edit.  This is the V1
# identity the IMMUTABLE V1 migration payloads were built under, so a historical
# V1 artifact stays readable under V1 only.  It is a genuine superseded
# production identity, unlike the rejected candidate below.
SUPERSEDED_V1_PROCESS_IDENTITY_SHA256 = (
    "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
)
# The Process-V2 identity computed by the FIRST, REJECTED candidate reading, in
# which inherited root/singleton/leaf candidates were exempted from the
# authoritative charge policy.  It was rejected before any downstream run and
# produced no downstream artifact: no payload, receipt, cache, checkpoint,
# Gate-0, T1, or P50 object was ever built under it.  It is recorded so the
# value can never be mistaken for a current or superseded production identity.
REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256 = (
    "9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b"
)
# The model-side atom-delete action-semantics modes.  The production constants
# live in ``compose_v4.model.factorized_tracelet_rate_model``; that module cannot
# be imported here (it imports the Process-V2 resolver, which imports this
# module through ``trace_shard_v3``), so the values are restated and
# ``scripts/verify_process_v2_hash_chain.py`` asserts them against the model
# source.  The contract binds the exact values so a mode string can never be
# introduced unbound.
LEGACY_ATOM_DELETE_ACTION_SEMANTICS = "legacy_acyclic_atom_delete_v1"
PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS = "process_v2_uniform_gated_atom_delete_v2"
# The candidate mode name the rejected reading used.  It falsely implied that
# inherited candidates were unfiltered, so it is recorded only as a removed
# name; nothing may alias it.
REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS = "process_v2_connected_nonleaf_atom_delete_v1"
# The corrected contract restructures the ``atom_delete`` body from a disjoint
# union of two admission rules into one admission authority.  ``schema_version``
# names the SEMANTIC PROCESS version (V1 -> 1, V2 -> 2), not the body shape, so
# the body revision is recorded separately rather than by bumping it.
PROCESS_V2_CONTRACT_REVISION = "process_v2_uniform_gated_admission_authority"
REJECTED_PROCESS_V2_CONTRACT_REVISION = "process_v2_disjoint_union_preserving_v1_admission"

# The frozen public API of the admission authority.  These names are contract
# values: the contract binds them, and
# ``scripts/verify_process_v2_hash_chain.py`` asserts each one is defined in the
# resolver module.
PROCESS_V2_ATOM_DELETE_RESOLVER = "resolve_process_v2_atom_delete"
PROCESS_V2_ATOM_DELETE_ENUMERATOR = "enumerate_process_v2_atom_deletes"
PROCESS_V2_ATOM_DELETE_MASK = "process_v2_atom_delete_mask"
# The two structural candidate sources.  They are DIAGNOSTIC LABELS: they select
# which additional gates apply, and they never constitute two admission rules.
PROCESS_V2_INHERITED_CANDIDATE_SOURCE = "inherited_root_singleton_leaf"
PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE = "connected_nonleaf"
# The complete reason-code list, frozen by the correction decision.  The live
# ``ProcessV2AtomDeleteRejectionCode`` enum is not imported here: the builder
# must stay deterministic while the resolver is being edited, so the contract is
# the authority and ``scripts/verify_process_v2_hash_chain.py`` asserts the enum
# against it.
# Declaration order, NOT evaluation order.  The gates are conjunctive, so the
# order changes only which code is reported for a slot that fails more than
# one gate, never whether it is admitted.  Two measured consequences of the
# frozen evaluation order are recorded in the resolver's docstring: an
# articulation exclusion reports ``successor_disconnected`` because
# connectivity is evaluated first, and three codes are defence in depth that
# never fire on the lead panel.
PROCESS_V2_ATOM_DELETE_REJECTION_CODES = (
    "invalid_source",
    "invalid_slot",
    "not_a_real_element",
    "aromatic_atom",
    "articulation_point",
    "scar_incident",
    "executor_rejected",
    "successor_disconnected",
    "charge_policy_violated",
    "successor_outside_declared_support",
    "successor_not_canonicalizable",
)
# Removed by the correction: the superseded reading needed a code for "this slot
# is not in the connected-nonleaf expansion", which only exists when the
# expansion is a separate admission rule.
REJECTED_PROCESS_V2_ATOM_DELETE_REJECTION_CODES = ("outside_connected_nonleaf_expansion",)

_PROCESS_V2_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v2.json"
_PROCESS_V2_ATOM_DELETE_RELATIVE_PATH = "src/compose_v4/rewrite/process_v2_atom_delete.py"
# Exactly the V1 implementation boundary plus the Process-V2 resolver.
_PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS = tuple(
    sorted({*_IMPLEMENTATION_RELATIVE_PATHS, _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH})
)
_UNCHANGED_EXECUTOR_RELATIVE_PATHS = (
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
)
_ATOM_DELETE_MASK_RELATIVE_PATHS = (
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH,
)

# ---- Frozen identity body shape (shared by both process versions) ----
_IDENTITY_BODY_FIELDS = (
    "schema",
    "schema_version",
    "process_semantics",
    "contract_relative_path",
    "contract_sha256",
    "contract_physical_sha256",
    "action_codec_schema_version",
    "action_codec_implementation_hash",
    "active_executor_rules",
    "implementation_source_sha256",
)
_IDENTITY_FIELDS = frozenset({*_IDENTITY_BODY_FIELDS, "process_identity_sha256"})
# Every declared identity schema and the triple it is definitionally bound to.
# These are schema constraints on the object itself; validating them reads no
# file and compares nothing against live source.
_FROZEN_IDENTITY_SCHEMAS: dict[str, tuple[int, str, str]] = {
    PROCESS_IDENTITY_SCHEMA: (
        PROCESS_IDENTITY_SCHEMA_VERSION,
        PROCESS_SEMANTICS,
        _CONTRACT_RELATIVE_PATH,
    ),
    PROCESS_V2_IDENTITY_SCHEMA: (
        PROCESS_V2_IDENTITY_SCHEMA_VERSION,
        PROCESS_V2_SEMANTICS,
        _PROCESS_V2_CONTRACT_RELATIVE_PATH,
    ),
}


class EditingV2ProcessIdentityError(ValueError):
    """The frozen process contract or implementation boundary is incomplete."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


def _contract_self_hash(contract: dict[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )


def _read_frozen_contract(contract_relative_path: str) -> tuple[bytes, dict[str, Any], str]:
    """Read one self-hashed semantic process contract and verify its self-hash."""

    contract_path = _repository_root() / contract_relative_path
    try:
        contract_bytes = contract_path.read_bytes()
        contract = json.loads(contract_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise EditingV2ProcessIdentityError(
            f"cannot read frozen semantic process contract {contract_path}"
        ) from error
    if not isinstance(contract, dict):
        raise EditingV2ProcessIdentityError(
            f"semantic process contract must be an object: {contract_relative_path}"
        )
    expected_contract_sha256 = _contract_self_hash(contract)
    if contract.get("contract_sha256") != expected_contract_sha256:
        raise EditingV2ProcessIdentityError(
            "semantic process contract self-hash does not match its contents: "
            f"{contract_relative_path}"
        )
    try:
        declared_rules = tuple(contract["action_codec"]["public_editing_v2_rules"])
    except (KeyError, TypeError) as error:
        raise EditingV2ProcessIdentityError(
            "semantic process contract does not declare "
            f"action_codec.public_editing_v2_rules: {contract_relative_path}"
        ) from error
    if frozenset(declared_rules) != frozenset(ACTIVE8_EXECUTOR_RULES):
        raise EditingV2ProcessIdentityError(
            "semantic process contract and ActionCodecV4 Active8 rules disagree: "
            f"{contract_relative_path}"
        )
    return contract_bytes, contract, expected_contract_sha256


def _implementation_source_sha256(relative_paths: tuple[str, ...]) -> dict[str, str]:
    root = _repository_root()
    source_sha256: dict[str, str] = {}
    for relative_path in relative_paths:
        try:
            source_sha256[relative_path] = _sha256_bytes((root / relative_path).read_bytes())
        except OSError as error:
            raise EditingV2ProcessIdentityError(
                f"semantic process identity source is missing: {relative_path}"
            ) from error
    return source_sha256


def _process_identity(
    *,
    schema: str,
    schema_version: int,
    process_semantics: str,
    contract_relative_path: str,
    implementation_relative_paths: tuple[str, ...],
) -> dict[str, object]:
    """Build one full-SHA process identity whose change invalidates artifacts."""

    contract_bytes, _, expected_contract_sha256 = _read_frozen_contract(contract_relative_path)
    body: dict[str, object] = {
        "schema": schema,
        "schema_version": schema_version,
        "process_semantics": process_semantics,
        "contract_relative_path": contract_relative_path,
        "contract_sha256": expected_contract_sha256,
        "contract_physical_sha256": _sha256_bytes(contract_bytes),
        "action_codec_schema_version": ACTION_CODEC_SCHEMA_VERSION,
        "action_codec_implementation_hash": codec_implementation_hash(),
        "active_executor_rules": list(ACTIVE8_EXECUTOR_RULES),
        "implementation_source_sha256": _implementation_source_sha256(
            implementation_relative_paths
        ),
    }
    return {**body, "process_identity_sha256": _canonical_sha256(body)}


@lru_cache(maxsize=1)
def editing_v2_process_identity() -> dict[str, object]:
    """Return the V1 full-SHA identity that invalidates semantic artifacts."""

    return _process_identity(
        schema=PROCESS_IDENTITY_SCHEMA,
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        process_semantics=PROCESS_SEMANTICS,
        contract_relative_path=_CONTRACT_RELATIVE_PATH,
        implementation_relative_paths=_IMPLEMENTATION_RELATIVE_PATHS,
    )


def require_editing_v2_process_identity(expected_sha256: str) -> dict[str, object]:
    identity = editing_v2_process_identity()
    if identity["process_identity_sha256"] != expected_sha256:
        raise EditingV2ProcessIdentityError(
            "Editing-V2 process identity differs from the bound artifact identity"
        )
    return identity


@lru_cache(maxsize=1)
def editing_process_v2_identity() -> dict[str, object]:
    """Return the Process-V2 full-SHA identity for the expanded delete fiber.

    It binds the V2 contract, the complete V1 implementation boundary, and the
    Process-V2 connected-nonleaf resolver.  Its ``schema`` and
    ``process_semantics`` differ from V1's, so the two identity objects are
    distinguishable without relying on their hashes.
    """

    return _process_identity(
        schema=PROCESS_V2_IDENTITY_SCHEMA,
        schema_version=PROCESS_V2_IDENTITY_SCHEMA_VERSION,
        process_semantics=PROCESS_V2_SEMANTICS,
        contract_relative_path=_PROCESS_V2_CONTRACT_RELATIVE_PATH,
        implementation_relative_paths=_PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS,
    )


def require_editing_process_v2_identity(expected_sha256: str) -> dict[str, object]:
    identity = editing_process_v2_identity()
    if identity["process_identity_sha256"] != expected_sha256:
        raise EditingV2ProcessIdentityError(
            "Process-V2 semantic process identity differs from the bound artifact identity"
        )
    return identity


def validate_frozen_process_identity(payload: Mapping[str, Any]) -> dict[str, object]:
    """Validate a HISTORICAL process-identity object for self-consistency only.

    This is how an immutable payload's own pinned identity is accepted, for
    example the superseded V1 identity carried by a V1 migration payload.  It
    checks that the object declares a known identity schema, that its schema
    version, ``process_semantics``, and ``contract_relative_path`` are the ones
    that schema is definitionally bound to, that it carries exactly the frozen
    field set, and that ``process_identity_sha256`` recomputes over its own body.

    It reads no file and compares nothing against live source, so it
    DELIBERATELY DOES NOT PROVE CURRENCY: a payload that passes here may still
    describe a superseded process.  Use
    :func:`require_editing_v2_process_identity` or
    :func:`require_editing_process_v2_identity` when currency is required.

    Raises:
        EditingV2ProcessIdentityError: on any inconsistency.
    """

    if not isinstance(payload, Mapping):
        raise EditingV2ProcessIdentityError(
            f"frozen process identity must be a mapping, got {type(payload).__name__}"
        )
    schema = payload.get("schema")
    if not isinstance(schema, str) or schema not in _FROZEN_IDENTITY_SCHEMAS:
        raise EditingV2ProcessIdentityError(
            f"unknown frozen process identity schema: {schema!r}"
        )
    present = frozenset(payload)
    if present != _IDENTITY_FIELDS:
        missing = sorted(_IDENTITY_FIELDS - present)
        unexpected = sorted(present - _IDENTITY_FIELDS)
        raise EditingV2ProcessIdentityError(
            f"frozen process identity {schema} field set is wrong: "
            f"missing={missing} unexpected={unexpected}"
        )
    schema_version, process_semantics, contract_relative_path = _FROZEN_IDENTITY_SCHEMAS[schema]
    for field, expected in (
        ("schema_version", schema_version),
        ("process_semantics", process_semantics),
        ("contract_relative_path", contract_relative_path),
    ):
        if payload[field] != expected:
            raise EditingV2ProcessIdentityError(
                f"frozen process identity {schema} declares {field}={payload[field]!r}, "
                f"which is not the {expected!r} that schema is bound to"
            )
    body = {field: payload[field] for field in _IDENTITY_BODY_FIELDS}
    try:
        recomputed = _canonical_sha256(body)
    except (TypeError, ValueError) as error:
        raise EditingV2ProcessIdentityError(
            f"frozen process identity {schema} body is not canonically serializable"
        ) from error
    if payload["process_identity_sha256"] != recomputed:
        raise EditingV2ProcessIdentityError(
            f"frozen process identity {schema} self-hash does not match its own body"
        )
    return dict(payload)


# ---- Prospective Process-V2 semantic process contract ----


def build_editing_process_v2_contract() -> dict[str, Any]:
    """Build the Process-V2 semantic process contract deterministically.

    Every settled value is read from its authority rather than retyped: the
    Active8 rule set and codec identity from :mod:`action_codec_v4`, the
    unchanged codec/scope/model blocks from the V1 contract file, the
    connected-nonleaf degree constant from
    :mod:`compose_v4.rewrite.process_v2_atom_delete`, the size bound from
    :mod:`compose_v4.rewrite.trace_shard_v3`, the vocabulary width from
    :mod:`compose_v4.chem.molecular_graph`, and the charge-policy version from
    :mod:`compose_v4.data.charge_policy`.

    Three groups of settled values are contract-owned rather than imported: the
    atom-delete mode strings, the resolver/enumerator/mask names, and the
    reason-code list.  Importing them would make the contract track whatever the
    resolver currently says, which is the wrong direction for a frozen decision
    and non-deterministic while the resolver is being edited.
    ``scripts/verify_process_v2_hash_chain.py`` asserts each of them against the
    live implementation, so the binding is enforced without inverting it.

    The returned mapping includes ``contract_sha256``, its own semantic
    self-hash over every other field, so the result can be serialized directly.
    """

    # Imported here, not at module scope: ``trace_shard_v3`` imports this module,
    # so ``process_v2_atom_delete`` (which imports ``trace_shard_v3``) would close
    # an import cycle.  Keeping the builder's imports local leaves this module's
    # import graph exactly as it was.
    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
    from compose_v4.rewrite.process_v2_atom_delete import CONNECTED_NONLEAF_MINIMUM_DEGREE
    from compose_v4.rewrite.trace_shard_v3 import MAX_ACTIVE_ATOMS

    _, v1_contract, v1_contract_sha256 = _read_frozen_contract(_CONTRACT_RELATIVE_PATH)
    v1_contract_physical_sha256 = _sha256_bytes(
        (_repository_root() / _CONTRACT_RELATIVE_PATH).read_bytes()
    )
    scope = dict(v1_contract["scope"])
    if int(scope["max_active_atoms"]) != int(MAX_ACTIVE_ATOMS):
        raise EditingV2ProcessIdentityError(
            "V1 contract scope.max_active_atoms and trace_shard_v3.MAX_ACTIVE_ATOMS disagree"
        )
    model_contract = dict(v1_contract["model_contract"])
    model_contract["action_semantics_modes"] = {
        **dict(model_contract["action_semantics_modes"]),
        "editing_v2": PROCESS_V2_SEMANTICS,
    }
    executor_sha256 = _implementation_source_sha256(_UNCHANGED_EXECUTOR_RELATIVE_PATHS)
    mask_sha256 = _implementation_source_sha256(_ATOM_DELETE_MASK_RELATIVE_PATHS)

    body: dict[str, Any] = {
        "action_codec": dict(v1_contract["action_codec"]),
        "action_codec_identity": {
            "active_executor_rules": list(ACTIVE8_EXECUTOR_RULES),
            "changed_by_process_v2": False,
            "implementation_hash": codec_implementation_hash(),
            "schema_version": ACTION_CODEC_SCHEMA_VERSION,
            "source_sha256": _implementation_source_sha256(
                ("src/compose_v4/rewrite/action_codec_v4.py",)
            ),
        },
        "atom_delete": {
            "action_type": "AtomDelete",
            "action_semantics_modes": {
                "legacy": LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
                "process_v2": PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
            },
            "admission_authority": {
                "candidate_sources_are_diagnostic_labels_only": True,
                "decides": "every_process_v2_atom_delete_candidate",
                "enumerator": PROCESS_V2_ATOM_DELETE_ENUMERATOR,
                "implementation_binding": (
                    "the_resolver_enumerator_mask_and_reason_codes_declared_here_are_"
                    "asserted_against_the_live_module_by_"
                    "scripts/verify_process_v2_hash_chain.py"
                ),
                "is_a_disjoint_union_of_two_admission_rules": False,
                "mask": PROCESS_V2_ATOM_DELETE_MASK,
                "mask_is_the_complete_effective_mask_not_an_extension": True,
                "module": _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH,
                "rejection_codes": list(
                    PROCESS_V2_ATOM_DELETE_REJECTION_CODES
                ),
                "resolver": PROCESS_V2_ATOM_DELETE_RESOLVER,
                "single_authority_over_all_candidates": True,
            },
            "aromatic_connected_nonleaf_policy": {
                "admitted": False,
                "perception": "resonance_invariant_bond_classes",
                "reason": (
                    "deleting_one_kekule_encoded_slot_of_a_perceived_aromatic_system_"
                    "yields_a_representation_sensitive_open_chain_successor"
                ),
                "requires_separate_future_semantic_decision_and_resolver": True,
            },
            "candidate_sources": {
                PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE: {
                    "definition": (
                        "real_slot_whose_real_atom_degree_is_at_least_the_minimum_degree"
                    ),
                    "diagnostic_label_only": True,
                    "introduced_by_process_v2": True,
                    "minimum_real_atom_degree": int(CONNECTED_NONLEAF_MINIMUM_DEGREE),
                    "minimum_real_atom_degree_constant": "CONNECTED_NONLEAF_MINIMUM_DEGREE",
                    "selects_additional_gates": True,
                },
                PROCESS_V2_INHERITED_CANDIDATE_SOURCE: {
                    "definition": "real_slot_whose_real_atom_degree_is_at_most_one",
                    "diagnostic_label_only": True,
                    "introduced_by_process_v2": False,
                    "maximum_real_atom_degree": int(CONNECTED_NONLEAF_MINIMUM_DEGREE) - 1,
                    "selects_additional_gates": False,
                },
            },
            "common_exact_gates_in_order": [
                "real_element_under_the_authoritative_element_predicate",
                "unchanged_production_atom_delete_executor_accepts_the_operation",
                "exact_persistent_slot_successor_is_connected_or_null",
                "authoritative_charge_policy_is_preserved_by_the_transition",
                "successor_is_within_the_declared_broad_organic_bounded_size_support",
                "successor_is_canonicalizable",
            ],
            "common_gates_apply_to_every_candidate_source": True,
            "connected_nonleaf_only_gates_in_order": [
                "non_aromatic_under_the_frozen_production_representation",
                "not_a_graph_articulation_point_of_the_real_atom_graph",
                "not_incident_to_any_scar_slot",
            ],
            "effective_mask_rule": {
                "batch_mask_equals_the_admission_authority_array": True,
                "definition": (
                    "the_process_v2_effective_atom_delete_mask_equals_"
                    "process_v2_atom_delete_mask_exactly"
                ),
                "model_forward_asserts_equality_not_subset": True,
                "v1_dense_mask_is_the_effective_process_v2_mask": False,
            },
            "executor_rule": "atom_delete",
            "executor_validity_is_a_connectivity_predicate": False,
            "family": "atom_delete",
            "frozen_non_goals": [
                "change_atom_delete_executor_semantics",
                "change_persistent_slot_identity_or_canonicalization",
                "change_formal_charge_policy",
                "admit_aromatic_connected_nonleaf_deletion",
                "admit_scar_incident_connected_nonleaf_deletion",
                "add_multi_neighbour_atom_insertion",
                "enable_ring_system_delete_or_ring_system_grow",
                "change_any_other_active8_operator_semantics",
                "relabel_v1_artifacts_as_v2",
            ],
            "inherited_capability": {
                "admission_set_preserved_bit_for_bit": False,
                "capability_remains_reachable": True,
                "covers": ["root", "singleton", "leaf"],
                "exempt_from_the_authoritative_charge_policy": False,
                "meaning": (
                    "preserving_the_capability_means_root_singleton_and_leaf_deletion_"
                    "stays_reachable_it_does_not_mean_preserving_an_unfiltered_"
                    "admission_set"
                ),
                "superseded_v1_dense_mask_rule": (
                    "real_slot_with_atom_topology_zero_and_not_an_articulation_point_and_"
                    "no_neighbour_implicit_hydrogen_exceeding_max_h_count"
                ),
            },
            "legality_authority": (
                "the_unchanged_production_atom_delete_executor_is_the_legality_authority_"
                "and_no_weaker_approximate_valence_test_may_be_substituted_for_it"
            ),
            "one_step_inverse_closure_claimed": False,
            "scar_incidence_policy": {
                "admitted": False,
                "applies_to_candidate_sources": [
                    PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE
                ],
                "constant": "compose_v4.chem.molecular_graph.SCAR_IDX",
                "connected_nonleaf_only_reason": (
                    "a_scar_adjacent_leaf_deletion_was_already_reachable_under_v1_so_"
                    "excluding_it_would_withdraw_an_inherited_capability_the_recorded_"
                    "decision_does_not_withdraw_while_a_scar_adjacent_ring_deletion_is_"
                    "newly_introduced_by_process_v2"
                ),
                "definition": "any_neighbour_slot_whose_atom_type_is_the_scar_marker",
                "reason": (
                    "apply_atom_delete_writes_implicit_hydrogen_onto_the_scar_neighbour_"
                    "which_is_unsettled"
                ),
                "requires_separate_future_scar_semantic_decision": True,
            },
            "superseded_reading": {
                "contract_revision": REJECTED_PROCESS_V2_CONTRACT_REVISION,
                "defect": (
                    "it_read_preserve_existing_root_singleton_and_leaf_behaviour_as_"
                    "preserve_the_admission_set_bit_for_bit_and_therefore_exempted_"
                    "inherited_candidates_from_the_authoritative_charge_policy_"
                    "preserving_a_legacy_defect"
                ),
                "removed_action_semantics_mode": (
                    REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
                ),
                "removed_action_semantics_mode_defect": (
                    "the_name_falsely_implied_that_inherited_candidates_were_unfiltered"
                ),
                "removed_action_semantics_mode_may_be_aliased": False,
                "removed_rejection_codes": list(
                    REJECTED_PROCESS_V2_ATOM_DELETE_REJECTION_CODES
                ),
            },
        },
        "atom_delete_mask_implementation": {
            "admission_authority_module": _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH,
            "dense_mask_module": "src/compose_v4/model/factorized_tracelet_rate_model.py",
            "dense_mask_symbol": "_graph_application_masks",
            "mask_properties": [
                "boolean",
                "deterministic",
                "persistent_slot_addressed",
                "independent_of_canonical_smiles_atom_order",
            ],
            "source_sha256": mask_sha256,
        },
        "authority": {
            "active8_materialization_authorized": False,
            "de_novo_run_authorized": False,
            "experiment_authorized": False,
            "gate0_authorized": False,
            "long_editing_run_authorized": False,
            "modal_launch_authorized": False,
            "p50_authorized": False,
            "t1_authorized": False,
            "training_authorized": False,
        },
        "canonicalizer": {
            "changed_by_process_v2": False,
            "function": "canonical_state_key",
            "module": "src/compose_v4/rewrite/kernel.py",
            "smiles_reconstruction_of_exact_states_forbidden": True,
            "source_sha256": _implementation_source_sha256(
                ("src/compose_v4/rewrite/kernel.py",)
            ),
        },
        "contract_revision": PROCESS_V2_CONTRACT_REVISION,
        "charge_policy": {
            "applied_to": "every_process_v2_atom_delete_candidate",
            "authoritative": True,
            "changed_by_process_v2": False,
            "constant": "compose_v4.data.charge_policy.CHARGE_POLICY_VERSION",
            "exempt_candidate_sources": [],
            "formal_charge_changes": "out_of_scope",
            "is_a_common_exact_gate": True,
            "predicate": "charge_policy_preserved",
            "version": str(CHARGE_POLICY_VERSION),
        },
        "downstream_invalidation": {
            "active8_inventory_rebuild_required": True,
            "affected_whole_trace_replay_required": True,
            "candidate_and_successor_cache_rebuild_required": True,
            "checkpoint_resume_across_process_versions": False,
            "gate0_and_t1_rebuild_required": True,
            "historical_artifacts_remain_readable_only_under_original_identity": True,
            "p50_authorized_by_this_contract": False,
            "source_and_scaffold_splits_changed": False,
            "superseded_v1_process_identity_sha256": SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
            "superseded_v1_process_semantics": PROCESS_SEMANTICS,
            "v1_artifacts_relabelled_as_v2": False,
            "v1_process_contract_edited_in_place": False,
        },
        "inherited_v1_process": {
            "contract_physical_sha256": v1_contract_physical_sha256,
            "contract_relative_path": _CONTRACT_RELATIVE_PATH,
            "contract_sha256": v1_contract_sha256,
            "inherited_blocks": ["action_codec", "model_contract", "scope"],
            "process_semantics": PROCESS_SEMANTICS,
            "unchanged_operator_semantics": [
                "atom_insert",
                "atom_restate",
                "bond_reorder",
                "bond_reroute",
                "cycle_insert",
                "cycle_attach",
                "ring_system_restate",
            ],
        },
        "lineage": {
            "current_process_v2_identity_source": (
                "editing_process_v2_identity()['process_identity_sha256'] recomputed "
                "from live source; this contract never pins its own identity value"
            ),
            "rejected_pre_run_candidate_identity": {
                "atom_delete_action_semantics": (
                    REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
                ),
                "contract_revision": REJECTED_PROCESS_V2_CONTRACT_REVISION,
                "downstream_artifacts_produced": [],
                "is_the_current_process_v2_identity": False,
                "process_identity_sha256": REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
                "process_semantics": PROCESS_V2_SEMANTICS,
                "produced_downstream_artifacts": False,
                "rejected_before_any_downstream_run": True,
                "rejection_reason": (
                    "it_exempted_inherited_root_singleton_and_leaf_candidates_from_the_"
                    "authoritative_charge_policy"
                ),
                "status": "REJECTED_PRE_RUN_CANDIDATE_IDENTITY",
                "was_a_superseded_production_identity": False,
            },
            "superseded_v1_payload_identity": {
                "contract_relative_path": _CONTRACT_RELATIVE_PATH,
                "immutable_v1_payloads_were_built_under_it": True,
                "is_the_current_process_v2_identity": False,
                "process_identity_sha256": SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
                "process_semantics": PROCESS_SEMANTICS,
                "readable_only_under_the_v1_identity_schema": True,
                "status": "SUPERSEDED_V1_PROCESS_IDENTITY",
                "was_a_superseded_production_identity": True,
            },
        },
        "model_contract": model_contract,
        "persistent_slot": {
            "changed_by_process_v2": False,
            "delete_nulls_the_slot_without_compacting_the_state": True,
            "max_active_atoms": int(MAX_ACTIVE_ATOMS),
            "max_active_atoms_constant": "compose_v4.rewrite.trace_shard_v3.MAX_ACTIVE_ATOMS",
            "representation": (
                "the_padded_persistent_slot_tensor_is_a_coordinate_representation_"
                "not_the_semantic_dimension"
            ),
        },
        "process_identity": {
            "identity_schema": PROCESS_V2_IDENTITY_SCHEMA,
            "identity_schema_version": PROCESS_V2_IDENTITY_SCHEMA_VERSION,
            "implementation_source_relative_paths": list(
                _PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS
            ),
            "module": "src/compose_v4/rewrite/editing_v2_process_identity.py",
            "provider": "editing_process_v2_identity",
        },
        "process_semantics": PROCESS_V2_SEMANTICS,
        "schema": PROCESS_V2_CONTRACT_SCHEMA,
        "schema_version": PROCESS_V2_CONTRACT_SCHEMA_VERSION,
        "schema_version_names": "the_semantic_process_version_not_the_body_shape",
        "scope": scope,
        "source_evidence": {
            "inherited_candidate_charge_violation_observation": {
                "authoritative_corpus_evidence": False,
                "conclusion": (
                    "the_authoritative_charge_policy_is_the_whole_gap_between_the_"
                    "superseded_inherited_admission_set_and_the_corrected_one_on_this_"
                    "bounded_panel"
                ),
                "denominator_inherited_candidates": 3319,
                "denominator_source_molecules": 800,
                "evidence_class": "bounded_measured_observation_not_a_corpus_result",
                "excluded_by_the_common_connectivity_gate_beyond_the_charge_gate": 0,
                "excluded_by_the_common_executor_gate_beyond_the_charge_gate": 0,
                "inherited_candidates_violating_the_authoritative_charge_policy": 125,
                "measured_on": "800_jin_qed_leads",
                "source_molecules_with_at_least_one_violation": 105,
                "status": (
                    "bounded_measured_motivation_for_the_correction_it_authorizes_"
                    "nothing_and_is_not_corpus_evidence"
                ),
                "superseded_rule_admitted_the_violating_candidates": True,
                "violating_fraction_percent_rounded_to_two_places": "3.77",
            },
        },
        "status": PROCESS_V2_CONTRACT_STATUS,
        "supersedes_for_editing_v2": [_CONTRACT_RELATIVE_PATH],
        "training_authorized": False,
        "unchanged_executor": {
            "apply_function": "apply_atom_delete",
            "changed_by_process_v2": False,
            "executor_rule": "atom_delete",
            "module": "src/compose_v4/rewrite/operators.py",
            "source_sha256": executor_sha256,
            "validity_function": "is_valid_atom_delete",
            "validity_function_is_a_connectivity_predicate": False,
        },
        "vocabulary": {
            "changed_by_process_v2": False,
            "class_count": len(ORGANIC_VOCABULARY),
            "legacy_cnof_vocabulary_is_production": False,
            "module": "src/compose_v4/chem/molecular_graph.py",
            "name": "ORGANIC_VOCABULARY",
        },
    }
    return {**body, "contract_sha256": _contract_self_hash(body)}


def serialize_editing_process_v2_contract(contract: Mapping[str, Any] | None = None) -> bytes:
    """Serialize the Process-V2 contract deterministically, byte-stable."""

    payload = dict(build_editing_process_v2_contract() if contract is None else contract)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def write_editing_process_v2_contract(path: Path | None = None) -> str:
    """Regenerate the committed Process-V2 contract; return its physical SHA-256.

    The contract records implementation source hashes, so it must be regenerated
    whenever a bound source changes -- that regeneration is the invalidation
    signal, not a convenience.
    """

    target = _repository_root() / _PROCESS_V2_CONTRACT_RELATIVE_PATH if path is None else path
    payload = serialize_editing_process_v2_contract()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.staged")
    try:
        staged.write_bytes(payload)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return _sha256_bytes(payload)


__all__ = [
    "LEGACY_ATOM_DELETE_ACTION_SEMANTICS",
    "PROCESS_IDENTITY_SCHEMA",
    "PROCESS_IDENTITY_SCHEMA_VERSION",
    "PROCESS_SEMANTICS",
    "PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS",
    "PROCESS_V2_ATOM_DELETE_ENUMERATOR",
    "PROCESS_V2_ATOM_DELETE_MASK",
    "PROCESS_V2_ATOM_DELETE_REJECTION_CODES",
    "PROCESS_V2_ATOM_DELETE_RESOLVER",
    "PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE",
    "PROCESS_V2_CONTRACT_REVISION",
    "PROCESS_V2_CONTRACT_SCHEMA",
    "PROCESS_V2_CONTRACT_SCHEMA_VERSION",
    "PROCESS_V2_CONTRACT_STATUS",
    "PROCESS_V2_IDENTITY_SCHEMA",
    "PROCESS_V2_IDENTITY_SCHEMA_VERSION",
    "PROCESS_V2_INHERITED_CANDIDATE_SOURCE",
    "PROCESS_V2_SEMANTICS",
    "REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS",
    "REJECTED_PROCESS_V2_ATOM_DELETE_REJECTION_CODES",
    "REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256",
    "REJECTED_PROCESS_V2_CONTRACT_REVISION",
    "SUPERSEDED_V1_PROCESS_IDENTITY_SHA256",
    "EditingV2ProcessIdentityError",
    "build_editing_process_v2_contract",
    "editing_process_v2_identity",
    "editing_v2_process_identity",
    "require_editing_process_v2_identity",
    "require_editing_v2_process_identity",
    "serialize_editing_process_v2_contract",
    "validate_frozen_process_identity",
    "write_editing_process_v2_contract",
]
