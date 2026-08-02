"""Content identity for the frozen Editing-V2 semantic processes.

Two semantic process *versions* share the ``editing_v2`` lane.  ``editing_v2``
names the LANE; the trailing ``_v1`` / ``_v2`` names the SEMANTIC PROCESS
VERSION.

* **V1** -- ``semantic_editing_v2_v1``, contract
  ``configs/editing_v2_semantic_process_v1.json``.  Its dense ``atom_delete``
  mask excludes every cyclic atom before executor validation, so it admits only
  root, singleton, and leaf slots.
* **V2** -- ``semantic_editing_v2_v2``, contract
  ``configs/editing_v2_semantic_process_v2.json``.  It preserves the V1 root,
  singleton, and leaf rule bit-for-bit and additionally admits executor-verified,
  charge-preserving, non-aromatic, non-articulation connected-nonleaf deletions
  resolved by :mod:`compose_v4.rewrite.process_v2_atom_delete`.

Both identities are computed by one private helper, so their field set and
derivation cannot drift apart.  They are distinguished by ``schema`` and by
``process_semantics``, never by the hash alone: a relabelled V1 object can never
be read as a V2 object.

The V1 identity *definition* -- schema string, schema version, semantics string,
contract path, implementation-source list, body key set -- is unchanged by the
V2 addition.  The V1 identity *value* does move whenever a listed implementation
source changes, which is the declared downstream-invalidation mechanism recorded
in the V2 contract, not a defect.
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
# Measured on the branch base, before any Process-V2 edit.  The two committed
# V1-lane binding configs pin this value; V2 records it as the identity it
# supersedes so a historical V1 artifact stays readable under V1 only.
SUPERSEDED_V1_PROCESS_IDENTITY_SHA256 = (
    "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
)
# The model-side atom-delete action-semantics modes.  The production constants
# live in ``compose_v4.model.factorized_tracelet_rate_model``; the contract binds
# their exact values so a mode string can never be introduced unbound.
LEGACY_ATOM_DELETE_ACTION_SEMANTICS = "legacy_acyclic_atom_delete_v1"
PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS = "process_v2_connected_nonleaf_atom_delete_v1"

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
    unchanged codec/scope/model blocks from the V1 contract file, the expansion
    constant and rejection codes from
    :mod:`compose_v4.rewrite.process_v2_atom_delete`, the size bound from
    :mod:`compose_v4.rewrite.trace_shard_v3`, the vocabulary width from
    :mod:`compose_v4.chem.molecular_graph`, and the charge-policy version from
    :mod:`compose_v4.data.charge_policy`.

    The returned mapping includes ``contract_sha256``, its own semantic
    self-hash over every other field, so the result can be serialized directly.
    """

    # Imported here, not at module scope: ``trace_shard_v3`` imports this module,
    # so ``process_v2_atom_delete`` (which imports ``trace_shard_v3``) would close
    # an import cycle.  Keeping the builder's imports local leaves this module's
    # import graph exactly as it was.
    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
    from compose_v4.rewrite.process_v2_atom_delete import (
        CONNECTED_NONLEAF_MINIMUM_DEGREE,
        ProcessV2AtomDeleteRejectionCode,
    )
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
            "aromatic_connected_nonleaf_policy": {
                "admitted": False,
                "perception": "resonance_invariant_bond_classes",
                "reason": (
                    "deleting_one_kekule_encoded_slot_of_a_perceived_aromatic_system_"
                    "yields_a_representation_sensitive_open_chain_successor"
                ),
                "requires_separate_future_semantic_decision_and_resolver": True,
            },
            "connected_nonleaf_expansion": {
                "admission_conditions": [
                    "real_element_under_the_authoritative_element_predicate",
                    "non_aromatic_under_the_frozen_production_representation",
                    "not_a_graph_articulation_point_of_the_real_atom_graph",
                    "unchanged_production_atom_delete_executor_accepts_the_operation",
                    "exact_persistent_slot_successor_is_connected",
                    "frozen_charge_policy_is_preserved_by_the_transition",
                    "successor_is_within_the_declared_support_and_is_canonicalizable",
                ],
                "disjointness_argument": (
                    "in_a_connected_real_atom_graph_an_acyclic_vertex_of_real_atom_degree_"
                    "at_least_two_is_always_a_cut_vertex_so_the_unchanged_v1_dense_mask_"
                    "admits_no_slot_of_real_atom_degree_at_least_two"
                ),
                "enumerator": "enumerate_process_v2_connected_nonleaf_atom_deletes",
                "mask": "process_v2_connected_nonleaf_atom_delete_mask",
                "minimum_real_atom_degree": int(CONNECTED_NONLEAF_MINIMUM_DEGREE),
                "minimum_real_atom_degree_constant": "CONNECTED_NONLEAF_MINIMUM_DEGREE",
                "module": _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH,
                "rejection_codes_in_evaluation_order": [
                    code.value for code in ProcessV2AtomDeleteRejectionCode
                ],
                "resolver": "resolve_process_v2_connected_nonleaf_atom_delete",
            },
            "effective_mask_rule": {
                "definition": (
                    "process_v2_effective_atom_delete_mask_equals_the_unchanged_v1_dense_"
                    "mask_union_the_connected_nonleaf_mask"
                ),
                "operands_are_disjoint": True,
                "union_is_a_disjoint_union": True,
                "v1_dense_mask_changed": False,
            },
            "executor_rule": "atom_delete",
            "executor_validity_is_a_connectivity_predicate": False,
            "family": "atom_delete",
            "frozen_non_goals": [
                "change_atom_delete_executor_semantics",
                "change_persistent_slot_identity_or_canonicalization",
                "change_formal_charge_policy",
                "admit_aromatic_connected_nonleaf_deletion",
                "add_multi_neighbour_atom_insertion",
                "enable_ring_system_delete_or_ring_system_grow",
                "change_any_other_active8_operator_semantics",
                "relabel_v1_artifacts_as_v2",
            ],
            "legality_authority": (
                "the_unchanged_production_atom_delete_executor_is_the_legality_authority_"
                "and_no_weaker_approximate_valence_test_may_be_substituted_for_it"
            ),
            "one_step_inverse_closure_claimed": False,
            "preserved_v1_admission": {
                "charge_policy_applied_to_preserved_v1_candidates": False,
                "covers": ["root", "singleton", "leaf"],
                "dense_mask_rule": (
                    "real_slot_with_atom_topology_zero_and_not_an_articulation_point_and_"
                    "no_neighbour_implicit_hydrogen_exceeding_max_h_count"
                ),
                "maximum_admitted_real_atom_degree": int(CONNECTED_NONLEAF_MINIMUM_DEGREE) - 1,
                "redecided_by_process_v2": False,
                "uniform_charge_application_would_remove_preserved_v1_candidates": True,
            },
        },
        "atom_delete_mask_implementation": {
            "connected_nonleaf_resolver_module": _PROCESS_V2_ATOM_DELETE_RELATIVE_PATH,
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
        "charge_policy": {
            "applied_to": "connected_nonleaf_expansion_candidates_only",
            "changed_by_process_v2": False,
            "constant": "compose_v4.data.charge_policy.CHARGE_POLICY_VERSION",
            "formal_charge_changes": "out_of_scope",
            "not_applied_to_preserved_v1_candidates_reason": (
                "uniform_application_would_remove_preserved_v1_leaf_candidates_and_"
                "silently_change_admission_behaviour_the_decision_declares_unchanged"
            ),
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
        "scope": scope,
        "source_evidence": {
            "connected_nonleaf_disjointness_development_panel": {
                "authoritative_corpus_evidence": False,
                "maximum_v1_admitted_real_atom_degree": 1,
                "panel_smiles": [
                    "C",
                    "C1CC2CCC1CC2",
                    "C1CCC(CC1)N",
                    "C1CCCCC1",
                    "C1CCOCC1",
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "CC1CCCCC1",
                    "CCCC",
                    "CCO",
                    "C[N+](C)(C)CC(=O)[O-]",
                    "Cc1ccccc1",
                    "O=C1NC(O)C2CCCCC12",
                    "c1ccc2ccccc2c1",
                    "c1ccccc1",
                ],
                "status": (
                    "bounded_development_observation_requires_the_registered_"
                    "process_v2_candidate_mask_gate"
                ),
                "v1_and_connected_nonleaf_overlap_count": 0,
            },
            "uniform_charge_policy_counterexample": {
                "authoritative_corpus_evidence": False,
                "charge_policy_preserving_v1_admitted_slots": [6],
                "conclusion": (
                    "applying_the_charge_policy_uniformly_would_remove_four_of_the_five_"
                    "preserved_v1_leaf_candidates_of_this_source"
                ),
                "connected_nonleaf_admitted_slots": [],
                "source_smiles": "C[N+](C)(C)CC(=O)[O-]",
                "v1_admitted_slots": [0, 2, 3, 6, 7],
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
    "PROCESS_V2_CONTRACT_SCHEMA",
    "PROCESS_V2_CONTRACT_SCHEMA_VERSION",
    "PROCESS_V2_CONTRACT_STATUS",
    "PROCESS_V2_IDENTITY_SCHEMA",
    "PROCESS_V2_IDENTITY_SCHEMA_VERSION",
    "PROCESS_V2_SEMANTICS",
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
