"""Corrected Process-V2 semantic process contract, identity, and lineage.

The correction round replaced a disjoint union of two admission rules with ONE
admission authority over every ``atom_delete`` candidate.  These tests pin the
contract structure that encodes that decision, the coexistence of the V1 and V2
identities, and the lineage record that keeps a rejected pre-run candidate
identity from ever being read as a production identity.

Deliberately NOT asserted here: agreement between the contract's declared
implementation names and the live resolver module.  The contract is the frozen
authority for those names and the resolver is being brought into line with it;
``scripts/verify_process_v2_hash_chain.py`` is what asserts the two agree, and it
is the tool run last, once every source edit is final.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from compose_v4.rewrite import editing_v2_process_identity as identity_module
from compose_v4.rewrite.editing_v2_process_identity import (
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_IDENTITY_SCHEMA,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    PROCESS_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ENUMERATOR,
    PROCESS_V2_ATOM_DELETE_MASK,
    PROCESS_V2_ATOM_DELETE_REJECTION_CODES,
    PROCESS_V2_ATOM_DELETE_RESOLVER,
    PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE,
    PROCESS_V2_CONTRACT_REVISION,
    PROCESS_V2_CONTRACT_SCHEMA,
    PROCESS_V2_CONTRACT_SCHEMA_VERSION,
    PROCESS_V2_CONTRACT_STATUS,
    PROCESS_V2_IDENTITY_SCHEMA,
    PROCESS_V2_IDENTITY_SCHEMA_VERSION,
    PROCESS_V2_INHERITED_CANDIDATE_SOURCE,
    PROCESS_V2_SEMANTICS,
    REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    REJECTED_PROCESS_V2_ATOM_DELETE_REJECTION_CODES,
    REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
    REJECTED_PROCESS_V2_CONTRACT_REVISION,
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    EditingV2ProcessIdentityError,
    build_editing_process_v2_contract,
    editing_process_v2_identity,
    editing_v2_process_identity,
    require_editing_process_v2_identity,
    require_editing_v2_process_identity,
    serialize_editing_process_v2_contract,
    validate_frozen_process_identity,
    write_editing_process_v2_contract,
)
from compose_v4.rewrite.process_v2_atom_delete import CONNECTED_NONLEAF_MINIMUM_DEGREE

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_CONTRACT = Path("configs/editing_v2_semantic_process_v1.json")
V2_CONTRACT = Path("configs/editing_v2_semantic_process_v2.json")
VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_process_v2_hash_chain.py"

# The V1 semantic-process contract is preserved unchanged by Process V2.
V1_CONTRACT_SHA256 = "f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288"
V1_CONTRACT_PHYSICAL_SHA256 = "43cb26e1ba33b27a0149a8142895ef9988853b458c0ace9bc9191f7b460dedbb"

# Lineage values.  Neither is ever the current identity of anything.
SUPERSEDED_V1_PROCESS_IDENTITY = (
    "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
)
REJECTED_PRE_RUN_CANDIDATE_IDENTITY = (
    "9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b"
)
REJECTED_ATOM_DELETE_MODE_NAME = "process_v2_connected_nonleaf_atom_delete_v1"

# The V1 identity *definition*: schema, semantics, contract path, and the exact
# implementation boundary it hashes.  Process V2 must not move any of these.
V1_IMPLEMENTATION_RELATIVE_PATHS = (
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
PROCESS_V2_RESOLVER_RELATIVE_PATH = "src/compose_v4/rewrite/process_v2_atom_delete.py"

IDENTITY_BODY_FIELDS = (
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
IDENTITY_FIELDS = frozenset({*IDENTITY_BODY_FIELDS, "process_identity_sha256"})


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hashed(body: dict[str, Any]) -> dict[str, Any]:
    """Return an identity body plus a self-hash that is internally consistent."""

    return {**body, "process_identity_sha256": _semantic_sha256(body)}


def _contract() -> dict[str, Any]:
    return json.loads(V2_CONTRACT.read_text())


def _v1_identity() -> dict[str, Any]:
    return dict(editing_v2_process_identity())


def _v2_identity() -> dict[str, Any]:
    return dict(editing_process_v2_identity())


def _clear_identity_caches() -> None:
    editing_v2_process_identity.cache_clear()
    editing_process_v2_identity.cache_clear()


@functools.lru_cache(maxsize=1)
def _load_verifier() -> ModuleType:
    """Import the verifier script by path, as a normal module.

    It must be registered in ``sys.modules`` before execution: it defines
    dataclasses under ``from __future__ import annotations``, and resolving those
    string annotations requires the defining module to be importable by name.
    """

    name = "verify_process_v2_hash_chain"
    spec = importlib.util.spec_from_file_location(name, VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---- The committed Process-V2 contract ----


def test_committed_v2_contract_matches_a_fresh_deterministic_rebuild() -> None:
    committed = V2_CONTRACT.read_bytes()
    rebuilt = serialize_editing_process_v2_contract()
    assert committed == rebuilt, (
        "configs/editing_v2_semantic_process_v2.json has drifted from "
        "build_editing_process_v2_contract(); regenerate it with "
        "write_editing_process_v2_contract()"
    )
    assert _file_sha256(V2_CONTRACT) == hashlib.sha256(rebuilt).hexdigest()
    # Byte-stable across repeated builds, not merely equal to the committed file.
    assert serialize_editing_process_v2_contract() == rebuilt


def test_contract_writer_publishes_atomically_and_idempotently(tmp_path: Path) -> None:
    """The documented regeneration entry point must leave no partial output."""

    target = tmp_path / "nested" / "editing_v2_semantic_process_v2.json"
    digest = write_editing_process_v2_contract(target)
    payload = target.read_bytes()
    assert digest == hashlib.sha256(payload).hexdigest()
    assert payload == serialize_editing_process_v2_contract()
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]
    assert write_editing_process_v2_contract(target) == digest
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]


def test_committed_v2_contract_semantic_self_hash_validates() -> None:
    contract = _contract()
    claimed = contract.pop("contract_sha256")
    assert _semantic_sha256(contract) == claimed
    assert claimed == editing_process_v2_identity()["contract_sha256"]
    assert build_editing_process_v2_contract()["contract_sha256"] == claimed


def test_v2_contract_declares_no_training_or_experiment_authority() -> None:
    contract = _contract()
    assert contract["training_authorized"] is False
    assert contract["status"] == PROCESS_V2_CONTRACT_STATUS
    assert "NOT_TRAINING_OR_EXPERIMENT_AUTHORIZED" in contract["status"]
    authority = contract["authority"]
    assert set(authority) == {
        "active8_materialization_authorized",
        "de_novo_run_authorized",
        "experiment_authorized",
        "gate0_authorized",
        "long_editing_run_authorized",
        "modal_launch_authorized",
        "p50_authorized",
        "t1_authorized",
        "training_authorized",
    }
    assert all(value is False for value in authority.values())
    assert contract["downstream_invalidation"]["p50_authorized_by_this_contract"] is False


def test_v2_contract_binds_the_declared_process_v2_schema_and_semantics() -> None:
    contract = _contract()
    assert contract["schema"] == PROCESS_V2_CONTRACT_SCHEMA
    assert contract["schema"] == "compose.editing.semantic_process_contract"
    assert contract["schema_version"] == PROCESS_V2_CONTRACT_SCHEMA_VERSION == 2
    assert contract["schema_version_names"] == "the_semantic_process_version_not_the_body_shape"
    assert contract["contract_revision"] == PROCESS_V2_CONTRACT_REVISION
    assert contract["contract_revision"] != REJECTED_PROCESS_V2_CONTRACT_REVISION
    assert contract["process_semantics"] == PROCESS_V2_SEMANTICS == "semantic_editing_v2_v2"
    assert contract["model_contract"]["action_semantics_modes"]["editing_v2"] == (
        PROCESS_V2_SEMANTICS
    )
    assert contract["process_identity"]["identity_schema"] == PROCESS_V2_IDENTITY_SCHEMA
    assert contract["process_identity"]["provider"] == "editing_process_v2_identity"


# ---- One admission authority, not a disjoint union ----


def test_v2_contract_binds_one_admission_authority_over_every_candidate() -> None:
    authority = _contract()["atom_delete"]["admission_authority"]

    assert authority["single_authority_over_all_candidates"] is True
    assert authority["is_a_disjoint_union_of_two_admission_rules"] is False
    assert authority["decides"] == "every_process_v2_atom_delete_candidate"
    assert authority["candidate_sources_are_diagnostic_labels_only"] is True
    assert authority["mask_is_the_complete_effective_mask_not_an_extension"] is True
    assert authority["module"] == PROCESS_V2_RESOLVER_RELATIVE_PATH
    assert authority["resolver"] == PROCESS_V2_ATOM_DELETE_RESOLVER
    assert authority["enumerator"] == PROCESS_V2_ATOM_DELETE_ENUMERATOR
    assert authority["mask"] == PROCESS_V2_ATOM_DELETE_MASK
    assert "verify_process_v2_hash_chain.py" in authority["implementation_binding"]


def test_v2_contract_candidate_sources_are_diagnostic_labels_only() -> None:
    sources = _contract()["atom_delete"]["candidate_sources"]

    assert set(sources) == {
        PROCESS_V2_INHERITED_CANDIDATE_SOURCE,
        PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE,
    }
    assert set(sources) == {"inherited_root_singleton_leaf", "connected_nonleaf"}
    assert all(source["diagnostic_label_only"] is True for source in sources.values())

    inherited = sources[PROCESS_V2_INHERITED_CANDIDATE_SOURCE]
    nonleaf = sources[PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE]
    assert inherited["introduced_by_process_v2"] is False
    assert inherited["selects_additional_gates"] is False
    assert nonleaf["introduced_by_process_v2"] is True
    assert nonleaf["selects_additional_gates"] is True
    assert nonleaf["minimum_real_atom_degree"] == CONNECTED_NONLEAF_MINIMUM_DEGREE == 2
    assert inherited["maximum_real_atom_degree"] == CONNECTED_NONLEAF_MINIMUM_DEGREE - 1


def test_v2_contract_applies_the_charge_policy_to_every_candidate() -> None:
    contract = _contract()
    atom_delete = contract["atom_delete"]

    assert atom_delete["common_gates_apply_to_every_candidate_source"] is True
    assert atom_delete["common_exact_gates_in_order"] == [
        "real_element_under_the_authoritative_element_predicate",
        "unchanged_production_atom_delete_executor_accepts_the_operation",
        "exact_persistent_slot_successor_is_connected_or_null",
        "authoritative_charge_policy_is_preserved_by_the_transition",
        "successor_is_within_the_declared_broad_organic_bounded_size_support",
        "successor_is_canonicalizable",
    ]
    charge = contract["charge_policy"]
    assert charge["applied_to"] == "every_process_v2_atom_delete_candidate"
    assert charge["authoritative"] is True
    assert charge["is_a_common_exact_gate"] is True
    assert charge["exempt_candidate_sources"] == []
    assert charge["changed_by_process_v2"] is False

    inherited = atom_delete["inherited_capability"]
    assert inherited["exempt_from_the_authoritative_charge_policy"] is False
    assert inherited["admission_set_preserved_bit_for_bit"] is False
    assert inherited["capability_remains_reachable"] is True
    assert inherited["covers"] == ["root", "singleton", "leaf"]


def test_v2_contract_gates_only_connected_nonleaf_on_aromaticity_articulation_and_scar() -> None:
    atom_delete = _contract()["atom_delete"]

    assert atom_delete["connected_nonleaf_only_gates_in_order"] == [
        "non_aromatic_under_the_frozen_production_representation",
        "not_a_graph_articulation_point_of_the_real_atom_graph",
        "not_incident_to_any_scar_slot",
    ]
    aromatic = atom_delete["aromatic_connected_nonleaf_policy"]
    assert aromatic["admitted"] is False
    assert aromatic["perception"] == "resonance_invariant_bond_classes"
    assert aromatic["requires_separate_future_semantic_decision_and_resolver"] is True

    scar = atom_delete["scar_incidence_policy"]
    assert scar["admitted"] is False
    assert scar["applies_to_candidate_sources"] == [
        PROCESS_V2_CONNECTED_NONLEAF_CANDIDATE_SOURCE
    ]
    assert scar["requires_separate_future_scar_semantic_decision"] is True
    assert scar["constant"] == "compose_v4.chem.molecular_graph.SCAR_IDX"
    # The rationale for the asymmetry has to survive, not just the flag.
    assert "already_reachable_under_v1" in scar["connected_nonleaf_only_reason"]
    assert "implicit_hydrogen" in scar["reason"]


def test_v2_contract_reason_codes_cover_scar_and_drop_the_expansion_code() -> None:
    authority = _contract()["atom_delete"]["admission_authority"]
    codes = authority["rejection_codes"]

    assert codes == list(PROCESS_V2_ATOM_DELETE_REJECTION_CODES)
    assert "scar_incident" in codes
    assert "charge_policy_violated" in codes
    assert "outside_connected_nonleaf_expansion" not in codes
    assert len(codes) == len(set(codes)) == 11
    assert _contract()["atom_delete"]["superseded_reading"]["removed_rejection_codes"] == list(
        REJECTED_PROCESS_V2_ATOM_DELETE_REJECTION_CODES
    )


def test_v2_effective_mask_is_an_equality_not_a_union() -> None:
    effective = _contract()["atom_delete"]["effective_mask_rule"]

    assert effective["batch_mask_equals_the_admission_authority_array"] is True
    assert effective["model_forward_asserts_equality_not_subset"] is True
    assert effective["v1_dense_mask_is_the_effective_process_v2_mask"] is False
    assert effective["definition"] == (
        "the_process_v2_effective_atom_delete_mask_equals_process_v2_atom_delete_mask_exactly"
    )
    assert "union" not in effective["definition"]
    assert "union" not in json.dumps(effective)


# ---- The renamed mode ----


def test_the_renamed_atom_delete_mode_appears_and_the_old_one_does_not() -> None:
    contract = _contract()
    modes = contract["atom_delete"]["action_semantics_modes"]

    assert PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS == "process_v2_uniform_gated_atom_delete_v2"
    assert modes == {
        "legacy": LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        "process_v2": PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    }
    assert modes["legacy"] == "legacy_acyclic_atom_delete_v1"

    # The removed name survives only as a recorded removed name, never as a mode.
    superseded = contract["atom_delete"]["superseded_reading"]
    assert superseded["removed_action_semantics_mode"] == (
        REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert superseded["removed_action_semantics_mode"] == REJECTED_ATOM_DELETE_MODE_NAME
    assert superseded["removed_action_semantics_mode_may_be_aliased"] is False
    assert "falsely_implied" in superseded["removed_action_semantics_mode_defect"]
    assert REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS not in modes.values()

    occurrences = [
        location
        for location, value in _walk(contract)
        if value == REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    ]
    assert occurrences, "the removed name must be recorded, not silently dropped"
    assert all(
        ".superseded_reading." in location or ".lineage." in location for location in occurrences
    ), occurrences


def _walk(node: object, path: str = "") -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = [(path, node)]
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


# ---- Lineage ----


def test_v2_contract_records_the_rejected_candidate_identity_as_rejected() -> None:
    rejected = _contract()["lineage"]["rejected_pre_run_candidate_identity"]

    assert rejected["process_identity_sha256"] == REJECTED_PRE_RUN_CANDIDATE_IDENTITY
    assert rejected["process_identity_sha256"] == REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256
    assert rejected["status"] == "REJECTED_PRE_RUN_CANDIDATE_IDENTITY"
    assert rejected["rejected_before_any_downstream_run"] is True
    assert rejected["produced_downstream_artifacts"] is False
    assert rejected["downstream_artifacts_produced"] == []
    assert rejected["was_a_superseded_production_identity"] is False
    assert rejected["is_the_current_process_v2_identity"] is False
    assert "authoritative_charge_policy" in rejected["rejection_reason"]
    assert rejected["contract_revision"] == REJECTED_PROCESS_V2_CONTRACT_REVISION

    # It is not, and can never become, the live identity.
    assert rejected["process_identity_sha256"] != _v2_identity()["process_identity_sha256"]
    assert rejected["process_identity_sha256"] != _v1_identity()["process_identity_sha256"]
    with pytest.raises(EditingV2ProcessIdentityError, match="Process-V2 semantic process identity"):
        require_editing_process_v2_identity(REJECTED_PRE_RUN_CANDIDATE_IDENTITY)


def test_the_rejected_candidate_identity_grants_no_authority() -> None:
    """A rejected identity must not smuggle in a training or experiment grant."""

    contract = _contract()
    rejected = contract["lineage"]["rejected_pre_run_candidate_identity"]
    assert not any("authoriz" in key for key in rejected)
    assert all(value is False for value in contract["authority"].values())
    assert contract["training_authorized"] is False
    # Nothing anywhere in the contract grants authority.
    grants = [
        (location, value)
        for location, value in _walk(contract)
        if "authoriz" in location.lower() and value is True
    ]
    assert grants == []


def test_v2_contract_keeps_the_superseded_v1_payload_identity_distinct() -> None:
    lineage = _contract()["lineage"]
    superseded = lineage["superseded_v1_payload_identity"]

    assert superseded["process_identity_sha256"] == SUPERSEDED_V1_PROCESS_IDENTITY
    assert superseded["process_identity_sha256"] == SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    assert superseded["status"] == "SUPERSEDED_V1_PROCESS_IDENTITY"
    assert superseded["process_semantics"] == PROCESS_SEMANTICS
    assert superseded["immutable_v1_payloads_were_built_under_it"] is True
    assert superseded["was_a_superseded_production_identity"] is True
    assert superseded["readable_only_under_the_v1_identity_schema"] is True
    assert superseded["is_the_current_process_v2_identity"] is False

    # The two lineage records are different KINDS of record, not synonyms.
    rejected = lineage["rejected_pre_run_candidate_identity"]
    assert superseded["process_identity_sha256"] != rejected["process_identity_sha256"]
    assert superseded["was_a_superseded_production_identity"] != (
        rejected["was_a_superseded_production_identity"]
    )
    assert superseded["process_semantics"] != rejected["process_semantics"]


def test_v2_contract_records_the_declared_v1_downstream_invalidation() -> None:
    invalidation = _contract()["downstream_invalidation"]
    assert invalidation["superseded_v1_process_identity_sha256"] == (
        SUPERSEDED_V1_PROCESS_IDENTITY
    )
    assert invalidation["superseded_v1_process_semantics"] == PROCESS_SEMANTICS
    assert invalidation["active8_inventory_rebuild_required"] is True
    assert invalidation["affected_whole_trace_replay_required"] is True
    assert invalidation["candidate_and_successor_cache_rebuild_required"] is True
    assert invalidation["gate0_and_t1_rebuild_required"] is True
    assert invalidation["checkpoint_resume_across_process_versions"] is False
    assert invalidation["historical_artifacts_remain_readable_only_under_original_identity"] is True
    assert invalidation["source_and_scaffold_splits_changed"] is False
    assert invalidation["v1_artifacts_relabelled_as_v2"] is False
    assert invalidation["v1_process_contract_edited_in_place"] is False


# ---- Measured motivation, recorded as a bounded observation ----


def test_the_correction_motivation_is_recorded_as_a_bounded_observation() -> None:
    observation = _contract()["source_evidence"][
        "inherited_candidate_charge_violation_observation"
    ]

    assert observation["authoritative_corpus_evidence"] is False
    assert observation["evidence_class"] == (
        "bounded_measured_observation_not_a_corpus_result"
    )
    assert observation["measured_on"] == "800_jin_qed_leads"
    assert observation["denominator_source_molecules"] == 800
    assert observation["denominator_inherited_candidates"] == 3319
    assert observation["inherited_candidates_violating_the_authoritative_charge_policy"] == 125
    assert observation["source_molecules_with_at_least_one_violation"] == 105
    assert observation["excluded_by_the_common_executor_gate_beyond_the_charge_gate"] == 0
    assert observation["excluded_by_the_common_connectivity_gate_beyond_the_charge_gate"] == 0
    assert observation["superseded_rule_admitted_the_violating_candidates"] is True
    assert observation["violating_fraction_percent_rounded_to_two_places"] == "3.77"
    # The recorded percentage must be the one the recorded counts imply.
    implied = 100.0 * observation[
        "inherited_candidates_violating_the_authoritative_charge_policy"
    ] / observation["denominator_inherited_candidates"]
    assert f"{implied:.2f}" == observation["violating_fraction_percent_rounded_to_two_places"]
    assert "authorizes_nothing" in observation["status"]


# ---- The V1 contract is untouched ----


def test_v2_contract_inherits_the_unchanged_v1_contract_by_hash() -> None:
    v1_contract = json.loads(V1_CONTRACT.read_text())
    contract = _contract()
    inherited = contract["inherited_v1_process"]

    assert _file_sha256(V1_CONTRACT) == V1_CONTRACT_PHYSICAL_SHA256
    assert v1_contract["contract_sha256"] == V1_CONTRACT_SHA256
    assert inherited["contract_relative_path"] == str(V1_CONTRACT)
    assert inherited["contract_sha256"] == V1_CONTRACT_SHA256
    assert inherited["contract_physical_sha256"] == V1_CONTRACT_PHYSICAL_SHA256
    assert inherited["process_semantics"] == PROCESS_SEMANTICS

    # The inherited blocks are the V1 blocks, so they cannot silently fork.
    assert contract["action_codec"] == v1_contract["action_codec"]
    assert contract["scope"] == v1_contract["scope"]
    v1_model = dict(v1_contract["model_contract"])
    v2_model = dict(contract["model_contract"])
    assert v2_model.pop("action_semantics_modes") == {
        **v1_model.pop("action_semantics_modes"),
        "editing_v2": PROCESS_V2_SEMANTICS,
    }
    assert v2_model == v1_model
    assert contract["supersedes_for_editing_v2"] == [str(V1_CONTRACT)]


def test_v2_contract_binds_the_unchanged_executor_and_mask_implementation() -> None:
    contract = _contract()
    executor = contract["unchanged_executor"]
    assert executor["executor_rule"] == "atom_delete"
    assert executor["validity_function"] == "is_valid_atom_delete"
    assert executor["apply_function"] == "apply_atom_delete"
    assert executor["module"] == "src/compose_v4/rewrite/operators.py"
    assert executor["changed_by_process_v2"] is False
    assert executor["source_sha256"] == {
        "src/compose_v4/rewrite/kernel.py": _file_sha256(Path("src/compose_v4/rewrite/kernel.py")),
        "src/compose_v4/rewrite/operators.py": _file_sha256(
            Path("src/compose_v4/rewrite/operators.py")
        ),
    }

    mask = contract["atom_delete_mask_implementation"]
    assert mask["dense_mask_module"] == "src/compose_v4/model/factorized_tracelet_rate_model.py"
    assert mask["admission_authority_module"] == PROCESS_V2_RESOLVER_RELATIVE_PATH
    assert mask["source_sha256"] == {
        "src/compose_v4/model/factorized_tracelet_rate_model.py": _file_sha256(
            Path("src/compose_v4/model/factorized_tracelet_rate_model.py")
        ),
        PROCESS_V2_RESOLVER_RELATIVE_PATH: _file_sha256(Path(PROCESS_V2_RESOLVER_RELATIVE_PATH)),
    }
    assert mask["mask_properties"] == [
        "boolean",
        "deterministic",
        "persistent_slot_addressed",
        "independent_of_canonical_smiles_atom_order",
    ]

    # A file bound by several blocks must carry one value inside a single build.
    identity_sources = editing_process_v2_identity()["implementation_source_sha256"]
    bound = {
        **executor["source_sha256"],
        **mask["source_sha256"],
        **contract["canonicalizer"]["source_sha256"],
        **contract["action_codec_identity"]["source_sha256"],
    }
    for relative_path, digest in bound.items():
        assert identity_sources[relative_path] == digest


def test_v2_contract_binds_the_frozen_representation_scope_identities() -> None:
    contract = _contract()
    assert contract["scope"]["max_active_atoms"] == 40
    assert contract["persistent_slot"]["max_active_atoms"] == 40
    assert contract["persistent_slot"]["changed_by_process_v2"] is False
    assert contract["persistent_slot"]["delete_nulls_the_slot_without_compacting_the_state"] is True
    assert contract["vocabulary"]["name"] == "ORGANIC_VOCABULARY"
    assert contract["vocabulary"]["class_count"] == 15
    assert contract["vocabulary"]["changed_by_process_v2"] is False
    assert contract["charge_policy"]["version"] == "exact_charged_center_preservation_v1"
    assert contract["charge_policy"]["formal_charge_changes"] == "out_of_scope"
    assert contract["canonicalizer"]["function"] == "canonical_state_key"
    assert contract["canonicalizer"]["changed_by_process_v2"] is False
    assert contract["scope"]["ring_system_delete_enabled"] is False
    assert contract["scope"]["ring_system_grow_enabled"] is False

    identity = editing_process_v2_identity()
    codec = contract["action_codec_identity"]
    assert codec["schema_version"] == identity["action_codec_schema_version"]
    assert codec["implementation_hash"] == identity["action_codec_implementation_hash"]
    assert codec["active_executor_rules"] == identity["active_executor_rules"]
    assert codec["changed_by_process_v2"] is False
    assert frozenset(contract["action_codec"]["public_editing_v2_rules"]) == frozenset(
        codec["active_executor_rules"]
    )


# ---- Coexisting V1 and V2 identities ----


def test_v1_identity_definition_is_unchanged_by_process_v2() -> None:
    identity = _v1_identity()
    assert identity["schema"] == PROCESS_IDENTITY_SCHEMA
    assert identity["schema"] == "compose.editing.semantic_process_identity"
    assert identity["schema_version"] == PROCESS_IDENTITY_SCHEMA_VERSION == 1
    assert identity["process_semantics"] == PROCESS_SEMANTICS == "semantic_editing_v2_v1"
    assert identity["contract_relative_path"] == str(V1_CONTRACT)
    assert identity["contract_sha256"] == V1_CONTRACT_SHA256
    assert identity["contract_physical_sha256"] == V1_CONTRACT_PHYSICAL_SHA256
    assert frozenset(identity) == IDENTITY_FIELDS
    assert tuple(sorted(identity["implementation_source_sha256"])) == (
        V1_IMPLEMENTATION_RELATIVE_PATHS
    )
    body = {field: identity[field] for field in IDENTITY_BODY_FIELDS}
    assert _semantic_sha256(body) == identity["process_identity_sha256"]


def test_v2_identity_extends_the_v1_boundary_with_the_admission_authority() -> None:
    identity = _v2_identity()
    assert identity["schema"] == PROCESS_V2_IDENTITY_SCHEMA
    assert identity["schema"] == "compose.editing.semantic_process_v2_identity"
    assert identity["schema_version"] == PROCESS_V2_IDENTITY_SCHEMA_VERSION
    assert identity["process_semantics"] == PROCESS_V2_SEMANTICS
    assert identity["contract_relative_path"] == str(V2_CONTRACT)
    assert identity["contract_physical_sha256"] == _file_sha256(V2_CONTRACT)
    assert frozenset(identity) == IDENTITY_FIELDS
    assert tuple(sorted(identity["implementation_source_sha256"])) == tuple(
        sorted({*V1_IMPLEMENTATION_RELATIVE_PATHS, PROCESS_V2_RESOLVER_RELATIVE_PATH})
    )
    body = {field: identity[field] for field in IDENTITY_BODY_FIELDS}
    assert _semantic_sha256(body) == identity["process_identity_sha256"]

    # The shared V1 sources are hashed identically by both identities.
    v1_sources = _v1_identity()["implementation_source_sha256"]
    v2_sources = identity["implementation_source_sha256"]
    assert {key: v2_sources[key] for key in v1_sources} == v1_sources


def test_the_two_identities_are_deterministic_and_differ() -> None:
    assert editing_v2_process_identity() == editing_v2_process_identity()
    assert editing_process_v2_identity() == editing_process_v2_identity()
    v1 = _v1_identity()
    v2 = _v2_identity()
    assert v1["process_identity_sha256"] != v2["process_identity_sha256"]
    assert v1["schema"] != v2["schema"]
    assert v1["process_semantics"] != v2["process_semantics"]
    assert v1["contract_relative_path"] != v2["contract_relative_path"]
    assert v1["contract_sha256"] != v2["contract_sha256"]
    # Everything a process identity is allowed to share stays shared.
    assert v1["action_codec_schema_version"] == v2["action_codec_schema_version"]
    assert v1["action_codec_implementation_hash"] == v2["action_codec_implementation_hash"]
    assert v1["active_executor_rules"] == v2["active_executor_rules"]


def test_each_require_function_rejects_the_other_process_version() -> None:
    v1_sha = str(_v1_identity()["process_identity_sha256"])
    v2_sha = str(_v2_identity()["process_identity_sha256"])

    assert require_editing_v2_process_identity(v1_sha)["process_semantics"] == PROCESS_SEMANTICS
    assert require_editing_process_v2_identity(v2_sha)["process_semantics"] == PROCESS_V2_SEMANTICS

    with pytest.raises(EditingV2ProcessIdentityError, match="Process-V2 semantic process identity"):
        require_editing_process_v2_identity(v1_sha)
    with pytest.raises(EditingV2ProcessIdentityError, match="Editing-V2 process identity"):
        require_editing_v2_process_identity(v2_sha)


def test_a_missing_or_mismatched_process_v2_identity_fails_loudly() -> None:
    with pytest.raises(
        EditingV2ProcessIdentityError,
        match="Process-V2 semantic process identity differs from the bound artifact identity",
    ):
        require_editing_process_v2_identity("0" * 64)
    with pytest.raises(EditingV2ProcessIdentityError):
        require_editing_process_v2_identity("")


def test_a_missing_process_v2_contract_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_identity_caches()
    monkeypatch.setattr(identity_module, "_repository_root", lambda: tmp_path)
    try:
        with pytest.raises(
            EditingV2ProcessIdentityError,
            match="cannot read frozen semantic process contract",
        ):
            editing_process_v2_identity()
    finally:
        monkeypatch.undo()
        _clear_identity_caches()


def test_a_tampered_process_v2_contract_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    contract["training_authorized"] = True
    staged = tmp_path / V2_CONTRACT
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(json.dumps(contract, sort_keys=True))

    _clear_identity_caches()
    monkeypatch.setattr(identity_module, "_repository_root", lambda: tmp_path)
    try:
        with pytest.raises(
            EditingV2ProcessIdentityError,
            match="self-hash does not match its contents",
        ):
            editing_process_v2_identity()
    finally:
        monkeypatch.undo()
        _clear_identity_caches()


# ---- A V1 artifact must not be able to masquerade as V2 ----


def test_a_relabelled_and_rehashed_v1_identity_is_still_not_process_v2() -> None:
    forged_body = {field: _v1_identity()[field] for field in IDENTITY_BODY_FIELDS}
    forged_body["schema"] = PROCESS_V2_IDENTITY_SCHEMA
    forged_body["schema_version"] = PROCESS_V2_IDENTITY_SCHEMA_VERSION
    forged_body["process_semantics"] = PROCESS_V2_SEMANTICS
    forged = _self_hashed(forged_body)

    # Internally consistent by construction, and still not the V2 identity.
    assert forged["process_identity_sha256"] != _v1_identity()["process_identity_sha256"]
    assert forged["process_identity_sha256"] != _v2_identity()["process_identity_sha256"]
    with pytest.raises(EditingV2ProcessIdentityError, match="Process-V2 semantic process identity"):
        require_editing_process_v2_identity(str(forged["process_identity_sha256"]))
    # The V1 contract path betrays the relabelling at the schema level.
    with pytest.raises(EditingV2ProcessIdentityError, match="contract_relative_path"):
        validate_frozen_process_identity(forged)


def test_a_fully_relabelled_v1_identity_is_self_consistent_but_still_not_current() -> None:
    """A self-consistent forgery is accepted as *historical* and rejected as V2.

    This is the boundary between the two validators: internal consistency is not
    currency, and only the live-source comparison can establish currency.
    """

    forged_body = {field: _v1_identity()[field] for field in IDENTITY_BODY_FIELDS}
    forged_body["schema"] = PROCESS_V2_IDENTITY_SCHEMA
    forged_body["schema_version"] = PROCESS_V2_IDENTITY_SCHEMA_VERSION
    forged_body["process_semantics"] = PROCESS_V2_SEMANTICS
    forged_body["contract_relative_path"] = str(V2_CONTRACT)
    forged = _self_hashed(forged_body)

    assert validate_frozen_process_identity(forged) == forged
    with pytest.raises(EditingV2ProcessIdentityError, match="Process-V2 semantic process identity"):
        require_editing_process_v2_identity(str(forged["process_identity_sha256"]))


def test_a_v2_identity_cannot_be_relabelled_as_v1_either() -> None:
    forged_body = {field: _v2_identity()[field] for field in IDENTITY_BODY_FIELDS}
    forged_body["schema"] = PROCESS_IDENTITY_SCHEMA
    forged_body["schema_version"] = PROCESS_IDENTITY_SCHEMA_VERSION
    forged_body["process_semantics"] = PROCESS_SEMANTICS
    forged = _self_hashed(forged_body)

    with pytest.raises(EditingV2ProcessIdentityError, match="contract_relative_path"):
        validate_frozen_process_identity(forged)
    with pytest.raises(EditingV2ProcessIdentityError, match="Editing-V2 process identity"):
        require_editing_v2_process_identity(str(forged["process_identity_sha256"]))


# ---- validate_frozen_process_identity ----


def test_validate_frozen_process_identity_accepts_genuine_historical_objects() -> None:
    for identity in (_v1_identity(), _v2_identity()):
        validated = validate_frozen_process_identity(identity)
        assert validated == identity
        assert validated is not identity


def test_validate_frozen_process_identity_rejects_a_mutated_body_without_rehash() -> None:
    identity = _v1_identity()
    identity["contract_sha256"] = "0" * 64
    with pytest.raises(EditingV2ProcessIdentityError, match="self-hash does not match its own body"):
        validate_frozen_process_identity(identity)


def test_validate_frozen_process_identity_rejects_a_missing_field() -> None:
    identity = _v1_identity()
    del identity["action_codec_implementation_hash"]
    with pytest.raises(EditingV2ProcessIdentityError, match="field set is wrong"):
        validate_frozen_process_identity(identity)


def test_validate_frozen_process_identity_rejects_an_extra_field() -> None:
    identity = _v1_identity()
    identity["training_authorized"] = False
    with pytest.raises(EditingV2ProcessIdentityError, match="field set is wrong"):
        validate_frozen_process_identity(identity)


@pytest.mark.parametrize("payload", [None, [], "identity", 7])
def test_validate_frozen_process_identity_rejects_a_non_mapping(payload: object) -> None:
    with pytest.raises(EditingV2ProcessIdentityError, match="must be a mapping"):
        validate_frozen_process_identity(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "schema",
    ["compose.editing.semantic_process_v3_identity", "", None, ["identity"]],
)
def test_validate_frozen_process_identity_rejects_an_unknown_schema(schema: object) -> None:
    identity = _v1_identity()
    identity["schema"] = schema
    with pytest.raises(EditingV2ProcessIdentityError, match="unknown frozen process identity"):
        validate_frozen_process_identity(identity)


def test_validate_frozen_process_identity_rejects_a_swapped_process_semantics() -> None:
    identity = _v1_identity()
    identity["process_semantics"] = PROCESS_V2_SEMANTICS
    with pytest.raises(EditingV2ProcessIdentityError, match="process_semantics"):
        validate_frozen_process_identity(identity)


def test_validate_frozen_process_identity_reads_no_live_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Historical validation must not depend on the working tree at all."""

    identity = _v1_identity()
    monkeypatch.setattr(identity_module, "_repository_root", lambda: tmp_path)
    assert validate_frozen_process_identity(identity) == identity


# ---- The hash-chain verifier's load-bearing discovery rules ----


def test_verifier_discovers_a_self_hash_without_knowing_its_field_name() -> None:
    verifier = _load_verifier()
    payload = {"policy": "x", "count": 3}
    sealed = {**payload, "policy_sha256": verifier._canonical_sha256(payload)}

    assert verifier._self_hash_fields(sealed) == {"policy_sha256": sealed["policy_sha256"]}
    # A drifted seal is discovered as a candidate but not as an agreeing self-hash.
    drifted = {**sealed, "count": 4}
    assert verifier._self_hash_fields(drifted) == {}
    assert set(verifier._self_hash_candidates(drifted)) == {"policy_sha256"}
    # The committed contracts are discovered the same way, with no name list.
    committed = json.loads(V2_CONTRACT.read_text())
    assert set(verifier._self_hash_fields(committed)) == {"contract_sha256"}


def test_verifier_discovers_pointer_edges_from_sibling_naming(tmp_path: Path) -> None:
    verifier = _load_verifier()
    (tmp_path / "configs").mkdir()
    target = tmp_path / "configs" / "target_v1.json"
    target.write_text("{}")
    payload = {
        "parents": {"a": {"path": "configs/target_v1.json", "file_sha256": "a" * 64}},
        "dependencies": {
            "thing": "configs/target_v1.json",
            "thing_sha256": "b" * 64,
            "unrelated_sha256": "c" * 64,
        },
    }

    edges = verifier._pointer_edges(tmp_path, "configs/source_v1.json", payload)
    assert {(edge.location, edge.value) for edge in edges} == {
        (".parents.a.file_sha256", "a" * 64),
        (".dependencies.thing_sha256", "b" * 64),
    }
    assert {edge.target for edge in edges} == {"configs/target_v1.json"}


def test_verifier_reports_a_stale_pointer_against_the_live_value(tmp_path: Path) -> None:
    verifier = _load_verifier()
    (tmp_path / "configs").mkdir()
    target = tmp_path / "configs" / "target_v1.json"
    target.write_text("{}")

    live = verifier.ValueIndex()
    live.add("configs/target_v1.json", "physical_sha256", "1" * 64)
    live.add("configs/target_v1.json", "policy_sha256", "2" * 64)
    base = verifier.ValueIndex()
    base.add("configs/target_v1.json", "policy_sha256", "3" * 64)

    edge = verifier.PointerEdge("configs/source.json", ".pin_sha256", "configs/target_v1.json",
                                "3" * 64)
    findings = verifier._check_pointer_edges(tmp_path, (edge,), live, base, True)
    assert [finding.category for finding in findings] == ["stale_pointer"]
    assert findings[0].severity == "FAIL"

    fresh = verifier.PointerEdge("configs/source.json", ".pin_sha256",
                                 "configs/target_v1.json", "2" * 64)
    assert verifier._check_pointer_edges(tmp_path, (fresh,), live, base, True) == []


def test_verifier_writes_nothing_and_reports_the_live_identities() -> None:
    """The verifier is read-only: it must not touch the tree it inspects."""

    verifier = _load_verifier()
    before = _file_sha256(V2_CONTRACT), _file_sha256(V1_CONTRACT)
    live, _findings = verifier._check_live_contract(REPO_ROOT)
    assert live["v1_process_identity_sha256"] == _v1_identity()["process_identity_sha256"]
    assert live["v2_process_identity_sha256"] == _v2_identity()["process_identity_sha256"]
    assert live["v2_contract_sha256"] == _contract()["contract_sha256"]
    assert live["v1_contract_physical_sha256"] == V1_CONTRACT_PHYSICAL_SHA256
    assert (_file_sha256(V2_CONTRACT), _file_sha256(V1_CONTRACT)) == before
