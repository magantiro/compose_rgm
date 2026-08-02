"""Prospective Process-V2 semantic process contract and coexisting identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite import editing_v2_process_identity as identity_module
from compose_v4.rewrite.editing_v2_process_identity import (
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_IDENTITY_SCHEMA,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    PROCESS_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_CONTRACT_SCHEMA,
    PROCESS_V2_CONTRACT_SCHEMA_VERSION,
    PROCESS_V2_CONTRACT_STATUS,
    PROCESS_V2_IDENTITY_SCHEMA,
    PROCESS_V2_IDENTITY_SCHEMA_VERSION,
    PROCESS_V2_SEMANTICS,
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
from compose_v4.rewrite.operators import AtomDelete, apply_atom_delete
from compose_v4.rewrite.process_v2_atom_delete import (
    CONNECTED_NONLEAF_MINIMUM_DEGREE,
    ProcessV2AtomDeleteRejectionCode,
    process_v2_connected_nonleaf_atom_delete_mask,
)

V1_CONTRACT = Path("configs/editing_v2_semantic_process_v1.json")
V2_CONTRACT = Path("configs/editing_v2_semantic_process_v2.json")

# The V1 semantic-process contract is preserved unchanged by Process V2.
V1_CONTRACT_SHA256 = "f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288"
V1_CONTRACT_PHYSICAL_SHA256 = "43cb26e1ba33b27a0149a8142895ef9988853b458c0ace9bc9191f7b460dedbb"

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


def _v1_identity() -> dict[str, Any]:
    return dict(editing_v2_process_identity())


def _v2_identity() -> dict[str, Any]:
    return dict(editing_process_v2_identity())


def _clear_identity_caches() -> None:
    editing_v2_process_identity.cache_clear()
    editing_process_v2_identity.cache_clear()


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
    contract = json.loads(V2_CONTRACT.read_text())
    claimed = contract.pop("contract_sha256")
    assert _semantic_sha256(contract) == claimed
    assert claimed == editing_process_v2_identity()["contract_sha256"]
    assert build_editing_process_v2_contract()["contract_sha256"] == claimed


def test_v2_contract_declares_no_training_or_experiment_authority() -> None:
    contract = json.loads(V2_CONTRACT.read_text())
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
    contract = json.loads(V2_CONTRACT.read_text())
    assert contract["schema"] == PROCESS_V2_CONTRACT_SCHEMA
    assert contract["schema"] == "compose.editing.semantic_process_contract"
    assert contract["schema_version"] == PROCESS_V2_CONTRACT_SCHEMA_VERSION == 2
    assert contract["process_semantics"] == PROCESS_V2_SEMANTICS == "semantic_editing_v2_v2"
    assert contract["model_contract"]["action_semantics_modes"]["editing_v2"] == (
        PROCESS_V2_SEMANTICS
    )
    assert contract["process_identity"]["identity_schema"] == PROCESS_V2_IDENTITY_SCHEMA
    assert contract["process_identity"]["provider"] == "editing_process_v2_identity"


def test_v2_contract_binds_the_connected_nonleaf_admission_semantics() -> None:
    atom_delete = json.loads(V2_CONTRACT.read_text())["atom_delete"]
    expansion = atom_delete["connected_nonleaf_expansion"]

    assert expansion["minimum_real_atom_degree"] == CONNECTED_NONLEAF_MINIMUM_DEGREE == 2
    assert expansion["minimum_real_atom_degree_constant"] == "CONNECTED_NONLEAF_MINIMUM_DEGREE"
    assert expansion["module"] == PROCESS_V2_RESOLVER_RELATIVE_PATH
    assert len(expansion["admission_conditions"]) == 7
    assert expansion["rejection_codes_in_evaluation_order"] == [
        code.value for code in ProcessV2AtomDeleteRejectionCode
    ]

    preserved = atom_delete["preserved_v1_admission"]
    assert preserved["covers"] == ["root", "singleton", "leaf"]
    assert preserved["maximum_admitted_real_atom_degree"] == CONNECTED_NONLEAF_MINIMUM_DEGREE - 1
    assert preserved["redecided_by_process_v2"] is False
    assert preserved["charge_policy_applied_to_preserved_v1_candidates"] is False
    assert preserved["uniform_charge_application_would_remove_preserved_v1_candidates"] is True

    effective = atom_delete["effective_mask_rule"]
    assert effective["v1_dense_mask_changed"] is False
    assert effective["operands_are_disjoint"] is True
    assert effective["union_is_a_disjoint_union"] is True
    assert "union" in effective["definition"]

    aromatic = atom_delete["aromatic_connected_nonleaf_policy"]
    assert aromatic["admitted"] is False
    assert "representation_sensitive_open_chain_successor" in aromatic["reason"]
    assert "kekule" in aromatic["reason"]
    assert aromatic["requires_separate_future_semantic_decision_and_resolver"] is True

    assert "legality_authority" in atom_delete
    assert "executor_is_the_legality_authority" in atom_delete["legality_authority"]
    assert atom_delete["executor_validity_is_a_connectivity_predicate"] is False
    assert atom_delete["one_step_inverse_closure_claimed"] is False
    assert atom_delete["action_semantics_modes"] == {
        "legacy": LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        "process_v2": PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    }


def test_v2_contract_binds_the_unchanged_executor_and_mask_implementation() -> None:
    contract = json.loads(V2_CONTRACT.read_text())
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
    assert mask["connected_nonleaf_resolver_module"] == PROCESS_V2_RESOLVER_RELATIVE_PATH
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
    contract = json.loads(V2_CONTRACT.read_text())
    assert contract["scope"]["max_active_atoms"] == 40
    assert contract["persistent_slot"]["max_active_atoms"] == 40
    assert contract["persistent_slot"]["changed_by_process_v2"] is False
    assert contract["persistent_slot"]["delete_nulls_the_slot_without_compacting_the_state"] is True
    assert contract["vocabulary"]["name"] == "ORGANIC_VOCABULARY"
    assert contract["vocabulary"]["class_count"] == 15
    assert contract["vocabulary"]["changed_by_process_v2"] is False
    assert contract["charge_policy"]["version"] == "exact_charged_center_preservation_v1"
    assert contract["charge_policy"]["changed_by_process_v2"] is False
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


def test_v2_contract_inherits_the_unchanged_v1_contract_by_hash() -> None:
    v1_contract = json.loads(V1_CONTRACT.read_text())
    contract = json.loads(V2_CONTRACT.read_text())
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


def test_v2_contract_records_the_declared_v1_downstream_invalidation() -> None:
    invalidation = json.loads(V2_CONTRACT.read_text())["downstream_invalidation"]
    assert invalidation["superseded_v1_process_identity_sha256"] == (
        SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    )
    assert invalidation["superseded_v1_process_identity_sha256"] == (
        "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
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


def test_v2_contract_development_evidence_is_not_promoted_and_still_holds() -> None:
    evidence = json.loads(V2_CONTRACT.read_text())["source_evidence"]
    panel = evidence["connected_nonleaf_disjointness_development_panel"]
    counterexample = evidence["uniform_charge_policy_counterexample"]
    assert panel["authoritative_corpus_evidence"] is False
    assert counterexample["authoritative_corpus_evidence"] is False

    # Re-measure the Process-V2 half of the recorded panel with the frozen
    # resolver: every admitted slot is a real, connected-nonleaf slot.
    for smiles in panel["panel_smiles"]:
        state = smiles_to_molecular_graph(smiles)
        mask = process_v2_connected_nonleaf_atom_delete_mask(state)
        assert mask.dtype == bool
        assert mask.shape == (state.n_atoms,)
        for slot in range(state.n_atoms):
            if not bool(mask[slot]):
                continue
            assert bool(is_element(state.atom_types[slot]))
            degree = sum(
                1
                for other in range(state.n_atoms)
                if bool(is_element(state.atom_types[other]))
                and int(state.bonds[slot, other]) != 0
            )
            assert degree >= CONNECTED_NONLEAF_MINIMUM_DEGREE

    zwitterion = smiles_to_molecular_graph(counterexample["source_smiles"])
    admitted = [
        slot
        for slot in range(zwitterion.n_atoms)
        if bool(process_v2_connected_nonleaf_atom_delete_mask(zwitterion)[slot])
    ]
    assert admitted == counterexample["connected_nonleaf_admitted_slots"] == []

    # Uniform charge-policy application really would remove preserved V1 leaves:
    # of the recorded V1-admitted slots only slot 6 preserves the charge policy.
    # The V1 dense mask itself is the candidate-mask owner's frozen fixture.
    charge_preserving = [
        slot
        for slot in counterexample["v1_admitted_slots"]
        if charge_policy_preserved(zwitterion, apply_atom_delete(zwitterion, AtomDelete(slot)))
    ]
    assert charge_preserving == counterexample["charge_policy_preserving_v1_admitted_slots"]
    assert len(charge_preserving) < len(counterexample["v1_admitted_slots"])


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


def test_v2_identity_extends_the_v1_boundary_with_the_process_v2_resolver() -> None:
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
    contract = json.loads(V2_CONTRACT.read_text())
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
