"""The Editing-V2 checkpoint identity is complete, exact, and fail-closed."""

from __future__ import annotations

from copy import deepcopy

import pytest

from compose_v4.experiments.editing_v2_scientific_identity import (
    ARTIFACT_KEYS,
    CHECKPOINT_IDENTITY_FIELD,
    PRODUCTIVE_TRAINING_OBJECT,
    SCIENTIFIC_IDENTITY_SCHEMA,
    SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    EditingV2ScientificIdentityError,
    LegacyOnlyCheckpointError,
    build_editing_v2_scientific_identity,
    canonical_sha256,
    editing_v2_identity_from_checkpoint,
    editing_v2_scientific_identities_equal,
    require_editing_v2_scientific_identity_equal,
    semantic_model_identity,
    validate_editing_v2_scientific_identity,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)

# The semantic Active8 modes intentionally have a different capability
# fingerprint from the legacy-action Gate-0 V2 configuration.
CAPABILITY_FINGERPRINT = "d246bc88d8440d31"


def _digest(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _artifact(label: str) -> dict[str, str]:
    return {
        "physical_sha256": _digest(f"{label}:physical"),
        "semantic_sha256": _digest(f"{label}:semantic"),
    }


def _model() -> dict:
    return semantic_model_identity(
        mark_dim=32,
        operator_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )


def _identity(
    *,
    corpus: dict[str, str] | None = None,
    model: dict | None = None,
) -> dict:
    artifacts = {key: _artifact(key) for key in ARTIFACT_KEYS}
    if corpus is not None:
        artifacts["semantic_corpus_global_registry"] = corpus
    return build_editing_v2_scientific_identity(
        model_identity=_model() if model is None else model,
        semantic_corpus_global_registry=artifacts["semantic_corpus_global_registry"],
        active8_admission=artifacts["active8_admission"],
        successor_cache=artifacts["successor_cache"],
        sampling_sidecar=artifacts["sampling_sidecar"],
        gate_zero=artifacts["gate_zero"],
        t1=artifacts["t1"],
        p50_recipe=artifacts["p50_recipe"],
    )


def _rehash(identity: dict) -> dict:
    body = {
        key: value
        for key, value in identity.items()
        if key != "scientific_identity_sha256"
    }
    return {**body, "scientific_identity_sha256": canonical_sha256(body)}


def test_complete_identity_round_trips_and_binds_every_boundary() -> None:
    first = _identity()
    second = _identity()

    assert first == second
    assert validate_editing_v2_scientific_identity(first) == first
    assert first["schema"] == SCIENTIFIC_IDENTITY_SCHEMA
    assert first["schema_version"] == SCIENTIFIC_IDENTITY_SCHEMA_VERSION
    assert first["training_object"] == PRODUCTIVE_TRAINING_OBJECT
    assert first["process_identity"] == editing_v2_process_identity()
    assert first["process_identity"]["process_identity_sha256"] == (
        editing_v2_process_identity()["process_identity_sha256"]
    )
    assert set(first["artifact_identities"]) == set(ARTIFACT_KEYS)
    assert first["model_identity"]["mark_dim"] == 32
    assert first["model_identity"]["ring_restate_scorer_mode"] == (
        "semantic_successor_group_v1"
    )
    assert first["model_identity"]["cycle_open_scorer_mode"] == "pair_linear"
    assert editing_v2_scientific_identities_equal(first, second)
    assert require_editing_v2_scientific_identity_equal(first, second) == second


@pytest.mark.parametrize("checkpoint", [{}, {"model_state": {}}, None])
def test_checkpoint_without_complete_identity_is_legacy_only(
    checkpoint: object,
) -> None:
    with pytest.raises(LegacyOnlyCheckpointError, match="legacy-only"):
        editing_v2_identity_from_checkpoint(checkpoint)


def test_partial_or_extra_identity_is_legacy_only() -> None:
    identity = _identity()
    identity.pop("artifact_identities")
    with pytest.raises(LegacyOnlyCheckpointError, match="missing"):
        validate_editing_v2_scientific_identity(identity)

    identity = _identity()
    identity["implementation_provenance_allowance"] = True
    with pytest.raises(LegacyOnlyCheckpointError, match="unexpected"):
        validate_editing_v2_scientific_identity(identity)


def test_checkpoint_extraction_returns_only_a_strict_identity() -> None:
    identity = _identity()
    checkpoint = {"model_state": {}, CHECKPOINT_IDENTITY_FIELD: identity}
    assert editing_v2_identity_from_checkpoint(checkpoint) == identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("editing_process_semantics", "legacy_raw_actions_v1"),
        ("ring_restate_scorer_mode", "legacy_changed_kekule_edges_v1"),
        ("cycle_open_action_semantics", "legacy_raw_bond_delete_v1"),
        ("cycle_open_scorer_mode", "exact_bond_contextual_probe"),
        ("use_aromatic_bond_view", False),
        ("compute_ring_system_delete", True),
    ],
)
def test_nonsemantic_or_incomplete_model_modes_are_rejected(
    field: str, value: object
) -> None:
    model = _model()
    model[field] = value
    with pytest.raises(EditingV2ScientificIdentityError, match="exact semantic"):
        _identity(model=model)


def test_mark_dimension_and_capability_fingerprint_are_exact() -> None:
    model = _model()
    model["mark_dim"] = 0
    with pytest.raises(EditingV2ScientificIdentityError, match="mark_dim"):
        _identity(model=model)

    model = _model()
    model["operator_capability_fingerprint"] = "0" * 16
    with pytest.raises(EditingV2ScientificIdentityError, match="disagrees"):
        _identity(model=model)


def test_rehashed_process_tampering_still_fails_closed() -> None:
    identity = _identity()
    tampered = deepcopy(identity)
    tampered["process_identity"]["process_semantics"] = "legacy_raw_actions_v1"
    tampered = _rehash(tampered)

    with pytest.raises(EditingV2ScientificIdentityError, match="process differs"):
        validate_editing_v2_scientific_identity(tampered)


def test_corpus_mismatch_is_scientific_inequality_and_cannot_be_waived() -> None:
    expected = _identity()
    observed = _identity(corpus=_artifact("different-corpus"))

    assert validate_editing_v2_scientific_identity(expected) == expected
    assert validate_editing_v2_scientific_identity(observed) == observed
    assert not editing_v2_scientific_identities_equal(expected, observed)
    with pytest.raises(EditingV2ScientificIdentityError, match="cannot be waived"):
        require_editing_v2_scientific_identity_equal(expected, observed)


@pytest.mark.parametrize("hash_field", ["physical_sha256", "semantic_sha256"])
def test_every_artifact_requires_physical_and_semantic_sha256(hash_field: str) -> None:
    identity = _identity()
    identity["artifact_identities"]["successor_cache"][hash_field] = "not-a-sha"
    identity = _rehash(identity)

    with pytest.raises(EditingV2ScientificIdentityError, match=hash_field):
        validate_editing_v2_scientific_identity(identity)


def test_missing_artifact_identity_never_validates_as_editing_v2() -> None:
    identity = _identity()
    identity["artifact_identities"].pop("sampling_sidecar")
    identity = _rehash(identity)

    with pytest.raises(LegacyOnlyCheckpointError, match="legacy-only"):
        validate_editing_v2_scientific_identity(identity)


def test_built_process_mapping_does_not_alias_the_cached_production_identity() -> None:
    identity = _identity()
    expected_process = deepcopy(editing_v2_process_identity())

    identity["process_identity"]["active_executor_rules"].append("not_real")

    assert editing_v2_process_identity() == expected_process
