"""A resolved evidence binding must pin measured provenance and grant nothing.

Two failure modes are the reason this artifact exists at all, and both are silent.

1. **Measured evidence smuggled into a frozen contract.** The seven prospective
   contracts used to advertise a fillable ``admitted_source`` slot that could never
   be filled. Anything that looks like a fillable slot on a deterministic artifact
   is either dead or a way to reseal policy under an old name, so measured
   provenance lives here instead and the contract is cited, never mutated.

2. **Authority acquired by accident.** A binding proves identity; it never permits
   training or a launch. Every authority field is asserted false, and the adapter's
   own authority block is asserted NOT to be copied in: it uses a different
   spelling for the bounded-P50 field, and reconciling the two vocabularies is an
   owner decision because the retired spelling lives inside a file whose bytes are
   hashed into the scientific Process-V2 identity. The guard that stays here is
   spelling-agnostic and refuses any granted ``*_authorized`` field.

Every fixture identity below is obviously synthetic. Reusing a real completion SHA
or census in a test would make a fixture indistinguishable from evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_admitted_source import (
    ADMITTED_SOURCE_SCHEMA,
    ADMITTED_SOURCE_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_policy_registry import FROZEN_POLICY_SOURCES
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    RESOLVED_EVIDENCE_BINDING_SCHEMA,
    RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
    RESOLVED_EVIDENCE_BINDING_STATUS,
    IdentityRole,
    PointerKind,
    typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    EDITING_CORPUS_V2_CONTRACT,
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
)
from compose_v4.experiments.editing_v2_process_v2_evidence_binding import (
    ADMITTED_SOURCE_IDENTITY_FIELDS,
    BINDING_SELF_HASH_FIELD,
    EVIDENCE_BINDING_STAGES,
    ProcessV2EvidenceBindingError,
    build_resolved_evidence_binding,
    load_resolved_evidence_binding,
    resolved_evidence_binding_self_hash,
    serialize_resolved_evidence_binding,
    validate_resolved_evidence_binding,
    write_resolved_evidence_binding,
)

_ROOT = Path(__file__).resolve().parents[1]

# Fixture identities. Deliberately not any real completion, run, or census.
_FIXTURE = {
    "adapter_implementation_sha256": "a1" * 32,
    "completion_sha256": "a2" * 32,
    "physical_inventory_sha256": "a3" * 32,
    "process_identity_sha256": "a4" * 32,
    "run_identity_sha256": "a5" * 32,
    "semantic_evidence_sha256": "a6" * 32,
}


def _admitted_source(**overrides: Any) -> dict[str, Any]:
    """A synthetic version-1 admitted-source identity descriptor."""

    payload: dict[str, Any] = {
        **_FIXTURE,
        "counts": {
            "source_entries": 30,
            "admitted_entries": 22,
            "rejected_entries": 8,
        },
        "rejected_traces_by_code": {
            "fixture_reason_alpha": 5,
            "fixture_reason_beta": 3,
        },
        # The adapter's own authority vocabulary, reproduced exactly as
        # `ProcessV2AdmittedSource.identity()` emits it today, including the
        # bounded-P50 field's older spelling. The binding neither translates nor
        # copies this block; renaming it at its source is an owner decision
        # because it would move the scientific Process-V2 identity.
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "p50_authorized": False,
    }
    payload.update(overrides)
    return payload


def _prerequisites(**overrides: Any) -> dict[str, dict[str, Any]]:
    prerequisites = {
        "chunk_cache_completion": typed_pointer(
            kind=PointerKind.REMOTE_ARTIFACT,
            provider="process_v2_chunk_cache",
            target="/editing_v2/fixture_chunk_cache/CACHE_COMPLETE.json",
            target_schema="compose.editing_v2.process_v2.chunk_cache_completion",
            identity_role=IdentityRole.SEMANTIC,
            hash_algorithm="self_hash_field_v1",
            sha256="b1" * 32,
        ),
        "editing_corpus_contract": typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="frozen_repository_artifact",
            target=EDITING_CORPUS_V2_CONTRACT,
            target_schema="compose.editing_corpus_contract",
            identity_role=IdentityRole.PHYSICAL,
            sha256=hashlib.sha256(
                (_ROOT / EDITING_CORPUS_V2_CONTRACT).read_bytes()
            ).hexdigest(),
        ),
    }
    prerequisites.update(overrides)
    return prerequisites


def _build(stage: str = "gate_zero", **kwargs: Any) -> dict[str, Any]:
    return build_resolved_evidence_binding(
        stage=stage,
        admitted_source=kwargs.pop("admitted_source", _admitted_source()),
        measured_prerequisites=kwargs.pop("measured_prerequisites", _prerequisites()),
        repo_root=kwargs.pop("repo_root", _ROOT),
        **kwargs,
    )


@contextmanager
def _repo_copy() -> Iterator[Path]:
    """A throwaway root holding the cited contracts and their frozen sources."""

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sources = [source.relative_path for source in FROZEN_POLICY_SOURCES.values()]
        for relative_path in (
            *PROCESS_V2_CHAIN_ARTIFACTS,
            EDITING_CORPUS_V2_CONTRACT,
            *sources,
        ):
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative_path, target)
        yield root


# ---- (a) round trip and determinism ----


def test_a_binding_round_trips_build_validate_load() -> None:
    built = _build()
    assert validate_resolved_evidence_binding(built, repo_root=_ROOT) == built
    with tempfile.TemporaryDirectory() as raw:
        path = write_resolved_evidence_binding(built, path=Path(raw) / "binding.json")
        assert load_resolved_evidence_binding(path, repo_root=_ROOT) == built


def test_building_the_same_inputs_twice_is_byte_identical() -> None:
    first = serialize_resolved_evidence_binding(_build())
    second = serialize_resolved_evidence_binding(_build())
    assert first == second
    assert first.endswith(b"\n")


def test_every_stage_binds_its_own_governing_contract() -> None:
    assert dict(EVIDENCE_BINDING_STAGES) == {
        "gate_zero": GATE_ZERO_STRUCTURAL,
        "t1_panel": T1_PANEL_POLICY,
        "t1_capacity": T1_CAPACITY_POLICY,
        "bounded_p50": P50_RECIPE_POLICY,
    }
    for stage, contract in sorted(EVIDENCE_BINDING_STAGES.items()):
        binding = _build(stage)
        assert binding["stage"] == stage
        assert binding["prospective_contract"]["physical"]["target"] == contract
        assert binding["prospective_contract"]["semantic"]["target"] == contract


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not a resolved-evidence"):
        _build("p2000")


def test_the_schema_envelope_is_the_frozen_one() -> None:
    binding = _build()
    assert binding["schema"] == RESOLVED_EVIDENCE_BINDING_SCHEMA
    assert binding["schema_version"] == RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION
    assert binding["status"] == RESOLVED_EVIDENCE_BINDING_STATUS
    assert binding[BINDING_SELF_HASH_FIELD] == resolved_evidence_binding_self_hash(binding)
    assert BINDING_SELF_HASH_FIELD != SELF_HASH_FIELD, (
        "a binding must not reuse a contract's self-hash field name"
    )


# ---- (b) the prospective contract is cited, never mutated ----


def test_the_binding_pins_path_physical_semantic_and_the_self_hash_field() -> None:
    binding = _build("t1_capacity")
    block = binding["prospective_contract"]
    assert set(block) == {"physical", "self_hash_field", "semantic"}
    assert block["self_hash_field"] == SELF_HASH_FIELD
    raw = (_ROOT / T1_CAPACITY_POLICY).read_bytes()
    assert block["physical"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert block["semantic"]["sha256"] == json.loads(raw)[SELF_HASH_FIELD]
    assert block["physical"]["identity_role"] == IdentityRole.PHYSICAL
    assert block["semantic"]["identity_role"] == IdentityRole.SEMANTIC
    assert block["semantic"]["hash_algorithm"] == "self_hash_field_v1"


def test_a_stale_contract_pin_is_refused() -> None:
    binding = _build()
    binding["prospective_contract"]["physical"]["sha256"] = "0" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="but the live value is"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_binding_that_cites_another_stages_contract_is_refused() -> None:
    binding = _build("gate_zero")
    other = _build("bounded_p50")
    binding["prospective_contract"] = other["prospective_contract"]
    with pytest.raises(ProcessV2EvidenceBindingError, match="is governed by"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_an_absent_contract_makes_the_binding_unbuildable() -> None:
    with _repo_copy() as root:
        (root / GATE_ZERO_STRUCTURAL).unlink()
        with pytest.raises(ProcessV2EvidenceBindingError, match="not readable"):
            _build("gate_zero", repo_root=root)


def test_an_edited_contract_makes_an_existing_binding_stale() -> None:
    binding = _build("t1_panel")
    with _repo_copy() as root:
        target = root / T1_PANEL_POLICY
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(ProcessV2EvidenceBindingError):
            validate_resolved_evidence_binding(binding, repo_root=root)


# ---- (c) the admitted-source schema comes from the adapter ----


def test_the_admitted_source_schema_is_the_adapters_own_constant() -> None:
    """Imported, never restated: a transcribed schema literal already drifted once."""

    binding = _build()
    assert binding["admitted_source"]["schema"] == ADMITTED_SOURCE_SCHEMA
    assert binding["admitted_source"]["schema_version"] == ADMITTED_SOURCE_SCHEMA_VERSION


def test_a_disagreeing_admitted_source_schema_is_refused() -> None:
    binding = _build()
    binding["admitted_source"]["schema"] = "compose.data.process_v2_admitted_source"
    with pytest.raises(ProcessV2EvidenceBindingError, match="the owning adapter declares"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_the_binding_pins_every_required_measured_identity() -> None:
    binding = _build()
    for name in ADMITTED_SOURCE_IDENTITY_FIELDS:
        assert binding["admitted_source"][name] == _FIXTURE[name], name


def test_an_incomplete_admitted_source_identity_is_refused() -> None:
    for name in ADMITTED_SOURCE_IDENTITY_FIELDS:
        source = _admitted_source()
        del source[name]
        with pytest.raises(ProcessV2EvidenceBindingError, match="omits"):
            _build(admitted_source=source)


def test_a_malformed_measured_identity_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="64-character SHA-256"):
        _build(admitted_source=_admitted_source(completion_sha256="short"))
    with pytest.raises(ProcessV2EvidenceBindingError, match="lowercase hexadecimal"):
        _build(admitted_source=_admitted_source(completion_sha256="A" * 64))


# ---- (d) the census must reconcile ----


def test_a_census_that_does_not_reconcile_is_refused() -> None:
    source = _admitted_source(
        counts={"source_entries": 30, "admitted_entries": 22, "rejected_entries": 7}
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="does not reconcile"):
        _build(admitted_source=source)


def test_a_reason_census_that_disagrees_with_the_count_is_refused() -> None:
    source = _admitted_source(rejected_traces_by_code={"fixture_reason_alpha": 5})
    with pytest.raises(ProcessV2EvidenceBindingError, match="disagrees with rejected_entries"):
        _build(admitted_source=source)


def test_a_validated_binding_still_refuses_a_broken_census() -> None:
    binding = _build()
    binding["admitted_source"]["counts"]["rejected_entries"] = 9
    with pytest.raises(ProcessV2EvidenceBindingError, match="does not reconcile"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


# ---- (e) authority ----


def test_every_authority_field_is_present_and_false() -> None:
    binding = _build()
    for name in AUTHORITY_FIELDS:
        assert binding[name] is False, name
    assert "bounded_p50_authorized" in AUTHORITY_FIELDS


def test_a_granted_authority_field_is_refused() -> None:
    for name in AUTHORITY_FIELDS:
        binding = _build()
        binding[name] = True
        with pytest.raises(ProcessV2EvidenceBindingError, # Either refusal is correct: the smuggled source carries the retired
        # spelling as well as the grant, and the shared walk reports whichever
        # it reaches first. Both refuse the binding, which is the property.
        match="grants authority|retired authority field"):
            validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_the_adapters_authority_block_is_not_copied_into_the_binding() -> None:
    """The binding carries exactly one authority block, its own.

    The adapter's descriptor spells its bounded-P50 field differently. Renaming
    it at its source would move the scientific Process-V2 identity, so that is an
    owner decision; translating it here would create a second vocabulary in a
    third place. The binding therefore ingests measured identities and the
    census, and nothing else.
    """

    source = _admitted_source()
    assert "p50_authorized" in source, "the fixture must reflect the adapter as it is"
    binding = _build(admitted_source=source)
    ingested = binding["admitted_source"]
    assert not any(key.endswith("_authorized") for key in ingested), sorted(ingested)
    assert "authority" not in ingested
    assert set(ingested) == {
        *ADMITTED_SOURCE_IDENTITY_FIELDS,
        "counts",
        "rejected_traces_by_code",
        "schema",
        "schema_version",
    }


def test_an_admitted_source_that_grants_authority_is_refused_whatever_the_spelling() -> None:
    """The guard is spelling-agnostic, so it needs no rename to be correct."""

    for field in ("p50_authorized", "bounded_p50_authorized", "training_authorized"):
        with pytest.raises(ProcessV2EvidenceBindingError, # Either refusal is correct: the smuggled source carries the retired
        # spelling as well as the grant, and the shared walk reports whichever
        # it reaches first. Both refuse the binding, which is the property.
        match="grants authority|retired authority field"):
            _build(admitted_source=_admitted_source(**{field: True}))


def test_a_validated_binding_refuses_a_granted_authority_smuggled_into_the_source() -> None:
    binding = _build()
    binding["admitted_source"]["p50_authorized"] = True
    with pytest.raises(ProcessV2EvidenceBindingError, # Either refusal is correct: the smuggled source carries the retired
        # spelling as well as the grant, and the shared walk reports whichever
        # it reaches first. Both refuse the binding, which is the property.
        match="grants authority|retired authority field"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


# ---- (f) counts are exact non-negative integers ----


def test_a_non_integer_count_is_refused() -> None:
    for value in ("22", 22.0, True, None):
        source = _admitted_source(
            counts={
                "source_entries": 30,
                "admitted_entries": value,
                "rejected_entries": 8,
            }
        )
        with pytest.raises(ProcessV2EvidenceBindingError, match="integer count"):
            _build(admitted_source=source)


def test_a_negative_count_is_refused() -> None:
    source = _admitted_source(
        counts={"source_entries": 30, "admitted_entries": 38, "rejected_entries": -8}
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="must not be negative"):
        _build(admitted_source=source)


def test_a_non_integer_rejection_count_is_refused() -> None:
    source = _admitted_source(rejected_traces_by_code={"fixture_reason_alpha": "8"})
    with pytest.raises(ProcessV2EvidenceBindingError, match="integer count"):
        _build(admitted_source=source)


def test_a_missing_census_field_is_refused() -> None:
    source = _admitted_source(counts={"source_entries": 30, "admitted_entries": 22})
    with pytest.raises(ProcessV2EvidenceBindingError, match="omits"):
        _build(admitted_source=source)


# ---- (g) measured prerequisites ----


def test_a_missing_repository_prerequisite_is_refused() -> None:
    binding = _build()
    binding["measured_prerequisites"]["editing_corpus_contract"]["target"] = (
        "configs/no_such_contract.json"
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="which is not present"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_stale_repository_prerequisite_is_refused() -> None:
    binding = _build()
    binding["measured_prerequisites"]["editing_corpus_contract"]["sha256"] = "1" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="the live physical hash is"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_remote_prerequisite_is_never_looked_for_locally() -> None:
    """A volume path is verified by its stage loader, not by this validator."""

    binding = _build()
    remote = binding["measured_prerequisites"]["chunk_cache_completion"]
    assert remote["kind"] == PointerKind.REMOTE_ARTIFACT
    assert not (_ROOT / remote["target"].lstrip("/")).exists()
    assert validate_resolved_evidence_binding(binding, repo_root=_ROOT) == binding


def test_a_malformed_prerequisite_pointer_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not a typed pointer"):
        _build(measured_prerequisites={"broken": {"path": "configs/x.json"}})


def test_prerequisites_are_serialized_in_a_stable_order() -> None:
    binding = _build()
    assert list(binding["measured_prerequisites"]) == sorted(
        binding["measured_prerequisites"]
    )


# ---- (h) self-hash and publication ----


def test_a_disagreeing_self_hash_is_refused() -> None:
    binding = _build()
    binding[BINDING_SELF_HASH_FIELD] = "2" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="self-hash"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_publication_leaves_no_partial_file_behind() -> None:
    binding = _build()
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        path = write_resolved_evidence_binding(binding, path=directory / "binding.json")
        assert path.is_file()
        assert [child.name for child in directory.iterdir()] == ["binding.json"]


def test_an_unparseable_binding_is_refused() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "binding.json"
        path.write_bytes(b"{nope")
        with pytest.raises(ProcessV2EvidenceBindingError, match="not valid JSON"):
            load_resolved_evidence_binding(path, repo_root=_ROOT)


def test_an_absent_binding_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not readable"):
        load_resolved_evidence_binding(Path("/nonexistent/binding.json"), repo_root=_ROOT)


def test_a_non_object_binding_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="must be a JSON object"):
        validate_resolved_evidence_binding([], repo_root=_ROOT)
