"""The frozen Process-V2 interfaces must be enforced, not merely declared.

An adversarial review found this module with zero importers and zero tests, and
drew the right conclusion: every guarantee in its docstring was unenforced, so
every mutation to it survived the suite trivially. A declaration nothing checks
is not a contract.

Each test below corresponds to an attack that succeeded against the first
version of this module. They are kept as tests rather than as fixes-with-a-
comment because the failure mode in every case was the same: a check that looked
right at the top level, or on a well-formed input, and was never given a
malformed one.
"""

from __future__ import annotations

import pytest

from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    CENSUS_FIELDS,
    RETIRED_AUTHORITY_FIELDS,
    SEMANTIC_HASH_ALGORITHMS,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    StructuralDecisionIndex,
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    require_census_reconciles,
    require_no_granted_authority,
    self_hashed,
    typed_pointer,
    validate_typed_pointer,
    verify_self_hash,
)

_HASH = "a" * 64


def _census(source: int = 4, admitted: int = 3, rejected: int = 1) -> dict[str, int]:
    return {
        "source_entries": source,
        "admitted_entries": admitted,
        "rejected_entries": rejected,
    }


# ---- Authority, at every depth ----


def test_authority_false_block_is_accepted_and_complete() -> None:
    block = authority_false_block()
    assert set(block) == set(AUTHORITY_FIELDS)
    require_authority_false(block, label="probe")


@pytest.mark.parametrize("field", AUTHORITY_FIELDS)
def test_a_granted_authority_field_is_refused_at_the_top_level(field: str) -> None:
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_authority_false({**authority_false_block(), field: True}, label="probe")


def test_a_nested_granted_authority_field_is_refused() -> None:
    """The place authority actually lives in this repository.

    `build_editing_process_v2_contract()` puts its authority under `.authority`
    and the census module under `.decisions`. A top-level-only scan passes a
    nested grant, which means the one function whose job is to refuse a granted
    field cannot see the grant.
    """

    payload = {
        **authority_false_block(),
        "authority": {"training_authorized": True, "bounded_p50_authorized": True},
    }
    with pytest.raises(ProcessV2SchemaError, match=r"grants authority at '\.authority\."):
        require_authority_false(payload, label="probe")


def test_a_list_nested_granted_authority_field_is_refused() -> None:
    payload = {
        **authority_false_block(),
        "stages": [{"name": "p50"}, {"t1_authorized": True}],
    }
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_authority_false(payload, label="probe")


@pytest.mark.parametrize("retired", sorted(RETIRED_AUTHORITY_FIELDS))
def test_a_retired_authority_spelling_is_refused_at_any_depth(retired: str) -> None:
    with pytest.raises(ProcessV2SchemaError, match="retired authority field"):
        require_authority_false({**authority_false_block(), retired: False}, label="p")
    with pytest.raises(ProcessV2SchemaError, match="retired authority field"):
        require_authority_false(
            {**authority_false_block(), "authority": {retired: False}}, label="p"
        )


@pytest.mark.parametrize("truthy", [True, 1, "true", [1], {"a": 1}])
def test_only_the_exact_false_singleton_satisfies_an_authority_field(truthy) -> None:
    """`is not False` rather than falsiness: a truthy-looking value is a grant."""

    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_authority_false(
            {**authority_false_block(), "t1_authorized": truthy}, label="probe"
        )


def test_an_omitted_top_level_authority_field_is_refused() -> None:
    block = authority_false_block()
    del block["t1_authorized"]
    with pytest.raises(ProcessV2SchemaError, match="omits authority fields"):
        require_authority_false(block, label="probe")


def test_an_out_of_vocabulary_authority_spelling_is_refused_by_the_free_guard() -> None:
    """The gap `require_authority_false` cannot close, by construction.

    It judges only the names it knows. `AUTHORITY_FIELDS` is the vocabulary of
    artifacts that must never grant, and it is deliberately not the only
    authority vocabulary here: `p500_authorized` is a real stage field, declared
    in the `p50_recipe_body` projection of the Process-V2 policy registry and
    enforced by the P50 execution contract, and it is correctly absent from
    `AUTHORITY_FIELDS`. Correctly absent is still invisible, so an artifact that
    must not grant needs a guard that reads no vocabulary at all.
    """

    payload = {**authority_false_block(), "p500_authorized": True}
    require_authority_false(payload, label="probe")  # blind to it, as designed
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_no_granted_authority(payload, label="probe")


def test_the_vocabulary_free_guard_refuses_a_grant_nested_in_a_mapping_or_a_list() -> None:
    """The flat predecessor of this guard passed both of these.

    A descriptor is embedded verbatim inside a binding and inside the Active8
    source identity, so a grant a consumer must not honour is nested by
    construction rather than by accident.
    """

    nested = {**authority_false_block(), "admitted_source": {"t1_authorized": True}}
    with pytest.raises(ProcessV2SchemaError, match=r"\.admitted_source\.t1_authorized"):
        require_no_granted_authority(nested, label="probe")

    in_list = {**authority_false_block(), "stages": [{"ok": 1}, {"gate_zero_authorized": 1}]}
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_no_granted_authority(in_list, label="probe")

    # A tuple, not only a list. These guards run on in-memory payloads at build
    # time, before anything is serialized, and a tuple is what a builder that
    # froze its collections naturally produces. An adversarial review found the
    # tuple branch of the walk had no test, so dropping it survived.
    in_tuple = {**authority_false_block(), "stages": ({"ok": 1}, {"t1_authorized": True})}
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_no_granted_authority(in_tuple, label="probe")
    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_authority_false(in_tuple, label="probe")


def test_the_vocabulary_free_guard_accepts_a_wholly_false_payload() -> None:
    """It refuses grants only. It must not also require a vocabulary."""

    require_no_granted_authority(
        {**authority_false_block(), "nested": {"p500_authorized": False}}, label="probe"
    )
    require_no_granted_authority({"unrelated": "value"}, label="probe")


@pytest.mark.parametrize("truthy", [True, 1, "false", [], {}, 0.0, None])
def test_the_vocabulary_free_guard_accepts_only_the_exact_false_singleton(truthy) -> None:
    """`is not False`, so an empty list and `None` are grants, not denials.

    A field spelled like authority whose value is not exactly `False` has not
    denied anything, whatever it evaluates to.
    """

    with pytest.raises(ProcessV2SchemaError, match="grants authority"):
        require_no_granted_authority({"future_stage_authorized": truthy}, label="probe")


# ---- Census: exact counts, never coerced ----


def test_a_reconciling_census_is_accepted() -> None:
    require_census_reconciles(_census(), label="probe")


def test_a_census_that_does_not_reconcile_is_refused() -> None:
    with pytest.raises(ProcessV2SchemaError, match="does not reconcile"):
        require_census_reconciles(_census(admitted=2), label="probe")


@pytest.mark.parametrize("value", ["646779", 3.0, True, b"3"])
def test_a_coerced_census_count_is_refused(value) -> None:
    """`int()` accepts a string, a float, a bool and bytes.

    `{"admitted_entries": "646779"}` reconciled while meaning something else,
    and `int(3.9) == 3` silently discarded a record.
    """

    with pytest.raises(ProcessV2SchemaError, match="not an\n?\\s*exact int|exact int"):
        require_census_reconciles(
            {**_census(), "admitted_entries": value}, label="probe"
        )


def test_a_negative_census_count_is_refused() -> None:
    """A negative rejection reconciles an inflated admitted total.

    That is the silent support inflation the census exists to stop: 646780
    admitted and -1 rejected reconciles against 646779 source.
    """

    with pytest.raises(ProcessV2SchemaError, match="negative"):
        require_census_reconciles(
            {"source_entries": 4, "admitted_entries": 5, "rejected_entries": -1},
            label="probe",
        )


@pytest.mark.parametrize("field", CENSUS_FIELDS)
def test_an_omitted_census_field_is_refused(field: str) -> None:
    counts = _census()
    del counts[field]
    with pytest.raises(ProcessV2SchemaError, match="omits"):
        require_census_reconciles(counts, label="probe")


# ---- Typed pointers ----


def _semantic_pointer(**overrides):
    keywords = {
        "kind": PointerKind.REPOSITORY_CONFIG,
        "provider": "probe",
        "target": "configs/x.json",
        "target_schema": "compose.x",
        "identity_role": IdentityRole.SEMANTIC,
        "sha256": _HASH,
        "hash_algorithm": SEMANTIC_HASH_ALGORITHMS[0],
    }
    keywords.update(overrides)
    return typed_pointer(**keywords)


def test_a_well_formed_pointer_round_trips() -> None:
    pointer = _semantic_pointer()
    assert validate_typed_pointer(pointer, label="p") == pointer


@pytest.mark.parametrize("bad", ["z" * 64, _HASH.upper(), "a" * 63, "a" * 65, " " + "a" * 63])
def test_a_malformed_or_uppercase_sha256_is_refused(bad: str) -> None:
    """Case matters: the same digest in two cases hashes to two pointers.

    A length-and-type check alone accepted `"z"*64` and both cases of one
    digest, so a chain rebuild was not byte-stable across a case change.
    """

    with pytest.raises(ProcessV2SchemaError, match="malformed SHA-256"):
        _semantic_pointer(sha256=bad)


def test_a_process_identity_pointer_cannot_address_a_file() -> None:
    """The role that addresses no file must not be pinned to a config path."""

    with pytest.raises(ProcessV2SchemaError, match="addresses no file"):
        typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="probe",
            target="configs/x.json",
            target_schema=None,
            identity_role=IdentityRole.PROCESS_IDENTITY,
            sha256=_HASH,
        )


def test_a_semantic_pointer_must_declare_a_known_algorithm() -> None:
    with pytest.raises(ProcessV2SchemaError, match="hash algorithm"):
        _semantic_pointer(hash_algorithm=None)
    with pytest.raises(ProcessV2SchemaError, match="hash algorithm"):
        _semantic_pointer(hash_algorithm="self_hash_field_v9")


def test_a_non_semantic_pointer_must_not_declare_an_algorithm() -> None:
    """An algorithm that is never applied misdescribes the pin."""

    with pytest.raises(ProcessV2SchemaError, match="must not declare"):
        typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="probe",
            target="configs/x.json",
            target_schema=None,
            identity_role=IdentityRole.PHYSICAL,
            sha256=_HASH,
            hash_algorithm=SEMANTIC_HASH_ALGORITHMS[0],
        )


def test_a_lineage_reference_may_carry_a_semantic_hash() -> None:
    """Recording a superseded contract's own self-hash IS lineage.

    An earlier version of this rule forbade it, which broke the single thing
    the specification requires lineage for: preserving the superseded design
    contract hashes. Only an external asset is excluded, because it declares no
    self-hash and is pinned by its bytes.
    """

    pointer = _semantic_pointer(kind=PointerKind.LINEAGE_REFERENCE)
    assert validate_typed_pointer(pointer, label="p") == pointer


def test_an_external_asset_has_no_self_hash_to_address() -> None:
    with pytest.raises(ProcessV2SchemaError, match="declares no self-hash"):
        _semantic_pointer(kind=PointerKind.EXTERNAL_ASSET)


def test_validate_type_checks_rather_than_coercing() -> None:
    """A coercing validator is weaker than its own builder.

    An object whose `__str__` returns 64 characters passed validation while
    failing `typed_pointer`'s own `isinstance` check.
    """

    class _LooksLikeAHash:
        def __str__(self) -> str:  # pragma: no cover - never reached
            return _HASH

    pointer = dict(_semantic_pointer())
    pointer["sha256"] = _LooksLikeAHash()
    with pytest.raises(ProcessV2SchemaError, match="not str"):
        validate_typed_pointer(pointer, label="p")

    pointer = dict(_semantic_pointer())
    pointer["target_schema"] = {"nested": "object"}
    with pytest.raises(ProcessV2SchemaError, match="not str or None"):
        validate_typed_pointer(pointer, label="p")


@pytest.mark.parametrize("mutation", [{"extra": 1}, {}])
def test_a_pointer_field_set_must_match_exactly(mutation: dict) -> None:
    pointer = dict(_semantic_pointer())
    pointer.update(mutation)
    if not mutation:
        pointer.pop("provider")
    with pytest.raises(ProcessV2SchemaError, match="not a typed pointer"):
        validate_typed_pointer(pointer, label="p")


# ---- Canonical hashing and self-hash sealing ----


def test_canonical_hash_is_order_stable_and_refuses_nonfinite() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_sha256({"x": value})


def test_seal_and_verify_round_trip_and_detect_mutation() -> None:
    sealed = self_hashed({"a": 1, "nested": {"b": 2}}, field="contract_sha256")
    verify_self_hash(sealed, field="contract_sha256", label="probe")
    mutated = {**sealed, "nested": {"b": 3}}
    with pytest.raises(ProcessV2SchemaError, match="disagrees"):
        verify_self_hash(mutated, field="contract_sha256", label="probe")


def test_sealing_a_body_that_already_carries_the_field_is_refused() -> None:
    with pytest.raises(ProcessV2SchemaError, match="already carries"):
        self_hashed({"contract_sha256": _HASH}, field="contract_sha256")


def test_a_non_string_self_hash_cannot_verify_any_body() -> None:
    """An object with a permissive `__eq__` otherwise verifies anything."""

    class _EqualsEverything:
        def __eq__(self, other: object) -> bool:
            return True

    with pytest.raises(ProcessV2SchemaError, match="not a SHA-256 string"):
        verify_self_hash(
            {"a": 1, "contract_sha256": _EqualsEverything()},
            field="contract_sha256",
            label="probe",
        )


# ---- The structural protocol ----


class _Index:
    process_identity_sha256 = "b" * 64
    completion_sha256 = "c" * 64

    def counts(self):
        return _census()

    def rejected_traces_by_code(self):
        return {}

    def identity(self):
        return {"probe": True}


def test_a_structural_index_satisfies_the_protocol_without_subclassing() -> None:
    index = _Index()
    assert isinstance(index, StructuralDecisionIndex)
    assert not any(
        "editing_v2_semantic_active8" in cls.__module__ for cls in type(index).__mro__
    )
    require_census_reconciles(index.counts(), label="probe")


def test_an_object_missing_a_protocol_member_is_not_an_index() -> None:
    class _Partial:
        process_identity_sha256 = "b" * 64

    assert not isinstance(_Partial(), StructuralDecisionIndex)


def test_the_protocol_alone_does_not_prove_which_process_produced_the_index() -> None:
    """A runtime-checkable protocol checks attribute presence, nothing more.

    Recorded deliberately: a V1 index pinning the live V1 identity satisfies
    this protocol. The protocol establishes shape so a V2 consumer need not
    subclass a V1 loader; it is NOT an identity check, and every consumer must
    assert the identity separately. Reading `isinstance` as proof of process is
    exactly the conflation this module exists to prevent.
    """

    class _V1Shaped(_Index):
        process_identity_sha256 = (
            "0a10a2fae24d51853dc30674e31842ea11313571d991e822f23dab7e1124445b"
        )

    assert isinstance(_V1Shaped(), StructuralDecisionIndex)
