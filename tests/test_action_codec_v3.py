"""Editing-V2 action-codec isolation and round-trip tests."""

from __future__ import annotations

import copy

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.rewrite import action_codec as v2
from compose_v4.rewrite import action_codec_v3 as v3
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import (
    AtomDelete,
    BondDelete,
    CycleOpenEdge,
    enumerate_cycle_open_edges,
)


def test_cycle_open_round_trip_preserves_new_action_and_ontology() -> None:
    action = CycleOpenEdge(2, 7)
    record = v3.encode_action("cycle_open", action)
    assert record == {
        "schema": "compose.rewrite.action",
        "schema_version": 3,
        "executor_rule": "cycle_open",
        "model_family": "cycle_attach",
        "payload_type": "CycleOpenEdge",
        "payload": {"a": 2, "b": 7},
    }
    assert v3.decode_action(record) == ("cycle_open", action)
    assert v3.canonical_json(v3.encode_action(*v3.decode_action(record))) == (
        v3.canonical_json(record)
    )


def test_unchanged_payload_round_trips_under_v3_without_changing_v2() -> None:
    action = AtomDelete(4)
    record_v2 = v2.encode_action("atom_delete", action)
    record_v3 = v3.encode_action("atom_delete", action)
    assert record_v2["schema_version"] == 2
    assert record_v3["schema_version"] == 3
    assert {
        key: value for key, value in record_v2.items() if key != "schema_version"
    } == {key: value for key, value in record_v3.items() if key != "schema_version"}
    assert v2.decode_action(record_v2) == ("atom_delete", action)
    assert v3.decode_action(record_v3) == ("atom_delete", action)


def test_v2_and_v3_readers_do_not_silently_cross_versions() -> None:
    v2_record = v2.encode_action("bond_delete", BondDelete(1, 4))
    v3_record = v3.encode_action("cycle_open", CycleOpenEdge(1, 4))
    with pytest.raises(v3.ActionCodecV3Error, match="schema_version"):
        v3.decode_action(v2_record)
    with pytest.raises(v2.ActionCodecError, match="schema_version"):
        v2.decode_action(v3_record)


def test_v3_rejects_raw_bond_delete_and_v2_rejects_semantic_cycle_open() -> None:
    with pytest.raises(v3.ActionCodecV3Error, match="raw bond_delete"):
        v3.encode_action("bond_delete", BondDelete(1, 4))
    with pytest.raises(v2.ActionCodecError, match="unknown executor rule"):
        v2.encode_action("cycle_open", CycleOpenEdge(1, 4))


def test_v3_supported_rules_replace_raw_delete_with_semantic_open() -> None:
    supported = set(v3.supported_executor_rules())
    assert "cycle_open" in supported
    assert "bond_delete" not in supported
    assert supported == set(v2.supported_executor_rules()) - {"bond_delete"} | {
        "cycle_open"
    }
    assert v3.canonical_family("cycle_open") == "cycle_attach"
    assert v3.public_operator_name("cycle_attach") == "cycle_open"


def test_cycle_open_codec_replay_matches_direct_runtime_execution() -> None:
    source = smiles_to_molecular_graph("Cc1cccc(Cl)c1")
    action = enumerate_cycle_open_edges(source)[0]
    runtime = editing_v2_rewrite_system()
    direct = runtime.apply(source, "cycle_open", action)
    rule, decoded = v3.decode_action(v3.encode_action("cycle_open", action))
    replayed = runtime.apply(source, rule, decoded)
    assert canonical_state_key(replayed) == canonical_state_key(direct)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda record: record["payload"].update({"a": True}), "expected int"),
        (lambda record: record["payload"].update({"extra": 1}), "fields disagree"),
        (lambda record: record.update({"model_family": "bond_reorder"}), "ontology"),
        (
            lambda record: record.update({"payload_type": "BondDelete"}),
            "requires payload",
        ),
    ),
)
def test_malformed_cycle_open_records_fail_loudly(mutation, match: str) -> None:
    record = v3.encode_action("cycle_open", CycleOpenEdge(1, 4))
    corrupted = copy.deepcopy(record)
    mutation(corrupted)
    with pytest.raises(v3.ActionCodecV3Error, match=match):
        v3.decode_action(corrupted)


def test_endpoint_order_and_payload_type_are_strict() -> None:
    with pytest.raises(v3.ActionCodecV3Error, match="strictly increasing"):
        v3.encode_action("cycle_open", CycleOpenEdge(4, 1))
    with pytest.raises(v3.ActionCodecV3Error, match="requires payload"):
        v3.encode_action("cycle_open", BondDelete(1, 4))


def test_v3_codec_identity_is_stable_and_distinct_from_v2() -> None:
    assert len(v3.codec_implementation_hash()) == 16
    assert v3.codec_implementation_hash() == v3.codec_implementation_hash()
    assert v3.codec_implementation_hash() != v2.codec_implementation_hash()
