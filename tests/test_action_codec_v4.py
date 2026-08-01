"""Strict compatibility tests for the complete Editing-V2 cycle codec."""

from __future__ import annotations

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.rewrite import action_codec as v2
from compose_v4.rewrite import action_codec_v3 as v3
from compose_v4.rewrite import action_codec_v4 as v4
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_cycle_rewrite_system,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.semantic_trace import invert_semantic_trace
from compose_v4.rewrite.trace import RewriteStep, execute_trace


def test_cycle_close_round_trip_preserves_action_and_ontology() -> None:
    action = CycleCloseEdge(2, 7, 2)
    record = v4.encode_action("cycle_close", action)
    assert record == {
        "schema": v4.SCHEMA,
        "schema_version": 4,
        "executor_rule": "cycle_close",
        "model_family": "cycle_insert",
        "payload_type": "CycleCloseEdge",
        "payload": {"a": 2, "b": 7, "order": 2},
    }
    assert v4.decode_action(record) == ("cycle_close", action)


def test_semantic_atom_restate_round_trip_preserves_target_class() -> None:
    action = SemanticAtomRestate(7, 6)
    record = v4.encode_action("atom_restate_semantic", action)
    assert record == {
        "schema": v4.SCHEMA,
        "schema_version": 4,
        "executor_rule": "atom_restate_semantic",
        "model_family": "atom_restate",
        "payload_type": "SemanticAtomRestate",
        "payload": {"v": 7, "target_class_index": 6},
    }
    assert v4.decode_action(record) == ("atom_restate_semantic", action)


def test_v4_preserves_semantic_cycle_open_without_reinterpreting_v3() -> None:
    action = CycleOpenEdge(1, 4)
    record = v4.encode_action("cycle_open", action)
    assert record["schema_version"] == 4
    assert v4.decode_action(record) == ("cycle_open", action)
    with pytest.raises(v3.ActionCodecV3Error, match="schema_version"):
        v3.decode_action(record)


def test_v4_rejects_both_raw_cycle_micro_actions() -> None:
    with pytest.raises(v4.ActionCodecV4Error, match="outside the frozen Active8"):
        v4.encode_action("bond_insert", BondInsert(0, 5, 1))
    with pytest.raises(v4.ActionCodecV4Error, match="outside the frozen Active8"):
        v4.encode_action("bond_delete", BondDelete(0, 1))
    supported = set(v4.supported_executor_rules())
    assert "cycle_close" in supported
    assert "cycle_open" in supported
    assert "bond_insert" not in supported
    assert "bond_delete" not in supported


def test_v4_rejects_raw_atom_restate_and_older_codecs_reject_semantic() -> None:
    raw = AtomRestate(1, 2, 0, 0)
    semantic = SemanticAtomRestate(1, 2)
    with pytest.raises(v4.ActionCodecV4Error, match="outside the frozen Active8"):
        v4.encode_action("atom_restate", raw)
    with pytest.raises(v3.ActionCodecV3Error):
        v3.encode_action("atom_restate_semantic", semantic)
    with pytest.raises(v2.ActionCodecError):
        v2.encode_action("atom_restate_semantic", semantic)
    supported = set(v4.supported_executor_rules())
    assert "atom_restate_semantic" in supported
    assert "atom_restate" not in supported


def test_v4_codec_and_semantic_runtime_are_exact_active8_allowlists() -> None:
    expected = {
        "atom_insert",
        "atom_delete",
        "atom_restate_semantic",
        "bond_reorder",
        "bond_reroute",
        "cycle_close",
        "cycle_open",
        "ring_system_restate",
    }
    assert set(v4.supported_executor_rules()) == expected
    assert set(editing_v2_semantic_rewrite_system().rules) == expected
    source = smiles_to_molecular_graph("C1CCCCC1")
    for disabled in ("ring_system_delete", "ring_system_grow", "cycle_insert"):
        with pytest.raises(v4.ActionCodecV4Error, match="outside the frozen Active8"):
            v4.canonical_family(disabled)
        with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
            editing_v2_semantic_rewrite_system().apply(source, disabled, object())


def test_v4_rejects_multi_neighbor_birth_from_codec_and_runtime() -> None:
    source = smiles_to_molecular_graph("CC")
    action = AtomInsert(
        slot=2,
        atom_type=2,
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    with pytest.raises(v4.ActionCodecV4Error, match="at most one"):
        v4.encode_action("atom_insert", action)
    with pytest.raises(InvalidRewrite, match="invalid atom_insert"):
        editing_v2_semantic_rewrite_system().apply(source, "atom_insert", action)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"model_family": "atom_insert"}, "ontology disagreement"),
        ({"payload_type": "AtomRestate"}, "requires payload SemanticAtomRestate"),
        ({"payload": {"v": -1, "target_class_index": 1}}, "non-negative"),
        (
            {"payload": {"v": 1, "target_class_index": 15}},
            "outside the broad-organic vocabulary",
        ),
        ({"payload": {"v": 1}}, "payload fields disagree"),
    ),
)
def test_malformed_semantic_atom_restate_records_fail_loudly(
    mutation,
    match: str,
) -> None:
    record = v4.encode_action(
        "atom_restate_semantic",
        SemanticAtomRestate(1, 2),
    )
    record.update(mutation)
    with pytest.raises(v4.ActionCodecV4Error, match=match):
        v4.decode_action(record)


def test_older_codecs_do_not_accept_semantic_cycle_close() -> None:
    action = CycleCloseEdge(0, 5, 1)
    with pytest.raises(v3.ActionCodecV3Error):
        v3.encode_action("cycle_close", action)
    with pytest.raises(v2.ActionCodecError):
        v2.encode_action("cycle_close", action)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"model_family": "cycle_attach"}, "ontology disagreement"),
        ({"payload_type": "BondInsert"}, "requires payload CycleCloseEdge"),
        ({"payload": {"a": 4, "b": 1, "order": 1}}, "strictly increasing"),
        ({"payload": {"a": 1, "b": 4, "order": 4}}, "one of 1, 2, or 3"),
        ({"payload": {"a": 1, "b": 4}}, "payload fields disagree"),
    ),
)
def test_malformed_cycle_close_records_fail_loudly(mutation, match: str) -> None:
    record = v4.encode_action("cycle_close", CycleCloseEdge(1, 4, 1))
    record.update(mutation)
    with pytest.raises(v4.ActionCodecV4Error, match=match):
        v4.decode_action(record)


def test_v4_cycle_close_codec_replay_matches_direct_execution() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    action = CycleCloseEdge(0, 5, 1)
    runtime = editing_v2_semantic_cycle_rewrite_system()
    direct = runtime.apply(source, "cycle_close", action)
    rule, decoded = v4.decode_action(v4.encode_action("cycle_close", action))
    replayed = runtime.apply(source, rule, decoded)
    assert canonical_state_key(replayed) == canonical_state_key(direct)


def test_codec_hash_changes_from_v3_and_is_deterministic() -> None:
    assert v4.codec_implementation_hash() == v4.codec_implementation_hash()
    assert v4.codec_implementation_hash() != v3.codec_implementation_hash()


def test_semantic_cycle_trace_inversion_is_v4_encodable_and_executable() -> None:
    runtime = editing_v2_semantic_cycle_rewrite_system()
    source = smiles_to_molecular_graph("CCCCCC")
    forward = (RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),)
    closed = execute_trace(source, forward, system=runtime)
    inverse = invert_semantic_trace(source, forward, system=runtime)
    assert inverse == (RewriteStep("cycle_open", CycleOpenEdge(0, 5)),)
    for step in (*forward, *inverse):
        assert v4.decode_action(v4.encode_action(step.rule_name, step.action)) == (
            step.rule_name,
            step.action,
        )
    restored = execute_trace(closed, inverse, system=runtime)
    assert canonical_state_key(restored) == canonical_state_key(source)
