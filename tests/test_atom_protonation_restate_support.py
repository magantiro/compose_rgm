"""Focused gates for the explicit Editing-V3 protonation support."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.protonation_restate_program import (
    compile_protonation_supported_target,
)
from compose_v4.rewrite import action_codec_v4 as v4
from compose_v4.rewrite import action_codec_v5 as v5
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
    editing_v3_protonation_rewrite_system,
)
from compose_v4.rewrite.operators import (
    PROTONATION_STATE_NEUTRAL_TERTIARY,
    PROTONATION_STATE_PROTONATED_TERTIARY,
    AtomProtonationRestate,
    AtomRestate,
    apply_atom_protonation_restate,
    enumerate_atom_protonation_restates,
    inverse_atom_protonation_restate,
    is_valid_atom_protonation_restate,
)
from scripts.audit_t4_protonation_restate_support import run_probe

SOURCE = "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1"
STRICT_ENDPOINTS = (
    "C1=CC2=NC=C(CCCn3nnnc3Cc3ccccc3)C2C=C1n1cnnc1",
    "C1=CC2=NC=C(CN3CCCN(CCCc4ccccc4)CC3)C2C=C1n1cnnc1",
    "C1=CC2=NC=C(CN3CCCN(CCc4ccccc4)CC3)C2C=C1n1cnnc1",
)


def _charged_tertiary_amine():
    graph = smiles_to_molecular_graph("C[NH+](C)C")
    vertex = int(np.flatnonzero(graph.formal_charges == 1)[0])
    return graph, vertex


def test_v5_codec_round_trip_and_historical_rejection() -> None:
    assert v4.codec_implementation_hash() == "fbd57cfdd1553424"
    action = AtomProtonationRestate(3, PROTONATION_STATE_NEUTRAL_TERTIARY)
    record = v5.encode_action("atom_protonation_restate", action)
    assert record == {
        "schema": v5.SCHEMA,
        "schema_version": 5,
        "executor_rule": "atom_protonation_restate",
        "model_family": "atom_protonation_restate",
        "payload_type": "AtomProtonationRestate",
        "payload": {"v": 3, "target_state": PROTONATION_STATE_NEUTRAL_TERTIARY},
    }
    assert v5.decode_action(record) == ("atom_protonation_restate", action)
    assert v5.public_operator_name("atom_protonation_restate") == (
        "atom_protonation_restate"
    )
    with pytest.raises(v4.ActionCodecV4Error, match="outside the frozen Active8"):
        v4.encode_action("atom_protonation_restate", action)
    with pytest.raises(v4.ActionCodecV4Error, match="schema_version"):
        v4.decode_action(record)
    assert "atom_protonation_restate" not in v4.supported_executor_rules()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"model_family": "atom_restate"}, "ontology disagreement"),
        ({"payload_type": "AtomRestate"}, "requires payload"),
        (
            {"payload": {"v": -1, "target_state": PROTONATION_STATE_NEUTRAL_TERTIARY}},
            "non-negative",
        ),
        (
            {"payload": {"v": 1, "target_state": "arbitrary_charge"}},
            "fixed tertiary-amine pair",
        ),
        ({"payload": {"v": 1}}, "payload fields disagree"),
    ),
)
def test_v5_malformed_records_fail_loudly(mutation: dict, match: str) -> None:
    record = v5.encode_action(
        "atom_protonation_restate",
        AtomProtonationRestate(1, PROTONATION_STATE_NEUTRAL_TERTIARY),
    )
    record.update(mutation)
    with pytest.raises(v5.ActionCodecV5Error, match=match):
        v5.decode_action(record)


def test_protonation_restate_changes_only_charge_and_h_and_has_exact_inverse() -> None:
    source, vertex = _charged_tertiary_amine()
    action = AtomProtonationRestate(vertex, PROTONATION_STATE_NEUTRAL_TERTIARY)
    assert is_valid_atom_protonation_restate(source, action)
    successor = editing_v3_protonation_rewrite_system().apply(
        source,
        "atom_protonation_restate",
        action,
    )
    assert canonical_state_key(successor) == "CN(C)C"
    assert np.array_equal(source.atom_types, successor.atom_types)
    assert np.array_equal(source.bonds, successor.bonds)
    assert np.array_equal(
        np.delete(source.formal_charges, vertex),
        np.delete(successor.formal_charges, vertex),
    )
    assert np.array_equal(
        np.delete(source.implicit_h_counts, vertex),
        np.delete(successor.implicit_h_counts, vertex),
    )
    assert (
        int(successor.formal_charges[vertex]),
        int(successor.implicit_h_counts[vertex]),
    ) == (0, 0)
    inverse = inverse_atom_protonation_restate(source, action)
    assert inverse == AtomProtonationRestate(
        vertex,
        PROTONATION_STATE_PROTONATED_TERTIARY,
    )
    restored = editing_v3_protonation_rewrite_system().apply(
        successor,
        "atom_protonation_restate",
        inverse,
    )
    assert canonical_state_key(restored) == canonical_state_key(source)


def test_narrow_validator_rejects_non_nitrogen_wrong_degree_and_self_event() -> None:
    carbon = smiles_to_molecular_graph("CC(C)C")
    assert not is_valid_atom_protonation_restate(
        carbon,
        AtomProtonationRestate(1, PROTONATION_STATE_PROTONATED_TERTIARY),
    )
    secondary = smiles_to_molecular_graph("C[NH2+]C")
    vertex = int(np.flatnonzero(secondary.formal_charges == 1)[0])
    assert not is_valid_atom_protonation_restate(
        secondary,
        AtomProtonationRestate(vertex, PROTONATION_STATE_NEUTRAL_TERTIARY),
    )
    source, vertex = _charged_tertiary_amine()
    assert not is_valid_atom_protonation_restate(
        source,
        AtomProtonationRestate(vertex, PROTONATION_STATE_PROTONATED_TERTIARY),
    )
    malformed = copy.copy(AtomProtonationRestate(vertex, "charge_minus_two"))
    with pytest.raises(ValueError, match="unsupported protonation target"):
        apply_atom_protonation_restate(source, malformed)


def test_old_runtime_rejects_new_action_and_new_runtime_rejects_raw_charge_change() -> (
    None
):
    source, vertex = _charged_tertiary_amine()
    action = AtomProtonationRestate(vertex, PROTONATION_STATE_NEUTRAL_TERTIARY)
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
        editing_v2_semantic_rewrite_system().apply(
            source,
            "atom_protonation_restate",
            action,
        )
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
        editing_v3_protonation_rewrite_system().apply(
            source,
            "atom_restate",
            AtomRestate(vertex, 3, 0, 0),
        )


def test_enumerator_is_exactly_the_named_reversible_pair() -> None:
    charged, vertex = _charged_tertiary_amine()
    assert enumerate_atom_protonation_restates(charged) == (
        AtomProtonationRestate(vertex, PROTONATION_STATE_NEUTRAL_TERTIARY),
    )
    neutral = smiles_to_molecular_graph("CN(C)C")
    neutral_vertex = next(
        index
        for index, atom in enumerate(neutral.atom_types)
        if int(atom) == 3 and int(neutral.bonds[index].sum()) == 3
    )
    assert enumerate_atom_protonation_restates(neutral) == (
        AtomProtonationRestate(
            neutral_vertex,
            PROTONATION_STATE_PROTONATED_TERTIARY,
        ),
    )


def test_teacher_forced_strict_endpoints_have_exact_complete_program_support() -> None:
    results = tuple(
        compile_protonation_supported_target(SOURCE, endpoint)
        for endpoint in STRICT_ENDPOINTS
    )
    assert [row["status"] for row in results] == ["supported"] * 3
    assert [row["primitive_edits"] for row in results] == [23, 21, 13]
    assert all(row["exact_endpoint"] for row in results)
    assert all(row["protonation_restate_count"] == 1 for row in results)
    assert all(
        row["teacher_endpoint_injected_into_autonomous_proposals"] is False
        for row in results
    )
    for row in results:
        assert row["exact_execution_precision_numerator"] == 1
        assert row["exact_execution_precision_denominator"] == 1
        assert len(row["actions"]) == len(row["states"]) - 1


def test_program_compiler_uses_supported_padded_source() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph(SOURCE), 48)
    assert source.n_atoms == 48
    result = compile_protonation_supported_target(SOURCE, STRICT_ENDPOINTS[2])
    assert result["source"] == canonical_state_key(source)


def test_contract_is_self_hashed_and_probe_reports_zero_oracle_exactness() -> None:
    contract_path = Path("configs/t4_atom_protonation_restate_support_v1.json")
    contract = json.loads(contract_path.read_text())
    canonical = json.dumps(
        contract["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == contract["payload_sha256"]
    result = run_probe(
        contract_path=contract_path,
        audit_path=Path(
            "diagnostics/t4_shared_retained_fiber_5ht1b_d06_v1/"
            "audit_20260919/result.json"
        ),
    )
    aggregate = result["payload"]["aggregate"]
    assert result["payload"]["costs"] == {
        "docking_calls": 0,
        "modal_launches": 0,
        "oracle_calls": 0,
    }
    assert aggregate["complete_program_support_numerator"] == 3
    assert aggregate["complete_program_support_denominator"] == 3
    assert aggregate["exact_execution_precision"] == 1.0
    assert aggregate["primitive_counts"] == [23, 21, 13]
    assert aggregate["autonomous_proposal_recovery_evaluated"] is False
