"""Production-boundary tests for the explicit semantic cycle-open action."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cycle_open_kekule_invariance import (
    build_alternate_kekule_pair,
)
from compose_v4.experiments.editing_cycle_open_global_equivalence import (
    semantic_sha256,
)
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_rewrite_system,
)
from compose_v4.rewrite.operators import (
    BondDelete,
    CycleOpenEdge,
    enumerate_cycle_open_edges,
    inverse_cycle_open_edge,
)

CONTRACT = Path("configs/editing_semantic_cycle_open_integration_v1.json")


def _cycle_rank(state) -> int:
    real = tuple(
        index for index, value in enumerate(is_element(state.atom_types)) if value
    )
    edges = sum(
        int(state.bonds[left, right]) != 0
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
    )
    return edges - len(real) + (1 if real else 0)


def test_frozen_integration_contract_is_self_hashed_and_non_authorizing() -> None:
    value = json.loads(CONTRACT.read_text())
    claimed = value.pop("contract_sha256")
    assert claimed == "34fe06c85ac4888b947a57482090105afb96cf320fcd410956b317234c78f451"
    assert semantic_sha256(value) == claimed
    assert value["training_authorized"] is False
    assert value["action_ontology"]["executor_rule"] == "cycle_open"
    assert value["action_ontology"]["model_family"] == "cycle_attach"


@pytest.mark.parametrize(
    "smiles",
    (
        "C1CCCCC1",
        "c1ccccc1",
        "Cc1cccc(Cl)c1",
        "c1ccc2ccccc2c1",
    ),
)
def test_enumerator_equals_runtime_admission_and_every_mark_executes(
    smiles: str,
) -> None:
    source = smiles_to_molecular_graph(smiles)
    runtime = editing_v2_rewrite_system()
    enumerated = set(enumerate_cycle_open_edges(source))
    candidates = {
        CycleOpenEdge(left, right)
        for left in range(source.n_atoms)
        for right in range(left + 1, source.n_atoms)
        if int(source.bonds[left, right]) != 0
    }
    admitted = set()
    for action in candidates:
        try:
            successor = runtime.apply(source, "cycle_open", action)
        except InvalidRewrite:
            continue
        admitted.add(action)
        assert canonical_state_key(successor)
        assert _cycle_rank(successor) == _cycle_rank(source) - 1
    assert enumerated == admitted


def test_semantic_action_is_not_interchangeable_with_raw_bond_delete() -> None:
    source = smiles_to_molecular_graph("Cc1cccc(Cl)c1")
    action = enumerate_cycle_open_edges(source)[0]
    runtime = editing_v2_rewrite_system()
    with pytest.raises(InvalidRewrite, match="expects CycleOpenEdge"):
        runtime.apply(source, "cycle_open", BondDelete(action.a, action.b))
    with pytest.raises(InvalidRewrite, match="expects BondDelete"):
        runtime.apply(source, "bond_delete", action)


def test_alternate_kekule_sources_induce_the_same_semantic_successor_support() -> None:
    pair = build_alternate_kekule_pair(
        pad_molecular_graph(smiles_to_molecular_graph("Cc1cccc(Cl)c1"), 16)
    )
    runtime = editing_v2_rewrite_system()

    def support(source) -> set[str]:
        return {
            canonical_state_key(runtime.apply(source, "cycle_open", action))
            for action in enumerate_cycle_open_edges(source)
        }

    assert support(pair.original) == support(pair.alternate)


def test_slot_relabeling_preserves_the_semantic_successor_support() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("c1ccc2ccccc2c1"), 16)
    permutation = (7, 2, 9, 0, 4, 1, 6, 3, 5, 8, 15, 10, 14, 11, 13, 12)
    relabeled = permute_persistent_slots(source, permutation)
    runtime = editing_v2_rewrite_system()

    def support(state) -> set[str]:
        return {
            canonical_state_key(runtime.apply(state, "cycle_open", action))
            for action in enumerate_cycle_open_edges(state)
        }

    assert support(source) == support(relabeled)


def test_cycle_close_inverse_recovers_canonical_source_identity() -> None:
    runtime = editing_v2_rewrite_system()
    for smiles in ("C1CCCCC1", "c1ccccc1", "c1ccc2ccccc2c1"):
        source = smiles_to_molecular_graph(smiles)
        for action in enumerate_cycle_open_edges(source):
            successor = runtime.apply(source, "cycle_open", action)
            inverse = inverse_cycle_open_edge(source, action)
            restored = runtime.apply(successor, "bond_insert", inverse)
            assert canonical_state_key(restored) == canonical_state_key(source)


def test_endpoint_order_is_part_of_the_new_action_contract() -> None:
    source = smiles_to_molecular_graph("C1CCCCC1")
    runtime = editing_v2_rewrite_system()
    action = enumerate_cycle_open_edges(source)[0]
    with pytest.raises(InvalidRewrite, match="invalid cycle_open"):
        runtime.apply(source, "cycle_open", CycleOpenEdge(action.b, action.a))


def test_legacy_and_de_novo_runtimes_do_not_silently_gain_the_new_rule() -> None:
    source = smiles_to_molecular_graph("C1CCCCC1")
    action = enumerate_cycle_open_edges(source)[0]
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
        de_novo_rewrite_system().apply(source, "cycle_open", action)
