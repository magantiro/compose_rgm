from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import compose_v4.experiments.atom_restate_neural_orbit_audit as audit_module
from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.atom_restate_neural_orbit_audit import (
    CURRENT_POLICY_ID,
    AtomRestateNeuralOrbitAuditError,
    TeacherProgressAddress,
    additive_message_orbit_classes,
    aggregate_teacher_audits,
    audit_atom_restate_teacher,
    enumerate_current_broad_restate_candidates,
    load_audit_contract,
    molecular_additive_message_orbits,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/editing_atom_restate_neural_orbit_audit_v1.json"


def _address(*, entry_index: int = 3, progress_index: int = 2) -> TeacherProgressAddress:
    return TeacherProgressAddress(
        packed_shard_content_sha256="a" * 64,
        packed_shard_name="shard_0000.jsonl.gz",
        entry_index=entry_index,
        trace_id=f"trace-{entry_index}",
        layer="operator_aware_real_endpoint",
        partition="train",
        progress_index=progress_index,
    )


def _editing_model(seed: int) -> FactorizedTraceletRateModel:
    torch.manual_seed(seed)
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=24,
        message_passing_steps=6,
        mark_dim=8,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


def test_symbolic_update_keeps_neighbor_and_edge_multisets_separate() -> None:
    # Roots 0 and 1 have the same self label, the same neighbor-label
    # multiset, and the same incident-edge-class multiset, but the edge classes
    # are paired with opposite neighbor labels.  The production additive
    # aggregation loses that pairing in one round, and so must this certificate.
    labels = (
        (2, 0, 2, 0),
        (2, 0, 2, 0),
        (3, 0, 1, 0),
        (4, 0, 0, 0),
        (3, 0, 1, 0),
        (4, 0, 0, 0),
    )
    bonds = np.zeros((6, 6), dtype=np.int64)
    for left, right, edge_class in (
        (0, 2, 1),
        (0, 3, 2),
        (1, 4, 2),
        (1, 5, 1),
    ):
        bonds[left, right] = bonds[right, left] = edge_class
    classes = additive_message_orbit_classes(
        labels,
        bonds,
        (True,) * 6,
        rounds=1,
    )
    assert classes[0] == classes[1]


@pytest.mark.parametrize("seed", (103, 907))
def test_six_round_certificate_implies_identical_scratch_node_and_restate_logits(
    seed: int,
) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    model = _editing_model(seed)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.37,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_system_delete=False,
    )
    with torch.no_grad():
        node, _, _ = model._encode_batch(batch)
        restate_logits = model.restate_head(node)[0]
    orbit_classes = molecular_additive_message_orbits(state, rounds=6)
    real = np.flatnonzero(is_element(state.atom_types))
    for orbit in sorted({orbit_classes[int(vertex)] for vertex in real}):
        members = [int(vertex) for vertex in real if orbit_classes[int(vertex)] == orbit]
        reference = members[0]
        for vertex in members[1:]:
            assert torch.allclose(node[0, reference], node[0, vertex], atol=1e-7, rtol=0.0)
            assert torch.allclose(
                restate_logits[reference],
                restate_logits[vertex],
                atol=1e-7,
                rtol=0.0,
            )


def test_current_enumerator_matches_production_model_mask_exactly() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1CCO"), 16)
    model = _editing_model(211)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_system_delete=False,
    )
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        masks, _, _ = model._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
    model_coordinates = {
        tuple(int(value) for value in coordinate)
        for coordinate in torch.nonzero(masks["atom_restate"][0], as_tuple=False).tolist()
    }
    audited_coordinates = {
        candidate.coordinate for candidate in enumerate_current_broad_restate_candidates(state)
    }
    assert audited_coordinates == model_coordinates


def test_audit_reports_exact_mixed_orbit_ceiling_without_selecting_policy() -> None:
    # The interior of this asymmetric long chain contains sites with identical
    # six-hop views but distinct complete molecular successors.  This provides
    # a deterministic fixture for a strict (<1) information ceiling.
    state = pad_molecular_graph(smiles_to_molecular_graph("N" + "C" * 18), 24)
    candidates = enumerate_current_broad_restate_candidates(state)
    oxygen = ELEMENT_TO_IDX["O"]
    grouped: dict[tuple[int, int], list] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.neural_class_key, []).append(candidate)
    mixed = next(
        members
        for members in grouped.values()
        if members[0].target_class[0] == oxygen
        and len({member.successor_key for member in members}) > 1
    )
    teacher = mixed[0]
    system = de_novo_rewrite_system()
    successor = system.apply(state, "atom_restate", teacher.action)
    row = audit_atom_restate_teacher(
        state,
        teacher.action,
        successor,
        _address(),
    )

    assert row["authorizes_training"] is False
    assert row["selects_support_policy"] is False
    assert row["teacher_coordinate"] == list(teacher.coordinate)
    current = row["policy_comparisons"][0]
    assert current["policy_id"] == CURRENT_POLICY_ID
    ceiling = current["family_conditional_teacher_probability_ceiling"]
    assert 0.0 < ceiling["value"] < 1.0
    assert ceiling["numerator"] < ceiling["denominator"]
    assert current["semantic_counts"]["source_to_target_element"]

    summary = aggregate_teacher_audits((row,))
    aggregate = summary["policy_aggregates"][CURRENT_POLICY_ID]
    assert aggregate["ceiling_below_one_count"] == 1
    assert aggregate["exact_ceiling_below_one_addresses"] == [row["address"]]


def test_contract_is_physically_and_logically_bound_and_train_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_audit_contract(CONTRACT)
    assert contract["authorizes_training"] is False
    assert contract["selects_support_policy"] is False
    assert contract["policies"][0]["hypothetical"] is False
    assert all(item["hypothetical"] for item in contract["policies"][1:])

    mutated = dict(contract)
    mutated["partition_role"] = "validation"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated))
    with pytest.raises(AtomRestateNeuralOrbitAuditError, match="physical SHA-256"):
        load_audit_contract(path)
    monkeypatch.setattr(audit_module, "PINNED_CONTRACT_FILE_SHA256", "")
    with pytest.raises(AtomRestateNeuralOrbitAuditError, match="schema, status"):
        load_audit_contract(path)

    with pytest.raises(ValueError, match="train-only"):
        TeacherProgressAddress(
            packed_shard_content_sha256="b" * 64,
            packed_shard_name="shard.jsonl.gz",
            entry_index=0,
            trace_id="trace",
            layer="lane",
            partition="validation",
            progress_index=0,
        )


def test_teacher_execution_mismatch_fails_loudly() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 8)
    candidate = enumerate_current_broad_restate_candidates(state)[0]
    wrong = pad_molecular_graph(smiles_to_molecular_graph("CCN"), 8)
    assert molecular_graph_to_smiles(wrong) != candidate.successor_key
    with pytest.raises(
        AtomRestateNeuralOrbitAuditError,
        match="exact stored successor",
    ):
        audit_atom_restate_teacher(
            state,
            candidate.action,
            wrong,
            _address(),
        )
