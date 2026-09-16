from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.complete_region_where_policy import (
    WhereMaskDecision,
    declared_where_support_size,
    decode_where_mask,
    encode_where_mask,
    source_component_count,
)
from compose_v4.control.docking_value import identity

CONTRACT = Path("configs/t4_complete_region_patch_policy_v2.json")


def _chain() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:4] = 2
    hydrogens[:4] = (3, 2, 2, 3)
    for left in range(3):
        bonds[left, left + 1] = bonds[left + 1, left] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def test_where_mask_roundtrips_connected_and_disconnected_role_sets() -> None:
    graph = _chain()
    for selected in ((0, 1, 2), (0, 3)):
        decision = encode_where_mask(graph, selected)
        assert set(decode_where_mask(graph, decision)) == set(selected)
    assert source_component_count(graph, (0, 1, 2)) == 1
    assert source_component_count(graph, (0, 3)) == 2
    assert declared_where_support_size(graph) == 15


def test_where_mask_rejects_empty_or_wrong_length() -> None:
    graph = _chain()
    with pytest.raises(ValueError, match="nonempty"):
        encode_where_mask(graph, ())
    with pytest.raises(ValueError, match="cannot select the empty"):
        WhereMaskDecision((0, 0, 0, 0))
    with pytest.raises(ValueError, match="length differs"):
        decode_where_mask(graph, WhereMaskDecision((1, 0)))


def test_attempt_2_contract_is_self_hashed_and_zero_cost() -> None:
    envelope = json.loads(CONTRACT.read_text())
    assert envelope["contract_sha256"] == identity(envelope["payload"])
    assert all(value == 0 for value in envelope["payload"]["costs"].values())
    assert envelope["payload"]["where"]["beam_budget"] == 128
