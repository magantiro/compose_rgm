"""Semantic partial transfer for the widened B->B-edit categorical heads (train_tracelet_cnof_gate).

Function-level contract tests (no training): shared (element,valence) rows copy by semantic LABEL, new rows
stay fresh, weights and biases both transfer, role-major grow_option maps per (class, role), a reordered
destination vocabulary still copies the correct semantic rows, and duplicate / dropped labels fail loudly.
restate_order_embedding is bond-order indexed and must never be treated as atom-vocabulary indexed. Training-
level gradient/optimizer/save-reload checks live with the wired trainer path."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.chem.molecular_graph import CNOF_VOCABULARY, ORGANIC_VOCABULARY  # noqa: E402
from train_tracelet_cnof_gate import (  # noqa: E402
    _ATOM_VOCAB_HEAD_LAYOUT,
    _semantic_partial_checkpoint_initialization,
    _semantic_row_map,
)

_HID, _MARK = 8, 4
_CNOF = CNOF_VOCABULARY.classes
_ORG = ORGANIC_VOCABULARY.classes
_NSRC, _NDEST = len(_CNOF), len(_ORG)


def _source_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "body.weight": torch.randn(_HID, _HID),
        "atom_embedding.weight": torch.randn(12, _HID),
        "restate_head.weight": torch.randn(_NSRC, _HID),
        "restate_head.bias": torch.randn(_NSRC),
        "grow_root_head.weight": torch.randn(_NSRC, _HID),
        "grow_root_head.bias": torch.randn(_NSRC),
        "grow_option.weight": torch.randn(3 * _NSRC, _MARK),
        "restate_order_embedding.weight": torch.randn(4, _HID),  # bond-order axis (NOT atom vocab)
    }


def _target_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    return {
        "body.weight": torch.randn(_HID, _HID),
        "atom_embedding.weight": torch.randn(12, _HID),
        "restate_head.weight": torch.randn(_NDEST, _HID),
        "restate_head.bias": torch.randn(_NDEST),
        "grow_root_head.weight": torch.randn(_NDEST, _HID),
        "grow_root_head.bias": torch.randn(_NDEST),
        "grow_option.weight": torch.randn(3 * _NDEST, _MARK),
        "restate_order_embedding.weight": torch.randn(4, _HID),
    }


def _run(source_classes=_CNOF, dest_classes=_ORG):
    src, tgt = _source_state(), _target_state()
    fresh = {k: v.clone() for k, v in tgt.items()}
    init, copied, partial, table, map_hash = _semantic_partial_checkpoint_initialization(
        tgt, {"state_dict": src}, source_classes=source_classes, dest_classes=dest_classes
    )
    return src, fresh, init, copied, partial, table, map_hash


def test_shared_rows_copy_by_semantic_label():
    src, _fresh, init, *_ = _run()
    idx = {c: i for i, c in enumerate(_CNOF)}
    for head in ("restate_head.weight", "restate_head.bias", "grow_root_head.weight", "grow_root_head.bias"):
        for j, label in enumerate(_ORG):
            if label in idx:
                assert torch.equal(init[head][j], src[head][idx[label]]), f"{head} row {j} {label}"


def test_grow_option_role_major_shared_rows_copy():
    src, _fresh, init, *_ = _run()
    idx = {c: i for i, c in enumerate(_CNOF)}
    for role in range(3):
        for j, label in enumerate(_ORG):
            if label in idx:
                dest_row = role * _NDEST + j
                src_row = role * _NSRC + idx[label]
                assert torch.equal(init["grow_option.weight"][dest_row], src["grow_option.weight"][src_row])


def test_new_class_rows_stay_fresh():
    _src, fresh, init, *_ = _run()
    shared = set(_CNOF)
    for head in ("restate_head.weight", "restate_head.bias", "grow_root_head.weight", "grow_root_head.bias"):
        for j, label in enumerate(_ORG):
            if label not in shared:
                assert torch.equal(init[head][j], fresh[head][j]), f"{head} new row {j} not fresh"
    for role in range(3):
        for j, label in enumerate(_ORG):
            if label not in shared:
                dr = role * _NDEST + j
                assert torch.equal(init["grow_option.weight"][dr], fresh["grow_option.weight"][dr])


def test_weight_and_bias_both_transfer():
    src, _fresh, init, *_ = _run()
    assert torch.equal(init["restate_head.bias"][:_NSRC], src["restate_head.bias"])
    assert torch.equal(init["grow_root_head.bias"][:_NSRC], src["grow_root_head.bias"])


def test_reordered_dest_vocab_maps_by_label_not_position():
    # Reverse the four CNOF slots so shared labels sit at NEW indices; copy must still be by label.
    reordered = (_CNOF[3], _CNOF[2], _CNOF[1], _CNOF[0]) + tuple(_ORG[_NSRC:])
    src, _fresh, init, *_ = _run(dest_classes=reordered)
    src_idx = {c: i for i, c in enumerate(_CNOF)}
    for j, label in enumerate(reordered):
        if label in src_idx:
            assert torch.equal(init["restate_head.weight"][j], src["restate_head.weight"][src_idx[label]])


def test_duplicate_label_fails_loudly():
    with pytest.raises(ValueError, match="duplicate"):
        _semantic_row_map((_CNOF[0], _CNOF[0], _CNOF[1]), _ORG)
    with pytest.raises(ValueError, match="duplicate"):
        _semantic_row_map(_CNOF, (_ORG[0],) + _ORG)


def test_source_class_absent_from_dest_fails_loudly():
    # a source class not present in dest would silently drop a learned row -> must raise
    with pytest.raises(ValueError, match="absent from destination"):
        _semantic_row_map(_CNOF, _ORG[1:])  # drop (C,4) from dest


def test_restate_order_embedding_is_not_atom_vocab_indexed():
    # Guard: bond-order-indexed tensors must never be in the atom-vocab head registry (would mis-map).
    assert "restate_order_embedding.weight" not in _ATOM_VOCAB_HEAD_LAYOUT
    src, fresh, init, *_ = _run()
    # it is 4-wide in both source and target here (bond order), so it is COPIED_EXACT, never row-mapped;
    # crucially it is NOT semantically remapped against the atom vocabulary.
    assert torch.equal(init["restate_order_embedding.weight"], src["restate_order_embedding.weight"])


def test_atom_embedding_copied_exact_not_in_semantic_table():
    src, _fresh, init, _copied, _partial, table, _h = _run()
    assert torch.equal(init["atom_embedding.weight"], src["atom_embedding.weight"])  # shape-exact copy
    assert not any(row["head"] == "atom_embedding.weight" for row in table)  # never semantic-mapped


def test_row_table_status_and_hash_deterministic():
    _src, _fresh, _init, _copied, _partial, table, map_hash = _run()
    restate = [r for r in table if r["head"] == "restate_head.weight"]
    shared = [r for r in restate if r["status"] == "COPIED_WITH_INDEX_MAP"]
    fresh = [r for r in restate if r["status"] == "FRESH_NEW_CLASS"]
    assert len(shared) == _NSRC and len(fresh) == _NDEST - _NSRC
    for r in shared:
        assert r["source_checksum"] == r["dest_checksum"]  # bit-exact copy proof
    # deterministic map hash
    _s2, _f2, _i2, _c2, _p2, _t2, h2 = _run()
    assert map_hash == h2
