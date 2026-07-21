"""Region-aware lipid rate model: additive, off-by-default, region-specific.

Verifies (a) region labels align with the batch's graph atom order, (b) a
zero-initialized region model with no prior table is bit-for-bit the base model
(the extension is purely additive / safe to merge), and (c) a region-conditioned
prior shifts the connected-insertion logits by exactly the per-atom region table.
"""

from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from compose_v4.lipids.region_labels import HEAD, TAIL
from compose_v4.lipids.region_aware_rate_model import (
    RegionAwareFactorizedTraceletRateModel,
    region_ids_for_state,
)

LIPID = "CCCCCCN(C)C"  # hexyl-dimethylamine: amine head + hexyl tail


def _catalog():
    records = build_tracelet_path_records((LIPID,), n_slots=12, typed_ring_payloads=True)
    return build_typed_ring_catalog((r.path.trace for r in records),
                                    max_cycle_templates=32, max_attach_templates=32,
                                    max_ear_templates=32)


def _state_and_batch(catalog):
    state = pad_molecular_graph(smiles_to_molecular_graph(LIPID), 12)
    batch = prepare_factorized_mark_batch(
        (state,), (0.5,), (None,), (None,), (0.0,), ring_catalog=catalog)
    return state, batch


def _logits(model, batch):
    model.eval()
    with torch.no_grad():
        node, glob, pair = model._encode_batch(batch)
        _, logits, _ = model._action_tables(batch, node, glob, pair)
    return node, logits


def test_region_ids_align_with_graph_order() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph(LIPID), 12)
    ids = region_ids_for_state(state)
    n_atoms = [i for i in range(state.n_atoms) if int(state.atom_types[i]) == ELEMENT_TO_IDX["N"]]
    assert n_atoms and all(ids[i] == HEAD for i in n_atoms)   # amine N is head
    assert (ids == TAIL).sum() >= 3                            # the hexyl chain is tail
    assert ids[0] == TAIL                                      # terminal chain carbon


def test_zero_region_model_equals_base() -> None:
    catalog = _catalog()
    _, batch = _state_and_batch(catalog)
    base = FactorizedTraceletRateModel(catalog, hidden_dim=8, message_passing_steps=1)
    region = RegionAwareFactorizedTraceletRateModel(catalog, hidden_dim=8, message_passing_steps=1)
    region.load_state_dict(base.state_dict(), strict=False)  # share base weights; region params stay at init
    base_node, base_logits = _logits(base, batch)
    region_node, region_logits = _logits(region, batch)
    # zero-init embedding + zero prior => identical encoder + logits
    assert torch.allclose(base_node, region_node, atol=1e-6)
    for key in ("grow_connected", "grow_root", "atom_restate", "bond_reorder"):
        assert torch.allclose(base_logits[key], region_logits[key], atol=1e-6), key


def test_region_prior_shifts_connected_logits_by_region() -> None:
    catalog = _catalog()
    state, batch = _state_and_batch(catalog)
    # distinct, non-uniform per-region (order x element) table, normalized per region
    table = np.arange(1, 4 * 3 * 4 + 1, dtype=np.float32).reshape(4, 3, 4)
    table /= table.sum(axis=(1, 2), keepdims=True)
    base = FactorizedTraceletRateModel(catalog, hidden_dim=8, message_passing_steps=1)
    region = RegionAwareFactorizedTraceletRateModel(
        catalog, hidden_dim=8, message_passing_steps=1, region_prior_table=table)
    region.load_state_dict(base.state_dict(), strict=False)
    _, base_logits = _logits(base, batch)
    _, region_logits = _logits(region, batch)
    diff = region_logits["grow_connected"] - base_logits["grow_connected"]  # [1, n_slots, 3, 4]
    region_ids = torch.from_numpy(region_ids_for_state(state)).long()
    expected = region.region_conditioned_prior[region_ids].unsqueeze(0)
    assert torch.allclose(diff, expected, atol=1e-5)
    assert diff.abs().max() > 0  # genuinely region-specific, nonzero
