"""Regression guard: AnalyticPancakeQuotientSampler must enumerate the checkpoint's editing vocabulary.

The pancake sampler (which every lead-editing controller uses) builds its legal-mark batch from the
base model's operator-capability flags.  A prior defect built the batch WITHOUT
``compute_ring_restates``/``compute_cyclic_graft``/``compute_ring_opening`` (they silently defaulted
``False``), so a B-edit checkpoint loaded its wide organic heads but sampled the DE-NOVO vocabulary --
de-aromatization / clean ring-opening were unreachable at inference.

These tests exercise the pancake sampling path (not the raw ``model.sample_rewrite_mark``), so removing
the propagated flags in ``canonical_successor_distillation`` makes them fail (the required mutation
test).  Clean ring-opening is catalog-template matched, so the model needs the carbon-tree-derived ring
templates (how B learns rings); ring restate needs no catalog.
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 16
_SYSTEM = de_novo_rewrite_system()


def _carbon_tree_trace(smiles: str):
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(37), n_slots=_SLOTS
    )
    return compile_carbon_tree_to_target(source, target, use_bond_reroute=True, align_source=True)


def _catalog():
    return build_typed_ring_catalog(
        tuple(_carbon_tree_trace(s) for s in ("c1ccccc1", "C1CCCCC1", "C1CCNCC1"))
    )


def _model(catalog, **flags):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, **flags
    ).eval()


def _sampler(model):
    return AnalyticPancakeQuotientSampler(model, calibration=PancakeQuotientCalibration())


def _state(smi):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)


def _family_rate(model, state, family, t=0.5):
    tbl = _sampler(model).rate_table(state, t)
    return float(tbl.productive_family_rates[MARK_RULE_TO_INDEX[family]])


def _sample_family_successors(model, state, family, *, n=300, t=0.5):
    sampler = _sampler(model)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        mark = sampler.sample_rewrite_mark(state, t, rng)
        if mark.action is not None and mark.rule_name == family:
            out.append(_SYSTEM.apply(state, mark.rule_name, mark.action))
    return out


def _is_aromatic(state):
    smi = molecular_graph_to_smiles(state)
    return smi is not None and any(c.islower() for c in smi if c.isalpha())


def test_pancake_sampler_enumerates_de_aromatization() -> None:
    # ring_system_restate is a family on/off capability: ON -> positive rate, OFF -> zero (mutation).
    catalog = _catalog()
    state = _state("Cc1ccccc1")
    fam = "ring_system_restate"
    assert _family_rate(_model(catalog, enable_ring_restates=True), state, fam) > 0.0
    assert _family_rate(_model(catalog, enable_ring_restates=False), state, fam) == 0.0

    succ = _sample_family_successors(_model(catalog, enable_ring_restates=True), state, fam)
    assert succ, "de-aromatization never sampled though rate > 0"
    assert all(is_valid_state(y) and is_connected_or_null(y) for y in succ)
    # de-aromatization changes perceived aromaticity for at least one successor.
    assert any(_is_aromatic(y) != _is_aromatic(state) for y in succ)


def test_pancake_sampler_de_aromatization_inverse_round_trip() -> None:
    catalog = _catalog()
    state = _state("Cc1ccccc1")
    model = _model(catalog, enable_ring_restates=True)
    succ = _sample_family_successors(model, state, "ring_system_restate")
    assert succ
    back = _sample_family_successors(model, succ[0], "ring_system_restate", n=600)
    assert any(canonical_state_key(b) == canonical_state_key(state) for b in back)


def test_pancake_sampler_clean_ring_opening_preserves_heteroatom() -> None:
    # enable_ring_opening SWAPS structured(carbon-izing)->clean(decoration-preserving): the family
    # fires either way, but ON keeps the ring N, OFF (mutation) carbon-izes it.
    catalog = _catalog()
    state = _state("C1CCNCC1")  # piperidine
    fam = "ring_system_delete"
    clean, struct = _model(catalog, enable_ring_opening=True), _model(catalog, enable_ring_opening=False)
    assert _family_rate(clean, state, fam) > 0.0
    assert _family_rate(struct, state, fam) > 0.0

    clean_succ = _sample_family_successors(clean, state, fam)
    struct_succ = _sample_family_successors(struct, state, fam)
    assert clean_succ and struct_succ
    assert all(is_valid_state(y) and is_connected_or_null(y) for y in clean_succ + struct_succ)
    clean_keeps_n = any("N" in (molecular_graph_to_smiles(y) or "").upper() for y in clean_succ)
    struct_keeps_n = any("N" in (molecular_graph_to_smiles(y) or "").upper() for y in struct_succ)
    assert clean_keeps_n, "clean ring-opening lost the ring N (capability not propagated)"
    assert not struct_keeps_n, "structured delete unexpectedly preserved N (mutation ineffective)"


def test_pancake_sampler_heteroatom_scan_reaches_ring_atom_bioisosterism() -> None:
    # enable_heteroatom_scan ungates ring atoms in the atom_restate dense mask, so ring-atom element
    # swaps (pyridine ring N -> C) become sampleable; without it only peripheral atoms restate.
    catalog = _catalog()
    state = _state("Cc1ccncc1")  # methylpyridine: one ring N

    def ring_n_eliminating_restates(flag):
        torch.manual_seed(0)
        model = FactorizedTraceletRateModel(
            catalog, hidden_dim=16, message_passing_steps=1,
            enable_heteroatom_scan=flag, atom_vocabulary=ORGANIC_VOCABULARY,
        ).eval()
        succ = _sample_family_successors(model, state, "atom_restate", n=400)
        assert all(is_valid_state(y) and is_connected_or_null(y) for y in succ)
        return sum(1 for y in succ if "N" not in (molecular_graph_to_smiles(y) or "").upper())

    assert ring_n_eliminating_restates(True) > 0, "heteroatom scan on: ring N->C restate must sample"
    assert ring_n_eliminating_restates(False) == 0, "heteroatom scan off: ring atoms must not restate"
