"""max_atoms=40 is a formal state-space bound: |V(x_k)| <= 40 for EVERY committed state in EVERY accepted
trajectory, not just the source + target.

The bound is enforced at ENUMERATION, not by executor rejection after sampling: the legal-action fiber only
offers atom_insert / grow into FREE padded slots, and a state padded to n_slots=40 that already holds 40
real atoms has none. These tests pin that across single inserts, multi-atom ring grows, the production
sampler, corruption, and compiled MMP traces (both directions).
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import make_edit_pair
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_CAP = 40
_SYS = de_novo_rewrite_system()


def _n_atoms(state) -> int:
    return int(is_element(state.atom_types).sum())


def _sampler():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), _CAP)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_CAP)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)

    catalog = build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "c1ccncc1", "c1ccsc1")))
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_restates=True,
        enable_cyclic_graft=True, enable_heteroatom_scan=True, enable_ring_opening=True,
        atom_vocabulary=ORGANIC_VOCABULARY).eval()
    return AnalyticPancakeQuotientSampler(model, calibration=PancakeQuotientCalibration())


def test_full_40_atom_state_exposes_no_size_increasing_edit() -> None:
    # a 40-carbon chain padded to exactly 40 slots has ZERO free slots -> the enumerator cannot offer an
    # atom_insert or a ring grow; sampling many times never produces a successor with > 40 atoms.
    state = pad_molecular_graph(smiles_to_molecular_graph("C" * _CAP), _CAP)
    assert _n_atoms(state) == _CAP and (state.atom_types == 0).sum() == 0  # full: no null slots
    sampler = _sampler()
    for step in range(12):
        mark = sampler.sample_rewrite_mark(state, frozen_time(0.2), np.random.default_rng(step))
        if mark.action is None:
            continue
        succ = _SYS.apply(state, mark.rule_name, mark.action)
        assert _n_atoms(succ) <= _CAP, f"{mark.rule_name} grew a full state to {_n_atoms(succ)}"


def test_39_atom_state_edits_stay_within_cap() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("C" * (_CAP - 1)), _CAP)
    assert _n_atoms(state) == _CAP - 1  # one free slot
    sampler = _sampler()
    seen = 0
    for lead_step in range(20):
        s = state
        for step in range(4):
            mark = sampler.sample_rewrite_mark(s, frozen_time(0.15 * (step + 1)),
                                               np.random.default_rng(lead_step * 10 + step))
            if mark.action is None:
                break
            succ = _SYS.apply(s, mark.rule_name, mark.action)
            seen += 1
            assert _n_atoms(succ) <= _CAP, f"{mark.rule_name}: {_n_atoms(s)} -> {_n_atoms(succ)} > {_CAP}"
            assert is_valid_state(succ) and is_connected_or_null(succ)
            s = succ
    assert seen > 0


def test_corruption_never_records_an_over_cap_state() -> None:
    # corruption walks real drug-like leads a few edits; EVERY recorded state must stay within the cap.
    rng = np.random.default_rng(4)
    leads = ["O=C(Nc1ccccc1)c1ccncc1", "CSc1ccc(CCNC(=O)c2ccco2)cc1", "Clc1ccc(-c2ccc(CCN)cc2)cc1"]
    seen = 0
    for smi in leads:
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), _CAP)
        for depth in (1, 2, 3, 4):
            trim, grow = make_edit_pair(target, depth, system=_SYS, rng=rng,
                                        vocabulary=ORGANIC_VOCABULARY)
            for trace in (trim, grow):
                if trace is None:
                    continue
                seen += 1
                state = trace.source
                assert _n_atoms(state) <= _CAP
                for step in trace.steps:
                    state = _SYS.apply(state, step.rule_name, step.action)
                    assert _n_atoms(state) <= _CAP, f"corruption intermediate {_n_atoms(state)} > {_CAP}"
    assert seen > 0


def test_reversed_corruption_trace_independently_within_cap() -> None:
    # a valid trim (A->corrupted) does not guarantee its grow inverse stays in cap; check the grow direction
    # from its OWN source, every intermediate.
    rng = np.random.default_rng(5)
    target = pad_molecular_graph(smiles_to_molecular_graph("CSc1ccc(CCNC(=O)c2ccco2)cc1"), _CAP)
    checked = 0
    for depth in (1, 2, 3):
        _, grow = make_edit_pair(target, depth, system=_SYS, rng=rng, vocabulary=ORGANIC_VOCABULARY)
        if grow is None:
            continue
        state = grow.source
        assert _n_atoms(state) <= _CAP
        for step in grow.steps:
            state = _SYS.apply(state, step.rule_name, step.action)
            assert _n_atoms(state) <= _CAP
        checked += 1
    assert checked > 0
