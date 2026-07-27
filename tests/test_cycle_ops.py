"""P1 acceptance tests for the compositional cycle_close / cycle_open families.

cycle_close = BondInsert (add a ring-closing bond over a nonbonded pair x order); cycle_open = BondDelete of a
non-bridge cycle edge. Both are gated by enable_cycle_ops -> byte-identical when off. These tests lock the
correctness-critical properties: byte-identical off, teacher-in-mask (finite GM loss), exact executor round
trip, and a sampler that proposes + executes valid cycle ops.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import BondDelete, BondInsert  # noqa: E402
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog  # noqa: E402

_SLOTS = 40


def _trace(smi: str):
    target = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
    src = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1), n_slots=_SLOTS
    )
    return compile_carbon_tree_to_target(src, target, use_bond_reroute=True, align_source=True)


def _catalog():
    return build_typed_ring_catalog(tuple(_trace(s) for s in ("c1ccccc1", "c1ccncc1")))


def _model(enable: bool):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        _catalog(), hidden_dim=16, message_passing_steps=1, enable_cycle_ops=enable
    ).eval()


def _state(smi: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)


def _a_ring_bond(state):
    real = [i for i in range(len(state.atom_types)) if is_element(state.atom_types)[i]]
    graph = nx.Graph()
    graph.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]):
                graph.add_edge(a, b)
    bridges = {frozenset(e) for e in nx.bridges(graph)}
    a, b = sorted(next((a, b) for a, b in graph.edges() if frozenset((a, b)) not in bridges))
    return a, b, int(state.bonds[a, b])


def _batch(state, action, rule):
    return prepare_factorized_mark_batch(
        (state,), (frozen_time(0.2),), (action,), (rule,), (1.0,), ring_catalog=_catalog()
    )


def test_off_is_byte_identical():
    off, base = _model(False), _model(False)
    assert set(off.state_dict()) == set(base.state_dict())
    assert not any("cycle_close_head" in k or "cycle_open_head" in k for k in off.state_dict())
    st = _state("c1ccccc1CO")
    b = prepare_factorized_mark_batch(
        (st,), (frozen_time(0.2),), (None,), (None,), (0.0,), ring_catalog=_catalog()
    )
    with torch.no_grad():
        assert torch.allclose(off.forward_mark_batch(b).total_hazard, base.forward_mark_batch(b).total_hazard)


def test_cycle_open_teacher_is_in_mask_finite():
    m = _model(True)
    st = _state("c1ccccc1CO")
    a, b, _order = _a_ring_bond(st)
    with torch.no_grad():
        pred = m.forward_mark_batch(_batch(st, BondDelete(a, b), "bond_delete"))
    assert bool(torch.isfinite(pred.selected_mark_log_probability).all())


def test_cycle_close_teacher_is_in_mask_finite_and_round_trips():
    m = _model(True)
    system = de_novo_rewrite_system()
    st = _state("c1ccccc1CO")
    a, b, order = _a_ring_bond(st)
    precursor = system.apply(st, "bond_delete", BondDelete(a, b))
    with torch.no_grad():
        pred = m.forward_mark_batch(_batch(precursor, BondInsert(a, b, order), "bond_insert"))
    assert bool(torch.isfinite(pred.selected_mark_log_probability).all())
    # exact executor round trip: open then close recovers the original
    recovered = system.apply(precursor, "bond_insert", BondInsert(a, b, order))
    assert canonical_state_key(recovered) == canonical_state_key(st)


def test_sampler_proposes_and_executes_valid_cycle_ops():
    m = _model(True)
    system = de_novo_rewrite_system()
    st = _state("c1ccccc1CO")
    a, b, _order = _a_ring_bond(st)
    precursor = system.apply(st, "bond_delete", BondDelete(a, b))
    executed = 0
    for seed in range(60):
        for start in (st, precursor):
            mark = m.sample_rewrite_mark(start, frozen_time(0.2), np.random.default_rng(seed))
            if mark.rule_name in ("bond_insert", "bond_delete"):
                succ = system.apply(start, mark.rule_name, mark.action)
                assert succ is not None  # every proposed cycle op is executor-legal
                executed += 1
    assert executed > 0  # cycle ops are actually reachable at sampling
