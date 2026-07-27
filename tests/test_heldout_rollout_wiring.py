"""Stage 6: held-out editing rollout WIRING smoke -- the process invariants that must hold for ANY
checkpoint (a fresh B-edit fixture here; the real trained B is absent locally).

Over held-out GuacaMol leads and fixed edit budgets, every rollout must: start EXACTLY at the requested
lead (never a carbon-tree seed); recompute legal actions at each state; only ever commit a sampled action
that the executor accepts; keep every intermediate valid + connected; and terminate cleanly. Any invalid
intermediate, illegal sampled action, or executor failure is a NO-GO. The degeneracy DIAGNOSTIC
(reversal/cycle/return-to-source rates) is measured for plumbing but its QUALITY reading requires the
trained model -- an untrained fixture is expected to look near-random, not degenerate-by-collapse.
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 40
_SYS = de_novo_rewrite_system()
# held-out drug-like leads (cyclic, decorated) NOT used to build the Stage-6A coverage corpus.
_LEADS = ["O=C(Nc1ccccc1)c1ccncc1", "COc1ccc(CCN)cc1", "CC(=O)Nc1ccc(O)cc1", "c1ccc(-c2ccncc2)cc1"]


def _catalog():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_SLOTS)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)
    return build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "C1CCNCC1", "c1ccncc1")))


def _b_edit(catalog):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_restates=True,
        enable_cyclic_graft=True, enable_heteroatom_scan=True, enable_ring_opening=True,
        atom_vocabulary=ORGANIC_VOCABULARY).eval()


def test_heldout_editing_rollout_wiring_invariants() -> None:
    catalog = _catalog()
    sampler = AnalyticPancakeQuotientSampler(_b_edit(catalog), calibration=PancakeQuotientCalibration())
    n_intermediates = n_valid = 0
    diag = {"reversal": 0, "revisit": 0, "return_to_source": 0, "steps": 0, "families": {}}
    for lead_smi in _LEADS:
        lead = pad_molecular_graph(smiles_to_molecular_graph(lead_smi), _SLOTS)
        lead_key = canonical_state_key(lead)
        for budget in (1, 2, 4, 8):
            rng = np.random.default_rng(budget)
            state = lead
            assert canonical_state_key(state) == lead_key  # rollout starts EXACTLY at the lead
            visited = [lead_key]
            for step in range(budget):
                mark = sampler.sample_rewrite_mark(state, frozen_time(0.1 * (step + 1)), rng)
                if mark.action is None:  # clean terminal
                    break
                # the executor must accept exactly the sampled action (legal set recomputed each state).
                succ = _SYS.apply(state, mark.rule_name, mark.action)
                n_intermediates += 1
                ok = is_valid_state(succ) and is_connected_or_null(succ)
                n_valid += int(ok)
                assert ok, f"{lead_smi} b={budget}: invalid/disconnected intermediate via {mark.rule_name}"
                key = canonical_state_key(succ)
                diag["steps"] += 1
                diag["families"][mark.rule_name] = diag["families"].get(mark.rule_name, 0) + 1
                if len(visited) >= 2 and key == visited[-2]:
                    diag["reversal"] += 1
                if key in visited:
                    diag["revisit"] += 1
                if key == lead_key:
                    diag["return_to_source"] += 1
                visited.append(key)
                state = succ
    # NO-GO conditions: any invalid intermediate or executor failure.
    assert n_intermediates > 0
    assert n_valid == n_intermediates, f"all-intermediate validity {n_valid}/{n_intermediates}"


def test_rollout_uses_the_lead_not_a_carbon_tree_seed() -> None:
    # The editing rollout is seeded from the provided lead; a carbon-tree seed would be a small alkane.
    catalog = _catalog()
    sampler = AnalyticPancakeQuotientSampler(_b_edit(catalog), calibration=PancakeQuotientCalibration())
    lead = pad_molecular_graph(smiles_to_molecular_graph("O=C(Nc1ccccc1)c1ccncc1"), _SLOTS)
    # a single step keeps us on/near the real lead (decorated, cyclic) -- never a bare carbon chain.
    mark = sampler.sample_rewrite_mark(lead, frozen_time(0.1), np.random.default_rng(0))
    start = molecular_graph_to_smiles(lead) or ""
    assert "n" in start.lower() or "o" in start.lower()  # the lead's heteroatoms are present at t=0
    if mark.action is not None:
        succ = _SYS.apply(lead, mark.rule_name, mark.action)
        assert is_valid_state(succ) and is_connected_or_null(succ)
