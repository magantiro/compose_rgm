"""Broad-organic vocabulary: chemistry-specific legality, replay, sampling, and CHARGE PRESERVATION.

The locked B-edit scope admits S/P/Cl/Br/I and representable charged states. These tests prove the model +
executor actually handle that chemistry: every class encodes/decodes, is in scope, and yields valid
executor successors under the sampler; and -- the load-bearing contract -- edits on a charged molecule
PRESERVE the charged center (net formal charge is invariant across every applied edit and every corruption
state). We never require the model to modify a protected charged atom; we require it to edit elsewhere while
the charge label survives. Sulfur and phosphorus get extra attention (valence patterns beyond terminal
halogens).
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, classify_smiles
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import make_edit_pair
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 40
_SYS = de_novo_rewrite_system()

# representative broad-vocab chemistry (drug-like, in scope)
_NEUTRAL = {
    "thioether": "CSc1ccccc1",
    "sulfoxide": "CS(=O)c1ccccc1",
    "sulfone": "CS(=O)(=O)c1ccccc1",
    "sulfonamide": "O=S(=O)(N)c1ccccc1",
    "phosphonate": "CCOP(=O)(OCC)C",
    "aryl_chloride": "Clc1ccccc1",
    "alkyl_chloride": "ClCCc1ccccc1",
    "aryl_bromide": "Brc1ccccc1",
    "iodobenzene": "Ic1ccccc1",
}
_CHARGED = {
    "charged_amine": "C[NH3+]",
    "carboxylate": "CC(=O)[O-]",
    "zwitterion": "C(C(=O)[O-])[NH3+]",
    "trimethylammonium": "C[N+](C)(C)C",
}


def _net_charge(state) -> int:
    # states are SLOT-STABLE: a deletion leaves a null slot mid-array, so real atoms are NOT the first
    # n_real_atoms. Sum over the is_element mask, never a [:n_real_atoms] slice (which drops trailing atoms).
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _b_edit():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_SLOTS)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)

    catalog = build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "c1ccncc1", "c1ccsc1")))
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_restates=True,
        enable_cyclic_graft=True, enable_heteroatom_scan=True, enable_ring_opening=True,
        atom_vocabulary=ORGANIC_VOCABULARY).eval()
    return AnalyticPancakeQuotientSampler(model, calibration=PancakeQuotientCalibration())


def test_broad_vocab_encode_decode_and_scope() -> None:
    for name, smi in {**_NEUTRAL, **_CHARGED}.items():
        graph = smiles_to_molecular_graph(smi)
        assert graph is not None, name
        # decode must recover a molecule (RDKit-parseable canonical SMILES)
        decoded = molecular_graph_to_smiles(graph)
        assert decoded, f"{name} did not decode"
        assert classify_smiles(smi, BROAD_ORGANIC_V1)[0], f"{name} not in broad scope"


def test_sampler_yields_valid_successors_on_broad_chemistry() -> None:
    sampler = _b_edit()
    n = valid = 0
    for name, smi in {**_NEUTRAL, **_CHARGED}.items():
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        for step in range(3):
            mark = sampler.sample_rewrite_mark(state, frozen_time(0.2 * (step + 1)),
                                               np.random.default_rng(step))
            if mark.action is None:
                break
            succ = _SYS.apply(state, mark.rule_name, mark.action)  # executor must accept the sampled mark
            n += 1
            ok = is_valid_state(succ) and is_connected_or_null(succ)
            valid += int(ok)
            assert ok, f"{name}: invalid successor via {mark.rule_name}"
            state = succ
    assert n > 0 and valid == n


def test_charge_preserving_rollout_keeps_charged_leads_valid() -> None:
    # The inference-side charge guarantee is a POLICY: a charged-source rollout rejects any proposed mark
    # that shifts the net formal charge (protecting the charged motif) and keeps a charge-preserving one.
    # This is the mechanism the rollout eval + preflight use; here we verify it yields valid, charge-stable
    # edits and that charged leads are editable at all (an untrained fixture is not charge-perfect on its
    # own -- the training corruption teaches preservation; this policy makes it hard at inference).
    sampler = _b_edit()
    preserved = rejected = 0
    for smi in _CHARGED.values():
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        q0 = _net_charge(state)
        for step in range(6):
            mark = sampler.sample_rewrite_mark(state, frozen_time(0.15 * (step + 1)),
                                               np.random.default_rng(step + 1))
            if mark.action is None:
                break
            succ = _SYS.apply(state, mark.rule_name, mark.action)
            if _net_charge(succ) != q0:
                rejected += 1
                continue  # charge-preserving policy: this mark is illegal on a charged source
            assert is_valid_state(succ) and is_connected_or_null(succ)
            preserved += 1
            state = succ
    assert preserved > 0, "no charge-preserving edit was available on any charged lead"


def test_corruption_preserves_net_charge_on_charged_leads() -> None:
    rng = np.random.default_rng(3)
    seen = 0
    for smi in _CHARGED.values():
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        q0 = _net_charge(target)
        for depth in (1, 2, 3):
            trim, grow = make_edit_pair(target, depth, system=_SYS, rng=rng,
                                        vocabulary=ORGANIC_VOCABULARY)
            for trace in (trim, grow):
                if trace is None:
                    continue
                seen += 1
                # the corrupted source prior protects charged atoms -> every endpoint keeps the net charge.
                assert _net_charge(trace.source) == q0
                assert _net_charge(trace.target) == q0
    assert seen > 0, "expected at least one charged-lead corruption trace"


def test_sulfur_phosphorus_rings_round_trip_through_catalog() -> None:
    # S/P valence is richer than halogens; ensure heteroaromatic + saturated S rings are representable.
    for smi in ("c1ccsc1", "c1ccoc1", "C1CCSCC1", "c1ccncc1"):
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        key = canonical_state_key(state)
        assert key  # canonical key computable
        assert classify_smiles(smi, BROAD_ORGANIC_V1)[0]
