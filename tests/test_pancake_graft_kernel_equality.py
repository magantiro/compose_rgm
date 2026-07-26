"""A7: the production pancake sampler must induce the SAME canonical molecular-successor kernel as the
GM-trained mark distribution -- which the raw ``model.sample_rewrite_mark`` sampler reproduces (verified
independently against the loss).

Full-Zq fidelity: the pancake graft family AND each successor group are normalized over the QUOTIENT
partition Zq (one convention for tree AND cyclic graft), so
    P_theta(x, y) = sum_{a : T(x,a)=y} p_theta(a | x)
matches the raw sampler at the canonical-successor level.  The historical Zr + survival convention
(``legacy_raw_graft_survival=True``) under-weights graft on self-graft states and is a NON-default
ablation only.

We prove kernel equality two ways: (1) EXACT -- the pancake's family distribution equals the raw
sampler's family distribution (same softmax over the same masked family logits) to machine precision,
across tree/symmetric-tree/cyclic/acyclic/post-ring-opening states, with the atom-delete calibration
zeroed so the graft convention is the only free variable; and (2) Monte-Carlo -- the raw and pancake
samplers' committed-successor frequencies agree within a bootstrap null band.  The completion criterion
is kernel equality, not merely "no NaNs".
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    _masked_family_logits,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.factorized_fiber import enumerate_pendant_graft_actions
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 16
_SYSTEM = de_novo_rewrite_system()
# atom-delete calibration ZEROED: graft convention is then the only thing that can differ raw<->pancake.
_ZQ = PancakeQuotientCalibration(atom_delete_log_rate_adjustment=0.0)
_LEGACY = PancakeQuotientCalibration(atom_delete_log_rate_adjustment=0.0, legacy_raw_graft_survival=True)

# tree unique-encoding, symmetric tree (neopentane), cyclic (graft+non-graft), acyclic, post-ring-opening.
_CASES = [
    ("butane_tree", "CCCC", {}),
    ("neopentane_symmetric_tree", "CC(C)(C)C", {}),
    ("octane_self_grafts", "CCCCCCCC", {}),
    ("propylbenzene_cyclic", "c1ccccc1CCC", {"enable_cyclic_graft": True}),
    ("propanal_acyclic", "CCC=O", {}),
    ("post_ring_opening_chain", "CCCCNC", {"enable_cyclic_graft": True}),
]


def _carbon_tree_trace(smiles):
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


def _state(smi):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)


def _raw_family_dist(model, state):
    """The raw sampler's family distribution = softmax over the masked family logits (model:3160),
    built from the exact batch the raw sampler builds."""
    batch = prepare_factorized_mark_batch(
        (state,), (0.5,), (None,), (None,), (0.0,),
        ring_catalog=model.ring_catalog,
        compute_ring_restates=model.enable_ring_restates,
        compute_cyclic_graft=model.enable_cyclic_graft,
        compute_ring_opening=model.enable_ring_opening,
    ).to(model.device)
    with torch.no_grad():
        node, glob, pair = model._encode_batch(batch)
        _, _, action_log_z = model._action_tables(batch, node, glob, pair, require_exact_ring_support=False)
        enabled = torch.isfinite(action_log_z[0])
        family_logits = _masked_family_logits(
            model._family_base_logits(batch, glob)[0], action_log_z[0], enabled,
            rate_factorization=model.rate_factorization,
        )
        return torch.softmax(family_logits, dim=-1).detach().cpu().numpy()


def _pancake_family_dist(model, state, calibration):
    tbl = AnalyticPancakeQuotientSampler(model, calibration=calibration).rate_table(state, 0.5)
    return (tbl.productive_family_rates / tbl.productive_total_hazard).detach().cpu().numpy()


def _successor_counts(sampler, state, *, n, seed):
    rng = np.random.default_rng(seed)
    counts: Counter = Counter()
    for _ in range(n):
        mark = sampler.sample_rewrite_mark(state, 0.5, rng)
        if mark.action is None:
            counts["<TERMINAL>"] += 1
        else:
            counts[canonical_state_key(_SYSTEM.apply(state, mark.rule_name, mark.action))] += 1
    return counts


def _tv(a: Counter, b: Counter, n) -> float:
    return 0.5 * sum(abs(a.get(k, 0) / n - b.get(k, 0) / n) for k in set(a) | set(b))


def test_pancake_family_kernel_equals_raw_sampler_exactly() -> None:
    # EXACT: the full-Zq pancake induces the raw sampler's family distribution to machine precision,
    # across every state class. (Within a family both samplers use the identical _sample_action_from_
    # family coordinate law and, for graft, the identical group-mass/Zq successor law, so family-dist
    # equality => full canonical-successor-kernel equality.)
    catalog = _catalog()
    for name, smi, flags in _CASES:
        model = _model(catalog, **flags)
        state = _state(smi)
        diff = float(np.max(np.abs(_raw_family_dist(model, state) - _pancake_family_dist(model, state, _ZQ))))
        assert diff < 1e-5, f"{name}: raw vs pancake family kernel differ by {diff:.2e}"


def test_pancake_graft_group_mass_equals_sum_of_raw_mark_masses() -> None:
    # survival == 1 is only correct if the post-quotient graft family mass equals the sum of the trained
    # raw-mark (successor-group) masses. Verify graft family rate == sum of successor-group rates, and
    # each group rate > 0, finite, on tree/cyclic states.
    catalog = _catalog()
    gi = MARK_RULE_TO_INDEX["bond_reroute"]
    for smi, flags in [("CCCCCCCC", {}), ("CC(C)(C)C", {}), ("c1ccccc1CCC", {"enable_cyclic_graft": True})]:
        model = _model(catalog, **flags)
        tbl = AnalyticPancakeQuotientSampler(model, calibration=_ZQ).rate_table(_state(smi), 0.5)
        gs = tbl.graft_successor_rates
        assert bool(torch.isfinite(gs).all()) and (gs >= 0).all()
        assert np.isclose(float(tbl.productive_family_rates[gi]), float(gs.sum()), rtol=1e-6)


def test_full_zq_matches_raw_where_legacy_diverges() -> None:
    # On a self-graft state (octane: 42 raw graft slots, 30 quotient) full-Zq matches the raw family
    # kernel exactly while the legacy Zr+survival convention under-weights graft and DIVERGES -- proving
    # the fix is correct AND that the family-kernel test genuinely discriminates a wrong convention.
    catalog = _catalog()
    model = _model(catalog)
    state = _state("CCCCCCCC")
    raw = _raw_family_dist(model, state)
    diff_zq = float(np.max(np.abs(raw - _pancake_family_dist(model, state, _ZQ))))
    diff_zr = float(np.max(np.abs(raw - _pancake_family_dist(model, state, _LEGACY))))
    assert diff_zq < 1e-5, f"full-Zq must match the raw family kernel (diff={diff_zq:.2e})"
    assert diff_zr > 1e-2, f"legacy Zr must diverge from the raw family kernel (diff={diff_zr:.2e})"


def test_raw_and_pancake_successor_frequencies_agree_within_null_band() -> None:
    # Monte-Carlo end-to-end: the raw and full-Zq pancake samplers' committed-successor frequencies
    # agree within a bootstrap null band (TV between two independent RAW samples), which auto-calibrates
    # to each state's successor-category count -- a fixed TV threshold is wrong for high-category states.
    catalog = _catalog()
    n = 3500  # the null band scales with n, so a modest n keeps the test fast AND robust.
    for smi, flags in [("CCCCCCCC", {}), ("c1ccccc1CCC", {"enable_cyclic_graft": True})]:
        model = _model(catalog, **flags)
        state = _state(smi)
        raw_a = _successor_counts(model, state, n=n, seed=1)
        raw_b = _successor_counts(model, state, n=n, seed=101)  # independent raw sample -> null band
        pancake = _successor_counts(
            AnalyticPancakeQuotientSampler(model, calibration=_ZQ), state, n=n, seed=2)
        null_tv = _tv(raw_a, raw_b, n)
        test_tv = _tv(raw_a, pancake, n)
        assert test_tv < 2.5 * null_tv + 0.01, (
            f"{smi}: raw-vs-pancake TV={test_tv:.4f} exceeds null band (raw-vs-raw TV={null_tv:.4f})")


def test_pancake_rate_table_normalized_and_finite() -> None:
    catalog = _catalog()
    model = _model(catalog, enable_cyclic_graft=True)
    gi = MARK_RULE_TO_INDEX["bond_reroute"]
    for smi in ("CC(C)(C)C", "c1ccccc1CCC", "CCC=O"):
        tbl = AnalyticPancakeQuotientSampler(model, calibration=_ZQ).rate_table(_state(smi), 0.5)
        pf = tbl.productive_family_rates
        assert bool(torch.isfinite(pf).all()) and float(pf.sum()) > 0.0
        assert float(tbl.productive_total_hazard) > 0.0 and np.isfinite(float(tbl.productive_total_hazard))
        assert np.isclose(float(pf[gi]), float(tbl.graft_successor_rates.sum()), rtol=1e-5)  # survival==1


def test_graft_group_representatives_share_one_canonical_successor() -> None:
    # Which canonical representative the sampler commits within a successor group cannot change the
    # future represented state / subsequent legal-action enumeration: every raw graft encoding in a
    # group executes to the SAME canonical successor and re-enumerates to the same canonical fiber.
    # (Property of the executor + shared enumerator; model-independent.)
    state = _state("c1ccccc1CCC")
    grafts = enumerate_pendant_graft_actions(state)
    assert grafts, "expected cyclic-graft encodings on propylbenzene"
    by_key: dict = {}
    for action in grafts:
        succ = _SYSTEM.apply(state, "bond_reroute", action)
        by_key.setdefault(canonical_state_key(succ), []).append(succ)
    alias_groups = [v for v in by_key.values() if len(v) > 1]
    assert alias_groups, "expected at least one multi-encoding graft successor group"
    for members in alias_groups:
        assert len({canonical_state_key(s) for s in members}) == 1
        fibers = {
            tuple(sorted(canonical_state_key(_SYSTEM.apply(s, "bond_reroute", a))
                         for a in enumerate_pendant_graft_actions(s)))
            for s in members
        }
        assert len(fibers) == 1, "re-enumeration differs across group representatives"


def test_legacy_ablation_available_and_finite() -> None:
    # The Zr+survival convention stays reachable behind the flag (reproducibility) and finite, including
    # on a cyclic lead (empty raw mask -> quotient fallback).
    catalog = _catalog()
    model = _model(catalog, enable_cyclic_graft=True)
    gi = MARK_RULE_TO_INDEX["bond_reroute"]
    for smi in ("CCCCCCCC", "c1ccccc1CCC"):
        tbl = AnalyticPancakeQuotientSampler(model, calibration=_LEGACY).rate_table(_state(smi), 0.5)
        assert bool(torch.isfinite(tbl.productive_family_rates).all())
        assert np.isfinite(float(tbl.productive_total_hazard))
        assert float(tbl.productive_family_rates[gi]) >= 0.0
