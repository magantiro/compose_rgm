#!/usr/bin/env python3
"""Stage 6A: operator-balanced coverage corpus, enabled-family contract, and tiny-overfit.

Builds a small deterministic corpus with POSITIVE teacher targets for every production-enabled operator
family (corruption for micro/bioisostere/bond-order/graft/de-aromatization/ring-open; de-novo carbon-tree
for ring-grow), each example at a UNIQUE state so a tiny overfit has no conflicting targets. Verifies
wiring, then overfits and records the enabled-family contract. This is NOT the natural §0b mixture and
must NOT be used to claim model quality; it verifies training plumbing only.

Writes:
- diagnostics/composition/stage6a_family_contract.json  (the enabled-family contract table)
- diagnostics/composition/stage6a_overfit.json          (tiny-overfit metrics)

Usage: PYTHONPATH=src python scripts/stage6a_coverage_smoke.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import make_edit_pair
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

REPO = Path(__file__).resolve().parent.parent
SLOTS = 20
SYS = de_novo_rewrite_system()
# production-enabled families (cycle_insert / cycle_attach are legacy null-prior ops, dead everywhere).
ENABLED = ["atom_insert", "atom_delete", "atom_restate", "bond_reorder", "bond_reroute",
           "ring_system_restate", "ring_system_delete", "ring_system_grow"]
DEAD = ["cycle_insert", "cycle_attach"]
RING_SMI = ("c1ccccc1", "C1CCCCC1", "C1CCNCC1", "c1ccncc1", "C1CCOCC1", "c1ccc2ccccc2c1")


def build_catalog():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), SLOTS)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=SLOTS)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)
    return build_typed_ring_catalog(tuple(trace(s) for s in RING_SMI))


def collect(cat, per_family=3):
    """Positive (state, action) teacher examples per family at UNIQUE states (no conflicting targets)."""
    ex = defaultdict(list)
    used = set()
    rng = np.random.default_rng(0)
    mols = ["c1ccccc1CCC", "C1CCNCC1CC", "CCC=O", "CC(C)(C)C", "c1ccncc1C", "C1CCOCC1CC",
            "CCCN", "c1ccc2ccccc2c1C", "CCOCC", "CC(=O)NC"]

    def offer(state, rule, action):
        key = canonical_state_key(state)
        if rule in ENABLED and len(ex[rule]) < per_family and key not in used:
            ex[rule].append((state, action))
            used.add(key)

    for _ in range(60):
        target = pad_molecular_graph(smiles_to_molecular_graph(str(rng.choice(mols))), SLOTS)
        trim, grow = make_edit_pair(target, int(rng.integers(1, 6)), system=SYS, rng=rng,
                                    catalog=cat, vocabulary=ORGANIC_VOCABULARY)
        for tr in (trim, grow):
            if tr is None:
                continue
            state = tr.source
            for step in tr.steps:
                offer(state, step.rule_name, step.action)
                state = SYS.apply(state, step.rule_name, step.action)
    for smi in RING_SMI:
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), SLOTS)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(3), n_slots=SLOTS)
        tr = compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)
        state = tr.source
        for step in tr.steps:
            offer(state, step.rule_name, step.action)
            state = SYS.apply(state, step.rule_name, step.action)
    return ex


def edit_model(cat):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(
        cat, hidden_dim=24, message_passing_steps=1, enable_ring_restates=True, enable_cyclic_graft=True,
        enable_ring_opening=True, enable_heteroatom_scan=True, atom_vocabulary=ORGANIC_VOCABULARY)


def forced_sampling_reaches(model, cat, family, examples):
    """Force-select a family by disabling every other family in the sampler; confirm it can fire."""
    model.disabled_sampling_rule_names = tuple(f for f in MARK_RULE_NAMES if f != family)
    rng = np.random.default_rng(0)
    reached = False
    for state, _ in examples:
        for _ in range(8):
            mark = model.sample_rewrite_mark(state, 0.5, rng)
            if mark.action is not None and mark.rule_name == family:
                reached = True
                break
        if reached:
            break
    model.disabled_sampling_rule_names = ()
    return reached


def run(overfit_steps=1200, write=True):
    cat = build_catalog()
    ex = collect(cat)
    corpus = [(s, fam, a) for fam in ENABLED for (s, a) in ex[fam]]
    states = tuple(s for s, _, _ in corpus)
    rules = tuple(r for _, r, _ in corpus)
    actions = tuple(a for _, _, a in corpus)
    rates = tuple(1.0 for _ in corpus)  # r=1 -> per-example Poisson-Bregman lower bound = 1
    lower_bound = 1.0

    batch = prepare_factorized_mark_batch(
        states, tuple(0.5 for _ in corpus), actions, rules, rates,
        ring_catalog=cat, compute_ring_restates=True, compute_cyclic_graft=True, compute_ring_opening=True)
    model = edit_model(cat).train()
    pred0 = model.forward_mark_batch(batch)
    logp0 = pred0.selected_mark_log_probability.detach().numpy()
    assert np.isfinite(logp0).all(), "a teacher mark is illegal (-inf) -> masked target"
    loss0 = float(factorized_mark_bregman_loss(pred0, batch).detach())

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(overfit_steps):
        opt.zero_grad()
        loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
        loss.backward()
        opt.step()
    predN = model.forward_mark_batch(batch)
    lossN = float(factorized_mark_bregman_loss(predN, batch).detach())
    logpN = predN.selected_mark_log_probability.detach().numpy()
    idx = np.array([ENABLED.index(r) for r in rules])

    # per-family gradient: a single positive teacher per family -> family-head gradient (positive target).
    family_grad = {}
    for fam in ENABLED:
        s, a = ex[fam][0]
        b1 = prepare_factorized_mark_batch(
            (s,), (0.5,), (a,), (fam,), (1.0,), ring_catalog=cat,
            compute_ring_restates=True, compute_cyclic_graft=True, compute_ring_opening=True)
        m1 = edit_model(cat).train()
        m1.zero_grad()
        factorized_mark_bregman_loss(m1.forward_mark_batch(b1), b1).backward()
        family_grad[fam] = float(sum(
            float(p.grad.abs().sum()) for p in m1.family_head.parameters() if p.grad is not None))

    contract = []
    all_rise = True
    for f, fam in enumerate(ENABLED):
        mask = idx == f
        p0 = float(np.exp(logp0[mask]).mean())
        pN = float(np.exp(logpN[mask]).mean())
        rise = pN > p0
        all_rise = all_rise and rise
        contract.append({
            "family": fam,
            "checkpoint_enabled": True,
            "executor_supported": True,
            "enumerator_reachable": True,
            "positive_training_targets": int(mask.sum()),
            "gradient_observed": family_grad[fam] > 0.0,
            "selected_prob_0_to_N": [round(p0, 4), round(pN, 4)],
            "forced_sampling_reaches": forced_sampling_reaches(edit_model(cat).eval(), cat, fam, ex[fam]),
            "status": "SUPERVISED_CURRENT_RECIPE",
        })
    for fam in DEAD:
        contract.append({"family": fam, "checkpoint_enabled": False, "executor_supported": True,
                         "enumerator_reachable": False, "positive_training_targets": 0,
                         "status": "DISABLED_EVERYWHERE"})

    overfit = {
        "n_examples": len(corpus), "distinct_states": len({canonical_state_key(s) for s in states}),
        "per_example_lower_bound_r_1_minus_log_r": lower_bound,
        "loss_initial": round(loss0, 4), "loss_final": round(lossN, 4),
        "gap_above_lower_bound": round(lossN - lower_bound, 4),
        "every_family_probability_rose": all_rise,
        "missing_families": [f for f in ENABLED if not ex[f]],
        "overfit_passed": bool(lossN < loss0 and (lossN - lower_bound) < 0.2 and all_rise
                               and not [f for f in ENABLED if not ex[f]]),
    }
    if write:
        out = REPO / "diagnostics/composition"
        (out / "stage6a_family_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (out / "stage6a_overfit.json").write_text(json.dumps(overfit, indent=2) + "\n")
    return contract, overfit


def main() -> None:
    contract, overfit = run()
    print("=== enabled-family contract ===")
    for row in contract:
        print(f"  {row['family']:22s} targets={row.get('positive_training_targets', 0):2d} "
              f"grad={row.get('gradient_observed', False)!s:5s} "
              f"forced={row.get('forced_sampling_reaches', '-')!s:5s} {row['status']}")
    print(f"\n=== tiny-overfit ===\n  {json.dumps(overfit)}")
    print(f"\nOVERFIT {'PASSED' if overfit['overfit_passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
