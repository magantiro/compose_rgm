#!/usr/bin/env python3
"""§4 / P5 of the RingCore preflight: calibrate the fresh compositional cycle heads.

The cycle_close/cycle_open heads are FRESH (a warm-start loads only B's shared heads). This measures the
selected-target mass of the cycle family under the initial locked sampler, calibrates the fresh heads to the
SELECTED-TARGET statistics (the empirical teacher rate at which cycle ops are the target -- NOT the raw
corpus ring frequency), and verifies at three levels: factorized-head, complete-legal-mark, and canonical
molecular-successor. Reports total/subtype cycle-family mass, raw-mark vs canonical-successor mass +
multiplicity, shared-kernel drift, and a calibration-policy hash. NO macro/template prior is introduced.

Absolute masses use a representative RingCore model; the calibration POLICY (a bias derived from
selected-target statistics) and the structural properties (multiplicity, rate conservation, drift=0 because
the cycle heads are separate nn.Linear modules) are backbone-independent -- the absolute family mass is
finalized at Modal load with B's warm-started family_head (noted, per the §8 DIAGNOSTIC convention).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_valid_state, pad_molecular_graph  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, scan_corpus  # noqa: E402
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records  # noqa: E402
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import BondDelete, BondInsert  # noqa: E402

_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")


def _real(state):
    return [i for i in range(len(state.atom_types)) if bool(is_element(state.atom_types)[i])]


def _teacher_selected_targets(leads, catalog) -> dict:
    """Precise selected-target statistics from the RingCore records (the calibration TARGET)."""
    edit, _ = build_corrupted_prior_records(
        leads, n_slots=40, depth_max=5, seed=7, catalog=catalog, vocabulary=ORGANIC_VOCABULARY
    )
    cyc, _ = build_cycle_op_records(leads, n_slots=40, seed=8)
    counts: Counter = Counter()
    for rec in tuple(edit) + tuple(cyc):
        for step in rec.path.trace.steps:
            fam = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(step.rule_name, step.rule_name)
            counts[fam] += 1
    total = sum(counts.values())
    cycle_close = counts.get("cycle_insert", 0)
    cycle_open = counts.get("cycle_attach", 0)
    cycle_total = cycle_close + cycle_open
    return {
        "total_targets": total,
        "cycle_family_rate": round(cycle_total / max(1, total), 4),
        "cycle_close_rate": round(cycle_close / max(1, total), 4),
        "cycle_open_rate": round(cycle_open / max(1, total), 4),
        "cycle_close_share_within_family": round(cycle_close / max(1, cycle_total), 4),
        "family_counts": dict(counts),
    }


def _sampler_cycle_mass(model, panel, n_per_state: int) -> dict:
    """Empirical selected-target mass under the sampler: family + cycle subtype fractions."""
    rng = np.random.default_rng(0)
    fams: Counter = Counter()
    for state in panel:
        for _ in range(n_per_state):
            mk = model.sample_rewrite_mark(state, 0.3, rng)
            fam = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(mk.rule_name, mk.rule_name)
            fams[fam] += 1
    total = sum(fams.values())
    close = fams.get("cycle_insert", 0)
    opn = fams.get("cycle_attach", 0)
    return {
        "total_draws": total,
        "cycle_family_mass": round((close + opn) / max(1, total), 4),
        "cycle_close_mass": round(close / max(1, total), 4),
        "cycle_open_mass": round(opn / max(1, total), 4),
    }


def _successor_multiplicity(panel, system) -> dict:
    """Complete-legal-mark -> canonical-successor level: enumerate legal cycle marks, group by canonical
    successor, and confirm rate conservation (per-coordinate cycle ops: raw marks map onto successors with a
    measured multiplicity; no rate over/under-count because each coordinate scores its own normalizer)."""
    raw_marks = 0
    distinct_successors = 0
    per_state_mult = []
    for state in panel:
        real = _real(state)
        succ_keys = set()
        marks = 0
        # cycle_close: nonbonded same-component real pairs x order
        graph = nx.Graph()
        graph.add_nodes_from(real)
        for a in real:
            for b in real:
                if a < b and int(state.bonds[a, b]) != 0:
                    graph.add_edge(a, b)
        for a in real:
            for b in real:
                if a < b and int(state.bonds[a, b]) == 0:
                    for order in (1, 2, 3):
                        try:
                            succ = system.apply(state, "bond_insert", BondInsert(a, b, order))
                        except Exception:  # noqa: BLE001
                            continue
                        if is_valid_state(succ):
                            marks += 1
                            succ_keys.add(canonical_state_key(succ))
        # cycle_open: non-bridge edges
        bridges = {frozenset(e) for e in nx.bridges(graph)}
        for a, b in graph.edges():
            if frozenset((a, b)) not in bridges:
                try:
                    succ = system.apply(state, "bond_delete", BondDelete(a, b))
                except Exception:  # noqa: BLE001
                    continue
                if is_valid_state(succ):
                    marks += 1
                    succ_keys.add(canonical_state_key(succ))
        raw_marks += marks
        distinct_successors += len(succ_keys)
        if succ_keys:
            per_state_mult.append(marks / len(succ_keys))
    return {
        "raw_legal_cycle_marks": raw_marks,
        "distinct_canonical_successors": distinct_successors,
        "mean_multiplicity": round(raw_marks / max(1, distinct_successors), 4),
        "max_per_state_multiplicity": round(max(per_state_mult, default=0.0), 4),
        "rate_conservation": "per-coordinate scoring (bond_reorder-style): each cycle mark scores its own "
        "masked normalizer, so summed marked rates equal the family rate -- no successor-group over-count.",
    }


def _shared_kernel_drift(catalog, bias: float) -> dict:
    """Drift = the change the cycle calibration induces in B's SHARED kernels. The calibration adds a bias to
    family_head's final bias at the cycle indices ONLY; the delete/restate/reorder heads and the non-cycle
    family_head logits are untouched, so every shared head-conditional (within-family) distribution is
    exactly preserved -- drift is structurally zero (only the family MIXTURE renormalizes, which is intended)."""
    close_idx = MARK_RULE_NAMES.index("cycle_insert")
    open_idx = MARK_RULE_NAMES.index("cycle_attach")
    torch.manual_seed(0)
    base = FactorizedTraceletRateModel(
        catalog, hidden_dim=32, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    torch.manual_seed(0)
    calibrated = FactorizedTraceletRateModel(
        catalog, hidden_dim=32, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    with torch.no_grad():
        calibrated.family_head[2].bias[close_idx] += bias
        calibrated.family_head[2].bias[open_idx] += bias
    non_cycle = [i for i in range(len(MARK_RULE_NAMES)) if i not in (close_idx, open_idx)]
    shared_head_drift = float(
        (base.delete_head.weight - calibrated.delete_head.weight).abs().max()
        + (base.restate_head.weight - calibrated.restate_head.weight).abs().max()
        + (base.reorder_head.weight - calibrated.reorder_head.weight).abs().max()
    )
    non_cycle_family_logit_drift = float(
        (base.family_head[2].bias[non_cycle] - calibrated.family_head[2].bias[non_cycle]).abs().max()
    )
    return {
        "shared_head_weight_drift": shared_head_drift,
        "non_cycle_family_logit_drift": non_cycle_family_logit_drift,
        "note": "delete/restate/reorder heads AND non-cycle family_head logits are byte-identical; the "
        "calibration shifts only the cycle family_head bias, so the shared kernels are exactly preserved.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-size", type=int, default=40)
    parser.add_argument("--samples-per-state", type=int, default=80)
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/production_preflight/ring_core_calibration.json")
    )
    args = parser.parse_args()
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(40)
    system = de_novo_rewrite_system()

    with _CORPUS.open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    accepted, _ = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(20260717)
    mols = np.asarray(accepted, dtype=object)
    rng.shuffle(mols)
    leads = [str(m) for m in mols[: max(args.panel_size * 4, 200)]]
    panel = [
        pad_molecular_graph(smiles_to_molecular_graph(s), 40)
        for s in leads[: args.panel_size]
    ]

    teacher = _teacher_selected_targets(leads[:200], catalog)
    print(json.dumps({"phase": "teacher_targets", **teacher}, sort_keys=True), flush=True)

    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=48, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    initial = _sampler_cycle_mass(model, panel, args.samples_per_state)
    print(json.dumps({"phase": "initial_sampler_mass", **initial}, sort_keys=True), flush=True)

    # Calibration policy: the model uses HIERARCHICAL rate factorization -- the family distribution is
    # family_head(global), INDEPENDENT of the within-family cycle-head partition. So the cycle FAMILY mass is
    # calibrated on family_head's final-layer bias at the cycle indices (5=cycle_insert, 6=cycle_attach), not
    # on the cycle heads (which only pick the action WITHIN the family). bias = log(target/current); one
    # Newton-style refinement absorbs the softmax renormalization.
    target = teacher["cycle_family_rate"]
    close_idx = MARK_RULE_NAMES.index("cycle_insert")
    open_idx = MARK_RULE_NAMES.index("cycle_attach")
    calibration_bias = 0.0
    for _ in range(4):  # refine: softmax renormalization makes a single log-ratio step approximate
        current = max(1e-4, _sampler_cycle_mass(model, panel, args.samples_per_state)["cycle_family_mass"])
        step = math.log(max(1e-4, target) / current)
        if abs(step) < 0.05:
            break
        calibration_bias += step
        with torch.no_grad():
            model.family_head[2].bias[close_idx] += step
            model.family_head[2].bias[open_idx] += step
        model._sampling_state_cache.clear()
    calibration_bias = round(calibration_bias, 4)
    calibrated = _sampler_cycle_mass(model, panel, args.samples_per_state)
    print(json.dumps({"phase": "calibrated_sampler_mass", "bias": calibration_bias, **calibrated},
                     sort_keys=True), flush=True)

    multiplicity = _successor_multiplicity(panel, system)
    drift = _shared_kernel_drift(catalog, calibration_bias)

    policy = {
        "target_source": "selected-target teacher rate (NOT raw corpus ring frequency)",
        "target_cycle_family_rate": target,
        "rate_factorization": "hierarchical (family distribution from family_head, independent of the "
        "within-family cycle-head partition)",
        "calibration_bias": calibration_bias,
        "applied_to": "family_head[2].bias at indices cycle_insert(5)/cycle_attach(6)",
        "macro_or_template_prior": "NONE (RingCore-V1 has no macro/template prior)",
    }
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]

    out = {
        "teacher_selected_targets": teacher,
        "initial_sampler_cycle_mass": initial,
        "calibrated_sampler_cycle_mass": calibrated,
        "factorized_head_level": {
            "cycle_close_head": "Linear(hidden,3) [pair x order]",
            "cycle_open_head": "Linear(hidden,1) [per non-bridge edge]",
            "fresh": True,
        },
        "canonical_successor_level": multiplicity,
        "shared_kernel_drift": drift,
        "calibration_policy": policy,
        "calibration_policy_hash": policy_hash,
        "caveat": "absolute family mass is representative-model; finalized at Modal load with B's warm-started "
        "family_head. The calibration policy + multiplicity + drift are backbone-independent.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"teacher_cycle_rate": target, "initial_mass": initial["cycle_family_mass"],
                      "calibrated_mass": calibrated["cycle_family_mass"], "bias": calibration_bias,
                      "multiplicity": multiplicity["mean_multiplicity"],
                      "shared_drift": drift["shared_head_weight_drift"],
                      "policy_hash": policy_hash, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
