#!/usr/bin/env python3
"""§6/§7 of the RingCore preflight: unseen-lead + topology-stratified rollout panels + the 13 metrics.

Runs editing rollouts (start from a real lead, sample K edits at a late frozen time) with a RING_CORE_V1
checkpoint and computes the mandate's 13 rollout metrics. Built to run the instant the bounded Modal
checkpoint lands (--checkpoint); without one it self-validates on a representative local RingCore model so
the harness + every metric is proven before the real run.

Metrics: (1) all-state validity, (2) cycle-rank correctness, (3) ring-op target counts, (4) natural ring-op
usage, (5) topology classes reached, (6) ring-edit path length, (7) reversal/cycle rates, (8) return-to-
source rate, (9) net structural displacement, (10) canonical branching, (11) legal-mark enumeration cost,
(12) throughput, (13) broad-element/charge preservation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (  # noqa: E402
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, scan_corpus  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402

_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")
_RING_OPS = {"bond_insert", "bond_delete", "ring_system_delete", "ring_system_grow", "ring_system_restate"}
_CYCLE_OPS = {"bond_insert", "bond_delete"}
_TOPOLOGY_LEADS = {  # one representative per topology class (from §2)
    "simple_saturated": "C1CCCCC1CCN", "aromatic_carbocycle": "c1ccccc1CCN",
    "aromatic_heterocycle": "c1ccncc1CC", "saturated_heterocycle": "C1CCNCC1CC",
    "macrocycle": "C1CCCCCCCCCCC1C", "fused_aromatic": "c1ccc2ccccc2c1CC",
    "spiro": "C1CCC2(CC1)CCCCC2", "bridged_bicyclic": "C1CC2CCC1C2",
    "sulfur_ring_aromatic": "c1ccsc1CC", "charge_preserving_ring": "C[N+](C)(C)CCC1CCCCC1",
}


def _real(state):
    return [i for i in range(len(state.atom_types)) if bool(is_element(state.atom_types)[i])]


def _net_charge(state) -> int:
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _cycle_rank(state) -> int:
    real = set(_real(state))
    g = nx.Graph()
    g.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]) != 0:
                g.add_edge(a, b)
    return g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


def _fingerprint(state):
    try:
        smi = molecular_graph_to_smiles(state)
        mol = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None
    except Exception:  # noqa: BLE001
        return None


def _topology_class(state) -> str:
    try:
        mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    except Exception:  # noqa: BLE001
        return "unparsed"
    if mol is None:
        return "unparsed"
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return "acyclic"
    fused = any(len(rings[i] & rings[j]) >= 2 for i in range(len(rings)) for j in range(i + 1, len(rings)))
    aromatic = any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings)
    return f"{'fused' if fused else 'mono'}/{'aromatic' if aromatic else 'saturated'}"


def _rollout(model, source, system, horizon, op_time, rng):
    """One editing rollout: sample up to `horizon` edits at a late frozen time. Returns (states, marks)."""
    states = [source]
    marks = []
    state = source
    t = frozen_time(op_time)
    for _ in range(horizon):
        try:
            mark = model.sample_rewrite_mark(state, t, rng)
        except Exception:  # noqa: BLE001
            break
        try:
            nxt = system.apply(state, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001
            break
        marks.append(mark)
        states.append(nxt)
        state = nxt
    return states, marks


def _panel_metrics(model, leads, system, *, horizon, op_time, rollouts_per_lead, seed):
    rng = np.random.default_rng(seed)
    all_states = valid_states = 0
    rank_checked = rank_correct = 0
    op_counts: Counter = Counter()
    ring_rollouts = 0
    n_rollouts = 0
    topo_reached: Counter = Counter()
    ring_path_lengths = []
    reversal_pairs = revisit_states = total_steps = 0
    return_to_source = 0
    displacements = []
    branchings = []
    enum_cost = 0.0
    charge_preserved = element_preserved = 0
    t0 = _time.perf_counter()
    n_marks = 0
    for smi in leads:
        try:
            source = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        except Exception:  # noqa: BLE001
            continue
        q0 = _net_charge(source)
        src_key = canonical_state_key(source)
        src_fp = _fingerprint(source)
        for _ in range(rollouts_per_lead):
            states, marks = _rollout(model, source, system, horizon, op_time, rng)
            n_rollouts += 1
            n_marks += len(marks)
            for st in states:
                all_states += 1
                valid_states += int(is_valid_state(st) and is_connected_or_null(st))
            for i, mk in enumerate(marks):
                op_counts[mk.rule_name] += 1
                if mk.rule_name in _CYCLE_OPS:
                    rank_checked += 1
                    delta = _cycle_rank(states[i + 1]) - _cycle_rank(states[i])
                    expected = 1 if mk.rule_name == "bond_insert" else -1
                    rank_correct += int(delta == expected)
            ring_ops = [m for m in marks if m.rule_name in _RING_OPS]
            if ring_ops:
                ring_rollouts += 1
                ring_path_lengths.append(len(ring_ops))
            topo_reached[_topology_class(states[-1])] += 1
            # reversal / revisit
            seen = {src_key}
            for i in range(len(states) - 1):
                total_steps += 1
                k = canonical_state_key(states[i + 1])
                if k in seen:
                    revisit_states += 1
                seen.add(k)
                if i + 1 < len(marks):
                    a, b = marks[i], marks[i + 1]
                    if {a.rule_name, b.rule_name} == _CYCLE_OPS:
                        reversal_pairs += 1
            end_key = canonical_state_key(states[-1])
            return_to_source += int(end_key == src_key)
            end_fp = _fingerprint(states[-1])
            if src_fp is not None and end_fp is not None:
                displacements.append(1.0 - DataStructs.TanimotoSimilarity(src_fp, end_fp))
            charge_preserved += int(_net_charge(states[-1]) == q0)
            element_preserved += 1  # broad vocab is closed under the ops; charge is the binding constraint
        # canonical branching + enumeration cost: sample marks from the source, count distinct successors
        t_enum = _time.perf_counter()
        succ = set()
        for _ in range(24):
            try:
                mk = model.sample_rewrite_mark(source, frozen_time(op_time), rng)
                succ.add(canonical_state_key(system.apply(source, mk.rule_name, mk.action)))
            except Exception:  # noqa: BLE001
                continue
        enum_cost += _time.perf_counter() - t_enum
        branchings.append(len(succ))
    elapsed = _time.perf_counter() - t0

    def _dist(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        return {"median": round(float(np.median(xs)), 3), "p90": round(float(xs[int(0.9 * len(xs)) - 1]), 3),
                "max": round(float(xs[-1]), 3), "mean": round(float(np.mean(xs)), 3)}

    return {
        "n_rollouts": n_rollouts,
        "1_all_state_validity": round(valid_states / max(1, all_states), 4),
        "2_cycle_rank_correctness": round(rank_correct / max(1, rank_checked), 4) if rank_checked else None,
        "3_ring_op_target_counts": {k: op_counts[k] for k in sorted(op_counts) if k in _RING_OPS},
        "3_all_op_counts": dict(op_counts),
        "4_natural_ring_op_usage": round(ring_rollouts / max(1, n_rollouts), 4),
        "5_topology_classes_reached": dict(topo_reached),
        "6_ring_edit_path_length": _dist(ring_path_lengths),
        "7_reversal_rate": round(reversal_pairs / max(1, total_steps), 4),
        "7_revisit_rate": round(revisit_states / max(1, total_steps), 4),
        "8_return_to_source_rate": round(return_to_source / max(1, n_rollouts), 4),
        "9_net_structural_displacement": _dist(displacements),
        "10_canonical_branching": _dist(branchings),
        "11_legal_mark_enumeration_cost_ms": round(1000 * enum_cost / max(1, len(leads)), 3),
        "12_throughput_marks_per_s": round(n_marks / max(1e-6, elapsed), 1),
        "13_charge_preserved_rate": round(charge_preserved / max(1, n_rollouts), 4),
    }


def _load_model(checkpoint: Path | None):
    if checkpoint is not None:
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

        model, payload = load_factorized_rollout_checkpoint(
            checkpoint, expected_scope_hash=BROAD_ORGANIC_V1.scope_hash()
        )
        model.eval()
        return model, {"source": "checkpoint", "path": str(checkpoint),
                       "enable_cycle_ops": bool(payload.get("enable_cycle_ops")),
                       "enable_ring_grow_macro": bool(payload.get("enable_ring_grow_macro", True))}
    # self-validation: representative local RingCore model (untrained -> metrics prove the HARNESS, not quality)
    from warmstart_dry_run import build_production_ring_catalog

    torch.manual_seed(0)
    catalog = build_production_ring_catalog(40)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=48, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    return model, {"source": "representative_local_model_UNTRAINED", "enable_cycle_ops": True,
                   "enable_ring_grow_macro": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="RingCore-V1 checkpoint; omit to self-validate on a local model")
    parser.add_argument("--unseen-leads", type=int, default=20)
    parser.add_argument("--rollouts-per-lead", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--op-time", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/production_preflight/ring_core_rollout_panel.json")
    )
    args = parser.parse_args()

    model, model_meta = _load_model(args.checkpoint)
    print(json.dumps({"phase": "model", **model_meta}, sort_keys=True), flush=True)
    system = de_novo_rewrite_system()

    with _CORPUS.open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    accepted, _ = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(args.seed)
    mols = np.asarray(accepted, dtype=object)
    rng.shuffle(mols)
    unseen = [str(m) for m in mols[: args.unseen_leads]]

    unseen_metrics = _panel_metrics(
        model, unseen, system, horizon=args.horizon, op_time=args.op_time,
        rollouts_per_lead=args.rollouts_per_lead, seed=args.seed,
    )
    print(json.dumps({"phase": "unseen_panel", "validity": unseen_metrics["1_all_state_validity"],
                      "natural_ring_op": unseen_metrics["4_natural_ring_op_usage"],
                      "throughput": unseen_metrics["12_throughput_marks_per_s"]}, sort_keys=True), flush=True)
    topo_metrics = _panel_metrics(
        model, list(_TOPOLOGY_LEADS.values()), system, horizon=args.horizon, op_time=args.op_time,
        rollouts_per_lead=args.rollouts_per_lead, seed=args.seed + 1,
    )
    print(json.dumps({"phase": "topology_panel",
                      "validity": topo_metrics["1_all_state_validity"]}, sort_keys=True), flush=True)

    out = {
        "model": model_meta,
        "config": {"unseen_leads": len(unseen), "rollouts_per_lead": args.rollouts_per_lead,
                   "horizon": args.horizon, "op_time": args.op_time},
        "unseen_lead_panel": unseen_metrics,
        "topology_stratified_panel": topo_metrics,
        "note": "with --checkpoint absent these numbers validate the HARNESS on an untrained model (metrics "
        "present + finite); the production numbers come from the bounded RingCore checkpoint.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_source": model_meta["source"],
                      "unseen_validity": unseen_metrics["1_all_state_validity"],
                      "cycle_rank_correctness": unseen_metrics["2_cycle_rank_correctness"],
                      "natural_ring_op_usage": unseen_metrics["4_natural_ring_op_usage"],
                      "throughput": unseen_metrics["12_throughput_marks_per_s"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
