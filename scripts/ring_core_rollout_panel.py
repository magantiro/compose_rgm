#!/usr/bin/env python3
"""RING_CORE_V1 rollout analysis harness -- ANALYSIS_READY_FOR_RINGCORE_CHECKPOINT.

Hardened per the owner mandate (10 requirements). Runs unseen-lead + topology-stratified editing rollouts on
a RingCore-V1 step-500 checkpoint and returns an interpretation verdict, with strict checkpoint-identity
verification, support-level legacy-macro exclusion, split charge-preservation semantics, a paired A/B/C
analysis (step-0 semantic init / step-500 trained / uniform-legal), compositional-ring behaviour + dead-head
detection, exact-zero hard correctness gates, learning/utilization diagnostics, a stable 13-metric schema
with provenance, and NON_SCIENTIFIC_PREFLIGHT quarantine.

Invoke ONLY after the step-500 checkpoint (+ training log) is present:
  python scripts/ring_core_rollout_panel.py --checkpoint <ckpt> [--step0-checkpoint <ckpt0>] \
      [--training-log <jsonl>]
--self-validate runs the structural machinery on a local model (proves the harness; not production numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import BondDelete, BondInsert  # noqa: E402
from ring_core_identity import verify_checkpoint_identity  # noqa: E402

STATUS = "ANALYSIS_READY_FOR_RINGCORE_CHECKPOINT"
SCHEMA_VERSION = "ringcore_rollout_v2"
QUARANTINE = "NON_SCIENTIFIC_PREFLIGHT"
_GROW_IDX = MARK_RULE_NAMES.index("ring_system_grow")
_CYCLE_OPS = {"bond_insert", "bond_delete"}
_RING_OPS = {"bond_insert", "bond_delete", "ring_system_delete", "ring_system_grow", "ring_system_restate"}
_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")
_TOPOLOGY_LEADS = {
    "simple_saturated": "C1CCCCC1CCN", "aromatic_carbocycle": "c1ccccc1CCN",
    "aromatic_heterocycle": "c1ccncc1CC", "saturated_heterocycle": "C1CCNCC1CC",
    "macrocycle": "C1CCCCCCCCCCC1C", "fused_aromatic": "c1ccc2ccccc2c1CC",
    "spiro": "C1CCC2(CC1)CCCCC2", "bridged_bicyclic": "C1CC2CCC1C2",
    "sulfur_ring_aromatic": "c1ccsc1CC", "charge_preserving_ring": "C[N+](C)(C)CCC1CCCCC1",
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _real(state):
    return [i for i in range(len(state.atom_types)) if bool(is_element(state.atom_types)[i])]


def _net_charge(state) -> int:
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _charged_centers(state) -> dict[int, int]:
    """Map real-atom slot -> nonzero formal charge (states are slot-stable, so slot identity persists)."""
    mask = is_element(state.atom_types)
    return {i: int(state.formal_charges[i]) for i in range(len(state.atom_types))
            if mask[i] and int(state.formal_charges[i]) != 0}


def _n_atoms(state) -> int:
    return int(is_element(state.atom_types).sum())


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
        mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
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
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return "acyclic"
    fused = any(len(rings[i] & rings[j]) >= 2 for i in range(len(rings)) for j in range(i + 1, len(rings)))
    aromatic = any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings)
    return f"{'fused' if fused else 'mono'}/{'aromatic' if aromatic else 'saturated'}"


def _grow_family_mass(model, state, catalog) -> dict:
    """Requirement 2 (support level): the legacy ring_system_grow macro must be absent from legal-mark
    enumeration, contribute zero canonical successors, receive zero mass, and be out of the capability set."""
    batch = prepare_factorized_mark_batch(
        (state,), (frozen_time(12.0),), (None,), (None,), (0.0,), ring_catalog=catalog,
        compute_ring_grow_support=bool(getattr(model, "enable_ring_grow_macro", True)),
    )
    with torch.no_grad():
        pred = model.forward_mark_batch(batch)
    grow_logprob = float(pred.family_log_probabilities[0, _GROW_IDX])
    support = batch.ring_grow_support_mask
    grow_legal = 0 if support is None else int(support[0].sum())
    return {
        "grow_family_log_prob": grow_logprob,
        "grow_family_mass": float(np.exp(grow_logprob)) if grow_logprob != float("-inf") else 0.0,
        "grow_legal_marks": grow_legal,
        "grow_in_capability_set": bool(getattr(model, "enable_ring_grow_macro", True)),
        "kernel_sum": round(float(pred.family_log_probabilities[0].exp().sum()), 6),
    }


class _UniformSampler:
    """Baseline C: sample uniformly over the executor-legal cycle_close/cycle_open marks at a state."""

    enable_ring_grow_macro = False

    def __init__(self, system):
        self.system = system

    def sample_rewrite_mark(self, state, time, rng):
        real = _real(state)
        marks = []
        for a in real:
            for b in real:
                if a < b:
                    if int(state.bonds[a, b]) == 0:
                        marks.append(("bond_insert", BondInsert(a, b, 1)))
                    else:
                        marks.append(("bond_delete", BondDelete(a, b)))
        if not marks:
            raise RuntimeError("no legal cycle mark")
        rule, action = marks[int(rng.integers(len(marks)))]

        class _M:  # minimal mark shim with the fields the harness reads
            def __init__(self, r, a):
                self.rule_name, self.action = r, a
        return _M(rule, action)


def _rollout(model, source, system, horizon, op_time, rng):
    states, marks = [source], []
    state = source
    t = frozen_time(op_time)
    for _ in range(horizon):
        try:
            mk = model.sample_rewrite_mark(state, t, rng)
            nxt = system.apply(state, mk.rule_name, mk.action)
        except Exception:  # noqa: BLE001
            break
        marks.append(mk)
        states.append(nxt)
        state = nxt
    return states, marks


def _panel_metrics(model, leads, system, *, horizon, op_time, rollouts_per_lead, seed, catalog):
    rng = np.random.default_rng(seed)
    all_states = valid_states = 0
    rank_checked = rank_correct = 0
    op_counts: Counter = Counter()
    ring_rollouts = n_rollouts = n_marks = 0
    topo_reached: Counter = Counter()
    ring_path_lengths, displacements, branchings, unique_per_path = [], [], [], []
    reversal_pairs = revisit_states = total_steps = return_to_source = 0
    # requirement 3 -- charge split
    net_charge_ok = centers_ok = charge_mutations = charged_context = 0
    # requirement 5 -- compositional-ring behaviour
    productive_close = productive_open = productive_delete = 0
    close_then_open = repeated_toggle = net_accum = net_loss = 0
    subtype_sources: dict[str, set] = {"bond_insert": set(), "bond_delete": set(),
                                       "ring_system_delete": set(), "ring_system_restate": set()}
    # requirement 6 -- hard-gate counters (must all end at 0)
    hard = Counter()
    grow_probe = None
    t0 = _time.perf_counter()
    for smi in leads:
        try:
            source = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        except Exception:  # noqa: BLE001
            continue
        if grow_probe is None and hasattr(model, "forward_mark_batch"):
            grow_probe = _grow_family_mass(model, source, catalog)
            if grow_probe["grow_family_mass"] > 0 or grow_probe["grow_legal_marks"] > 0 \
                    or grow_probe["grow_in_capability_set"]:
                hard["legacy_macro_access"] += 1
            if abs(grow_probe["kernel_sum"] - 1.0) > 1e-3:
                hard["non_normalized_kernel"] += 1
        q0 = _net_charge(source)
        src_centers = _charged_centers(source)
        if src_centers:
            charged_context += rollouts_per_lead
        src_key = canonical_state_key(source)
        src_fp = _fingerprint(source)
        src_rank = _cycle_rank(source)
        for _ in range(rollouts_per_lead):
            states, marks = _rollout(model, source, system, horizon, op_time, rng)
            n_rollouts += 1
            n_marks += len(marks)
            for st in states[1:]:
                all_states += 1
                ok = is_valid_state(st) and is_connected_or_null(st)
                valid_states += int(ok)
                if not ok:
                    hard["invalid_or_disconnected_committed"] += 1
                if _n_atoms(st) > 40:
                    hard["over_40_atoms"] += 1
            for i, mk in enumerate(marks):
                op_counts[mk.rule_name] += 1
                if mk.rule_name in subtype_sources:
                    subtype_sources[mk.rule_name].add(smi)
                if mk.rule_name in _CYCLE_OPS:
                    rank_checked += 1
                    delta = _cycle_rank(states[i + 1]) - _cycle_rank(states[i])
                    expected = 1 if mk.rule_name == "bond_insert" else -1
                    rank_correct += int(delta == expected)
                    if delta == 1:
                        productive_close += 1
                    elif delta == -1:
                        productive_open += 1
                elif mk.rule_name == "ring_system_delete":
                    productive_delete += 1
                # executor/successor identity: re-apply must reproduce the same canonical key
                try:
                    re_key = canonical_state_key(system.apply(states[i], mk.rule_name, mk.action))
                    if re_key != canonical_state_key(states[i + 1]):
                        hard["executor_successor_mismatch"] += 1
                except Exception:  # noqa: BLE001
                    hard["illegal_committed_action"] += 1
            # toggling
            for i in range(len(marks) - 1):
                a, b = marks[i], marks[i + 1]
                if {a.rule_name, b.rule_name} == _CYCLE_OPS and getattr(a.action, "a", None) == \
                        getattr(b.action, "a", None) and getattr(a.action, "b", None) == \
                        getattr(b.action, "b", None):
                    close_then_open += 1
            ring_ops = [m for m in marks if m.rule_name in _RING_OPS]
            if ring_ops:
                ring_rollouts += 1
                ring_path_lengths.append(len(ring_ops))
            end = states[-1]
            end_rank = _cycle_rank(end)
            net_accum += int(end_rank > src_rank)
            net_loss += int(end_rank < src_rank)
            repeated_toggle += int(end_rank == src_rank and len([m for m in marks if m.rule_name in _CYCLE_OPS]) >= 2)
            topo_reached[_topology_class(end)] += 1
            seen = {src_key}
            for st in states[1:]:
                total_steps += 1
                k = canonical_state_key(st)
                if k in seen:
                    revisit_states += 1
                seen.add(k)
            unique_per_path.append(len(seen))
            reversal_pairs += close_then_open
            if canonical_state_key(end) == src_key:
                return_to_source += 1
            end_fp = _fingerprint(end)
            if src_fp is not None and end_fp is not None:
                displacements.append(1.0 - DataStructs.TanimotoSimilarity(src_fp, end_fp))
            # charge split
            if _net_charge(end) == q0:
                net_charge_ok += 1
            else:
                charge_mutations += 1
                hard["unexplained_charge_change"] += 1
            end_centers = _charged_centers(end)
            if all(end_centers.get(i) == c for i, c in src_centers.items()):
                centers_ok += 1
        t_enum = _time.perf_counter()
        succ = set()
        for _ in range(24):
            try:
                mk = model.sample_rewrite_mark(source, frozen_time(op_time), rng)
                succ.add(canonical_state_key(system.apply(source, mk.rule_name, mk.action)))
            except Exception:  # noqa: BLE001
                continue
        branchings.append(len(succ))
        _panel_metrics._enum_ms = _panel_metrics.__dict__.get("_enum_ms", 0.0) + \
            (_time.perf_counter() - t_enum) * 1000
    elapsed = _time.perf_counter() - t0

    def _d(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        return {"median": round(float(np.median(xs)), 3),
                "p90": round(float(xs[min(len(xs) - 1, int(0.9 * len(xs)))]), 3),
                "max": round(float(xs[-1]), 3), "mean": round(float(np.mean(xs)), 3)}

    cycle_total = op_counts.get("bond_insert", 0) + op_counts.get("bond_delete", 0)
    return {
        "n_rollouts": n_rollouts,
        "1_all_state_validity": round(valid_states / max(1, all_states), 4),
        "2_cycle_rank_correctness": round(rank_correct / max(1, rank_checked), 4) if rank_checked else None,
        "3_ring_op_target_counts": {k: op_counts[k] for k in sorted(op_counts) if k in _RING_OPS},
        "3_all_op_counts": dict(op_counts),
        "4_natural_ring_op_usage": round(ring_rollouts / max(1, n_rollouts), 4),
        "5_topology_classes_reached": dict(topo_reached),
        "6_ring_edit_path_length": _d(ring_path_lengths),
        "7_reversal_rate": round(reversal_pairs / max(1, total_steps), 4),
        "7_revisit_rate": round(revisit_states / max(1, total_steps), 4),
        "8_return_to_source_rate": round(return_to_source / max(1, n_rollouts), 4),
        "9_net_structural_displacement": _d(displacements),
        "10_canonical_branching": _d(branchings),
        "11_legal_mark_enumeration_cost_ms": round(
            _panel_metrics.__dict__.pop("_enum_ms", 0.0) / max(1, len(leads)), 3),
        "12_throughput_marks_per_s": round(n_marks / max(1e-6, elapsed), 1),
        "13_charge_preserved_rate": round(net_charge_ok / max(1, n_rollouts), 4),
        # requirement 3
        "charge": {
            "net_formal_charge_preserved": round(net_charge_ok / max(1, n_rollouts), 4),
            "protected_charged_centers_preserved": round(centers_ok / max(1, n_rollouts), 4),
            "unexplained_charge_mutations": charge_mutations,
            "charged_context_rollouts": charged_context,
        },
        # requirement 5
        "compositional_ring_behavior": {
            "cycle_close_count": op_counts.get("bond_insert", 0),
            "cycle_open_count": op_counts.get("bond_delete", 0),
            "ring_delete_count": op_counts.get("ring_system_delete", 0),
            "productive_close": productive_close,
            "productive_open": productive_open,
            "productive_delete": productive_delete,
            "close_then_inverse_open": close_then_open,
            "repeated_ring_toggle_rollouts": repeated_toggle,
            "net_ring_accumulation_rollouts": net_accum,
            "net_ring_loss_rollouts": net_loss,
            "distinct_sources_per_subtype": {k: len(v) for k, v in subtype_sources.items()},
            "cycle_ops_naturally_sampled": cycle_total > 0,
            "toggle_dominated": close_then_open > 0.5 * max(1, cycle_total),
        },
        "unique_states_per_path": _d(unique_per_path),
        "grow_exclusion_probe": grow_probe,
        # requirement 6
        "hard_gate_violations": dict(hard),
        "hard_gates_all_zero": sum(hard.values()) == 0,
    }


def _learning_diagnostics(training_log: Path | None) -> dict:
    """Requirement 7: extract fixed-diagnostic loss + gradient/update norms at steps 0/250/500 from the
    training log (JSONL). Best-effort against the trainer's emitted keys; reports what is present."""
    if training_log is None or not training_log.exists():
        return {"available": False,
                "note": "provide --training-log <jsonl> from the bounded run to fill steps 0/250/500 "
                "diagnostic loss, target log-prob, hazard error, and per-cycle-subtype grad/update norms."}
    records = []
    for line in training_log.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_step = {}
    for r in records:
        step = r.get("step", r.get("global_step"))
        if step in (0, 250, 500):
            by_step[step] = {k: v for k, v in r.items()
                             if isinstance(v, (int, float)) or "grad" in k or "loss" in k or "norm" in k}
    return {"available": True, "steps": by_step, "n_records": len(records)}


def _interpretation(unseen: dict, trained_vs_init: dict, learning: dict) -> dict:
    """Requirement 9: the 4-verdict interpretation gate + the smallest required next action."""
    reasons = []
    hard_ok = unseen["hard_gates_all_zero"]
    if not hard_ok:
        return {"verdict": "NO_GO_RINGCORE_TRAINING",
                "reason": f"hard correctness gates nonzero: {unseen['hard_gate_violations']}",
                "next_action": "fix the failing executor/validity/charge/kernel invariant"}
    beh = unseen["compositional_ring_behavior"]
    cycle_sampled = beh["cycle_ops_naturally_sampled"]
    toggle_dom = beh["toggle_dominated"]
    non_ring_active = sum(unseen["3_all_op_counts"].get(k, 0)
                          for k in ("atom_insert", "atom_delete", "atom_restate", "bond_reorder")) > 0
    diagnostic_improves = None
    steps = learning.get("steps", {}) if learning.get("available") else {}
    if 0 in steps and 500 in steps:
        l0 = steps[0].get("loss", steps[0].get("diagnostic_loss"))
        l5 = steps[500].get("loss", steps[500].get("diagnostic_loss"))
        if l0 is not None and l5 is not None:
            diagnostic_improves = l5 < l0
    if not cycle_sampled:
        return {"verdict": "GO_AFTER_RECIPE_ADJUSTMENT",
                "reason": "no compositional cycle op is naturally sampled after training -- cycle heads are "
                "potentially dead/underweighted (inspect their gradients + updates + calibration)",
                "smallest_correction": "cycle-op mixture weight OR family-mass calibration (raise the cycle "
                "family_head bias per §4) -- NOT a GO merely because states are valid"}
    if toggle_dom:
        return {"verdict": "GO_AFTER_RECIPE_ADJUSTMENT",
                "reason": "ring behaviour is dominated by close->inverse-open toggling (net-zero)",
                "smallest_correction": "subtype calibration or curriculum scheduling to reward productive "
                "restructuring over toggling"}
    if not non_ring_active:
        reasons.append("ordinary non-ring editing is inactive")
    if diagnostic_improves is False:
        return {"verdict": "GO_AFTER_RECIPE_ADJUSTMENT",
                "reason": "fixed diagnostic did not improve step-0->500",
                "smallest_correction": "family-mass calibration or learning-rate/curriculum schedule"}
    if diagnostic_improves is None:
        return {"verdict": "GO_AFTER_PRODUCTION_SCHEDULE_CHECK",
                "reason": "correctness + natural compositional editing hold, but the 500-step diagnostic "
                "trace (training log) was not supplied to confirm the schedule is representative",
                "next_action": "supply --training-log and confirm the compressed 500-step schedule matches "
                "the intended full-run schedule"}
    if reasons:
        return {"verdict": "GO_AFTER_RECIPE_ADJUSTMENT", "reason": "; ".join(reasons),
                "smallest_correction": "raise non-ring editing weight in the mixture"}
    return {"verdict": "GO_FOR_FULL_RINGCORE_TRAINING",
            "reason": "hard gates zero; diagnostic improves; compositional ring ops occur naturally and are "
            "not toggle-dominated; non-ring editing active; legacy macro fully excluded",
            "next_action": "proceed to the full RingCore-V1 training run (schedule confirmed representative)"}


def _load(checkpoint, self_validate):
    if checkpoint is not None:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload = dict(raw) if isinstance(raw, dict) else {}
        identity = verify_checkpoint_identity(payload, checkpoint_path=checkpoint)
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
        model, _ = load_factorized_rollout_checkpoint(
            checkpoint, expected_scope_hash=BROAD_ORGANIC_V1.scope_hash())
        model.eval()
        return model, identity
    if not self_validate:
        raise SystemExit("--checkpoint is required (or pass --self-validate to test the harness)")
    from warmstart_dry_run import build_production_ring_catalog
    torch.manual_seed(0)
    cat = build_production_ring_catalog(40)
    model = FactorizedTraceletRateModel(
        cat, hidden_dim=48, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False).eval()
    return model, {"identity_ok": None, "self_validate": True, "model": "representative_UNTRAINED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="RingCore-V1 step-500 checkpoint (B)")
    parser.add_argument("--step0-checkpoint", type=Path, default=None,
                        help="semantically-initialized step-0 checkpoint (A) for the paired analysis")
    parser.add_argument("--training-log", type=Path, default=None, help="training JSONL for requirement 7")
    parser.add_argument("--self-validate", action="store_true")
    parser.add_argument("--unseen-leads", type=int, default=20)
    parser.add_argument("--rollouts-per-lead", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--op-time", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/production_preflight/ring_core_rollout_panel.json"))
    args = parser.parse_args()

    from warmstart_dry_run import build_production_ring_catalog
    catalog = build_production_ring_catalog(40)
    system = de_novo_rewrite_system()
    model_b, identity = _load(args.checkpoint, args.self_validate)
    print(json.dumps({"phase": "identity", **{k: identity.get(k) for k in
          ("identity_ok", "global_step", "checkpoint_sha256", "capability_hash",
           "enable_cycle_ops", "enable_ring_grow_macro")}}, sort_keys=True), flush=True)

    with _CORPUS.open() as h:
        texts = [ln.strip().split()[0] for ln in h if ln.strip()]
    accepted, _ = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(args.seed)
    mols = np.asarray(accepted, dtype=object)
    rng.shuffle(mols)
    unseen = [str(m) for m in mols[: args.unseen_leads]]
    lead_panel_hash = hashlib.sha256(json.dumps(unseen, sort_keys=True).encode()).hexdigest()[:16]

    def _panels(model, tag, seed):
        u = _panel_metrics(model, unseen, system, horizon=args.horizon, op_time=args.op_time,
                           rollouts_per_lead=args.rollouts_per_lead, seed=seed, catalog=catalog)
        t = _panel_metrics(model, list(_TOPOLOGY_LEADS.values()), system, horizon=args.horizon,
                           op_time=args.op_time, rollouts_per_lead=args.rollouts_per_lead,
                           seed=seed + 1, catalog=catalog)
        print(json.dumps({"phase": tag, "validity": u["1_all_state_validity"],
                          "natural_ring_op": u["4_natural_ring_op_usage"],
                          "hard_gates_all_zero": u["hard_gates_all_zero"]}, sort_keys=True), flush=True)
        return {"unseen": u, "topology": t}

    # requirement 4 -- paired A/B/C on identical leads + seeds
    processes = {"B_trained_step500": _panels(model_b, "B_trained_step500", args.seed)}
    if args.step0_checkpoint is not None:
        model_a, _ = _load(args.step0_checkpoint, False)
        processes["A_semantic_init_step0"] = _panels(model_a, "A_semantic_init_step0", args.seed)
    processes["C_uniform_legal"] = _panels(_UniformSampler(system), "C_uniform_legal", args.seed)

    learning = _learning_diagnostics(args.training_log)
    verdict = _interpretation(processes["B_trained_step500"]["unseen"],
                              {k: v["unseen"]["compositional_ring_behavior"] for k, v in processes.items()},
                              learning)

    out = {
        "status": STATUS,
        "quarantine": QUARANTINE,
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "harness_commit": _git_commit(),
            "lead_panel_hash": lead_panel_hash,
            "seed_list_hash": hashlib.sha256(str(args.seed).encode()).hexdigest()[:16],
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "evaluation_config_hash": hashlib.sha256(json.dumps(
                {"horizon": args.horizon, "op_time": args.op_time,
                 "rollouts_per_lead": args.rollouts_per_lead, "unseen_leads": args.unseen_leads},
                sort_keys=True).encode()).hexdigest()[:16],
        },
        "checkpoint_identity": identity,
        "config": {"horizon": args.horizon, "op_time": args.op_time,
                   "rollouts_per_lead": args.rollouts_per_lead, "unseen_leads": len(unseen)},
        "paired_processes": processes,
        "learning_diagnostics": learning,
        "interpretation": verdict,
        "note": "NON_SCIENTIFIC_PREFLIGHT -- these rollouts/summaries must NOT enter manuscript results, "
        "benchmark aggregation, final model selection, or headline figures. With --self-validate the numbers "
        "prove the HARNESS on an untrained model; production numbers require the step-500 checkpoint.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": STATUS, "verdict": verdict["verdict"],
                      "hard_gates_all_zero": processes["B_trained_step500"]["unseen"]["hard_gates_all_zero"],
                      "cycle_ops_naturally_sampled":
                      processes["B_trained_step500"]["unseen"]["compositional_ring_behavior"][
                          "cycle_ops_naturally_sampled"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
