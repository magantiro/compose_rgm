#!/usr/bin/env python3
"""Canonical-successor checkpoint selection for RingCore-V1 EDITING checkpoints.

Implements the FROZEN selection rule in ``configs/ringcore_v1_checkpoint_selection.json``. The editing
controller acts on the canonical-successor embedded jump chain -- not raw mark coordinates and not the
continuous-time hazard -- so the load-bearing metric is the canonical-successor negative log-likelihood

    L_edit = -mean log sum_{a : canonical(T(x,a)) == y_teacher} p_theta(a | x, t)

which aggregates every legal mark that lands on the teacher's molecule. Selecting by the hazard-inclusive
GM training loss instead optimizes a term the deployment discards (CHECKPOINT_SELECTION_METRIC_MISMATCH).

Every checkpoint is scored on ONE fixed held-out edit set (built once, seeded, cached to disk), so the
comparison is paired. Hazard calibration is reported SEPARATELY -- never folded into the selection metric.

Usage:
  python scripts/ring_core_checkpoint_selection.py \
      --checkpoint step500=/path/ckpt.pt --checkpoint step2000=/path/rec.pt:current \
      --examples 200 --output diagnostics/coherence/checkpoint_selection.json

A ``label=path`` pair scores the checkpoint's own ``state_dict``; ``label=path:current`` overrides it with
that file's ``current_state_dict`` (the step-N model inside an exact-recovery payload, whose ``state_dict``
would otherwise be the BEST -- i.e. a different step).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, classify_smiles  # noqa: E402
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records  # noqa: E402
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    MARK_RULE_NAMES,
    prepare_factorized_mark_batch,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.fiber import (  # noqa: E402
    ActionFiberSpec,
    AtomState,
    _candidate_actions,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.ring_system_fiber import enumerate_clean_ring_system_deletes  # noqa: E402
from compose_v4.rewrite.tracelet_fiber import (  # noqa: E402
    enumerate_ring_system_restate_actions,
)
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint  # noqa: E402

HELDOUT_SMILES = REPO / "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles"
N_SLOTS = 40
OP_TIME = 0.5

# Executor rule_name -> the dense FAMILY slot that scores it (cycle ops are recorded under executor names).
_CYCLE_OP_EXECUTOR_TO_FAMILY = {"bond_insert": "cycle_insert", "bond_delete": "cycle_attach"}


def _state_fiber_spec(state) -> ActionFiberSpec:
    """Alias-enumeration language RESTRICTED to the atom states already present in this molecule.

    An alias must produce the SAME molecule as the teacher, so it can only use atom states the target
    molecule already contains -- enumerating all 10 elements x 5 H-counts is combinatorially wasteful
    (it made the naive version intractable)."""
    from compose_v4.chem.molecular_graph import is_element

    real = is_element(state.atom_types)
    types = sorted({int(t) for t, ok in zip(state.atom_types, real) if ok})
    hs = sorted({int(h) for h, ok in zip(state.implicit_h_counts, real) if ok} | {0, 1})
    return ActionFiberSpec(
        atom_states=tuple(AtomState(t, 0, h) for t in types for h in hs),
        allow_bond_reroute=False,
    )


def _invariants(state) -> tuple:
    """Graph invariants: (atom-type multiset, bond count, bond-order sum).

    Two graphs that are the SAME molecule must agree on all three, so a mismatch is a sound rejection.
    Measured cost decomposition of a naive alias search: executor 96%, canonicalization 3%, candidate
    generation 0% -- so the useful filter is one applied BEFORE execution, which is what these enable."""
    from compose_v4.chem.molecular_graph import is_element

    real = is_element(state.atom_types)
    idx = [i for i, ok in enumerate(real) if ok]
    types = tuple(sorted(int(state.atom_types[i]) for i in idx))
    bonds = state.bonds
    n_bonds = sum(1 for i in idx for j in idx if i < j and int(bonds[i, j]))
    order_sum = int(sum(int(bonds[i, j]) for i in idx for j in idx if i < j))
    return (types, n_bonds, order_sum)


def _predicted_invariants(state, rule_name: str, action, base: tuple):
    """Predict the successor's invariants WITHOUT executing, or None when the effect is not modelled
    (ring/graft macros) -- None means 'must execute to know', never 'reject'."""
    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.rewrite.operators import (
        AtomDelete,
        AtomInsert,
        AtomRestate,
        BondDelete,
        BondInsert,
        BondReorder,
    )

    types, n_bonds, order_sum = base
    bonds = state.bonds
    if isinstance(action, BondInsert):
        return (types, n_bonds + 1, order_sum + int(action.order))
    if isinstance(action, BondDelete):
        return (types, n_bonds - 1, order_sum - int(bonds[action.a, action.b]))
    if isinstance(action, BondReorder):
        old = int(bonds[action.a, action.b])
        return (types, n_bonds, order_sum - old + int(action.new_order))
    if isinstance(action, AtomInsert):
        new_types = tuple(sorted(types + (int(action.atom_type),)))
        deg = len(action.neighbors)
        added = sum(int(o) for _, o in action.neighbors)
        return (new_types, n_bonds + deg, order_sum + added)
    if isinstance(action, AtomDelete):
        v = int(action.v)
        real = is_element(state.atom_types)
        idx = [i for i, ok in enumerate(real) if ok]
        deg = sum(1 for j in idx if int(bonds[v, j]))
        lost = sum(int(bonds[v, j]) for j in idx)
        rest = list(types)
        rest.remove(int(state.atom_types[v]))
        return (tuple(rest), n_bonds - deg, order_sum - lost)
    if isinstance(action, AtomRestate):
        rest = list(types)
        rest.remove(int(state.atom_types[action.v]))
        return (tuple(sorted(rest + [int(action.atom_type)])), n_bonds, order_sum)
    return None  # ring-system / graft macros: effect not modelled -> execute


# ---- Fixed held-out edit set (built once, cached; identical for every checkpoint) ----


def build_eval_examples(n_molecules: int, seed: int, catalog, cache: Path | None) -> list[dict]:
    if cache is not None and cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        print(json.dumps({"phase": "eval_set_cache_loaded", "path": str(cache),
                          "examples": len(payload["examples"])}), flush=True)
        return payload["examples"]
    text = HELDOUT_SMILES.read_text().split()
    keep: list[str] = []
    for smi in text:
        ok, _ = classify_smiles(smi, BROAD_ORGANIC_V1)
        if ok:
            keep.append(smi)
        if len(keep) >= n_molecules:
            break
    records, _log = build_corrupted_prior_records(
        keep, n_slots=N_SLOTS, depth_max=5, seed=seed, catalog=catalog,
        vocabulary=ORGANIC_VOCABULARY, couplings_per_target=2,
    )
    cyc, _c = build_cycle_op_records(tuple(keep), n_slots=N_SLOTS, seed=seed)
    examples: list[dict] = []
    for rec in tuple(records) + tuple(cyc):
        trace = rec.path.trace
        for i, step in enumerate(trace.steps):
            examples.append({
                "state": rec.path.states[i],
                "action": step.action,
                "rule_name": step.rule_name,
                "rate": float(rec.path.operational_jump_rate(i)),
            })
    # Precompute the alias sets ONCE. They depend only on (molecule, teacher mark) and the executor --
    # no model term enters -- so this expensive step is shared by every checkpoint scored below.
    system = de_novo_rewrite_system()
    t0 = time.perf_counter()
    for ex in examples:
        try:
            ex["aliases"] = find_aliases(
                ex["state"], ex["rule_name"], ex["action"], catalog, system
            )
        except Exception:  # noqa: BLE001
            ex["aliases"] = []
    alias_secs = time.perf_counter() - t0
    if cache is not None:
        torch.save({"examples": examples}, cache)
    print(json.dumps({"phase": "eval_set_built", "molecules": len(keep),
                      "examples": len(examples),
                      "alias_search_seconds": round(alias_secs, 1),
                      "examples_with_alias": sum(1 for e in examples if e.get("aliases"))},
                     sort_keys=True), flush=True)
    return examples


# ---- Alias enumeration: every legal mark landing on the teacher's canonical successor ----


def find_aliases(state, teacher_rule: str, teacher_action, catalog, system) -> list[tuple[str, object]]:
    """Every OTHER legal mark whose canonical successor equals the teacher's -- i.e. the marks the
    canonical-successor kernel aggregates with the teacher.

    MODEL-INDEPENDENT (depends only on the molecule + teacher), so it is computed once with the fixed eval
    set and reused for every checkpoint. ``enumerate_action_fiber`` already executes and canonicalizes each
    candidate and returns ``successor_key``, so we compare keys directly instead of re-executing. Out-of-mask
    candidates later score to probability zero, so a superset is safe; a subset would UNDER-count."""
    teacher_succ = system.apply(state, teacher_rule, teacher_action)
    y_star = canonical_state_key(teacher_succ)
    target_inv = _invariants(teacher_succ)
    base = _invariants(state)
    aliases: list[tuple[str, object]] = []

    # micro / bond families: filter on predicted invariants BEFORE executing (execution is 96% of cost)
    try:
        for rule_name, action in _candidate_actions(state, _state_fiber_spec(state)):
            if rule_name == teacher_rule and action == teacher_action:
                continue
            predicted = _predicted_invariants(state, rule_name, action, base)
            if predicted is not None and predicted != target_inv:
                continue  # sound rejection: equal molecules must share these invariants
            try:
                succ = system.apply(state, rule_name, action)
            except Exception:  # noqa: BLE001 -- illegal candidate
                continue
            if succ is not None and canonical_state_key(succ) == y_star:
                aliases.append((rule_name, action))
    except Exception:  # noqa: BLE001 -- best-effort; the teacher itself is always scored
        pass

    # ring families the micro fiber does not cover (small enumerated sets -> execute directly).
    # bond_reroute (Graft) is deliberately EXCLUDED: _teacher_action_score already returns a logsumexp
    # over graft_successor_groups, i.e. the model aggregates graft aliases internally. Adding them here
    # would DOUBLE-COUNT them and inflate the canonical-successor probability.
    for enum, name in (
        (lambda: enumerate_ring_system_restate_actions(state), "ring_system_restate"),
        (lambda: enumerate_clean_ring_system_deletes(state, catalog), "ring_system_delete"),
    ):
        try:
            for action in enum():
                if name == teacher_rule and action == teacher_action:
                    continue
                succ = system.apply(state, name, action)
                if succ is None or _invariants(succ) != target_inv:
                    continue
                if canonical_state_key(succ) == y_star:
                    aliases.append((name, action))
        except Exception:  # noqa: BLE001
            continue
    return aliases


def _score_marks(model, state, marks: list[tuple[str, object]], catalog) -> list[float]:
    """log p_theta(a | x, t) for each mark; -inf when the mark is outside the model's dense candidates."""
    caps = model.operator_capabilities
    logps: list[float] = []
    for rule_name, action in marks:
        family = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(rule_name, rule_name)
        if family not in MARK_RULE_NAMES:
            logps.append(float("-inf"))
            continue
        try:
            batch = prepare_factorized_mark_batch(
                (state,), (frozen_time(OP_TIME),), (action,), (rule_name,), (1.0,),
                ring_catalog=catalog,
                compute_ring_grow_support=caps.compute_ring_grow_support,
                compute_ring_restates=caps.compute_ring_restates,
                compute_cyclic_graft=caps.compute_cyclic_graft,
                compute_ring_opening=caps.compute_ring_opening,
            )
            with torch.no_grad():
                pred = model.forward_mark_batch(batch)
            logps.append(float(pred.selected_mark_log_probability[0]))
        except Exception:  # noqa: BLE001 -- outside the dense mask => zero probability
            logps.append(float("-inf"))
    return logps


def _logsumexp(values: list[float]) -> float:
    finite = [v for v in values if v > float("-inf")]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


# ---- Scoring one checkpoint on the fixed set ----


def score_checkpoint(model, examples: list[dict], *, alias_search: bool) -> dict:
    catalog = model.ring_catalog
    system = de_novo_rewrite_system()
    caps = model.operator_capabilities
    per_family_canon: dict[str, list[float]] = defaultdict(list)
    canon_nlls: list[float] = []
    raw_nlls: list[float] = []
    sel_probs: list[float] = []
    hazards: list[float] = []
    teacher_rates: list[float] = []
    ranks: list[int] = []
    per_family_rank: dict[str, list[int]] = defaultdict(list)
    alias_counts: Counter = Counter()
    teacher_family_counts: Counter = Counter()
    model_family_mass: dict[str, float] = defaultdict(float)
    skipped = 0

    for ex in examples:
        state, action, rule_name = ex["state"], ex["action"], ex["rule_name"]
        family = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(rule_name, rule_name)
        try:
            batch = prepare_factorized_mark_batch(
                (state,), (frozen_time(OP_TIME),), (action,), (rule_name,), (ex["rate"],),
                ring_catalog=catalog,
                compute_ring_grow_support=caps.compute_ring_grow_support,
                compute_ring_restates=caps.compute_ring_restates,
                compute_cyclic_graft=caps.compute_cyclic_graft,
                compute_ring_opening=caps.compute_ring_opening,
            )
            with torch.no_grad():
                pred = model.forward_mark_batch(batch)
            raw_logp = float(pred.selected_mark_log_probability[0])
            hazard = float(pred.total_hazard[0])
            fam_logp = pred.family_log_probabilities[0]
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if not math.isfinite(raw_logp):
            skipped += 1
            continue

        # family rank of the teacher family + the model's family mass (for calibration)
        fam_probs = fam_logp.exp().tolist()
        order = sorted(range(len(fam_probs)), key=lambda i: -fam_probs[i])
        fam_index = MARK_RULE_NAMES.index(family)
        ranks.append(order.index(fam_index) + 1)
        per_family_rank[family].append(order.index(fam_index) + 1)
        for i, name in enumerate(MARK_RULE_NAMES):
            model_family_mass[name] += fam_probs[i]
        teacher_family_counts[family] += 1

        # canonical-successor aggregation: add every OTHER legal mark landing on the same molecule
        canon_logp = raw_logp
        n_alias = 1
        if alias_search:
            try:
                # aliases are MODEL-INDEPENDENT: precomputed with the fixed eval set when available, so
                # the expensive search runs once instead of once per checkpoint
                aliases = ex.get("aliases")
                if aliases is None:
                    aliases = find_aliases(state, rule_name, action, catalog, system)
                if aliases:
                    extra = _score_marks(model, state, aliases, catalog)
                    canon_logp = _logsumexp([raw_logp, *extra])
                    n_alias += sum(1 for v in extra if v > float("-inf"))
            except Exception:  # noqa: BLE001
                pass
        alias_counts[n_alias] += 1

        canon_nlls.append(-canon_logp)
        raw_nlls.append(-raw_logp)
        sel_probs.append(math.exp(raw_logp))
        hazards.append(hazard)
        teacher_rates.append(float(ex["rate"]))
        per_family_canon[family].append(-canon_logp)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    # family-mass calibration: teacher family distribution vs model predicted family mass
    tot_t = sum(teacher_family_counts.values()) or 1
    q_target = {f: teacher_family_counts.get(f, 0) / tot_t for f in MARK_RULE_NAMES}
    tot_m = sum(model_family_mass.values()) or 1.0
    q_model = {f: model_family_mass.get(f, 0.0) / tot_m for f in MARK_RULE_NAMES}
    tv = 0.5 * sum(abs(q_target[f] - q_model[f]) for f in MARK_RULE_NAMES)
    kl = sum(
        q_target[f] * math.log(q_target[f] / max(q_model[f], 1e-12))
        for f in MARK_RULE_NAMES if q_target[f] > 0
    )

    return {
        # PRIMARY + SECONDARY selection metrics
        "production_weighted_canonical_successor_nll": _mean(canon_nlls),
        "balanced_family_canonical_successor_nll": _mean(
            [_mean(v) for v in per_family_canon.values()]
        ),
        # supporting diagnostics
        "raw_mark_nll": _mean(raw_nlls),
        "selected_mark_probability": _mean(sel_probs),
        "family_calibration_tv": tv,
        "family_calibration_kl": kl,
        "mean_family_target_rank": _mean([float(r) for r in ranks]),
        "family_target_rank_by_family": {k: _mean([float(x) for x in v]) for k, v in per_family_rank.items()},
        "per_family_canonical_nll": {k: _mean(v) for k, v in per_family_canon.items()},
        "teacher_examples_by_family": dict(teacher_family_counts),
        "alias_multiplicity_histogram": dict(alias_counts),
        # hazard reported SEPARATELY -- never part of the selection metric
        "hazard_calibration": {
            "mean_predicted_hazard": _mean(hazards),
            "mean_teacher_rate": _mean(teacher_rates),
            "mean_absolute_hazard_error": _mean(
                [abs(h - r) for h, r in zip(hazards, teacher_rates)]
            ),
        },
        "scored_examples": len(canon_nlls),
        "skipped_examples": skipped,
    }


def _load(spec: str):
    label, _, rest = spec.partition("=")
    path, _, mode = rest.partition(":")
    model, meta = load_factorized_rollout_checkpoint(
        Path(path), expected_scope_hash=BROAD_ORGANIC_V1.scope_hash()
    )
    step = meta.get("completed_steps") if isinstance(meta, dict) else None
    if mode == "current":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["current_state_dict"])
        step = payload.get("completed_steps")
    model.eval()
    return label, model, {"path": path, "state_source": mode or "state_dict", "completed_steps": step}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="label=path[:current] (repeatable)")
    parser.add_argument("--molecules", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--no-alias-search", action="store_true",
                        help="skip canonical aggregation (raw-mark NLL only) -- diagnostic")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    loaded = [_load(spec) for spec in args.checkpoint]
    catalog = loaded[0][1].ring_catalog
    examples = build_eval_examples(args.molecules, args.seed, catalog, args.cache)

    report: dict = {
        "rule": "configs/ringcore_v1_checkpoint_selection.json",
        "primary_metric": "production_weighted_canonical_successor_nll",
        "secondary_metric": "balanced_family_canonical_successor_nll",
        "fixed_eval_set": {
            "source": str(HELDOUT_SMILES.relative_to(REPO)),
            "molecules": args.molecules, "seed": args.seed, "examples": len(examples),
            "alias_search": not args.no_alias_search,
        },
        "checkpoints": {},
    }
    for label, model, meta in loaded:
        print(json.dumps({"phase": "scoring", "checkpoint": label}), flush=True)
        result = score_checkpoint(model, examples, alias_search=not args.no_alias_search)
        report["checkpoints"][label] = {**meta, **result}

    ranked = sorted(
        report["checkpoints"].items(),
        key=lambda kv: kv[1]["production_weighted_canonical_successor_nll"],
    )
    report["ranking_by_primary_metric"] = [k for k, _ in ranked]
    report["best_by_primary_metric"] = ranked[0][0] if ranked else None
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
