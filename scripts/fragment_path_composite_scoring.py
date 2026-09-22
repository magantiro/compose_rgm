#!/usr/bin/env python3
"""Is each constituent of the v3 composite transaction rankable by the proposal law?

A composite whose constituents are each individually in support is an
ACCELERATION: it bundles moves the prior could make but does not make in
sequence.  A composite that needs a constituent the prior cannot rank is a
capability EXTENSION and has to be declared as one rather than wrapped.  v3
changed the route -- ``atom_insert`` then ``bond_reroute`` instead of
``atom_insert``, ring-close, bond-open -- so the v2-era answer does not carry.

BUDGET, and why it is not a ladder.  The v2 probe swept a guessed ladder of
[256, 2048, 16384] draws and left one constituent UNRESOLVED.  A zero at an
underpowered budget is no evidence at all, and the budget that settles it is not
guessable: ``bond_reroute`` may be common or rare at these states and neither was
known.  This probe therefore MEASURES the per-draw rate first and derives the
budget from it:

1.  draw ``rate_draws`` production marks at the constituent's own source state
    and count the family histogram -- this is a measured per-draw family rate,
    p_family, with a Wilson interval;
2.  count D, the number of DISTINCT successors that family produced in those
    draws, and take r = p_family / D as the per-draw rate of any one particular
    successor.  This is an approximation and is labelled as one: it assumes the
    within-family law is not strongly concentrated.  Where it is concentrated on
    the target, r understates and the budget is conservative;
3.  the decisive budget is N* = ceil(log(0.05) / log(1 - r)), the draws needed
    to see a rate-r event at least once with 95% probability.

When p_family is measured as ZERO the rate estimate has no point value, and the
probe reports the 95% upper bound on p_family instead, which is what bounds the
claim.  A family never drawn at a state cannot be said to rank a member of it.

Instrumentation is by WRAPPING and never by transcription.  The production
``_attempt_path_transaction`` still decides everything; the probe replaces the
executor it is handed with a recorder that forwards every call to the real
system and keeps the (source, rule, action, successor) it saw.  The constituents
scored are therefore exactly the transitions production executed.

Transitions are compared on the SUCCESSOR molecule, not on the action object:
two different actions producing the same committed molecule are the same
transition to everything downstream, and the benchmark scores molecules.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.benchmark import fragment_conditioned_sampler as sampler_module
from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
SEED = 20260921
# The transaction samples its payload at time 0.0, so the proposal law is
# interrogated at the same time feature it was drawn under.
TIME_FEATURE = 0.0


class _RecordingSystem:
    """Forwards every call to the real executor and keeps what it saw.

    The production transaction is handed this instead of the rewrite system, so
    the ORIGINAL function still performs and decides every step; nothing here
    re-implements it.
    """

    def __init__(self, system):
        self._system = system
        self.calls: list[tuple] = []

    def apply(self, state, rule_name, action):
        successor = self._system.apply(state, rule_name, action)
        self.calls.append((state, rule_name, action, successor))
        return successor

    def __getattr__(self, name):
        return getattr(self._system, name)


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _draw_census(model, state, rng, draws: int, system) -> tuple[Counter, dict]:
    """Draw production marks at ``state``; return the family census and successors."""
    families: Counter[str] = Counter()
    successors: dict[str, set[str]] = {}
    for _ in range(draws):
        try:
            mark = model.sample_rewrite_mark(state, TIME_FEATURE, rng)
        except Exception:  # noqa: BLE001
            families["<refused>"] += 1
            continue
        if mark.action is None:
            families[mark.rule_name] += 1
            continue
        families[mark.rule_name] += 1
        try:
            successor = system.apply(state, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001, S112 -- a refused mark is simply not a successor
            continue
        smiles = molecular_graph_to_smiles(successor)
        if smiles:
            successors.setdefault(mark.rule_name, set()).add(smiles)
    return families, successors


def _score_constituent(
    model, system, source, rule_name, target_smiles, rng, *, rate_draws: int, cap: int
) -> dict:
    """Measure the per-draw rate first, then spend the budget the rate implies."""
    started = time.time()
    families, successors = _draw_census(model, source, rng, rate_draws, system)
    family_hits = families.get(rule_name, 0)
    p_family = family_hits / rate_draws
    low, high = _wilson(family_hits, rate_draws)
    distinct = len(successors.get(rule_name, ()))
    hit_in_rate_phase = target_smiles in successors.get(rule_name, set())
    # Any family may produce the target successor; the question is about the
    # transition, not about which head emitted it.
    hit_any_family = any(target_smiles in s for s in successors.values())

    if p_family > 0 and distinct > 0:
        rate = p_family / distinct
        decisive = math.ceil(math.log(0.05) / math.log(1 - rate)) if rate < 1 else 1
    else:
        rate = None
        decisive = None

    budget = min(decisive, cap) if decisive is not None else cap
    spent = rate_draws
    hit = hit_any_family

    def _spend(draws: int) -> bool:
        nonlocal spent
        extra_families, extra_successors = _draw_census(
            model, source, rng, draws, system
        )
        spent += draws
        families.update(extra_families)
        for key, values in extra_successors.items():
            successors.setdefault(key, set()).update(values)
        return any(target_smiles in s for s in extra_successors.values())

    if not hit and budget > 0:
        hit = _spend(budget)
    hit_at_decisive_budget = hit
    spent_at_decisive_budget = spent
    # A zero at the decisive budget rests on the uniform-within-family step, and
    # that step is optimistic wherever the family law concentrates away from the
    # target. Escalating to the cap is what separates "rarer than the estimate"
    # from "not in the support at all", and it is the standard this workstream
    # already adopted: a zero surviving ONE budget increase is not a hard zero.
    escalated = False
    if not hit and cap > spent:
        escalated = True
        hit = _spend(cap - spent)

    detection_power = (
        1.0 - (1.0 - rate) ** spent if rate is not None else None
    )
    _ = hit_at_decisive_budget
    return {
        "rule": rule_name,
        "target_successor": target_smiles,
        "measured_family_rate": p_family,
        "family_rate_wilson_95": [low, high],
        "family_draws": family_hits,
        "distinct_family_successors": distinct,
        "estimated_per_successor_rate": rate,
        "decisive_budget_for_95_percent_detection": decisive,
        "budget_spent": spent,
        "spent_at_decisive_budget": spent_at_decisive_budget,
        "recovered_at_decisive_budget": hit_at_decisive_budget,
        "escalated_to_cap_after_a_miss": escalated,
        "budget_was_capped": decisive is not None and decisive > cap,
        "realized_detection_power": detection_power,
        "hit_in_rate_phase": hit_in_rate_phase,
        "recovered": hit,
        "verdict": (
            "IN_SUPPORT_AT_DECISIVE_BUDGET" if hit_at_decisive_budget
            else "IN_SUPPORT_ONLY_AFTER_ESCALATION" if hit
            else "NO_HIT_AT_DECISIVE_BUDGET_AND_AT_THE_CAP"
        ),
        "family_census": dict(families.most_common()),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts-per-drug", type=int, default=25)
    parser.add_argument("--transactions-per-drug", type=int, default=1)
    parser.add_argument("--rate-draws", type=int, default=400)
    parser.add_argument("--cap", type=int, default=16384)
    parser.add_argument("--rate-only", action="store_true")
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True, path_program=True)
    prompts = [
        p for p in load_genmol_prompts(MANIFEST) if p.task is FragmentTask.LINKER_DESIGN
    ]

    import rdkit

    rows = []
    started = time.time()
    original = sampler_module._attempt_path_transaction
    for prompt in prompts:
        captured: list[list[tuple]] = []

        def wrapped(
            model_, system_, state_, controller_, lock_, rng_, receipt_,
            _sink=captured, **kw,
        ):
            recorder = _RecordingSystem(system_)
            out = original(
                model_, recorder, state_, controller_, lock_, rng_, receipt_, **kw
            )
            if out is not None:
                _sink.append(list(recorder.calls))
            return out

        sampler_module._attempt_path_transaction = wrapped
        try:
            context = build_prompt_context(
                prompt, config=config, control=control, linker_bridge_atoms=1
            )
            rng = np.random.default_rng(SEED)
            receipt = SamplingReceipt()
            for _ in range(args.attempts_per_drug):
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
                if len(captured) >= args.transactions_per_drug:
                    break
        finally:
            sampler_module._attempt_path_transaction = original

        if not captured:
            rows.append({"drug": prompt.drug_name, "transactions_captured": 0})
            print(f"{prompt.drug_name:14s} no transaction captured", flush=True)
            continue

        calls = captured[0]
        constituents = []
        score_rng = np.random.default_rng(SEED + 1)
        for source, rule_name, _action, successor in calls:
            target = molecular_graph_to_smiles(successor)
            if not target:
                continue
            if args.rate_only:
                families, successors = _draw_census(
                    model, source, score_rng, args.rate_draws, system
                )
                constituents.append({
                    "rule": rule_name,
                    "measured_family_rate": families.get(rule_name, 0) / args.rate_draws,
                    "family_draws": families.get(rule_name, 0),
                    "rate_draws": args.rate_draws,
                    "distinct_family_successors": len(successors.get(rule_name, ())),
                    "hit_in_rate_phase": any(
                        target in s for s in successors.values()
                    ),
                    "family_census": dict(families.most_common(8)),
                })
            else:
                constituents.append(
                    _score_constituent(
                        model, system, source, rule_name, target, score_rng,
                        rate_draws=args.rate_draws, cap=args.cap,
                    )
                )
        rows.append({
            "drug": prompt.drug_name,
            "transactions_captured": len(captured),
            "constituents": constituents,
        })
        summary = " | ".join(
            f"{c['rule']}:{c.get('verdict', 'rate-only')}"
            f" p={c['measured_family_rate']:.4f}"
            for c in constituents
        )
        print(f"{prompt.drug_name:14s} {summary}", flush=True)

    payload = {
        "schema": "compose_fragment_path_composite_scoring_v1",
        "question": (
            "is each constituent of the v3 atom_insert + bond_reroute composite "
            "rankable by the proposal law, at a budget derived from a measured "
            "per-draw rate rather than a guessed ladder"
        ),
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
        },
        "mode": "rate-only" if args.rate_only else "scored",
        "rate_draws": args.rate_draws,
        "cap": args.cap,
        "budget_rule": (
            "N* = ceil(log(0.05)/log(1-r)) with r = measured family rate divided "
            "by the number of distinct successors that family produced; the "
            "uniform-within-family step is an APPROXIMATION and is conservative "
            "where the family law concentrates on the target"
        ),
        "comparison": "on the successor molecule, never on the action object",
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
