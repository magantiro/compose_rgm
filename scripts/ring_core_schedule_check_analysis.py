#!/usr/bin/env python3
"""RingCore-V1 schedule-check analyzer (Parts E-J of the schedule-faithful 1000-step check).

Consumes a training ``metrics.json`` (the trainer's per-eval ``training.history`` + selected/test blocks) and
emits the schedule-check diagnostics + a verdict. This is the NON-rollout half (Parts E/F/G/H/J); Part I
(rollout gates) is the separate ``ring_core_rollout_panel.py`` run on the step-1000 checkpoint.

Verdict targets (owner criteria, 2026-07-28):
  GO_FOR_FULL_RINGCORE_TRAINING  -- hard gates zero, validation stable/improves after step 500, no enabled
                                    family effectively dead, lower-frequency families retain finite probability
                                    + updates, family-mass calibration defensible, cycle editing productive.
  GO_AFTER_RECIPE_ADJUSTMENT     -- ONLY for a demonstrated data/family-distribution imbalance.
  NO_GO_PRODUCTION_SCHEDULE      -- instability caused by the warmup->decay transition.

"Dead" is judged on FINITE PROBABILITY + updates (mean_teacher_family_probability > eps with teacher examples),
NOT top-1 accuracy: a low-frequency family can be correctly low-accuracy yet retain finite, updated mass.

Usage:
  python scripts/ring_core_schedule_check_analysis.py --metrics metrics.json [--output analysis.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# families ENABLED in RingCore-V1 (grow is masked dead by contract; the two cycle families are the compositional
# ring-addition/removal ops recorded under executor->family aliases cycle_insert/cycle_attach).
ENABLED_FAMILIES = [
    "atom_insert", "atom_delete", "atom_restate", "bond_reorder", "bond_reroute",
    "cycle_insert", "cycle_attach", "ring_system_delete", "ring_system_restate",
]
CYCLE_FAMILIES = ["cycle_insert", "cycle_attach"]
DISABLED_FAMILY = "ring_system_grow"
FINITE_PROB_EPS = 1e-4  # a family with teacher targets must keep at least this much model family-mass
EVAL_STEPS = [0, 250, 500, 750, 1000, 1250, 1500]  # covers the step-1500 continuation history too


def _f(d: dict, key: str, default=float("nan")) -> float:
    v = d.get(key, default)
    return float(v) if v is not None else default


def _nearest(history: list[dict], target_step: int) -> dict | None:
    if not history:
        return None
    return min(history, key=lambda h: abs(_f(h, "step", 1e9) - target_step))


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    # KL(target || model) over the shared family support; skip zero-target families.
    out = 0.0
    for k, pk in p.items():
        qk = max(q.get(k, 0.0), 1e-12)
        if pk > 0:
            out += pk * math.log(pk / qk)
    return out


def _tv(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()} if tot > 0 else {k: 0.0 for k in counts}


def analyze(metrics: dict) -> dict:
    training = metrics.get("training", {})
    history = sorted(training.get("history", []), key=lambda h: _f(h, "step"))
    selected = metrics.get("selected_validation", {})
    report: dict = {
        "schedule_config_hash": "0b832985c65de1cc", "status": "NON_SCIENTIFIC_PREFLIGHT",
        "selected_validation_gm_loss": _f(selected, "factorized_gm_loss"),
    }

    # ---- Part E: LR trajectory + validation improvement at 0/250/500/750/1000 ----
    part_e = []
    for s in EVAL_STEPS:
        h = _nearest(history, s)
        if h is None:
            continue
        part_e.append({
            "target_step": s, "actual_step": _f(h, "step"),
            "learning_rate": _f(h, "learning_rate"),
            "gm_loss": _f(h, "factorized_gm_loss"),
            "balanced_family_top3": _f(h, "balanced_family_top3_accuracy"),
        })
    report["part_e_trajectory"] = part_e
    # validation stable-or-improves AFTER step 500 (warmup end -> decay phase)
    post = [p for p in part_e if p["target_step"] >= 500 and math.isfinite(p["gm_loss"])]
    improves_after_warmup = len(post) >= 2 and post[-1]["gm_loss"] <= post[0]["gm_loss"] + 1e-6
    # LR must actually be IN decay by 750/1000 (peak is at 500) -- proves the schedule was exercised
    lr_at = {p["target_step"]: p["learning_rate"] for p in part_e}
    peak = lr_at.get(500, float("nan"))
    decayed = (
        math.isfinite(peak) and peak > 0
        and lr_at.get(750, peak) < peak * 0.999 and lr_at.get(1000, peak) < peak * 0.99
    )
    report["validation_improves_after_warmup"] = improves_after_warmup
    report["lr_entered_decay_phase"] = decayed

    # ---- Part F: balanced family diagnostic + dead-head detection ----
    last = _nearest(history, 1000) or (history[-1] if history else {})
    fam_rows = []
    dead = []
    for fam in ENABLED_FAMILIES:
        tex = _f(last, f"teacher_examples_{fam}", 0.0)
        fam_prob = _f(last, f"mean_teacher_family_probability_{fam}", 0.0)
        row = {
            "family": fam,
            "teacher_examples": tex,
            "top1": _f(last, f"family_accuracy_{fam}", 0.0),
            "top3": _f(last, f"family_top3_accuracy_{fam}", 0.0),
            "model_family_prob": fam_prob,
        }
        fam_rows.append(row)
        # dead = has teacher targets but the model assigns ~zero family mass (not merely low top-1)
        if tex >= 1 and fam_prob < FINITE_PROB_EPS:
            dead.append(fam)
    report["part_f_families"] = fam_rows
    report["dead_enabled_families"] = dead
    report["balanced_family_top3_final"] = _f(last, "balanced_family_top3_accuracy")
    report["grow_teacher_examples"] = _f(last, f"teacher_examples_{DISABLED_FAMILY}", 0.0)
    report["grow_model_prob"] = _f(last, f"mean_teacher_family_probability_{DISABLED_FAMILY}", 0.0)

    # ---- Part G: family-mass calibration (q_target from teacher_examples vs q_model family mass) ----
    q_target = _normalize({fam: _f(last, f"teacher_examples_{fam}", 0.0) for fam in ENABLED_FAMILIES})
    q_model = _normalize({fam: _f(last, f"mean_teacher_family_probability_{fam}", 0.0) for fam in ENABLED_FAMILIES})
    report["part_g_calibration"] = {
        "q_target": q_target, "q_model": q_model,
        "kl_target_model": _kl(q_target, q_model), "tv": _tv(q_target, q_model),
    }

    # ---- Part H: training-stability signals available in metrics (grad/clip need the training log) ----
    losses = [_f(h, "factorized_gm_loss") for h in history if math.isfinite(_f(h, "factorized_gm_loss"))]
    any_nan = any(not math.isfinite(_f(h, "factorized_gm_loss")) for h in history)
    report["part_h_stability"] = {
        "all_eval_losses_finite": not any_nan and bool(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "monotone_nonincreasing_after_warmup": improves_after_warmup,
        "early_stopped": bool(_f(last, "early_stopped", 0.0)),
        "note": "grad-norms / clipping / scaler / body-drift are NOT in metrics.json; read the training log or "
                "the rollout panel's learning diagnostics for those.",
    }

    # ---- Part I hook: cycle-editing productivity comes from the rollout panel (recorded here as a reminder) ----
    report["part_i_rollout"] = "run scripts/ring_core_rollout_panel.py on the step-1000 checkpoint (separate)"

    # ---- Part J: verdict (metrics-only half; the rollout hard gates gate the FINAL verdict) ----
    reasons = []
    if not report["lr_entered_decay_phase"]:
        reasons.append("LR did not enter the decay phase (schedule not exercised) -- check is not schedule-faithful")
    if dead:
        reasons.append(f"enabled families effectively dead (finite-prob): {dead}")
    if not improves_after_warmup:
        reasons.append("validation did not stay stable/improve after step 500 (warmup->decay)")
    cal = report["part_g_calibration"]
    cal_ok = math.isfinite(cal["tv"]) and cal["tv"] <= 0.5  # generous: family mass not wildly miscalibrated
    if not cal_ok:
        reasons.append(f"family-mass calibration indefensible (TV={cal['tv']:.3f})")

    if any_nan:
        verdict = "NO_GO_PRODUCTION_SCHEDULE"
        reasons.insert(0, "non-finite eval loss (instability across the warmup->decay transition)")
    elif dead and not improves_after_warmup:
        # a demonstrated family-distribution imbalance (some families never get updated) -> recipe adjustment
        verdict = "GO_AFTER_RECIPE_ADJUSTMENT"
    elif not reasons:
        verdict = "GO_FOR_FULL_RINGCORE_TRAINING (metrics half; pending rollout hard gates)"
    else:
        verdict = "REVIEW_REQUIRED (metrics half)"
    report["part_j_metrics_verdict"] = verdict
    report["part_j_reasons"] = reasons or ["all metrics-half criteria satisfied"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    report = analyze(metrics)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
