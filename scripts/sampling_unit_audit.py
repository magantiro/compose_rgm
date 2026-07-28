"""Resolve the atomic sampling semantics of the production training draw, empirically.

The exposure horizon must be derived from the unit that actually contributes gradient, not from
whichever artifact count is easiest to quote. This script answers, against the REAL sampler code
(no reimplementation):

  1. does one draw select a trace or a transition?
  2. do all steps of the drawn trace contribute loss, or exactly one?
  3. how are long traces weighted relative to short ones?
  4. record-level exposure  E_rec = S * B * w_layer / N_records(layer)
  5. transition-level exposure E_tr = expected draws landing on a given transition

It drives `_sample_tracelet_progress` (the real function `FactorizedMarkDataset.__getitem__` calls)
over real records from all three layers and measures the realized landing distribution, separating
SAMPLING frequency from EXPECTED GRADIENT MASS -- the family stratification deliberately oversamples
rare families and divides the excess back out through an importance weight, so the two differ.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts \
        python scripts/sampling_unit_audit.py --draws 40000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import exp
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.analogue_prior import build_analogue_prior_records
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.tracelet_conditional import _sample_tracelet_progress

# Production draw parameters (schedcheck manifest.training.json, run 48af1bd).
LATE_TIME_FRACTION = 0.5
OPERATIONAL_HORIZON = 16.0
PROGRESS_STRATIFICATION_FRACTION = 0.5
BATCH_SIZE = 64

# Owner-locked first-run layer weights (never inferred from artifact size).
LAYER_WEIGHTS = {"general_corruption": 0.40, "cycle_operations": 0.25, "mmp_analogue": 0.35}

# Full-corpus census from the validated Stage 1 build (edit_precompile_v1, train partition).
CENSUS = {
    "general_corruption": {"records": 74_949, "transitions": 218_641},
    "cycle_operations": {"records": 255_168, "transitions": 255_168},
    "mmp_analogue": {"records": 363_456, "transitions": 2_142_591},
}


def _draw_time(rng: np.random.Generator) -> float:
    """The exact time law from FactorizedMarkDataset.__getitem__."""
    if rng.random() < LATE_TIME_FRACTION:
        return 1.0 - exp(-float(rng.uniform(0.0, OPERATIONAL_HORIZON)))
    return float(rng.uniform(0.01, 0.99))


def measure_layer(records, draws: int, seed: int) -> dict:
    """Monte-Carlo the real progress sampler over a layer's records.

    Returns realized landing frequencies and expected importance weight per family, plus the
    terminal (no-jump) share -- the quantities that convert a trace draw into transition exposure.
    """
    rng = np.random.default_rng(seed)
    landings: Counter[str] = Counter()
    weight_sum: dict[str, float] = defaultdict(float)
    path_lengths: Counter[int] = Counter()
    for _ in range(draws):
        record = records[int(rng.integers(len(records)))]
        path = record.path
        path_lengths[path.path_length] += 1
        progress, importance = _sample_tracelet_progress(
            path,
            time=_draw_time(rng),
            rng=rng,
            stratification_fraction=PROGRESS_STRATIFICATION_FRACTION,
        )
        family = (
            path.trace.steps[progress].rule_name
            if progress < path.path_length
            else "<TERMINAL>"
        )
        landings[family] += 1
        weight_sum[family] += float(importance)
    mean_len = sum(k * v for k, v in path_lengths.items()) / max(1, sum(path_lengths.values()))
    return {
        "draws": draws,
        "mean_path_length": round(mean_len, 4),
        "terminal_share": round(landings["<TERMINAL>"] / draws, 5),
        "landing_share": {k: round(v / draws, 5) for k, v in landings.most_common()},
        "mean_importance_weight": {
            k: round(weight_sum[k] / landings[k], 5) for k in landings if landings[k]
        },
        "gradient_share": {
            k: round(weight_sum[k] / sum(weight_sum.values()), 5)
            for k in sorted(weight_sum, key=lambda x: -weight_sum[x])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--sources", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--pool",
        default="diagnostics/composition/analogue_trace_pool.jsonl",
        help="local verified analogue-trace sample (structure identical to the Modal pool)",
    )
    parser.add_argument("--out", default="diagnostics/coherence/sampling_unit_audit.json")
    args = parser.parse_args()

    # Sources come from the verified analogue pool's own targets: real broad-organic corpus molecules
    # under the same standardization, so the fixture needs no Modal-hosted corpus file.
    pool_rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    smiles = tuple(dict.fromkeys(row["target_smiles"] for row in pool_rows))[: args.sources]

    corruption, _ = build_corrupted_prior_records(
        smiles, n_slots=40, depth_max=5, seed=args.seed + 7, vocabulary=ORGANIC_VOCABULARY
    )
    cycle, _ = build_cycle_op_records(smiles, n_slots=40, seed=args.seed + 8)
    mmp = build_analogue_prior_records(args.pool)

    layers = {
        "general_corruption": corruption,
        "cycle_operations": cycle,
        "mmp_analogue": mmp,
    }
    result = {"fixture_records": {k: len(v) for k, v in layers.items()}, "layers": {}}
    for i, (name, records) in enumerate(layers.items()):
        if not records:
            raise SystemExit(f"layer {name} produced no fixture records")
        result["layers"][name] = measure_layer(records, args.draws, args.seed + 100 * i)

    # Convert to exposure at BOTH levels for the candidate horizons.
    result["exposure"] = {}
    for steps in (3000, 8000, 9000, 10000):
        total_draws = steps * BATCH_SIZE
        per_layer = {}
        for name, weights in LAYER_WEIGHTS.items():
            layer_draws = total_draws * weights
            n_rec = CENSUS[name]["records"]
            n_tr = CENSUS[name]["transitions"]
            jump = 1.0 - result["layers"][name]["terminal_share"]
            per_layer[name] = {
                "weight": weights,
                "draws": round(layer_draws),
                "record_exposure": round(layer_draws / n_rec, 4),
                "transition_exposure": round(layer_draws * jump / n_tr, 4),
                "jump_share": round(jump, 5),
            }
        result["exposure"][f"{steps}_steps"] = per_layer

    # A layer weight governs TRACE draws. Because a draw may land on the terminal (no-jump) position,
    # and single-step traces spend proportionally more mass there, the realized share of teacher
    # TRANSITIONS differs from the configured weight. Report both so the mixture is auditable.
    jump_mass = {
        name: LAYER_WEIGHTS[name] * (1.0 - result["layers"][name]["terminal_share"])
        for name in LAYER_WEIGHTS
    }
    total_jump = sum(jump_mass.values())
    result["realized_mixture"] = {
        "terminal_share_of_all_draws": round(1.0 - total_jump, 5),
        "teacher_transition_share_of_all_draws": round(total_jump, 5),
        "configured_trace_weight": dict(LAYER_WEIGHTS),
        "realized_transition_weight": {
            name: round(mass / total_jump, 5) for name, mass in jump_mass.items()
        },
    }
    result["census_consistency"] = {
        name: {
            "fixture_mean_path_length": result["layers"][name]["mean_path_length"],
            "census_mean_path_length": round(
                CENSUS[name]["transitions"] / CENSUS[name]["records"], 4
            ),
        }
        for name in CENSUS
    }

    result["semantics"] = {
        "draw_unit": "ONE trace (PathRecord), then ONE progress position inside it",
        "steps_contributing_loss_per_draw": 1,
        "terminal_position": "path_length is a legal landing -> a no-jump (rate-only) example",
        "long_trace_weighting": (
            "a trace is drawn with probability independent of its length, so each of its transitions "
            "receives roughly 1/path_length of that probability: long MMP traces dilute per-transition "
            "exposure relative to single-step cycle records"
        ),
        "family_stratification": (
            "with probability 0.5 the position is drawn family-uniformly over the families present on "
            "THAT trace (terminal counts as a family); the returned importance weight divides the "
            "oversampling back out, so the gradient EXPECTATION remains the Generator Matching law -- "
            "stratification buys coverage/variance, not expected family mass"
        ),
        "parameters": {
            "late_time_fraction": LATE_TIME_FRACTION,
            "operational_horizon": OPERATIONAL_HORIZON,
            "progress_stratification_fraction": PROGRESS_STRATIFICATION_FRACTION,
            "batch_size": BATCH_SIZE,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
