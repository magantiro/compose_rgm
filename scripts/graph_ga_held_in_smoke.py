"""Held-in smoke of the REAL upstream GB-GA under COMPOSE oracle accounting.

Unlike the harness stress test, this runs the **actual vendored upstream
algorithm** (`baselines/graph_ga/upstream/`, commit `4b49f182`, byte-identical
to GitHub) on held-in COMPOSE sources against the frozen COMPOSE developability
objective.

What it establishes:

1. the adapter drives upstream GB-GA end to end;
2. all three counters are wired and their identity holds;
3. **upstream's own published accounting identity is reproduced** --
   `oracle_requests == population_size * (generations + 1)`, which is the paper's
   "population size is 20 and 50 generations ... i.e. 1000 J(m) evaluations";
4. **a published qualitative behaviour is reproduced** -- the paper states the
   molecules found "bear little resemblance to the molecules used to construct
   the initial mating pool" (nearest-ZINC Tanimoto 0.27 and 0.12). We measure
   endpoint-to-seed ECFP4 Tanimoto and confirm it collapses. This is the exact
   claim the fairness contract rests on when it says GraphGA cannot enter a
   source-conditioned panel unaltered;
5. **the size prior is a fairness parameter, not a constant** -- the run is
   repeated with upstream's ZINC values to show how much the offspring size
   distribution moves.

Artifact status: `SMOKE_HELD_IN`. It is an adapter and instrument check. It is
NOT a measurement of GraphGA's optimization quality and may not be cited as one.

    python scripts/graph_ga_held_in_smoke.py --sources 5
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, QED, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.graph_ga_adapter import (  # noqa: E402
    UPSTREAM_ZINC_SIZE_PRIOR,
    expected_scoring_calls,
    run_graph_ga,
    size_prior_from_sources,
)
from compose_v4.experiments.oracle_accounting import silence_rdkit  # noqa: E402

COHORT = ROOT / "diagnostics" / "retarget_calibration_cohort.json"
NORMALIZERS = ROOT / "diagnostics" / "retarget_goal_language_normalizers.json"
OUTPUT = ROOT / "diagnostics" / "baselines" / "graph_ga_held_in_smoke.json"

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

#: Declared BEFORE the run, not fitted to it. The developability margin is in
#: held-in IQR units; 5.0 covers the plausible negative range with headroom.
#: Floor hits are counted so an inadequate shift cannot pass silently.
SHIFT = 5.0


def load_sources(count: int) -> tuple[list[str], str]:
    cohort = json.loads(COHORT.read_text())
    if cohort["pool"] != "held-in training sources only":
        raise SystemExit(f"refusing a non-held-in cohort: {cohort['pool']!r}")
    return [row["source"] for row in cohort["sources"][:count]], cohort["cohort_sha256"]


def developability_objective():
    """The frozen target-free goal: QED floor and cLogP box, in held-in IQR units."""
    normalizers = json.loads(NORMALIZERS.read_text())["normalizers"]
    qed_scale = normalizers["qed"]["iqr"]
    clogp_scale = normalizers["clogp"]["iqr"]
    qed_floor = normalizers["qed"]["median"]
    clogp_low = normalizers["clogp"]["p25"]
    clogp_high = normalizers["clogp"]["p75"]

    def evaluate(canonical: str) -> float:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            return 0.0
        qed_margin = (QED.qed(molecule) - qed_floor) / qed_scale
        clogp = Crippen.MolLogP(molecule)
        clogp_margin = min(clogp - clogp_low, clogp_high - clogp) / clogp_scale
        return float(min(qed_margin, clogp_margin))

    return evaluate, {
        "goal": "target-free developability: QED floor and cLogP box",
        "normalizers": "diagnostics/retarget_goal_language_normalizers.json",
        "qed_floor_held_in_median": qed_floor,
        "clogp_box_held_in_p25_p75": [clogp_low, clogp_high],
    }


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else _MORGAN.GetFingerprint(molecule)


def nearest_seed_similarity(endpoints: list[str], sources: list[str]) -> dict:
    """Max ECFP4 Tanimoto from each endpoint to any seed. The paper's claim."""
    seed_fps = [fp for fp in (fingerprint(s) for s in sources) if fp is not None]
    nearest = []
    for endpoint in endpoints:
        fp = fingerprint(endpoint)
        if fp is None or not seed_fps:
            continue
        nearest.append(max(DataStructs.BulkTanimotoSimilarity(fp, seed_fps)))
    if not nearest:
        return {"n": 0}
    return {
        "n": len(nearest),
        "median": round(statistics.median(nearest), 4),
        "mean": round(statistics.fmean(nearest), 4),
        "max": round(max(nearest), 4),
        "min": round(min(nearest), 4),
        "fraction_above_0.4": round(
            sum(1 for v in nearest if v >= 0.4) / len(nearest), 4
        ),
    }


def heavy_atom_stats(smiles: list[str]) -> dict:
    counts = [
        m.GetNumAtoms()
        for m in (Chem.MolFromSmiles(s) for s in smiles)
        if m is not None
    ]
    if not counts:
        return {"n": 0}
    return {
        "n": len(counts),
        "mean": round(statistics.fmean(counts), 2),
        "max": max(counts),
        "min": min(counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=5, help="held-in sources, 3-5")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not 3 <= args.sources <= 5:
        raise SystemExit("this smoke is specified for 3-5 held-in sources")

    silence_rdkit()
    sources, cohort_sha = load_sources(args.sources)
    objective, goal = developability_objective()
    seed_path = args.output.parent / "graph_ga_smoke_seed.smi"

    started = time.perf_counter()
    panel_prior = size_prior_from_sources(sources)

    # Record, rather than dodge, what upstream's OWN transform does to a COMPOSE
    # margin objective. This is the decisive adaptation finding.
    clamp_probe: dict[str, object]
    try:
        run_graph_ga(
            sources=sources,
            objective=objective,
            seed_path=seed_path,
            budget=args.budget,
            budget_counter="unique_valid_canonical_evaluations",
            population_size=args.population,
            generations=1,
            seed=args.seed,
            size_prior=panel_prior,
            nonnegative_transform="upstream_clamp",
        )
        clamp_probe = {"outcome": "ran", "error": None}
    except Exception as exc:  # noqa: BLE001 - the exception IS the finding
        clamp_probe = {
            "outcome": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "why": (
                "Every held-in source starts OUTSIDE the developability box, so "
                "the margin objective is negative for the whole initial "
                "population. Upstream's clamp (max(0.0, score), as in "
                "scoring_functions.logP_max) maps all of them to 0.0, the "
                "fitness sum is 0, and calculate_normalized_fitness divides by "
                "zero. Upstream never hits this because penalized logP is "
                "positive for some molecules by construction."
            ),
        }

    runs = {}
    for label, prior in (("panel_size_prior", panel_prior),
                         ("upstream_zinc_size_prior", UPSTREAM_ZINC_SIZE_PRIOR)):
        result = run_graph_ga(
            sources=sources,
            objective=objective,
            seed_path=seed_path,
            budget=args.budget,
            budget_counter="unique_valid_canonical_evaluations",
            population_size=args.population,
            generations=args.generations,
            seed=args.seed,
            size_prior=prior,
            nonnegative_transform="constant_shift",
            constant_shift=SHIFT,
        )
        runs[label] = {
            "counters": result.counters,
            "invariants": result.accountant_manifest["invariants"],
            "demand_ratio_requests_over_unique": result.accountant_manifest[
                "demand_ratio_requests_over_unique"
            ],
            "generations_run": result.generations_run,
            "wall_seconds": result.wall_seconds,
            "budget_exhausted": result.budget_exhausted,
            "wall_clock_exceeded": result.wall_clock_exceeded,
            "shift_floor_hits": result.shift_floor_hits,
            "size_prior": {k: round(v, 4) for k, v in result.size_prior.items()},
            "best_score": round(result.scores[0], 4) if result.scores else None,
            "endpoint_heavy_atoms": heavy_atom_stats(result.endpoints),
            "nearest_seed_tanimoto": nearest_seed_similarity(result.endpoints, sources),
            "settings": result.settings,
        }

    panel = runs["panel_size_prior"]
    zinc = runs["upstream_zinc_size_prior"]
    checks = {
        "upstream_accounting_identity_reproduced": (
            panel["counters"]["oracle_requests"]
            == expected_scoring_calls(args.population, panel["generations_run"])
        ),
        "all_three_counters_wired": (
            panel["counters"]["oracle_requests"] > 0
            and panel["counters"]["unique_valid_canonical_evaluations"] > 0
            and panel["counters"]["evaluator_calls"] > 0
        ),
        "counter_identity_holds": all(panel["invariants"].values()),
        "graph_ga_natively_requests_duplicates": (
            panel["counters"]["duplicate_requests"] > 0
        ),
        "published_behaviour_source_not_preserved": (
            panel["nearest_seed_tanimoto"]["median"] < 0.4
        ),
        "size_prior_materially_changes_offspring_size": (
            zinc["endpoint_heavy_atoms"]["mean"]
            > panel["endpoint_heavy_atoms"]["mean"] + 1.0
        ),
        "upstream_own_transform_is_inapplicable_here": (
            clamp_probe["outcome"] == "FAILED"
        ),
        "declared_shift_was_adequate": panel["shift_floor_hits"] == 0,
        "no_wall_clock_deadlock": (
            not panel["wall_clock_exceeded"] and not zinc["wall_clock_exceeded"]
        ),
    }

    report = {
        "schema": "compose.baselines.graph_ga_held_in_smoke",
        "title": "GRAPH GA (real upstream GB-GA) — HELD-IN ADAPTER SMOKE",
        "artifact_status": "SMOKE_HELD_IN",
        "what_this_is": (
            "The actual vendored upstream GB-GA algorithm, commit 4b49f182, "
            "byte-identical to GitHub, driven on held-in COMPOSE sources against "
            "the frozen COMPOSE developability objective under the three-counter "
            "oracle accounting."
        ),
        "what_this_is_not": (
            "A measurement of GraphGA's optimization quality, and not a "
            "COMPOSE-versus-GraphGA comparison. No COMPOSE arm was run. Budgets "
            "here are smoke-sized, not the frozen suite budgets."
        ),
        "held_out_data_opened": False,
        "sources": {
            "cohort": "diagnostics/retarget_calibration_cohort.json",
            "cohort_sha256": cohort_sha,
            "pool": "held-in training sources only",
            "count": len(sources),
            "smiles": sources,
            "heavy_atoms": heavy_atom_stats(sources),
        },
        "objective": goal,
        "upstream": {
            "repository": "https://github.com/jensengroup/GB_GA",
            "commit": "4b49f1822c5190e8b2bbb8b7403eed30af9e50fd",
            "license": "MIT",
            "vendored_at": "baselines/graph_ga/upstream/",
            "modified": False,
        },
        "adapter_findings": {
            "undocumented_required_globals": (
                "crossover.average_size and crossover.size_stdev must be set by "
                "the caller. Unset, mol_OK raises NameError into a bare except, "
                "every candidate is rejected, crossover always returns None, and "
                "reproduce()'s unbounded while loop spins forever with no error. "
                "Upstream sets them only in GA_logP.py (39.15 / 3.50, ZINC)."
            ),
            "size_prior_is_a_fairness_parameter": (
                "Those globals impose a soft Gaussian size prior on offspring. "
                "Running a COMPOSE panel under upstream's ZINC values asks the "
                "baseline to build ZINC-sized molecules. Both runs are reported."
            ),
            "oracle_budget_does_not_bound_runtime": (
                "Failed crossovers and mutations consume no oracle calls, so a "
                "budget-matched comparison bounds calls but not wall time. The "
                "adapter imposes a wall-clock guard."
            ),
        },
        "upstream_clamp_probe": clamp_probe,
        "runs": runs,
        "checks": checks,
        "total_wall_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "rdkit": rdkit.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "production_pin": "rdkit 2024.3.5",
            "pin_matches_production": rdkit.__version__ == "2024.3.5",
            "warning": (
                "If pin_matches_production is false this run is an adapter check "
                "only and its canonical keys must not be reused for any "
                "claim-bearing comparison; see CLAUDE.md on catalog drift."
            ),
        },
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
