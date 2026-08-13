"""ORACLE-ACCOUNTING HARNESS STRESS TEST.

**This is not a GraphGA run and produces no GraphGA result.** The upstream GB-GA
crossover/mutation code is not vendored in this repository and nothing external
was installed. What runs here is a GA-*shaped*, deliberately adversarial
candidate stream -- BRICS fragment recombination, survivor rescores, alternate
SMILES spellings and malformed strings over held-in sources -- whose only job is
to drive the shared accountant hard enough to prove the plumbing:

* the frozen COMPOSE objective is reached through the wrapper and nowhere else;
* canonicalization collapses equivalent SMILES spellings to one key;
* duplicates charge ``oracle_requests`` and not
  ``unique_valid_canonical_evaluations``;
* a cache reduces ``evaluator_calls`` and cannot erase a wasteful request;
* invalid candidates charge ``oracle_requests`` and are counted as failures;
* every counting convention terminates a run at its declared budget.

An accountant that has never been shown an adversarial stream has not been
tested, which is why the stream is built to be hostile rather than realistic.

Artifact status: ``SMOKE_HELD_IN``. It is an instrument check on the harness and
can never be cited for or against any scientific claim, about GraphGA or
anything else.

Before any claim-bearing GraphGA comparison, three things are still required:
the actual upstream implementation vendored; the production RDKit pin 2024.3.5,
or an explicitly isolated environment whose canonicalization is reconciled
against it; and the real algorithm terminating against every counter.

    python scripts/oracle_accounting_stress_test.py --sources 5
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import rdkit
from rdkit import Chem
from rdkit.Chem import BRICS, Crippen, QED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.oracle_accounting import (  # noqa: E402
    OracleAccountant,
    silence_rdkit,
)

COHORT = ROOT / "diagnostics" / "retarget_calibration_cohort.json"
NORMALIZERS = ROOT / "diagnostics" / "retarget_goal_language_normalizers.json"
OUTPUT = (
    ROOT / "diagnostics" / "baselines" / "oracle_accounting_harness_stress_test.json"
)

#: Deliberately malformed strings mixed into the stream so the invalid path is
#: exercised rather than assumed.
MALFORMED = ["C1CC", "not_a_molecule", "", "[C@@H](", "CC(((C"]


def load_sources(count: int) -> tuple[list[str], str]:
    cohort = json.loads(COHORT.read_text())
    if cohort["pool"] != "held-in training sources only":
        raise SystemExit(f"refusing a non-held-in cohort: {cohort['pool']!r}")
    smiles = [row["source"] for row in cohort["sources"][:count]]
    return smiles, cohort["cohort_sha256"]


def developability_objective() -> tuple[object, dict]:
    """Frozen target-free goal: QED floor and cLogP box, in held-in IQR units.

    This is the goal language the main lane calibrated on, so the smoke calls
    the same normalizers the comparison will use rather than an ad-hoc score.
    """
    normalizers = json.loads(NORMALIZERS.read_text())["normalizers"]
    qed_scale = normalizers["qed"]["iqr"]
    clogp_scale = normalizers["clogp"]["iqr"]
    # Held-in medians, used only as the box centre for this instrument check.
    # No threshold is selected here and none is implied.
    qed_floor = normalizers["qed"]["median"]
    clogp_low = normalizers["clogp"]["p25"]
    clogp_high = normalizers["clogp"]["p75"]

    def evaluate(canonical: str) -> float:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:  # pragma: no cover - accountant filters these
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
        "note": "no threshold is selected by this smoke; the box is the held-in IQR",
    }


def ga_shaped_stream(sources: list[str], rng: random.Random, length: int) -> list[str]:
    """A GA-shaped candidate stream. NOT GB-GA -- see the module docstring."""
    fragments: set[str] = set()
    for smiles in sources:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            fragments.update(BRICS.BRICSDecompose(molecule))

    offspring: list[str] = []
    if fragments:
        builder = BRICS.BRICSBuild(
            [Chem.MolFromSmiles(f) for f in sorted(fragments)],
            scrambleReagents=False,
        )
        for _ in range(length):
            try:
                child = next(builder)
            except StopIteration:
                break
            except Exception:  # BRICSBuild can raise on odd fragment pairs
                continue
            offspring.append(Chem.MolToSmiles(child))

    stream: list[str] = []
    for index in range(length):
        roll = rng.random()
        if roll < 0.30 and offspring:
            stream.append(rng.choice(offspring))          # recombination
        elif roll < 0.55:
            stream.append(rng.choice(sources))            # survivor rescore
        elif roll < 0.75:
            # Same molecule, different spelling: the canonicalization test.
            molecule = Chem.MolFromSmiles(rng.choice(sources))
            stream.append(Chem.MolToSmiles(molecule, doRandom=True, canonical=False))
        elif roll < 0.85:
            stream.append(rng.choice(MALFORMED))          # failed proposal
        else:
            stream.append(rng.choice(offspring or sources))
        del index
    return stream


def run(
    sources: list[str],
    stream: list[str],
    budget: int,
    counter: str,
    *,
    cache: bool = True,
) -> dict:
    evaluate, goal = developability_objective()
    calls: list[str] = []

    def counted(canonical: str) -> float:
        calls.append(canonical)
        return evaluate(canonical)

    accountant = OracleAccountant(
        evaluate=counted, budget=budget, budget_counter=counter, cache=cache
    )
    started = time.perf_counter()
    accountant.score_many(stream)
    elapsed = time.perf_counter() - started

    manifest = accountant.manifest()
    manifest["goal"] = goal
    manifest["wall_seconds"] = round(elapsed, 4)
    manifest["candidates_offered"] = len(stream)
    manifest["underlying_evaluator_invocations"] = len(calls)
    manifest["evaluator_saw_only_canonical_input"] = all(
        Chem.MolToSmiles(Chem.MolFromSmiles(c)) == c for c in calls
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=5, help="held-in sources, 3-5")
    parser.add_argument("--stream", type=int, default=400, help="candidates offered")
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not 3 <= args.sources <= 5:
        raise SystemExit("this smoke is specified for 3-5 held-in sources")

    silence_rdkit()
    sources, cohort_sha = load_sources(args.sources)
    rng = random.Random(args.seed)
    stream = ga_shaped_stream(sources, rng, args.stream)

    # The same stream under every counting convention. The point of the stress
    # test is that these runs disagree in a predictable direction.
    by_counter = {
        counter: run(sources, stream, args.budget, counter)
        for counter in ("unique_valid_canonical_evaluations", "oracle_requests")
    }
    # A cache-disabled run isolates the third counter: the same stream must move
    # evaluator_calls and leave the other two alone.
    uncached = run(
        sources,
        stream,
        args.budget,
        "unique_valid_canonical_evaluations",
        cache=False,
    )

    unique_run = by_counter["unique_valid_canonical_evaluations"]["counters"]
    demand_run = by_counter["oracle_requests"]["counters"]
    checks = {
        "canonicalization_collapses_spellings": unique_run["duplicate_requests"] > 0,
        "duplicates_charge_requests_not_unique_evaluations": (
            unique_run["oracle_requests"]
            > unique_run["unique_valid_canonical_evaluations"]
        ),
        "invalids_counted_and_never_reach_the_evaluator": (
            unique_run["failed_proposals"] > 0
            and unique_run["evaluator_calls"]
            == unique_run["unique_valid_canonical_evaluations"]
        ),
        "caching_reduces_evaluator_calls": (
            uncached["counters"]["evaluator_calls"] > unique_run["evaluator_calls"]
        ),
        "caching_cannot_erase_a_wasteful_request": (
            uncached["counters"]["oracle_requests"] == unique_run["oracle_requests"]
            and uncached["counters"]["unique_valid_canonical_evaluations"]
            == unique_run["unique_valid_canonical_evaluations"]
        ),
        "evaluator_only_ever_saw_canonical_smiles": (
            by_counter["unique_valid_canonical_evaluations"][
                "evaluator_saw_only_canonical_input"
            ]
        ),
        "every_budget_convention_terminates": all(
            run_report["budget_exhausted"] for run_report in by_counter.values()
        ),
        "conventions_disagree_as_predicted": (
            demand_run["unique_valid_canonical_evaluations"]
            < unique_run["unique_valid_canonical_evaluations"]
        ),
        "counter_identity_holds_in_every_run": all(
            all(report["invariants"].values())
            for report in (*by_counter.values(), uncached)
        ),
    }

    report = {
        "schema": "compose.baselines.oracle_accounting_harness_stress_test",
        "title": "ORACLE-ACCOUNTING HARNESS STRESS TEST",
        "artifact_status": "SMOKE_HELD_IN",
        "THIS_IS_NOT_A_GRAPHGA_RESULT": (
            "Upstream GB-GA was never vendored and nothing external was "
            "installed. The candidate stream is GA-SHAPED and adversarial by "
            "construction (BRICS recombination, survivor rescores, alternate "
            "SMILES spellings, malformed strings). It exists only to drive the "
            "accountant. Do not cite this as a GraphGA smoke, a GraphGA result, "
            "or evidence about GraphGA's behaviour."
        ),
        "purpose": (
            "Instrument check on the frozen three-counter oracle accounting: "
            "oracle wrapper, canonicalization, duplicate treatment, cache "
            "semantics, every counting convention, budget termination."
        ),
        "required_before_any_claim_bearing_graphga_comparison": [
            "the actual upstream GB-GA implementation vendored",
            "the production RDKit pin 2024.3.5, or an explicitly isolated "
            "environment whose canonicalization is reconciled against it",
            "the real algorithm terminating against every counter",
        ],
        "held_out_data_opened": False,
        "sources": {
            "cohort": "diagnostics/retarget_calibration_cohort.json",
            "cohort_sha256": cohort_sha,
            "pool": "held-in training sources only",
            "count": len(sources),
            "smiles": sources,
        },
        "seed": args.seed,
        "determinism": (
            "Counters, checks and verdict are deterministic under a fixed seed "
            "and rdkit version; only wall_seconds varies between runs, so the "
            "artifact is not byte-identical across reruns by design."
        ),
        "environment": {
            "rdkit": rdkit.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "production_pin": "rdkit 2024.3.5",
            "pin_matches_production": rdkit.__version__ == "2024.3.5",
            "warning": (
                "If pin_matches_production is false this run is an instrument "
                "check only and its canonical keys must not be reused for any "
                "claim-bearing comparison; see CLAUDE.md on catalog drift."
            ),
        },
        "runs": by_counter,
        "cache_disabled_control_run": uncached,
        "checks": checks,
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
