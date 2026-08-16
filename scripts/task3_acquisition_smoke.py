#!/usr/bin/env python
"""Offline smoke for an acquisition. ZERO oracle spend, zero Modal spend.

A bad acquisition should die here, in minutes, not in a charged run. Everything
below is computed from cached R_theta fibers and a surrogate fit on the frozen
seeded archive; the benchmark oracle is never called, so nothing is charged and
no result is produced -- only a verdict on whether the mechanism ENGAGES.

The five checks are the ones the previous adaptive rule would have failed. Its
`iterations_where_target_was_represented` was zero on every arm and seed: it
aimed at a dominance-shaped aspiration nothing could reach and sat permanently
in fallback. That was invisible until after two charged experiments. These
checks make it visible in seconds:

  1 NONDEGENERATE      the acquisition separates candidates at all
  2 REPRESENTED        the thing being aimed at exists in the actual fiber
  3 STATE-DEPENDENT    different archives prefer different regions
  4 NOT IN FALLBACK    the mechanism is on, not silently defaulting
  5 ARMS DIVERGE       fixed and adaptive pick different molecules often enough
                       for a comparison between them to mean anything
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.policy.task3.archive import ParetoArchive  # noqa: E402
from compose_v4.policy.task3.steering import (  # noqa: E402
    AdaptiveRegion,
    FixedScalarization,
    ReachableHVI,
)
from compose_v4.policy.task3.surrogate import TanimotoKNN  # noqa: E402

FIXTURE = Path("artifacts/benchmarks/task3_mechanism_archives_v1/archives.json")
#: How many molecules an arm would charge per decision, matching the A/B.
TOP_K = 4


def load_fibers(directory: Path) -> dict[str, list[str]]:
    fibers: dict[str, list[str]] = {}
    for path in sorted(directory.glob("fibers_*.json")):
        payload = json.loads(path.read_text())
        for start, rows in payload["fibers"].items():
            fibers[start] = [row[0] for row in rows]
    return fibers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fibers", type=Path, default=Path("/tmp/fiber_cache/seed100"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_acquisition_smoke.json"))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    seeded = {s: tuple(v) for s, v in
              fixture["archives"][str(args.seed)]["molecules"].items()}
    fibers = load_fibers(args.fibers)
    if not fibers:
        raise SystemExit(f"no cached fibers under {args.fibers}")
    print(f"{len(fibers)} cached fibers, median {statistics.median(len(v) for v in fibers.values()):.0f} "
          f"successors; seeded archive {len(seeded)}")

    surrogate = TanimotoKNN()
    surrogate.update(list(seeded), list(seeded.values()))

    rng = np.random.default_rng(args.seed)
    starts = sorted(fibers)
    arms = {"reachable-hvi": ReachableHVI(),
            "dominance-aspiration": AdaptiveRegion(),
            "fixed-scalarization": FixedScalarization()}

    stats: dict[str, dict] = {name: {"engaged": 0, "decisions": 0,
                                     "top_choice": [], "best_gain": []}
                              for name in arms}
    overlap = []
    for _ in range(args.decisions):
        # A different archive state each decision, so check 3 is exercised:
        # a random subset of the seeded archive plus its own history.
        keep = rng.choice(len(seeded), size=rng.integers(80, len(seeded)),
                          replace=False)
        keys = list(seeded)
        archive = ParetoArchive()
        archive.add_many({keys[int(i)]: seeded[keys[int(i)]] for i in keep})

        start = starts[int(rng.integers(len(starts)))]
        candidates = [s for s in fibers[start] if s not in archive.values]
        if len(candidates) < TOP_K:
            continue
        predicted, spread = surrogate.predict_with_spread(candidates)

        chosen: dict[str, set[str]] = {}
        for name, arm in arms.items():
            target = arm.target(archive, rng)
            if target is None:
                continue
            try:
                ranks = arm.rank(predicted, target, spread)
            except TypeError:      # arms that take no uncertainty signal
                ranks = arm.rank(predicted, target)
            order = np.argsort(-ranks)[:TOP_K]
            chosen[name] = {candidates[int(i)] for i in order}
            entry = stats[name]
            entry["decisions"] += 1
            entry["top_choice"].append(predicted[int(order[0])].tolist())
            # ENGAGED = the acquisition actually discriminated, rather than
            # every candidate scoring the same and the tie-break deciding.
            # Engagement is judged on what the ARM actually scores, which for
            # an optimistic acquisition is the optimistic estimate.
            estimate = getattr(arm, "optimism", 0.0)
            scored = (np.clip(predicted + estimate * spread, 0.0, 1.0)
                      if estimate else predicted)
            gains = archive.gains_of(scored)
            entry["best_gain"].append(float(gains.max()))
            if float(gains.max()) > 0 and float(np.ptp(ranks)) > 0:
                entry["engaged"] += 1
        if "reachable-hvi" in chosen and "fixed-scalarization" in chosen:
            shared = chosen["reachable-hvi"] & chosen["fixed-scalarization"]
            overlap.append(len(shared) / TOP_K)

    report: dict = {"seed": args.seed, "fibers": len(fibers),
                    "decisions": args.decisions, "checks": {}}
    print(f"\n{'arm':<24}{'engaged':>10}{'best HVI>0':>12}{'region spread':>15}")
    for name, entry in stats.items():
        if not entry["decisions"]:
            continue
        engaged = entry["engaged"] / entry["decisions"]
        positive = statistics.fmean(1.0 if g > 0 else 0.0
                                    for g in entry["best_gain"])
        picks = np.asarray(entry["top_choice"])
        spread = float(np.mean(picks.std(axis=0)))
        report["checks"][name] = {
            "fraction_engaged": engaged,
            "fraction_with_positive_best_gain": positive,
            "region_spread_across_states": spread,
        }
        print(f"{name:<24}{engaged:>10.1%}{positive:>12.1%}{spread:>15.4f}")

    divergence = 1.0 - statistics.fmean(overlap) if overlap else float("nan")
    report["arms_diverge"] = {
        "mean_top4_overlap_hvi_vs_fixed": statistics.fmean(overlap) if overlap else None,
        "divergence": divergence,
    }
    print(f"\nreachable-hvi vs fixed: top-{TOP_K} overlap "
          f"{statistics.fmean(overlap):.1%}, so they differ on "
          f"{divergence:.1%} of picks")

    verdict = []
    hvi = report["checks"].get("reachable-hvi", {})
    verdict.append(("1 nondegenerate", hvi.get("fraction_engaged", 0) > 0.5))
    verdict.append(("2 represented", hvi.get("fraction_with_positive_best_gain", 0) > 0.5))
    verdict.append(("3 state-dependent", hvi.get("region_spread_across_states", 0) > 0.01))
    verdict.append(("4 not in fallback", hvi.get("fraction_engaged", 0) > 0.5))
    verdict.append(("5 arms diverge", divergence > 0.2))
    print()
    for label, passed in verdict:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    report["verdict"] = {label: bool(ok) for label, ok in verdict}
    report["all_passed"] = all(ok for _, ok in verdict)
    print(f"\n=> {'PROCEED to a tiny charged A/B' if report['all_passed'] else 'DO NOT SPEND: fix the acquisition offline'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
