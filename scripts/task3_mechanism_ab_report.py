#!/usr/bin/env python
"""Report the seeded mechanism A/B: paired official HV, then why.

DEVELOPMENT-ONLY. The initialization is seeded and departs from the benchmark's
random-120, so nothing here is Task 3 performance.

Two questions, in order, because answering the second first is how a lucky point
gets mistaken for a mechanism:

1. PAIRED OFFICIAL-HV IMPROVEMENT, which is the only criterion the A/B is judged
   by. Paired because both arms of a seed start from the identical frozen
   archive, so the seed-to-seed variation -- which is large on this task --
   cancels in the difference.

2. WHY, from artifacts only. Recompute each arm's lift without its single
   highest-JNK3 new molecule, and count how many of its new front members are
   individually load-bearing. One lucky point and broad Pareto improvement both
   count for official HV, but only the second supports the adaptive-navigation
   story, and they are trivial to confuse if nobody looks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.benchmark.molleo_task3 import _pareto_mask, hypervolume_qmc  # noqa: E402

FIXTURE = Path("artifacts/benchmarks/task3_mechanism_archives_v1/archives.json")
ARMS = ("adaptive-region", "fixed-scalarization")
#: Below this a paired difference is noise on this benchmark, not a result.
TIE_BAND = 0.001


def load_records(logs: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for path in logs:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith('{"arm"'):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row["arm"], row["seed"])
                if key not in seen:          # a rerun supersedes nothing silently
                    seen.add(key)
                    rows.append(row)
    return rows


def why(seed: int, arm: str, ledger: Path, fixture: dict) -> dict | None:
    """One lucky point, or many? Needs the arm's ledger; skipped if absent."""

    if not ledger.exists():
        return None
    seeded = np.asarray(list(fixture["archives"][str(seed)]["molecules"].values()))
    new = np.asarray([json.loads(line)["v"] for line in
                      ledger.read_text().splitlines() if line.strip()])
    base = hypervolume_qmc(seeded, log2_samples=17)
    after = hypervolume_qmc(np.vstack([seeded, new]), log2_samples=17)
    drop = int(np.argmax(new[:, 1]))
    without = hypervolume_qmc(np.vstack([seeded, np.delete(new, drop, axis=0)]),
                              log2_samples=17)
    lift = after - base
    combined = np.vstack([seeded, new])
    mask = _pareto_mask(combined)
    new_front = [i for i in np.flatnonzero(mask) if i >= len(seeded)]
    carrying = sum(
        1 for i in new_front[:120]
        if after - hypervolume_qmc(np.delete(combined, i, axis=0),
                                   log2_samples=14) > 0.001 * after)
    return {
        "lift": lift,
        "lift_without_best_point": without - base,
        "share_from_best_point": (lift - (without - base)) / lift if lift else 0.0,
        "new_on_front": len(new_front),
        "new_front_members_load_bearing": carrying,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--ledgers", type=Path, default=Path("/tmp/ab_ledgers"))
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_rtheta_mechanism_ab.json"))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    rows = load_records(args.logs)
    by = {(r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows
                    if all((a, r["seed"]) in by for a in ARMS)})
    if not seeds:
        raise SystemExit("no complete pairs found")

    print("PAIRED OFFICIAL HYPERVOLUME  (development-only; not Task 3 performance)")
    print(f"{'seed':>6}{'adaptive':>11}{'fixed':>10}{'difference':>13}")
    differences = {}
    for seed in seeds:
        lift = {a: by[(a, seed)]["hv_after"] - by[(a, seed)]["hv_seeded_only"]
                for a in ARMS}
        differences[seed] = lift["adaptive-region"] - lift["fixed-scalarization"]
        print(f"{seed:>6}{lift['adaptive-region']:>+11.4f}"
              f"{lift['fixed-scalarization']:>+10.4f}{differences[seed]:>+13.4f}")

    values = list(differences.values())
    wins = sum(1 for d in values if d > TIE_BAND)
    losses = sum(1 for d in values if d < -TIE_BAND)
    print(f"\nmean paired difference {statistics.fmean(values):+.4f} over "
          f"{len(values)} seeds | adaptive ahead {wins}, behind {losses}, "
          f"tied {len(values) - wins - losses}")

    print("\nWHY -- one lucky point, or broad improvement?")
    print(f"{'arm/seed':<32}{'lift':>9}{'w/o best':>10}{'share':>8}"
          f"{'on front':>10}{'load-bearing':>14}")
    reasons = {}
    for seed in seeds:
        for arm in ARMS:
            detail = why(seed, arm, args.ledgers / f"{arm}_seed{seed}.jsonl",
                         fixture)
            if detail is None:
                continue
            reasons[f"{arm}_seed{seed}"] = detail
            print(f"{arm + '/' + str(seed):<32}{detail['lift']:>+9.4f}"
                  f"{detail['lift_without_best_point']:>+10.4f}"
                  f"{detail['share_from_best_point']:>8.1%}"
                  f"{detail['new_on_front']:>10}"
                  f"{detail['new_front_members_load_bearing']:>14}")

    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    report["STATUS"] = "DEVELOPMENT-ONLY -- NOT TASK 3 PERFORMANCE (seeded init)"
    report["results"] = rows
    report["paired_official_hv"] = {
        "per_seed_difference": differences,
        "mean_paired_difference": statistics.fmean(values),
        "seeds": len(values), "adaptive_ahead": wins, "adaptive_behind": losses,
        "tied_within_band": len(values) - wins - losses, "tie_band": TIE_BAND,
    }
    report["why"] = reasons
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
