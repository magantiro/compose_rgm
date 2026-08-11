"""Decision-state coverage of the h_phi teacher probe.

The unit is the DECISION STATE, not the pair. A pair's aggregate recovery rate
is coarse: one pair can hold both completely uninformative states and the single
branching state that decides everything.

A decision state teaches the recovery head only when its candidates DISAGREE:

    contrastive    at least one candidate recovers and at least one does not
    all-positive   every candidate recovers   -- no ranking signal
    all-negative   no candidate recovers      -- no ranking signal

Homogeneous states are not useless: they teach calibration, and keep h_phi from
learning that there is always a clever escape. But contrastive states are the
ones that teach the ranking that produced the rescues.

The headline number is the last one: how often does a state exist where greedy
and some alternative have DIFFERENT eventual recoverability? Those are literally
the examples h_phi exists to learn.

Reads the probe artifact only. Measures; decides nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def decision_states(pair: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Group a pair's labels into decision states.

    Every candidate labelled at one decision shares the same `remaining`, and
    `remaining` strictly decreases along the trajectory, so it identifies the
    decision step within a pair without needing an explicit index.
    """

    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in pair.get("label_rows", []):
        grouped[int(row["remaining"])].append(row)
    return [grouped[k] for k in sorted(grouped, reverse=True)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()

    probe = json.loads(args.probe.read_text())
    pairs = probe["per_pair"]
    states = [s for pair in pairs for s in decision_states(pair)]
    if not states:
        raise SystemExit("no decision states in the probe artifact")

    kinds = collections.Counter()
    contrastive_positive: list[int] = []
    contrastive_negative: list[int] = []
    greedy_differs = 0
    greedy_worse = 0
    greedy_missing = 0

    for candidates in states:
        recoveries = [bool(c["recovery"]) for c in candidates]
        if all(recoveries):
            kinds["all_positive"] += 1
        elif not any(recoveries):
            kinds["all_negative"] += 1
        else:
            kinds["contrastive"] += 1
            contrastive_positive.append(sum(recoveries))
            contrastive_negative.append(len(recoveries) - sum(recoveries))

        greedy = [c for c in candidates if c["is_greedy_action"]]
        if not greedy:
            greedy_missing += 1
            continue
        greedy_recovers = bool(greedy[0]["recovery"])
        best_alternative = max(
            (bool(c["recovery"]) for c in candidates if not c["is_greedy_action"]),
            default=False)
        if greedy_recovers != best_alternative:
            greedy_differs += 1
            if best_alternative and not greedy_recovers:
                greedy_worse += 1

    total = len(states)
    labels = sum(len(s) for s in states)
    non_greedy = sum(1 for s in states for c in s if not c["is_greedy_action"])
    calls = sum(p["kernel_calls"] for p in pairs)

    print(f"DECISION STATES {total} over {len(pairs)} pairs, {labels} labels")
    for kind in ("contrastive", "all_positive", "all_negative"):
        print(f"  {kind:14} {kinds[kind]:4d}  ({kinds[kind]/total:5.1%})")
    if contrastive_positive:
        import statistics
        print(f"\n  within contrastive states: median {statistics.median(contrastive_positive):.1f} "
              f"positive vs {statistics.median(contrastive_negative):.1f} negative candidates")

    print(f"\n  non-greedy candidate fraction  {non_greedy}/{labels} "
          f"({non_greedy/labels:.1%})")
    print(f"  kernel calls {calls:,}; labels per call {labels/max(calls,1):.2f}")
    print(f"  enumeration cache hits {sum(p['cache_hits'] for p in pairs):,}; "
          f"teacher-value cache hits {sum(p['teacher_cache_hits'] for p in pairs):,}")

    by_horizon = collections.defaultdict(lambda: [0, 0])
    for pair in pairs:
        cell = by_horizon[pair["steps"]]
        cell[0] += pair["cache_hits"] + pair["teacher_cache_hits"]
        cell[1] += pair["kernel_calls"]
    print("  cache reuse by verified horizon (hits per kernel call):")
    for horizon in sorted(by_horizon):
        hits, kcalls = by_horizon[horizon]
        print(f"    {horizon} steps: {hits/max(kcalls,1):.2f}")

    print(f"\n  THE NUMBER THAT MATTERS")
    print(f"  states where greedy and the best alternative differ in eventual "
          f"recoverability: {greedy_differs}/{total} ({greedy_differs/total:.1%})")
    print(f"    of which greedy is the one that FAILS: {greedy_worse} "
          f"({greedy_worse/total:.1%} of all states)")
    if greedy_missing:
        print(f"    ({greedy_missing} states had no greedy candidate recorded)")

    print(f"\n  reading: contrastive states teach the recovery ranking; "
          f"homogeneous ones teach calibration. Oversample the former, keep a "
          f"calibration sample of the latter. Per-pair prevalence is the wrong "
          f"unit and is not used here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
