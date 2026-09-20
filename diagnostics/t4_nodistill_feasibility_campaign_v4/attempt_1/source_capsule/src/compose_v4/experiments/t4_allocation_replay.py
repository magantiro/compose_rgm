"""Would this allocator have spent the budget better, on runs that already happened?

The gate before any new docking call. Thousands of scored T4 transitions exist with their
charged-call indices, so a controller can be replayed against a real search history:
reveal observations in the order they were actually paid for, and at every call ask where
the next one would have gone. Nothing is predicted, no docking label is invented, and no
oracle call is spent.

The honest limit of a replay is that it cannot conjure candidates the historical run never
generated. So this does not ask "what would the controller have found"; it asks the
narrower and still decisive question: among the structural hypotheses the run DID sample,
would the allocator have concentrated its budget on the branch that eventually produced
the win, earlier than the policy that actually ran?

If it cannot do that on the cells where Dynamic already won, it is not ready for JAK2.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compose_v4.control.hypothesis_allocation import HypothesisTree, allocate

SCHEMA_VERSION = "t4_allocation_replay_v1"

# Coarse-to-fine labels are read off fields the archive already records. This is a
# retrospective LABELLING of what each historical program was, not a new representation:
# the structural hypothesis object itself is `structural_subgoal.StructuralSubgoal`.
RING_FAMILIES = ("cycle_close", "cycle_open", "ring_system_restate", "ring_system_delete")


def hypothesis_path(row: dict) -> tuple[str, ...]:
    """The coarse-to-fine hypothesis a historical program realises."""
    counts = row.get("rule_counts") or {}
    created = row.get("net_created") or 0
    deleted = row.get("net_deleted") or 0
    ring_change = row.get("ring_count_change") or 0
    touches_ring = any(counts.get(f) for f in RING_FAMILIES) or ring_change

    if touches_ring and created >= 2:
        family = "ring_construction"
    elif touches_ring and deleted:
        family = "ring_remodelling"
    elif touches_ring:
        family = "ring_restatement"
    elif created >= 2 and deleted >= 2:
        family = "fragment_replacement"
    elif created >= 2:
        family = "fragment_growth"
    elif deleted >= 2:
        family = "fragment_deletion"
    else:
        family = "local_refinement"

    primitives = row.get("primitive_count") or 0
    scale = "compact" if primitives <= 4 else ("extended" if primitives <= 11 else "large")
    sites = row.get("changed_slot_count") or 0
    boundary = "single_site" if sites <= 1 else ("two_site" if sites == 2 else "multi_site")
    lane = "structured" if primitives > 4 or touches_ring else "shallow"
    return (lane, family, scale, boundary)


def read_run(path: Path, run: str) -> list[dict]:
    with gzip.open(Path(path), "rt") as handle:
        rows = [r for r in map(json.loads, handle) if r.get("run") == run]
    return sorted((r for r in rows if r.get("query") is not None), key=lambda r: r["query"])


def winning_path(ordered) -> tuple[str, ...]:
    """The hypothesis of the program that produced the run's best molecule."""
    return hypothesis_path(min(ordered, key=lambda r: r["score"]))


def replay(ordered, *, rule: str, seed: int = 0, warmup: int = 20) -> dict:
    """Replay one run, revealing outcomes chronologically.

    At each charged call the tree has seen only what was paid for earlier. The allocator
    names the branch it would fund next; we record whether that is the branch that
    eventually produced the run's best molecule. `warmup` calls are revealed before any
    allocation is scored, because an allocator with no observations is uninformative by
    construction rather than by fault.
    """
    rng = np.random.default_rng(seed)
    tree = HypothesisTree()
    target = winning_path(ordered)
    horizon = len(ordered)
    incumbent = float("inf")

    chosen_target, decisions = 0, 0
    first_hit = None
    available: set[tuple[str, ...]] = set()
    for index, row in enumerate(ordered):
        path = hypothesis_path(row)
        available.add(path)
        if index >= warmup and available:
            pick = allocate(
                tree,
                sorted(available),
                rng,
                rule=rule,
                budget_remaining=horizon - index,
                horizon=horizon,
            )
            decisions += 1
            if pick == target:
                chosen_target += 1
                if first_hit is None:
                    first_hit = index
        tree.record(
            path,
            score=row["score"],
            parent_score=row.get("parent_score"),
            incumbent=None if incumbent == float("inf") else incumbent,
        )
        incumbent = min(incumbent, row["score"])

    historical = sum(1 for r in ordered if hypothesis_path(r) == target) / max(len(ordered), 1)
    return {
        "calls": len(ordered),
        "decisions_scored": decisions,
        "winning_hypothesis": list(target),
        "branches_seen": len(available),
        "share_allocated_to_winning_branch": chosen_target / decisions if decisions else 0.0,
        "share_the_run_actually_spent_there": historical,
        "first_call_allocated_there": first_hit,
        "best_score": min(r["score"] for r in ordered),
        "new_oracle_calls": 0,
    }


def compare(path: Path, runs, *, rules=("thompson", "ucb", "uniform"), seeds=(0, 1, 2)) -> dict:
    """Every rule on every run, averaged over seeds."""
    report = defaultdict(dict)
    for run in runs:
        ordered = read_run(path, run)
        if len(ordered) < 60:
            continue
        for rule in rules:
            shares = [
                replay(ordered, rule=rule, seed=s)["share_allocated_to_winning_branch"]
                for s in seeds
            ]
            report[run][rule] = float(np.mean(shares))
        report[run]["historical"] = replay(ordered, rule="uniform", seed=0)[
            "share_the_run_actually_spent_there"
        ]
        report[run]["branches"] = replay(ordered, rule="uniform", seed=0)["branches_seen"]
    return dict(report)
