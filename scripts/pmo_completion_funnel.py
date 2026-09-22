#!/usr/bin/env python
"""WHERE does a repaired completion get lost between synthesis and the charged batch?

The scored A/B rejected the completion repair while the repair reached only 3.8%
of charged candidates.  Nine molecules cannot arithmetically produce a 22% AUC
gap, so before the REJECT can be read as evidence about large structured
completions, the funnel has to say whether the mechanism was throttled.

Every count below is read from the run's OWN durable artifacts -- the per-round
``pending.json`` batch, which carries each attempt's status and each pool
candidate's full provenance -- so this is the path the campaign executed, not a
replay of it.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/pmo_completion_repair_v1"
RUNS = OUT / "scored_ab/celecoxib_rediscovery"
ARMS = ("A_deployed_b", "B_completion")


def configure(runs: Path, arms: tuple[str, ...]) -> None:
    """Point the analysis at another run root; the hop logic is unchanged."""

    global RUNS, ARMS
    RUNS, ARMS = Path(runs), tuple(arms)

#: Signatures of a failure inside a component install, so a rejection can be
#: attributed to the repaired path rather than guessed at.
COMPONENT_SIGNATURES = (
    ("cycle_close", re.compile(r"invalid cycle_close instance")),
    ("atom_insert_multi_order", re.compile(
        r"invalid atom_insert instance.*neighbors=\(\((\d+), ([23])\)\)")),
    ("atom_insert_single_order", re.compile(
        r"invalid atom_insert instance.*neighbors=\(\((\d+), 1\)\)")),
)


def carries_completion(node) -> tuple[bool, int, list[str]]:
    """Does this metadata subtree carry a repaired completion, and at what size?"""

    kinds, sizes, stack = [], [], [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "completion" in item and "requested_size" in item:
                kinds.append(str(item["completion"]))
                sizes.append(int(item.get("requested_size") or 0))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return bool(kinds), (max(sizes) if sizes else 0), kinds


def completion_families(node) -> list[str]:
    """Which generic families the program used, for the share denominator."""

    out, stack = [], [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "family" in item and "primitive_edits" in item:
                out.append(str(item["family"]))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return out


def classify_rejection(reason: str) -> str:
    for name, pattern in COMPONENT_SIGNATURES:
        if pattern.search(reason):
            return name
    if "work_limit" in reason:
        return "work_limit"
    if "no generic module executed" in reason or "no v1 module executed" in reason:
        return "no_module_executed"
    if "invalid" in reason:
        return "other_invalid_primitive"
    return "other"


def charged_endpoints(folder: Path) -> tuple[set[str], int]:
    endpoints, n = set(), 0
    for path in sorted(folder.glob("oracle/query_*/result.json")):
        row = json.loads(path.read_text())
        if row.get("role") == "initialization":
            continue
        endpoints.add(row["endpoint"])
        n += 1
    return endpoints, n


def analyse(arm: str) -> dict:
    folder = RUNS / arm
    batches = sorted(folder.glob("campaign/round_*/pending.json"))
    hops = {
        "attempts": collections.Counter(),
        "attempts_with_completion": collections.Counter(),
        "eligible_attempts": 0,
        "eligible_attempts_with_completion": 0,
        "eligible_pool": 0,
        "eligible_pool_with_completion": 0,
        "selected": 0,
        "selected_with_completion": 0,
    }
    rejections = collections.Counter()
    eligible_completion_family = 0
    family_counts = collections.Counter()
    eligible_families = collections.Counter()
    sizes_eligible, sizes_selected = [], []
    selected_endpoints, pool_endpoints = set(), set()
    completion_pool_endpoints, completion_selected_endpoints = set(), set()

    for path in batches:
        batch = json.loads(path.read_text())["batch"]
        for attempt in batch["attempts"]:
            status = attempt.get("status", "unknown")
            hops["attempts"][status] += 1
            metadata = attempt.get("metadata")
            if metadata is None:
                if status == "execution_rejected":
                    rejections[classify_rejection(str(attempt.get("reason", "")))] += 1
                continue
            has, size, _kinds = carries_completion(metadata)
            for family in completion_families(metadata):
                family_counts[family] += 1
                if status == "eligible":
                    eligible_families[family] += 1
            if has:
                hops["attempts_with_completion"][status] += 1
            if status == "eligible":
                hops["eligible_attempts"] += 1
                if {"segment_grow", "segment_replace"} & set(
                    completion_families(metadata)
                ):
                    eligible_completion_family += 1
                if has:
                    hops["eligible_attempts_with_completion"] += 1
                    sizes_eligible.append(size)
        pool = (batch.get("eligible_pool") or {}).get("candidates")
        if pool is None:
            # The bootstrap batch is built by initial_dynamic_program_batch_v21 and
            # carries no merged pool; count it as a hop the funnel skips rather than
            # silently folding it into the pooled rounds.
            hops["bootstrap_batches"] = hops.get("bootstrap_batches", 0) + 1
            pool = batch.get("candidates", [])
        for candidate in pool:
            hops["eligible_pool"] += 1
            pool_endpoints.add(candidate["endpoint"])
            has, size, _ = carries_completion(candidate.get("provenance", {}))
            if has:
                hops["eligible_pool_with_completion"] += 1
                completion_pool_endpoints.add(candidate["endpoint"])
        for candidate in (batch.get("proposal_pool") or batch)["candidates"]:
            hops["selected"] += 1
            selected_endpoints.add(candidate["endpoint"])
            has, size, _ = carries_completion(candidate.get("provenance", {}))
            if has:
                hops["selected_with_completion"] += 1
                completion_selected_endpoints.add(candidate["endpoint"])
                sizes_selected.append(size)

    charged, n_charged = charged_endpoints(folder)
    return {
        "arm": arm,
        "rounds": len(batches),
        "bootstrap_batches_without_a_merged_pool": hops.get("bootstrap_batches", 0),
        "hop_1_synthesis_attempts": dict(hops["attempts"]),
        "hop_1_attempts_total": sum(hops["attempts"].values()),
        "hop_1_with_completion_by_status": dict(hops["attempts_with_completion"]),
        "hop_2_eligible_attempts": hops["eligible_attempts"],
        "hop_2_eligible_with_completion": hops["eligible_attempts_with_completion"],
        "hop_2_completion_share_of_eligible": round(
            hops["eligible_attempts_with_completion"] / max(1, hops["eligible_attempts"]), 4
        ),
        "hop_3_eligible_pool": hops["eligible_pool"],
        "hop_3_pool_with_completion": hops["eligible_pool_with_completion"],
        "hop_3_completion_share_of_pool": round(
            hops["eligible_pool_with_completion"] / max(1, hops["eligible_pool"]), 4
        ),
        "hop_4_selected_by_credit_allocate": hops["selected"],
        "hop_4_selected_with_completion": hops["selected_with_completion"],
        "hop_4_completion_share_of_selected": round(
            hops["selected_with_completion"] / max(1, hops["selected"]), 4
        ),
        "hop_5_charged_non_initialization": n_charged,
        "hop_5_charged_distinct_endpoints": len(charged),
        "hop_5_charged_with_completion": len(completion_selected_endpoints & charged),
        "allocator_discard_rate": round(
            1 - hops["selected"] / max(1, hops["eligible_pool"]), 4
        ),
        "allocator_discard_rate_for_completions": (
            round(
                1
                - hops["selected_with_completion"]
                / max(1, hops["eligible_pool_with_completion"]),
                4,
            )
            if hops["eligible_pool_with_completion"]
            else None
        ),
        "execution_rejection_reasons": dict(rejections.most_common()),
        "family_counts_over_attempts_with_metadata": dict(family_counts.most_common()),
        "eligible_family_counts": dict(eligible_families.most_common()),
        "hop_1b_eligible_using_a_completion_family": eligible_completion_family,
        "hop_1b_of_those_carrying_the_law": hops["eligible_attempts_with_completion"],
        "hop_1b_FOOTPRINT": (
            round(hops["eligible_attempts_with_completion"] / eligible_completion_family, 4)
            if eligible_completion_family
            else None
        ),
        "requested_size_eligible": sorted(sizes_eligible),
        "requested_size_selected": sorted(sizes_selected),
    }


def main() -> None:
    report = {
        "schema_version": "pmo_completion_funnel_v1",
        "new_oracle_calls": 0,
        "question": (
            "Why does a repaired completion reach only 3.8% of charged candidates "
            "when roughly a fifth of synthesized proposals carry a completion module?"
        ),
        "source": "each run's own per-round pending.json batches and oracle receipts",
        "arms": {arm: analyse(arm) for arm in ARMS},
    }
    (OUT / "funnel_v1.json").write_text(json.dumps(report, sort_keys=True, indent=1))

    for arm in ARMS:
        a = report["arms"][arm]
        print(f"\n=== {arm} ({a['rounds']} rounds) ===")
        print(f"  hop 1 synthesis attempts      {a['hop_1_attempts_total']:5d}  "
              f"{a['hop_1_synthesis_attempts']}")
        print(f"        with a completion       {a['hop_1_with_completion_by_status']}")
        print(f"  hop 1b using a completion fam {a['hop_1b_eligible_using_a_completion_family']:5d}  "
              f"carrying the law {a['hop_1b_of_those_carrying_the_law']:4d}  "
              f"FOOTPRINT {a['hop_1b_FOOTPRINT']}")
        print(f"  hop 2 eligible attempts       {a['hop_2_eligible_attempts']:5d}  "
              f"with completion {a['hop_2_eligible_with_completion']:4d} "
              f"({a['hop_2_completion_share_of_eligible']:.1%})")
        print(f"  hop 3 merged eligible pool    {a['hop_3_eligible_pool']:5d}  "
              f"with completion {a['hop_3_pool_with_completion']:4d} "
              f"({a['hop_3_completion_share_of_pool']:.1%})")
        print(f"  hop 4 credit_allocate picks   {a['hop_4_selected_by_credit_allocate']:5d}  "
              f"with completion {a['hop_4_selected_with_completion']:4d} "
              f"({a['hop_4_completion_share_of_selected']:.1%})")
        print(f"  hop 5 charged                 {a['hop_5_charged_non_initialization']:5d}  "
              f"with completion {a['hop_5_charged_with_completion']:4d}")
        print(f"  allocator discard  ALL {a['allocator_discard_rate']:.1%}   "
              f"COMPLETIONS {a['allocator_discard_rate_for_completions']}")
        print(f"  execution rejections: {a['execution_rejection_reasons']}")
        print(f"  eligible families   : {a['eligible_family_counts']}")
    print("\nwrote", OUT / "funnel_v1.json")


if __name__ == "__main__":
    main()
